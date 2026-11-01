//! Actual router joins and runtime lifecycle, with model-free controlled workers.
#[path = "lifecycle_tests/reentrancy.rs"]
mod reentrancy;
use super::*;
use crate::{Runtime, diagnostics::RuntimeStatus, scheduler::SchedulerLimits};
use std::sync::mpsc as signals;

const WAIT: Duration = Duration::from_secs(5);
fn received<T>(receiver: &signals::Receiver<T>) -> T {
    receiver
        .recv_timeout(WAIT)
        .expect("lifecycle signal timed out")
}

/// A real consumer in the original join list; it probes status only after queue
/// closure, then deliberately holds retirement until the test releases it.
fn status_worker(
    router: &mut Router,
    runtime: Runtime,
    fail: bool,
) -> (signals::Receiver<RuntimeStatus>, signals::SyncSender<()>) {
    let queue = router.queue.worker().unwrap();
    let (observed_tx, observed_rx) = signals::sync_channel(1);
    let (release_tx, release_rx) = signals::sync_channel(1);
    router.workers.push(thread::spawn(move || {
        assert!(
            matches!(queue.pop_wait(WAIT, |_| false), Pop::Closed),
            "admission must close before join"
        );
        observed_tx.send(runtime.status()).unwrap();
        received(&release_rx);
        assert!(!fail, "injected lifecycle worker panic");
    }));
    (observed_rx, release_tx)
}
fn shutting_down(
    runtime: Runtime,
) -> (
    thread::JoinHandle<()>,
    signals::Receiver<Result<(), String>>,
) {
    let (done_tx, done_rx) = signals::sync_channel(1);
    let handle = thread::spawn(move || {
        done_tx
            .send(runtime.shutdown().map_err(|error| error.to_string()))
            .unwrap();
    });
    (handle, done_rx)
}
fn pending<T>(receiver: &signals::Receiver<T>) {
    assert!(
        matches!(receiver.try_recv(), Err(signals::TryRecvError::Empty)),
        "returned before the terminal join"
    );
}

#[test]
fn worker_status_is_unlocked_and_every_shutdown_waits_for_the_same_failed_join() {
    let runtime = Runtime::builder().build().unwrap();
    let (mut router, host) = Router::transport_test();
    let (worker_status, release) = status_worker(&mut router, runtime.clone(), true);
    runtime.configure_text(|| Ok(router)).unwrap();
    let (first, first_done) = shutting_down(runtime.clone());
    assert!(received(&worker_status).closed);
    let (second, second_done) = shutting_down(runtime.clone());
    runtime.wait_for_shutdown_waiters(1);
    pending(&first_done);
    pending(&second_done);
    assert_eq!(
        runtime
            .request_text(Request {
                model: "not-loaded".into(),
                input: "x".into(),
                chat: false,
                max_tokens: 1,
                stream: false,
                token_metrics: None,
            })
            .err()
            .unwrap()
            .code,
        "runtime_closed"
    );
    release.send(()).unwrap();
    let error = received(&first_done).unwrap_err();
    assert_eq!(received(&second_done).unwrap_err(), error);
    first.join().unwrap();
    second.join().unwrap();
    assert_eq!(
        runtime.shutdown().unwrap_err().to_string(),
        error,
        "failure must remain sticky for later clones"
    );
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn shutdown_waits_for_unlocked_loader_and_its_raced_returned_router_retirement() {
    let runtime = Runtime::builder().build().unwrap();
    let (mut router, host) = Router::transport_test();
    let (worker_status, release_worker) = status_worker(&mut router, runtime.clone(), false);
    let (started_tx, started_rx) = signals::sync_channel(1);
    let (probe_tx, probe_rx) = signals::sync_channel(1);
    let (loader_status_tx, loader_status_rx) = signals::sync_channel(1);
    let (return_tx, return_rx) = signals::sync_channel(1);
    let (configured_tx, configured_rx) = signals::sync_channel(1);
    let loading_runtime = runtime.clone();
    let loading = thread::spawn(move || {
        let result = loading_runtime.configure_text(|| {
            started_tx.send(()).unwrap();
            received(&probe_rx);
            loader_status_tx.send(loading_runtime.status()).unwrap();
            received(&return_rx);
            Ok(router)
        });
        configured_tx
            .send(result.map_err(|error| (error.status, error.code)))
            .unwrap();
    });
    received(&started_rx);
    let (first, first_done) = shutting_down(runtime.clone());
    let (second, second_done) = shutting_down(runtime.clone());
    runtime.wait_for_shutdown_waiters(2);
    probe_tx.send(()).unwrap();
    assert!(received(&loader_status_rx).closed);
    pending(&first_done);
    pending(&second_done);
    return_tx.send(()).unwrap();
    assert!(received(&worker_status).closed);
    runtime.wait_for_shutdown_waiters(2);
    pending(&configured_rx);
    pending(&first_done);
    pending(&second_done);
    release_worker.send(()).unwrap();
    assert_eq!(
        received(&configured_rx).unwrap_err(),
        (503, "runtime_closed")
    );
    assert!(received(&first_done).is_ok());
    assert!(received(&second_done).is_ok());
    loading.join().unwrap();
    first.join().unwrap();
    second.join().unwrap();
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn rejected_configuration_joins_unlocked_and_retains_any_worker_failure() {
    for fail in [false, true] {
        let runtime = Runtime::builder()
            .limits(SchedulerLimits {
                max_active: 1,
                max_queued: Some(1),
                max_input_bytes: 1024,
            })
            .build()
            .unwrap();
        // The original idle fixture has pending limit 8: this runtime rejects it.
        let (mut router, host) = Router::transport_test();
        let (worker_status, release) = status_worker(&mut router, runtime.clone(), fail);
        let (configured_tx, configured_rx) = signals::sync_channel(1);
        let loading_runtime = runtime.clone();
        let loading = thread::spawn(move || {
            let result = loading_runtime.configure_text(|| {
                assert!(!loading_runtime.status().closed);
                Ok(router)
            });
            configured_tx
                .send(result.map_err(|error| (error.status, error.code, error.message)))
                .unwrap();
        });
        assert!(!received(&worker_status).closed);
        pending(&configured_rx);
        assert_eq!(
            runtime
                .configure_text(|| panic!("second loader must not run during retirement"))
                .unwrap_err()
                .code,
            "runtime_configured"
        );
        release.send(()).unwrap();
        let (status, code, message) = received(&configured_rx).unwrap_err();
        loading.join().unwrap();
        if fail {
            assert_eq!((status, code), (500, "worker_shutdown"));
            let error = runtime.shutdown().unwrap_err().to_string();
            assert!(error.contains(&message));
            assert_eq!(runtime.shutdown().unwrap_err().to_string(), error);
        } else {
            assert_eq!((status, code), (400, "invalid_config"));
            // A normal load error also releases the slot, without running joins
            // or user code while the runtime lock is held.
            assert_eq!(
                runtime
                    .configure_text(|| {
                        assert!(!runtime.status().closed);
                        Err(ServiceError::new(
                            400,
                            "source_refused",
                            "test load refusal",
                        ))
                    })
                    .unwrap_err()
                    .code,
                "source_refused"
            );
            runtime.shutdown().unwrap();
        }
        assert_eq!(host.snapshot().unwrap().1, 0);
    }
}

#[test]
fn loader_unwind_retires_its_router_before_waking_shutdown_waiters() {
    let runtime = Runtime::builder().build().unwrap();
    let (mut router, host) = Router::transport_test();
    let (worker_status, release_worker) = status_worker(&mut router, runtime.clone(), false);
    let (started_tx, started_rx) = signals::sync_channel(1);
    let (panic_tx, panic_rx) = signals::sync_channel(1);
    let (unwound_tx, unwound_rx) = signals::sync_channel(1);
    let loading_runtime = runtime.clone();
    let loading = thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loading_runtime.configure_text(|| {
                let _owned_router = router;
                started_tx.send(()).unwrap();
                received(&panic_rx);
                panic!("injected loader panic");
            })
        }));
        unwound_tx.send(result.is_err()).unwrap();
    });
    received(&started_rx);
    let (shutdown, shutdown_done) = shutting_down(runtime.clone());
    runtime.wait_for_shutdown_waiters(1);
    panic_tx.send(()).unwrap();
    assert!(received(&worker_status).closed);
    pending(&shutdown_done);
    pending(&unwound_rx);
    release_worker.send(()).unwrap();
    assert!(received(&unwound_rx));
    let error = received(&shutdown_done).unwrap_err();
    assert!(error.contains("text loader panicked"));
    loading.join().unwrap();
    shutdown.join().unwrap();
    assert_eq!(runtime.shutdown().unwrap_err().to_string(), error);
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn failed_loader_leaves_an_open_slot_for_a_later_real_router() {
    let runtime = Runtime::builder().build().unwrap();
    assert_eq!(
        runtime
            .configure_text(|| {
                assert!(!runtime.status().closed);
                Err(ServiceError::new(
                    400,
                    "source_refused",
                    "test load refusal",
                ))
            })
            .unwrap_err()
            .code,
        "source_refused"
    );
    let (router, host) = Router::transport_test();
    runtime
        .configure_text(|| {
            assert!(!runtime.status().closed);
            Ok(router)
        })
        .unwrap();
    assert_eq!(runtime.status().execution_adapters, 1);
    runtime.shutdown().unwrap();
    assert_eq!(host.snapshot().unwrap().1, 0);
}
