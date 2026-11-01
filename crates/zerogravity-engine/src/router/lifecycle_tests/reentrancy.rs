//! Explicit self-join refusals; no successful inference or device execution.
use super::*;
use crate::router::{WorkerScope, on_worker_thread};
use zerogravity_core::identity::SourceIdentity;

type Observation = (Result<(), String>, RuntimeStatus);

fn probe(runtime: &Runtime) -> Observation {
    (
        runtime.shutdown().map_err(|error| error.to_string()),
        runtime.status(),
    )
}
fn refused(observation: Observation, context: &str, closed: bool) {
    let (result, status) = observation;
    assert!(result.unwrap_err().contains(context));
    assert_eq!(status.closed, closed);
}

#[test]
fn loader_self_shutdown_refuses_before_mutating_or_waiting_on_admission() {
    for racing in [false, true] {
        let runtime = Runtime::builder().build().unwrap();
        let loading_runtime = runtime.clone();
        let (entered_tx, entered_rx) = signals::sync_channel(1);
        let (release_tx, release_rx) = signals::sync_channel(1);
        let (observed_tx, observed_rx) = signals::sync_channel(1);
        let (done_tx, done_rx) = signals::sync_channel(1);
        let loading = thread::spawn(move || {
            let result = loading_runtime.configure_text(|| {
                entered_tx.send(()).unwrap();
                received(&release_rx);
                observed_tx.send(probe(&loading_runtime)).unwrap();
                Err(ServiceError::new(400, "source_refused", "no test source"))
            });
            done_tx.send(result.unwrap_err().code).unwrap();
        });
        received(&entered_rx);
        let shutdown = racing.then(|| shutting_down(runtime.clone()));
        if racing {
            runtime.wait_for_shutdown_waiters(1);
        }
        release_tx.send(()).unwrap();
        refused(received(&observed_rx), "loader/retirement owner", racing);
        assert_eq!(received(&done_rx), "source_refused");
        loading.join().unwrap();
        if let Some((handle, done)) = shutdown {
            received(&done).unwrap();
            handle.join().unwrap();
        } else {
            assert!(!runtime.status().closed);
            let (router, host) = Router::transport_test();
            runtime.configure_text(|| Ok(router)).unwrap();
            runtime.shutdown().unwrap();
            assert_eq!(host.snapshot().unwrap().1, 0);
        }
    }
}

struct WorkerProbe {
    entered: signals::SyncSender<()>,
    release: std::sync::Mutex<signals::Receiver<()>>,
    observed: signals::SyncSender<Observation>,
}
struct RefusingModel {
    source: SourceIdentity,
    runtime: Runtime,
    worker: Option<WorkerProbe>,
    dropped: Option<signals::SyncSender<Observation>>,
    _source_charge: GlobalResourceLease,
}
impl Drop for RefusingModel {
    fn drop(&mut self) {
        if let Some(observed) = &self.dropped {
            observed.send(probe(&self.runtime)).unwrap();
        }
    }
}
impl PreparedModel for RefusingModel {
    fn source(&self) -> SourceIdentity {
        self.source
    }
    fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
        Arc::new(super::super::scheduler_tests::MockTokenizer)
    }
    fn max_context(&self) -> usize {
        16
    }
    fn max_session_bytes(&self) -> u64 {
        4096
    }
    fn session_scratch_bytes(&self) -> u64 {
        1024
    }
    fn max_token_bytes(&self) -> usize {
        1
    }
    fn vocabulary_size(&self) -> u64 {
        4
    }
    fn new_session(self: Arc<Self>) -> crate::Result<Box<dyn TextSession>> {
        let probe = self.worker.as_ref().expect("expected worker probe");
        probe.entered.send(()).unwrap();
        received(&probe.release.lock().unwrap());
        probe.observed.send(self::probe(&self.runtime)).unwrap();
        Err(crate::Error::Unsupported("no test inference".into()))
    }
}
fn model_router(
    runtime: &Runtime,
    worker: Option<WorkerProbe>,
    dropped: Option<signals::SyncSender<Observation>>,
    pending: Option<usize>,
) -> (Router, HostPool) {
    let host = HostPool::isolated(256).unwrap();
    let model: Arc<dyn PreparedModel> = Arc::new(RefusingModel {
        source: SourceIdentity::for_new_allocation().unwrap(),
        runtime: runtime.clone(),
        worker,
        dropped,
        _source_charge: host
            .admit("probe", "source", AllocationClass::ImmutableWeights, 64)
            .unwrap(),
    });
    let router = Router::new(
        BTreeMap::from([("probe".into(), model)]),
        1,
        1,
        16,
        1,
        host.clone(),
        pending,
    )
    .unwrap();
    (router, host)
}

#[test]
fn actual_worker_self_shutdown_refuses_in_ready_and_joining_phases() {
    for racing in [false, true] {
        let runtime = Runtime::builder().build().unwrap();
        let (entered_tx, entered_rx) = signals::sync_channel(1);
        let (release_tx, release_rx) = signals::sync_channel(1);
        let (observed_tx, observed_rx) = signals::sync_channel(1);
        let (router, host) = model_router(
            &runtime,
            Some(WorkerProbe {
                entered: entered_tx,
                release: std::sync::Mutex::new(release_rx),
                observed: observed_tx,
            }),
            None,
            Some(1),
        );
        runtime.configure_text(|| Ok(router)).unwrap();
        let receipt = runtime
            .request_text(Request {
                model: "probe".into(),
                input: "x".into(),
                chat: false,
                max_tokens: 1,
                stream: false,
                token_metrics: None,
            })
            .unwrap();
        received(&entered_rx);
        let shutdown = racing.then(|| shutting_down(runtime.clone()));
        if racing {
            // A second caller proves the first owns Joining and is waiting for
            // this actual model callback; neither may finish before release.
            let (second, second_done) = shutting_down(runtime.clone());
            runtime.wait_for_shutdown_waiters(1);
            release_tx.send(()).unwrap();
            refused(received(&observed_rx), "engine worker", true);
            received(&second_done).unwrap();
            second.join().unwrap();
        } else {
            release_tx.send(()).unwrap();
            refused(received(&observed_rx), "engine worker", false);
        }
        assert!(matches!(
            receipt.events.recv_timeout(WAIT).unwrap(),
            ResponseEvent::Error(_)
        ));
        drop(receipt);
        if let Some((handle, done)) = shutdown {
            received(&done).unwrap();
            handle.join().unwrap();
        }
        runtime.shutdown().unwrap();
        assert_eq!(host.snapshot().unwrap().1, 0);
    }
}

#[test]
fn source_destructor_self_shutdown_refuses_during_load_rejection_and_retirement() {
    for accepted in [false, true] {
        let runtime = Runtime::builder().build().unwrap();
        let (observed_tx, observed_rx) = signals::sync_channel(1);
        // None exceeds the runtime's default count ceiling: rejected load must
        // still retire its real worker/source outside the mutex while Loading.
        let (router, host) = model_router(&runtime, None, Some(observed_tx), accepted.then_some(1));
        let owned = runtime.clone();
        let (done_tx, done_rx) = signals::sync_channel(1);
        let operation = thread::spawn(move || {
            let loaded = owned.configure_text(|| Ok(router));
            if accepted {
                loaded.unwrap();
                owned.shutdown().unwrap();
            } else {
                assert_eq!(loaded.unwrap_err().code, "invalid_config");
            }
            done_tx.send(()).unwrap();
        });
        let observation = received(&observed_rx);
        assert_eq!(observation.1.loaded_models, 0);
        refused(observation, "loader/retirement owner", accepted);
        received(&done_rx);
        operation.join().unwrap();
        assert_eq!(host.snapshot().unwrap().1, 0);
        runtime.shutdown().unwrap();
    }
}

#[test]
fn worker_context_is_scoped_through_nested_entries_and_unwind() {
    assert!(!on_worker_thread());
    let result = std::panic::catch_unwind(|| {
        let outer = WorkerScope::enter();
        assert!(on_worker_thread());
        {
            let _inner = WorkerScope::enter();
            assert!(on_worker_thread());
        }
        assert!(on_worker_thread());
        drop(outer);
        assert!(!on_worker_thread());
        let _unwinding = WorkerScope::enter();
        panic!("injected worker unwind");
    });
    assert!(result.is_err());
    assert!(!on_worker_thread());
}
