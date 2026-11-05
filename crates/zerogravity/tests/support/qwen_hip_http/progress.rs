//! One ordinary HTTP request; read-only progress samples, not performance acceptance.
use super::*;
use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::mpsc,
};

fn sample(h: &Harness, started: Instant, phase: &str) {
    let admission = h.admission();
    let mut record = json!({"elapsed_ms":started.elapsed().as_millis(), "phase":phase});
    // Independent atomic samples, not a transactional state snapshot or GPU read.
    for key in [
        "active_sessions",
        "queued_requests",
        "pending_admission_in_progress",
        "gpu_step_attempts",
        "gpu_forward_calls_completed",
        "completed_model_token_steps",
        "completed_forward_kernel_dispatches",
        "committed_gpu_roots",
        "accepted_output_tokens",
        "output_credit_parked_requests",
        "failed_gpu_quanta",
        "cancelled_requests",
    ] {
        record[key] = admission[key].clone();
    }
    println!("SOURCE_HTTP_PROGRESS {record}");
}

#[test]
#[ignore = "diagnostic: approved populated source/gfx1151/private loopback, fresh release process"]
fn diagnose_populated_http_deadline() {
    let started = Instant::now();
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let capacity: usize = std::env::var("ZG_QWEN_CONTEXT").unwrap().parse().unwrap();
    let case = [1024, 2048, 4096, 8192, 16384]
        .iter()
        .position(|n| *n == capacity)
        .unwrap();
    assert_eq!(std::env::var("ZG_PRIVATE_LOOPBACK").as_deref(), Ok("1"));
    let h = Harness::open(&label, 2, capacity, 3, 3);
    println!(
        "SOURCE_HTTP_STAGE stage=model_loaded elapsed_ms={}",
        started.elapsed().as_millis()
    );
    let reference = super::super::contexts::ContextReference::open(&label);
    let (prompt, expected) = reference.trajectory(case, capacity, h.tokenizer.as_ref());
    let text = h.tokenizer.decode(&expected, true).unwrap();
    let server = h
        .runtime
        .server()
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(zerogravity::serving::ingress::IngressLimits {
            connections: 2,
            body_bytes: 128 * 1024,
            output_bytes: 64 * 1024,
        })
        .enable_per_request_metrics(true)
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    let body = json!({"model":"source", "prompt":prompt, "max_tokens":3,
        "stream":true, "return_token_ids":true});
    assert!(serde_json::to_vec(&body).unwrap().len() <= 128 * 1024);
    println!(
        "SOURCE_HTTP_STAGE stage=request_start model={label} context={capacity} elapsed_ms={} performance=false serving_acceptance=false",
        started.elapsed().as_millis()
    );
    let request_started = Instant::now();
    let result = std::thread::scope(|scope| {
        let (stop, stopped) = mpsc::channel::<()>();
        let observer_h = &h;
        let observer = scope.spawn(move || {
            loop {
                sample(observer_h, request_started, "waiting");
                match stopped.recv_timeout(Duration::from_secs(5)) {
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    _ => break,
                }
            }
            sample(observer_h, request_started, "request_returned");
        });
        // Preserve the original assertion/panic, but join observation and retire
        // the original runtime before reporting the failed diagnostic request.
        let result = catch_unwind(AssertUnwindSafe(|| {
            completion(
                address,
                body,
                &expected,
                &text,
                "context-0",
                false,
                capacity - 2,
            )
        }));
        stop.send(()).unwrap();
        observer.join().unwrap();
        result
    });
    server.shutdown().unwrap();
    sample(&h, request_started, "http_closed");
    drop(reference);
    h.retire();
    println!(
        "SOURCE_HTTP_PROGRESS_RETIRED request_completed={} elapsed_ms={} performance=false serving_acceptance=false",
        result.is_ok(),
        request_started.elapsed().as_millis()
    );
    match result {
        Ok(outcome) => assert!(outcome.is_some()),
        Err(error) => resume_unwind(error),
    }
}
