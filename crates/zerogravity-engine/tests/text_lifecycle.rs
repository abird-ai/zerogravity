use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_engine::{
    Runtime,
    router::{Request, ServiceError},
};

#[test]
fn unavailable_or_closed_runtime_never_calls_transport_or_loader() {
    let runtime = Runtime::builder().build().unwrap();
    let called = AtomicBool::new(false);
    let error = runtime
        .with_text_router(|_| {
            called.store(true, Ordering::Release);
            Ok(())
        })
        .unwrap_err();
    assert_eq!((error.status, error.code), (503, "runtime_unconfigured"));
    assert!(!called.load(Ordering::Acquire));
    runtime.shutdown().unwrap();
    let error = runtime
        .configure_text(|| {
            called.store(true, Ordering::Release);
            Err(ServiceError::new(
                500,
                "unexpected_loader",
                "must not load after shutdown",
            ))
        })
        .unwrap_err();
    assert_eq!((error.status, error.code), (503, "runtime_closed"));
    let error = runtime
        .with_text_router(|_| {
            called.store(true, Ordering::Release);
            Ok(())
        })
        .unwrap_err();
    assert_eq!(error.code, "runtime_closed");
    assert!(!called.load(Ordering::Acquire));
    let error = runtime
        .request_text(Request {
            model: "unloaded".into(),
            input: "input".into(),
            chat: false,
            max_tokens: 1,
            stream: false,
            token_metrics: None,
        })
        .err()
        .unwrap();
    assert_eq!(error.code, "runtime_closed");
}
