//! Real scheduler/lifetime test with a deliberately non-executable adapter.
//! No model tensors, logits, committed output or numerical qualification.
use super::*;
use crate::test_support::{self, HostBytes};
use std::{
    sync::{Condvar, Mutex, atomic::AtomicBool},
    time::{Duration, Instant},
};
use zerogravity_core::identity::SourceIdentity;
use zerogravity_engine::{
    model_instance::{PreparedModel, TextSession, TextTokenizer},
    router::{Request, ResponseEvent, Router},
};

struct Gate {
    released: Mutex<bool>,
    ready: Condvar,
    entered: AtomicBool,
}
impl Gate {
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.ready.notify_all();
    }
}
struct Release(Arc<Gate>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}
struct RefusingModel {
    source: SourceIdentity,
    gate: Arc<Gate>,
}
impl TextTokenizer for RefusingModel {
    fn eos(&self) -> u32 {
        1
    }
    fn decode(&self, _: &[u32], _: bool) -> Result<String> {
        Err(Error::Unsupported("no fixture decoder".into()))
    }
    fn prompt_tokens(&self, _: &str, _: bool) -> Result<Vec<u32>> {
        Ok(vec![0])
    }
    fn prompt_token_bound(&self, _: usize, _: bool) -> usize {
        1
    }
    fn admitted_prompt_scratch_bound(&self, _: usize, _: bool) -> Result<u64> {
        Ok(0)
    }
    fn decode_scratch_bound(&self, _: usize) -> usize {
        0
    }
}
impl PreparedModel for RefusingModel {
    fn source(&self) -> SourceIdentity {
        self.source
    }
    fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
        Arc::new(RefusingModel {
            source: self.source,
            gate: self.gate.clone(),
        })
    }
    fn max_context(&self) -> usize {
        8
    }
    fn max_session_bytes(&self) -> u64 {
        16
    }
    fn session_scratch_bytes(&self) -> u64 {
        16
    }
    fn max_token_bytes(&self) -> usize {
        1
    }
    fn vocabulary_size(&self) -> u64 {
        2
    }
    fn new_session(self: Arc<Self>) -> Result<Box<dyn TextSession>> {
        self.gate.entered.store(true, Ordering::Release);
        let released = self.gate.released.lock().unwrap();
        let _wait = self
            .gate
            .ready
            .wait_timeout_while(released, Duration::from_secs(5), |released| !*released)
            .unwrap();
        Err(Error::Unsupported(
            "transport fixture has no model math".into(),
        ))
    }
}
fn request() -> Request {
    Request {
        model: "fixture".into(),
        input: "x".into(),
        chat: false,
        max_tokens: 1,
        stream: false,
        token_metrics: None,
    }
}

#[test]
fn server_stop_preserves_callers_runtime_active_and_queued_sdk_tasks() {
    let _lock = test_support::lock();
    let host = test_support::host();
    let baseline = host.serving_bytes();
    let gate = Arc::new(Gate {
        released: Mutex::new(false),
        ready: Condvar::new(),
        entered: AtomicBool::new(false),
    });
    let runtime = Runtime::builder().build().unwrap();
    let _release_on_unwind = Release(gate.clone());
    let model: Arc<dyn PreparedModel> = Arc::new(RefusingModel {
        source: SourceIdentity::for_new_allocation().unwrap(),
        gate: gate.clone(),
    });
    runtime
        .configure_text(|| {
            Router::new(
                [("fixture".into(), model)].into_iter().collect(),
                1,
                1,
                8,
                1,
                host.clone(),
                Some(4),
            )
        })
        .unwrap();
    let server = Server::builder(runtime.clone())
        .bind("127.0.0.1:0".parse().unwrap())
        .build()
        .unwrap()
        .start()
        .unwrap();
    let active = runtime.request_text(request()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while !gate.entered.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let queued = runtime.request_text(request()).unwrap();
    server.shutdown().unwrap();
    assert!(!runtime.status().closed);
    assert_eq!(
        runtime
            .with_text_router(|router| Ok(router.queue_snapshot().pending))
            .unwrap(),
        1
    );
    gate.release();
    for events in [active, queued] {
        match events.events.recv_timeout(Duration::from_secs(3)).unwrap() {
            ResponseEvent::Error(error) => {
                assert_eq!(error.code, "inference_error");
                assert!(
                    error
                        .message
                        .contains("transport fixture has no model math")
                );
            }
            _ => panic!("non-executable fixture must not produce output"),
        }
    }
    runtime.shutdown().unwrap();
    drop(runtime);
    assert_eq!(host.serving_bytes(), baseline);
}
