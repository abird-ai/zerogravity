//! Model-free scheduler tests; no model math or physical execution claim.
use super::*;
use std::sync::{Condvar, Mutex};
use zerogravity_core::identity::SourceIdentity;

#[derive(Default)]
struct Gate {
    state: Mutex<(bool, bool)>,
    changed: Condvar,
}
impl Gate {
    fn worker(&self) {
        let mut state = self.state.lock().unwrap();
        state.0 = true;
        self.changed.notify_all();
        while !state.1 {
            let (next, timeout) = self
                .changed
                .wait_timeout(state, Duration::from_secs(5))
                .unwrap();
            state = next;
            if timeout.timed_out() {
                break;
            }
        }
    }
    fn entered(&self) {
        let state = self.state.lock().unwrap();
        let (state, _) = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(5), |s| !s.0)
            .unwrap();
        assert!(state.0, "worker did not enter its first quantum");
    }
    fn release(&self) {
        self.state.lock().unwrap().1 = true;
        self.changed.notify_all();
    }
}

pub(super) struct MockTokenizer;
impl TextTokenizer for MockTokenizer {
    fn eos(&self) -> u32 {
        1
    }
    fn decode(&self, tokens: &[u32], _: bool) -> crate::Result<String> {
        Ok(tokens
            .iter()
            .map(|id| if *id == 2 { 'A' } else { 'B' })
            .collect())
    }
    fn prompt_tokens(&self, input: &str, _: bool) -> crate::Result<Vec<u32>> {
        Ok(vec![2; input.len()])
    }
    fn prompt_token_bound(&self, bytes: usize, _: bool) -> usize {
        bytes
    }
    fn admitted_prompt_scratch_bound(&self, _: usize, _: bool) -> crate::Result<u64> {
        Ok(0)
    }
    fn decode_scratch_bound(&self, tokens: usize) -> usize {
        tokens * 4
    }
}
struct MockModel {
    source: SourceIdentity,
    next: AtomicUsize,
    log: Mutex<Vec<(usize, &'static str, usize)>>,
    gate: Option<Arc<Gate>>,
    _source_charge: GlobalResourceLease,
}
impl MockModel {
    fn new(host: &HostPool, gate: Option<Arc<Gate>>) -> Arc<Self> {
        Arc::new(Self {
            source: SourceIdentity::for_new_allocation().unwrap(),
            next: AtomicUsize::new(0),
            log: Mutex::new(Vec::new()),
            gate,
            _source_charge: host
                .admit("mock", "source", AllocationClass::ImmutableWeights, 64)
                .unwrap(),
        })
    }
}
struct MockSession {
    id: usize,
    position: usize,
    model: Arc<MockModel>,
}
impl MockSession {
    fn prefill(&mut self, tokens: &[u32], stage: &'static str) {
        if self.id == 0
            && self.position == 0
            && let Some(gate) = &self.model.gate
        {
            gate.worker();
        }
        self.model
            .log
            .lock()
            .unwrap()
            .push((self.id, stage, tokens.len()));
        self.position += tokens.len();
    }
}
impl TextSession for MockSession {
    fn prefill_chunk_without_logits(&mut self, tokens: &[u32]) -> crate::Result<()> {
        self.prefill(tokens, "chunk");
        Ok(())
    }
    fn prefill_transaction(&mut self, tokens: &[u32]) -> crate::Result<Vec<f32>> {
        self.prefill(tokens, "last");
        Ok(vec![0.0, 0.0, 2.0, 1.0])
    }
    fn forward(&mut self, _: u32) -> crate::Result<Vec<f32>> {
        self.model
            .log
            .lock()
            .unwrap()
            .push((self.id, "decode", self.position));
        self.position += 1;
        Ok(vec![0.0, 0.0, 1.0, 2.0])
    }
}
impl PreparedModel for MockModel {
    fn source(&self) -> SourceIdentity {
        self.source
    }
    fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
        Arc::new(MockTokenizer)
    }
    fn max_context(&self) -> usize {
        128
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
        let id = self.next.fetch_add(1, Ordering::AcqRel);
        self.log.lock().unwrap().push((id, "new", 0));
        Ok(Box::new(MockSession {
            id,
            position: 0,
            model: self,
        }))
    }
}
fn router(host: &HostPool, model: &Arc<MockModel>) -> Router {
    let model: Arc<dyn PreparedModel> = model.clone();
    Router::new(
        BTreeMap::from([("model".into(), model.clone()), ("alias".into(), model)]),
        1,
        2,
        128,
        16,
        host.clone(),
        Some(16),
    )
    .unwrap()
}
fn request(input: &str) -> Request {
    Request {
        model: "model".into(),
        input: input.into(),
        chat: false,
        max_tokens: 2,
        stream: false,
        token_metrics: None,
    }
}
fn done(receipt: &RequestEvents) -> String {
    match receipt.events.recv_timeout(Duration::from_secs(5)).unwrap() {
        ResponseEvent::Done {
            output,
            completion_tokens,
            ..
        } => {
            assert_eq!(completion_tokens, 2);
            output
        }
        ResponseEvent::Error(error) => panic!("{error}"),
        _ => panic!("expected nonstream terminal"),
    }
}

#[test]
fn scheduler_bounds_prefill_quanta_and_round_robins_independent_sessions() {
    let host = HostPool::isolated(256).unwrap();
    let gate = Arc::new(Gate::default());
    let model = MockModel::new(&host, Some(gate.clone()));
    let mut router = router(&host, &model);
    assert!(
        router
            .model("model")
            .unwrap()
            .shares_source(&router.model("alias").unwrap())
    );
    let first = router.request(request(&"a".repeat(65))).unwrap();
    gate.entered();
    let mut second_request = request(&"b".repeat(65));
    second_request.model = "alias".into();
    let second = router.request(second_request).unwrap();
    gate.release();
    assert_eq!(done(&first), "AB");
    assert_eq!(done(&second), "AB");
    router.shutdown().unwrap();
    let log = model.log.lock().unwrap();
    let second_prefill = log
        .iter()
        .position(|row| row.0 == 1 && row.1 == "chunk")
        .unwrap();
    let first_decode = log
        .iter()
        .position(|row| row.0 == 0 && row.1 == "decode")
        .unwrap();
    assert!(
        second_prefill < first_decode,
        "second request progresses before first finishes"
    );
    for id in [0, 1] {
        assert!(log.contains(&(id, "new", 0)));
        assert_eq!(
            log.iter()
                .filter(|r| r.0 == id && matches!(r.1, "chunk" | "last"))
                .map(|r| r.2)
                .collect::<Vec<_>>(),
            [32, 32, 1]
        );
        assert!(
            log.contains(&(id, "decode", 65)),
            "sessions have independent positions"
        );
    }
    drop(log);
    drop((first, second, router, model));
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn cancellation_retires_queued_and_active_work_but_not_the_shared_source() {
    let host = HostPool::isolated(256).unwrap();
    let gate = Arc::new(Gate::default());
    let model = MockModel::new(&host, Some(gate.clone()));
    let mut router = router(&host, &model);
    let first = router.request(request(&"a".repeat(65))).unwrap();
    gate.entered();
    let queued = router.request(request("later")).unwrap();
    first.cancel();
    drop(queued);
    gate.release();
    router.shutdown().unwrap();
    assert_eq!(
        model.next.load(Ordering::Acquire),
        1,
        "cancelled queued request never creates a session"
    );
    assert_eq!(
        *model.log.lock().unwrap(),
        [(0, "new", 0), (0, "chunk", 32)]
    );
    drop((first, router));
    assert_eq!(host.serving_bytes(), 0);
    assert!(
        host.snapshot().unwrap().1 > 0,
        "source owner remains independent of requests"
    );
    drop(model);
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn exhausted_ids_never_wrap_and_lost_worker_rejects_admission() {
    let host = HostPool::isolated(256).unwrap();
    let model = MockModel::new(&host, None);
    let mut router = router(&host, &model);
    router.next_id.store(u64::MAX, Ordering::Relaxed);
    for _ in 0..2 {
        assert_eq!(
            router.request(request("x")).err().unwrap().code,
            "id_exhausted"
        );
    }
    router.shutdown().unwrap();
    router.next_id.store(1, Ordering::Relaxed);
    assert_eq!(
        router.request(request("x")).err().unwrap().code,
        "worker_unavailable"
    );
    drop((router, model));
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn held_queue_ticket_refuses_with_429_then_release_admits_a_request() {
    let host = HostPool::isolated(256).unwrap();
    let model = MockModel::new(&host, None);
    let router = Router::new(
        BTreeMap::from([("model".into(), model.clone() as Arc<dyn PreparedModel>)]),
        1,
        1,
        128,
        16,
        host.clone(),
        Some(1),
    )
    .unwrap();
    let ticket = router.queue.reserve().unwrap();
    let held = router.queue_snapshot();
    assert_eq!(
        (held.pending, held.in_progress, held.limit),
        (0, 1, Some(1))
    );
    let refusal = router.request(request("x")).err().unwrap();
    assert_eq!((refusal.status, refusal.code), (429, "queue_full"));
    assert_eq!(router.queue_snapshot().in_progress, 1);
    drop(ticket);
    assert_eq!(router.queue_snapshot().in_progress, 0);
    let receipt = router.request(request("x")).unwrap();
    assert_eq!(done(&receipt), "AB");
    drop((receipt, router, model));
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn runtime_direct_text_and_task_stream_share_scheduler_and_shutdown() {
    let host = HostPool::isolated(256).unwrap();
    let model = MockModel::new(&host, None);
    let runtime = crate::Runtime::builder()
        .limits(crate::scheduler::SchedulerLimits {
            max_active: 2,
            max_queued: Some(16),
            max_input_bytes: 1024,
        })
        .build()
        .unwrap();
    runtime
        .configure_text(|| Ok(router(&host, &model)))
        .unwrap();
    let clone = runtime.clone();
    assert_eq!(runtime.status().loaded_models, 2);
    assert_eq!(runtime.models().list().unwrap().len(), 2);
    assert_eq!(runtime.status().execution_adapters, 1);
    assert!(
        crate::router::resolve(&runtime, "model")
            .unwrap()
            .shares_source(&crate::router::resolve(&clone, "alias").unwrap())
    );
    let receipt = runtime.request_text(request("x")).unwrap();
    assert_eq!(done(&receipt), "AB");
    let mut stream = clone
        .run(&crate::TaskRequest {
            model: "alias".into(),
            input: crate::TaskInput::Text("y".into()),
            max_output: 2.try_into().unwrap(),
        })
        .unwrap();
    let mut terminal = 0;
    while let Some(event) = stream.next().unwrap() {
        if let ResponseEvent::Done { output, .. } = event {
            assert_eq!(output, "AB");
            terminal += 1;
        }
    }
    assert_eq!(terminal, 1);
    drop((receipt, stream));
    clone.shutdown().unwrap();
    assert!(runtime.status().closed);
    assert_eq!(runtime.status().loaded_models, 0);
    assert_eq!(
        runtime.request_text(request("x")).err().unwrap().code,
        "runtime_closed"
    );
    drop((runtime, clone, model));
    assert_eq!(host.snapshot().unwrap().1, 0);
}
