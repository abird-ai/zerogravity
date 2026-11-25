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

    fn max_token_bytes(&self) -> usize {
        1
    }
    fn vocabulary_size(&self) -> u64 {
        4
    }
}
impl LogitModel for MockModel {
    fn max_session_bytes(&self) -> u64 {
        4096
    }
    fn session_scratch_bytes(&self) -> u64 {
        1024
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
    let model: Arc<dyn LogitModel> = model.clone();
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
        BTreeMap::from([("model".into(), model.clone() as Arc<dyn LogitModel>)]),
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

#[test]
fn catalog_unloads_exact_owner_and_drains_before_cold_reload() {
    use crate::{Runtime, model_instance::ModelInstance};
    use std::time::Instant;
    let host = HostPool::isolated(256).unwrap();
    let gate = Arc::new(Gate::default());
    let old = MockModel::new(&host, Some(gate.clone()));
    let fresh = MockModel::new(&host, None);
    let runtime = Runtime::builder()
        .limits(crate::scheduler::SchedulerLimits {
            max_active: 2,
            ..Default::default()
        })
        .build()
        .unwrap();
    runtime.configure_text(|| Ok(router(&host, &old))).unwrap();
    let registered = runtime.models().resolve("alias").unwrap();
    let foreign = ModelInstance::new(fresh.clone());
    assert_eq!(
        runtime.models().aliases(&registered).unwrap(),
        ["alias", "model"]
    );
    assert!(runtime.models().unload(&foreign).is_err());
    let first = runtime.request_text(request("first")).unwrap();
    gate.entered();
    let mut queued = request("second");
    queued.model = "alias".into();
    let second = runtime.request_text(queued).unwrap();
    let retire = runtime.clone();
    let handle = registered.clone();
    let join = std::thread::spawn(move || retire.models().unload(&handle));
    let deadline = Instant::now() + Duration::from_secs(5);
    while runtime.status().execution_adapters != 0 {
        assert!(
            Instant::now() < deadline,
            "retirement did not close admission"
        );
        std::thread::yield_now();
    }
    assert_eq!(
        runtime.request_text(request("late")).err().unwrap().code,
        "runtime_unconfigured"
    );
    assert_eq!(
        runtime
            .configure_text(|| panic!("cannot load during join"))
            .unwrap_err()
            .code,
        "runtime_configured"
    );
    gate.release();
    assert_eq!(done(&first), "AB");
    assert_eq!(done(&second), "AB");
    join.join().unwrap().unwrap();
    assert!(!runtime.status().closed);
    assert!(runtime.models().list().unwrap().is_empty());
    runtime
        .configure_text(|| Ok(router(&host, &fresh)))
        .unwrap();
    assert!(runtime.models().unload(&registered).is_err());
    assert_eq!(done(&runtime.request_text(request("fresh")).unwrap()), "AB");
    runtime.models().unload(&foreign).unwrap();
    runtime.shutdown().unwrap();
    drop((first, second, registered, foreign, old, fresh, runtime));
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn selective_catalog_retirement_preserves_other_workers_and_shutdown_order() {
    use crate::Runtime;
    use std::time::Instant;
    for race_shutdown in [false, true] {
        let host = HostPool::isolated(256).unwrap();
        let retiring_gate = Arc::new(Gate::default());
        let survivor_gate = Arc::new(Gate::default());
        let retiring = MockModel::new(&host, Some(retiring_gate.clone()));
        let survivor = MockModel::new(&host, Some(survivor_gate.clone()));
        let runtime = Runtime::builder()
            .limits(crate::scheduler::SchedulerLimits {
                max_active: 2,
                ..Default::default()
            })
            .build()
            .unwrap();
        runtime
            .configure_text(|| {
                let a: Arc<dyn LogitModel> = retiring.clone();
                let b: Arc<dyn LogitModel> = survivor.clone();
                Router::new(
                    BTreeMap::from([
                        ("model".into(), a.clone()),
                        ("alias".into(), a),
                        ("other".into(), b),
                    ]),
                    2,
                    1,
                    128,
                    16,
                    host.clone(),
                    Some(16),
                )
            })
            .unwrap();
        let model = runtime.models().resolve("model").unwrap();
        let other = runtime.models().resolve("other").unwrap();
        assert_eq!(
            runtime.models().aliases(&model).unwrap(),
            ["alias", "model"]
        );
        use crate::diagnostics::{ModelLifecycle, ModelMemory, RuntimePhase};
        let initial = runtime.snapshot().unwrap();
        assert_eq!(initial.status.phase, RuntimePhase::Ready);
        assert_eq!(
            (initial.status.loaded_models, initial.status.loaded_owners),
            (3, 2)
        );
        assert_eq!(initial.catalog.len(), 2);
        assert_eq!(initial.catalog[0].aliases, ["alias", "model"]);
        let capabilities = initial.catalog[0].capabilities.as_ref().unwrap();
        assert!(matches!(
            capabilities.details.memory,
            Some(ModelMemory::Host { .. })
        ));
        assert!(
            capabilities.details.execution.is_none(),
            "unknown adapter profile not inferred"
        );
        assert_eq!(initial.scheduler.as_ref().unwrap().config.workers, 2);
        let first = runtime.request_text(request("first target")).unwrap();
        retiring_gate.entered();
        let other_request = || {
            let mut req = request("survivor");
            req.model = "other".into();
            req
        };
        let first_other = runtime.request_text(other_request()).unwrap();
        survivor_gate.entered();
        let mut alias = request("queued target");
        alias.model = "alias".into();
        let second = runtime.request_text(alias).unwrap();
        let cancelled = runtime
            .request_text(request("cancelled queued target"))
            .unwrap();
        cancelled.cancel();
        drop(cancelled);
        let retire = runtime.clone();
        let handle = model.clone();
        let joining = std::thread::spawn(move || retire.models().unload(&handle));
        let deadline = Instant::now() + Duration::from_secs(5);
        while runtime.status().loaded_models != 1 {
            assert!(
                Instant::now() < deadline,
                "selective admission did not close"
            );
            std::thread::yield_now();
        }
        let draining = runtime.snapshot().unwrap();
        assert_eq!(draining.status.phase, RuntimePhase::Retiring);
        assert_eq!(
            (draining.status.loaded_models, draining.status.loaded_owners),
            (1, 1)
        );
        assert_eq!(draining.catalog[0].lifecycle, ModelLifecycle::Draining);
        assert!(draining.catalog[0].admitted_requests >= 2);
        assert!(draining.catalog[0].capabilities.is_none());
        assert_eq!(draining.catalog[1].lifecycle, ModelLifecycle::Ready);
        assert!(draining.catalog[1].capabilities.is_some());
        assert_eq!(runtime.status().execution_adapters, 1);
        assert!(runtime.models().aliases(&model).is_err());
        assert!(runtime.models().resolve("alias").is_err());
        assert_eq!(
            runtime
                .request_text(request("late target"))
                .err()
                .unwrap()
                .code,
            "model_not_found"
        );
        assert!(
            runtime.models().unload(&other).is_err(),
            "one lifecycle operation at a time"
        );
        assert!(
            runtime
                .models()
                .resolve("other")
                .unwrap()
                .shares_source(&other)
        );
        let new_other = runtime.request_text(other_request()).unwrap();
        survivor_gate.release();
        assert_eq!(done(&first_other), "AB");
        assert_eq!(
            done(&second),
            "AB",
            "accepted alias survives admission close"
        );
        assert_eq!(
            done(&new_other),
            "AB",
            "unrelated owner continues during drain"
        );
        assert!(!joining.is_finished(), "target has a live session");

        let shutdown = race_shutdown.then(|| {
            let runtime = runtime.clone();
            std::thread::spawn(move || runtime.shutdown())
        });
        if shutdown.is_some() {
            runtime.wait_for_shutdown_waiters(1);
            assert!(runtime.status().closed);
        }
        retiring_gate.release();
        assert_eq!(done(&first), "AB");
        joining.join().unwrap().unwrap();
        if let Some(shutdown) = shutdown {
            shutdown.join().unwrap().unwrap();
        } else {
            let retired = runtime.models().snapshot().unwrap();
            assert_eq!(retired[0].lifecycle, ModelLifecycle::Retired);
            assert_eq!(retired[0].admitted_requests, 0);
            assert!(retired[0].capabilities.is_none());
            assert_eq!(retired[1].source, other.source());
            assert!(!runtime.status().closed);
            assert_eq!(runtime.models().list().unwrap().len(), 1);
            assert!(runtime.models().unload(&model).is_err());
            assert_eq!(done(&runtime.request_text(other_request()).unwrap()), "AB");
            runtime.models().unload(&other).unwrap();
            assert!(runtime.models().list().unwrap().is_empty());
        }
        runtime.shutdown().unwrap();
        let closed = runtime.snapshot().unwrap();
        assert_eq!(closed.status.phase, RuntimePhase::Closed);
        assert!(closed.scheduler.is_none() && closed.catalog.is_empty());
        drop((
            first,
            second,
            first_other,
            new_other,
            model,
            other,
            runtime,
            retiring,
            survivor,
        ));
        assert_eq!(host.snapshot().unwrap().1, 0);
    }
}

fn language_chain() -> zerogravity_core::pipeline::Config {
    use zerogravity_core::pipeline::{
        Config, ModelInstance, Pipeline, PipelineEdge, PipelineNode, ValueKind,
    };
    let declaration = |id: &str| ModelInstance {
        id: id.into(),
        target: "qwen3.5-2b".into(),
        source: None,
        model_config: None,
        tokenizer: None,
        snapshot: None,
        native_artifacts: None,
    };
    let stage = |id: &str, model: &str| PipelineNode {
        id: id.into(),
        kind: "llm".into(),
        model: Some(model.into()),
    };
    let edge = |from: &str, to: &str| PipelineEdge {
        from: from.into(),
        to: to.into(),
    };
    Config {
        models: vec![declaration("first"), declaration("second")],
        pipelines: vec![Pipeline {
            name: "chain".into(),
            input: ValueKind::Text,
            output: ValueKind::Text,
            realtime: false,
            chat_context: None,
            // Declaration order is deliberately different from edge order.
            steps: vec![stage("finish", "second"), stage("start", "first")],
            links: vec![
                edge("finish.text", "output.value"),
                edge("input.value", "start.text"),
                edge("start.text", "finish.text"),
            ],
        }],
        routes: vec![],
    }
}

#[test]
fn serial_pipeline_uses_exact_registrations_and_original_text_stream() {
    use crate::{Runtime, pipeline::LanguageBinding};
    let host = HostPool::isolated(256).unwrap();
    let model = MockModel::new(&host, None);
    let runtime = Runtime::builder()
        .limits(crate::scheduler::SchedulerLimits {
            max_active: 2,
            ..Default::default()
        })
        .build()
        .unwrap();
    runtime
        .configure_text(|| Ok(router(&host, &model)))
        .unwrap();
    let graph = language_chain();
    let owner = runtime.models().resolve("model").unwrap();
    let other = MockModel::new(&host, None);
    let wrong = crate::model_instance::ModelInstance::new(other.clone());
    let bindings = |second| {
        [
            LanguageBinding {
                model: "first",
                alias: "model",
                owner: &owner,
            },
            LanguageBinding {
                model: "second",
                alias: "alias",
                owner: second,
            },
        ]
    };
    let before = host.snapshot().unwrap().1;
    assert!(
        runtime
            .prepare_pipeline(&graph, "chain", &bindings(&wrong))
            .is_err()
    );
    assert!(
        runtime
            .prepare_pipeline(&graph, "chain", &bindings(&owner)[..1])
            .is_err()
    );
    assert_eq!(host.snapshot().unwrap().1, before);
    assert_eq!(model.next.load(Ordering::Acquire), 0);
    let plan = runtime
        .prepare_pipeline(&graph, "chain", &bindings(&owner))
        .unwrap();
    assert_eq!(plan.name(), "chain");
    assert_eq!(
        plan.stages().iter().map(|s| s.id()).collect::<Vec<_>>(),
        ["start", "finish"]
    );
    assert_eq!(plan.stages()[0].source(), owner.source());
    assert_eq!(plan.stages()[1].source(), owner.source());
    assert_eq!(plan.stages()[1].alias(), "alias");
    assert!(
        runtime
            .run_pipeline(&plan, "x", 17.try_into().unwrap())
            .is_err()
    );
    assert_eq!(model.next.load(Ordering::Acquire), 0);
    let mut stream = runtime
        .run_pipeline(&plan, "x", 2.try_into().unwrap())
        .unwrap();
    let mut text = String::new();
    let mut terminal = 0;
    while let Some(event) = stream.next().unwrap() {
        match event {
            ResponseEvent::Text(delta) => text.push_str(&delta),
            ResponseEvent::Done {
                output,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                assert_eq!(output, "AB");
                assert_eq!(
                    prompt_tokens, 2,
                    "second stage consumes first terminal text"
                );
                assert_eq!(completion_tokens, 2);
                terminal += 1;
            }
            ResponseEvent::Error(error) => panic!("{error}"),
            _ => panic!("unexpected event"),
        }
    }
    assert_eq!(text, "AB", "only final stage deltas");
    assert_eq!(terminal, 1);
    drop(stream);
    assert_eq!(model.next.load(Ordering::Acquire), 2);
    {
        let log = model.log.lock().unwrap();
        assert!(log.contains(&(0, "last", 1)));
        assert!(log.contains(&(1, "last", 2)));
    }
    runtime.models().unload(&owner).unwrap();
    // Re-register even the same exact model Arc: the old plan is stale.
    runtime
        .configure_text(|| Ok(router(&host, &model)))
        .unwrap();
    assert!(matches!(
        runtime.run_pipeline(&plan, "x", 2.try_into().unwrap()),
        Err(crate::Error::State(_))
    ));
    assert_eq!(model.next.load(Ordering::Acquire), 2);
    runtime.shutdown().unwrap();
    drop((plan, owner, wrong, other, runtime, model));
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn pipeline_cancel_and_retirement_never_submit_a_future_stage() {
    use crate::{Runtime, pipeline::LanguageBinding};
    let host = HostPool::isolated(256).unwrap();
    let gate = Arc::new(Gate::default());
    let first = MockModel::new(&host, Some(gate.clone()));
    let second = MockModel::new(&host, None);
    let runtime = Runtime::builder().build().unwrap();
    runtime
        .configure_text(|| {
            let a: Arc<dyn LogitModel> = first.clone();
            let b: Arc<dyn LogitModel> = second.clone();
            Router::new(
                BTreeMap::from([("model".into(), a), ("alias".into(), b)]),
                1,
                1,
                128,
                16,
                host.clone(),
                Some(16),
            )
        })
        .unwrap();
    let a = runtime.models().resolve("model").unwrap();
    let b = runtime.models().resolve("alias").unwrap();
    let plan = runtime
        .prepare_pipeline(
            &language_chain(),
            "chain",
            &[
                LanguageBinding {
                    model: "first",
                    alias: "model",
                    owner: &a,
                },
                LanguageBinding {
                    model: "second",
                    alias: "alias",
                    owner: &b,
                },
            ],
        )
        .unwrap();
    let mut cancelled = runtime
        .run_pipeline(&plan, "x", 2.try_into().unwrap())
        .unwrap();
    gate.entered();
    cancelled.cancel().unwrap();
    assert!(cancelled.next().unwrap().is_none());
    gate.release();
    drop(cancelled);
    assert_eq!(second.next.load(Ordering::Acquire), 0);
    let mut active = runtime
        .run_pipeline(&plan, "y", 2.try_into().unwrap())
        .unwrap();
    // This second owner has no accepted stage yet. Retire it while the
    // predecessor runs or has a terminal event waiting for its consumer.
    runtime.models().unload(&b).unwrap();
    let weak = Arc::downgrade(b.prepared());
    assert_eq!(second.next.load(Ordering::Acquire), 0);
    drop((b, second));
    assert!(
        weak.upgrade().is_none(),
        "prepared pipeline retains no model"
    );
    assert!(matches!(active.next(), Err(crate::Error::State(_))));
    assert!(active.next().unwrap().is_none());
    drop(active);
    assert_eq!(
        done(&runtime.request_text(request("survivor")).unwrap()),
        "AB"
    );
    runtime.shutdown().unwrap();
    drop((plan, a, first, runtime));
    assert_eq!(host.snapshot().unwrap().1, 0);
}
