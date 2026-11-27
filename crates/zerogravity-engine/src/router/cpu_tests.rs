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
            // The two aliases share exact registration timing. The
            // cancelled queued request contributes no invented token or
            // completion; the other owner has its independent samples.
            let measured = retired[0].performance;
            assert_eq!(measured.output_tokens, 4);
            assert_eq!(measured.ttft_samples, 2);
            assert_eq!(measured.decode_intervals, 2);
            assert_eq!(measured.completed_requests, 2);
            assert_eq!(measured.last_completed.unwrap().output_tokens, 2);
            assert_eq!(retired[1].performance.output_tokens, 4);

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
fn declared_pipeline_load_publishes_only_after_exact_admitted_binding() {
    use crate::{Runtime, diagnostics::RuntimePhase};
    let host = HostPool::isolated(256).unwrap();
    let runtime = Runtime::builder()
        .limits(crate::scheduler::SchedulerLimits {
            max_active: 2,
            ..Default::default()
        })
        .build()
        .unwrap();
    let mut graph = language_chain();
    graph.models[0].source = Some("/explicit/first.gguf".into());
    graph.models[1].source = Some("/explicit/second.gguf".into());
    let mut unused = graph.models[0].clone();
    unused.id = "unused".into();
    unused.source = Some("/not-opened/unused.gguf".into());
    graph.models.push(unused);
    assert!(graph.language_chain("chain").is_err());
    let create = |models: &[&zerogravity_core::pipeline::ModelInstance]| {
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert_eq!(runtime.status().phase, RuntimePhase::Loading);
        assert!(runtime.models().list().unwrap().is_empty());
        assert!(runtime.request_text(request("never submitted")).is_err());
        let owner: Arc<dyn LogitModel> = MockModel::new(&host, None);
        Router::new(
            models
                .iter()
                .map(|m| (m.id.clone(), owner.clone()))
                .collect(),
            1,
            1,
            128,
            16,
            host.clone(),
            Some(16),
        )
    };
    let error = runtime
        .configure_language_pipeline(&graph, "chain", |_| {
            Ok(router(&host, &MockModel::new(&host, None)))
        })
        .err()
        .unwrap();
    assert_eq!(error.code, "pipeline_binding");
    assert_eq!(runtime.status().phase, RuntimePhase::Empty);
    assert_eq!(host.snapshot().unwrap().1, 0);

    // The router is fully prepared, but the original host ledger cannot
    // admit its retained pipeline metadata. The private router must retire.
    let mut pressure = None;
    let error = runtime
        .configure_language_pipeline(&graph, "chain", |models| {
            let router = create(models)?;
            let admission = host.transfer_admission("pressure", "pipeline").unwrap();
            let remaining = admission
                .arbiter
                .snapshot(&admission.resource)
                .unwrap()
                .remaining;
            pressure = Some(
                admission
                    .arbiter
                    .admit(
                        &admission.resource,
                        admission.owner,
                        &[zerogravity_core::resource_arbiter::AllocationRequest {
                            kind: AllocationClass::Scratch,
                            bytes: remaining,
                        }],
                    )
                    .unwrap(),
            );
            Ok(router)
        })
        .err()
        .unwrap();
    assert_eq!(error.status, 503);
    assert_eq!(runtime.status().phase, RuntimePhase::Empty);
    drop(pressure);
    assert_eq!(host.snapshot().unwrap().1, 0);

    let plan = runtime
        .configure_language_pipeline(&graph, "chain", create)
        .unwrap();
    assert_eq!(runtime.status().phase, RuntimePhase::Ready);
    assert_eq!(
        (
            runtime.status().loaded_models,
            runtime.status().loaded_owners
        ),
        (2, 1)
    );
    assert_eq!(
        plan.stages().iter().map(|s| s.alias()).collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(plan.stages()[0].source(), plan.stages()[1].source());
    assert!(runtime.models().resolve("unused").is_err());
    assert_eq!(
        runtime
            .configure_language_pipeline(&graph, "chain", |_| panic!("already configured"))
            .err()
            .unwrap()
            .code,
        "runtime_configured"
    );
    let mut output = runtime
        .run_pipeline(&plan, "x", 2.try_into().unwrap())
        .unwrap();
    let mut terminal = false;
    while let Some(event) = output.next().unwrap() {
        if let ResponseEvent::Done {
            output,
            prompt_tokens,
            ..
        } = event
        {
            assert_eq!(output, "AB");
            assert_eq!(prompt_tokens, 2);
            terminal = true;
        }
    }
    assert!(terminal);
    drop(output);
    runtime.shutdown().unwrap();
    drop(plan);
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn serial_pipeline_uses_exact_registrations_and_original_text_stream() {
    use crate::{
        Runtime, TaskTarget, TextControls, TextTask, output::OutputPoll, pipeline::LanguageBinding,
    };
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
    let task = TextTask {
        target: TaskTarget::Pipeline(&plan),
        input: "x",
        controls: TextControls {
            observe_tokens: true,
            ..TextControls::new(2.try_into().unwrap())
        },
    };
    let before = host.snapshot().unwrap();
    let selected = runtime.inspect_task(task).unwrap();
    assert_eq!(selected.pipeline.as_deref(), Some("chain"));
    assert_eq!(selected.controls, task.controls);
    assert_eq!(selected.input_bytes, 1);
    assert_eq!(selected.scheduler.config.output, 16);
    assert_eq!(selected.models.len(), 1);
    assert_eq!(selected.models[0].source, owner.source());
    assert_eq!(selected.models[0].aliases, ["alias", "model"]);
    for (index, stage) in selected.stages.iter().enumerate() {
        assert_eq!(stage.step.as_deref(), Some(plan.stages()[index].id()));
        assert_eq!(stage.alias, plan.stages()[index].alias());
        assert_eq!(stage.registration_id, selected.models[0].registration_id);
    }
    let model_task = TextTask {
        target: TaskTarget::Model("alias"),
        ..task
    };
    let single = runtime.inspect_task(model_task).unwrap();
    assert!(single.pipeline.is_none() && single.stages[0].step.is_none());
    assert_eq!(
        single.stages[0].registration_id,
        selected.stages[1].registration_id
    );
    let bad = TextTask {
        controls: TextControls {
            chat: true,
            ..task.controls
        },
        ..task
    };
    assert_eq!(runtime.submit_task(bad).err().unwrap().code, "unsupported");
    let bad = TextTask {
        target: TaskTarget::Model("missing"),
        ..task
    };
    assert_eq!(
        runtime.inspect_task(bad).err().unwrap().code,
        "model_not_found"
    );
    let bad = TextTask {
        controls: TextControls::new(17.try_into().unwrap()),
        ..task
    };
    assert_eq!(
        runtime.submit_task(bad).err().unwrap().code,
        "limit_exceeded"
    );
    assert_eq!(host.snapshot().unwrap(), before);
    assert_eq!(model.next.load(Ordering::Acquire), 0);

    let mut stream = runtime.submit_task(task).unwrap();
    let mut text = String::new();
    let mut observed = Vec::new();
    let mut terminal = 0;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        assert!(Instant::now() < deadline);
        match stream.poll_next(Duration::from_millis(100)).unwrap() {
            OutputPoll::Pending => continue,
            OutputPoll::Finished => break,
            OutputPoll::Event(event) => match event {
                ResponseEvent::TokenText { text: delta, token } => {
                    assert_eq!(token.index, observed.len());
                    text.push_str(&delta);
                    observed.push(token.id);
                }
                ResponseEvent::Done {
                    output,
                    prompt_tokens,
                    completion_tokens,
                    ..
                } => {
                    assert_eq!(output, "AB");
                    assert_eq!(
                        prompt_tokens, 2,
                        "second stage consumes committed first text"
                    );
                    assert_eq!(completion_tokens, 2);
                    terminal += 1;
                }
                ResponseEvent::Error(error) => panic!("{error}"),
                _ => panic!("unexpected unobserved delta"),
            },
        }
    }
    assert_eq!(observed, [2, 3], "final stage only");
    assert_eq!(text, "AB");
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
    assert_eq!(
        runtime.inspect_task(task).err().unwrap().code,
        "stale_pipeline"
    );
    assert_ne!(
        runtime.inspect_task(model_task).unwrap().stages[0].registration_id,
        selected.stages[0].registration_id
    );
    assert_eq!(model.next.load(Ordering::Acquire), 2);
    runtime.shutdown().unwrap();
    drop((plan, owner, wrong, other, runtime, model));
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn pipeline_cancel_and_retirement_never_submit_a_future_stage() {
    use crate::{
        Runtime, TaskTarget, TextControls, TextTask, output::OutputPoll, pipeline::LanguageBinding,
    };
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
        .submit_task(TextTask {
            target: TaskTarget::Pipeline(&plan),
            input: "x",
            controls: TextControls::new(2.try_into().unwrap()),
        })
        .unwrap();
    gate.entered();
    assert!(matches!(
        cancelled.poll_next(Duration::ZERO).unwrap(),
        OutputPoll::Pending
    ));
    cancelled.cancel().unwrap();
    assert!(matches!(
        cancelled.poll_next(Duration::ZERO).unwrap(),
        OutputPoll::Finished
    ));
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
    let deadline = Instant::now() + Duration::from_secs(5);
    let error = loop {
        assert!(Instant::now() < deadline);
        match active.poll_next(Duration::from_millis(100)) {
            Ok(OutputPoll::Pending) => continue,
            Err(error) => break error,
            Ok(_) => panic!("future retired stage must not emit"),
        }
    };
    assert_eq!(error.code, "stale_pipeline");
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

#[test]
fn legacy_only_model_refuses_structured_conversation_before_enqueue_and_retires_credit() {
    let host = HostPool::isolated(256).unwrap();
    let model = MockModel::new(&host, None);
    let mut router = router(&host, &model);
    let baseline = host.snapshot().unwrap();
    let conversation = SemanticRequest {
        model: "model".into(),
        input: TextInput::Conversation(vec![ChatMessage {
            role: ChatRole::User,
            text: "hello".into(),
        }]),
        max_tokens: 2,
        stream: false,
        token_metrics: None,
    };
    let error = match router.request_semantic(conversation) {
        Ok(_) => panic!("legacy-only tokenizer accepted structured chat"),
        Err(error) => error,
    };
    assert_eq!(error.code, "unsupported_chat");
    assert_eq!(host.snapshot().unwrap(), baseline);
    assert!(model.log.lock().unwrap().is_empty());
    let receipt = router.request_semantic(request("hello").into()).unwrap();
    assert_eq!(done(&receipt), "AB");
    drop(receipt);
    router.shutdown().unwrap();
}

#[test]
fn exact_input_count_is_charged_but_does_not_admit_generation() {
    let host = HostPool::isolated(256).unwrap();
    let model = MockModel::new(&host, None);
    let mut router = router(&host, &model);
    let baseline = host.snapshot().unwrap();
    let tickets = (0..16)
        .map(|_| router.queue.reserve().unwrap())
        .collect::<Vec<_>>();
    let queue = router.queue_snapshot();
    let input = TextInput::Raw("hello".into());
    assert_eq!(
        router.request(request("x")).err().unwrap().code,
        "queue_full"
    );
    // A full generation queue cannot prevent this independent prompt read.
    let count_id = router.next_id.load(Ordering::Relaxed);
    assert_eq!(router.count_input_tokens("alias", &input).unwrap(), 5);
    assert_eq!(router.count_input_tokens("model", &input).unwrap(), 5);
    assert_eq!(router.next_id.load(Ordering::Relaxed), count_id);
    assert_eq!(router.queue_snapshot(), queue);
    assert_eq!(host.snapshot().unwrap(), baseline);
    assert_eq!(model.next.load(Ordering::Acquire), 0);
    assert_eq!(router.catalog_snapshot().unwrap()[0].admitted_requests, 0);
    assert_eq!(router.scheduler_snapshot().counters.attempted, 0);
    assert_eq!(
        router
            .count_input_tokens("missing", &input)
            .unwrap_err()
            .code,
        "model_not_found"
    );
    let conversation = TextInput::Conversation(vec![ChatMessage {
        role: ChatRole::User,
        text: "x".into(),
    }]);
    assert_eq!(
        router
            .count_input_tokens("model", &conversation)
            .unwrap_err()
            .code,
        "unsupported_chat"
    );
    assert_eq!(
        router
            .count_input_tokens("model", &TextInput::Raw("x".repeat(128)))
            .unwrap(),
        128
    );
    assert_eq!(
        router
            .count_input_tokens("model", &TextInput::Raw("x".repeat(129)))
            .unwrap_err()
            .code,
        "context_overflow"
    );

    // Exhaust the actual original physical ledger, not a simulated quota.
    let admission = host.transfer_admission("pressure", "count").unwrap();
    let remaining = admission
        .arbiter
        .snapshot(&admission.resource)
        .unwrap()
        .remaining;
    let pressure = admission
        .arbiter
        .admit(
            &admission.resource,
            admission.owner,
            &[zerogravity_core::resource_arbiter::AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: remaining,
            }],
        )
        .unwrap();
    assert_eq!(
        router
            .count_input_tokens("model", &input)
            .unwrap_err()
            .status,
        503
    );
    assert_eq!(router.catalog_snapshot().unwrap()[0].admitted_requests, 0);
    assert_eq!(router.queue_snapshot(), queue);
    drop((pressure, tickets));
    assert_eq!(host.snapshot().unwrap(), baseline);
    let counted = router.count_input_tokens("model", &input).unwrap();
    let receipt = router.request(request("hello")).unwrap();
    match receipt.events.recv_timeout(Duration::from_secs(5)).unwrap() {
        ResponseEvent::Done { prompt_tokens, .. } => assert_eq!(prompt_tokens, counted),
        ResponseEvent::Error(error) => panic!("{error}"),
        _ => panic!("expected terminal generation"),
    }
    drop(receipt);
    router.shutdown().unwrap();
    drop((router, model));
    assert_eq!(host.snapshot().unwrap().1, 0);
}

// Scheduler-control adapter only. Actual encoder math is covered by the
// source-bound BERT SDK consumer, not by these deterministic gate values.
struct EncoderProbe {
    source: SourceIdentity,
    charge: GlobalResourceLease,
    calls: AtomicUsize,
    first: Option<Arc<Gate>>,
    third: Option<Arc<Gate>>,
}
impl EncoderProbe {
    fn new(host: &HostPool, first: Option<Arc<Gate>>, third: Option<Arc<Gate>>) -> Arc<Self> {
        Arc::new(Self {
            source: SourceIdentity::for_new_allocation().unwrap(),
            charge: host
                .admit("encoder", "source", AllocationClass::ImmutableWeights, 64)
                .unwrap(),
            calls: AtomicUsize::new(0),
            first,
            third,
        })
    }
}
impl EmbeddingEncoder for EncoderProbe {
    fn source(&self) -> SourceIdentity {
        self.source
    }
    fn max_context(&self) -> usize {
        8
    }
    fn vocabulary_size(&self) -> u64 {
        4
    }
    fn dimensions(&self) -> usize {
        2
    }
    fn validate_admission(
        &self,
        admission: &zerogravity_core::resource_arbiter::TransferAdmission,
    ) -> crate::Result<()> {
        self.charge
            .bytes_in_domain(&admission.arbiter, &admission.resource)
            .map(|_| ())
    }
    fn embed_into(
        &self,
        input: &crate::embedding::EmbeddingInput,
        max_tokens: usize,
        output: &mut [f32],
        _: &AtomicBool,
    ) -> crate::Result<usize> {
        let call = self.calls.fetch_add(1, Ordering::AcqRel);
        if call == 0
            && let Some(gate) = &self.first
        {
            gate.worker();
        }
        if call == 2
            && let Some(gate) = &self.third
        {
            gate.worker();
        }
        let crate::embedding::EmbeddingInput::Tokens(ids) = input else {
            return Err(crate::Error::Unsupported(
                "probe supports only explicit IDs".into(),
            ));
        };
        assert!(ids.len() <= max_tokens);
        output.copy_from_slice(&[ids[0] as f32, ids.len() as f32]);
        Ok(ids.len())
    }
}
fn embedding_request(alias: &str, ids: &[u32]) -> crate::embedding::EmbeddingRequest {
    crate::embedding::EmbeddingRequest {
        model: alias.into(),
        inputs: ids
            .iter()
            .map(|&id| crate::embedding::EmbeddingInput::Tokens(vec![id]))
            .collect(),
    }
}
fn mixed_config() -> WorkerConfig {
    WorkerConfig {
        workers: 1,
        active: 2,
        context: 128,
        output: 16,
        max_pending: Some(1),
        policy: WorkPolicy::Sequential,
    }
}

#[test]
fn mixed_encoder_and_generation_share_fifo_slots_and_rounds() {
    let host = HostPool::isolated(256).unwrap();
    let baseline = host.serving_bytes();
    let first = Arc::new(Gate::default());
    let third = Arc::new(Gate::default());
    let text = MockModel::new(&host, None);
    let encoder = EncoderProbe::new(&host, Some(first.clone()), Some(third.clone()));
    let model: Arc<dyn LogitModel> = text.clone();
    let embeddings: Arc<dyn EmbeddingEncoder> = encoder.clone();
    let mut router = Router::with_executor_and_embeddings::<HostLogits>(
        BTreeMap::from([("model".into(), model)]),
        BTreeMap::from([
            ("embed".into(), embeddings.clone()),
            ("embed-alias".into(), embeddings),
        ]),
        host.clone(),
        mixed_config(),
    )
    .unwrap();
    assert_eq!(router.model_ids().collect::<Vec<_>>(), ["model"]);
    let snapshots = router.catalog_snapshot().unwrap();
    assert_eq!(snapshots.len(), 2);
    assert_eq!(snapshots[0].aliases, ["embed", "embed-alias"]);
    assert_eq!(
        snapshots[0]
            .capabilities
            .as_ref()
            .unwrap()
            .embedding_dimensions,
        Some(2)
    );
    let mut wrong = request("hello");
    wrong.model = "embed".into();
    assert_eq!(
        router.request(wrong).err().unwrap().code,
        "unsupported_task"
    );
    assert_eq!(
        router
            .request_embeddings(embedding_request("model", &[1]))
            .err()
            .unwrap()
            .code,
        "unsupported_task"
    );
    let batch = router
        .request_embeddings(embedding_request("embed-alias", &[1, 2, 3]))
        .unwrap();
    first.entered();
    let text_reply = router.request(request("a")).unwrap();
    assert_eq!(text_reply.id, batch.id + 1);
    assert_eq!(router.queue_snapshot().pending, 1);
    assert_eq!(
        router
            .request_embeddings(embedding_request("embed", &[1]))
            .err()
            .unwrap()
            .code,
        "queue_full"
    );
    first.release();
    third.entered();
    assert_eq!(router.active_sessions(), 2);
    assert!(
        text.log
            .lock()
            .unwrap()
            .iter()
            .any(|(_, stage, _)| *stage == "last"),
        "a text quantum must run between the encoder sequences"
    );
    third.release();
    let result = batch
        .events
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(result.values(), [1., 1., 2., 1., 3., 1.]);
    assert_eq!(result.input_tokens(), 3);
    assert_eq!(done(&text_reply), "AB");
    drop((batch, text_reply));
    let counters = router.scheduler_snapshot().counters;
    assert_eq!(counters.embedding_requests, 1);
    assert_eq!(counters.embedding_input_tokens, 3);
    assert_eq!(counters.output_tokens, 2);
    assert_eq!(counters.peak, 2);
    let id = snapshots[0].registration_id;
    router
        .begin_unload_registration("embed", id)
        .unwrap()
        .unwrap()
        .finish()
        .unwrap();
    assert!(router.registration("embed-alias").is_none());
    assert_eq!(
        router
            .request_embeddings(embedding_request("embed", &[1]))
            .err()
            .unwrap()
            .code,
        "model_not_found"
    );
    assert!(router.begin_unload_registration("embed", id).is_err());
    assert_eq!(done(&router.request(request("b")).unwrap()), "AB");
    router.shutdown().unwrap();
    drop((router, encoder, text));
    assert!(host.serving_bytes() > baseline);
    drop(result);
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn encoder_cancel_and_domain_refusal_never_publish_a_partial_batch() {
    let host = HostPool::isolated(256).unwrap();
    let baseline = host.serving_bytes();
    let foreign = HostPool::isolated(256).unwrap();
    let wrong: Arc<dyn EmbeddingEncoder> = EncoderProbe::new(&foreign, None, None);
    assert!(
        Router::with_embeddings(
            BTreeMap::from([("wrong".into(), wrong)]),
            host.clone(),
            mixed_config()
        )
        .is_err()
    );
    assert_eq!(host.serving_bytes(), baseline);
    let first = Arc::new(Gate::default());
    let encoder = EncoderProbe::new(&host, Some(first.clone()), None);
    let model: Arc<dyn EmbeddingEncoder> = encoder.clone();
    let mut router = Router::with_embeddings(
        BTreeMap::from([("embed".into(), model)]),
        host.clone(),
        mixed_config(),
    )
    .unwrap();
    let reply = router
        .request_embeddings(embedding_request("embed", &[1, 2]))
        .unwrap();
    first.entered();
    let queued = router
        .request_embeddings(embedding_request("embed", &[3]))
        .unwrap();
    queued.cancel();
    reply.cancel();
    first.release();
    assert!(reply.events.recv_timeout(Duration::from_secs(5)).is_err());
    assert!(queued.events.recv_timeout(Duration::from_secs(5)).is_err());
    assert_eq!(encoder.calls.load(Ordering::Acquire), 1);
    assert_eq!(router.scheduler_snapshot().counters.embedding_requests, 0);
    drop((reply, queued));
    router.shutdown().unwrap();
    drop((router, encoder));
    assert_eq!(host.serving_bytes(), baseline);
}

use crate::transcription::{
    SpeechRecognizer, SpeechSession, TranscriptionFinish, TranscriptionRequest, TranscriptionUsage,
};

struct SpeechProbe {
    source: SourceIdentity,
    memory: GlobalResourceLease,
    start: Arc<Gate>,
    terminal: Arc<Gate>,
}
impl SpeechProbe {
    fn new(host: &HostPool, start: Arc<Gate>, terminal: Arc<Gate>) -> Arc<Self> {
        Arc::new(Self {
            source: SourceIdentity::for_new_allocation().unwrap(),
            memory: host
                .admit("speech", "source", AllocationClass::ImmutableWeights, 64)
                .unwrap(),
            start,
            terminal,
        })
    }
}
impl SpeechRecognizer for SpeechProbe {
    fn source(&self) -> SourceIdentity {
        self.source
    }
    fn max_samples(&self) -> usize {
        16000
    }
    fn max_context(&self) -> usize {
        128
    }
    fn vocabulary_size(&self) -> u64 {
        4
    }
    fn max_output_tokens(&self) -> usize {
        8
    }
    fn max_output_bytes(&self) -> usize {
        128
    }
    fn validate_admission(
        &self,
        a: &zerogravity_core::resource_arbiter::TransferAdmission,
    ) -> crate::Result<()> {
        self.memory
            .bytes_in_domain(&a.arbiter, &a.resource)
            .map(|_| ())
    }
    fn start(
        self: Arc<Self>,
        _: &[f32],
        _: u32,
        _: usize,
        _: usize,
        _: &AtomicBool,
    ) -> crate::Result<Box<dyn SpeechSession>> {
        self.start.worker();
        Ok(Box::new(SpeechProbeSession {
            model: self,
            steps: 0,
        }))
    }
}
struct SpeechProbeSession {
    model: Arc<SpeechProbe>,
    steps: usize,
}
impl SpeechSession for SpeechProbeSession {
    fn step(
        &mut self,
        out: &mut String,
        _: &AtomicBool,
    ) -> crate::Result<Option<TranscriptionUsage>> {
        self.steps += 1;
        if self.steps < 2 {
            return Ok(None);
        }
        self.model.terminal.worker();
        out.push_str("probe");
        Ok(Some(TranscriptionUsage {
            input_tokens: 5,
            output_tokens: 2,
            finish: TranscriptionFinish::Stop,
        }))
    }
}
fn speech_request() -> TranscriptionRequest {
    TranscriptionRequest {
        model: "speech-alias".into(),
        samples: vec![0.; 320],
        sample_rate: 16000,
        max_tokens: 2,
    }
}

#[test]
fn speech_uses_original_mixed_worker_fifo_and_retirement() {
    let host = HostPool::isolated(256).unwrap();
    let baseline = host.serving_bytes();
    let first = Arc::new(Gate::default());
    let terminal = Arc::new(Gate::default());
    let speech: Arc<dyn SpeechRecognizer> =
        SpeechProbe::new(&host, first.clone(), terminal.clone());
    let text = MockModel::new(&host, None);
    let encoder: Arc<dyn EmbeddingEncoder> = EncoderProbe::new(&host, None, None);
    let mut router = Router::with_executor_and_finite_tasks::<HostLogits>(
        BTreeMap::from([("model".into(), text.clone() as Arc<dyn LogitModel>)]),
        BTreeMap::from([("embed".into(), encoder)]),
        BTreeMap::from([
            ("speech".into(), speech.clone()),
            ("speech-alias".into(), speech),
        ]),
        host.clone(),
        mixed_config(),
    )
    .unwrap();
    assert_eq!(router.serving_model_ids().collect::<Vec<_>>(), ["model"]);
    let speech_snapshot = router
        .catalog_snapshot()
        .unwrap()
        .into_iter()
        .find(|s| s.aliases.contains(&"speech".to_owned()))
        .unwrap();
    assert!(speech_snapshot.capabilities.as_ref().unwrap().transcription);
    let mut wrong = speech_request();
    wrong.model = "embed".into();
    assert_eq!(
        router.request_transcription(wrong).err().unwrap().code,
        "unsupported_task"
    );
    let mut wrong = speech_request();
    wrong.sample_rate = 48000;
    assert_eq!(
        router.request_transcription(wrong).err().unwrap().code,
        "invalid_input"
    );
    let speech = router.request_transcription(speech_request()).unwrap();
    first.entered();
    let generation = router.request(request("a")).unwrap();
    assert_eq!(generation.id, speech.id + 1);
    assert_eq!(router.queue_snapshot().pending, 1);
    assert_eq!(
        router
            .request_embeddings(embedding_request("embed", &[1]))
            .err()
            .unwrap()
            .code,
        "queue_full"
    );
    first.release();
    terminal.entered();
    assert!(
        text.log
            .lock()
            .unwrap()
            .iter()
            .any(|(_, stage, _)| *stage == "last")
    );
    let embedding = router
        .request_embeddings(embedding_request("embed", &[2]))
        .unwrap();
    terminal.release();
    let result = speech
        .events
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(result.text(), "probe");
    assert_eq!(
        result.usage(),
        TranscriptionUsage {
            input_tokens: 5,
            output_tokens: 2,
            finish: TranscriptionFinish::Stop
        }
    );
    assert_eq!(done(&generation), "AB");
    let vectors = embedding
        .events
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(vectors.values(), [2., 1.]);
    drop((speech, generation, embedding, vectors));
    let stats = router.scheduler_snapshot().counters;
    assert_eq!(stats.transcription_requests, 1);
    assert_eq!(stats.transcription_input_tokens, 5);
    assert_eq!(stats.transcription_output_tokens, 2);
    assert_eq!(stats.embedding_requests, 1);
    assert_eq!(stats.peak, 2);
    router
        .begin_unload_registration("speech-alias", speech_snapshot.registration_id)
        .unwrap()
        .unwrap()
        .finish()
        .unwrap();
    assert!(router.registration("speech").is_none());
    router.shutdown().unwrap();
    drop((router, text));
    assert!(host.serving_bytes() > baseline);
    drop(result);
    assert_eq!(host.serving_bytes(), baseline);
}

#[test]
fn speech_cancelled_terminal_and_foreign_domain_refuse() {
    let host = HostPool::isolated(256).unwrap();
    let foreign = HostPool::isolated(256).unwrap();
    let baseline = host.serving_bytes();
    let start = Arc::new(Gate::default());
    let terminal = Arc::new(Gate::default());
    let wrong: Arc<dyn SpeechRecognizer> =
        SpeechProbe::new(&foreign, start.clone(), terminal.clone());
    assert!(
        Router::with_transcription(
            BTreeMap::from([("wrong".into(), wrong)]),
            host.clone(),
            mixed_config()
        )
        .is_err()
    );
    assert_eq!(host.serving_bytes(), baseline);
    let speech: Arc<dyn SpeechRecognizer> =
        SpeechProbe::new(&host, start.clone(), terminal.clone());
    let mut router = Router::with_transcription(
        BTreeMap::from([("speech-alias".into(), speech)]),
        host.clone(),
        mixed_config(),
    )
    .unwrap();
    let reply = router.request_transcription(speech_request()).unwrap();
    start.entered();
    start.release();
    terminal.entered();
    reply.cancel();
    terminal.release();
    assert!(reply.events.recv_timeout(Duration::from_secs(5)).is_err());
    assert_eq!(
        router.scheduler_snapshot().counters.transcription_requests,
        0
    );
    drop(reply);
    router.shutdown().unwrap();
    drop(router);
    assert_eq!(host.serving_bytes(), baseline);
}
