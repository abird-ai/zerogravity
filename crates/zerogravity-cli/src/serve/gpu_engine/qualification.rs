//! Opt-in release CUDA serving proof over original source/session owners.
use super::*;
use zerogravity::accelerator::program::StateSlot;
use zerogravity::gpu::ResidentSnapshot;
use zerogravity::qwen35::QwenDenseCpu;
use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter};

type Domains = Vec<(StateSlot, Vec<f32>)>;

fn exact(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "source bits at {index}"
        );
    }
}

fn same_domains(expected: &Domains, actual: &Domains) {
    assert_eq!(expected.len(), 42);
    assert_eq!(actual.len(), 42);
    let expected = expected
        .iter()
        .map(|(slot, values)| (*slot, values))
        .collect::<BTreeMap<_, _>>();
    let actual = actual
        .iter()
        .map(|(slot, values)| (*slot, values))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(expected.len(), 42);
    assert_eq!(actual.len(), 42);
    for (slot, expected) in expected {
        exact(
            expected,
            actual.get(&slot).expect("missing physical state slot"),
        );
    }
}

fn best(values: &[f32]) -> (u32, u32) {
    assert_eq!(values.len(), 248_320);
    assert!(values.iter().all(|value| value.is_finite()));
    let index = (0..values.len())
        .max_by(|a, b| values[*a].total_cmp(&values[*b]).then_with(|| b.cmp(a)))
        .unwrap();
    (index as u32, values[index].to_bits())
}

struct Fixture {
    model: Arc<GpuModel>,
    source: Arc<LoadedQwenModel>,
    stats: Arc<WaveStats>,
    key: PhysicalDeviceKey,
}
impl Fixture {
    fn new(name: &str) -> Self {
        Self::with_native_rows(name, 2)
    }

    fn with_native_rows(name: &str, native_rows: usize) -> Self {
        assert!(matches!(native_rows, 2 | 4 | 8));
        if cfg!(debug_assertions) {
            panic!("physical serving qualification requires --release");
        }
        preflight_cuda_toolkit(GpuBackend::Cuda).unwrap();
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/models")
            .join(name);
        let source = LoadedQwenModel::open(
            &directory.join("model.gguf"),
            &directory.join("config.json"),
        )
        .unwrap();
        let tokenizer =
            Arc::new(QwenTokenizer::from_file(&directory.join("tokenizer.json")).unwrap());
        let device = GpuBackend::Cuda.device().unwrap();
        let client = device.client();
        let properties = client.properties();
        let pci = properties
            .identity
            .physical
            .as_ref()
            .and_then(|identity| identity.pci_address)
            .expect("qualified NVIDIA PCI identity")
            .to_string();
        let key = PhysicalDeviceKey::new("pci", &pci).unwrap();
        if let Ok(prior) = ResourceArbiter::process_shared().snapshot(&key) {
            assert_eq!(prior.reserved, 0, "physical tests must run sequentially");
            assert!(!prior.fenced);
        }
        let plan = source.forward_plan(GPU_CAPACITY, true, true).unwrap();
        let resident = block_on(ResidentModel::load(
            device,
            source.clone(),
            plan,
            4096 * 1024 * 1024,
            5120 * 1024 * 1024,
        ))
        .unwrap();
        let model = Arc::new(GpuModel {
            engine: ModelEngine::Gguf(resident),
            tokenizer: TextTokenizer::Shared(tokenizer),
            group: 0,
            hidden_width: source.geometry().hidden_size as usize,
            vocabulary_size: source.geometry().vocab_size,
            native_rows,
        });
        Self {
            model,
            source,
            stats: Arc::new(WaveStats::default()),
            key,
        }
    }

    fn lane(&self, input: &str, maximum: usize, capacity: usize) -> (GpuActive, Trace) {
        self.lane_configured(input, maximum, capacity, false)
    }

    fn lane_configured(
        &self,
        input: &str,
        maximum: usize,
        capacity: usize,
        observations: bool,
    ) -> (GpuActive, Trace) {
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let cancelled = Arc::new(AtomicBool::new(false));
        let lane = GpuActive::new(
            Scheduled {
                request: Request {
                    model: "qualified".into(),
                    input: input.into(),
                    chat: false,
                    max_tokens: maximum,
                    stream: true,
                    token_metrics: observations.then(|| {
                        super::super::engine::TokenMetricsRequest {
                            started: Instant::now(),
                            wire_request_id: Some("worker:unequal".into()),
                        }
                    }),
                },
                events: sender,
                cancelled,
            },
            Arc::clone(&self.model),
            Arc::clone(&self.stats),
        )
        .unwrap();
        (
            lane,
            Trace {
                receiver,
                text: String::new(),
                done: Vec::new(),
                tokens: observations.then(Vec::new),
            },
        )
    }

    fn close(self) {
        assert_eq!(self.stats.active.load(Ordering::Acquire), 0);
        assert_eq!(self.stats.failed.load(Ordering::Acquire), 0);
        let charged = ResourceArbiter::process_shared()
            .snapshot(&self.key)
            .unwrap();
        assert_eq!(charged.reserved, 4096 * 1024 * 1024);
        assert!(!charged.fenced);
        let key = self.key.clone();
        drop(self);
        let budget = ResourceArbiter::process_shared().snapshot(&key).unwrap();
        assert_eq!(budget.reserved, 0);
        assert!(!budget.fenced);
    }
}

struct Trace {
    receiver: mpsc::Receiver<ResponseEvent>,
    text: String,
    done: Vec<(String, usize, usize, &'static str)>,
    tokens: Option<Vec<super::super::engine::TokenObservation>>,
}
impl Trace {
    fn drain(&mut self) {
        while let Ok(event) = self.receiver.try_recv() {
            self.accept(event);
        }
    }
    fn one_credit(&mut self) {
        self.accept(self.receiver.recv_timeout(Duration::from_secs(1)).unwrap());
    }
    fn accept(&mut self, event: ResponseEvent) {
        match event {
            ResponseEvent::TokenText { text, token } => {
                assert!(self.done.is_empty(), "token after terminal event");
                let tokens = self
                    .tokens
                    .as_mut()
                    .expect("unrequested token observations in default profile");
                assert_eq!(token.index, tokens.len());
                tokens.push(token);
                self.text.push_str(&text);
            }
            ResponseEvent::Text(text) => {
                assert!(self.done.is_empty(), "text after terminal event");
                self.text.push_str(&text);
            }
            ResponseEvent::Done {
                output,
                prompt_tokens,
                completion_tokens,
                finish_reason,
            } => {
                self.done
                    .push((output, prompt_tokens, completion_tokens, finish_reason));
            }
            ResponseEvent::Error(error) => panic!("physical GPU worker error: {error:?}"),
        }
    }
    fn verify(&mut self, reference: &Reference, tokenizer: &QwenTokenizer, maximum: usize) {
        self.drain();
        let selected = reference
            .sequence
            .iter()
            .map(|&(id, _)| id)
            .take(maximum)
            .take_while(|id| *id != tokenizer.eos())
            .collect::<Vec<_>>();
        let stopped = selected.len() < maximum;
        assert_eq!(
            reference.sequence.len(),
            selected.len() + usize::from(stopped),
            "extra model forward after the terminal token"
        );
        if stopped {
            assert_eq!(reference.sequence[selected.len()].0, tokenizer.eos());
        }
        if let Some(tokens) = &self.tokens {
            assert_eq!(tokens.len(), selected.len());
            for (index, (token, expected)) in tokens.iter().zip(&selected).enumerate() {
                assert_eq!(token.index, index);
                assert_eq!(token.id, *expected);
                let gpu = token.gpu.expect("actual GPU producer binding");
                assert_eq!(
                    gpu.producer_frontier,
                    (reference.prompt_tokens + index) as u64
                );
                if index > 0 {
                    assert!(token.commit_observed_at >= tokens[index - 1].commit_observed_at);
                }
            }
        }
        let expected = tokenizer.decode(&selected, true).unwrap();
        assert_eq!(self.text, expected, "duplicate/missing streamed text");
        assert_eq!(
            self.done,
            vec![(
                expected,
                reference.prompt_tokens,
                selected.len(),
                if stopped { "stop" } else { "length" }
            )]
        );
    }
}

struct Reference {
    cpu: QwenDenseCpu,
    gpu: ResidentSession,
    packet: ResidentToken,
    sequence: Vec<(u32, u32)>,
    prompt_tokens: usize,
}
impl Reference {
    fn new(fixture: &Fixture, lane: &GpuActive) -> Self {
        assert_eq!(lane.cursor, lane.prompt.len());
        let mut cpu = QwenDenseCpu::from_shared(Arc::clone(&fixture.source)).unwrap();
        let selected = best(&cpu.prefill_transaction(&lane.prompt).unwrap());
        let mut gpu = block_on(
            fixture
                .model
                .engine
                .legacy_model()
                .expect("legacy qualification")
                .session(),
        )
        .unwrap();
        let read = block_on(gpu.forward_greedy(&lane.prompt)).unwrap();
        assert!(read.logits.is_empty());
        assert_eq!(read.bytes, 16);
        let packet = read.device_token.unwrap();
        assert_eq!((packet.id, packet.max_bits), selected);
        let reference = Self {
            cpu,
            gpu,
            packet,
            sequence: vec![selected],
            prompt_tokens: lane.prompt.len(),
        };
        reference.check(lane);
        reference
    }
    fn sync(&mut self, lane: &GpuActive) {
        let position = lane.engine.binding().frontier() as usize;
        while self.cpu.position() < position {
            let selected = best(&self.cpu.forward(self.packet.id).unwrap());
            let read = block_on(self.gpu.forward_device_token(&self.packet)).unwrap();
            assert!(read.logits.is_empty());
            assert_eq!(read.bytes, 16);
            self.packet = read.device_token.unwrap();
            assert_eq!((self.packet.id, self.packet.max_bits), selected);
            self.sequence.push(selected);
        }
        assert_eq!(self.cpu.position(), position);
        self.check(lane);
    }
    fn check(&self, lane: &GpuActive) {
        let packet = lane.device_token.as_ref().expect("live worker packet");
        assert_eq!((packet.id, packet.max_bits), *self.sequence.last().unwrap());
        let expected = self.cpu.snapshot().domain_values();
        same_domains(
            &expected,
            &block_on(
                lane.engine
                    .legacy()
                    .expect("legacy qualification")
                    .snapshot()
                    .read_domains(),
            )
            .unwrap(),
        );
        same_domains(
            &expected,
            &block_on(self.gpu.snapshot().read_domains()).unwrap(),
        );
    }
    fn snapshot(&self, snapshot: &ResidentSnapshot) {
        same_domains(
            &self.cpu.snapshot().domain_values(),
            &block_on(snapshot.read_domains()).unwrap(),
        );
    }
}

fn b1_prompt(lane: &mut GpuActive, stats: &WaveStats) {
    while lane.cursor < lane.prompt.len() {
        assert!(matches!(lane.step(stats).unwrap(), MemberStep::Forwarded));
    }
    assert!(lane.device_token.is_some());
}

/// The actual worker eligibility/wave/helper boundaries, without a detached thread.
fn round(active: &mut VecDeque<GpuActive>, stats: &WaveStats) {
    let mut parked = VecDeque::new();
    let mut eligible = VecDeque::new();
    for lane in active.drain(..) {
        if let Some(lane) = eligible_member(lane, &mut parked, stats) {
            eligible.push_back(lane);
        }
    }
    if eligible.is_empty() {
        active.extend(parked);
        return;
    }
    let native_rows = eligible[0].native_rows;
    assert!(eligible.iter().all(|lane| lane.native_rows == native_rows));
    let ready = eligible.iter().map(GpuActive::ready).collect::<Vec<_>>();
    let waves = gpu_batch::form_decode(
        &ready,
        Limits {
            max_members: native_rows,
            max_tokens: 64,
            max_scratch: 256 * 1024 * 1024,
        },
    )
    .unwrap();
    let mut pending = eligible.into_iter().map(Some).collect::<Vec<_>>();
    for wave in waves {
        let members = wave
            .indices
            .into_iter()
            .map(|index| pending[index].take().unwrap())
            .collect::<Vec<_>>();
        if members.iter().all(|lane| lane.cursor == lane.prompt.len()) {
            advance_decode_cohort(members, active, stats, native_rows);
        } else if members.len() > 1
            && members
                .iter()
                .all(|lane| lane.cursor + 1 == lane.prompt.len())
        {
            finish_prompt_cohort(members, active, stats, native_rows);
        } else {
            for lane in members {
                advance_member(lane, active, stats);
            }
        }
    }
    active.extend(parked);
}

fn update(active: &VecDeque<GpuActive>, references: &mut BTreeMap<u64, Reference>) {
    for lane in active {
        if lane.cursor == lane.prompt.len() {
            let id = lane.engine.binding().session().get();
            if let Some(reference) = references.get_mut(&id) {
                reference.sync(lane);
            }
        }
    }
}

fn finish(
    active: &mut VecDeque<GpuActive>,
    fixture: &Fixture,
    references: &mut BTreeMap<u64, Reference>,
    traces: &mut [&mut Trace],
) {
    for _ in 0..64 {
        if active.is_empty() {
            break;
        }
        for trace in traces.iter_mut() {
            trace.drain();
        }
        round(active, &fixture.stats);
        update(active, references);
    }
    assert!(active.is_empty(), "bounded worker did not finish");
}

fn handoff_and_lengths(fixture: &Fixture) {
    let (mut left, mut left_trace) = fixture.lane("a", 2, EVENTS_PER_REQUEST);
    let (mut right, mut right_trace) = fixture.lane("b", 5, EVENTS_PER_REQUEST);
    b1_prompt(&mut left, &fixture.stats);
    b1_prompt(&mut right, &fixture.stats);
    let left_id = left.engine.binding().session().get();
    let right_id = right.engine.binding().session().get();
    let mut references = BTreeMap::from([
        (left_id, Reference::new(fixture, &left)),
        (right_id, Reference::new(fixture, &right)),
    ]);
    let fused_before = fixture.stats.fused_batches.load(Ordering::Acquire);
    let model_tokens = fixture.stats.model_tokens.load(Ordering::Acquire);
    let committed = fixture.stats.committed_roots.load(Ordering::Acquire);
    let sampler = fixture.stats.native_sampler.load(Ordering::Acquire);
    let mut active = VecDeque::from([left, right]);
    round(&mut active, &fixture.stats);
    update(&active, &mut references);
    assert!(
        fixture.stats.fused_batches.load(Ordering::Acquire) > fused_before,
        "B1-produced packets were not consumed by a native model cohort"
    );
    assert_eq!(
        fixture.stats.model_tokens.load(Ordering::Acquire),
        model_tokens + 2
    );
    assert_eq!(
        fixture.stats.committed_roots.load(Ordering::Acquire),
        committed + 2
    );
    assert_eq!(
        fixture.stats.native_sampler.load(Ordering::Acquire),
        sampler + 4
    );
    assert!(
        active
            .iter()
            .all(|lane| lane.engine.binding().frontier() == 2)
    );
    finish(
        &mut active,
        fixture,
        &mut references,
        &mut [&mut left_trace, &mut right_trace],
    );
    left_trace.verify(&references[&left_id], &fixture.model.tokenizer, 2);
    right_trace.verify(&references[&right_id], &fixture.model.tokenizer, 5);
    assert!(
        references[&right_id].sequence.len() > references[&left_id].sequence.len(),
        "fixture did not exercise unequal generation lengths/B1 survivor"
    );
    println!(
        "P12_B5_B1_NATIVE_B1_HANDOFF_PASS full_packets_and_all42domains=exact \
        unequal_lengths=2,5 event_text_usage_Done=exact"
    );
}

fn cancel_before_feedback(fixture: &Fixture) {
    let (mut left, mut left_trace) = fixture.lane("a", 4, EVENTS_PER_REQUEST);
    let (mut right, mut right_trace) = fixture.lane("b", 3, EVENTS_PER_REQUEST);
    b1_prompt(&mut left, &fixture.stats);
    b1_prompt(&mut right, &fixture.stats);
    let saved = left
        .engine
        .legacy()
        .expect("legacy qualification")
        .snapshot();
    let left_reference = Reference::new(fixture, &left);
    let right_id = right.engine.binding().session().get();
    let mut references = BTreeMap::from([(right_id, Reference::new(fixture, &right))]);
    left.cancelled.store(true, Ordering::Release);
    let fused = fixture.stats.fused_batches.load(Ordering::Acquire);
    let cancelled = fixture.stats.cancelled.load(Ordering::Acquire);
    let mut active = VecDeque::from([left, right]);
    round(&mut active, &fixture.stats);
    update(&active, &mut references);
    assert_eq!(fixture.stats.fused_batches.load(Ordering::Acquire), fused);
    assert_eq!(
        fixture.stats.cancelled.load(Ordering::Acquire),
        cancelled + 1
    );
    left_reference.snapshot(&saved);
    left_trace.drain();
    assert!(left_trace.text.is_empty() && left_trace.done.is_empty());
    finish(
        &mut active,
        fixture,
        &mut references,
        &mut [&mut right_trace],
    );
    right_trace.verify(&references[&right_id], &fixture.model.tokenizer, 3);
    println!("P12_B5_CANCEL_BEFORE_FEEDBACK_PASS cancelled_prior=exact survivor_B1=exact");
}

fn cancel_after_ready(fixture: &Fixture) {
    let (mut left, mut left_trace) = fixture.lane("a", 4, EVENTS_PER_REQUEST);
    let (mut right, mut right_trace) = fixture.lane("b", 4, EVENTS_PER_REQUEST);
    b1_prompt(&mut left, &fixture.stats);
    b1_prompt(&mut right, &fixture.stats);
    let saved = left
        .engine
        .legacy()
        .expect("legacy qualification")
        .snapshot();
    let left_reference = Reference::new(fixture, &left);
    let right_id = right.engine.binding().session().get();
    let initial_right = right.engine.binding();
    let mut references = BTreeMap::from([(right_id, Reference::new(fixture, &right))]);
    let cancel = Arc::clone(&left.cancelled);
    AFTER_COHORT_READY.with(|hook| {
        assert!(hook.borrow().is_none());
        *hook.borrow_mut() = Some(Box::new(move || cancel.store(true, Ordering::Release)));
    });
    let discarded = fixture.stats.discarded_cohorts.load(Ordering::Acquire);
    let committed = fixture.stats.committed_roots.load(Ordering::Acquire);
    let model_tokens = fixture.stats.model_tokens.load(Ordering::Acquire);
    let sampler = fixture.stats.native_sampler.load(Ordering::Acquire);
    let mut active = VecDeque::from([left, right]);
    round(&mut active, &fixture.stats);
    AFTER_COHORT_READY
        .with(|hook| assert!(hook.borrow().is_none(), "actual Ready hook not reached"));
    assert_eq!(
        fixture.stats.discarded_cohorts.load(Ordering::Acquire),
        discarded + 1
    );
    assert_eq!(active.len(), 1);
    let survivor = &active[0];
    assert_eq!(
        survivor.engine.binding().root(),
        initial_right.root() + 1,
        "discarded native proposal published before B1 retry"
    );
    assert_eq!(
        survivor.engine.binding().frontier(),
        initial_right.frontier() + 1
    );
    assert_eq!(
        fixture.stats.committed_roots.load(Ordering::Acquire),
        committed + 1
    );
    assert_eq!(
        fixture.stats.model_tokens.load(Ordering::Acquire),
        model_tokens + 3,
        "two discarded physical rows plus one committed B1 retry"
    );
    assert_eq!(
        fixture.stats.native_sampler.load(Ordering::Acquire),
        sampler + 4
    );
    update(&active, &mut references);
    left_reference.snapshot(&saved);
    finish(
        &mut active,
        fixture,
        &mut references,
        &mut [&mut right_trace],
    );
    right_trace.verify(&references[&right_id], &fixture.model.tokenizer, 4);
    left_trace.drain();
    assert!(left_trace.done.is_empty());
    println!(
        "P12_B5_CANCEL_AFTER_READY_PASS discarded_whole_cohort=true \
        actual_original_prior=exact survivor_retry_single_root=true text_usage_Done=no_duplicates"
    );
}

fn credit_park(fixture: &Fixture) {
    let (mut left, mut left_trace) = fixture.lane("a", 8, 1);
    let (mut right, mut right_trace) = fixture.lane("b", 8, EVENTS_PER_REQUEST);
    b1_prompt(&mut left, &fixture.stats);
    b1_prompt(&mut right, &fixture.stats);
    let left_id = left.engine.binding().session().get();
    let right_id = right.engine.binding().session().get();
    let mut references = BTreeMap::from([
        (left_id, Reference::new(fixture, &left)),
        (right_id, Reference::new(fixture, &right)),
    ]);
    let mut active = VecDeque::from([left, right]);
    for _ in 0..12 {
        round(&mut active, &fixture.stats);
        update(&active, &mut references);
        right_trace.drain();
        if active.iter().any(|lane| {
            lane.engine.binding().session().get() == left_id && lane.blocked_since.is_some()
        }) {
            break;
        }
    }
    let parked = active
        .iter()
        .find(|lane| lane.engine.binding().session().get() == left_id)
        .expect("slow source request finished without a credit boundary");
    assert!(parked.blocked_since.is_some());
    assert!(parked.packet_presented);
    let binding = parked.engine.binding();
    let packet = parked
        .device_token
        .as_ref()
        .map(|token| (token.id, token.max_bits));
    for _ in 0..3 {
        round(&mut active, &fixture.stats);
        update(&active, &mut references);
        right_trace.drain();
        let parked = active
            .iter()
            .find(|lane| lane.engine.binding().session().get() == left_id)
            .unwrap();
        assert_eq!(parked.engine.binding(), binding);
        assert_eq!(
            parked
                .device_token
                .as_ref()
                .map(|token| (token.id, token.max_bits)),
            packet
        );
    }
    left_trace.one_credit();
    finish(
        &mut active,
        fixture,
        &mut references,
        &mut [&mut left_trace, &mut right_trace],
    );
    left_trace.verify(&references[&left_id], &fixture.model.tokenizer, 8);
    right_trace.verify(&references[&right_id], &fixture.model.tokenizer, 8);
    println!(
        "P12_B5_REAL_CREDIT_PARK_PASS original_bounded_channel=1 \
        parked_packet_root_unchanged=true recovered_text_usage_Done=exact"
    );
}

fn find_source_eos(fixture: &Fixture) -> Option<String> {
    let prefixes = [
        "<|im_start|>user\nSay only OK.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nOK",
        "<|im_start|>user\nWhat is 1+1? Give only the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n2",
        "<|im_start|>user\nReply with nothing.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>user\nSay only Done.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nDone",
    ];
    for input in prefixes {
        let tokens = fixture.model.tokenizer.prompt_tokens(input, false).unwrap();
        let mut cpu = QwenDenseCpu::from_shared(Arc::clone(&fixture.source)).unwrap();
        let mut logits = cpu.prefill_transaction(&tokens).unwrap();
        for _ in 0..8 {
            let id = best(&logits).0;
            if id == fixture.model.tokenizer.eos() {
                return Some(input.into());
            }
            logits = cpu.forward(id).unwrap();
        }
    }
    None
}

fn source_eos(fixture: &Fixture) {
    let input = find_source_eos(fixture).expect(
        "P12_B5_TRAINED_EOS_NOT_QUALIFIED: no actual original CPU-selected EOS \
        in four source-real prefixes with at most eight greedy selections each",
    );
    let length = fixture
        .model
        .tokenizer
        .prompt_tokens(&input, false)
        .unwrap()
        .len();
    let sibling = " a".repeat(length);
    assert_eq!(
        fixture
            .model
            .tokenizer
            .prompt_tokens(&sibling, false)
            .unwrap()
            .len(),
        length
    );
    let (left, mut left_trace) = fixture.lane(&input, 9, EVENTS_PER_REQUEST);
    let (right, mut right_trace) = fixture.lane(&sibling, 12, EVENTS_PER_REQUEST);
    let left_id = left.engine.binding().session().get();
    let right_id = right.engine.binding().session().get();
    let mut active = VecDeque::from([left, right]);
    let fused_before = fixture.stats.fused_batches.load(Ordering::Acquire);
    for _ in 0..32 {
        if active.iter().all(|lane| lane.cursor == lane.prompt.len()) {
            break;
        }
        round(&mut active, &fixture.stats);
    }
    assert_eq!(active.len(), 2);
    assert!(active.iter().all(|lane| lane.cursor == lane.prompt.len()));
    assert_eq!(
        fixture.stats.fused_batches.load(Ordering::Acquire),
        fused_before + 1,
        "actual final prompt tokens did not execute the native model cohort"
    );
    let mut references = active
        .iter()
        .map(|lane| {
            (
                lane.engine.binding().session().get(),
                Reference::new(fixture, lane),
            )
        })
        .collect::<BTreeMap<_, _>>();
    finish(
        &mut active,
        fixture,
        &mut references,
        &mut [&mut left_trace, &mut right_trace],
    );
    left_trace.verify(&references[&left_id], &fixture.model.tokenizer, 9);
    right_trace.verify(&references[&right_id], &fixture.model.tokenizer, 12);
    assert_eq!(left_trace.done[0].3, "stop");
    println!(
        "P12_B5_TRAINED_EOS_GPU_PASS actual_original_source_selected_EOS=true \
        CPU_B1_packets_all42=exact stop_excludes_EOS_usage=true sibling_independent=true"
    );
}

fn qualify(name: &str) {
    let fixture = Fixture::new(name);
    handoff_and_lengths(&fixture);
    cancel_before_feedback(&fixture);
    cancel_after_ready(&fixture);
    credit_park(&fixture);
    source_eos(&fixture);
    assert!(fixture.stats.kernel_dispatches.load(Ordering::Acquire) > 0);
    assert!(fixture.stats.native_projections.load(Ordering::Acquire) > 0);
    assert!(fixture.stats.native_sampler.load(Ordering::Acquire) > 0);
    fixture.close();
    println!(
        "P12_B5_NATIVE_WORKER_PHYSICAL_PASS source={name} final_active=0 final_reserved=0 \
        original_GPU_authorities_CPU_B1_state=exact actual_HTTP_wire=separate_gate vLLM_performance=not_qualified"
    );
}

struct Observed {
    id: u64,
    maximum: usize,
    trace: Trace,
}

fn observed_lane(
    fixture: &Fixture,
    input: &str,
    maximum: usize,
    capacity: usize,
    b1: bool,
) -> (GpuActive, Observed) {
    let (mut lane, trace) = fixture.lane(input, maximum, capacity);
    if b1 {
        b1_prompt(&mut lane, &fixture.stats);
    }
    let id = lane.engine.binding().session().get();
    (lane, Observed { id, maximum, trace })
}

fn reference_rows(fixture: &Fixture, active: &VecDeque<GpuActive>) -> BTreeMap<u64, Reference> {
    let references = active
        .iter()
        .map(|lane| {
            (
                lane.engine.binding().session().get(),
                Reference::new(fixture, lane),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(references.len(), active.len(), "independent session IDs");
    references
}

fn finish_observed(
    active: &mut VecDeque<GpuActive>,
    fixture: &Fixture,
    references: &mut BTreeMap<u64, Reference>,
    observed: &mut [Observed],
) {
    finish(
        active,
        fixture,
        references,
        &mut observed
            .iter_mut()
            .map(|row| &mut row.trace)
            .collect::<Vec<_>>(),
    );
    for row in observed {
        row.trace
            .verify(&references[&row.id], &fixture.model.tokenizer, row.maximum);
    }
}

fn band_counts(stats: &WaveStats) -> [u64; 3] {
    stats
        .native_bands
        .each_ref()
        .map(|count| count.load(Ordering::Acquire))
}

fn require_bands(stats: &WaveStats, before: [u64; 3], expected: &[usize]) {
    let actual = band_counts(stats);
    for (index, width) in [2, 4, 8].into_iter().enumerate() {
        assert_eq!(
            actual[index] - before[index],
            expected.iter().filter(|&&band| band == width).count() as u64,
            "actual completed native B{width} cohorts"
        );
    }
}

fn prior_roots(active: &VecDeque<GpuActive>) -> BTreeMap<u64, (u64, u64)> {
    active
        .iter()
        .map(|lane| {
            let binding = lane.engine.binding();
            (
                binding.session().get(),
                (binding.root(), binding.frontier()),
            )
        })
        .collect()
}

fn require_single_transition(active: &VecDeque<GpuActive>, prior: &BTreeMap<u64, (u64, u64)>) {
    for lane in active {
        let binding = lane.engine.binding();
        let (root, frontier) = prior[&binding.session().get()];
        assert_eq!(binding.root(), root + 1, "one original root publication");
        assert_eq!(binding.frontier(), frontier + 1, "one consumed token");
    }
}

fn survivor_bands(width: usize) -> &'static [usize] {
    match width {
        4 => &[2, 1],
        8 => &[4, 2, 1],
        _ => panic!("wide fixture requires B4/B8"),
    }
}

fn wide_prompt_lengths(fixture: &Fixture) {
    let width = fixture.model.native_rows;
    let mut active = VecDeque::new();
    let mut observed = Vec::new();
    for (index, input) in ["a", "b", "c", "d", "e", "f", "g", "h"]
        .into_iter()
        .take(width)
        .enumerate()
    {
        let (lane, trace) = observed_lane(fixture, input, index + 1, EVENTS_PER_REQUEST, false);
        assert_eq!(lane.prompt.len(), 1, "equal-frontier original token inputs");
        active.push_back(lane);
        observed.push(trace);
    }
    assert_eq!(
        active
            .iter()
            .map(|lane| lane.prompt[0])
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        width,
        "distinct source input token IDs"
    );
    let prior = prior_roots(&active);
    let bands = band_counts(&fixture.stats);
    round(&mut active, &fixture.stats);
    assert_eq!(active.len(), width);
    require_bands(&fixture.stats, bands, &[width]);
    require_single_transition(&active, &prior);
    let mut references = reference_rows(fixture, &active);
    assert!(
        references
            .values()
            .all(|reference| { reference.sequence[0].0 != fixture.model.tokenizer.eos() })
    );
    let prior = prior_roots(&active);
    let bands = band_counts(&fixture.stats);
    let committed = fixture.stats.committed_roots.load(Ordering::Acquire);
    let forwards = fixture.stats.gpu_forwards.load(Ordering::Acquire);
    round(&mut active, &fixture.stats);
    assert_eq!(
        active.len(),
        width - 1,
        "maximum=1 row retires before feedback"
    );
    require_bands(&fixture.stats, bands, survivor_bands(width));
    require_single_transition(&active, &prior);
    assert_eq!(
        fixture.stats.committed_roots.load(Ordering::Acquire),
        committed + (width - 1) as u64
    );
    assert_eq!(
        fixture.stats.gpu_forwards.load(Ordering::Acquire),
        forwards + survivor_bands(width).len() as u64,
        "native packet includes the exact B1 residual handoff"
    );
    update(&active, &mut references);
    finish_observed(&mut active, fixture, &mut references, &mut observed);
    println!(
        "P12_WIDE_PROMPT_RETIRE_PASS rows={width} distinct_source_tokens_and_sessions=true \
        maxima=1..{width} native_to_B1=true CPU_B1_packets_all42=exact text_usage_Done=once"
    );
}

fn wide_b1_handoff(fixture: &Fixture) {
    let width = fixture.model.native_rows;
    let mut active = VecDeque::new();
    let mut observed = Vec::new();
    for _ in 0..width {
        let (lane, trace) = observed_lane(fixture, "a", 3, EVENTS_PER_REQUEST, true);
        active.push_back(lane);
        observed.push(trace);
    }
    let mut references = reference_rows(fixture, &active);
    let prior = prior_roots(&active);
    let bands = band_counts(&fixture.stats);
    round(&mut active, &fixture.stats);
    assert_eq!(active.len(), width);
    require_bands(&fixture.stats, bands, &[width]);
    require_single_transition(&active, &prior);
    update(&active, &mut references);
    finish_observed(&mut active, fixture, &mut references, &mut observed);
    println!("P12_WIDE_B1_NATIVE_HANDOFF_PASS rows={width} CPU_B1_packets_all42=exact");
}

fn wide_cancel_after_ready(fixture: &Fixture) {
    let width = fixture.model.native_rows;
    let mut active = VecDeque::new();
    let mut observed = Vec::new();
    for _ in 0..width {
        let (lane, trace) = observed_lane(fixture, "a", 4, EVENTS_PER_REQUEST, true);
        active.push_back(lane);
        observed.push(trace);
    }
    let mut references = reference_rows(fixture, &active);
    let cancelled_index = width / 2;
    let mut cancelled = observed.remove(cancelled_index);
    let cancelled_reference = references.remove(&cancelled.id).unwrap();
    let saved = active[cancelled_index]
        .engine
        .legacy()
        .expect("legacy qualification")
        .snapshot();
    let prior = prior_roots(&active);
    let cancel = Arc::clone(&active[cancelled_index].cancelled);
    AFTER_COHORT_READY.with(|hook| {
        assert!(hook.borrow().is_none());
        *hook.borrow_mut() = Some(Box::new(move || cancel.store(true, Ordering::Release)));
    });
    let bands = band_counts(&fixture.stats);
    let discarded = fixture.stats.discarded_cohorts.load(Ordering::Acquire);
    let committed = fixture.stats.committed_roots.load(Ordering::Acquire);
    let model_tokens = fixture.stats.model_tokens.load(Ordering::Acquire);
    let cancelled_before = fixture.stats.cancelled.load(Ordering::Acquire);
    round(&mut active, &fixture.stats);
    AFTER_COHORT_READY.with(|hook| assert!(hook.borrow().is_none(), "real Ready hook reached"));
    assert_eq!(active.len(), width - 1);
    assert!(
        active
            .iter()
            .all(|lane| lane.engine.binding().session().get() != cancelled.id)
    );
    let mut physical = vec![width];
    physical.extend_from_slice(survivor_bands(width));
    require_bands(&fixture.stats, bands, &physical);
    require_single_transition(&active, &prior);
    assert_eq!(
        fixture.stats.discarded_cohorts.load(Ordering::Acquire),
        discarded + 1
    );
    assert_eq!(
        fixture.stats.cancelled.load(Ordering::Acquire),
        cancelled_before + 1
    );
    assert_eq!(
        fixture.stats.committed_roots.load(Ordering::Acquire),
        committed + (width - 1) as u64
    );
    assert_eq!(
        fixture.stats.model_tokens.load(Ordering::Acquire),
        model_tokens + (2 * width - 1) as u64,
        "whole discarded native cohort plus actual survivor forwards"
    );
    update(&active, &mut references);
    cancelled_reference.snapshot(&saved);
    cancelled.trace.drain();
    assert!(cancelled.trace.done.is_empty());
    let presented = fixture
        .model
        .tokenizer
        .decode(&[cancelled_reference.sequence[0].0], true)
        .unwrap();
    let expected_text = if presented.contains('�') {
        ""
    } else {
        &presented
    };
    assert_eq!(cancelled.trace.text, expected_text);
    finish_observed(&mut active, fixture, &mut references, &mut observed);
    cancelled_reference.snapshot(&saved);
    cancelled.trace.drain();
    assert!(cancelled.trace.done.is_empty());
    assert_eq!(cancelled.trace.text, expected_text);
    println!(
        "P12_WIDE_READY_CANCEL_PASS rows={width} whole_candidate_discard=true \
        survivors_regrouped=true original_roots_packets_all42=exact text_usage_Done=once"
    );
}

fn wide_credit_park(fixture: &Fixture) {
    let width = fixture.model.native_rows;
    let mut active = VecDeque::new();
    let mut observed = Vec::new();
    for index in 0..width {
        let (lane, trace) = observed_lane(
            fixture,
            if index == 0 { "a" } else { "b" },
            8,
            if index == 0 { 1 } else { EVENTS_PER_REQUEST },
            true,
        );
        active.push_back(lane);
        observed.push(trace);
    }
    let slow_id = observed[0].id;
    let mut references = reference_rows(fixture, &active);
    for _ in 0..12 {
        round(&mut active, &fixture.stats);
        update(&active, &mut references);
        for row in observed.iter_mut().skip(1) {
            row.trace.drain();
        }
        if active.iter().any(|lane| {
            lane.engine.binding().session().get() == slow_id && lane.blocked_since.is_some()
        }) {
            break;
        }
    }
    assert_eq!(
        active.len(),
        width,
        "healthy rows remain live at slow-reader park"
    );
    let slow = active
        .iter()
        .find(|lane| lane.engine.binding().session().get() == slow_id)
        .unwrap();
    assert!(slow.blocked_since.is_some());
    assert!(slow.packet_presented);
    let binding = slow.engine.binding();
    let token = slow.device_token.as_ref().unwrap();
    let packet = (token.id, token.max_bits);
    for _ in 0..2 {
        let prior = prior_roots(&active);
        let bands = band_counts(&fixture.stats);
        round(&mut active, &fixture.stats);
        update(&active, &mut references);
        require_bands(&fixture.stats, bands, survivor_bands(width));
        let mut healthy = 0;
        for lane in &active {
            let current = lane.engine.binding();
            if current.session().get() != slow_id {
                assert_eq!(current.root(), prior[&current.session().get()].0 + 1);
                assert_eq!(current.frontier(), prior[&current.session().get()].1 + 1);
                healthy += 1;
            }
        }
        assert_eq!(healthy, width - 1, "every healthy survivor progresses");
        let slow = active
            .iter()
            .find(|lane| lane.engine.binding().session().get() == slow_id)
            .unwrap();
        assert_eq!(slow.engine.binding(), binding);
        assert!(slow.packet_presented && slow.blocked_since.is_some());
        let token = slow.device_token.as_ref().unwrap();
        assert_eq!((token.id, token.max_bits), packet);
        for row in observed.iter_mut().skip(1) {
            row.trace.drain();
        }
    }
    observed[0].trace.one_credit();
    finish_observed(&mut active, fixture, &mut references, &mut observed);
    println!(
        "P12_WIDE_REAL_CREDIT_PASS rows={width} channel_capacity=1 \
        healthy_survivors_progress=true parked_packet_root_all42=exact resumed_events=once"
    );
}

fn wide_source_eos(fixture: &Fixture) {
    let width = fixture.model.native_rows;
    let input =
        find_source_eos(fixture).expect("wide cohort requires an actual trained-source EOS");
    let length = fixture
        .model
        .tokenizer
        .prompt_tokens(&input, false)
        .unwrap()
        .len();
    let sibling = " a".repeat(length);
    assert_eq!(
        fixture
            .model
            .tokenizer
            .prompt_tokens(&sibling, false)
            .unwrap()
            .len(),
        length
    );
    let mut active = VecDeque::new();
    let mut observed = Vec::new();
    for index in 0..width {
        let (lane, trace) = observed_lane(
            fixture,
            if index == 0 { &input } else { &sibling },
            if index == 0 { 9 } else { 10 },
            EVENTS_PER_REQUEST,
            false,
        );
        active.push_back(lane);
        observed.push(trace);
    }
    let bands = band_counts(&fixture.stats);
    for _ in 0..32 {
        if active.iter().all(|lane| lane.cursor == lane.prompt.len()) {
            break;
        }
        round(&mut active, &fixture.stats);
    }
    assert_eq!(active.len(), width);
    assert!(active.iter().all(|lane| lane.cursor == lane.prompt.len()));
    require_bands(&fixture.stats, bands, &[width]);
    let mut references = reference_rows(fixture, &active);
    finish_observed(&mut active, fixture, &mut references, &mut observed);
    assert_eq!(
        observed[0].trace.done[0].3, "stop",
        "trained EOS must retire its row"
    );
    println!(
        "P12_WIDE_TRAINED_EOS_PASS rows={width} source_selected_EOS=true \
        CPU_B1_packets_all42=exact EOS_excluded_usage=true terminal_Done=once"
    );
}

fn qualify_wide(name: &str, native_rows: usize) {
    let fixture = Fixture::with_native_rows(name, native_rows);
    wide_prompt_lengths(&fixture);
    wide_b1_handoff(&fixture);
    wide_cancel_after_ready(&fixture);
    wide_credit_park(&fixture);
    wide_source_eos(&fixture);
    assert!(fixture.stats.kernel_dispatches.load(Ordering::Acquire) > 0);
    assert!(fixture.stats.native_projections.load(Ordering::Acquire) > 0);
    assert!(fixture.stats.native_sampler.load(Ordering::Acquire) > 0);
    assert!(band_counts(&fixture.stats)[if native_rows == 4 { 1 } else { 2 }] > 0);
    assert_eq!(
        fixture
            .stats
            .native_admission_refusals
            .load(Ordering::Acquire),
        0
    );
    fixture.close();
    println!(
        "P12_WIDE_NATIVE_WORKER_PHYSICAL_PASS source={name} rows={native_rows} \
        original_authorities_CPU_B1_all42=exact final_active=0 final_reserved=0 \
        actual_HTTP_wire=separate_gate vLLM_performance=not_qualified"
    );
}

#[test]
#[ignore = "physical release CUDA; approved original .8B; isolated sequential GPU coordinator"]
fn original_qwen08b_native_b2_worker() {
    qualify("qwen35-0.8b");
}

#[test]
#[ignore = "physical release CUDA; approved original 2B; isolated sequential GPU coordinator"]
fn original_qwen2b_native_b2_worker() {
    qualify("qwen35-2b");
}

#[test]
#[ignore = "physical release CUDA B4; approved original .8B; isolated sequential GPU coordinator"]
fn original_qwen08b_native_b4_worker() {
    qualify_wide("qwen35-0.8b", 4);
}

#[test]
#[ignore = "physical release CUDA B8; approved original .8B; isolated sequential GPU coordinator"]
fn original_qwen08b_native_b8_worker() {
    qualify_wide("qwen35-0.8b", 8);
}

#[test]
#[ignore = "physical release CUDA B4; approved original 2B; isolated sequential GPU coordinator"]
fn original_qwen2b_native_b4_worker() {
    qualify_wide("qwen35-2b", 4);
}

#[test]
#[ignore = "physical release CUDA B8; approved original 2B; isolated sequential GPU coordinator"]
fn original_qwen2b_native_b8_worker() {
    qualify_wide("qwen35-2b", 8);
}

fn unequal_arrival_metrics(name: &str) {
    let fixture = Fixture::new(name);
    // The first request has genuinely more source tokens and is already
    // decoding before the second request is created.
    let (mut first, mut first_trace) = fixture.lane_configured("a b c d e f g h", 8, 1, true);
    b1_prompt(&mut first, &fixture.stats);
    let first_id = first.engine.binding().session().get();
    let mut first_reference = Reference::new(&fixture, &first);
    assert_ne!(
        first.device_token.as_ref().unwrap().id,
        fixture.model.tokenizer.eos()
    );
    assert!(matches!(
        first.step(&fixture.stats).unwrap(),
        MemberStep::Forwarded
    ));
    first_reference.sync(&first);
    assert!(matches!(
        first.step(&fixture.stats).unwrap(),
        MemberStep::Parked
    ));
    assert!(first.packet_presented && first.blocked_since.is_some());
    let parked = first.engine.binding();

    let (mut second, mut second_trace) = fixture.lane_configured("b", 6, EVENTS_PER_REQUEST, true);
    b1_prompt(&mut second, &fixture.stats);
    let second_id = second.engine.binding().session().get();
    assert_ne!(first.prompt.len(), second.prompt.len());
    assert_ne!(parked.frontier(), second.engine.binding().frontier());
    let second_reference = Reference::new(&fixture, &second);
    assert_ne!(
        second.device_token.as_ref().unwrap().id,
        fixture.model.tokenizer.eos()
    );
    let mut references =
        BTreeMap::from([(first_id, first_reference), (second_id, second_reference)]);
    first_trace.one_credit();
    let mut active = VecDeque::from([first, second]);
    let prior = prior_roots(&active);
    let bands = band_counts(&fixture.stats);
    round(&mut active, &fixture.stats);
    require_bands(&fixture.stats, bands, &[2]);
    require_single_transition(&active, &prior);
    update(&active, &mut references);
    assert_eq!(active.len(), 2);
    let produced = active
        .iter()
        .map(|lane| lane.token_gpu_observation.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(produced[0].native_cohort, produced[1].native_cohort);
    assert_eq!(produced[0].native_cohort.unwrap().width, 2);
    assert_ne!(produced[0].producer_frontier, produced[1].producer_frontier);
    finish(
        &mut active,
        &fixture,
        &mut references,
        &mut [&mut first_trace, &mut second_trace],
    );
    first_trace.verify(&references[&first_id], &fixture.model.tokenizer, 8);
    second_trace.verify(&references[&second_id], &fixture.model.tokenizer, 6);
    let first_tokens = first_trace.tokens.as_ref().unwrap();
    let second_tokens = second_trace.tokens.as_ref().unwrap();
    assert!(
        first_tokens
            .iter()
            .any(|left| second_tokens.iter().any(|right| {
                let (a, b) = (left.gpu.unwrap(), right.gpu.unwrap());
                a.native_cohort.is_some()
                    && a.native_cohort == b.native_cohort
                    && a.producer_frontier != b.producer_frontier
            })),
        "wire observations must retain actual unequal-frontier native membership"
    );
    drop(references);
    drop(first_trace);
    drop(second_trace);
    fixture.close();
    println!(
        "P12_UNEQUAL_ARRIVAL_WORKER_PASS source={name} actual_GPU_prefill=true bounded_credit_park=true native_membership_frontiers=observed CPU_B1_packets_all42=exact final_reserved=0"
    );
}

#[test]
#[ignore = "release physical CUDA unequal-frontier worker; approved .8B source; serial"]
fn original_qwen08b_native_unequal_arrival_token_observations() {
    unequal_arrival_metrics("qwen35-0.8b");
}

#[test]
#[ignore = "release physical CUDA unequal-frontier worker; approved 2B source; serial"]
fn original_qwen2b_native_unequal_arrival_token_observations() {
    unequal_arrival_metrics("qwen35-2b");
}

#[path = "qualification/prefix.rs"]
mod prefix;
