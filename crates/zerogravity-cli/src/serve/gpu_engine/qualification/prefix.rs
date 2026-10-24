//! Real teacher-prefix cohorts, mounted under the original worker qualifier.
use super::*;

struct PrefixOracle {
    cpu: QwenDenseCpu,
    gpu: ResidentSession,
    consumed: usize,
    packet: Option<ResidentToken>,
}
impl PrefixOracle {
    fn new(fixture: &Fixture) -> Self {
        Self {
            cpu: QwenDenseCpu::from_shared(Arc::clone(&fixture.source)).unwrap(),
            gpu: block_on(
                fixture
                    .model
                    .engine
                    .legacy_model()
                    .expect("legacy qualification")
                    .session(),
            )
            .unwrap(),
            consumed: 0,
            packet: None,
        }
    }
    fn observe(&mut self, lane: &GpuActive) {
        assert_eq!(
            lane.cursor,
            self.consumed + 1,
            "one truthful teacher wave credit"
        );
        let id = lane.prompt[self.consumed];
        let expected = self.cpu.forward(id).unwrap();
        let prior = self.gpu.snapshot();
        let read = block_on(self.gpu.forward(&[id], false)).unwrap();
        exact(&expected, &read.logits);
        self.consumed += 1;
        same_domains(
            &self.cpu.snapshot().domain_values(),
            &block_on(self.gpu.snapshot().read_domains()).unwrap(),
        );
        same_domains(
            &self.cpu.snapshot().domain_values(),
            &block_on(
                lane.engine
                    .legacy()
                    .expect("legacy qualification")
                    .snapshot()
                    .read_domains(),
            )
            .unwrap(),
        );
        assert_eq!(lane.engine.binding().frontier(), self.consumed as u64);
        if self.consumed < lane.prompt.len() {
            assert!(lane.device_token.is_none());
            assert!(lane.token_commit_observed_at.is_none());
            assert!(lane.token_gpu_observation.is_none());
            assert!(!lane.packet_presented);
            assert!(lane.blocked_since.is_none());
            assert_eq!(lane.output.generated_token_count(), 0);
        } else {
            let expected = best(&expected);
            let actual = lane
                .device_token
                .as_ref()
                .expect("prompt tail greedy packet");
            assert_eq!((actual.id, actual.max_bits), expected);
            // Actual accepted B1 greedy selection is an independent packet oracle.
            self.gpu.restore(&prior).unwrap();
            let read = block_on(self.gpu.forward_greedy(&[id])).unwrap();
            let packet = read.device_token.unwrap();
            assert_eq!((packet.id, packet.max_bits), expected);
            same_domains(
                &self.cpu.snapshot().domain_values(),
                &block_on(self.gpu.snapshot().read_domains()).unwrap(),
            );
            self.packet = Some(packet);
        }
    }
    fn into_reference(self, lane: &GpuActive) -> Reference {
        assert_eq!(self.consumed, lane.prompt.len());
        let packet = self.packet.expect("independently produced B1 tail packet");
        let reference = Reference {
            cpu: self.cpu,
            gpu: self.gpu,
            sequence: vec![(packet.id, packet.max_bits)],
            packet,
            prompt_tokens: lane.prompt.len(),
        };
        reference.check(lane);
        reference
    }
}

/// Same eligibility, wave formation and cohort helpers as the actual worker.
fn teacher_round(active: &mut VecDeque<GpuActive>, stats: &WaveStats) {
    let mut parked = VecDeque::new();
    let eligible = active
        .drain(..)
        .filter_map(|lane| eligible_member(lane, &mut parked, stats))
        .collect::<Vec<_>>();
    active.extend(parked);
    if eligible.is_empty() {
        return;
    }
    let width = eligible[0].native_rows;
    let ready = eligible.iter().map(GpuActive::ready).collect::<Vec<_>>();
    assert!(ready.iter().all(|row| row.tokens == 1));
    let waves = gpu_batch::form_native(
        &ready,
        Limits {
            max_members: width,
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
        if members.len() > 1 && members.iter().all(|lane| lane.cursor < lane.prompt.len()) {
            finish_prompt_cohort(members, active, stats, width);
        } else if members.len() > 1 && members.iter().all(|lane| lane.cursor == lane.prompt.len()) {
            advance_decode_cohort(members, active, stats, width);
        } else {
            for lane in members {
                advance_member(lane, active, stats);
            }
        }
    }
    assert!(pending.iter().all(Option::is_none));
}

fn no_prefix_events(trace: &Trace) {
    assert!(
        matches!(
            trace.receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected)
        ),
        "teacher prefix emitted Text/TokenText/Done/Error"
    );
    assert!(trace.text.is_empty());
    assert!(trace.done.is_empty());
    assert!(trace.tokens.as_ref().is_some_and(Vec::is_empty));
}

const PROMPTS: [&str; 8] = [
    "a b c d e f g",
    "one two three four five six seven eight nine",
    "Continue the sequence 1, 2, 3, 4, 5, 6,",
    "The answer to the following simple question is",
    "Red green blue yellow orange purple white black",
    "A small bird flies above a wide green field.",
    "Count all the whole numbers starting at one and then",
    "This original source request has an independent teacher history.",
];

fn ragged_prefixes(fixture: &Fixture) {
    let width = fixture.model.native_rows;
    let mut active = VecDeque::new();
    let mut traces = BTreeMap::new();
    let mut prefix = BTreeMap::new();
    let mut references = BTreeMap::new();
    for (row, prompt) in PROMPTS.iter().take(width).enumerate() {
        let (mut lane, trace) =
            fixture.lane_configured(prompt, 2 + row % 3, EVENTS_PER_REQUEST, true);
        assert!(lane.prompt.len() > 2);
        if row == 0 {
            // Genuine source EOS as a teacher token cannot end generation.
            lane.prompt[1] = fixture.model.tokenizer.eos();
        }
        let id = lane.engine.binding().session().get();
        traces.insert(id, trace);
        prefix.insert(id, PrefixOracle::new(fixture));
        active.push_back(lane);
    }
    let before = band_counts(&fixture.stats);
    let sampler = fixture.stats.native_sampler.load(Ordering::Acquire);
    let state_batches = fixture
        .stats
        .fused_state_only_batches
        .load(Ordering::Acquire);
    let state_rows = fixture.stats.fused_state_only_rows.load(Ordering::Acquire);
    let greedy_batches = fixture.stats.fused_greedy_batches.load(Ordering::Acquire);
    teacher_round(&mut active, &fixture.stats);
    require_bands(&fixture.stats, before, &[width]);
    assert_eq!(
        fixture.stats.native_sampler.load(Ordering::Acquire),
        sampler
    );
    assert_eq!(
        fixture
            .stats
            .fused_state_only_batches
            .load(Ordering::Acquire),
        state_batches + 1
    );
    assert_eq!(
        fixture.stats.fused_state_only_rows.load(Ordering::Acquire),
        state_rows + width as u64
    );
    assert_eq!(
        fixture.stats.fused_greedy_batches.load(Ordering::Acquire),
        greedy_batches
    );
    assert_eq!(fixture.stats.output_tokens.load(Ordering::Acquire), 0);
    assert_eq!(
        fixture.stats.model_tokens.load(Ordering::Acquire),
        width as u64
    );
    assert_eq!(
        fixture.stats.committed_roots.load(Ordering::Acquire),
        width as u64
    );
    for lane in &active {
        let id = lane.engine.binding().session().get();
        prefix.get_mut(&id).unwrap().observe(lane);
        no_prefix_events(&traces[&id]);
    }
    if width == 8 {
        // The real worker admits at most8 active requests. Retire a cancelled
        // teacher row before creating its arrival replacement.
        let cancelled = active.pop_back().unwrap();
        let id = cancelled.engine.binding().session().get();
        cancelled.cancelled.store(true, Ordering::Release);
        assert!(eligible_member(cancelled, &mut VecDeque::new(), &fixture.stats).is_none());
        drop(prefix.remove(&id).unwrap());
        no_prefix_events(&traces.remove(&id).unwrap());
    }
    // Arriving source-real prompt starts at frontier0 while original rows are1.
    let (arrival, trace) = fixture.lane_configured(
        "a late teacher prompt with independent positions",
        3,
        EVENTS_PER_REQUEST,
        true,
    );
    assert!(arrival.prompt.len() > 2);
    let arrival_id = arrival.engine.binding().session().get();
    traces.insert(arrival_id, trace);
    prefix.insert(arrival_id, PrefixOracle::new(fixture));
    active.push_front(arrival);
    for _ in 0..128 {
        if active.is_empty() {
            break;
        }
        teacher_round(&mut active, &fixture.stats);
        for lane in &active {
            let id = lane.engine.binding().session().get();
            if let Some(oracle) = prefix.get_mut(&id) {
                oracle.observe(lane);
                if lane.cursor == lane.prompt.len() {
                    let oracle = prefix.remove(&id).unwrap();
                    references.insert(id, oracle.into_reference(lane));
                } else {
                    no_prefix_events(&traces[&id]);
                }
            } else {
                references.get_mut(&id).unwrap().sync(lane);
            }
        }
        for (id, trace) in &mut traces {
            if references.contains_key(id) {
                trace.drain();
            }
        }
    }
    assert!(
        active.is_empty(),
        "bounded native prefix/decode worker stalled"
    );
    assert!(prefix.is_empty());
    for (id, trace) in &mut traces {
        let reference = &references[id];
        // The request limit is encoded by its independently observed output.
        let maximum = if *id == arrival_id {
            3
        } else {
            // Original IDs retain insertion order (global IDs are monotonic).
            let row = traces_row_index(&references, *id, arrival_id);
            2 + row % 3
        };
        trace.verify(reference, &fixture.model.tokenizer, maximum);
    }
    assert_eq!(fixture.stats.active.load(Ordering::Acquire), 0);
    assert!(fixture.stats.fused_batches.load(Ordering::Acquire) > 1);
    assert!(fixture.stats.native_sampler.load(Ordering::Acquire) > 0);
    drop(references);
    println!(
        "P12_NATIVE_PREFIX_WORKER_PASS rows={width} actual_teacher_EOS=true each_teacher_CPU_B1_all42=exact ragged_arrival=true no_prefix_Text_ID_Done_usage=true tail_fullV_packets_and_final_events=exact"
    );
}

fn traces_row_index(references: &BTreeMap<u64, Reference>, id: u64, arrival: u64) -> usize {
    references
        .keys()
        .filter(|&&key| key != arrival)
        .position(|&key| key == id)
        .unwrap()
}

fn cancel_after_prefix_ready(fixture: &Fixture) {
    let (left, left_trace) = fixture.lane_configured(PROMPTS[0], 2, EVENTS_PER_REQUEST, true);
    let (right, mut right_trace) = fixture.lane_configured(PROMPTS[1], 2, EVENTS_PER_REQUEST, true);
    let left_saved = left
        .engine
        .legacy()
        .expect("legacy qualification")
        .snapshot();
    let left_root = left.engine.binding();
    let right_root = right.engine.binding();
    let flag = Arc::clone(&left.cancelled);
    let fired = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&fired);
    let model_tokens = fixture.stats.model_tokens.load(Ordering::Acquire);
    let committed = fixture.stats.committed_roots.load(Ordering::Acquire);
    let discarded = fixture.stats.discarded_cohorts.load(Ordering::Acquire);
    let sampler = fixture.stats.native_sampler.load(Ordering::Acquire);
    let output = fixture.stats.output_tokens.load(Ordering::Acquire);
    AFTER_COHORT_READY.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(move || {
            seen.store(true, Ordering::Release);
            flag.store(true, Ordering::Release);
        }))
    });
    let mut active = VecDeque::from([left, right]);
    teacher_round(&mut active, &fixture.stats);
    assert!(
        fired.load(Ordering::Acquire),
        "no real state-only Ready hook"
    );
    assert_eq!(active.len(), 1);
    let mut survivor = active.pop_front().unwrap();
    assert_eq!(survivor.engine.binding().session(), right_root.session());
    assert_eq!(survivor.engine.binding().root(), right_root.root() + 1);
    assert_eq!(survivor.cursor, 1);
    assert_eq!(left_root.frontier(), 0);
    let empty = QwenDenseCpu::from_shared(Arc::clone(&fixture.source)).unwrap();
    same_domains(
        &empty.snapshot().domain_values(),
        &block_on(left_saved.read_domains()).unwrap(),
    );
    let mut reference = PrefixOracle::new(fixture);
    reference.observe(&survivor);
    no_prefix_events(&left_trace);
    no_prefix_events(&right_trace);
    assert_eq!(
        fixture.stats.model_tokens.load(Ordering::Acquire),
        model_tokens + 3,
        "actual discarded B2 launch plus survivor B1 must remain counted"
    );
    assert_eq!(
        fixture.stats.committed_roots.load(Ordering::Acquire),
        committed + 1
    );
    assert_eq!(
        fixture.stats.discarded_cohorts.load(Ordering::Acquire),
        discarded + 1
    );
    assert_eq!(
        fixture.stats.native_sampler.load(Ordering::Acquire),
        sampler
    );
    assert_eq!(fixture.stats.output_tokens.load(Ordering::Acquire), output);
    drop(left_saved);
    while survivor.cursor < survivor.prompt.len() {
        assert!(matches!(
            survivor.step(&fixture.stats).unwrap(),
            MemberStep::Forwarded
        ));
        reference.observe(&survivor);
        no_prefix_events(&right_trace);
    }
    let id = survivor.engine.binding().session().get();
    let mut references = BTreeMap::from([(id, reference.into_reference(&survivor))]);
    let mut active = VecDeque::from([survivor]);
    finish(
        &mut active,
        fixture,
        &mut references,
        &mut [&mut right_trace],
    );
    right_trace.verify(&references[&id], &fixture.model.tokenizer, 2);
    no_prefix_events(&left_trace);
    drop(references);
    println!(
        "P12_NATIVE_PREFIX_READY_CANCEL_PASS real_Ready=true whole_cohort_discarded=true survivor_original_teacher_once=true no_prefix_output=true"
    );
}

fn parked_decode_does_not_block_prefix(fixture: &Fixture) {
    let (mut decode, mut decode_trace) = fixture.lane_configured("a", 4, 1, true);
    b1_prompt(&mut decode, &fixture.stats);
    let decode_id = decode.engine.binding().session().get();
    let mut reference = Reference::new(fixture, &decode);
    assert_ne!(reference.packet.id, fixture.model.tokenizer.eos());
    assert!(matches!(
        decode.step(&fixture.stats).unwrap(),
        MemberStep::Forwarded
    ));
    reference.sync(&decode);
    assert_ne!(reference.packet.id, fixture.model.tokenizer.eos());
    assert!(matches!(
        decode.step(&fixture.stats).unwrap(),
        MemberStep::Parked
    ));
    let parked_root = decode.engine.binding();
    let parked_snapshot = decode
        .engine
        .legacy()
        .expect("legacy qualification")
        .snapshot();
    let (left, left_trace) = fixture.lane_configured(PROMPTS[0], 2, EVENTS_PER_REQUEST, true);
    let (right, right_trace) = fixture.lane_configured(PROMPTS[1], 2, EVENTS_PER_REQUEST, true);
    let (cancelled, cancelled_trace) =
        fixture.lane_configured(PROMPTS[2], 2, EVENTS_PER_REQUEST, true);
    cancelled.cancelled.store(true, Ordering::Release);
    let left_id = left.engine.binding().session().get();
    let right_id = right.engine.binding().session().get();
    let mut prefix = BTreeMap::from([
        (left_id, PrefixOracle::new(fixture)),
        (right_id, PrefixOracle::new(fixture)),
    ]);
    let mut traces = BTreeMap::from([(left_id, left_trace), (right_id, right_trace)]);
    let mut references = BTreeMap::from([(decode_id, reference)]);
    let sampler = fixture.stats.native_sampler.load(Ordering::Acquire);
    let cancelled_before = fixture.stats.cancelled.load(Ordering::Acquire);
    let before = band_counts(&fixture.stats);
    let mut active = VecDeque::from([decode, left, right, cancelled]);
    teacher_round(&mut active, &fixture.stats);
    require_bands(&fixture.stats, before, &[2]);
    assert_eq!(
        fixture.stats.native_sampler.load(Ordering::Acquire),
        sampler
    );
    assert_eq!(
        fixture.stats.cancelled.load(Ordering::Acquire),
        cancelled_before + 1
    );
    let parked = active
        .iter()
        .find(|lane| lane.engine.binding().session().get() == decode_id)
        .unwrap();
    assert_eq!(parked.engine.binding(), parked_root);
    assert!(parked.packet_presented && parked.blocked_since.is_some());
    references[&decode_id].snapshot(&parked_snapshot);
    for lane in &active {
        let id = lane.engine.binding().session().get();
        if let Some(oracle) = prefix.get_mut(&id) {
            oracle.observe(lane);
        }
    }
    no_prefix_events(&traces[&left_id]);
    no_prefix_events(&traces[&right_id]);
    no_prefix_events(&cancelled_trace);
    drop(parked_snapshot);
    for _ in 0..128 {
        if active.is_empty() {
            break;
        }
        decode_trace.drain();
        for trace in traces.values_mut() {
            trace.drain();
        }
        teacher_round(&mut active, &fixture.stats);
        for lane in &active {
            let id = lane.engine.binding().session().get();
            if let Some(oracle) = prefix.get_mut(&id) {
                oracle.observe(lane);
                if lane.cursor == lane.prompt.len() {
                    let oracle = prefix.remove(&id).unwrap();
                    references.insert(id, oracle.into_reference(lane));
                }
            } else {
                references.get_mut(&id).unwrap().sync(lane);
            }
        }
    }
    assert!(active.is_empty());
    assert!(prefix.is_empty());
    decode_trace.verify(&references[&decode_id], &fixture.model.tokenizer, 4);
    for (id, trace) in &mut traces {
        trace.verify(&references[id], &fixture.model.tokenizer, 2);
    }
    no_prefix_events(&cancelled_trace);
    drop(references);
    println!(
        "P12_NATIVE_PREFIX_FAIRNESS_PASS parked_decode_root=unchanged new_prefix_native_B2=true pre_cancelled_row=not_launched output_credit_resume_text_usage_Done=once"
    );
}

fn qualify(name: &str, width: usize) {
    let fixture = Fixture::with_native_rows(name, width);
    ragged_prefixes(&fixture);
    cancel_after_prefix_ready(&fixture);
    parked_decode_does_not_block_prefix(&fixture);
    assert_eq!(
        fixture.stats.fused_batches.load(Ordering::Acquire),
        fixture
            .stats
            .fused_state_only_batches
            .load(Ordering::Acquire)
            + fixture.stats.fused_greedy_batches.load(Ordering::Acquire)
    );
    assert_eq!(
        fixture.stats.native_sampler.load(Ordering::Acquire),
        4 * fixture.stats.fused_greedy_batches.load(Ordering::Acquire)
    );
    fixture.close();
}
macro_rules! case {
    ($test:ident, $name:literal, $rows:literal) => {
        #[test]
        #[ignore = "release physical CUDA native teacher prefix worker; serial"]
        fn $test() {
            qualify($name, $rows);
        }
    };
}
case!(original_qwen08b_native_prefix_b2_worker, "qwen35-0.8b", 2);
case!(original_qwen08b_native_prefix_b4_worker, "qwen35-0.8b", 4);
case!(original_qwen08b_native_prefix_b8_worker, "qwen35-0.8b", 8);
case!(original_qwen2b_native_prefix_b2_worker, "qwen35-2b", 2);
case!(original_qwen2b_native_prefix_b4_worker, "qwen35-2b", 4);
case!(original_qwen2b_native_prefix_b8_worker, "qwen35-2b", 8);
