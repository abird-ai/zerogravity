//! Actual two-position natural-route diagnostics against pinned original natural reference.
#[allow(dead_code)]
#[path = "qwen27b_hip_matrix/expected.rs"]
mod expected;
#[allow(dead_code)]
#[path = "gemma4_cpu_matrix/io.rs"]
mod io;
use expected::{Expected, packet_matches};
use io::{Proof, need, write_new};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    future::Future,
    path::PathBuf,
    pin::pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread,
};
use zerogravity::{
    Error,
    accelerator::program::StateSlot,
    cache::{CacheOptions, CacheStore},
    qwen27b_gpu::{Qwen27bGpuContext, Qwen27bGpuLoaded},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_gpu::{
    Bf16TextSourceCandidate, Bf16TextSourceSession, Bf16TextSourceSnapshot, Device,
    TextSourceFrame, TextSourceFrameKind, TextSourceObserver,
};
const GIB: u64 = 1024 * 1024 * 1024;
const SNAPSHOT: &str = "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
/// The 384MiB harness covers one phase-owned extent: the comparator inventory
/// reports admission/observed/snapshot/committed/report maxima from actual
/// typed descriptors and buffers; they are never summed or guessed.
const ROOT_BYTES: u64 = 159_121_408;
const DIAGNOSTIC_HOST_BYTES: u64 = 384 * 1024 * 1024;
const RETAINED_SOURCE_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const RETAINED_CONTEXT_CONTROL_BYTES: u64 = 4 * 1024 * 1024;
struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}
struct Observed<'a, 'e> {
    expected: &'e mut Expected,
    repeat: usize,
    call: usize,
    layers: BTreeSet<usize>,
    proposed: BTreeSet<(String, usize)>,
    prior: BTreeSet<(String, usize)>,
    logits_words: usize,
    late: Option<&'a AtomicBool>,
    unknown: bool,
    deliberate: bool,
}
impl<'a, 'e> Observed<'a, 'e> {
    fn new(expected: &'e mut Expected, repeat: usize, call: usize) -> Proof<Self> {
        expected.begin(repeat, call)?;
        Ok(Self {
            expected,
            repeat,
            call,
            layers: BTreeSet::new(),
            proposed: BTreeSet::new(),
            prior: BTreeSet::new(),
            logits_words: 0,
            late: None,
            unknown: false,
            deliberate: false,
        })
    }
    fn finish(&mut self) -> Proof<(Value, (u32, u32))> {
        need(
            self.layers.len() == 64
                && self.proposed.len() == 128
                && self.prior.len() == if self.call == 0 { 96 } else { 128 }
                && self.logits_words == 248320,
            "actual full64/all128-domain planes/fullV observation",
        )?;
        let finished = self.expected.finish()?;
        need(
            finished["exact"] == true && finished["mismatches"] == 0,
            "independent original comparison exact BEFORE any natural commit",
        )?;
        let selection = self.expected.selection(self.repeat, self.call)?;
        Ok((finished, selection))
    }
}
impl TextSourceObserver for Observed<'_, '_> {
    fn completed(&mut self, frame: TextSourceFrame, actual: &[u8]) -> zerogravity::Result<()> {
        let words = actual.len() / 4;
        if frame.position != self.call
            || actual.is_empty()
            || !actual.len().is_multiple_of(4)
            || frame
                .offset_words
                .checked_add(words)
                .is_none_or(|n| n > frame.total_words)
            || actual
                .chunks_exact(4)
                .any(|b| !f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())).is_finite())
        {
            return Err(Error::State(
                "natural actual finite producer frame extent".into(),
            ));
        }
        match frame.kind {
            TextSourceFrameKind::Value {
                layer,
                stage: "layer_output",
            } => {
                if layer >= 64
                    || frame.total_words != 5120
                    || frame.offset_words != 0
                    || !self.layers.insert(layer)
                {
                    return Err(Error::State("natural actual complete layer once".into()));
                }
                if self.unknown && layer == 0 {
                    // Compare the real embedding+layer0 bytes against the fresh
                    // natural expectation BEFORE the deliberate armed refusal,
                    // and only refuse deliberately when those bytes actually match.
                    self.expected
                        .completed(frame, actual)
                        .map_err(|e| Error::State(format!("original natural comparison: {e}")))?;
                    if self
                        .expected
                        .active_mismatches()
                        .map_err(|e| Error::State(e.to_string()))?
                        != 0
                    {
                        return Err(Error::State(
                            "natural cached actual bytes differ from held original".into(),
                        ));
                    }
                    self.deliberate = true;
                    return Err(Error::State(
                        "deliberate natural cached armed producer refusal".into(),
                    ));
                }
            }
            TextSourceFrameKind::Value {
                stage: "logits", ..
            } => {
                if frame.total_words != 248320 || frame.offset_words != self.logits_words {
                    return Err(Error::State("natural fullV ordered observation".into()));
                }
                self.logits_words += words;
                if self.logits_words == 248320
                    && let Some(flag) = self.late
                {
                    flag.store(true, Ordering::Release);
                }
            }
            TextSourceFrameKind::State { slot, plane, prior }
                if frame.offset_words + words == frame.total_words =>
            {
                let set = if prior {
                    &mut self.prior
                } else {
                    &mut self.proposed
                };
                if !set.insert((format!("{slot:?}"), plane)) {
                    return Err(Error::State("natural actual state plane once".into()));
                }
            }
            _ => {}
        }
        self.expected
            .completed(frame, actual)
            .map_err(|e| Error::State(format!("original natural comparison: {e}")))
    }
}
fn completed(
    session: &mut Bf16TextSourceSession,
    seed: Option<u32>,
    expected: &mut Expected,
    repeat: usize,
    call: usize,
    late: Option<&AtomicBool>,
) -> Proof<(Value, (u32, u32))> {
    let before = session.binding();
    need(
        before.frontier() == call as u64,
        "observed proposal frontier equals independently bound call",
    )?;
    let mut observer = Observed::new(expected, repeat, call)?;
    observer.late = late;
    let pending = match seed {
        Some(id) => block_on(session.prepare_seed(id, 2, Some(&mut observer), None))?,
        None => block_on(session.prepare_next(Some(&mut observer), None))?,
    };
    let (observed, selection) = observer.finish()?;
    drop(observer);
    let selected = pending.selected()?;
    need(
        packet_matches(selected.0, selected.1, selection) && session.binding() == before,
        "GPU packet ID/maxbits equal independent original natural fullV BEFORE publication",
    )?;
    let output = session.commit_natural(pending)?;
    need(
        packet_matches(output.selected_id, output.max_bits, selection)
            && output.root == session.binding()
            && output.root.frontier() == before.frontier() + 1
            && output.root.root() == before.root() + 1,
        "one original natural root/feedback publication",
    )?;
    Ok((
        json!({"before":format!("{before:?}"),"after":format!("{:?}", output.root),
        "selected_id":output.selected_id,"max_bits":output.max_bits,"layers":64,
        "state_proposed_planes":128,"state_prior_planes":if call==0 {96} else {128},
        "fullV_words":248320,"original":observed,"dispatches":output.dispatches,
        "readback_bytes":output.terminal_readback_bytes,"capacity_exhausted":output.capacity_exhausted}),
        selection,
    ))
}
fn snapshot_domains(snapshot: &Bf16TextSourceSnapshot) -> Proof<Vec<(StateSlot, Vec<f32>)>> {
    let domains = block_on(snapshot.read_domains())?;
    let bytes = domains
        .iter()
        .map(|(_, values)| values.len() * 4)
        .sum::<usize>();
    let (mut conv, mut recurrent, mut kv) = (0usize, 0usize, 0usize);
    for (slot, _) in &domains {
        match slot {
            StateSlot::Conv(_) => conv += 1,
            StateSlot::Recurrent(_) => recurrent += 1,
            StateSlot::Kv(_) => kv += 1,
        }
    }
    need(
        domains.len() == 112
            && (conv, recurrent, kv) == (48, 48, 16)
            && bytes as u64 == ROOT_BYTES
            && domains
                .iter()
                .all(|(_, values)| values.iter().all(|v| v.is_finite())),
        "exact112 snapshot domains flatten to the actual 159121408B root",
    )?;
    Ok(domains)
}
fn exercise(
    candidate: Arc<Bf16TextSourceCandidate>,
    mode: &str,
    reference: &Arc<expected::Reference>,
) -> Proof<Value> {
    need(
        candidate.prepared_natural_available() && !candidate.trained_model_qualified(),
        "natural candidate has no trained/serving grant",
    )?;
    let mut a = block_on(candidate.session())?;
    let mut b = block_on(candidate.session())?;
    need(
        a.binding().session() != b.binding().session() && block_on(candidate.session()).is_err(),
        "two independent original sessions/third refusal",
    )?;
    let empty = a.binding();
    let cancel = AtomicBool::new(true);
    need(
        block_on(a.prepare_seed(11, 2, None, Some(&cancel))).is_err() && a.binding() == empty,
        "pre-cancel before natural effects",
    )?;
    need(
        block_on(a.prepare_seed(11, 3, None, None)).is_err() && a.binding() == empty,
        "wider unsupported temporal scope refuses",
    )?;
    let discarded = block_on(a.prepare_seed(11, 2, None, None))?;
    drop(discarded);
    need(
        a.binding() == empty && !candidate.physical_fenced(),
        "actual packet-only Ready discard preserves current root",
    )?;
    let mut main = Expected::new_shared(reference, true)?;
    let (first, first_selection) = completed(&mut a, Some(11), &mut main, 0, 0, None)?;
    let snapshot = a.snapshot()?;
    let untouched = b.binding();
    need(
        block_on(b.restore(&snapshot)).is_err() && b.binding() == untouched,
        "foreign original snapshot refuses",
    )?;
    if mode == "cached-unknown" {
        let before = a.binding();
        need(before.frontier() == 1, "cached observed proposal frontier")?;
        let mut observed = Observed::new(&mut main, 0, 1)?;
        observed.unknown = true;
        let result = block_on(a.prepare_next(Some(&mut observed), None));
        let deliberate = observed.deliberate;
        drop(observed);
        let (pending, words) = main.in_progress();
        need(
            matches!(result, Err(Error::State(ref why)) if why == "deliberate natural cached armed producer refusal")
                && deliberate
                && a.binding() == before
                && b.binding() == untouched
                && candidate.physical_fenced(),
            "actual cached unknown has no publication and fences original service",
        )?;
        let closure = main.finish_partial(&[(0, 0), (0, 1)])?;
        need(
            closure["exact_completed_prefix"] == true && closure["interrupted"] == true,
            "actual cached fault closes only its exact completed prefix",
        )?;
        return Ok(json!({"first":first,
            "selected_output_tuples":{"seed_input":11,
                "seed":[first["selected_id"].clone(),first["max_bits"].clone()],
                "next":Value::Null,"distinct_from_seed_input":true},
            "cached_unknown_root_unchanged":true,"published_cached_roots":0,
            "cached_expected_records":pending,"cached_expected_words":words,
            "cached_expected_closure":closure}));
    }
    let flag = AtomicBool::new(false);
    let late = {
        let mut late_expected = Expected::new_shared(reference, true)?;
        need(
            a.binding().frontier() == 1,
            "late observed proposal frontier",
        )?;
        let mut observed = Observed::new(&mut late_expected, 0, 1)?;
        observed.late = Some(&flag);
        let result = block_on(a.prepare_next(Some(&mut observed), Some(&flag)));
        need(
            result.is_err() && flag.load(Ordering::Acquire),
            "fullV completed late cancellation refuses publication",
        )?;
        let (late_scope, late_selection) = observed.finish()?;
        drop(observed);
        let late_closure = late_expected.finish_scoped(&[(0, 1)])?;
        need(
            a.binding().frontier() == 1 && !candidate.physical_fenced(),
            "late cancellation discards known Ready without publication/fence",
        )?;
        json!({"scope":late_scope,"scope_closure":late_closure,"selection":late_selection})
    };
    let (second, second_selection) = completed(&mut a, None, &mut main, 0, 1, None)?;
    need(
        a.natural_stopped() && block_on(a.prepare_next(None, None)).is_err(),
        "finite natural scope exhausted exactly",
    )?;
    let (peer, peer_selection) = completed(&mut b, Some(11), &mut main, 1, 0, None)?;
    let (peer_cached, peer_cached_selection) = completed(&mut b, None, &mut main, 1, 1, None)?;
    let full4 = main.after_natural([
        [first_selection, second_selection],
        [peer_selection, peer_cached_selection],
    ])?;
    block_on(a.restore(&snapshot))?;
    need(
        a.binding().frontier() == 1 && !a.natural_stopped(),
        "own snapshot restores current natural feedback",
    )?;
    let replay = {
        let mut replay_expected = Expected::new_shared(reference, true)?;
        let (replay, selection) = completed(&mut a, None, &mut replay_expected, 0, 1, None)?;
        need(
            selection == second_selection,
            "restamped original feedback reproduces cached selection",
        )?;
        let replay_closure = replay_expected.finish_scoped(&[(0, 1)])?;
        json!({"scope":replay,"scope_closure":replay_closure,"selection":selection})
    };
    block_on(a.reset())?;
    need(
        a.binding().frontier() == 0
            && b.binding().frontier() == 2
            && !a.natural_stopped()
            && block_on(a.prepare_next(None, None)).is_err(),
        "reset clears private feedback without changing peer",
    )?;
    // Ordinary observer=None pair after reset; expected numeric packet and diagnostic full state.
    let seed = block_on(a.prepare_seed(11, 2, None, None))?;
    let ordinary_seed_selection = seed.selected()?;
    need(
        ordinary_seed_selection == main.selection(0, 0)?,
        "ordinary seed16B packet equals independent original ID/maxbits",
    )?;
    let seed_out = a.commit_natural(seed)?;
    let next = block_on(a.prepare_next(None, None))?;
    let ordinary_selection = next.selected()?;
    need(
        ordinary_selection == main.selection(0, 1)?,
        "ordinary continuation16B packet equals independent original ID/maxbits",
    )?;
    let next_out = a.commit_natural(next)?;
    need(
        ordinary_seed_selection.0 != 11 && ordinary_selection.0 != 11,
        "seed input 11 is distinct from both selected output tuples",
    )?;
    main.close_ordinary()?;
    need(
        seed_out.terminal_readback_bytes == 16 && next_out.terminal_readback_bytes == 16,
        "ordinary pair reads status+actual12B packet only",
    )?;
    // Geometry and the complete inventory precede the one full-root allocation:
    // the expected frontier/capacity/112-domain/root extent is checked against
    // the actual binding, and the 2R + 2*3MiB + metadata peak must already fit
    // the reserved 384MiB harness before `read_domains` runs.
    let bound = a.binding();
    main.preflight_domains(1, bound.frontier(), 2)?;
    let ordinary_snapshot = a.snapshot()?;
    let domains = snapshot_domains(&ordinary_snapshot)?;
    let ordinary_committed = main.compare_committed(0, 1, &domains)?;
    drop(ordinary_snapshot);
    main.verify_held()?;
    Ok(
        json!({"first":first,"cached":second,"independent_peer":peer,
        "independent_peer_cached":peer_cached,"first_selection":first_selection,
        "cached_selection":second_selection,"peer_selection":peer_selection,
        "peer_cached_selection":peer_cached_selection,"late_cancel":late,"restore_replay":replay,
        "full4":full4,"ordinary_packet":{"seed":ordinary_seed_selection,"next":ordinary_selection,
        "seed_readback_bytes":seed_out.terminal_readback_bytes,
        "next_readback_bytes":next_out.terminal_readback_bytes},
        "selected_output_tuples":{"seed_input":11,
            "seed":ordinary_seed_selection,"next":ordinary_selection,
            "distinct_from_seed_input":true},
        "ordinary_committed":ordinary_committed,
        "actual_full64_calls":9,"source_capacity":2,"eos_text_policy_implemented":false,
        "diagnostic_committed_state_compared":true}),
    )
}
fn final_reader(
    candidate: Arc<Bf16TextSourceCandidate>,
    snapshot_case: bool,
    fault: bool,
    reference: &Arc<expected::Reference>,
) -> Proof<Value> {
    let mut session = block_on(candidate.session())?;
    let mut expected = Expected::new_shared(reference, true)?;
    let (first, selection) = completed(&mut session, Some(11), &mut expected, 0, 0, None)?;
    // Seed [(0,0)] scope closes exactly from held bytes before the final reader.
    let closure = expected.finish_scoped(&[(0, 0)])?;
    need(
        closure["exact"] == true && closure["mismatches"] == 0,
        "final natural call0 scope closes exactly before the last reader",
    )?;
    let receipt = if snapshot_case {
        let snapshot = session.snapshot()?;
        drop((session, candidate));
        snapshot.qualify_final_feedback_drop(fault)?
    } else {
        drop(candidate);
        session.qualify_final_feedback_drop(fault)?
    };
    need(
        receipt.lease_bytes == 12
            && receipt.aliases_inside_wrapper == 1
            && receipt.candidate_retired
            && receipt.release_hits == 1
            && receipt.reader_hits_before_native_retirement > 0,
        "true final12B reader releases/refuses BEFORE actual native retirement",
    )?;
    need(
        first["selected_id"].as_u64() != Some(11) && selection.0 != 11,
        "held seed input 11 is distinct from both final-path output tuples",
    )?;
    let seed_tuple = json!([first["selected_id"].clone(), first["max_bits"].clone()]);
    Ok(
        json!({"first":first,"selection":selection,"scope_closure":closure,
        "selected_output_tuples":{"seed_input":11,"seed":seed_tuple,
            "next":[selection.0,selection.1],
            "distinct_from_seed_input":true},
        "original_final_reader":{
            "wrapper":receipt.wrapper,"fault":receipt.fault,"source":receipt.source,
            "lease":receipt.lease,"service_generation":receipt.service_generation,
            "lease_bytes":receipt.lease_bytes,"aliases_inside_wrapper":receipt.aliases_inside_wrapper,
            "candidate_owners_before_drop":receipt.candidate_owners_before_drop,
            "model_owners_before_drop":receipt.model_owners_before_drop,
            "candidate_workspace_target_aliases":receipt.candidate_workspace_target_aliases,
            "candidate_retired":receipt.candidate_retired,"release_hits":receipt.release_hits,
            "native_retirement_attempts":receipt.native_retirement_attempts,
            "reader_hits_before_native_retirement":receipt.reader_hits_before_native_retirement,
            "native_modules_before":receipt.native_modules_before,
            "native_code_bytes_before":receipt.native_code_bytes_before,
            "native_host_bytes_before":receipt.native_host_bytes_before,
            "native_device_bytes_before":receipt.native_device_bytes_before,
            "native_admission_retained_after":receipt.native_admission_retained_after,
            "native_modules_after":receipt.native_modules_after,
            "original_service_retained_after":receipt.original_service_retained_after,
            "quarantined_handles_after":receipt.quarantined_handles_after}}),
    )
}
fn run() -> Proof<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    // GPU-free source serialization for the build receipt and grouped control:
    // the runtime itself writes the actual serde fixture shape, never Python.
    if args.first().is_some_and(|arg| arg == "--source-fixtures") {
        need(
            args.len() == 2,
            "usage: qwen27b_hip_natural --source-fixtures ABS_PATH",
        )?;
        let fixtures = PathBuf::from(&args[1]);
        need(
            fixtures.is_absolute() && !fixtures.exists(),
            "fresh absolute source fixture path",
        )?;
        write_new(&fixtures, &expected::natural_report_fixtures())?;
        return Ok(());
    }
    need(
        args.len() == 3 || args.len() == 5,
        "usage: qwen27b_hip_natural MODE ABS_FRESH_REPORT ABS_NATURAL_EXPECTED [EXTERNAL_RECEIPT_BYTES EXTERNAL_RECEIPT_SHA256]",
    )?;
    let mode = args[0].as_str();
    let (snapshot_case, fault, final_case) = match mode {
        "healthy" => (false, false, false),
        "cached-unknown" => (false, true, false),
        "session-known" => (false, false, true),
        "session-release-fault" => (false, true, true),
        "snapshot-known" => (true, false, true),
        "snapshot-release-fault" => (true, true, true),
        _ => return Err("six explicit natural qualification diagnostics only".into()),
    };
    let report = PathBuf::from(&args[1]);
    let expected_root = PathBuf::from(&args[2]);
    need(
        report.is_absolute() && !report.exists() && expected_root.is_absolute(),
        "fresh absolute report/natural expected directory",
    )?;
    let refusal = |reason: &str| {
        write_new(
            &report,
            &json!({"schema":"zg.qwen27b-natural-two-position-diagnostic.v2",
            "status":"REFUSED_NATURAL_CAPTURE_ABSENT","mode":mode,"snapshot":SNAPSHOT,
            "join":Value::Null,"failure":reason,"model_loaded":false,"device_work":false,
            "natural_accepted_pin":Value::Null,
            "mathematical_original_reference_compared":false,
            "held_reference_fullv_and_join_verified":false,
            "ordinary_full_state_compared":false,"full4_compared":false,
            "trained_native_natural_serving_or_performance_qualified":false}),
        )
    };
    // CPU-only held-receipt join evidence: the extent/SHA are an external
    // operand from the trusted launch boundary, never measured here. While the
    // authority is absent/awaiting or the receipt is absent, refuse with its own
    // report before any context/arbiter/model allocation and never fall back to
    // a teacher receipt.
    if args.len() != 5 {
        let reason = "accepted original natural capture absent: run the frozen natural tool, then validate --receipt";
        refusal(reason)?;
        return Err(reason.into());
    }
    let external_bytes = match args[3].parse::<u64>() {
        Ok(bytes) if bytes > 0 && bytes <= 1024 * 1024 => bytes,
        _ => {
            let reason = "accepted original natural capture absent: run the frozen natural tool, then validate --receipt";
            refusal(reason)?;
            return Err(reason.into());
        }
    };
    let external_sha = args[4].as_str();
    // Prove the pinned metadata/admission layout with fallible arithmetic and
    // reserve the original 384MiB host harness BEFORE the presence path join.
    // A missing external pin still refuses before GPU context/model work.
    let harness_bytes = expected::natural_admission_preflight()?;
    let arbiter = ResourceArbiter::process_shared();
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let physical_key = PhysicalDeviceKey::new("pci", "0000:c6:00.0")?;
    let harness = arbiter.admit(
        &host_key,
        ResourceOwner::new("Qwen27B-natural", "bounded-actual-frames-report")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: harness_bytes,
        }],
    )?;
    let present = expected::natural_receipt_present(&expected_root, (external_bytes, external_sha));
    if present.is_err() {
        let reason = "accepted original natural capture absent: run the frozen natural tool, then validate --receipt";
        refusal(reason)?;
        return Err(reason.into());
    }
    let context = Qwen27bGpuContext::new()?;
    let external_pin = (external_bytes, external_sha);
    let (reference, inventory) = expected::natural_reference(&expected_root, external_pin)?;
    need(
        inventory.admission_bytes <= harness_bytes,
        "actual admission high-water inside the proved 384MiB harness",
    )?;
    let join = reference.join_evidence(None)?;
    if join["independent_original_natural_trajectory"] != true {
        let reason = join["reason"]
            .as_str()
            .unwrap_or("held natural join unavailable")
            .to_owned();
        write_new(
            &report,
            &json!({"schema":"zg.qwen27b-natural-two-position-diagnostic.v2",
            "status":"REFUSED_HELD_NATURAL_JOIN_UNAVAILABLE","mode":mode,"snapshot":SNAPSHOT,
            "join":join,"failure":reason,"model_loaded":false,"device_work":false,
            "natural_accepted_pin":{"bytes":external_bytes,"sha256":external_sha},
            "mathematical_original_reference_compared":false,
            "held_reference_fullv_and_join_verified":true,
            "ordinary_full_state_compared":false,"full4_compared":false,
            "trained_native_natural_serving_or_performance_qualified":false}),
        )?;
        return Err(reason.into());
    }
    // One shared immutable reference; each comparator owns only its progress.
    // The per-phase maximum (never a guessed constant, never their sum) must
    // fit the 384MiB harness before any model allocation.
    need(
        inventory.peak_bytes <= DIAGNOSTIC_HOST_BYTES,
        "actual phase-owned diagnostic inventory fits the 384MiB host harness",
    )?;
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * GIB))?;
    let snapshot = Arc::new(store.open_snapshot(SNAPSHOT)?);
    let loaded = Qwen27bGpuLoaded::open(snapshot.clone(), context.clone(), None)?;
    let candidate =
        block_on(loaded.load_hip_natural_qualification(Device::Hip(Default::default()), None))?;
    let source_inventory = format!("{:?}", candidate.inventory());
    let witness = candidate.retention_witness();
    let result = if final_case {
        final_reader(candidate.clone(), snapshot_case, fault, &reference)
    } else {
        exercise(candidate.clone(), mode, &reference)
    };
    drop(candidate);
    let retention = witness.observe()?;
    let close = loaded.clone().close(None);
    drop((loaded, snapshot, store, context));
    let physical = arbiter.snapshot(&physical_key)?;
    let host = arbiter.snapshot(&host_key)?;
    // Healthy release must show no original retention and the exact held
    // 2GiB+384MiB host domain. A fault is bounded, unfenced and source-backed:
    // it may keep the actual native modules/service/quarantine, so HOST is at
    // most the original 16GiB reserve and at least the held model remainder plus
    // the retained module HOST (final receipt when a wrapper exists, else the
    // weak/value-only observation) and the context/metadata controls.
    let receipt = result
        .as_ref()
        .ok()
        .and_then(|v| v.get("original_final_reader"))
        .cloned();
    let retained_native_host = receipt
        .as_ref()
        .and_then(|r| r["native_host_bytes_before"].as_u64())
        .unwrap_or(retention.native_host_bytes as u64);
    let retained_evidence = receipt
        .as_ref()
        .map(|r| {
            r["native_admission_retained_after"] == true
                && r["original_service_retained_after"] == true
                && r["quarantined_handles_after"]
                    .as_u64()
                    .is_some_and(|n| n > 0)
        })
        .unwrap_or(
            retention.native_admission_retained
                && retention.original_service_retained
                && retention.quarantined_handles > 0,
        );
    let host_ok = if fault {
        !host.fenced
            && host.reserved <= 16 * GIB
            && host.reserved
                >= 2 * GIB
                    + RETAINED_SOURCE_METADATA_BYTES
                    + RETAINED_CONTEXT_CONTROL_BYTES
                    + retained_native_host
            && retained_evidence
    } else {
        !host.fenced
            && host.reserved == 2 * GIB + DIAGNOSTIC_HOST_BYTES
            && !retention.native_admission_retained
    };
    let expected = close.is_err() == fault
        && physical.fenced == fault
        && physical.reserved == if fault { 16 * GIB } else { 4 * GIB }
        && host_ok;
    // Final before-report lease recheck of the exact five original held FDs.
    reference.verify_lease()?;
    // The disposition is not a mode-string label: a final wrapper must show a
    // released known root, a fault must show the actual retained native modules
    // and HOST bytes through the crate's typed retention join.
    let typed_final_join = if fault {
        retention.retention_agrees(false)
    } else if final_case {
        retention.retention_agrees(true)
    } else {
        true
    };
    need(
        typed_final_join,
        "typed final native retention mode agrees with the actual observation",
    )?;
    let retains_source = mode.ends_with("fault") || mode == "cached-unknown";
    let lease = reference.lease_evidence();
    write_new(
        &report,
        &json!({"schema":"zg.qwen27b-natural-two-position-diagnostic.v2",
        "status":if result.is_ok() && expected {"PASS_DIAGNOSTIC_NATURAL_UNQUALIFIED"} else {"REFUSED_UNQUALIFIED"},
        "mode":mode,"retention_mode":if retains_source {"unknown"} else {"known"},"snapshot":SNAPSHOT,
        "typed_final_retention_join":typed_final_join,
        "inventory":source_inventory.clone(),"actual":result.as_ref().ok(),
        "failure":result.as_ref().err().map(ToString::to_string),"source_close_refused":close.is_err(),
        "physical_reserved":physical.reserved,"physical_fenced":physical.fenced,"host_reserved":host.reserved,"host_fenced":host.fenced,
        "diagnostic_host_bytes":DIAGNOSTIC_HOST_BYTES,
        "natural_accepted_pin":{"bytes":external_bytes,"sha256":external_sha},
        "diagnostic_inventory":{"root_bytes":inventory.root_bytes,
            "readback_window_bytes":inventory.readback_window_bytes,
            "domain_metadata_bytes":inventory.domain_metadata_bytes,
            "reference_bytes":inventory.reference_bytes,
            "manifest_input_bytes":inventory.manifest_input_bytes,
            "manifest_tree_bytes":inventory.manifest_tree_bytes,
            "admission_bytes":inventory.admission_bytes,
            "observed_bytes":inventory.observed_bytes,
            "snapshot_bytes":inventory.snapshot_bytes,
            "committed_bytes":inventory.committed_bytes,
            "report_bytes":inventory.report_bytes,
            "peak_bytes":inventory.peak_bytes,"host_bytes":DIAGNOSTIC_HOST_BYTES},
        "reference_lease":lease,
        "retained_source_context_bytes":if retains_source {RETAINED_CONTEXT_CONTROL_BYTES} else {0},
        "retained_source_metadata_bytes":if retains_source {RETAINED_SOURCE_METADATA_BYTES} else {0},
        "retained_native_inventory":if retains_source {Value::String(source_inventory)} else {Value::Null},
        "released_host_assertion":if retains_source {Value::Null} else {json!(true)},
        "mathematical_original_reference_compared":result.is_ok() && mode == "healthy",
        "ordinary_full_state_compared":result.is_ok() && mode == "healthy",
        "full4_compared":result.is_ok() && mode == "healthy",
        "lifecycle_cases_qualified":false,
        "trained_native_natural_serving_or_performance_qualified":false}),
    )?;
    result?;
    need(expected, "original source/native/model/context disposition")?;
    drop(harness);
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("QWEN27B_NATURAL_REFUSED: {error}");
        std::process::exit(1);
    }
}
