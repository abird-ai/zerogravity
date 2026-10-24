//! Actual two-position natural-route diagnostics; independent original math pending.
#[allow(dead_code)]
#[path = "gemma4_cpu_matrix/io.rs"]
mod io;
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
    accelerator::program::Bf16TextGreedy,
    cache::{CacheOptions, CacheStore},
    qwen27b_gpu::{Qwen27bGpuContext, Qwen27bGpuLoaded},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_gpu::{
    Device, Qwen27bSourceGpuCandidate, Qwen27bSourceGpuSession, QwenSourceFrame,
    QwenSourceFrameKind, QwenSourceObserver,
};
const GIB: u64 = 1024 * 1024 * 1024;
const SNAPSHOT: &str = "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
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
struct Observed<'a> {
    position: usize,
    layers: BTreeSet<usize>,
    proposed: BTreeSet<(String, usize)>,
    prior: BTreeSet<(String, usize)>,
    logits_words: usize,
    best: Option<(u32, u32)>,
    late: Option<&'a AtomicBool>,
    unknown: bool,
}
impl<'a> Observed<'a> {
    fn new(position: usize) -> Self {
        Self {
            position,
            layers: BTreeSet::new(),
            proposed: BTreeSet::new(),
            prior: BTreeSet::new(),
            logits_words: 0,
            best: None,
            late: None,
            unknown: false,
        }
    }
    fn finish(&self) -> Proof<(u32, u32)> {
        need(
            self.layers.len() == 64
                && self.proposed.len() == 128
                && self.prior.len() == if self.position == 0 { 96 } else { 128 }
                && self.logits_words == 248320,
            "actual full64/all112-domain planes/fullV observation",
        )?;
        self.best
            .ok_or_else(|| "actual fullV greedy observation absent".into())
    }
}
impl QwenSourceObserver for Observed<'_> {
    fn completed(&mut self, frame: QwenSourceFrame, actual: &[u8]) -> zerogravity::Result<()> {
        let words = actual.len() / 4;
        if frame.position != self.position
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
            QwenSourceFrameKind::Value {
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
                    return Err(Error::State(
                        "deliberate natural cached armed producer refusal".into(),
                    ));
                }
            }
            QwenSourceFrameKind::Value {
                stage: "logits", ..
            } => {
                if frame.total_words != 248320 || frame.offset_words != self.logits_words {
                    return Err(Error::State("natural fullV ordered observation".into()));
                }
                for (i, bytes) in actual.chunks_exact(4).enumerate() {
                    let candidate = (
                        (frame.offset_words + i) as u32,
                        u32::from_le_bytes(bytes.try_into().unwrap()),
                    );
                    Bf16TextGreedy::prefer(candidate, candidate)?;
                    if match self.best {
                        Some(prior) => Bf16TextGreedy::prefer(candidate, prior)?,
                        None => true,
                    } {
                        self.best = Some(candidate);
                    }
                }
                self.logits_words += words;
                if self.logits_words == 248320
                    && let Some(flag) = self.late
                {
                    flag.store(true, Ordering::Release);
                }
            }
            QwenSourceFrameKind::State { slot, plane, prior }
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
        Ok(())
    }
}
fn completed(session: &mut Qwen27bSourceGpuSession, seed: Option<u32>) -> Proof<Value> {
    let before = session.binding();
    let mut observer = Observed::new(before.frontier() as usize);
    let pending = match seed {
        Some(id) => block_on(session.prepare_seed(id, 2, Some(&mut observer), None))?,
        None => block_on(session.prepare_next(Some(&mut observer), None))?,
    };
    let selected = observer.finish()?;
    need(
        pending.selected()? == selected && session.binding() == before,
        "GPU packet equals complete actual fullV source-numeric argmax BEFORE publication",
    )?;
    let output = session.commit_natural(pending)?;
    need(
        output.selected_id == selected.0
            && output.max_bits == selected.1
            && output.root == session.binding()
            && output.root.frontier() == before.frontier() + 1
            && output.root.root() == before.root() + 1,
        "one original natural root/feedback publication",
    )?;
    Ok(
        json!({"before":format!("{before:?}"),"after":format!("{:?}", output.root),"selected_id":output.selected_id,
        "max_bits":output.max_bits,"layers":observer.layers.len(),"state_proposed_planes":observer.proposed.len(),
        "state_prior_planes":observer.prior.len(),"fullV_words":observer.logits_words,"dispatches":output.dispatches,
        "readback_bytes":output.terminal_readback_bytes,"capacity_exhausted":output.capacity_exhausted}),
    )
}
fn exercise(candidate: Arc<Qwen27bSourceGpuCandidate>, mode: &str) -> Proof<Value> {
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
    let first = completed(&mut a, Some(11))?;
    let snapshot = a.snapshot()?;
    let untouched = b.binding();
    need(
        block_on(b.restore(&snapshot)).is_err() && b.binding() == untouched,
        "foreign original snapshot refuses",
    )?;
    if mode == "cached-unknown" {
        let before = a.binding();
        let mut observed = Observed::new(1);
        observed.unknown = true;
        let result = block_on(a.prepare_next(Some(&mut observed), None));
        need(
            matches!(result, Err(Error::State(ref why)) if why == "deliberate natural cached armed producer refusal")
                && a.binding() == before
                && b.binding() == untouched
                && candidate.physical_fenced(),
            "actual cached unknown has no publication and fences original service",
        )?;
        return Ok(
            json!({"first":first,"cached_unknown_root_unchanged":true,"completed_cached_layers":observed.layers.len(),"published_cached_roots":0}),
        );
    }
    let flag = AtomicBool::new(false);
    let mut observed = Observed::new(1);
    observed.late = Some(&flag);
    need(
        block_on(a.prepare_next(Some(&mut observed), Some(&flag))).is_err()
            && observed.finish().is_ok()
            && a.binding().frontier() == 1
            && !candidate.physical_fenced(),
        "fullV completed late cancellation discards known Ready without publication/fence",
    )?;
    let second = completed(&mut a, None)?;
    need(
        a.natural_stopped() && block_on(a.prepare_next(None, None)).is_err(),
        "finite natural scope exhausted exactly",
    )?;
    let peer = completed(&mut b, Some(11))?;
    let peer_cached = completed(&mut b, None)?;
    block_on(a.restore(&snapshot))?;
    need(
        a.binding().frontier() == 1 && !a.natural_stopped(),
        "own snapshot restores current natural feedback",
    )?;
    let replay = block_on(a.prepare_next(None, None))?;
    need(
        replay.selected()?
            == (
                second["selected_id"].as_u64().unwrap() as u32,
                second["max_bits"].as_u64().unwrap() as u32,
            ),
        "restamped original feedback reproduces cached selection",
    )?;
    let packet_only = a.commit_natural(replay)?;
    need(
        packet_only.terminal_readback_bytes == 16,
        "ordinary continuation reads status+actual12B packet only",
    )?;
    let peer_root = b.binding();
    block_on(a.reset())?;
    need(
        a.binding().frontier() == 0
            && b.binding() == peer_root
            && !a.natural_stopped()
            && block_on(a.prepare_next(None, None)).is_err(),
        "reset clears private feedback without changing peer",
    )?;
    Ok(
        json!({"first":first,"cached":second,"independent_peer":peer,"independent_peer_cached":peer_cached,"restored_cached_packet_readback_bytes":packet_only.terminal_readback_bytes,
        "actual_full64_calls":7,"source_capacity":2,"eos_text_policy_implemented":false,"mathematical_reference_compared":false}),
    )
}
fn final_reader(
    candidate: Arc<Qwen27bSourceGpuCandidate>,
    snapshot_case: bool,
    fault: bool,
) -> Proof<Value> {
    let mut session = block_on(candidate.session())?;
    let first = completed(&mut session, Some(11))?;
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
    Ok(json!({"first":first,"original_final_reader":format!("{receipt:?}")}))
}
fn run() -> Proof<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    need(
        args.len() == 2,
        "usage: qwen27b_hip_natural MODE ABS_FRESH_REPORT",
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
    need(
        report.is_absolute() && !report.exists(),
        "fresh absolute report",
    )?;
    let context = Qwen27bGpuContext::new()?;
    let arbiter = ResourceArbiter::process_shared();
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let physical_key = PhysicalDeviceKey::new("pci", "0000:c6:00.0")?;
    let harness = arbiter.admit(
        &host_key,
        ResourceOwner::new("Qwen27B-natural", "bounded-actual-frames-report")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 64 * 1024 * 1024,
        }],
    )?;
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * GIB))?;
    let snapshot = Arc::new(store.open_snapshot(SNAPSHOT)?);
    let loaded = Qwen27bGpuLoaded::open(snapshot.clone(), context.clone(), None)?;
    let candidate =
        block_on(loaded.load_hip_natural_qualification(Device::Hip(Default::default()), None))?;
    let inventory = format!("{:?}", candidate.inventory());
    let result = if final_case {
        final_reader(candidate, snapshot_case, fault)
    } else {
        exercise(candidate, mode)
    };
    let close = loaded.clone().close(None);
    drop((loaded, snapshot, store, context));
    let physical = arbiter.snapshot(&physical_key)?;
    let host = arbiter.snapshot(&host_key)?;
    let expected = if fault {
        close.is_err()
            && physical.fenced
            && physical.reserved == 16 * GIB
            && host.reserved > 2 * GIB
    } else {
        close.is_ok()
            && !physical.fenced
            && physical.reserved == 4 * GIB
            && !host.fenced
            && host.reserved == 2 * GIB + 64 * 1024 * 1024
    };
    write_new(
        &report,
        &json!({"schema":"zg.qwen27b-natural-two-position-diagnostic.v1",
        "status":if result.is_ok() && expected {"PASS_DIAGNOSTIC_NATURAL_UNQUALIFIED"} else {"REFUSED_UNQUALIFIED"},
        "mode":mode,"snapshot":SNAPSHOT,"inventory":inventory,"actual":result.as_ref().ok(),
        "failure":result.as_ref().err().map(ToString::to_string),"source_close_refused":close.is_err(),
        "physical_reserved":physical.reserved,"physical_fenced":physical.fenced,"host_reserved":host.reserved,"host_fenced":host.fenced,
        "mathematical_original_reference_compared":false,"trained_native_natural_serving_or_performance_qualified":false}),
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
