use super::{
    expected::{Expected, Matrix},
    io::*,
};
use serde_json::{Value, json};
use std::{
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
    cache::{CacheOptions, CacheStore},
    gemma4_cpu::{Gemma4CpuContext, Gemma4CpuInput, Gemma4CpuLoaded},
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter},
};
use zerogravity_gpu::{Device, Gemma4Bf16GpuCandidate, Gemma4Bf16GpuSession, GemmaTextHostOwners};
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
    let mut cx = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}
fn binding(root: zerogravity::session_root::RootBinding) -> Value {
    json!({"session":root.session().get(),"root":root.root(),"root_epoch":root.session_epoch(),"frontier":root.frontier(),"source_bound":root.source().is_bound()})
}
/// Completed-callback counters for one proposal: prior prefixes, final planes.
#[derive(Default)]
struct Observation {
    frontier: usize,
    prior: usize,
    final_planes: [bool; 30],
}
impl Observation {
    fn planes(&self) -> usize {
        self.final_planes.iter().filter(|v| **v).count()
    }
}
struct Tracing<'a> {
    inner: &'a mut Expected,
    observed: &'a mut Observation,
}
impl zerogravity_gpu::GemmaTextObserver for Tracing<'_> {
    fn completed_producer(
        &mut self,
        p: usize,
        o: &zerogravity::gemma4::text::Gemma4Bf16Observation,
        b: &[u8],
    ) -> zerogravity::Result<()> {
        zerogravity_gpu::GemmaTextObserver::completed_producer(self.inner, p, o, b)
    }
    fn completed_kv(
        &mut self,
        prior: bool,
        d: zerogravity::accelerator::program::KvDomain,
        v: bool,
        active: usize,
        b: &[u8],
    ) -> zerogravity::Result<()> {
        zerogravity_gpu::GemmaTextObserver::completed_kv(self.inner, prior, d, v, active, b)?;
        if prior && active > 0 {
            self.observed.prior += 1;
        } else if !prior && active == self.observed.frontier {
            self.observed.final_planes[d.0 * 2 + usize::from(v)] = true;
        }
        Ok(())
    }
}
/// One diagnostic proposal against the immutable expected cell. The original
/// expected report owns call/exact/mismatched/boundary/frontier/packet fields;
/// only the diagnostic-only counters are added here.
fn compared(
    e: &mut Expected,
    call: usize,
    actual: &zerogravity_gpu::InferenceReadback,
    before: zerogravity::session_root::RootBinding,
    observed: &Observation,
    phase: &str,
    schedule: usize,
    sequence: usize,
) -> Proof<Value> {
    let name = e.cell.request.name.clone();
    let call_ids = e.cell.calls[call].input_ids.clone();
    let base = e.cell.calls[call].base;
    let max_bits = actual
        .device_token
        .as_ref()
        .ok_or("actual completed private packet")?
        .max_bits;
    let mut report = e.completed(actual, before)?;
    let object = report
        .as_object_mut()
        .ok_or("actual completed proposal report")?;
    object.insert("case".into(), json!(name));
    object.insert("phase".into(), json!(phase));
    object.insert("call".into(), json!(call));
    object.insert("schedule".into(), json!(schedule));
    object.insert("sequence".into(), json!(sequence));
    object.insert("input_ids".into(), json!(call_ids));
    object.insert("base".into(), json!(base));
    object.insert("source_bound".into(), json!(true));
    object.insert("unique_order".into(), json!(true));
    object.insert("prior_kv_comparisons".into(), json!(observed.prior));
    object.insert("final_kv_planes".into(), json!(observed.planes()));
    object.insert("full_vocab_words".into(), json!(262144));
    object.insert("max_bits".into(), json!(max_bits));
    Ok(report)
}
fn completed(
    s: &mut Gemma4Bf16GpuSession,
    e: &mut Expected,
    call: usize,
    events: &mut Vec<Value>,
) -> Proof<bool> {
    let base = e.cell.calls[call].base;
    let frontier = e.cell.calls[call].frontier;
    let selected = e.cell.calls[call].selected_id;
    let eos = e.cell.calls[call].eos;
    let name = e.cell.request.name.clone();
    let call_ids = e.cell.calls[call].input_ids.clone();
    e.begin(call);
    let before = s.binding();
    let mut observed = Observation {
        frontier,
        ..Observation::default()
    };
    let pending = {
        let mut tracing = Tracing {
            inner: &mut *e,
            observed: &mut observed,
        };
        if call == 0 {
            block_on(s.prepare_prompt(8, Some(&mut tracing), None))?
        } else {
            block_on(s.prepare_next(Some(&mut tracing), None))?
        }
    };
    need(
        pending.readback().device_token.is_some()
            && observed.planes() == 30
            && (observed.prior > 0 || base == 0),
        "actual completed proposal observed all15 domains",
    )?;
    let sequence = events
        .iter()
        .filter(|v| v["kind"].as_str() == Some("actual-completed-proposal"))
        .count();
    let comparison = compared(
        e,
        call,
        pending.readback(),
        before,
        &observed,
        "proposal",
        call,
        sequence,
    )?;
    let exact = comparison["exact"] == true;
    events.push(
        json!({"kind":"actual-completed-proposal","case":name,"phase":"proposal",
                       "call":call,"input_ids":call_ids,"before":binding(before),
                       "comparison":comparison}),
    );
    if !exact {
        drop(pending);
        need(
            s.binding() == before,
            "fatal numerical refusal preserves original root/private feedback",
        )?;
        return Ok(false);
    }
    need(
        s.binding() == before,
        "exact completed Ready has not published root/private feedback",
    )?;
    let receipt = s.commit(pending)?;
    need(
        receipt.root == s.binding()
            && receipt.root.root() == before.root() + 1
            && receipt.root.frontier() == frontier as u64
            && receipt.selected_id == selected
            && receipt.eos == eos,
        "complete source/state/private packet join before publication",
    )?;
    events.push(json!({"kind":"actual-committed-call","case":name,"call":call,"selected_id":receipt.selected_id,"eos":receipt.eos,"root":binding(receipt.root)}));
    Ok(true)
}
fn trajectory(
    s: &mut Gemma4Bf16GpuSession,
    e: &mut Expected,
    start: usize,
    events: &mut Vec<Value>,
    decoder: &Gemma4CpuInput,
) -> Proof<bool> {
    let name = e.cell.request.name.clone();
    for call in start..e.cell.calls.len() {
        if !completed(s, e, call, events)? {
            e.after()?;
            return Ok(false);
        }
    }
    need(
        s.stopped(),
        "genuine EOS/8call stop without extra forced token",
    )?;
    let mut generated = Vec::with_capacity(8);
    for call in 0..e.cell.calls.len() {
        let event = events
            .iter()
            .rev()
            .find(|v| {
                v["kind"].as_str() == Some("actual-committed-call")
                    && v["case"].as_str() == Some(name.as_str())
                    && v["call"].as_u64() == Some(call as u64)
            })
            .ok_or("actual committed output record")?;
        if event["eos"] != true {
            generated.push(u32::try_from(
                event["selected_id"].as_u64().ok_or("actual generated ID")?,
            )?);
        }
    }
    let text = decoder.decode_output(&generated, true, None)?;
    events.push(json!({"kind":"source-ID-decoded-output","case":name,"ids":generated,"text":text.as_str(),"sequence_decode_only":true}));
    e.after()?;
    Ok(true)
}
struct BorrowedObserver<'a, 'b>(&'a std::cell::RefCell<&'b mut Expected>);
impl zerogravity_gpu::GemmaTextObserver for BorrowedObserver<'_, '_> {
    fn completed_producer(
        &mut self,
        p: usize,
        o: &zerogravity::gemma4::text::Gemma4Bf16Observation,
        b: &[u8],
    ) -> zerogravity::Result<()> {
        zerogravity_gpu::GemmaTextObserver::completed_producer(&mut **self.0.borrow_mut(), p, o, b)
    }
    fn completed_kv(
        &mut self,
        p: bool,
        d: zerogravity::accelerator::program::KvDomain,
        v: bool,
        a: usize,
        b: &[u8],
    ) -> zerogravity::Result<()> {
        zerogravity_gpu::GemmaTextObserver::completed_kv(&mut **self.0.borrow_mut(), p, d, v, a, b)
    }
}
/// Pre-cancel, completed-Ready Drop and post-completion refusal/late-cancel on
/// the ORIGINAL session: no publication, no root or private-feedback change.
fn cancelled_and_discarded(
    s: &mut Gemma4Bf16GpuSession,
    e: &mut Expected,
    loaded: &Arc<Gemma4CpuLoaded>,
    events: &mut Vec<Value>,
) -> Proof<bool> {
    let before = s.binding();
    let flag = AtomicBool::new(true);
    e.begin(0);
    need(
        matches!(block_on(s.prepare_prompt(8,Some(&mut *e),Some(&flag))),Err(Error::State(why))if why=="Gemma source-bound GPU call cancelled")
            && s.binding() == before,
        "pre-cancel has no root/private feedback change",
    )?;
    events.push(json!({"kind":"pre-cancel","root":binding(before)}));
    e.begin(0);
    let pending = block_on(s.prepare_prompt(8, Some(&mut *e), None))?;
    need(
        s.binding() == before,
        "completed Ready has not published root/private feedback",
    )?;
    let report = e.completed(pending.readback(), before).ok();
    let exact = report.as_ref().is_some_and(|r| r["exact"] == true);
    drop(pending);
    need(
        s.binding() == before,
        "actual Ready pending Drop cannot publish root/private feedback",
    )?;
    events.push(json!({"kind":"completed-Ready-drop","exact":exact,"comparison":report,"root":binding(before)}));
    if !exact {
        return Ok(false);
    }
    for cancel in [false, true] {
        flag.store(false, Ordering::Release);
        e.begin(0);
        let observed = std::cell::RefCell::new(&mut *e);
        let mut observer = BorrowedObserver(&observed);
        let mut comparison: Option<Value> = None;
        let mut exact = false;
        let result = block_on(
            s.infer_prompt(8, Some(&mut observer), Some(&flag), |actual| {
                let report = observed
                    .borrow_mut()
                    .completed(actual, before)
                    .map_err(|e| Error::State(e.to_string()))?;
                exact = report["exact"] == true;
                comparison = Some(report);
                if !exact {
                    return Err(Error::State(
                        "fatal complete Gemma numerical mismatch".into(),
                    ));
                }
                if cancel {
                    flag.store(true, Ordering::Release);
                    Ok(())
                } else {
                    Err(Error::State("deliberate completed Gemma refusal".into()))
                }
            }),
        );
        need(
            s.binding() == before && loaded.clone().close(None).is_err(),
            "actual completed refusal/cancel preserves original root and authentic managed-source guard",
        )?;
        let kind = if cancel {
            "actual-completed-late-cancel"
        } else {
            "actual-completed-refusal"
        };
        events.push(json!({"kind":kind,"comparison":comparison,"root":binding(before),"error":result.as_ref().err().map(ToString::to_string)}));
        if !exact {
            return Ok(false);
        }
        let expected = if cancel {
            "Gemma source-bound GPU call cancelled"
        } else {
            "deliberate completed Gemma refusal"
        };
        need(
            matches!(result,Err(Error::State(why))if why==expected),
            "original actual completed callback/cancel disposition",
        )?;
    }
    Ok(true)
}
struct UnknownObserver {
    completed: usize,
    stage: &'static str,
}
impl zerogravity_gpu::GemmaTextObserver for UnknownObserver {
    fn completed_producer(
        &mut self,
        _: usize,
        o: &zerogravity::gemma4::text::Gemma4Bf16Observation,
        _: &[u8],
    ) -> zerogravity::Result<()> {
        if o.stage == self.stage {
            self.completed += 1;
            return Err(Error::State(
                "deliberate actual armed Gemma producer refusal".into(),
            ));
        }
        Ok(())
    }
    fn completed_kv(
        &mut self,
        _: bool,
        _: zerogravity::accelerator::program::KvDomain,
        _: bool,
        _: usize,
        _: &[u8],
    ) -> zerogravity::Result<()> {
        Ok(())
    }
}
/// Resident PLE row page evidence for every real call of every real case, plus
/// three full traversals of one prompt: ordinary, instrumented all-hit and
/// ordinary observer-free hit. Every completed counter precedes its snapshot.
fn resident_rows(
    candidate: &Arc<Gemma4Bf16GpuCandidate>,
    matrix: &Matrix,
    loaded: &Arc<Gemma4CpuLoaded>,
    ids: &[Arc<[u32]>],
    events: &mut Vec<Value>,
) -> Proof<bool> {
    for (index, input_ids) in ids.iter().enumerate() {
        let mut session = block_on(candidate.session(input_ids.clone()))?;
        let expected = matrix.cell(index, loaded.source_identity())?;
        for (call, reference) in expected.cell.calls.iter().enumerate() {
            let before = session.binding();
            let traffic = block_on(candidate.traffic())?;
            let pending = if call == 0 {
                block_on(session.prepare_prompt(8, None, None))?
            } else {
                block_on(session.prepare_next(None, None))?
            };
            expected.normal_packet(call, pending.readback(), before)?;
            let packet = pending
                .readback()
                .device_token
                .as_ref()
                .ok_or("normal actual private packet")?;
            let (actual_id, actual_bits) = (packet.id, packet.max_bits);
            let after = block_on(candidate.traffic())?;
            let uploads = after
                .selected_row_uploads
                .checked_sub(traffic.selected_row_uploads)
                .ok_or("normal selected row upload underflow")?;
            let reuses = after
                .selected_row_reuses
                .checked_sub(traffic.selected_row_reuses)
                .ok_or("normal selected row reuse underflow")?;
            let avoided = after
                .selected_row_avoided_upload_bytes
                .checked_sub(traffic.selected_row_avoided_upload_bytes)
                .ok_or("normal selected row avoided underflow")?;
            let total_value_h2d = after
                .value_upload_bytes
                .checked_sub(traffic.value_upload_bytes)
                .ok_or("normal value upload underflow")?;
            need(
                after.producer_readback_bytes == traffic.producer_readback_bytes
                    && after.kv_diagnostic_readback_bytes == traffic.kv_diagnostic_readback_bytes
                    && uploads
                        .checked_add(reuses)
                        .ok_or("normal row counter overflow")?
                        == reference.input_ids.len() as u64
                    && total_value_h2d
                        == uploads
                            .checked_mul(17_920)
                            .ok_or("normal row upload bytes overflow")?
                    && avoided
                        == reuses
                            .checked_mul(17_920)
                            .ok_or("normal row avoided bytes overflow")?
                    && after.resident_weight_bytes == 4_558_946_304,
                "normal no producer/KV D2H or repeated selected weight H2D; PLE row only",
            )?;
            need(
                session.binding() == before,
                "ordinary completed Ready has not published root/private feedback",
            )?;
            let receipt = session.commit(pending)?;
            need(
                receipt.root == session.binding()
                    && receipt.root.frontier() == reference.frontier as u64
                    && receipt.selected_id == reference.selected_id
                    && receipt.eos == reference.eos,
                "normal original authority/private feedback parity",
            )?;
            let snapshot = session.snapshot()?;
            expected.normal_state(call, &block_on(snapshot.read_domains())?)?;
            drop(snapshot);
            events.push(json!({"kind":"resident-normal-parity","case":expected.cell.request.name,"phase":"proposal","call":call,"schedule":call,"input_ids":reference.input_ids,"base":before.frontier(),"frontier":receipt.root.frontier(),"selected_id":receipt.selected_id,"actual_selected_id":actual_id,"actual_max_bits":format!("{:08x}",actual_bits),"max_bits":actual_bits,"observer":false,"source_bound":true,"state_domains_exact":true,"producer_kv_d2h":0,"resident_weight_bytes":after.resident_weight_bytes,"ple_row_h2d":uploads.checked_mul(17_920).ok_or("normal row upload bytes overflow")?,"value_upload_total":total_value_h2d,"ple_row_uploads":uploads,"ple_row_reuses":reuses,"ple_row_avoided_h2d":avoided}));
        }
        expected.after()?;
    }
    // A short authentic prompt fits the existing page without eviction; its
    // next two full traversals must consume only completed source rows.
    let index = ids
        .iter()
        .enumerate()
        .min_by_key(|(_, ids)| ids.len())
        .map(|(i, _)| i)
        .ok_or("actual source prompt set")?;
    need(
        ids[index].len() <= candidate.inventory().selected_row_cache_capacity,
        "actual repeated prompt fits existing page rows",
    )?;
    for repeat in 0..3 {
        let mut session = block_on(candidate.session(ids[index].clone()))?;
        let mut expected = matrix.cell(index, loaded.source_identity())?;
        let before = session.binding();
        let traffic = block_on(candidate.traffic())?;
        let diagnostic = repeat == 1;
        let mut observed = Observation {
            frontier: expected.cell.calls[0].frontier,
            ..Observation::default()
        };
        let pending = if diagnostic {
            expected.begin(0);
            let mut tracing = Tracing {
                inner: &mut expected,
                observed: &mut observed,
            };
            block_on(session.prepare_prompt(8, Some(&mut tracing), None))?
        } else {
            block_on(session.prepare_prompt(8, None, None))?
        };
        let after = block_on(candidate.traffic())?;
        let hits = after
            .selected_row_reuses
            .checked_sub(traffic.selected_row_reuses)
            .ok_or("warm selected row reuse underflow")?;
        let misses = after
            .selected_row_uploads
            .checked_sub(traffic.selected_row_uploads)
            .ok_or("warm selected row upload underflow")?;
        let avoided = after
            .selected_row_avoided_upload_bytes
            .checked_sub(traffic.selected_row_avoided_upload_bytes)
            .ok_or("warm selected row avoided underflow")?;
        let total_value_h2d = after
            .value_upload_bytes
            .checked_sub(traffic.value_upload_bytes)
            .ok_or("warm value upload underflow")?;
        let producer_d2h = after
            .producer_readback_bytes
            .checked_sub(traffic.producer_readback_bytes)
            .ok_or("warm producer readback underflow")?;
        let kv_diag_d2h = after
            .kv_diagnostic_readback_bytes
            .checked_sub(traffic.kv_diagnostic_readback_bytes)
            .ok_or("warm KV readback underflow")?;
        need(
            hits.checked_add(misses)
                .ok_or("warm row counter overflow")?
                == ids[index].len() as u64
                && total_value_h2d
                    == misses
                        .checked_mul(17_920)
                        .ok_or("warm row upload bytes overflow")?
                && avoided
                    == hits
                        .checked_mul(17_920)
                        .ok_or("warm row avoided bytes overflow")?,
            "real producer-side source row traffic conservation",
        )?;
        if repeat > 0 {
            need(
                misses == 0 && hits == ids[index].len() as u64,
                "completed immutable PLE rows actually reused without H2D",
            )?;
        }
        let comparison = if diagnostic {
            need(
                total_value_h2d == 0
                    && producer_d2h > 0
                    && producer_d2h % 4 == 0
                    && producer_d2h <= 64 * 1024 * 1024
                    && kv_diag_d2h > 0
                    && kv_diag_d2h % 4 == 0
                    && kv_diag_d2h <= 64 * 1024 * 1024,
                "diagnostic hit reads bounded actual producer/fullKV D2H only",
            )?;
            let sequence = events
                .iter()
                .filter(|v| v["kind"].as_str() == Some("actual-completed-proposal"))
                .count();
            let report = compared(
                &mut expected,
                0,
                pending.readback(),
                before,
                &observed,
                "proposal",
                repeat,
                sequence,
            )?;
            need(
                report["exact"] == true
                    && report["final_kv_planes"] == 30
                    && report["full_vocab_words"] == 262144,
                "actual warmed PLE hit full35/fullV/all15KV exact before commit",
            )?;
            Some(report)
        } else {
            expected.normal_packet(0, pending.readback(), before)?;
            need(
                total_value_h2d == 0 && producer_d2h == 0 && kv_diag_d2h == 0,
                "ordinary completed hit has no optional producer/KV D2H",
            )?;
            None
        };
        need(
            session.binding() == before,
            "warmed hit Ready has not published root/private feedback",
        )?;
        let receipt = session.commit(pending)?;
        need(
            receipt.selected_id == expected.cell.calls[0].selected_id
                && receipt.root.frontier() == expected.cell.calls[0].frontier as u64,
            "full genuine prompt natural selection/root unchanged",
        )?;
        let snapshot = session.snapshot()?;
        expected.normal_state(0, &block_on(snapshot.read_domains())?)?;
        drop(snapshot);
        events.push(json!({"kind":"resident-ple-row-reuse","case":expected.cell.request.name,"phase":"proposal","call":0,"schedule":repeat,"repeat":repeat,"input_ids":expected.cell.calls[0].input_ids,"base":before.frontier(),"frontier":receipt.root.frontier(),"observer":diagnostic,"actual_rows":ids[index].len(),"uploads":misses,"reuses":hits,"ple_row_h2d":misses.checked_mul(17_920).ok_or("warm row upload bytes overflow")?,"value_upload_total":total_value_h2d,"avoided_h2d":avoided,"producer_d2h":producer_d2h,"kv_diagnostic_d2h":kv_diag_d2h,"resident_weight_bytes":after.resident_weight_bytes,"source_bound":true,"state_domains_exact":true,"full_kv_exact":true,"selected_id":receipt.selected_id,"max_bits":receipt.max_bits,"actual_max_bits":format!("{:08x}",receipt.max_bits),"comparison":comparison}));
        expected.after()?;
    }
    Ok(true)
}
pub fn run() -> Proof<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() == 2 && args[0] == "--import-source" {
        return super::cache_import::publish(&PathBuf::from(&args[1]));
    }
    need(
        (args.len() == 2)
            || (args.len() == 3
                && matches!(
                    args[2].as_str(),
                    "unknown" | "resident" | "resident-unknown" | "resident-row-unknown"
                )),
        "usage: gemma4_hip_matrix EXPECTED_INVOCATION FRESH_REPORT [unknown|resident|resident-unknown|resident-row-unknown] | --import-source ABS_STAGE",
    )?;
    let mode = args.get(2).map(String::as_str).unwrap_or("unknown");
    let unknown = matches!(
        mode,
        "unknown" | "resident-unknown" | "resident-row-unknown"
    );
    let resident = matches!(
        mode,
        "resident" | "resident-unknown" | "resident-row-unknown"
    );
    let storage = if resident {
        zerogravity_gpu::GemmaTextStorage::ResidentProjections
    } else {
        zerogravity_gpu::GemmaTextStorage::Streamed
    };
    let provider = if resident { 20u64 } else { 16u64 } * 1024 * 1024 * 1024;
    let invocation = PathBuf::from(&args[0]);
    let report = PathBuf::from(&args[1]);
    need(
        invocation.is_absolute() && report.is_absolute() && !report.exists(),
        "absolute expected/fresh report",
    )?;
    // The canonical larger host contract precedes source and expected buffers.
    let context = Gemma4CpuContext::new_hip_qualification()?;
    let harness = ResourceArbiter::process_shared().admit(
        &PhysicalDeviceKey::new("host", "system-ram")?,
        zerogravity::resource_arbiter::ResourceOwner::new(
            "Gemma-qualification",
            "bounded-expected-frames-and-report",
        )?,
        &[zerogravity::resource_arbiter::AllocationRequest {
            kind: zerogravity::resource_arbiter::AllocationClass::Scratch,
            bytes: 32 * 1024 * 1024,
        }],
    )?;
    let matrix = Matrix::open(&invocation)?;
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * 1024 * 1024 * 1024))?;
    let snapshot = Arc::new(
        store.open_snapshot("ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1")?,
    );
    let mut inputs = Vec::with_capacity(7);
    let mut ids = Vec::with_capacity(7);
    for case in &matrix.request.cases {
        let input = context.prepare_text(&snapshot, &case.text, None)?;
        need(
            input.ids().len() == case.rows,
            "actual source Rust tokenizer row count",
        )?;
        ids.push(Arc::<[u32]>::from(input.ids()));
        inputs.push(Some(input));
    }
    let decoder = context.prepare_text(&snapshot, "One", None)?;
    let loaded = Gemma4CpuLoaded::open(snapshot.clone(), context.clone(), None)?;
    let (candidate, mut a) = block_on(loaded.load_hip_qualification(
        Device::Hip(Default::default()),
        inputs[0].take().ok_or("genuine first prepared text")?,
        GemmaTextHostOwners {
            additional_recorded_calls: 7,
            additional_frame_bytes: 32 * 1024 * 1024,
        },
        storage,
        None,
    ))?;
    need(
        !candidate.trained_model_qualified(),
        "explicit candidate scope without model/serving permit",
    )?;
    // Actual source-issued cold inventory: resident/streamed geometry and each
    // owned device/host owner equation must hold before any inference.
    let cold = candidate.inventory();
    let complete_source_bytes = cold.complete_source_bytes as u64;
    // Storage policy changes only the owned page/resident binding and the
    // projection scratch rows: immutable tables, value slots, gamma and math
    // scratch are plan-owned and identical for both policies.
    let (resident_owner, projection_scratch, projection_rows, largest_binding) = if resident {
        (
            4_558_946_304usize,
            2_121_728usize,
            262_144usize,
            805_306_368usize,
        )
    } else {
        (0usize, 40_960usize, 2_048usize, 1_048_576usize)
    };
    let owned_sum = cold
        .source_page_bytes
        .checked_add(resident_owner)
        .and_then(|v| v.checked_add(cold.value_slot_bytes))
        .and_then(|v| v.checked_add(cold.source_gamma_bytes))
        .and_then(|v| v.checked_add(cold.math_scratch_bytes))
        .and_then(|v| v.checked_add(projection_scratch))
        .and_then(|v| v.checked_add(cold.status_and_packet_bytes))
        .and_then(|v| v.checked_add(2_048))
        .ok_or("actual owned cold device owner sum")?;
    need(
        cold.unique_source_matrices == 278
            && complete_source_bytes == 9_256_566_784
            && cold.single_root_payload_bytes == 2_359_296
            && cold.six_root_device_bytes == cold.single_root_payload_bytes * 6
            && cold.root_initialization_host_bytes == cold.single_root_payload_bytes * 3
            && cold.source_page_bytes == 1_048_576
            && cold.value_slots == 24
            && cold.value_slot_bytes == 2_502_656
            && cold.source_gamma_bytes == 1_146_880
            && cold.math_scratch_bytes == 114_688
            && cold.projection_scratch_bytes == projection_scratch
            && cold.projection_max_rows == projection_rows
            && cold.resident_matrix_bytes == resident_owner
            && cold.largest_binding_bytes == largest_binding
            && cold.status_and_packet_bytes == 2_560
            && cold.original_greedy_workspace_bytes == 12_336
            && cold.root_pool_metadata_bytes == 256_000
            && cold.owned_device_bytes == owned_sum
            && cold.device_with_six_roots_bytes
                == cold.owned_device_bytes
                    + cold.six_root_device_bytes
                    + cold.original_greedy_workspace_bytes
                    + 6 * 256
                    + cold.native_device_bytes
            && cold.complete_physical_device_bytes
                == cold.device_with_six_roots_bytes + cold.persistent_context_device_bytes
            && cold.persistent_context_device_bytes == 4 * 1024 * 1024 * 1024
            && cold.persistent_context_host_bytes == 2 * 1024 * 1024 * 1024
            && cold.native_device_bytes >= 8 * 1024 * 1024 * 1024
            && cold.native_device_bytes <= 9 * 1024 * 1024 * 1024
            && cold.native_host_bytes >= 8 * 1024 * 1024 * 1024
            && cold.native_host_bytes <= 10 * 1024 * 1024 * 1024 + 2 * 1024 * 1024
            && cold.host_upper_bytes > cold.native_host_bytes
            && cold.host_upper_bytes > 2 * 1024 * 1024 * 1024
            && cold.host_upper_bytes <= 16 * 1024 * 1024 * 1024
            && cold.complete_physical_device_bytes as u64 <= provider,
        "actual source-issued cold device/host owner equations and caps",
    )?;
    need(
        if resident {
            cold.resident_matrices == 277
                && cold.resident_upload_windows == 4_516
                && cold.resident_control_host_bytes == 18_153_472
                && cold.selected_row_cache_capacity == 58
                && cold.selected_row_cache_row_bytes == 17_920
        } else {
            // Streamed owns no resident binding/page window at all.
            cold.resident_matrices == 0
                && cold.resident_upload_windows == 0
                && cold.resident_control_host_bytes == 0
                && cold.selected_row_cache_capacity == 0
                && cold.selected_row_cache_row_bytes == 0
        },
        "mode-exact cold source resident/streamed geometry",
    )?;
    let inventory = format!("{cold:?}");
    let mut events = Vec::with_capacity(128);
    let mut exact = true;
    if unknown {
        let cached_row_fault = mode == "resident-row-unknown";
        if cached_row_fault {
            // A genuine cached PLE row needs one completed warm publication
            // first; the armed refusal then happens on that same hit row.
            need(
                ids[0].len() <= candidate.inventory().selected_row_cache_capacity,
                "authentic warm prompt fits source row page",
            )?;
            let expected = matrix.cell(0, loaded.source_identity())?;
            let before = a.binding();
            let pending = block_on(a.prepare_prompt(8, None, None))?;
            expected.normal_packet(0, pending.readback(), before)?;
            need(
                a.binding() == before,
                "warm row Ready has not published root/private feedback",
            )?;
            let warm = a.commit(pending)?;
            need(
                warm.root.frontier() == expected.cell.calls[0].frontier as u64,
                "warm PLE row publication before armed refusal",
            )?;
            let snapshot = a.snapshot()?;
            expected.normal_state(0, &block_on(snapshot.read_domains())?)?;
            drop(snapshot);
            expected.after()?;
            block_on(a.reset())?;
        }
        let traffic = block_on(candidate.traffic())?;
        let before = a.binding();
        let mut observer = UnknownObserver {
            completed: 0,
            stage: if cached_row_fault {
                "ple.embedding"
            } else {
                "block.output"
            },
        };
        let result = block_on(a.prepare_prompt(8, Some(&mut observer), None));
        need(
            matches!(result,Err(Error::State(ref why))if why=="deliberate actual armed Gemma producer refusal")
                && observer.completed == 1
                && a.binding() == before,
            "one actual submitted/completed producer refusal before Ready/root",
        )?;
        let after = block_on(candidate.traffic())?;
        let row_uploads = after
            .selected_row_uploads
            .checked_sub(traffic.selected_row_uploads)
            .ok_or("selected row upload counter underflow")?;
        let row_reuses = after
            .selected_row_reuses
            .checked_sub(traffic.selected_row_reuses)
            .ok_or("selected row reuse counter underflow")?;
        let avoided = after
            .selected_row_avoided_upload_bytes
            .checked_sub(traffic.selected_row_avoided_upload_bytes)
            .ok_or("selected row avoided counter underflow")?;
        // Whole-service value H2D, distinct from the PLE selected-row counter.
        let total_value_h2d = after
            .value_upload_bytes
            .checked_sub(traffic.value_upload_bytes)
            .ok_or("value upload counter underflow")?;
        let row_h2d = row_uploads
            .checked_mul(17_920)
            .ok_or("selected row upload bytes overflow")?;
        need(
            match (cached_row_fault, resident) {
                // Completed cached PLE consumer before refusal: one genuine
                // reuse, its avoided row bytes, no H2D and no publication.
                (true, _) => {
                    row_reuses == 1
                        && avoided == 17_920
                        && row_uploads == 0
                        && row_h2d == 0
                        && total_value_h2d == 0
                }
                // Fresh resident unknown faults after its actual PLE row upload.
                (false, true) => {
                    row_reuses == 0
                        && avoided == 0
                        && row_uploads == ids[0].len() as u64
                        && row_uploads <= 64
                        && row_h2d == 17_920 * row_uploads
                        && total_value_h2d == row_h2d
                }
                // Fresh streamed unknown reaches block.output through streamed
                // embedding/first-block projections: total service H2D is
                // positive, source-byte bounded, and never PLE-row traffic.
                (false, false) => {
                    row_reuses == 0
                        && avoided == 0
                        && row_uploads == 0
                        && row_h2d == 0
                        && total_value_h2d > 0
                        && total_value_h2d <= complete_source_bytes
                }
            },
            "actual cached hit, fresh resident PLE row upload or bounded streamed first-block H2D",
        )?;
        // The original envelope, source, cache and native owners stay retained
        // and charged: the live session refuses a managed source close.
        let close = loaded.clone().close(None);
        let arbiter = ResourceArbiter::process_shared();
        let physical = arbiter.snapshot(&PhysicalDeviceKey::new("pci", "0000:c6:00.0")?)?;
        let host = arbiter.snapshot(&PhysicalDeviceKey::new("host", "system-ram")?)?;
        write_new(
            &report,
            &json!({"schema":"zg.gemma4-e2b-original-hip-unknown.v1","status":"ACTUAL_UNKNOWN_RETAINED","completed_producers":observer.completed,"cached_row_fault":cached_row_fault,"warmup_calls":usize::from(cached_row_fault),"warmup_commits":usize::from(cached_row_fault),"faulted_call_commits":0,"selected_row_uploads":row_uploads,"selected_row_reuses":row_reuses,"selected_row_avoided_h2d":avoided,"selected_row_h2d":row_h2d,"total_value_upload_h2d":total_value_h2d,"complete_source_bytes":complete_source_bytes,"published_roots":0,"inventory":inventory,"source_close_refused":close.is_err(),"source_close_error":close.as_ref().err().map(ToString::to_string),"physical_reserved":physical.reserved,"host_reserved":host.reserved,"diagnostic_host_reserved":32u64*1024*1024,"physical_fenced":physical.fenced,"host_fenced":host.fenced,"context_retirement":"process lifetime; source/native/flight owners intentionally retained","trained_native_serving_or_performance_qualified":false}),
        )?;
        // Coarse retained HOST envelope, not a per-owner retained-byte proof:
        // the original envelope/source/cache/native owners stay charged/fenced.
        need(
            close.is_err()
                && physical.fenced
                && physical.reserved == provider
                && host.reserved >= 2 * 1024 * 1024 * 1024 + 32 * 1024 * 1024
                && host.reserved <= 16 * 1024 * 1024 * 1024
                && !host.fenced,
            "unknown same original envelope/source/cache owners retained and fenced",
        )?;
        println!(
            "GEMMA_COMPLETE_UNKNOWN_PASS actual_completed_producer=1 published_roots=0 original_envelope=retained original_source_close=refused original_context=process_lifetime"
        );
        return Ok(());
    }
    let mut b = block_on(candidate.session(ids[1].clone()))?;
    let untouched = b.binding();
    let mut ea = matrix.cell(0, loaded.source_identity())?;
    let mut eb = matrix.cell(1, loaded.source_identity())?;
    // Case0 then case1 keep the original two-session population; further cases
    // are admitted only after both are dropped.
    let mut result: Proof<()> = (|| -> Proof<()> {
        need(
            a.binding().session() != b.binding().session(),
            "two independent real SessionAuthority owners",
        )?;
        need(
            matches!(
                block_on(candidate.session(ids[0].clone())),
                Err(Error::Limit(_))
            ),
            "third session pressure before allocation/publication",
        )?;
        need(
            loaded.clone().close(None).is_err(),
            "GPU/native owners retain authentic SnapshotLease/managed lifecycle",
        )?;
        need(
            &ids[0][..] == ea.cell.prompt_ids.as_slice()
                && &ids[1][..] == eb.cell.prompt_ids.as_slice(),
            "source-computed token IDs compared only",
        )?;
        exact &= cancelled_and_discarded(&mut a, &mut ea, &loaded, &mut events)?;
        if !exact {
            return Ok(());
        }
        exact &= completed(&mut a, &mut ea, 0, &mut events)?;
        need(
            b.binding() == untouched,
            "peer root unchanged after other actual publication",
        )?;
        if !exact {
            return Ok(());
        }
        let snapshot_a = a.snapshot()?;
        if ea.cell.calls.len() > 1 {
            exact &= completed(&mut a, &mut ea, 1, &mut events)?;
            if !exact {
                return Ok(());
            }
            // Original snapshot state after call0; restoring it must replay
            // exactly one further call from the same completed prefix.
            block_on(a.restore(&snapshot_a))?;
            need(
                a.binding().frontier() == ea.cell.calls[0].frontier as u64,
                "source-independent state restore frontier",
            )?;
            exact &= completed(&mut a, &mut ea, 1, &mut events)?;
            if !exact {
                return Ok(());
            }
            exact &= trajectory(&mut a, &mut ea, 2, &mut events, &decoder)?;
        } else {
            exact &= trajectory(&mut a, &mut ea, 1, &mut events, &decoder)?;
        }
        if !exact {
            return Ok(());
        }
        // Independent second real session: its own first completed proposal
        // must not move the first session's published root.
        let a_before_b = a.binding();
        exact &= completed(&mut b, &mut eb, 0, &mut events)?;
        need(
            a.binding() == a_before_b,
            "independent session root isolation",
        )?;
        if !exact {
            return Ok(());
        }
        need(
            block_on(b.restore(&snapshot_a)).is_err(),
            "foreign-session snapshot refused before publication",
        )?;
        exact &= trajectory(&mut b, &mut eb, 1, &mut events, &decoder)?;
        drop(snapshot_a);
        if !exact {
            return Ok(());
        }
        ea.after()?;
        eb.after()?;
        Ok(())
    })();
    // The second permit is released before the remaining genuine texts.
    drop(b);
    if result.is_ok() && exact {
        result = (|| -> Proof<()> {
            for (i, input_ids) in ids.iter().enumerate().skip(2) {
                let mut session = block_on(candidate.session(input_ids.clone()))?;
                let mut expected = matrix.cell(i, loaded.source_identity())?;
                need(
                    &input_ids[..] == expected.cell.prompt_ids.as_slice(),
                    "genuine text token IDs compared only",
                )?;
                exact &= trajectory(&mut session, &mut expected, 0, &mut events, &decoder)?;
                if !exact {
                    return Ok(());
                }
            }
            // Reset publishes empty original state and clears private feedback.
            // The following completed call0 is a genuine root-relative
            // comparison on the same original authority; it is deliberately not
            // appended to the two-cell proposal schedule the expected cells define.
            block_on(a.reset())?;
            let reset_binding = a.binding();
            need(
                reset_binding.frontier() == 0 && reset_binding.root() > 0 && !a.stopped(),
                "reset publishes empty original state and clears private feedback",
            )?;
            exact &= completed(&mut a, &mut ea, 0, &mut events)?;
            if !exact {
                return Ok(());
            }
            need(
                a.binding().root() == reset_binding.root() + 1
                    && a.binding().session_epoch() == reset_binding.session_epoch(),
                "post-reset genuine root advance from the same empty-state epoch",
            )?;
            ea.after()?;
            Ok(())
        })();
    }
    // Both real session owners are dropped before the page-coverage pass: the
    // original two-session permit is not widened and no root is recreated.
    drop(a);
    if resident && result.is_ok() && exact {
        match resident_rows(&candidate, &matrix, &loaded, &ids, &mut events) {
            Ok(ok) => exact = ok,
            Err(error) => {
                exact = false;
                result = Err(error);
            }
        }
    }
    drop(candidate);
    drop(decoder);
    drop(inputs);
    drop(ids);
    // Last native owner performs actual synchronise/module unload. Context's
    // separately partitioned source-neutral baseline deliberately remains live.
    let close = loaded.clone().close(None);
    drop(loaded);
    drop(snapshot);
    drop(store);
    drop(context);
    let arbiter = ResourceArbiter::process_shared();
    let physical = arbiter.snapshot(&PhysicalDeviceKey::new("pci", "0000:c6:00.0")?)?;
    let host = arbiter.snapshot(&PhysicalDeviceKey::new("host", "system-ram")?)?;
    let failure = result.as_ref().err().map(ToString::to_string);
    let source_closed = close.is_ok();
    let status = if result.is_ok() && exact && source_closed {
        "PASS_EXACT_DIAGNOSTIC_MATRIX"
    } else {
        "REFUSED_UNQUALIFIED"
    };
    write_new(
        &report,
        &json!({"schema":"zg.gemma4-e2b-original-hip-complete-qualification.v1","status":status,"failure":failure,"exact":result.is_ok()&&exact,"events":events,"inventory":inventory,"managed_source_closed":source_closed,"source_close_error":close.as_ref().err().map(ToString::to_string),"physical_reserved":physical.reserved,"host_reserved":host.reserved,"diagnostic_host_reserved":32u64*1024*1024,"physical_fenced":physical.fenced,"host_fenced":host.fenced,"expected_persistent_context_device":4u64*1024*1024*1024,"expected_persistent_context_host":2u64*1024*1024*1024,"context_retirement":"process lifetime; model0 differs from global0","storage":format!("{storage:?}"),"source_page_policy":"all277 repeated projections resident; tied embedding shares exact allocation; PLE selected row page only; ordinary observerNone","trained_native_serving_or_performance_qualified":false}),
    )?;
    result?;
    need(exact, "fatal complete Gemma numerical mismatch")?;
    close?;
    drop(matrix);
    // Healthy report phase: model reservations released, the diagnostic
    // harness frame still held, the separate context baseline truthful.
    need(
        physical.reserved == 4 * 1024 * 1024 * 1024
            && host.reserved == 2 * 1024 * 1024 * 1024 + 32 * 1024 * 1024
            && !physical.fenced
            && !host.fenced,
        "report-phase DEVICE4GiB/HOST2GiB+32MiB with held harness frame",
    )?;
    drop(harness);
    let final_host = ResourceArbiter::process_shared()
        .snapshot(&PhysicalDeviceKey::new("host", "system-ram")?)?;
    need(
        physical.reserved == 4 * 1024 * 1024 * 1024
            && final_host.reserved == 2 * 1024 * 1024 * 1024
            && !physical.fenced
            && !host.fenced,
        "actual model reservations released; honest separate persistent context baseline retained",
    )?;
    println!("GEMMA_FINAL_HOST_RETIRED reserved={}", final_host.reserved);
    println!(
        "GEMMA_COMPLETE_SOURCE_MATRIX_PASS cases=7 full35=true all_boundary_KV_fullV_words=exact source_guard=actual native_modules=actual_unloaded model_reservations=0 context_baseline=retained_until_process_exit serving_performance=false"
    );
    Ok(())
}
