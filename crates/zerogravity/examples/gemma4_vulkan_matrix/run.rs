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
    gemma4_cpu::{Gemma4CpuContext, Gemma4CpuLoaded},
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter},
};
use zerogravity_gpu::{Device, Gemma4Bf16GpuSession, GemmaTextHostOwners};
const GIB: u64 = 1024 * 1024 * 1024;
const PERSISTENT_CONTEXT_HOST: u64 = 2 * GIB;
const PERSISTENT_COMPILER_HOST: u64 = 8 * GIB;
const PERSISTENT_HOST_BASELINE: u64 = PERSISTENT_CONTEXT_HOST + PERSISTENT_COMPILER_HOST;
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
    json!({"session":root.session().get(),"root":root.root(),"frontier":root.frontier()})
}
fn completed(
    s: &mut Gemma4Bf16GpuSession,
    e: &mut Expected,
    call: usize,
    events: &mut Vec<Value>,
) -> Proof<bool> {
    e.begin(call);
    let before = s.binding();
    let pending = if call == 0 {
        block_on(s.prepare_prompt(8, Some(e), None))?
    } else {
        block_on(s.prepare_next(Some(e), None))?
    };
    let comparison = e.completed(pending.readback(), before)?;
    let exact = comparison["exact"] == true;
    events.push(json!({"kind":"actual-completed-proposal","before":binding(before),"comparison":comparison}));
    if !exact {
        drop(pending);
        need(
            s.binding() == before,
            "fatal numerical refusal preserves original root/private feedback",
        )?;
        return Ok(false);
    }
    let receipt = s.commit(pending)?;
    let c = &e.cell.calls[call];
    need(
        receipt.root == s.binding()
            && receipt.root.root() == before.root() + 1
            && receipt.root.frontier() == c.frontier as u64
            && receipt.selected_id == c.selected_id
            && receipt.eos == c.eos,
        "complete source/state/private packet join before publication",
    )?;
    events.push(json!({"kind":"actual-committed-call","case":e.cell.request.name,"call":call,"selected_id":receipt.selected_id,"eos":receipt.eos,"root":binding(receipt.root)}));
    Ok(true)
}
fn trajectory(
    s: &mut Gemma4Bf16GpuSession,
    e: &mut Expected,
    start: usize,
    events: &mut Vec<Value>,
    decoder: &zerogravity::gemma4_cpu::Gemma4CpuInput,
) -> Proof<bool> {
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
                v["kind"] == "actual-committed-call"
                    && v["case"] == e.cell.request.name
                    && v["call"] == call
            })
            .ok_or("actual committed output record")?;
        if event["eos"] != true {
            generated.push(u32::try_from(
                event["selected_id"].as_u64().ok_or("actual generated ID")?,
            )?);
        }
    }
    let text = decoder.decode_output(&generated, true, None)?;
    events.push(json!({"kind":"source-ID-decoded-output","case":e.cell.request.name,"ids":generated,"text":text.as_str(),"sequence_decode_only":true}));
    drop(text);
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
        matches!(block_on(s.prepare_prompt(8,Some(e),Some(&flag))),Err(Error::State(why))if why=="Gemma source-bound GPU call cancelled")
            && s.binding() == before,
        "pre-cancel has no root/private feedback change",
    )?;
    events.push(json!({"kind":"pre-cancel","root":binding(before)}));
    e.begin(0);
    let pending = block_on(s.prepare_prompt(8, Some(e), None))?;
    let comparison = e.completed(pending.readback(), before)?;
    let exact = comparison["exact"] == true;
    drop(pending);
    need(
        s.binding() == before,
        "actual Ready pending Drop cannot publish root/private feedback",
    )?;
    events.push(
        json!({"kind":"completed-Ready-drop","comparison":comparison,"root":binding(before)}),
    );
    if !exact {
        return Ok(false);
    }
    for cancel in [false, true] {
        flag.store(false, Ordering::Release);
        e.begin(0);
        let observed = std::cell::RefCell::new(&mut *e);
        let mut observer = BorrowedObserver(&observed);
        let mut comparison = None;
        let mut exact = false;
        let result = block_on(
            s.infer_prompt(8, Some(&mut observer), Some(&flag), |actual| {
                let report = observed
                    .borrow()
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
        events.push(json!({"kind":if cancel{"actual-completed-late-cancel"}else{"actual-completed-refusal"},"comparison":comparison,"root":binding(before),"error":result.as_ref().err().map(ToString::to_string)}));
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
                    "resident" | "resident-unknown" | "resident-row-unknown"
                )),
        "usage: gemma4_vulkan_matrix EXPECTED_INVOCATION FRESH_REPORT [resident|resident-unknown|resident-row-unknown] | --import-source ABS_STAGE",
    )?;
    let mode = args.get(2).map(String::as_str).unwrap_or("resident");
    let unknown = matches!(mode, "resident-unknown" | "resident-row-unknown");
    let resident = matches!(
        mode,
        "resident" | "resident-unknown" | "resident-row-unknown"
    );
    need(
        resident,
        "Vulkan shared/multi-set reload requires resident projections",
    )?;
    let storage = zerogravity_gpu::GemmaTextStorage::ResidentProjectionsVulkan;
    let provider = 20u64 * 1024 * 1024 * 1024;
    let invocation = PathBuf::from(&args[0]);
    let report = PathBuf::from(&args[1]);
    need(
        invocation.is_absolute() && report.is_absolute() && !report.exists(),
        "absolute expected/fresh report",
    )?;
    // The canonical larger host contract precedes source and expected buffers.
    let context = Gemma4CpuContext::new_vulkan_qualification()?;
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
    let device = Device::vulkan(zerogravity::gpu::WgpuDeviceKind::IntegratedGpu(0))
        .map_err(|error| Error::State(format!("original Vulkan device: {error}")))?;
    let (candidate, mut a) = {
        block_on(loaded.load_vulkan_qualification(
            device,
            inputs[0].take().ok_or("genuine first prepared text")?,
            GemmaTextHostOwners {
                additional_recorded_calls: 7,
                additional_frame_bytes: 32 * 1024 * 1024,
            },
            storage,
            None,
        ))?
    };
    need(
        !candidate.trained_model_qualified(),
        "explicit candidate scope without model/serving permit",
    )?;
    let inventory = format!("{:?}", candidate.inventory());
    let mut events = Vec::with_capacity(128);
    let mut exact = true;
    if unknown {
        let cached_row_fault = mode == "resident-row-unknown";
        if cached_row_fault {
            need(
                ids[0].len() <= candidate.inventory().selected_row_cache_capacity,
                "authentic warm prompt fits source row page",
            )?;
            let expected = matrix.cell(0, loaded.source_identity())?;
            let before = a.binding();
            let pending = block_on(a.prepare_prompt(8, None, None))?;
            expected.normal_packet(0, pending.readback(), before)?;
            a.commit(pending)?;
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
        if cached_row_fault {
            need(
                after.selected_row_reuses - traffic.selected_row_reuses == 1
                    && after.selected_row_avoided_upload_bytes
                        - traffic.selected_row_avoided_upload_bytes
                        == 17920
                    && after.value_upload_bytes == traffic.value_upload_bytes,
                "actual cached PLE consumer completed before refusal; no new H2D or publication",
            )?;
        }
        drop((a, candidate, inputs, ids, decoder));
        let close = loaded.clone().close(None);
        drop(loaded);
        drop(snapshot);
        drop(store);
        drop(context);
        let arbiter = ResourceArbiter::process_shared();
        let physical = arbiter.snapshot(&PhysicalDeviceKey::new("pci", "0000:c6:00.0")?)?;
        let host = arbiter.snapshot(&PhysicalDeviceKey::new("host", "system-ram")?)?;
        write_new(
            &report,
            &json!({"schema":"zg.gemma4-original-vulkan-unknown-native-owner.v1","status":"ACTUAL_UNKNOWN_RETAINED","completed_producers":observer.completed,"cached_row_fault":cached_row_fault,"warmup_calls":usize::from(cached_row_fault),"selected_row_reuses":after.selected_row_reuses-traffic.selected_row_reuses,"selected_row_avoided_h2d":after.selected_row_avoided_upload_bytes-traffic.selected_row_avoided_upload_bytes,"published_roots":0,"source_close_refused":close.is_err(),"physical_reserved":physical.reserved,"host_reserved":host.reserved,"diagnostic_host_reserved":32u64*1024*1024,"expected_persistent_context_host":PERSISTENT_CONTEXT_HOST,"persistent_compiler_host":PERSISTENT_COMPILER_HOST,"expected_persistent_host_baseline":PERSISTENT_HOST_BASELINE,"physical_fenced":physical.fenced,"context_retirement":"process lifetime; source/native/flight owners intentionally retained","trained_native_serving_or_performance_qualified":false}),
        )?;
        need(
            close.is_err()
                && physical.fenced
                && physical.reserved == provider
                && host.reserved > 2 * 1024 * 1024 * 1024,
            "unknown same original envelope/source/cache owners retained and fenced",
        )?;
        println!(
            "GEMMA_VULKAN_COMPLETE_UNKNOWN_PASS actual_completed_producer=1 published_roots=0 original_envelope=retained original_source_close=refused original_context=process_lifetime"
        );
        return Ok(());
    }
    let result = (|| -> Proof<()> {
        let mut b = block_on(candidate.session(ids[1].clone()))?;
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
        let mut ea = matrix.cell(0, loaded.source_identity())?;
        let mut eb = matrix.cell(1, loaded.source_identity())?;
        need(
            ids[0].as_ref() == ea.cell.prompt_ids && ids[1].as_ref() == eb.cell.prompt_ids,
            "source-computed token IDs compared only",
        )?;
        exact &= cancelled_and_discarded(&mut a, &mut ea, &loaded, &mut events)?;
        if !exact {
            return Ok(());
        }
        let untouched = b.binding();
        exact &= completed(&mut a, &mut ea, 0, &mut events)?;
        need(
            b.binding() == untouched,
            "peer root unchanged after other actual publication",
        )?;
        if !exact {
            return Ok(());
        }
        let snapshot_a = a.snapshot()?;
        let before_b = a.binding();
        exact &= completed(&mut b, &mut eb, 0, &mut events)?;
        need(
            a.binding() == before_b,
            "independent session root isolation",
        )?;
        if !exact {
            return Ok(());
        }
        need(
            block_on(b.restore(&snapshot_a)).is_err(),
            "foreign-session snapshot refused before publication",
        )?;
        if ea.cell.calls.len() > 1 {
            exact &= completed(&mut a, &mut ea, 1, &mut events)?;
            if !exact {
                return Ok(());
            }
            block_on(a.restore(&snapshot_a))?;
            need(
                a.binding().frontier() == ea.cell.calls[0].frontier as u64,
                "source-independent state restore frontier",
            )?;
            exact &= trajectory(&mut a, &mut ea, 1, &mut events, &decoder)?;
        } else {
            exact &= trajectory(&mut a, &mut ea, 1, &mut events, &decoder)?;
        }
        drop(snapshot_a);
        if !exact {
            return Ok(());
        }
        exact &= trajectory(&mut b, &mut eb, 1, &mut events, &decoder)?;
        if !exact {
            return Ok(());
        }
        block_on(a.reset())?;
        need(
            a.binding().frontier() == 0 && !a.stopped(),
            "reset publishes empty original state and clears private feedback",
        )?;
        exact &= completed(&mut a, &mut ea, 0, &mut events)?;
        ea.after()?;
        eb.after()?;
        drop((a, b, ea, eb));
        if !exact {
            return Ok(());
        }
        for (i, input_ids) in ids.iter().enumerate().skip(2) {
            let mut session = block_on(candidate.session(input_ids.clone()))?;
            let mut expected = matrix.cell(i, loaded.source_identity())?;
            need(
                input_ids.as_ref() == expected.cell.prompt_ids,
                "genuine text token IDs compared only",
            )?;
            exact &= trajectory(&mut session, &mut expected, 0, &mut events, &decoder)?;
            if !exact {
                break;
            }
        }
        if resident && exact {
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
                    let after = block_on(candidate.traffic())?;
                    need(
                        after.producer_readback_bytes == traffic.producer_readback_bytes
                            && after.kv_diagnostic_readback_bytes
                                == traffic.kv_diagnostic_readback_bytes
                            && (after.selected_row_uploads - traffic.selected_row_uploads)
                                + (after.selected_row_reuses - traffic.selected_row_reuses)
                                == reference.input_ids.len() as u64
                            && after.value_upload_bytes - traffic.value_upload_bytes
                                == (after.selected_row_uploads - traffic.selected_row_uploads)
                                    * 17920
                            && after.selected_row_avoided_upload_bytes
                                - traffic.selected_row_avoided_upload_bytes
                                == (after.selected_row_reuses - traffic.selected_row_reuses)
                                    * 17920
                            && after.resident_weight_bytes == 4_558_946_304,
                        "normal no producer/KV D2H or repeated selected weight H2D; PLE row only",
                    )?;
                    let receipt = session.commit(pending)?;
                    need(
                        receipt.root.frontier() == reference.frontier as u64
                            && receipt.selected_id == reference.selected_id
                            && receipt.eos == reference.eos,
                        "normal original authority/private feedback parity",
                    )?;
                    let snapshot = session.snapshot()?;
                    expected.normal_state(call, &block_on(snapshot.read_domains())?)?;
                    drop(snapshot);
                    events.push(json!({"kind":"resident-normal-parity","case":expected.cell.request.name,"call":call,"frontier":reference.frontier,"selected_id":receipt.selected_id,"observer":false,"producer_kv_d2h":0,"selected_weight_h2d":0,"ple_row_h2d":after.value_upload_bytes-traffic.value_upload_bytes,"ple_row_uploads":after.selected_row_uploads-traffic.selected_row_uploads,"ple_row_reuses":after.selected_row_reuses-traffic.selected_row_reuses,"ple_row_avoided_h2d":after.selected_row_avoided_upload_bytes-traffic.selected_row_avoided_upload_bytes}));
                }
                expected.after()?;
            }
            // Two real fresh-session prompts, no teacher IDs or output imports.
            // A short authentic prompt fits the existing page without eviction;
            // its second full traversal must consume only completed source rows.
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
            for repeat in 0..2 {
                let mut session = block_on(candidate.session(ids[index].clone()))?;
                let mut expected = matrix.cell(index, loaded.source_identity())?;
                if repeat == 0 {
                    let before = session.binding();
                    let traffic = block_on(candidate.traffic())?;
                    let pending = block_on(session.prepare_prompt(8, None, None))?;
                    expected.normal_packet(0, pending.readback(), before)?;
                    let after = block_on(candidate.traffic())?;
                    let hits = after.selected_row_reuses - traffic.selected_row_reuses;
                    let misses = after.selected_row_uploads - traffic.selected_row_uploads;
                    need(
                        hits + misses == ids[index].len() as u64
                            && after.value_upload_bytes - traffic.value_upload_bytes
                                == misses * 17920
                            && after.selected_row_avoided_upload_bytes
                                - traffic.selected_row_avoided_upload_bytes
                                == hits * 17920,
                        "real producer-side source row traffic conservation",
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
                    events.push(json!({"kind":"resident-ple-row-reuse","repeat":repeat,"actual_rows":ids[index].len(),"uploads":misses,"reuses":hits,"ple_row_h2d":after.value_upload_bytes-traffic.value_upload_bytes,"avoided_h2d":after.selected_row_avoided_upload_bytes-traffic.selected_row_avoided_upload_bytes,"full_kv_exact":false,"observed":false,"selected_id":receipt.selected_id}));
                    expected.after()?;
                    continue;
                }
                // The warmed producer must reuse every completed immutable PLE
                // row. Reset publishes an empty root, then an OBSERVED full
                // traversal recompares full35/fullV/all15 KV against the
                // unchanged oracle before this hit may commit.
                let warmed = session.binding();
                let unobserved = block_on(session.prepare_prompt(8, None, None))?;
                need(
                    session.binding() == warmed,
                    "warm observerNone proposal preserves original root",
                )?;
                drop(unobserved);
                block_on(session.reset())?;
                need(
                    session.binding().frontier() == 0,
                    "warm reset publishes empty original state",
                )?;
                let before = session.binding();
                expected.begin(0);
                let traffic = block_on(candidate.traffic())?;
                let pending = block_on(session.prepare_prompt(8, Some(&mut expected), None))?;
                let comparison = expected.completed(pending.readback(), before)?;
                need(
                    comparison["exact"] == true,
                    "learned full35/fullV/all15KV PLE warm hit before commit",
                )?;
                let after = block_on(candidate.traffic())?;
                let hits = after.selected_row_reuses - traffic.selected_row_reuses;
                let misses = after.selected_row_uploads - traffic.selected_row_uploads;
                need(
                    misses == 0
                        && hits == ids[index].len() as u64
                        && after.value_upload_bytes == traffic.value_upload_bytes
                        && after.selected_row_avoided_upload_bytes
                            - traffic.selected_row_avoided_upload_bytes
                            == hits * 17920,
                    "completed immutable PLE rows reused without H2D on observed hit",
                )?;
                let receipt = session.commit(pending)?;
                need(
                    receipt.selected_id == expected.cell.calls[0].selected_id
                        && receipt.root.frontier() == expected.cell.calls[0].frontier as u64,
                    "learned warm hit natural selection/root unchanged",
                )?;
                let snapshot = session.snapshot()?;
                expected.normal_state(0, &block_on(snapshot.read_domains())?)?;
                drop(snapshot);
                events.push(json!({"kind":"resident-ple-row-reuse","repeat":repeat,"actual_rows":ids[index].len(),"uploads":misses,"reuses":hits,"ple_row_h2d":after.value_upload_bytes-traffic.value_upload_bytes,"avoided_h2d":after.selected_row_avoided_upload_bytes-traffic.selected_row_avoided_upload_bytes,"full_kv_exact":comparison["exact"],"observed":true,"learned_full35":comparison["exact"],"selected_id":receipt.selected_id}));
                expected.after()?;
            }
        }
        Ok(())
    })();
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
    write_new(
        &report,
        &json!({"schema":"zg.gemma4-e2b-original-vulkan-complete-qualification.v1","status":if result.is_ok()&&exact&&source_closed{"PASS_EXACT_DIAGNOSTIC_MATRIX"}else{"REFUSED_UNQUALIFIED"},"failure":failure,"exact":result.is_ok()&&exact,"events":events,"inventory":inventory,"managed_source_closed":source_closed,"source_close_error":close.as_ref().err().map(ToString::to_string),"physical_reserved":physical.reserved,"host_reserved":host.reserved,"diagnostic_host_reserved":32u64*1024*1024,"physical_fenced":physical.fenced,"host_fenced":host.fenced,"expected_persistent_context_device":4u64*1024*1024*1024,"expected_persistent_context_host":PERSISTENT_CONTEXT_HOST,"persistent_compiler_host":PERSISTENT_COMPILER_HOST,"expected_persistent_host_baseline":PERSISTENT_HOST_BASELINE,"context_retirement":"process lifetime; model0 differs from global0","storage":format!("{storage:?}"),"source_page_policy":"resident projections bounded to <=8192 selected rows per Vulkan dispatch; PLE selected row page only; ordinary observerNone","trained_native_serving_or_performance_qualified":false}),
    )?;
    result?;
    need(exact, "fatal complete Gemma numerical mismatch")?;
    close?;
    drop(events);
    drop(matrix);
    drop(harness);
    let final_host = ResourceArbiter::process_shared()
        .snapshot(&PhysicalDeviceKey::new("host", "system-ram")?)?;
    need(
        physical.reserved == 4 * 1024 * 1024 * 1024
            && final_host.reserved == PERSISTENT_HOST_BASELINE
            && !physical.fenced
            && !host.fenced,
        "actual model reservations released; honest separate persistent context/compiler baseline retained",
    )?;
    println!(
        "GEMMA_VULKAN_COMPLETE_SOURCE_MATRIX_PASS cases=7 full35=true all_boundary_KV_fullV_words=exact source_guard=actual native_modules=actual_unloaded model_reservations=0 context_baseline=retained_until_process_exit serving_performance=false"
    );
    Ok(())
}
