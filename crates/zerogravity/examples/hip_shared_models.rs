//! Finite same-actor genuine Qwen/Gemma lifecycle; no serving/speed eligibility.
#[allow(dead_code)]
#[path = "gemma4_cpu_matrix/comparison.rs"]
mod comparison;
#[allow(dead_code)]
#[path = "gemma4_hip_matrix/expected.rs"]
mod gexpected;
#[allow(dead_code)]
#[path = "gemma4_cpu_matrix/io.rs"]
mod io;
#[allow(dead_code)]
#[path = "qwen27b_hip_matrix/expected.rs"]
mod qexpected;
use io::{Proof, need, write_new};
use serde_json::{Value, json};
use std::{
    future::Future,
    path::PathBuf,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread,
};
use zerogravity::{
    cache::{CacheOptions, CacheStore},
    gemma4_cpu::{Gemma4CpuContext, Gemma4CpuLoaded},
    qwen27b_gpu::{Qwen27bGpuContext, Qwen27bGpuLoaded},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_gpu::{
    Device, Gemma4Bf16GpuCandidate, Gemma4Bf16GpuSession, GemmaTextHostOwners,
    HipSourceSetIdentity, Qwen27bSourceGpuSession, SharedHipNative,
};
const GIB: u64 = 1024 * 1024 * 1024;
const QSNAP: &str = "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
const GSNAP: &str = "ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1";
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
fn identity(i: HipSourceSetIdentity) -> Value {
    json!({"source":format!("{:?}",i.source),"source_service_generation":i.source_service_generation,"actor":format!("{:?}",i.actor),"set_key":i.set_key,"logical_stream":i.logical_stream})
}
macro_rules! receipt_json {
    ($r:expr) => {{ let r=&$r; json!({
        "wrapper":r.wrapper,"fault":r.fault,"source":r.source,"lease":r.lease,
        "service_generation":r.service_generation,"lease_bytes":r.lease_bytes,
        "aliases_inside_wrapper":r.aliases_inside_wrapper,"candidate_owners_before_drop":r.candidate_owners_before_drop,
        "model_owners_before_drop":r.model_owners_before_drop,"candidate_workspace_target_aliases":r.candidate_workspace_target_aliases,
        "candidate_retired":r.candidate_retired,"release_hits":r.release_hits,
        "native_retirement_attempts":r.native_retirement_attempts,"reader_hits_before_native_retirement":r.reader_hits_before_native_retirement,
        "native_modules_before":r.native_modules_before,"native_code_bytes_before":r.native_code_bytes_before,
        "native_host_bytes_before":r.native_host_bytes_before,"native_device_bytes_before":r.native_device_bytes_before,
        "native_admission_retained_after":r.native_admission_retained_after,"native_modules_after":r.native_modules_after,
        "original_service_retained_after":r.original_service_retained_after,"quarantined_handles_after":r.quarantined_handles_after }) }};
}
fn native(shared: &SharedHipNative) -> Proof<Value> {
    let n = shared.inspect_native()?;
    Ok(
        json!({"sets":n.sets.iter().map(|s| json!({"set_key":s.set_key,"logical_stream":s.logical_stream,"catalogue_keys":s.catalogue_keys,"non_null_modules":s.non_null_modules,"admitted_modules":s.admitted_native.modules,"PT_LOAD_bytes":s.admitted_native.code_object_load_bytes,"retained_code_host_bytes":s.admitted_native.retained_host_bytes,"scoped_metadata_bytes":s.scoped_metadata_bytes,"sealed":s.sealed,"usable":s.usable,"sealed_miss_refused":s.sealed_miss_refused})).collect::<Vec<_>>(),"original_key_overlap":n.original_key_overlap,"default_stream_refused":n.default_stream_refused,"compiler_host_reserved":shared.compiler_host_reserved_bytes()?,"available_model_capacity":shared.available_model_capacity()?,"resident_bytes":shared.resident_bytes()?}),
    )
}
fn live(
    shared: &SharedHipNative,
    q: HipSourceSetIdentity,
    g: HipSourceSetIdentity,
) -> Proof<Value> {
    need(
        q.actor == g.actor
            && q.source != g.source
            && q.source_service_generation != g.source_service_generation
            && q.set_key != g.set_key
            && q.logical_stream.is_some()
            && g.logical_stream.is_some()
            && q.logical_stream != g.logical_stream,
        "same physical actor; distinct actual source service and scoped logical identity",
    )?;
    let n = native(shared)?;
    let rows = n["sets"]
        .as_array()
        .ok_or("actual native set observations")?;
    need(
        rows.len() == 2
            && n["original_key_overlap"].as_u64().is_some_and(|n| n > 0)
            && n["default_stream_refused"] == true,
        "two real sets overlap original full keys but default never routes",
    )?;
    for (id, count) in [(q, 117), (g, 184)] {
        let row = rows
            .iter()
            .find(|s| s["set_key"] == id.set_key)
            .ok_or("actual source set key")?;
        need(
            row["logical_stream"] == id.logical_stream.unwrap()
                && row["catalogue_keys"] == count
                && row["non_null_modules"] == count
                && row["admitted_modules"] == count
                && row["sealed"] == true
                && row["usable"] == true
                && row["sealed_miss_refused"] == true,
            "complete actual source-native catalogue and sealed preflight",
        )?;
    }
    Ok(n)
}
#[derive(Default)]
struct Counts {
    q_calls: usize,
    q_commits: usize,
    g_calls: usize,
    g_instrumented: usize,
    g_commits: usize,
}
fn qstep(
    s: &mut Qwen27bSourceGpuSession,
    expected: &mut qexpected::Expected,
    repeat: usize,
    call: usize,
    counts: &mut Counts,
    events: &mut Vec<Value>,
) -> Proof<()> {
    expected.begin(repeat, call)?;
    let before = s.binding();
    let pending = block_on(s.prepare_teacher(expected, None))?;
    counts.q_calls += 1;
    let comparison = expected.finish()?;
    need(
        comparison["exact"] == true,
        "actual complete Qwen full64/fullV/all112 state domains equal original",
    )?;
    need(s.binding() == before, "Qwen Ready has no publication")?;
    let output = s.commit(pending)?;
    counts.q_commits += 1;
    drop(output);
    need(
        s.binding().root() == before.root() + 1 && s.binding().frontier() == (call + 1) as u64,
        "actual Qwen teacher commit frontier",
    )?;
    events.push(json!({"kind":"qwen-completed-committed","comparison":comparison,"before":binding(before),"after":binding(s.binding())}));
    Ok(())
}
fn gstep(
    s: &mut Gemma4Bf16GpuSession,
    expected: &mut gexpected::Expected,
    normal: bool,
    counts: &mut Counts,
    events: &mut Vec<Value>,
) -> Proof<()> {
    let before = s.binding();
    expected.begin(0);
    let pending = if normal {
        block_on(s.prepare_prompt(1, None, None))?
    } else {
        block_on(s.prepare_prompt(1, Some(expected), None))?
    };
    counts.g_calls += 1;
    if !normal {
        counts.g_instrumented += 1;
    }
    let comparison = if normal {
        expected.normal_packet(0, pending.readback(), before)?;
        json!({"normal_observer_none":true,"packet_exact":true,"ordinary_chunk_bytes":pending.readback().chunk_bytes})
    } else {
        let c = expected.completed(pending.readback(), before)?;
        need(
            c["exact"] == true,
            "all35/fullV/15KV diagnostic comparisons",
        )?;
        c
    };
    need(s.binding() == before, "Gemma Ready has no publication")?;
    let call = s.commit(pending)?;
    counts.g_commits += 1;
    need(
        call.root == s.binding()
            && call.root.root() == before.root() + 1
            && call.root.frontier() == 1
            && call.selected_id == expected.cell.calls[0].selected_id,
        "actual Gemma source/root/private feedback publication",
    )?;
    if normal {
        let snapshot = s.snapshot()?;
        expected.normal_state(0, &block_on(snapshot.read_domains())?)?;
        drop(snapshot);
    }
    events.push(json!({"kind":"gemma-completed-committed","comparison":comparison,"before":binding(before),"after":binding(s.binding()),"selected_id":call.selected_id,"max_bits":call.max_bits}));
    Ok(())
}
fn new_gstep(
    g: &Arc<Gemma4Bf16GpuCandidate>,
    ids: Arc<[u32]>,
    expected: &mut gexpected::Expected,
    counts: &mut Counts,
    events: &mut Vec<Value>,
) -> Proof<()> {
    let mut s = block_on(g.session(ids))?;
    gstep(&mut s, expected, true, counts, events)?;
    drop(s);
    Ok(())
}
fn run() -> Proof<()> {
    let a = std::env::args().skip(1).collect::<Vec<_>>();
    need(
        a.len() == 6,
        "usage: hip_shared_models MODE ABS_QCACHE ABS_GCACHE ABS_QEXPECTED ABS_GEXPECTED_RECEIPT ABS_FRESH_REPORT",
    )?;
    need(
        matches!(
            a[0].as_str(),
            "known-qwen-reload" | "qwen-final-reader-unknown" | "gemma-feedback-unknown"
        ),
        "three finite shared modes only",
    )?;
    let paths = a[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    need(
        paths.iter().all(|p| p.is_absolute()) && !paths[4].exists(),
        "absolute source/cache/reference and fresh report paths",
    )?;
    let q_scope: &[(usize, usize)] = match a[0].as_str() {
        "known-qwen-reload" => &[(0, 0), (0, 1), (1, 0)],
        "qwen-final-reader-unknown" | "gemma-feedback-unknown" => &[(0, 0)],
        _ => unreachable!("validated three finite shared modes"),
    };
    let (want_q_calls, want_q_commits, want_g_calls, want_g_instrumented, want_g_commits) =
        match a[0].as_str() {
            "known-qwen-reload" => (3, 3, 4, 1, 4),
            "qwen-final-reader-unknown" => (1, 0, 1, 1, 1),
            "gemma-feedback-unknown" => (1, 1, 1, 1, 1),
            _ => unreachable!("validated three finite shared modes"),
        };
    let wanted_events: &[(&str, usize)] = match a[0].as_str() {
        "known-qwen-reload" => &[
            ("both-original-sources-live", 1),
            ("qwen-completed-committed", 3),
            ("gemma-completed-committed", 4),
            ("known-qwen-retired", 1),
            ("actual-qwen-reloaded", 1),
            ("fourth-preparation-refused-preeffect", 1),
        ],
        "qwen-final-reader-unknown" => &[
            ("both-original-sources-live", 1),
            ("gemma-completed-committed", 1),
            ("actual-qwen-final-reader-fault", 1),
        ],
        "gemma-feedback-unknown" => &[
            ("both-original-sources-live", 1),
            ("qwen-completed-committed", 1),
            ("gemma-completed-committed", 1),
            ("actual-gemma-final-feedback-fault", 1),
        ],
        _ => unreachable!("validated three finite shared modes"),
    };
    let qctx = Qwen27bGpuContext::new_shared_hip_qualification()?;
    let gctx = Gemma4CpuContext::new_shared_hip_qualification()?;
    let arbiter = ResourceArbiter::process_shared();
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let gpu_key = PhysicalDeviceKey::new("pci", "0000:c6:00.0")?;
    let harness = arbiter.admit(
        &host_key,
        ResourceOwner::new("shared-HIP-fixture", "bounded-source-frame-report")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 128 * 1024 * 1024,
        }],
    )?;
    let qs = CacheStore::open_existing(CacheOptions::new(Some(paths[0].clone()), 96 * GIB))?;
    let gs = CacheStore::open_existing(CacheOptions::new(Some(paths[1].clone()), 96 * GIB))?;
    let q_snapshot = Arc::new(qs.open_snapshot(QSNAP)?);
    let g_snapshot = Arc::new(gs.open_snapshot(GSNAP)?);
    let input = gctx.prepare_text(&g_snapshot, "One", None)?;
    let g_ids: Arc<[u32]> = Arc::from(input.ids());
    need(g_ids.as_ref() == [4906], "actual managed One token")?;
    let q_source = Qwen27bGpuLoaded::open(q_snapshot.clone(), qctx.clone(), None)?;
    let g_source = Gemma4CpuLoaded::open(g_snapshot.clone(), gctx.clone(), None)?;
    let host = GemmaTextHostOwners {
        additional_recorded_calls: 1,
        additional_frame_bytes: 32 * 1024 * 1024,
    };
    let plan = q_source.shared_hip_cold_plan(&g_source, &input, &host)?;
    let cold = json!({"qwen":format!("{:?}",plan.qwen()),"gemma":format!("{:?}",plan.gemma()),"provider_upper":plan.complete_provider_upper_bytes(),"host_upper":plan.complete_host_upper_bytes()});
    need(
        plan.complete_host_upper_bytes()
            .checked_add(128 * 1024 * 1024)
            .and_then(|n| n.checked_add(512 * 1024 * 1024))
            .is_some_and(|n| n <= 32 * GIB as usize),
        "complete cold host plus Rust fixture and external supervisor before bootstrap",
    )?;
    let shared = SharedHipNative::bootstrap(Device::Hip(Default::default()), plan)?;
    let mut events = Vec::new();
    let mut counts = Counts::default();
    let mut oracle_scope = None;
    // All finite failures after bootstrap retain an actual partial report.
    let result = (|| -> Proof<()> {
        let q = block_on(q_source.load_shared_hip_qualification(shared.clone(), None))?;
        let (g, mut g_session) =
            block_on(g_source.load_shared_hip_qualification(shared.clone(), input, host, None))?;
        let qi = q.native_identity();
        let gi = g.native_identity()?;
        need(
            q.peer_native_view_refused(&g),
            "actual prepared token rejects sibling pinned view on same actor",
        )?;
        events.push(json!({"kind":"both-original-sources-live","qwen":identity(qi),"gemma":identity(gi),"native":live(&shared,qi,gi)?,"qwen_actual":format!("{:?}",q.inventory()),"gemma_actual":format!("{:?}",g.inventory())}));
        let mut qe = qexpected::Expected::new(paths[2].clone())?;
        let matrix = gexpected::Matrix::open(&paths[3])?;
        let mut ge = matrix.cell(0, g_source.source_identity())?;
        need(
            ge.cell.prompt_ids.as_slice() == g_ids.as_ref(),
            "actual tokenizer equals expected-only source sequence",
        )?;
        let mut q_session = block_on(q.session())?;
        if a[0] == "qwen-final-reader-unknown" {
            qe.begin(0, 0)?;
            let before = q_session.binding();
            let pending = block_on(q_session.prepare_teacher(&mut qe, None))?;
            counts.q_calls += 1;
            let comparison = qe.finish()?;
            need(
                comparison["exact"] == true && q_session.binding() == before,
                "real full64 Ready before fault",
            )?;
            gstep(&mut g_session, &mut ge, false, &mut counts, &mut events)?;
            drop((q_session, q));
            let receipt = pending.qualify_final_owner_drop(true)?;
            events.push(json!({"kind":"actual-qwen-final-reader-fault","comparison":comparison,"receipt":receipt_json!(receipt),"native":native(&shared)?}));
            let sibling = block_on(g.session(g_ids.clone()));
            need(
                sibling.is_err(),
                "global unknown refuses sibling new root before enqueue",
            )?;
            drop(sibling);
            need(
                q_source.clone().close(None).is_err() && g_source.clone().close(None).is_err(),
                "both authentic sources retained after shared unknown",
            )?;
            drop((g_session, g));
        } else if a[0] == "gemma-feedback-unknown" {
            qstep(&mut q_session, &mut qe, 0, 0, &mut counts, &mut events)?;
            gstep(&mut g_session, &mut ge, false, &mut counts, &mut events)?;
            drop(g);
            let receipt = g_session.qualify_final_feedback_drop(true)?;
            events.push(json!({"kind":"actual-gemma-final-feedback-fault","receipt":receipt_json!(receipt),"native":native(&shared)?}));
            let before = q_session.binding();
            let before_scope = qe.summary();
            need(
                qe.in_progress() == (0, 0),
                "no Qwen producer progress before refused declaration",
            )?;
            qe.begin(0, 1)?;
            let refused = block_on(q_session.prepare_teacher(&mut qe, None));
            need(
                refused.is_err()
                    && q_session.binding() == before
                    && qe.summary() == before_scope
                    && qe.in_progress() == (0, 0),
                "global unknown refuses actual sibling cached proposal/publication without progress",
            )?;
            drop(refused);
            need(
                q_source.clone().close(None).is_err() && g_source.clone().close(None).is_err(),
                "both authentic sources retained after shared unknown",
            )?;
            drop((q_session, q));
        } else {
            qstep(&mut q_session, &mut qe, 0, 0, &mut counts, &mut events)?;
            gstep(&mut g_session, &mut ge, false, &mut counts, &mut events)?;
            drop(g_session);
            qstep(&mut q_session, &mut qe, 0, 1, &mut counts, &mut events)?;
            let snapshot = q_session.snapshot()?;
            drop((q_session, q));
            let receipt = snapshot.qualify_final_owner_drop(false)?;
            need(
                q_source.native_owner_count()? == 0
                    && !shared.stream_preflight_usable(qi.logical_stream.unwrap())?,
                "actual Qwen last source reader retired; stale source view refuses",
            )?;
            let after = native(&shared)?;
            need(
                after["sets"].as_array().is_some_and(|s| s.len() == 1)
                    && after["compiler_host_reserved"] == 16 * GIB
                    && after["available_model_capacity"] == 12 * GIB,
                "Qwen native/model quota retired, compiler16GiB retained, Gemma sibling still live",
            )?;
            events.push(json!({"kind":"known-qwen-retired","receipt":receipt_json!(receipt),"native":after}));
            new_gstep(&g, g_ids.clone(), &mut ge, &mut counts, &mut events)?;
            let reload = block_on(q_source.load_shared_hip_qualification(shared.clone(), None))?;
            let ri = reload.native_identity();
            need(
                ri.source == qi.source
                    && ri.source_service_generation != qi.source_service_generation
                    && ri.set_key > qi.set_key
                    && ri.logical_stream != qi.logical_stream,
                "same authentic source reload has new real service/key/logical identity",
            )?;
            events.push(json!({"kind":"actual-qwen-reloaded","identity":identity(ri),"native":live(&shared,ri,gi)?}));
            let mut rs = block_on(reload.session())?;
            qstep(&mut rs, &mut qe, 1, 0, &mut counts, &mut events)?;
            drop((rs, reload));
            need(
                q_source.native_owner_count()? == 0
                    && !shared.stream_preflight_usable(ri.logical_stream.unwrap())?,
                "reloaded Qwen readers really retire",
            )?;
            new_gstep(&g, g_ids.clone(), &mut ge, &mut counts, &mut events)?;
            let before = native(&shared)?;
            let refused = block_on(q_source.load_shared_hip_qualification(shared.clone(), None));
            need(
                matches!(&refused,Err(zerogravity::Error::Limit(why)) if why=="HIP cold cumulative compiler preparation generations exhausted"),
                "fourth admitted preparation refused before queue submission/RTC",
            )?;
            drop(refused);
            let after = native(&shared)?;
            need(
                before == after
                    && q_source.native_owner_count()? == 0
                    && after["compiler_host_reserved"] == 24 * GIB,
                "ordinary precompiler refusal has no set/ledger/compiler credit or sibling fencing",
            )?;
            events.push(json!({"kind":"fourth-preparation-refused-preeffect","native":after}));
            new_gstep(&g, g_ids.clone(), &mut ge, &mut counts, &mut events)?;
            drop(g);
            need(
                shared.inspect_native()?.sets.is_empty(),
                "all actual native sets retired",
            )?;
        }
        oracle_scope = Some(qe.finish_scoped(q_scope)?);
        need(
            counts.q_calls == want_q_calls
                && counts.q_commits == want_q_commits
                && counts.g_calls == want_g_calls
                && counts.g_instrumented == want_g_instrumented
                && counts.g_commits == want_g_commits,
            "mode-specific actual Qwen/Gemma calls/commits and full35 evidence scope",
        )?;
        for &(kind, expected) in wanted_events {
            need(
                events.iter().filter(|e| e["kind"] == kind).count() == expected,
                format!("mode-specific lifecycle event {kind} exactly {expected}"),
            )?;
        }
        ge.after()?;
        Ok(())
    })();
    let fault = a[0] != "known-qwen-reload";
    let qclose = q_source.clone().close(None);
    let gclose = g_source.clone().close(None);
    let final_native = native(&shared);
    let actual_compiler_host = shared.compiler_host_reserved_bytes()?;
    drop((
        q_source, g_source, q_snapshot, g_snapshot, qs, gs, qctx, gctx, shared,
    ));
    let gpu = arbiter.snapshot(&gpu_key)?;
    let host = arbiter.snapshot(&host_key)?;
    let final_ok = if fault {
        gpu.fenced && gpu.reserved == 32 * GIB && qclose.is_err() && gclose.is_err()
    } else {
        !gpu.fenced
            && gpu.reserved == 4 * GIB
            && qclose.is_ok()
            && gclose.is_ok()
            && host.reserved == 26 * GIB + 64 * 1024 + 128 * 1024 * 1024
    };
    let success = result.is_ok() && final_ok;
    write_new(
        &paths[4],
        &json!({"schema":"zg.hip-two-source-original-lifecycle.v1","mode":a[0],"status":if success{"PASS_FINITE_SHARED_SOURCE_SCOPE"}else{"REFUSED_UNQUALIFIED"},"failure":result.as_ref().err().map(ToString::to_string),"cold":cold,"events":events,"qwen_declared_scope":q_scope.iter().map(|&(repeat,call)|json!([repeat,call])).collect::<Vec<_>>(),"qwen_oracle_scope":oracle_scope,"actual_completed_full64_calls":counts.q_calls,"actual_qwen_commits":counts.q_commits,"actual_completed_full35_calls":counts.g_calls,"actual_gemma_calls":counts.g_calls,"actual_instrumented_full35_calls":counts.g_instrumented,"actual_ordinary_packet_fullkv_calls":counts.g_calls-counts.g_instrumented,"actual_gemma_commits":counts.g_commits,"qwen_source_close_refused":qclose.is_err(),"gemma_source_close_refused":gclose.is_err(),"final_native":final_native.ok(),"physical_reserved":gpu.reserved,"physical_fenced":gpu.fenced,"host_reserved":host.reserved,"host_fenced":host.fenced,"diagnostic_host":128*1024*1024,"persistent_compiler_host":actual_compiler_host,"trained_serving_speed_or_general_fit_qualified":false}),
    )?;
    drop(harness);
    result?;
    need(success, "actual final source/native/physical/host outcome")
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
