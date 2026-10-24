use super::{expected::Expected, io::*};
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
    qwen27b_gpu::{Qwen27bGpuContext, Qwen27bGpuLoaded},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_gpu::{
    Device, Qwen27bSourceGpuSession, QwenSourceFrame, QwenSourceFrameKind, QwenSourceObserver,
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
fn binding(root: zerogravity::session_root::RootBinding) -> Value {
    json!({"session":root.session().get(),"root":root.root(),"frontier":root.frontier()})
}
struct DiscardObserver;
impl QwenSourceObserver for DiscardObserver {
    fn completed(&mut self, _: QwenSourceFrame, _: &[u8]) -> zerogravity::Result<()> {
        Ok(())
    }
}
struct LateCancelObserver<'a> {
    flag: &'a AtomicBool,
    completed_logits: usize,
}
impl QwenSourceObserver for LateCancelObserver<'_> {
    fn completed(&mut self, frame: QwenSourceFrame, _: &[u8]) -> zerogravity::Result<()> {
        if matches!(
            frame.kind,
            QwenSourceFrameKind::Value {
                stage: "logits",
                ..
            }
        ) {
            self.completed_logits += 1;
            self.flag.store(true, Ordering::Release);
        }
        Ok(())
    }
}
struct UnknownObserver {
    completed: usize,
}
impl QwenSourceObserver for UnknownObserver {
    fn completed(&mut self, frame: QwenSourceFrame, _: &[u8]) -> zerogravity::Result<()> {
        if matches!(
            frame.kind,
            QwenSourceFrameKind::Value {
                layer: 0,
                stage: "layer_output"
            }
        ) {
            self.completed += 1;
            return Err(Error::State(
                "deliberate actual armed Qwen producer refusal".into(),
            ));
        }
        Ok(())
    }
}
fn completed(
    session: &mut Qwen27bSourceGpuSession,
    expected: &mut Expected,
    repeat: usize,
    call: usize,
    events: &mut Vec<Value>,
) -> Proof<zerogravity_gpu::Qwen27bSourceGpuOutput> {
    expected.begin(repeat, call)?;
    let before = session.binding();
    need(
        before.frontier() == call as u64,
        "actual finite teacher frontier before submission",
    )?;
    let pending = block_on(session.prepare_teacher(expected, None))?;
    let comparison = expected.finish()?;
    events.push(json!({"kind":"actual-completed-full64-proposal","teacher_id":([11,353][call]),"before":binding(before),"comparison":comparison}));
    need(
        comparison["exact"] == true,
        "fatal bit-exact full64/state/fullV mismatch before publication",
    )?;
    let output = session.commit(pending)?;
    need(
        session.binding().frontier() == call as u64 + 1
            && session.binding().root() == before.root() + 1,
        "all fallible comparison/Ready joins precede original root publication",
    )?;
    events.push(json!({"kind":"actual-committed-call","repeat":repeat,"call":call,"root":binding(session.binding()),"dispatches":output.dispatches()}));
    Ok(output)
}
pub fn run() -> Proof<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() == 2 && args[0] == "--import-source" {
        return super::cache_import::publish(&PathBuf::from(&args[1]));
    }
    need(
        args.len() == 2 || (args.len() == 3 && args[2] == "unknown"),
        "usage: qwen27b_hip_matrix ABS_ORIGINAL_EXPECTED_DIR ABS_FRESH_REPORT [unknown] | --import-source ABS_STAGE",
    )?;
    let expected_dir = PathBuf::from(&args[0]);
    let report = PathBuf::from(&args[1]);
    let unknown = args.len() == 3;
    need(
        expected_dir.is_absolute() && report.is_absolute() && !report.exists(),
        "explicit expected/fresh report paths",
    )?;
    // SAME canonical host contract precedes every source, native and oracle owner.
    let context = Qwen27bGpuContext::new()?;
    let arbiter = ResourceArbiter::process_shared();
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let harness = arbiter.admit(
        &host_key,
        ResourceOwner::new("Qwen27B-qualification", "bounded-expected-frames-report")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 64 * 1024 * 1024,
        }],
    )?;
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * GIB))?;
    let snapshot = Arc::new(store.open_snapshot(SNAPSHOT)?);
    let loaded = Qwen27bGpuLoaded::open(snapshot.clone(), context.clone(), None)?;
    let candidate = block_on(loaded.load_hip_qualification(Device::Hip(Default::default()), None))?;
    need(
        !candidate.trained_model_qualified(),
        "finite candidate has no native/serving qualification grant",
    )?;
    let inventory = format!("{:?}", candidate.inventory());
    need(
        candidate.inventory().complete_device_bytes <= 12 * GIB as usize,
        "complete paired bodies/six roots/native quota",
    )?;
    let mut events = Vec::with_capacity(32);
    if unknown {
        let session = block_on(candidate.session())?;
        let before = session.binding();
        let mut observer = UnknownObserver { completed: 0 };
        let result = block_on(session.prepare_teacher(&mut observer, None));
        need(
            matches!(result,Err(Error::State(ref why))if why=="deliberate actual armed Qwen producer refusal")
                && observer.completed == 1
                && session.binding() == before,
            "actual completed first layer while original flight armed/no Ready/no root",
        )?;
        drop((session, candidate));
        let close = loaded.clone().close(None);
        drop((loaded, snapshot, store, context));
        let physical = arbiter.snapshot(&PhysicalDeviceKey::new("pci", "0000:c6:00.0")?)?;
        let host = arbiter.snapshot(&host_key)?;
        write_new(
            &report,
            &json!({"schema":"zg.qwen27b-original-hip-unknown.v1","status":"ACTUAL_UNKNOWN_RETAINED","completed_producers":observer.completed,"published_roots":0,"source_close_refused":close.is_err(),"physical_reserved":physical.reserved,"host_reserved":host.reserved,"physical_fenced":physical.fenced,"inventory":inventory,"context_retirement":"process lifetime; source/native/armed flight retained","trained_native_serving_or_performance_qualified":false}),
        )?;
        need(
            close.is_err()
                && physical.fenced
                && physical.reserved == 16 * GIB
                && host.reserved > 2 * GIB,
            "whole original physical envelope/source retained on unknown",
        )?;
        println!(
            "QWEN_FULL64_UNKNOWN_PASS actual_completed_producer=1 published_roots=0 original_envelope=retained source_close=refused"
        );
        return Ok(());
    }
    let mut expected = Expected::new(expected_dir)?;
    let result = (|| -> Proof<zerogravity_gpu::Qwen27bSourceGpuOutput> {
        // Exhaust original canonical admission without allocating untracked payload.
        let free = 16 * GIB - context.reserved_bytes()?;
        let pressure = arbiter.admit(
            &host_key,
            ResourceOwner::new("Qwen27B-qualification", "actual-host-pressure")?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: free,
            }],
        )?;
        need(
            block_on(candidate.session()).is_err(),
            "real exhausted host refuses session before device/root work",
        )?;
        drop(pressure);
        events.push(json!({"kind":"actual-host-pressure-refused-before-work"}));
        let mut a = block_on(candidate.session())?;
        let mut b = block_on(candidate.session())?;
        need(
            a.binding().session() != b.binding().session()
                && block_on(candidate.session()).is_err(),
            "independent two sessions/third bounded refusal",
        )?;
        events.push(json!({"kind":"two-sessions-third-refused"}));
        need(
            loaded.clone().close(None).is_err(),
            "actual native guard retains original SnapshotLease",
        )?;
        let before = a.binding();
        let flag = AtomicBool::new(true);
        let mut discard = DiscardObserver;
        need(
            block_on(a.prepare_teacher(&mut discard, Some(&flag))).is_err()
                && a.binding() == before,
            "pre-cancel leaves original root unchanged",
        )?;
        events.push(json!({"kind":"pre-cancel","root":binding(before)}));
        let pending = block_on(a.prepare_teacher(&mut discard, None))?;
        drop(pending);
        need(
            a.binding() == before,
            "actual completed Ready discard cannot publish original root",
        )?;
        events.push(json!({"kind":"actual-Ready-discard","root":binding(before),"full_extra_initial_call":true}));
        let flag = AtomicBool::new(false);
        let mut late = LateCancelObserver {
            flag: &flag,
            completed_logits: 0,
        };
        let refused = block_on(a.prepare_teacher(&mut late, Some(&flag)));
        need(
            matches!(refused, Err(Error::State(ref why)) if why == "Qwen source call cancelled")
                && late.completed_logits == 1
                && a.binding() == before
                && !candidate.physical_fenced(),
            "late cancellation after actual Ready/fullV refuses publication without unknown fence",
        )?;
        events.push(json!({"kind":"actual-completed-late-cancel","root":binding(before),"full_extra_initial_call":true}));
        let untouched = b.binding();
        let output = completed(&mut a, &mut expected, 0, 0, &mut events)?;
        need(
            b.binding() == untouched,
            "peer root unchanged after first publication",
        )?;
        drop(output);
        let snap_a = a.snapshot()?;
        let snap_b = b.snapshot()?;
        need(
            a.snapshot().is_err(),
            "third retained snapshot bounded refusal",
        )?;
        need(
            block_on(b.restore(&snap_a)).is_err() && b.binding() == untouched,
            "foreign-session snapshot refusal before publication",
        )?;
        events.push(json!({"kind":"snapshot-pressure-and-foreign-refusal"}));
        let output = completed(&mut b, &mut expected, 1, 0, &mut events)?;
        drop(output);
        let output = completed(&mut a, &mut expected, 0, 1, &mut events)?;
        drop(output);
        let output = completed(&mut b, &mut expected, 1, 1, &mut events)?;
        block_on(a.restore(&snap_a))?;
        need(
            a.binding().frontier() == 1 && b.binding().frontier() == 2,
            "snapshot restore preserves independent peer root",
        )?;
        events.push(json!({"kind":"own-snapshot-restored","root":binding(a.binding())}));
        block_on(a.reset())?;
        need(
            a.binding().frontier() == 0 && b.binding().frontier() == 2,
            "reset publishes empty root independently",
        )?;
        events.push(json!({"kind":"reset-independent-root","root":binding(a.binding())}));
        drop((snap_a, snap_b, a, b));
        expected.after()?;
        Ok(output)
    })();
    // Caller candidate/sessions/snapshots are gone. Only actual retained output
    // can keep this native/source guard live; avoid shadowing that proof.
    drop(candidate);
    let result = result.and_then(|output| {
        need(
            output.logits().len() == 248320 && loaded.clone().close(None).is_err(),
            "retained output alone keeps actual source/native/readback owner",
        )?;
        events.push(json!({"kind":"output-retains-source-after-caller-drop"}));
        drop(output);
        Ok(())
    });
    let close = loaded.clone().close(None);
    drop((loaded, snapshot, store, context));
    let physical = arbiter.snapshot(&PhysicalDeviceKey::new("pci", "0000:c6:00.0")?)?;
    let host = arbiter.snapshot(&host_key)?;
    let commits = events
        .iter()
        .filter(|e| e["kind"] == "actual-committed-call")
        .count();
    let compared = events
        .iter()
        .filter(|e| e["kind"] == "actual-completed-full64-proposal")
        .count();
    let late_cancelled = events
        .iter()
        .filter(|e| e["kind"] == "actual-completed-late-cancel")
        .count();
    let discarded = events
        .iter()
        .filter(|e| e["kind"] == "actual-Ready-discard")
        .count();
    write_new(
        &report,
        &json!({"schema":"zg.qwen27b-original-hip-full64.v1","status":if result.is_ok()&&close.is_ok(){"PASS_EXACT_FINITE_TEACHER"}else{"REFUSED_UNQUALIFIED"},"failure":result.as_ref().err().map(ToString::to_string),"source_close_error":close.as_ref().err().map(ToString::to_string),"inventory":inventory,"events":events,"comparison":expected.summary(),"positions_basis":"original I64 compared to actual bound source call frontier; no GPU I64 tensor","compared_calls":compared,"additional_Ready_discard_calls":discarded,"additional_late_cancel_calls":late_cancelled,"commits":commits,"managed_source_closed":close.is_ok(),"physical_reserved":physical.reserved,"physical_fenced":physical.fenced,"host_reserved":host.reserved,"host_fenced":host.fenced,"diagnostic_host_reserved":64*1024*1024,"context_retirement":"model0 retains device4GiB/host2GiB until actual process retirement","source_page_policy":"single synchronous admitted page; correctness baseline","trained_native_natural_serving_or_performance_qualified":false}),
    )?;
    result?;
    close?;
    drop((expected, harness, events));
    let final_host = arbiter.snapshot(&host_key)?;
    need(
        physical.reserved == 4 * GIB
            && !physical.fenced
            && final_host.reserved == 2 * GIB
            && !final_host.fenced,
        "actual module/model retirement; separate persistent context retained",
    )?;
    println!(
        "QWEN_FULL64_SOURCE_STATE_PASS proposals=4 Ready_discard_calls=1 late_cancel_calls=1 commits=4 records=1040 float_words=240896000 positions=12 mismatches=0 model_reservations=0 context_baseline=retained_until_process_exit serving_performance=false"
    );
    Ok(())
}
