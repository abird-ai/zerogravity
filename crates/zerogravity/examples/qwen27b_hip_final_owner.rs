//! Actual last Pending/Snapshot reader and source/native retirement boundary.
#[path = "qwen27b_hip_matrix/cache_import.rs"]
mod cache_import;
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
    sync::Arc,
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
    Device, QwenFinalOwnerReceipt, QwenSourceFrame, QwenSourceFrameKind, QwenSourceObserver,
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
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}
#[derive(Default)]
struct Completed {
    layers: BTreeSet<usize>,
    states: BTreeSet<(String, usize)>,
    full_vocab: usize,
}
impl QwenSourceObserver for Completed {
    fn completed(&mut self, frame: QwenSourceFrame, actual: &[u8]) -> zerogravity::Result<()> {
        let words = actual.len() / 4;
        if frame.position != 0
            || !actual.len().is_multiple_of(4)
            || frame
                .offset_words
                .checked_add(words)
                .is_none_or(|n| n > frame.total_words)
        {
            return Err(Error::State(
                "final-owner actual INITIAL1 frame geometry".into(),
            ));
        }
        if frame.offset_words + words != frame.total_words {
            return Ok(());
        }
        match frame.kind {
            QwenSourceFrameKind::Value {
                layer,
                stage: "layer_output",
            } if layer >= 64 || frame.total_words != 5120 || !self.layers.insert(layer) => {
                return Err(Error::State(
                    "final-owner original full64 completed layer".into(),
                ));
            }
            QwenSourceFrameKind::Value {
                stage: "logits", ..
            } => {
                if frame.total_words != 248320 || self.full_vocab != 0 {
                    return Err(Error::State("final-owner original fullV completion".into()));
                }
                self.full_vocab += 1;
            }
            QwenSourceFrameKind::State {
                slot,
                plane,
                prior: false,
            } if !self.states.insert((format!("{slot:?}"), plane)) => {
                return Err(Error::State(
                    "final-owner duplicate proposed state plane".into(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}
fn receipt_json(r: &QwenFinalOwnerReceipt) -> Value {
    json!({
        "wrapper":r.wrapper,"fault":r.fault,"source":r.source,
        "target_lease":r.lease,"service_generation":r.service_generation,
        "target_lease_bytes":r.lease_bytes,"aliases_inside_only_wrapper":r.aliases_inside_wrapper,
        "candidate_owners_before_drop":r.candidate_owners_before_drop,
        "model_owners_before_drop":r.model_owners_before_drop,
        "candidate_workspace_target_aliases":r.candidate_workspace_target_aliases,
        "candidate_retired":r.candidate_retired,"target_final_reader_hits":r.release_hits,
        "native_retirement_attempts":r.native_retirement_attempts,
        "reader_hits_before_native_retirement":r.reader_hits_before_native_retirement,
        "actual_native_modules_before":r.native_modules_before,
        "actual_native_PT_LOAD_bytes_before":r.native_code_bytes_before,
        "actual_native_host_bytes_before":r.native_host_bytes_before,
        "actual_native_device_bytes_before":r.native_device_bytes_before,
        "native_admission_retained_after":r.native_admission_retained_after,
        "actual_native_modules_after":r.native_modules_after,
        "original_service_retained_after":r.original_service_retained_after,
        "actual_quarantined_handles_after":r.quarantined_handles_after
    })
}
fn run() -> Proof<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() == 2 && args[0] == "--import-source" {
        return cache_import::publish(&PathBuf::from(&args[1]));
    }
    need(
        args.len() == 2,
        "usage: qwen27b_hip_final_owner MODE ABS_FRESH_REPORT | --import-source ABS_STAGE",
    )?;
    let (snapshot_case, fault) = match args[0].as_str() {
        "pending-known" => (false, false),
        "pending-release-fault" => (false, true),
        "snapshot-known" => (true, false),
        "snapshot-release-fault" => (true, true),
        _ => return Err("four explicit final-owner modes only".into()),
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
        ResourceOwner::new("Qwen27B-final-owner", "bounded-frames-report")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 64 * 1024 * 1024,
        }],
    )?;
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * GIB))?;
    let snapshot = Arc::new(store.open_snapshot(SNAPSHOT)?);
    let loaded = Qwen27bGpuLoaded::open(snapshot.clone(), context.clone(), None)?;
    let candidate = block_on(loaded.load_hip_qualification(Device::Hip(Default::default()), None))?;
    let inventory = format!("{:?}", candidate.inventory());
    need(
        candidate.inventory().complete_device_bytes <= 12 * GIB as usize,
        "unchanged complete cold model/native/root cap",
    )?;
    let mut observer = Completed::default();
    let mut completed_calls = 0;
    let mut commits = 0;
    let result = (|| -> Proof<QwenFinalOwnerReceipt> {
        let mut session = block_on(candidate.session())?;
        let before = session.binding();
        let pending = block_on(session.prepare_teacher(&mut observer, None))?;
        completed_calls += 1;
        need(
            observer.layers.len() == 64 && observer.states.len() == 128 && observer.full_vocab == 1,
            "actual full64/128proposed state planes/fullV producer completed",
        )?;
        need(
            session.binding() == before && !candidate.physical_fenced(),
            "Ready has not published/fenced",
        )?;
        need(
            loaded.clone().close(None).is_err(),
            "actual managed native owner live before final wrapper",
        )?;
        if snapshot_case {
            let output = session.commit(pending)?;
            commits += 1;
            need(
                session.binding().frontier() == 1,
                "actual snapshot committed frontier1",
            )?;
            let retained = session.snapshot()?;
            drop((output, session, candidate));
            Ok(retained.qualify_final_owner_drop(fault)?)
        } else {
            drop((session, candidate));
            Ok(pending.qualify_final_owner_drop(fault)?)
        }
    })();
    let close = loaded.clone().close(None);
    let expected_close = close.is_err() == fault;
    drop((loaded, snapshot, store, context));
    let physical = arbiter.snapshot(&physical_key)?;
    let host = arbiter.snapshot(&host_key)?;
    let resources_match = if fault {
        physical.fenced && physical.reserved == 16 * GIB && host.reserved > 2 * GIB
    } else {
        !physical.fenced
            && physical.reserved == 4 * GIB
            && !host.fenced
            && host.reserved == 2 * GIB + 64 * 1024 * 1024
    };
    let success = result.is_ok() && expected_close && resources_match;
    write_new(
        &report,
        &json!({
            "schema":"zg.qwen27b-original-hip-final-owner.v1",
            "status":if success {"PASS_ACTUAL_FINAL_OWNER"} else {"REFUSED_UNQUALIFIED"},
            "mode":args[0],"teacher_id":11,"source_frontier":0,
            "actual_completed_full64_calls":completed_calls,"actual_completed_layers":observer.layers.len(),
            "actual_completed_proposed_state_planes":observer.states.len(),"actual_completed_fullV":observer.full_vocab,
            "commits":commits,"failure":result.as_ref().err().map(ToString::to_string),
            "original_final_owner":result.as_ref().ok().map(receipt_json),
            "source_close_refused":close.is_err(),"source_close_error":close.as_ref().err().map(ToString::to_string),
            "inventory":inventory,"physical_reserved":physical.reserved,"physical_fenced":physical.fenced,
            "host_reserved":host.reserved,"host_fenced":host.fenced,"diagnostic_host_reserved":64*1024*1024,
            "context_retirement":"known:model0/persistentdevice4GiBhost2GiB;fault:originalprovider/source/native retained to process exit",
            "numerical_replay_performed":false,"old_order_physical_negative_performed":false,
            "trained_native_natural_serving_or_performance_qualified":false
        }),
    )?;
    result?;
    need(
        expected_close && resources_match,
        "final source close/fence/original resource boundary",
    )?;
    drop(harness);
    if !fault {
        need(
            arbiter.snapshot(&host_key)?.reserved == 2 * GIB,
            "known final diagnostic host retirement",
        )?;
    }
    println!(
        "QWEN_FINAL_OWNER_PASS mode={} real_producer=full64 target_final_reader=1 native_retirement_after_reader=true source_close_refused={fault} process_retirement=true serving_performance=false",
        args[0]
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("QWEN_FINAL_OWNER_REFUSED: {error}");
        std::process::exit(1);
    }
}
