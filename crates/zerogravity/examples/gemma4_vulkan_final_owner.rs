//! Actual final committed Gemma feedback lease and native/source retirement.
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
    gemma4_cpu::{Gemma4CpuContext, Gemma4CpuLoaded},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_gpu::{
    Device, GemmaFinalOwnerReceipt, GemmaTextHostOwners, GemmaTextObserver, GemmaTextStorage,
};
const GIB: u64 = 1024 * 1024 * 1024;
const PERSISTENT_CONTEXT_HOST: u64 = 2 * GIB;
const PERSISTENT_COMPILER_HOST: u64 = 8 * GIB;
const PERSISTENT_HOST_BASELINE: u64 = PERSISTENT_CONTEXT_HOST + PERSISTENT_COMPILER_HOST;
const SNAPSHOT: &str = "ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1";
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
    best: Option<(u32, u32)>,
}
impl GemmaTextObserver for Completed {
    fn completed_producer(
        &mut self,
        position: usize,
        frame: &zerogravity::gemma4::text::Gemma4Bf16Observation,
        actual: &[u8],
    ) -> zerogravity::Result<()> {
        if position != 0 || !actual.len().is_multiple_of(4) {
            return Err(Error::State(
                "Gemma final-owner actual INITIAL1 frame".into(),
            ));
        }
        if matches!(frame.stage, "block.input" | "block.output" | "logits")
            && actual
                .chunks_exact(4)
                .any(|b| !f32::from_ne_bytes(b.try_into().unwrap()).is_finite())
        {
            return Err(Error::State(
                "Gemma final-owner nonfinite actual producer".into(),
            ));
        }
        if frame.stage == "block.output"
            && (frame.layer >= 35 || actual.len() != 1536 * 4 || !self.layers.insert(frame.layer))
        {
            return Err(Error::State(
                "Gemma final-owner full35 completed layers".into(),
            ));
        }
        if frame.stage == "logits" {
            if actual.len() != 262144 * 4 || self.full_vocab != 0 {
                return Err(Error::State(
                    "Gemma final-owner original fullV completion".into(),
                ));
            }
            let mut best = (0u32, f32::NEG_INFINITY);
            for (index, word) in actual.chunks_exact(4).enumerate() {
                let value = f32::from_ne_bytes(word.try_into().unwrap());
                if value > best.1 {
                    best = (index as u32, value);
                }
            }
            self.best = Some((best.0, best.1.to_bits()));
            self.full_vocab += 1;
        }
        Ok(())
    }
    fn completed_kv(
        &mut self,
        prior: bool,
        domain: zerogravity::accelerator::program::KvDomain,
        value: bool,
        active: usize,
        actual: &[u8],
    ) -> zerogravity::Result<()> {
        if prior
            || active != 1
            || actual.is_empty()
            || !actual.len().is_multiple_of(4)
            || !self
                .states
                .insert((format!("{domain:?}"), usize::from(value)))
        {
            return Err(Error::State(
                "Gemma final-owner actual proposed KV planes".into(),
            ));
        }
        Ok(())
    }
}
fn receipt_json(r: &GemmaFinalOwnerReceipt) -> Value {
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
        "actual_native_spirv_representation_bytes_before":r.native_code_bytes_before,
        "actual_native_host_bytes_before":r.native_host_bytes_before,
        "actual_native_device_bytes_before":r.native_device_bytes_before,
        "native_admission_retained_after":r.native_admission_retained_after,
        "actual_native_modules_after":r.native_modules_after,
        "original_service_retained_after":r.original_service_retained_after,
        "actual_quarantined_handles_after":r.quarantined_handles_after,
        "source_frontier":r.source_frontier
    })
}
fn run() -> Proof<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    need(
        args.len() == 2,
        "usage: gemma4_vulkan_final_owner MODE ABS_FRESH_REPORT",
    )?;
    let (snapshot_case, fault, partial_native) = match args[0].as_str() {
        "session-known" => (false, false, false),
        "session-release-fault" => (false, true, false),
        "snapshot-known" => (true, false, false),
        "snapshot-release-fault" => (true, true, false),
        "partial-native" => (false, true, true),
        _ => return Err("five explicit final-owner modes only".into()),
    };
    let report = PathBuf::from(&args[1]);
    need(
        report.is_absolute() && !report.exists(),
        "fresh absolute report",
    )?;
    let context = Gemma4CpuContext::new_vulkan_qualification()?;
    let arbiter = ResourceArbiter::process_shared();
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let physical_key = PhysicalDeviceKey::new("pci", "0000:c6:00.0")?;
    let harness = arbiter.admit(
        &host_key,
        ResourceOwner::new("Gemma-final-feedback", "bounded-frames-report")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 32 * 1024 * 1024,
        }],
    )?;
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * GIB))?;
    let snapshot = Arc::new(store.open_snapshot(SNAPSHOT)?);
    let loaded = Gemma4CpuLoaded::open(snapshot.clone(), context.clone(), None)?;
    let input = context.prepare_text(&snapshot, "One", None)?;
    let prompt_ids = input.ids().to_vec();
    need(
        prompt_ids.len() == 1,
        "genuine original text One is INITIAL1",
    )?;
    let device = Device::vulkan(zerogravity::gpu::WgpuDeviceKind::IntegratedGpu(0))
        .map_err(|error| Error::State(format!("original Vulkan device: {error}")))?;
    if partial_native {
        // Qualification-only actual partial construction: the vendor actor has
        // already installed real compiled modules before this fault. It is not
        // a preflight rejection, and every partial module/admission/source owner
        // must survive to the process boundary.
        unsafe {
            std::env::set_var("GEMMA_VULKAN_NATIVE_FAULT_AFTER_MODULES", "2");
        }
        let partial = {
            block_on(loaded.load_vulkan_qualification(
                device,
                input,
                GemmaTextHostOwners {
                    additional_recorded_calls: 1,
                    additional_frame_bytes: 32 * 1024 * 1024,
                },
                GemmaTextStorage::ResidentProjectionsVulkan,
                None,
            ))
        };
        let close = loaded.clone().close(None);
        let partial_error = partial.as_ref().err().map(ToString::to_string);
        let partial_modules = zerogravity_gpu::retained_partial_native_modules();
        let partial_ok = partial.is_err()
            && partial_error
                .as_deref()
                .is_some_and(|error| error.contains("actual partial native preparation fault"))
            && partial_modules.is_some_and(|modules| modules > 0)
            && close.is_err();
        drop((loaded, snapshot, store, context));
        let physical = arbiter.snapshot(&physical_key)?;
        let host = arbiter.snapshot(&host_key)?;
        let resources_match = physical.fenced && physical.reserved == 20 * GIB;
        write_new(
            &report,
            &json!({
                "schema":"zg.gemma4-original-vulkan-partial-native-owner.v1",
                "status":if partial_ok && resources_match {"PASS_ACTUAL_PARTIAL_NATIVE_RETAINED"} else {"REFUSED_UNQUALIFIED"},
                "mode":args[0],"prompt_text":"One","prompt_ids":prompt_ids,
                "partial_native_error":partial_error,
                "partial_modules_constructed_before_fault":partial_modules,
                "source_close_refused":close.is_err(),
                "source_close_error":close.as_ref().err().map(ToString::to_string),
                "physical_reserved":physical.reserved,"physical_fenced":physical.fenced,
                "host_reserved":host.reserved,"host_fenced":host.fenced,
                "context_retirement":"unknown:actual partial modules/admissions/source retained;PCI fenced to process exit",
                "numerical_replay_performed":false,"old_order_physical_negative_performed":false,
                "trained_native_serving_or_performance_qualified":false,
                "trained_native_natural_serving_or_performance_qualified":false
            }),
        )?;
        need(
            partial_ok && resources_match,
            "actual partial module retention/source fence/original resource boundary",
        )?;
        println!(
            "GEMMA_VULKAN_PARTIAL_NATIVE_PASS modules_before_fault={} source_close_refused=true pci_fenced=true process_retirement=false serving_performance=false",
            partial_modules.unwrap_or(0)
        );
        return Ok(());
    }
    let (candidate, mut session) = {
        block_on(loaded.load_vulkan_qualification(
            device,
            input,
            GemmaTextHostOwners {
                additional_recorded_calls: 1,
                additional_frame_bytes: 32 * 1024 * 1024,
            },
            GemmaTextStorage::ResidentProjectionsVulkan,
            None,
        ))?
    };
    let inventory = format!("{:?}", candidate.inventory());
    need(
        candidate.inventory().device_with_six_roots_bytes <= 16 * GIB as usize,
        "unchanged complete cold model/native/root cap",
    )?;
    let mut observer = Completed::default();
    let mut completed_calls = 0;
    let mut commits = 0;
    let mut actual_selected = None;
    let result = (|| -> Proof<GemmaFinalOwnerReceipt> {
        let before = session.binding();
        let pending = block_on(session.prepare_prompt(1, Some(&mut observer), None))?;
        completed_calls += 1;
        need(
            observer.layers.len() == 35 && observer.states.len() == 30 && observer.full_vocab == 1,
            "actual full35/all15K+V/fullV producers completed",
        )?;
        need(session.binding() == before, "Ready has not published")?;
        need(
            loaded.clone().close(None).is_err(),
            "actual managed native owner live before final wrapper",
        )?;
        let call = session.commit(pending)?;
        commits += 1;
        need(
            observer.best == Some((call.selected_id, call.max_bits)),
            "actual greedy feedback equals this whole fullV",
        )?;
        actual_selected = Some((call.selected_id, call.max_bits));
        need(
            call.root == session.binding() && call.root.frontier() == 1 && call.root.root() == 1,
            "actual committed source/root/feedback",
        )?;
        if snapshot_case {
            let retained = session.snapshot()?;
            drop((session, candidate));
            Ok(retained.qualify_final_feedback_drop(fault)?)
        } else {
            drop(candidate);
            Ok(session.qualify_final_feedback_drop(fault)?)
        }
    })();
    let close = loaded.clone().close(None);
    let expected_close = close.is_err() == fault;
    drop((loaded, snapshot, store, context));
    let physical = arbiter.snapshot(&physical_key)?;
    let host = arbiter.snapshot(&host_key)?;
    let resources_match = if fault {
        physical.fenced && physical.reserved == 20 * GIB && host.reserved > PERSISTENT_HOST_BASELINE
    } else {
        !physical.fenced
            && physical.reserved == 4 * GIB
            && !host.fenced
            && host.reserved == PERSISTENT_HOST_BASELINE + 32 * 1024 * 1024
    };
    let success = result.is_ok() && expected_close && resources_match;
    write_new(
        &report,
        &json!({
            "schema":"zg.gemma4-original-vulkan-final-feedback-owner.v1",
            "status":if success {"PASS_ACTUAL_FINAL_OWNER"} else {"REFUSED_UNQUALIFIED"},
            "mode":args[0],"prompt_text":"One","prompt_ids":prompt_ids,
            "source_frontier":result.as_ref().ok().map(|r| r.source_frontier).unwrap_or(0),
            "actual_completed_full35_calls":completed_calls,"actual_completed_layers":observer.layers.len(),
            "actual_completed_proposed_state_planes":observer.states.len(),"actual_completed_fullV":observer.full_vocab,
            "commits":commits,"actual_selected":actual_selected,"failure":result.as_ref().err().map(ToString::to_string),
            "original_final_owner":result.as_ref().ok().map(receipt_json),
            "source_close_refused":close.is_err(),"source_close_error":close.as_ref().err().map(ToString::to_string),
            "inventory":inventory,"physical_reserved":physical.reserved,"physical_fenced":physical.fenced,
            "host_reserved":host.reserved,"host_fenced":host.fenced,"diagnostic_host_reserved":32*1024*1024,"expected_persistent_context_host":PERSISTENT_CONTEXT_HOST,"persistent_compiler_host":PERSISTENT_COMPILER_HOST,"expected_persistent_host_baseline":PERSISTENT_HOST_BASELINE,
            "context_retirement":"known:model0/persistentdevice4GiBhost2GiB;fault:originalprovider/source/native retained to process exit",
            "numerical_replay_performed":false,"old_order_physical_negative_performed":false,
            "trained_native_serving_or_performance_qualified":false,
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
            arbiter.snapshot(&host_key)?.reserved == PERSISTENT_HOST_BASELINE,
            "known final diagnostic host retirement",
        )?;
    }
    println!(
        "GEMMA_VULKAN_FINAL_OWNER_PASS mode={} real_producer=full35 target_final_reader=1 native_retirement_after_reader=true source_close_refused={fault} process_retirement=true serving_performance=false",
        args[0]
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("GEMMA_VULKAN_FINAL_OWNER_REFUSED: {error}");
        std::process::exit(1);
    }
}
