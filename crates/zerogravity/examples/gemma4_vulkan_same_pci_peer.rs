//! Learned Gemma Vulkan same-PCI peer receipt; no serving/numerical permit.
#![recursion_limit = "256"]
use serde_json::json;
use std::{
    env, fs,
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
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter},
};
use zerogravity_gpu::{Device, GemmaTextHostOwners, GemmaTextStorage};

const GIB: u64 = 1024 * 1024 * 1024;
const SNAPSHOT: &str = "ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1";
const PCI: &str = "0000:c6:00.0";

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

fn main() {
    if let Err(error) = run() {
        eprintln!("LEARNED_SAME_PCI_PEER_REFUSED: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args_os().skip(1);
    let direction = args
        .next()
        .and_then(|v| v.into_string().ok())
        .unwrap_or_else(|| "vulkan-then-hip".into());
    let output = args.next().map(PathBuf::from);
    if args.next().is_some() {
        eprintln!(
            "usage: gemma4_vulkan_same_pci_peer [vulkan-then-hip|hip-then-vulkan] [receipt.json]"
        );
        std::process::exit(2);
    }
    let context = Gemma4CpuContext::new_vulkan_qualification()
        .map_err(|e| format!("original Vulkan Gemma context: {e}"))?;
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * GIB))
        .map_err(|e| format!("existing canonical Gemma cache: {e}"))?;
    let snapshot = Arc::new(
        store
            .open_snapshot(SNAPSHOT)
            .map_err(|e| format!("canonical Gemma snapshot: {e}"))?,
    );
    let loaded = Gemma4CpuLoaded::open(snapshot.clone(), context.clone(), None)
        .map_err(|e| format!("managed Gemma source owner: {e}"))?;
    let input = context
        .prepare_text(&snapshot, "One", None)
        .map_err(|e| format!("genuine Gemma INITIAL1 text: {e}"))?;
    let device = Device::vulkan(zerogravity::gpu::WgpuDeviceKind::IntegratedGpu(0))
        .map_err(|e| format!("original explicit Vulkan device: {e}"))?;
    let (candidate, session) = {
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
    }
    .map_err(|e| format!("sealed learned Vulkan source/native owner: {e}"))?;
    let inventory = format!("{:?}", candidate.inventory());
    let receipt = session
        .same_pci_peer(&direction)
        .map_err(|e| format!("learned same-PCI actual peer probe: {e}"))?;
    drop(candidate);
    let close = loaded.clone().close(None);
    let physical_key = PhysicalDeviceKey::new("pci", PCI).map_err(|e| e.to_string())?;
    let host_key = PhysicalDeviceKey::new("host", "system-ram").map_err(|e| e.to_string())?;
    let arbiter = ResourceArbiter::process_shared();
    let physical = arbiter
        .snapshot(&physical_key)
        .map_err(|e| format!("physical disposition: {e}"))?;
    let host = arbiter
        .snapshot(&host_key)
        .map_err(|e| format!("host disposition: {e}"))?;
    let actual = receipt.same_process
        && receipt.same_pci
        && receipt.same_arbiter
        && receipt.same_queue
        && receipt.distinct_services
        && receipt.learned_vulkan_owner
        && receipt.finite_hip_peer
        && receipt.host_pre_admitted
        && receipt.submitted_fault
        && receipt.submitter_retained_handles > 0
        && receipt.peer_refused
        && receipt.peer_new_service_refused
        && receipt.peer_root_unchanged
        && receipt.peer_pending_unchanged
        && receipt.peer_resident_unchanged
        && receipt.peer_uploads_unchanged
        && receipt.peer_queue_unchanged
        && receipt.vulkan_native_retained
        && receipt.vulkan_native_modules > 0
        && receipt.hip_source_retained
        && receipt.hip_service_retained
        && receipt.physical_fenced
        && receipt.original_source_retained
        && receipt.original_native_retained
        && receipt.original_handle_retained
        && close.is_err()
        && physical.fenced
        && physical.reserved == receipt.physical_reserved
        && host.reserved <= 16 * GIB;
    let report = json!({
        "schema": "zg.gemma4-original-vulkan-learned-same-pci-peer.v1",
        "status": if actual { "PASS_ACTUAL_LEARNED_SAME_PCI_PEER" } else { "REFUSED_UNQUALIFIED" },
        "direction": receipt.direction,
        "same_process": receipt.same_process,
        "same_pci": receipt.same_pci,
        "same_arbiter": receipt.same_arbiter,
        "same_queue": receipt.same_queue,
        "distinct_services": receipt.distinct_services,
        "learned_vulkan_owner": receipt.learned_vulkan_owner,
        "finite_hip_peer": receipt.finite_hip_peer,
        "source_host_limit": receipt.source_host_limit,
        "host_limit": receipt.host_limit,
        "host_reserved": receipt.host_reserved,
        "host_pre_admitted": receipt.host_pre_admitted,
        "physical_limit": receipt.physical_limit,
        "physical_reserved": receipt.physical_reserved,
        "vulkan_provider_device": receipt.vulkan_provider_device,
        "vulkan_context_host": receipt.vulkan_context_host,
        "vulkan_context_device": receipt.vulkan_context_device,
        "vulkan_native_host": receipt.vulkan_native_host,
        "vulkan_native_device": receipt.vulkan_native_device,
        "hip_source_host": receipt.hip_source_host,
        "hip_provider_device": receipt.hip_provider_device,
        "hip_context_host": receipt.hip_context_host,
        "hip_compiler_host": receipt.hip_compiler_host,
        "hip_context_device": receipt.hip_context_device,
        "hip_compiler_device": receipt.hip_compiler_device,
        "submitted_fault": receipt.submitted_fault,
        "submitter_retained_handles": receipt.submitter_retained_handles,
        "peer_refused": receipt.peer_refused,
        "peer_new_service_refused": receipt.peer_new_service_refused,
        "peer_root_unchanged": receipt.peer_root_unchanged,
        "peer_pending_unchanged": receipt.peer_pending_unchanged,
        "peer_resident_unchanged": receipt.peer_resident_unchanged,
        "peer_uploads_unchanged": receipt.peer_uploads_unchanged,
        "peer_queue_unchanged": receipt.peer_queue_unchanged,
        "vulkan_native_retained": receipt.vulkan_native_retained,
        "vulkan_set_key": receipt.vulkan_set_key,
        "vulkan_actor": receipt.vulkan_actor,
        "vulkan_native_modules": receipt.vulkan_native_modules,
        "vulkan_native_code_bytes": receipt.vulkan_native_code_bytes,
        "vulkan_native_host_bytes": receipt.vulkan_native_host_bytes,
        "vulkan_native_device_bytes": receipt.vulkan_native_device_bytes,
        "vulkan_managed_handles": receipt.vulkan_managed_handles,
        "hip_source_retained": receipt.hip_source_retained,
        "hip_service_retained": receipt.hip_service_retained,
        "hip_managed_handles": receipt.hip_managed_handles,
        "hip_managed_handles_added": receipt.hip_managed_handles_added,
        "physical_fenced": receipt.physical_fenced,
        "original_source_retained": receipt.original_source_retained,
        "original_native_retained": receipt.original_native_retained,
        "original_handle_retained": receipt.original_handle_retained,
        "source_close_refused": close.is_err(),
        "source_close_error": close.as_ref().err().map(ToString::to_string),
        "inventory": inventory,
        "physical_snapshot_reserved": physical.reserved,
        "physical_snapshot_fenced": physical.fenced,
        "host_snapshot_reserved": host.reserved,
        "host_snapshot_fenced": host.fenced,
        "trained_native_serving_or_performance_qualified": false,
        "trained_native_natural_serving_or_performance_qualified": false,
    });
    let rendered = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    if let Some(path) = output {
        fs::write(path, rendered.as_bytes()).map_err(|e| format!("peer receipt write: {e}"))?;
    }
    println!("{rendered}");
    if actual {
        Ok(())
    } else {
        Err("learned same-PCI source/native/host/physical peer join is incomplete".into())
    }
}
