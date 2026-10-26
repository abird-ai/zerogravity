//! Learned Gemma Vulkan same-process/same-PCI peer probe.
//!
//! One sealed learned Vulkan owner and one finite HIP peer share the original
//! process arbiter/physical card. Both are priced before the submitter launches
//! an unknown; the opposite provider must then refuse. No portable finite
//! fixture is retagged as the learned Vulkan primary, and no serving permit is
//! produced.
use super::*;
#[path = "../portable_qualification/finite_peer.rs"]
mod finite_peer;
use finite_peer::{FiniteHipPeer, PeerRefusal};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

const SAME_PCI_HOST_LIMIT: u64 = 16 * 1024 * 1024 * 1024;
const SAME_PCI_DEVICE_LIMIT: u64 = 64 * 1024 * 1024 * 1024;

/// Actual typed Vulkan native owner/retention observed through the sealed
/// Gemma programme. It is a source owner proof, not a performance permit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextVulkanNativeRetention {
    pub set_key: u64,
    pub source_service_generation: u64,
    pub actor: String,
    pub retained: bool,
    pub modules: usize,
    pub code_bytes: usize,
    pub host_bytes: usize,
    pub device_bytes: usize,
    pub persistent_context_host_bytes: usize,
    pub persistent_context_device_bytes: usize,
    pub native_host_bytes: usize,
    pub native_device_bytes: usize,
}

/// Same-process/same-PCI learned-Vulkan + finite-HIP peer disposition.
///
/// Capability fields stay false: successful source preparation is not natural
/// Gemma serving or performance qualification.
#[derive(Clone, Debug)]
pub struct LearnedSamePciPeerReceipt {
    pub same_process: bool,
    pub same_pci: bool,
    pub same_arbiter: bool,
    pub same_queue: bool,
    pub distinct_services: bool,
    pub learned_vulkan_owner: bool,
    pub finite_hip_peer: bool,
    pub direction: &'static str,
    pub source_host_limit: u64,
    pub host_limit: u64,
    pub host_reserved: u64,
    pub host_pre_admitted: bool,
    pub physical_limit: u64,
    pub physical_reserved: u64,
    pub vulkan_provider_device: u64,
    pub vulkan_context_host: u64,
    pub vulkan_context_device: u64,
    pub vulkan_native_host: u64,
    pub vulkan_native_device: u64,
    pub hip_source_host: u64,
    pub hip_provider_device: u64,
    pub hip_context_host: u64,
    pub hip_compiler_host: u64,
    pub hip_context_device: u64,
    pub hip_compiler_device: u64,
    pub submitted_fault: bool,
    pub submitter_retained_handles: usize,
    pub peer_refused: bool,
    pub peer_new_service_refused: bool,
    pub peer_root_unchanged: bool,
    pub peer_pending_unchanged: bool,
    pub peer_resident_unchanged: bool,
    pub peer_uploads_unchanged: bool,
    pub peer_queue_unchanged: bool,
    pub vulkan_native_retained: bool,
    pub vulkan_set_key: u64,
    pub vulkan_actor: String,
    pub vulkan_native_modules: usize,
    pub vulkan_native_code_bytes: usize,
    pub vulkan_native_host_bytes: usize,
    pub vulkan_native_device_bytes: usize,
    pub vulkan_managed_handles: usize,
    pub hip_source_retained: bool,
    pub hip_service_retained: bool,
    pub hip_managed_handles: usize,
    pub hip_managed_handles_added: usize,
    pub physical_fenced: bool,
    pub original_source_retained: bool,
    pub original_native_retained: bool,
    pub original_handle_retained: bool,
    pub trained_native_serving_or_performance_qualified: bool,
    pub trained_native_natural_serving_or_performance_qualified: bool,
}

impl Bf16TextCandidate {
    /// Read the actual sealed Vulkan prepared set, its service identity and
    /// the retained module/representation/device admission. No new allocation
    /// or compilation is performed by this observation.
    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        feature = "hardware-abort-probe",
        target_os = "linux"
    ))]
    pub fn vulkan_native_retention(&self) -> Result<TextVulkanNativeRetention> {
        let identity = self.vulkan_native_identity()?;
        let workspace = self.model.workspace.try_lock().ok_or_else(|| {
            Error::State("BF16 text Vulkan native observation workspace busy".into())
        })?;
        let observation = workspace
            .program
            .bf16_text_final_owner_native_probe()?
            .observe()?;
        Ok(TextVulkanNativeRetention {
            set_key: identity.set_key,
            source_service_generation: identity.source_service_generation,
            actor: format!("{:?}", identity.actor),
            retained: observation.retained,
            modules: observation.modules,
            code_bytes: observation.code_bytes,
            host_bytes: observation.host_bytes,
            device_bytes: observation.device_bytes,
            persistent_context_host_bytes: self.inventory.persistent_context_host_bytes,
            persistent_context_device_bytes: self.inventory.persistent_context_device_bytes,
            native_host_bytes: self.inventory.native_host_bytes,
            native_device_bytes: self.inventory.native_device_bytes,
        })
    }

    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        feature = "hardware-abort-probe",
        target_os = "linux"
    ))]
    pub fn vulkan_quarantined_handles(&self) -> Result<usize> {
        crate::physical_quarantine::quarantined_handles_for(self.model.service.client())
    }
}

impl Bf16TextSession {
    /// Launch the original learned Gemma prompt and drop its completed Future
    /// only after the actual native launch is observable. This deliberately
    /// leaves the original flight/queue RAII owning the unknown.
    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        feature = "hip",
        feature = "hardware-abort-probe",
        target_os = "linux"
    ))]
    pub async fn abandon_after_launch(&self) -> Result<usize> {
        let loaded = self.loaded.clone();
        let before = self.loaded.vulkan_quarantined_handles()?;
        loaded
            .model
            .launch_seen
            .store(0, std::sync::atomic::Ordering::Release);
        loaded
            .model
            .abort_after_launch
            .store(1, std::sync::atomic::Ordering::Release);
        let mut future = Box::pin(self.prepare_prompt(1, None, None));
        let launch = std::future::poll_fn(|cx| match future.as_mut().poll(cx) {
            std::task::Poll::Ready(_) => std::task::Poll::Ready(Err(Error::State(
                "learned BF16 text Vulkan fault completed without the real launch pause".into(),
            ))),
            std::task::Poll::Pending => {
                let launch = loaded
                    .model
                    .launch_seen
                    .load(std::sync::atomic::Ordering::Acquire);
                if launch == 0 {
                    std::task::Poll::Pending
                } else {
                    std::task::Poll::Ready(Ok(launch))
                }
            }
        })
        .await?;
        if launch == 0 {
            return Err(Error::State(
                "learned BF16 text Vulkan launch count absent".into(),
            ));
        }
        drop(future);
        let handles = self.loaded.vulkan_quarantined_handles()?;
        if handles <= before {
            return Err(Error::State(
                "learned BF16 text Vulkan unknown retained no new managed handle".into(),
            ));
        }
        Ok(handles - before)
    }

    /// Actual same-process/same-PCI learned Vulkan primary plus finite HIP peer.
    /// The session is consumed; callers cannot publish after the shared fence.
    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        feature = "hip",
        feature = "hardware-abort-probe",
        target_os = "linux"
    ))]
    pub fn same_pci_peer(self, direction: &str) -> Result<LearnedSamePciPeerReceipt> {
        let direction = match direction {
            "vulkan-then-hip" => "vulkan-then-hip",
            "hip-then-vulkan" => "hip-then-vulkan",
            _ => return Err(Error::Malformed("learned same-PCI peer direction".into())),
        };
        futures_lite::future::block_on(learned_same_pci_peer(self, direction))
    }
}

#[cfg(all(
    feature = "gemma-bf16-qualification",
    feature = "vulkan",
    feature = "hip",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
async fn learned_same_pci_peer(
    session: Bf16TextSession,
    direction: &'static str,
) -> Result<LearnedSamePciPeerReceipt> {
    let candidate = session.loaded.clone();
    let physical = candidate.model.service.physical_key();
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let mut peer = FiniteHipPeer::load(physical.clone()).await?;
    let charges = peer.charges();
    let budget = ResourceArbiter::process_shared().snapshot(&physical)?;
    let host = ResourceArbiter::process_shared().snapshot(&host_key)?;
    let provider_capacity = candidate.model.service.capacity() as u64;
    let expected_physical = provider_capacity
        .checked_add(candidate.inventory.persistent_context_device_bytes as u64)
        .and_then(|n| n.checked_add(charges.total_device()))
        .ok_or_else(|| Error::Limit("learned same-PCI physical envelope sum".into()))?;
    if physical != peer.physical_key()
        || !Arc::ptr_eq(candidate.model.service.queue(), peer.queue())
        || candidate.model.service.client().service_id() == peer.service_id()
        || budget.limit != SAME_PCI_DEVICE_LIMIT
        || budget.fenced
        || budget.reserved != expected_physical
        || host.limit != SAME_PCI_HOST_LIMIT
        || host.reserved > host.limit
        || host.reserved
            < 10 * 1024 * 1024 * 1024
                + charges.source_host
                + charges.context_host
                + charges.compiler_host
        || charges.provider == 0
        || charges.context_host == 0
        || charges.compiler_host == 0
        || charges.context_device == 0
        || charges.compiler_device == 0
    {
        return Err(Error::State(
            "learned same-PCI actual provider/queue/arbiter/admission join".into(),
        ));
    }
    let before_native = candidate.vulkan_native_retention()?;
    if !before_native.retained
        || before_native.modules == 0
        || before_native.device_bytes == 0
        || before_native.native_host_bytes == 0
    {
        return Err(Error::State(
            "learned Vulkan sealed native set absent before peer fault".into(),
        ));
    }
    let vulkan_before_handles = candidate.vulkan_quarantined_handles()?;
    let hip_before_handles = peer.quarantine_handles()?;
    let submitted_fault;
    let submitter_retained_handles;
    let refusal: PeerRefusal;
    let vulkan_after_handles;

    match direction {
        "vulkan-then-hip" => {
            let retained = session.abandon_after_launch().await?;
            submitted_fault = retained > 0;
            submitter_retained_handles = retained;
            if candidate.model.service.ensure_usable().is_ok()
                || candidate.model.service.pending_events()? == 0
            {
                return Err(Error::State(
                    "learned Vulkan actual unknown did not fence its own service".into(),
                ));
            }
            vulkan_after_handles = candidate.vulkan_quarantined_handles()?;
            if vulkan_after_handles < vulkan_before_handles + retained {
                return Err(Error::State(
                    "learned Vulkan native handles not retained by quarantine".into(),
                ));
            }
            refusal = peer.refuse_after_fence().await?;
        }
        "hip-then-vulkan" => {
            peer.submit_known().await?;
            let retained = peer.abandon_after_launch().await?;
            submitted_fault = retained > 0;
            submitter_retained_handles = retained;
            if peer.pending_events()? == 0 {
                return Err(Error::State(
                    "finite HIP actual unknown has no pending physical work".into(),
                ));
            }
            let before = session.binding();
            let pending = candidate.model.service.pending_events()?;
            let resident = candidate.model.service.resident_bytes()?;
            let uploads = candidate.model.service.uploaded_value_bytes();
            let queue = candidate.model.service.queue().stats()?;
            let first = session.prepare_prompt(1, None, None).await;
            let prepare_refused = first.is_err();
            drop(first);
            let refusal_row = PeerRefusal {
                prepare_refused,
                new_service_refused: ResidentService::new(
                    session.loaded.model.service.physical_owner().device(),
                    session.loaded.source.source_identity(),
                    ResourceOwner::new("learned-same-pci", "new-after-shared-fence")?,
                    session.loaded.model.service.capacity(),
                    SAME_PCI_DEVICE_LIMIT,
                )
                .is_err(),
                root_unchanged: session.binding() == before,
                pending_unchanged: candidate.model.service.pending_events()? == pending,
                resident_unchanged: candidate.model.service.resident_bytes()? == resident,
                uploads_unchanged: candidate.model.service.uploaded_value_bytes() == uploads,
                queue_unchanged: candidate.model.service.queue().stats()? == queue,
            };
            if !refusal_row.prepare_refused
                || !refusal_row.new_service_refused
                || !refusal_row.root_unchanged
                || !refusal_row.pending_unchanged
                || !refusal_row.resident_unchanged
                || !refusal_row.uploads_unchanged
                || !refusal_row.queue_unchanged
                || candidate.model.service.ensure_usable().is_ok()
            {
                return Err(Error::State(
                    "learned Vulkan accepted work after finite HIP shared fence".into(),
                ));
            }
            refusal = refusal_row;
            vulkan_after_handles = candidate.vulkan_quarantined_handles()?;
            if vulkan_after_handles != vulkan_before_handles {
                return Err(Error::State(
                    "learned Vulkan peer gained handles without submitting".into(),
                ));
            }
        }
        _ => unreachable!(),
    }

    let after_native = candidate.vulkan_native_retention()?;
    if after_native != before_native {
        return Err(Error::State(
            "learned Vulkan native set changed across peer fault".into(),
        ));
    }
    let hip_retention = peer.retire_and_observe()?;
    // The original runtime/quarantine must retain the authentic peer source
    // admission and service after the shared physical uncertainty.
    if !hip_retention.service_retained || !hip_retention.source_retained {
        return Err(Error::State(
            "finite HIP peer source/service not retained after fault".into(),
        ));
    }
    if direction == "hip-then-vulkan" && hip_retention.handles <= hip_before_handles {
        return Err(Error::State(
            "finite HIP peer handles not conservatively quarantined".into(),
        ));
    }
    let final_budget = ResourceArbiter::process_shared().snapshot(&physical)?;
    let final_host = ResourceArbiter::process_shared().snapshot(&host_key)?;
    if !final_budget.fenced
        || final_budget.reserved != expected_physical
        || final_host.reserved > final_host.limit
    {
        return Err(Error::State(
            "learned same-PCI terminal physical/host disposition".into(),
        ));
    }
    let vulkan_after_native_handles = candidate.vulkan_quarantined_handles()?;
    Ok(LearnedSamePciPeerReceipt {
        same_process: true,
        same_pci: true,
        same_arbiter: true,
        same_queue: true,
        distinct_services: true,
        learned_vulkan_owner: true,
        finite_hip_peer: true,
        direction,
        source_host_limit: SAME_PCI_HOST_LIMIT,
        host_limit: host.limit,
        host_reserved: host.reserved,
        host_pre_admitted: true,
        physical_limit: budget.limit,
        physical_reserved: budget.reserved,
        vulkan_provider_device: provider_capacity
            + candidate.inventory.persistent_context_device_bytes as u64,
        vulkan_context_host: before_native.persistent_context_host_bytes as u64,
        vulkan_context_device: before_native.persistent_context_device_bytes as u64,
        vulkan_native_host: before_native.native_host_bytes as u64,
        vulkan_native_device: before_native.native_device_bytes as u64,
        hip_source_host: charges.source_host,
        hip_provider_device: charges.provider,
        hip_context_host: charges.context_host,
        hip_compiler_host: charges.compiler_host,
        hip_context_device: charges.context_device,
        hip_compiler_device: charges.compiler_device,
        submitted_fault,
        submitter_retained_handles,
        peer_refused: refusal.prepare_refused,
        peer_new_service_refused: refusal.new_service_refused,
        peer_root_unchanged: refusal.root_unchanged,
        peer_pending_unchanged: refusal.pending_unchanged,
        peer_resident_unchanged: refusal.resident_unchanged,
        peer_uploads_unchanged: refusal.uploads_unchanged,
        peer_queue_unchanged: refusal.queue_unchanged,
        vulkan_native_retained: after_native.retained,
        vulkan_set_key: after_native.set_key,
        vulkan_actor: after_native.actor,
        vulkan_native_modules: after_native.modules,
        vulkan_native_code_bytes: after_native.code_bytes,
        vulkan_native_host_bytes: after_native.host_bytes,
        vulkan_native_device_bytes: after_native.device_bytes,
        vulkan_managed_handles: vulkan_after_native_handles,
        hip_source_retained: hip_retention.source_retained,
        hip_service_retained: hip_retention.service_retained,
        hip_managed_handles: hip_retention.handles,
        hip_managed_handles_added: hip_retention.handles_added,
        physical_fenced: final_budget.fenced,
        original_source_retained: after_native.retained,
        original_native_retained: after_native.retained && after_native.modules > 0,
        original_handle_retained: vulkan_after_native_handles >= vulkan_before_handles
            && hip_retention.handles >= hip_before_handles,
        trained_native_serving_or_performance_qualified: false,
        trained_native_natural_serving_or_performance_qualified: false,
    })
}
