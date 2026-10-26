//! Explicit physical qualification probes over the original resident owners.
use super::*;
use std::sync::Weak;
use zerogravity_core::resource_arbiter::PhysicalDeviceKey;

/// Retains the original service; never owns an independent queue or admission.
pub struct QualificationService(Arc<ResidentService>);
pub struct QualificationWitness(Weak<ResidentService>);
pub struct QualificationAllocation {
    _lease: ResidentLease,
}
impl QualificationWitness {
    pub fn upgrade(&self) -> Option<QualificationService> {
        self.0.upgrade().map(QualificationService)
    }
}
impl QualificationService {
    pub fn new(
        device: Device,
        source: zerogravity_core::accelerator::SourceIdentity,
        owner: ResourceOwner,
        capacity: usize,
        device_cap: u64,
    ) -> Result<Self> {
        ResidentService::new(device, source, owner, capacity, device_cap).map(Self)
    }
    pub fn client(&self) -> cubecl::client::Client {
        self.0.client().clone()
    }
    pub fn source_identity(&self) -> zerogravity_core::accelerator::SourceIdentity {
        self.0.source_identity()
    }
    pub fn physical_key(&self) -> PhysicalDeviceKey {
        self.0.physical_key()
    }
    pub fn resident_bytes(&self) -> Result<usize> {
        self.0.resident_bytes()
    }
    pub fn remaining_capacity(&self) -> Result<usize> {
        self.0.remaining_capacity()
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub fn physical_remaining(&self) -> Result<u64> {
        self.0.physical_remaining()
    }
    pub fn uploaded_value_bytes(&self) -> u64 {
        self.0.uploaded_value_bytes()
    }
    pub fn pending_events(&self) -> Result<usize> {
        self.0.pending_events()
    }
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }
    pub fn ensure_usable(&self) -> Result<()> {
        self.0.ensure_usable()
    }
    pub fn allocate_uninitialized(
        &self,
        bytes: usize,
        role: BufferRole,
    ) -> Result<QualificationAllocation> {
        self.0
            .allocate_uninitialized(bytes, role)
            .map(|lease| QualificationAllocation { _lease: lease })
    }
}
impl ResidentModel {
    pub fn qualification(&self) -> QualificationService {
        QualificationService(self.service.clone())
    }
    pub fn qualification_witness(&self) -> QualificationWitness {
        QualificationWitness(Arc::downgrade(&self.service))
    }
    pub fn state_bytes(&self) -> usize {
        self.state_bytes
    }
    pub fn state_requirements(&self) -> &[StateRequirement] {
        &self.states
    }
    pub fn vocabulary(&self) -> u64 {
        self.vocab
    }
    pub fn is_cuda_service(&self) -> bool {
        crate::quantization::activation_fused::is_cuda_service(self.service.client())
    }
    pub async fn qualify_aliases(self: &Arc<Self>) -> Result<()> {
        alias_qualification::qualify(self).await
    }
    pub async fn qualify_ranges(self: &Arc<Self>) -> Result<()> {
        range_qualification::qualify(self).await
    }
    pub async fn qualify_views(self: &Arc<Self>) -> Result<()> {
        view_qualification::qualify(self).await
    }
    pub async fn qualify_greedy(self: &Arc<Self>) -> Result<()> {
        let _admission = self.allocation.lock().await;
        crate::resident_greedy_qualification::run(&self.service)
            .await
            .map(|_| ())
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub fn qualification_arm_abort(&self, after: usize) {
        self.abort_after_launch
            .store(after, std::sync::atomic::Ordering::Release);
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub fn qualification_reset_launch_seen(&self) {
        self.launch_seen
            .store(0, std::sync::atomic::Ordering::Release);
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub fn qualification_launch_seen(&self) -> usize {
        self.launch_seen.load(std::sync::atomic::Ordering::Acquire)
    }
    #[cfg(feature = "cuda")]
    pub async fn qualification_release_graph_band(&self) -> Result<()> {
        let _allocation = self.allocation.lock().await;
        let mut workspace = self.workspace.lock().await;
        if workspace
            .batch
            .as_ref()
            .is_none_or(|b| b.program.graph_pinned())
        {
            return Err(Error::State(
                "retired graph still pins the original batch band".into(),
            ));
        }
        workspace.batch = None;
        Ok(())
    }
}
impl ResidentSnapshot {
    pub fn binding(&self) -> RootBinding {
        self.binding
    }
    pub fn qualification_stale_generation(&self) -> Self {
        let mut changed = self.clone();
        changed.domains[0].generations[0] += 1;
        changed
    }
}
impl PreparedInference {
    pub fn qualification_readback(&self) -> &InferenceReadback {
        &self.readback
    }
    pub fn qualification_expected_root(&self) -> u64 {
        self.ready.expected_root()
    }
}
impl ResidentToken {
    pub fn qualification_producer(&self) -> Option<RootBinding> {
        self.producer
    }
}

impl ResidentSession {
    pub async fn qualification_prepare_state_only(
        &self,
        tokens: &[u32],
    ) -> Result<PreparedInference> {
        self.prepare_mode(tokens, false, Readout::None).await
    }
}

pub use super::Readout;
pub use super::qualification::{compare_domains, exact};
pub use crate::physical_quarantine::quarantined_handles_for;
#[cfg(feature = "cuda")]
pub fn quarantined_completion_owners_for(client: &cubecl::client::Client) -> Result<usize> {
    crate::physical_quarantine::quarantined_completion_owners_for(client)
}
pub use crate::resident_program::{ResidentOutput, StateLayout, StateRequirement};
pub mod prefill {
    pub use super::super::prefill_qualification::*;
}
pub mod fused {
    pub use super::super::fused_qualification::*;
}
pub mod fused_greedy {
    pub use super::super::fused_greedy_qualification::*;
}
#[cfg(feature = "cuda")]
pub mod graph {
    pub use super::super::graph::qualification::*;
}
#[cfg(feature = "cuda")]
pub mod graph_wide {
    pub use super::super::graph::wider::*;
}
#[cfg(all(feature = "portable-checkpoint", feature = "hardware-abort-probe"))]
pub use super::checkpoint_abort::abort_upload;
#[cfg(feature = "cuda")]
pub use super::graph::{CapturedResidentGraph, GraphCaptureReceipt};

impl ResidentModel {
    #[cfg(feature = "cuda")]
    pub async fn qualification_load_graph(
        device: Device,
        source: Arc<dyn TensorSource>,
        plan: ForwardPlan,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<Arc<Self>> {
        Self::load_inner(device, source, plan, provider_cap, device_cap, true).await
    }
    pub async fn qualification_source_signature(
        &self,
    ) -> Result<(
        Vec<(zerogravity_core::device_events::BufferLeaseId, usize)>,
        usize,
    )> {
        let workspace = self.workspace.lock().await;
        let signature = workspace
            .leases
            .iter()
            .filter(|(_, access)| *access == DeviceAccess::Read)
            .map(|(lease, _)| (lease.id(), lease.bytes()))
            .collect();
        Ok((signature, workspace.program.budget().source_bytes))
    }
    pub async fn qualification_require_initialized_suffix(&self) -> Result<()> {
        let workspace = self.workspace.lock().await;
        for node in
            &workspace.program.plan().nodes()[workspace.program.plan().state_nodes().len()..]
        {
            for id in &node.outputs {
                let value = &workspace.program.values()[id.0 as usize];
                if !lease_for(&workspace.leases, &value.handle)?.is_initialized() {
                    return Err(Error::State(
                        "H17 full graph failed to initialize suffix outputs".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    pub async fn qualification_batch_bytes(&self) -> Result<usize> {
        let workspace = self.workspace.lock().await;
        workspace
            .batch
            .as_ref()
            .ok_or_else(|| Error::State("fused boundary lost native workspace".into()))?
            .program
            .budget()
            .owned_bytes()
    }
}

#[cfg(feature = "hardware-abort-probe")]
pub async fn qualify_preserved_prior(
    client: &cubecl::client::Client,
    saved: &ResidentSnapshot,
    expected: &[(StateSlot, Vec<f32>)],
    binding: RootBinding,
) -> Result<()> {
    for (cursor, domain) in saved.domains.iter().enumerate() {
        let handles = domain.leases.iter().map(ResidentLease::handle).collect();
        let data = client
            .read_async(handles)
            .await
            .map_err(|e| Error::State(format!("diagnostic prior-root readback failed: {e}")))?;
        if data.len() != domain.leases.len()
            || data
                .iter()
                .zip(&domain.leases)
                .any(|(bytes, lease)| bytes.len() != lease.bytes())
        {
            return Err(Error::State(
                "diagnostic prior-root full-capacity readback incomplete".into(),
            ));
        }
        let mut flat = Vec::new();
        for (l, b) in domain.leases.iter().zip(data) {
            let v = words(&b)?;
            if matches!(&domain.buffers, StateBuffers::Kv { .. }) {
                let active = l.bytes() / 4 / saved.capacity * binding.frontier() as usize;
                if v[active..].iter().any(|x| x.to_bits() != 0) {
                    return Err(Error::State(
                        "aborted graph changed prior canonical KV tail".into(),
                    ));
                }
                flat.extend_from_slice(&v[..active]);
            } else {
                flat.extend(v);
            }
        }
        exact(
            &expected[cursor].1,
            &flat,
            "physically preserved nonzero root after abort",
        )?;
    }
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
pub async fn qualify_preserved_packets(
    client: &cubecl::client::Client,
    incoming: &[ResidentToken],
    selected: &[(u32, u32)],
) -> Result<()> {
    let raw = client
        .read_async(incoming.iter().map(|token| token.lease.handle()).collect())
        .await
        .map_err(|error| Error::State(format!("greedy abort original packet read: {error}")))?;
    if raw.len() != 2 {
        return Err(Error::State("greedy abort packet read coverage".into()));
    }
    for (bytes, &(id, bits)) in raw.iter().zip(selected) {
        if bytes.len() != 12
            || u32::from_ne_bytes(bytes[0..4].try_into().unwrap()) != id
            || u32::from_ne_bytes(bytes[4..8].try_into().unwrap()) != bits
            || bytes[8..12].iter().any(|byte| *byte != 0)
        {
            return Err(Error::State(
                "greedy aborted feedback mutated original packet".into(),
            ));
        }
    }
    Ok(())
}

impl ResidentSession {
    pub fn qualification_model(&self) -> &Arc<ResidentModel> {
        &self.model
    }
}
