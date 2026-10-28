//! Same canonical GDN bit transports; distinct full-layer source/root binding.
use super::state_transport::{StateCopy, StateTransport, StateTransportDefinition, plane_handle};
use super::*;
use crate::cuda::native::model_gdn::ModelGdnBinding;
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue, model_state::BoundModelState,
};
use crate::resident_leases::ResidentReadback;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::StateEffect;

const CONV_ELEMENTS: usize = 6144 * 3;
const H_ELEMENTS: usize = 16 * 128 * 128;
pub(crate) struct ModelGdnAdapter {
    copies: [StateTransport; 4],
    domains: Arc<BoundModelState>,
    binding: Arc<ModelGdnBinding>,
    cache: ResidentLease,
    status: ResidentLease,
    slot: GdnCacheSlot,
    _transport: Option<Arc<ModelGdnTransportStorage>>,
    _metadata: WindowWorkspace,
}
impl ModelGdnAdapter {
    pub(crate) async fn prepare(
        binding: Arc<ModelGdnBinding>,
        domains: Arc<BoundModelState>,
        prepared: Arc<GdnPrepared>,
        cache: ResidentLease,
        phase: GdnPhase,
        slot: GdnCacheSlot,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        binding.require_prepared(&prepared, &domains)?;
        slot.validate(phase)?;
        let service = cache.service();
        if domains.conv_domain()?
            != zerogravity_core::accelerator::program::ConvDomain(binding.layer())
            || domains.recurrent_domain()?
                != zerogravity_core::accelerator::program::RecurrentDomain(binding.layer())
            || binding.call() != domains.call()
            || binding.layer() != domains.layer()
            || !Arc::ptr_eq(service, &prepared.service)
            || service.source_identity() != binding.source().source_identity()
            || cache.bytes() != GdnBuffer::CacheArena.bytes(phase)
            || !cache.is_initialized()
            || prepared.phase != phase
            || prepared.slot != slot
            || prepared.bound_lease(GdnBuffer::CacheArena)?.id() != cache.id()
        {
            return Err(Error::State(
                "full24 original GDN source/root/native preparation".into(),
            ));
        }
        let leases = domains.leases();
        for (index, (lease, _)) in leases.iter().enumerate() {
            if !Arc::ptr_eq(lease.service(), service)
                || lease.source_identity() != service.source_identity()
                || lease.service_generation() != service.generation()
                || lease.bytes()
                    != if index < 2 {
                        CONV_ELEMENTS * 4
                    } else {
                        H_ELEMENTS * 4
                    }
                || lease.id() == cache.id()
            {
                return Err(Error::State(
                    "full24 exact original canonical GDN transport leases".into(),
                ));
            }
        }
        let metadata = binding.source().reserve_workspace(64 * 1024)?;
        let mut init = service.begin_initialization(domains.base().root())?;
        init.retain_completion_owner(binding.clone())?;
        init.retain_completion_owner(domains.clone())?;
        let status = init.upload_bytes(&[0; 4], BufferRole::Scratch)?;
        let conv = plane_handle(&cache, slot.convolution_offset(), CONV_STATE_BYTES)?;
        let recurrent = plane_handle(&cache, slot.recurrent_offset(), RECURRENT_STATE_BYTES)?;
        // These are the SAME StateCopy kernels and exact bit layouts used by
        // existing prefixes; no recurrence/normalization equation is copied.
        let copies = [
            StateTransport::prepare(
                service,
                StateCopy::ConvIn,
                leases[0].0.handle(),
                conv.clone(),
                &status,
            )?,
            StateTransport::prepare(
                service,
                StateCopy::ConvOut,
                conv,
                leases[1].0.handle(),
                &status,
            )?,
            StateTransport::prepare(
                service,
                StateCopy::RecurrentIn,
                leases[2].0.handle(),
                recurrent.clone(),
                &status,
            )?,
            StateTransport::prepare(
                service,
                StateCopy::RecurrentOut,
                recurrent,
                leases[3].0.handle(),
                &status,
            )?,
        ];
        drop(init.complete_sync().await?);
        not_cancelled(cancelled)?;
        Ok(Arc::new(Self {
            copies,
            domains,
            binding,
            cache,
            status,
            slot,
            _transport: None,
            _metadata: metadata,
        }))
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.domains
            .leases()
            .into_iter()
            .chain([
                (self.cache.clone(), DeviceAccess::Write),
                (self.status.clone(), DeviceAccess::Write),
            ])
            .collect()
    }
    pub(crate) fn status_readback(&self) -> Result<ResidentReadback> {
        self.status.readback(0, 4)
    }
    fn require_prepared(&self, prepared: &GdnPrepared) -> Result<()> {
        self.binding.require_prepared(prepared, &self.domains)?;
        if !Arc::ptr_eq(&prepared.service, self.cache.service())
            || prepared.bound_lease(GdnBuffer::CacheArena)?.id() != self.cache.id()
            || prepared.slot != self.slot
        {
            return Err(Error::State(
                "full24 original GDN math/cache adapter changed".into(),
            ));
        }
        self.cache.service().ensure_usable()
    }
    fn require_enqueue(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        effect: StateEffect,
        recurrent: bool,
    ) -> Result<()> {
        self.domains.require_enqueue(self.cache.service(), permit)?;
        self.domains.require_effect(effect, recurrent)
    }
    fn copy(
        &self,
        kind: StateCopy,
        prepared: &GdnPrepared,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.require_prepared(prepared)?;
        self.copies[kind.tag()].launch(&prepared.service)?;
        Ok(1)
    }
}

#[must_use = "same armed original flight; Conv is not a complete full-model proposal"]
pub(crate) struct ModelGdnConvEnqueued {
    prepared: Arc<GdnPrepared>,
    adapter: Arc<ModelGdnAdapter>,
    dispatches: usize,
}
impl GdnPrepared {
    pub(crate) fn launch_model_convolution(
        self: &Arc<Self>,
        adapter: Arc<ModelGdnAdapter>,
        effect: StateEffect,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ModelGdnConvEnqueued> {
        not_cancelled(cancelled)?;
        adapter.require_enqueue(permit, effect, false)?;
        adapter.require_prepared(self)?;
        let dispatches = adapter.copy(StateCopy::ConvIn, self, cancelled)?
            + self.launch_partition(StagePartition::Convolution, cancelled)?
            + adapter.copy(StateCopy::ConvOut, self, cancelled)?;
        Ok(ModelGdnConvEnqueued {
            prepared: self.clone(),
            adapter,
            dispatches,
        })
    }
}
impl ModelGdnConvEnqueued {
    pub(crate) fn convolution_dispatches(&self) -> usize {
        self.dispatches
    }
    pub(crate) fn launch_recurrent(
        self,
        effect: StateEffect,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.adapter.require_enqueue(permit, effect, true)?;
        let count = self
            .adapter
            .copy(StateCopy::RecurrentIn, &self.prepared, cancelled)?
            + self
                .prepared
                .launch_partition(StagePartition::Recurrent, cancelled)?
            + self
                .adapter
                .copy(StateCopy::RecurrentOut, &self.prepared, cancelled)?;
        not_cancelled(cancelled)?;
        Ok(count)
    }
}

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod natural;

mod fixed_prompt_natural;
pub(crate) use fixed_prompt_natural::ModelGdnTransportStorage;

impl ModelGdnAdapter {
    pub(crate) fn storage_accesses_partition(
        &self,
        recurrent: bool,
    ) -> Vec<(ResidentLease, DeviceAccess)> {
        let states = self.domains.leases();
        let offset = if recurrent { 2 } else { 0 };
        vec![
            states[offset].clone(),
            states[offset + 1].clone(),
            (self.cache.clone(), DeviceAccess::Write),
            (self.status.clone(), DeviceAccess::Write),
        ]
    }
}
