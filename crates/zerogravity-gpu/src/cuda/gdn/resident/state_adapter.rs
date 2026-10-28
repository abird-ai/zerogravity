//! Staged exact canonical-root transport, never a StateRoot/Ready issuer.
use super::*;
use crate::cuda::native::native_gdn::NativeGdnBinding;
use crate::resident_inference::gdn_state::BoundGdnDomains;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_leases::ResidentReadback;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{SourceOperatorKind, SourceStateAccess, StateSlot};

const METADATA_BYTES: usize = 2 * 1024 * 1024;

use super::state_transport::{
    CONV_ELEMENTS, RECURRENT_ELEMENTS, StateCopy, StateTransport, plane_handle,
};

struct AdapterSource {
    binding: Arc<NativeGdnBinding>,
    // Actual config/descriptor metadata drops before its original admission.
    _metadata: WindowWorkspace,
}
/// Caller retains this in the SAME armed original flight before either node.
pub(crate) struct GdnStateAdapter {
    copies: [StateTransport; 4],
    domains: Arc<BoundGdnDomains>,
    cache: ResidentLease,
    status: ResidentLease,
    slot: GdnCacheSlot,
    source: Arc<AdapterSource>,
}
impl GdnStateAdapter {
    /// Existing fair preparation slot; original host metadata and init guard.
    /// This never imports captured state or initializes a prior root by itself.
    pub(crate) async fn prepare(
        service: Arc<ResidentService>,
        binding: Arc<NativeGdnBinding>,
        domains: Arc<BoundGdnDomains>,
        cache: ResidentLease,
        slot: GdnCacheSlot,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        service.ensure_usable()?;
        if binding.source_binding().source().source_identity() != service.source_identity()
            || binding.invocation() != domains.invocation()
            || !matches!(binding.operators()[0].kind(), SourceOperatorKind::CausalConvSplit { domain, .. } if domain == domains.conv_domain()?)
            || !matches!(binding.operators()[1].kind(), SourceOperatorKind::GatedDelta { domain, .. } if domain == domains.recurrent_domain()?)
            || binding.operators()[0].state_access()
                != SourceStateAccess::Write(StateSlot::Conv(domains.conv_domain()?))
            || binding.operators()[1].state_access()
                != SourceStateAccess::Write(StateSlot::Recurrent(domains.recurrent_domain()?))
            || domains.base().source() != service.source_identity()
            || !Arc::ptr_eq(cache.service(), &service)
            || cache.service_generation() != service.generation()
            || !cache.is_initialized()
            || cache.bytes() != GdnBuffer::CacheArena.bytes(domains.phase())
        {
            return Err(Error::State(
                "same original GDN root/source/service/cache".into(),
            ));
        }
        slot.validate(domains.phase())?;
        let leases = domains.leases();
        for (index, (lease, access)) in leases.iter().enumerate() {
            if *access == DeviceAccess::Read {
                domains.require_prepared_prior(lease)?;
            }
            if !Arc::ptr_eq(lease.service(), &service)
                || lease.source_identity() != service.source_identity()
                || lease.service_generation() != service.generation()
                || lease.bytes()
                    != if index < 2 {
                        CONV_ELEMENTS * 4
                    } else {
                        RECURRENT_ELEMENTS * 4
                    }
                || lease.id() == cache.id()
            {
                return Err(Error::State("original GDN root domain lease/layout".into()));
            }
        }
        let metadata = binding
            .source_binding()
            .source()
            .reserve_workspace(METADATA_BYTES)?;
        let source = Arc::new(AdapterSource {
            binding,
            _metadata: metadata,
        });
        let mut init = service.begin_initialization(domains.base().root())?;
        init.retain_completion_owner(source.clone())?;
        init.retain_completion_owner(domains.clone())?;
        let status = init.upload_bytes(&[0; 4], BufferRole::Scratch)?;
        let conv_plane = plane_handle(&cache, slot.convolution_offset(), CONV_STATE_BYTES)?;
        let h_plane = plane_handle(&cache, slot.recurrent_offset(), RECURRENT_STATE_BYTES)?;
        let copies = [
            StateTransport::prepare(
                &service,
                StateCopy::ConvIn,
                leases[0].0.handle(),
                conv_plane.clone(),
                &status,
            )?,
            StateTransport::prepare(
                &service,
                StateCopy::ConvOut,
                conv_plane,
                leases[1].0.handle(),
                &status,
            )?,
            StateTransport::prepare(
                &service,
                StateCopy::RecurrentIn,
                leases[2].0.handle(),
                h_plane.clone(),
                &status,
            )?,
            StateTransport::prepare(
                &service,
                StateCopy::RecurrentOut,
                h_plane,
                leases[3].0.handle(),
                &status,
            )?,
        ];
        drop(init.complete_sync().await?);
        not_cancelled(cancelled)?;
        Ok(Arc::new(Self {
            copies,
            domains,
            cache,
            status,
            slot,
            source,
        }))
    }
    pub(crate) fn domains(&self) -> &BoundGdnDomains {
        &self.domains
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
    pub(crate) fn validate_status(actual: &[u8]) -> Result<()> {
        if actual != [0u8; 4] {
            return Err(Error::State(
                "exact finite BF16/F32 canonical GDN state transport".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn require_effect(
        &self,
        effect: zerogravity_core::accelerator::program::StateEffect,
        convolution: bool,
    ) -> Result<()> {
        let expected = self.domains.effect(convolution);
        if effect.read != expected.read || effect.write != expected.write {
            return Err(Error::State(
                "GDN node differs from original bound domain effect".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn require_prepared(&self, prepared: &GdnPrepared) -> Result<()> {
        let service = self.cache.service();
        service.ensure_usable()?;
        self.source
            .binding
            .require_prepared(prepared, &self.domains)?;
        if !Arc::ptr_eq(service, &prepared.service)
            || prepared.phase != self.domains.phase()
            || prepared.bound_lease(GdnBuffer::CacheArena)?.id() != self.cache.id()
            || prepared.slot != self.slot
        {
            return Err(Error::State(
                "original GDN prepared/root adapter binding".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn require_enqueue(&self, permit: Option<&PrefixPhaseEnqueue<'_>>) -> Result<()> {
        self.domains.require_enqueue(self.cache.service(), permit)
    }
    pub(super) fn copy(
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_logical_transport_indices_are_exact_inverse_permutations() {
        for (forward, inverse) in [
            (StateCopy::ConvIn, StateCopy::ConvOut),
            (StateCopy::RecurrentIn, StateCopy::RecurrentOut),
        ] {
            for index in 0..forward.elements() {
                let mapped = forward.source_index(index).unwrap();
                assert!(mapped < inverse.elements());
                assert_eq!(inverse.source_index(mapped).unwrap(), index);
            }
            assert!(forward.source_index(forward.elements()).is_err());
            assert!(inverse.source_index(usize::MAX).is_err());
        }
    }
    #[test]
    fn native_planes_preserve_pitch_and_full_extent_and_transport_sizes() {
        assert_eq!(StateCopy::ConvIn.elements() * 4, 73728);
        assert_eq!(StateCopy::ConvIn.elements() * 2, CONV_STATE_BYTES);
        assert_eq!(StateCopy::RecurrentIn.elements() * 4, RECURRENT_STATE_BYTES);
        for slot in [GdnCacheSlot::One, GdnCacheSlot::Four] {
            assert_eq!(
                slot.recurrent_offset() - slot.convolution_offset(),
                CONV_STATE_BYTES
            );
            assert!(
                slot.recurrent_offset() + RECURRENT_STATE_BYTES <= CACHE_SLOTS * CACHE_SLOT_BYTES
            );
        }
        assert!(GdnStateAdapter::validate_status(&[0; 4]).is_ok());
        for value in [1u32, 2, 4, 8, 15] {
            assert!(GdnStateAdapter::validate_status(&value.to_le_bytes()).is_err());
        }
        assert!(GdnStateAdapter::validate_status(&[0; 3]).is_err());
    }
}
