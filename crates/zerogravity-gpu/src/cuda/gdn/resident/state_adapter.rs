//! Staged exact canonical-root transport, never a StateRoot/Ready issuer.
use super::*;
use crate::cuda::native::native_gdn::NativeGdnBinding;
use crate::resident_inference::gdn_state::BoundGdnDomains;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_leases::ResidentReadback;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{SourceOperatorKind, SourceStateAccess, StateSlot};

const STATE_COPY_SOURCE: &str = include_str!("state_copy.cu");
const METADATA_BYTES: usize = 2 * 1024 * 1024;
const CONV_ELEMENTS: usize = 6144 * 3;
const RECURRENT_ELEMENTS: usize = 16 * 128 * 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StateCopy {
    ConvIn,
    ConvOut,
    RecurrentIn,
    RecurrentOut,
}
impl StateCopy {
    pub(super) fn tag(self) -> usize {
        match self {
            Self::ConvIn => 0,
            Self::ConvOut => 1,
            Self::RecurrentIn => 2,
            Self::RecurrentOut => 3,
        }
    }
    fn elements(self) -> usize {
        if matches!(self, Self::ConvIn | Self::ConvOut) {
            CONV_ELEMENTS
        } else {
            RECURRENT_ELEMENTS
        }
    }
    fn entry(self) -> &'static str {
        match self {
            Self::ConvIn => "zg_vllm_root_conv_in",
            Self::ConvOut => "zg_vllm_root_conv_out",
            Self::RecurrentIn => "zg_vllm_root_h_in",
            Self::RecurrentOut => "zg_vllm_root_h_out",
        }
    }
    fn source_index(self, index: usize) -> Result<usize> {
        if index >= self.elements() {
            return Err(Error::Limit("state copy index".into()));
        }
        Ok(match self {
            Self::ConvIn => (index % 6144) * 3 + index / 6144,
            Self::ConvOut => (index % 3) * 6144 + index / 3,
            Self::RecurrentIn | Self::RecurrentOut => {
                (index / 16384) * 16384 + (index % 128) * 128 + (index / 128) % 128
            }
        })
    }
    fn widths(self) -> (usize, usize) {
        match self {
            Self::ConvIn => (4, 2),
            Self::ConvOut => (2, 4),
            _ => (4, 4),
        }
    }
    fn source(self) -> Arc<str> {
        let (input, output) = match self {
            Self::ConvIn => ("unsigned int", "unsigned short"),
            Self::ConvOut => ("unsigned short", "unsigned int"),
            _ => ("unsigned int", "unsigned int"),
        };
        format!("// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_ENTRY {}\n#define ZG_COPY {}\n#define ZG_ELEMENTS {}u\n#define ZG_INPUT {input}\n#define ZG_OUTPUT {output}\n{STATE_COPY_SOURCE}", self.entry(), self.tag(), self.elements()).into()
    }
}
pub(super) struct StateTransportDefinition {
    kind: StateCopy,
    kernel: PinnedKernel,
    grid: [u32; 3],
}
pub(super) struct StateTransport {
    kernel: PinnedKernel,
    grid: [u32; 3],
    handles: Vec<Handle>,
}
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
pub(super) fn plane_handle(cache: &ResidentLease, offset: usize, bytes: usize) -> Result<Handle> {
    let end = offset
        .checked_add(bytes)
        .filter(|&end| end <= cache.bytes())
        .ok_or_else(|| Error::State("native state cache plane range".into()))?;
    Ok(cache
        .handle()
        .offset_start(offset as u64)
        .offset_end((cache.bytes() - end) as u64))
}
impl StateTransportDefinition {
    /// Complete source/ABI/key plus cold construction peak. The generated
    /// header and entry suffix are bounded512B; no source text is rendered here.
    pub(super) fn metadata_bound() -> usize {
        4 * (3 * (STATE_COPY_SOURCE.len() + 512)
            + PinnedKernel::KEY_METADATA_BYTES
            + 3 * std::mem::size_of::<(usize, BufferIOAttr)>()
            + std::mem::size_of::<Self>()
            + 64)
    }
    pub(super) fn prepare(kind: StateCopy) -> Result<Self> {
        let (input_width, output_width) = kind.widths();
        if kind.source_index(kind.elements() - 1)? >= kind.elements() {
            return Err(Error::State("state transport source extent".into()));
        }
        let bindings = Arc::from([
            (kind.elements() * input_width, BufferIOAttr::ReadOnly),
            (kind.elements() * output_width, BufferIOAttr::WriteOnly),
            (4, BufferIOAttr::ReadWrite),
        ]);
        let kernel = PinnedKernel::new(kind.entry(), kind.source(), 128, bindings)?;
        let grid = [(kind.elements() as u32).div_ceil(128), 1, 1];
        if kernel.retained_metadata_bytes()? + std::mem::size_of::<Self>()
            > Self::metadata_bound() / 4
        {
            return Err(Error::Limit(
                "state transport cold source/ABI/key owner bound".into(),
            ));
        }
        Ok(Self { kind, kernel, grid })
    }
    pub(super) fn bind(
        &self,
        service: &Arc<ResidentService>,
        kind: StateCopy,
        input: Handle,
        output: Handle,
        status: &ResidentLease,
    ) -> Result<StateTransport> {
        if kind != self.kind {
            return Err(Error::State(
                "state transport cold definition kind differs".into(),
            ));
        }
        let handles = vec![input, output, status.handle()];
        self.kernel
            .validate(service.client(), self.grid, &handles)?;
        Ok(StateTransport {
            kernel: self.kernel.clone(),
            grid: self.grid,
            handles,
        })
    }
}
impl StateTransport {
    pub(super) fn prepare(
        service: &Arc<ResidentService>,
        kind: StateCopy,
        input: Handle,
        output: Handle,
        status: &ResidentLease,
    ) -> Result<Self> {
        StateTransportDefinition::prepare(kind)?.bind(service, kind, input, output, status)
    }
    pub(super) fn launch(&self, service: &Arc<ResidentService>) -> Result<()> {
        self.kernel
            .launch(service.client(), self.grid, &self.handles)
    }
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

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
pub(super) fn prepare_code(queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
    for kind in [
        StateCopy::ConvIn,
        StateCopy::ConvOut,
        StateCopy::RecurrentIn,
        StateCopy::RecurrentOut,
    ] {
        StateTransportDefinition::prepare(kind)?
            .kernel
            .prepare_code(queue)?;
    }
    Ok(4)
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
