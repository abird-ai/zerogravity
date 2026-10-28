//! SAME GDN partitions and canonical transport, sealed to actual 2B producers.
use super::*;
use crate::cuda::gdn::state_transport::{StateCopy, StateTransport, plane_handle};
use crate::cuda::native::qwen2b_component::ComponentGdnBinding;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use zerogravity_core::accelerator::program::StateEffect;

pub(crate) struct ComponentGdnAdapter {
    copies: [StateTransport; 4],
    binding: Arc<ComponentGdnBinding>,
}
#[must_use = "the original armed flight owns an incomplete proposal until recurrence"]
pub(crate) struct ComponentConvEnqueued {
    prepared: Arc<GdnPrepared>,
    adapter: Arc<ComponentGdnAdapter>,
    dispatches: usize,
}
impl ComponentGdnAdapter {
    pub(crate) fn prepare(
        binding: Arc<ComponentGdnBinding>,
        prepared: &GdnPrepared,
    ) -> Result<Arc<Self>> {
        binding.require_prepared(prepared)?;
        let cache = binding.cache();
        let service = binding.values().service();
        let status = binding.values().status();
        let slot = binding.slot();
        let domains = binding.domains().leases();
        for (lease, access) in &domains {
            if !Arc::ptr_eq(lease.service(), service)
                || lease.service_generation() != service.generation()
                || lease.source_identity() != service.source_identity()
                || lease.id() == cache.id()
                || lease.id() == status.id()
            {
                return Err(Error::State(
                    "source2B genuine canonical domain service/generation/nonalias".into(),
                ));
            }
            if *access == DeviceAccess::Read {
                binding.domains().require_prepared_prior(lease)?;
            }
        }
        let conv = plane_handle(cache, slot.convolution_offset(), CONV_STATE_BYTES)?;
        let h = plane_handle(cache, slot.recurrent_offset(), RECURRENT_STATE_BYTES)?;
        let copies = [
            StateTransport::prepare(
                service,
                StateCopy::ConvIn,
                domains[0].0.handle(),
                conv.clone(),
                status,
            )?,
            StateTransport::prepare(
                service,
                StateCopy::ConvOut,
                conv,
                domains[1].0.handle(),
                status,
            )?,
            StateTransport::prepare(
                service,
                StateCopy::RecurrentIn,
                domains[2].0.handle(),
                h.clone(),
                status,
            )?,
            StateTransport::prepare(
                service,
                StateCopy::RecurrentOut,
                h,
                domains[3].0.handle(),
                status,
            )?,
        ];
        Ok(Arc::new(Self { copies, binding }))
    }
    fn require(
        &self,
        prepared: &GdnPrepared,
        effect: StateEffect,
        convolution: bool,
        permit: &PrefixPhaseEnqueue<'_>,
    ) -> Result<()> {
        let service = self.binding.values().service();
        self.binding
            .domains()
            .require_enqueue(service, Some(permit))?;
        self.binding.require_prepared(prepared)?;
        let expected = self.binding.domains().effect(convolution);
        if effect.read != expected.read
            || effect.write != expected.write
            || !Arc::ptr_eq(&prepared.service, service)
            || prepared.phase != self.binding.phase()?
            || prepared.slot != self.binding.slot()
            || !prepared.typed_outputs
            || !match (&prepared.code, self.binding.values().recipe().kind()) {
                (
                    ComponentCode::Qwen2b(b),
                    crate::resident_program::qwen2b_component::ComponentKind::Qwen2b,
                ) => Arc::ptr_eq(b, &self.binding),
                (
                    ComponentCode::FixedPrompt(b),
                    crate::resident_program::qwen2b_component::ComponentKind::FixedPrompt(_),
                ) => Arc::ptr_eq(b, &self.binding),
                _ => false,
            }
        {
            return Err(Error::State(
                "source2B exact code/parameter/producer/root effect seal".into(),
            ));
        }
        Ok(())
    }
    fn copy(
        &self,
        kind: StateCopy,
        prepared: &GdnPrepared,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.binding.require_prepared(prepared)?;
        self.copies[kind.tag()].launch(&prepared.service)?;
        Ok(1)
    }
}
impl GdnPrepared {
    pub(crate) fn qwen2b_component_code_bytes(phase: GdnPhase) -> Result<usize> {
        if !matches!(phase, GdnPhase::InitialOne | GdnPhase::Initial64) {
            return Err(Error::Unsupported(
                "source2B component initial code only".into(),
            ));
        }
        phase
            .stages()
            .iter()
            .try_fold(256 * 1024usize, |sum, &stage| {
                sum.checked_add(
                    assets::select_qwen2b(phase, stage)?
                        .source
                        .len()
                        .checked_mul(4)
                        .ok_or_else(|| Error::Limit("source2B code bytes".into()))?,
                )
                .and_then(|v| v.checked_add(4096))
                .ok_or_else(|| Error::Limit("source2B source-code admission".into()))
            })
    }
    pub(crate) async fn prepare_qwen2b_component_outputs(
        binding: Arc<ComponentGdnBinding>,
        inputs: GdnInputs,
        outputs: GdnTypedOutputs,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        binding.require_source(binding.values().service())?;
        if binding.values().recipe().kind()
            != crate::resident_program::qwen2b_component::ComponentKind::Qwen2b
        {
            return Err(Error::Unsupported(
                "original2B component code remains closed".into(),
            ));
        }
        Self::prepare_impl(
            binding.values().service().clone(),
            binding.phase()?,
            binding.slot(),
            inputs,
            Some(outputs),
            ComponentCode::Qwen2b(binding),
            cancelled,
        )
        .await
    }
    pub(crate) fn fixed_prompt_component_code_bytes(
        arithmetic: super::super::FixedPromptGdnArithmetic,
    ) -> Result<usize> {
        arithmetic.require_owned()?;
        GdnPhase::FixedPrompt
            .stages()
            .iter()
            .try_fold(256 * 1024usize, |sum, &stage| {
                sum.checked_add(
                    assets::select(GdnPhase::FixedPrompt, stage)?
                        .source
                        .len()
                        .checked_mul(4)
                        .ok_or_else(|| Error::Limit("fixed_prompt source bytes".into()))?,
                )
                .and_then(|n| n.checked_add(4096))
                .ok_or_else(|| Error::Limit("fixed_prompt bounded source code admission".into()))
            })
    }
    pub(crate) async fn prepare_fixed_prompt_component_outputs(
        binding: Arc<ComponentGdnBinding>,
        inputs: GdnInputs,
        outputs: GdnTypedOutputs,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let crate::resident_program::qwen2b_component::ComponentKind::FixedPrompt(arithmetic) =
            binding.values().recipe().kind()
        else {
            return Err(Error::Unsupported(
                "fixed_prompt source binding required".into(),
            ));
        };
        arithmetic.require_owned()?;
        binding.require_source(binding.values().service())?;
        let prepared = Self::prepare_impl(
            binding.values().service().clone(),
            GdnPhase::FixedPrompt,
            GdnCacheSlot::One,
            inputs,
            Some(outputs),
            ComponentCode::FixedPrompt(binding),
            cancelled,
        )
        .await?;
        arithmetic.require_prepared(prepared.arithmetic())?;
        Ok(prepared)
    }
    pub(crate) fn launch_component_convolution(
        self: &Arc<Self>,
        adapter: Arc<ComponentGdnAdapter>,
        effect: StateEffect,
        cancelled: Option<&AtomicBool>,
        permit: &PrefixPhaseEnqueue<'_>,
    ) -> Result<ComponentConvEnqueued> {
        not_cancelled(cancelled)?;
        adapter.require(self, effect, true, permit)?;
        let dispatches = adapter.copy(StateCopy::ConvIn, self, cancelled)?
            + self.launch_partition(StagePartition::Convolution, cancelled)?
            + adapter.copy(StateCopy::ConvOut, self, cancelled)?;
        Ok(ComponentConvEnqueued {
            prepared: self.clone(),
            adapter,
            dispatches,
        })
    }
}
impl ComponentConvEnqueued {
    pub(crate) fn dispatches(&self) -> usize {
        self.dispatches
    }
    pub(crate) fn launch_recurrent(
        self,
        effect: StateEffect,
        cancelled: Option<&AtomicBool>,
        permit: &PrefixPhaseEnqueue<'_>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.adapter
            .require(&self.prepared, effect, false, permit)?;
        let dispatches = self
            .adapter
            .copy(StateCopy::RecurrentIn, &self.prepared, cancelled)?
            + self
                .prepared
                .launch_partition(StagePartition::Recurrent, cancelled)?
            + self
                .adapter
                .copy(StateCopy::RecurrentOut, &self.prepared, cancelled)?;
        not_cancelled(cancelled)?;
        Ok(dispatches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn component_code_is_exact_initial2b_and_default_assets_stay_separate() {
        for phase in [
            GdnPhase::CachedDecode,
            GdnPhase::ContinuedTwo,
            GdnPhase::FixedPrompt,
        ] {
            assert!(GdnPrepared::qwen2b_component_code_bytes(phase).is_err());
        }
        assert_eq!(
            GdnPrepared::qwen2b_component_code_bytes(GdnPhase::InitialOne).unwrap(),
            2_431_424
        );
        assert_eq!(
            GdnPrepared::qwen2b_component_code_bytes(GdnPhase::Initial64).unwrap(),
            2_497_184
        );
        let old = kernel(GdnPhase::InitialOne, GdnStage::Conv).unwrap();
        let new = kernel_qwen2b_component(GdnPhase::InitialOne, GdnStage::Conv).unwrap();
        assert_eq!(old.grid, new.grid);
        assert_eq!(
            old.bindings
                .iter()
                .map(|b| (b.role, b.offset, b.physical_bytes))
                .collect::<Vec<_>>(),
            new.bindings
                .iter()
                .map(|b| (b.role, b.offset, b.physical_bytes))
                .collect::<Vec<_>>()
        );
        assert_ne!(
            assets::select(GdnPhase::InitialOne, GdnStage::Conv)
                .unwrap()
                .source,
            assets::select_qwen2b(GdnPhase::InitialOne, GdnStage::Conv)
                .unwrap()
                .source
        );
        assert_eq!(
            assets::select(GdnPhase::InitialOne, GdnStage::Cumsum)
                .unwrap()
                .threads,
            128
        );
        assert_eq!(
            assets::select_qwen2b(GdnPhase::InitialOne, GdnStage::Cumsum)
                .unwrap()
                .threads,
            64
        );
    }
}
