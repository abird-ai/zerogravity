//! Genuine FinalRMS253 full47 producer, exact existing N1 vocabulary packet.
use super::*;
use crate::cuda::rms::FusedRmsPrepared;
use crate::resident_leases::{TIED_BYTES, TIED_WINDOWS, TiedWeight};
use cubecl::cuda::{Bf16VocabularyHead, PinnedBf16Cubin, VocabularyHeadSelection};
use zerogravity_core::accelerator::program::PlanOperation;

pub(crate) fn required_images() -> BTreeSet<ProjectionImage> {
    let mut set = super::super::fixed_prompt_mlp::required_images();
    set.extend([
        ProjectionImage::AttentionAmpere256x128Sm80,
        ProjectionImage::AttentionQkSm86,
        ProjectionImage::AttentionKvSm86,
        ProjectionImage::AttentionFaCausalCombineSm80,
        ProjectionImage::AttentionSigmoidSm86,
        ProjectionImage::AttentionMultiplySm86,
        ProjectionImage::Bf16Gemv2TSm86,
    ]);
    set
}
pub(crate) fn validate_inputs_whole(
    cut: &FixedPromptPrefixCut,
    inputs: &super::super::fixed_prompt_mlp::FixedPromptMlpQualificationInputs,
) -> Result<()> {
    if !cut.is_whole() || inputs.projections.images.capacity() > 11 {
        return Err(Error::State(
            "fixed_prompt whole exact finite image scope".into(),
        ));
    }
    let required = required_images();
    let source = cut.scope().source();
    super::super::projection_cubin::validate_fixed_prompt_full_image_set(
        source,
        &required,
        &inputs.projections.images,
    )?;
    for image in &inputs.projections.images {
        let cap = match image.kind {
            ProjectionImage::AttentionQkSm86 | ProjectionImage::AttentionKvSm86 => 2 * 1024 * 1024,
            ProjectionImage::AttentionFaCausalCombineSm80
            | ProjectionImage::AttentionSigmoidSm86
            | ProjectionImage::AttentionMultiplySm86 => 4 * 1024 * 1024,
            _ => 16 * 1024 * 1024,
        };
        if image.bytes.capacity() > cap || image.admission.bytes() > cap {
            return Err(Error::Limit(
                "fixed_prompt whole bounded image capacity/admission".into(),
            ));
        }
    }
    let act = inputs
        .activation
        .as_ref()
        .ok_or_else(|| Error::State("fixed_prompt whole activation image absent".into()))?;
    if act.bytes.capacity() > 2 * 1024 * 1024 || act.admission.bytes() > 2 * 1024 * 1024 {
        return Err(Error::Limit(
            "fixed_prompt whole activation source cap".into(),
        ));
    }
    validate_activation_image(source, act)
}
pub(crate) fn prepare_images_whole(
    cut: &Arc<FixedPromptPrefixCut>,
    service: &Arc<crate::resident_leases::ResidentService>,
    inputs: super::super::fixed_prompt_mlp::FixedPromptMlpQualificationInputs,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<PreparedImages>> {
    not_cancelled(cancelled)?;
    validate_inputs_whole(cut, &inputs)?;
    init.require_service(service)?;
    if zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&service.physical_key())?
        .limit
        != inputs.projections.device_cap
    {
        return Err(Error::State(
            "fixed_prompt whole original device cap".into(),
        ));
    }
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.projections.cublas, &inputs.projections.cublas_lt)
        .map_err(native_error)?;
    let owner: crate::physical_quarantine::CompletionOwner = cut.clone();
    let admitted = AdmittedImages::reserve_with_activation(
        service,
        owner,
        inputs.projections.images,
        inputs.activation,
    )?;
    init.retain_completion_owner(admitted.completion_owner())?;
    admitted.prepare_fixed_prompt_full_images(sources, init, cancelled)
}

pub(crate) struct FixedPromptHeadPrograms {
    cut: Arc<FixedPromptPrefixCut>,
    values: Arc<ModelValues>,
    tied: Arc<TiedWeight>,
    final_normalizer: Arc<FusedRmsPrepared>,
    input: ResidentLease,
    output: ResidentLease,
    kernel: Arc<OwnedCudaKernel>,
    images: Arc<PreparedImages>,
    enqueues: AtomicUsize,
    _metadata: Arc<WindowWorkspace>,
}
impl FixedPromptHeadPrograms {
    pub(crate) fn prepare(
        cut: Arc<FixedPromptPrefixCut>,
        values: Arc<ModelValues>,
        tied: Arc<TiedWeight>,
        final_normalizer: Arc<FusedRmsPrepared>,
        images: Arc<PreparedImages>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        cut.scope().require_values(&values)?;
        if !cut.is_whole()
            || cut.nodes().len() != 255
            || cut.values().len() != 327
            || !Arc::ptr_eq(values.record(), cut.scope().record())
            || !tied.source().same_issuer(cut.scope().source())
            || tied.copy_counts()? != (TIED_WINDOWS, TIED_WINDOWS)
            || !tied.lease().is_initialized()
        {
            return Err(Error::State(
                "fixed_prompt actual whole source and initialized tied owner".into(),
            ));
        }
        let record = values.record();
        let head = &record.plan().nodes()[254];
        let norm = &record.plan().nodes()[253];
        let Some(hd) = record.operator(254)? else {
            return Err(Error::State(
                "fixed_prompt actual head descriptor absent".into(),
            ));
        };
        let Some(nd) = record.operator(253)? else {
            return Err(Error::State("fixed_prompt actual final RMS absent".into()));
        };
        hd.validate()?;
        nd.validate()?;
        final_normalizer.require_full_model_operator(nd)?;
        if !matches!(
            hd.kind(),
            SourceOperatorKind::Projection {
                selected_row: Some(46)
            }
        ) || hd.call() != record.call()
            || head.inputs != [ValueId(324)]
            || head.outputs != [ValueId(326)]
            || !matches!(
                &head.operation,
                PlanOperation::SourceOperator { state: None, .. }
            )
            || head.state.is_some()
            || !matches!(nd.kind(), SourceOperatorKind::ResidualRms(_))
            || nd.call() != record.call()
            || norm.outputs != [ValueId(324), ValueId(325)]
            || !matches!(
                &norm.operation,
                PlanOperation::SourceOperator { state: None, .. }
            )
            || norm.state.is_some()
            || record.selection().normalized().ok_or_else(|| {
                Error::State("fixed prompt actual final normalized selection absent".into())
            })? != ValueId(324)
            || record.selection().output() != ValueId(326)
        {
            return Err(Error::State(
                "fixed_prompt genuine Final253/head254 producer SSA".into(),
            ));
        }
        let input = values
            .lease(record.selection().normalized().ok_or_else(|| {
                Error::State("fixed prompt actual final normalized selection absent".into())
            })?)?
            .clone();
        let output = values.lease(record.selection().output())?.clone();
        if input.id() != final_normalizer.normalized().id()
            || input.allocation_generation()
                != final_normalizer.normalized().allocation_generation()
            || input.bytes() != 96_256
            || output.bytes() != 496_640
            || tied.lease().bytes() != TIED_BYTES
            || input.id() == output.id()
            || input.id() == tied.lease().id()
            || output.id() == tied.lease().id()
            || !Arc::ptr_eq(input.service(), tied.lease().service())
            || !Arc::ptr_eq(output.service(), tied.lease().service())
        {
            return Err(Error::State(
                "fixed_prompt complete last47 head lease geometry/nonalias".into(),
            ));
        }
        init.require_service(values.service())?;
        images.require_service(values.service())?;
        let metadata = Arc::new(cut.scope().source().reserve_workspace(64 * 1024)?);
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(cut.clone())?;
        init.retain_completion_owner(values.clone())?;
        init.retain_completion_owner(tied.clone())?;
        init.retain_completion_owner(final_normalizer.clone())?;
        init.retain_completion_owner(images.clone())?;
        let image = images.image(ProjectionImage::Bf16Gemv2TSm86)?;
        let pin = PinnedBf16Cubin::Bf16Gemv2TSm86.derivation();
        if image.derived_from() != Some(PinnedBf16Cubin::Bf16Gemv2TSm86)
            || image.captured_from().is_some()
            || image.byte_len() != pin.image_bytes
            || image.sha256() != pin.image_sha256
            || image.binary_sm() != 86
        {
            return Err(Error::State(
                "fixed_prompt unchanged exact N1 head image".into(),
            ));
        }
        // SAFETY: exact existing manifest; all real full-range owners retained before load.
        let kernel = unsafe { images.set().prepare(image, Bf16VocabularyHead::manifest()) }
            .map_err(native_error)?;
        init.retain_completion_owner(kernel.clone())?;
        let result = Arc::new(Self {
            cut,
            values,
            tied,
            final_normalizer,
            input,
            output,
            kernel,
            images,
            enqueues: AtomicUsize::new(0),
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        vec![
            (self.tied.lease().clone(), DeviceAccess::Read),
            (self.input.clone(), DeviceAccess::Read),
            (self.output.clone(), DeviceAccess::Write),
        ]
    }
    pub(crate) fn enqueue(&self, index: usize, cancelled: Option<&AtomicBool>) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.cut.scope().require_values(&self.values)?;
        self.images.require_service(self.values.service())?;
        if index != 254
            || self.enqueues() != 0
            || !self.output_unpromoted()?
            || self.tied.copy_counts()? != (TIED_WINDOWS, TIED_WINDOWS)
            || !self.tied.lease().is_initialized()
        {
            return Err(Error::State(
                "fixed_prompt head actual final cursor/initialized source".into(),
            ));
        }
        // SAFETY: same original armed whole flight; actual Final253 producer precedes Head254.
        unsafe {
            self.kernel.enqueue_vocabulary_head_selected(
                self.tied.lease().handle(),
                self.input.handle(),
                self.output.handle(),
                VocabularyHeadSelection::FixedPromptLast,
            )
        }
        .map_err(native_error)?;
        self.enqueues.fetch_add(1, Ordering::AcqRel);
        if !self.output_unpromoted()? {
            return Err(Error::State(
                "fixed_prompt head output promoted before completion".into(),
            ));
        }
        Ok(1)
    }
    pub(crate) fn retained_values(
        &self,
        cut: &Arc<FixedPromptPrefixCut>,
        values: &Arc<ModelValues>,
    ) -> Result<bool> {
        // Metadata only: usable/Ready checks are forbidden after original quarantine.
        let input = values.lease(ValueId(324))?;
        let output = values.lease(ValueId(326))?;
        Ok(Arc::ptr_eq(cut, &self.cut)
            && Arc::ptr_eq(values, &self.values)
            && Arc::ptr_eq(values.record(), cut.scope().record())
            && self.tied.source().same_issuer(cut.scope().source())
            && self.tied.copy_counts()? == (TIED_WINDOWS, TIED_WINDOWS)
            && self.tied.lease().is_initialized()
            && self.tied.lease().role() == BufferRole::ImmutableWeights
            && self.tied.lease().bytes() == TIED_BYTES
            && self.input.id() == input.id()
            && self.input.allocation_generation() == input.allocation_generation()
            && self.output.id() == output.id()
            && self.output.allocation_generation() == output.allocation_generation()
            && self.input.id() == self.final_normalizer.normalized().id()
            && self.input.allocation_generation()
                == self.final_normalizer.normalized().allocation_generation()
            && self.input.bytes() == 96_256
            && self.output.bytes() == 496_640
            && [self.tied.lease(), &self.input, &self.output]
                .iter()
                .all(|lease| {
                    Arc::ptr_eq(lease.service(), values.service())
                        && lease.service_generation() == values.service().generation()
                        && lease.source_identity() == values.record().call().source()
                }))
    }
    pub(crate) fn enqueues(&self) -> usize {
        self.enqueues.load(Ordering::Acquire)
    }
    pub(crate) fn output_unpromoted(&self) -> Result<bool> {
        Ok(!self.output.is_initialized())
    }
    pub(crate) fn tied(&self) -> &Arc<TiedWeight> {
        &self.tied
    }
    pub(crate) fn kernel(&self) -> &Arc<OwnedCudaKernel> {
        &self.kernel
    }
    pub(crate) fn require_images(&self, images: &Arc<PreparedImages>) -> Result<()> {
        if !Arc::ptr_eq(&self.images, images) {
            return Err(Error::State(
                "fixed_prompt head must share the exact original full set".into(),
            ));
        }
        Ok(())
    }
    pub(crate) unsafe fn retire(&self) -> Result<()> {
        // SAFETY: caller owns original read-only whole-source retirement flight.
        unsafe { self.kernel.retire_after_physical_completion() }.map_err(native_error)
    }
}
