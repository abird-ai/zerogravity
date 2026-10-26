//! Family-owned qualification entrypoints; physical preparation stays in GPU.
use super::source::{retained, rms_leaf_source};
use super::*;
use crate::gpu::cuda::native;
use crate::qwen35::{plan::QwenSourceProfile, source::binding::BoundFullModelSource};

fn preempt(cancelled: Option<&AtomicBool>, reason: &'static str) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(reason.into()));
    }
    Ok(())
}
fn require_release_cuda(device: &Device, reason: &'static str) -> Result<()> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(reason.into()));
    }
    Ok(())
}
pub async fn qualify_source_rotary_cubins<F>(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    device_cap: u64,
    inputs: Option<native::CubinQualificationInputs>,
    mode: native::RotaryMode,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<native::RotaryOutcome>
where
    F: FnOnce(&[(native::RotaryOutput, &[u8])]) -> Result<()>,
{
    preempt(cancelled, "closed original cubin qualification cancelled")?;
    let source = super::prepare_rotary_source(&model, false)?;
    drop(model);
    native::qualify_source_rotary_cubins(
        device, source, device_cap, inputs, mode, cancelled, validate,
    )
    .await
}

pub async fn qualify_text_rotary_cubins<F>(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    device_cap: u64,
    inputs: Option<native::CubinQualificationInputs>,
    mode: native::RotaryMode,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<native::RotaryOutcome>
where
    F: FnOnce(&[(native::RotaryOutput, &[u8])]) -> Result<()>,
{
    preempt(cancelled, "closed original cubin qualification cancelled")?;
    let source = super::prepare_rotary_source(&model, true)?;
    drop(model);
    native::qualify_text_rotary_cubins(
        device, source, device_cap, inputs, mode, cancelled, validate,
    )
    .await
}

pub async fn qualify_fixed_prompt_projections<F>(
    device: Device,
    source: BoundFullModelSource,
    fixtures: Vec<native::FixedPromptProjectionFixture>,
    inputs: native::CubinQualificationInputs,
    validate: F,
    cancelled: Option<&AtomicBool>,
) -> Result<native::FixedPromptProjectionReceipt>
where
    F: FnMut(native::FixedPromptProjectionReadback<'_>) -> Result<()>,
{
    preempt(cancelled, "closed original cubin qualification cancelled")?;
    require_release_cuda(&device, "release original CUDA47 projection leaf")?;
    native::FixedPromptProjectionFixture::preflight_route(&device)?;
    let source = retained(source.prepare_qualification_source(QwenSourceProfile::Qwen08)?);
    native::qualify_fixed_prompt_projections(device, source, fixtures, inputs, validate, cancelled)
        .await
}

pub async fn qualify_fixed_prompt_projection_fault(
    device: Device,
    source: BoundFullModelSource,
    fixtures: Vec<native::FixedPromptProjectionFixture>,
    inputs: native::CubinQualificationInputs,
    fault: native::FixedPromptProjectionFault,
    cancelled: Option<&AtomicBool>,
) -> Result<native::FixedPromptProjectionFaultOutcome> {
    preempt(cancelled, "closed original cubin qualification cancelled")?;
    require_release_cuda(&device, "release original CUDA47 projection fault")?;
    if fault != native::FixedPromptProjectionFault::Pressure {
        native::FixedPromptProjectionFixture::preflight_route(&device)?;
    }
    let source = retained(source.prepare_qualification_source(QwenSourceProfile::Qwen08)?);
    native::qualify_fixed_prompt_projection_fault(
        device, source, fixtures, inputs, fault, cancelled,
    )
    .await
}

pub async fn qualify_qwen2b_projections<F>(
    device: Device,
    source: BoundFullModelSource,
    fixtures: Vec<native::Qwen2bProjectionFixture>,
    inputs: native::CubinQualificationInputs,
    validate: F,
    cancelled: Option<&AtomicBool>,
) -> Result<native::Qwen2bProjectionReceipt>
where
    F: FnMut(native::Qwen2bProjectionReadback<'_>) -> Result<()>,
{
    preempt(cancelled, "closed original cubin qualification cancelled")?;
    require_release_cuda(&device, "release original CUDA2B projection leaf")?;
    native::Qwen2bProjectionFixture::preflight_route(&device)?;
    let source = retained(source.prepare_qualification_source(QwenSourceProfile::Qwen2B)?);
    native::qualify_qwen2b_projections(device, source, fixtures, inputs, validate, cancelled).await
}

pub async fn qualify_qwen2b_projection_fault(
    device: Device,
    source: BoundFullModelSource,
    fixtures: Vec<native::Qwen2bProjectionFixture>,
    inputs: native::CubinQualificationInputs,
    fault: native::Qwen2bProjectionFault,
    cancelled: Option<&AtomicBool>,
) -> Result<native::Qwen2bProjectionFaultOutcome> {
    preempt(cancelled, "closed original cubin qualification cancelled")?;
    require_release_cuda(&device, "release original CUDA2B projection fault")?;
    if fault != native::Qwen2bProjectionFault::Pressure {
        native::Qwen2bProjectionFixture::preflight_route(&device)?;
    }
    let source = retained(source.prepare_qualification_source(QwenSourceProfile::Qwen2B)?);
    native::qualify_qwen2b_projection_fault(device, source, fixtures, inputs, fault, cancelled)
        .await
}

pub async fn qualify_fixed_prompt_solve_leaves(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixture: native::FixedPromptSolveFixture,
    device_cap: u64,
    validate: impl FnMut(native::FixedPromptSolveReadback<'_>) -> Result<()>,
    cancelled: Option<&AtomicBool>,
) -> Result<native::FixedPromptSolveReceipt> {
    preempt(cancelled, "original47 solve leaf cancelled")?;
    let source = super::prepare_leaf_source(&model, QwenSourceProfile::Qwen08)?;
    drop(model);
    native::qualify_fixed_prompt_solve_leaves(
        device, source, fixture, device_cap, validate, cancelled,
    )
    .await
}

pub async fn qualify_fixed_prompt_solve_fault(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixture: Option<native::FixedPromptSolveFixture>,
    fault: native::FixedPromptSolveFault,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<native::FixedPromptSolveFaultOutcome> {
    preempt(cancelled, "original47 solve leaf cancelled")?;
    require_release_cuda(&device, "release CUDA required for original47 solve fault")?;
    if fault == native::FixedPromptSolveFault::Pressure && fixture.is_some() {
        return Err(Error::State(
            "fixed-prompt solve pressure has no fixture/H2D prerequisite".into(),
        ));
    }
    let source = super::prepare_leaf_source(&model, QwenSourceProfile::Qwen08)?;
    drop(model);
    native::qualify_fixed_prompt_solve_fault(device, source, fixture, fault, device_cap, cancelled)
        .await
}

pub async fn qualify_solve2b_owned_leaves(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixture: native::Solve2bFixture,
    device_cap: u64,
    validate: impl FnMut(native::Solve2bReadback<'_>) -> Result<()>,
    cancelled: Option<&AtomicBool>,
) -> Result<native::Solve2bReceipt> {
    preempt(cancelled, "original2B/T64 solve leaf cancelled")?;
    let source = super::prepare_leaf_source(&model, QwenSourceProfile::Qwen2B)?;
    drop(model);
    native::qualify_solve2b_owned_leaves(device, source, fixture, device_cap, validate, cancelled)
        .await
}

pub async fn qualify_solve2b_owned_fault(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixture: Option<native::Solve2bFixture>,
    fault: native::Solve2bFault,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<native::Solve2bFaultOutcome> {
    preempt(cancelled, "original2B/T64 solve leaf cancelled")?;
    require_release_cuda(
        &device,
        "release CUDA required for original2B/T64 solve fault",
    )?;
    if fault == native::Solve2bFault::Pressure && fixture.is_some() {
        return Err(Error::State(
            "solve2b pressure has no fixture/H2D prerequisite".into(),
        ));
    }
    let source = super::prepare_leaf_source(&model, QwenSourceProfile::Qwen2B)?;
    drop(model);
    native::qualify_solve2b_owned_fault(device, source, fixture, fault, device_cap, cancelled).await
}

pub async fn qualify_fixed_prompt_rms(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    teachers: [u32; 47],
    leaf: native::FixedPromptDualRmsLeaf,
    device_cap: u64,
    validate: impl FnMut(native::FixedPromptRmsReadback<'_>) -> Result<()>,
    cancelled: Option<&AtomicBool>,
) -> Result<native::FixedPromptRmsReceipt> {
    preempt(cancelled, "fixed_prompt RMS cancelled")?;
    let prepared = Arc::new(model.prepare_rms_source(QwenSourceProfile::Qwen08)?);
    let ids_source = prepared.clone();
    let source = rms_leaf_source(&model, prepared)?;
    drop(model);
    native::qualify_fixed_prompt_rms(
        device,
        source,
        teachers,
        leaf,
        device_cap,
        validate,
        cancelled,
        move |ids| ids_source.require_ids(ids),
    )
    .await
}

pub async fn qualify_qwen2b_rms(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    teachers: [Vec<u32>; 5],
    leaves: [native::Qwen2bDualRmsLeaf; 5],
    device_cap: u64,
    validate: impl FnMut(native::Qwen2bRmsReadback<'_>) -> Result<()>,
    cancelled: Option<&AtomicBool>,
) -> Result<native::Qwen2bRmsReceipt> {
    preempt(cancelled, "original2B RMS cancelled")?;
    let prepared = Arc::new(model.prepare_rms_source(QwenSourceProfile::Qwen2B)?);
    let ids_source = prepared.clone();
    let source = rms_leaf_source(&model, prepared)?;
    drop(model);
    native::qualify_qwen2b_rms(
        device,
        source,
        teachers,
        leaves,
        device_cap,
        validate,
        cancelled,
        move |ids| ids_source.require_ids(ids),
    )
    .await
}

pub async fn qualify_fixed_prompt_rms_fault(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fault: native::FixedPromptRmsFault,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<native::FixedPromptRmsFaultOutcome> {
    preempt(cancelled, "fixed_prompt RMS cancelled")?;
    require_release_cuda(
        &device,
        "release original CUDA required forfixed_prompt RMS fault",
    )?;
    let source = super::prepare_rms_leaf_source(&model, QwenSourceProfile::Qwen08)?;
    drop(model);
    native::qualify_fixed_prompt_rms_fault(device, source, fault, device_cap, cancelled).await
}

pub async fn qualify_qwen2b_rms_fault(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fault: native::Qwen2bRmsFault,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<native::Qwen2bRmsFaultOutcome> {
    preempt(cancelled, "original2B RMS cancelled")?;
    require_release_cuda(&device, "release original CUDA required for2B RMS fault")?;
    let source = super::prepare_rms_leaf_source(&model, QwenSourceProfile::Qwen2B)?;
    drop(model);
    native::qualify_qwen2b_rms_fault(device, source, fault, device_cap, cancelled).await
}

pub async fn qualify_tied_weight_upload(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    device_cap: u64,
    mode: native::TiedWeightUploadMode,
    cancelled: Option<&AtomicBool>,
) -> Result<native::TiedWeightUploadOutcome> {
    preempt(cancelled, "original tied upload qualification cancelled")?;
    let source = native::RetainedSource::new(model.source().clone(), model.clone(), model.clone());
    native::qualify_tied_weight_upload(device, source, device_cap, mode, cancelled, move || {
        super::prepare_tied_weight_source(&model)
    })
    .await
}

pub async fn qualify_vocabulary_head_cubins(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixtures: native::VocabularyHeadFixtures,
    inputs: native::CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<native::VocabularyHeadComparison> {
    preempt(cancelled, "closed original cubin qualification cancelled")?;
    require_release_cuda(&device, "release original CUDA native head qualification")?;
    let source = native::RetainedSource::new(model.source().clone(), model.clone(), model.clone());
    native::qualify_vocabulary_head_cubins(device, source, fixtures, inputs, cancelled, move || {
        super::prepare_tied_weight_source(&model)
    })
    .await
}

pub async fn qualify_final_norm_head_cubins(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixtures: native::VocabularyHeadFixtures,
    inputs: native::CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<native::VocabularyHeadComparison> {
    preempt(cancelled, "closed original cubin qualification cancelled")?;
    require_release_cuda(&device, "release original CUDA native head qualification")?;
    let source = native::RetainedSource::new(model.source().clone(), model.clone(), model.clone());
    let final_model = model.clone();
    native::qualify_final_norm_head_cubins(
        device,
        source,
        fixtures,
        inputs,
        cancelled,
        move || super::source::prepare_final_rms_source(&final_model),
        move || super::prepare_tied_weight_source(&model),
    )
    .await
}

pub async fn qualify_vocabulary_head_fault(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixtures: native::VocabularyHeadFixtures,
    inputs: native::CubinQualificationInputs,
    scope: native::VocabularyHeadScope,
    fault: native::VocabularyHeadFault,
    cancelled: Option<&AtomicBool>,
) -> Result<native::VocabularyHeadFaultOutcome> {
    preempt(cancelled, "closed original cubin qualification cancelled")?;
    require_release_cuda(&device, "release original CUDA native head qualification")?;
    let source = native::RetainedSource::new(model.source().clone(), model.clone(), model.clone());
    let final_model = model.clone();
    native::qualify_vocabulary_head_fault(
        device,
        source,
        fixtures,
        inputs,
        scope,
        fault,
        cancelled,
        move || super::source::prepare_final_rms_source(&final_model),
        move || super::prepare_tied_weight_source(&model),
    )
    .await
}
