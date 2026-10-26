//! Family eligibility/metadata factories for the existing physical qualification paths.
use super::*;
use crate::gpu::cuda::{native, rms::RmsLeafSource};
use crate::qwen35::{plan::QwenSourceProfile, source::binding::BoundFullModelSource};
use zerogravity_artifacts::source_weights::WindowWorkspace;

pub(super) use crate::qwen35_cuda::bindings::retained;

/// Exact family source/profile before physical projection/solve preparation.
pub fn prepare_leaf_source(
    model: &Arc<OriginalWindowModelSource>,
    profile: QwenSourceProfile,
) -> Result<Arc<native::RetainedSource>> {
    Ok(retained(model.prepare_qualification_source(profile)?))
}

/// Original fixed-prompt or finite text position eligibility, before CUDA lookup.
pub fn prepare_rotary_source(
    model: &Arc<OriginalWindowModelSource>,
    text: bool,
) -> Result<Arc<native::RetainedSource>> {
    Ok(retained(model.prepare_rotary_source(text)?))
}

pub fn prepare_rms_leaf_source(
    model: &Arc<OriginalWindowModelSource>,
    profile: QwenSourceProfile,
) -> Result<RmsLeafSource> {
    let prepared = Arc::new(model.prepare_rms_source(profile)?);
    rms_leaf_source(model, prepared)
}
pub(super) fn rms_leaf_source(
    model: &Arc<OriginalWindowModelSource>,
    prepared: Arc<crate::qwen35::source::binding::PreparedRmsSource>,
) -> Result<RmsLeafSource> {
    let source = RmsLeafSource::bind(
        prepared.source().clone(),
        prepared.embedding().clone(),
        prepared.input_gamma().clone(),
        prepared.final_gamma().clone(),
        prepared.geometry(),
        prepared.numerical_contract(),
        prepared.metadata(),
    )?;
    source.retain_source(native::RetainedSource::new(
        model.source().clone(),
        model.clone(),
        prepared,
    ))
}

pub fn prepare_tied_weight_source(
    model: &Arc<OriginalWindowModelSource>,
) -> Result<Arc<native::TiedWeightSource>> {
    let output = model.tied_output_descriptor()?;
    let retained =
        native::RetainedSource::new(model.source().clone(), model.clone(), model.clone());
    native::TiedWeightSource::bind(retained, output)
}

pub(super) fn prepare_final_rms_source(
    model: &Arc<OriginalWindowModelSource>,
) -> Result<crate::gpu::cuda::rms::BoundFinalRmsSource> {
    let prepared = model.prepare_final_rms_source()?;
    crate::gpu::cuda::rms::BoundFinalRmsSource::bind(
        prepared.source().clone(),
        prepared.parameter().clone(),
        prepared.geometry(),
        prepared.numerical_contract(),
        prepared.metadata(),
    )
}

/// The legacy .8B uploader's actual 2B refusal is family policy; W transfer,
/// complete source readback and uncertain retirement still use the GPU owner.
pub async fn qualify_qwen2b_mlp_weights(
    device: Device,
    source: BoundFullModelSource,
    recipe: Arc<crate::gpu::ComponentRecipe>,
    record_admission: Arc<WindowWorkspace>,
    mode: native::Qwen2bMlpWeightMode,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<native::Qwen2bMlpWeightOutcome> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "original full-model learned upload cancelled".into(),
        ));
    }
    if mode == native::Qwen2bMlpWeightMode::LegacyUploadRefusal {
        let rows = recipe.mlp_weight_rows(source.source().source(), &record_admission)?;
        let result = source.source().require_small_model_source();
        if !matches!(result,Err(Error::Unsupported(ref why)) if why=="closed original .8B24 source/config/rotary geometry")
        {
            return Err(Error::State(
                "legacy .8B upload did not reject exact actual2B source geometry".into(),
            ));
        }
        return Ok(native::Qwen2bMlpWeightOutcome {
            mode,
            status: "LEGACY_08_UPLOAD_REJECTED_ACTUAL_2B_SOURCE",
            rows,
            copies_enqueued: 0,
            copy_checkpoints: 0,
            initialized_weights: false,
            serial_reads_completed: 0,
            source_bytes_compared: 0,
            leading_readback_bytes: 0,
            trusted_validator_entered: false,
            read_only_flight_retired: false,
            diagnostic_ready_outputs: 0,
            native_launches: 0,
            prepared_buffers: 0,
            submitted_buffers: 0,
            resident_bytes: 0,
            uploaded_value_bytes: 0,
            pending_events: 0,
            quarantined_handles: 0,
            quarantined_owners: 0,
            source_record_weights_plan_retained: false,
            record_admission_retained: false,
            physical_reserved_bytes: 0,
            physical_fenced: false,
            root_published: false,
            native_mlp_qualified: false,
        });
    }
    // Move the existing bound metadata; do not create another source admission.
    let actual = native::RetainedSource::new(
        source.source().source().clone(),
        source.source().clone(),
        Arc::new(source),
    );
    native::qualify_qwen2b_mlp_weights(
        device,
        actual,
        recipe,
        record_admission,
        mode,
        device_cap,
        cancelled,
    )
    .await
}
