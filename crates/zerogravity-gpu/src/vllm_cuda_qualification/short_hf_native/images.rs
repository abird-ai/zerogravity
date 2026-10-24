//! Exact compiled profile image cohort on one original actor/set.
use super::{
    super::{
        canonical47_mlp::Mlp47QualificationInputs,
        projection_cubin::{
            native_error, not_cancelled,
            resident::{AdmittedImages, PreparedImages},
            validate_activation_image, validate_short_hf_image_set,
        },
    },
    ShortHfNativeGeometry,
};
use crate::{physical_quarantine::CompletionOwner, resident_leases::ResidentInitialization};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::BoundFullModelSource;

pub(super) fn prepare(
    source: &BoundFullModelSource,
    geometries: &[ShortHfNativeGeometry],
    inputs: Mlp47QualificationInputs,
    cohort_b4: bool,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<PreparedImages>> {
    not_cancelled(cancelled)?;
    let first = geometries
        .first()
        .ok_or_else(|| Error::State("short-HF source geometry absent".into()))?;
    let service = first.values.service();
    init.require_service(service)?;
    let required = super::short_hf_required_images_for_cohort(source.model_profile(), cohort_b4)?;
    if inputs.projections.images.capacity() > required.len()
        || inputs
            .projections
            .images
            .iter()
            .any(|i| i.bytes.capacity() > 32 * 1024 * 1024)
    {
        return Err(Error::Limit(
            "short-HF exact immutable image capacity cohort".into(),
        ));
    }
    validate_short_hf_image_set(
        source.source().source(),
        source.model_profile(),
        cohort_b4,
        &inputs.projections.images,
    )?;
    let activation = inputs
        .activation
        .as_ref()
        .ok_or_else(|| Error::State("short-HF actual ACT image absent".into()))?;
    if activation.bytes.capacity() > 2 * 1024 * 1024 {
        return Err(Error::Limit(
            "short-HF exact activation image capacity".into(),
        ));
    }
    validate_activation_image(source.source().source(), activation)?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.projections.cublas, &inputs.projections.cublas_lt)
        .map_err(native_error)?;
    let owner: CompletionOwner = Arc::new((
        source.clone(),
        geometries
            .iter()
            .map(|g| g.scope.clone())
            .collect::<Vec<_>>(),
    ));
    let admitted = AdmittedImages::reserve_short_hf(
        service,
        owner,
        inputs.projections.images,
        inputs.activation,
        source.model_profile(),
        cohort_b4,
    )?;
    init.retain_completion_owner(admitted.completion_owner())?;
    admitted.prepare_short_hf_images(source.model_profile(), cohort_b4, sources, init, cancelled)
}
