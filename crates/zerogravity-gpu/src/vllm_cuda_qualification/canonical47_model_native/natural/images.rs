//! One exact source/image set for both geometries. No raw entry/image authority.
use super::*;
use crate::vllm_cuda_qualification::{
    canonical47_mlp::Mlp47QualificationInputs,
    projection_cubin::{validate_activation_image, validate_canonical47_natural_image_set},
};
pub(crate) fn required_images() -> BTreeSet<ProjectionImage> {
    let mut s = super::super::head::required_images();
    s.extend([
        ProjectionImage::CutlassWmmaSm80,
        ProjectionImage::Bf16GemvxSm86,
        ProjectionImage::AttentionFaSplitSm80,
    ]);
    s
}
pub(crate) fn prepare(
    scopes: &[Arc<super::super::super::canonical47_model::Canonical47ModelBinding>; 2],
    service: &Arc<crate::resident_leases::ResidentService>,
    inputs: Mlp47QualificationInputs,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<PreparedImages>> {
    not_cancelled(cancelled)?;
    init.require_service(service)?;
    let source = scopes[0].source().source().source();
    if inputs.projections.images.capacity() > 14 {
        return Err(Error::Limit(
            "canonical natural fourteen image Vec admission".into(),
        ));
    }
    validate_canonical47_natural_image_set(source, &required_images(), &inputs.projections.images)?;
    let activation = inputs
        .activation
        .as_ref()
        .ok_or_else(|| Error::State("canonical natural ACT source absent".into()))?;
    validate_activation_image(source, activation)?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.projections.cublas, &inputs.projections.cublas_lt)
        .map_err(native_error)?;
    let owner: crate::physical_quarantine::CompletionOwner = Arc::new(scopes.clone());
    let admitted = AdmittedImages::reserve_with_activation(
        service,
        owner,
        inputs.projections.images,
        inputs.activation,
    )?;
    init.retain_completion_owner(admitted.completion_owner())?;
    admitted.prepare_canonical47_natural_images(sources, init, cancelled)
}
