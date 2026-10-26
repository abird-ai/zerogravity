//! Native captured-operand attention leaf; never a model/profile permit.
mod descriptor;
mod faults;
pub(crate) mod fixed_prompt_fa_packet;
pub(crate) mod fixed_prompt_layer3_packets;
mod gate_manifests;
mod manifests;
mod qualify;
mod resident;
mod weights;
pub use faults::{
    CubinAttentionFault, CubinAttentionFaultOutcome, qualify_attention_cubin_fault,
    qualify_attention_gated_cubin_fault,
};
pub use manifests::AttentionCase as CubinAttentionCase;
pub use qualify::{
    CubinAttentionComparison, CubinAttentionExpected, CubinAttentionFixture,
    CubinAttentionGatedFixture, CubinAttentionOutput, qualify_attention_cubins,
    qualify_attention_gated_cubins,
};
#[cfg(test)]
mod tests;

// Complete immutable key for record dedup, not a caller hash or image issuer.
fn manifest_identity(m: &cubecl::cuda::CubinKernelManifest, image: [u8; 32]) -> Vec<u8> {
    let mut bytes = Vec::from(image);
    bytes.extend_from_slice(format!("{m:?}").as_bytes());
    bytes
}

pub(super) fn required_images()
-> std::collections::BTreeSet<super::cubin_manifests::ProjectionImage> {
    manifests::required_images()
}

pub(super) fn required_gate_images()
-> std::collections::BTreeSet<super::cubin_manifests::ProjectionImage> {
    use super::cubin_manifests::ProjectionImage;
    let mut images = required_images();
    images.insert(ProjectionImage::AttentionSigmoidSm86);
    images.insert(ProjectionImage::AttentionMultiplySm86);
    images
}

/// Reuses original image/source admissions and the SAME prepaid native charge.
/// The distinct scope cannot enlarge either existing set policy.
pub(super) fn prepare_images(
    source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    service: &std::sync::Arc<crate::resident_leases::ResidentService>,
    source_owner: crate::physical_quarantine::CompletionOwner,
    inputs: super::projection_cubin::CubinQualificationInputs,
    init: &mut crate::resident_leases::ResidentInitialization<'_>,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> zerogravity_core::Result<std::sync::Arc<super::projection_cubin::resident::PreparedImages>> {
    prepare_images_scoped(
        source,
        service,
        source_owner,
        inputs,
        init,
        cancelled,
        descriptor::AttentionScope::CapturedOutInput,
    )
}
pub(super) fn prepare_gate_images(
    source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    service: &std::sync::Arc<crate::resident_leases::ResidentService>,
    source_owner: crate::physical_quarantine::CompletionOwner,
    inputs: super::projection_cubin::CubinQualificationInputs,
    init: &mut crate::resident_leases::ResidentInitialization<'_>,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> zerogravity_core::Result<std::sync::Arc<super::projection_cubin::resident::PreparedImages>> {
    prepare_images_scoped(
        source,
        service,
        source_owner,
        inputs,
        init,
        cancelled,
        descriptor::AttentionScope::ComputedGate,
    )
}
fn prepare_images_scoped(
    source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    service: &std::sync::Arc<crate::resident_leases::ResidentService>,
    source_owner: crate::physical_quarantine::CompletionOwner,
    inputs: super::projection_cubin::CubinQualificationInputs,
    init: &mut crate::resident_leases::ResidentInitialization<'_>,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
    scope: descriptor::AttentionScope,
) -> zerogravity_core::Result<std::sync::Arc<super::projection_cubin::resident::PreparedImages>> {
    use super::{
        cubin_manifests::ProjectionImageSource,
        projection_cubin::{native_error, not_cancelled, resident::AdmittedImages},
    };
    use std::collections::BTreeSet;
    use zerogravity_core::{Error, resource_arbiter::ResourceArbiter};
    not_cancelled(cancelled)?;
    service.ensure_usable()?;
    init.require_service(service)?;
    if service.source_identity() != source.source_identity()
        || inputs.images.len() != scope.images()
        || inputs.images.capacity() > 16
        || ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.device_cap
    {
        return Err(Error::State(
            "attention original source/service/capacity".into(),
        ));
    }
    let expected = if scope == descriptor::AttentionScope::ComputedGate {
        required_gate_images()
    } else {
        required_images()
    };
    let mut seen = BTreeSet::new();
    let mut total = 0usize;
    for image in &inputs.images {
        let kind = image.kind;
        let (bytes, sha, sm) = match kind.provenance() {
            ProjectionImageSource::Derived(p) => {
                let r = p.derivation();
                (r.image_bytes, r.image_sha256, r.binary_sm)
            }
            ProjectionImageSource::Captured(p) => {
                let r = p.capture();
                (r.image_bytes, r.image_sha256, r.binary_sm)
            }
        };
        if !seen.insert(kind)
            || !expected.contains(&kind)
            || bytes != kind.bytes()
            || sha != kind.sha256()
            || sm != kind.binary_sm()
            || image.bytes.len() != bytes
            || image.bytes.capacity() > 32 * 1024 * 1024
            || !image.admission.source().same_issuer(source)
            || image.admission.bytes() < image.bytes.capacity()
        {
            return Err(Error::Malformed(
                "attention exact admitted source-image set".into(),
            ));
        }
        total = total
            .checked_add(image.bytes.capacity())
            .ok_or_else(|| Error::Limit("attention image capacities overflow".into()))?;
    }
    if seen != expected || total > 128 * 1024 * 1024 {
        return Err(Error::Limit(
            "attention full original image capacity".into(),
        ));
    }
    init.retain_completion_owner(source_owner.clone())?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    not_cancelled(cancelled)?;
    let admitted = AdmittedImages::reserve(service, source_owner, inputs.images)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let images = match scope {
        descriptor::AttentionScope::CapturedOutInput => {
            admitted.prepare_attention_images(sources, init, cancelled)?
        }
        descriptor::AttentionScope::ComputedGate => {
            admitted.prepare_attention_gate_images(sources, init, cancelled)?
        }
    };
    images.require_service(service)?;
    Ok(images)
}

mod model;
pub(crate) use model::{
    ModelAttentionBinding, ModelAttentionInputs, ModelAttentionParameters, full_model_private_bound,
};
pub(crate) use resident::ModelAttentionPrograms;

pub(crate) mod fixed_prompt_packets;

pub(crate) mod fixed_prompt_natural;
