//! One exact known full24 image set; no mutable limits or runtime issuer.
use super::cubin_manifests::{Projection, ProjectionImage, ProjectionImageSource};
use super::projection_cubin::{
    CubinActivationImage, CubinQualificationInputs, native_error, not_cancelled,
    resident::{AdmittedImages, PreparedImages},
    validate_activation_image,
};
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::{ResidentInitialization, ResidentService};
use std::collections::BTreeSet;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::resource_arbiter::ResourceArbiter;
use zerogravity_core::{Error, Result};

pub(super) fn required_images() -> BTreeSet<ProjectionImage> {
    // Old scopes stay exact; this finite union is a distinct actor policy.
    let mut images = super::attention_cubin::required_gate_images();
    for projection in [
        Projection::Qkvz1,
        Projection::Ba1,
        Projection::Out1,
        Projection::Qkvz64,
        Projection::Ba64,
        Projection::Out64,
        Projection::Qkvz2,
        Projection::Ba2,
        Projection::Out2,
    ] {
        images.extend(
            projection
                .launches()
                .into_iter()
                .map(|launch| launch.image()),
        );
    }
    images
}

fn require_metadata(
    source: &VerifiedTensorSource,
    inputs: &CubinQualificationInputs,
    activation: &CubinActivationImage,
) -> Result<()> {
    let required = required_images();
    if required.len() != 13 || inputs.images.len() != 13 || inputs.images.capacity() > 16 {
        return Err(Error::Malformed(
            "closed full24 thirteen source images".into(),
        ));
    }
    validate_activation_image(source, activation)?;
    let mut seen = BTreeSet::new();
    let mut capacities = activation.bytes.capacity();
    for image in &inputs.images {
        let kind = image.kind;
        let (bytes, sha, sm) = match kind.provenance() {
            ProjectionImageSource::Derived(pin) => {
                let record = pin.derivation();
                (record.image_bytes, record.image_sha256, record.binary_sm)
            }
            ProjectionImageSource::Captured(pin) => {
                let record = pin.capture();
                (record.image_bytes, record.image_sha256, record.binary_sm)
            }
        };
        if !seen.insert(kind)
            || !required.contains(&kind)
            || (bytes, sha, sm) != (kind.bytes(), kind.sha256(), kind.binary_sm())
            || image.bytes.len() != bytes
            || image.bytes.capacity() > 32 * 1024 * 1024
            || !image.admission.source().same_issuer(source)
            || image.admission.bytes() < image.bytes.capacity()
        {
            return Err(Error::Malformed(
                "full24 exact original image category/metadata/admission".into(),
            ));
        }
        capacities = capacities
            .checked_add(image.bytes.capacity())
            .ok_or_else(|| Error::Limit("full24 original image capacity overflow".into()))?;
    }
    if seen != required || capacities > 128 * 1024 * 1024 {
        return Err(Error::Limit("full24 finite original image capacity".into()));
    }
    Ok(())
}

/// All source/control/value/state/W leases must already be retained by init.
pub(crate) fn prepare_full_model_images(
    source: VerifiedTensorSource,
    service: &Arc<ResidentService>,
    inputs: CubinQualificationInputs,
    activation: CubinActivationImage,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<PreparedImages>> {
    not_cancelled(cancelled)?;
    require_metadata(&source, &inputs, &activation)?;
    init.require_service(service)?;
    service.ensure_usable()?;
    if service.source_identity() != source.source_identity()
        || ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.device_cap
    {
        return Err(Error::State(
            "full24 actual original service/source/device admission".into(),
        ));
    }
    // Retain the immutable model, never the body that will own these images.
    // Deriving this owner here prevents a NativeBody→images→NativeBody cycle.
    let source_owner: CompletionOwner = Arc::new(source);
    init.retain_completion_owner(source_owner.clone())?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    not_cancelled(cancelled)?;
    let admitted = AdmittedImages::reserve_with_activation(
        service,
        source_owner,
        inputs.images,
        Some(activation),
    )?;
    init.retain_completion_owner(admitted.completion_owner())?;
    admitted.prepare_full_model_images(sources, init, cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_full24_inventory_is_distinct_from_existing_leaf_policies() {
        let all = required_images();
        assert_eq!(all.len(), 13);
        let attention = super::super::attention_cubin::required_gate_images();
        assert_eq!(attention.len(), 11);
        assert!(attention.is_subset(&all));
        assert_eq!(
            all.difference(&attention).copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([
                ProjectionImage::Bf16GemvxSm86,
                ProjectionImage::AmpereBf16Gemm128x64Sm80,
            ])
        );
        assert_eq!(
            all.iter().map(|image| image.bytes()).sum::<usize>()
                + cubecl::cuda::PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86
                    .capture()
                    .image_bytes,
            51_873_680
        );
    }
}
