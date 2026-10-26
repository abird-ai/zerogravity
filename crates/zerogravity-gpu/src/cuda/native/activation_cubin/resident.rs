//! Closed activation records share the original images, actor and admission.
use super::super::activation_manifest::{Activation, IMAGE_BYTES, IMAGE_SHA256};
use super::super::cubin_manifests::ProjectionBinding;
use super::super::projection_cubin::resident::{
    NativeRetentionState, NativeRetentionWitness, PreparedImages,
};
use super::super::projection_cubin::{native_error, not_cancelled};
use crate::resident_leases::{ResidentInitialization, ResidentLease};
use cubecl::cuda::{OwnedCudaKernel, PinnedCapturedBf16Cubin};
use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_core::{Error, Result};

pub(in crate::cuda::native) struct ActivationPrograms {
    kernels: BTreeMap<Activation, Arc<OwnedCudaKernel>>,
    images: Arc<PreparedImages>,
}

fn validate_activations(activations: &[Activation]) -> Result<()> {
    if activations.is_empty()
        || activations.len() > 3
        || activations
            .iter()
            .enumerate()
            .any(|(i, a)| activations[..i].contains(a))
    {
        return Err(Error::Malformed(
            "exact unique closed activation records1..3".into(),
        ));
    }
    Ok(())
}

impl ActivationPrograms {
    pub(in crate::cuda::native) fn prepare(
        images: Arc<PreparedImages>,
        activations: &[Activation],
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        validate_activations(activations)?;
        // Attach before any native prepare, including every partial failure.
        init.retain_completion_owner(images.clone())?;
        let image = images.activation()?;
        if image.byte_len() != IMAGE_BYTES
            || image.sha256() != IMAGE_SHA256
            || image.binary_sm() != 86
            || image.derived_from().is_some()
            || image.captured_from() != Some(PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86)
        {
            return Err(Error::State(
                "closed selected activation resource identity".into(),
            ));
        }
        let mut kernels = BTreeMap::new();
        for &activation in activations {
            not_cancelled(cancelled)?;
            if activation.bindings() != [ProjectionBinding::X, ProjectionBinding::Y] {
                return Err(Error::State(
                    "closed activation original X/Y bindings".into(),
                ));
            }
            // SAFETY: only the fixed source-derived ABI reaches this original
            // shared set. Its real initialization retains all image/admission
            // owners before any native load, including a partial prepare error.
            let kernel = unsafe { images.set().prepare(image.clone(), activation.manifest()) }
                .map_err(native_error)?;
            init.retain_completion_owner(kernel.clone())?;
            kernels.insert(activation, kernel);
        }
        let prepared = Arc::new(Self { kernels, images });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }

    pub(in crate::cuda::native) fn retention_witness(&self) -> Result<ActivationRetentionWitness> {
        let kernels = self.kernels.values().cloned().collect::<Vec<_>>();
        Ok(ActivationRetentionWitness {
            images: Arc::downgrade(&self.images),
            native: self.images.retention_witness(&kernels)?,
        })
    }

    pub(in crate::cuda::native) fn enqueue(
        &self,
        activation: Activation,
        input: &ResidentLease,
        output: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        input.service().ensure_usable()?;
        if !Arc::ptr_eq(input.service(), output.service())
            || input.source_identity() != output.source_identity()
            || input.id() == output.id()
            || input.bytes() != activation.input_bytes()
            || output.bytes() != activation.output_bytes()
        {
            return Err(Error::State(
                "closed activation original source/service/geometry".into(),
            ));
        }
        let kernel = self
            .kernels
            .get(&activation)
            .ok_or_else(|| Error::State("unprepared closed activation".into()))?;
        let handles = vec![input.handle(), output.handle()];
        not_cancelled(cancelled)?;
        // SAFETY: caller's armed original flight retains this program, source
        // and both original leases with exclusive output access. The kernel's
        // canonical-handle gate verifies the same original queue before launch.
        unsafe { kernel.enqueue(handles) }.map_err(native_error)?;
        Ok(1)
    }

    /// Caller proved real completion and retains self in the original armed
    /// read-only flight. Any error requires its original quarantine; no retry.
    pub(in crate::cuda::native) unsafe fn retire_kernels_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        for kernel in self.kernels.values() {
            // SAFETY: original caller proof covers every record; no enqueue race.
            unsafe { kernel.retire_after_physical_completion() }?;
        }
        Ok(())
    }

    /// Standalone only: every shared-set record belongs to these activations.
    /// Caller retains the original read-only flight through unload/readback.
    pub(in crate::cuda::native) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        // SAFETY: same caller proof covers all activation records and the set.
        unsafe { self.retire_kernels_after_physical_completion() }?;
        unsafe { self.images.set().retire_after_physical_completion() }
    }
}

pub(in crate::cuda::native) struct ActivationRetentionWitness {
    images: std::sync::Weak<PreparedImages>,
    native: NativeRetentionWitness,
}
impl ActivationRetentionWitness {
    pub(in crate::cuda::native) fn inspect(&self) -> Result<NativeRetentionState> {
        if self.images.upgrade().is_none() {
            return Err(Error::State(
                "activation shared images lost from original quarantine".into(),
            ));
        }
        self.native.inspect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_closed_records_prevent_lost_native_retirement_owners() {
        assert!(validate_activations(&[]).is_err());
        assert!(validate_activations(&[Activation::Rows1, Activation::Rows1]).is_err());
        assert!(
            validate_activations(&[
                Activation::Rows1,
                Activation::Rows2,
                Activation::Rows64,
                Activation::Rows1
            ])
            .is_err()
        );
        assert!(
            validate_activations(&[Activation::Rows1, Activation::Rows2, Activation::Rows64])
                .is_ok()
        );
    }
}
