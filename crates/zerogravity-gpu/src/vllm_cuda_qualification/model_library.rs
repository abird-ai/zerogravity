//! One immutable native record set, no completed body/state authority.
use super::attention_cubin::{ModelAttentionBinding, ModelAttentionPrograms};
use super::model_native::ModelNativePrograms;
use super::model_values::ModelValues;
use super::model_weights::ModelWeights;
use super::projection_cubin::native_error;
use super::projection_cubin::resident::PreparedImages;
use super::vocabulary_head::ModelHeadPrograms;
use super::*;
use crate::resident_leases::{ResidentInitialization, ResidentLease, TiedWeight};
use crate::resident_program::native_full::FullModelLibrary;
use crate::vllm_cuda_input_rms::FusedRmsPrepared;
use std::sync::atomic::AtomicBool;

pub(crate) struct FullModelFactoryInputs {
    pub(crate) images: Arc<PreparedImages>,
    pub(crate) weights: Arc<ModelWeights>,
    pub(crate) tied: Arc<TiedWeight>,
    pub(crate) values: Vec<Arc<ModelValues>>,
    pub(crate) final_normalizers: Vec<Arc<FusedRmsPrepared>>,
    pub(crate) attention: Vec<Arc<ModelAttentionBinding>>,
    pub(crate) workspace: ResidentLease,
}
impl FullModelLibrary {
    /// The initializer already retains every actual body/source/control/cache.
    /// All learned W and managed outputs must exist before this first load.
    pub(crate) fn prepare(
        input: FullModelFactoryInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        projection_cubin::not_cancelled(cancelled)?;
        input.weights.require_initialized()?;
        if input.values.len() != 5
            || input.final_normalizers.len() != 5
            || !Arc::ptr_eq(input.weights.source(), input.tied.source())
        {
            return Err(Error::State(
                "full24 one actual source/phase factory cohort".into(),
            ));
        }
        let source = input.weights.source().clone();
        let phases = input
            .values
            .iter()
            .cloned()
            .zip(input.final_normalizers)
            .collect();
        let projections = ModelNativePrograms::prepare(
            input.images.clone(),
            input.weights,
            input.values,
            input.workspace,
            init,
            cancelled,
        )?;
        let attention = ModelAttentionPrograms::prepare(
            input.images.clone(),
            input.attention,
            init,
            cancelled,
        )?;
        let head = ModelHeadPrograms::prepare(
            input.images.clone(),
            input.tied.source().clone(),
            input.tied,
            phases,
            init,
            cancelled,
        )?;
        let library = Arc::new(Self {
            source,
            projections,
            attention,
            head,
            images: input.images,
        });
        init.retain_completion_owner(library.clone())?;
        Ok(library)
    }
    /// All five physical producer completions have succeeded. Caller supplies
    /// ONE original initialized learned-W Read flight covering unload failures.
    pub(crate) unsafe fn retire_after_physical_completion(&self) -> Result<()> {
        unsafe { self.projections.retire_kernels_after_physical_completion() }
            .map_err(native_error)?;
        unsafe { self.attention.retire_kernels_after_physical_completion() }
            .map_err(native_error)?;
        unsafe { self.head.retire_kernels_after_physical_completion() }.map_err(native_error)?;
        unsafe { self.images.set().retire_after_physical_completion() }.map_err(native_error)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FullModelNativeRetention {
    pub(crate) images: usize,
    pub(crate) head_kernels: usize,
    pub(crate) image_admission_bytes: usize,
    pub(crate) native_charge_bytes: usize,
}
impl FullModelLibrary {
    pub(crate) fn inspect_native_retention(&self) -> Result<FullModelNativeRetention> {
        let actual = self.head.retention_witness()?.inspect()?;
        if actual.image_owners != 14
            || actual.kernel_owners != 1
            || actual.native_charge_bytes != 64 * 1024 * 1024
        {
            return Err(Error::State(
                "full24 original native image/set/admission retention differs".into(),
            ));
        }
        Ok(FullModelNativeRetention {
            images: actual.image_owners,
            head_kernels: actual.kernel_owners,
            image_admission_bytes: actual.image_admission_bytes,
            native_charge_bytes: actual.native_charge_bytes,
        })
    }
}
