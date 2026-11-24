//! Authenticated forward original-HF source seal and static plan factory.
//! No native backend, session root, invocation IDs or completion authority.
use super::*;
use crate::plan::{QwenForwardGeometry, QwenForwardPlan, QwenModelVariant, build_forward_plan};
use zerogravity_core::numerics::NumericalContract;

struct ForwardSourceOwner {
    model: Arc<OriginalWindowModelSource>,
    profile: QwenForwardGeometry,
    numerical: NumericalContract,
    _metadata: WindowWorkspace,
}

/// Exact original source/config/role authentication retained for the lifetime
/// of forward plans. This seal grants source semantics only; no device,
/// allocation, root, queue, completion or publication permission.
#[derive(Clone)]
pub struct ForwardSourceSeal(Arc<ForwardSourceOwner>);
impl std::fmt::Debug for ForwardSourceSeal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ForwardSourceSeal")
            .field("source", &self.source_identity())
            .field("profile", &self.model_profile())
            .finish_non_exhaustive()
    }
}
impl ForwardSourceSeal {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.0.model.source_identity()
    }
    pub fn model_profile(&self) -> QwenForwardGeometry {
        self.0.profile
    }
    pub fn numerical_contract(&self) -> NumericalContract {
        self.0.numerical
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        self.0.model.source()
    }
    pub fn build_plan(&self, context_capacity_tokens: usize) -> Result<QwenForwardPlan> {
        let model = &self.0.model;
        build_forward_plan(
            self,
            context_capacity_tokens,
            |role| {
                let descriptor = model.matrix_descriptor(role)?;
                model.validate_matrix(role, &descriptor)?;
                Ok(descriptor)
            },
            |role| {
                let descriptor = model.parameter_descriptor(role)?;
                model.validate_parameter(role, &descriptor)?;
                Ok(descriptor)
            },
        )
    }
    #[cfg(all(test, feature = "cpu"))]
    pub(crate) fn for_test(
        model: Arc<OriginalWindowModelSource>,
        profile: QwenModelVariant,
    ) -> Result<Self> {
        let metadata = model.source().reserve_workspace(64 * 1024)?;
        Ok(Self(Arc::new(ForwardSourceOwner {
            model,
            profile: profile.into(),
            numerical: NumericalContract::CudaSm86Bf16ReferenceV1,
            _metadata: metadata,
        })))
    }
}

impl OriginalWindowModelSource {
    /// Authenticate once without creating request data, a prompt cache,
    /// native image or execution owner.
    pub fn bind_cuda_forward_source(self: &Arc<Self>) -> Result<ForwardSourceSeal> {
        self.bind_forward_source(NumericalContract::CudaSm86Bf16ReferenceV1)
    }
    /// Original mixed CPU-source arithmetic; no native/model execution permit.
    pub fn bind_cpu_forward_source(self: &Arc<Self>) -> Result<ForwardSourceSeal> {
        self.bind_forward_source(NumericalContract::TorchCpuAvx2Bf16MixedV1)
    }
    /// Same authenticated tensors with the explicit native HIP projection profile.
    pub fn bind_hip_native_forward_source(self: &Arc<Self>) -> Result<ForwardSourceSeal> {
        self.bind_forward_source(NumericalContract::HipRocBlasBf16MixedV1)
    }
    pub fn bind_hip_native_solve_forward_source(self: &Arc<Self>) -> Result<ForwardSourceSeal> {
        self.bind_forward_source(NumericalContract::HipRocBlasBf16MixedV2)
    }
    fn bind_forward_source(
        self: &Arc<Self>,
        numerical: NumericalContract,
    ) -> Result<ForwardSourceSeal> {
        let profile = if self.owner().config_sha256() == CONFIG_LARGE {
            require_large_pins(self)?;
            let geometry = QwenForwardGeometry::from_config(self.owner().config())?;
            // Original large-source A_log and GDN norm are stored as BF16.
            super::source_geometry::require_roles(
                self,
                geometry,
                zerogravity_core::encoding::ScalarDType::Bf16,
            )?;
            return self.seal_forward_source(geometry, numerical);
        } else if self.owner().config_sha256() == super::full_model_source::CONFIG2B {
            QwenModelVariant::Qwen2B
        } else {
            QwenModelVariant::Qwen800M
        };
        super::full_model_source::require_source(self, profile)?;
        self.seal_forward_source(profile.into(), numerical)
    }
    fn seal_forward_source(
        self: &Arc<Self>,
        profile: QwenForwardGeometry,
        numerical: NumericalContract,
    ) -> Result<ForwardSourceSeal> {
        let metadata = self.source().reserve_workspace(64 * 1024)?;
        Ok(ForwardSourceSeal(Arc::new(ForwardSourceOwner {
            model: self.clone(),
            profile,
            numerical,
            _metadata: metadata,
        })))
    }
}

const CONFIG_LARGE: &str = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab";
fn require_large_pins(model: &OriginalWindowModelSource) -> Result<()> {
    if model.owner().config_sha256() != CONFIG_LARGE
        || model.owner().verified_index_sha256()
            != "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
        || model.source().source_bytes() != 55_563_006_776
        || model.source().staging_chunk_bytes() != 1024 * 1024
        || model.owner().text_tensor_count() != 851
        || model.owner().other_tensor_count() != 348
    {
        return Err(Error::Unsupported(
            "closed original large-model source/config/index/extent/1MiB".into(),
        ));
    }
    Ok(())
}
