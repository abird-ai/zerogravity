//! Authenticated ordinary original-HF source seal and static plan factory.
//! No native backend, session root, invocation IDs or completion authority.
use super::*;
use crate::plan::{OrdinaryQwenPlan, QwenSourceProfile, build_ordinary_plan};
use zerogravity_core::numerics::SourceNumericalContract;

struct OrdinarySourceOwner {
    model: Arc<OriginalWindowModelSource>,
    profile: QwenSourceProfile,
    numerical: SourceNumericalContract,
    _metadata: WindowWorkspace,
}

/// Exact original source/config/role authentication retained for the lifetime
/// of ordinary plans. This seal grants source semantics only; no device,
/// allocation, root, queue, completion or publication permission.
#[derive(Clone)]
pub struct OrdinarySourceSeal(Arc<OrdinarySourceOwner>);
/// Historical CUDA spelling; same original source owner.
pub type OrdinaryCudaSourceSeal = OrdinarySourceSeal;
impl std::fmt::Debug for OrdinarySourceSeal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OrdinarySourceSeal")
            .field("source", &self.source_identity())
            .field("profile", &self.model_profile())
            .finish_non_exhaustive()
    }
}
impl OrdinarySourceSeal {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.0.model.source_identity()
    }
    pub fn model_profile(&self) -> QwenSourceProfile {
        self.0.profile
    }
    pub fn numerical_contract(&self) -> SourceNumericalContract {
        self.0.numerical
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        self.0.model.source()
    }
    pub fn build_plan(&self, context_capacity_tokens: usize) -> Result<OrdinaryQwenPlan> {
        let model = &self.0.model;
        build_ordinary_plan(
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
        profile: QwenSourceProfile,
    ) -> Result<Self> {
        let metadata = model.source().reserve_workspace(64 * 1024)?;
        Ok(Self(Arc::new(OrdinarySourceOwner {
            model,
            profile,
            numerical: SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            _metadata: metadata,
        })))
    }
}

impl OriginalWindowModelSource {
    /// Authenticate once without creating request data, a prompt cache,
    /// native image or execution owner.
    pub fn bind_ordinary_cuda_source(self: &Arc<Self>) -> Result<OrdinaryCudaSourceSeal> {
        self.bind_ordinary_source(SourceNumericalContract::ReferenceCudaSm86Bf16V1)
    }
    /// Original mixed CPU-source arithmetic; no native/model execution permit.
    pub fn bind_ordinary_cpu_source(self: &Arc<Self>) -> Result<OrdinarySourceSeal> {
        self.bind_ordinary_source(SourceNumericalContract::TorchCpuBf16Avx2MixedSourceV1)
    }
    fn bind_ordinary_source(
        self: &Arc<Self>,
        numerical: SourceNumericalContract,
    ) -> Result<OrdinarySourceSeal> {
        let profile = if self.owner().config_sha256() == super::full_model_source::CONFIG2B {
            QwenSourceProfile::Qwen2B
        } else {
            QwenSourceProfile::Qwen08
        };
        super::full_model_source::require_source(self, profile)?;
        let metadata = self.source().reserve_workspace(64 * 1024)?;
        Ok(OrdinarySourceSeal(Arc::new(OrdinarySourceOwner {
            model: self.clone(),
            profile,
            numerical,
            _metadata: metadata,
        })))
    }
}
