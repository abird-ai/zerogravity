//! Authenticated ordinary original-HF CUDA source seal and static plan factory.
//! No native backend, session root, invocation IDs or completion authority.
use super::*;
use crate::plan::{OrdinaryQwenPlan, QwenSourceProfile, build_ordinary_plan};

struct OrdinaryCudaOwner {
    model: Arc<OriginalWindowModelSource>,
    profile: QwenSourceProfile,
    _metadata: WindowWorkspace,
}

/// Exact original source/config/role authentication retained for the lifetime
/// of ordinary CUDA plans. This seal grants source semantics only; no device,
/// allocation, root, queue, completion or publication permission.
#[derive(Clone)]
pub struct OrdinaryCudaSourceSeal(Arc<OrdinaryCudaOwner>);
impl OrdinaryCudaSourceSeal {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.0.model.source_identity()
    }
    pub fn model_profile(&self) -> QwenSourceProfile {
        self.0.profile
    }
    pub fn build_plan(&self, context_capacity_tokens: usize) -> Result<OrdinaryQwenPlan> {
        let model = &self.0.model;
        build_ordinary_plan(
            model.source_identity(),
            self.0.profile,
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
}

impl OriginalWindowModelSource {
    /// Authenticate once without creating request data, a prompt cache,
    /// native image or execution owner.
    pub fn bind_ordinary_cuda_source(self: &Arc<Self>) -> Result<OrdinaryCudaSourceSeal> {
        let profile = if self.owner().config_sha256() == super::full_model_source::CONFIG2B {
            QwenSourceProfile::Qwen2B
        } else {
            QwenSourceProfile::Qwen08
        };
        super::full_model_source::require_source(self, profile)?;
        let metadata = self.source().reserve_workspace(64 * 1024)?;
        Ok(OrdinaryCudaSourceSeal(Arc::new(OrdinaryCudaOwner {
            model: self.clone(),
            profile,
            _metadata: metadata,
        })))
    }
}
