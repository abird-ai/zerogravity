//! Original source plan + resident owners on the existing engine worker.
use crate::{
    gpu::{ResidentTextModel, ResidentTextSession},
    identity::SourceIdentity,
    qwen35::plan::QwenForwardPlan,
    resource_arbiter::{AllocationClass, GlobalResourceLease},
};
use std::sync::{Arc, Mutex};
#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
use zerogravity_engine::state_cache::PrefixCache;
use zerogravity_engine::{
    model_instance::{PreparedModel, TextTokenizer},
    router::{ServiceError, pending::ServingHost},
};
mod execute;
mod load;
pub(crate) use load::{SourceBackend, load_router};

/// One cold raw-text source, shared by aliases. CUDA may opt into an aligned
/// four-row decode cohort; host/provider/context bounds remain backend-owned.
#[derive(Clone, Copy, Debug)]
pub struct QwenServingProfile {
    pub active: usize,
    pub context: usize,
    pub output: usize,
    pub max_pending: Option<usize>,
    /// Legacy/public cohort-width value. Under CUDA smart config, nonzero values
    /// are treated as user overrides; zero leaves the knob eligible for Auto.
    /// Use `QwenCudaSmartOverrides::cohort_width = Some(0)` to explicitly force
    /// sequential execution while keeping smart selection enabled for other knobs.
    pub cohort_width: usize,
}
impl QwenServingProfile {
    pub fn validate_base(self) -> Result<(), ServiceError> {
        use crate::qwen35::plan::{FORWARD_SOURCE_MAX_CONTEXT, FORWARD_SOURCE_MIN_CONTEXT};
        if !(1..=zerogravity_engine::router::MAX_ACTIVE_PER_WORKER).contains(&self.active)
            || !(FORWARD_SOURCE_MIN_CONTEXT..=FORWARD_SOURCE_MAX_CONTEXT).contains(&self.context)
            || self.output == 0
            || self.output > self.context
            || self.output > zerogravity_engine::router::MAX_OUTPUT_TOKENS
            || self.max_pending == Some(0)
        {
            return Err(ServiceError::new(
                400,
                "invalid_source_profile",
                "source GPU requires active=1..8, context=512..32768, output=1..512 and a positive pending bound",
            ));
        }
        // Keep every shared worker refusal before source IO or device creation.
        zerogravity_engine::router::validate_config(
            1,
            1,
            self.active,
            self.context,
            self.output,
            self.max_pending,
        )
    }

    pub(crate) fn validate_cohort_width(self, cohort_width: usize) -> Result<(), ServiceError> {
        if !matches!(cohort_width, 0 | 4 | 8) || (cohort_width != 0 && self.active < cohort_width) {
            return Err(ServiceError::new(
                400,
                "invalid_source_profile",
                "source GPU requires cohort_width=0|4|8 with active>=cohort_width",
            ));
        }
        Ok(())
    }
}
struct Source {
    plan: QwenForwardPlan,
    metadata: crate::qwen35_source::Metadata,
    _snapshot: Arc<crate::cache::SnapshotDescriptorLease>,
    // Source/canonical metadata and plan allocation, not device/root storage.
    _control: GlobalResourceLease,
}
#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
type TextPrefixCache =
    PrefixCache<u32, crate::gpu::ResidentTextSnapshot, crate::gpu::ResidentTextStateIdentity>;
#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
struct CachedPrefixes {
    cache: TextPrefixCache,
    _storage: GlobalResourceLease,
}
struct Model {
    resident: Mutex<Option<Arc<ResidentTextModel>>>,
    #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
    prefix_cache: Mutex<Option<CachedPrefixes>>,
    source: Arc<Source>,
    tokenizer: Arc<dyn TextTokenizer>,
    max_token_bytes: usize,
    _metadata: Arc<zerogravity_engine::router::pending::ServingCharge>,
}
impl Model {
    fn resident(&self) -> Result<Arc<ResidentTextModel>, ServiceError> {
        self.resident
            .lock()
            .map_err(|_| poison())?
            .as_ref()
            .cloned()
            .ok_or_else(|| ServiceError::new(503, "model_retired", "original source model retired"))
    }
}
fn poison() -> ServiceError {
    ServiceError::new(500, "source_owner", "original model owner lock poisoned")
}
impl PreparedModel for Model {
    fn source(&self) -> SourceIdentity {
        self.source.plan.source_seal().source_identity()
    }
    fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
        self.tokenizer.clone()
    }
    fn max_context(&self) -> usize {
        self.source.plan.context_capacity_tokens()
    }
    fn max_token_bytes(&self) -> usize {
        self.max_token_bytes
    }
    fn vocabulary_size(&self) -> u64 {
        self.source.plan.model_profile().vocabulary_size() as u64
    }
    fn describe(&self) -> crate::Result<zerogravity_engine::diagnostics::ModelDetails> {
        use zerogravity_core::provider::PreparedExecutable;
        use zerogravity_engine::diagnostics::{
            ArtifactDescription, ModelDetails, ModelMemory, SourceRevision,
        };
        let resident = self.resident().map_err(ServiceError::into_core)?;
        let prepared = resident.prepared_text().ok_or_else(|| {
            crate::Error::State("original resident text preparation absent".into())
        })?;
        let metadata = &self.source.metadata;
        #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
        let prefix_sharing = self
            .prefix_cache
            .lock()
            .map_err(|_| poison().into_core())?
            .is_some();
        #[cfg(not(all(feature = "model-qwen35", feature = "cuda")))]
        let prefix_sharing = false;
        Ok(ModelDetails {
            artifact: Some(ArtifactDescription {
                family: metadata.family.into(),
                format: zerogravity_artifacts::artifact::ArtifactFormat::Safetensors,
                cache_snapshot: None,
                revision: Some(SourceRevision {
                    repository: metadata.repository.into(),
                    revision: metadata.revision.into(),
                }),
            }),
            execution: Some(prepared.description()),
            state_policy: Some(resident.state_policy()),
            prefix_sharing: Some(prefix_sharing),
            source_host_reserved_bytes: None,
            memory: Some(ModelMemory::ResidentText {
                admission: resident.admission().clone(),
                resident_bytes: resident.resident_bytes()?,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serving_profile_refuses_unsupported_limits_before_io() {
        let good = QwenServingProfile {
            active: 2,
            context: 514,
            output: 2,
            max_pending: Some(4),
            cohort_width: 0,
        };
        let valid = |profile: QwenServingProfile| {
            profile
                .validate_base()
                .and_then(|_| profile.validate_cohort_width(profile.cohort_width))
        };
        assert!(valid(good).is_ok());
        for context in [8192, 16384, 32768] {
            assert!(valid(QwenServingProfile { context, ..good }).is_ok());
        }
        for bad in [
            QwenServingProfile { active: 0, ..good },
            QwenServingProfile { active: 9, ..good },
            QwenServingProfile {
                context: 511,
                ..good
            },
            QwenServingProfile {
                context: 32769,
                ..good
            },
            QwenServingProfile {
                context: 16384,
                output: 513,
                ..good
            },
            QwenServingProfile {
                output: 515,
                ..good
            },
            QwenServingProfile {
                max_pending: Some(0),
                ..good
            },
            QwenServingProfile {
                cohort_width: 2,
                ..good
            },
            QwenServingProfile {
                active: 2,
                cohort_width: 4,
                ..good
            },
        ] {
            assert!(valid(bad).is_err());
        }
    }

    #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
    #[test]
    fn paged_cuda_policy_refuses_invalid_shape_before_source_io() {
        let base = QwenServingProfile {
            active: 4,
            context: 1024,
            output: 2,
            max_pending: Some(4),
            cohort_width: 0,
        };
        let load = |profile, page_tokens, max_append_tokens| {
            load::load_router(
                vec!["paged-policy-preflight".into()],
                crate::cache::CacheOptions::new(None, 1),
                "not-a-source-digest".into(),
                profile,
                load::SourceBackend::Cuda {
                    provider_bytes: 1,
                    device_bytes: 1,
                    smart_config: crate::models::SmartConfig::enabled(
                        crate::models::QwenCudaSmartOverrides {
                            state_policy: Some(crate::models::QwenCudaStatePolicy::Paged {
                                page_tokens,
                                max_append_tokens,
                            }),
                            cohort_width: None,
                        },
                    ),
                },
            )
        };
        for (profile, page_tokens, max_append_tokens) in [
            (base, 0, 64),
            (base, 4, 0),
            (base, 1025, 64),
            (base, 4, 1025),
        ] {
            let error = match load(profile, page_tokens, max_append_tokens) {
                Ok(_) => panic!("invalid paged policy reached source loading"),
                Err(error) => error,
            };
            assert_eq!(error.status, 400);
            assert_eq!(error.code, "invalid_source_state_policy");
        }
    }
}
