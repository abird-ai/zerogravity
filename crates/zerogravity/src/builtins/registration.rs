//! One cold composition seam; only family/provider adapters know source loading.
use crate::models::ModelLoadRequest;
use zerogravity_engine::{
    router::{Router, ServiceError},
    scheduler::SchedulerLimits,
};

pub(crate) fn load_text(
    request: ModelLoadRequest,
    limits: SchedulerLimits,
) -> Result<Router, ServiceError> {
    match request {
        #[cfg(all(feature = "qwen35-cuda", target_os = "linux"))]
        ModelLoadRequest::QwenCuda {
            aliases,
            cache,
            snapshot,
            profile,
            provider_bytes,
            device_bytes,
        } => load_qwen_cuda(
            aliases,
            cache,
            snapshot,
            profile,
            provider_bytes,
            device_bytes,
            crate::models::QwenCudaStatePolicy::Contiguous,
            limits,
        ),
        #[cfg(all(feature = "qwen35-cuda", target_os = "linux"))]
        ModelLoadRequest::QwenCudaWithStatePolicy {
            aliases,
            cache,
            snapshot,
            profile,
            provider_bytes,
            device_bytes,
            state_policy,
        } => load_qwen_cuda(
            aliases,
            cache,
            snapshot,
            profile,
            provider_bytes,
            device_bytes,
            state_policy,
            limits,
        ),
        #[cfg(all(feature = "qwen35-hip", target_os = "linux"))]
        ModelLoadRequest::QwenHipWithBudget {
            aliases,
            cache,
            snapshot,
            profile,
            budget,
        } => {
            profile.validate()?;
            validate_limits(1, profile.active, profile.max_pending, limits)?;
            super::text::qwen_source::load_router(
                aliases,
                cache,
                snapshot,
                profile,
                super::text::qwen_source::SourceBackend::Hip(budget),
            )
        }
        #[cfg(all(feature = "qwen35-hip", target_os = "linux"))]
        ModelLoadRequest::QwenHip {
            aliases,
            cache,
            snapshot,
            profile,
        } => {
            profile.validate()?;
            validate_limits(1, profile.active, profile.max_pending, limits)?;
            super::text::qwen_source::load_router(
                aliases,
                cache,
                snapshot,
                profile,
                super::text::qwen_source::SourceBackend::Hip(Default::default()),
            )
        }
        #[cfg(feature = "serve")]
        ModelLoadRequest::ServingText { declared, profile } => {
            profile.validate()?;
            validate_limits(
                profile.workers,
                profile.active_per_worker,
                profile.max_pending_requests,
                limits,
            )?;
            profile.load(declared)
        }
        ModelLoadRequest::CpuText { declared, config } => {
            validate_limits(
                config.workers,
                config.active_per_worker,
                config.max_pending,
                limits,
            )?;
            #[cfg(all(feature = "cpu", feature = "text", feature = "cache"))]
            {
                super::text::load_router(
                    declared,
                    config.workers,
                    config.active_per_worker,
                    config.max_context,
                    config.max_output,
                    config.host_mib,
                    config.max_pending,
                )
            }
            #[cfg(not(all(feature = "cpu", feature = "text", feature = "cache")))]
            {
                let _ = declared;
                Err(ServiceError::new(
                    400,
                    "backend_not_compiled",
                    "CPU text loading requires the cpu, text and cache features",
                ))
            }
        }
        #[cfg(all(
            feature = "text",
            any(feature = "vulkan", feature = "cuda", feature = "hip")
        ))]
        ModelLoadRequest::GpuText {
            declared,
            backend,
            profile,
        } => {
            validate_limits(profile.workers, profile.active, profile.max_pending, limits)?;
            super::text::load_gpu_router(declared, backend, profile)
        }
    }
}

#[cfg(all(feature = "qwen35-cuda", target_os = "linux"))]
#[allow(clippy::too_many_arguments)]
fn load_qwen_cuda(
    aliases: Vec<String>,
    cache: crate::cache::CacheOptions,
    snapshot: String,
    profile: crate::models::QwenSourceProfile,
    provider_bytes: usize,
    device_bytes: u64,
    state_policy: crate::models::QwenCudaStatePolicy,
    limits: SchedulerLimits,
) -> Result<Router, ServiceError> {
    profile.validate()?;
    validate_limits(1, profile.active, profile.max_pending, limits)?;
    super::text::qwen_source::load_router(
        aliases,
        cache,
        snapshot,
        profile,
        super::text::qwen_source::SourceBackend::Cuda {
            provider_bytes,
            device_bytes,
            state_policy: state_policy.into(),
        },
    )
}

fn validate_limits(
    workers: usize,
    active: usize,
    pending: Option<usize>,
    limits: SchedulerLimits,
) -> Result<(), ServiceError> {
    if workers
        .checked_mul(active)
        .is_none_or(|n| n > limits.max_active)
        || !limits.allows_pending(pending)
    {
        Err(ServiceError::new(
            400,
            "invalid_config",
            "text scheduler exceeds runtime limits",
        ))
    } else {
        Ok(())
    }
}

#[cfg(all(test, feature = "qwen35-cuda", target_os = "linux"))]
mod tests {
    use super::*;
    use crate::{
        cache::CacheOptions,
        models::{QwenCudaStatePolicy, QwenSourceProfile},
    };

    fn limits() -> SchedulerLimits {
        SchedulerLimits {
            max_active: 4,
            ..Default::default()
        }
    }

    fn profile(active: usize, native_rows: usize) -> QwenSourceProfile {
        QwenSourceProfile {
            active,
            context: 512,
            output: 1,
            max_pending: Some(1),
            native_rows,
        }
    }

    fn expect_pre_source_error(request: ModelLoadRequest) -> ServiceError {
        match load_text(request, limits()) {
            Ok(_) => panic!("invalid digest reached source IO"),
            Err(error) => error,
        }
    }

    #[test]
    fn qwen_cuda_default_request_remains_contiguous() {
        let error = expect_pre_source_error(ModelLoadRequest::QwenCuda {
            aliases: vec!["default-contiguous".into()],
            cache: CacheOptions::new(None, 1),
            snapshot: "not-a-source-digest".into(),
            profile: profile(4, 4),
            provider_bytes: 1,
            device_bytes: 1,
        });
        assert_ne!(error.code, "invalid_source_state_policy");
    }

    #[test]
    fn explicit_paged_request_validates_policy_before_source_io() {
        let request = |profile, state_policy| ModelLoadRequest::QwenCudaWithStatePolicy {
            aliases: vec!["explicit-paged".into()],
            cache: CacheOptions::new(None, 1),
            snapshot: "not-a-source-digest".into(),
            profile,
            provider_bytes: 1,
            device_bytes: 1,
            state_policy,
        };

        for invalid in [
            request(
                profile(4, 4),
                QwenCudaStatePolicy::Paged {
                    page_tokens: 4,
                    max_append_tokens: 4,
                },
            ),
            request(
                profile(1, 0),
                QwenCudaStatePolicy::Paged {
                    page_tokens: 0,
                    max_append_tokens: 4,
                },
            ),
        ] {
            let error = expect_pre_source_error(invalid);
            assert_eq!(error.status, 400);
            assert_eq!(error.code, "invalid_source_state_policy");
        }

        let valid = expect_pre_source_error(request(
            profile(1, 0),
            QwenCudaStatePolicy::Paged {
                page_tokens: 4,
                max_append_tokens: 4,
            },
        ));
        assert_ne!(valid.code, "invalid_source_state_policy");
    }
}
