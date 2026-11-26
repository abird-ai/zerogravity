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
        #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
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
            crate::models::SmartConfig::default(),
            limits,
        ),
        #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
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
            crate::models::SmartConfig::enabled(crate::models::QwenCudaSmartOverrides {
                state_policy: Some(state_policy),
                cohort_width: None,
            }),
            limits,
        ),
        #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
        ModelLoadRequest::QwenCudaWithSmartConfig {
            aliases,
            cache,
            snapshot,
            profile,
            provider_bytes,
            device_bytes,
            smart_config,
        } => load_qwen_cuda(
            aliases,
            cache,
            snapshot,
            profile,
            provider_bytes,
            device_bytes,
            smart_config,
            limits,
        ),
        #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
        ModelLoadRequest::QwenHipNative {
            aliases,
            cache,
            snapshot,
            profile,
            budget,
        } => load_qwen_source(
            aliases,
            cache,
            snapshot,
            profile,
            super::text::qwen_source::SourceBackend::Hip(
                budget,
                crate::accelerator::program::NumericalContract::HipRocBlasBf16MixedV2,
            ),
            limits,
        ),
        #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
        ModelLoadRequest::QwenHipWithBudget {
            aliases,
            cache,
            snapshot,
            profile,
            budget,
        } => load_qwen_source(
            aliases,
            cache,
            snapshot,
            profile,
            super::text::qwen_source::SourceBackend::Hip(
                budget,
                crate::accelerator::program::NumericalContract::TorchCpuAvx2Bf16MixedV1,
            ),
            limits,
        ),
        #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
        ModelLoadRequest::QwenHip {
            aliases,
            cache,
            snapshot,
            profile,
        } => load_qwen_source(
            aliases,
            cache,
            snapshot,
            profile,
            super::text::qwen_source::SourceBackend::Hip(
                Default::default(),
                crate::accelerator::program::NumericalContract::TorchCpuAvx2Bf16MixedV1,
            ),
            limits,
        ),
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
                super::text::load_router_declared(
                    super::text::SourceDeclarations::Paths(declared),
                    config,
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
            super::text::gpu::load_gpu_router_declared(
                super::text::SourceDeclarations::Paths(declared),
                backend,
                profile,
            )
        }
    }
}

#[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
#[allow(clippy::too_many_arguments)]
fn load_qwen_cuda(
    aliases: Vec<String>,
    cache: crate::cache::CacheOptions,
    snapshot: String,
    profile: crate::models::QwenServingProfile,
    provider_bytes: usize,
    device_bytes: u64,
    smart_config: crate::models::SmartConfig<crate::models::QwenCudaSmartOverrides>,
    limits: SchedulerLimits,
) -> Result<Router, ServiceError> {
    load_qwen_source(
        aliases,
        cache,
        snapshot,
        profile,
        super::text::qwen_source::SourceBackend::Cuda {
            provider_bytes,
            device_bytes,
            smart_config,
        },
        limits,
    )
}

/// One original HF text registration path, regardless of the selected
/// arithmetic/device policy. Retain compatibility request variants above.
#[cfg(all(
    feature = "model-qwen35",
    any(feature = "cuda", feature = "hip"),
    target_os = "linux"
))]
fn load_qwen_source(
    aliases: Vec<String>,
    cache: crate::cache::CacheOptions,
    snapshot: String,
    profile: crate::models::QwenServingProfile,
    backend: super::text::qwen_source::SourceBackend,
    limits: SchedulerLimits,
) -> Result<Router, ServiceError> {
    profile.validate_base()?;
    #[cfg(feature = "hip")]
    if matches!(backend, super::text::qwen_source::SourceBackend::Hip(..)) {
        profile.validate_cohort_width(profile.cohort_width)?;
    }
    validate_limits(1, profile.active, profile.max_pending, limits)?;
    super::text::qwen_source::load_router(aliases, cache, snapshot, profile, backend)
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

#[cfg(all(
    test,
    all(feature = "model-qwen35", feature = "cuda"),
    target_os = "linux"
))]
mod tests {
    use super::*;
    use crate::{
        cache::CacheOptions,
        models::{QwenCudaSmartOverrides, QwenCudaStatePolicy, QwenServingProfile, SmartConfig},
    };

    fn limits() -> SchedulerLimits {
        SchedulerLimits {
            max_active: 4,
            ..Default::default()
        }
    }

    fn profile(active: usize, cohort_width: usize) -> QwenServingProfile {
        QwenServingProfile {
            active,
            context: 512,
            output: 1,
            max_pending: Some(1),
            cohort_width,
        }
    }

    fn expect_pre_source_error(request: ModelLoadRequest) -> ServiceError {
        match load_text(request, limits()) {
            Ok(_) => panic!("invalid digest reached source IO"),
            Err(error) => error,
        }
    }

    #[test]
    fn qwen_cuda_default_request_enables_smart_config() {
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

        for invalid in [request(
            profile(1, 0),
            QwenCudaStatePolicy::Paged {
                page_tokens: 0,
                max_append_tokens: 4,
            },
        )] {
            let error = expect_pre_source_error(invalid);
            assert_eq!(error.status, 400);
            assert_eq!(error.code, "invalid_source_state_policy");
        }

        for profile in [profile(1, 0), profile(4, 4)] {
            let valid = expect_pre_source_error(request(
                profile,
                QwenCudaStatePolicy::Paged {
                    page_tokens: 4,
                    max_append_tokens: 4,
                },
            ));
            assert_ne!(valid.code, "invalid_source_state_policy");
        }
    }

    #[test]
    fn smart_config_modes_apply_user_precedence_before_source_io() {
        let invalid = QwenCudaStatePolicy::Paged {
            page_tokens: 0,
            max_append_tokens: 4,
        };
        let request = |smart_config| ModelLoadRequest::QwenCudaWithSmartConfig {
            aliases: vec!["smart-mode".into()],
            cache: CacheOptions::new(None, 1),
            snapshot: "not-a-source-digest".into(),
            profile: profile(1, 0),
            provider_bytes: 1,
            device_bytes: 1,
            smart_config,
        };
        let overrides = QwenCudaSmartOverrides {
            state_policy: Some(invalid),
            cohort_width: None,
        };

        for config in [
            SmartConfig::enabled(overrides),
            SmartConfig::disabled(overrides),
        ] {
            let error = expect_pre_source_error(request(config));
            assert_eq!(error.code, "invalid_source_state_policy");
        }

        let exclusive = expect_pre_source_error(request(SmartConfig::exclusive(overrides)));
        assert_ne!(exclusive.code, "invalid_source_state_policy");
    }
}

#[cfg(all(
    test,
    feature = "model-qwen35",
    any(feature = "cuda", feature = "hip"),
    target_os = "linux"
))]
#[test]
fn unapproved_snapshot_refuses_before_cache_or_physical_preparation() {
    use crate::{cache::CacheOptions, models::QwenServingProfile};
    let profile = QwenServingProfile {
        active: 1,
        context: 512,
        output: 1,
        max_pending: Some(1),
        cohort_width: 0,
    };
    let requests = [
        #[cfg(feature = "cuda")]
        ModelLoadRequest::QwenCuda {
            aliases: vec!["source".into()],
            cache: CacheOptions::new(Some("/source-not-opened".into()), 1),
            snapshot: "0".repeat(64),
            profile,
            provider_bytes: 1,
            device_bytes: 1,
        },
        #[cfg(feature = "hip")]
        ModelLoadRequest::QwenHip {
            aliases: vec!["source".into()],
            cache: CacheOptions::new(Some("/source-not-opened".into()), 1),
            snapshot: "0".repeat(64),
            profile,
        },
    ];
    for request in requests {
        let error = load_text(request, SchedulerLimits::default())
            .err()
            .unwrap();
        assert_eq!((error.status, error.code), (400, "unsupported"));
        assert!(error.message.contains("exact approved snapshot"));
    }
}

/// Shared declarations for SDK and selected graph consumers. Resolve only
/// after the original lifecycle slot and the explicit policy's bounds.
pub(crate) fn load_declared_text(
    models: &[&crate::pipeline::ModelInstance],
    policy: crate::models::TextLoadPolicy,
    cache_root: Option<&std::path::Path>,
    limits: SchedulerLimits,
) -> Result<Router, ServiceError> {
    use crate::models::TextLoadPolicy;
    zerogravity_engine::router::validate_aliases(models.iter().map(|m| m.id.as_str()))?;
    match policy {
        TextLoadPolicy::Cpu(config) => {
            validate_limits(
                config.workers,
                config.active_per_worker,
                config.max_pending,
                limits,
            )?;
            #[cfg(all(feature = "cpu", feature = "text", feature = "cache"))]
            {
                super::text::load_router_declared(
                    super::text::SourceDeclarations::Models(models, cache_root),
                    config,
                )
            }
            #[cfg(not(all(feature = "cpu", feature = "text", feature = "cache")))]
            {
                let _ = cache_root;
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
        TextLoadPolicy::Gpu { backend, profile } => {
            validate_limits(profile.workers, profile.active, profile.max_pending, limits)?;
            super::text::gpu::load_gpu_router_declared(
                super::text::SourceDeclarations::Models(models, cache_root),
                backend,
                profile,
            )
        }
        #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
        TextLoadPolicy::QwenHipNative {
            cache,
            profile,
            budget,
        } => {
            if cache_root.is_some() {
                return Err(ServiceError::new(
                    400,
                    "invalid_config",
                    "original HF policy owns its cache options",
                ));
            }
            let (aliases, snapshot) = original_snapshot(models)?;
            load_text(
                ModelLoadRequest::qwen_hip_native(aliases, cache, snapshot, profile, budget),
                limits,
            )
        }
        #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
        TextLoadPolicy::QwenCuda { cache, profile } => {
            if cache_root.is_some() {
                return Err(ServiceError::new(
                    400,
                    "invalid_config",
                    "original HF policy owns its cache options",
                ));
            }
            let (aliases, snapshot) = original_snapshot(models)?;
            load_text(
                ModelLoadRequest::qwen_cuda(aliases, cache, snapshot, profile),
                limits,
            )
        }
    }
}

#[cfg(all(
    feature = "model-qwen35",
    any(feature = "cuda", feature = "hip"),
    target_os = "linux"
))]
fn original_snapshot(
    models: &[&crate::pipeline::ModelInstance],
) -> Result<(Vec<String>, String), ServiceError> {
    use crate::{models::qwen_source_inventory::select_snapshot, run_policy::SourceSelector};
    let mut selected = None;
    let mut aliases = Vec::with_capacity(models.len());
    for model in models {
        if model.model_config.is_some()
            || model.tokenizer.is_some()
            || model.native_artifacts.is_some()
        {
            return Err(ServiceError::new(
                400,
                "unsupported",
                "original HF requires its pinned role-bound sidecars",
            ));
        }
        let pin = model.snapshot.as_deref().ok_or_else(|| {
            ServiceError::new(
                400,
                "invalid_model",
                "original HF pipeline requires an exact snapshot pin",
            )
        })?;
        let spec = select_snapshot(pin).map_err(ServiceError::inference)?;
        if spec.target != model.target {
            return Err(ServiceError::new(
                400,
                "invalid_model",
                "declared target differs from approved snapshot",
            ));
        }
        if let Some(source) = &model.source
            && !matches!(SourceSelector::parse(source).map_err(ServiceError::inference)?,
                SourceSelector::HuggingFace(repository) if repository == spec.repository)
        {
            return Err(ServiceError::new(
                400,
                "unsupported",
                "HF source selector must match the pinned approved repository",
            ));
        }
        if selected.is_some_and(|prior| prior != spec.snapshot_sha) {
            return Err(ServiceError::new(
                400,
                "unsupported",
                "original HF supports one distinct source per runtime",
            ));
        }
        selected = Some(spec.snapshot_sha);
        aliases.push(model.id.clone());
    }
    Ok((
        aliases,
        selected
            .ok_or_else(|| {
                ServiceError::new(400, "invalid_model", "empty pipeline source selection")
            })?
            .into(),
    ))
}
