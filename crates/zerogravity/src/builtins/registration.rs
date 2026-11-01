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
