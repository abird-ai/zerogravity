//! Qwen CUDA route selection and ordinary capability refusal.
use super::*;

pub(super) fn selected(model: &zerogravity::pipeline::ModelInstance) -> bool {
    model.native_artifacts.is_some()
        || model
            .source
            .as_deref()
            .and_then(|s| match SourceSelector::parse(s).ok()? {
                SourceSelector::Local(p) => Some(p.is_dir()),
                _ => None,
            })
            .unwrap_or(false)
}

/// Ordinary builds refuse before inventory/NVML/CUDA metadata. Only the
/// explicit diagnostic target may enter the captured finite route.
pub(super) fn require_route_capability() -> Result<()> {
    #[cfg(feature = "qwen35-cuda-qualification")]
    {
        Ok(())
    }
    #[cfg(not(feature = "qwen35-cuda-qualification"))]
    {
        zerogravity::qwen35_cuda::require_ordinary_capability()
    }
}

#[cfg(not(feature = "qwen35-cuda-qualification"))]
pub(super) fn run_route(
    _: Request,
    _: devices::Inventory,
    _: Vec<devices::Observation>,
) -> Result<()> {
    zerogravity::qwen35_cuda::require_ordinary_capability()
}

#[cfg(feature = "qwen35-cuda-qualification")]
mod qualification;
#[cfg(feature = "qwen35-cuda-qualification")]
pub(super) use qualification::run as run_route;
#[cfg(all(feature = "qwen35-cuda-qualification", test))]
pub(super) use qualification::{
    FiniteRouteProfile, admit_pre_client_card, dry_run_admission_trace, dry_run_trace_status,
    executing_admission_trace, no_route_refusal, select_sm86_card,
};
