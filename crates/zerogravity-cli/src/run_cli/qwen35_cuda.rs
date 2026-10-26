//! Qwen CUDA route selection and ordinary capability refusal.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
struct FiniteTargetProfile {
    target: &'static str,
    #[cfg(feature = "qwen35-cuda-qualification")]
    source_family: &'static str,
}

const FINITE_TARGET_PROFILES: &[FiniteTargetProfile] = &[
    FiniteTargetProfile {
        target: "qwen3.5-0.8b",
        #[cfg(feature = "qwen35-cuda-qualification")]
        source_family: "0.8b",
    },
    FiniteTargetProfile {
        target: "qwen3.5-2b",
        #[cfg(feature = "qwen35-cuda-qualification")]
        source_family: "2b",
    },
];

fn finite_target_profile(target: &str) -> Option<&'static FiniteTargetProfile> {
    FINITE_TARGET_PROFILES
        .iter()
        .find(|profile| profile.target == target)
}

#[cfg(feature = "qwen35-cuda-qualification")]
fn finite_source_profile(source_family: &str) -> Option<&'static FiniteTargetProfile> {
    FINITE_TARGET_PROFILES
        .iter()
        .find(|profile| profile.source_family == source_family)
}

pub(super) fn selected(backend: &Backend, model: &zerogravity::pipeline::ModelInstance) -> bool {
    if !matches!(backend, Backend::Auto | Backend::Cuda) {
        return false;
    }
    if model.native_artifacts.is_some() {
        return true;
    }
    // Untargeted direct sources remain generic until their contents identify a family.
    *backend == Backend::Cuda && finite_target_profile(&model.target).is_some()
}

/// Ordinary builds refuse before inventory/NVML/CUDA metadata. Only the
/// explicit diagnostic target may enter the captured finite route.
pub(super) fn require_route_capability() -> Result<()> {
    #[cfg(feature = "qwen35-cuda-qualification")]
    {
        Ok(())
    }
    #[cfg(all(feature = "qwen35-cuda", not(feature = "qwen35-cuda-qualification")))]
    {
        zerogravity::qwen35_cuda::require_ordinary_capability()
    }
    #[cfg(not(feature = "qwen35-cuda"))]
    {
        Err(Error::Unsupported(
            "Qwen CUDA requires the qwen35-cuda feature".into(),
        ))
    }
}

#[cfg(not(feature = "qwen35-cuda-qualification"))]
pub(super) fn run_route(
    _: Request,
    _: devices::Inventory,
    _: Vec<devices::Observation>,
) -> Result<()> {
    require_route_capability()
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
