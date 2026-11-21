//! Qwen CUDA route selection and ordinary capability refusal.
use crate::{
    Error, Result, devices,
    run_policy::{Backend, Request},
};

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

pub fn selected(backend: &Backend, model: &crate::pipeline::ModelInstance) -> bool {
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
pub fn require_route_capability() -> Result<()> {
    #[cfg(feature = "qwen35-cuda-qualification")]
    {
        Ok(())
    }
    #[cfg(all(feature = "qwen35-cuda", not(feature = "qwen35-cuda-qualification")))]
    {
        crate::qwen35_cuda::require_forward_capability()
    }
    #[cfg(not(feature = "qwen35-cuda"))]
    {
        Err(Error::Unsupported(
            "Qwen CUDA requires the qwen35-cuda feature".into(),
        ))
    }
}

/// Caller-owned presentation. Borrowed data is never a physical Ready witness.
pub enum Report<'a> {
    Decision {
        value: &'a serde_json::Value,
        pretty: bool,
    },
    Text(&'a str),
}
/// Explicit finite diagnostic only. The caller supplies stdin/presentation; this
/// library never installs signals, reads process input, or selects a substitute.
pub fn run_route(
    request: Request,
    inventory: devices::Inventory,
    read_prompt: impl FnOnce() -> Result<String>,
    report: impl FnMut(Report<'_>) -> Result<()>,
) -> Result<()> {
    crate::run::validate_source_request(&request)?;
    require_route_capability()?;
    let crate::run_policy::Entrypoint::Model(id) = &request.entrypoint else {
        return Err(Error::Malformed(
            "finite HF requires model entrypoint".into(),
        ));
    };
    let model = request
        .graph
        .models
        .iter()
        .find(|model| &model.id == id)
        .ok_or_else(|| Error::Malformed("entrypoint absent".into()))?;
    if !selected(&request.backend, model) {
        return Err(Error::Unsupported(
            "finite CUDA diagnostic route is not selected".into(),
        ));
    }
    // The SDK owns allowlist validation; an observation is not execution authority.
    devices::allowed(&request, &inventory)?;
    #[cfg(feature = "qwen35-cuda-qualification")]
    {
        qualification::run(request, inventory, read_prompt, report)
    }
    #[cfg(not(feature = "qwen35-cuda-qualification"))]
    {
        let _ = (request, inventory, read_prompt, report);
        Err(Error::Unsupported(
            "finite CUDA route requires explicit qualification feature".into(),
        ))
    }
}
#[cfg(feature = "qwen35-cuda-qualification")]
mod qualification;
#[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux", test))]
mod qwen35_cuda_route_tests;
