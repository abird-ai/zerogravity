//! Qwen CUDA route selection and ordinary capability refusal.
use super::*;

pub(super) fn selected(backend: &Backend, model: &zerogravity::pipeline::ModelInstance) -> bool {
    if !matches!(backend, Backend::Auto | Backend::Cuda) {
        return false;
    }
    if model.native_artifacts.is_some() {
        return true;
    }
    if *backend != Backend::Cuda {
        return false;
    }
    let qwen_target = model.target.is_empty()
        || model.target.starts_with("qwen3.5-")
        || model.target.starts_with("qwen35-");
    qwen_target
        && model.source.as_deref().is_some_and(|source| {
            matches!(
                SourceSelector::parse(source),
                Ok(SourceSelector::Local(path))
                    if !path.extension().is_some_and(|extension| {
                        extension
                            .to_str()
                            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
                    })
            )
        })
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
