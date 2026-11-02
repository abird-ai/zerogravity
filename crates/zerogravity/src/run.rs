//! Shared runtime requests and explicit source planning/diagnostics.
pub use zerogravity_engine::output::{CommittedOutput, OutputMilestone, OutputStream};
pub use zerogravity_engine::session::{RootBinding, TaskSession};
pub use zerogravity_engine::{TaskInput, TaskRequest};

/// Cold source-request policy, not source or physical execution authority.
pub fn validate_source_request(request: &crate::run_policy::Request) -> crate::Result<()> {
    use crate::{
        Error,
        run_policy::{Backend, Speculation},
    };
    request.validate_policy()?;
    if !matches!(request.speculation, Speculation::Auto | Speculation::Off) {
        return Err(Error::Unsupported(
            "required speculative execution has no qualified local adapter; use auto/off".into(),
        ));
    }
    if request.numerical.as_deref().is_some_and(|p| p != "source") {
        return Err(Error::Unsupported(
            "requested numerical profile has no qualified direct adapter; source is available"
                .into(),
        ));
    }
    let compiled = match request.backend {
        Backend::Auto => cfg!(any(
            feature = "cpu",
            feature = "cuda",
            feature = "hip",
            feature = "vulkan"
        )),
        Backend::Cpu => cfg!(all(
            feature = "cpu",
            feature = "model-qwen35",
            feature = "text"
        )),
        Backend::Cuda => cfg!(feature = "cuda"),
        Backend::Rocm => cfg!(feature = "hip"),
        Backend::Vulkan => cfg!(feature = "vulkan"),
    };
    if !compiled {
        return Err(Error::Unsupported(format!(
            "{} backend requires compiled features for its backend/model/text capabilities; its CPU adapter requires cpu, model-qwen35 and text",
            request.backend.label()
        )));
    }
    Ok(())
}
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
pub mod cpu;

pub mod config;
