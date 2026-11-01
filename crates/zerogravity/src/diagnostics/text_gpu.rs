//! Explicit finite captured-source serving diagnostics. No ordinary capability
//! claim; original recipe guards and refusals remain authoritative. Uses the
//! shared engine, never a diagnostic-owned queue, scheduler or worker thread.
use crate::{builtins::text::GpuProfile, devices::GpuBackend};
use std::path::{Path, PathBuf};
use zerogravity_engine::router::{Router, ServiceError};
mod qwen35_cuda;
pub use qwen35_cuda::load_qwen35_cuda;
#[cfg(feature = "qwen35-cuda-qualification")]
mod fixed_prompt;
#[cfg(feature = "qwen35-cuda-qualification")]
pub use fixed_prompt::load_fixed_prompt;
#[cfg(feature = "qwen35-cuda-qualification")]
mod captured;
