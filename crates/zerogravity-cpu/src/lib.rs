//! CPU operations and numerical profiles for the shared mathematical engine.

pub use zerogravity_artifacts::{artifact, quant};
pub use zerogravity_core::{Error, Result, accelerator};

pub mod numerics;
pub mod ops;
pub mod reference_text;
pub mod runtime;
pub mod source;
