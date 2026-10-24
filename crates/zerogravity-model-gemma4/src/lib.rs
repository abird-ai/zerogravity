//! Source-pinned Gemma 4 architecture and tensor contracts.
//!
//! Contract validation does not establish executable model support or trained
//! source parity. Numerical execution stays in the shared mathematical engine.
pub use zerogravity_core::{Error, Result};
pub mod config;
pub mod moe;
pub mod variant_26b;
pub mod variant_31b;
pub use config::{AttentionKind, Gemma4Config, Gemma4LayerSpec, Gemma4Variant, RotaryKind};
pub mod tensors;
pub use tensors::Gemma4TensorSpec;
