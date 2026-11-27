//! Native sentence encoder, deliberately separate from decoder generation.
//! The CPU source-bound candidate is not a loaded Runtime model capability.
pub use zerogravity_core::{Error, Result};
mod config;
pub use config::BertSentenceConfig;
#[cfg(all(feature = "source-binding", target_os = "linux"))]
mod source;
#[cfg(all(feature = "source-binding", target_os = "linux"))]
pub use source::{BertSentenceEncoder, Embedding};

/// Source-neutral family intent, not authorization to acquire/serve weights.
pub const FIRST_SOURCE: &str = "sentence-transformers/all-MiniLM-L6-v2";
