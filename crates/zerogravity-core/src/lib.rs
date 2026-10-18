//! Shared logical state, device completion and physical resource contracts.
//! No model family, tokenizer, GPU SDK or execution backend dependencies.

pub mod accelerator;
pub mod context_filter;
pub mod context_snapshot;
pub mod device_events;
pub mod device_queue;
pub mod pipeline;
pub mod quant;
pub mod resource_arbiter;
pub mod session_root;
pub mod source;
pub mod state;
pub mod targets;

use std::fmt;
use std::io;

/// Typed failures distinguish unsupported semantics from corrupt input.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Malformed(String),
    Unsupported(String),
    Limit(String),
    State(String),
}
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Malformed(s) => write!(f, "malformed model artifact: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Limit(s) => write!(f, "resource limit: {s}"),
            Self::State(s) => write!(f, "invalid session state: {s}"),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
