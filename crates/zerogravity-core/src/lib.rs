//! Shared logical state, device completion and physical resource contracts.
//! No model family, tokenizer, GPU SDK or execution backend dependencies.

pub mod accelerator;
pub mod context_filter;
pub mod context_snapshot;
pub mod device_events;
pub mod device_queue;
pub mod encoding;
pub mod identity;
pub mod model;
pub mod numerics;
pub mod optimization;
pub mod paged_state;
pub mod physical;
pub mod pipeline;
pub mod placement;
pub mod provider;
pub mod resource_arbiter;
pub mod semantic;
pub mod session_root;
pub mod source;
pub mod state;
pub mod targets;
pub mod task;

use std::fmt;
use std::io;

/// A concrete resource admission failure.
///
/// Keep arithmetic/configuration bounds in Error::Limit. Use this only when a
/// real resource budget is known and the attempted reservation does not fit,
/// so public callers can see what resource was short and by how much.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceLimit {
    pub resource: String,
    pub operation: String,
    pub requested: u64,
    pub available: u64,
    pub scope: Option<String>,
}
impl ResourceLimit {
    pub fn new(
        resource: impl Into<String>,
        operation: impl Into<String>,
        requested: u64,
        available: u64,
    ) -> Self {
        Self {
            resource: resource.into(),
            operation: operation.into(),
            requested,
            available,
            scope: None,
        }
    }

    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }
}
impl fmt::Display for ResourceLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn bytes(f: &mut fmt::Formatter<'_>, value: u64) -> fmt::Result {
            const MIB: u64 = 1024 * 1024;
            const GIB: u64 = 1024 * MIB;
            if value >= GIB && value.is_multiple_of(GIB) {
                write!(f, "{value} bytes ({} GiB)", value / GIB)
            } else if value >= MIB {
                write!(f, "{value} bytes ({} MiB)", value / MIB)
            } else {
                write!(f, "{value} bytes")
            }
        }

        write!(
            f,
            "{} exhausted during {}: requested ",
            self.resource, self.operation
        )?;
        bytes(f, self.requested)?;
        write!(f, ", available ")?;
        bytes(f, self.available)?;
        if let Some(scope) = &self.scope {
            write!(f, "; {scope}")?;
        }
        Ok(())
    }
}

/// Typed failures distinguish unsupported semantics from corrupt input.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Malformed(String),
    Unsupported(String),
    Resource(ResourceLimit),
    Limit(String),
    State(String),
}
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn resource(
        resource: impl Into<String>,
        operation: impl Into<String>,
        requested: u64,
        available: u64,
    ) -> Self {
        Self::Resource(ResourceLimit::new(
            resource, operation, requested, available,
        ))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Malformed(s) => write!(f, "malformed model artifact: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Resource(limit) => write!(f, "{limit}"),
            Self::Limit(s) => write!(f, "limit exceeded: {s}"),
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

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn resource_limit_reports_resource_operation_and_exact_budget() {
        let error = Error::Resource(
            ResourceLimit::new(
                "device memory / VRAM budget",
                "model admission",
                5 * 1024 * 1024 * 1024,
                4608 * 1024 * 1024,
            )
            .with_scope("device=pci/0000:01:00.0 owner=model/session"),
        );
        let message = error.to_string();
        assert!(message.contains("device memory / VRAM budget exhausted during model admission"));
        assert!(message.contains("5368709120 bytes (5 GiB)"));
        assert!(message.contains("4831838208 bytes (4608 MiB)"));
        assert!(message.contains("device=pci/0000:01:00.0 owner=model/session"));
    }
}

pub mod run_policy;
