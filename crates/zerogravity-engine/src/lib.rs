//! Model-neutral runtime, typed execution adapters and shared fair workers.
//! Concrete source/provider composition stays in callers; physical authority stays in core.
pub use zerogravity_core::{Error, Result};
pub mod catalog;
pub mod diagnostics;
pub mod embedding;
pub mod executable;
pub mod model_instance;
pub mod output;
pub mod pipeline;
pub mod placement;
pub mod prepare;
pub mod request;
pub mod residency;
pub mod router;
pub mod runtime;
pub mod scheduler;
pub mod session;
pub mod state_cache;
pub mod task;
pub use request::{TaskInput, TaskRequest, TaskTarget, TextControls, TextTask};
pub use runtime::{Runtime, RuntimeBuilder};
