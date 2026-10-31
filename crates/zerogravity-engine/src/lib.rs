//! Cold Fabric shell. Existing execution is migrated here in F0, not duplicated.
//! No concrete model/backend, source load, worker, socket or new physical ledger.
pub use zerogravity_core::{Error, Result};
pub mod catalog;
pub mod diagnostics;
pub mod executable;
pub mod model_instance;
pub mod output;
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
pub use request::{TaskInput, TaskRequest};
pub use runtime::{Runtime, RuntimeBuilder};
