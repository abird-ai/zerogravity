//! Adapter to the original logical publisher; no new root creation/publication.
pub use zerogravity_core::session_root::{RootAttempt, RootBinding, SessionId};
pub use zerogravity_core::state::{PreparedDomain, StateRoot, TaskCoordinator};
pub trait TaskSession: Send {
    fn root(&self) -> RootBinding;
    fn cancel(&mut self) -> crate::Result<()>;
}
