//! Existing live issuers and roots; content equality never recreates ownership.
pub use crate::accelerator::SourceIdentity;
pub use crate::session_root::{RootBinding, SessionId};

/// Artifact metadata only, not a verified source lease or live model identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactDigest(pub [u8; 32]);
