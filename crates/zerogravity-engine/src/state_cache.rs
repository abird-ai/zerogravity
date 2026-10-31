//! Full-domain restore required; a KV-only match is not a hybrid prefix hit.
use crate::{Error, Result};
use zerogravity_core::identity::RootBinding;
pub struct PrefixRequest {
    pub root: RootBinding,
    pub requested_frontier: u64,
}
pub fn restore(_request: &PrefixRequest) -> Result<()> {
    Err(Error::Unsupported(
        "Fabric complete-domain prefix restore is not implemented".into(),
    ))
}
