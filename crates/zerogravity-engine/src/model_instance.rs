//! Prepared model extension seam. Instances cannot be minted from target metadata.
use crate::{Result, request::TaskRequest, session::TaskSession};
use std::sync::Arc;
use zerogravity_core::identity::SourceIdentity;

pub trait PreparedModel: Send + Sync {
    fn source(&self) -> SourceIdentity;
    fn prepare_session(&self, request: &TaskRequest) -> Result<Box<dyn TaskSession>>;
}
#[derive(Clone)]
pub struct ModelInstance {
    model: Arc<dyn PreparedModel>,
}
impl ModelInstance {
    pub fn source(&self) -> SourceIdentity {
        self.model.source()
    }
}
