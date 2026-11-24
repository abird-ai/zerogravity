//! One original prepared owner; no erased downcast, new IR, queue or publisher.
use std::sync::Arc;
use zerogravity_core::{Result, provider::PreparedExecutable, semantic::ForwardPlan};

pub struct ExecutableBundle<E: PreparedExecutable> {
    pub(crate) prepared: E,
}
impl<E: PreparedExecutable> ExecutableBundle<E> {
    pub async fn source(&self) -> Result<E::Source> {
        self.prepared.source().await
    }
    pub async fn program(&self) -> Result<Arc<ForwardPlan>> {
        self.prepared.program().await
    }
    pub fn prepared(&self) -> &E {
        &self.prepared
    }
    /// Transfer the original owner into its existing typed execution adapter.
    pub fn into_prepared(self) -> E {
        self.prepared
    }
}
