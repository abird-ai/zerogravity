//! One original prepared owner; no erased downcast, new IR, queue or publisher.
use std::sync::Arc;
use zerogravity_core::{
    Result,
    provider::{PreparedExecutable, PreparedRegion},
    semantic::ForwardPlan,
};

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
    pub fn prepared_region(&self) -> Option<&PreparedRegion<E::Source>> {
        self.prepared.prepared_region()
    }
    pub fn region_nodes(&self) -> Option<std::ops::Range<usize>> {
        self.prepared_region()
            .map(|region| region.model().region().range())
    }
    pub fn prepared(&self) -> &E {
        &self.prepared
    }
    /// Transfer the original owner into its existing typed execution adapter.
    pub fn into_prepared(self) -> E {
        self.prepared
    }
}
