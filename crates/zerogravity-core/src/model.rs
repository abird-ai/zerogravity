//! Family semantics. Registration and metadata alone grant no execution rights.
use crate::{Result, identity::SourceIdentity, semantic::ForwardPlan, task::TaskKind};

pub trait ModelDefinition: Send + Sync {
    fn source(&self) -> SourceIdentity;
    fn tasks(&self) -> &[TaskKind];
    fn program(&self, task: TaskKind) -> Result<ForwardPlan>;
}

/// Source-defined components, distinct from a cross-model task pipeline.
/// Each component must independently retain its authenticated source owner.
pub struct ModelAssembly {
    pub components: Vec<std::sync::Arc<dyn ModelDefinition>>,
}
