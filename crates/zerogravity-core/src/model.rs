//! Source-owned family recording; a semantic program is never execution or Ready.
use crate::{
    Error, Result, identity::SourceIdentity, semantic::ForwardPlan, source::TensorSource,
    task::TaskKind,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct ProgramRequest {
    pub task: TaskKind,
    pub context: usize,
    pub logits: bool,
    pub observations: bool,
}

/// Family roles/math only. Providers independently validate source, numerical,
/// layout, kernel and physical eligibility before preparing any execution.
/// Implementations validate their source; this trait is not authentication or enrollment.
pub trait ModelDefinition: Send + Sync {
    fn source(&self) -> SourceIdentity;
    fn tasks(&self) -> &[TaskKind];
    fn program(&self, request: ProgramRequest) -> Result<ForwardPlan>;
}

/// Original recorded graph and its actual source owner, not another model IR.
/// Construction requires a family recorder and retained source, never a bare plan or ID.
pub struct ModelProgram {
    plan: Arc<ForwardPlan>,
    source: Arc<dyn TensorSource>,
}
impl ModelProgram {
    /// This provider binding needs borrowed tensors; the general family contract
    /// does not. Window-bound sources keep their own source-retaining adapters.
    pub fn record<M: ModelDefinition + TensorSource + 'static>(
        model: Arc<M>,
        request: ProgramRequest,
    ) -> Result<Self> {
        if request.context == 0 {
            return Err(Error::Limit("model program context is zero".into()));
        }
        if !model.tasks().contains(&request.task) {
            return Err(Error::Unsupported(
                "model task has no recording implementation".into(),
            ));
        }
        let source = model.source_identity();
        if !source.is_bound() || model.source() != source {
            return Err(Error::State(
                "model program requires an original source owner".into(),
            ));
        }
        let plan = model.program(request)?;
        if plan.source() != source
            || model.source() != source
            || model.source_identity() != source
            || plan.prefix_capacity() != request.context
            || plan.logits().is_some() != request.logits
            || plan.is_operator_slice()
        {
            return Err(Error::State(
                "model recording differs from its source/task request".into(),
            ));
        }
        Ok(Self {
            plan: Arc::new(plan),
            source: model,
        })
    }
    pub fn program(&self) -> &Arc<ForwardPlan> {
        &self.plan
    }
    pub fn source(&self) -> &Arc<dyn TensorSource> {
        &self.source
    }
    /// Transfer the original owners to the physical provider; no reconstruction.
    pub fn into_parts(self) -> (Arc<dyn TensorSource>, Arc<ForwardPlan>) {
        (self.source, self.plan)
    }
}

/// Source-defined components, not a cross-model task pipeline or residency grant.
pub struct ModelAssembly {
    pub components: Vec<Arc<dyn ModelDefinition>>,
}
