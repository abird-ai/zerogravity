//! Source-owned family recording; a semantic program is never execution or Ready.
use crate::{
    Error, Result,
    identity::SourceIdentity,
    semantic::ForwardPlan,
    source::{ProgramSource, TensorSource},
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

/// Original recorded graph and retained source binding, not another model IR.
/// Source access remains typed: borrowed tensors need not stand in for windows.
pub struct ModelProgram<S: ProgramSource = Arc<dyn TensorSource>> {
    plan: Arc<ForwardPlan>,
    source: S,
}
impl ModelProgram {
    /// Record a borrowed-tensor family. Window-backed families transfer their
    /// existing authenticated recording with `from_recorded`.
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
        Self::from_recorded(model, Arc::new(plan))
    }
}
impl<S: ProgramSource> ModelProgram<S> {
    /// Transfer an already source-bound whole-model recording, preserving both
    /// owners. This checks the binding, not source authentication or model math;
    /// the family recorder and physical provider still validate their contracts.
    pub fn from_recorded(source: S, plan: Arc<ForwardPlan>) -> Result<Self> {
        let identity = source.source_identity();
        if !identity.is_bound() || plan.source() != identity || plan.is_operator_slice() {
            return Err(Error::State(
                "recorded model program source/region differs".into(),
            ));
        }
        Ok(Self { plan, source })
    }
    pub fn program(&self) -> &Arc<ForwardPlan> {
        &self.plan
    }
    pub fn source(&self) -> &S {
        &self.source
    }
    /// Transfer the original owners to the physical provider; no reconstruction.
    pub fn into_parts(self) -> (S, Arc<ForwardPlan>) {
        (self.source, self.plan)
    }
}

/// Source-defined components, not a cross-model task pipeline or residency grant.
pub struct ModelAssembly {
    pub components: Vec<Arc<dyn ModelDefinition>>,
}
