//! Source-owned family recording; a semantic program is never execution or Ready.
use crate::{
    Error, Result,
    accelerator::program::{SourceParameter, SourceTensor},
    identity::SourceIdentity,
    semantic::{ClosedRegion, ForwardPlan, PlanNode, PlanOperation, SourceOperatorTemplate},
    source::{ProgramSource, RegionSource, TensorSource},
    task::TaskKind,
};
use std::{ops::Range, sync::Arc};

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
    /// Derive a closed region from the original program and retain its typed source.
    pub fn closed_region(&self, nodes: Range<usize>) -> Result<ModelRegion<S>> {
        Ok(ModelRegion {
            source: self.source.clone(),
            region: ClosedRegion::derive(self.plan.clone(), nodes)?,
        })
    }
    /// Transfer the original owners to the physical provider; no reconstruction.
    pub fn into_parts(self) -> (S, Arc<ForwardPlan>) {
        (self.source, self.plan)
    }
}

/// Source-bound region contract, not an executable or physical admission.
pub struct ModelRegion<S: ProgramSource = Arc<dyn TensorSource>> {
    source: S,
    region: ClosedRegion,
}
impl<S: ProgramSource> ModelRegion<S> {
    pub fn source(&self) -> &S {
        &self.source
    }
    pub fn region(&self) -> &ClosedRegion {
        &self.region
    }
    pub fn program(&self) -> &Arc<ForwardPlan> {
        self.region.program()
    }
}

impl<S: RegionSource> ModelRegion<S> {
    /// Ordered cold lowering view of the original nodes and their learned
    /// operands. No graph serialization, new IR, cache or byte materialization.
    pub fn operators(&self) -> impl ExactSizeIterator<Item = SourceRegionOperator<'_, S>> + '_ {
        let first = self.region.range().start;
        self.region
            .nodes()
            .iter()
            .enumerate()
            .map(move |(offset, node)| {
                let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation
                else {
                    unreachable!("ClosedRegion admits only original source templates")
                };
                SourceRegionOperator {
                    source: &self.source,
                    index: first + offset,
                    node,
                    template: descriptor,
                }
            })
    }
}

/// Borrowed view of an original closed-region node. Bind operands in their
/// declared order, preserving duplicates/roles and source-specific backing.
pub struct SourceRegionOperator<'a, S: RegionSource> {
    source: &'a S,
    index: usize,
    node: &'a PlanNode,
    template: &'a SourceOperatorTemplate,
}
impl<'a, S: RegionSource> SourceRegionOperator<'a, S> {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn node(&self) -> &'a PlanNode {
        self.node
    }
    pub fn template(&self) -> &'a SourceOperatorTemplate {
        self.template
    }
    pub fn matrices(
        &self,
    ) -> impl ExactSizeIterator<Item = Result<(&'a SourceTensor, S::Matrix<'a>)>> + 'a {
        let source = self.source;
        self.template
            .matrices()
            .iter()
            .map(move |descriptor| Ok((descriptor, source.bind_matrix(descriptor)?)))
    }
    pub fn parameters(
        &self,
    ) -> impl ExactSizeIterator<Item = Result<(&'a SourceParameter, S::Parameter<'a>)>> + 'a {
        let source = self.source;
        self.template
            .parameters()
            .iter()
            .map(move |descriptor| Ok((descriptor, source.bind_parameter(descriptor)?)))
    }
}

/// Source-defined components, not a cross-model task pipeline or residency grant.
pub struct ModelAssembly {
    pub components: Vec<Arc<dyn ModelDefinition>>,
}
