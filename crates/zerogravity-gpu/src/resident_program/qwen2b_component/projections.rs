//! Typed prepared operations indexed by actual source nodes; one existing node loop.
use super::*;
use crate::cuda::native::fixed_prompt_mlp::{
    FixedPromptMlpPrograms, FixedPromptMlpQualificationInputs, FixedPromptMlpWeights,
};
use crate::cuda::native::fixed_prompt_projection::component::ConnectedProjectionPrograms;
use crate::cuda::native::post_attention_projection::OutPrograms;
use crate::cuda::native::qwen2b_mlp::{Qwen2bMlpPrograms, Qwen2bMlpQualificationInputs};
use crate::resident_leases::Qwen2bMlpWeights;
use std::collections::{BTreeMap, BTreeSet};
use zerogravity_core::accelerator::program::SourceOperator;

// Real distinct native packet owners, not model bodies or another evaluator.
#[derive(Clone)]
enum PreparedOperation {
    PackedReference(Arc<ProjectionPrograms>),
    PackedFixedPrompt(Arc<ConnectedProjectionPrograms>),
    Output(Arc<OutPrograms>),
    MlpFixedPrompt(Arc<FixedPromptMlpPrograms>),
    MlpReference(Arc<Qwen2bMlpPrograms>),
}
impl PreparedOperation {
    fn identity(&self) -> (u8, usize) {
        match self {
            Self::PackedReference(p) => (0, Arc::as_ptr(p) as usize),
            Self::PackedFixedPrompt(p) => (1, Arc::as_ptr(p) as usize),
            Self::Output(p) => (2, Arc::as_ptr(p) as usize),
            Self::MlpFixedPrompt(p) => (3, Arc::as_ptr(p) as usize),
            Self::MlpReference(p) => (4, Arc::as_ptr(p) as usize),
        }
    }
    fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        match self {
            Self::PackedReference(p) => p.immutable_weights(),
            Self::PackedFixedPrompt(p) => p.immutable_weights(),
            Self::Output(p) => p.immutable_weights(),
            Self::MlpFixedPrompt(p) => p.immutable_weights(),
            Self::MlpReference(p) => p.immutable_weights(),
        }
    }
    fn accesses(
        &self,
        op: &SourceOperator,
        x: &ResidentLease,
        y: &ResidentLease,
        workspace: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        match self {
            Self::PackedReference(p) => p.accesses(op, x, y, workspace),
            Self::PackedFixedPrompt(p) => p.accesses(op, x, y, workspace),
            Self::Output(p) => {
                p.require_operator(op, x, y)?;
                p.accesses()
            }
            Self::MlpFixedPrompt(p) => p.accesses(op, x, y, workspace),
            Self::MlpReference(p) => p.accesses(op, x, y, workspace),
        }
    }
    fn enqueue(
        &self,
        op: &SourceOperator,
        x: &ResidentLease,
        y: &ResidentLease,
        workspace: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        match self {
            Self::PackedReference(p) => p.enqueue(op, x, y, workspace, cancelled),
            Self::PackedFixedPrompt(p) => p.enqueue(op, x, y, workspace, cancelled),
            Self::Output(p) => {
                p.require_operator(op, x, y)?;
                p.enqueue(cancelled)
            }
            Self::MlpFixedPrompt(p) => p.enqueue(op, x, y, workspace, cancelled),
            Self::MlpReference(p) => p.enqueue(op, x, y, workspace, cancelled),
        }
    }
    unsafe fn retire(&self) -> Result<()> {
        unsafe {
            match self {
                Self::PackedReference(p) => p.retire_after_physical_completion(),
                Self::PackedFixedPrompt(p) => p.retire_after_physical_completion(),
                Self::Output(p) => p.retire_kernels_after_physical_completion(),
                Self::MlpFixedPrompt(p) => p.retire_after_physical_completion(),
                Self::MlpReference(p) => p.retire_after_physical_completion(),
            }
        }
    }
}
struct PreparedNode {
    operator: SourceOperator,
    operation: PreparedOperation,
}
pub(super) struct ComponentProjections {
    nodes: BTreeMap<usize, PreparedNode>,
    _metadata: WindowWorkspace,
}
impl ComponentProjections {
    fn from_nodes(
        values: &ComponentValues,
        metadata: WindowWorkspace,
        init: &mut ResidentInitialization<'_>,
        operations: impl IntoIterator<Item = (usize, PreparedOperation)>,
    ) -> Result<Arc<Self>> {
        // Admit map/Arc/selector controls before allocating them; no additional graph.
        let mut nodes = BTreeMap::new();
        let mut retained = BTreeSet::new();
        for (index, operation) in operations {
            if retained.insert(operation.identity()) {
                match &operation {
                    PreparedOperation::PackedReference(p) => {
                        init.retain_completion_owner(p.clone())?
                    }
                    PreparedOperation::PackedFixedPrompt(p) => {
                        init.retain_completion_owner(p.clone())?
                    }
                    PreparedOperation::Output(p) => init.retain_completion_owner(p.clone())?,
                    PreparedOperation::MlpFixedPrompt(p) => {
                        init.retain_completion_owner(p.clone())?
                    }
                    PreparedOperation::MlpReference(p) => {
                        init.retain_completion_owner(p.clone())?
                    }
                };
            }
            if !matches!(index, 2 | 3 | 6 | 8 | 9 | 10) || index >= values.recipe().node_end() {
                return Err(Error::State(
                    "prepared native node outside component extent".into(),
                ));
            }
            if nodes
                .insert(
                    index,
                    PreparedNode {
                        operator: values.recipe().operator(index)?.clone(),
                        operation,
                    },
                )
                .is_some()
            {
                return Err(Error::State(
                    "duplicate prepared native component node".into(),
                ));
            }
        }
        let expected: &[usize] = if values.recipe().cut().includes_mlp() {
            &[2, 3, 6, 8, 9, 10]
        } else if values.recipe().cut().includes_post_attention() {
            &[2, 3, 6]
        } else {
            &[2, 3]
        };
        if nodes.keys().copied().collect::<Vec<_>>() != expected {
            return Err(Error::State(
                "complete prepared native component node set differs".into(),
            ));
        }
        Ok(Arc::new(Self {
            nodes,
            _metadata: metadata,
        }))
    }
    pub(super) fn prepare(
        values: Arc<ComponentValues>,
        service: &Arc<ResidentService>,
        inputs: CubinQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let metadata = values.recipe().source().reserve_workspace(64 * 1024)?;
        let recipe = values.recipe();
        if recipe.cut().includes_mlp() {
            return Err(Error::Unsupported(
                "MLP requires separate exact activation/source factory".into(),
            ));
        }
        let (prefix, out) = match (recipe.kind(), recipe.cut()) {
            (ComponentKind::Qwen2b, ComponentCut::Gated) => (
                PreparedOperation::PackedReference(
                    crate::cuda::native::qwen2b_component::prepare_projections(
                        recipe.clone(),
                        service,
                        inputs,
                        init,
                        cancelled,
                    )?,
                ),
                None,
            ),
            (ComponentKind::FixedPrompt(_), ComponentCut::Gated) => (
                PreparedOperation::PackedFixedPrompt(
                    crate::cuda::native::fixed_prompt_projection::component::prepare_projections(
                        recipe.clone(),
                        service,
                        inputs,
                        init,
                        cancelled,
                    )?,
                ),
                None,
            ),
            (ComponentKind::Qwen2b, ComponentCut::PostAttention) => {
                let (p, o) =
                    crate::cuda::native::qwen2b_component::prepare_post_attention_projections(
                        values.clone(),
                        service,
                        inputs,
                        init,
                        cancelled,
                    )?;
                (PreparedOperation::PackedReference(p), Some(o))
            }
            (ComponentKind::FixedPrompt(_), ComponentCut::PostAttention) => {
                let(p,o)=crate::cuda::native::fixed_prompt_projection::component::prepare_post_attention_projections(values.clone(),service,inputs,init,cancelled)?;
                (PreparedOperation::PackedFixedPrompt(p), Some(o))
            }
            _ => {
                return Err(Error::Unsupported(
                    "native component input set absent".into(),
                ));
            }
        };
        let mut operations = vec![(2, prefix.clone()), (3, prefix)];
        if let Some(out) = out {
            operations.push((6, PreparedOperation::Output(out)));
        }
        Self::from_nodes(&values, metadata, init, operations)
    }
    pub(super) fn prepare_mlp(
        values: Arc<ComponentValues>,
        weights: Arc<FixedPromptMlpWeights>,
        inputs: FixedPromptMlpQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        if values.recipe().cut() != ComponentCut::Mlp {
            return Err(Error::Unsupported(
                "MLP factory exact eleven-node cut required".into(),
            ));
        }
        let metadata = values.recipe().source().reserve_workspace(64 * 1024)?;
        let p = FixedPromptMlpPrograms::prepare(values.clone(), weights, inputs, init, cancelled)?;
        Self::from_nodes(
            &values,
            metadata,
            init,
            [2, 3, 6, 8, 9, 10].map(|index| (index, PreparedOperation::MlpFixedPrompt(p.clone()))),
        )
    }
    pub(super) fn prepare_qwen2b_mlp(
        values: Arc<ComponentValues>,
        weights: Arc<Qwen2bMlpWeights>,
        inputs: Qwen2bMlpQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        crate::cuda::native::qwen2b_mlp::require_recipe(values.recipe())?;
        weights.require_bindings(values.recipe().bindings(), values.service())?;
        let metadata = values.recipe().source().reserve_workspace(64 * 1024)?;
        let p = Qwen2bMlpPrograms::prepare(values.clone(), weights, inputs, init, cancelled)?;
        Self::from_nodes(
            &values,
            metadata,
            init,
            [2, 3, 6, 8, 9, 10].map(|index| (index, PreparedOperation::MlpReference(p.clone()))),
        )
    }
    fn operation(&self, op: &SourceOperator) -> Result<&PreparedOperation> {
        let mut found = self.nodes.values().filter(|node| &node.operator == op);
        let first = found.next().ok_or_else(|| {
            Error::State("actual prepared native component operator absent".into())
        })?;
        if found.next().is_some() {
            return Err(Error::State(
                "ambiguous prepared native source operator".into(),
            ));
        }
        Ok(&first.operation)
    }
    fn reference_mlp(&self) -> Result<&Arc<Qwen2bMlpPrograms>> {
        match self.nodes.get(&10).map(|node| &node.operation) {
            Some(PreparedOperation::MlpReference(p)) => Ok(p),
            _ => Err(Error::Unsupported(
                "reference Down native operation absent".into(),
            )),
        }
    }
    pub(super) fn arm_qwen2b_down1_fault(&self, cancel: bool) -> Result<()> {
        self.reference_mlp()?.arm_down1_fault(cancel)
    }
    pub(super) fn qwen2b_native_launches(&self) -> Result<usize> {
        Ok(self.reference_mlp()?.native_launches())
    }
    pub(super) fn down1_enqueues(&self) -> Result<usize> {
        Ok(self.reference_mlp()?.down1_enqueues())
    }
    pub(super) fn down1_output_unpromoted(&self) -> Result<bool> {
        self.reference_mlp()?.down1_output_unpromoted()
    }
    pub(super) fn accesses(
        &self,
        op: &SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.operation(op)?.accesses(op, input, output, workspace)
    }
    pub(super) fn enqueue(
        &self,
        op: &SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.operation(op)?
            .enqueue(op, input, output, workspace, cancelled)
    }
    pub(super) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        let mut seen = BTreeSet::new();
        let mut weights = Vec::new();
        for node in self.nodes.values() {
            if seen.insert(node.operation.identity()) {
                weights.extend(node.operation.immutable_weights());
            }
        }
        weights
    }
    pub(super) fn retained_post_attention_values(&self, values: &Arc<ComponentValues>) -> bool {
        match self.nodes.get(&6).map(|node| &node.operation) {
            Some(PreparedOperation::Output(p)) => p.retained_values(values),
            Some(PreparedOperation::MlpFixedPrompt(p)) => p.retained_values(values),
            Some(PreparedOperation::MlpReference(p)) => p.retained_values(values),
            _ => false,
        }
    }
    pub(super) unsafe fn retire_after_physical_completion(&self) -> Result<()> {
        // Out kernels retire before the prefix's shared native set, exactly once.
        let mut seen = BTreeSet::new();
        for output_pass in [true, false] {
            for node in self.nodes.values() {
                if matches!(&node.operation, PreparedOperation::Output(_)) != output_pass {
                    continue;
                }
                if seen.insert(node.operation.identity()) {
                    unsafe { node.operation.retire()? };
                }
            }
        }
        Ok(())
    }
}
