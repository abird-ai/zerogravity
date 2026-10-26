//! Bind common mathematical source metadata to the actual verified snapshot.
use std::{collections::BTreeMap, sync::Arc};
use zerogravity_artifacts::source_weights::{VerifiedMatrix, VerifiedTensorSource};
use zerogravity_core::accelerator::program::{
    Bf16TextCall, Bf16TextProjectionGeometry, Bf16TextRow, ForwardPlan, PlanOperation, SourceNorm,
    SourceParameter, TextObservation, TextPlanBindings, TextPlanInventory,
};
use zerogravity_core::{Error, Result};

pub(crate) struct TextSourcePlan {
    bindings: Arc<TextPlanBindings>,
    matrices: BTreeMap<String, VerifiedMatrix>,
    source: VerifiedTensorSource,
}
impl TextSourcePlan {
    pub(crate) fn bind(
        source: VerifiedTensorSource,
        bindings: Arc<TextPlanBindings>,
    ) -> Result<Arc<Self>> {
        if source.source_identity() != bindings.source_identity() {
            return Err(Error::State(
                "text plan actual source snapshot differs".into(),
            ));
        }
        let mut matrices = BTreeMap::new();
        for node in bindings.plan().nodes() {
            if let PlanOperation::Project { tensor } | PlanOperation::Embed { tensor, .. } =
                &node.operation
            {
                let matrix = source.matrix(&tensor.name, tensor.axes)?;
                if tensor.source != source.source_identity()
                    || matrix.source_identity() != tensor.source
                    || matrix.rows() != tensor.rows
                    || matrix.columns() != tensor.columns
                    || matrix.tensor().kind != tensor.kind
                {
                    return Err(Error::State(
                        "text actual matrix source/shape/codec differs".into(),
                    ));
                }
                if let Some(prior) = matrices.insert(tensor.name.clone(), matrix.clone()) {
                    if !prior.same_binding(&matrix) {
                        return Err(Error::State("text repeated matrix binding differs".into()));
                    }
                }
            }
        }
        for index in 0..bindings.plan().nodes().len() {
            if let Ok(SourceNorm::Direct(parameter)) = bindings.normalization_source(index) {
                require_parameter(&source, parameter)?;
            }
            if let Some(parameter) = bindings.scalar_source(index) {
                require_parameter(&source, parameter)?;
            }
        }
        Ok(Arc::new(Self {
            bindings,
            matrices,
            source,
        }))
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn plan(&self) -> &ForwardPlan {
        self.bindings.plan()
    }
    pub(crate) fn shared_plan(&self) -> &Arc<ForwardPlan> {
        self.bindings.shared_plan()
    }
    pub(crate) fn call(&self) -> &Bf16TextCall {
        self.bindings.call()
    }
    pub(crate) fn inventory(&self) -> &TextPlanInventory {
        self.bindings.inventory()
    }
    pub(crate) fn normalization_source(&self, node: usize) -> Result<&SourceNorm> {
        self.bindings.normalization_source(node)
    }
    pub(crate) fn observations(&self) -> &[TextObservation] {
        self.bindings.observations()
    }
    pub(crate) fn projection_for_row(
        &self,
        node: usize,
        row: &Bf16TextRow<'_>,
    ) -> Result<(&VerifiedMatrix, Bf16TextProjectionGeometry)> {
        let geometry = self.bindings.projection_geometry_for_row(node, row)?;
        let PlanOperation::Project { tensor } = &self.plan().nodes()[node].operation else {
            unreachable!("checked source projection node")
        };
        let matrix = self
            .matrices
            .get(&tensor.name)
            .ok_or_else(|| Error::State("text prepared matrix binding absent".into()))?;
        Ok((matrix, geometry))
    }
}
fn require_parameter(source: &VerifiedTensorSource, expected: &SourceParameter) -> Result<()> {
    if source.parameter(&expected.name)?.source_parameter() != *expected {
        return Err(Error::State("text actual parameter binding differs".into()));
    }
    Ok(())
}
