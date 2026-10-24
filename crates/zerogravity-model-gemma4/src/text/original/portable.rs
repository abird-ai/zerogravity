//! Authentic whole-text recording and bounded source tiles; no GPU execution.
use super::*;
use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedMatrix};
use zerogravity_core::accelerator::program::TextCallGeometry;
use zerogravity_core::session_root::RootAttempt;

mod inventory;
mod rows;
pub use inventory::{Gemma4Bf16Inventory, Gemma4Bf16Observation};
#[cfg(test)]
mod tests;

const METADATA: usize = 4 * 1024 * 1024;
const PAGE: usize = 1024 * 1024;

#[derive(Clone, Debug)]
pub enum Gemma4Bf16NormSource {
    /// Original checkpoint parameter, without Qwen's1+weight transformation.
    Direct(SourceParameter),
    /// Model-defined unit value normalization; no checkpoint gamma exists.
    Unit { width: usize },
}

pub struct Gemma4Bf16TextPlan {
    model: Arc<OriginalGemma4Text>,
    plan: ForwardPlan,
    call: Bf16TextCall,
    norms: BTreeMap<usize, Gemma4Bf16NormSource>,
    scalars: BTreeMap<usize, SourceParameter>,
    tiles: BTreeMap<String, Gemma4SourceTiles>,
    inventory: Gemma4Bf16Inventory,
    _metadata: WindowWorkspace,
}
impl OriginalGemma4Text {
    /// The same family traversal and real attempt, never a model/Ready permit.
    pub fn record_bf16_text_call(
        self: &Arc<Self>,
        attempt: RootAttempt,
        ids: Arc<[u32]>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4Bf16TextPlan> {
        cancelled_check(cancelled)?;
        require_geometry(self)?;
        let geometry = TextCallGeometry::new(self.source_identity(), self.geometry.vocab, 64)?;
        let call = Bf16TextCall::bind(geometry, attempt, ids)?;
        let metadata = self.source.source().reserve_workspace(METADATA)?;
        let weights = ArithmeticWeights {
            base: self.as_ref(),
            arithmetic: ReferenceTextArithmetic::Bf16EagerCandidate,
        };
        let plan = record(&weights)?;
        let (norms, scalars) = bind_parameters(self, &plan)?;
        let mut tiles = BTreeMap::new();
        for descriptor in self.matrices.values() {
            if tiles.contains_key(&descriptor.name) {
                continue;
            }
            let matrix = self.source().matrix(&descriptor.name, descriptor.axes)?;
            if matrix.source_identity() != self.source_identity()
                || matrix.rows() != descriptor.rows
                || matrix.columns() != descriptor.columns
                || matrix.tensor().kind != descriptor.kind
            {
                return Err(Error::State(
                    "Gemma BF16 actual complete matrix source".into(),
                ));
            }
            tiles.insert(
                descriptor.name.clone(),
                Gemma4SourceTiles::new(self.source().clone(), matrix)?,
            );
        }
        let inventory = Gemma4Bf16Inventory::derive(self, &plan, &call, &tiles)?;
        cancelled_check(cancelled)?;
        Ok(Gemma4Bf16TextPlan {
            model: self.clone(),
            plan,
            call,
            norms,
            scalars,
            tiles,
            inventory,
            _metadata: metadata,
        })
    }
}
fn require_geometry(model: &OriginalGemma4Text) -> Result<()> {
    let g = &model.geometry;
    if g.hidden != 1536
        || g.vocab != 262144
        || g.layers.len() != 35
        || g.ple != 256
        || g.moe.is_some()
        || model.matrices.values().any(|m| m.kind != QuantKind::Bf16)
        || model
            .parameters
            .values()
            .any(|p| p.descriptor.kind != QuantKind::Bf16)
    {
        return Err(Error::Unsupported(
            "Gemma BF16 text complete authentic E2B geometry".into(),
        ));
    }
    Ok(())
}
fn record(weights: &impl Weights) -> Result<ForwardPlan> {
    let mut builder = PlanBuilder::new_with_profile(
        weights.source_identity(),
        64,
        weights
            .geometry()
            .domains()
            .into_iter()
            .map(StateSlot::Kv)
            .collect(),
        true,
        ProgramNumericalProfile::Gemma4E2bBf16TextCandidateV1,
    )?;
    let output = forward(weights, &mut builder, PlanBuilder::inputs(), true)?;
    let plan = builder.finish(output)?;
    if plan.frontier_delta() != 1
        || plan.logits().is_none()
        || plan.proposed_domains().len() != 15
        || plan.nodes().len() > 8192
        || plan.values().len() > 4096
    {
        return Err(Error::State(
            "Gemma BF16 complete recorded source traversal".into(),
        ));
    }
    Ok(plan)
}
fn bind_parameters(
    model: &OriginalGemma4Text,
    plan: &ForwardPlan,
) -> Result<(
    BTreeMap<usize, Gemma4Bf16NormSource>,
    BTreeMap<usize, SourceParameter>,
)> {
    let mut norms = BTreeMap::new();
    let mut scalars = BTreeMap::new();
    for (index, node) in plan.nodes().iter().enumerate() {
        if let PlanOperation::Rms { gamma, eps, .. } = node.operation {
            let vector = &plan.constants()[gamma.0 as usize];
            if vector.source != model.source_identity() || eps.to_bits() != 1e-6f32.to_bits() {
                return Err(Error::State("Gemma BF16 direct norm source/epsilon".into()));
            }
            let source = if let Some(parameter) = model
                .parameters
                .values()
                .find(|p| p.descriptor.name == vector.name)
            {
                if parameter.values.len() != vector.values.len()
                    || parameter
                        .values
                        .iter()
                        .zip(vector.values.iter())
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err(Error::State(
                        "Gemma BF16 norm raw source words differ".into(),
                    ));
                }
                Gemma4Bf16NormSource::Direct(parameter.descriptor.clone())
            } else if matches!(vector.values.len(), 256 | 512)
                && vector.name == format!("gemma4.pure_value_rms.{}", vector.values.len())
                && vector
                    .values
                    .iter()
                    .all(|v| v.to_bits() == 1.0f32.to_bits())
            {
                Gemma4Bf16NormSource::Unit {
                    width: vector.values.len(),
                }
            } else {
                return Err(Error::State(
                    "Gemma BF16 norm lacks raw parameter or source unit role".into(),
                ));
            };
            norms.insert(index, source);
        }
        if let PlanOperation::TextElementwise(TextElementwise::Scale(value)) = node.operation
            && let Some(parameter) = model
                .parameters
                .values()
                .find(|p| p.descriptor.name == node.name)
        {
            if parameter.descriptor.shape != [1]
                || parameter.values.len() != 1
                || parameter.values[0].to_bits() != value.to_bits()
            {
                return Err(Error::State("Gemma BF16 actual layer scalar source".into()));
            }
            scalars.insert(index, parameter.descriptor.clone());
        }
    }
    Ok((norms, scalars))
}
impl Gemma4Bf16TextPlan {
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn call(&self) -> &Bf16TextCall {
        &self.call
    }
    pub fn source(&self) -> &Arc<OriginalGemma4Text> {
        &self.model
    }
    pub fn inventory(&self) -> &Gemma4Bf16Inventory {
        &self.inventory
    }
    pub fn normalization_source(&self, node: usize) -> Result<&Gemma4Bf16NormSource> {
        self.norms
            .get(&node)
            .ok_or_else(|| Error::State("Gemma actual normalization node absent".into()))
    }
    pub fn scalar_source(&self, node: usize) -> Option<&SourceParameter> {
        self.scalars.get(&node)
    }
    pub fn matrix_tiles(&self, role: Gemma4MatrixRole) -> Result<&Gemma4SourceTiles> {
        let descriptor = self.model.matrix_descriptor(role)?;
        self.tiles
            .get(&descriptor.name)
            .ok_or_else(|| Error::State("Gemma actual matrix tile owner absent".into()))
    }
    pub fn embedding_row(
        &self,
        role: Gemma4MatrixRole,
        index: usize,
    ) -> Result<Gemma4SourceTile<'_>> {
        if !matches!(
            role,
            Gemma4MatrixRole::Embedding | Gemma4MatrixRole::PerLayerEmbedding
        ) {
            return Err(Error::State(
                "Gemma selected token row requires an embedding role".into(),
            ));
        }
        self.matrix_tiles(role)?
            .selected_row(self.call.row(index)?.token())
    }
    pub fn projection_tiles(
        &self,
        node: usize,
        index: usize,
    ) -> Result<(&Gemma4SourceTiles, Bf16TextProjectionGeometry)> {
        let row = self.call.row(index)?;
        self.projection_tiles_for_row(node, &row)
    }
    /// Reuse this immutable recorded traversal with a fresh actual source/root
    /// call row; no repeated full plan allocation or new evaluator.
    pub fn projection_tiles_for_row(
        &self,
        node: usize,
        row: &Bf16TextRow<'_>,
    ) -> Result<(&Gemma4SourceTiles, Bf16TextProjectionGeometry)> {
        if node >= self.plan.state_nodes().len() && !row.needs_logits() {
            return Err(Error::State(
                "Gemma state-only prompt row has no final output producer".into(),
            ));
        }
        let Some(PlanNode {
            operation: PlanOperation::Project { tensor },
            ..
        }) = self.plan.nodes().get(node)
        else {
            return Err(Error::State(
                "Gemma actual projection source node absent".into(),
            ));
        };
        let geometry = row.projection_geometry(tensor)?;
        let tiles = self
            .tiles
            .get(&tensor.name)
            .ok_or_else(|| Error::State("Gemma projection tile source absent".into()))?;
        Ok((tiles, geometry))
    }
    pub fn trained_model_qualified(&self) -> bool {
        false
    }
}

/// Whole original matrix plus a bounded full-K output-row recipe. No fake full-W lease.
pub struct Gemma4SourceTiles {
    source: VerifiedTensorSource,
    matrix: VerifiedMatrix,
    layout: rows::Rows,
}
impl Gemma4SourceTiles {
    fn new(source: VerifiedTensorSource, matrix: VerifiedMatrix) -> Result<Self> {
        let layout = rows::Rows::new(matrix.rows(), matrix.columns())?;
        if source.source_identity() != matrix.source_identity()
            || matrix.tensor().kind != QuantKind::Bf16
            || matrix.axes() != MatrixAxes::OutputInput
            || matrix.tensor().byte_len != layout.bytes() as u64
        {
            return Err(Error::State("Gemma tile actual BF16 matrix source".into()));
        }
        Ok(Self {
            source,
            matrix,
            layout,
        })
    }
    pub fn matrix(&self) -> &VerifiedMatrix {
        &self.matrix
    }
    pub fn count(&self) -> usize {
        self.layout.count()
    }
    pub fn tile(&self, ordinal: usize) -> Result<Gemma4SourceTile<'_>> {
        Ok(Gemma4SourceTile {
            tiles: self,
            rows: self.layout.tile(ordinal)?,
        })
    }
    pub fn selected_row(&self, id: u32) -> Result<Gemma4SourceTile<'_>> {
        Ok(Gemma4SourceTile {
            tiles: self,
            rows: self.layout.selected(id)?,
        })
    }
}
/// Borrowed authenticated source range. Read bytes retain their original Window owner.
pub struct Gemma4SourceTile<'a> {
    tiles: &'a Gemma4SourceTiles,
    rows: std::ops::Range<usize>,
}
impl Gemma4SourceTile<'_> {
    pub fn rows(&self) -> std::ops::Range<usize> {
        self.rows.clone()
    }
    pub fn bytes(&self) -> usize {
        self.rows.len() * self.tiles.layout.row_bytes()
    }
    pub fn source_offset(&self) -> Result<u64> {
        self.tiles
            .matrix
            .tensor()
            .byte_offset
            .checked_add((self.rows.start * self.tiles.layout.row_bytes()) as u64)
            .ok_or_else(|| Error::Limit("Gemma source row offset overflow".into()))
    }
    pub fn read(&self, cancelled: Option<&AtomicBool>) -> Result<TensorWindow> {
        cancelled_check(cancelled)?;
        let window = match cancelled {
            Some(c) => {
                self.tiles
                    .source
                    .read_rows_cancellable(&self.tiles.matrix, self.rows.clone(), c)?
            }
            None => self
                .tiles
                .source
                .read_rows(&self.tiles.matrix, self.rows.clone())?,
        };
        if window.matrix().source_identity() != self.tiles.source.source_identity()
            || window.row_range() != self.rows
            || window.source_offset() != self.source_offset()?
            || window.as_slice().len() != self.bytes()
        {
            return Err(Error::State(
                "Gemma exact authenticated full-K row window".into(),
            ));
        }
        Ok(window)
    }
}
