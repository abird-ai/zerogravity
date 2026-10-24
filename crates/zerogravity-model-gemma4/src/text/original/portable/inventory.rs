//! Actual source/plan layout; missing lowerer allocations are not a fit claim.
use super::*;

#[derive(Clone, Debug)]
pub struct Gemma4Bf16Inventory {
    pub unique_source_matrices: usize,
    pub full_source_matrix_bytes: usize,
    pub retained_source_parameter_bytes: usize,
    pub recorded_value_bytes: usize,
    pub kv_root_bytes: usize,
    /// Two current roots, two proposals and two retained snapshots.
    pub six_root_payload_bytes: usize,
    pub source_page_bytes: usize,
    pub source_page_host_staging_bytes: usize,
    pub observation_frames: usize,
    pub largest_observation_bytes: usize,
    /// One device value remains live until its actual D2H checkpoint finishes.
    pub observation_host_upper_bytes: usize,
    pub known_host_upper_bytes: usize,
    pub projection_pages_with_head: usize,
    pub projection_pages_state_only: usize,
}
impl Gemma4Bf16Inventory {
    pub(super) fn derive(
        model: &OriginalGemma4Text,
        plan: &ForwardPlan,
        call: &Bf16TextCall,
        matrices: &BTreeMap<String, Gemma4SourceTiles>,
    ) -> Result<Self> {
        let matrix_bytes = matrices.values().try_fold(0usize, |sum, t| {
            add(sum, t.matrix.tensor().byte_len as usize)
        })?;
        let parameters = model
            .parameters
            .values()
            .try_fold(0usize, |sum, p| add(sum, p.descriptor.bytes))?;
        let values = plan
            .values()
            .iter()
            .try_fold(0usize, |sum, s| add(sum, times(s.capacity_elements()?, 4)?))?;
        let widths = model
            .geometry
            .layers
            .iter()
            .enumerate()
            .filter(|(i, s)| s.kv_source_layer == *i)
            .try_fold(0usize, |sum, (_, s)| {
                add(sum, times(s.kv_heads, s.head_dim)?)
            })?;
        let kv = times(times(2 * 64, widths)?, 4)?;
        let observations = observations(plan)?;
        let mut largest = 0;
        for observation in &observations {
            for index in 0..call.rows() {
                largest = largest.max(observation.bytes(&call.row(index)?)?);
            }
        }
        let mut pages = 0;
        let mut state_pages = 0;
        for (index, node) in plan.nodes().iter().enumerate() {
            if let PlanOperation::Project { tensor } = &node.operation {
                let t = matrices.get(&tensor.name).ok_or_else(|| {
                    Error::State("Gemma source projection tile recipe absent".into())
                })?;
                pages = add(pages, t.count())?;
                if index < plan.state_nodes().len() {
                    state_pages = add(state_pages, t.count())?;
                }
            }
        }
        // Existing whole CPU layout conservatively includes source/parameter/decoder/
        // traversal owners, even though a future GPU lowerer may not retain all of them.
        let retained_parameters = model
            .parameters
            .values()
            .try_fold(0usize, |sum, p| add(sum, times(p.values.len(), 4)?))?;
        let baseline = super::super::resources::layout(
            &model.geometry,
            model.source().metadata_reserved_bytes(),
            retained_parameters as u64,
        )?;
        let baseline = usize::try_from(baseline.total_upper_bytes)
            .map_err(|_| Error::Limit("Gemma host baseline conversion".into()))?;
        let observation_host = times(largest, 3)?; // actual D2H, expected window, decoding/compare scratch
        let known_host = add(add(add(baseline, METADATA)?, 3 * PAGE)?, observation_host)?;
        if known_host > 128 * 1024 * 1024 {
            return Err(Error::Limit(
                "Gemma known source/streaming-observation layout exceeds128MiB".into(),
            ));
        }
        Ok(Self {
            unique_source_matrices: matrices.len(),
            full_source_matrix_bytes: matrix_bytes,
            retained_source_parameter_bytes: parameters,
            recorded_value_bytes: values,
            kv_root_bytes: kv,
            six_root_payload_bytes: times(kv, 6)?,
            source_page_bytes: PAGE,
            source_page_host_staging_bytes: 3 * PAGE,
            observation_frames: observations.len(),
            largest_observation_bytes: largest,
            observation_host_upper_bytes: observation_host,
            known_host_upper_bytes: known_host,
            projection_pages_with_head: pages,
            projection_pages_state_only: state_pages,
        })
    }
    /// Planning arithmetic only; callers must price actual shared lowerer/client owners.
    pub fn host_with_lowerers(&self, actual_additional_bytes: usize) -> Result<usize> {
        // Derived cost only. The engine compares against its genuine source
        // host contract and admits actual owners; no family backend limit here.
        add(self.known_host_upper_bytes, actual_additional_bytes)
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error::Limit("Gemma prepared resource sum overflow".into()))
}
fn times(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Limit("Gemma prepared resource product overflow".into()))
}

/// A single completed producer frame. No late lease read or device completion proof.
#[derive(Clone, Debug)]
pub struct Gemma4Bf16Observation {
    pub node: usize,
    pub value: ValueId,
    pub layer: usize,
    pub stage: &'static str,
    shape: ValueShape,
    pub before_frontier: bool,
    pub allows_attention_mask: bool,
}
impl Gemma4Bf16Observation {
    pub fn bytes(&self, row: &Bf16TextRow<'_>) -> Result<usize> {
        let elements = match self.shape {
            ValueShape::CausalRows { heads, capacity } => {
                if row.valid_tokens() > capacity {
                    return Err(Error::State(
                        "Gemma actual observation row exceeds recorded capacity".into(),
                    ));
                }
                times(heads, row.valid_tokens())?
            }
            shape => shape.capacity_elements()?,
        };
        let bytes = times(elements, 4)?;
        if bytes == 0 || bytes > PAGE {
            return Err(Error::Limit(
                "Gemma single actual producer frame exceeds1MiB".into(),
            ));
        }
        Ok(bytes)
    }
}
pub(super) fn observations(plan: &ForwardPlan) -> Result<Vec<Gemma4Bf16Observation>> {
    let mut result = Vec::new();
    let masked = plan
        .nodes()
        .iter()
        .filter(|n| matches!(n.operation, PlanOperation::TextWindowQk { .. }))
        .flat_map(|n| n.outputs.iter().copied())
        .map(|v| v.0)
        .collect::<std::collections::BTreeSet<_>>();
    for (index, node) in plan.nodes().iter().enumerate() {
        if let PlanOperation::Observe { layer, stage } = node.operation {
            let [value] = node.inputs.as_slice() else {
                return Err(Error::State("Gemma observer exact producer arity".into()));
            };
            let shape = *plan
                .values()
                .get(value.0 as usize)
                .ok_or_else(|| Error::State("Gemma observer producer extent absent".into()))?;
            result.push(Gemma4Bf16Observation {
                node: index,
                value: *value,
                layer,
                stage,
                shape,
                before_frontier: index < plan.state_nodes().len(),
                allows_attention_mask: masked.contains(&value.0),
            });
        }
    }
    Ok(result)
}
impl Gemma4Bf16TextPlan {
    /// Execute each read/borrowed comparison at its real same-flight checkpoint,
    /// before any storage reuse. A checkpoint cannot publish or finish the root.
    pub fn observations(&self) -> Result<Vec<Gemma4Bf16Observation>> {
        observations(&self.plan)
    }
    pub fn observations_for_row(&self, index: usize) -> Result<Vec<Gemma4Bf16Observation>> {
        let row = self.call.row(index)?;
        Ok(observations(&self.plan)?
            .into_iter()
            .filter(|o| o.before_frontier || row.needs_logits())
            .collect())
    }
}
