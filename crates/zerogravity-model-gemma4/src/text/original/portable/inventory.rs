//! Family source/plan resource derivation; physical storage is backend-owned.
use super::*;

pub(super) fn derive(
    model: &OriginalGemma4Text,
    plan: &ForwardPlan,
    call: &Bf16TextCall,
    matrices: &BTreeMap<String, Gemma4SourceTiles>,
) -> Result<TextPlanInventory> {
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
    let observations = TextObservation::for_plan(plan, PAGE)?;
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
            let t = matrices
                .get(&tensor.name)
                .ok_or_else(|| Error::State("Gemma source projection tile recipe absent".into()))?;
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
    Ok(TextPlanInventory {
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

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error::Limit("Gemma prepared resource sum overflow".into()))
}
fn times(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Limit("Gemma prepared resource product overflow".into()))
}
