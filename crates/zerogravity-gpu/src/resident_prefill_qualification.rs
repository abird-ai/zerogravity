//! H17 physical qualification of the shared plan's state-only prefix.
//! CPU results are diagnostic oracles only; no expected answer is uploaded.
use super::*;

pub fn publication(before: RootBinding, after: RootBinding, tokens: u64) -> Result<()> {
    if before.source() != after.source()
        || before.session() != after.session()
        || before.root().checked_add(1) != Some(after.root())
        || before.frontier().checked_add(tokens) != Some(after.frontier())
        || before.session_epoch() != after.session_epoch()
    {
        return Err(Error::State(
            "H17 chunk did not publish one root at the exact frontier".into(),
        ));
    }
    Ok(())
}
pub fn restoration(before: RootBinding, after: RootBinding, saved: RootBinding) -> Result<()> {
    if before.source() != after.source()
        || before.session() != after.session()
        || before.root().checked_add(1) != Some(after.root())
        || before.session_epoch().checked_add(1) != Some(after.session_epoch())
        || after.frontier() != saved.frontier()
    {
        return Err(Error::State(
            "H17 restore root/epoch/frontier mismatch".into(),
        ));
    }
    Ok(())
}
pub fn observations(
    expected: &[LayerObservation],
    actual: &[LayerObservation],
    state_only: bool,
) -> Result<usize> {
    let expected = expected
        .iter()
        .filter(|o| !state_only || o.stage != "final_norm")
        .collect::<Vec<_>>();
    if expected.len() != actual.len() {
        return Err(Error::State(format!(
            "H17 trace count {} != {}",
            expected.len(),
            actual.len()
        )));
    }
    let mut compared = 0;
    for (e, a) in expected.into_iter().zip(actual) {
        if e.position != a.position || e.layer != a.layer || e.stage != a.stage {
            return Err(Error::State("H17 ordered trace metadata differs".into()));
        }
        exact(
            &e.values,
            &a.values,
            &format!("H17 layer {} {}", e.layer, e.stage),
        )?;
        compared += e.values.len();
    }
    Ok(compared)
}
pub fn state_receipt(
    read: &InferenceReadback,
    steps: usize,
    dispatches: usize,
    diagnostic: bool,
) -> Result<()> {
    if !read.logits.is_empty()
        || read.device_token.is_some()
        || read.state_only_steps != steps
        || read.logit_steps != 0
        || read.dispatches != dispatches
        || (!diagnostic
            && (read.bytes != 4
                || read.chunk_bytes != steps * 4
                || !read.domains.is_empty()
                || !read.observations.is_empty()))
    {
        return Err(Error::State(
            "H17 state-only execution/readback receipt differs".into(),
        ));
    }
    Ok(())
}
pub async fn cold_suffix(resident: &Arc<ResidentModel>) -> Result<(usize, usize)> {
    let workspace = resident.workspace.lock().await;
    let plan = workspace.program.plan();
    let prefix = plan.state_nodes().len();
    if prefix == 0 || prefix >= plan.nodes().len() {
        return Err(Error::State(
            "H17 missing state/output plan boundary".into(),
        ));
    }
    let selected = workspace
        .state_accesses
        .iter()
        .map(|(l, _)| l.id())
        .collect::<std::collections::BTreeSet<_>>();
    let mut outputs = 0;
    for node in &plan.nodes()[prefix..] {
        if node.state.is_some() {
            return Err(Error::State(
                "H17 output suffix contains a state effect".into(),
            ));
        }
        for id in &node.outputs {
            let value = &workspace.program.values()[id.0 as usize];
            let lease = lease_for(&workspace.leases, &value.handle)?;
            if lease.is_initialized() || selected.contains(&lease.id()) {
                return Err(Error::State(
                    "H17 state-only work initialized/admitted an output-only value".into(),
                ));
            }
            outputs += 1;
        }
    }
    if outputs == 0 {
        return Err(Error::State(
            "H17 output suffix has no physical values".into(),
        ));
    }
    for handle in workspace
        .greedy
        .as_ref()
        .ok_or_else(|| Error::State("H17 model greedy workspace absent".into()))?
    {
        let lease = lease_for(&workspace.leases, handle)?;
        if lease.is_initialized() || selected.contains(&lease.id()) {
            return Err(Error::State(
                "H17 state-only work initialized/admitted greedy scratch".into(),
            ));
        }
    }
    // This also covers suffix-only quant status leases without exposing the
    // lowerer's private NodeStatus representation to this qualification owner.
    let mut skipped = 0;
    for (lease, access) in &workspace.leases {
        if *access == DeviceAccess::Write && !selected.contains(&lease.id()) {
            if lease.is_initialized() {
                return Err(Error::State(
                    "H17 skipped mutable suffix allocation became Ready".into(),
                ));
            }
            skipped += 1;
        }
    }
    Ok((outputs, skipped))
}
