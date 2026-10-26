//! P12 full-model native batch proof. CPU/B1 outputs remain host oracles.
use super::*;

pub type DomainValues = Vec<(StateSlot, Vec<f32>)>;

pub const IDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];

pub fn bindings(sessions: &[ResidentSession]) -> Vec<RootBinding> {
    sessions.iter().map(ResidentSession::binding).collect()
}

pub fn unchanged(sessions: &[ResidentSession], before: &[RootBinding]) -> Result<()> {
    if bindings(sessions) != before {
        return Err(Error::State(
            "unpublished fused batch changed a committed session root".into(),
        ));
    }
    Ok(())
}

pub async fn domain_values(sessions: &[ResidentSession]) -> Result<Vec<DomainValues>> {
    let mut output = Vec::with_capacity(sessions.len());
    for session in sessions {
        output.push(session.snapshot().read_domains().await?);
    }
    Ok(output)
}

pub async fn unchanged_domains(
    sessions: &[ResidentSession],
    before: &[DomainValues],
) -> Result<()> {
    if sessions.len() != before.len() {
        return Err(Error::State("fused rollback state row coverage".into()));
    }
    for (session, expected) in sessions.iter().zip(before) {
        compare_domains(expected, &session.snapshot().read_domains().await?)?;
    }
    Ok(())
}

pub async fn prepare(
    resident: &Arc<ResidentModel>,
    sessions: &[ResidentSession],
    tokens: &[u32],
) -> Result<PreparedResidentCohort> {
    if sessions.len() != tokens.len() {
        return Err(Error::State("fused qualifier token/row coverage".into()));
    }
    let rows = sessions
        .iter()
        .zip(tokens)
        .map(|(session, &token)| (session, token))
        .collect::<Vec<_>>();
    resident.prepare_logits_fused_cohort(&rows).await
}

pub fn receipt(candidate: &PreparedResidentCohort, rows: usize, frontier: u64) -> Result<()> {
    if candidate.len() != rows
        || candidate.prior_frontier() != Some(frontier)
        || candidate.dispatches() == 0
        || candidate.fused_dispatches() == 0
        || candidate.fused_dispatches() > candidate.dispatches()
    {
        return Err(Error::State(
            "fused cohort lacks native multirow physical dispatch evidence".into(),
        ));
    }
    Ok(())
}

pub fn idle(resident: &ResidentModel, bytes: usize) -> Result<()> {
    if resident.service.pending_events()? != 0 || resident.service.resident_bytes()? != bytes {
        return Err(Error::State(
            "completed fused cohort retained candidate leases or pending events".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
pub async fn physical_prior_after_abort(
    client: &cubecl::client::Client,
    snapshot: &ResidentSnapshot,
    expected: &DomainValues,
) -> Result<()> {
    if snapshot.domains.len() != expected.len() {
        return Err(Error::State(
            "fused abort physical prior domain coverage".into(),
        ));
    }
    for (domain, (expected_slot, expected_values)) in snapshot.domains.iter().zip(expected) {
        if domain.slot != *expected_slot {
            return Err(Error::State(
                "fused abort physical prior slot ordering".into(),
            ));
        }
        let handles = domain.leases.iter().map(ResidentLease::handle).collect();
        let data = client
            .read_async(handles)
            .await
            .map_err(|error| Error::State(format!("fused abort diagnostic prior read: {error}")))?;
        if data.len() != domain.leases.len()
            || data
                .iter()
                .zip(&domain.leases)
                .any(|(bytes, lease)| bytes.len() != lease.bytes())
        {
            return Err(Error::State(
                "fused abort full-capacity prior read incomplete".into(),
            ));
        }
        let mut actual = Vec::new();
        for (lease, bytes) in domain.leases.iter().zip(data) {
            let values = words(&bytes)?;
            if matches!(&domain.buffers, StateBuffers::Kv { .. }) {
                let active =
                    lease.bytes() / 4 / snapshot.capacity * snapshot.binding.frontier() as usize;
                if values[active..].iter().any(|value| value.to_bits() != 0) {
                    return Err(Error::State(
                        "fused abort changed canonical prior KV tail".into(),
                    ));
                }
                actual.extend_from_slice(&values[..active]);
            } else {
                actual.extend(values);
            }
        }
        exact(
            expected_values,
            &actual,
            "fused abort immutable physical prior",
        )?;
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
#[path = "resident_inference/prefix_qualification.rs"]
mod prefix_qualification;
