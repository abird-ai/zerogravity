//! Actual original CPU/B1/nativeB2/graphB2 qualification; ROOT executes only.
use super::*;

pub type Domains = Vec<(StateSlot, Vec<f32>)>;
pub const IDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];

pub async fn states(sessions: &[ResidentSession]) -> Result<Vec<Domains>> {
    let mut values = Vec::with_capacity(sessions.len());
    for session in sessions {
        values.push(session.snapshot().read_domains().await?);
    }
    Ok(values)
}
pub fn roots(sessions: &[ResidentSession]) -> Vec<RootBinding> {
    sessions.iter().map(ResidentSession::binding).collect()
}
pub async fn unchanged(
    sessions: &[ResidentSession],
    bindings: &[RootBinding],
    expected: &[Domains],
) -> Result<()> {
    if roots(sessions) != bindings || sessions.len() != expected.len() {
        return Err(Error::State("graph refusal changed root/row count".into()));
    }
    for (s, domains) in sessions.iter().zip(expected) {
        compare_domains(domains, &s.snapshot().read_domains().await?)?;
    }
    Ok(())
}
pub fn idle(model: &ResidentModel, bytes: usize) -> Result<()> {
    let queue = model.service.queue().stats()?;
    if model.service.pending_events()? != 0
        || model.service.resident_bytes()? != bytes
        || queue.active
        || queue.queued != 0
    {
        return Err(Error::State(
            "graph idle retained event/candidate/queue item".into(),
        ));
    }
    model.service.ensure_usable()
}
