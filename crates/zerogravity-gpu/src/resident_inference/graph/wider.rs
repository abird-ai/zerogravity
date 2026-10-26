//! NEXT source-only qualification: typed StateOnly/Logits bodies, B2/B4/B8.
//! ROOT must first pass the frozen original B2/logits prototype physically.
use super::qualification::{idle, roots, states, unchanged};
use super::*;

pub const IDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];

pub fn selected(logits: &[f32]) -> Result<(u32, u32)> {
    if logits.len() != 248_320 || logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::State("wide graph CPU fullV selection".into()));
    }
    let mut best = 0;
    for (at, &value) in logits.iter().enumerate().skip(1) {
        if value > logits[best] {
            best = at;
        }
    }
    Ok((best as u32, logits[best].to_bits()))
}
#[cfg(feature = "hardware-abort-probe")]
pub fn body(readout: Readout) -> ResidentOutput {
    match readout {
        Readout::None => ResidentOutput::StateOnly,
        Readout::Logits | Readout::Greedy => ResidentOutput::Logits,
    }
}
#[cfg(feature = "hardware-abort-probe")]
pub fn mode_name(readout: Readout) -> &'static str {
    match readout {
        Readout::None => "state",
        Readout::Logits => "logits",
        Readout::Greedy => "greedy",
    }
}
#[cfg(feature = "hardware-abort-probe")]
pub fn parse_mode(mode: &str) -> Result<Readout> {
    match mode {
        "state" => Ok(Readout::None),
        "logits" => Ok(Readout::Logits),
        "greedy" => Ok(Readout::Greedy),
        _ => Err(Error::Malformed(
            "wide graph mode must be state/logits/greedy".into(),
        )),
    }
}
pub fn rows<'a>(
    sessions: &'a [ResidentSession],
    tokens: &[u32],
    order: &[usize],
) -> Result<Vec<(&'a ResidentSession, u32)>> {
    if sessions.len() != tokens.len() || order.len() != tokens.len() {
        return Err(Error::State("wide graph exact row coverage".into()));
    }
    let mut seen = std::collections::BTreeSet::new();
    order
        .iter()
        .map(|&row| {
            if row >= sessions.len() || !seen.insert(row) {
                return Err(Error::State("wide graph row permutation".into()));
            }
            Ok((&sessions[row], tokens[row]))
        })
        .collect()
}
pub fn ordered_sessions<'a>(
    sessions: &'a mut [ResidentSession],
    order: &[usize],
) -> Result<Vec<&'a mut ResidentSession>> {
    let mut remaining = sessions.iter_mut().map(Some).collect::<Vec<_>>();
    order
        .iter()
        .map(|&row| {
            remaining
                .get_mut(row)
                .and_then(Option::take)
                .ok_or_else(|| Error::State("wide graph mutable row permutation".into()))
        })
        .collect()
}
pub fn packet(
    read: &InferenceReadback,
    session: &ResidentSession,
    expected: (u32, u32),
) -> Result<()> {
    let token = read
        .device_token
        .as_ref()
        .ok_or_else(|| Error::State("wide graph missing real GPU packet".into()))?;
    if token.producer != Some(session.binding())
        || (token.id, token.max_bits) != expected
        || token.lease.bytes() != 12
        || !token.lease.is_initialized()
        || !Arc::ptr_eq(token.lease.service(), &session.model.service)
    {
        return Err(Error::State(
            "wide graph packet/source/root/physical owner".into(),
        ));
    }
    Ok(())
}
pub fn candidate_shape(
    candidate: &PreparedResidentCohort,
    count: usize,
    readout: Readout,
) -> Result<()> {
    if candidate.len() != count
        || candidate.dispatches() == 0
        || candidate.fused_dispatches() == 0
        || candidate.native_greedy_dispatches()
            != if matches!(readout, Readout::Greedy) {
                4
            } else {
                0
            }
    {
        return Err(Error::State(
            "wide graph real cohort/body/sampler receipt".into(),
        ));
    }
    for (row, prepared) in candidate.prepared.iter().enumerate() {
        let read = &prepared.readback;
        let row_bytes = match readout {
            Readout::None => 0,
            Readout::Logits => 248_320 * 4,
            Readout::Greedy => 12,
        };
        if read.bytes != row_bytes + usize::from(row == 0) * 4
            || read.chunk_bytes != read.bytes
            || read.state_only_steps != usize::from(matches!(readout, Readout::None))
            || read.logit_steps != usize::from(!matches!(readout, Readout::None))
            || read.logits.len()
                != if matches!(readout, Readout::Logits) {
                    248_320
                } else {
                    0
                }
            || read.device_token.is_some() != matches!(readout, Readout::Greedy)
        {
            return Err(Error::State(
                "wide graph state-only/logit/packet readback geometry".into(),
            ));
        }
    }
    Ok(())
}

/// Explicit qualifier read of the already Ready, source-owned full logits.
/// Greedy API still returns only12B packets; diagnostic transfer is separate.
pub async fn completed_logits(
    model: &Arc<ResidentModel>,
    root: u64,
    count: usize,
) -> Result<Vec<f32>> {
    let bytes = count
        .checked_mul(248_320 * 4)
        .ok_or_else(|| Error::Limit("wide graph diagnostic logits".into()))?;
    let ticket = model.service.queue().submit(
        model.service.budget(),
        model.service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: (bytes + 4) as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    let workspace = model.workspace.lock().await;
    let batch = workspace
        .batch
        .as_ref()
        .ok_or_else(|| Error::State("wide graph diagnostic batch absent".into()))?;
    if batch.rows != count {
        return Err(Error::State("wide graph diagnostic row band".into()));
    }
    let logits = batch
        .program
        .logits()
        .ok_or_else(|| Error::State("wide graph diagnostic planned logits absent".into()))?;
    if logits.elements != bytes / 4 {
        return Err(Error::State("wide graph diagnostic fullV geometry".into()));
    }
    let status = lease_for(&batch.leases, &batch.program.status())?.clone();
    let logits = lease_for(&batch.leases, &logits.handle)?.clone();
    let accesses = [
        (status.clone(), DeviceAccess::Read),
        (logits.clone(), DeviceAccess::Read),
    ];
    slot.begin_submission()?;
    let flight = model.service.submit(root, &accesses)?;
    let read = flight
        .complete_read_only_readback(
            &[status.readback(0, 4)?, logits.readback(0, bytes)?],
            |values| {
                if values.len() != 2
                    || !all_zero(&values[0])
                    || words(&values[1])?.iter().any(|x| !x.is_finite())
                {
                    return Err(Error::State(
                        "wide graph diagnostic fullV status/finite".into(),
                    ));
                }
                Ok(())
            },
        )
        .await?;
    slot.finish()?;
    words(&read[1])
}

pub async fn pressure_and_abandon(
    model: &Arc<ResidentModel>,
    sessions: &mut [ResidentSession],
    graph: &Arc<CapturedResidentGraph>,
    readout: Readout,
) -> Result<()> {
    let count = sessions.len();
    let tokens = (0..count).map(|r| IDS[(r + 3) % 8]).collect::<Vec<_>>();
    let order = (0..count).collect::<Vec<_>>();
    let rows = rows(sessions, &tokens, &order)?;
    let bindings = roots(sessions);
    let prior = states(sessions).await?;
    let bytes = model.service.resident_bytes()?;
    let enqueues = graph.replay_enqueues();
    let required = model
        .state_bytes
        .checked_mul(count)
        .and_then(|n| {
            n.checked_add(if matches!(readout, Readout::Greedy) {
                12 * count
            } else {
                0
            })
        })
        .ok_or_else(|| Error::Limit("wide graph pressure footprint".into()))?;
    let fill = model
        .service
        .remaining_capacity()?
        .checked_sub(required)
        .and_then(|n| n.checked_add(1))
        .filter(|&n| n > 0)
        .ok_or_else(|| Error::Limit("wide graph qualifier pressure headroom".into()))?;
    let filler = model
        .service
        .allocate_uninitialized(fill, BufferRole::Scratch)?;
    let queue = model.service.queue().stats()?;
    let refused = model
        .prepare_fused_execution(
            &rows,
            None,
            readout,
            super::super::fused::FusedExecution::Graph(graph.clone()),
        )
        .await?;
    if refused.is_some()
        || graph.replay_enqueues() != enqueues
        || model.service.queue().stats()? != queue
        || model.service.resident_bytes()? != bytes + fill
        || model.service.pending_events()? != 0
    {
        return Err(Error::State(
            "wide graph capacity refusal was not prelaunch None".into(),
        ));
    }
    drop(filler);
    idle(model, bytes)?;
    unchanged(sessions, &bindings, &prior).await?;
    let discarded = model.prepare_graph_readout(&rows, graph, readout).await?;
    candidate_shape(&discarded, count, readout)?;
    if graph.replay_enqueues() != enqueues + 1 {
        return Err(Error::State(
            "wide graph Ready cancellation actual enqueue".into(),
        ));
    }
    drop(discarded);
    unchanged(sessions, &bindings, &prior).await?;
    idle(model, bytes)?;
    // Actual Ready proposal loses every sibling's publication when one row
    // restores its own original snapshot (same frontier/state, new epoch).
    let stale = model.prepare_graph_readout(&rows, graph, readout).await?;
    drop(rows);
    candidate_shape(&stale, count, readout)?;
    let snapshot = sessions[count - 1].snapshot();
    sessions[count - 1].restore(&snapshot)?;
    let changed = roots(sessions);
    if changed[..count - 1] != bindings[..count - 1]
        || changed[count - 1] == bindings[count - 1]
        || changed[count - 1].frontier() != bindings[count - 1].frontier()
        || stale
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
    {
        return Err(Error::State(
            "wide graph stale per-row epoch published cohort".into(),
        ));
    }
    drop(snapshot);
    unchanged(sessions, &changed, &prior).await?;
    if graph.replay_enqueues() != enqueues + 2 {
        return Err(Error::State(
            "wide graph abandoned/stale actual enqueue count".into(),
        ));
    }
    idle(model, bytes)
}
