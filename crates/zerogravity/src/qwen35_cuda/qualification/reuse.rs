use super::*;
use std::collections::BTreeSet;
use zerogravity_core::device_events::BufferLeaseId;
/// Same accepted1→1 and64→2→1 coordinates on two independent Session pairs.
/// Expected archives remain callback-only after actual original completion.
/// No claim for other lengths, repeated decode beyond2/67 or concurrent GPU math.
pub async fn qualify_natural_reuse<F>(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [QwenPlan; 5],
    inputs: SourceModelInputs,
    rotary: Option<RotaryOutcome>,
    cancelled: Option<&AtomicBool>,
    mut validate: F,
) -> Result<NaturalReuseReceipt>
where
    F: FnMut(SourceModelReadback<'_>) -> Result<()>,
{
    not_cancelled(cancelled)?;
    let _controls = source.source().reserve_workspace(2 * 1024 * 1024)?;
    let hot = PreparedScopes::new(source.clone(), records)?;
    let prepared =
        prepare_reusable_natural_cold(device, hot.cold_scopes()?, inputs, rotary, cancelled)
            .await?;
    let model = prepared.model;
    let storage = prepared.storage;
    let observer = storage.observer()?;
    let mut sessions = Vec::with_capacity(4);
    let mut rounds = Vec::with_capacity(2);
    let mut geometry_ids: [Option<Vec<BufferLeaseId>>; 5] = std::array::from_fn(|_| None);
    let mut packets = BTreeSet::new();
    let mut candidate_ids = BTreeSet::new();
    let mut kept_feedback = Vec::with_capacity(8);
    let mut calls = 0;
    let mut reused_geometries = 0;
    let mut prefix64_emissions = 0;
    let mut native_launches = 0;
    let mut workspace_clears = 0;
    let mut device_copies = 0;
    let mut dispatches = 0;
    for round in 0..2 {
        not_cancelled(cancelled)?;
        let start = sessions.len();
        sessions.push(storage.session(cancelled).await?);
        sessions.push(storage.session(cancelled).await?);
        let initial = [sessions[start].binding(), sessions[start + 1].binding()];
        if initial.iter().any(|r| {
            r.source() != source.source().source_identity()
                || r.frontier() != 0
                || r.root() != 0
                || r.session_epoch() != 0
        }) || sessions.iter().enumerate().any(|(i, a)| {
            sessions[..i]
                .iter()
                .any(|b| a.binding().session() == b.binding().session())
        }) {
            return Err(Error::State(
                "reusable natural requires four independent original Sessions".into(),
            ));
        }
        let mut feedback: [Option<Arc<CommittedBf16Feedback>>; 2] = [None, None];
        let mut current = initial;
        let mut short = Vec::with_capacity(2);
        let mut long = Vec::with_capacity(2);
        let mut roots = Vec::with_capacity(5);
        for slot in NaturalSlot::ALL {
            let h = usize::from(slot.index() >= 2);
            let s = start + h;
            storage.require_open(&sessions[s])?;
            let before = sessions[s].binding();
            let mut prepared = if slot.feedback() {
                let packet = feedback[h]
                    .as_ref()
                    .ok_or_else(|| Error::State("natural actual feedback missing".into()))?;
                let request = packet.request(&sessions[s])?;
                let record = hot.bind(slot, Some(&request))?;
                storage
                    .prepare_feedback(
                        &sessions[s],
                        record,
                        slot,
                        request,
                        &mut validate,
                        cancelled,
                    )
                    .await?
            } else {
                let record = hot.bind(slot, None)?;
                storage
                    .prepare_prompt(&sessions[s], record, slot, &mut validate, cancelled)
                    .await?
            };
            let counts = (
                prepared.native_launches,
                prepared.workspace_clears,
                prepared.device_copies,
            );
            for id in prepared
                .candidate_ids
                .take()
                .ok_or_else(|| Error::State("reusable candidate IDs absent".into()))?
            {
                if !candidate_ids.insert(id) {
                    return Err(Error::State(
                        "reusable natural candidate was reused across calls".into(),
                    ));
                }
            }
            let ids = prepared
                .value_ids
                .take()
                .ok_or_else(|| Error::State("reusable scratch IDs absent".into()))?;
            if round == 0 {
                geometry_ids[slot.index()] = Some(ids);
            } else if geometry_ids[slot.index()].as_ref() == Some(&ids) {
                reused_geometries += 1;
            } else {
                return Err(Error::State(
                    "reusable call replaced the original geometry scratch".into(),
                ));
            }
            if !packets.insert(
                prepared
                    .packet_id
                    .ok_or_else(|| Error::State("reusable packet ID absent".into()))?,
            ) {
                return Err(Error::State(
                    "reusable immutable packet allocation was reused".into(),
                ));
            }
            if sessions[s].binding() != before {
                return Err(Error::State(
                    "reusable physical completion published early".into(),
                ));
            }
            let committed = prepared.commit(&mut sessions[s], cancelled)?;
            let after = committed.root();
            if after.session() != before.session()
                || after.session_epoch() != before.session_epoch()
                || after.root() != before.root() + 1
                || after.frontier() != (slot.base() + slot.rows()) as u64
                || sessions[start + 1 - h].binding() != current[1 - h]
            {
                return Err(Error::State(
                    "reusable original roots crossed request boundaries".into(),
                ));
            }
            if let Some(next) = committed.feedback() {
                let (id, _) = next.selected();
                if h == 0 {
                    short.push(id);
                } else {
                    long.push(id);
                }
                feedback[h] = Some(next.clone());
                kept_feedback.push(next.clone());
            } else if slot.emits() {
                return Err(Error::State(
                    "reusable emitted phase lacks actual packet".into(),
                ));
            }
            if slot == NaturalSlot::Prefix64 {
                prefix64_emissions += usize::from(committed.readback().device_token.is_some());
                if committed.feedback().is_some() || prefix64_emissions != 0 {
                    return Err(Error::State("reusable prefix64 emission".into()));
                }
            }
            native_launches += counts.0;
            workspace_clears += counts.1;
            device_copies += counts.2;
            dispatches += committed.readback().dispatches;
            roots.push(after);
            current[h] = after;
            calls += 1;
        }
        if short.len() != 2
            || long.len() != 2
            || sessions[start].binding().frontier() != 2
            || sessions[start + 1].binding().frontier() != 67
        {
            return Err(Error::State(
                "reusable complete original request pair".into(),
            ));
        }
        rounds.push(NaturalReuseRound {
            roots: [sessions[start].binding(), sessions[start + 1].binding()],
            phase_roots: roots,
            short_tokens: [short[0], short[1]],
            long_tokens: [long[0], long[1]],
        });
    }
    if calls != 10
        || reused_geometries != 5
        || packets.len() != 10
        || candidate_ids.len() != 480
        || kept_feedback.len() != 8
    {
        return Err(Error::State(
            "reusable actual storage/state/packet coverage".into(),
        ));
    }
    storage.verify_packets(&kept_feedback, cancelled).await?;
    let (source_weight_windows, tied_weight_windows) =
        storage.verify_source_and_retire(cancelled).await?;
    drop(kept_feedback);
    drop(sessions);
    drop(storage);
    drop(model);
    let (final_resident_bytes, final_physical_reserved_bytes) = observer.finish()?;
    Ok(NaturalReuseReceipt {
        rounds,
        calls,
        reused_geometries,
        distinct_packets: packets.len(),
        preserved_committed_packets: 8,
        prefix64_emissions,
        native_launches,
        workspace_clears,
        device_copies,
        dispatches,
        source_weight_windows,
        tied_weight_windows,
        final_resident_bytes,
        final_physical_reserved_bytes,
    })
}
