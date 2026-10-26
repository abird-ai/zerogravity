use super::*;
/// The supplied records establish cold storage geometry only. Their cached
/// teacher IDs are NEVER natural decode inputs: only actual committed private
/// BF16 packets may mint the subsequent one-token source requests.
/// Expected archives are opened only by `validate` after genuine arithmetic.
pub async fn qualify_natural_model<F>(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [QwenPlan; 5],
    inputs: SourceModelInputs,
    rotary: Option<RotaryOutcome>,
    cancelled: Option<&AtomicBool>,
    mut validate: F,
) -> Result<NaturalReceipt>
where
    F: FnMut(SourceModelReadback<'_>) -> Result<()>,
{
    // Wins even when no rotary producer or provider exists. No fake prefix.
    not_cancelled(cancelled)?;
    let _controls = source.source().reserve_workspace(1024 * 1024)?;
    let hot = PreparedScopes::new(source.clone(), records)?;
    let cold = prepare_natural_cold(device, hot.cold_scopes()?, inputs, rotary, cancelled).await?;
    let observer = cold.observer();
    let model = cold.model;
    let mut sessions = cold.sessions;
    let slots = cold.slots;
    let state_pool = cold.state_pool;
    let initial = [sessions[0].binding(), sessions[1].binding()];
    if initial.iter().any(|root| {
        root.source() != source.source().source_identity()
            || root.frontier() != 0
            || root.root() != 0
            || root.session_epoch() != 0
    }) || initial[0].session() == initial[1].session()
    {
        return Err(Error::State(
            "natural two genuine independent original sessions".into(),
        ));
    }
    let mut feedback: [Option<Arc<CommittedBf16Feedback>>; 2] = [None, None];
    let mut current = initial;
    let mut short = Vec::with_capacity(2);
    let mut long = Vec::with_capacity(2);
    let mut phase_roots = Vec::with_capacity(5);
    let mut dispatches = 0usize;
    let mut native_launches = 0usize;
    let mut workspace_clears = 0usize;
    let mut device_copies = 0usize;
    let mut prefix64_emissions = 0usize;
    for slot in NaturalSlot::ALL {
        not_cancelled(cancelled)?;
        let h = usize::from(slot.index() >= 2);
        let before = sessions[h].binding();
        let prepared = if slot.feedback() {
            let packet = feedback[h]
                .as_ref()
                .ok_or_else(|| Error::State("natural actual feedback missing".into()))?;
            let request = packet.request(&sessions[h])?;
            let record = hot.bind(slot, Some(&request))?;
            slots
                .prepare_feedback(
                    &sessions[h],
                    record,
                    slot,
                    request,
                    &state_pool,
                    &mut validate,
                    cancelled,
                )
                .await?
        } else {
            let record = hot.bind(slot, None)?;
            slots
                .prepare_prompt(
                    &sessions[h],
                    record,
                    slot,
                    &state_pool,
                    &mut validate,
                    cancelled,
                )
                .await?
        };
        let counts = (
            prepared.native_launches,
            prepared.workspace_clears,
            prepared.device_copies,
        );
        // The physical result is still private; only original commit publishes.
        if sessions[h].binding() != before {
            return Err(Error::State(
                "natural physical completion published before original commit".into(),
            ));
        }
        let committed = prepared.commit(&mut sessions[h], cancelled)?;
        let after = committed.root();
        if after.root() != before.root() + 1
            || after.frontier() != (slot.base() + slot.rows()) as u64
            || after.session() != before.session()
            || after.session_epoch() != before.session_epoch()
            || sessions[1 - h].binding() != current[1 - h]
        {
            return Err(Error::State(
                "natural original current-root progression differs".into(),
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
        } else if slot.emits() {
            return Err(Error::State(
                "natural emitted phase omitted actual committed packet".into(),
            ));
        }
        if slot == NaturalSlot::Prefix64 {
            prefix64_emissions += usize::from(committed.readback().device_token.is_some());
            if committed.feedback().is_some() || prefix64_emissions != 0 {
                return Err(Error::State(
                    "natural prompt prefix must publish state without token emission".into(),
                ));
            }
        }
        dispatches = dispatches
            .checked_add(committed.readback().dispatches)
            .ok_or_else(|| Error::Limit("natural dispatch count overflow".into()))?;
        native_launches = native_launches
            .checked_add(counts.0)
            .ok_or_else(|| Error::Limit("natural native count overflow".into()))?;
        workspace_clears = workspace_clears
            .checked_add(counts.1)
            .ok_or_else(|| Error::Limit("natural clear count overflow".into()))?;
        device_copies = device_copies
            .checked_add(counts.2)
            .ok_or_else(|| Error::Limit("natural copy count overflow".into()))?;
        phase_roots.push(after);
        current[h] = after;
        drop(committed);
    }
    if short.len() != 2
        || long.len() != 2
        || phase_roots.len() != 5
        || prefix64_emissions != 0
        || sessions[0].binding().frontier() != 2
        || sessions[1].binding().frontier() != 67
    {
        return Err(Error::State(
            "natural complete finite publication/emission coverage".into(),
        ));
    }
    let roots = [sessions[0].binding(), sessions[1].binding()];
    let (source_weight_windows, tied_weight_windows) =
        slots.verify_source_and_retire(&sessions, cancelled).await?;
    let short_tokens = [short[0], short[1]];
    let long_tokens = [long[0], long[1]];
    drop(feedback);
    drop(state_pool);
    drop(slots);
    drop(sessions);
    drop(model);
    let (final_resident_bytes, final_physical_reserved_bytes) = observer.finish()?;
    Ok(NaturalReceipt {
        phases: 5,
        roots,
        phase_roots,
        short_tokens,
        long_tokens,
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
