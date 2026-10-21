//! H17 physical qualification of the shared plan's state-only prefix.
//! CPU results are diagnostic oracles only; no expected answer is uploaded.
use super::*;

fn publication(before: RootBinding, after: RootBinding, tokens: u64) -> Result<()> {
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
fn restoration(before: RootBinding, after: RootBinding, saved: RootBinding) -> Result<()> {
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
fn observations(
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
fn state_receipt(
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
async fn cold_suffix(resident: &Arc<ResidentModel>) -> Result<(usize, usize)> {
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
    for handle in &workspace.greedy {
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

/// Must run immediately after load, before any vocabulary suffix/greedy launch.
pub(super) async fn qualify(
    resident: &Arc<ResidentModel>,
    model: &Arc<LoadedQwenModel>,
) -> Result<()> {
    if resident.service.source_identity() != model.source_identity() || resident.states.len() != 42
    {
        return Err(Error::State(
            "H17 source/state-domain qualification binding".into(),
        ));
    }
    let expected_savings = match model.tensor_info("token_embd.weight")?.kind {
        zerogravity_artifacts::quant::QuantKind::Q8_0 => 17,
        zerogravity_artifacts::quant::QuantKind::Q6K => 15,
        _ => {
            return Err(Error::Unsupported(
                "H17 vocabulary source codec is not qualified".into(),
            ));
        }
    };
    let original_bytes = resident.service.resident_bytes()?;
    let cold_uploads = resident.service.uploaded_value_bytes();
    let (source_signature, source_bytes) = {
        let workspace = resident.workspace.lock().await;
        let signature = workspace
            .leases
            .iter()
            .filter(|(_, a)| *a == DeviceAccess::Read)
            .map(|(l, _)| (l.id(), l.bytes()))
            .collect::<Vec<_>>();
        let bytes = signature.iter().map(|(_, n)| *n).sum::<usize>();
        if bytes != workspace.program.budget().source_bytes || cold_uploads != bytes as u64 {
            return Err(Error::State(
                "H17 requires one cold source upload before session initialization".into(),
            ));
        }
        (signature, bytes)
    };
    let (tail_outputs, skipped_leases) = cold_suffix(resident).await?;
    let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(model.clone())?;
    let mut session = resident.session().await?;
    let warm_uploads = resident.service.uploaded_value_bytes();
    if warm_uploads.checked_sub(cold_uploads) != Some(resident.state_bytes as u64) {
        return Err(Error::State(
            "H17 cold session upload exceeded its zero-state allocation".into(),
        ));
    }
    let initial = session.binding();
    if initial.root() != 0 || initial.frontier() != 0 || initial.session_epoch() != 0 {
        return Err(Error::State("H17 cold session root is not initial".into()));
    }
    let mut state_dispatches = 0;
    let mut compared_trace = 0;
    for token in [11, 353] {
        let (_, draft, trace) = cpu.next_logits_with_layer_trace(token)?;
        if trace.iter().filter(|o| o.stage == "final_norm").count() != 1 {
            return Err(Error::State(
                "H17 CPU full trace lacks its one optional final norm".into(),
            ));
        }
        let before = session.binding();
        let actual = session.prefill_without_logits(&[token], true).await?;
        if state_dispatches == 0 {
            state_dispatches = actual.dispatches;
        }
        if state_dispatches == 0 {
            return Err(Error::State("H17 state prefix launched no GPU work".into()));
        }
        state_receipt(&actual, 1, state_dispatches, true)?;
        if actual.chunk_bytes != actual.bytes {
            return Err(Error::State("H17 single-step readback sum".into()));
        }
        publication(before, session.binding(), 1)?;
        compare_domains(&draft.domain_values(), &actual.domains)?;
        compared_trace += observations(&trace, &actual.observations, true)?;
        cpu.commit(draft)?;
        cold_suffix(resident).await?;
        println!(
            "H17_RESIDENT_STATE_ONLY_TOKEN_PASS token={token} frontier={} dispatches={} final_step_readback_bytes={} logits_absent=true all42_domains=exact ordered_trace=exact skipped_tail_outputs_uninitialized={tail_outputs} skipped_mutable_leases_uninitialized={skipped_leases}",
            session.binding().frontier(),
            actual.dispatches,
            actual.bytes
        );
    }
    let (expected, draft, trace) = cpu.next_logits_with_layer_trace(17)?;
    let before = session.binding();
    let actual = session.forward(&[17], true).await?;
    publication(before, session.binding(), 1)?;
    exact(&expected, &actual.logits, "H17 first full suffix logits")?;
    compare_domains(&draft.domain_values(), &actual.domains)?;
    compared_trace += observations(&trace, &actual.observations, false)?;
    cpu.commit(draft)?;
    let full_dispatches = actual.dispatches;
    if full_dispatches.checked_sub(state_dispatches) != Some(expected_savings)
        || actual.state_only_steps != 0
        || actual.logit_steps != 1
        || actual.device_token.is_some()
    {
        return Err(Error::State(
            "H17 physical vocabulary suffix dispatch saving differs".into(),
        ));
    }
    {
        let workspace = resident.workspace.lock().await;
        for node in
            &workspace.program.plan().nodes()[workspace.program.plan().state_nodes().len()..]
        {
            for id in &node.outputs {
                let value = &workspace.program.values()[id.0 as usize];
                if !lease_for(&workspace.leases, &value.handle)?.is_initialized() {
                    return Err(Error::State(
                        "H17 full graph failed to initialize suffix outputs".into(),
                    ));
                }
            }
        }
    }
    drop(actual);
    let saved = session.snapshot();
    let saved_cpu = cpu.snapshot();
    let (expected_final, expected_trace) =
        cpu.prefill_transaction_with_layer_trace(&[42, 77, 19])?;
    let expected_domains = cpu.snapshot().domain_values();
    let before = session.binding();
    let atomic = session.forward(&[42, 77, 19], true).await?;
    publication(before, session.binding(), 3)?;
    exact(
        &expected_final,
        &atomic.logits,
        "H17 atomic nonzero prefill logits",
    )?;
    compare_domains(&expected_domains, &atomic.domains)?;
    compared_trace += observations(&expected_trace, &atomic.observations, false)?;
    if atomic.dispatches != 2 * state_dispatches + full_dispatches
        || atomic.state_only_steps != 2
        || atomic.logit_steps != 1
        || atomic.chunk_bytes != atomic.bytes + 8
    {
        return Err(Error::State(
            "H17 atomic prefill launched/read optional suffixes".into(),
        ));
    }
    let atomic_dispatches = atomic.dispatches;
    let atomic_bytes = atomic.chunk_bytes;
    let atomic_last_bytes = atomic.bytes;
    drop(atomic);
    println!(
        "H17_RESIDENT_ATOMIC_PREFILL_PASS tokens=3 state_only_steps=2 logit_steps=1 dispatches={atomic_dispatches} skipped_dispatches={} chunk_readback_bytes={atomic_bytes} final_step_readback_bytes={atomic_last_bytes} intermediate_status_readback_bytes=8 root_increment=1 all42_domains=exact last_token_trace=exact",
        2 * expected_savings
    );

    let before = session.binding();
    session.restore(&saved)?;
    restoration(before, session.binding(), saved.binding)?;
    cpu.restore(&saved_cpu)?;
    let before = session.binding();
    let prefix = session.prefill_without_logits(&[42, 77], false).await?;
    state_receipt(&prefix, 2, 2 * state_dispatches, false)?;
    publication(before, session.binding(), 2)?;
    cpu.prefill_chunk_without_logits(&[42, 77])?;
    compare_domains(
        &cpu.snapshot().domain_values(),
        &session.snapshot().read_domains().await?,
    )?;
    let prefix_bytes = prefix.chunk_bytes;
    let prefix_dispatches = prefix.dispatches;
    drop(prefix);
    let (expected, draft, trace) = cpu.next_logits_with_layer_trace(19)?;
    let before = session.binding();
    let segmented = session.forward(&[19], true).await?;
    publication(before, session.binding(), 1)?;
    exact(
        &expected,
        &segmented.logits,
        "H17 segmented CPU replay logits",
    )?;
    exact(
        &expected_final,
        &segmented.logits,
        "H17 atomic/segmented replay logits",
    )?;
    compare_domains(&expected_domains, &segmented.domains)?;
    compare_domains(&draft.domain_values(), &segmented.domains)?;
    compared_trace += observations(&trace, &segmented.observations, false)?;
    compared_trace += observations(&expected_trace, &segmented.observations, false)?;
    cpu.commit(draft)?;
    if segmented.dispatches != full_dispatches
        || segmented.state_only_steps != 0
        || segmented.logit_steps != 1
        || prefix_dispatches + segmented.dispatches != atomic_dispatches
        || prefix_bytes + segmented.chunk_bytes != atomic_bytes
        || segmented.chunk_bytes != segmented.bytes
    {
        return Err(Error::State(
            "H17 segmented prefill scheduling/readback differs".into(),
        ));
    }
    let replay = session.binding();
    drop(segmented);
    println!(
        "H17_RESIDENT_SEGMENTED_PREFILL_PASS prefix_tokens=2 prefix_state_only_steps=2 prefix_chunk_readback_bytes={prefix_bytes} logits_tokens=1 exact_atomic_replay=true all42_domains=exact ordered_trace=exact final_frontier={} restored_session_epoch={} publishes=2",
        replay.frontier(),
        replay.session_epoch()
    );

    let before = session.binding();
    session.restore(&saved)?;
    restoration(before, session.binding(), saved.binding)?;
    let before = session.binding();
    let prepared = session
        .prepare_mode(&[42, 77], false, Readout::None)
        .await?;
    state_receipt(&prepared.readback, 2, 2 * state_dispatches, false)?;
    if session.binding() != before || prepared.ready.expected_root() != before.root() {
        return Err(Error::State(
            "H17 private state-only Ready mutated visible root".into(),
        ));
    }
    session.restore(&saved)?;
    let restored = session.binding();
    restoration(before, restored, saved.binding)?;
    if session.commit(prepared).is_ok() || session.binding() != restored {
        return Err(Error::State(
            "H17 stale state-only actual Ready published after restore".into(),
        ));
    }
    compare_domains(
        &saved_cpu.domain_values(),
        &session.snapshot().read_domains().await?,
    )?;
    if resident.service.uploaded_value_bytes() != warm_uploads
        || resident.service.pending_events()? != 0
    {
        return Err(Error::State(
            "H17 warm prefill uploaded values or leaked physical events".into(),
        ));
    }
    drop(saved);
    drop(session);
    if resident.service.resident_bytes()? != original_bytes {
        return Err(Error::State(
            "H17 completed private/root state leaked admission".into(),
        ));
    }
    {
        let workspace = resident.workspace.lock().await;
        let signature = workspace
            .leases
            .iter()
            .filter(|(_, a)| *a == DeviceAccess::Read)
            .map(|(l, _)| (l.id(), l.bytes()))
            .collect::<Vec<_>>();
        if signature != source_signature || workspace.program.budget().source_bytes != source_bytes
        {
            return Err(Error::State(
                "H17 source lease identity/bytes changed".into(),
            ));
        }
    }
    resident.service.ensure_usable()?;
    println!(
        "H17_RESIDENT_PREFILL_PASS physical={:?} state_dispatches={state_dispatches} full_dispatches={full_dispatches} saved_dispatches_per_state_token={expected_savings} compared_trace_f32={compared_trace} cold_source_uploaded_bytes={cold_uploads} source_lease_bytes={source_bytes} cold_session_zero_state_uploaded_bytes={} warm_value_uploads=0 private_actual_ready_preserves_root=true stale_state_only_ready_after_restore=refused restored_state=exact final_resident_bytes={original_bytes} pending_events=0",
        resident.service.physical_key(),
        warm_uploads - cold_uploads
    );
    Ok(())
}
