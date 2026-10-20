//! Actual original CPU/B1/nativeB2/graphB2 qualification; ROOT executes only.
use super::*;
use zerogravity_model_qwen35::QwenDenseCpu;

pub(super) type Domains = Vec<(StateSlot, Vec<f32>)>;
const IDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];

pub(super) async fn states(sessions: &[ResidentSession]) -> Result<Vec<Domains>> {
    let mut values = Vec::with_capacity(sessions.len());
    for session in sessions {
        values.push(session.snapshot().read_domains().await?);
    }
    Ok(values)
}
pub(super) fn roots(sessions: &[ResidentSession]) -> Vec<RootBinding> {
    sessions.iter().map(ResidentSession::binding).collect()
}
pub(super) async fn unchanged(
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
pub(super) fn idle(model: &ResidentModel, bytes: usize) -> Result<()> {
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
pub(super) async fn loaded(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<(Arc<LoadedQwenModel>, Arc<ResidentModel>)> {
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "graph physical qualification requires release".into(),
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("graph original GGUF path".into()))?;
    let source = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    let plan = source.forward_plan(512, true, true)?;
    let resident =
        ResidentModel::load_inner(device, source.clone(), plan, provider_cap, device_cap, true)
            .await?;
    if resident.states.len() != 42 || resident.vocab != 248_320 {
        return Err(Error::Unsupported(
            "graph qualifier supports approved original Qwen3.5 .8B/2B".into(),
        ));
    }
    Ok((source, resident))
}

pub(crate) async fn qualify(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    let (source, resident) = loaded(device, path, provider_cap, device_cap).await?;
    let key = resident.service.physical_key();
    let mut sessions = Vec::new();
    let mut singles = Vec::new();
    let mut native = Vec::new();
    let mut cpus = Vec::new();
    for history in [&IDS[..2], &IDS[2..5]] {
        let mut cpu = QwenDenseCpu::from_shared(source.clone())?;
        let expected = cpu.prefill_transaction(history)?;
        for group in [&mut sessions, &mut singles, &mut native] {
            let mut session = resident.session().await?;
            exact(
                &expected,
                &session.forward(history, false).await?.logits,
                "graph independent initial history fullV",
            )?;
            compare_domains(
                &cpu.snapshot().domain_values(),
                &session.snapshot().read_domains().await?,
            )?;
            group.push(session);
        }
        cpus.push(cpu);
    }
    let prior_roots = roots(&sessions);
    let prior = states(&sessions).await?;
    let mut graph = resident
        .capture_logits_b2(&[(&sessions[0], IDS[5]), (&sessions[1], IDS[6])])
        .await?;
    unchanged(&sessions, &prior_roots, &prior).await?;
    let receipt = graph.receipt();
    if receipt.warm_dispatches.contains(&0)
        || receipt.captured_kernels == 0
        || receipt.projection_dots == 0
        || receipt.body_nodes == 0
        || receipt.retained_lease_bytes == 0
        || receipt.owned_lease_vector_bytes == 0
        || graph.replay_enqueues() != 0
    {
        return Err(Error::State("graph cold receipt incomplete".into()));
    }
    let warm_bytes = resident.service.resident_bytes()?;
    let uploads = resident.service.uploaded_value_bytes();
    idle(&resident, warm_bytes)?;
    if resident
        .capture_logits_b2(&[(&sessions[0], 11), (&sessions[1], 17)])
        .await
        .is_ok()
    {
        return Err(Error::State("second live graph admitted".into()));
    }
    // Every refusal must precede a physical graph enqueue or candidate lease.
    for rows in [
        vec![(&sessions[0], u32::MAX), (&sessions[1], 11)],
        vec![(&sessions[0], 11), (&sessions[0], 17)],
    ] {
        if resident.prepare_graph_logits(&rows, &graph).await.is_ok() {
            return Err(Error::State("invalid graph rows accepted".into()));
        }
    }
    if resident
        .prepare_logits_fused_cohort(&[
            (&sessions[0], 11),
            (&sessions[1], 17),
            (&singles[0], 42),
            (&singles[1], 55),
        ])
        .await
        .is_ok()
    {
        return Err(Error::State("live graph band counted reclaimable".into()));
    }
    let old_generation = graph.profile.generation;
    Arc::get_mut(&mut graph)
        .ok_or_else(|| Error::State("unexpected graph owner after Ready".into()))?
        .profile
        .generation = old_generation
        .checked_add(1)
        .ok_or_else(|| Error::Limit("test graph generation".into()))?;
    if resident
        .prepare_graph_logits(&[(&sessions[0], 11), (&sessions[1], 17)], &graph)
        .await
        .is_ok()
    {
        return Err(Error::State("stale graph workspace key accepted".into()));
    }
    Arc::get_mut(&mut graph).unwrap().profile.generation = old_generation;
    // Synthetic metadata boundary only; no claim of 256-token GPU prefill.
    if graph
        .validate(&resident, 2, &[256, 2], Readout::Logits)
        .is_ok()
        || graph.validate(&resident, 2, &[2, 3], Readout::None).is_ok()
    {
        return Err(Error::State(
            "unqualified graph readout/padding accepted".into(),
        ));
    }
    if graph.replay_enqueues() != 0 {
        return Err(Error::State("refusal enqueued graph".into()));
    }
    unchanged(&sessions, &prior_roots, &prior).await?;
    idle(&resident, warm_bytes)?;
    // Actual BufferLease pressure; no synthetic byte-credit admission.
    let staging = resident
        .state_bytes
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("graph pressure state bytes".into()))?;
    let available = resident.service.remaining_capacity()?;
    let filler = resident.service.allocate_uninitialized(
        available
            .checked_sub(staging)
            .and_then(|n| n.checked_add(1))
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::Limit("graph pressure headroom".into()))?,
        BufferRole::Scratch,
    )?;
    if resident
        .prepare_graph_logits(&[(&sessions[0], 11), (&sessions[1], 17)], &graph)
        .await
        .is_ok()
        || graph.replay_enqueues() != 0
    {
        return Err(Error::State(
            "graph pressure submitted unadmitted candidates".into(),
        ));
    }
    drop(filler);
    idle(&resident, warm_bytes)?;
    let abandoned = resident
        .prepare_graph_logits(&[(&sessions[0], 55), (&sessions[1], 78)], &graph)
        .await?;
    let total_kernels = abandoned.dispatches();
    if total_kernels <= receipt.captured_kernels
        || abandoned.fused_dispatches() != receipt.projection_dots
        || abandoned.native_greedy_dispatches() != 0
    {
        return Err(Error::State("graph kernel/outside/sampler receipt".into()));
    }
    drop(abandoned);
    unchanged(&sessions, &prior_roots, &prior).await?;
    idle(&resident, warm_bytes)?;
    // Restore changes epoch after an actual B1 state transition. The old
    // graph proposal cannot publish any sibling; graph itself remains reusable.
    let stale = resident
        .prepare_graph_logits(&[(&sessions[0], 55), (&sessions[1], 78)], &graph)
        .await?;
    let snapshot = sessions[0].snapshot();
    sessions[0].forward(&[101], false).await?;
    let changed_roots = roots(&sessions);
    let changed = states(&sessions).await?;
    if stale
        .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
        .is_ok()
    {
        return Err(Error::State("stale graph cohort published".into()));
    }
    unchanged(&sessions, &changed_roots, &changed).await?;
    sessions[0].restore(&snapshot)?;
    drop(snapshot);
    idle(&resident, warm_bytes)?;
    let mut committed_kernels = 0usize;
    for step in 0..4 {
        let ids = [IDS[(step + 6) % 8], IDS[(step + 1) % 8]];
        let order = if step % 2 == 0 { [0, 1] } else { [1, 0] };
        let expected = cpus
            .iter_mut()
            .zip(ids)
            .map(|(cpu, id)| cpu.forward(id))
            .collect::<Result<Vec<_>>>()?;
        for row in 0..2 {
            let got = singles[row].forward(&[ids[row]], false).await?;
            exact(
                &expected[row],
                &got.logits,
                "graph independent accepted B1 fullV",
            )?;
            compare_domains(
                &cpus[row].snapshot().domain_values(),
                &singles[row].snapshot().read_domains().await?,
            )?;
        }
        let shadow = resident
            .prepare_logits_fused_cohort(&[
                (&native[order[0]], ids[order[0]]),
                (&native[order[1]], ids[order[1]]),
            ])
            .await?;
        let mut ordered = native.iter_mut().collect::<Vec<_>>();
        if order[0] == 1 {
            ordered.swap(0, 1);
        }
        let shadow = shadow.commit(&mut ordered)?;
        drop(ordered);
        let candidate = resident
            .prepare_graph_logits(
                &[
                    (&sessions[order[0]], ids[order[0]]),
                    (&sessions[order[1]], ids[order[1]]),
                ],
                &graph,
            )
            .await?;
        if candidate.len() != 2
            || candidate.dispatches() != total_kernels
            || candidate.fused_dispatches() != receipt.projection_dots
            || candidate
                .prior_frontiers()
                .ne(order.into_iter().map(|r| sessions[r].binding().frontier()))
        {
            return Err(Error::State(
                "graph replay per-row/dispatch receipt changed".into(),
            ));
        }
        committed_kernels += candidate.dispatches();
        let mut ordered = sessions.iter_mut().collect::<Vec<_>>();
        if order[0] == 1 {
            ordered.swap(0, 1);
        }
        let reads = candidate.commit(&mut ordered)?;
        drop(ordered);
        if reads.len() != 2 || shadow.len() != 2 {
            return Err(Error::State("graph/full native row coverage".into()));
        }
        for (index, row) in order.into_iter().enumerate() {
            if reads[index].logits.len() != 248_320
                || reads[index].device_token.is_some()
                || reads[index].bytes != 248_320 * 4 + usize::from(index == 0) * 4
            {
                return Err(Error::State(
                    "graph full vocabulary/private row readout".into(),
                ));
            }
            exact(&expected[row], &reads[index].logits, "graph CPU fullV")?;
            exact(
                &shadow[index].logits,
                &reads[index].logits,
                "graph nativeB2 fullV",
            )?;
            let expected = cpus[row].snapshot().domain_values();
            if expected.len() != 42 {
                return Err(Error::State("graph original all42 coverage".into()));
            }
            compare_domains(&expected, &sessions[row].snapshot().read_domains().await?)?;
            compare_domains(&expected, &native[row].snapshot().read_domains().await?)?;
        }
        if resident.service.uploaded_value_bytes() != uploads {
            return Err(Error::State("graph warm replay uploaded values".into()));
        }
        idle(&resident, warm_bytes)?;
    }
    let enqueues = graph.replay_enqueues();
    if enqueues != 6 {
        return Err(Error::State(
            "graph host enqueue count excludes/multiplies actual replays".into(),
        ));
    }
    graph.retire(sessions[0].binding().root()).await?;
    idle(&resident, warm_bytes)?;
    {
        let _allocation = resident.allocation.lock().await;
        let mut workspace = resident.workspace.lock().await;
        if workspace
            .batch
            .as_ref()
            .is_none_or(|b| b.program.graph_pinned())
        {
            return Err(Error::State("retired graph still pins band".into()));
        }
        workspace.batch = None;
    }
    let live = resident.service.resident_bytes()?;
    if live >= warm_bytes {
        return Err(Error::State(
            "retired graph batch bytes not reclaimable".into(),
        ));
    }
    drop(cpus);
    drop(native);
    drop(singles);
    drop(sessions);
    drop(resident);
    drop(source);
    let end =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if end.reserved != 0 || end.fenced {
        return Err(Error::State(
            "graph final source/lease/envelope reclamation".into(),
        ));
    }
    println!(
        "P12_GRAPH_B2_LOGITS_PASS source=original_GGUF rows=2 independent_frontiers=true swapped_rows=true full_vocab=248320 all_domains=42 steps=4 warm_kernels={:?} captured_body_kernel_launch_sites={} outside_kernels_per_replay={} graph_host_enqueues={} committed_kernel_launches={} recorded_model_execution=0 retained_engine_lease_bytes={} owned_lease_vector_bytes={} driver_internal_bytes=unqualified driver_graph_node_enumeration=unqualified synthetic_padding_guard=true sampler=outside_unused final_reserved=0",
        receipt.warm_dispatches,
        receipt.captured_kernels,
        total_kernels - receipt.captured_kernels,
        enqueues,
        committed_kernels,
        receipt.retained_lease_bytes,
        receipt.owned_lease_vector_bytes
    );
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
pub(crate) async fn qualify_abort(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    let (source, resident) = loaded(device.clone(), path, provider_cap, device_cap).await?;
    let mut sessions = vec![resident.session().await?, resident.session().await?];
    for (session, history) in sessions.iter_mut().zip([&IDS[..2], &IDS[2..5]]) {
        let mut cpu = QwenDenseCpu::from_shared(source.clone())?;
        exact(
            &cpu.prefill_transaction(history)?,
            &session.forward(history, false).await?.logits,
            "graph abort independent CPU/B1 fullV prefix",
        )?;
        compare_domains(
            &cpu.snapshot().domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
    }
    let bindings = roots(&sessions);
    let prior = states(&sessions).await?;
    let snapshots = sessions
        .iter()
        .map(ResidentSession::snapshot)
        .collect::<Vec<_>>();
    let graph = resident
        .capture_logits_b2(&[(&sessions[0], 55), (&sessions[1], 78)])
        .await?;
    let old_ready = resident
        .prepare_logits_fused_cohort(&[(&sessions[0], 55), (&sessions[1], 78)])
        .await?;
    let client = resident.service.client().clone();
    let key = resident.service.physical_key();
    let cap = resident.service.capacity() as u64;
    let before = crate::physical_quarantine::quarantined_handles_for(&client)?;
    let owners_before = crate::physical_quarantine::quarantined_completion_owners_for(&client)?;
    let weak_graph = Arc::downgrade(&graph);
    let weak_service = Arc::downgrade(&resident.service);
    resident.launch_seen.store(0, Ordering::Release);
    resident.abort_after_launch.store(1, Ordering::Release);
    let rows = [(&sessions[0], 101), (&sessions[1], 135)];
    let mut future = Box::pin(resident.prepare_graph_logits(&rows, &graph));
    loop {
        if let Some(result) = futures_lite::future::poll_once(future.as_mut()).await {
            return Err(Error::State(format!(
                "graph abort before actual replay pending: success={}",
                result.is_ok()
            )));
        }
        if resident.launch_seen.load(Ordering::Acquire) > 0 {
            break;
        }
        futures_lite::future::yield_now().await;
    }
    if graph.replay_enqueues() != 1 {
        return Err(Error::State(
            "graph abort did not enqueue one real executable".into(),
        ));
    }
    drop(future);
    let handles = resident.launch_seen.load(Ordering::Acquire);
    let retained = crate::physical_quarantine::quarantined_handles_for(&client)?;
    if handles == 0
        || retained.saturating_sub(before) < handles
        || resident.service.ensure_usable().is_ok()
        || roots(&sessions) != bindings
        || old_ready
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
        || crate::physical_quarantine::quarantined_completion_owners_for(&client)? <= owners_before
    {
        return Err(Error::State(
            "unknown graph completion lost original ownership/fence".into(),
        ));
    }
    // No graph destroy, client.sync, diagnostic recovery or unknown readback.
    // Prior complete domains were captured before the abort; roots stay exact.
    if prior.len() != 2
        || snapshots.len() != 2
        || prior.iter().any(|d| d.len() != 42)
        || resident
            .prepare_graph_logits(&[(&sessions[0], 11), (&sessions[1], 17)], &graph)
            .await
            .is_ok()
        || sessions[0].prepare(&[11], false).await.is_ok()
        || ResidentService::new(
            device,
            source.source_identity(),
            ResourceOwner::new("graph-abort", "refused")?,
            1024 * 1024,
            device_cap,
        )
        .is_ok()
    {
        return Err(Error::State("unknown graph service admitted work".into()));
    }
    drop(graph);
    drop(snapshots);
    drop(sessions);
    drop(resident);
    drop(source);
    if weak_graph.upgrade().is_none()
        || weak_service.upgrade().is_none()
        || crate::physical_quarantine::quarantined_handles_for(&client)? < retained
    {
        return Err(Error::State(
            "graph/model drop destroyed quarantined executable/resources".into(),
        ));
    }
    let final_budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if final_budget.reserved != cap || !final_budget.fenced {
        return Err(Error::State("unknown graph envelope released".into()));
    }
    println!(
        "P12_GRAPH_B2_ABORT_PASS actual_graph_host_enqueue=1 completion=unknown independent_roots_unchanged=2 quarantined_original_handles={retained} executable_owner=retained driver_destroy=not_requested trusted_recovery=false retained_envelope_bytes={cap} prior_all42=observed_before_abort post_abort_state_readback=not_attempted"
    );
    Ok(())
}
