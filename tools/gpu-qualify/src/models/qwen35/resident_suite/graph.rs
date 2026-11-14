//! Qwen host/reference driver over the original physical probes.
use super::*;
use zerogravity_gpu::qualification::resident::graph::*;

fn idle_at(model: &ResidentModel, bytes: usize, stage: &str) -> Result<()> {
    idle(model, bytes).map_err(|error| Error::State(format!("{stage}: {error}")))
}

pub async fn qualify(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    let (source, resident) = loaded(device, path, provider_cap, device_cap).await?;
    let key = resident.qualification().physical_key();
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
        || receipt.resource_charge_bytes == 0
        || graph.replay_enqueues() != 0
    {
        return Err(Error::State("graph cold receipt incomplete".into()));
    }
    let warm_bytes = resident.qualification().resident_bytes()?;
    let warm_foreign = resident.qualification().foreign_bytes()?;
    let resource_witness = graph.qualification_resource_witness();
    if warm_foreign != receipt.resource_charge_bytes
        || resource_witness.0 != receipt.resource_charge_bytes
        || resource_witness.1 == 0
        || resource_witness.2.iter().any(|&owners| owners == 0)
    {
        return Err(Error::State(format!(
            "graph retained charge witness differs: ledger_foreign_bytes={warm_foreign} receipt_bytes={} graph_resource_bytes={} graph_charge_count={} graph_charge_strong_counts={:?}",
            receipt.resource_charge_bytes,
            resource_witness.0,
            resource_witness.1,
            resource_witness.2
        )));
    }
    let uploads = resident.qualification().uploaded_value_bytes();
    if let Err(error) = idle_at(&resident, warm_bytes, "post-capture") {
        let steady_bytes = resident.qualification().resident_bytes()?;
        let steady_foreign = resident.qualification().foreign_bytes()?;
        return Err(Error::State(format!(
            "{error}; warm_foreign_bytes={warm_foreign} steady_foreign_bytes={steady_foreign} graph_resource_bytes={} graph_charge_count={} graph_charge_strong_counts={:?} resident_delta={}",
            resource_witness.0,
            resource_witness.1,
            resource_witness.2,
            warm_bytes.saturating_sub(steady_bytes)
        )));
    }
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
    let old_generation = graph.qualification_generation();
    Arc::get_mut(&mut graph)
        .ok_or_else(|| Error::State("unexpected graph owner after Ready".into()))?
        .qualification_set_generation(
            old_generation
                .checked_add(1)
                .ok_or_else(|| Error::Limit("test graph generation".into()))?,
        );
    if resident
        .prepare_graph_logits(&[(&sessions[0], 11), (&sessions[1], 17)], &graph)
        .await
        .is_ok()
    {
        return Err(Error::State("stale graph workspace key accepted".into()));
    }
    Arc::get_mut(&mut graph)
        .unwrap()
        .qualification_set_generation(old_generation);
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
    idle_at(&resident, warm_bytes, "post-refusal")?;
    // Actual BufferLease pressure; no synthetic byte-credit admission.
    let staging = resident
        .state_bytes()
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("graph pressure state bytes".into()))?;
    let available = resident.qualification().remaining_capacity()?;
    let filler = resident.qualification().allocate_uninitialized(
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
    idle_at(&resident, warm_bytes, "post-pressure")?;
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
    idle_at(&resident, warm_bytes, "post-abandon")?;
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
    idle_at(&resident, warm_bytes, "post-stale-restore")?;
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
        if resident.qualification().uploaded_value_bytes() != uploads {
            return Err(Error::State("graph warm replay uploaded values".into()));
        }
        idle_at(
            &resident,
            warm_bytes,
            &format!("post-committed-replay-step-{step}"),
        )?;
    }
    let enqueues = graph.replay_enqueues();
    if enqueues != 6 {
        return Err(Error::State(
            "graph host enqueue count excludes/multiplies actual replays".into(),
        ));
    }
    let retired_bytes = warm_bytes
        .checked_sub(receipt.resource_charge_bytes)
        .ok_or_else(|| Error::State("graph policy charge exceeds warm resident bytes".into()))?;
    graph.retire(sessions[0].binding().root()).await?;
    let retired_foreign = resident.qualification().foreign_bytes()?;
    if retired_foreign != 0 {
        return Err(Error::State(format!(
            "graph trusted retirement retained foreign charge: foreign_bytes={retired_foreign}"
        )));
    }
    idle_at(&resident, retired_bytes, "post-graph-retire")?;
    resident.qualification_release_graph_band().await?;
    let live = resident.qualification().resident_bytes()?;
    if live >= retired_bytes {
        return Err(Error::State(
            "retired graph batch bytes not reclaimable".into(),
        ));
    }

    // Production lifecycle/cache qualification stays private: capture installs
    // one live graph in the batch workspace, a second capture refuses, and a
    // different native band must retire/destroy the graph before replacement.
    let production_rows = [(&sessions[0], 11), (&sessions[1], 17)];
    let production_receipt = resident
        .qualification_capture_production_graph(&production_rows, ResidentOutput::Logits)
        .await?;
    if resident
        .qualification_production_graph_cache_state()
        .await?
        != "live"
        || resident.qualification().foreign_bytes()? != production_receipt.resource_charge_bytes
        || !resident
            .qualification_production_graph_cache_compatible(
                &production_rows,
                ResidentOutput::Logits,
            )
            .await?
        || resident
            .qualification_production_graph_cache_compatible(
                &production_rows,
                ResidentOutput::StateOnly,
            )
            .await?
    {
        return Err(Error::State(
            "production graph cache live identity/profile witness failed".into(),
        ));
    }
    let production_live_bytes = resident.qualification().resident_bytes()?;
    let selector_roots = roots(&sessions);
    let selector_states = states(&sessions).await?;
    let selector_before = resident
        .qualification_production_graph_replay_enqueues()
        .await?
        .ok_or_else(|| Error::State("production selector live graph witness absent".into()))?;
    let selected = resident
        .qualification_prepare_production_selected(&production_rows, Readout::Logits)
        .await?;
    if selected.len() != 2 || selected.fused_dispatches() != production_receipt.projection_dots {
        return Err(Error::State(
            "production selector graph receipt changed".into(),
        ));
    }
    drop(selected);
    let selector_after = resident
        .qualification_production_graph_replay_enqueues()
        .await?
        .ok_or_else(|| Error::State("production selector graph disappeared".into()))?;
    if selector_after != selector_before + 1 {
        return Err(Error::State(
            "compatible production selector did not replay exactly once".into(),
        ));
    }
    let fallback = resident
        .qualification_prepare_production_selected(&production_rows, Readout::None)
        .await?;
    if fallback.len() != 2 || fallback.fused_dispatches() == 0 {
        return Err(Error::State(
            "production selector native fallback receipt incomplete".into(),
        ));
    }
    drop(fallback);
    if resident
        .qualification_production_graph_replay_enqueues()
        .await?
        != Some(selector_after)
        || resident
            .qualification_production_graph_cache_state()
            .await?
            != "live"
        || resident.qualification().foreign_bytes()? != production_receipt.resource_charge_bytes
    {
        return Err(Error::State(
            "incompatible production selector replayed or retired live graph".into(),
        ));
    }
    unchanged(&sessions, &selector_roots, &selector_states).await?;
    idle_at(&resident, production_live_bytes, "post-production-selector")?;

    // Same process/model/workspace/frontier and identical full-logit readback.
    // Alternate order to expose clock drift; timing stays diagnostic until
    // serving selection is separately admitted.
    for round in 0..4 {
        if round % 2 == 0 {
            drop(
                resident
                    .prepare_logits_fused_cohort(&production_rows)
                    .await?,
            );
            drop(
                resident
                    .qualification_prepare_production_selected(&production_rows, Readout::Logits)
                    .await?,
            );
        } else {
            drop(
                resident
                    .qualification_prepare_production_selected(&production_rows, Readout::Logits)
                    .await?,
            );
            drop(
                resident
                    .prepare_logits_fused_cohort(&production_rows)
                    .await?,
            );
        }
    }
    let benchmark_enqueues_before = resident
        .qualification_production_graph_replay_enqueues()
        .await?
        .ok_or_else(|| Error::State("production benchmark graph witness absent".into()))?;
    let mut native_us = Vec::with_capacity(12);
    let mut graph_us = Vec::with_capacity(12);
    for round in 0..12 {
        if round % 2 == 0 {
            let start = std::time::Instant::now();
            drop(
                resident
                    .prepare_logits_fused_cohort(&production_rows)
                    .await?,
            );
            native_us.push(start.elapsed().as_micros());
            let start = std::time::Instant::now();
            drop(
                resident
                    .qualification_prepare_production_selected(&production_rows, Readout::Logits)
                    .await?,
            );
            graph_us.push(start.elapsed().as_micros());
        } else {
            let start = std::time::Instant::now();
            drop(
                resident
                    .qualification_prepare_production_selected(&production_rows, Readout::Logits)
                    .await?,
            );
            graph_us.push(start.elapsed().as_micros());
            let start = std::time::Instant::now();
            drop(
                resident
                    .prepare_logits_fused_cohort(&production_rows)
                    .await?,
            );
            native_us.push(start.elapsed().as_micros());
        }
    }
    let benchmark_enqueues_after = resident
        .qualification_production_graph_replay_enqueues()
        .await?
        .ok_or_else(|| Error::State("production benchmark graph disappeared".into()))?;
    if benchmark_enqueues_after != benchmark_enqueues_before + graph_us.len()
        || resident.qualification().foreign_bytes()? != production_receipt.resource_charge_bytes
    {
        return Err(Error::State(
            "production graph benchmark selector/lifetime witness changed".into(),
        ));
    }
    idle_at(
        &resident,
        production_live_bytes,
        "post-production-selector-benchmark",
    )?;
    let native_mean_us = native_us.iter().sum::<u128>() / native_us.len() as u128;
    let graph_mean_us = graph_us.iter().sum::<u128>() / graph_us.len() as u128;
    native_us.sort_unstable();
    graph_us.sort_unstable();
    let native_median_us = native_us[native_us.len() / 2];
    let graph_median_us = graph_us[graph_us.len() / 2];

    if resident
        .qualification_capture_production_graph(&production_rows, ResidentOutput::Logits)
        .await
        .is_ok()
    {
        return Err(Error::State(
            "production graph cache admitted a second cold capture".into(),
        ));
    }
    idle_at(
        &resident,
        production_live_bytes,
        "post-production-cache-refusal",
    )?;
    if !resident.qualification_retire_production_graph().await?
        || resident
            .qualification_production_graph_cache_state()
            .await?
            != "spent"
        || resident.qualification().foreign_bytes()? != 0
    {
        return Err(Error::State(
            "explicit production graph retirement retained stale cache/charge".into(),
        ));
    }
    resident.qualification_release_graph_band().await?;

    // A new live batch workspace may capture once again. Replacing that band
    // must retire the second cached graph before any new batch storage is built.
    let replacement_receipt = resident
        .qualification_capture_production_graph(&production_rows, ResidentOutput::Logits)
        .await?;
    if resident
        .qualification_production_graph_cache_state()
        .await?
        != "live"
        || resident.qualification().foreign_bytes()? != replacement_receipt.resource_charge_bytes
    {
        return Err(Error::State(
            "fresh production graph workspace failed to own its cache/charge".into(),
        ));
    }
    let replacement = resident
        .prepare_logits_fused_cohort(&[
            (&sessions[0], 19),
            (&sessions[1], 23),
            (&singles[0], 29),
            (&singles[1], 31),
        ])
        .await?;
    drop(replacement);
    if resident
        .qualification_production_graph_cache_state()
        .await?
        != "cold"
        || resident.qualification().foreign_bytes()? != 0
    {
        return Err(Error::State(
            "production graph replacement retained stale cache/charge".into(),
        ));
    }
    resident.qualification_release_graph_band().await?;

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
        "P12_GRAPH_B2_LOGITS_PASS source=original_GGUF rows=2 independent_frontiers=true swapped_rows=true full_vocab=248320 all_domains=42 steps=4 warm_kernels={:?} captured_body_kernel_launch_sites={} outside_kernels_per_replay={} graph_host_enqueues={} committed_kernel_launches={} recorded_model_execution=0 retained_engine_lease_bytes={} owned_lease_vector_bytes={} graph_policy_charge_bytes={} graph_policy_charge_retained_while_live=true graph_policy_charge_released_on_retire=true production_cache_capture_once=true production_cache_charge_bytes={} production_cache_compatible_lookup=true production_cache_incompatible_profile_miss=true production_selector_compatible_graph_replay=true production_selector_incompatible_native_fallback=true production_selector_replay_enqueues={} production_selector_public_serving=false selector_benchmark_samples=12 selector_native_mean_us={} selector_graph_mean_us={} selector_native_median_us={} selector_graph_median_us={} selector_benchmark_acceptance=diagnostic production_cache_second_capture_refused=true production_cache_explicit_retire=true production_cache_explicit_retire_state=spent production_cache_explicit_retire_foreign_bytes=0 production_cache_fresh_workspace_recapture=true production_cache_replacement_charge_bytes={} production_cache_replacement=B2_to_B4 production_cache_replacement_retired_before_release=true production_cache_replacement_foreign_bytes=0 replacement_cache_state=cold driver_internal_bytes=unqualified driver_graph_node_enumeration=unqualified synthetic_padding_guard=true sampler=outside_unused final_reserved=0",
        receipt.warm_dispatches,
        receipt.captured_kernels,
        total_kernels - receipt.captured_kernels,
        enqueues,
        committed_kernels,
        receipt.retained_lease_bytes,
        receipt.owned_lease_vector_bytes,
        receipt.resource_charge_bytes,
        production_receipt.resource_charge_bytes,
        selector_after,
        native_mean_us,
        graph_mean_us,
        native_median_us,
        graph_median_us,
        replacement_receipt.resource_charge_bytes
    );
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
pub async fn qualify_abort(
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
    let client = resident.qualification().client().clone();
    let key = resident.qualification().physical_key();
    let cap = resident.qualification().capacity() as u64;
    let before = quarantined_handles_for(&client)?;
    let owners_before = quarantined_completion_owners_for(&client)?;
    let weak_graph = Arc::downgrade(&graph);
    let weak_service = resident.qualification_witness();
    resident.qualification_reset_launch_seen();
    resident.qualification_arm_abort(1);
    let rows = [(&sessions[0], 101), (&sessions[1], 135)];
    let mut future = Box::pin(resident.prepare_graph_logits(&rows, &graph));
    loop {
        if let Some(result) = futures_lite::future::poll_once(future.as_mut()).await {
            return Err(Error::State(format!(
                "graph abort before actual replay pending: success={}",
                result.is_ok()
            )));
        }
        if resident.qualification_launch_seen() > 0 {
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
    let handles = resident.qualification_launch_seen();
    let retained = quarantined_handles_for(&client)?;
    if handles == 0
        || retained.saturating_sub(before) < handles
        || resident.qualification().ensure_usable().is_ok()
        || roots(&sessions) != bindings
        || old_ready
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
        || quarantined_completion_owners_for(&client)? <= owners_before
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
        || QualificationService::new(
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
        || quarantined_handles_for(&client)? < retained
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
    let resident = ResidentModel::qualification_load_graph(
        device,
        source.clone(),
        plan,
        provider_cap,
        device_cap,
    )
    .await?;
    if resident.state_requirements().len() != 42 || resident.vocabulary() != 248_320 {
        return Err(Error::Unsupported(
            "graph qualifier supports approved original Qwen3.5 .8B/2B".into(),
        ));
    }
    Ok((source, resident))
}
