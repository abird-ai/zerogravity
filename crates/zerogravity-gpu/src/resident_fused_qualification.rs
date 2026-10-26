//! P12 full-model native batch proof. CPU/B1 outputs remain host oracles.
use super::*;

type DomainValues = Vec<(StateSlot, Vec<f32>)>;

const IDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];

struct Reference {
    logits: Vec<f32>,
    domains: DomainValues,
    dispatches: usize,
    initialization_bytes: u64,
}

async fn reference(
    resident: &Arc<ResidentModel>,
    model: &Arc<LoadedQwenModel>,
    history: &[u32],
    token: u32,
) -> Result<Reference> {
    let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(model))?;
    let before_initialization = resident.service.uploaded_value_bytes();
    let mut gpu = resident.session().await?;
    let initialized = resident.service.uploaded_value_bytes();
    let initialization_bytes = initialized
        .checked_sub(before_initialization)
        .ok_or_else(|| Error::State("B1 oracle upload accounting".into()))?;
    if initialization_bytes != resident.state_bytes as u64 {
        return Err(Error::State("B1 oracle initial-state upload size".into()));
    }
    let expected_prefix = cpu.prefill_transaction(history)?;
    let actual_prefix = gpu.forward(history, false).await?;
    exact(
        &expected_prefix,
        &actual_prefix.logits,
        "fused oracle independent B1 prefix logits",
    )?;
    compare_domains(
        &cpu.snapshot().domain_values(),
        &gpu.snapshot().read_domains().await?,
    )?;
    let logits = cpu.forward(token)?;
    let actual = gpu.forward(&[token], false).await?;
    let domains = cpu.snapshot().domain_values();
    if logits.len() != 248_320 || domains.len() != 42 || actual.dispatches == 0 {
        return Err(Error::State(
            "fused oracle lacks original full vocabulary/state/physical dispatches".into(),
        ));
    }
    exact(
        &logits,
        &actual.logits,
        "fused oracle independent B1 logits",
    )?;
    compare_domains(&domains, &gpu.snapshot().read_domains().await?)?;
    if resident.service.uploaded_value_bytes() != initialized {
        return Err(Error::State(
            "warm B1 oracle uploaded model/activation values".into(),
        ));
    }
    Ok(Reference {
        logits,
        domains,
        dispatches: actual.dispatches,
        initialization_bytes,
    })
}

fn bindings(sessions: &[ResidentSession]) -> Vec<RootBinding> {
    sessions.iter().map(ResidentSession::binding).collect()
}

fn unchanged(sessions: &[ResidentSession], before: &[RootBinding]) -> Result<()> {
    if bindings(sessions) != before {
        return Err(Error::State(
            "unpublished fused batch changed a committed session root".into(),
        ));
    }
    Ok(())
}

async fn domain_values(sessions: &[ResidentSession]) -> Result<Vec<DomainValues>> {
    let mut output = Vec::with_capacity(sessions.len());
    for session in sessions {
        output.push(session.snapshot().read_domains().await?);
    }
    Ok(output)
}

async fn unchanged_domains(sessions: &[ResidentSession], before: &[DomainValues]) -> Result<()> {
    if sessions.len() != before.len() {
        return Err(Error::State("fused rollback state row coverage".into()));
    }
    for (session, expected) in sessions.iter().zip(before) {
        compare_domains(expected, &session.snapshot().read_domains().await?)?;
    }
    Ok(())
}

async fn prepare(
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

fn receipt(candidate: &PreparedResidentCohort, rows: usize, frontier: u64) -> Result<()> {
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

fn idle(resident: &ResidentModel, bytes: usize) -> Result<()> {
    if resident.service.pending_events()? != 0 || resident.service.resident_bytes()? != bytes {
        return Err(Error::State(
            "completed fused cohort retained candidate leases or pending events".into(),
        ));
    }
    Ok(())
}

pub async fn qualify(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    count: usize,
) -> Result<()> {
    if !matches!(count, 2 | 4 | 8) {
        return Err(Error::Limit(
            "full-model fused qualification requires B2/B4/B8".into(),
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("fused model source path".into()))?;
    let model = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    let resident =
        load_qualified(device, Arc::clone(&model), 512, provider_cap, device_cap).await?;
    let mut sessions = Vec::with_capacity(count);
    let mut histories = Vec::with_capacity(count);
    for (row, &first) in IDS.iter().take(count).enumerate() {
        let history = vec![first, IDS[(row + 3) % IDS.len()]];
        let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&model))?;
        let mut session = resident.session().await?;
        let expected = cpu.prefill_transaction(&history)?;
        exact(
            &expected,
            &session.forward(&history, false).await?.logits,
            "fused row distinct nonzero initial history",
        )?;
        compare_domains(
            &cpu.snapshot().domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
        sessions.push(session);
        histories.push(history);
    }
    let mut permitted_uploads = resident.service.uploaded_value_bytes();
    let initial = bindings(&sessions);
    let ids = initial
        .iter()
        .map(|root| root.session())
        .collect::<std::collections::BTreeSet<_>>();
    if ids.len() != count
        || initial
            .iter()
            .any(|root| root.frontier() != 2 || root.source() != resident.service.source_identity())
    {
        return Err(Error::State(
            "fused source/session/history isolation".into(),
        ));
    }
    let tokens = IDS[..count].to_vec();
    let prior_domains = domain_values(&sessions).await?;
    let abandoned = prepare(&resident, &sessions, &tokens).await?;
    receipt(&abandoned, count, 2)?;
    unchanged(&sessions, &initial)?;
    unchanged_domains(&sessions, &prior_domains).await?;
    drop(abandoned);
    unchanged(&sessions, &initial)?;
    unchanged_domains(&sessions, &prior_domains).await?;
    let warm_bytes = resident.service.resident_bytes()?;
    idle(&resident, warm_bytes)?;

    let mut invalid = tokens.clone();
    invalid[count - 1] = u32::MAX;
    if prepare(&resident, &sessions, &invalid).await.is_ok()
        || resident
            .prepare_logits_fused_cohort(&[(&sessions[0], tokens[0]), (&sessions[0], tokens[1])])
            .await
            .is_ok()
    {
        return Err(Error::State("bad/duplicate fused row accepted".into()));
    }
    unchanged(&sessions, &initial)?;
    unchanged_domains(&sessions, &prior_domains).await?;
    idle(&resident, warm_bytes)?;
    drop(prior_domains);

    let mut compared_state_elements = 0usize;
    let mut fused_dispatches = 0usize;
    let mut physical_dispatches = 0usize;
    let mut independent_dispatches = 0usize;
    for step in 0..2 {
        let tokens = (0..count)
            .map(|row| IDS[(row + step + 1) % IDS.len()])
            .collect::<Vec<_>>();
        let mut expected = Vec::with_capacity(count);
        for (history, &token) in histories.iter().zip(&tokens) {
            let source_reference = reference(&resident, &model, history, token).await?;
            permitted_uploads += source_reference.initialization_bytes;
            expected.push(source_reference);
        }
        idle(&resident, warm_bytes)?;
        if resident.service.uploaded_value_bytes() != permitted_uploads {
            return Err(Error::State(
                "warm fused graph uploaded source/activation values".into(),
            ));
        }
        let before = bindings(&sessions);
        let candidate = prepare(&resident, &sessions, &tokens).await?;
        receipt(&candidate, count, before[0].frontier())?;
        unchanged(&sessions, &before)?;
        physical_dispatches += candidate.dispatches();
        fused_dispatches += candidate.fused_dispatches();
        let results = candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
        if results.len() != count {
            return Err(Error::State(
                "fused full-model readback row coverage".into(),
            ));
        }
        for (row, (actual, expected)) in results.iter().zip(&expected).enumerate() {
            exact(
                &expected.logits,
                &actual.logits,
                "fused complete CPU/B1 logits",
            )?;
            let domains = sessions[row].snapshot().read_domains().await?;
            if domains.len() != 42 {
                return Err(Error::State("fused complete StateRoot coverage".into()));
            }
            compare_domains(&expected.domains, &domains)?;
            compared_state_elements += domains
                .iter()
                .map(|(_, values)| values.len())
                .sum::<usize>();
            independent_dispatches += expected.dispatches;
            let after = sessions[row].binding();
            let prior = before[row];
            if after.session() != prior.session()
                || after.source() != prior.source()
                || after.session_epoch() != prior.session_epoch()
                || after.root() != prior.root() + 1
                || after.frontier() != prior.frontier() + 1
            {
                return Err(Error::State("fused independent root transition".into()));
            }
            histories[row].push(tokens[row]);
        }
        idle(&resident, warm_bytes)?;
    }

    let tokens = IDS[..count].to_vec();
    let before = bindings(&sessions);
    let domains = domain_values(&sessions).await?;
    let candidate = prepare(&resident, &sessions, &tokens).await?;
    let mut reordered = sessions.iter_mut().collect::<Vec<_>>();
    reordered.swap(0, 1);
    if candidate.commit(&mut reordered).is_ok() {
        return Err(Error::State("foreign/reordered fused row published".into()));
    }
    drop(reordered);
    unchanged(&sessions, &before)?;
    unchanged_domains(&sessions, &domains).await?;
    idle(&resident, warm_bytes)?;

    let mut foreign = resident.session().await?;
    permitted_uploads += resident.state_bytes as u64;
    foreign.forward(&histories[count - 1], false).await?;
    let foreign_root = foreign.binding();
    let foreign_domains = foreign.snapshot().read_domains().await?;
    let candidate = prepare(&resident, &sessions, &tokens).await?;
    let mut substituted = sessions.iter_mut().collect::<Vec<_>>();
    substituted[count - 1] = &mut foreign;
    if candidate.commit(&mut substituted).is_ok() {
        return Err(Error::State(
            "foreign independent fused session published".into(),
        ));
    }
    drop(substituted);
    unchanged(&sessions, &before)?;
    unchanged_domains(&sessions, &domains).await?;
    if foreign.binding() != foreign_root {
        return Err(Error::State(
            "foreign fused session changed on refusal".into(),
        ));
    }
    compare_domains(&foreign_domains, &foreign.snapshot().read_domains().await?)?;
    drop(foreign);
    idle(&resident, warm_bytes)?;

    let candidate = prepare(&resident, &sessions, &tokens).await?;
    let saved = sessions[0].snapshot();
    sessions[0].restore(&saved)?;
    drop(saved);
    let restored = bindings(&sessions);
    if restored[0].session_epoch() != before[0].session_epoch() + 1
        || candidate
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
    {
        return Err(Error::State("stale fused session epoch published".into()));
    }
    unchanged(&sessions, &restored)?;
    unchanged_domains(&sessions, &domains).await?;
    idle(&resident, warm_bytes)?;

    let candidate = prepare(&resident, &sessions, &tokens).await?;
    sessions[0].forward(&[tokens[0]], false).await?;
    let advanced = bindings(&sessions);
    if candidate
        .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
        .is_ok()
    {
        return Err(Error::State("stale fused root published a sibling".into()));
    }
    unchanged(&sessions, &advanced)?;
    unchanged_domains(&sessions[1..], &domains[1..]).await?;
    idle(&resident, warm_bytes)?;
    drop(domains);
    let surviving = reference(&resident, &model, &histories[1], tokens[1]).await?;
    permitted_uploads += surviving.initialization_bytes;
    let actual = sessions[1].forward(&[tokens[1]], false).await?;
    exact(
        &surviving.logits,
        &actual.logits,
        "fused rollback surviving sibling",
    )?;
    compare_domains(
        &surviving.domains,
        &sessions[1].snapshot().read_domains().await?,
    )?;
    idle(&resident, warm_bytes)?;
    if resident.service.uploaded_value_bytes() != permitted_uploads {
        return Err(Error::State(
            "warm fused graph uploaded source/activation values".into(),
        ));
    }
    if physical_dispatches >= independent_dispatches {
        return Err(Error::State(format!(
            "native full-model dispatches {physical_dispatches} do not reduce independent B1 count {independent_dispatches}"
        )));
    }
    let without_sessions = warm_bytes
        .checked_sub(count * resident.state_bytes)
        .ok_or_else(|| Error::State("fused session lease accounting".into()))?;
    drop(sessions);
    idle(&resident, without_sessions)?;
    let key = resident.service.physical_key();
    drop(resident);
    let final_budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if final_budget.reserved != 0 || final_budget.fenced {
        return Err(Error::State(
            "fused final physical envelope not reclaimed".into(),
        ));
    }
    println!(
        "P12_RESIDENT_FUSED_GPU_PASS B={count} independent_sessions={count} \
         initial_history_tokens=2 distinct_row_histories=true fused_decode_steps=2 \
         complete_vocabulary=248320 CPU_and_B1_logits=bit_exact \
         all42domains=bit_exact compared_state_elements={compared_state_elements} \
         physical_dispatches={physical_dispatches} native_multirow_projection_dispatches={fused_dispatches} \
         independent_B1_dispatches={independent_dispatches} abandoned=unpublished \
         bad_row=refused duplicate_row=refused reordered_row=refused foreign_row=refused \
         stale_epoch=all_row_rollback stale_root=all_row_rollback \
         surviving_sibling=source_exact source_reuploads=0 pending_events=0 \
         session_leases=reclaimed final_reserved_bytes=0 source={} \
         continuous_serving_and_vllm_performance=not_qualified",
        path.display()
    );
    Ok(())
}

/// Separate process only: deliberate unknown completion retains the envelope.
#[cfg(feature = "hardware-abort-probe")]
pub async fn qualify_abort(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    use std::sync::atomic::Ordering;

    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("fused abort source path".into()))?;
    let source = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    let resident = load_qualified(
        device.clone(),
        Arc::clone(&source),
        512,
        provider_cap,
        device_cap,
    )
    .await?;
    let client = device.client();
    let mut sessions = Vec::with_capacity(2);
    for history in [[11, 353], [17, 42]] {
        let mut session = resident.session().await?;
        let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&source))?;
        exact(
            &cpu.prefill_transaction(&history)?,
            &session.forward(&history, false).await?.logits,
            "fused abort distinct nonzero source prefix",
        )?;
        compare_domains(
            &cpu.snapshot().domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
        sessions.push(session);
    }
    let roots = bindings(&sessions);
    let saved = sessions
        .iter()
        .map(ResidentSession::snapshot)
        .collect::<Vec<_>>();
    let expected = domain_values(&sessions).await?;
    if expected.iter().any(|domains| domains.len() != 42)
        || roots[0].session() == roots[1].session()
        || roots.iter().any(|root| root.frontier() != 2)
    {
        return Err(Error::State(
            "fused abort original source/root coverage".into(),
        ));
    }
    // A genuinely completed old proposal must also lose publication authority
    // when the same physical card later acquires an unknown completion.
    let stale_ready = prepare(&resident, &sessions, &[55, 78]).await?;
    receipt(&stale_ready, 2, 2)?;
    unchanged(&sessions, &roots)?;
    let quarantined_before = crate::physical_quarantine::quarantined_handles_for(&client)?;
    resident.launch_seen.store(0, Ordering::Release);
    resident.abort_after_launch.store(1, Ordering::Release);
    let rows = [(&sessions[0], 101), (&sessions[1], 135)];
    let mut pending = Box::pin(resident.prepare_logits_fused_cohort(&rows));
    loop {
        if let Some(result) = futures_lite::future::poll_once(pending.as_mut()).await {
            return Err(Error::State(format!(
                "fused abort completed before pending launch hook: success={}",
                result.is_ok()
            )));
        }
        if resident.launch_seen.load(Ordering::Acquire) > 0 {
            break;
        }
        futures_lite::future::yield_now().await;
    }
    // Original ResidentSubmission/PhysicalWork/ActiveSlot RAII owns this drop.
    drop(pending);
    let submitted_handles = resident.launch_seen.load(Ordering::Acquire);
    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
    if submitted_handles == 0
        || quarantined.saturating_sub(quarantined_before) < submitted_handles
        || resident.service.ensure_usable().is_ok()
        || stale_ready
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
    {
        return Err(Error::State(
            "fused unknown completion retention/publication".into(),
        ));
    }
    unchanged(&sessions, &roots)?;
    if sessions[0].prepare(&[55], false).await.is_ok()
        || prepare(&resident, &sessions, &[55, 78]).await.is_ok()
        || ResidentService::new(
            device,
            source.source_identity(),
            ResourceOwner::new("other-model", "after-fused-abort")?,
            1024 * 1024,
            device_cap,
        )
        .is_ok()
    {
        return Err(Error::State(
            "fused fenced service admitted more work/model".into(),
        ));
    }
    unchanged(&sessions, &roots)?;

    // Diagnostic reads only: never retire unknown events or clear a fence.
    client
        .sync()
        .await
        .map_err(|error| Error::State(format!("fused abort diagnostic barrier: {error}")))?;
    for (snapshot, expected) in saved.iter().zip(&expected) {
        physical_prior_after_abort(&client, snapshot, expected).await?;
    }
    unchanged(&sessions, &roots)?;
    let key = resident.service.physical_key();
    let weak = Arc::downgrade(resident.service());
    let cap = resident.service.capacity() as u64;
    drop(saved);
    drop(sessions);
    drop(resident);
    let retained = weak
        .upgrade()
        .ok_or_else(|| Error::State("fused abort lost original charged service".into()))?;
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != cap
        || !budget.fenced
        || retained.physical_remaining()? != device_cap - cap
        || retained.ensure_usable().is_ok()
        || crate::physical_quarantine::quarantined_handles_for(&client)? < quarantined
    {
        return Err(Error::State(
            "fused abort released unknown physical ownership".into(),
        ));
    }
    println!(
        "P12_RESIDENT_FUSED_ABORT_PASS B=2 independent_sessions=2 prior_frontier=2 \
         actual_native_full_model_launch=true completion=unknown \
         submitted_original_handles={submitted_handles} quarantined_original_handles={quarantined} \
         all42_prior_domains_per_row=bit_exact canonical_future_kv_zero=exact \
         roots_unchanged=true old_ready_after_fence=refused subsequent_step=refused \
         other_model=refused retained_envelope_bytes={cap} trusted_recovery=false"
    );
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
async fn physical_prior_after_abort(
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

/// Source-real CPU checkpoint hydration, followed by native B2 GPU decode.
/// This does not establish continuous GPU prefill of the imported prefix.
#[cfg(feature = "portable-checkpoint")]
pub async fn qualify_boundary(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("fused boundary source path".into()))?;
    let source = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    if source.geometry().vocab_size != 248_320 {
        return Err(Error::Unsupported(
            "fused boundary requires original vocabulary".into(),
        ));
    }
    let full_layers = source
        .geometry()
        .layers
        .iter()
        .filter(|kind| {
            matches!(
                kind,
                zerogravity_model_qwen35::config::LayerKind::FullAttention
            )
        })
        .count();
    let boundary_launches = full_layers * (256 / 32) * 3;
    if boundary_launches == 0 {
        return Err(Error::Unsupported(
            "fused source has no attention padding boundary".into(),
        ));
    }
    let resident =
        load_qualified(device, Arc::clone(&source), 512, provider_cap, device_cap).await?;
    let original_bytes = resident.service.resident_bytes()?;
    let streams: [Vec<u32>; 2] = std::array::from_fn(|row| {
        (0..512)
            .map(|position| IDS[(position + row * 3) % IDS.len()])
            .collect()
    });
    let mut cpus = (0..2)
        .map(|_| zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&source)))
        .collect::<Result<Vec<_>>>()?;
    let mut total_dispatches = 0usize;
    let mut total_projections = 0usize;
    let mut total_b1_dispatches = 0usize;
    for frontier in [0usize, 255, 511] {
        let mut sessions = Vec::with_capacity(2);
        let mut b1_sessions = Vec::with_capacity(2);
        for (row, cpu) in cpus.iter_mut().enumerate() {
            if cpu.position() > frontier {
                return Err(Error::State(
                    "fused boundary CPU prefix moved backwards".into(),
                ));
            }
            if cpu.position() < frontier {
                cpu.prefill_chunk_without_logits(&streams[row][cpu.position()..frontier])?;
            }
            let cpu_checkpoint = cpu.checkpoint()?;
            let domains = cpu.snapshot().domain_values();
            let rebuilt = source.checkpoint_from_domains(frontier, &domains)?;
            if domains.len() != 42
                || cpu_checkpoint.position() != frontier
                || cpu_checkpoint.source_identity() != source.source_identity()
                || rebuilt.source_identity() != source.source_identity()
            {
                return Err(Error::State(
                    "fused boundary CPU checkpoint/source coverage".into(),
                ));
            }
            compare_domains(&domains, &cpu_checkpoint.domain_values())?;
            compare_domains(&domains, &rebuilt.domain_values())?;
            let mut batch_lane = resident.session().await?;
            let mut b1_lane = resident.session().await?;
            batch_lane.import_checkpoint(&rebuilt).await?;
            b1_lane.import_checkpoint(&cpu_checkpoint).await?;
            compare_domains(&domains, &batch_lane.snapshot().read_domains().await?)?;
            compare_domains(&domains, &b1_lane.snapshot().read_domains().await?)?;
            if batch_lane.binding().frontier() != frontier as u64
                || b1_lane.binding().frontier() != frontier as u64
            {
                return Err(Error::State("fused boundary imported frontier".into()));
            }
            sessions.push(batch_lane);
            b1_sessions.push(b1_lane);
            println!(
                "P12_FUSED_BOUNDARY_CPU_PREFIX_READY row={row} frontier={frontier} \
                 source_math=original_CPU checkpoint_payload_and_GPU_import=bit_exact"
            );
        }
        let identities = sessions
            .iter()
            .chain(&b1_sessions)
            .map(|session| session.binding().session())
            .collect::<std::collections::BTreeSet<_>>();
        if identities.len() != 4 {
            return Err(Error::State(
                "fused boundary B1/batch session isolation".into(),
            ));
        }
        let uploads = resident.service.uploaded_value_bytes();
        let mut at256 = None;
        for offset in 0..if frontier == 255 { 2 } else { 1 } {
            let before = bindings(&sessions);
            let tokens = streams
                .iter()
                .map(|stream| stream[frontier + offset])
                .collect::<Vec<_>>();
            let mut expected = Vec::with_capacity(2);
            let mut b1_dispatches = 0usize;
            for ((cpu, b1), &token) in cpus.iter_mut().zip(&mut b1_sessions).zip(&tokens) {
                let logits = cpu.forward(token)?;
                let domains = cpu.snapshot().domain_values();
                let actual = b1.forward(&[token], false).await?;
                if logits.len() != 248_320 || domains.len() != 42 || actual.dispatches == 0 {
                    return Err(Error::State(
                        "fused boundary independent oracle coverage".into(),
                    ));
                }
                exact(
                    &logits,
                    &actual.logits,
                    "fused boundary independent CPU/B1 logits",
                )?;
                compare_domains(&domains, &b1.snapshot().read_domains().await?)?;
                b1_dispatches += actual.dispatches;
                expected.push((logits, domains));
            }
            let candidate = prepare(&resident, &sessions, &tokens).await?;
            receipt(&candidate, 2, (frontier + offset) as u64)?;
            unchanged(&sessions, &before)?;
            let dispatches = candidate.dispatches();
            let projections = candidate.fused_dispatches();
            let results = candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
            if results.len() != 2 || dispatches >= b1_dispatches {
                return Err(Error::State("fused boundary row/dispatch receipt".into()));
            }
            for (row, (actual, (logits, domains))) in results.iter().zip(&expected).enumerate() {
                exact(
                    logits,
                    &actual.logits,
                    "fused boundary complete native logits",
                )?;
                compare_domains(domains, &sessions[row].snapshot().read_domains().await?)?;
                let after = sessions[row].binding();
                let prior = before[row];
                if after.session() != prior.session()
                    || after.source() != prior.source()
                    || after.session_epoch() != prior.session_epoch()
                    || after.root() != prior.root() + 1
                    || after.frontier() != prior.frontier() + 1
                {
                    return Err(Error::State(
                        "fused boundary independent root publication".into(),
                    ));
                }
            }
            if frontier == 255 {
                if offset == 0 {
                    at256 = Some((dispatches, b1_dispatches));
                } else {
                    let (prior_fused, prior_b1) = at256.ok_or_else(|| {
                        Error::State("fused boundary missing valid256 receipt".into())
                    })?;
                    if dispatches != prior_fused + boundary_launches
                        || b1_dispatches != prior_b1 + 2 * boundary_launches
                    {
                        return Err(Error::State(
                            "fused valid257 attention dispatch boundary differs".into(),
                        ));
                    }
                }
            }
            if resident.service.uploaded_value_bytes() != uploads
                || resident.service.pending_events()? != 0
            {
                return Err(Error::State(
                    "fused boundary warm upload/event regression".into(),
                ));
            }
            total_dispatches += dispatches;
            total_projections += projections;
            total_b1_dispatches += b1_dispatches;
            println!(
                "P12_FUSED_BOUNDARY_TOKEN_PASS imported_prefix={frontier} valid_tokens={} \
                 B=2 full248320_logits_CPU_B1_native=bit_exact all42domains=bit_exact \
                 canonical_KV_tails=positive_zero physical_dispatches={dispatches} \
                 native_multirow_projection_dispatches={projections} independent_B1_dispatches={b1_dispatches}",
                frontier + offset + 1,
            );
        }
        if frontier == 511 {
            let before = bindings(&sessions);
            let domains = domain_values(&sessions).await?;
            let bytes = resident.service.resident_bytes()?;
            if prepare(&resident, &sessions, &[11, 353]).await.is_ok()
                || b1_sessions[0].prepare(&[11], false).await.is_ok()
            {
                return Err(Error::State("fused exhausted512 session advanced".into()));
            }
            unchanged(&sessions, &before)?;
            unchanged_domains(&sessions, &domains).await?;
            idle(&resident, bytes)?;
            if resident.service.uploaded_value_bytes() != uploads {
                return Err(Error::State(
                    "fused exhausted512 refusal uploaded values".into(),
                ));
            }
        }
        drop(sessions);
        drop(b1_sessions);
        let persistent_batch_bytes = resident
            .workspace
            .lock()
            .await
            .batch
            .as_ref()
            .ok_or_else(|| Error::State("fused boundary lost native workspace".into()))?
            .program
            .budget()
            .owned_bytes()?;
        idle(&resident, original_bytes + persistent_batch_bytes)?;
    }
    let key = resident.service.physical_key();
    drop(resident);
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != 0 || budget.fenced {
        return Err(Error::State(
            "fused boundary physical envelope not reclaimed".into(),
        ));
    }
    println!(
        "P12_FUSED_CHECKPOINT_BOUNDARY_PASS B=2 CPU_hydrated_prefixes=0,255,511 \
         native_GPU_valid_tokens=1,256,257,512 distinct_histories=true \
         full248320_logits_CPU_B1_native=bit_exact all42domains=bit_exact \
         exhausted512=refused roots_unchanged_on_refusal=true private_leases=reclaimed \
         pending_events=0 final_reserved_bytes=0 physical_dispatches={total_dispatches} \
         native_multirow_projection_dispatches={total_projections} independent_B1_dispatches={total_b1_dispatches} \
         GPU_full_prefix_prefill=not_qualified"
    );
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
#[path = "resident_inference/prefix_qualification.rs"]
mod prefix_qualification;
