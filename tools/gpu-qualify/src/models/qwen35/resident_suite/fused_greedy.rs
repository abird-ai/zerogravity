//! Qwen host/reference driver over the original physical probes.
use super::*;
use zerogravity_gpu::qualification::resident::fused_greedy::*;

pub async fn qualify(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    count: usize,
) -> Result<()> {
    if !matches!(count, 2 | 4 | 8) {
        return Err(Error::Limit("native greedy requires B2/B4/B8".into()));
    }
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("greedy source path".into()))?;
    let source = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    if source.geometry().vocab_size != VOCABULARY as u64 {
        return Err(Error::Unsupported(
            "native greedy original vocabulary".into(),
        ));
    }
    let tokenizer = zerogravity_model_qwen35::tokenize::QwenTokenizer::from_file(
        &parent.join("tokenizer.json"),
    )?;
    let eos = tokenizer.eos();
    if eos as usize >= VOCABULARY {
        return Err(Error::State(
            "greedy tokenizer EOS exceeds source vocabulary".into(),
        ));
    }
    let resident =
        load_qualified(device, Arc::clone(&source), 512, provider_cap, device_cap).await?;
    let original_bytes = resident.qualification().resident_bytes()?;
    let mut cpus = Vec::with_capacity(count);
    let mut sessions = Vec::with_capacity(count);
    let mut shadows = Vec::with_capacity(count);
    let mut b1 = Vec::with_capacity(count);
    for (row, &first) in IDS.iter().take(count).enumerate() {
        let history = [first, IDS[(row + 3) % IDS.len()]];
        let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&source))?;
        let expected = cpu.prefill_transaction(&history)?;
        let mut lane = resident.session().await?;
        let mut shadow = resident.session().await?;
        let mut reference = resident.session().await?;
        lane.prefill_without_logits(&history, false).await?;
        shadow.prefill_without_logits(&history, false).await?;
        exact(
            &expected,
            &reference.forward(&history, false).await?.logits,
            "greedy independent CPU/B1 nonzero prefix",
        )?;
        let state = cpu.snapshot().domain_values();
        for physical in [&lane, &shadow, &reference] {
            compare_domains(&state, &physical.snapshot().read_domains().await?)?;
        }
        cpus.push(cpu);
        sessions.push(lane);
        shadows.push(shadow);
        b1.push(reference);
    }
    let uploads = resident.qualification().uploaded_value_bytes();
    let mut device_packets = Vec::new();
    let mut next_ids = IDS[..count].to_vec();
    next_ids[count - 1] = eos; // Real EOS input, not a fabricated trained winner.
    let mut stable_bytes = None;
    let mut total_dispatches = 0usize;
    let mut total_projections = 0usize;
    let mut total_sampler = 0usize;
    let mut total_readback = 0usize;
    for step in 0..6 {
        if step == 1 {
            refusals(&resident, &mut sessions, &device_packets).await?;
        }
        if step == 5 {
            let before = roots(&sessions);
            let state = domains(&sessions).await?;
            let bytes = resident.qualification().resident_bytes()?;
            let candidate = feedback(&resident, &sessions, &device_packets).await?;
            let saved = sessions[0].snapshot();
            sessions[0].restore(&saved)?;
            drop(saved);
            let restored = roots(&sessions);
            if restored[0].session_epoch() != before[0].session_epoch() + 1
                || candidate
                    .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
                    .is_ok()
                || feedback(&resident, &sessions, &device_packets)
                    .await
                    .is_ok()
            {
                return Err(Error::State(
                    "greedy restore epoch accepted stale packet/proposal".into(),
                ));
            }
            unchanged(&sessions, &restored)?;
            same_domains(&sessions, &state).await?;
            quiet(&resident, bytes)?;
        }
        let before = roots(&sessions);
        let mut expected_logits = Vec::with_capacity(count);
        let mut expected_states = Vec::with_capacity(count);
        let mut selected = Vec::with_capacity(count);
        for ((cpu, reference), &id) in cpus.iter_mut().zip(&mut b1).zip(&next_ids) {
            let logits = cpu.forward(id)?;
            let prior = reference.snapshot();
            let actual = reference.forward(&[id], false).await?;
            if logits.len() != VOCABULARY {
                return Err(Error::State("greedy full source vocabulary".into()));
            }
            exact(
                &logits,
                &actual.logits,
                "greedy CPU/B1 complete source logits",
            )?;
            let state = cpu.snapshot().domain_values();
            if state.len() != 42 {
                return Err(Error::State("greedy original state coverage".into()));
            }
            compare_domains(&state, &reference.snapshot().read_domains().await?)?;
            let want = greedy(&logits)?;
            reference.restore(&prior)?;
            drop(prior);
            let mut b1_greedy = reference.forward_greedy(&[id]).await?;
            let b1_packet = b1_greedy
                .device_token
                .take()
                .ok_or_else(|| Error::State("accepted B1 greedy packet absent".into()))?;
            if !b1_greedy.logits.is_empty()
                || b1_greedy.bytes != 16
                || b1_greedy.chunk_bytes != 16
                || (b1_packet.id, b1_packet.max_bits) != want
                || b1_packet.qualification_producer() != Some(reference.binding())
            {
                return Err(Error::State(
                    "accepted B1 greedy packet differs from full logits".into(),
                ));
            }
            compare_domains(&state, &reference.snapshot().read_domains().await?)?;
            drop(b1_packet);
            drop(b1_greedy);
            selected.push(want);
            expected_logits.push(logits);
            expected_states.push(state);
        }
        let shadow_rows = shadows
            .iter()
            .zip(&next_ids)
            .map(|(s, &id)| (s, id))
            .collect::<Vec<_>>();
        let full = resident.prepare_logits_fused_cohort(&shadow_rows).await?;
        if full.fused_dispatches() == 0 {
            return Err(Error::State(
                "greedy native logits shadow was serial".into(),
            ));
        }
        let full = full.commit(&mut shadows.iter_mut().collect::<Vec<_>>())?;
        for (row, actual) in full.iter().enumerate() {
            exact(
                &expected_logits[row],
                &actual.logits,
                "greedy native full-vocabulary shadow",
            )?;
        }
        same_domains(&shadows, &expected_states).await?;
        let candidate = if step == 0 || step == 5 {
            host(&resident, &sessions, &next_ids).await?
        } else if step == 2 {
            let mut tampered = device_packets.clone();
            tampered[0].id = u32::MAX;
            tampered[0].max_bits = f32::NAN.to_bits();
            feedback(&resident, &sessions, &tampered).await?
        } else {
            feedback(&resident, &sessions, &device_packets).await?
        };
        private_receipt(&candidate, count, before[0].frontier())?;
        unchanged(&sessions, &before)?;
        total_dispatches += candidate.dispatches();
        total_projections += candidate.fused_dispatches();
        total_sampler += candidate.native_greedy_dispatches();
        let result = candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
        total_readback += result.iter().map(|read| read.bytes).sum::<usize>();
        publication(&before, &sessions)?;
        let next = packets(result, &sessions, &selected)?;
        same_domains(&sessions, &expected_states).await?;
        if step == 1 {
            let current = roots(&sessions);
            let last = count - 1;
            let previous = device_packets[last]
                .qualification_producer()
                .ok_or_else(|| Error::State("previous greedy packet has no producer".into()))?;
            if previous.session_epoch() != current[last].session_epoch()
                || previous.root() >= current[last].root()
                || previous.session() != current[last].session()
                || previous.source() != current[last].source()
            {
                return Err(Error::State(
                    "greedy stale-root fixture is not same-epoch".into(),
                ));
            }
            let bytes = resident.qualification().resident_bytes()?;
            let pending = resident.qualification().pending_events()?;
            let mut mixed = next.clone();
            mixed[last] = device_packets[last].clone();
            if feedback(&resident, &sessions, &mixed).await.is_ok() {
                return Err(Error::State(
                    "same-epoch previous-root greedy packet accepted".into(),
                ));
            }
            unchanged(&sessions, &current)?;
            same_domains(&sessions, &expected_states).await?;
            quiet(&resident, bytes)?;
            if resident.qualification().pending_events()? != pending {
                return Err(Error::State(
                    "same-epoch packet refusal changed pending events".into(),
                ));
            }
        }
        if step == 5
            && feedback(&resident, &sessions, &device_packets)
                .await
                .is_ok()
        {
            return Err(Error::State(
                "greedy root advance accepted previous packets".into(),
            ));
        }
        device_packets = next;
        next_ids = selected.iter().map(|&(id, _)| id).collect();
        if resident.qualification().uploaded_value_bytes() != uploads {
            return Err(Error::State(
                "native greedy/feedback uploaded warm values".into(),
            ));
        }
        let bytes = *stable_bytes.get_or_insert(resident.qualification().resident_bytes()?);
        quiet(&resident, bytes)?;
    }
    let bytes_with_packets =
        stable_bytes.ok_or_else(|| Error::State("missing greedy live admission".into()))?;
    drop(device_packets);
    drop(sessions);
    drop(shadows);
    drop(b1);
    quiet(
        &resident,
        bytes_with_packets - count * (12 + 3 * resident.state_bytes()),
    )?;
    identical_packet_bindings(&resident, &source).await?;
    // Explicit fixture uploads begin only after the source-model zero-upload gate.
    selector_fixtures(&resident, count, eos).await?;
    let persistent = resident.qualification().resident_bytes()?;
    if persistent < original_bytes {
        return Err(Error::State("greedy lost original resident owner".into()));
    }
    let key = resident.qualification().physical_key();
    drop(resident);
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != 0 || budget.fenced {
        return Err(Error::State("greedy final physical envelope leaked".into()));
    }
    println!(
        "P12_FUSED_GREEDY_GPU_PASS B={count} source_steps=6 host_seed_steps=2 device_feedback_steps=4 \
         CPU_B1_native_full248320_logits=bit_exact selected_ID_max_bits=exact all42domains=exact \
         compact_readback_bytes={total_readback} bytes_per_cohort={} physical_dispatches={total_dispatches} \
         native_projection_dispatches={total_projections} native_sampler_dispatches={total_sampler} \
         swapped_duplicate_foreign_stale_restore_packets=refused same_epoch_old_root=refused \
         host_metadata=ignored_by_device_feedback \
         abandoned_reordered_stale_proposals=unpublished genuine_EOS_input=source_exact \
         private_packet_state_leases=reclaimed final_reserved_bytes=0 HTTP_fused_serving=not_qualified",
        4 + 12 * count,
    );
    Ok(())
}

async fn identical_packet_bindings(
    resident: &Arc<ResidentModel>,
    source: &Arc<LoadedQwenModel>,
) -> Result<()> {
    let mut sessions = Vec::with_capacity(2);
    let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(source))?;
    let expected = greedy(&cpu.prefill_transaction(&[11, 353, 17])?)?;
    for _ in 0..2 {
        let mut session = resident.session().await?;
        session.prefill_without_logits(&[11, 353], false).await?;
        sessions.push(session);
    }
    let candidate = host(resident, &sessions, &[17, 17]).await?;
    private_receipt(&candidate, 2, 2)?;
    let result = candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
    let mut tokens = packets(result, &sessions, &[expected, expected])?;
    let cpu_state = cpu.snapshot().domain_values();
    for session in &sessions {
        compare_domains(&cpu_state, &session.snapshot().read_domains().await?)?;
    }
    let before = roots(&sessions);
    let state = domains(&sessions).await?;
    let bytes = resident.qualification().resident_bytes()?;
    tokens.swap(0, 1);
    if feedback(resident, &sessions, &tokens).await.is_ok() {
        return Err(Error::State(
            "identical-ID foreign producer packet accepted".into(),
        ));
    }
    unchanged(&sessions, &before)?;
    same_domains(&sessions, &state).await?;
    quiet(resident, bytes)?;
    drop(tokens);
    drop(sessions);
    quiet(resident, bytes - 2 * (resident.state_bytes() + 12))?;
    Ok(())
}

/// Separate-process unknown completion proof includes original input/output packets.
#[cfg(feature = "hardware-abort-probe")]
pub async fn qualify_abort(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("greedy abort source path".into()))?;
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
    let mut selected = Vec::with_capacity(2);
    let mut expected = Vec::with_capacity(2);
    for (history, seed) in [([11, 353], 55), ([17, 42], 78)] {
        let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&source))?;
        let mut session = resident.session().await?;
        exact(
            &cpu.prefill_transaction(&history)?,
            &session.forward(&history, false).await?.logits,
            "greedy abort accepted CPU/B1 prefix",
        )?;
        selected.push(greedy(&cpu.forward(seed)?)?);
        expected.push(cpu.snapshot().domain_values());
        sessions.push(session);
    }
    let seed = host(&resident, &sessions, &[55, 78]).await?;
    private_receipt(&seed, 2, 2)?;
    let seed = seed.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
    let incoming = packets(seed, &sessions, &selected)?;
    same_domains(&sessions, &expected).await?;
    let before = roots(&sessions);
    let saved = sessions
        .iter()
        .map(ResidentSession::snapshot)
        .collect::<Vec<_>>();
    let stale_ready = feedback(&resident, &sessions, &incoming).await?;
    private_receipt(&stale_ready, 2, 3)?;
    unchanged(&sessions, &before)?;
    let quarantined_before = quarantined_handles_for(&client)?;
    resident.qualification_reset_launch_seen();
    resident.qualification_arm_abort(1);
    let mut pending = Box::pin(feedback(&resident, &sessions, &incoming));
    loop {
        if let Some(completed) = futures_lite::future::poll_once(pending.as_mut()).await {
            return Err(Error::State(format!(
                "greedy feedback abort missed launch hook: success={}",
                completed.is_ok()
            )));
        }
        if resident.qualification_launch_seen() != 0 {
            break;
        }
        futures_lite::future::yield_now().await;
    }
    drop(pending);
    let submitted = resident.qualification_launch_seen();
    let quarantined = quarantined_handles_for(&client)?;
    if submitted == 0
        || quarantined.saturating_sub(quarantined_before) < submitted
        || resident.qualification().ensure_usable().is_ok()
        || stale_ready
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
    {
        return Err(Error::State(
            "greedy abort original packet/state retention".into(),
        ));
    }
    unchanged(&sessions, &before)?;
    if feedback(&resident, &sessions, &incoming).await.is_ok()
        || host(&resident, &sessions, &[11, 353]).await.is_ok()
        || QualificationService::new(
            device,
            source.source_identity(),
            ResourceOwner::new("other-model", "after-greedy-feedback-abort")?,
            1024 * 1024,
            device_cap,
        )
        .is_ok()
    {
        return Err(Error::State(
            "greedy feedback fenced card admitted work/model".into(),
        ));
    }
    unchanged(&sessions, &before)?;
    // Diagnostic physical reads do not retire unknown events or permit reuse.
    client
        .sync()
        .await
        .map_err(|error| Error::State(format!("greedy abort diagnostic barrier: {error}")))?;
    for (snapshot, expected) in saved.iter().zip(&expected) {
        abort_prior_bits(&client, snapshot, expected).await?;
    }
    qualify_preserved_packets(&client, &incoming, &selected).await?;
    let key = resident.qualification().physical_key();
    let weak = resident.qualification_witness();
    let cap = resident.qualification().capacity() as u64;
    drop(incoming);
    drop(saved);
    drop(sessions);
    drop(resident);
    let retained = weak
        .upgrade()
        .ok_or_else(|| Error::State("greedy abort lost charged owner".into()))?;
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != cap
        || !budget.fenced
        || retained.physical_remaining()? != device_cap - cap
        || retained.ensure_usable().is_ok()
        || quarantined_handles_for(&client)? < quarantined
    {
        return Err(Error::State(
            "greedy abort physical envelope/quarantine released".into(),
        ));
    }
    println!(
        "P12_FUSED_GREEDY_FEEDBACK_ABORT_PASS B=2 prior_frontier=3 \
        actual_model_and_native_sampler_launch=true completion=unknown \
        submitted_original_handles={submitted} quarantined_original_handles={quarantined} \
        incoming_packet_bits=exact all42_prior_domains_per_row=bit_exact \
        canonical_future_KV_zero=exact roots_unchanged=true old_ready=refused \
        further_work_other_model=refused retained_envelope_bytes={cap} trusted_recovery=false"
    );
    Ok(())
}
