//! Qwen host/reference driver over the original physical probes.
use super::graph::loaded;
use super::*;
use zerogravity_gpu::qualification::resident::graph::{idle, roots, states, unchanged};
use zerogravity_gpu::qualification::resident::graph_wide::*;
use zerogravity_model_qwen35::tokenize::QwenTokenizer;

async fn reference_rows(
    model: &Arc<ResidentModel>,
    cpus: &mut [QwenDenseCpu],
    histories: &[Vec<u32>],
    tokens: &[u32],
    readout: Readout,
) -> Result<Vec<Vec<f32>>> {
    if cpus.len() != histories.len() || cpus.len() != tokens.len() {
        return Err(Error::State("wide graph oracle coverage".into()));
    }
    let mut expected = Vec::with_capacity(tokens.len());
    for ((cpu, history), &token) in cpus.iter_mut().zip(histories).zip(tokens) {
        let logits = cpu.forward(token)?;
        let uploads = model.qualification().uploaded_value_bytes();
        let mut b1 = model.session().await?;
        if model
            .qualification()
            .uploaded_value_bytes()
            .checked_sub(uploads)
            != Some(model.state_bytes() as u64)
        {
            return Err(Error::State(
                "wide graph independent B1 state upload".into(),
            ));
        }
        b1.forward(history, false).await?;
        let before = b1.snapshot();
        let got = b1.forward(&[token], false).await?;
        exact(
            &logits,
            &got.logits,
            "wide graph accepted independent B1 fullV",
        )?;
        let state = cpu.snapshot().domain_values();
        if state.len() != 42 {
            return Err(Error::State("wide graph CPU all42 coverage".into()));
        }
        compare_domains(&state, &b1.snapshot().read_domains().await?)?;
        if matches!(readout, Readout::Greedy) {
            b1.restore(&before)?;
            let greedy = b1.forward_greedy(&[token]).await?;
            if !greedy.logits.is_empty() || greedy.bytes != 16 {
                return Err(Error::State(
                    "wide graph independent B1 greedy packet geometry".into(),
                ));
            }
            packet(&greedy, &b1, selected(&logits)?)?;
            compare_domains(&state, &b1.snapshot().read_domains().await?)?;
            drop(greedy);
        }
        drop(before);
        if model.qualification().uploaded_value_bytes() != uploads + model.state_bytes() as u64 {
            return Err(Error::State("wide graph B1 warm value upload".into()));
        }
        expected.push(logits);
    }
    Ok(expected)
}

async fn teacher_step(
    model: &Arc<ResidentModel>,
    sessions: &mut [ResidentSession],
    cpus: &mut [QwenDenseCpu],
    histories: &mut [Vec<u32>],
    tokens: &[u32],
    graph: &Arc<CapturedResidentGraph>,
    policy: TeacherPolicy,
) -> Result<usize> {
    let TeacherPolicy { readout, reverse } = policy;
    let count = sessions.len();
    let expected = reference_rows(model, cpus, histories, tokens, readout).await?;
    let uploads = model.qualification().uploaded_value_bytes();
    let prior = states(sessions).await?;
    let snapshots = sessions
        .iter()
        .map(ResidentSession::snapshot)
        .collect::<Vec<_>>();
    let order = if reverse {
        (0..count).rev().collect::<Vec<_>>()
    } else {
        (0..count).collect::<Vec<_>>()
    };
    let before_frontiers = sessions
        .iter()
        .map(|s| s.binding().frontier())
        .collect::<Vec<_>>();
    // SAME original native cohort/publisher, followed by checked authority
    // restore. No second B-shaped GPU lane or invented checkpoint/root.
    let native = match readout {
        Readout::None => {
            model
                .prepare_state_fused_cohort(&rows(sessions, tokens, &order)?)
                .await?
        }
        Readout::Logits => {
            model
                .prepare_logits_fused_cohort(&rows(sessions, tokens, &order)?)
                .await?
        }
        Readout::Greedy => {
            model
                .prepare_greedy_fused_cohort(&rows(sessions, tokens, &order)?)
                .await?
        }
    };
    candidate_shape(&native, count, readout)?;
    let native_dispatches = native.dispatches();
    let native_dots = native.fused_dispatches();
    let reads = native.commit(&mut ordered_sessions(sessions, &order)?)?;
    if reads.len() != count {
        return Err(Error::State(
            "wide graph native committed row coverage".into(),
        ));
    }
    for (index, &row) in order.iter().enumerate() {
        match readout {
            Readout::Logits => exact(
                &expected[row],
                &reads[index].logits,
                "wide graph ordinary native fullV",
            )?,
            Readout::Greedy => packet(&reads[index], &sessions[row], selected(&expected[row])?)?,
            Readout::None => {}
        }
        compare_domains(
            &cpus[row].snapshot().domain_values(),
            &sessions[row].snapshot().read_domains().await?,
        )?;
    }
    if matches!(readout, Readout::Greedy) {
        let raw = completed_logits(model, sessions[0].binding().root(), count).await?;
        for (index, &row) in order.iter().enumerate() {
            exact(
                &expected[row],
                &raw[index * 248_320..(index + 1) * 248_320],
                "wide graph ordinary native greedy diagnostic fullV",
            )?;
        }
    }
    let native_states = states(sessions).await?;
    drop(reads);
    for ((session, snapshot), expected_prior) in sessions.iter_mut().zip(&snapshots).zip(&prior) {
        session.restore(snapshot)?;
        compare_domains(expected_prior, &session.snapshot().read_domains().await?)?;
    }
    drop(snapshots);
    if sessions
        .iter()
        .map(|s| s.binding().frontier())
        .ne(before_frontiers.iter().copied())
    {
        return Err(Error::State("wide graph restore frontier changed".into()));
    }
    let bytes = model.qualification().resident_bytes()?;
    let candidate = model
        .prepare_graph_readout(&rows(sessions, tokens, &order)?, graph, readout)
        .await?;
    candidate_shape(&candidate, count, readout)?;
    if candidate.dispatches() != native_dispatches
        || candidate.fused_dispatches() != native_dots
        || candidate
            .prior_frontiers()
            .ne(order.iter().map(|&row| sessions[row].binding().frontier()))
    {
        return Err(Error::State(
            "wide graph exact body/outside/independent frontier receipt".into(),
        ));
    }
    let kernels = candidate.dispatches();
    let reads = candidate.commit(&mut ordered_sessions(sessions, &order)?)?;
    if reads.len() != count {
        return Err(Error::State("wide graph committed row omission".into()));
    }
    for (index, &row) in order.iter().enumerate() {
        match readout {
            Readout::Logits => exact(
                &expected[row],
                &reads[index].logits,
                "wide graph actual graph fullV",
            )?,
            Readout::Greedy => packet(&reads[index], &sessions[row], selected(&expected[row])?)?,
            Readout::None => {}
        }
        compare_domains(
            &cpus[row].snapshot().domain_values(),
            &sessions[row].snapshot().read_domains().await?,
        )?;
        compare_domains(
            &native_states[row],
            &sessions[row].snapshot().read_domains().await?,
        )?;
    }
    if matches!(readout, Readout::Greedy) {
        let raw = completed_logits(model, sessions[0].binding().root(), count).await?;
        for (index, &row) in order.iter().enumerate() {
            exact(
                &expected[row],
                &raw[index * 248_320..(index + 1) * 248_320],
                "wide graph actual greedy diagnostic fullV",
            )?;
        }
    }
    drop(reads);
    for (history, &token) in histories.iter_mut().zip(tokens) {
        history.push(token);
    }
    if model.qualification().uploaded_value_bytes() != uploads {
        return Err(Error::State(
            "wide graph warm native/graph uploaded values".into(),
        ));
    }
    idle(model, bytes)?;
    Ok(kernels)
}

pub async fn qualify(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    count: usize,
) -> Result<()> {
    if !matches!(count, 2 | 4 | 8) {
        return Err(Error::Unsupported("wide graph rows must be 2/4/8".into()));
    }
    let (source, model) = loaded(device, path, provider_cap, device_cap).await?;
    let key = model.qualification().physical_key();
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("wide graph tokenizer path".into()))?;
    let eos = QwenTokenizer::from_file(&parent.join("tokenizer.json"))?.eos();
    if u64::from(eos) >= model.vocabulary() {
        return Err(Error::State("wide graph original trained EOS input".into()));
    }
    let mut cpus = Vec::with_capacity(count);
    let mut sessions = Vec::with_capacity(count);
    let mut histories = Vec::with_capacity(count);
    for row in 0..count {
        let history = (0..2 + row % 3)
            .map(|at| IDS[(row + at) % 8])
            .collect::<Vec<_>>();
        let mut cpu = QwenDenseCpu::from_shared(source.clone())?;
        let expected = cpu.prefill_transaction(&history)?;
        let mut session = model.session().await?;
        exact(
            &expected,
            &session.forward(&history, false).await?.logits,
            "wide graph distinct real histories",
        )?;
        compare_domains(
            &cpu.snapshot().domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
        cpus.push(cpu);
        sessions.push(session);
        histories.push(history);
    }
    let tokens = (0..count).map(|row| IDS[(row + 3) % 8]).collect::<Vec<_>>();
    let order = (0..count).collect::<Vec<_>>();
    let prefix = model
        .capture_body(
            &rows(&sessions, &tokens, &order)?,
            ResidentOutput::StateOnly,
        )
        .await?;
    if prefix.output() != ResidentOutput::StateOnly || prefix.qualification_rows() != count {
        return Err(Error::State("wide graph typed StateOnly body key".into()));
    }
    let before = roots(&sessions);
    let original = states(&sessions).await?;
    let bytes = model.qualification().resident_bytes()?;
    if model
        .prepare_graph_readout(&rows(&sessions, &tokens, &order)?, &prefix, Readout::Logits)
        .await
        .is_ok()
        || model
            .prepare_graph_readout(&rows(&sessions, &tokens, &order)?, &prefix, Readout::Greedy)
            .await
            .is_ok()
        || model
            .prepare_graph_readout(
                &rows(
                    &sessions[..count / 2],
                    &tokens[..count / 2],
                    &(0..count / 2).collect::<Vec<_>>(),
                )?,
                &prefix,
                Readout::None,
            )
            .await
            .is_ok()
        || prefix.replay_enqueues() != 0
    {
        return Err(Error::State(
            "wide graph typed StateOnly/readout/band refusal".into(),
        ));
    }
    unchanged(&sessions, &before, &original).await?;
    idle(&model, bytes)?;
    pressure_and_abandon(&model, &mut sessions, &prefix, Readout::None).await?;
    let prefix_receipt = prefix.receipt();
    let mut kernel_total = 0;
    for step in 0..2 {
        let mut tokens = (0..count)
            .map(|row| IDS[(row + step + 4) % 8])
            .collect::<Vec<_>>();
        if step == 1 {
            tokens[count - 1] = eos;
        } // Genuine tokenizer EOS is a teacher, not generated output.
        kernel_total += teacher_step(
            &model,
            &mut sessions,
            &mut cpus,
            &mut histories,
            &tokens,
            &prefix,
            TeacherPolicy {
                readout: Readout::None,
                reverse: step % 2 != 0,
            },
        )
        .await?;
    }
    if prefix.replay_enqueues() != 4 {
        return Err(Error::State(
            "wide graph StateOnly host enqueue count".into(),
        ));
    }
    let prefix_enqueues = prefix.replay_enqueues();
    prefix.retire(sessions[0].binding().root()).await?;
    let graph = model
        .capture_body(&rows(&sessions, &tokens, &order)?, ResidentOutput::Logits)
        .await?;
    let logits_receipt = graph.receipt();
    let bytes = model.qualification().resident_bytes()?;
    let before = roots(&sessions);
    let prior = states(&sessions).await?;
    if model
        .prepare_graph_readout(&rows(&sessions, &tokens, &order)?, &graph, Readout::None)
        .await
        .is_ok()
        || graph.replay_enqueues() != 0
    {
        return Err(Error::State(
            "wide graph Logits body accepted StateOnly key".into(),
        ));
    }
    unchanged(&sessions, &before, &prior).await?;
    idle(&model, bytes)?;
    for (step, readout) in [Readout::Logits, Readout::Greedy].into_iter().enumerate() {
        pressure_and_abandon(&model, &mut sessions, &graph, readout).await?;
        let tokens = (0..count)
            .map(|row| IDS[(row + step + 6) % 8])
            .collect::<Vec<_>>();
        kernel_total += teacher_step(
            &model,
            &mut sessions,
            &mut cpus,
            &mut histories,
            &tokens,
            &graph,
            TeacherPolicy {
                readout,
                reverse: step % 2 == 0,
            },
        )
        .await?;
    }
    if graph.replay_enqueues() != 6 {
        return Err(Error::State(
            "wide graph Logits/Greedy host enqueue count".into(),
        ));
    }
    let logit_enqueues = graph.replay_enqueues();
    graph.retire(sessions[0].binding().root()).await?;
    model.qualification_release_graph_band().await?;
    drop(cpus);
    drop(sessions);
    drop(histories);
    drop(model);
    drop(source);
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != 0 || budget.fenced {
        return Err(Error::State(
            "wide graph final original resource ownership".into(),
        ));
    }
    println!(
        "P12_GRAPH_WIDE_PASS B={count} original=GGUF capacity=512 AV_padding=256 independent_CPU_B1_native=true all_vocab=248320 all_domains=42 state_only_teacher_steps=2 genuine_EOS_teacher=true prefix_generated_output=0 future_logits_steps=1 outside_greedy_steps=1 sampler_kernels=4 prefix_body_K={} logits_body_K={} private_warm_K={:?}/{:?} actual_graph_host_enqueues={prefix_enqueues}/{logit_enqueues} committed_kernel_sites={kernel_total} greedy_fullV=separate_readonly_diagnostic capture_global_barrier=UNQUALIFIED metadata_admission=UNQUALIFIED final_reserved=0",
        prefix_receipt.captured_kernels,
        logits_receipt.captured_kernels,
        prefix_receipt.warm_dispatches,
        logits_receipt.warm_dispatches
    );
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
pub async fn qualify_abort(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    count: usize,
    mode: &str,
) -> Result<()> {
    let readout = parse_mode(mode)?;
    if !matches!(count, 2 | 4 | 8) {
        return Err(Error::Unsupported("wide graph abort rows2/4/8".into()));
    }
    let (source, model) = loaded(device.clone(), path, provider_cap, device_cap).await?;
    let key = model.qualification().physical_key();
    let cap = model.qualification().capacity() as u64;
    let mut sessions = Vec::with_capacity(count);
    for row in 0..count {
        let history = (0..2 + row % 3)
            .map(|at| IDS[(row + at) % 8])
            .collect::<Vec<_>>();
        let mut cpu = QwenDenseCpu::from_shared(source.clone())?;
        let mut session = model.session().await?;
        exact(
            &cpu.prefill_transaction(&history)?,
            &session.forward(&history, false).await?.logits,
            "wide graph abort CPU/B1 prefix",
        )?;
        compare_domains(
            &cpu.snapshot().domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
        sessions.push(session);
    }
    let tokens = (0..count).map(|row| IDS[(row + 4) % 8]).collect::<Vec<_>>();
    let order = (0..count).collect::<Vec<_>>();
    let graph = model
        .capture_body(&rows(&sessions, &tokens, &order)?, body(readout))
        .await?;
    let bindings = roots(&sessions);
    let prior = states(&sessions).await?;
    let old_ready = match readout {
        Readout::None => {
            model
                .prepare_state_fused_cohort(&rows(&sessions, &tokens, &order)?)
                .await?
        }
        Readout::Logits => {
            model
                .prepare_logits_fused_cohort(&rows(&sessions, &tokens, &order)?)
                .await?
        }
        Readout::Greedy => {
            model
                .prepare_greedy_fused_cohort(&rows(&sessions, &tokens, &order)?)
                .await?
        }
    };
    let client = model.qualification().client().clone();
    let before = quarantined_handles_for(&client)?;
    let owners = quarantined_completion_owners_for(&client)?;
    let weak_graph = Arc::downgrade(&graph);
    let weak_service = model.qualification_witness();
    model.qualification_reset_launch_seen();
    model.qualification_arm_abort(1);
    let pending_rows = rows(&sessions, &tokens, &order)?;
    let mut future = Box::pin(model.prepare_graph_readout(&pending_rows, &graph, readout));
    loop {
        if let Some(result) = futures_lite::future::poll_once(future.as_mut()).await {
            return Err(Error::State(format!(
                "wide graph abort returned before actual pending: success={}",
                result.is_ok()
            )));
        }
        if model.qualification_launch_seen() > 0 {
            break;
        }
        futures_lite::future::yield_now().await;
    }
    if graph.replay_enqueues() != 1 {
        return Err(Error::State(
            "wide graph abort actual executable enqueue".into(),
        ));
    }
    drop(future);
    drop(pending_rows);
    let handles = model.qualification_launch_seen();
    let retained = quarantined_handles_for(&client)?;
    if handles == 0
        || retained.saturating_sub(before) < handles
        || roots(&sessions) != bindings
        || model.qualification().ensure_usable().is_ok()
        || prior.len() != count
        || prior.iter().any(|d| d.len() != 42)
        || old_ready
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_ok()
        || quarantined_completion_owners_for(&client)? <= owners
    {
        return Err(Error::State(
            "wide graph unknown completion ownership/publication".into(),
        ));
    }
    if sessions[0].prepare(&[11], false).await.is_ok()
        || model
            .prepare_graph_readout(&rows(&sessions, &tokens, &order)?, &graph, readout)
            .await
            .is_ok()
        || QualificationService::new(
            device,
            source.source_identity(),
            ResourceOwner::new("wide-graph-abort", "refused")?,
            1024 * 1024,
            device_cap,
        )
        .is_ok()
    {
        return Err(Error::State(
            "wide graph unknown service admitted work/model".into(),
        ));
    }
    drop(graph);
    drop(sessions);
    drop(model);
    drop(source);
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if weak_graph.upgrade().is_none()
        || weak_service.upgrade().is_none()
        || budget.reserved != cap
        || !budget.fenced
        || quarantined_handles_for(&client)? < retained
    {
        return Err(Error::State(
            "wide graph unknown executable/source/envelope dropped".into(),
        ));
    }
    println!(
        "P12_GRAPH_WIDE_ABORT_PASS B={count} mode={} actual_graph_enqueue=1 roots_unchanged=true prior_all42=observed_before_abort post_abort_readback=NOT_ATTEMPTED completion=unknown executable=retained driver_destroy=not_requested subsequent_model=refused retained_envelope={cap} trusted_recovery=false",
        mode_name(readout)
    );
    Ok(())
}

struct TeacherPolicy {
    readout: Readout,
    reverse: bool,
}
