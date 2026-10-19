//! Native teacher-token state-only proof; no generated output is requested.
//! Mounted beneath the accepted fused qualifier to reuse its exact raw checks.
use super::*;
use zerogravity_core::resource_arbiter::ResourceArbiter;
use zerogravity_model_qwen35::{QwenDenseCpu, tokenize::QwenTokenizer};

const PROVIDER: usize = 4 * 1024 * 1024 * 1024;
const DEVICE_CAP: u64 = 5 * 1024 * 1024 * 1024;

async fn load(name: &str) -> Result<(Arc<LoadedQwenModel>, Arc<ResidentModel>, u32)> {
    if cfg!(debug_assertions) {
        panic!("physical native prefix qualification requires --release");
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/models")
        .join(name);
    let source = LoadedQwenModel::open(&path.join("model.gguf"), &path.join("config.json"))?;
    let eos = QwenTokenizer::from_file(&path.join("tokenizer.json"))?.eos();
    assert_eq!(source.geometry().vocab_size, 248_320);
    assert!(u64::from(eos) < source.geometry().vocab_size);
    let model = load_qualified(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        Arc::clone(&source),
        512,
        PROVIDER,
        DEVICE_CAP,
    )
    .await?;
    Ok((source, model, eos))
}

async fn state(sessions: &[ResidentSession], cpu: &[QwenDenseCpu]) -> Result<()> {
    assert_eq!(sessions.len(), cpu.len());
    for (session, cpu) in sessions.iter().zip(cpu) {
        let expected = cpu.snapshot().domain_values();
        assert_eq!(expected.len(), 42);
        assert_eq!(session.binding().frontier(), cpu.position() as u64);
        compare_domains(&expected, &session.snapshot().read_domains().await?)?;
    }
    Ok(())
}

async fn teacher(
    model: &Arc<ResidentModel>,
    sessions: &[ResidentSession],
    ids: &[u32],
) -> Result<PreparedResidentCohort> {
    let rows = sessions
        .iter()
        .zip(ids)
        .map(|(s, &id)| (s, id))
        .collect::<Vec<_>>();
    model.prepare_state_fused_cohort(&rows).await
}

fn state_receipt(candidate: &PreparedResidentCohort, before: &[RootBinding]) {
    assert_eq!(candidate.len(), before.len());
    assert_eq!(
        candidate.prior_frontiers().collect::<Vec<_>>(),
        before
            .iter()
            .map(|root| root.frontier())
            .collect::<Vec<_>>()
    );
    assert!(candidate.dispatches() > candidate.fused_dispatches());
    assert!(candidate.fused_dispatches() > 0);
    assert_eq!(candidate.native_greedy_dispatches(), 0);
}

fn state_reads(reads: &[InferenceReadback], dispatches: usize) {
    assert!(!reads.is_empty());
    assert_eq!(reads.iter().map(|r| r.bytes).sum::<usize>(), 4);
    assert_eq!(reads.iter().map(|r| r.chunk_bytes).sum::<usize>(), 4);
    assert_eq!(
        reads.iter().map(|r| r.dispatches).sum::<usize>(),
        dispatches
    );
    for read in reads {
        assert_eq!(read.state_only_steps, 1);
        assert_eq!(read.logit_steps, 0);
        assert!(read.logits.is_empty());
        assert!(read.device_token.is_none());
    }
}

async fn reclaim(model: Arc<ResidentModel>) -> Result<()> {
    let service = Arc::clone(&model.service);
    let key = service.physical_key();
    assert_eq!(service.pending_events()?, 0);
    service.ensure_usable()?;
    drop(model);
    assert_eq!(service.resident_bytes()?, 0);
    assert_eq!(
        ResourceArbiter::process_shared().snapshot(&key)?.reserved,
        PROVIDER as u64
    );
    drop(service);
    let final_state = ResourceArbiter::process_shared().snapshot(&key)?;
    assert_eq!(final_state.reserved, 0);
    assert!(!final_state.fenced);
    Ok(())
}

async fn qualify_prefix(name: &str, count: usize) -> Result<()> {
    let (source, model, eos) = load(name).await?;
    let mut sessions = Vec::new();
    let mut single = Vec::new();
    let mut cpu = Vec::new();
    for row in 0..count {
        let mut reference = QwenDenseCpu::from_shared(Arc::clone(&source))?;
        let mut b1 = model.session().await?;
        let mut native = model.session().await?;
        // Real independent GPU prefixes, including unequal frontiers; no import.
        for step in 0..row % 3 {
            let id = IDS[(row + step) % IDS.len()];
            let expected = reference.forward(id)?;
            exact(
                &expected,
                &b1.forward(&[id], false).await?.logits,
                "prefix B1 seed",
            )?;
            exact(
                &expected,
                &native.forward(&[id], false).await?.logits,
                "prefix native seed",
            )?;
        }
        cpu.push(reference);
        single.push(b1);
        sessions.push(native);
    }
    state(&sessions, &cpu).await?;
    state(&single, &cpu).await?;
    let uploaded = model.service.uploaded_value_bytes();
    let mut state_dispatches = 0;
    let mut state_projections = 0;
    for step in 0..4 {
        let ids = (0..count)
            .map(|row| {
                if step == 2 {
                    eos
                } else {
                    IDS[(row + step + 2) % IDS.len()]
                }
            })
            .collect::<Vec<_>>();
        let before = bindings(&sessions);
        let original_domains = domain_values(&sessions).await?;
        let queue = model.service.queue().stats()?;
        let mut invalid = sessions
            .iter()
            .zip(&ids)
            .map(|(s, &id)| (s, id))
            .collect::<Vec<_>>();
        invalid.last_mut().unwrap().1 = u32::MAX;
        assert!(
            model
                .prepare_state_fused_cohort_admitted(&invalid)
                .await
                .is_err()
        );
        let duplicate = vec![(&sessions[0], ids[0]); count];
        assert!(
            model
                .prepare_state_fused_cohort_admitted(&duplicate)
                .await
                .is_err()
        );
        assert_eq!(model.service.queue().stats()?, queue);
        unchanged(&sessions, &before)?;
        unchanged_domains(&sessions, &original_domains).await?;
        if step == 0 {
            let abandoned = teacher(&model, &sessions, &ids).await?;
            state_receipt(&abandoned, &before);
            unchanged(&sessions, &before)?;
            drop(abandoned);
            unchanged_domains(&sessions, &original_domains).await?;
            let bytes = model.service.resident_bytes()?;
            drop(teacher(&model, &sessions, &ids).await?);
            idle(&model, bytes)?;
        }
        if step == 1 {
            let saved = sessions[count - 1].snapshot();
            let stale = teacher(&model, &sessions, &ids).await?;
            state_receipt(&stale, &before);
            sessions[count - 1].restore(&saved)?;
            let after_restore = bindings(&sessions);
            assert_ne!(
                before[count - 1].session_epoch(),
                after_restore[count - 1].session_epoch()
            );
            assert!(
                stale
                    .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
                    .is_err()
            );
            unchanged(&sessions, &after_restore)?;
            unchanged_domains(&sessions, &original_domains).await?;
            drop(saved);
        }
        // Independent CPU and accepted complete-vocabulary B1 reference per row.
        for ((reference, b1), &id) in cpu.iter_mut().zip(&mut single).zip(&ids) {
            let expected = reference.forward(id)?;
            exact(
                &expected,
                &b1.forward(&[id], false).await?.logits,
                "prefix each teacher full CPU/B1",
            )?;
        }
        state(&single, &cpu).await?;
        let before = bindings(&sessions);
        let candidate = teacher(&model, &sessions, &ids).await?;
        state_receipt(&candidate, &before);
        state_dispatches = candidate.dispatches();
        state_projections = candidate.fused_dispatches();
        unchanged(&sessions, &before)?;
        let reads = candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
        state_reads(&reads, state_dispatches);
        for (session, root) in sessions.iter().zip(before) {
            assert_eq!(session.binding().session(), root.session());
            assert_eq!(session.binding().session_epoch(), root.session_epoch());
            assert_eq!(session.binding().root(), root.root() + 1);
            assert_eq!(session.binding().frontier(), root.frontier() + 1);
        }
        state(&sessions, &cpu).await?;
        assert_eq!(model.service.pending_events()?, 0);
        assert_eq!(
            model.service.uploaded_value_bytes(),
            uploaded,
            "warm prefix uploaded values"
        );
    }
    // A later complete native forward proves skipped tail values were not used.
    let ids = (0..count)
        .map(|row| IDS[(row + 6) % IDS.len()])
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for ((reference, b1), &id) in cpu.iter_mut().zip(&mut single).zip(&ids) {
        let logits = reference.forward(id)?;
        exact(
            &logits,
            &b1.forward(&[id], false).await?.logits,
            "prefix future CPU/B1 logits",
        )?;
        expected.push(logits);
    }
    let rows = sessions
        .iter()
        .zip(&ids)
        .map(|(s, &id)| (s, id))
        .collect::<Vec<_>>();
    let candidate = model.prepare_logits_fused_cohort(&rows).await?;
    assert!(candidate.dispatches() > state_dispatches);
    assert!(candidate.fused_dispatches() > state_projections);
    let reads = candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
    assert_eq!(reads.len(), count, "future full-vocabulary row coverage");
    for (read, expected) in reads.iter().zip(&expected) {
        assert_eq!(read.state_only_steps, 0);
        assert_eq!(read.logit_steps, 1);
        assert_eq!(expected.len(), 248_320);
        exact(
            expected,
            &read.logits,
            "prefix future full native vocabulary",
        )?;
    }
    state(&sessions, &cpu).await?;
    state(&single, &cpu).await?;
    drop(reads);
    drop(expected);
    let ids = (0..count)
        .map(|row| IDS[(row + 1) % IDS.len()])
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    for ((reference, b1), &id) in cpu.iter_mut().zip(&mut single).zip(&ids) {
        let logits = reference.forward(id)?;
        let prior = b1.snapshot();
        exact(
            &logits,
            &b1.forward(&[id], false).await?.logits,
            "prefix later greedy full CPU/B1",
        )?;
        let index = (0..logits.len())
            .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then_with(|| b.cmp(a)))
            .unwrap();
        let expected = (index as u32, logits[index].to_bits());
        b1.restore(&prior)?;
        let read = b1.forward_greedy(&[id]).await?;
        let packet = read.device_token.unwrap();
        assert_eq!((packet.id, packet.max_bits), expected);
        selected.push(expected);
    }
    let rows = sessions
        .iter()
        .zip(&ids)
        .map(|(s, &id)| (s, id))
        .collect::<Vec<_>>();
    let candidate = model.prepare_greedy_fused_cohort(&rows).await?;
    assert_eq!(candidate.native_greedy_dispatches(), 4);
    let reads = candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
    assert_eq!(reads.len(), count, "future greedy row coverage");
    for ((read, session), expected) in reads.iter().zip(&sessions).zip(selected) {
        let packet = read.device_token.as_ref().unwrap();
        assert!(read.logits.is_empty());
        assert_eq!((packet.id, packet.max_bits), expected);
        assert_eq!(packet.producer, Some(session.binding()));
    }
    state(&sessions, &cpu).await?;
    state(&single, &cpu).await?;
    drop(reads);
    drop(cpu);
    drop(single);
    drop(sessions);
    reclaim(model).await?;
    println!(
        "P12_NATIVE_PREFIX_PASS source={name} B={count} teacher_steps_per_row=4 teacher_EOS=true unequal_frontiers=true each_all42=CPU_B1_exact state_only_status_bytes=4 sampler=0 future_full248320=exact abandoned_stale_invalid=unchanged final_reserved=0"
    );
    Ok(())
}

async fn pressure_advance(
    model: &Arc<ResidentModel>,
    sessions: &mut [ResidentSession],
    single: &mut [ResidentSession],
    cpu: &mut [QwenDenseCpu],
    ids: &[u32],
) -> Result<()> {
    for ((reference, b1), &id) in cpu.iter_mut().zip(single.iter_mut()).zip(ids) {
        exact(
            &reference.forward(id)?,
            &b1.forward(&[id], false).await?.logits,
            "prefix pressure CPU/B1 fullV",
        )?;
    }
    let before = bindings(sessions);
    let candidate = teacher(model, sessions, ids).await?;
    state_receipt(&candidate, &before);
    let dispatches = candidate.dispatches();
    state_reads(
        &candidate.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?,
        dispatches,
    );
    state(sessions, cpu).await?;
    state(single, cpu).await?;
    Ok(())
}

async fn qualify_prefix_pressure(name: &str) -> Result<()> {
    let (source, model, _) = load(name).await?;
    let mut sessions = Vec::new();
    let mut single = Vec::new();
    let mut cpu = Vec::new();
    for _ in 0..8 {
        sessions.push(model.session().await?);
        single.push(model.session().await?);
        cpu.push(QwenDenseCpu::from_shared(Arc::clone(&source))?);
    }
    let (need4, need8) = {
        let workspace = model.workspace.lock().await;
        assert!(workspace.batch.is_none());
        let required = |rows| -> Result<usize> {
            BatchProgram::preflight(&workspace.program, rows)?
                .owned_bytes()?
                .checked_add(model.state_bytes * rows)
                .ok_or_else(|| Error::Limit("prefix pressure footprint".into()))
        };
        (required(4)?, required(8)?)
    };
    let available = model.service.remaining_capacity()?;
    assert!(need4 < need8 && need8 <= available);
    let pressure = model
        .service
        .allocate_uninitialized(available - need4, BufferRole::Scratch)?;
    assert_eq!(model.service.remaining_capacity()?, need4);
    let before = bindings(&sessions);
    let original = domain_values(&sessions).await?;
    let bytes = model.service.resident_bytes()?;
    let uploads = model.service.uploaded_value_bytes();
    let queue = model.service.queue().stats()?;
    let rows = sessions.iter().zip(IDS).collect::<Vec<_>>();
    assert!(
        model
            .prepare_state_fused_cohort_admitted(&rows)
            .await?
            .is_none()
    );
    assert!(matches!(
        model.prepare_state_fused_cohort(&rows).await,
        Err(Error::Limit(_))
    ));
    unchanged(&sessions, &before)?;
    assert_eq!(model.service.queue().stats()?, queue);
    unchanged_domains(&sessions, &original).await?;
    assert_eq!(model.service.resident_bytes()?, bytes);
    assert_eq!(model.service.uploaded_value_bytes(), uploads);
    assert_eq!(model.service.pending_events()?, 0);
    assert_eq!(
        model.workspace.lock().await.batch.as_ref().map(|b| b.rows),
        None
    );
    pressure_advance(
        &model,
        &mut sessions[..4],
        &mut single[..4],
        &mut cpu[..4],
        &IDS[..4],
    )
    .await?;
    pressure_advance(
        &model,
        &mut sessions[4..],
        &mut single[4..],
        &mut cpu[4..],
        &IDS[4..],
    )
    .await?;
    let before = bindings(&sessions);
    let bytes = model.service.resident_bytes()?;
    let queue = model.service.queue().stats()?;
    let rows = sessions.iter().zip(IDS).collect::<Vec<_>>();
    assert!(
        model
            .prepare_state_fused_cohort_admitted(&rows)
            .await?
            .is_none()
    );
    unchanged(&sessions, &before)?;
    assert_eq!(model.service.resident_bytes()?, bytes);
    assert_eq!(model.service.queue().stats()?, queue);
    assert_eq!(
        model.workspace.lock().await.batch.as_ref().map(|b| b.rows),
        Some(4)
    );
    drop(pressure);
    pressure_advance(&model, &mut sessions, &mut single, &mut cpu, &IDS).await?;
    assert_eq!(
        model.workspace.lock().await.batch.as_ref().map(|b| b.rows),
        Some(8)
    );
    drop(cpu);
    drop(single);
    drop(sessions);
    reclaim(model).await?;
    println!(
        "P12_NATIVE_PREFIX_PRESSURE_PASS source={name} actual_scratch_lease=true B8_refusal_before_launch=true B4_under_pressure_B8_after_release=CPU_B1_all42_exact state_only_sampler=0 final_reserved=0"
    );
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
async fn qualify_prefix_abort(name: &str) -> Result<()> {
    use std::sync::atomic::Ordering;
    let (source, model, _) = load(name).await?;
    let mut sessions = Vec::new();
    for history in [[11, 353], [17, 42]] {
        let mut cpu = QwenDenseCpu::from_shared(Arc::clone(&source))?;
        let mut session = model.session().await?;
        exact(
            &cpu.prefill_transaction(&history)?,
            &session.forward(&history, false).await?.logits,
            "prefix abort prior",
        )?;
        compare_domains(
            &cpu.snapshot().domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
        sessions.push(session);
    }
    let before = bindings(&sessions);
    let saved = sessions
        .iter()
        .map(ResidentSession::snapshot)
        .collect::<Vec<_>>();
    let expected = domain_values(&sessions).await?;
    let ready = teacher(&model, &sessions, &[55, 78]).await?;
    state_receipt(&ready, &before);
    let client = model.service.client().clone();
    let initial_quarantine = crate::physical_quarantine::quarantined_handles_for(&client)?;
    model.launch_seen.store(0, Ordering::Release);
    model.abort_after_launch.store(1, Ordering::Release);
    let rows = [(&sessions[0], 101), (&sessions[1], 135)];
    let mut pending = Box::pin(model.prepare_state_fused_cohort(&rows));
    loop {
        if let Some(result) = futures_lite::future::poll_once(pending.as_mut()).await {
            return Err(Error::State(format!(
                "prefix abort reached completion before hook: success={}",
                result.is_ok()
            )));
        }
        if model.launch_seen.load(Ordering::Acquire) != 0 {
            break;
        }
        futures_lite::future::yield_now().await;
    }
    drop(pending);
    let handles = model.launch_seen.load(Ordering::Acquire);
    let quarantine = crate::physical_quarantine::quarantined_handles_for(&client)?;
    assert!(handles > 0 && quarantine >= initial_quarantine + handles);
    assert!(model.service.ensure_usable().is_err());
    assert!(
        ready
            .commit(&mut sessions.iter_mut().collect::<Vec<_>>())
            .is_err()
    );
    unchanged(&sessions, &before)?;
    assert!(teacher(&model, &sessions, &[55, 78]).await.is_err());
    assert!(sessions[0].prepare(&[55], false).await.is_err());
    assert!(
        ResidentService::new(
            Device::Cuda(cubecl::device::CudaDevice::new(0)),
            source.source_identity(),
            ResourceOwner::new("other-model", "after-prefix-abort")?,
            1024 * 1024,
            DEVICE_CAP,
        )
        .is_err()
    );
    client
        .sync()
        .await
        .map_err(|e| Error::State(format!("prefix abort diagnostic barrier: {e}")))?;
    for (snapshot, domains) in saved.iter().zip(&expected) {
        physical_prior_after_abort(&client, snapshot, domains).await?;
    }
    unchanged(&sessions, &before)?;
    let service = Arc::clone(&model.service);
    let key = service.physical_key();
    drop(saved);
    drop(sessions);
    drop(model);
    assert!(service.ensure_usable().is_err());
    let budget = ResourceArbiter::process_shared().snapshot(&key)?;
    assert_eq!(budget.reserved, PROVIDER as u64);
    assert!(budget.fenced);
    assert!(crate::physical_quarantine::quarantined_handles_for(&client)? >= quarantine);
    println!(
        "P12_NATIVE_PREFIX_ABORT_PASS source={name} B=2 actual_state_only_launch=true roots_all42_prior=unchanged original_handles={handles} quarantine_retained=true service_fenced=true provider_cap_retained=true recovery=false"
    );
    Ok(())
}

macro_rules! case {
    ($test:ident, $name:literal, $rows:literal) => {
        #[test]
        #[ignore = "release original CUDA prefix; run serially"]
        fn $test() -> Result<()> {
            futures_lite::future::block_on(qualify_prefix($name, $rows))
        }
    };
}
case!(original_qwen35_08b_native_prefix_b2, "qwen35-0.8b", 2);
case!(original_qwen35_08b_native_prefix_b4, "qwen35-0.8b", 4);
case!(original_qwen35_08b_native_prefix_b8, "qwen35-0.8b", 8);
case!(original_qwen35_2b_native_prefix_b2, "qwen35-2b", 2);
case!(original_qwen35_2b_native_prefix_b4, "qwen35-2b", 4);
case!(original_qwen35_2b_native_prefix_b8, "qwen35-2b", 8);
#[cfg(feature = "hardware-abort-probe")]
#[test]
#[ignore = "release unknown CUDA prefix completion; isolated process"]
fn original_qwen35_08b_native_prefix_abort() -> Result<()> {
    futures_lite::future::block_on(qualify_prefix_abort("qwen35-0.8b"))
}
#[cfg(feature = "hardware-abort-probe")]
#[test]
#[ignore = "release unknown CUDA prefix completion; isolated process"]
fn original_qwen35_2b_native_prefix_abort() -> Result<()> {
    futures_lite::future::block_on(qualify_prefix_abort("qwen35-2b"))
}

#[test]
#[ignore = "release actual CUDA native prefix lease pressure; serial"]
fn original_qwen35_08b_native_prefix_pressure() -> Result<()> {
    futures_lite::future::block_on(qualify_prefix_pressure("qwen35-0.8b"))
}
#[test]
#[ignore = "release actual CUDA native prefix lease pressure; serial"]
fn original_qwen35_2b_native_prefix_pressure() -> Result<()> {
    futures_lite::future::block_on(qualify_prefix_pressure("qwen35-2b"))
}
