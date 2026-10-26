//! P12 resident-session cohort integration over the ONE existing GPU
//! program and ResourceArbiter/SessionAuthority. Important: each row's
//! physically qualified forward still launches separately at this stage.
//! This probes a single original multi-root publication barrier, not a
//! batch-native fused model graph nor a vLLM performance claim.
use super::*;

pub async fn qualify(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    count: usize,
) -> Result<()> {
    if !matches!(count, 2 | 4 | 8) {
        return Err(Error::Limit(
            "physical resident cohort supports B2/B4/B8".into(),
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("resident cohort source path".into()))?;
    let model = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    let resident =
        load_qualified(device, Arc::clone(&model), 512, provider_cap, device_cap).await?;
    let mut left = resident.session().await?;
    let mut right = resident.session().await?;
    let inputs = [11u32, 353u32];
    let roots = [left.binding(), right.binding()];
    if roots[0].session() == roots[1].session() || roots[0].source() != roots[1].source() {
        return Err(Error::State(
            "GPU cohort needs separate sessions under one source".into(),
        ));
    }
    let mut cpus = [
        zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&model))?,
        zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&model))?,
    ];
    let expected = [
        cpus[0].prefill_transaction(&[inputs[0]])?,
        cpus[1].prefill_transaction(&[inputs[1]])?,
    ];
    let uploaded = resident.service.uploaded_value_bytes();
    // The already completed private state must not become a root merely
    // because the model source has executed physical GPU work.
    let abandoned = resident
        .prepare_logits_cohort(&[(&left, inputs[0]), (&right, inputs[1])])
        .await?;
    if abandoned.len() != 2
        || abandoned.prior_frontier() != Some(0)
        || abandoned.source() != resident.service.source_identity()
        || [left.binding(), right.binding()] != roots
    {
        return Err(Error::State(
            "tentative resident cohort published state".into(),
        ));
    }
    drop(abandoned);
    if [left.binding(), right.binding()] != roots {
        return Err(Error::State(
            "dropping prepared batch published state".into(),
        ));
    }
    if resident
        .prepare_logits_cohort(&[(&left, inputs[0]), (&left, inputs[1])])
        .await
        .is_ok()
    {
        return Err(Error::State(
            "duplicate resident cohort session accepted".into(),
        ));
    }
    if resident
        .prepare_logits_cohort(&[(&left, u32::MAX), (&right, inputs[1])])
        .await
        .is_ok()
        || [left.binding(), right.binding()] != roots
    {
        return Err(Error::State(
            "invalid batch input crossed publication barrier".into(),
        ));
    }
    let cohort = resident
        .prepare_logits_cohort(&[(&left, inputs[0]), (&right, inputs[1])])
        .await?;
    if [left.binding(), right.binding()] != roots {
        return Err(Error::State(
            "prepared cohort mutated committed root".into(),
        ));
    }
    let readback = cohort.commit(&mut [&mut left, &mut right])?;
    if readback.len() != 2
        || left.binding().root() != 1
        || right.binding().root() != 1
        || left.binding().frontier() != 1
        || right.binding().frontier() != 1
        || left.binding().session() == right.binding().session()
    {
        return Err(Error::State(
            "GPU B2 cohort publication lost root isolation".into(),
        ));
    }
    for ((actual, want), session) in readback.iter().zip(&expected).zip([&left, &right]) {
        exact(want, &actual.logits, "GPU cohort complete source logits")?;
        compare_domains(
            &cpus[usize::from(session.binding().session() == right.binding().session())]
                .snapshot()
                .domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
    }
    if resident.service.uploaded_value_bytes() != uploaded
        || resident.service.pending_events()? != 0
    {
        return Err(Error::State(
            "warm GPU cohort uploaded source/reused pending event".into(),
        ));
    }

    // Both rows share frontier one. Stale sibling must refuse the ENTIRE
    // logical publication, preserving the other row's previous root.
    let candidate = resident
        .prepare_logits_cohort(&[(&left, 42), (&right, 42)])
        .await?;
    let old_right = right.binding();
    left.forward(&[42], false).await?;
    if candidate.commit(&mut [&mut left, &mut right]).is_ok()
        || right.binding() != old_right
        || left.binding().frontier() != 2
    {
        return Err(Error::State(
            "stale GPU row partially published sibling root".into(),
        ));
    }
    let reference_right = &mut cpus[1];
    let expected_second = reference_right.forward(42)?;
    let actual_second = right.forward(&[42], false).await?;
    exact(
        &expected_second,
        &actual_second.logits,
        "surviving GPU sibling after stale cohort refusal",
    )?;
    compare_domains(
        &reference_right.snapshot().domain_values(),
        &right.snapshot().read_domains().await?,
    )?;
    if resident.service.pending_events()? != 0 {
        return Err(Error::State(
            "physical resident cohort left device event pending".into(),
        ));
    }
    // Verify larger logical batch ownership with the very same resident
    // source and one original StateRoot per row, not a new GPU runtime.
    drop(left);
    drop(right);
    if count > 2 {
        let original_usage = resident.service.resident_bytes()?;
        let ids = [11u32, 353, 17, 42, 55, 78, 101, 135];
        let mut sessions = Vec::with_capacity(count);
        let mut references = Vec::with_capacity(count);
        for &id in &ids[..count] {
            sessions.push(resident.session().await?);
            let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&model))?;
            references.push(cpu.prefill_transaction(&[id])?);
        }
        let original = sessions
            .iter()
            .map(ResidentSession::binding)
            .collect::<Vec<_>>();
        let candidate = {
            let rows = sessions
                .iter()
                .zip(&ids[..count])
                .map(|(session, &token)| (session, token))
                .collect::<Vec<_>>();
            resident.prepare_logits_cohort(&rows).await?
        };
        if sessions
            .iter()
            .map(ResidentSession::binding)
            .collect::<Vec<_>>()
            != original
        {
            return Err(Error::State(
                "multirow private resident candidate published early".into(),
            ));
        }
        let mut live = sessions.iter_mut().collect::<Vec<_>>();
        let results = candidate.commit(&mut live)?;
        drop(live);
        if results.len() != count {
            return Err(Error::State("multirow cohort completion count".into()));
        }
        let mut state_bits = 0usize;
        for (row, (session, result)) in sessions.iter().zip(results.iter()).enumerate() {
            exact(
                &references[row],
                &result.logits,
                "multirow source real logits",
            )?;
            let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(Arc::clone(&model))?;
            cpu.prefill_transaction(&[ids[row]])?;
            let expected = cpu.snapshot().domain_values();
            let actual = session.snapshot().read_domains().await?;
            compare_domains(&expected, &actual)?;
            state_bits += expected.iter().map(|(_, v)| v.len()).sum::<usize>();
            if session.binding().root() != 1
                || session.binding().frontier() != 1
                || session.binding().session() != original[row].session()
            {
                return Err(Error::State("multirow root/session binding changed".into()));
            }
        }
        drop(sessions);
        if resident.service.resident_bytes()? != original_usage {
            return Err(Error::State(
                "multirow resident session lease not released".into(),
            ));
        }
        println!(
            "P12_RESIDENT_COHORT_GPU_PASS B={count} independent_sessions={count}              complete_logits=exact all42domains=exact state_elements={state_bits}              source_weight_uploads=unchanged gpu_fused_model_kernels=0"
        );
    }
    println!(
        "P12_RESIDENT_COHORT_GPU_PASS B=2 independent_sessions=2 complete_logits=exact \
         all42domains=exact abandoned=unpublished invalid_token=refused \
         stale_root=all_row_rollback physical_ready=2_per_cohort \
         gpu_fused_model_kernels=0"
    );
    Ok(())
}
