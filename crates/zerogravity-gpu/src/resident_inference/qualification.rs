//! Source/physical diagnostics, compiled only by explicit qualification.
use super::*;

pub async fn qualify_bounded_context(
    device: Device,
    path: &Path,
    teacher_ids: Vec<u32>,
    capacity: usize,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    context_qualification::qualify_bounded(
        device,
        path,
        teacher_ids,
        capacity,
        provider_cap,
        device_cap,
    )
    .await
}

pub async fn qualify_context(
    device: Device,
    path: &Path,
    prompt: &Path,
    ids: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    context_qualification::qualify(device, path, prompt, ids, provider_cap, device_cap).await
}

pub async fn qualify(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    let model = LoadedQwenModel::open(
        path,
        &path
            .parent()
            .ok_or_else(|| Error::Malformed("source path".into()))?
            .join("config.json"),
    )?;
    let resident = load_qualified(device, model.clone(), 256, provider_cap, device_cap).await?;
    alias_qualification::qualify(&resident).await?;
    prefill_qualification::qualify(&resident, &model).await?;
    alias_qualification::qualify(&resident).await?;
    range_qualification::qualify(&resident).await?;
    view_qualification::qualify(&resident).await?;
    let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(model.clone())?;
    let mut session = resident.session().await?;
    let mut comparisons = 0usize;
    for token in [11, 353, 17, 42] {
        let (expected, draft, trace) = cpu.next_logits_with_layer_trace(token)?;
        let uploads = resident.service.uploaded_value_bytes();
        let actual = session.forward(&[token], true).await?;
        if resident.service.uploaded_value_bytes() != uploads {
            return Err(Error::State("warm graph uploaded model values".into()));
        }
        exact(&expected, &actual.logits, "full logits")?;
        let domains = draft.domain_values();
        if domains.len() != actual.domains.len() {
            return Err(Error::State("domain count mismatch".into()));
        }
        for (slot, expected) in &domains {
            let (_, got) = actual
                .domains
                .iter()
                .find(|(candidate, _)| candidate == slot)
                .ok_or_else(|| Error::State("missing physical domain slot".into()))?;
            exact(expected, got, &format!("state {slot:?}"))?;
            comparisons += expected.len();
        }
        if trace.len() != actual.observations.len() {
            return Err(Error::State(format!(
                "trace count {} != {}",
                trace.len(),
                actual.observations.len()
            )));
        }
        for (e, a) in trace.iter().zip(&actual.observations) {
            if e.layer != a.layer || e.stage != a.stage || e.position != a.position {
                return Err(Error::State("trace order mismatch".into()));
            }
            exact(
                &e.values,
                &a.values,
                &format!("layer {} {}", e.layer, e.stage),
            )?;
            comparisons += e.values.len();
        }
        cpu.commit(draft)?;
        println!(
            "H16_RESIDENT_TOKEN_PASS token={token} frontier={} dispatches={} actual_d2h_bytes={} logits_bits=exact all42_state_domains=exact ordered_trace=exact",
            session.binding().frontier(),
            actual.dispatches,
            actual.bytes
        );
    }
    let saved = session.snapshot();
    let saved_cpu = cpu.snapshot();
    compare_domains(&saved_cpu.domain_values(), &saved.read_domains().await?)?;
    let expected = cpu.forward(55)?;
    let stale = session.prepare(&[55], false).await?;
    if session.binding() != saved.binding {
        return Err(Error::State("private Ready changed visible root".into()));
    }
    session.restore(&saved)?;
    let restored = session.binding();
    if session.commit(stale).is_ok() || session.binding() != restored {
        return Err(Error::State("stale actualReady published".into()));
    }
    exact(
        &expected,
        &session.forward(&[55], false).await?.logits,
        "restored nonzero replay",
    )?;
    compare_domains(
        &cpu.snapshot().domain_values(),
        &session.snapshot().read_domains().await?,
    )?;
    let unchanged = session.binding();
    if session.prepare(&[11, u32::MAX], false).await.is_ok() || session.binding() != unchanged {
        return Err(Error::State("invalid host token changed root".into()));
    }
    let mut wrong_generation = session.snapshot();
    wrong_generation.domains[0].generations[0] += 1;
    if session.restore(&wrong_generation).is_ok() || session.binding() != unchanged {
        return Err(Error::State(
            "stale physical allocation generation accepted".into(),
        ));
    }
    let bytes = resident.service.resident_bytes()?;
    if resident
        .service
        .allocate_uninitialized(
            resident.service.remaining_capacity()? + 1,
            BufferRole::Scratch,
        )
        .is_ok()
        || resident.service.resident_bytes()? != bytes
    {
        return Err(Error::State("resident overbudget admission".into()));
    }
    resident.service.ensure_usable()?;
    let mut sibling = resident.session().await?;
    if sibling.restore(&saved).is_ok() {
        return Err(Error::State("foreign resident snapshot accepted".into()));
    }
    let mut sibling_cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(model.clone())?;
    let expected = sibling_cpu.prefill_transaction(&[11, 353, 17])?;
    let actual = sibling.forward(&[11, 353, 17], true).await?;
    exact(&expected, &actual.logits, "atomic three-token GPU prefill")?;
    compare_domains(&sibling_cpu.snapshot().domain_values(), &actual.domains)?;
    if sibling.binding().root() != 1
        || sibling.binding().frontier() != 3
        || session.binding() != unchanged
    {
        return Err(Error::State("prefill publication/sibling isolation".into()));
    }
    sibling.reset().await?;
    sibling_cpu.reset()?;
    exact(
        &sibling_cpu.forward(42)?,
        &sibling.forward(&[42], false).await?.logits,
        "GPU reset replay",
    )?;
    if resident.service.pending_events()? != 0 {
        return Err(Error::State("completed resident events leaked".into()));
    }
    sibling.reset().await?;
    sibling_cpu.reset()?;
    let mut cpu_logits = sibling_cpu.prefill_transaction(&[11, 353])?;
    let uploads = resident.service.uploaded_value_bytes();
    let first = sibling.forward_greedy(&[11, 353]).await?;
    if first.bytes != 16 || !first.logits.is_empty() {
        return Err(Error::State("greedy readback is not compact".into()));
    }
    let mut token = first
        .device_token
        .ok_or_else(|| Error::State("missing device token".into()))?;
    for iteration in 0..5 {
        let best = (0..cpu_logits.len())
            .max_by(|a, b| {
                cpu_logits[*a]
                    .total_cmp(&cpu_logits[*b])
                    .then_with(|| b.cmp(a))
            })
            .unwrap();
        if token.id as usize != best || token.max_bits != cpu_logits[best].to_bits() {
            return Err(Error::State(
                "GPU greedy source token/max bits differ".into(),
            ));
        }
        if iteration != 4 {
            cpu_logits = sibling_cpu.forward(token.id)?;
            let next = sibling.forward_device_token(&token).await?;
            if next.bytes != 16 || !next.logits.is_empty() {
                return Err(Error::State(
                    "device-token decode readback is not compact".into(),
                ));
            }
            token = next
                .device_token
                .ok_or_else(|| Error::State("missing next GPU token".into()))?;
        }
    }
    if resident.service.uploaded_value_bytes() != uploads {
        return Err(Error::State("device-token loop reuploaded values".into()));
    }
    compare_domains(
        &sibling_cpu.snapshot().domain_values(),
        &sibling.snapshot().read_domains().await?,
    )?;
    let before = sibling.snapshot();
    sibling.restore(&before)?;
    if sibling.forward_device_token(&token).await.is_ok() {
        return Err(Error::State(
            "stale device token accepted after rewind".into(),
        ));
    }
    drop(before);
    drop(token);
    println!(
        "H16_RESIDENT_GREEDY_PASS greedy_tokens=5 device_to_device_token_bindings=4 exact_ids_and_max_bits=true final_state=exact actual_step_d2h_bytes=16 warm_buffer_h2d_bytes=0 stale_token_after_restore=refused"
    );
    {
        let _admission = resident.allocation.lock().await;
        crate::resident_greedy_qualification::run(&resident.service).await?;
    }
    let mut concurrent_a = resident.session().await?;
    let mut concurrent_b = resident.session().await?;
    let uploads = resident.service.uploaded_value_bytes();
    let (a, b) = futures_lite::future::zip(
        concurrent_a.forward(&[11, 353], false),
        concurrent_b.forward(&[17, 42], false),
    )
    .await;
    let mut cpu_a = zerogravity_model_qwen35::QwenDenseCpu::from_shared(model.clone())?;
    let mut cpu_b = zerogravity_model_qwen35::QwenDenseCpu::from_shared(model.clone())?;
    exact(
        &cpu_a.prefill_transaction(&[11, 353])?,
        &a?.logits,
        "concurrent GPU session A",
    )?;
    exact(
        &cpu_b.prefill_transaction(&[17, 42])?,
        &b?.logits,
        "concurrent GPU session B",
    )?;
    let ready_a = concurrent_a.prepare(&[77], false).await?;
    let rewind_b = concurrent_b.snapshot();
    concurrent_b.restore(&rewind_b)?;
    exact(
        &cpu_a.forward(77)?,
        &concurrent_a.commit(ready_a)?.logits,
        "independent session epoch",
    )?;
    compare_domains(
        &cpu_a.snapshot().domain_values(),
        &concurrent_a.snapshot().read_domains().await?,
    )?;
    compare_domains(
        &cpu_b.snapshot().domain_values(),
        &concurrent_b.snapshot().read_domains().await?,
    )?;
    if resident.service.uploaded_value_bytes() != uploads || session.binding() != unchanged {
        return Err(Error::State(
            "concurrent sessions changed source or sibling root".into(),
        ));
    }
    drop(rewind_b);
    drop(concurrent_a);
    drop(concurrent_b);
    println!(
        "H16_RESIDENT_CONCURRENT_PASS independently_polled_sessions=2 distinct_committed_roots=true exact_logits_and_states=true sibling_restore_does_not_revoke_ready=true warm_buffer_h2d_bytes=0"
    );
    compare_domains(&saved_cpu.domain_values(), &saved.read_domains().await?)?;
    println!(
        "H16_RESIDENT_OWNERSHIP_PASS stale_actual_ready=refused sibling_state=independent snapshot_readback=exact nonzero_restore_replay=exact private_allocation_generation=checked invalid_host_chunk=prelaunch_refused memory_limit=prelaunch_refused chunk_root_increment=one reset=exact"
    );
    println!(
        "H16_RESIDENT_GRAPH_PASS physical={:?} compared_state_trace_f32={comparisons} resident_bytes={} source_reuploads=0 intermediate_activation_transfers=0 cpu_activation_quantization=0 diagnostic_end_of_graph_readback=true",
        resident.service.physical_key(),
        resident.service.resident_bytes()?
    );
    let surviving = session.snapshot();
    let expected_surviving = surviving.read_domains().await?;
    let service_weak = Arc::downgrade(resident.service());
    let key = resident.service.physical_key();
    let charged = resident.service.capacity() as u64;
    drop(saved);
    drop(wrong_generation);
    drop(sibling);
    drop(session);
    drop(resident);
    if service_weak.upgrade().is_none()
        || zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&key)?
            .reserved
            != charged
    {
        return Err(Error::State("snapshot lost its physical envelope".into()));
    }
    compare_domains(&expected_surviving, &surviving.read_domains().await?)?;
    drop(surviving);
    if service_weak.upgrade().is_some()
        || zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&key)?
            .reserved
            != 0
    {
        return Err(Error::State(
            "completed resident teardown leaked ownership/charge".into(),
        ));
    }
    println!(
        "H16_RESIDENT_LIFETIME_PASS snapshot_outlives_provider=true physical_charge_retained=true immutable_state_readback=exact final_reserved_bytes=0"
    );
    Ok(())
}
pub(super) fn compare_domains(
    expected: &[(StateSlot, Vec<f32>)],
    actual: &[(StateSlot, Vec<f32>)],
) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(Error::State("physical domain coverage".into()));
    }
    for (slot, values) in expected {
        let (_, got) = actual
            .iter()
            .find(|(s, _)| s == slot)
            .ok_or_else(|| Error::State("missing physical state slot".into()))?;
        exact(values, got, &format!("state {slot:?}"))?;
    }
    Ok(())
}

pub(super) fn exact(expected: &[f32], actual: &[f32], name: &str) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(Error::State(format!(
            "{name} length {} != {}",
            expected.len(),
            actual.len()
        )));
    }
    if let Some((index, (e, a))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (e, a))| e.to_bits() != a.to_bits())
    {
        return Err(Error::State(format!(
            "{name} bits differ at {index}: expected {:08x}, got {:08x}",
            e.to_bits(),
            a.to_bits()
        )));
    }
    Ok(())
}

/// Process-isolated diagnostic: real graph launch, then drop before readback.
#[cfg(feature = "hardware-abort-probe")]
pub async fn qualify_abort(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    qualify_abort_inner(device, path, provider_cap, device_cap, None).await
}

#[cfg(feature = "hardware-abort-probe")]
pub async fn qualify_context_abort(
    device: Device,
    path: &Path,
    prompt: &Path,
    ids: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    qualify_abort_inner(device, path, provider_cap, device_cap, Some((prompt, ids))).await
}

#[cfg(feature = "hardware-abort-probe")]
async fn qualify_abort_inner(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    context: Option<(&Path, &Path)>,
) -> Result<()> {
    use std::sync::atomic::Ordering;
    let client = device.client();
    let source = LoadedQwenModel::open(
        path,
        &path
            .parent()
            .ok_or_else(|| Error::Malformed("source path".into()))?
            .join("config.json"),
    )?;
    let context_tokens = context
        .map(|(prompt, ids)| context_qualification::source_tokens(path, prompt, ids))
        .transpose()?;
    let resident = load_qualified(
        device.clone(),
        source.clone(),
        if context.is_some() { 512 } else { 256 },
        provider_cap,
        device_cap,
    )
    .await?;
    let mut session = resident.session().await?;
    if let Some(tokens) = &context_tokens {
        session
            .prefill_without_logits(&tokens[..255], false)
            .await?;
    } else {
        session.forward(&[11, 353], false).await?;
    }
    let saved = session.snapshot();
    let expected = saved.read_domains().await?;
    let binding = session.binding();
    let chunk = context_tokens
        .as_ref()
        .map_or(&[42, 77, 19][..], |t| &t[255..258]);
    let stale_ready = session.prepare(&chunk[..1], false).await?;
    resident.abort_after_launch.store(2, Ordering::Release);
    let mut future = Box::pin(session.prepare(chunk, false));
    loop {
        if let Some(result) = futures_lite::future::poll_once(future.as_mut()).await {
            return Err(Error::State(format!(
                "abort graph unexpectedly completed: {}",
                result.is_ok()
            )));
        }
        if resident.launch_seen.load(Ordering::Acquire) > 0 {
            break;
        }
        futures_lite::future::yield_now().await;
    }
    drop(future);
    let expected_handles = resident.launch_seen.load(Ordering::Acquire);
    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
    if quarantined < expected_handles
        || session.binding() != binding
        || resident.service.ensure_usable().is_ok()
        || session.commit(stale_ready).is_ok()
        || session.binding() != binding
    {
        return Err(Error::State(
            "unknown resident graph retained/publication contract failed".into(),
        ));
    }
    if ResidentService::new(
        device,
        source.source_identity(),
        ResourceOwner::new("other-model", "after-abort")?,
        1024 * 1024,
        device_cap,
    )
    .is_ok()
    {
        return Err(Error::State(
            "fenced resident card admitted another model".into(),
        ));
    }
    // Diagnostic driver barrier only: do not retire unknown events, release
    // admissions, clear fences, report Ready or execute more model operations.
    client
        .sync()
        .await
        .map_err(|e| Error::State(format!("diagnostic abort quiescence failed: {e}")))?;
    for (cursor, domain) in saved.domains.iter().enumerate() {
        let handles = domain.leases.iter().map(ResidentLease::handle).collect();
        let data = client
            .read_async(handles)
            .await
            .map_err(|e| Error::State(format!("diagnostic prior-root readback failed: {e}")))?;
        if data.len() != domain.leases.len()
            || data
                .iter()
                .zip(&domain.leases)
                .any(|(bytes, lease)| bytes.len() != lease.bytes())
        {
            return Err(Error::State(
                "diagnostic prior-root full-capacity readback incomplete".into(),
            ));
        }
        let mut flat = Vec::new();
        for (l, b) in domain.leases.iter().zip(data) {
            let v = words(&b)?;
            if matches!(&domain.buffers, StateBuffers::Kv { .. }) {
                let active = l.bytes() / 4 / saved.capacity * binding.frontier() as usize;
                if v[active..].iter().any(|x| x.to_bits() != 0) {
                    return Err(Error::State(
                        "aborted graph changed prior canonical KV tail".into(),
                    ));
                }
                flat.extend_from_slice(&v[..active]);
            } else {
                flat.extend(v);
            }
        }
        exact(
            &expected[cursor].1,
            &flat,
            "physically preserved nonzero root after abort",
        )?;
    }
    let key = resident.service.physical_key();
    let weak = Arc::downgrade(resident.service());
    let cap = resident.service.capacity() as u64;
    drop(saved);
    drop(session);
    drop(resident);
    let kept = weak
        .upgrade()
        .ok_or_else(|| Error::State("unknown graph lost charged service".into()))?;
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != cap || !budget.fenced || kept.physical_remaining()? != device_cap - cap {
        return Err(Error::State(
            "unknown graph physical charge/fence lost".into(),
        ));
    }
    let label = if context.is_some() {
        "H18_RESIDENT_CONTEXT_ABORT_PASS"
    } else {
        "H17_RESIDENT_PREFILL_ABORT_PASS"
    };
    println!(
        "{label} prior_frontier={} cancelled_valid_tokens={} actual_state_only_graph_launched=true tentative_completed_state_only_tokens=1 cancelled_chunk_tokens=3 quarantined_original_handles={quarantined} submitted_original_handles={expected_handles} prior_nonzero_state_physical_bits=exact canonical_future_kv_zero=exact root_unchanged=true stale_ready_after_fence=refused other_model=refused retained_envelope_bytes={cap} trusted_recovery=false",
        binding.frontier(),
        binding.frontier() + 2,
    );
    Ok(())
}
