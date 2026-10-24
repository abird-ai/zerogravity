//! Genuine2B N1 roots→cachedB4; original same-schedule continuation judges math.
use super::*;

// Accepted original2B discovery1 plain/result.json,4003B,
// SHA d91984eb4e8c7792d5efeee653c4b87762193405d27af5b64e10a4629606d667.
// All four actual public prompts agree; no output/reference state is imported.
const PROMPT: [u32; 47] = [
    248045, 846, 198, 22791, 411, 1103, 314, 24959, 11, 18101, 539, 73982, 11, 364, 1599, 777,
    3665, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 248046, 198, 248045, 74455, 198,
];
fn host() -> Result<TransferAdmission> {
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?;
    arbiter.register(resource.clone(), (4096 * MIB) as u64)?;
    Ok(TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("short-hf-2b-b4-physical", "genuine-n1-cached-transition")?,
    })
}
fn load(host: &TransferAdmission) -> Result<Arc<short_hf::ShortHfModel>> {
    let device =
        zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("actual CUDA: {e}")))?;
    let model = block_on(short_hf::load_b4(
        device,
        &env("ZG_SHORT_HF_SOURCE")?,
        &env("ZG_SHORT_HF_NATIVE")?,
        host,
        4608 * MIB,
        None,
    ))?;
    ensure(
        model.source_metadata().family == "2b"
            && model.source_metadata().hidden_width == 2048
            && model.source_metadata().manifest_sha256
                == "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52"
            && model.resident().native_cohort_width() == 4
            && model.resident().initial_cohort_width() == 0
            && model.resident().resources().root_count == 18,
        "exact prepared2B cached-only source/phase/root authority",
    )?;
    conservation(model.resident())?;
    println!(
        "SHORT_HF_2B_CACHED_RESOURCE_SOURCE_BOUND {:?} debug={} initial_b4_qualified=false performance_qualified=false",
        model.resident().resources(),
        cfg!(debug_assertions)
    );
    Ok(model)
}
fn closed(
    model: Arc<short_hf::ShortHfModel>,
    host: &TransferAdmission,
    result: Result<()>,
) -> Result<()> {
    if let Err(error) = &result {
        eprintln!("SHORT_HF_2B_CACHED_FAILURE {error:?}");
    }
    let observation = model.resident().observation();
    let close = block_on(model.resident().clone().close(None));
    drop(model);
    let retired = observation.snapshot()?;
    drop(observation);
    eprintln!(
        "SHORT_HF_2B_CACHED_RETIREMENT result={result:?} closed={close:?} retired={retired:?}"
    );
    ensure(
        retired.resident_bytes == 0
            && retired.pending_events == 0
            && retired.root_pool_allocated_sets == 0
            && retired.root_pool_backing_bytes == 0,
        "actual2B model/source/native/root retirement",
    )?;
    let device = host.arbiter.snapshot(&retired.physical_key)?;
    let cpu = host.arbiter.snapshot(&host.resource)?;
    ensure(
        retired.backend_device_reserved_bytes == (512 * MIB) as u64
            && retired.backend_host_reserved_bytes == (384 * MIB) as u64
            && device.reserved == retired.backend_device_reserved_bytes
            && cpu.reserved == retired.backend_host_reserved_bytes
            && device
                .active_models
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                == ["backend/cuda"]
            && cpu
                .active_models
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                == ["backend/cuda"],
        "only actual admitted original CUDA context remains after2B source closure",
    )?;
    close?;
    result
}
#[test]
#[ignore = "DIAGNOSTIC ONLY: ROOT original owner, authentic2B source/native, independent original continuation required"]
fn diagnostic_short_hf_2b_compositional_trajectory() -> Result<()> {
    drop(env("ZG_SHORT_HF_SAME_PRIOR_EXPORT")?);
    let host = host()?;
    let model = load(&host)?;
    let result = block_on(continuous::diagnostic_trajectory_with_prompt(
        &model, &host, &PROMPT,
    ));
    closed(model, &host, result)?;
    println!(
        "SHORT_HF_2B_DIAGNOSTIC_UNQUALIFIED_CLOSED source_roots=true mathematical_accepted=false serving_qualified=false performance_qualified=false"
    );
    Ok(())
}
#[test]
#[ignore = "fresh ROOT-owned CUDA process per actual2B cached B4 unknown cut; deliberate retained quarantine"]
fn actual_short_hf_2b_cohort_cached_fault() -> Result<()> {
    let fault = match std::env::var("ZG_SHORT_HF_COHORT_FAULT").as_deref() {
        Ok("conv-unknown") => native::ShortHfCohortFault::UnknownAfterFirstConvolution,
        Ok("last-fa-unknown") => native::ShortHfCohortFault::UnknownAfterLastFa,
        Ok("head-unknown") => native::ShortHfCohortFault::UnknownAfterHead,
        _ => return Err(fail("exact2B cached cohort fault mode")),
    };
    let host = host()?;
    let model = load(&host)?;
    let loaded = model.resident().clone();
    let mut sessions = Vec::with_capacity(4);
    for _ in 0..4 {
        sessions.push(block_on(loaded.session(None))?);
    }
    let empty = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    let flags = [const { AtomicBool::new(false) }; 4];
    let cancelled = flags.iter().collect::<Vec<_>>();
    let before = loaded.observation().snapshot()?;
    ensure(
        matches!(
            block_on(loaded.prepare_canonical47_cohort_status(
                &sessions.iter().collect::<Vec<_>>(),
                &PROMPT,
                &cancelled,
                None
            )),
            Err(Error::Unsupported(_))
        ) && sessions
            .iter()
            .zip(&empty)
            .all(|(s, root)| s.binding() == *root),
        "unsupported2B INITIAL B4 refuses before candidate reservation/publication",
    )?;
    let after = loaded.observation().snapshot()?;
    ensure(
        before.resident_bytes == after.resident_bytes
            && before.pending_events == after.pending_events,
        "unsupported initial has no physical state effect",
    )?;
    for session in &mut sessions {
        block_on(session.infer_prompt(&PROMPT, None))?;
    }
    ensure(
        sessions
            .iter()
            .all(|s| s.binding().frontier() == 47 && s.cached_cohort_eligible()),
        "four genuine source-issued N1prefill roots/current feedback before cached unknown",
    )?;
    let roots = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    let healthy = loaded.observation().snapshot()?;
    let healthy_host = host.arbiter.snapshot(&host.resource)?;
    let healthy_device = host.arbiter.snapshot(&healthy.physical_key)?;
    eprintln!(
        "SHORT_HF_2B_CACHED_FAULT_BEFORE fault={fault:?} roots={roots:?} healthy={healthy:?} host={healthy_host:?} device={healthy_device:?}"
    );
    let facade_weak = Arc::downgrade(&model);
    let loaded_weak = Arc::downgrade(&loaded);
    let outcome = block_on(loaded.probe_feedback_cohort_fault(
        &sessions.iter().collect::<Vec<_>>(),
        &cancelled,
        fault,
    ))?;
    let actual_roots = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    eprintln!(
        "SHORT_HF_2B_CACHED_FAULT_NO_PUBLICATION fault={fault:?} observed_fault={:?} expected_roots={roots:?} outcome_roots={:?} current_roots={actual_roots:?}",
        outcome.fault, outcome.roots
    );
    ensure(
        outcome.fault == fault
            && outcome.roots == roots
            && sessions
                .iter()
                .zip(&roots)
                .all(|(s, root)| s.binding() == *root),
        "actual2B submitted unknown has no row publication",
    )?;
    drop(sessions);
    drop(loaded);
    drop(model);
    let facade_gone = facade_weak.upgrade().is_none();
    let loaded_gone = loaded_weak.upgrade().is_none();
    eprintln!(
        "SHORT_HF_2B_CACHED_FAULT_EXTERNAL_DROP facade_gone={facade_gone} loaded_gone={loaded_gone}"
    );
    ensure(
        facade_gone && loaded_gone,
        "no external2B model/session/Loaded keeps failed graph",
    )?;
    let retained = outcome.retained_after_external_drop()?;
    let host_retained = host.arbiter.snapshot(&host.resource)?;
    let device_retained = host.arbiter.snapshot(&healthy.physical_key)?;
    // Emit every actual field BEFORE assertions, including the reused N1 source
    // scope and all unknown cuts. No prefill/warmup counter is subtracted.
    eprintln!(
        "SHORT_HF_2B_CACHED_FAULT_RETAINED fault={fault:?} actual_retention={retained:?} healthy={healthy:?} host={host_retained:?} device={device_retained:?}"
    );
    // Weak upgrades retain this SAME body/all actual flight leases after caller
    // drop. The registry also counts other service retirements (N1 logical and
    // backing refusals), so its global references only bound this flight below.
    ensure(
        retained.submitted_nodes > 0
            && retained.native_launches > 0
            && retained.projection_launches > 0
            && retained.sampler_dispatches == 0
            && retained.candidate_planes_unpromoted == 4 * 48
            && retained.retained_actual_leases > 4 * 96
            && retained.quarantined_handles >= retained.retained_actual_leases
            && retained.quarantined_owners >= 5
            && retained.pending_events == 1
            && retained.physical_fenced
            && !retained.service_usable
            && healthy.backend_device_reserved_bytes == (512 * MIB) as u64
            && retained.physical_reserved_bytes
                == (4608 * MIB) as u64 + healthy.backend_device_reserved_bytes
            && retained.resident_bytes > 0
            && retained.resident_bytes <= 4608 * MIB,
        "same actual2B prior/candidate/source/native owners retained and provider fenced",
    )?;
    ensure(
        healthy.backend_host_reserved_bytes == (384 * MIB) as u64
            && host_retained.reserved > healthy.backend_host_reserved_bytes
            && host_retained.reserved <= (4096 * MIB) as u64
            && host_retained.active_models.contains(&host.owner.model)
            && host_retained
                .active_models
                .iter()
                .any(|owner| owner == "backend/cuda"),
        "actual original HOST source owner and detached backend both remain",
    )?;
    println!(
        "SHORT_HF_2B_B4_CACHED_FAULT_PASS fault={fault:?} actual_retention={retained:?} host={host_retained:?} published=0 recovery_qualified=false performance_qualified=false"
    );
    Ok(())
}
