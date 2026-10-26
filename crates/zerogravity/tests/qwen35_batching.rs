#![cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
//! Explicit genuine B188 initial prefill -> same private B4 roots -> cached epochs.
//! A fresh 47-token prompt commits frontier 47; frontiers 48–50 consume one cached token each.
//! Fresh 48–50-token prompts are not implemented or qualified by this suite.
// Former names: short_hf, canonical47, case5. Historical oracle labels remain exact.

use std::{
    future::Future,
    path::{Path, PathBuf},
    pin::pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread,
};
use zerogravity::gpu::cuda::native;
use zerogravity::{
    Error, Result,
    cache::TransferAdmission,
    qwen35_cuda,
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
const MIB: usize = 1024 * 1024;
#[path = "qwen35_cuda/host.rs"]
mod fixture_host;
use fixture_host::FixtureHost;
#[path = "qwen35_batching/cached2b.rs"]
mod cached2b;
#[path = "qwen35_batching/compare.rs"]
mod compare;
#[path = "qwen35_batching/continuous.rs"]
mod continuous;
#[path = "qwen35_batching/oracle.rs"]
mod oracle;
fn fail(message: impl Into<String>) -> Error {
    Error::State(message.into())
}
fn ensure(value: bool, message: &str) -> Result<()> {
    if value { Ok(()) } else { Err(fail(message)) }
}
fn env(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .ok_or_else(|| fail(format!("required absolute {name}")))
}
struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark()
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark()
    }
}
fn block_on<F: Future>(future: F) -> F::Output {
    let w = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut cx = Context::from_waker(&w);
    let mut f = pin!(future);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}
fn conservation(model: &qwen35_cuda::Model, host: &TransferAdmission) -> Result<()> {
    let loaded = model.resident();
    let s = loaded.observation().snapshot()?;
    let r = loaded.resources();
    let root = r.root_initialization_host_bytes / 3;
    let (roots, provider_mib) = match model.source_profile() {
        zerogravity::qwen35::plan::QwenSourceProfile::Qwen08 => (22, 4096),
        zerogravity::qwen35::plan::QwenSourceProfile::Qwen2B => (18, 4608),
    };
    ensure(
        r.root_count == roots
            && r.host_upper_bytes as u64 <= fixture_host::declared_limit(host)?
            && r.provider_upper_bytes <= provider_mib * MIB
            && s.usable
            && s.pending_events == 0
            && s.root_pool_idle_sets <= s.root_pool_allocated_sets
            && s.root_pool_allocated_sets <= roots
            && s.root_pool_backing_bytes == s.root_pool_allocated_sets * root,
        "actual source-bound root/provider/host bound and healthy completion",
    )
}
async fn matrix(
    model: &Arc<qwen35_cuda::Model>,
    host: &TransferAdmission,
    original: &oracle::Original,
) -> Result<()> {
    let ids = original.prompt()?;
    let mut sessions = Vec::new();
    for _ in 0..4 {
        sessions.push(model.session(None).await?);
    }
    for a in 0..4 {
        for b in a + 1..4 {
            ensure(
                sessions[a].binding().session() != sessions[b].binding().session(),
                "four genuine source-issued independent roots",
            )?;
        }
    }
    let flags = std::array::from_fn::<_, 4, _>(|_| AtomicBool::new(false));
    let cancelled = flags.iter().collect::<Vec<_>>();
    for call in 0..8 {
        let before = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
        let mut callbacks = 0;
        let mut validate = |read: native::TextCohortReadback<'_>| {
            let _expected = host.arbiter.admit(
                &host.resource,
                host.owner.clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: (256 * MIB) as u64,
                }],
            )?;
            callbacks += 1;
            compare::completed(read, original, call)
        };
        let refs = sessions.iter().collect::<Vec<_>>();
        let pending = if call == 0 {
            model
                .prepare_fixed_prompt_cohort(&refs, &ids, &cancelled, Some(&mut validate))
                .await?
        } else {
            model
                .prepare_feedback_cohort(&refs, &cancelled, Some(&mut validate))
                .await?
        }
        .ok_or_else(|| fail("complete admitted B4 unexpectedly refused before launch"))?;
        ensure(
            callbacks == 1 && sessions.iter().zip(&before).all(|(s, b)| s.binding() == *b),
            "known completion compares once before ANY row publication",
        )?;
        let receipt = pending
            .commit(
                &mut sessions
                    .iter_mut()
                    .map(|s| s.engine_mut())
                    .collect::<Vec<_>>(),
                &cancelled,
            )?
            .ok_or_else(|| fail("healthy B4 publication discarded"))?;
        ensure(
            receipt.physical_rows == 4
                && receipt.executed_nodes == 255
                && receipt.native_projection_launches == 115
                && receipt.sampler_dispatches == 2 + receipt.physical_rows
                && receipt.rows.len() == 4
                && receipt.native_launches > 0
                && receipt.readback_bytes <= 640 * MIB,
            &format!(
                "one genuine B-shaped255-node flight/head/reducer: two reductions + four packet copies; receipt={receipt:?}"
            ),
        )?;
        for (row, (session, receipt)) in sessions.iter().zip(&receipt.rows).enumerate() {
            ensure(
                receipt.root == session.binding()
                    && receipt.root.session() == before[row].session()
                    && receipt.root.root() == before[row].root() + 1
                    && receipt.root.frontier() == (47 + call) as u64
                    && receipt.selected_id == original.selected(row, call)?
                    && receipt.consumed_tokens == if call == 0 { 47 } else { 1 }
                    && session.cohort_feedback_eligible(),
                "same genuine roots advance exactly once with committed native feedback lineage",
            )?;
        }
        conservation(model, host)?;
        println!(
            "QWEN35_BATCHING_COMMIT_PASS call={call} request_batch_size=4 nodes=255 projection_packets={} sampler_dispatches={} readback_bytes={}",
            receipt.native_projection_launches, receipt.sampler_dispatches, receipt.readback_bytes
        );
    }
    drop(sessions);
    conservation(model, host)
}
async fn lifecycle(
    model: &Arc<qwen35_cuda::Model>,
    host: &TransferAdmission,
    original: &oracle::Original,
) -> Result<()> {
    let ids = original.prompt()?;
    let mut sessions = Vec::new();
    for _ in 0..8 {
        sessions.push(model.session(None).await?);
    }
    ensure(
        model.session(None).await.is_err(),
        "ninth actual session cannot invalidate22-root bound",
    )?;
    let flags = std::array::from_fn::<_, 4, _>(|_| AtomicBool::new(false));
    let cancelled = flags.iter().collect::<Vec<_>>();
    let before = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    flags[1].store(true, Ordering::Release);
    ensure(
        model
            .prepare_fixed_prompt_cohort(
                &sessions[..4].iter().collect::<Vec<_>>(),
                &ids,
                &cancelled,
                None,
            )
            .await
            .is_err()
            && sessions.iter().zip(&before).all(|(s, b)| s.binding() == *b),
        "pre-cancel has no launch/publication",
    )?;
    flags[1].store(false, Ordering::Release);
    let pending = model
        .prepare_fixed_prompt_cohort(
            &sessions[..4].iter().collect::<Vec<_>>(),
            &ids,
            &cancelled,
            None,
        )
        .await?
        .ok_or_else(|| fail("first real pending cohort"))?;
    ensure(
        model
            .prepare_fixed_prompt_cohort(
                &sessions[4..].iter().collect::<Vec<_>>(),
                &ids,
                &cancelled,
                None,
            )
            .await?
            .is_none(),
        "completed pending retains sole reservation; typed prelaunch capacity fallback",
    )?;
    flags[2].store(true, Ordering::Release);
    ensure(
        pending
            .commit(
                &mut sessions[..4]
                    .iter_mut()
                    .map(|s| s.engine_mut())
                    .collect::<Vec<_>>(),
                &cancelled,
            )?
            .is_none()
            && sessions.iter().zip(&before).all(|(s, b)| s.binding() == *b)
            && sessions.iter().all(|s| !s.cohort_feedback_eligible()),
        "known late cancellation discards ALL private candidates and grants no lineage",
    )?;
    flags[2].store(false, Ordering::Release);
    let pending = model
        .prepare_fixed_prompt_cohort(
            &sessions[..4].iter().collect::<Vec<_>>(),
            &ids,
            &cancelled,
            None,
        )
        .await?
        .ok_or_else(|| fail("healthy successor after known discard"))?;
    pending
        .commit(
            &mut sessions[..4]
                .iter_mut()
                .map(|s| s.engine_mut())
                .collect::<Vec<_>>(),
            &cancelled,
        )?
        .ok_or_else(|| fail("healthy all-row publication"))?;
    let committed = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    let pending = model
        .prepare_feedback_cohort(&sessions[..4].iter().collect::<Vec<_>>(), &cancelled, None)
        .await?
        .ok_or_else(|| fail("real cached pending for stale guard"))?;
    sessions[0].infer_next(None).await?;
    ensure(
        !sessions[0].cohort_feedback_eligible(),
        "N1 successor cannot mint B4 native lineage",
    )?;
    let advanced = sessions[0].binding();
    ensure(
        pending
            .commit(
                &mut sessions[..4]
                    .iter_mut()
                    .map(|s| s.engine_mut())
                    .collect::<Vec<_>>(),
                &cancelled,
            )
            .is_err()
            && sessions[0].binding() == advanced
            && sessions[1..]
                .iter()
                .zip(&committed[1..])
                .all(|(s, b)| s.binding() == *b),
        "one stale sibling fatally refuses all-row publication without partial swaps",
    )?;
    drop(sessions);
    conservation(model, host)?;
    println!(
        "QWEN35_BATCHING_LIFECYCLE_PASS sessions=8 ninth_refused=true completed_pending_capacity=true precancel=true latecancel_all_none=true stale_all_none=true lineage_only_commit=true roots_bound=22"
    );
    Ok(())
}
#[test]
#[ignore = "explicit original approved source/native packets; ROOT-owned isolated required-release CUDA process; no perf claim"]
fn prompt_batch_and_decode_preserve_request_state() -> Result<()> {
    let host = FixtureHost::new(ResourceOwner::new(
        "qwen35-batching-b4-physical",
        "same-root-whole-cohort",
    )?)?;
    let original = oracle::Original::open(&env("ZG_QWEN35_BATCHING_REFERENCE_DIR")?, &host)?;
    original.preflight()?;
    let device =
        zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("actual CUDA: {e}")))?;
    let model = block_on(qwen35_cuda::qualification::load_b4(
        device,
        &env("ZG_QWEN35_MODEL_DIR")?,
        &env("ZG_QWEN35_CUDA_KERNEL_DIR")?,
        &host,
        4096 * MIB,
        None,
    ))?;
    println!(
        "QWEN35_BATCHING_RESOURCE_SOURCE_BOUND {:?} debug={} performance_qualified=false",
        model.resident().resources(),
        cfg!(debug_assertions)
    );
    ensure(
        model.resident().native_cohort_width() == 4 && model.source_metadata().family == "0.8b",
        "only explicit actual B4 .8B profile",
    )?;
    let observation = model.resident().observation();
    let result = block_on(async {
        matrix(&model, &host, &original).await?;
        lifecycle(&model, &host, &original).await
    });
    if let Err(error) = &result {
        eprintln!("QWEN35_BATCHING_FAILURE {error:?}");
    }
    let closed = block_on(model.resident().clone().close(None));
    drop(model);
    let retired = observation.snapshot()?;
    eprintln!("QWEN35_BATCHING_RETIREMENT result={result:?} closed={closed:?} retired={retired:?}");
    drop(observation);
    ensure(
        retired.resident_bytes == 0
            && retired.pending_events == 0
            && retired.root_pool_allocated_sets == 0
            && retired.root_pool_backing_bytes == 0,
        "actual B4 source/state/native physical retirement",
    )?;
    original.close()?;
    ensure(
        retired.backend_device_reserved_bytes == (512 * MIB) as u64
            && retired.backend_host_reserved_bytes == (384 * MIB) as u64
            && host.arbiter.snapshot(&retired.physical_key)?.reserved
                == retired.backend_device_reserved_bytes
            && host
                .arbiter
                .snapshot(&retired.physical_key)?
                .active_models
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                == ["backend/cuda"],
        "actual final B4 host/provider/device conservation",
    )?;
    host.require_final(retired.backend_host_reserved_bytes)?;
    closed?;
    result?;
    println!(
        "QWEN35_BATCHING_WHOLE_PASS request_batch_size=4 outputs=32 same_roots=true performance_qualified=false ordinary_mixed_frontier_qualified=false"
    );
    Ok(())
}

#[test]
#[ignore = "explicit genuine N1 same-prior continuous B4; ROOT-owned isolated required-release CUDA process; no perf claim"]
fn cached_decode_matches_single_request() -> Result<()> {
    let host = FixtureHost::new(ResourceOwner::new(
        "qwen35-batching-b4-physical",
        "same-root-whole-cohort",
    )?)?;
    let original = oracle::Original::open(&env("ZG_QWEN35_BATCHING_REFERENCE_DIR")?, &host)?;
    original.preflight()?;
    let device =
        zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("actual CUDA: {e}")))?;
    let model = block_on(qwen35_cuda::qualification::load_b4(
        device,
        &env("ZG_QWEN35_MODEL_DIR")?,
        &env("ZG_QWEN35_CUDA_KERNEL_DIR")?,
        &host,
        4096 * MIB,
        None,
    ))?;
    println!(
        "QWEN35_BATCHING_RESOURCE_SOURCE_BOUND {:?} debug={} performance_qualified=false",
        model.resident().resources(),
        cfg!(debug_assertions)
    );
    ensure(
        model.resident().native_cohort_width() == 4 && model.source_metadata().family == "0.8b",
        "only explicit actual B4 .8B profile",
    )?;
    let observation = model.resident().observation();
    let result = block_on(async { continuous::matrix(&model, &host, &original).await });
    if let Err(error) = &result {
        eprintln!("QWEN35_BATCHING_FAILURE {error:?}");
    }
    let closed = block_on(model.resident().clone().close(None));
    drop(model);
    let retired = observation.snapshot()?;
    eprintln!(
        "QWEN35_BATCHING_CONTINUOUS_RETIREMENT result={result:?} closed={closed:?} retired={retired:?}"
    );
    drop(observation);
    ensure(
        retired.resident_bytes == 0
            && retired.pending_events == 0
            && retired.root_pool_allocated_sets == 0
            && retired.root_pool_backing_bytes == 0,
        "actual B4 source/state/native physical retirement",
    )?;
    original.close()?;
    ensure(
        retired.backend_device_reserved_bytes == (512 * MIB) as u64
            && retired.backend_host_reserved_bytes == (384 * MIB) as u64
            && host.arbiter.snapshot(&retired.physical_key)?.reserved
                == retired.backend_device_reserved_bytes
            && host
                .arbiter
                .snapshot(&retired.physical_key)?
                .active_models
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                == ["backend/cuda"],
        "actual final B4 host/provider/device conservation",
    )?;
    host.require_final(retired.backend_host_reserved_bytes)?;
    closed?;
    result?;
    println!(
        "QWEN35_BATCHING_CONTINUOUS_WHOLE_PASS source_roots=true same_prior_n1=true fixed_prompt_cached=true wider_shapes_qualified=false performance_qualified=false"
    );
    Ok(())
}

#[test]
#[ignore = "DIAGNOSTIC ONLY: exports own unqualified trajectory; never numerical acceptance or serving eligibility"]
fn export_batched_decode_reference_composition() -> Result<()> {
    drop(env("ZG_QWEN35_BATCHING_EXPORT_DIR")?);
    let host = FixtureHost::new(ResourceOwner::new(
        "qwen35-batching-b4-physical",
        "same-root-whole-cohort",
    )?)?;
    let original = oracle::Original::open(&env("ZG_QWEN35_BATCHING_REFERENCE_DIR")?, &host)?;
    original.preflight()?;
    let device =
        zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("actual CUDA: {e}")))?;
    let model = block_on(qwen35_cuda::qualification::load_b4(
        device,
        &env("ZG_QWEN35_MODEL_DIR")?,
        &env("ZG_QWEN35_CUDA_KERNEL_DIR")?,
        &host,
        4096 * MIB,
        None,
    ))?;
    println!(
        "QWEN35_BATCHING_RESOURCE_SOURCE_BOUND {:?} debug={} performance_qualified=false",
        model.resident().resources(),
        cfg!(debug_assertions)
    );
    ensure(
        model.resident().native_cohort_width() == 4 && model.source_metadata().family == "0.8b",
        "only explicit actual B4 .8B profile",
    )?;
    let observation = model.resident().observation();
    let result =
        block_on(async { continuous::diagnostic_trajectory(&model, &host, &original).await });
    if let Err(error) = &result {
        eprintln!("QWEN35_BATCHING_DIAGNOSTIC_UNQUALIFIED_FAILURE {error:?}");
    }
    let closed = block_on(model.resident().clone().close(None));
    drop(model);
    let retired = observation.snapshot()?;
    eprintln!(
        "QWEN35_BATCHING_DIAGNOSTIC_UNQUALIFIED_RETIREMENT result={result:?} closed={closed:?} retired={retired:?}"
    );
    drop(observation);
    ensure(
        retired.resident_bytes == 0
            && retired.pending_events == 0
            && retired.root_pool_allocated_sets == 0
            && retired.root_pool_backing_bytes == 0,
        "actual B4 source/state/native physical retirement",
    )?;
    original.close()?;
    ensure(
        retired.backend_device_reserved_bytes == (512 * MIB) as u64
            && retired.backend_host_reserved_bytes == (384 * MIB) as u64
            && host.arbiter.snapshot(&retired.physical_key)?.reserved
                == retired.backend_device_reserved_bytes
            && host
                .arbiter
                .snapshot(&retired.physical_key)?
                .active_models
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                == ["backend/cuda"],
        "actual final B4 host/provider/device conservation",
    )?;
    host.require_final(retired.backend_host_reserved_bytes)?;
    closed?;
    result?;
    println!(
        "QWEN35_BATCHING_DIAGNOSTIC_UNQUALIFIED_CLOSED source_roots=true mathematical_accepted=false serving_qualified=false performance_qualified=false"
    );
    Ok(())
}

#[test]
#[ignore = "fresh original ROOT-owned isolated process per submitted B4 fault; deliberate quarantine, no recovery"]
fn cached_decode_fault_retains_unfinished_work() -> Result<()> {
    let fault = match std::env::var("ZG_QWEN35_BATCHING_FAULT_POINT").as_deref() {
        Ok("conv-unknown") => native::TextCohortFault::UnknownAfterFirstConvolution,
        Ok("last-fa-unknown") => native::TextCohortFault::UnknownAfterLastFa,
        Ok("head-unknown") => native::TextCohortFault::UnknownAfterHead,
        _ => {
            return Err(fail(
                "exact B4 fault mode conv-unknown/last-fa-unknown/head-unknown",
            ));
        }
    };
    let host = FixtureHost::new(ResourceOwner::new(
        "qwen35-batching-b4-physical",
        "cached-unknown",
    )?)?;
    let original = oracle::Original::open(&env("ZG_QWEN35_BATCHING_REFERENCE_DIR")?, &host)?;
    original.preflight()?;
    let device =
        zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("actual CUDA: {e}")))?;
    let model = block_on(qwen35_cuda::qualification::load_b4(
        device,
        &env("ZG_QWEN35_MODEL_DIR")?,
        &env("ZG_QWEN35_CUDA_KERNEL_DIR")?,
        &host,
        4096 * MIB,
        None,
    ))?;
    let loaded = model.resident().clone();
    let ids = original.prompt()?;
    let mut sessions = Vec::new();
    for _ in 0..4 {
        sessions.push(block_on(model.session(None))?);
    }
    let flags = std::array::from_fn::<_, 4, _>(|_| AtomicBool::new(false));
    let cancelled = flags.iter().collect::<Vec<_>>();
    let mut callbacks = 0;
    let mut compare = |read: native::TextCohortReadback<'_>| {
        let _expected = host.arbiter.admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: (256 * MIB) as u64,
            }],
        )?;
        callbacks += 1;
        compare::completed(read, &original, 0)
    };
    let initial = block_on(model.prepare_fixed_prompt_cohort(
        &sessions.iter().collect::<Vec<_>>(),
        &ids,
        &cancelled,
        Some(&mut compare),
    ))?
    .ok_or_else(|| fail("B4 initial before deliberate unknown"))?;
    initial
        .commit(
            &mut sessions
                .iter_mut()
                .map(|s| s.engine_mut())
                .collect::<Vec<_>>(),
            &cancelled,
        )?
        .ok_or_else(|| fail("B4 genuine initial roots before cached fault"))?;
    ensure(
        callbacks == 1
            && sessions
                .iter()
                .all(|s| s.binding().frontier() == 47 && s.cohort_feedback_eligible()),
        "actual compared B188 creates all4 genuine cached roots",
    )?;
    original.close()?;
    let roots = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    let facade_weak = Arc::downgrade(&model);
    let loaded_weak = Arc::downgrade(&loaded);
    let outcome = block_on(model.probe_feedback_cohort_fault(
        &sessions.iter().collect::<Vec<_>>(),
        &cancelled,
        fault,
    ))?;
    ensure(
        outcome.fault == fault
            && outcome.roots == roots
            && sessions.iter().zip(&roots).all(|(s, r)| s.binding() == *r),
        "submitted cached unknown has no partial publication",
    )?;
    drop(sessions);
    drop(loaded);
    drop(model);
    ensure(
        facade_weak.upgrade().is_none() && loaded_weak.upgrade().is_none(),
        "no external facade/Loaded/session retains submitted graph",
    )?;
    let retained = outcome.retained_after_external_drop()?;
    ensure(
        retained.submitted_nodes > 0
            && retained.native_launches > 0
            && retained.projection_launches > 0
            && retained.sampler_dispatches == 0
            && retained.candidate_planes_unpromoted == 4 * 48
            && retained.retained_actual_leases > 4 * 96
            && retained.quarantined_handles == retained.retained_actual_leases
            && retained.quarantined_owners >= 5
            && retained.pending_events == 1
            && retained.physical_fenced
            && !retained.service_usable
            && retained.physical_reserved_bytes == ((4096 + 512) * MIB) as u64
            && retained.resident_bytes > 0
            && retained.resident_bytes <= 4096 * MIB,
        "all actual prior/candidate/source/native/epoch owners remain quarantined; no Ready or feedback",
    )?;
    host.require_retained((384 * MIB) as u64)?;
    println!(
        "QWEN35_BATCHING_CACHED_FAULT_PASS fault={fault:?} actual_retention={retained:?} published=0 recovery_qualified=false performance_qualified=false"
    );
    // Original process/group/container/EOF supervisor is the final owner.
    Ok(())
}

#[test]
#[ignore = "CPU-only complete accepted original archive/typed-consumer preflight; no GPU/model execution"]
fn reference_archive_covers_all_consumed_values() -> Result<()> {
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?;
    arbiter.register(resource.clone(), (4096 * MIB) as u64)?;
    let host = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("qwen35-batching-b4-reference", "cpu-preflight")?,
    };
    let original = oracle::Original::open(&env("ZG_QWEN35_BATCHING_REFERENCE_DIR")?, &host)?;
    original.preflight()?;
    original.close()?;
    println!(
        "QWEN35_BATCHING_ORIGINAL_PREFLIGHT_PASS request_batch_size=4 calls=8 all4344records=true source_held=true native_math_qualified=false"
    );
    Ok(())
}

#[test]
#[ignore = "explicit grouped-transport release CUDA qualification; ROOT-owned isolated process; no perf claim"]
fn grouped_state_transport_preserves_each_request() -> Result<()> {
    ensure(
        !cfg!(debug_assertions),
        "release grouped transport scope only",
    )?;
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?;
    arbiter.register(resource.clone(), (4096 * MIB) as u64)?;
    let owner = ResourceOwner::new("qwen35-batching-cohort-physical", "grouped-transport")?;
    let device =
        zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("actual CUDA device: {e}")))?;
    let cases = block_on(native::qualify_grouped_state_transport(
        device, &resource, owner,
    ))?;
    ensure(cases > 0, "grouped transport case count")?;
    println!(
        "GROUPED_TRANSPORT_PHYSICAL_PASS cases={cases} case_unit=grouped_launches per_row_vs_grouped=true canaries=true status_sticky=true guards=true future_padding=true"
    );
    Ok(())
}
