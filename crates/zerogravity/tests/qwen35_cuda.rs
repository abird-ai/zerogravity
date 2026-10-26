#![cfg(all(feature = "qwen35-cuda", target_os = "linux"))]
//! Explicit real-source/GPU suite; never part of ordinary CPU test execution.
//! A fresh 47-token prompt commits frontier 47; frontiers 48–50 consume one cached token each.
//! Fresh 48–50-token prompts are not implemented or qualified by this suite.
// Former names: short_hf, canonical47, case5. Historical oracle labels remain exact.

use std::{
    future::Future,
    path::{Path, PathBuf},
    pin::pin,
    sync::{Arc, atomic::AtomicBool},
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
#[path = "qwen35_cuda/compare.rs"]
mod compare;
#[path = "qwen35_cuda/faults.rs"]
mod faults;
#[path = "qwen35_cuda/host.rs"]
mod fixture_host;
use fixture_host::FixtureHost;
#[path = "qwen35_cuda/oracle.rs"]
mod oracle;
fn fail(message: impl Into<String>) -> Error {
    Error::State(message.into())
}
fn ensure(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(fail(message)) }
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
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(result) => return result,
            Poll::Pending => thread::park(),
        }
    }
}
fn fixture_prompt(public: &serde_json::Value, rows: usize) -> Result<Vec<u32>> {
    let ids = public[rows.to_string()]["prompt_token_ids"]
        .as_array()
        .ok_or_else(|| fail("public prompt IDs"))?;
    ensure(ids.len() == rows, "genuine public prompt length")?;
    ids.iter()
        .map(|id| {
            id.as_u64()
                .filter(|n| *n < 248320)
                .map(|n| n as u32)
                .ok_or_else(|| fail("public source token ID"))
        })
        .collect()
}
fn pool_conservation(model: &qwen35_cuda::Model) -> Result<()> {
    let loaded = model.resident();
    let observed = loaded.observation().snapshot()?;
    let resources = loaded.resources();
    let root_bytes = resources.root_initialization_host_bytes / 3;
    ensure(
        observed.root_pool_idle_sets <= observed.root_pool_allocated_sets
            && observed.root_pool_allocated_sets <= resources.root_count
            && observed.root_pool_backing_bytes == observed.root_pool_allocated_sets * root_bytes
            && observed.resident_bytes >= observed.root_pool_backing_bytes,
        "live/idle original backing remains charged inside the whole root bound",
    )
}
async fn matrix(
    model: &Arc<qwen35_cuda::Model>,
    host: &TransferAdmission,
    public: &serde_json::Value,
    reference: &Path,
) -> Result<()> {
    for rows in [47, 57, 62, 67, 72] {
        let ids = fixture_prompt(public, rows)?;
        let cell = reference.join(format!("{}-{rows}", model.source_metadata().family));
        let mut session = model.session(None).await?;
        for call in 0..8 {
            let before = session.binding();
            let mut callbacks = 0;
            let mut validate = |read: native::TextReadback<'_>| {
                let _reference = host.arbiter.admit(
                    &host.resource,
                    host.owner.clone(),
                    &[AllocationRequest {
                        kind: AllocationClass::Scratch,
                        bytes: (256 * MIB) as u64,
                    }],
                )?;
                callbacks += 1;
                compare::completed(read, &cell, call, rows, model.source_profile())
            };
            let receipt = if call == 0 {
                session.infer_prompt_with(&ids, None, &mut validate).await?
            } else {
                session.infer_next_with(None, &mut validate).await?
            };
            ensure(
                callbacks == 1
                    && receipt.root == session.binding()
                    && receipt.executed_nodes == 255
                    && receipt.native_launches > 0
                    && receipt.root.session() == before.session()
                    && receipt.root.root() == before.root() + 1
                    && receipt.root.frontier() == (rows + call) as u64
                    && receipt.consumed_tokens == if call == 0 { rows } else { 1 },
                "one completed original current-root commit per natural output",
            )?;
            let [issued, retired, queries, waits, pending] = receipt.transfer_retirement;
            let windows = if model.resident().resources().resident_head_bytes != 0 {
                0
            } else {
                (model.source_metadata().vocabulary_size as usize
                    * model.source_metadata().hidden_width
                    * 2)
                .div_ceil(MIB)
            };
            ensure(
                issued == windows
                    && retired == issued
                    && pending == 0
                    && queries >= retired
                    && waits <= retired,
                "actual terminal retirement precedes successful natural publication",
            )?;
            pool_conservation(model)?;
            println!(
                "QWEN35_CUDA_TRANSFER_RETIREMENT_PASS rows={rows} call={call} counts={:?}",
                receipt.transfer_retirement,
            );
            println!(
                "QWEN35_CUDA_NUMERIC_CALL_PASS rows={rows} call={call} dispatches={} finite_certification={:?}",
                receipt.dispatches, receipt.finite_certification
            );
        }
        drop(session);
        pool_conservation(model)?;
        let observed = model.resident().observation().snapshot()?;
        ensure(
            observed.pending_events == 0
                && observed.usable
                && observed.root_pool_allocated_sets <= 4,
            "whole trajectory completes while reusing bounded root backing",
        )?;
    }
    Ok(())
}
async fn lifecycle(model: &Arc<qwen35_cuda::Model>, public: &serde_json::Value) -> Result<()> {
    let ids = fixture_prompt(public, 47)?;
    let loaded = model.resident();
    let mut sessions = Vec::with_capacity(8);
    for _ in 0..8 {
        sessions.push(model.session(None).await?);
    }
    for i in 0..8 {
        for j in i + 1..8 {
            ensure(
                sessions[i].binding().session() != sessions[j].binding().session(),
                "eight independent actual SessionAuthority roots",
            )?;
        }
    }
    pool_conservation(model)?;
    let before = loaded.observation().snapshot()?;
    ensure(
        model.session(None).await.is_err(),
        "ninth actual session pressure refused",
    )?;
    let after = loaded.observation().snapshot()?;
    ensure(
        before.resident_bytes == after.resident_bytes && after.pending_events == 0 && after.usable,
        "session pressure leaves healthy physical owners unchanged",
    )?;
    let first = sessions[0].snapshot_empty()?;
    let second = sessions[0].snapshot_empty()?;
    ensure(
        sessions[0].snapshot_empty().is_err(),
        "third actual snapshot pressure refused",
    )?;
    let left = sessions[0]
        .prepare_prompt_with(&ids, None, |_| Ok(()))
        .await?;
    let right = sessions[0]
        .prepare_prompt_with(&ids, None, |_| Ok(()))
        .await?;
    ensure(
        sessions[0]
            .prepare_prompt_with(&ids, None, |_| Ok(()))
            .await
            .is_err(),
        "third actual Pending pressure refused",
    )?;
    let receipt = sessions[0].commit(right, None)?;
    ensure(
        receipt.root == sessions[0].binding(),
        "actual ready proposal committed",
    )?;
    ensure(
        sessions[0].commit(left, None).is_err(),
        "actual completed sibling proposal stale after commit",
    )?;
    sessions[0].restore_empty(&first, None)?;
    ensure(
        sessions[0].binding().frontier() == 0,
        "original empty restore",
    )?;
    let pending = sessions[0]
        .prepare_prompt_with(&ids, None, |_| Ok(()))
        .await?;
    sessions[0].restore_empty(&second, None)?;
    ensure(
        sessions[0].commit(pending, None).is_err(),
        "actual completed proposal stale after restore",
    )?;
    drop(first);
    drop(second);
    let cancelled = AtomicBool::new(true);
    let root = sessions[0].binding();
    ensure(
        sessions[0]
            .infer_prompt(&ids, Some(&cancelled))
            .await
            .is_err()
            && sessions[0].binding() == root,
        "pre-cancellation cannot publish or change original root",
    )?;
    for session in &mut sessions {
        let receipt = session.infer_prompt(&ids, None).await?;
        ensure(
            receipt.root == session.binding() && receipt.root.frontier() == 47,
            "all eight real roots advance independently",
        )?;
    }
    drop(sessions);
    pool_conservation(model)?;
    let observed = loaded.observation().snapshot()?;
    ensure(
        observed.usable && observed.pending_events == 0,
        "healthy survivor after pressure/stale/restore/cancellation",
    )?;
    println!(
        "QWEN35_CUDA_LIFECYCLE_PASS actual_sessions=8 independent_roots=true ninth_refused=true pending_limit=2 snapshot_limit=2 stale_commit_restore=refused precancel_no_publish=true survivors=8"
    );
    Ok(())
}

#[test]
#[ignore = "explicit original CUDA physical qualification; separate fresh owned process before model scopes"]
fn finite_batch_transport_and_state_ownership() -> Result<()> {
    ensure(
        !cfg!(debug_assertions),
        "release finite physical scope only",
    )?;
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?;
    arbiter.register(resource.clone(), (4096 * MIB) as u64)?;
    let owner = ResourceOwner::new("qwen35-cuda-physical", "whole-matrix")?;
    let device =
        zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("actual CUDA device: {e}")))?;
    let finite_cases = block_on(native::qualify_bf16_finite_cohorts(
        device, &resource, owner,
    ))?;
    ensure(
        finite_cases == 301,
        "unchanged complete finite physical case count",
    )?;
    println!(
        "FINITE_COHORT_PHYSICAL_PASS cases={finite_cases} all_slots=true finite_nonfinite=true sticky=true poison_tails=true original_completion=true"
    );
    Ok(())
}

#[test]
#[ignore = "explicit real approved source/native artifacts, release CUDA, owned isolated process only"]
fn source_model_state_and_output_agreement() -> Result<()> {
    ensure(!cfg!(debug_assertions), "release physical matrix only")?;
    let source = env("ZG_QWEN35_MODEL_DIR")?;
    let bundle = env("ZG_QWEN35_CUDA_KERNEL_DIR")?;
    let reference = env("ZG_QWEN35_REFERENCE_DIR")?;
    let prompts = env("ZG_QWEN35_PROMPT_FIXTURES")?;
    let public = oracle::json(&prompts, MIB)?;
    let host = FixtureHost::new(ResourceOwner::new("qwen35-cuda-physical", "whole-matrix")?)?;
    let device =
        zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("actual CUDA device: {e}")))?;
    let model = block_on(qwen35_cuda::load(
        device,
        &source,
        &bundle,
        &host,
        qwen35_cuda::PROVIDER_MIB * MIB,
        None,
    ))?;
    println!(
        "QWEN35_CUDA_RESOURCE_SOURCE_BOUND {:?}",
        model.resident().resources()
    );
    let observation = model.resident().observation();
    let observed = observation.snapshot()?;
    ensure(
        observed.provider_capacity_bytes == qwen35_cuda::PROVIDER_MIB * MIB
            && observed.resident_bytes <= qwen35_cuda::PROVIDER_MIB * MIB
            && model.resident().resources().host_upper_bytes as u64
                <= fixture_host::declared_limit(&host)?
            && observed.usable
            && observed.pending_events == 0,
        "actual configured provider load",
    )?;
    let physical = host.arbiter.snapshot(&observed.physical_key)?;
    ensure(
        physical.limit == (5120 * MIB) as u64 && physical.reserved <= physical.limit,
        "actual5GiB physical admission",
    )?;
    ensure(
        model.resident().resources().resident_head_bytes
            == model.source_metadata().vocabulary_size as usize
                * model.source_metadata().hidden_width
                * 2,
        "actual source-shaped full head selected within configured allowance",
    )?;
    let family = model.source_metadata().family;
    let result = block_on(async {
        matrix(&model, &host, &public, &reference).await?;
        lifecycle(&model, &public).await
    });
    if let Err(error) = &result {
        eprintln!("QWEN35_CUDA_EXECUTION_FAILURE {error:?}");
    }
    let closed = block_on(model.resident().clone().close(None));
    if let Err(error) = &closed {
        eprintln!("QWEN35_CUDA_CLOSE_FAILURE {error:?}");
    }
    drop(model);
    let retired = observation.snapshot()?;
    drop(observation); // Release the service's prepaid physical envelope too.
    ensure(
        retired.resident_bytes == 0
            && retired.pending_events == 0
            && retired.root_pool_allocated_sets == 0
            && retired.root_pool_backing_bytes == 0,
        "actual whole-model native retirement",
    )?;
    let physical_retired = host.arbiter.snapshot(&retired.physical_key)?.reserved;
    let host_retired = host.arbiter.snapshot(&host.resource)?.reserved;
    println!("QWEN35_CUDA_FINAL_RESERVATIONS physical={physical_retired} host={host_retired}");
    ensure(
        retired.backend_device_reserved_bytes == (512 * MIB) as u64
            && retired.backend_host_reserved_bytes == (384 * MIB) as u64
            && physical_retired == retired.backend_device_reserved_bytes
            && host
                .arbiter
                .snapshot(&retired.physical_key)?
                .active_models
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                == ["backend/cuda"],
        "actual final host/provider/physical conservation",
    )?;
    host.require_final(retired.backend_host_reserved_bytes)?;
    closed?;
    result?;
    println!(
        "QWEN35_CUDA_WHOLE_MATRIX_PASS model={family} prompts=47,57,62,67,72 calls=40 native_performance_qualified=false"
    );
    Ok(())
}

#[test]
#[ignore = "fresh owned isolated release process per deliberate cached fault"]
fn decode_fault_retains_unfinished_work() -> Result<()> {
    faults::run()
}
