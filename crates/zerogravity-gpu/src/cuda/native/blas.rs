//! Genuine projection leaf comparison on the original queue and envelope.
use super::*;
use crate::resident_leases::ForeignProviderCharge;
use cubecl::cuda::{Bf16GemmBindings, ObservedBf16Gemm};
use std::path::Path;
use zerogravity_artifacts::source_weights::TensorWindow;

const FOREIGN_BYTES: usize = 64 * 1024 * 1024;

/// Complete genuine RMS input / gated input and three public projection outputs.
/// This leaf seam does not qualify a connected layer or permit a model route.
pub struct ProjectionFixture {
    pub normalized: Vec<u8>,
    pub gated: Vec<u8>,
    pub qkvz: Vec<u8>,
    pub ba: Vec<u8>,
    pub output: Vec<u8>,
}

struct ProjectionOwner {
    _source: VerifiedTensorSource,
    _windows: Vec<Arc<TensorWindow>>,
    weights: [Vec<u8>; 3],
    fixtures: [ProjectionFixture; 5],
    _host: [WindowWorkspace; 3],
}
struct Job {
    operation: ObservedBf16Gemm,
    weights: ResidentLease,
    input: ResidentLease,
    output: ResidentLease,
    case: usize,
    projection: usize,
}

fn server_error(error: impl std::fmt::Display) -> Error {
    Error::State(format!("original cuBLAS provider: {error}"))
}

pub(super) type OriginalProjectionWeights = (Vec<Arc<TensorWindow>>, [Vec<u8>; 3]);

pub(super) fn source_weights(source: &VerifiedTensorSource) -> Result<OriginalProjectionWeights> {
    let mut windows = Vec::new();
    let mut weights = std::array::from_fn(|_| Vec::new());
    for (index, value) in weights.iter_mut().enumerate() {
        let (original, raw) = super::layer_common::projection_weight(source, index)?;
        windows.extend(original);
        *value = raw;
    }
    Ok((windows, weights))
}

pub(super) fn operation(rows: usize, projection: usize) -> ObservedBf16Gemm {
    use ObservedBf16Gemm::*;
    match (rows, projection) {
        (1, 0) => Qwen08Layer0QkvzDecode,
        (1, 1) => Qwen08Layer0BaDecode,
        (1, _) => Qwen08Layer0OutProjectionDecode,
        (64, 0) => Qwen08Layer0QkvzPrefill,
        (64, 1) => Qwen08Layer0BaPrefill,
        (64, _) => Qwen08Layer0OutProjectionPrefill,
        (2, 0) => Qwen08Layer0QkvzContinuedPrefill,
        (2, 1) => Qwen08Layer0BaContinuedPrefill,
        (2, _) => Qwen08Layer0OutProjectionContinuedPrefill,
        _ => unreachable!("closed observed rows"),
    }
}

/// Fresh isolated process only: resolver hooks and the opaque pool are one-shot.
/// Provider modules/driver metadata retain the existing coarse-envelope scope;
/// explicit allocation counters are not whole-process GPU/RAM accounting.
pub async fn qualify_projections(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [ProjectionFixture; 5],
    cublas: &Path,
    cublas_lt: &Path,
    device_cap: u64,
) -> Result<usize> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release CUDA required for vLLM projection leaf".into(),
        ));
    }
    let shapes = [1, 1, 64, 2, 1];
    for (fixture, rows) in fixtures.iter().zip(shapes) {
        for (raw, words) in [
            (&fixture.normalized, rows * 1024),
            (&fixture.gated, rows * 2048),
            (&fixture.qkvz, rows * 8192),
            (&fixture.ba, rows * 32),
            (&fixture.output, rows * 1024),
        ] {
            if raw.len() != words * 2 || !finite_bf16(raw) {
                return Err(Error::Malformed(
                    "complete actual projection leaf fixture".into(),
                ));
            }
        }
    }
    let host = [
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
    ];
    let (windows, weights) = source_weights(&source)?;
    let owner = Arc::new(ProjectionOwner {
        _source: source.clone(),
        _windows: windows,
        _host: host,
        weights,
        fixtures,
    });
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("qwen08-cuda-leaf", "original-projections")?,
        PROVIDER_CAP,
        device_cap,
    )?;
    let key = service.physical_key();
    let queue = service.pinned_cuda_queue()?.clone();
    let sources = queue
        .verify_bf16_blas_sources(cublas, cublas_lt)
        .map_err(server_error)?;
    let charge: Arc<ForeignProviderCharge> = service.reserve_foreign_provider(FOREIGN_BYTES)?;
    if charge.bytes() != FOREIGN_BYTES {
        return Err(Error::State("original opaque cuBLAS charge extent".into()));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: (PROVIDER_CAP - FOREIGN_BYTES) as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(1)?;
    init.retain_completion_owner(owner.clone())?;
    let weights = owner
        .weights
        .iter()
        .map(|raw| init.upload_bytes(raw, BufferRole::ImmutableWeights))
        .collect::<Result<Vec<_>>>()?;
    let workspace = init.allocate_empty(8_519_680, BufferRole::Scratch)?;
    let status = init.upload_bytes(&0x13579bdfu32.to_le_bytes(), BufferRole::ImmutableWeights)?;
    let mut jobs = Vec::new();
    for (case, rows) in shapes.into_iter().enumerate() {
        for (projection, weights) in weights.iter().enumerate() {
            let op = operation(rows, projection);
            let input = if projection == 2 {
                &owner.fixtures[case].gated
            } else {
                &owner.fixtures[case].normalized
            };
            jobs.push(Job {
                operation: op,
                weights: weights.clone(),
                input: init.upload_bytes(input, BufferRole::Activation)?,
                output: init.allocate_empty(op.output_bytes(), BufferRole::Output)?,
                case,
                projection,
            });
        }
    }
    let initialized = init.complete_sync().await?;
    slot.finish()?;
    drop(initialized);
    // No ordinary buffer is allocated after activation. The original global
    // keepalive owns this exact prepaid charge even if partial creation fails.
    let pool = unsafe { queue.activate_blas_allocation_pool(FOREIGN_BYTES as u64, charge.clone()) }
        .map_err(server_error)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: FOREIGN_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(2, &[(status.clone(), DeviceAccess::Read)])?;
    flight.retain_completion_owner(owner.clone())?;
    flight.retain_completion_owner(pool.clone())?;
    flight.retain_completion_owner(charge.clone())?;
    let provider = unsafe { queue.create_bf16_blas_provider(sources.clone(), pool.clone()) }
        .map_err(|error| {
            Error::State(format!(
                "actual cuBLAS creation: {error}; pool={:?}",
                pool.snapshot()
            ))
        })?;
    flight.retain_completion_owner(provider.clone())?;
    flight
        .complete_read_only_readback(&[status.readback(0, 4)?], |_| Ok(()))
        .await?;
    slot.finish()?;
    let mut compared = 0;
    for (index, job) in jobs.iter().enumerate() {
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: job.operation.output_bytes() as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(
            index as u64 + 3,
            &[
                (job.weights.clone(), DeviceAccess::Read),
                (job.input.clone(), DeviceAccess::Read),
                (job.output.clone(), DeviceAccess::Write),
                (workspace.clone(), DeviceAccess::Write),
            ],
        )?;
        flight.retain_completion_owner(owner.clone())?;
        flight.retain_completion_owner(provider.clone())?;
        flight.retain_completion_owner(pool.clone())?;
        let bindings = Bf16GemmBindings {
            weights: job.weights.handle(),
            input: job.input.handle(),
            output: job.output.handle(),
            workspace: workspace.handle(),
        };
        unsafe { provider.enqueue(job.operation, bindings) }.map_err(|error| {
            Error::State(format!(
                "actual cuBLAS enqueue{index}: {error}; pool={:?}",
                pool.snapshot()
            ))
        })?;
        let expected = match job.projection {
            0 => &owner.fixtures[job.case].qkvz,
            1 => &owner.fixtures[job.case].ba,
            _ => &owner.fixtures[job.case].output,
        };
        let (ready, bytes) = flight
            .complete_with_readback(
                std::slice::from_ref(&job.output),
                &[job.output.readback(0, job.output.bytes())?],
                |actual| {
                    if &actual[0][..] != expected.as_slice() {
                        let first = actual[0]
                            .iter()
                            .zip(expected)
                            .position(|(a, b)| a != b)
                            .unwrap_or(0);
                        return Err(Error::State(format!(
                            "actual cuBLAS {index} projection differs at byte{first}"
                        )));
                    }
                    Ok(())
                },
            )
            .await?;
        compared += expected.len() / 2;
        slot.finish()?;
        drop((ready, bytes));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: FOREIGN_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(99, &[(status.clone(), DeviceAccess::Read)])?;
    flight.retain_completion_owner(provider.clone())?;
    flight.retain_completion_owner(pool.clone())?;
    flight.retain_completion_owner(charge.clone())?;
    unsafe { provider.destroy_after_physical_completion() }.map_err(server_error)?;
    flight
        .complete_read_only_readback(&[status.readback(0, 4)?], |_| Ok(()))
        .await?;
    slot.finish()?;
    let retired = unsafe { pool.retire_after_physical_completion() }.map_err(|error| {
        Error::State(format!(
            "actual cuBLAS pool retirement: {error}; pool={:?}",
            pool.snapshot()
        ))
    })?;
    if !retired.retired
        || retired.charged_bytes != 0
        || retired.live_allocations != 0
        || retired.provider_live
    {
        return Err(Error::State(
            "actual cuBLAS pool retirement incomplete".into(),
        ));
    }
    drop((
        jobs, weights, workspace, status, provider, pool, sources, queue, charge, owner,
    ));
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "actual cuBLAS original resident allocations/events not retired".into(),
        ));
    }
    drop(service);
    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != 0 {
        return Err(Error::State(
            "actual cuBLAS original envelope did not retire".into(),
        ));
    }
    Ok(compared)
}

#[cfg(test)]
mod tests {
    use cubecl::cuda::{
        Bf16GemmOperation, ObservedBf16Gemm, REFERENCE_BF16_GEMM_API,
        REFERENCE_BF16_GEMM_WORKSPACE_BYTES,
    };

    #[test]
    fn owned_bf16_gemm_descriptor_preserves_observed_calls_and_accepts_dynamic_rows() {
        use ObservedBf16Gemm::*;
        for observed in [
            Qwen08Layer0QkvzDecode,
            Qwen08Layer0BaDecode,
            Qwen08Layer0QkvzPrefill,
            Qwen08Layer0BaPrefill,
            Qwen08Layer0OutProjectionDecode,
            Qwen08Layer0OutProjectionPrefill,
            Qwen08Layer0QkvzContinuedPrefill,
            Qwen08Layer0BaContinuedPrefill,
            Qwen08Layer0OutProjectionContinuedPrefill,
        ] {
            let operation = observed.operation();
            assert_eq!(
                (operation.m(), operation.n(), operation.k()),
                (observed.m(), observed.n(), observed.k())
            );
            assert_eq!(operation.api(), observed.observed_api());
            assert_eq!(operation.workspace_bytes(), observed.workspace_bytes());
        }

        for rows in 1..=64 {
            let operation = Bf16GemmOperation::reference(8192, rows, 1024).unwrap();
            assert_eq!(operation.n(), rows);
            assert_eq!(operation.api(), REFERENCE_BF16_GEMM_API);
            assert_eq!(
                operation.workspace_bytes(),
                REFERENCE_BF16_GEMM_WORKSPACE_BYTES
            );
            assert_eq!(
                operation.sizes().unwrap(),
                [
                    8192 * 1024 * 2,
                    1024 * rows * 2,
                    8192 * rows * 2,
                    REFERENCE_BF16_GEMM_WORKSPACE_BYTES,
                ]
            );
        }
    }
}
