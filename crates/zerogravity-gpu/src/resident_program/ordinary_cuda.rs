//! Production ordinary-CUDA physical components.
//!
//! This module deliberately owns no model loop and no StateRoot publication.
//! It composes the existing pinned ResidentService/PhysicalQueue/flight owner;
//! every submitted native object is retained by that original flight until
//! actual completion or quarantine.
#![allow(dead_code)] // Private unselected implementation until the full body lands.

use crate::resident_leases::{
    ForeignProviderCharge, ResidentInitialization, ResidentLease, ResidentService,
    ResidentSubmission, ordinary_projection_upload::PackedProjectionWeights,
};
use cubecl::cuda::{
    Bf16BlasSources, Bf16GemmBindings, Bf16GemmOperation, OwnedBf16BlasProvider,
    OwnedBlasAllocationPool, OwnedCudaQueue, REFERENCE_BF16_GEMM_WORKSPACE_BYTES,
};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
mod attention;

use zerogravity_core::{
    Error, Result,
    accelerator::program::{SourceNumericalContract, SourceOperator, SourceOperatorKind},
    device_events::{BufferRole, DeviceAccess},
    resource_arbiter::{AllocationClass, AllocationRequest},
};

const FOREIGN_BYTES: usize = 64 * 1024 * 1024;
const SENTINEL: u32 = 0x1357_9bdf;

fn server_error(error: impl std::fmt::Display) -> Error {
    Error::State(format!("ordinary CUDA cuBLAS owner: {error}"))
}

fn exact_view(
    lease: &ResidentLease,
    offset: usize,
    bytes: usize,
    role: &str,
) -> Result<cubecl::server::Handle> {
    let end = offset
        .checked_add(bytes)
        .filter(|end| *end <= lease.bytes())
        .ok_or_else(|| Error::Limit(format!("ordinary CUDA {role} view exceeds lease")))?;
    Ok(lease
        .handle()
        .offset_start(offset as u64)
        .offset_end((lease.bytes() - end) as u64))
}

/// Explicit runtime provider binding. These paths are never selected by model,
/// prompt, call/node ordinals, packets or captured recipe metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OrdinaryCudaProviderPaths {
    cublas: PathBuf,
    cublas_lt: PathBuf,
}
impl OrdinaryCudaProviderPaths {
    pub(crate) fn new(cublas: impl Into<PathBuf>, cublas_lt: impl Into<PathBuf>) -> Result<Self> {
        let cublas = cublas.into();
        let cublas_lt = cublas_lt.into();
        if cublas.as_os_str().is_empty() || cublas_lt.as_os_str().is_empty() || cublas == cublas_lt
        {
            return Err(Error::Malformed(
                "ordinary CUDA distinct cuBLAS/cuBLASLt paths required".into(),
            ));
        }
        Ok(Self { cublas, cublas_lt })
    }
    fn cublas(&self) -> &Path {
        &self.cublas
    }
    fn cublas_lt(&self) -> &Path {
        &self.cublas_lt
    }
}

/// The one active opaque BLAS provider for an ordinary ResidentService.
/// Explicit retirement is mandatory; Drop never pretends the provider retired.
pub(crate) struct OrdinaryCudaBlas {
    service: Arc<ResidentService>,
    queue: Arc<OwnedCudaQueue>,
    sources: Arc<Bf16BlasSources>,
    charge: Arc<ForeignProviderCharge>,
    pool: Arc<OwnedBlasAllocationPool>,
    provider: Arc<OwnedBf16BlasProvider>,
    workspace: ResidentLease,
    sentinel: ResidentLease,
    retired: AtomicBool,
}
impl OrdinaryCudaBlas {
    /// Verify provider source first, then allocate/activate only through the
    /// existing pinned service and physical queue.
    pub(crate) async fn prepare(
        service: Arc<ResidentService>,
        expected_root: u64,
        paths: &OrdinaryCudaProviderPaths,
    ) -> Result<Arc<Self>> {
        service.ensure_usable()?;
        let queue = service.pinned_cuda_queue()?.clone();
        let sources = queue
            .verify_bf16_blas_sources(paths.cublas(), paths.cublas_lt())
            .map_err(server_error)?;
        let charge: Arc<ForeignProviderCharge> = service.reserve_foreign_provider(FOREIGN_BYTES)?;
        if charge.bytes() != FOREIGN_BYTES {
            return Err(Error::State(
                "ordinary CUDA exact opaque provider charge differs".into(),
            ));
        }
        charge.require_service(&service)?;

        let init_bytes = REFERENCE_BF16_GEMM_WORKSPACE_BYTES
            .checked_add(4)
            .ok_or_else(|| {
                Error::Limit("ordinary CUDA BLAS initialization bytes overflow".into())
            })?;
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: init_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut init = service.begin_initialization(expected_root)?;
        init.require_service(&service)?;
        init.retain_completion_owner(sources.clone())?;
        init.retain_completion_owner(charge.clone())?;
        let workspace =
            init.allocate_empty(REFERENCE_BF16_GEMM_WORKSPACE_BYTES, BufferRole::Scratch)?;
        let sentinel = init.upload_bytes(&SENTINEL.to_le_bytes(), BufferRole::ImmutableWeights)?;
        drop(init.complete_sync().await?);
        slot.finish()?;

        // No native provider can exist before the exact non-Handle charge.
        let pool =
            unsafe { queue.activate_blas_allocation_pool(FOREIGN_BYTES as u64, charge.clone()) }
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
        let mut flight =
            service.submit(expected_root, &[(sentinel.clone(), DeviceAccess::Read)])?;
        flight.retain_completion_owner(sources.clone())?;
        flight.retain_completion_owner(pool.clone())?;
        flight.retain_completion_owner(charge.clone())?;
        let provider = unsafe { queue.create_bf16_blas_provider(sources.clone(), pool.clone()) }
            .map_err(server_error)?;
        // Attach before the completion fence so any later error keeps the
        // provider alive behind the original service quarantine.
        flight.retain_completion_owner(provider.clone())?;
        flight
            .complete_read_only_readback(&[sentinel.readback(0, 4)?], |actual| {
                if actual.len() != 1 || actual[0].as_ref() != SENTINEL.to_le_bytes() {
                    return Err(Error::State(
                        "ordinary CUDA BLAS creation sentinel differs".into(),
                    ));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;

        Ok(Arc::new(Self {
            service,
            queue,
            sources,
            charge,
            pool,
            provider,
            workspace,
            sentinel,
            retired: AtomicBool::new(false),
        }))
    }

    fn require_active(&self) -> Result<()> {
        self.service.ensure_usable()?;
        self.charge.require_service(&self.service)?;
        if self.retired.load(Ordering::Acquire) {
            return Err(Error::State("ordinary CUDA BLAS owner retired".into()));
        }
        let pool = self.pool.snapshot().map_err(server_error)?;
        if pool.poisoned || pool.retired || !pool.provider_live {
            return Err(Error::State(
                "ordinary CUDA BLAS allocation/provider state unavailable".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn workspace(&self) -> &ResidentLease {
        &self.workspace
    }

    /// Enqueue one exact full-matrix GemmEx into an already-armed original
    /// submission. Input/output subviews are bounds checked but their parent
    /// leases remain the retained physical owners.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(
        &self,
        flight: &mut ResidentSubmission<'_>,
        operation: Bf16GemmOperation,
        weights: &ResidentLease,
        input: &ResidentLease,
        input_offset_bytes: usize,
        output: &ResidentLease,
        output_offset_bytes: usize,
    ) -> Result<()> {
        self.require_active()?;
        let [weight_bytes, input_bytes, output_bytes, workspace_bytes] =
            operation.sizes().map_err(server_error)?;
        if weights.bytes() != weight_bytes || self.workspace.bytes() != workspace_bytes {
            return Err(Error::State(
                "ordinary CUDA BLAS weights/workspace extent differs".into(),
            ));
        }
        flight.require_accesses([
            (weights, DeviceAccess::Read),
            (input, DeviceAccess::Read),
            (output, DeviceAccess::Write),
            (&self.workspace, DeviceAccess::Write),
        ])?;
        let bindings = Bf16GemmBindings {
            weights: weights.handle(),
            input: exact_view(input, input_offset_bytes, input_bytes, "GEMM input")?,
            output: exact_view(output, output_offset_bytes, output_bytes, "GEMM output")?,
            workspace: self.workspace.handle(),
        };
        // Attach all opaque/native owners before the actor accepts work.
        flight.retain_completion_owner(self.sources.clone())?;
        flight.retain_completion_owner(self.provider.clone())?;
        flight.retain_completion_owner(self.pool.clone())?;
        flight.retain_completion_owner(self.charge.clone())?;
        unsafe { self.provider.enqueue(operation, bindings) }.map_err(server_error)
    }

    /// Destroy/unload only after a trusted completion fence on the same service.
    /// Any error leaves the owner non-reusable and the service/flight retains
    /// native objects through the ordinary quarantine path.
    pub(crate) async fn retire(&self, expected_root: u64) -> Result<()> {
        self.require_active()?;
        self.retired
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("ordinary CUDA BLAS retirement raced".into()))?;
        let ticket = self.service.queue().submit(
            self.service.budget(),
            self.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: FOREIGN_BYTES as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = self.service.submit(
            expected_root,
            &[(self.sentinel.clone(), DeviceAccess::Read)],
        )?;
        flight.retain_completion_owner(self.sources.clone())?;
        flight.retain_completion_owner(self.provider.clone())?;
        flight.retain_completion_owner(self.pool.clone())?;
        flight.retain_completion_owner(self.charge.clone())?;
        unsafe { self.provider.destroy_after_physical_completion() }.map_err(server_error)?;
        flight
            .complete_read_only_readback(&[self.sentinel.readback(0, 4)?], |actual| {
                if actual.len() != 1 || actual[0].as_ref() != SENTINEL.to_le_bytes() {
                    return Err(Error::State(
                        "ordinary CUDA BLAS retirement sentinel differs".into(),
                    ));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        let retired =
            unsafe { self.pool.retire_after_physical_completion() }.map_err(server_error)?;
        if !retired.retired
            || retired.charged_bytes != 0
            || retired.live_allocations != 0
            || retired.provider_live
        {
            return Err(Error::State(
                "ordinary CUDA BLAS pool retirement incomplete".into(),
            ));
        }
        Ok(())
    }
}

/// One source-derived full projection. Weights are packed in SourceOperator
/// member order and remain source-authenticated through the compute flight.
pub(crate) struct OrdinaryCudaProjection {
    descriptor: SourceOperator,
    weights: Arc<PackedProjectionWeights>,
    operation: Bf16GemmOperation,
    selected_row: Option<usize>,
}
impl OrdinaryCudaProjection {
    pub(crate) async fn prepare(
        service: &Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        source: VerifiedTensorSource,
        descriptor: &SourceOperator,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        if descriptor.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1 {
            return Err(Error::Unsupported(
                "ordinary CUDA projection numerical contract differs".into(),
            ));
        }
        let SourceOperatorKind::Projection { selected_row } = descriptor.kind() else {
            return Err(Error::State(
                "ordinary CUDA projection requires projection descriptor".into(),
            ));
        };
        if descriptor.call().source() != source.source_identity()
            || descriptor.matrices().is_empty()
            || !descriptor.parameters().is_empty()
        {
            return Err(Error::State(
                "ordinary CUDA projection source/parameter contract differs".into(),
            ));
        }
        let rows = descriptor.call().rows().len();
        let active_rows = if selected_row.is_some() { 1 } else { rows };
        if selected_row.is_some_and(|row| row >= rows) {
            return Err(Error::State(
                "ordinary CUDA selected projection row outside call".into(),
            ));
        }
        let weights = init
            .upload_ordinary_projection(service, source, descriptor.matrices(), cancelled)
            .await?;
        weights.require_members(descriptor.matrices())?;
        let operation =
            Bf16GemmOperation::reference(weights.rows(), active_rows, weights.columns())
                .map_err(server_error)?;
        if descriptor.input_elements()?.as_slice() != [rows * weights.columns()]
            || descriptor.output_elements()?.as_slice() != [active_rows * weights.rows()]
        {
            return Err(Error::State(
                "ordinary CUDA projection operator/packed geometry differs".into(),
            ));
        }
        Ok(Arc::new(Self {
            descriptor: descriptor.clone(),
            weights,
            operation,
            selected_row,
        }))
    }

    pub(crate) fn output_bytes(&self) -> Result<usize> {
        self.operation.output_bytes().map_err(server_error)
    }

    pub(crate) fn accesses(
        &self,
        blas: &OrdinaryCudaBlas,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let input_bytes = self.operation.input_bytes().map_err(server_error)?;
        let input_offset = self
            .selected_row
            .unwrap_or(0)
            .checked_mul(self.weights.columns())
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA projection input offset overflow".into()))?;
        if input_offset
            .checked_add(input_bytes)
            .is_none_or(|end| end > input.bytes())
            || output.bytes() != self.operation.output_bytes().map_err(server_error)?
        {
            return Err(Error::State(
                "ordinary CUDA projection input/output lease extent differs".into(),
            ));
        }
        Ok(vec![
            (self.weights.lease().clone(), DeviceAccess::Read),
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
            (blas.workspace().clone(), DeviceAccess::Write),
        ])
    }

    pub(crate) fn enqueue(
        &self,
        blas: &OrdinaryCudaBlas,
        flight: &mut ResidentSubmission<'_>,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<()> {
        if self.descriptor.call().source() != self.weights.source_identity() {
            return Err(Error::State(
                "ordinary CUDA projection source owner changed".into(),
            ));
        }
        self.weights.require_members(self.descriptor.matrices())?;
        let input_offset = self
            .selected_row
            .unwrap_or(0)
            .checked_mul(self.weights.columns())
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA projection input offset overflow".into()))?;
        blas.enqueue(
            flight,
            self.operation,
            self.weights.lease(),
            input,
            input_offset,
            output,
            0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_paths_are_explicit_and_never_inferred() {
        let paths =
            OrdinaryCudaProviderPaths::new("/a/libcublas.so.12", "/b/libcublasLt.so.12").unwrap();
        assert_eq!(paths.cublas(), Path::new("/a/libcublas.so.12"));
        assert_eq!(paths.cublas_lt(), Path::new("/b/libcublasLt.so.12"));
        assert!(OrdinaryCudaProviderPaths::new("", "/b/libcublasLt.so.12").is_err());
        assert!(OrdinaryCudaProviderPaths::new("/same", "/same").is_err());
    }
}
