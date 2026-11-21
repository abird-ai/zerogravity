//! Pinned opaque-provider diagnostic; not part of forward execution.
use super::{exact_view, server_error};
use crate::resident_leases::{
    ForeignProviderCharge, ResidentLease, ResidentService, ResidentSubmission,
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
use zerogravity_core::{
    Error, Result,
    device_events::{BufferRole, DeviceAccess},
    resource_arbiter::{AllocationClass, AllocationRequest},
};
const FOREIGN_BYTES: usize = 64 * 1024 * 1024;
const SENTINEL: u32 = 0x1357_9bdf;
/// Explicit runtime provider binding. These paths are never selected by model,
/// prompt, call/node ordinals, packets or captured recipe metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaForwardProviderPaths {
    cublas: PathBuf,
    cublas_lt: PathBuf,
}
impl CudaForwardProviderPaths {
    pub fn new(cublas: impl Into<PathBuf>, cublas_lt: impl Into<PathBuf>) -> Result<Self> {
        let cublas = cublas.into();
        let cublas_lt = cublas_lt.into();
        if cublas.as_os_str().is_empty() || cublas_lt.as_os_str().is_empty() || cublas == cublas_lt
        {
            return Err(Error::Malformed(
                "CUDA forward distinct cuBLAS/cuBLASLt paths required".into(),
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

fn quarantine_blas_activation_failure(
    service: &Arc<ResidentService>,
    workspace: &ResidentLease,
    sentinel: &ResidentLease,
    sources: &Arc<Bf16BlasSources>,
    charge: &Arc<ForeignProviderCharge>,
    pool: Option<&Arc<OwnedBlasAllocationPool>>,
    provider: Option<&Arc<OwnedBf16BlasProvider>>,
) {
    let mut owners: Vec<crate::physical_quarantine::CompletionOwner> =
        vec![sources.clone(), charge.clone()];
    if let Some(pool) = pool {
        owners.push(pool.clone());
    }
    if let Some(provider) = provider {
        owners.push(provider.clone());
    }
    service.fence_and_retain_owners(&[workspace.clone(), sentinel.clone()], owners);
}

struct BlasRetirementReservation<'a> {
    flag: &'a AtomicBool,
    irreversible: bool,
}
impl<'a> BlasRetirementReservation<'a> {
    fn reserve(flag: &'a AtomicBool) -> Result<Self> {
        flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("CUDA forward BLAS retirement raced".into()))?;
        Ok(Self {
            flag,
            irreversible: false,
        })
    }

    fn make_irreversible(&mut self) {
        self.irreversible = true;
    }
}
impl Drop for BlasRetirementReservation<'_> {
    fn drop(&mut self) {
        if !self.irreversible {
            self.flag.store(false, Ordering::Release);
        }
    }
}

/// The one active opaque BLAS provider for an forward ResidentService.
/// Explicit retirement is mandatory; Drop never pretends the provider retired.
pub(crate) struct CudaForwardBlas {
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
impl CudaForwardBlas {
    /// Verify provider source first, then allocate/activate only through the
    /// existing pinned service and physical queue.
    pub(crate) async fn prepare(
        service: Arc<ResidentService>,
        expected_root: u64,
        paths: &CudaForwardProviderPaths,
    ) -> Result<Arc<Self>> {
        service.ensure_usable()?;
        let queue = service.pinned_cuda_queue()?.clone();
        let sources = queue
            .verify_bf16_blas_sources(paths.cublas(), paths.cublas_lt())
            .map_err(server_error)?;
        let charge: Arc<ForeignProviderCharge> = service.reserve_foreign_provider(FOREIGN_BYTES)?;
        if charge.bytes() != FOREIGN_BYTES {
            return Err(Error::State(
                "CUDA forward exact opaque provider charge differs".into(),
            ));
        }
        charge.require_service(&service)?;

        let init_bytes = REFERENCE_BF16_GEMM_WORKSPACE_BYTES
            .checked_add(4)
            .ok_or_else(|| {
                Error::Limit("CUDA forward BLAS initialization bytes overflow".into())
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
        let pool = match unsafe {
            queue.activate_blas_allocation_pool(FOREIGN_BYTES as u64, charge.clone())
        } {
            Ok(pool) => pool,
            Err(error) => {
                quarantine_blas_activation_failure(
                    &service, &workspace, &sentinel, &sources, &charge, None, None,
                );
                return Err(server_error(error));
            }
        };
        let ticket = match service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: FOREIGN_BYTES as u64,
            }],
            None,
        ) {
            Ok(ticket) => ticket,
            Err(error) => {
                quarantine_blas_activation_failure(
                    &service,
                    &workspace,
                    &sentinel,
                    &sources,
                    &charge,
                    Some(&pool),
                    None,
                );
                return Err(error);
            }
        };
        let mut slot = match ticket.wait().await {
            Ok(slot) => slot,
            Err(error) => {
                quarantine_blas_activation_failure(
                    &service,
                    &workspace,
                    &sentinel,
                    &sources,
                    &charge,
                    Some(&pool),
                    None,
                );
                return Err(error);
            }
        };
        if let Err(error) = slot.begin_submission() {
            quarantine_blas_activation_failure(
                &service,
                &workspace,
                &sentinel,
                &sources,
                &charge,
                Some(&pool),
                None,
            );
            return Err(error);
        }
        let mut flight =
            match service.submit(expected_root, &[(sentinel.clone(), DeviceAccess::Read)]) {
                Ok(flight) => flight,
                Err(error) => {
                    quarantine_blas_activation_failure(
                        &service,
                        &workspace,
                        &sentinel,
                        &sources,
                        &charge,
                        Some(&pool),
                        None,
                    );
                    return Err(error);
                }
            };
        if let Err(error) = flight.retain_completion_owner(sources.clone()) {
            service.fence_and_retain_owners(&[], vec![pool.clone(), charge.clone()]);
            return Err(error);
        }
        if let Err(error) = flight.retain_completion_owner(pool.clone()) {
            service.fence_and_retain_owners(&[], vec![charge.clone()]);
            return Err(error);
        }
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
                        "CUDA forward BLAS creation sentinel differs".into(),
                    ));
                }
                Ok(())
            })
            .await?;
        if let Err(error) = slot.finish() {
            quarantine_blas_activation_failure(
                &service,
                &workspace,
                &sentinel,
                &sources,
                &charge,
                Some(&pool),
                Some(&provider),
            );
            return Err(error);
        }

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

    pub(crate) fn require_service(&self, service: &Arc<ResidentService>) -> Result<()> {
        self.require_active()?;
        if !Arc::ptr_eq(&self.service, service) {
            return Err(Error::State(
                "CUDA forward BLAS physical service differs".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        vec![
            (self.workspace.clone(), DeviceAccess::Write),
            (self.sentinel.clone(), DeviceAccess::Read),
        ]
    }

    pub(crate) fn execution_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        vec![(self.workspace.clone(), DeviceAccess::Write)]
    }

    fn require_active(&self) -> Result<()> {
        self.service.ensure_usable()?;
        self.charge.require_service(&self.service)?;
        if self.retired.load(Ordering::Acquire) {
            return Err(Error::State("CUDA forward BLAS owner retired".into()));
        }
        let pool = self.pool.snapshot().map_err(server_error)?;
        if pool.poisoned || pool.retired || !pool.provider_live {
            return Err(Error::State(
                "CUDA forward BLAS allocation/provider state unavailable".into(),
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
                "CUDA forward BLAS weights/workspace extent differs".into(),
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

    fn quarantine_retirement_failure(&self) {
        self.service.fence_and_retain_owners(
            &[self.workspace.clone(), self.sentinel.clone()],
            vec![
                self.sources.clone(),
                self.provider.clone(),
                self.pool.clone(),
                self.charge.clone(),
            ],
        );
    }

    /// Destroy/unload only after a trusted completion fence on the same service.
    /// Pre-flight admission/queue failures remain retryable. Once the retirement
    /// flight retains every opaque owner, any later failure is irreversible and
    /// stays on the original quarantine path.
    pub(crate) async fn retire(&self, expected_root: u64) -> Result<()> {
        self.require_active()?;
        let mut reservation = BlasRetirementReservation::reserve(&self.retired)?;
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
        reservation.make_irreversible();
        unsafe { self.provider.destroy_after_physical_completion() }.map_err(server_error)?;
        flight
            .complete_read_only_readback(&[self.sentinel.readback(0, 4)?], |actual| {
                if actual.len() != 1 || actual[0].as_ref() != SENTINEL.to_le_bytes() {
                    return Err(Error::State(
                        "CUDA forward BLAS retirement sentinel differs".into(),
                    ));
                }
                Ok(())
            })
            .await?;
        if let Err(error) = slot.finish() {
            self.quarantine_retirement_failure();
            return Err(error);
        }
        let retired = match unsafe { self.pool.retire_after_physical_completion() } {
            Ok(retired) => retired,
            Err(error) => {
                self.quarantine_retirement_failure();
                return Err(server_error(error));
            }
        };
        if !retired.retired
            || retired.charged_bytes != 0
            || retired.live_allocations != 0
            || retired.provider_live
        {
            self.quarantine_retirement_failure();
            return Err(Error::State(
                "CUDA forward BLAS pool retirement incomplete".into(),
            ));
        }
        Ok(())
    }
}

mod tests {
    use super::*;

    #[test]
    fn retirement_reservation_retries_before_irreversible_boundary() {
        let retired = AtomicBool::new(false);
        {
            let _reservation = BlasRetirementReservation::reserve(&retired).unwrap();
            assert!(retired.load(Ordering::Acquire));
            assert!(BlasRetirementReservation::reserve(&retired).is_err());
        }
        assert!(!retired.load(Ordering::Acquire));

        {
            let mut reservation = BlasRetirementReservation::reserve(&retired).unwrap();
            reservation.make_irreversible();
        }
        assert!(retired.load(Ordering::Acquire));
        assert!(BlasRetirementReservation::reserve(&retired).is_err());
    }

    #[test]
    fn provider_paths_are_explicit_and_never_inferred() {
        let paths =
            CudaForwardProviderPaths::new("/a/libcublas.so.12", "/b/libcublasLt.so.12").unwrap();
        assert_eq!(paths.cublas(), Path::new("/a/libcublas.so.12"));
        assert_eq!(paths.cublas_lt(), Path::new("/b/libcublasLt.so.12"));
        assert!(CudaForwardProviderPaths::new("", "/b/libcublasLt.so.12").is_err());
        assert!(CudaForwardProviderPaths::new("/same", "/same").is_err());
    }
}
