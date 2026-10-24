//! Unsafe diagnostic bridge; original ResidentService owns admission/completion.
use super::*;
use crate::OwnedBlasAllocationPool;
use std::sync::atomic::{AtomicBool, Ordering};

/// No provider handle, physical pointer, stream or completion authority escapes.
/// Dropping an unretired token does not destroy or unload its actor-owned state.
pub struct OwnedBf16BlasProvider {
    queue: Arc<OwnedCudaQueue>,
    sources: Arc<Bf16BlasSources>,
    pool: Arc<OwnedBlasAllocationPool>,
    retired: AtomicBool,
}
impl OwnedBf16BlasProvider {
    pub fn sources(&self) -> Bf16BlasSourceIdentity {
        self.sources.identity()
    }
    fn require_active(&self) -> Result<(), ServerError> {
        if self.retired.load(Ordering::Acquire) {
            return Err(failure("cuBLAS provider already retired"));
        }
        self.sources.require_queue(&self.queue)?;
        self.pool.require_queue(&self.queue)
    }
    /// # Safety
    /// Original submission guard and CompletionOwner already retain this exact
    /// provider, pool, opaque charge, and all original buffer leases. The caller
    /// owns exclusive output/workspace access through real completion. This is
    /// enqueue only: success is neither Ready nor numerical qualification.
    pub unsafe fn enqueue(
        &self,
        operation: ObservedBf16Gemm,
        bindings: Bf16GemmBindings,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        validate_bindings(&self.queue, operation, &bindings)?;
        let stream = self.queue.stream;
        self.queue
            .actor
            .submit_blocking(move |server| {
                server.enqueue_owned_bf16_gemm(stream, operation, bindings)
            })
            .map_err(|e| failure(format!("original cuBLAS enqueue actor: {e:?}")))?
    }
    /// # Safety
    /// All original submissions completed physically and no later enqueue can
    /// race this retirement. Original guard/CompletionOwner and opaque charge
    /// remain alive. Destroy synchronizes on the original actor; caller must
    /// complete its original trusted guard before retiring the allocation pool.
    pub unsafe fn destroy_after_physical_completion(&self) -> Result<(), ServerError> {
        self.require_active()?;
        self.retired
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| failure("cuBLAS provider retirement raced"))?;
        self.queue
            .actor
            .submit_blocking(|server| server.destroy_owned_bf16_blas())
            .map_err(|e| failure(format!("original cuBLAS destroy actor: {e:?}")))?
    }
}
fn validate_bindings(
    queue: &Arc<OwnedCudaQueue>,
    operation: ObservedBf16Gemm,
    bindings: &Bf16GemmBindings,
) -> Result<(), ServerError> {
    if bindings
        .handles()
        .iter()
        .zip(Bf16GemmBindings::sizes(operation))
        .any(|(handle, bytes)| {
            handle.service != queue.service_id()
                || handle.size() != bytes as u64
                || handle.offset_start.is_some()
                || handle.offset_end.is_some()
        })
    {
        return Err(failure(
            "cuBLAS exact unoffset original service/binding extent",
        ));
    }
    Ok(())
}
impl OwnedCudaQueue {
    /// Load exact retained DSO FDs and create on this original actor only.
    ///
    /// # Safety
    /// Pool was activated before any BLAS loading, backed by the exact original
    /// opaque ResidentService charge. An original initialization guard is armed
    /// before this call and retains pool/source owners on errors or abandonment.
    /// Ordinary buffers are already allocated; no other provider runs in this
    /// isolated process. Module/driver metadata stays in the existing coarse
    /// ProviderEnvelope; pool counters describe explicit driver allocations.
    pub unsafe fn create_bf16_blas_provider(
        self: &Arc<Self>,
        sources: Arc<Bf16BlasSources>,
        pool: Arc<OwnedBlasAllocationPool>,
    ) -> Result<Arc<OwnedBf16BlasProvider>, ServerError> {
        sources.require_queue(self)?;
        pool.require_queue(self)?;
        let retained_sources = sources.clone();
        let retained_pool = pool.clone();
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.create_owned_bf16_blas(stream, retained_sources, retained_pool)
            })
            .map_err(|e| failure(format!("original cuBLAS create actor: {e:?}")))??;
        Ok(Arc::new(OwnedBf16BlasProvider {
            queue: self.clone(),
            sources,
            pool,
            retired: AtomicBool::new(false),
        }))
    }
}
