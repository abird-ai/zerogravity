//! No inference/buffer side effect: use the original actor/compiler with Result.
use super::*;
use cubecl_core::prelude::CubeKernel;
use std::{any::Any, sync::Arc};

impl OwnedCudaQueue {
    /// Bounded cumulative charges, not a persisted key database or timing trace.
    pub fn compiler_context_usage(
        &self,
    ) -> Result<Option<crate::CompilerContextUsage>, ServerError> {
        self.actor
            .submit_blocking(|server| server.owned_compiler_context_usage())
            .map_err(|e| ServerError::graph_state(format!("original compiler actor: {e:?}")))
    }
    /// Source-neutral lifetime owner already installed on this original context.
    pub fn compiler_context_owner<T: Any + Send + Sync>(
        &self,
    ) -> Result<Option<Arc<T>>, ServerError> {
        self.actor
            .submit_blocking(|server| {
                server
                    .owned_compiler_context_owner()
                    .map(|owner| {
                        owner.downcast::<T>().map_err(|_| {
                            ServerError::graph_state("CUDA compiler context owner type differs")
                        })
                    })
                    .transpose()
            })
            .map_err(|e| ServerError::graph_state(format!("original compiler actor: {e:?}")))?
    }
    /// # Safety
    /// `owner` contains the fixed policy's actual original host/device leases,
    /// no Client/source/model backedge. Caller has retained it on the SAME
    /// original armed physical owner before compiler effects. No unpriced cache owner.
    pub unsafe fn install_compiler_context<T: Any + Send + Sync>(
        &self,
        owner: Arc<T>,
        adopted: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), ServerError> {
        self.actor
            .submit_blocking(move |server| server.install_owned_compiler_context(owner, adopted))
            .map_err(|e| ServerError::graph_state(format!("original compiler actor: {e:?}")))?
    }
    /// # Safety
    /// Same admitted compiler/context and original armed physical compiler scope retains
    /// all immutable definition/charge owners before this call. Definition is
    /// from the SAME checked factory as actual unchecked launch. No dummy run.
    pub unsafe fn prepare_kernel(&self, kernel: Box<dyn CubeKernel>) -> Result<(), ServerError> {
        self.actor
            .submit_blocking(move |server| server.prepare_owned_kernel(kernel))
            .map_err(|e| ServerError::graph_state(format!("original compiler actor: {e:?}")))?
    }

    /// Compile/load without launch, then return the driver-reflected formal ABI.
    ///
    /// # Safety
    /// Same admitted compiler/context and retained immutable definition contract
    /// as [`Self::prepare_kernel`].
    #[cfg(target_os = "linux")]
    pub unsafe fn diagnose_kernel_parameters(
        &self,
        kernel: Box<dyn CubeKernel>,
        parameter_count: usize,
    ) -> Result<Vec<(usize, usize)>, ServerError> {
        self.actor
            .submit_blocking(move |server| {
                server.diagnose_owned_kernel_parameters(kernel, parameter_count)
            })
            .map_err(|e| ServerError::graph_state(format!("original compiler actor: {e:?}")))?
    }
}
