//! Typed commands on an existing CUDA actor and its pinned logical stream.
//! No context, physical stream or raw device pointer leaves that actor.

use crate::compute::CudaServer;
use cubecl_common::{device::ServiceId, device_handle::DeviceHandle};
use cubecl_core::{
    client::Client,
    server::{Handle, ServerError},
};
use cubecl_environment::stream::StreamId;
use std::sync::Arc;

mod compiler;
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
mod event;
#[cfg(all(feature = "flash-attn", target_os = "linux"))]
mod flash_attn;
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub(crate) mod embedding_rows;
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub(crate) mod source_upload;
pub(crate) mod weight_upload;
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub use weight_upload::OwnedCudaTransferRetirement;
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub use event::OwnedCudaEvent;
pub use weight_upload::SourceWeightLayout;
#[cfg(all(feature = "flash-attn", target_os = "linux"))]
pub use flash_attn::{Bf16FlashAttention, Bf16FlashAttentionBindings};

#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub(crate) mod cubin;
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub use cubin::{
    Bf16VocabularyHead, CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes,
    CubinParameter, CubinPointerFixup, CubinProvider, CubinWorkspaceClear, CubinWorkspaceView,
    OwnedCudaBinding, OwnedCudaKernel, OwnedCudaModuleSet, VerifiedCubin, VocabularyHeadSelection,
};

#[cfg(target_os = "linux")]
mod bf16_gemm;
#[cfg(target_os = "linux")]
pub use bf16_gemm::{
    Bf16BlasSourceIdentity, Bf16BlasSources, Bf16GemmApi, Bf16GemmBindings, Bf16GemmInspection,
    Bf16GemmOperation, ObservedBf16Gemm, REFERENCE_BF16_GEMM_API,
    REFERENCE_BF16_GEMM_WORKSPACE_BYTES,
};
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub use bf16_gemm::{
    CapturedBf16CubinRecord, PinnedBf16Cubin, PinnedBf16CubinDerivation, PinnedCapturedBf16Cubin,
};

/// Retains the original CUDA actor and the stream selected before client use.
/// This token does not establish process-wide or device-wide capture isolation.
pub struct OwnedCudaQueue {
    pub(crate) actor: DeviceHandle<CudaServer>,
    pub(crate) stream: StreamId,
}

impl core::fmt::Debug for OwnedCudaQueue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OwnedCudaQueue")
            .field("service", &self.service_id())
            .field("stream", &self.stream)
            .finish()
    }
}

impl OwnedCudaQueue {
    /// Pin a moved, unused client to this thread's current logical stream.
    /// The live client keeps the same registry service present while its typed
    /// actor handle is acquired; this never initializes a second service.
    ///
    /// # Safety
    ///
    /// `client` must be freshly created, never cloned and never used for any
    /// allocation, transfer, submission or stream override. All subsequent own
    /// work must use the returned client (or clones made after this call).
    /// Other clients of this actor can exist; global graph capture additionally
    /// requires the caller to establish an isolated process/device boundary.
    pub unsafe fn pin_cold(mut client: Client) -> Result<(Client, Self), ServerError> {
        let service = client.service_id();
        if service != ServiceId::of::<CudaServer>(service.device) {
            return Err(ServerError::graph_state(
                "original queue pin requires a CUDA client",
            ));
        }
        let stream = StreamId::current();
        // SAFETY: the caller transfers a never-used/never-cloned client. No
        // earlier binding or operation can depend on its former stream choice.
        unsafe { client.set_stream(stream) };
        let actor = DeviceHandle::<CudaServer>::new(service.device);
        Ok((client, Self { actor, stream }))
    }

    /// Original registry actor identity, including its concrete server type.
    pub fn service_id(&self) -> ServiceId {
        self.actor.service_id()
    }

    /// Logical stream identity; the underlying `CUstream` remains server-owned.
    pub fn logical_stream_id(&self) -> StreamId {
        self.stream
    }

    /// Current device-wide CUDA free/total memory from the original actor/context.
    /// This is an instantaneous admission observation, never a reservation.
    pub fn memory_info(&self) -> Result<(usize, usize), ServerError> {
        self.actor
            .submit_blocking(|server| server.owned_memory_info())
            .map_err(|error| {
                ServerError::graph_state(format!("CUDA memory info original actor: {error:?}"))
            })?
    }

    /// Enqueue one bounded immutable host window into an exact view of a full
    /// original allocation. This does not retain the host owner after enqueue.
    ///
    /// # Safety
    /// The caller must retain the source, complete destination allocation and
    /// every downstream consumer until actual completion/quarantine.
    pub unsafe fn enqueue_owned_source_window<T>(
        &self,
        destination: Handle,
        full_bytes: usize,
        offset: usize,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.upload_owned_original_window(
                    stream,
                    destination,
                    full_bytes,
                    offset,
                    source,
                )
            })
            .map_err(|error| {
                ServerError::graph_state(format!("owned source window original actor: {error:?}"))
            })?
    }

    /// Bind one same-service client clone to a fresh logical CUDA stream while
    /// retaining the original actor/context. This creates no second service.
    ///
    /// # Safety
    ///
    /// The caller must provide a client for this exact CUDA service and enforce
    /// all cross-stream buffer dependencies. Rebinding affects only this client
    /// value; existing clones retain their prior stream.
    pub unsafe fn fork_stream(&self, mut client: Client) -> Result<(Client, Self), ServerError> {
        if client.service_id() != self.service_id() {
            return Err(ServerError::graph_state(
                "owned CUDA secondary stream requires the same service",
            ));
        }
        let stream = loop {
            let candidate = StreamId::allocate();
            if candidate != self.stream {
                break candidate;
            }
        };
        // SAFETY: this client clone is intentionally dedicated to the new
        // logical stream; caller owns cross-stream dependency correctness.
        unsafe { client.set_stream(stream) };
        Ok((
            client,
            Self {
                actor: self.actor.clone(),
                stream,
            },
        ))
    }
}

#[cfg(all(
    feature = "owned-cublas-allocator",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub use bf16_gemm::OwnedBf16BlasProvider;

#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub use embedding_rows::{Bf16MatrixLayout, TextEmbeddingWidth};
