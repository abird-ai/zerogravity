//! Typed commands on an existing CUDA actor and its pinned logical stream.
//! No context, physical stream or raw device pointer leaves that actor.

use crate::compute::CudaServer;
use cubecl_common::{device::ServiceId, device_handle::DeviceHandle};
use cubecl_core::{client::Client, server::ServerError};
use cubecl_environment::stream::StreamId;

mod compiler;
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub(crate) mod embedding_rows;
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub(crate) mod source_upload;
pub(crate) mod weight_upload;
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub use weight_upload::OwnedCudaTransferRetirement;
pub use weight_upload::SourceWeightLayout;

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
