//! Exact device-to-device transport on an owned actor stream.
use super::OwnedCudaQueue;
use cubecl_core::server::{Handle, ServerError};

impl OwnedCudaQueue {
    /// Bind an existing same-service client clone to this exact queue.
    ///
    /// # Safety
    /// The caller establishes all cross-stream dependencies and retains the
    /// original buffer and execution owners until physical completion.
    pub unsafe fn bind_client(
        &self,
        mut client: cubecl_core::client::Client,
    ) -> Result<cubecl_core::client::Client, ServerError> {
        if client.service_id() != self.service_id() {
            return Err(ServerError::graph_state(
                "owned CUDA client service differs",
            ));
        }
        unsafe { client.set_stream(self.stream) };
        Ok(client)
    }

    /// Copy an exact nonoverlapping byte span; no allocation or conversion.
    ///
    /// # Safety
    /// Both views are retained by the original admitted Read/Write flight.
    /// The caller owns producer/consumer ordering and destination initialization.
    pub unsafe fn enqueue_copy(
        &self,
        source: Handle,
        destination: Handle,
        bytes: usize,
        alignment: usize,
    ) -> Result<(), ServerError> {
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.copy_owned_views(stream, source, destination, bytes, alignment)
            })
            .map_err(|error| {
                ServerError::graph_state(format!("owned copy original actor: {error:?}"))
            })?
    }
}
