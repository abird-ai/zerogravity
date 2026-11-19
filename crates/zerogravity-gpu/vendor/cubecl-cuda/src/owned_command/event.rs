use super::OwnedCudaQueue;
use crate::compute::CudaServer;
use cubecl_common::device_handle::DeviceHandle;
use cubecl_core::server::ServerError;
use cubecl_environment::stream::StreamId;

/// Non-clone CUDA event owner bound to one original actor/logical stream.
///
/// This proves only physical stream completion. It never owns model leases,
/// initializes buffers or grants logical publication authority.
pub struct OwnedCudaEvent {
    actor: DeviceHandle<CudaServer>,
    stream: StreamId,
    event: usize,
    complete: bool,
}

impl OwnedCudaEvent {
    pub fn stream_id(&self) -> StreamId {
        self.stream
    }

    pub fn ready(&mut self) -> Result<bool, ServerError> {
        if self.complete {
            return Ok(true);
        }
        let stream = self.stream;
        let event = self.event;
        let ready = self
            .actor
            .submit_blocking(move |server| server.query_owned_event(stream, event, false))
            .map_err(|error| {
                ServerError::graph_state(format!("owned event query original actor: {error:?}"))
            })??;
        self.complete = ready;
        Ok(ready)
    }

    pub fn wait(&mut self) -> Result<(), ServerError> {
        if self.complete {
            return Ok(());
        }
        let stream = self.stream;
        let event = self.event;
        self.actor
            .submit_blocking(move |server| server.query_owned_event(stream, event, true))
            .map_err(|error| {
                ServerError::graph_state(format!("owned event wait original actor: {error:?}"))
            })??;
        self.complete = true;
        Ok(())
    }
}

impl Drop for OwnedCudaEvent {
    fn drop(&mut self) {
        let stream = self.stream;
        let event = self.event;
        let _ = self
            .actor
            .submit_blocking(move |server| server.destroy_owned_event(stream, event));
    }
}

impl OwnedCudaQueue {
    /// Record completion after all commands already enqueued on this exact
    /// logical stream. The returned event is only a physical completion witness.
    pub fn record_event(&self) -> Result<OwnedCudaEvent, ServerError> {
        let stream = self.stream;
        let event = self
            .actor
            .submit_blocking(move |server| server.record_owned_event(stream))
            .map_err(|error| {
                ServerError::graph_state(format!("owned event record original actor: {error:?}"))
            })??;
        Ok(OwnedCudaEvent {
            actor: self.actor.clone(),
            stream,
            event,
            complete: false,
        })
    }
}
