use super::CudaServer;
use cubecl_core::server::ServerError;
use cubecl_environment::stream::StreamId;

impl CudaServer {
    pub(crate) fn wait_owned_event(
        &mut self,
        stream: StreamId,
        event: usize,
    ) -> Result<(), ServerError> {
        let mut command = self.command(stream, std::iter::empty())?;
        if command.stream().capturing.is_recording() || event == 0 {
            return Err(ServerError::graph_state(
                "owned event dependency capture/event",
            ));
        }
        // SAFETY: the owned event is live on this actor/context and has been
        // recorded; the wait is enqueued before the owner can be dropped.
        unsafe {
            cudarc::driver::result::stream::wait_event(
                command.stream().sys,
                event as cudarc::driver::sys::CUevent,
                cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }
        .map_err(|error| ServerError::graph_state(format!("owned event dependency: {error}")))
    }

    pub(crate) fn record_owned_event(&mut self, stream: StreamId) -> Result<usize, ServerError> {
        let mut command = self.command(stream, std::iter::empty())?;
        if command.stream().capturing.is_recording() {
            return Err(ServerError::graph_state(
                "owned event cannot record during graph capture",
            ));
        }
        let event = cudarc::driver::result::event::create(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        )
        .map_err(|error| ServerError::graph_state(format!("owned event create: {error}")))?;
        // SAFETY: the actor owns the CUDA context, event and exact stream.
        if let Err(error) =
            unsafe { cudarc::driver::result::event::record(event, command.stream().sys) }
        {
            // SAFETY: failed first record has no other event owner.
            let _ = unsafe { cudarc::driver::result::event::destroy(event) };
            return Err(ServerError::graph_state(format!(
                "owned event record: {error}"
            )));
        }
        Ok(event as usize)
    }

    pub(crate) fn query_owned_event(
        &mut self,
        stream: StreamId,
        event: usize,
        wait: bool,
    ) -> Result<bool, ServerError> {
        let mut command = self.command(stream, std::iter::empty())?;
        if command.stream().capturing.is_recording() || event == 0 {
            return Err(ServerError::graph_state("owned event query capture/event"));
        }
        let event = event as cudarc::driver::sys::CUevent;
        if wait {
            // SAFETY: live actor-owned event; waiting needs no later actor work.
            unsafe { cudarc::driver::result::event::synchronize(event) }
                .map_err(|error| ServerError::graph_state(format!("owned event wait: {error}")))?;
            return Ok(true);
        }
        // SAFETY: live actor-owned event in the same CUDA context.
        match unsafe { cudarc::driver::result::event::query(event) } {
            Ok(()) => Ok(true),
            Err(error) if error.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                Ok(false)
            }
            Err(error) => Err(ServerError::graph_state(format!(
                "owned event query: {error}"
            ))),
        }
    }

    pub(crate) fn destroy_owned_event(
        &mut self,
        stream: StreamId,
        event: usize,
    ) -> Result<(), ServerError> {
        let _command = self.command(stream, std::iter::empty())?;
        if event == 0 {
            return Err(ServerError::graph_state("owned event destroy zero event"));
        }
        // SAFETY: non-clone owner drops the event on the original actor/context.
        unsafe { cudarc::driver::result::event::destroy(event as cudarc::driver::sys::CUevent) }
            .map_err(|error| ServerError::graph_state(format!("owned event destroy: {error}")))
    }
}
