//! Shared original H2D implementation; only closed layout wrappers call it.
use super::*;
use crate::owned_command::weight_upload::Vllm08WeightLayout;

fn full_geometry(
    expected: usize,
    size: u64,
    start: Option<u64>,
    end: Option<u64>,
) -> Result<(), ServerError> {
    if size != expected as u64 || start.unwrap_or(0) != 0 || end.unwrap_or(0) != 0 {
        return Err(ServerError::graph_state(
            "source upload requires exact original full allocation",
        ));
    }
    Ok(())
}
enum UploadDestination {
    Original(Handle),
    #[cfg(all(feature = "owned-cubin", target_os = "linux"))]
    Workspace(crate::OwnedCudaBinding),
}
impl CudaServer {
    pub(crate) fn upload_owned_vllm08_weight_window<T>(
        &mut self,
        stream: StreamId,
        destination: Handle,
        layout: Vllm08WeightLayout,
        offset: usize,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        layout.window(offset, source.as_ref().as_ref().len())?;
        self.upload_owned_original_window(stream, destination, layout.bytes(), offset, source)
    }
    /// Both callers validate their exact original source layout BEFORE entering.
    /// No public arbitrary-extent upload method or new native provider exists.
    pub(crate) fn upload_owned_original_window<T>(
        &mut self,
        stream: StreamId,
        destination: Handle,
        full_bytes: usize,
        offset: usize,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        self.upload_owned_window_checked(
            stream,
            UploadDestination::Original(destination),
            full_bytes,
            offset,
            source,
        )
    }
    #[cfg(all(feature = "owned-cubin", target_os = "linux"))]
    pub(crate) fn upload_owned_workspace_rows<T>(
        &mut self,
        stream: StreamId,
        destination: crate::OwnedCudaBinding,
        bytes: usize,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        if source.as_ref().as_ref().len() != bytes {
            return Err(ServerError::graph_state(
                "workspace authentic complete row image differs",
            ));
        }
        self.upload_owned_window_checked(
            stream,
            UploadDestination::Workspace(destination),
            bytes,
            0,
            source,
        )
    }
    fn upload_owned_window_checked<T>(
        &mut self,
        stream: StreamId,
        destination: UploadDestination,
        full_bytes: usize,
        offset: usize,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        let bytes = source.as_ref().as_ref().len();
        if bytes == 0
            || bytes > 1024 * 1024
            || offset.checked_add(bytes).is_none_or(|end| end > full_bytes)
        {
            return Err(ServerError::graph_state(
                "bounded original source copy range",
            ));
        }
        let (original_service, full) = match &destination {
            UploadDestination::Original(handle) => {
                let full = handle.clone().binding();
                full_geometry(full_bytes, full.size, full.offset_start, full.offset_end)?;
                (handle.service == self.utilities.service, full)
            }
            #[cfg(all(feature = "owned-cubin", target_os = "linux"))]
            UploadDestination::Workspace(binding) => {
                binding.require_copy(self.utilities.service, full_bytes as u64)?;
                (true, binding.handle().binding())
            }
        };
        let original_stream = full.stream == stream;
        let mut region = full.clone();
        region.offset_start = Some(
            full.offset_start
                .unwrap_or(0)
                .checked_add(offset as u64)
                .ok_or_else(|| {
                    ServerError::graph_state("source upload exact view start overflow")
                })?,
        );
        region.offset_end = Some(
            full.offset_end
                .unwrap_or(0)
                .checked_add((full_bytes - offset - bytes) as u64)
                .ok_or_else(|| ServerError::graph_state("source upload exact view end overflow"))?,
        );
        let mut written = self.write_set();
        written.push(region.clone());
        ExecuteScope::over(self, stream, written)
            .execute(move |server| {
                let _destination = destination; // Retain actual view/run through acknowledgement.
                if !original_service || !original_stream {
                    return Err(ServerError::graph_state(
                        "source upload original service/stream differs",
                    ));
                }
                let mut command = server.command(stream, [&region].into_iter())?;
                if command.stream().capturing.is_recording() {
                    return Err(ServerError::graph_state(
                        "source upload graph capture unsupported",
                    ));
                }
                let original = command.resource(full)?;
                let target = command.resource(region)?;
                if original.size != full_bytes as u64
                    || target.size != bytes as u64
                    || original.ptr == 0
                    || original.ptr.checked_add(offset as u64) != Some(target.ptr)
                    || target.ptr.checked_add(bytes as u64).is_none()
                    || original.ptr.checked_add(full_bytes as u64).is_none()
                {
                    return Err(ServerError::graph_state(
                        "source upload native allocation/view extent",
                    ));
                }
                let host = source.as_ref().as_ref();
                if host.len() != bytes {
                    return Err(ServerError::graph_state(
                        "source upload host extent changed",
                    ));
                }
                // Exact original host-window Arc is retained in THIS actor closure
                // and original initialization until its real barrier or quarantine.
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        target.ptr,
                        host,
                        command.stream().sys,
                    )
                }
                .map_err(|error| ServerError::graph_state(format!("original source H2D: {error}")))
            })
            .into_result()
    }
    #[cfg(all(feature = "owned-cubin", target_os = "linux"))]
    pub(crate) fn checkpoint_owned_streamed_head<T>(
        &mut self,
        stream: StreamId,
        weight: Handle,
        output: Handle,
        range: std::ops::Range<u64>,
        source: Arc<T>,
        verification: Option<Arc<std::sync::Mutex<Vec<u8>>>>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        let original_service =
            weight.service == self.utilities.service && output.service == self.utilities.service;
        let w = weight.binding();
        let y = output.binding();
        full_geometry(1024 * 1024, w.size, w.offset_start, w.offset_end)?;
        full_geometry(248320 * 2, y.size, y.offset_start, y.offset_end)?;
        if range.start >= range.end
            || range.end > 248320 * 2
            || range.start % 16 != 0
            || ![512, 1024].contains(&(range.end - range.start))
        {
            return Err(ServerError::graph_state(
                "streamed head exact physical checkpoint range",
            ));
        }
        let mut region = y.clone();
        region.offset_start = Some(range.start);
        region.offset_end = Some(248320 * 2 - range.end);
        self.streams.ensure_written([&w, &region].into_iter())?;
        let mut written = self.write_set();
        written.push(w.clone());
        written.push(region.clone());
        ExecuteScope::over(self, stream, written)
            .execute(move |server| {
                if !original_service || w.stream != stream || y.stream != stream {
                    return Err(ServerError::graph_state(
                        "streamed head checkpoint original queue",
                    ));
                }
                let mut command = server.command(stream, [&w, &region].into_iter())?;
                if command.stream().capturing.is_recording() {
                    return Err(ServerError::graph_state(
                        "streamed head checkpoint graph capture unsupported",
                    ));
                }
                // Audit buffer was admitted/allocated before launch. Production
                // passesNone and performs no D2H or output comparison here.
                let mut bounce = verification
                    .as_ref()
                    .map(|b| {
                        b.lock().map_err(|_| {
                            ServerError::graph_state("streamed head audit buffer poisoned")
                        })
                    })
                    .transpose()?;
                if let Some(bytes) = bounce.as_mut() {
                    if bytes.len() != 1024 * 1024
                        || bytes.capacity() != 1024 * 1024
                        || source.as_ref().as_ref().len() != 1024 * 1024
                    {
                        return Err(ServerError::graph_state(
                            "streamed head admitted audit/source exact1MiB",
                        ));
                    }
                    let resource = command.resource(w.clone())?;
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_async(
                            bytes.as_mut_slice(),
                            resource.ptr,
                            command.stream().sys,
                        )
                    }
                    .map_err(|e| {
                        ServerError::graph_state(format!("streamed head audit copy: {e}"))
                    })?;
                }
                // No initialization flag or event/physical retirement. Failure
                // leaves the original flight armed with its exact pending window.
                unsafe { cudarc::driver::result::stream::synchronize(command.stream().sys) }
                    .map_err(|e| {
                        ServerError::graph_state(format!("streamed head physical checkpoint: {e}"))
                    })?;
                if bounce
                    .as_ref()
                    .is_some_and(|bytes| bytes.as_slice() != source.as_ref().as_ref())
                {
                    return Err(ServerError::graph_state(
                        "streamed head actual consumed W source bytes differ",
                    ));
                }
                Ok(())
            })
            .into_result()
    }
    /// Private shared retirement primitive; typed callers supply previously
    /// checked complete allocation/range geometry. No copy or kernel permit.
    #[cfg(all(feature = "owned-cubin", target_os = "linux"))]
    pub(crate) fn record_owned_source_retirement(
        &mut self,
        stream: StreamId,
        handles: [Handle; 2],
        full_bytes: [usize; 2],
        range: std::ops::Range<u64>,
        reusable: Option<usize>,
    ) -> Result<usize, ServerError> {
        let [weight, output] = handles;
        let original_service =
            weight.service == self.utilities.service && output.service == self.utilities.service;
        let w = weight.binding();
        let y = output.binding();
        full_geometry(full_bytes[0], w.size, w.offset_start, w.offset_end)?;
        full_geometry(full_bytes[1], y.size, y.offset_start, y.offset_end)?;
        if range.start >= range.end || range.end > full_bytes[1] as u64 {
            return Err(ServerError::graph_state(
                "source retirement exact output range",
            ));
        }
        let mut region = y.clone();
        region.offset_start = Some(range.start);
        region.offset_end = Some(full_bytes[1] as u64 - range.end);
        self.streams.ensure_written([&w, &region].into_iter())?;
        let mut written = self.write_set();
        written.push(w.clone());
        written.push(region.clone());
        ExecuteScope::over(self, stream, written)
            .execute(move |server| {
                if !original_service || w.stream != stream || y.stream != stream {
                    return Err(ServerError::graph_state(
                        "source retirement original service/stream",
                    ));
                }
                let mut command = server.command(stream, [&w, &region].into_iter())?;
                if command.stream().capturing.is_recording() {
                    return Err(ServerError::graph_state(
                        "source retirement capture unsupported",
                    ));
                }
                let wr = command.resource(w)?;
                let original_y = command.resource(y)?;
                let yr = command.resource(region)?;
                if original_y.size != full_bytes[1] as u64
                    || original_y.ptr == 0
                    || original_y.ptr.checked_add(range.start) != Some(yr.ptr)
                    || original_y.ptr.checked_add(original_y.size).is_none()
                    || !(wr
                        .ptr
                        .checked_add(wr.size)
                        .is_some_and(|end| end <= original_y.ptr)
                        || original_y
                            .ptr
                            .checked_add(original_y.size)
                            .is_some_and(|end| end <= wr.ptr))
                    || wr.size != full_bytes[0] as u64
                    || wr.ptr == 0
                    || yr.size != range.end - range.start
                    || yr.ptr == 0
                    || wr.ptr.checked_add(wr.size).is_none()
                    || yr.ptr.checked_add(yr.size).is_none()
                {
                    return Err(ServerError::graph_state(
                        "source retirement native allocation/range",
                    ));
                }
                let event = match reusable {
                    Some(event) => event as cudarc::driver::sys::CUevent,
                    None => cudarc::driver::result::event::create(
                        cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
                    )
                    .map_err(|e| {
                        ServerError::graph_state(format!("source retirement event create: {e}"))
                    })?,
                };
                // SAFETY: original actor owns this event and the original logical
                // stream; prior H2D/native commands are on this exact CUstream.
                let recorded =
                    unsafe { cudarc::driver::result::event::record(event, command.stream().sys) };
                if let Err(error) = recorded {
                    if reusable.is_none() {
                        // SAFETY: failed first record has no other event owner.
                        let _ = unsafe { cudarc::driver::result::event::destroy(event) };
                    }
                    return Err(ServerError::graph_state(format!(
                        "source retirement event record: {error}"
                    )));
                }
                Ok(event as usize)
            })
            .into_result()
    }

    #[cfg(all(feature = "owned-cubin", target_os = "linux"))]
    pub(crate) fn destroy_owned_source_retirement(
        &mut self,
        stream: StreamId,
        event: usize,
    ) -> Result<(), ServerError> {
        let _command = self.command(stream, std::iter::empty())?;
        // SAFETY: non-Clone opaque owner consumed its last token on the
        // original actor/context. CUDA destruction never cancels stream work.
        unsafe { cudarc::driver::result::event::destroy(event as cudarc::driver::sys::CUevent) }
            .map_err(|e| ServerError::graph_state(format!("source retirement event destroy: {e}")))
    }

    #[cfg(all(feature = "owned-cubin", target_os = "linux"))]
    pub(crate) fn query_owned_source_retirement(
        &mut self,
        stream: StreamId,
        event: usize,
        wait: bool,
    ) -> Result<bool, ServerError> {
        // Only the opaque original actor/stream owner can supply an event.
        // ensure_stream obtains that same original context before driver calls.
        let mut command = self.command(stream, std::iter::empty())?;
        if command.stream().capturing.is_recording() || event == 0 {
            return Err(ServerError::graph_state(
                "source retirement query capture/event",
            ));
        }
        let event = event as cudarc::driver::sys::CUevent;
        if wait {
            // SAFETY: mark is live/non-Clone; every dependency precedes its
            // record. Waiting requires no queued actor command to progress.
            unsafe { cudarc::driver::result::event::synchronize(event) }.map_err(|e| {
                ServerError::graph_state(format!("source retirement event wait: {e}"))
            })?;
            return Ok(true);
        }
        // SAFETY: same original actor context and live recorded event owner.
        match unsafe { cudarc::driver::result::event::query(event) } {
            Ok(()) => Ok(true),
            Err(e) if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            Err(e) => Err(ServerError::graph_state(format!(
                "source retirement event query: {e}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closed_full_layouts_cannot_be_reinterpreted_or_supplied_as_subviews() {
        for layout in [
            Vllm08WeightLayout::GdnQkvz,
            Vllm08WeightLayout::GdnBa,
            Vllm08WeightLayout::GdnOutput,
            Vllm08WeightLayout::AttentionQkv,
            Vllm08WeightLayout::AttentionOutput,
            Vllm08WeightLayout::MlpGateUp,
            Vllm08WeightLayout::MlpDown,
        ] {
            let bytes = layout.bytes();
            full_geometry(bytes, bytes as u64, None, None).unwrap();
            full_geometry(bytes, bytes as u64, Some(0), Some(0)).unwrap();
            for (size, start, end) in [
                (bytes as u64 - 1, None, None),
                (bytes as u64 + 1, None, None),
                (bytes as u64, Some(1), None),
                (bytes as u64, None, Some(1)),
                (bytes as u64, Some(u64::MAX), None),
            ] {
                assert!(full_geometry(bytes, size, start, end).is_err());
            }
        }
    }
}
