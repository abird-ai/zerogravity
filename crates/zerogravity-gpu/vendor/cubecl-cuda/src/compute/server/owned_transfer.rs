//! Same-context byte transfer using the existing driver and write scope.
use super::*;
use cubecl_server::command::Driver;

struct OwnedCopy;

fn exact_extent(binding: &BufferBinding, bytes: usize) -> bool {
    binding
        .size
        .checked_sub(binding.offset_start.unwrap_or(0))
        .and_then(|n| n.checked_sub(binding.offset_end.unwrap_or(0)))
        == Some(bytes as u64)
}

pub(super) fn require_copy_ranges(
    source: u64,
    destination: u64,
    bytes: usize,
    alignment: usize,
) -> Result<(), ServerError> {
    let end = |start: u64| {
        start
            .checked_add(bytes as u64)
            .ok_or_else(|| ServerError::graph_state("owned copy address overflow"))
    };
    let source_end = end(source)?;
    let destination_end = end(destination)?;
    if bytes == 0
        || !alignment.is_power_of_two()
        || source == 0
        || destination == 0
        || !source.is_multiple_of(alignment as u64)
        || !destination.is_multiple_of(alignment as u64)
        || (source < destination_end && destination < source_end)
    {
        return Err(ServerError::graph_state(
            "owned copy nonalias/alignment/extent differs",
        ));
    }
    Ok(())
}

/// Reuse the original command resolver and device copy driver. The enclosing
/// write scope validates source taint and marks failures on the destination.
pub(super) fn copy_exact(
    command: &mut Command<'_>,
    source: BufferBinding,
    destination: BufferBinding,
    bytes: usize,
    alignment: usize,
) -> Result<(), ServerError> {
    if command.stream().capturing.is_recording()
        || !exact_extent(&source, bytes)
        || !exact_extent(&destination, bytes)
    {
        return Err(ServerError::graph_state(
            "owned copy capture/view extent differs",
        ));
    }
    let source = command.resource(source)?;
    let destination = command.resource(destination)?;
    if source.size != bytes as u64 || destination.size != bytes as u64 {
        return Err(ServerError::graph_state(
            "owned copy actual resource extent differs",
        ));
    }
    require_copy_ranges(source.ptr, destination.ptr, bytes, alignment)?;
    // SAFETY: checked original nonoverlapping device ranges and their exact
    // stream. The enclosing physical flight retains both allocations.
    unsafe { Cuda::copy_on_device(&source, &destination, command.stream().sys) }?;
    Ok(())
}

impl CudaServer {
    pub(crate) fn copy_owned_views(
        &mut self,
        stream: StreamId,
        source: Handle,
        destination: Handle,
        bytes: usize,
        alignment: usize,
    ) -> Result<(), ServerError> {
        if source.service != self.utilities.service || destination.service != self.utilities.service
        {
            return Err(ServerError::graph_state(
                "owned copy original service differs",
            ));
        }
        let native = [source.binding(), destination.binding()];
        let mut written = self.write_set();
        written.push(native[1].clone());
        ExecuteScope::launching(
            self,
            cubecl_server::id::KernelId::new::<OwnedCopy>(),
            stream,
            std::iter::once(&native[0]),
            written,
        )
        .execute(move |server| {
            let mut command = server.command(stream, native.iter())?;
            copy_exact(
                &mut command,
                native[0].clone(),
                native[1].clone(),
                bytes,
                alignment,
            )
        })
        .into_result()
    }
}
