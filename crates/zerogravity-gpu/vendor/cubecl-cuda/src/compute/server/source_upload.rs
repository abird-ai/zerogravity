//! SAME original actor/context/stream and exact destination write tracking.
use super::*;
use crate::owned_command::source_upload::{TABLE_BYTES, window_range};

fn full_geometry(
    size: u64,
    offset_start: Option<u64>,
    offset_end: Option<u64>,
) -> Result<(), ServerError> {
    if size != TABLE_BYTES as u64 || offset_start.unwrap_or(0) != 0 || offset_end.unwrap_or(0) != 0
    {
        return Err(ServerError::graph_state(
            "source upload requires full original table handle",
        ));
    }
    Ok(())
}

impl CudaServer {
    pub(crate) fn upload_owned_tied_window<T>(
        &mut self,
        stream: StreamId,
        destination: Handle,
        offset: usize,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        let bytes = source.as_ref().as_ref();
        window_range(offset, bytes.len())?;
        let full = destination.clone().binding();
        full_geometry(full.size, full.offset_start, full.offset_end)?;
        self.upload_owned_original_window(stream, destination, TABLE_BYTES, offset, source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_subview_or_wrong_full_allocation_before_native_lookup() {
        full_geometry(TABLE_BYTES as u64, None, None).unwrap();
        full_geometry(TABLE_BYTES as u64, Some(0), Some(0)).unwrap();
        for (size, start, end) in [
            (TABLE_BYTES as u64 - 1, None, None),
            (TABLE_BYTES as u64 + 1, None, None),
            (TABLE_BYTES as u64, Some(1), None),
            (TABLE_BYTES as u64, None, Some(1)),
            (TABLE_BYTES as u64, Some(u64::MAX), None),
        ] {
            assert!(full_geometry(size, start, end).is_err());
        }
    }
}
