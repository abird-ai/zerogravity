//! Bounded original-host H2D; no allocator, extra stream or Ready authority.
use super::OwnedCudaQueue;
use cubecl_core::server::{Handle, ServerError};
use std::sync::Arc;

pub(crate) const TABLE_BYTES: usize = 508_559_360;
pub(crate) const WINDOW_BYTES: usize = 1024 * 1024;

pub(crate) fn window_range(offset: usize, bytes: usize) -> Result<(), ServerError> {
    if bytes != WINDOW_BYTES
        || !offset.is_multiple_of(WINDOW_BYTES)
        || offset
            .checked_add(bytes)
            .is_none_or(|end| end > TABLE_BYTES)
    {
        return Err(ServerError::graph_state(
            "closed original BF16 table window",
        ));
    }
    Ok(())
}

impl OwnedCudaQueue {
    /// Enqueue one1MiB of the original485MiB BF16 table, without a host bounce.
    ///
    /// # Safety
    /// `source` exposes immutable, already-admitted ordinary host memory. The
    /// original armed initialization must retain its SAME Arc BEFORE this call
    /// and until an actual original-stream `sync_buffers` succeeds, or retain
    /// it in original unknown-work quarantine. This method only enqueues; it
    /// does not report physical completion, coverage or initialized contents.
    /// The full destination must be this original service's admitted,
    /// uninitialized lease with the original initialization's exclusive Write;
    /// no mathematical reader or initialized immutable weight may be present.
    pub unsafe fn enqueue_tied_bf16_window<T>(
        &self,
        destination: Handle,
        offset: usize,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        window_range(offset, source.as_ref().as_ref().len())?;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.upload_owned_tied_window(stream, destination, offset, source)
            })
            .map_err(|e| ServerError::graph_state(format!("source upload original actor: {e:?}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_485_exact_windows_and_last_byte_are_bounded() {
        for index in 0..485 {
            window_range(index * WINDOW_BYTES, WINDOW_BYTES).unwrap();
        }
        assert_eq!(485 * WINDOW_BYTES, TABLE_BYTES);
        assert!(window_range(TABLE_BYTES, WINDOW_BYTES).is_err());
    }

    #[test]
    fn refuses_partial_misaligned_oversized_and_overflowing_windows() {
        for (offset, bytes) in [
            (0, 0),
            (0, WINDOW_BYTES - 1),
            (0, WINDOW_BYTES + 1),
            (1, WINDOW_BYTES),
            (2048, WINDOW_BYTES),
            (TABLE_BYTES - 2048, WINDOW_BYTES),
            (usize::MAX - WINDOW_BYTES + 1, WINDOW_BYTES),
        ] {
            assert!(window_range(offset, bytes).is_err());
        }
    }
}
