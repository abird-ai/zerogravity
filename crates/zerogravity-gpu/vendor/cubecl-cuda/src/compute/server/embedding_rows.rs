//! Source-exact original tied-table→Activation copies on the original stream.
use super::*;
use crate::owned_command::embedding_rows::{
    Bf16MatrixLayout, ROW_BYTES, TABLE_BYTES, validate_canonical47_ids, validate_ids,
    validate_short_hf_resident_ids,
};

fn full_geometry(handle: &BufferBinding, expected: usize) -> Result<(), ServerError> {
    if handle.size != expected as u64
        || handle.offset_start.unwrap_or(0) != 0
        || handle.offset_end.unwrap_or(0) != 0
    {
        return Err(ServerError::graph_state(
            "embedding copy exact full original handles",
        ));
    }
    Ok(())
}
#[cfg(test)]
fn ranges(table: u64, destination: u64, bytes: usize) -> Result<(), ServerError> {
    matrix_ranges(table, destination, TABLE_BYTES, bytes)
}
fn matrix_ranges(
    table: u64,
    destination: u64,
    table_bytes: usize,
    bytes: usize,
) -> Result<(), ServerError> {
    let a = table
        .checked_add(table_bytes as u64)
        .ok_or_else(|| ServerError::graph_state("embedding table end overflow"))?;
    let b = destination
        .checked_add(bytes as u64)
        .ok_or_else(|| ServerError::graph_state("embedding output end overflow"))?;
    if table == 0
        || destination == 0
        || !table.is_multiple_of(16)
        || !destination.is_multiple_of(16)
        || (table < b && destination < a)
    {
        return Err(ServerError::graph_state(
            "embedding original aligned nonalias ranges",
        ));
    }
    Ok(())
}
enum RowDestination {
    Original(Handle),
    Workspace(crate::OwnedCudaBinding),
}
impl CudaServer {
    pub(crate) fn copy_owned_short_hf_resident_embedding_rows(
        &mut self,
        stream: StreamId,
        table: Handle,
        destination: Handle,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        validate_short_hf_resident_ids(&ids)?;
        self.copy_owned_embedding_rows(stream, table, destination, ids)
    }
    pub(crate) fn copy_owned_vllm08_embedding_rows(
        &mut self,
        stream: StreamId,
        table: Handle,
        destination: Handle,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        validate_ids(&ids)?;
        self.copy_owned_embedding_rows(stream, table, destination, ids)
    }
    pub(crate) fn copy_owned_canonical47_embedding_rows(
        &mut self,
        stream: StreamId,
        table: Handle,
        destination: Handle,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        validate_canonical47_ids(&ids)?;
        self.copy_owned_embedding_rows(stream, table, destination, ids)
    }
    fn copy_owned_embedding_rows(
        &mut self,
        stream: StreamId,
        table: Handle,
        destination: Handle,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        self.copy_owned_bf16_matrix_rows(
            stream,
            table,
            destination,
            Bf16MatrixLayout::new(TABLE_BYTES / ROW_BYTES, ROW_BYTES / 2)?,
            ids,
        )
    }
    pub(crate) fn copy_owned_bf16_matrix_rows(
        &mut self,
        stream: StreamId,
        table: Handle,
        destination: Handle,
        layout: Bf16MatrixLayout,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        self.copy_owned_matrix_rows_inner(
            stream,
            table,
            RowDestination::Original(destination),
            layout,
            ids,
        )
    }
    pub(crate) fn copy_owned_bf16_matrix_workspace_rows(
        &mut self,
        stream: StreamId,
        table: Handle,
        destination: crate::OwnedCudaBinding,
        layout: Bf16MatrixLayout,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        self.copy_owned_matrix_rows_inner(
            stream,
            table,
            RowDestination::Workspace(destination),
            layout,
            ids,
        )
    }
    fn copy_owned_matrix_rows_inner(
        &mut self,
        stream: StreamId,
        table: Handle,
        destination: RowDestination,
        layout: Bf16MatrixLayout,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        let bytes = layout.output_bytes(&ids)?;
        let (destination_service, dst) = match &destination {
            RowDestination::Original(handle) => {
                let binding = handle.clone().binding();
                full_geometry(&binding, bytes)?;
                (handle.service == self.utilities.service, binding)
            }
            RowDestination::Workspace(binding) => {
                binding.require_copy(self.utilities.service, bytes as u64)?;
                (true, binding.handle().binding())
            }
        };
        let same_service = table.service == self.utilities.service && destination_service;
        let native = [table.binding(), dst];
        full_geometry(&native[0], layout.bytes())?;
        let same_stream = native.iter().all(|binding| binding.stream == stream);
        // launching validates the original source read BEFORE provisional
        // destination Write taint; every later error preserves failure taint.
        let mut written = self.write_set();
        written.push(native[1].clone());
        ExecuteScope::launching(
            self,
            cubecl_server::id::KernelId::new::<EmbeddingRows>(),
            stream,
            std::iter::once(&native[0]),
            written,
        )
        .execute(move |server| {
            let _destination = destination;
            if !same_service || !same_stream {
                return Err(ServerError::graph_state(
                    "embedding original service/stream differs",
                ));
            }
            let mut command = server.command(stream, native.iter())?;
            if command.stream().capturing.is_recording() {
                return Err(ServerError::graph_state(
                    "embedding source copy graph capture unsupported",
                ));
            }
            let source = command.resource(native[0].clone())?;
            let target = command.resource(native[1].clone())?;
            if source.size != layout.bytes() as u64 || target.size != bytes as u64 {
                return Err(ServerError::graph_state(
                    "embedding exact original native extents",
                ));
            }
            matrix_ranges(source.ptr, target.ptr, layout.bytes(), bytes)?;
            for (row, id) in ids.iter().enumerate() {
                // All IDs/ranges were checked before the first native copy.
                let from = source.ptr + u64::from(*id) * layout.row_bytes() as u64;
                let to = target.ptr + (row * layout.row_bytes()) as u64;
                let status = unsafe {
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        to,
                        from,
                        layout.row_bytes(),
                        command.stream().sys,
                    )
                };
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(ServerError::graph_state(format!(
                        "source embedding D2D: {status:?}"
                    )));
                }
            }
            Ok(())
        })
        .into_result()
    }
}
struct EmbeddingRows;

impl CudaServer {
    pub(crate) fn copy_owned_short_hf_source_rows<T>(
        &mut self,
        stream: StreamId,
        destination: Handle,
        width: crate::owned_command::embedding_rows::ShortHfEmbeddingWidth,
        ids: Arc<[u32]>,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        let bytes = width.bytes(&ids)?;
        if source.as_ref().as_ref().len() != bytes {
            return Err(ServerError::graph_state(
                "short-HF authentic full source row image",
            ));
        }
        // Existing implementation validates exact full native allocation, original
        // stream/service, retained host ownership and actual bounded H2D lifetime.
        self.upload_owned_original_window(stream, destination, bytes, 0, source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generic_matrix_full_ranges_use_actual_table_extent() {
        let small = Bf16MatrixLayout::new(16, 1536).unwrap();
        matrix_ranges(0x1000, 0x10_0000, small.bytes(), small.row_bytes()).unwrap();
        assert!(
            matrix_ranges(
                0x1000,
                0x1000 + small.bytes() as u64 - 16,
                small.bytes(),
                small.row_bytes()
            )
            .is_err()
        );
        assert!(matrix_ranges(u64::MAX - 15, 0x10_0000, small.bytes(), small.row_bytes()).is_err());
    }
    #[test]
    fn full_table_copy_ranges_refuse_alias_zero_overflow() {
        ranges(0x1000, 0x4000_0000, 64 * ROW_BYTES).unwrap();
        for (a, b) in [
            (0, 0x4000_0000),
            (0x1000, 0),
            (0x1000, 0x1010),
            (0x1001, 0x4000_0000),
            (u64::MAX - 15, 0x1000),
        ] {
            assert!(ranges(a, b, ROW_BYTES).is_err());
        }
    }
    #[test]
    fn short_resident_full_ranges_keep_last_vocabulary_row_inside_exact_table() {
        for rows in [1, 47, 57, 62, 67, 72] {
            ranges(0x1000, 0x4000_0000, rows * ROW_BYTES).unwrap();
            assert!(ranges(0x1000, 0x1000 + TABLE_BYTES as u64 - 16, rows * ROW_BYTES).is_err());
        }
        assert_eq!(248319 * ROW_BYTES + ROW_BYTES, TABLE_BYTES);
    }
}
