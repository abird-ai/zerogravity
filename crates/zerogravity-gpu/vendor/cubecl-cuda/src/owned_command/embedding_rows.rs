//! Closed byte-copy source embedding; no arithmetic, allocation or raw pointer API.
use super::OwnedCudaQueue;
use cubecl_core::server::{Handle, ServerError};
use std::sync::Arc;

pub(crate) const TABLE_BYTES: usize = 508_559_360;
pub(crate) const ROW_BYTES: usize = 2048;

/// Contiguous BF16 OutputInput matrix geometry. It authorizes bounded byte
/// transport only; callers retain authenticated source/lease/flight owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16MatrixLayout {
    rows: usize,
    row_bytes: usize,
    bytes: usize,
}
impl Bf16MatrixLayout {
    pub fn new(rows: usize, columns: usize) -> Result<Self, ServerError> {
        let row_bytes = columns
            .checked_mul(2)
            .filter(|n| *n != 0 && *n <= 1024 * 1024 && n.is_multiple_of(16))
            .ok_or_else(|| ServerError::graph_state("BF16 matrix complete aligned row"))?;
        let bytes = rows
            .checked_mul(row_bytes)
            .filter(|_| rows != 0 && rows - 1 <= u32::MAX as usize)
            .ok_or_else(|| ServerError::graph_state("BF16 matrix row/byte overflow"))?;
        Ok(Self {
            rows,
            row_bytes,
            bytes,
        })
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn row_bytes(self) -> usize {
        self.row_bytes
    }
    pub fn bytes(self) -> usize {
        self.bytes
    }
    pub(crate) fn output_bytes(self, ids: &[u32]) -> Result<usize, ServerError> {
        ids.len()
            .checked_mul(self.row_bytes)
            .filter(|bytes| {
                *bytes != 0
                    && *bytes <= 1024 * 1024
                    && ids.iter().all(|id| (*id as usize) < self.rows)
            })
            .ok_or_else(|| ServerError::graph_state("BF16 matrix bounded actual row IDs"))
    }
    pub(crate) fn window(self, offset: usize, bytes: usize) -> Result<(), ServerError> {
        if bytes == 0
            || bytes > 1024 * 1024
            || !bytes.is_multiple_of(self.row_bytes)
            || !offset.is_multiple_of(self.row_bytes)
            || offset.checked_add(bytes).is_none_or(|end| end > self.bytes)
        {
            return Err(ServerError::graph_state("BF16 matrix exact source window"));
        }
        Ok(())
    }
}
impl OwnedCudaQueue {
    /// # Safety
    /// Same original source/IDs/flight contract as resident row copy. The exact
    /// destination seal is minted from the actual currently active workspace;
    /// it remains owned by this actor command and the original completion flight.
    pub unsafe fn enqueue_resident_bf16_rows_workspace(
        &self,
        table: Handle,
        destination: crate::OwnedCudaBinding,
        layout: Bf16MatrixLayout,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        layout.output_bytes(&ids)?;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.copy_owned_bf16_matrix_workspace_rows(
                    stream,
                    table,
                    destination,
                    layout,
                    ids,
                )
            })
            .map_err(|e| ServerError::graph_state(format!("resident workspace row actor: {e:?}")))?
    }
    /// # Safety
    /// Complete authenticated immutable matrix Read and exclusive destination
    /// Write belong to this original armed flight; retain all owners to completion.
    pub unsafe fn enqueue_resident_bf16_rows(
        &self,
        table: Handle,
        destination: Handle,
        layout: Bf16MatrixLayout,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        layout.output_bytes(&ids)?;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.copy_owned_bf16_matrix_rows(stream, table, destination, layout, ids)
            })
            .map_err(|e| ServerError::graph_state(format!("resident matrix row actor: {e:?}")))?
    }
    /// # Safety
    /// The original armed initialization owns the complete uninitialized Write
    /// and admitted immutable source window before enqueue through checkpoint/quarantine.
    pub unsafe fn enqueue_bf16_matrix_window<T>(
        &self,
        destination: Handle,
        layout: Bf16MatrixLayout,
        offset: usize,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        layout.window(offset, source.as_ref().as_ref().len())?;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.upload_owned_original_window(
                    stream,
                    destination,
                    layout.bytes(),
                    offset,
                    source,
                )
            })
            .map_err(|e| ServerError::graph_state(format!("resident matrix source actor: {e:?}")))?
    }
}
pub(crate) fn validate_ids(ids: &[u32]) -> Result<(), ServerError> {
    if !matches!(ids.len(), 1 | 2 | 64) || ids.iter().any(|id| *id >= 248_320) {
        return Err(ServerError::graph_state(
            "closed original .8B embedding row geometry",
        ));
    }
    Ok(())
}
pub(crate) fn validate_fixed_prompt_ids(ids: &[u32]) -> Result<(), ServerError> {
    if ids.len() != 47 || ids.iter().any(|id| *id >= 248_320) {
        return Err(ServerError::graph_state(
            "closed canonical47 source embedding rows",
        ));
    }
    Ok(())
}
pub(crate) fn validate_text_resident_ids(ids: &[u32]) -> Result<(), ServerError> {
    TextEmbeddingWidth::H1024.bytes(ids).map(|_| ())
}
impl OwnedCudaQueue {
    /// # Safety
    /// Exact source-row owner and active workspace destination belong to this
    /// SAME armed original flight through completion/quarantine. No Ready here.
    pub unsafe fn enqueue_text_source_rows_workspace<T>(
        &self,
        destination: crate::OwnedCudaBinding,
        width: TextEmbeddingWidth,
        ids: Arc<[u32]>,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        let bytes = width.bytes(&ids)?;
        if source.as_ref().as_ref().len() != bytes {
            return Err(ServerError::graph_state(
                "workspace authentic complete source rows differ",
            ));
        }
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.upload_owned_workspace_rows(stream, destination, bytes, source)
            })
            .map_err(|e| {
                ServerError::graph_state(format!("workspace source embedding actor: {e:?}"))
            })?
    }
    /// # Safety
    /// SAME complete initialized .8B tied source, actual immutable short-call IDs,
    /// and original registered table Read/exclusive output Write flight. No Ready.
    pub unsafe fn enqueue_text_resident_embedding_rows(
        &self,
        table: Handle,
        destination: Handle,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        validate_text_resident_ids(&ids)?;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.copy_owned_text_resident_embedding_rows(stream, table, destination, ids)
            })
            .map_err(|error| {
                ServerError::graph_state(format!(
                    "short-HF resident source embedding actor: {error:?}"
                ))
            })?
    }
    /// # Safety
    /// Same source/flight/read/write contract as the legacy method; caller seals
    /// the genuine complete47-row prompt. This method issues no call/Ready permit.
    pub unsafe fn enqueue_fixed_prompt_embedding_rows(
        &self,
        table: Handle,
        destination: Handle,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        validate_fixed_prompt_ids(&ids)?;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.copy_owned_fixed_prompt_embedding_rows(stream, table, destination, ids)
            })
            .map_err(|error| {
                ServerError::graph_state(format!("canonical47 source embedding actor: {error:?}"))
            })?
    }

    /// # Safety
    /// Caller supplies the original fully initialized immutable tied table and
    /// IDs from its private authenticated call. The SAME armed flight retains
    /// both complete original leases and the call/table owner before enqueue;
    /// table is Read, exclusive destination Activation is Write. No Ready here.
    pub unsafe fn enqueue_qwen08_embedding_rows(
        &self,
        table: Handle,
        destination: Handle,
        ids: Arc<[u32]>,
    ) -> Result<(), ServerError> {
        validate_ids(&ids)?;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.copy_owned_vllm08_embedding_rows(stream, table, destination, ids)
            })
            .map_err(|error| {
                ServerError::graph_state(format!("source embedding original actor: {error:?}"))
            })?
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextEmbeddingWidth {
    H1024,
    H2048,
}
impl TextEmbeddingWidth {
    pub(crate) const fn row_bytes(self) -> usize {
        match self {
            Self::H1024 => 2048,
            Self::H2048 => 4096,
        }
    }
    pub(crate) fn bytes(self, ids: &[u32]) -> Result<usize, ServerError> {
        if !matches!(ids.len(), 1 | 47 | 57 | 62 | 67 | 72) || ids.iter().any(|id| *id >= 248320) {
            return Err(ServerError::graph_state(
                "finite genuine short-HF source embedding rows",
            ));
        }
        Ok(ids.len() * self.row_bytes())
    }
}
impl OwnedCudaQueue {
    /// # Safety
    /// Authentic tied embedding source rows and actual call/IDs are retained by
    /// the SAME original armed flight before upload. Exclusive full destination
    /// is Write; source owner includes its real FD/windows/host admission.
    pub unsafe fn enqueue_text_source_rows<T>(
        &self,
        destination: Handle,
        width: TextEmbeddingWidth,
        ids: Arc<[u32]>,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        let bytes = width.bytes(&ids)?;
        if source.as_ref().as_ref().len() != bytes {
            return Err(ServerError::graph_state(
                "short-HF complete authentic source row bytes",
            ));
        }
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.copy_owned_text_source_rows(stream, destination, width, ids, source)
            })
            .map_err(|e| {
                ServerError::graph_state(format!("short-HF source embedding actor: {e:?}"))
            })?
    }
}
#[cfg(test)]
mod text_tests {
    use super::*;
    #[test]
    fn source_neutral_matrix_layout_prices_whole_storage_and_actual_rows() {
        // Different vocabulary and non-power-of-two hidden width need no family tag.
        for (vocabulary, hidden) in [(248320, 1024), (248320, 2048), (262144, 1536)] {
            let layout = Bf16MatrixLayout::new(vocabulary, hidden).unwrap();
            assert_eq!(layout.bytes(), vocabulary * hidden * 2);
            assert_eq!(
                layout.output_bytes(&[0, vocabulary as u32 - 1, 0]).unwrap(),
                hidden * 6
            );
            let window = (1024 * 1024 / layout.row_bytes()) * layout.row_bytes();
            layout.window(0, window).unwrap();
            layout
                .window(layout.bytes() - layout.row_bytes(), layout.row_bytes())
                .unwrap();
            assert!(layout.window(1, layout.row_bytes()).is_err());
            assert!(layout.window(layout.bytes(), layout.row_bytes()).is_err());
            assert!(layout.output_bytes(&[vocabulary as u32]).is_err());
            assert!(layout.output_bytes(&[]).is_err());
            assert!(
                layout
                    .output_bytes(&vec![0; 1024 * 1024 / layout.row_bytes() + 1])
                    .is_err()
            );
        }
        for (rows, columns) in [
            (0, 1024),
            (1, 0),
            (1, 7),
            (1, usize::MAX),
            (usize::MAX, 1024),
        ] {
            assert!(Bf16MatrixLayout::new(rows, columns).is_err());
        }
    }
    #[test]
    fn both_actual_source_widths_copy_only_exact_complete_finite_rows() {
        for width in [TextEmbeddingWidth::H1024, TextEmbeddingWidth::H2048] {
            for rows in [1, 47, 57, 62, 67, 72] {
                assert_eq!(
                    width.bytes(&vec![248319; rows]).unwrap(),
                    rows * width.row_bytes()
                );
            }
            for rows in [0, 2, 46, 48, 56, 58, 61, 63, 64, 66, 68, 71, 73] {
                assert!(width.bytes(&vec![11; rows]).is_err());
            }
            assert!(width.bytes(&[248320]).is_err());
        }
        assert!(validate_ids(&[11; 57]).is_err());
        assert!(validate_fixed_prompt_ids(&[11; 57]).is_err());
    }
    #[test]
    fn resident_route_adds_only_short_rows_without_widening_original_entrypoints() {
        for rows in [1, 47, 57, 62, 67, 72] {
            validate_text_resident_ids(&vec![248319; rows]).unwrap();
        }
        for rows in [0, 2, 46, 48, 56, 58, 61, 63, 64, 66, 68, 71, 73] {
            assert!(validate_text_resident_ids(&vec![11; rows]).is_err());
        }
        for rows in [1, 47, 57, 62, 67, 72] {
            let mut ids = vec![11; rows];
            ids[rows - 1] = 248320;
            assert!(validate_text_resident_ids(&ids).is_err());
        }
        assert!(validate_ids(&[11; 57]).is_err());
        assert!(validate_fixed_prompt_ids(&[11; 57]).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepts_only_complete_closed_rows_and_vocabulary() {
        for count in [1, 2, 64] {
            validate_ids(&vec![248_319; count]).unwrap();
        }
        for count in [0, 3, 63, 65] {
            assert!(validate_ids(&vec![0; count]).is_err());
        }
        assert!(validate_ids(&[248_320]).is_err());
        assert_eq!(248_320 * ROW_BYTES, TABLE_BYTES);
        assert!(validate_ids(&vec![0; 47]).is_err());
        validate_fixed_prompt_ids(&vec![248_319; 47]).unwrap();
        for count in [0, 1, 2, 46, 48, 64] {
            assert!(validate_fixed_prompt_ids(&vec![0; count]).is_err());
        }
        let mut invalid = vec![0; 47];
        invalid[46] = 248_320;
        assert!(validate_fixed_prompt_ids(&invalid).is_err());
    }
}
