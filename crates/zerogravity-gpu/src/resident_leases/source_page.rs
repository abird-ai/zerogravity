//! Authenticated full-K row pages on the original armed flight.
use super::*;
use std::ops::Range;
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_artifacts::source_weights::{
    TensorWindow, VerifiedMatrix, VerifiedTensorSource, WindowWorkspace,
};
use zerogravity_core::accelerator::program::MatrixAxes;

pub(crate) const PAGE_BYTES: usize = 1024 * 1024;
const CONTROL_BYTES: usize = 64 * 1024;
mod row_cache;
use row_cache::RowCache;
#[cfg(feature = "gemma-bf16-qualification")]
pub(crate) fn cached_row_capacity(bytes: usize) -> Result<usize> {
    row_cache::capacity(bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PageGeometry {
    pub rows: usize,
    pub columns: usize,
    pub row_bytes: usize,
    pub rows_per_page: usize,
}
impl PageGeometry {
    pub(crate) fn new(rows: usize, columns: usize) -> Result<Self> {
        let row_bytes = columns
            .checked_mul(2)
            .filter(|n| rows != 0 && *n != 0 && *n <= PAGE_BYTES && n.is_multiple_of(16))
            .ok_or_else(|| Error::Limit("source page complete aligned BF16 row".into()))?;
        rows.checked_mul(row_bytes)
            .ok_or_else(|| Error::Limit("source page matrix extent".into()))?;
        Ok(Self {
            rows,
            columns,
            row_bytes,
            rows_per_page: PAGE_BYTES / row_bytes,
        })
    }
    #[cfg(test)]
    pub(crate) fn pages(self) -> usize {
        self.rows.div_ceil(self.rows_per_page)
    }
    #[cfg(test)]
    pub(crate) fn range(self, ordinal: usize) -> Result<Range<usize>> {
        let first = ordinal
            .checked_mul(self.rows_per_page)
            .filter(|n| *n < self.rows)
            .ok_or_else(|| Error::State("source page outside actual matrix".into()))?;
        Ok(first..first + (self.rows - first).min(self.rows_per_page))
    }
    fn bytes(self, rows: &Range<usize>) -> Result<usize> {
        if rows.start >= rows.end || rows.end > self.rows || rows.len() > self.rows_per_page {
            return Err(Error::State(
                "source page actual complete row extent".into(),
            ));
        }
        Ok(rows.len() * self.row_bytes)
    }
}

struct PendingPage {
    _window: Option<TensorWindow>,
    _bytes: Option<Bytes>,
    _staging: Option<WindowWorkspace>,
    serial: u64,
}
#[derive(Default)]
struct Progress {
    pending: Option<PendingPage>,
    submitted: u64,
    completed: u64,
    registered_event: Option<DeviceEventId>,
    cache: RowCache,
    traffic: SourceRowTraffic,
}
/// Completed original consumer checkpoints only; no numerical or completion permit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SourceRowTraffic {
    pub completed_uploads: u64,
    pub completed_reuses: u64,
    pub avoided_upload_bytes: u64,
}
/// One dedicated mutable packed-source page, never a whole-matrix certificate.
pub(crate) struct ResidentSourcePage {
    source: VerifiedTensorSource,
    service: Arc<ResidentService>,
    lease: ResidentLease,
    progress: Mutex<Progress>,
    _control: WindowWorkspace,
}
pub(crate) struct SourcePageView<'a> {
    handle: Handle,
    matrix: &'a VerifiedMatrix,
    rows: Range<usize>,
    bytes: usize,
}
impl SourcePageView<'_> {
    pub(crate) fn handle(&self) -> &Handle {
        &self.handle
    }
    pub(crate) fn matrix(&self) -> &VerifiedMatrix {
        self.matrix
    }
    pub(crate) fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}
fn cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("source page cancelled".into()));
    }
    Ok(())
}
impl ResidentSourcePage {
    pub(crate) fn allocate(
        source: VerifiedTensorSource,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
        if source.source_identity() != service.source_identity() || service.client().is_capturing()
        {
            return Err(Error::State(
                "source page original source/service/capture".into(),
            ));
        }
        if service.client().properties().memory.max_page_size < PAGE_BYTES as u64 {
            return Err(Error::Unsupported(
                "source page effective binding capacity".into(),
            ));
        }
        let control = source.reserve_workspace(CONTROL_BYTES)?;
        let lease = init.allocate_empty(PAGE_BYTES, BufferRole::Scratch)?;
        let page = Arc::new(Self {
            source,
            service,
            lease,
            progress: Mutex::new(Progress::default()),
            _control: control,
        });
        init.retain_completion_owner(page.clone())?;
        Ok(page)
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn require_operand(
        &self,
        source: &VerifiedTensorSource,
        service: &Arc<ResidentService>,
        matrix: &VerifiedMatrix,
    ) -> Result<()> {
        service.ensure_usable()?;
        source.require_matrix_binding(matrix)?;
        let geometry = PageGeometry::new(matrix.rows(), matrix.columns())?;
        let range = matrix.byte_range()?;
        let progress = self
            .progress
            .lock()
            .map_err(|_| Error::State("source page progress poisoned".into()))?;
        if !self.source.same_issuer(source)
            || !Arc::ptr_eq(&self.service, service)
            || matrix.axes() != MatrixAxes::OutputInput
            || matrix.tensor().kind != QuantKind::Bf16
            || range.end - range.start != (geometry.rows * geometry.row_bytes) as u64
            || progress.pending.is_some()
            || progress.submitted != progress.completed
        {
            return Err(Error::State(
                "source page actual issuer/service/range/retirement differs".into(),
            ));
        }
        Ok(())
    }
    /// Closure queues only consumers of this exact page. Its returned original
    /// output handles join the barrier before any page/staging reuse.
    pub(crate) async fn with_rows(
        self: &Arc<Self>,
        flight: &mut ResidentSubmission<'_>,
        matrix: &VerifiedMatrix,
        rows: Range<usize>,
        flag: Option<&AtomicBool>,
        enqueue: impl FnOnce(&SourcePageView<'_>) -> Result<Vec<ResidentLease>>,
    ) -> Result<()> {
        self.with_rows_inner(flight, matrix, rows, flag, false, enqueue)
            .await
    }
    /// Explicit immutable single-row reuse in this SAME page. Another matrix,
    /// ordinary page write or failed/uncertain consumer cannot produce a hit.
    #[cfg(feature = "gemma-bf16-qualification")]
    pub(crate) async fn with_cached_row(
        self: &Arc<Self>,
        flight: &mut ResidentSubmission<'_>,
        matrix: &VerifiedMatrix,
        rows: Range<usize>,
        flag: Option<&AtomicBool>,
        enqueue: impl FnOnce(&SourcePageView<'_>) -> Result<Vec<ResidentLease>>,
    ) -> Result<()> {
        if rows.len() != 1 {
            return Err(Error::State(
                "source row reuse requires exactly one checked row".into(),
            ));
        }
        self.with_rows_inner(flight, matrix, rows, flag, true, enqueue)
            .await
    }
    #[cfg(feature = "gemma-bf16-qualification")]
    pub(crate) fn row_traffic(&self) -> Result<SourceRowTraffic> {
        let p = self
            .progress
            .lock()
            .map_err(|_| Error::State("source page progress poisoned".into()))?;
        Ok(p.traffic)
    }
    async fn with_rows_inner(
        self: &Arc<Self>,
        flight: &mut ResidentSubmission<'_>,
        matrix: &VerifiedMatrix,
        rows: Range<usize>,
        flag: Option<&AtomicBool>,
        reuse: bool,
        enqueue: impl FnOnce(&SourcePageView<'_>) -> Result<Vec<ResidentLease>>,
    ) -> Result<()> {
        cancelled(flag)?;
        self.require_operand(&self.source, &self.service, matrix)?;
        if reuse {
            self.source.require_current_matrix(matrix)?;
        }
        flight.require_accesses([(&self.lease, DeviceAccess::Write)])?;
        {
            let mut p = self
                .progress
                .lock()
                .map_err(|_| Error::State("source page progress poisoned".into()))?;
            if p.registered_event != Some(flight.event) {
                flight.retain_completion_owner(self.clone())?;
                p.registered_event = Some(flight.event);
            }
        }
        let geometry = PageGeometry::new(matrix.rows(), matrix.columns())?;
        let n = geometry.bytes(&rows)?;
        let tensor = matrix.tensor();
        if matrix.source_identity() != self.source.source_identity()
            || matrix.axes() != MatrixAxes::OutputInput
            || tensor.kind != QuantKind::Bf16
            || matrix.byte_range()?.end - matrix.byte_range()?.start
                != (geometry.rows * geometry.row_bytes) as u64
        {
            return Err(Error::State(
                "source page authentic full matrix issuer/geometry".into(),
            ));
        }
        let (previous, placement) = {
            let p = self
                .progress
                .lock()
                .map_err(|_| Error::State("source page progress poisoned".into()))?;
            if p.pending.is_some() || p.submitted != p.completed {
                return Err(Error::State(
                    "source page reuse before actual checkpoint".into(),
                ));
            }
            (
                p.completed,
                if reuse {
                    Some(p.cache.select(matrix, rows.start, geometry.row_bytes)?)
                } else {
                    None
                },
            )
        };
        let reused = placement.is_some_and(|p| p.hit);
        let upload = if reused {
            None
        } else {
            let window = match flag {
                Some(c) => self.source.read_rows_cancellable(matrix, rows.clone(), c)?,
                None => self.source.read_rows(matrix, rows.clone())?,
            };
            let expected_offset = matrix
                .byte_range()?
                .start
                .checked_add((rows.start * geometry.row_bytes) as u64)
                .ok_or_else(|| Error::Limit("source page authentic offset".into()))?;
            let actual = window.matrix().tensor();
            if !window.matrix().same_binding(matrix)
                || window.matrix().source_identity() != matrix.source_identity()
                || actual.name != tensor.name
                || actual.kind != tensor.kind
                || actual.shape != tensor.shape
                || actual.byte_offset != tensor.byte_offset
                || actual.byte_len != tensor.byte_len
                || window.row_range() != rows
                || window.source_offset() != expected_offset
                || window.as_slice().len() != n
            {
                return Err(Error::State(
                    "source page authentic original window differs".into(),
                ));
            }
            let staging = self.source.reserve_workspace(2 * n)?;
            let copy = window.as_slice().to_vec();
            if copy.capacity() != n {
                return Err(Error::Limit("source page copy capacity".into()));
            }
            let bytes = Bytes::from_bytes_vec(copy);
            if bytes.capacity() > n {
                return Err(Error::Limit("source page aligned staging capacity".into()));
            }
            Some((window, bytes, staging))
        };
        let (serial, bytes) = {
            let mut p = self
                .progress
                .lock()
                .map_err(|_| Error::State("source page progress poisoned".into()))?;
            if p.pending.is_some() || p.submitted != previous || p.completed != previous {
                return Err(Error::State("source page concurrent reuse".into()));
            }
            let serial = p
                .submitted
                .checked_add(1)
                .ok_or_else(|| Error::Limit("source page serial exhausted".into()))?;
            // Invalidate the destination BEFORE any write/consumer. Only its
            // real output barrier below can publish a reusable row again.
            if let Some(place) = placement {
                p.cache.begin(matrix, place);
            } else {
                p.cache.clear();
            }
            let (window, bytes, staging) = match upload {
                Some((window, bytes, staging)) => (Some(window), Some(bytes), Some(staging)),
                None => (None, None, None),
            };
            let transfer = bytes.clone();
            p.pending = Some(PendingPage {
                _window: window,
                _bytes: bytes,
                _staging: staging,
                serial,
            });
            p.submitted = serial;
            (serial, transfer)
        };
        cancelled(flag)?;
        let offset = placement.map_or(0, |p| p.offset);
        let handle = self
            .lease
            .handle()
            .offset_start(offset as u64)
            .offset_end((PAGE_BYTES - offset - n) as u64);
        if let Some(bytes) = bytes {
            self.service.client().write(&handle, bytes);
            self.service.record_value_upload(n);
        }
        let view = SourcePageView {
            handle: handle.clone(),
            matrix,
            rows: rows.clone(),
            bytes: n,
        };
        let outputs = enqueue(&view)?;
        if outputs.is_empty() {
            return Err(Error::State(
                "source page requires original consumer output barrier".into(),
            ));
        }
        let page_range = crate::resident_program::BufferRange::new(&self.lease.handle())?;
        for output in &outputs {
            if page_range.overlaps(&crate::resident_program::BufferRange::new(
                &output.handle(),
            )?) {
                return Err(Error::State(
                    "source page consumer output aliases immutable cached rows".into(),
                ));
            }
        }
        flight.require_accesses(outputs.iter().map(|l| (l, DeviceAccess::Write)))?;
        let mut barriers = outputs
            .iter()
            .map(ResidentLease::handle)
            .collect::<Vec<_>>();
        barriers.push(handle);
        self.service
            .client()
            .sync_buffers(barriers.iter())
            .await
            .map_err(|e| Error::State(format!("source page actual consumer checkpoint: {e}")))?;
        flight.require_accesses([(&self.lease, DeviceAccess::Write)])?;
        if reuse {
            self.source.require_current_matrix(matrix)?;
        }
        let retired = {
            let mut p = self
                .progress
                .lock()
                .map_err(|_| Error::State("source page progress poisoned".into()))?;
            if p.pending.as_ref().map(|v| v.serial) != Some(serial) || p.submitted != serial {
                return Err(Error::State("source page checkpoint foreign epoch".into()));
            }
            if let Some(place) = placement {
                p.cache.complete(place, rows.start)?;
                if reused {
                    p.traffic.completed_reuses = p
                        .traffic
                        .completed_reuses
                        .checked_add(1)
                        .ok_or_else(|| Error::Limit("source row reuse count".into()))?;
                    p.traffic.avoided_upload_bytes = p
                        .traffic
                        .avoided_upload_bytes
                        .checked_add(n as u64)
                        .ok_or_else(|| Error::Limit("source row reused bytes".into()))?;
                } else {
                    p.traffic.completed_uploads = p
                        .traffic
                        .completed_uploads
                        .checked_add(1)
                        .ok_or_else(|| Error::Limit("source row upload count".into()))?;
                }
            }
            p.completed = serial;
            p.pending.take()
        };
        drop(retired);
        cancelled(flag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_rows_cover_large_tables_and_real_tail_without_k_split() {
        for (rows, columns, pages, tail) in [
            (262144, 8960, 4520, 42),
            (262144, 1536, 769, 256),
            (1536, 12288, 37, 24),
        ] {
            let g = PageGeometry::new(rows, columns).unwrap();
            assert_eq!(g.pages(), pages);
            let mut end = 0;
            for p in 0..g.pages() {
                let r = g.range(p).unwrap();
                assert_eq!(r.start, end);
                assert!(g.bytes(&r).unwrap() <= PAGE_BYTES);
                end = r.end;
            }
            assert_eq!(end, rows);
            assert_eq!(g.range(pages - 1).unwrap().len(), tail);
            assert!(g.range(pages).is_err());
            assert!(g.bytes(&(0..g.rows_per_page + 1)).is_err());
        }
    }
    #[test]
    fn malformed_or_partial_rows_refuse() {
        for (r, c) in [
            (0, 1536),
            (1, 0),
            (1, 1535),
            (1, usize::MAX),
            (usize::MAX, 1536),
            (1, PAGE_BYTES),
        ] {
            assert!(PageGeometry::new(r, c).is_err());
        }
    }
}
