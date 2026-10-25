//! Authenticated full-K row pages on the original armed flight.
use super::*;
#[cfg(feature = "hip")]
use crate::transfer_retirement::{RetirementMark, TransferRetirementRing};
use std::ops::Range;
#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
use std::sync::Weak;
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_artifacts::source_weights::{
    TensorWindow, VerifiedMatrix, VerifiedTensorSource, WindowWorkspace,
};
#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::MatrixAxes;

pub(crate) const PAGE_BYTES: usize = 1024 * 1024;
const CONTROL_BYTES: usize = 64 * 1024;
/// Ordinary source windows retained behind the same 1 MiB device page.
pub(crate) const RETIRED_PAGE_TRANSFERS: usize = 2;

/// Exact host charge of the retained ordinary transfers. On the HIP route at
/// most two page-sized source windows live; each staged read is charged by its
/// whole intersecting integrity chunks plus offset metadata, exactly as
/// `TensorWindow::charged_staging_bytes` charges it. The persistent window
/// cache budget is charged once and is not returned by ring retirement. A
/// shared cached window adds no second payload. Ring pending/reusable slots,
/// issue marks and their handle vectors, the run owner and the bounded caller
/// output leases are counted exactly; the old/new page metadata stays inside
/// the existing 64 KiB control reservation, proved at construction.
///
/// The serial route has no ring/page retirement: one ordinary window is staged
/// by its whole intersecting integrity chunks plus offset metadata, and the
/// shared three-page family baseline still covers the window payload, its
/// `Bytes` copy and the backend bounce. A HIP-compiled binary can still select
/// this serial route when its prepared set holds no live scoped HIP
/// source-page actor, so both compiled routes are priced and the larger
/// source-derived bound is charged. The routes are never summed; the original
/// source base inventory already charges the metadata envelope the persistent
/// cache is prepaid from, so the cache budget is neither duplicated nor
/// omitted. No page counts or caps change.
pub(crate) fn transport_host_bytes(source: &VerifiedTensorSource) -> Result<usize> {
    let chunk = source.staging_chunk_bytes();
    let serial = serial_transport_host_charge(chunk)?;
    #[cfg(feature = "hip")]
    {
        let cache = source
            .window_cache_stats()
            .map_err(|error| Error::State(format!("source window cache charge: {error}")))?
            .budget_bytes;
        return Ok(serial.max(transport_host_charge(chunk, cache)?));
    }
    #[cfg(not(feature = "hip"))]
    {
        return Ok(serial);
    }
}
/// One serial source window: the unchanged three-page family baseline plus
/// whole intersecting integrity chunks and per-chunk offset metadata.
fn serial_transport_host_charge(chunk: usize) -> Result<usize> {
    (3 * PAGE_BYTES)
        .checked_add(window_staging_charge(chunk)?)
        .ok_or_else(|| Error::Limit("serial source transport host".into()))
}
/// Whole intersecting integrity chunks plus per-chunk offset metadata for one
/// page-sized window, matching the staged-read charge of both routes.
fn window_staging_charge(chunk: usize) -> Result<usize> {
    if chunk == 0 {
        return Err(Error::Limit(
            "source page retained host chunk geometry".into(),
        ));
    }
    // A page-sized window can intersect at most page/chunk + 1 integrity
    // chunks (the extra when it straddles a chunk boundary).
    let chunks = PAGE_BYTES
        .div_ceil(chunk)
        .checked_add(1)
        .ok_or_else(|| Error::Limit("source page retained host chunk count".into()))?;
    let offset_metadata = std::mem::size_of::<(usize, Range<usize>)>();
    chunks
        .checked_mul(
            chunk
                .checked_add(offset_metadata)
                .ok_or_else(|| Error::Limit("source page retained host chunk".into()))?,
        )
        .ok_or_else(|| Error::Limit("source page retained host window".into()))
}
#[cfg(feature = "hip")]
fn transport_host_charge(chunk: usize, cache: usize) -> Result<usize> {
    let per_window = window_staging_charge(chunk)?;
    let payload = per_window
        .checked_mul(RETIRED_PAGE_TRANSFERS)
        .ok_or_else(|| Error::Limit("source page retained host payload".into()))?;
    let mark = std::mem::size_of::<PageMark>();
    let bindings = 5usize
        .checked_mul(std::mem::size_of::<cubecl::server::BufferBinding>())
        .ok_or_else(|| Error::Limit("source page retained host bindings".into()))?;
    let owner_run = probe_arc_bytes::<PageRunOwner>()?;
    let owner_window = probe_arc_bytes::<WindowBytes>()?;
    let ring_owner = std::mem::size_of::<Arc<PageRunOwner>>();
    let pending = mark
        .checked_add(bindings)
        .and_then(|n| n.checked_add(owner_run))
        .and_then(|n| n.checked_add(owner_window))
        .and_then(|n| n.checked_add(ring_owner))
        .and_then(|n| n.checked_mul(RETIRED_PAGE_TRANSFERS))
        .ok_or_else(|| Error::Limit("source page retained host pending".into()))?;
    let reusable = mark
        .checked_add(bindings)
        .and_then(|n| n.checked_mul(RETIRED_PAGE_TRANSFERS))
        .ok_or_else(|| Error::Limit("source page retained host reusable".into()))?;
    let outputs = std::mem::size_of::<ResidentLease>()
        .checked_mul(4 * RETIRED_PAGE_TRANSFERS)
        .ok_or_else(|| Error::Limit("source page retained host outputs".into()))?;
    let records = std::mem::size_of::<PageCompletion>()
        .checked_mul(RETIRED_PAGE_TRANSFERS)
        .ok_or_else(|| Error::Limit("source page retained host records".into()))?;
    let run = std::mem::size_of::<PageRunState>()
        .checked_add(std::mem::size_of::<RunInner>())
        .ok_or_else(|| Error::Limit("source page retained host run".into()))?;
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    let probe = source_page_event_probe_control_bytes()?;
    #[cfg(not(all(feature = "hip", feature = "hardware-abort-probe")))]
    let probe = 0usize;
    let metadata = [pending, reusable, outputs, records, run, probe]
        .into_iter()
        .try_fold(0usize, |acc, n| {
            acc.checked_add(n)
                .ok_or_else(|| Error::Limit("source page retained host metadata".into()))
        })?;
    if metadata > CONTROL_BYTES {
        return Err(Error::Limit(
            "source page retained host metadata exceeds control".into(),
        ));
    }
    // The persistent window-cache budget is charged once; a shared cached
    // window is inside it and is not charged again by the ring.
    [payload, cache, CONTROL_BYTES]
        .into_iter()
        .try_fold(0usize, |acc, n| {
            acc.checked_add(n)
                .ok_or_else(|| Error::Limit("source page retained host sum".into()))
        })
}
/// The original actor view when the prepared HIP set owns a scoped stream.
#[cfg(feature = "hip")]
pub(crate) type PageActor = cubecl::hip::HipSourcePageActor;
#[cfg(not(feature = "hip"))]
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct PageActor;
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
    #[cfg(feature = "hip")]
    run: Option<Arc<PageRunState>>,
    #[cfg(feature = "hip")]
    failed: bool,
}
#[cfg(feature = "hip")]
struct PageMark(cubecl::hip::OwnedHipTransferRetirement);
#[cfg(feature = "hip")]
impl RetirementMark for PageMark {
    fn ready(&mut self) -> Result<bool> {
        self.0
            .ready()
            .map_err(|e| Error::State(format!("source page HIP event query: {e}")))
    }
    fn wait(&mut self) -> Result<()> {
        self.0
            .wait()
            .map_err(|e| Error::State(format!("source page HIP event wait: {e}")))
    }
}
/// Retains the exact source-owned window bytes until the original actor proves
/// the recorded write plus consumers completed.
#[cfg(feature = "hip")]
struct WindowBytes(TensorWindow);
#[cfg(feature = "hip")]
impl AsRef<[u8]> for WindowBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}
#[cfg(feature = "hip")]
struct PageRunOwner {
    _window: Option<Arc<WindowBytes>>,
    outputs: Mutex<Vec<ResidentLease>>,
}
#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
#[derive(Clone)]
pub struct SourcePageProbeParts {
    pub source: SourceIdentity,
    state: Arc<cubecl::hip::SourcePageEventProbeState>,
    witness: Arc<SourcePageProbeWitness>,
}
#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
impl SourcePageProbeParts {
    /// Scalar state only; still valid when the weak run/window witness expired.
    pub fn snapshot_state(&self) -> Result<cubecl::hip::SourcePageEventProbeSnapshot> {
        let mut snapshot = self
            .state
            .snapshot()
            .map_err(|error| Error::State(format!("source page probe snapshot: {error}")))?;
        snapshot.page_allocation_id = self.witness.page_allocation_id();
        Ok(snapshot)
    }
    /// Bounded pair witness for one scalar snapshot; missing/poison is an error,
    /// never a false dead-weak result.
    pub fn witness_live(
        &self,
        snapshot: &cubecl::hip::SourcePageEventProbeSnapshot,
    ) -> Result<(bool, bool)> {
        self.witness.live(
            snapshot.event_identity.unwrap_or(0),
            snapshot.current_serial,
            snapshot.window_payload_bytes,
            snapshot.source_window_charge_bytes,
        )
    }
    pub fn snapshot(&self) -> Result<(cubecl::hip::SourcePageEventProbeSnapshot, bool, bool)> {
        let snapshot = self.snapshot_state()?;
        let (run, window) = self.witness_live(&snapshot)?;
        Ok((snapshot, run, window))
    }
}
#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
#[derive(Clone)]
struct ProbeSlot {
    event: u64,
    serial: u64,
    run: Weak<PageRunOwner>,
    window: Option<Weak<WindowBytes>>,
}
/// Weak-only diagnostic witnesses. The slots never own a run or window, so a
/// probe cannot extend the original ring/consumer retirement lifetime.
#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
pub(crate) struct SourcePageProbeWitness {
    page_allocation_id: u64,
    source: SourceIdentity,
    slots: Mutex<[Option<ProbeSlot>; RETIRED_PAGE_TRANSFERS]>,
    next: std::sync::atomic::AtomicUsize,
}
#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
impl SourcePageProbeWitness {
    pub(crate) fn new(page_allocation_id: u64, source: SourceIdentity) -> Self {
        Self {
            page_allocation_id,
            source,
            slots: Mutex::new([None, None]),
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    pub(crate) fn page_allocation_id(&self) -> u64 {
        self.page_allocation_id
    }
    pub(crate) fn capture(&self, event: u64, serial: u64, owner: &Arc<PageRunOwner>) -> Result<()> {
        let slot = ProbeSlot {
            event,
            serial,
            run: Arc::downgrade(owner),
            window: owner._window.as_ref().map(Arc::downgrade),
        };
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| Error::State("source page probe witness poisoned".into()))?;
        if let Some(existing) = slots
            .iter_mut()
            .find(|slot| slot.as_ref().is_some_and(|slot| slot.event == event))
        {
            *existing = Some(slot);
            return Ok(());
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed) % RETIRED_PAGE_TRANSFERS;
        slots[index] = Some(slot);
        Ok(())
    }
    pub(crate) fn live(
        &self,
        event: u64,
        serial: u64,
        payload: u64,
        charge: u64,
    ) -> Result<(bool, bool)> {
        let slots = self
            .slots
            .lock()
            .map_err(|_| Error::State("source page probe witness poisoned".into()))?;
        let Some(slot) = slots
            .iter()
            .flatten()
            .find(|slot| slot.event == event && slot.serial == serial)
        else {
            return Err(Error::State(
                "source page probe selected event absent".into(),
            ));
        };
        let run = slot.run.upgrade().is_some();
        let window = match slot.window.as_ref().and_then(|window| window.upgrade()) {
            Some(window) => {
                if window.0.matrix().source_identity() != self.source
                    || window.as_ref().as_ref().len() as u64 != payload
                    || window.0.charged_staging_bytes() as u64 != charge
                {
                    return Err(Error::State(
                        "source page probe selected window extent differs".into(),
                    ));
                }
                true
            }
            None => false,
        };
        Ok((run, window))
    }
}
/// Conservative size of one `Arc<T>` allocation: the two control words, the
/// inline value, and worst-case alignment padding between/after them.
#[cfg(feature = "hip")]
fn probe_arc_bytes<T>() -> Result<usize> {
    let word = std::mem::size_of::<usize>();
    let align = word.max(std::mem::align_of::<T>());
    // ArcInner starts with two control words; T begins at the next alignment
    // boundary, not merely at a two-word header. Round the combined size to
    // the allocation alignment as well.
    let header = word
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("source page probe Arc header".into()))?;
    let value = header
        .checked_add(align - 1)
        .map(|n| n & !(align - 1))
        .and_then(|n| n.checked_add(std::mem::size_of::<T>()))
        .ok_or_else(|| Error::Limit("source page probe Arc body".into()))?;
    value
        .checked_add(align - 1)
        .map(|n| n & !(align - 1))
        .ok_or_else(|| Error::Limit("source page probe Arc layout".into()))
}
#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
pub(crate) fn source_page_event_probe_control_bytes() -> Result<usize> {
    // Parts are stored inline by both the page and the original actor. The
    // probe state and weak witness are each one Arc allocation. A weak-only
    // slot still retains the whole old run/window Arc allocation after its
    // payload is destroyed, so charge both slot bodies here.
    let page_probe = std::mem::size_of::<std::sync::Mutex<Option<SourcePageProbeParts>>>();
    let actor_probe = std::mem::size_of::<
        std::sync::Mutex<Option<Arc<cubecl::hip::SourcePageEventProbeState>>>,
    >();
    let state = probe_arc_bytes::<cubecl::hip::SourcePageEventProbeState>()?;
    let witness = probe_arc_bytes::<SourcePageProbeWitness>()?;
    let weak_run = probe_arc_bytes::<PageRunOwner>()?;
    let weak_window = probe_arc_bytes::<WindowBytes>()?;
    let weak_slots = weak_run
        .checked_add(weak_window)
        .and_then(|n| n.checked_mul(RETIRED_PAGE_TRANSFERS))
        .ok_or_else(|| Error::Limit("source page probe weak slots".into()))?;
    page_probe
        .checked_add(actor_probe)
        .and_then(|n| n.checked_add(state))
        .and_then(|n| n.checked_add(witness))
        .and_then(|n| n.checked_add(weak_slots))
        .ok_or_else(|| Error::Limit("source page probe control".into()))
}
#[cfg(feature = "hip")]
#[derive(Clone)]
struct PageCompletion {
    serial: u64,
    placement: Option<row_cache::RowPlacement>,
    row: usize,
    bytes: usize,
    reused: bool,
}
#[cfg(feature = "hip")]
struct PageRunState<M: RetirementMark = PageMark> {
    flight: Option<DeviceEventId>,
    inner: Mutex<RunInner<M>>,
    quarantine: Option<Box<dyn FnOnce(Arc<PageRunOwner>) + Send + Sync>>,
}
#[cfg(feature = "hip")]
struct RunInner<M: RetirementMark = PageMark> {
    /// The exact owner installed before the first H2D and before its mark.
    pending: Option<Arc<PageRunOwner>>,
    /// Temporarily `None` only while this run retires outside its mutex.
    ring: Option<TransferRetirementRing<M, Arc<PageRunOwner>>>,
    completed_runs: Vec<PageCompletion>,
}
#[cfg(feature = "hip")]
impl<M: RetirementMark> Drop for PageRunState<M> {
    fn drop(&mut self) {
        let Ok(inner) = self.inner.get_mut() else {
            return;
        };
        let Some(owner) = inner.pending.take() else {
            return;
        };
        if let Some(quarantine) = self.quarantine.take() {
            quarantine(owner);
        } else {
            // Already failed closed; never release an unresolved owner.
            std::mem::forget(owner);
        }
    }
}
#[cfg(feature = "hip")]
fn run_has_pending<M: RetirementMark>(run: &PageRunState<M>) -> bool {
    run.inner
        .lock()
        .map(|inner| {
            inner.pending.is_some()
                || !inner.completed_runs.is_empty()
                || inner
                    .ring
                    .as_ref()
                    .is_none_or(|ring| ring.counts().pending != 0)
        })
        .unwrap_or(true)
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
    transport: Option<PageActor>,
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    probe: Mutex<Option<SourcePageProbeParts>>,
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
        Self::allocate_with(source, service, init, None)
    }
    /// Shared original-actor page for ordinary windows; `None` keeps the
    /// existing serial generic path.
    #[cfg(feature = "hip")]
    pub(crate) fn allocate_observed(
        source: VerifiedTensorSource,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        actor: Option<PageActor>,
    ) -> Result<Arc<Self>> {
        Self::allocate_with(source, service, init, actor)
    }
    fn allocate_with(
        source: VerifiedTensorSource,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        transport: Option<PageActor>,
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
        #[cfg(feature = "hip")]
        if let Some(actor) = &transport {
            if actor.service() != service.client().service_id()
                || actor.generation() != service.generation()
            {
                return Err(Error::State(
                    "source page original actor/generation differs".into(),
                ));
            }
        }
        let page = Arc::new(Self {
            source,
            service,
            lease,
            progress: Mutex::new(Progress::default()),
            transport,
            #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
            probe: Mutex::new(None),
            _control: control,
        });
        init.retain_completion_owner(page.clone())?;
        Ok(page)
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    pub(crate) fn arm_source_page_event_probe(
        &self,
        mode: cubecl::hip::SourcePageEventProbeMode,
    ) -> Result<SourcePageProbeParts> {
        let mut held = self
            .probe
            .lock()
            .map_err(|_| Error::State("source page probe poisoned".into()))?;
        if held.is_some() {
            return Err(Error::State("source page probe already armed".into()));
        }
        let actor = self
            .transport
            .as_ref()
            .ok_or_else(|| Error::State("source page original actor absent".into()))?;
        let state = actor
            .arm_source_page_event_probe(mode)
            .map_err(|error| Error::State(format!("source page probe arm: {error}")))?;
        let page_allocation_id = self.lease.handle().binding().memory.id();
        let parts = SourcePageProbeParts {
            source: self.source.source_identity(),
            state,
            witness: Arc::new(SourcePageProbeWitness::new(
                page_allocation_id,
                self.source.source_identity(),
            )),
        };
        *held = Some(parts.clone());
        Ok(parts)
    }
    /// Real accessor for the same armed parts; the original program forwarding
    /// therefore observes the page's actual one-shot state, not a second mock.
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    pub(crate) fn source_page_event_probe(
        &self,
    ) -> Result<(cubecl::hip::SourcePageEventProbeSnapshot, bool, bool)> {
        let parts = self
            .probe
            .lock()
            .map_err(|_| Error::State("source page probe poisoned".into()))?
            .as_ref()
            .cloned()
            .ok_or_else(|| Error::State("source page probe not armed".into()))?;
        parts.snapshot()
    }
    pub(crate) fn require_operand(
        &self,
        source: &VerifiedTensorSource,
        service: &Arc<ResidentService>,
        matrix: &VerifiedMatrix,
    ) -> Result<()> {
        service.ensure_usable()?;
        #[cfg(feature = "hip")]
        self.make_room()?;
        self.require_operand_current(source, service, matrix)
    }
    fn require_operand_current(
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
        #[cfg(feature = "hip")]
        if progress.failed {
            return Err(Error::State("source page retired transfer failed".into()));
        }
        if !self.source.same_issuer(source)
            || !Arc::ptr_eq(&self.service, service)
            || matrix.axes() != MatrixAxes::OutputInput
            || matrix.tensor().kind != QuantKind::Bf16
            || range.end - range.start != (geometry.rows * geometry.row_bytes) as u64
            || self.transport.is_none()
                && (progress.pending.is_some() || progress.submitted != progress.completed)
        {
            return Err(Error::State(
                "source page actual issuer/service/range/retirement differs".into(),
            ));
        }
        Ok(())
    }
    /// Retire the oldest retained ordinary window before the next read or host
    /// owner is allocated.
    #[cfg(feature = "hip")]
    fn make_room(&self) -> Result<()> {
        self.reap(false)
    }
    /// Wait for every retained ordinary window; used at an operand boundary,
    /// before a cached row, and at the original actual terminal boundaries.
    pub(crate) fn drain(&self) -> Result<()> {
        #[cfg(feature = "hip")]
        return self.reap(true);
        #[cfg(not(feature = "hip"))]
        Ok(())
    }
    /// Terminal body readiness: every ordinary window issued by this original
    /// flight is retired; no forced pending count. On the known-healthy path
    /// this also destroys the retired HIP events on the ORIGINAL actor and drops
    /// this page's per-flight run reference while the source/owner/service are
    /// still usable, so event retirement cannot follow native SET retirement.
    pub(crate) fn finish(&self) -> Result<()> {
        #[cfg(feature = "hip")]
        {
            self.reap(true)?;
            let run = {
                let p = self
                    .progress
                    .lock()
                    .map_err(|_| Error::State("source page progress poisoned".into()))?;
                if p.failed
                    || p.submitted != p.completed
                    || p.run.as_ref().is_some_and(|r| run_has_pending(r))
                {
                    return Err(Error::State(
                        "source page terminal retirement differs".into(),
                    ));
                }
                p.run.as_ref().cloned()
            };
            if let Some(run) = run {
                // Release every known-retired mark here: dropping each mark
                // performs the real event destroy on the original actor while
                // the service generation is still live.
                let retired = {
                    let mut inner = run
                        .inner
                        .lock()
                        .map_err(|_| Error::State("source page run poisoned".into()))?;
                    inner
                        .ring
                        .as_mut()
                        .ok_or_else(|| Error::State("source page run ring absent".into()))?
                        .take_retired_marks()?
                };
                drop(retired);
                // A refused event destroy fences the original service from the
                // mark Drop above. Finishing must not report readiness with an
                // authority it can no longer prove, so recheck the actual
                // original service and actor generation BEFORE releasing the
                // run. On refusal the run/owners stay with this page and the
                // existing completion authority fences the service.
                if let Err(error) = self.require_finish_authority() {
                    self.service.fence_and_retain_owners(&[], Vec::new());
                    return Err(Error::State(format!(
                        "source page finish original authority: {error}"
                    )));
                }
                let mut p = self
                    .progress
                    .lock()
                    .map_err(|_| Error::State("source page progress poisoned".into()))?;
                if p.failed {
                    return Err(Error::State("source page retired transfer failed".into()));
                }
                // The armed flight keeps its own run reference until consumer
                // completion; this page no longer prolongs event/owner life.
                p.run = None;
            }
        }
        Ok(())
    }
    /// The original-service authority must still hold after the retired events
    /// were destroyed: a failed destroy fences the service, and a generation
    /// change means the native SET this page recorded on is no longer live.
    #[cfg(feature = "hip")]
    fn require_finish_authority(&self) -> Result<()> {
        self.service.ensure_usable()?;
        if let Some(actor) = &self.transport {
            if actor.service() != self.service.client().service_id()
                || actor.generation() != self.service.generation()
            {
                return Err(Error::State(
                    "source page finish actor/generation differs".into(),
                ));
            }
        }
        Ok(())
    }
    #[cfg(feature = "hip")]
    fn reap(&self, all: bool) -> Result<()> {
        let run = {
            let progress = self
                .progress
                .lock()
                .map_err(|_| Error::State("source page progress poisoned".into()))?;
            if progress.failed {
                return Err(Error::State("source page retired transfer failed".into()));
            }
            let Some(run) = progress.run.as_ref().cloned() else {
                return Ok(());
            };
            run
        };
        // Retire on the original actor without holding a mutex another entry
        // needs for its query/record/quarantine.
        let (before, mut ring) = loop {
            let mut inner = run
                .inner
                .lock()
                .map_err(|_| Error::State("source page run poisoned".into()))?;
            if let Some(ring) = inner.ring.take() {
                break (ring.counts(), ring);
            }
            drop(inner);
            std::thread::yield_now();
        };
        let result = if all { ring.drain() } else { ring.make_room() };
        if let Err(error) = result {
            let mut inner = run
                .inner
                .lock()
                .map_err(|_| Error::State("source page run poisoned".into()))?;
            inner.ring = Some(ring);
            drop(inner);
            if let Ok(mut progress) = self.progress.lock() {
                progress.failed = true;
            }
            return Err(error);
        }
        let after = ring.counts();
        let retired = after.retired.saturating_sub(before.retired);
        let mut progress = self
            .progress
            .lock()
            .map_err(|_| Error::State("source page progress poisoned".into()))?;
        if progress.failed {
            let mut inner = run
                .inner
                .lock()
                .map_err(|_| Error::State("source page run poisoned".into()))?;
            inner.ring = Some(ring);
            return Err(Error::State("source page retired transfer failed".into()));
        }
        let Progress {
            cache,
            traffic,
            completed,
            failed,
            ..
        } = &mut *progress;
        let mut inner = run
            .inner
            .lock()
            .map_err(|_| Error::State("source page run poisoned".into()))?;
        inner.ring = Some(ring);
        for _ in 0..retired {
            let Some(done) = inner.completed_runs.first().cloned() else {
                *failed = true;
                return Err(Error::State(
                    "source page completion without retained serial".into(),
                ));
            };
            inner.completed_runs.remove(0);
            if let Some(place) = done.placement {
                let completed_row = cache.complete(place, done.row);
                let counted = match completed_row {
                    Err(error) => Err(error),
                    Ok(()) if done.reused => traffic
                        .completed_reuses
                        .checked_add(1)
                        .map(|n| traffic.completed_reuses = n)
                        .and_then(|()| {
                            traffic
                                .avoided_upload_bytes
                                .checked_add(done.bytes as u64)
                                .map(|n| traffic.avoided_upload_bytes = n)
                        })
                        .ok_or_else(|| Error::Limit("source row reuse count".into())),
                    Ok(()) => traffic
                        .completed_uploads
                        .checked_add(1)
                        .map(|n| traffic.completed_uploads = n)
                        .ok_or_else(|| Error::Limit("source row upload count".into())),
                };
                if let Err(error) = counted {
                    *failed = true;
                    return Err(error);
                }
            }
            *completed = done.serial;
        }
        Ok(())
    }
    /// Queue up to two ordinary windows on the same original page. Each write
    /// is recorded with its exact consumers; the host retires the oldest window
    /// with the real event, never by assuming enqueue equals completion.
    #[cfg(feature = "hip")]
    async fn with_rows_pipelined(
        self: &Arc<Self>,
        flight: &mut ResidentSubmission<'_>,
        matrix: &VerifiedMatrix,
        rows: Range<usize>,
        flag: Option<&AtomicBool>,
        reuse: bool,
        enqueue: impl FnOnce(&SourcePageView<'_>) -> Result<Vec<ResidentLease>>,
    ) -> Result<()> {
        cancelled(flag)?;
        self.require_operand_current(&self.source, &self.service, matrix)?;
        if reuse {
            self.source.require_current_matrix(matrix)?;
            self.drain()?;
        }
        flight.require_accesses([(&self.lease, DeviceAccess::Write)])?;
        // Retire the oldest retained window before this entry reads a new
        // source window or allocates a new host owner.
        self.make_room()?;
        let run = {
            let mut p = self
                .progress
                .lock()
                .map_err(|_| Error::State("source page progress poisoned".into()))?;
            if p.failed {
                return Err(Error::State("source page retired transfer failed".into()));
            }
            if p.registered_event != Some(flight.event) {
                if p.run.as_ref().is_some_and(|run| run_has_pending(run)) {
                    return Err(Error::State(
                        "source page stale/foreign armed flight".into(),
                    ));
                }
                // A run belongs to one original flight. Once its ring and
                // pending owners are empty, replace it rather than carrying
                // an old flight's container into a new armed flight.
                p.run = None;
                flight.retain_completion_owner(self.clone())?;
                p.registered_event = Some(flight.event);
            }
            self.ensure_run(&mut p, flight)?
        };
        if run.flight != Some(flight.event) {
            return Err(Error::State(
                "source page run belongs to another flight".into(),
            ));
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
        let placement = {
            let p = self
                .progress
                .lock()
                .map_err(|_| Error::State("source page progress poisoned".into()))?;
            if reuse {
                Some(p.cache.select(matrix, rows.start, geometry.row_bytes)?)
            } else {
                None
            }
        };
        let reused = placement.is_some_and(|p| p.hit);
        let window = if reused {
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
                || window.charged_staging_bytes() < n
            {
                return Err(Error::State(
                    "source page authentic original window differs".into(),
                ));
            }
            Some(Arc::new(WindowBytes(window)))
        };
        let (serial, offset) = {
            let mut p = self
                .progress
                .lock()
                .map_err(|_| Error::State("source page progress poisoned".into()))?;
            let serial = p
                .submitted
                .checked_add(1)
                .ok_or_else(|| Error::Limit("source page serial exhausted".into()))?;
            if let Some(place) = placement {
                p.cache.begin(matrix, place);
            } else {
                p.cache.clear();
            }
            p.submitted = serial;
            (serial, placement.map_or(0, |p| p.offset))
        };
        let handle = self
            .lease
            .handle()
            .offset_start(offset as u64)
            .offset_end((PAGE_BYTES - offset - n) as u64);
        let owner = Arc::new(PageRunOwner {
            _window: window.clone(),
            outputs: Mutex::new(Vec::new()),
        });
        // The per-flight run retains this exact owner BEFORE the first H2D and
        // before any mark exists. The flight retains only the run container, so
        // a healthy retirement can drop this window before the whole flight.
        {
            let mut inner = run
                .inner
                .lock()
                .map_err(|_| Error::State("source page run poisoned".into()))?;
            if inner.pending.is_some() {
                return Err(Error::State(
                    "source page run already has a pending owner".into(),
                ));
            }
            inner.pending = Some(owner.clone());
        }
        if let Some(window) = &window {
            let page = self.lease.handle().binding();
            let actor = self
                .transport
                .as_ref()
                .ok_or_else(|| Error::State("source page original actor absent".into()))?;
            if let Err(error) = actor.upload(page, offset, n, window.clone()) {
                self.quarantine_page(&owner);
                return Err(Error::State(format!(
                    "source page original upload: {error}"
                )));
            }
            self.service.record_value_upload(n);
        }
        let view = SourcePageView {
            handle: handle.clone(),
            matrix,
            rows: rows.clone(),
            bytes: n,
        };
        let completion = PageCompletion {
            serial,
            placement,
            row: rows.start,
            bytes: n,
            reused,
        };
        let outputs = match enqueue(&view) {
            Ok(outputs) => outputs,
            Err(error) => {
                // Consumers may already be queued behind this write; retain the
                // page/window with a real event before returning the failure.
                self.record_and_push(flight, owner, &[], completion)?;
                return Err(error);
            }
        };
        if outputs.is_empty() {
            self.record_and_push(flight, owner, &[], completion)?;
            return Err(Error::State(
                "source page requires original consumer output barrier".into(),
            ));
        }
        *owner
            .outputs
            .lock()
            .map_err(|_| Error::State("source page output owner poisoned".into()))? =
            outputs.clone();
        let page_range = crate::resident_program::BufferRange::new(&self.lease.handle())?;
        for output in &outputs {
            if page_range.overlaps(&crate::resident_program::BufferRange::new(
                &output.handle(),
            )?) {
                self.record_and_push(flight, owner, &outputs, completion)?;
                return Err(Error::State(
                    "source page consumer output aliases immutable cached rows".into(),
                ));
            }
        }
        flight.require_accesses(outputs.iter().map(|l| (l, DeviceAccess::Write)))?;
        self.record_and_push(flight, owner, &outputs, completion)?;
        if reuse {
            self.source.require_current_matrix(matrix)?;
        }
        cancelled(flag)
    }
    /// Install `owner` on the armed flight before this call; record the real
    /// page/consumer event on the original actor and retain the exact source
    /// window until a later query/wait proves it completed.
    #[cfg(feature = "hip")]
    fn record_and_push(
        &self,
        flight: &mut ResidentSubmission<'_>,
        owner: Arc<PageRunOwner>,
        outputs: &[ResidentLease],
        completion: PageCompletion,
    ) -> Result<()> {
        let page = self.lease.handle().binding();
        if page.size != PAGE_BYTES as u64
            || page.offset_start.unwrap_or(0) != 0
            || page.offset_end.unwrap_or(0) != 0
        {
            self.quarantine_page(&owner);
            return Err(Error::State(
                "source page is not one complete original allocation".into(),
            ));
        }
        let bindings = outputs
            .iter()
            .map(|lease| lease.handle().binding())
            .collect::<Vec<_>>();
        let run = {
            let mut p = self
                .progress
                .lock()
                .map_err(|_| Error::State("source page progress poisoned".into()))?;
            if p.failed {
                drop(p);
                self.quarantine_page(&owner);
                return Err(Error::State("source page retired transfer failed".into()));
            }
            self.ensure_run(&mut p, flight)?
        };
        if run.flight != Some(flight.event) {
            self.quarantine_page(&owner);
            return Err(Error::State(
                "source page run belongs to another flight".into(),
            ));
        }
        let reusable = loop {
            let mut inner = run
                .inner
                .lock()
                .map_err(|_| Error::State("source page run poisoned".into()))?;
            if let Some(ring) = inner.ring.as_mut() {
                break ring.take_reusable().map(|mark| mark.0);
            }
            drop(inner);
            std::thread::yield_now();
        };
        let actor = self
            .transport
            .as_ref()
            .ok_or_else(|| Error::State("source page original actor absent".into()))?;
        #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
        let probe_parts = match self.probe.lock() {
            Ok(held) => held.as_ref().cloned(),
            Err(_) => {
                self.fence_probe();
                self.quarantine_page(&owner);
                return Err(Error::State("source page probe poisoned".into()));
            }
        };
        #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
        let (payload, charge) = owner
            ._window
            .as_ref()
            .map(|window| {
                (
                    window.as_ref().as_ref().len() as u64,
                    window.0.charged_staging_bytes() as u64,
                )
            })
            .unwrap_or((0, 0));
        #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
        if let Some(parts) = &probe_parts {
            if owner._window.is_some()
                && let Err(error) = parts.state.note_copy(payload, charge)
            {
                self.fence_probe();
                self.quarantine_page(&owner);
                return Err(Error::State(format!("source page probe copy: {error}")));
            }
            if !bindings.is_empty()
                && let Err(error) = parts.state.note_consumer()
            {
                self.fence_probe();
                self.quarantine_page(&owner);
                return Err(Error::State(format!("source page probe consumer: {error}")));
            }
        }
        #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
        let recorded = match (&probe_parts, reusable) {
            (Some(parts), reusable) => actor.record_probed(
                page,
                bindings,
                reusable,
                parts.state.clone(),
                payload,
                charge,
            ),
            (None, reusable) => actor.record(page, bindings, reusable),
        };
        #[cfg(not(all(feature = "hip", feature = "hardware-abort-probe")))]
        let recorded = actor.record(page, bindings, reusable);
        let mark = match recorded {
            Ok(mark) => mark,
            Err(error) => {
                self.quarantine_page(&owner);
                return Err(Error::State(format!(
                    "source page original event record: {error}"
                )));
            }
        };
        #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
        let captured = if let Some(parts) = &probe_parts {
            let event = mark.event_identity();
            parts.witness.capture(event, mark.serial(), &owner)
        } else {
            Ok(())
        };
        let pushed = loop {
            let mut inner = run
                .inner
                .lock()
                .map_err(|_| Error::State("source page run poisoned".into()))?;
            if inner.ring.is_none() {
                drop(inner);
                std::thread::yield_now();
                continue;
            }
            if !inner
                .pending
                .as_ref()
                .is_some_and(|pending| Arc::ptr_eq(pending, &owner))
            {
                drop(inner);
                self.quarantine_page(&owner);
                return Err(Error::State("source page run pending owner differs".into()));
            }
            inner.pending = None;
            let pushed = inner
                .ring
                .as_mut()
                .expect("ring checked above")
                .push(PageMark(mark), owner);
            if pushed.is_ok() {
                inner.completed_runs.push(completion);
            }
            break pushed;
        };
        match pushed {
            Ok(()) => {
                #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
                if let Err(error) = captured {
                    self.fence_probe();
                    if let Ok(mut p) = self.progress.lock() {
                        p.failed = true;
                    }
                    return Err(error);
                }
                Ok(())
            }
            // `push` preserves the incoming mark and owner in quarantine on
            // refusal; do not release or quarantine them a second time.
            Err(error) => {
                if let Ok(mut p) = self.progress.lock() {
                    p.failed = true;
                }
                Err(error)
            }
        }
    }
    #[cfg(feature = "hip")]
    fn ensure_run(
        &self,
        p: &mut Progress,
        flight: &mut ResidentSubmission<'_>,
    ) -> Result<Arc<PageRunState>> {
        if let Some(run) = &p.run {
            return Ok(run.clone());
        }
        if self.transport.is_none() {
            return Err(Error::State("source page original actor absent".into()));
        }
        let service = self.service.clone();
        let lease = self.lease.clone();
        let client = service.client().clone();
        let ring_client = client.clone();
        let ring_service = service.clone();
        let ring_lease = lease.clone();
        let ring = TransferRetirementRing::new(
            RETIRED_PAGE_TRANSFERS,
            Vec::new(),
            move |pending: Vec<(PageMark, Arc<PageRunOwner>)>| {
                let mut handles = vec![ring_lease.handle()];
                for (_, owner) in &pending {
                    if let Ok(outputs) = owner.outputs.lock() {
                        handles.extend(outputs.iter().map(ResidentLease::handle));
                    }
                }
                let keepalive: crate::physical_quarantine::CompletionOwner =
                    Arc::new((ring_service.clone(), ring_lease.clone(), pending));
                crate::physical_quarantine::quarantine_uncertain_with_owners(
                    ring_client,
                    handles,
                    vec![keepalive],
                );
            },
        )?;
        let pending_client = client;
        let pending = Box::new(move |owner: Arc<PageRunOwner>| {
            let mut handles = vec![lease.handle()];
            if let Ok(outputs) = owner.outputs.lock() {
                handles.extend(outputs.iter().map(ResidentLease::handle));
            }
            let keepalive: crate::physical_quarantine::CompletionOwner =
                Arc::new((service.clone(), lease.clone(), owner));
            crate::physical_quarantine::quarantine_uncertain_with_owners(
                pending_client,
                handles,
                vec![keepalive],
            );
        });
        let run = Arc::new(PageRunState {
            flight: Some(flight.event),
            inner: Mutex::new(RunInner {
                pending: None,
                ring: Some(ring),
                completed_runs: Vec::new(),
            }),
            quarantine: Some(pending),
        });
        flight.retain_completion_owner(run.clone())?;
        p.run = Some(run.clone());
        Ok(run)
    }
    /// Diagnostic failures also fence the original native actor; coarse
    /// quarantine alone is a distinct authority.
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    fn fence_probe(&self) {
        if let Some(actor) = self.transport.as_ref() {
            let _ = actor.fence_diagnostic();
        }
    }
    #[cfg(feature = "hip")]
    fn quarantine_page(&self, owner: &Arc<PageRunOwner>) {
        let mut handles = vec![self.lease.handle()];
        if let Ok(outputs) = owner.outputs.lock() {
            handles.extend(outputs.iter().map(ResidentLease::handle));
        }
        let keepalive: crate::physical_quarantine::CompletionOwner = Arc::new((
            self.service.clone(),
            self.lease.clone(),
            vec![owner.clone()],
        ));
        crate::physical_quarantine::quarantine_uncertain_with_owners(
            self.service.client().clone(),
            handles,
            vec![keepalive],
        );
        if let Ok(mut p) = self.progress.lock() {
            p.failed = true;
            if let Some(run) = p.run.as_ref() {
                if let Ok(mut inner) = run.inner.lock() {
                    if inner
                        .pending
                        .as_ref()
                        .is_some_and(|pending| Arc::ptr_eq(pending, owner))
                    {
                        inner.pending = None;
                    }
                }
            }
        }
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
        #[cfg(feature = "hip")]
        if self.transport.is_some() {
            if !reuse {
                return self
                    .with_rows_pipelined(flight, matrix, rows, flag, reuse, enqueue)
                    .await;
            }
            // Cached-row reuse remains the original completed-only synchronous
            // path: retire the ordinary run first, then execute the serial
            // consumer checkpoint. No whole-flight Ready comes from a mark.
            self.drain()?;
        }
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

#[cfg(all(test, feature = "hip"))]
mod transport_tests {
    use super::*;

    #[test]
    fn ordinary_windows_charge_actual_integrity_chunks_once() {
        // Default production scan chunk is 4 MiB. An unaligned page-sized
        // window intersects two whole chunks; two retained windows charge four
        // chunks, not two nominal page payloads.
        let chunk = 4 * 1024 * 1024;
        let meta = std::mem::size_of::<(usize, Range<usize>)>();
        let exact = transport_host_charge(chunk, 0).unwrap();
        let staged = 2 * 2 * (chunk + meta);
        assert!(exact >= staged + CONTROL_BYTES);
        assert!(exact < staged + CONTROL_BYTES + 64 * 1024);
        // A 1 MiB integrity chunk has the same two-chunk straddle bound.
        let one = transport_host_charge(1024 * 1024, 0).unwrap();
        assert!(one >= 2 * 2 * (1024 * 1024 + meta) + CONTROL_BYTES);
        // The persistent cache budget is charged once, not per retained window.
        let cached = transport_host_charge(chunk, 8 * 1024 * 1024).unwrap();
        assert!(cached >= exact + 8 * 1024 * 1024);
        assert!(cached < exact + 8 * 1024 * 1024 + 1024 * 1024);
        assert!(transport_host_charge(0, 0).is_err());
    }
}
/// Shared CPU fixture: a real on-disk verified source with a tiny chunk
/// geometry. Used by the HIP owner controls and by the serial charge control.
#[cfg(all(
    test,
    any(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification"
    )
))]
mod source_fixture {
    use sha2::{Digest, Sha256};
    use std::fs::{self, File};
    use std::sync::atomic::{AtomicU64, Ordering};
    use zerogravity_artifacts::artifact::Artifact;
    use zerogravity_artifacts::cache::TransferAdmission;
    use zerogravity_artifacts::source_weights::{
        ExpectedSourceShard, SourceWeights, VerifiedWindowPlan, WindowOpenLimits,
    };
    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

    const SHARD: &str = "model.safetensors";
    const INDEX: &[u8] =
        br#"{"metadata":{"total_size":32},"weight_map":{"m":"model.safetensors"}}"#;
    const HEADER: &[u8] = br#"{"m":{"dtype":"BF16","shape":[2,8],"data_offsets":[0,32]}}"#;

    fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn limits(chunk: usize) -> WindowOpenLimits {
        WindowOpenLimits {
            max_source_bytes: 1024 * 1024,
            max_metadata_bytes: 16 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 1,
            scan_chunk_bytes: chunk,
        }
    }
    pub(super) fn admission() -> TransferAdmission {
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("host", "source-page-owner-tests").unwrap();
        arbiter
            .register(resource.clone(), 64 * 1024 * 1024)
            .unwrap();
        TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("source-page-fixture", "windows").unwrap(),
        }
    }
    pub(super) struct Fixture {
        dir: std::path::PathBuf,
        pub(super) bytes: Vec<u8>,
    }
    impl Fixture {
        pub(super) fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let dir = std::env::temp_dir().join(format!(
                "zg-source-page-owner-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&dir).unwrap();
            let mut bytes = (HEADER.len() as u64).to_le_bytes().to_vec();
            bytes.extend(HEADER);
            bytes.extend((0..16u16).flat_map(u16::to_le_bytes));
            fs::write(dir.join(SHARD), &bytes).unwrap();
            Self { dir, bytes }
        }
        pub(super) fn plan(&self, admission: &TransferAdmission) -> VerifiedWindowPlan {
            self.plan_chunk(admission, 16)
        }
        pub(super) fn plan_chunk(
            &self,
            admission: &TransferAdmission,
            chunk: usize,
        ) -> VerifiedWindowPlan {
            let shard = ExpectedSourceShard::new(
                SHARD.into(),
                Artifact::from_file(
                    File::open(self.dir.join(SHARD)).unwrap(),
                    std::path::Path::new(SHARD),
                )
                .unwrap(),
                self.bytes.len() as u64,
                sha(&self.bytes),
            )
            .unwrap();
            SourceWeights::prepare_verified_windows(
                INDEX,
                &sha(INDEX),
                vec![shard],
                limits(chunk),
                admission,
            )
            .unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

/// Serial (non-HIP) source transport has no ring: one ordinary window is
/// staged by whole intersecting integrity chunks. The host charge must be that
/// source-derived amount on top of the unchanged three-page family baseline.
#[cfg(all(test, not(feature = "hip"), feature = "gemma-bf16-qualification"))]
mod serial_transport_tests {
    use super::source_fixture::{Fixture, admission};
    use super::*;

    #[test]
    fn serial_transport_charge_uses_whole_source_chunks() {
        let fixture = Fixture::new();
        let admission = admission();
        let source = fixture.plan(&admission).open().unwrap();
        let chunk = source.staging_chunk_bytes();
        assert!(chunk > 0);
        let chunks = PAGE_BYTES.div_ceil(chunk) + 1;
        let meta = std::mem::size_of::<(usize, Range<usize>)>();
        let window = chunks * (chunk + meta);
        let host = transport_host_bytes(&source).unwrap();
        // The non-HIP build prices exactly the serial route: unchanged
        // three-page baseline plus whole chunks/offset metadata.
        assert_eq!(host, serial_transport_host_charge(chunk).unwrap());
        assert_eq!(host, 3 * PAGE_BYTES + window);
        // The persistent cache budget lives in the original metadata envelope;
        // the serial transport term does not credit it a second time.
        assert!(source.window_cache_stats().unwrap().budget_bytes > 0);
        // Full serial window staging, not the empty 64 KiB control charge this
        // route used to return; the page payload alone is not enough.
        assert!(host >= 3 * PAGE_BYTES + PAGE_BYTES);
        assert!(host > CONTROL_BYTES * 16);
        assert!(window_staging_charge(0).is_err());
    }
}

/// A HIP-compiled binary can still select the serial page transport when its
/// prepared set holds no live scoped HIP source-page actor. The host charge
/// must be the larger of the two actual source-derived routes, and the
/// persistent window-cache budget must not be summed twice.
#[cfg(all(test, feature = "hip", feature = "gemma-bf16-qualification"))]
mod combined_route_transport_tests {
    use super::source_fixture::{Fixture, admission};
    use super::*;

    #[test]
    fn combined_route_charges_the_larger_actual_serial_or_hip_bound() {
        let fixture = Fixture::new();
        let admission = admission();
        let meta = std::mem::size_of::<(usize, Range<usize>)>();
        for chunk in [
            16usize,
            PAGE_BYTES / 2,
            PAGE_BYTES,
            PAGE_BYTES + 16,
            4 * 1024 * 1024,
        ] {
            // Whole intersecting chunks plus the straddle: the exact staging
            // charge of one page-sized window at this scan geometry.
            let chunks = PAGE_BYTES.div_ceil(chunk) + 1;
            let window = chunks * (chunk + meta);
            for cached in [false, true] {
                let plan = fixture.plan_chunk(&admission, chunk);
                let plan = if cached {
                    plan
                } else {
                    plan.without_window_cache()
                };
                let source = plan.open().unwrap();
                let budget = source.window_cache_stats().unwrap().budget_bytes;
                assert_eq!(budget > 0, cached);
                let serial = serial_transport_host_charge(chunk).unwrap();
                assert_eq!(serial, 3 * PAGE_BYTES + window);
                // The HIP term keeps the actual persistent cache budget once.
                let hip = transport_host_charge(chunk, budget).unwrap();
                let host = transport_host_bytes(&source).unwrap();
                // The compiled helper never returns less than either route the
                // same binary can actually select, and never sums both.
                assert_eq!(host, serial.max(hip));
                assert!(host >= serial && host >= hip);
                if chunk == PAGE_BYTES {
                    // The reported gap: with no retained cache budget the
                    // serial route needs 5 MiB+48 B while HIP prices
                    // 4 MiB+96 B+64 KiB. HIP-only selection underpaid it.
                    if !cached {
                        assert!(serial > hip);
                        assert!(host > hip);
                    }
                }
                if chunk == 4 * 1024 * 1024 {
                    // Two whole 4 MiB chunks per retained window make the HIP
                    // bound exceed the serial three-page baseline + window.
                    assert!(hip >= serial);
                }
            }
        }
        assert!(window_staging_charge(0).is_err());
        assert!(serial_transport_host_charge(0).is_err());
        assert!(serial_transport_host_charge(usize::MAX).is_err());
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

#[cfg(all(test, feature = "qwen-source-qualification"))]
mod real_owner_tests {
    use super::source_fixture::{Fixture, admission};
    use super::*;
    use cubecl_common::{
        device::{DeviceId, DeviceService, ServerUtilitiesHandle},
        device_handle::DeviceHandle,
    };
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    #[test]
    fn real_pending_window_drops_before_run_container() {
        let fixture = Fixture::new();
        let admission = admission();
        let source = fixture.plan(&admission).open().unwrap();
        let matrix = source.matrix("m", MatrixAxes::OutputInput).unwrap();
        let window = source.read_rows(&matrix, 0..1).unwrap();
        let window = Arc::new(WindowBytes(window));
        let weak = Arc::downgrade(&window);
        let owner = Arc::new(PageRunOwner {
            _window: Some(window),
            outputs: Mutex::new(Vec::new()),
        });
        let run: Arc<PageRunState> = Arc::new(PageRunState {
            flight: None,
            inner: Mutex::new(RunInner {
                pending: Some(owner),
                ring: None,
                completed_runs: Vec::new(),
            }),
            quarantine: None,
        });
        // A healthy record transfers the exact pending owner into the ring;
        // dropping that owner releases the real source window while the run
        // container and its flight are still alive.
        assert!(run.inner.lock().unwrap().pending.take().is_some());
        assert!(weak.upgrade().is_none());
        assert_eq!(Arc::strong_count(&run), 1);
    }

    /// The only part of the production retirement path a CPU-only control
    /// cannot execute is the HIP event query itself. Retirement stays on the
    /// real `TransferRetirementRing`, `RunInner` and `PageRunOwner` types.
    struct HostMark {
        ready: Arc<AtomicBool>,
        failing: bool,
        /// Set by the drop of a mark whose real event destroy would refuse;
        /// production Drop then fences the original service and retains the
        /// event instead of publishing completion.
        destroy_fenced: Option<Arc<AtomicBool>>,
    }
    impl Drop for HostMark {
        fn drop(&mut self) {
            if let Some(fenced) = &self.destroy_fenced {
                fenced.store(true, Ordering::Release);
            }
        }
    }
    impl RetirementMark for HostMark {
        fn ready(&mut self) -> Result<bool> {
            if self.failing {
                return Err(Error::State("host event query failure".into()));
            }
            Ok(self.ready.load(Ordering::Acquire))
        }
        fn wait(&mut self) -> Result<()> {
            self.ready.store(true, Ordering::Release);
            Ok(())
        }
    }
    fn host_mark(ready: &Arc<AtomicBool>, failing: bool) -> HostMark {
        HostMark {
            ready: ready.clone(),
            failing,
            destroy_fenced: None,
        }
    }
    fn fenced_host_mark(ready: &Arc<AtomicBool>, fenced: &Arc<AtomicBool>) -> HostMark {
        HostMark {
            ready: ready.clone(),
            failing: false,
            destroy_fenced: Some(fenced.clone()),
        }
    }
    fn real_window(
        source: &VerifiedTensorSource,
        matrix: &VerifiedMatrix,
        row: usize,
    ) -> Arc<WindowBytes> {
        Arc::new(WindowBytes(source.read_rows(matrix, row..row + 1).unwrap()))
    }
    fn run_with_ring(
        ring: TransferRetirementRing<HostMark, Arc<PageRunOwner>>,
    ) -> Arc<PageRunState<HostMark>> {
        Arc::new(PageRunState {
            flight: None,
            inner: Mutex::new(RunInner {
                pending: None,
                ring: Some(ring),
                completed_runs: Vec::new(),
            }),
            quarantine: None,
        })
    }

    #[test]
    fn many_admitted_windows_share_one_flight_run_and_retire_before_it() {
        let fixture = Fixture::new();
        let admission = admission();
        let source = fixture.plan(&admission).open().unwrap();
        let matrix = source.matrix("m", MatrixAxes::OutputInput).unwrap();
        let ready = Arc::new(AtomicBool::new(false));
        let held = Arc::new(Mutex::new(Vec::new()));
        let captured = held.clone();
        let ring: TransferRetirementRing<HostMark, Arc<PageRunOwner>> =
            TransferRetirementRing::new(RETIRED_PAGE_TRANSFERS, Vec::new(), move |owners| {
                captured.lock().unwrap().extend(owners)
            })
            .unwrap();
        let run = run_with_ring(ring);
        let mut weaks = Vec::new();
        let mut windows = 0usize;
        for index in 0..8usize {
            let window = real_window(&source, &matrix, index % matrix.rows());
            weaks.push(Arc::downgrade(&window));
            let mut inner = run.inner.lock().unwrap();
            let ring = inner.ring.as_mut().unwrap();
            // One bound for every admitted window: at most the ring capacity
            // survives, and the oldest REAL window is retired before the next
            // source read allocates a new host owner.
            ring.make_room().unwrap();
            ring.push(
                host_mark(&ready, false),
                Arc::new(PageRunOwner {
                    _window: Some(window),
                    outputs: Mutex::new(Vec::new()),
                }),
            )
            .unwrap();
            assert!(ring.counts().pending <= RETIRED_PAGE_TRANSFERS);
            assert_eq!(ring.counts().issued, windows + 1);
            windows += 1;
            drop(inner);
            // The flight/run container is singular; windows never extend it.
            assert_eq!(Arc::strong_count(&run), 1);
        }
        assert!(weaks.iter().filter(|w| w.upgrade().is_none()).count() >= 6);
        assert!(held.lock().unwrap().is_empty());
        // Healthy terminal close, exactly as `finish`: drain, take retired
        // marks (HIP events would be destroyed here on the original actor),
        // then release the run reference while the flight still holds it.
        let retired = {
            let mut inner = run.inner.lock().unwrap();
            inner.ring.as_mut().unwrap().drain().unwrap();
            inner.ring.as_mut().unwrap().take_retired_marks().unwrap()
        };
        assert_eq!(retired.len(), windows);
        drop(retired);
        assert_eq!(Arc::strong_count(&run), 1);
        drop(run);
        assert!(weaks.iter().all(|w| w.upgrade().is_none()));
        assert!(held.lock().unwrap().is_empty());
    }

    #[test]
    fn unresolved_ring_owner_keeps_real_window_through_run_close() {
        let fixture = Fixture::new();
        let admission = admission();
        let source = fixture.plan(&admission).open().unwrap();
        let matrix = source.matrix("m", MatrixAxes::OutputInput).unwrap();
        let ready = Arc::new(AtomicBool::new(false));
        let held = Arc::new(Mutex::new(Vec::new()));
        let captured = held.clone();
        let ring: TransferRetirementRing<HostMark, Arc<PageRunOwner>> =
            TransferRetirementRing::new(RETIRED_PAGE_TRANSFERS, Vec::new(), move |owners| {
                captured.lock().unwrap().extend(owners)
            })
            .unwrap();
        let run = run_with_ring(ring);
        let window = real_window(&source, &matrix, 0);
        let weak = Arc::downgrade(&window);
        {
            let mut inner = run.inner.lock().unwrap();
            inner
                .ring
                .as_mut()
                .unwrap()
                .push(
                    host_mark(&ready, false),
                    Arc::new(PageRunOwner {
                        _window: Some(window),
                        outputs: Mutex::new(Vec::new()),
                    }),
                )
                .unwrap();
            // Below capacity: no wait happens, the owner stays pending.
            assert!(inner.ring.as_mut().unwrap().make_room().is_ok());
            assert_eq!(inner.ring.as_ref().unwrap().counts().pending, 1);
        }
        drop(run);
        // Run close with an unresolved owner quarantines it; the real window
        // is never released on an unproven transfer.
        assert_eq!(held.lock().unwrap().len(), 1);
        assert!(weak.upgrade().is_some());
        held.lock().unwrap().clear();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn event_query_failure_retains_real_window_owner() {
        let fixture = Fixture::new();
        let admission = admission();
        let source = fixture.plan(&admission).open().unwrap();
        let matrix = source.matrix("m", MatrixAxes::OutputInput).unwrap();
        let ready = Arc::new(AtomicBool::new(false));
        let held = Arc::new(Mutex::new(Vec::new()));
        let captured = held.clone();
        let ring: TransferRetirementRing<HostMark, Arc<PageRunOwner>> =
            TransferRetirementRing::new(RETIRED_PAGE_TRANSFERS, Vec::new(), move |owners| {
                captured.lock().unwrap().extend(owners)
            })
            .unwrap();
        let run = run_with_ring(ring);
        let window = real_window(&source, &matrix, 1);
        let weak = Arc::downgrade(&window);
        {
            let mut inner = run.inner.lock().unwrap();
            inner
                .ring
                .as_mut()
                .unwrap()
                .push(
                    host_mark(&ready, true),
                    Arc::new(PageRunOwner {
                        _window: Some(window),
                        outputs: Mutex::new(Vec::new()),
                    }),
                )
                .unwrap();
            assert!(inner.ring.as_mut().unwrap().make_room().is_err());
            assert_eq!(inner.ring.as_ref().unwrap().counts().retired, 0);
            assert_eq!(inner.ring.as_ref().unwrap().counts().pending, 1);
        }
        drop(run);
        assert_eq!(held.lock().unwrap().len(), 1);
        assert!(weak.upgrade().is_some());
        held.lock().unwrap().clear();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn unresolved_pending_quarantine_retains_real_window() {
        let fixture = Fixture::new();
        let admission = admission();
        let source = fixture.plan(&admission).open().unwrap();
        let matrix = source.matrix("m", MatrixAxes::OutputInput).unwrap();
        let window = source.read_rows(&matrix, 0..1).unwrap();
        let window = Arc::new(WindowBytes(window));
        let weak = Arc::downgrade(&window);
        let retained = Arc::new(Mutex::new(Vec::new()));
        let captured = retained.clone();
        let owner = Arc::new(PageRunOwner {
            _window: Some(window),
            outputs: Mutex::new(Vec::new()),
        });
        let run: Arc<PageRunState> = Arc::new(PageRunState {
            flight: None,
            inner: Mutex::new(RunInner {
                pending: Some(owner),
                ring: None,
                completed_runs: Vec::new(),
            }),
            quarantine: Some(Box::new(move |owner| {
                captured.lock().unwrap().push(owner);
            })),
        });
        drop(run);
        assert_eq!(retained.lock().unwrap().len(), 1);
        assert!(weak.upgrade().is_some());
        retained.lock().unwrap().clear();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn reusable_mark_is_dropped_outside_the_original_actor_refusal_seam() {
        // Minimal real `DeviceHandle` service: the CPU-observable actor seam.
        // The token stands for the production HIP event and is destroyed only
        // from the caller frame by a nested `submit_blocking`, exactly like
        // `MarkDrop`. A drop inside the running task would instead reach the
        // same runner with the service state already borrowed.
        struct SeamProbe;
        impl DeviceService for SeamProbe {
            fn init(_: DeviceId) -> Self {
                Self
            }
            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }
        }
        struct SeamShutdown(DeviceId);
        impl Drop for SeamShutdown {
            fn drop(&mut self) {
                DeviceHandle::<SeamProbe>::shutdown(self.0);
            }
        }
        struct SeamMark {
            handle: DeviceHandle<SeamProbe>,
            event: Option<u64>,
            unknown: Arc<AtomicBool>,
            destroyed: Arc<AtomicUsize>,
            retained: Arc<AtomicUsize>,
        }
        impl SeamMark {
            fn recycle(
                &mut self,
                generation: u64,
                expected: u64,
            ) -> std::result::Result<u64, &'static str> {
                if generation != expected {
                    return Err("generation differs");
                }
                self.event.take().ok_or("mark already recycled")
            }
        }
        impl Drop for SeamMark {
            fn drop(&mut self) {
                let Some(event) = self.event.take() else {
                    return;
                };
                if self.unknown.load(Ordering::Acquire) {
                    self.retained.fetch_add(1, Ordering::AcqRel);
                    return;
                }
                if self.handle.submit_blocking(move |_| event).is_ok() {
                    self.destroyed.fetch_add(1, Ordering::AcqRel);
                } else {
                    self.retained.fetch_add(1, Ordering::AcqRel);
                }
            }
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = DeviceId {
            type_id: u16::MAX - 1,
            index_id: NEXT.fetch_add(1, Ordering::Relaxed) as u16,
        };
        let shutdown = SeamShutdown(id);
        let handle = DeviceHandle::<SeamProbe>::new(id);
        let destroyed = Arc::new(AtomicUsize::new(0));
        let retained = Arc::new(AtomicUsize::new(0));
        let unknown = Arc::new(AtomicBool::new(false));
        let make_mark = || SeamMark {
            handle: handle.clone(),
            event: Some(7),
            unknown: unknown.clone(),
            destroyed: destroyed.clone(),
            retained: retained.clone(),
        };

        // Early refusal before the event is taken: the mark stays owned by
        // this caller frame, and its destroy runs after the actor task
        // returned (not from inside the borrowed service state).
        let mut mark = make_mark();
        let refused = {
            let reusable = Some(&mut mark);
            handle
                .submit_blocking(move |_| match reusable {
                    Some(mark) => mark.recycle(9, 3),
                    None => Err("mark absent"),
                })
                .unwrap()
        };
        assert_eq!(refused, Err("generation differs"));
        assert_eq!(mark.event, Some(7), "refusal keeps the real event");
        drop(mark);
        assert_eq!(destroyed.load(Ordering::Acquire), 1);
        assert_eq!(retained.load(Ordering::Acquire), 0);

        // Unknown actor/record state retains the event; nothing is destroyed.
        unknown.store(true, Ordering::Release);
        drop(make_mark());
        assert_eq!(retained.load(Ordering::Acquire), 1);
        assert_eq!(destroyed.load(Ordering::Acquire), 1);
        unknown.store(false, Ordering::Release);

        // Successful recycle transfers the raw event exactly once; the empty
        // mark must not destroy anything on drop.
        let mut mark = make_mark();
        let event = {
            let reusable = Some(&mut mark);
            handle
                .submit_blocking(move |_| match reusable {
                    Some(mark) => mark.recycle(3, 3),
                    None => Err("mark absent"),
                })
                .unwrap()
        };
        assert_eq!(event, Ok(7));
        drop(mark);
        assert_eq!(destroyed.load(Ordering::Acquire), 1);

        drop(handle);
        drop(shutdown);
    }

    #[test]
    fn finish_destroy_refusal_fences_before_releasing_the_run() {
        let fixture = Fixture::new();
        let admission = admission();
        let source = fixture.plan(&admission).open().unwrap();
        let matrix = source.matrix("m", MatrixAxes::OutputInput).unwrap();
        let ready = Arc::new(AtomicBool::new(false));
        let fenced = Arc::new(AtomicBool::new(false));
        let retained = Arc::new(Mutex::new(Vec::new()));
        let captured = retained.clone();
        let ring: TransferRetirementRing<HostMark, Arc<PageRunOwner>> =
            TransferRetirementRing::new(RETIRED_PAGE_TRANSFERS, Vec::new(), move |owners| {
                captured.lock().unwrap().extend(owners)
            })
            .unwrap();
        let run = run_with_ring(ring);
        for index in 0..RETIRED_PAGE_TRANSFERS {
            let window = real_window(&source, &matrix, index);
            run.inner
                .lock()
                .unwrap()
                .ring
                .as_mut()
                .unwrap()
                .push(
                    fenced_host_mark(&ready, &fenced),
                    Arc::new(PageRunOwner {
                        _window: Some(window),
                        outputs: Mutex::new(Vec::new()),
                    }),
                )
                .unwrap();
        }
        // Production finish order: drain, take the retired marks, drop them
        // (the real event destroy on the original actor), then recheck the
        // original authority before releasing this page's run reference.
        let run_ref = run.clone();
        let retired = {
            let mut inner = run.inner.lock().unwrap();
            inner.ring.as_mut().unwrap().drain().unwrap();
            inner.ring.as_mut().unwrap().take_retired_marks().unwrap()
        };
        assert_eq!(retired.len(), RETIRED_PAGE_TRANSFERS);
        drop(retired);
        // A refused destroy fences the original service. Finish must observe
        // that instead of publishing Ready, and keep the run/owner container
        // alive so the armed flight still owns completion.
        let authority_live = !fenced.load(Ordering::Acquire);
        assert!(!authority_live);
        assert_eq!(Arc::strong_count(&run), 2, "refused finish retains the run");
        assert!(retained.lock().unwrap().is_empty());
        drop(run_ref);
        drop(run);
    }

    #[test]
    fn real_source_charge_uses_actual_chunks_and_shared_cache_once() {
        let fixture = Fixture::new();
        let admission = admission();
        let source = fixture.plan(&admission).open().unwrap();
        let matrix = source.matrix("m", MatrixAxes::OutputInput).unwrap();
        let window = source.read_rows(&matrix, 0..2).unwrap();
        let staged = window.charged_staging_bytes();
        assert!(staged > 0);
        let cache = source.window_cache_stats().unwrap().budget_bytes;
        let charged = transport_host_bytes(&source).unwrap();
        // Two retained windows charge the actual intersecting whole chunks;
        // the persistent cache budget is included once, not per window.
        let two = staged
            .checked_mul(RETIRED_PAGE_TRANSFERS)
            .unwrap()
            .max(2 * 2 * (source.staging_chunk_bytes() + 16));
        assert!(charged >= two + cache + CONTROL_BYTES);
    }
}
