//! Original-actor ownership for source-page H2D and consumer retirement.
//!
//! A source page is one existing 1 MiB allocation. Ordinary projection windows
//! reuse it in place, so a write must not be treated as complete when the host
//! enqueue returns. The actor records one event after the page write and the
//! consumers already queued behind it on the page's logical stream; the ring
//! only retires the retained source window once that event reports completion.
//!
//! This is deliberately not a second queue or stream: every operation resolves
//! handles on the existing original stream, and any unknown storage state fails
//! closed before a new write can overwrite bytes an earlier fault still owns.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cubecl_common::device::ServiceId;
use cubecl_common::device_handle::DeviceHandle;
use cubecl_core::server::{BufferBinding, ServerError};
use cubecl_environment::stream::StreamId;

use crate::compute::HipServer;
use crate::compute::status::checked;
use crate::prepared_native::SourcePageAuthority;
use cubecl_server::stream::{ExecuteScope, FailureStore};

fn same_page(left: &BufferBinding, right: &BufferBinding) -> bool {
    left.service == right.service
        && left.stream == right.stream
        && left.memory.id() == right.memory.id()
        && left.size == right.size
        && left.offset_start == right.offset_start
        && left.offset_end == right.offset_end
}

#[cfg(not(feature = "source-page-event-probe"))]
struct SourcePageEventProbeState;

#[cfg(feature = "source-page-event-probe")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourcePageEventProbeMode {
    QueryUnknown,
    WaitUnknown,
    DestroyUnknown,
}

#[cfg(feature = "source-page-event-probe")]
impl SourcePageEventProbeMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::QueryUnknown => "query",
            Self::WaitUnknown => "wait",
            Self::DestroyUnknown => "destroy",
        }
    }
    /// Canonical finite fixture mode; `name` is only the injection stage.
    pub fn full_name(self) -> &'static str {
        match self {
            Self::QueryUnknown => "source-page-query-unknown",
            Self::WaitUnknown => "source-page-wait-unknown",
            Self::DestroyUnknown => "source-page-destroy-unknown",
        }
    }
    fn result(self) -> &'static str {
        match self {
            Self::QueryUnknown => "injected-policy-unknown",
            Self::WaitUnknown => "injected-policy-unknown",
            Self::DestroyUnknown => "injected-destroy-refused",
        }
    }
}

/// Fixed observation of one actual source-page event. No history is retained.
#[cfg(feature = "source-page-event-probe")]
#[derive(Clone, Debug)]
pub struct SourcePageEventProbeSnapshot {
    pub mode: &'static str,
    pub service: ServiceId,
    pub generation: u64,
    pub actor: ServiceId,
    pub set_key: u64,
    pub logical_stream: u64,
    pub page_allocation_id: u64,
    pub event_identity: Option<u64>,
    pub first_serial: u64,
    pub current_serial: u64,
    pub window_payload_bytes: u64,
    pub source_window_charge_bytes: u64,
    pub reached_copy: bool,
    pub reached_consumer: bool,
    pub reached_record: bool,
    pub query_first: Option<i32>,
    pub query_last: Option<i32>,
    pub query_success_count: u64,
    pub query_not_ready_count: u64,
    pub query_error_count: u64,
    pub synchronize: Option<i32>,
    pub destroy: Option<i32>,
    pub injection_hits: u64,
    pub injection_stage: &'static str,
    pub injection_result: &'static str,
    pub retirement_hold_count: u64,
    pub selected_retire_success: u64,
    pub selected_recycle_success: u64,
    pub destroy_calls: u64,
    pub destroy_success: u64,
}

#[cfg(feature = "source-page-event-probe")]
struct ProbeSelected {
    event: u64,
    serial: u64,
    first_serial: u64,
    current_serial: u64,
    payload: u64,
    charge: u64,
    query_first: Option<i32>,
    query_last: Option<i32>,
    query_success: u64,
    query_not_ready: u64,
    query_error: u64,
    synchronize: Option<i32>,
    destroy: Option<i32>,
    retire_success: u64,
    recycle_success: u64,
    destroy_calls: u64,
    destroy_success: u64,
}

#[cfg(feature = "source-page-event-probe")]
struct ProbeInner {
    mode: SourcePageEventProbeMode,
    service: ServiceId,
    generation: u64,
    actor: ServiceId,
    set_key: u64,
    logical_stream: u64,
    reached_copy: bool,
    reached_consumer: bool,
    reached_record: bool,
    payload: u64,
    charge: u64,
    selected: Option<ProbeSelected>,
    hits: u64,
    holds: u64,
}

/// Bounded diagnostic state attached to one existing source-page actor. It
/// stores only scalar identities and the selected event; no Arc/Weak owner is
/// held here, so it never extends a run, window or native SET lifetime.
#[cfg(feature = "source-page-event-probe")]
pub struct SourcePageEventProbeState {
    inner: Mutex<ProbeInner>,
}

#[cfg(feature = "source-page-event-probe")]
impl SourcePageEventProbeState {
    fn new(
        mode: SourcePageEventProbeMode,
        service: ServiceId,
        generation: u64,
        actor: ServiceId,
        set_key: u64,
        logical_stream: u64,
    ) -> Self {
        Self {
            inner: Mutex::new(ProbeInner {
                mode,
                service,
                generation,
                actor,
                set_key,
                logical_stream,
                reached_copy: false,
                reached_consumer: false,
                reached_record: false,
                payload: 0,
                charge: 0,
                selected: None,
                hits: 0,
                holds: 0,
            }),
        }
    }
    fn with<R>(&self, f: impl FnOnce(&mut ProbeInner) -> R) -> Result<R, ServerError> {
        self.inner
            .lock()
            .map(|mut inner| f(&mut inner))
            .map_err(|_| ServerError::graph_state("HIP source page probe state poisoned"))
    }
    pub fn mode(&self) -> Result<Option<SourcePageEventProbeMode>, ServerError> {
        self.with(|inner| Some(inner.mode))
    }
    pub fn note_copy(&self, payload: u64, charge: u64) -> Result<(), ServerError> {
        self.with(|inner| {
            inner.reached_copy = true;
            inner.payload = payload;
            inner.charge = charge;
        })
    }
    pub fn note_consumer(&self) -> Result<(), ServerError> {
        self.with(|inner| inner.reached_consumer = true)
    }
    pub fn note_record(
        &self,
        event: u64,
        serial: u64,
        payload: u64,
        charge: u64,
    ) -> Result<(), ServerError> {
        self.with(|inner| {
            inner.reached_record = true;
            if let Some(selected) = inner.selected.as_mut() {
                if selected.event == event {
                    selected.serial = serial;
                    selected.current_serial = serial;
                    selected.payload = payload;
                    selected.charge = charge;
                    inner.payload = payload;
                    inner.charge = charge;
                }
                return;
            }
            inner.payload = payload;
            inner.charge = charge;
            inner.selected = Some(ProbeSelected {
                event,
                serial,
                // The first successful native record's actual issued serial.
                first_serial: serial,
                current_serial: serial,
                payload,
                charge,
                query_first: None,
                query_last: None,
                query_success: 0,
                query_not_ready: 0,
                query_error: 0,
                synchronize: None,
                destroy: None,
                retire_success: 0,
                recycle_success: 0,
                destroy_calls: 0,
                destroy_success: 0,
            });
        })
    }
    pub fn hold_ready(&self, event: u64, serial: u64) -> Result<bool, ServerError> {
        self.with(|inner| {
            if inner.mode != SourcePageEventProbeMode::WaitUnknown || inner.hits != 0 {
                return false;
            }
            let Some(selected) = inner.selected.as_ref() else {
                return false;
            };
            if selected.event != event || selected.serial != serial {
                return false;
            }
            inner.holds += 1;
            true
        })
    }
    fn inject(&self, event: u64, serial: u64) -> Result<bool, ServerError> {
        self.with(|inner| {
            let Some(selected) = inner.selected.as_ref() else {
                return false;
            };
            if selected.event != event || selected.serial != serial || inner.hits != 0 {
                return false;
            }
            inner.hits = 1;
            true
        })
    }
    pub fn inject_query(&self, event: u64, serial: u64) -> Result<bool, ServerError> {
        if self.mode()? != Some(SourcePageEventProbeMode::QueryUnknown) {
            return Ok(false);
        }
        self.inject(event, serial)
    }
    pub fn inject_wait(&self, event: u64, serial: u64) -> Result<bool, ServerError> {
        if self.mode()? != Some(SourcePageEventProbeMode::WaitUnknown) {
            return Ok(false);
        }
        self.inject(event, serial)
    }
    pub fn inject_destroy(&self, event: u64, serial: u64) -> Result<bool, ServerError> {
        self.with(|inner| {
            if inner.mode != SourcePageEventProbeMode::DestroyUnknown || inner.hits != 0 {
                return false;
            }
            let Some(selected) = inner.selected.as_ref() else {
                return false;
            };
            if selected.event != event || selected.serial != serial || selected.retire_success == 0
            {
                return false;
            }
            inner.hits = 1;
            true
        })
    }
    pub fn observe_query(&self, event: u64, serial: u64, status: i32) -> Result<(), ServerError> {
        self.with(|inner| {
            let Some(selected) = inner.selected.as_mut() else {
                return;
            };
            if selected.event != event || selected.serial != serial {
                return;
            }
            selected.query_first.get_or_insert(status);
            selected.query_last = Some(status);
            if status == cubecl_hip_sys::hipError_t_hipSuccess as i32 {
                selected.query_success += 1;
            } else if status == cubecl_hip_sys::hipError_t_hipErrorNotReady as i32 {
                selected.query_not_ready += 1;
            } else {
                selected.query_error += 1;
            }
        })
    }
    pub fn observe_synchronize(
        &self,
        event: u64,
        serial: u64,
        status: i32,
    ) -> Result<(), ServerError> {
        self.with(|inner| {
            if let Some(selected) = inner.selected.as_mut()
                && selected.event == event
                && selected.serial == serial
            {
                selected.synchronize = Some(status);
            }
        })
    }
    pub fn note_retired(&self, event: u64, serial: u64) -> Result<(), ServerError> {
        self.with(|inner| {
            if let Some(selected) = inner.selected.as_mut()
                && selected.event == event
                && selected.serial == serial
            {
                selected.retire_success += 1;
            }
        })
    }
    pub fn note_recycled(&self, event: u64, serial: u64) -> Result<(), ServerError> {
        self.with(|inner| {
            if let Some(selected) = inner.selected.as_mut()
                && selected.event == event
                && selected.serial == serial
            {
                selected.recycle_success += 1;
            }
        })
    }
    pub fn note_destroy_call(&self, event: u64, serial: u64) -> Result<(), ServerError> {
        self.with(|inner| {
            if let Some(selected) = inner.selected.as_mut()
                && selected.event == event
                && selected.serial == serial
            {
                selected.destroy_calls += 1;
            }
        })
    }
    pub fn note_destroy_status(
        &self,
        event: u64,
        serial: u64,
        status: i32,
    ) -> Result<(), ServerError> {
        self.with(|inner| {
            if let Some(selected) = inner.selected.as_mut()
                && selected.event == event
                && selected.serial == serial
            {
                selected.destroy = Some(status);
                if status == cubecl_hip_sys::hipError_t_hipSuccess as i32 {
                    selected.destroy_success += 1;
                }
            }
        })
    }
    /// A poisoned probe cannot prove its one-shot state; callers fence/refuse.
    pub fn require_healthy(&self) -> Result<(), ServerError> {
        self.inner
            .lock()
            .map(|_| ())
            .map_err(|_| ServerError::graph_state("HIP source page probe state poisoned"))
    }
    pub fn snapshot(&self) -> Result<SourcePageEventProbeSnapshot, ServerError> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| ServerError::graph_state("HIP source page probe state poisoned"))?;
        let selected = inner.selected.as_ref();
        Ok(SourcePageEventProbeSnapshot {
            mode: inner.mode.full_name(),
            service: inner.service,
            generation: inner.generation,
            actor: inner.actor,
            set_key: inner.set_key,
            logical_stream: inner.logical_stream,
            page_allocation_id: 0,
            event_identity: selected.map(|s| s.event),
            first_serial: selected.map(|s| s.first_serial).unwrap_or(0),
            current_serial: selected.map(|s| s.current_serial).unwrap_or(0),
            window_payload_bytes: selected.map(|s| s.payload).unwrap_or(inner.payload),
            source_window_charge_bytes: selected.map(|s| s.charge).unwrap_or(inner.charge),
            reached_copy: inner.reached_copy,
            reached_consumer: inner.reached_consumer,
            reached_record: inner.reached_record,
            query_first: selected.and_then(|s| s.query_first),
            query_last: selected.and_then(|s| s.query_last),
            query_success_count: selected.map(|s| s.query_success).unwrap_or(0),
            query_not_ready_count: selected.map(|s| s.query_not_ready).unwrap_or(0),
            query_error_count: selected.map(|s| s.query_error).unwrap_or(0),
            synchronize: selected.and_then(|s| s.synchronize),
            destroy: selected.and_then(|s| s.destroy),
            injection_hits: inner.hits,
            injection_stage: if inner.hits == 1 {
                inner.mode.name()
            } else {
                "none"
            },
            injection_result: if inner.hits == 1 {
                inner.mode.result()
            } else {
                "none"
            },
            retirement_hold_count: inner.holds,
            selected_retire_success: selected.map(|s| s.retire_success).unwrap_or(0),
            selected_recycle_success: selected.map(|s| s.recycle_success).unwrap_or(0),
            destroy_calls: selected.map(|s| s.destroy_calls).unwrap_or(0),
            destroy_success: selected.map(|s| s.destroy_success).unwrap_or(0),
        })
    }
}

/// The one serialized owner of HIP source-page events. The actual `HipServer`
/// service, logical stream, native SET key, page-allocation binding and service
/// generation are checked before every actor operation, every status query,
/// every reuse and every destroy.
///
/// The native SET key and the page-allocation identity are distinct domains.
/// The page binding is installed once, at preparation or the first actor call,
/// and every later call must match that exact retained handle.
pub struct HipSourcePageActor {
    actor: DeviceHandle<HipServer>,
    service: ServiceId,
    logical: StreamId,
    set_key: u64,
    generation: u64,
    authority: Arc<dyn SourcePageAuthority>,
    page: Mutex<Option<BufferBinding>>,
    issued: Arc<AtomicU64>,
    #[cfg(feature = "source-page-event-probe")]
    probe: Mutex<Option<Arc<SourcePageEventProbeState>>>,
}
impl HipSourcePageActor {
    pub(crate) fn new(
        actor: DeviceHandle<HipServer>,
        service: ServiceId,
        logical: StreamId,
        set_key: u64,
        authority: Arc<dyn SourcePageAuthority>,
    ) -> Self {
        let generation = authority.generation();
        Self {
            actor,
            service,
            logical,
            set_key,
            generation,
            authority,
            page: Mutex::new(None),
            issued: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "source-page-event-probe")]
            probe: Mutex::new(None),
        }
    }
    /// Arm exactly one one-shot diagnostic on this existing actor. The actor is
    /// not replaced and no probe-owned stream/event/allocation is created.
    #[cfg(feature = "source-page-event-probe")]
    pub fn arm_source_page_event_probe(
        &self,
        mode: SourcePageEventProbeMode,
    ) -> Result<Arc<SourcePageEventProbeState>, ServerError> {
        self.require_authority()?;
        if self.issued.load(Ordering::Acquire) != 0 {
            return Err(ServerError::graph_state(
                "HIP source page probe actor already issued",
            ));
        }
        let mut held = self
            .probe
            .lock()
            .map_err(|_| ServerError::graph_state("HIP source page probe poisoned"))?;
        if held.is_some() {
            return Err(ServerError::graph_state(
                "HIP source page probe already armed",
            ));
        }
        let state = Arc::new(SourcePageEventProbeState::new(
            mode,
            self.service,
            self.generation,
            self.actor.service_id(),
            self.set_key,
            self.logical.value,
        ));
        *held = Some(state.clone());
        Ok(state)
    }
    fn require_authority(&self) -> Result<(), ServerError> {
        self.authority.require_usable()?;
        if self.authority.generation() != self.generation {
            return Err(ServerError::graph_state(
                "HIP source page service generation differs",
            ));
        }
        Ok(())
    }
    /// Run the diagnostic native fence on the original serialized actor and
    /// poison its original authority. Transport refusal is returned, never
    /// reported as a successful fence; the caller then retains unknown owners.
    pub fn fence_diagnostic(&self) -> Result<(), ServerError> {
        self.authority.fence_unknown();
        let actor = self.actor.clone();
        let authority = self.authority.clone();
        actor
            .submit_blocking(move |server| {
                server.ctx.native.fence_all();
                authority.fence_unknown();
            })
            .map_err(|error| {
                ServerError::graph_state(format!(
                    "HIP source page diagnostic fence call: {error:?}"
                ))
            })
    }
    /// Bind the exact original page allocation once. Re-binding is checked
    /// against the retained handle; a different allocation or a foreign stream
    /// is refused. Page-allocation and native SET identities are distinct
    /// domains, so their numeric values are never compared.
    pub fn bind_page(&self, page: BufferBinding) -> Result<(), ServerError> {
        self.require_authority()?;
        full_geometry(page.size, page.offset_start, page.offset_end)?;
        if page.service != self.service || page.stream != self.logical {
            return Err(ServerError::graph_state(
                "HIP source page original service/stream differs",
            ));
        }
        let mut held = self
            .page
            .lock()
            .map_err(|_| ServerError::graph_state("HIP source page binding poisoned"))?;
        match held.as_ref() {
            Some(bound) if !same_page(bound, &page) => Err(ServerError::graph_state(
                "HIP source page bound allocation differs",
            )),
            Some(_) => Ok(()),
            None => {
                *held = Some(page);
                Ok(())
            }
        }
    }
    fn page_identity(&self, page: &BufferBinding) -> Result<BufferBinding, ServerError> {
        self.require_authority()?;
        let held = self
            .page
            .lock()
            .map_err(|_| ServerError::graph_state("HIP source page binding poisoned"))?;
        let bound = held
            .as_ref()
            .ok_or_else(|| ServerError::graph_state("HIP source page allocation is not bound"))?;
        if !same_page(bound, page) {
            return Err(ServerError::graph_state(
                "HIP source page allocation identity differs",
            ));
        }
        Ok(bound.clone())
    }
    /// Copy one ordinary window from a retained source owner into the original
    /// page. The declared outputs are checked before the page is overwritten,
    /// so an earlier consumer failure cannot be cleared by this copy.
    pub fn upload<T: AsRef<[u8]> + Send + Sync + 'static>(
        &self,
        page: BufferBinding,
        offset: usize,
        bytes: usize,
        window: Arc<T>,
    ) -> Result<Arc<T>, ServerError> {
        self.bind_page(page.clone())?;
        let page = self.page_identity(&page)?;
        let authority = self.authority.clone();
        let service = self.service;
        let logical = self.logical;
        let set_key = self.set_key;
        let generation = self.generation;
        self.actor
            .submit_blocking(move |server| {
                server.upload_source_page_window(
                    authority, service, logical, set_key, generation, page, offset, bytes, window,
                )
            })
            .map_err(|error| {
                ServerError::graph_state(format!("HIP source page upload call: {error:?}"))
            })?
    }
    /// Record the page write, its consumer handles and the exact owner after
    /// they were queued on the original stream. `page` must be the complete
    /// original allocation; `outputs` are the consumer handles queued behind
    /// this write, checked again before an event can make the run look complete.
    pub fn record(
        &self,
        page: BufferBinding,
        outputs: Vec<BufferBinding>,
        reusable: Option<OwnedHipTransferRetirement>,
    ) -> Result<OwnedHipTransferRetirement, ServerError> {
        self.record_impl(page, outputs, reusable, None, 0, 0)
    }
    /// Diagnostic record carrying the selected actual event lineage and scalar
    /// source-window payload/charge into the same one-shot mark.
    #[cfg(feature = "source-page-event-probe")]
    pub fn record_probed(
        &self,
        page: BufferBinding,
        outputs: Vec<BufferBinding>,
        reusable: Option<OwnedHipTransferRetirement>,
        probe: Arc<SourcePageEventProbeState>,
        payload: u64,
        charge: u64,
    ) -> Result<OwnedHipTransferRetirement, ServerError> {
        self.record_impl(page, outputs, reusable, Some(probe), payload, charge)
    }
    #[allow(clippy::too_many_arguments)]
    fn record_impl(
        &self,
        page: BufferBinding,
        outputs: Vec<BufferBinding>,
        mut reusable: Option<OwnedHipTransferRetirement>,
        probe: Option<Arc<SourcePageEventProbeState>>,
        payload: u64,
        charge: u64,
    ) -> Result<OwnedHipTransferRetirement, ServerError> {
        #[cfg(not(feature = "source-page-event-probe"))]
        let _ = (&probe, payload, charge);
        self.bind_page(page.clone())?;
        let page = self.page_identity(&page)?;
        #[cfg(feature = "source-page-event-probe")]
        if let Some(probe) = &probe {
            if let Err(error) = probe.require_healthy() {
                self.authority.fence_unknown();
                self.fence_diagnostic()?;
                return Err(error);
            }
            let armed = match self.probe.lock() {
                Ok(armed) => armed,
                Err(_) => {
                    self.authority.fence_unknown();
                    self.fence_diagnostic()?;
                    return Err(ServerError::graph_state("HIP source page probe poisoned"));
                }
            };
            if armed
                .as_ref()
                .is_none_or(|state| !Arc::ptr_eq(state, probe))
            {
                drop(armed);
                self.authority.fence_unknown();
                self.fence_diagnostic()?;
                return Err(ServerError::graph_state(
                    "HIP source page probe actor differs",
                ));
            }
        }
        let actor = self.actor.clone();
        let authority = self.authority.clone();
        let service = self.service;
        let logical = self.logical;
        let set_key = self.set_key;
        let generation = self.generation;
        let issued = self.issued.clone();
        // Borrow the reusable mark on THIS stack frame. An early refusal or a
        // recycle failure then drops it here, outside the serialized actor task
        // and after the actor released its mutable service borrow, instead of
        // re-entering `submit_blocking` from inside `act_on`.
        let result = {
            let reusable = reusable.as_mut();
            self.actor.submit_blocking(move |server| {
                server.record_source_page_run(
                    actor, authority, service, logical, set_key, generation, issued, page, outputs,
                    reusable, probe, payload, charge,
                )
            })
        };
        // `reusable` is still alive here: if the record refused before taking
        // the event, its Drop performs the real destroy/fence now.
        result.map_err(|error| {
            self.authority.fence_unknown();
            ServerError::graph_state(format!("HIP source page record call: {error:?}"))
        })?
    }
    pub fn set_key(&self) -> u64 {
        self.set_key
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn service(&self) -> ServiceId {
        self.service
    }
    pub fn logical(&self) -> StreamId {
        self.logical
    }
}
/// One recorded, not yet retired page write plus its consumer handles.
///
/// The event is created on the original actor and destroyed by the last owner
/// of this mark. The exact retained page allocation, every consumer handle and
/// the live generation are authenticated again on every status query, so a
/// retired or foreign handle is a real failure rather than a stale success.
/// Page-allocation identity is never compared numerically with the SET key.
pub struct OwnedHipTransferRetirement {
    actor: DeviceHandle<HipServer>,
    authority: Arc<dyn SourcePageAuthority>,
    service: ServiceId,
    logical: StreamId,
    set_key: u64,
    generation: u64,
    issued: Arc<AtomicU64>,
    event: cubecl_hip_sys::hipEvent_t,
    /// The exact retained page allocation, then the declared outputs in the
    /// exact record order.
    page: BufferBinding,
    outputs: Vec<BufferBinding>,
    serial: u64,
    retired: bool,
    #[cfg(feature = "source-page-event-probe")]
    probe: Option<Arc<SourcePageEventProbeState>>,
}
// SAFETY: the event handle is owned by this mark and only used through the
// original actor's serialized status query/wait/destroy. The bindings are
// plain copies of service-stamped handles.
unsafe impl Send for OwnedHipTransferRetirement {}
unsafe impl Sync for OwnedHipTransferRetirement {}
impl OwnedHipTransferRetirement {
    /// Run the native fence on the original actor and poison the original
    /// authority. Transport refusal is retained, never reported as success.
    fn fence_diagnostic(&self) -> Result<(), ServerError> {
        self.authority.fence_unknown();
        let actor = self.actor.clone();
        let authority = self.authority.clone();
        actor
            .submit_blocking(move |server| {
                server.ctx.native.fence_all();
                authority.fence_unknown();
            })
            .map_err(|error| {
                ServerError::graph_state(format!(
                    "HIP source page diagnostic fence call: {error:?}"
                ))
            })
    }
    fn require_identity(&self) -> Result<(), ServerError> {
        if self.serial == 0
            || self.generation != self.authority.generation()
            || self.set_key == 0
            || self.page.service != self.service
            || self.page.stream != self.logical
            || self.serial > self.issued.load(Ordering::Acquire)
        {
            return Err(ServerError::graph_state(
                "HIP source page mark page/generation identity differs",
            ));
        }
        self.authority.require_usable()
    }
    /// Whether the recorded write and its consumers completed. The original
    /// event query runs on the actor thread, together with the actual page and
    /// output failure claims.
    pub fn ready(&mut self) -> Result<bool, ServerError> {
        self.require_identity()?;
        // A previously retired event still executes its real query and every
        // original authority/handle check. Only the diagnostic retirement
        // counter is suppressed below.
        #[cfg(feature = "source-page-event-probe")]
        let was_retired = self.retired;
        #[cfg(feature = "source-page-event-probe")]
        let probe = self.probe.clone();
        #[cfg(feature = "source-page-event-probe")]
        let event_id = self.event as usize as u64;
        #[cfg(feature = "source-page-event-probe")]
        let serial = self.serial;
        let actor = self.actor.clone();
        let authority = self.authority.clone();
        let service = self.service;
        let logical = self.logical;
        let set_key = self.set_key;
        let generation = self.generation;
        let page = self.page.clone();
        let outputs = self.outputs.clone();
        let event = self.event as usize;
        #[cfg(feature = "source-page-event-probe")]
        let probe_for_actor = probe.clone();
        let submitted = actor.submit_blocking(move |server| {
            server
                .require_source_page_authority(&*authority, generation)
                .map_err(|error| error.to_string())?;
            server
                .require_source_page_handles(service, logical, set_key, &page, outputs.iter())
                .map_err(|error| error.to_string())?;
            if let Err(failure) = server
                .streams
                .ensure_written(std::iter::once(&page).chain(outputs.iter()))
            {
                return Err(failure.to_string());
            }
            #[cfg(feature = "source-page-event-probe")]
            if let Some(probe) = &probe_for_actor {
                if let Err(error) = probe.require_healthy() {
                    server.ctx.native.fence_all();
                    authority.fence_unknown();
                    return Err(error.to_string());
                }
            }
            // Order: real query and raw observation first; genuine driver
            // errors return before injection, holds or retirement.
            let status = unsafe { cubecl_hip_sys::hipEventQuery(event as _) };
            #[cfg(feature = "source-page-event-probe")]
            if let Some(probe) = &probe_for_actor {
                if let Err(error) = probe.observe_query(event_id, serial, status as i32) {
                    server.ctx.native.fence_all();
                    authority.fence_unknown();
                    return Err(error.to_string());
                }
            }
            let legal = status == cubecl_hip_sys::hipError_t_hipSuccess
                || status == cubecl_hip_sys::hipError_t_hipErrorNotReady;
            if !legal {
                return Ok::<_, String>((status, false, false));
            }
            #[cfg(feature = "source-page-event-probe")]
            if let Some(probe) = &probe_for_actor {
                let injected = match probe.inject_query(event_id, serial) {
                    Ok(injected) => injected,
                    Err(error) => {
                        server.ctx.native.fence_all();
                        authority.fence_unknown();
                        return Err(error.to_string());
                    }
                };
                if injected {
                    server.ctx.native.fence_all();
                    authority.fence_unknown();
                    return Err("HIP source page injected query unknown".to_string());
                }
            }
            let held = {
                #[cfg(feature = "source-page-event-probe")]
                if let Some(probe) = &probe_for_actor {
                    match probe.hold_ready(event_id, serial) {
                        Ok(held) => held,
                        Err(error) => {
                            server.ctx.native.fence_all();
                            authority.fence_unknown();
                            return Err(error.to_string());
                        }
                    }
                } else {
                    false
                }
                #[cfg(not(feature = "source-page-event-probe"))]
                {
                    false
                }
            };
            // A held WaitUnknown mark has not retired yet. Count retirement
            // only for a real false->true transition after the hold check.
            let retired = if !held && status == cubecl_hip_sys::hipError_t_hipSuccess {
                #[cfg(feature = "source-page-event-probe")]
                if let Some(probe) = &probe_for_actor {
                    if !was_retired && let Err(error) = probe.note_retired(event_id, serial) {
                        server.ctx.native.fence_all();
                        authority.fence_unknown();
                        return Err(error.to_string());
                    }
                }
                true
            } else {
                false
            };
            Ok::<_, String>((status, held, retired))
        });
        let (status, held, retired) = submitted
            .map_err(|error| {
                self.authority.fence_unknown();
                let _ = self.fence_diagnostic();
                ServerError::graph_state(format!("HIP source page query: {error:?}"))
            })?
            .map_err(ServerError::graph_state)?;
        if held {
            return Ok(false);
        }
        match status {
            cubecl_hip_sys::hipError_t_hipSuccess => {}
            cubecl_hip_sys::hipError_t_hipErrorNotReady => return Ok(false),
            status => {
                return Err(checked("hipEventQuery", status)
                    .err()
                    .map(ServerError::from)
                    .unwrap_or_else(|| ServerError::graph_state("HIP source page query differs")));
            }
        }
        if retired {
            self.retired = true;
            Ok(true)
        } else {
            Err(ServerError::graph_state(
                "HIP source page query success without retirement",
            ))
        }
    }
    /// Wait for the real event on the actor thread, then re-check the handles.
    pub fn wait(&mut self) -> Result<(), ServerError> {
        self.require_identity()?;
        // As with ready, a retired mark still runs the original synchronize,
        // follow-up query and handle checks; only its diagnostic retire count
        // is suppressed.
        #[cfg(feature = "source-page-event-probe")]
        let was_retired = self.retired;
        #[cfg(feature = "source-page-event-probe")]
        let probe = self.probe.clone();
        #[cfg(feature = "source-page-event-probe")]
        let event_id = self.event as usize as u64;
        #[cfg(feature = "source-page-event-probe")]
        let serial = self.serial;
        let actor = self.actor.clone();
        let authority = self.authority.clone();
        let service = self.service;
        let logical = self.logical;
        let set_key = self.set_key;
        let generation = self.generation;
        let page = self.page.clone();
        let outputs = self.outputs.clone();
        let event = self.event as usize;
        #[cfg(feature = "source-page-event-probe")]
        let probe_for_actor = probe.clone();
        let submitted = actor.submit_blocking(move |server| {
            server
                .require_source_page_authority(&*authority, generation)
                .map_err(|error| error.to_string())?;
            server
                .require_source_page_handles(service, logical, set_key, &page, outputs.iter())
                .map_err(|error| error.to_string())?;
            if let Err(failure) = server
                .streams
                .ensure_written(std::iter::once(&page).chain(outputs.iter()))
            {
                return Err(failure.to_string());
            }
            #[cfg(feature = "source-page-event-probe")]
            if let Some(probe) = &probe_for_actor {
                if let Err(error) = probe.require_healthy() {
                    server.ctx.native.fence_all();
                    authority.fence_unknown();
                    return Err(error.to_string());
                }
            }
            let synchronize = unsafe { cubecl_hip_sys::hipEventSynchronize(event as _) };
            #[cfg(feature = "source-page-event-probe")]
            if let Some(probe) = &probe_for_actor {
                if let Err(error) = probe.observe_synchronize(event_id, serial, synchronize as i32)
                {
                    server.ctx.native.fence_all();
                    authority.fence_unknown();
                    return Err(error.to_string());
                }
            }
            if synchronize != cubecl_hip_sys::hipError_t_hipSuccess {
                return Ok::<_, String>((synchronize, None, false));
            }
            #[cfg(feature = "source-page-event-probe")]
            if let Some(probe) = &probe_for_actor {
                let injected = match probe.inject_wait(event_id, serial) {
                    Ok(injected) => injected,
                    Err(error) => {
                        server.ctx.native.fence_all();
                        authority.fence_unknown();
                        return Err(error.to_string());
                    }
                };
                // Injection is immediately after actual synchronize success
                // and before the follow-up query/retirement.
                if injected {
                    server.ctx.native.fence_all();
                    authority.fence_unknown();
                    return Err("HIP source page injected wait unknown".to_string());
                }
            }
            let query = unsafe { cubecl_hip_sys::hipEventQuery(event as _) };
            #[cfg(feature = "source-page-event-probe")]
            if let Some(probe) = &probe_for_actor {
                if let Err(error) = probe.observe_query(event_id, serial, query as i32) {
                    server.ctx.native.fence_all();
                    authority.fence_unknown();
                    return Err(error.to_string());
                }
            }
            let retired = if query == cubecl_hip_sys::hipError_t_hipSuccess {
                #[cfg(feature = "source-page-event-probe")]
                if let Some(probe) = &probe_for_actor {
                    if !was_retired && let Err(error) = probe.note_retired(event_id, serial) {
                        server.ctx.native.fence_all();
                        authority.fence_unknown();
                        return Err(error.to_string());
                    }
                }
                true
            } else {
                false
            };
            Ok::<_, String>((synchronize, Some(query), retired))
        });
        let (synchronize, query, retired) = submitted
            .map_err(|error| {
                self.authority.fence_unknown();
                let _ = self.fence_diagnostic();
                ServerError::graph_state(format!("HIP source page wait: {error:?}"))
            })?
            .map_err(ServerError::graph_state)?;
        if synchronize != cubecl_hip_sys::hipError_t_hipSuccess {
            return Err(checked("hipEventSynchronize", synchronize)
                .err()
                .map(ServerError::from)
                .unwrap_or_else(|| ServerError::graph_state("HIP source page wait differs")));
        }
        let query = query.ok_or_else(|| {
            ServerError::graph_state("HIP source page event query was not called")
        })?;
        if query != cubecl_hip_sys::hipError_t_hipSuccess {
            return Err(checked("hipEventQuery", query)
                .err()
                .map(ServerError::from)
                .unwrap_or_else(|| {
                    ServerError::graph_state("HIP source page wait query differs")
                }));
        }
        if !retired {
            return Err(ServerError::graph_state(
                "HIP source page wait success without retirement",
            ));
        }
        self.retired = true;
        Ok(())
    }
    /// The ring reuses a completed mark's event for the next write. A mark that
    /// was never retired, or whose generation/issue serial/page identity
    /// differs, fails closed instead of recycling a foreign event.
    fn recycle(
        &mut self,
        service: ServiceId,
        logical: StreamId,
        set_key: u64,
        page: &BufferBinding,
        generation: u64,
    ) -> Result<cubecl_hip_sys::hipEvent_t, ServerError> {
        if !self.retired {
            return Err(ServerError::graph_state(
                "HIP source page reusable mark is not retired",
            ));
        }
        self.require_identity()?;
        if self.service != service
            || self.logical != logical
            || self.set_key != set_key
            || self.generation != generation
            || !same_page(&self.page, page)
        {
            return Err(ServerError::graph_state(
                "HIP source page reusable mark page identity differs",
            ));
        }
        self.outputs.clear();
        #[cfg(feature = "source-page-event-probe")]
        if let Some(probe) = &self.probe {
            if let Err(error) = probe.note_recycled(self.event as usize as u64, self.serial) {
                // Inside the original actor: poison authority and let the
                // record wrapper directly fence native after this call returns.
                self.authority.fence_unknown();
                return Err(error);
            }
        }
        let event = self.event;
        self.event = std::ptr::null_mut();
        Ok(event)
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn set_key(&self) -> u64 {
        self.set_key
    }
    pub fn serial(&self) -> u64 {
        self.serial
    }
    #[cfg(feature = "source-page-event-probe")]
    pub fn event_identity(&self) -> u64 {
        self.event as usize as u64
    }
}
impl Drop for OwnedHipTransferRetirement {
    fn drop(&mut self) {
        if self.event.is_null() {
            return;
        }
        #[cfg(feature = "source-page-event-probe")]
        let probe = self.probe.clone();
        #[cfg(not(feature = "source-page-event-probe"))]
        let probe = None::<Arc<SourcePageEventProbeState>>;
        // Identity unknown: never destroy an unproven event. Fence the original
        // physical service and retain the handle; host owners were only released
        // after a proved retirement, so this is event retention, not release.
        if self.require_identity().is_err() {
            self.authority.fence_unknown();
            let _ = self.fence_diagnostic();
            return;
        }
        #[cfg(feature = "source-page-event-probe")]
        if let Some(probe) = &probe {
            if probe.require_healthy().is_err() {
                self.authority.fence_unknown();
                // Keep the event retained if the original actor transport is
                // itself unknown; the helper result is deliberately not a
                // successful-fence claim.
                let _ = self.fence_diagnostic();
                return;
            }
        }
        let retired = self.retired;
        let serial = self.serial;
        let event = std::mem::replace(&mut self.event, std::ptr::null_mut());
        let actor = self.actor.clone();
        let authority = self.authority.clone();
        let service = self.service;
        let logical = self.logical;
        let set_key = self.set_key;
        let generation = self.generation;
        let page = self.page.clone();
        let outputs = self.outputs.clone();
        let event = event as usize;
        let event_id = event as u64;
        // Destroy only through the original serialized actor while the exact
        // live authority/generation and page/output identities are known.
        let destroyed = actor.submit_blocking(move |server| {
            server.destroy_source_page_event(
                &*authority,
                service,
                logical,
                set_key,
                generation,
                &page,
                &outputs,
                event,
                probe,
                event_id,
                serial,
                retired,
            )
        });
        match destroyed {
            Ok(Ok(())) => {}
            // Inner refusal and actor transport failure are both unknown: the
            // event stays retained and the original service is fenced.
            Ok(Err(_)) | Err(_) => {
                self.authority.fence_unknown();
                let _ = self.fence_diagnostic();
            }
        }
    }
}
impl HipServer {
    /// Live authority/generation check executed ON the original actor thread.
    /// Every page upload/record/query/wait/destroy closure calls this, so a
    /// retirement or native-set change between submission and execution is a
    /// real refusal rather than an outer-only check.
    pub(crate) fn require_source_page_authority(
        &mut self,
        authority: &dyn SourcePageAuthority,
        generation: u64,
    ) -> Result<(), ServerError> {
        let result = authority.require_usable().and_then(|()| {
            if authority.generation() != generation {
                Err(ServerError::graph_state(
                    "HIP source page live service generation differs",
                ))
            } else {
                Ok(())
            }
        });
        if result.is_err() {
            self.ctx.native.fence_all();
        }
        result
    }
    /// Resolve the exact original page/consumer handles, without recording any
    /// new work. Every query goes through the actor so it cannot observe a
    /// handle on another service, stream, generation or retired native SET.
    pub(crate) fn require_source_page_handles<'a>(
        &mut self,
        service: ServiceId,
        logical: StreamId,
        set_key: u64,
        page: &BufferBinding,
        outputs: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<(), ServerError> {
        let result = self.source_page_handles(service, logical, set_key, page, outputs);
        if result.is_err() {
            self.ctx.native.fence_all();
        }
        result
    }
    /// Destroy one retired page event on the original actor while its exact
    /// live authority/generation and page/consumer/native identities are still
    /// known. Any refusal leaves the event retained and fences the service.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn destroy_source_page_event(
        &mut self,
        authority: &dyn SourcePageAuthority,
        service: ServiceId,
        logical: StreamId,
        set_key: u64,
        generation: u64,
        page: &BufferBinding,
        outputs: &[BufferBinding],
        event: usize,
        probe: Option<Arc<SourcePageEventProbeState>>,
        event_id: u64,
        serial: u64,
        retired: bool,
    ) -> Result<(), ServerError> {
        #[cfg(not(feature = "source-page-event-probe"))]
        let _ = (&probe, event_id, serial);
        let result = (|| {
            self.require_source_page_authority(authority, generation)?;
            self.require_source_page_handles(service, logical, set_key, page, outputs.iter())?;
            if !retired {
                self.ctx.native.fence_all();
                return Err(ServerError::graph_state(
                    "HIP source page event destroy before retirement",
                ));
            }
            #[cfg(feature = "source-page-event-probe")]
            if let Some(probe) = &probe {
                probe.require_healthy()?;
                if probe.inject_destroy(event_id, serial)? {
                    // Actual native fence inside the serialized destroy, after
                    // authority/handles/retired and before hipEventDestroy.
                    self.ctx.native.fence_all();
                    authority.fence_unknown();
                    return Err(ServerError::graph_state(
                        "HIP source page injected destroy refused",
                    ));
                }
                probe.note_destroy_call(event_id, serial)?;
            }
            let status = unsafe { cubecl_hip_sys::hipEventDestroy(event as _) };
            #[cfg(feature = "source-page-event-probe")]
            if let Some(probe) = &probe {
                probe.note_destroy_status(event_id, serial, status as i32)?;
            }
            checked("hipEventDestroy", status)?;
            Ok(())
        })();
        if result.is_err() {
            self.ctx.native.fence_all();
            authority.fence_unknown();
        }
        result
    }
    fn source_page_handles<'a>(
        &mut self,
        service: ServiceId,
        logical: StreamId,
        set_key: u64,
        page: &BufferBinding,
        outputs: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<(), ServerError> {
        if service != self.utilities.service {
            return Err(ServerError::graph_state(
                "HIP source page actor/service differs",
            ));
        }
        // The SET key is native-set authority; page-allocation identity is a
        // separate domain and is never compared numerically with it.
        if set_key == 0 {
            return Err(ServerError::graph_state(
                "HIP source page native SET key absent",
            ));
        }
        self.require_native(set_key)?;
        if page.service != self.utilities.service || page.stream != logical {
            return Err(ServerError::graph_state(
                "HIP source page original service/stream differs",
            ));
        }
        full_geometry(page.size, page.offset_start, page.offset_end)?;
        let page_id = page.memory.id();
        let actor_service = self.utilities.service;
        let mut command = self.command_no_inputs(logical)?;
        command.resource(page.clone())?.require_known_retirement()?;
        for output in outputs {
            let start = output.offset_start.unwrap_or(0);
            let end = output.offset_end.unwrap_or(0);
            if output.service != actor_service
                || output.stream != logical
                || output.size == 0
                || start.checked_add(end).is_none_or(|used| used > output.size)
            {
                return Err(ServerError::graph_state(
                    "HIP source page consumer output is foreign/invalid",
                ));
            }
            if output.memory.id() == page_id {
                return Err(ServerError::graph_state(
                    "HIP source page consumer output aliases the page allocation",
                ));
            }
            command
                .resource(output.clone())?
                .require_known_retirement()?;
        }
        Ok(())
    }
    pub(crate) fn upload_source_page_window<T: AsRef<[u8]> + Send + Sync + 'static>(
        &mut self,
        authority: Arc<dyn SourcePageAuthority>,
        service: ServiceId,
        logical: StreamId,
        set_key: u64,
        generation: u64,
        page: BufferBinding,
        offset: usize,
        bytes: usize,
        window: Arc<T>,
    ) -> Result<Arc<T>, ServerError> {
        let result = self.upload_source_page_window_inner(
            &*authority,
            service,
            logical,
            set_key,
            generation,
            page,
            offset,
            bytes,
            window,
        );
        if result.is_err() {
            self.ctx.native.fence_all();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn upload_source_page_window_inner<T: AsRef<[u8]> + Send + Sync + 'static>(
        &mut self,
        authority: &dyn SourcePageAuthority,
        service: ServiceId,
        logical: StreamId,
        set_key: u64,
        generation: u64,
        page: BufferBinding,
        offset: usize,
        bytes: usize,
        window: Arc<T>,
    ) -> Result<Arc<T>, ServerError> {
        self.require_source_page_authority(authority, generation)?;
        full_geometry(page.size, page.offset_start, page.offset_end)?;
        let host = window.as_ref().as_ref();
        if host.len() != bytes {
            return Err(ServerError::graph_state(
                "HIP source page host extent changed",
            ));
        }
        let (offset, bytes) = (offset as u64, bytes as u64);
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|end| end > page.size) {
            return Err(ServerError::graph_state(
                "HIP source page window range exceeds original allocation",
            ));
        }
        // Check the actual page claim BEFORE the copy overwrites the page. A
        // later overwrite must not clear an earlier write failure.
        self.require_source_page_handles(
            service,
            logical,
            set_key,
            &page,
            std::iter::empty::<&BufferBinding>(),
        )?;
        if let Err(failure) = self.streams.ensure_written(std::iter::once(&page)) {
            return Err(ServerError::graph_state(failure.to_string()));
        }
        let view = BufferBinding {
            offset_start: Some(offset),
            offset_end: Some(page.size - offset - bytes),
            ..page.clone()
        };
        let executed = ExecuteScope::over(self, logical, vec![page.clone()]).execute(|server| {
            let mut command = server.command_no_inputs(logical)?;
            if command.stream().capturing.is_recording() {
                return Err(ServerError::graph_state(
                    "HIP source page graph capture unsupported",
                ));
            }
            let original = command.resource(page.clone())?;
            let resource = command.resource(view)?;
            original.require_known_retirement()?;
            resource.require_known_retirement()?;
            let (base, target) = (original.ptr as u64, resource.ptr as u64);
            if base == 0
                || resource.size != bytes
                || base.checked_add(offset) != Some(target)
                || target.checked_add(bytes).is_none()
            {
                return Err(ServerError::graph_state(
                    "HIP source page native allocation/view extent",
                ));
            }
            checked("hipMemcpyHtoDAsync", unsafe {
                cubecl_hip_sys::hipMemcpyHtoDAsync(
                    resource.ptr,
                    host.as_ptr() as *mut core::ffi::c_void,
                    bytes as usize,
                    command.stream().sys,
                )
            })?;
            Ok(())
        });
        executed.into_result()?;
        Ok(window)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_source_page_run(
        &mut self,
        actor: DeviceHandle<HipServer>,
        authority: Arc<dyn SourcePageAuthority>,
        service: ServiceId,
        logical: StreamId,
        set_key: u64,
        generation: u64,
        issued: Arc<AtomicU64>,
        page: BufferBinding,
        outputs: Vec<BufferBinding>,
        reusable: Option<&mut OwnedHipTransferRetirement>,
        probe: Option<Arc<SourcePageEventProbeState>>,
        payload: u64,
        charge: u64,
    ) -> Result<OwnedHipTransferRetirement, ServerError> {
        let result = self.record_source_page_run_inner(
            actor, authority, service, logical, set_key, generation, issued, page, outputs,
            reusable, probe, payload, charge,
        );
        if result.is_err() {
            self.ctx.native.fence_all();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn record_source_page_run_inner(
        &mut self,
        actor: DeviceHandle<HipServer>,
        authority: Arc<dyn SourcePageAuthority>,
        service: ServiceId,
        logical: StreamId,
        set_key: u64,
        generation: u64,
        issued: Arc<AtomicU64>,
        page: BufferBinding,
        outputs: Vec<BufferBinding>,
        reusable: Option<&mut OwnedHipTransferRetirement>,
        probe: Option<Arc<SourcePageEventProbeState>>,
        payload: u64,
        charge: u64,
    ) -> Result<OwnedHipTransferRetirement, ServerError> {
        #[cfg(not(feature = "source-page-event-probe"))]
        let _ = (&probe, payload, charge);
        #[cfg(feature = "source-page-event-probe")]
        if let Some(probe) = &probe {
            if let Err(error) = probe.require_healthy() {
                self.ctx.native.fence_all();
                authority.fence_unknown();
                return Err(error);
            }
        }
        full_geometry(page.size, page.offset_start, page.offset_end)?;
        self.require_source_page_authority(&*authority, generation)?;
        let page_id = page.memory.id();
        for (index, output) in outputs.iter().enumerate() {
            let start = output.offset_start.unwrap_or(0);
            let end = output.offset_end.unwrap_or(0);
            if output.service != self.utilities.service
                || output.stream != logical
                || output.size == 0
                || start.checked_add(end).is_none_or(|used| used > output.size)
            {
                return Err(ServerError::graph_state(
                    "HIP source page consumer output is foreign/invalid",
                ));
            }
            if output.memory.id() == page_id
                || outputs[..index]
                    .iter()
                    .any(|prior| prior.memory.id() == output.memory.id())
            {
                return Err(ServerError::graph_state(
                    "HIP source page output aliases page or sibling",
                ));
            }
        }
        self.require_source_page_handles(service, logical, set_key, &page, outputs.iter())?;
        // The page and declared outputs are checked again AT record, after the
        // consumers were queued and before any later entry can overwrite the
        // page. A failed page write or consumer is a real refusal here.
        if let Err(failure) = self
            .streams
            .ensure_written(std::iter::once(&page).chain(outputs.iter()))
        {
            return Err(ServerError::graph_state(failure.to_string()));
        }
        let mut command = self.command_no_inputs(logical)?;
        command.resource(page.clone())?.require_known_retirement()?;
        for output in &outputs {
            command
                .resource(output.clone())?
                .require_known_retirement()?;
        }
        let serial = issued
            .fetch_add(1, Ordering::AcqRel)
            .checked_add(1)
            .ok_or_else(|| ServerError::graph_state("HIP source page issue serial overflow"))?;
        let event = match reusable {
            Some(mark) => match mark.recycle(service, logical, set_key, &page, generation) {
                Ok(event) => event,
                Err(error) => {
                    authority.fence_unknown();
                    return Err(error);
                }
            },
            None => {
                let mut event = std::ptr::null_mut();
                if let Err(error) = checked("hipEventCreateWithFlags", unsafe {
                    cubecl_hip_sys::hipEventCreateWithFlags(
                        &mut event,
                        cubecl_hip_sys::hipEventDefault,
                    )
                }) {
                    authority.fence_unknown();
                    return Err(error.into());
                }
                event
            }
        };
        if let Err(error) = checked("hipEventRecord", unsafe {
            cubecl_hip_sys::hipEventRecord(event, command.stream().sys)
        }) {
            // Unknown record: never destroy an event whose record may already be
            // observed. The wrapper fences the original service, the product
            // keeps page/output/window owners, and this event stays retained.
            authority.fence_unknown();
            return Err(error.into());
        }
        #[cfg(feature = "source-page-event-probe")]
        if let Some(probe) = &probe {
            probe
                .note_record(event as usize as u64, serial, payload, charge)
                .map_err(|error| {
                    authority.fence_unknown();
                    error
                })?;
        }
        Ok(OwnedHipTransferRetirement {
            actor,
            authority,
            service,
            logical,
            set_key,
            generation,
            issued,
            event,
            page,
            outputs,
            serial,
            retired: false,
            #[cfg(feature = "source-page-event-probe")]
            probe,
        })
    }
}

/// Exact original allocation geometry: size is the whole page, and the binding
/// has no window offsets of its own (the window is supplied separately).
fn full_geometry(size: u64, start: Option<u64>, end: Option<u64>) -> Result<(), ServerError> {
    if size == 0 || start.unwrap_or(0) != 0 || end.unwrap_or(0) != 0 {
        return Err(ServerError::graph_state(
            "HIP source page is not one complete original allocation",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_geometry_requires_one_complete_original_allocation() {
        assert!(full_geometry(1024 * 1024, None, None).is_ok());
        assert!(full_geometry(0, None, None).is_err());
        assert!(full_geometry(4, Some(1), None).is_err());
        assert!(full_geometry(4, None, Some(1)).is_err());
    }
}
