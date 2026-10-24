//! Diagnostic-only driver resolver interposition; one original-queue pool/process.
//! Successful requested bytes remain charged until trusted final retirement.
//! Driver/module metadata stays in the original coarse ProviderEnvelope; these
//! counters do not report allocator granularity, driver-internal or host RAM.
use crate::OwnedCudaQueue;
mod context;
mod elf;
mod events;
mod export_queries;
use export_queries::ExportQueries;
mod typed;
use cubecl_core::server::ServerError;
use cudarc::driver::sys::{
    CUdeviceptr, CUdriverProcAddressQueryResult, CUmemoryPool, CUresult, CUstream,
};
use events::{EventKind, ResolverEvents};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::io::Read;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

const SLOTS: usize = 128;
const SUCCESS: CUresult = CUresult::CUDA_SUCCESS;
const REFUSED: CUresult = CUresult::CUDA_ERROR_OUT_OF_MEMORY;
const UNSUPPORTED: CUresult = CUresult::CUDA_ERROR_NOT_SUPPORTED;
const INVALID: CUresult = CUresult::CUDA_ERROR_INVALID_VALUE;
type Dlsym = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
type GetProc = unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, u64) -> CUresult;
type GetProcV2 = unsafe extern "C" fn(
    *const c_char,
    *mut *mut c_void,
    c_int,
    u64,
    *mut CUdriverProcAddressQueryResult,
) -> CUresult;
type Dlvsym = unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> *mut c_void;

static DRIVER_LIBRARY: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static REAL_DLVSYM: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static REAL_DLSYM: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static GET_PROC: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static GET_PROC_V2: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static ALLOC: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static FREE: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static ALLOC_ASYNC: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static ALLOC_ASYNC_PT: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static ALLOC_POOL: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static ALLOC_POOL_PT: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static FREE_ASYNC: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static FREE_ASYNC_PT: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

#[derive(Clone, Copy)]
struct Entry {
    pointer: u64,
    bytes: u64,
    state: u8,
}
const EMPTY: Entry = Entry {
    pointer: 0,
    bytes: 0,
    state: 0,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Cold,
    Active,
    Poisoned,
    Retired,
}
struct State {
    phase: Phase,
    cap: u64,
    charged: u64,
    live: u64,
    attempts: u64,
    refunds: u64,
    refusals: u64,
    unsupported: u64,
    resolver_calls: u64,
    redirected: u64,
    provider_live: bool,
    first_unsupported: [u8; 64],
    denied_lookups: u64,
    events: ResolverEvents,
    export_queries: ExportQueries,
    original_device: i32,
    original_context: usize,
    primary_retains: u64,
    context_inflight: u64,
    entries: [Entry; SLOTS],
    admission_owner: Option<Arc<dyn Send + Sync>>,
    queue_owner: Option<Arc<OwnedCudaQueue>>,
}
static STATE: Mutex<State> = Mutex::new(State {
    phase: Phase::Cold,
    cap: 0,
    charged: 0,
    live: 0,
    attempts: 0,
    refunds: 0,
    refusals: 0,
    unsupported: 0,
    resolver_calls: 0,
    redirected: 0,
    provider_live: false,
    first_unsupported: [0; 64],
    denied_lookups: 0,
    events: ResolverEvents::new(),
    export_queries: ExportQueries::new(),
    original_device: -1,
    original_context: 0,
    primary_retains: 0,
    context_inflight: 0,
    entries: [EMPTY; SLOTS],
    admission_owner: None,
    queue_owner: None,
});

#[derive(Clone, Copy, Debug)]
pub struct BlasAllocationSnapshot {
    pub limit_bytes: u64,
    pub charged_bytes: u64,
    pub live_bytes: u64,
    pub allocation_attempts: u64,
    pub refunded_bytes: u64,
    pub refused_calls: u64,
    pub unsupported_calls: u64,
    pub resolver_calls: u64,
    pub redirected_calls: u64,
    pub first_unsupported: [u8; 64],
    pub denied_lookups: u64,
    pub resolver_events: ResolverEvents,
    pub resolver_events_overflow: bool,
    pub export_table_queries: ExportQueries,
    pub export_table_queries_overflow: bool,
    pub primary_retains: u64,
    pub context_calls_inflight: u64,
    pub live_allocations: usize,
    pub inflight_allocations: usize,
    pub poisoned: bool,
    pub retired: bool,
    pub provider_live: bool,
}
fn snapshot(state: &State) -> BlasAllocationSnapshot {
    BlasAllocationSnapshot {
        limit_bytes: state.cap,
        charged_bytes: state.charged,
        live_bytes: state.live,
        allocation_attempts: state.attempts,
        refunded_bytes: state.refunds,
        refused_calls: state.refusals,
        unsupported_calls: state.unsupported,
        resolver_calls: state.resolver_calls,
        redirected_calls: state.redirected,
        first_unsupported: state.first_unsupported,
        denied_lookups: state.denied_lookups,
        resolver_events: state.events,
        resolver_events_overflow: state.events.overflow,
        export_table_queries: state.export_queries,
        export_table_queries_overflow: state.export_queries.overflow,
        primary_retains: state.primary_retains,
        context_calls_inflight: state.context_inflight,
        live_allocations: state
            .entries
            .iter()
            .filter(|entry| entry.state == 2)
            .count(),
        inflight_allocations: state
            .entries
            .iter()
            .filter(|entry| entry.state == 1)
            .count(),
        poisoned: state.phase == Phase::Poisoned,
        retired: state.phase == Phase::Retired,
        provider_live: state.provider_live,
    }
}
fn error(message: impl Into<String>) -> ServerError {
    ServerError::graph_state(message)
}

/// Retain this token AND the original non-Handle ledger charge through original
/// CompletionOwner/quarantine. Dropping it never releases/rewinds the pool.
pub struct OwnedBlasAllocationPool {
    queue: Arc<OwnedCudaQueue>,
}
impl OwnedBlasAllocationPool {
    pub fn snapshot(&self) -> Result<BlasAllocationSnapshot, ServerError> {
        STATE
            .lock()
            .map(|state| snapshot(&state))
            .map_err(|_| error("BLAS pool lock poisoned"))
    }
    pub(crate) fn require_queue(&self, queue: &Arc<OwnedCudaQueue>) -> Result<(), ServerError> {
        if !Arc::ptr_eq(&self.queue, queue) {
            return Err(error(
                "BLAS allocation pool belongs to another original queue",
            ));
        }
        let state = STATE.lock().map_err(|_| error("BLAS pool lock poisoned"))?;
        if state.phase != Phase::Active {
            return Err(error("BLAS allocation pool is not active"));
        }
        Ok(())
    }
    pub(crate) fn begin_provider(&self) -> Result<(), ServerError> {
        let mut state = STATE.lock().map_err(|_| error("BLAS pool lock poisoned"))?;
        if state.phase != Phase::Active || state.provider_live {
            return Err(error("BLAS pool provider already present/inactive"));
        }
        state.provider_live = true;
        Ok(())
    }
    pub(crate) fn finish_provider(&self) -> Result<(), ServerError> {
        let mut state = STATE.lock().map_err(|_| error("BLAS pool lock poisoned"))?;
        if state.phase != Phase::Active || !state.provider_live {
            return Err(error("BLAS pool provider lifecycle invalid"));
        }
        state.provider_live = false;
        Ok(())
    }
    /// # Safety
    /// All provider handles/libraries have been destroyed/unloaded on the
    /// original actor; actual provider destruction plus original trusted
    /// completion has finished. No later CUDA allocation can use this process.
    /// This returns metadata only, never physical Ready or StateRoot authority.
    pub unsafe fn retire_after_physical_completion(
        &self,
    ) -> Result<BlasAllocationSnapshot, ServerError> {
        let mut state = STATE.lock().map_err(|_| error("BLAS pool lock poisoned"))?;
        if state.phase != Phase::Active
            || state.provider_live
            || state.primary_retains != 0
            || state.context_inflight != 0
            || state
                .entries
                .iter()
                .any(|entry| matches!(entry.state, 1 | 2))
        {
            return Err(error(
                "BLAS pool retains unknown/live/inflight provider allocations",
            ));
        }
        state.phase = Phase::Retired;
        state.charged = 0;
        state.entries.fill(EMPTY);
        let result = snapshot(&state);
        let admission_owner = state.admission_owner.take();
        let queue_owner = state.queue_owner.take();
        drop(state);
        drop((admission_owner, queue_owner));
        Ok(result)
    }
}
impl Drop for OwnedBlasAllocationPool {
    fn drop(&mut self) {
        if let Ok(mut state) = STATE.lock() {
            if state.phase == Phase::Active {
                state.phase = Phase::Poisoned;
            }
        }
    }
}
impl OwnedCudaQueue {
    /// Activate BEFORE the first pinned BLAS dlopen, after all ordinary buffers
    /// are prepared. Supported driver allocations from any thread are counted.
    /// Async/pool/other opaque allocators are refused while the pool is active.
    ///
    /// # Safety
    /// Caller already holds `device_bytes` as a NON-Handle reservation inside
    /// this original ResidentService ledger/envelope. No other device/provider
    /// or uninstrumented allocator runs concurrently. Retain that exact charge
    /// and this token until trusted retirement or permanent original quarantine.
    pub unsafe fn activate_blas_allocation_pool(
        self: &Arc<Self>,
        device_bytes: u64,
        admission_owner: Arc<dyn Send + Sync>,
    ) -> Result<Arc<OwnedBlasAllocationPool>, ServerError> {
        if device_bytes == 0 || device_bytes > 16 * 1024 * 1024 * 1024 {
            return Err(error("BLAS explicit bounded prepaid pool required"));
        }
        let mut maps = String::new();
        std::fs::File::open("/proc/self/maps")
            .map_err(|e| error(format!("BLAS pre-load maps: {e}")))?
            .take(4 * 1024 * 1024 + 1)
            .read_to_string(&mut maps)
            .map_err(|e| error(format!("BLAS pre-load maps: {e}")))?;
        if maps.len() > 4 * 1024 * 1024 || maps.contains("libcublas") {
            return Err(error("BLAS pool must precede all BLAS library loading"));
        }
        // Exact loaded-glibc ELF bootstrap; no recursive resolver or loader fallback.
        let bound = unsafe { real_dlsym(ptr::null_mut(), c"dlsym".as_ptr()) };
        let bound_versioned = unsafe { real_dlsym(ptr::null_mut(), c"dlvsym".as_ptr()) };
        if bound != owned_dlsym as *const () as *mut c_void
            || bound_versioned != owned_dlvsym as *const () as *mut c_void
        {
            return Err(error(
                "diagnostic executable must export owned dlsym AND dlvsym interposers",
            ));
        }
        self.actor
            .submit_blocking(|server| server.bind_owned_blas_pool())
            .map_err(|e| error(format!("original BLAS context binding actor: {e:?}")))??;
        let driver = DRIVER_LIBRARY.load(Ordering::Acquire);
        if driver.is_null() {
            return Err(error("original CUDA driver resolver handle absent"));
        }
        unsafe {
            typed::seed(driver);
        }
        let mut state = STATE.lock().map_err(|_| error("BLAS pool lock poisoned"))?;
        if state.phase != Phase::Cold {
            return Err(error(
                "one BLAS allocation activation per diagnostic process",
            ));
        }
        state.cap = device_bytes;
        state.admission_owner = Some(admission_owner);
        state.queue_owner = Some(self.clone());
        state.phase = Phase::Active;
        Ok(Arc::new(OwnedBlasAllocationPool {
            queue: self.clone(),
        }))
    }
}

unsafe fn real_dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
    let mut pointer = REAL_DLSYM.load(Ordering::Acquire);
    if pointer.is_null() {
        pointer = elf::glibc_symbol(b"dlsym");
        if pointer.is_null() {
            return ptr::null_mut();
        }
        let _ = REAL_DLSYM.compare_exchange(
            ptr::null_mut(),
            pointer,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        pointer = REAL_DLSYM.load(Ordering::Acquire);
    }
    let function: Dlsym = unsafe { std::mem::transmute(pointer) };
    unsafe { function(handle, name) }
}
fn record(slot: &AtomicPtr<c_void>, actual: *mut c_void, wrapper: *mut c_void) -> *mut c_void {
    if actual.is_null() {
        return actual;
    }
    match slot.compare_exchange(ptr::null_mut(), actual, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => wrapper,
        Err(prior) if prior == actual => wrapper,
        Err(_) => {
            if let Ok(mut state) = STATE.lock() {
                state.phase = Phase::Poisoned;
            }
            ptr::null_mut()
        }
    }
}
fn rewrite(name: &[u8], actual: *mut c_void, version: c_int, flags: u64) -> *mut c_void {
    if let Ok(mut state) = STATE.lock() {
        state.resolver_calls = state.resolver_calls.saturating_add(1);
    }
    let threaded = flags & 2 != 0;
    if flags > 2 {
        return ptr::null_mut();
    }
    if let Some(result) = context::rewrite(name, actual, version)
        .or_else(|| typed::rewrite(name, actual, version, flags))
    {
        if let Ok(mut state) = STATE.lock() {
            if state.phase != Phase::Cold {
                if result.is_null() {
                    state.denied_lookups = state.denied_lookups.saturating_add(1);
                    note_name(&mut state, name);
                    state.events.record(name, EventKind::NullLookup);
                } else {
                    state.events.record(name, EventKind::StubLookup);
                }
            }
            if !result.is_null() {
                state.redirected = state.redirected.saturating_add(1);
            }
            if state.events.overflow {
                state.phase = Phase::Poisoned;
                return ptr::null_mut();
            }
        }
        return result;
    }
    let result = match name {
        b"dlsym" if version == 0 && !actual.is_null() => owned_dlsym as *const () as *mut c_void,
        b"dlvsym" if version == 0 && !actual.is_null() => owned_dlvsym as *const () as *mut c_void,
        b"cuGetProcAddress" if version >= 12000 => record(
            &GET_PROC_V2,
            actual,
            owned_get_proc_v2 as *const () as *mut c_void,
        ),
        b"cuGetProcAddress" => record(
            &GET_PROC,
            actual,
            owned_get_proc as *const () as *mut c_void,
        ),
        b"cuGetProcAddress_v2" if version == 0 || version >= 12000 => record(
            &GET_PROC_V2,
            actual,
            owned_get_proc_v2 as *const () as *mut c_void,
        ),
        b"cuMemAlloc_v2" => record(&ALLOC, actual, alloc as *const () as *mut c_void),
        b"cuMemAlloc" if version >= 3020 => {
            record(&ALLOC, actual, alloc as *const () as *mut c_void)
        }
        b"cuMemFree_v2" => record(&FREE, actual, free as *const () as *mut c_void),
        b"cuMemFree" if version >= 3020 => record(&FREE, actual, free as *const () as *mut c_void),
        b"cuMemAllocAsync_ptsz" => record(
            &ALLOC_ASYNC_PT,
            actual,
            alloc_async_pt as *const () as *mut c_void,
        ),
        b"cuMemAllocAsync" if threaded => record(
            &ALLOC_ASYNC_PT,
            actual,
            alloc_async_pt as *const () as *mut c_void,
        ),
        b"cuMemAllocAsync" => record(
            &ALLOC_ASYNC,
            actual,
            alloc_async as *const () as *mut c_void,
        ),
        b"cuMemAllocFromPoolAsync_ptsz" => record(
            &ALLOC_POOL_PT,
            actual,
            alloc_pool_pt as *const () as *mut c_void,
        ),
        b"cuMemAllocFromPoolAsync" if threaded => record(
            &ALLOC_POOL_PT,
            actual,
            alloc_pool_pt as *const () as *mut c_void,
        ),
        b"cuMemAllocFromPoolAsync" => {
            record(&ALLOC_POOL, actual, alloc_pool as *const () as *mut c_void)
        }
        b"cuMemFreeAsync_ptsz" => record(
            &FREE_ASYNC_PT,
            actual,
            free_async_pt as *const () as *mut c_void,
        ),
        b"cuMemFreeAsync" if threaded => record(
            &FREE_ASYNC_PT,
            actual,
            free_async_pt as *const () as *mut c_void,
        ),
        b"cuMemFreeAsync" => record(&FREE_ASYNC, actual, free_async as *const () as *mut c_void),
        b"cuGetProcAddress_v2" => ptr::null_mut(),
        _ => {
            // No unknown allocating/mapping/importing driver route may bypass
            // accounting. Cold original runtime use remains unchanged.
            let opaque = name.starts_with(b"cuMemAlloc")
                || name.starts_with(b"cuMemHost")
                || name.starts_with(b"cuMemCreate")
                || name.starts_with(b"cuMemMap")
                || name.starts_with(b"cuMemAddressReserve")
                || name.starts_with(b"cuMemImport")
                || name.starts_with(b"cuMemPool")
                || name.starts_with(b"cuArrayCreate")
                || name.starts_with(b"cuArray3DCreate")
                || name.starts_with(b"cuMipmappedArrayCreate")
                || name.starts_with(b"cuImportExternalMemory")
                || name.starts_with(b"cuExternalMemoryGetMapped")
                || name.starts_with(b"cuMulticastCreate")
                || name.starts_with(b"cuGraphAddMemAllocNode")
                || name.starts_with(b"cuGraphInstantiate")
                || name.starts_with(b"cuCtxCreate")
                || name.starts_with(b"cuDevicePrimaryCtxRetain");
            if opaque {
                if let Ok(mut state) = STATE.lock() {
                    if state.phase != Phase::Cold {
                        state.denied_lookups = state.denied_lookups.saturating_add(1);
                        note_name(&mut state, name);
                        state.events.record(name, EventKind::NullLookup);
                        if state.events.overflow {
                            state.phase = Phase::Poisoned;
                        }
                        return ptr::null_mut();
                    }
                } else {
                    return ptr::null_mut();
                }
            }
            actual
        }
    };
    if result != actual && !result.is_null() {
        if let Ok(mut state) = STATE.lock() {
            state.redirected = state.redirected.saturating_add(1);
            if state.phase != Phase::Cold {
                state.events.record(name, EventKind::StubLookup);
            }
            if state.events.overflow {
                state.phase = Phase::Poisoned;
                return ptr::null_mut();
            }
        }
    }
    result
}

/// ROOT opts in with the feature and exports both dlsym/dlvsym in its executable.
#[unsafe(export_name = "dlsym")]
pub unsafe extern "C" fn owned_dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
    if name.is_null() {
        return ptr::null_mut();
    }
    let actual = unsafe { real_dlsym(handle, name) };
    capture_driver(handle, unsafe { CStr::from_ptr(name) }.to_bytes(), actual);
    rewrite(unsafe { CStr::from_ptr(name) }.to_bytes(), actual, 0, 0)
}
/// Versioned resolver imports must pass the same allocation gate.
#[unsafe(export_name = "dlvsym")]
pub unsafe extern "C" fn owned_dlvsym(
    handle: *mut c_void,
    name: *const c_char,
    version: *const c_char,
) -> *mut c_void {
    if name.is_null() || version.is_null() {
        return ptr::null_mut();
    }
    let mut pointer = REAL_DLVSYM.load(Ordering::Acquire);
    if pointer.is_null() {
        pointer = elf::glibc_symbol(b"dlvsym");
        if pointer.is_null() {
            return ptr::null_mut();
        }
        let _ = REAL_DLVSYM.compare_exchange(
            ptr::null_mut(),
            pointer,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        pointer = REAL_DLVSYM.load(Ordering::Acquire);
    }
    let function: Dlvsym = unsafe { std::mem::transmute(pointer) };
    let actual = unsafe { function(handle, name, version) };
    capture_driver(handle, unsafe { CStr::from_ptr(name) }.to_bytes(), actual);
    rewrite(unsafe { CStr::from_ptr(name) }.to_bytes(), actual, 0, 0)
}
unsafe extern "C" fn owned_get_proc(
    name: *const c_char,
    output: *mut *mut c_void,
    version: c_int,
    flags: u64,
) -> CUresult {
    if name.is_null() || output.is_null() {
        return INVALID;
    }
    let pointer = GET_PROC.load(Ordering::Acquire);
    if pointer.is_null() {
        return UNSUPPORTED;
    }
    let function: GetProc = unsafe { std::mem::transmute(pointer) };
    let status = unsafe { function(name, output, version, flags) };
    if status != SUCCESS {
        return status;
    }
    let value = rewrite(
        unsafe { CStr::from_ptr(name) }.to_bytes(),
        unsafe { *output },
        version,
        flags,
    );
    unsafe {
        *output = value;
    }
    if value.is_null() {
        UNSUPPORTED
    } else {
        SUCCESS
    }
}
unsafe extern "C" fn owned_get_proc_v2(
    name: *const c_char,
    output: *mut *mut c_void,
    version: c_int,
    flags: u64,
    query: *mut CUdriverProcAddressQueryResult,
) -> CUresult {
    if name.is_null() || output.is_null() {
        return INVALID;
    }
    let pointer = GET_PROC_V2.load(Ordering::Acquire);
    if pointer.is_null() {
        return UNSUPPORTED;
    }
    let function: GetProcV2 = unsafe { std::mem::transmute(pointer) };
    let status = unsafe { function(name, output, version, flags, query) };
    if status != SUCCESS {
        return status;
    }
    let value = rewrite(
        unsafe { CStr::from_ptr(name) }.to_bytes(),
        unsafe { *output },
        version,
        flags,
    );
    unsafe {
        *output = value;
    }
    if value.is_null() {
        if !query.is_null() {
            unsafe {
                *query = CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            }
        }
        UNSUPPORTED
    } else {
        SUCCESS
    }
}

enum Reservation {
    Cold,
    Active(usize),
    Refused,
}
fn reserve(bytes: usize) -> Reservation {
    let Ok(mut state) = STATE.lock() else {
        return Reservation::Refused;
    };
    reserve_in(&mut state, bytes)
}
fn reserve_in(state: &mut State, bytes: usize) -> Reservation {
    if state.phase == Phase::Cold {
        return Reservation::Cold;
    }
    state.attempts = state.attempts.saturating_add(1);
    if state.phase != Phase::Active
        || bytes == 0
        || state
            .charged
            .checked_add(bytes as u64)
            .is_none_or(|sum| sum > state.cap)
    {
        state.refusals = state.refusals.saturating_add(1);
        return Reservation::Refused;
    }
    let Some(index) = state.entries.iter().position(|entry| entry.state == 0) else {
        state.refusals = state.refusals.saturating_add(1);
        return Reservation::Refused;
    };
    state.entries[index] = Entry {
        pointer: 0,
        bytes: bytes as u64,
        state: 1,
    };
    state.charged += bytes as u64;
    Reservation::Active(index)
}
fn finish(index: usize, pointer: u64, status: CUresult) -> CUresult {
    let Ok(mut state) = STATE.lock() else {
        return CUresult::CUDA_ERROR_UNKNOWN;
    };
    finish_in(&mut state, index, pointer, status)
}
fn finish_in(state: &mut State, index: usize, pointer: u64, status: CUresult) -> CUresult {
    let bytes = state.entries[index].bytes;
    if status != SUCCESS {
        state.charged -= bytes;
        state.refunds = state.refunds.saturating_add(bytes);
        state.entries[index] = EMPTY;
        return status;
    }
    if pointer == 0
        || state
            .entries
            .iter()
            .any(|entry| entry.state == 2 && entry.pointer == pointer)
    {
        state.phase = Phase::Poisoned;
        return CUresult::CUDA_ERROR_UNKNOWN;
    }
    state.entries[index].pointer = pointer;
    state.entries[index].state = 2;
    state.live += bytes;
    status
}
unsafe extern "C" fn alloc(output: *mut CUdeviceptr, bytes: usize) -> CUresult {
    if output.is_null() {
        return INVALID;
    }
    let pointer = ALLOC.load(Ordering::Acquire);
    if pointer.is_null() {
        return UNSUPPORTED;
    }
    let function: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult =
        unsafe { std::mem::transmute(pointer) };
    match reserve(bytes) {
        Reservation::Refused => REFUSED,
        Reservation::Cold => unsafe { function(output, bytes) },
        Reservation::Active(index) => {
            let status = unsafe { function(output, bytes) };
            finish(
                index,
                if status == SUCCESS {
                    unsafe { *output }
                } else {
                    0
                },
                status,
            )
        }
    }
}
unsafe extern "C" fn free(pointer: CUdeviceptr) -> CUresult {
    let actual = FREE.load(Ordering::Acquire);
    if actual.is_null() {
        return UNSUPPORTED;
    }
    let function: unsafe extern "C" fn(CUdeviceptr) -> CUresult =
        unsafe { std::mem::transmute(actual) };
    let status = unsafe { function(pointer) };
    if let Ok(mut state) = STATE.lock() {
        free_in(&mut state, pointer, status);
    }
    status
}
fn free_in(state: &mut State, pointer: CUdeviceptr, status: CUresult) {
    if let Some(index) = state
        .entries
        .iter()
        .position(|entry| entry.state == 2 && entry.pointer == pointer)
    {
        if status == SUCCESS {
            state.live -= state.entries[index].bytes;
            state.entries[index].state = 3; // Charged until original final retirement.
        } else {
            state.phase = Phase::Poisoned;
        }
    }
}
fn note_name(state: &mut State, name: &[u8]) {
    if state.first_unsupported[0] == 0 {
        let count = name.len().min(63);
        state.first_unsupported[..count].copy_from_slice(&name[..count]);
    }
}
fn deny_named(name: &[u8]) -> bool {
    let Ok(mut state) = STATE.lock() else {
        return true;
    };
    if state.phase == Phase::Cold {
        return false;
    }
    state.unsupported = state.unsupported.saturating_add(1);
    note_name(&mut state, name);
    state.events.record(name, EventKind::DeniedCall);
    if state.events.overflow {
        state.phase = Phase::Poisoned;
    }
    true
}
macro_rules! denied {
    ($name:ident, $slot:ident, $label:literal, ($($arg:ident:$ty:ty),*)) => {
        unsafe extern "C" fn $name($($arg:$ty),*) -> CUresult {
            if deny_named($label) { return UNSUPPORTED; }
            let pointer = $slot.load(Ordering::Acquire);
            if pointer.is_null() { return UNSUPPORTED; }
            let function: unsafe extern "C" fn($($ty),*) -> CUresult = unsafe { std::mem::transmute(pointer) };
            unsafe { function($($arg),*) }
        }
    };
}
denied!(alloc_async, ALLOC_ASYNC, b"cuMemAllocAsync", (output:*mut CUdeviceptr, bytes:usize, stream:CUstream));
denied!(alloc_async_pt, ALLOC_ASYNC_PT, b"cuMemAllocAsync_ptsz", (output:*mut CUdeviceptr, bytes:usize, stream:CUstream));
denied!(alloc_pool, ALLOC_POOL, b"cuMemAllocFromPoolAsync", (output:*mut CUdeviceptr, bytes:usize, pool:CUmemoryPool, stream:CUstream));
denied!(alloc_pool_pt, ALLOC_POOL_PT, b"cuMemAllocFromPoolAsync_ptsz", (output:*mut CUdeviceptr, bytes:usize, pool:CUmemoryPool, stream:CUstream));
denied!(free_async, FREE_ASYNC, b"cuMemFreeAsync", (pointer:CUdeviceptr, stream:CUstream));
denied!(free_async_pt, FREE_ASYNC_PT, b"cuMemFreeAsync_ptsz", (pointer:CUdeviceptr, stream:CUstream));

#[cfg(test)]
mod tests {
    use super::*;
    fn state(cap: u64) -> State {
        State {
            phase: Phase::Active,
            cap,
            charged: 0,
            live: 0,
            attempts: 0,
            refunds: 0,
            refusals: 0,
            unsupported: 0,
            resolver_calls: 0,
            redirected: 0,
            provider_live: false,
            first_unsupported: [0; 64],
            denied_lookups: 0,
            events: ResolverEvents::new(),
            export_queries: ExportQueries::new(),
            original_device: -1,
            original_context: 0,
            primary_retains: 0,
            context_inflight: 0,
            entries: [EMPTY; SLOTS],
            admission_owner: None,
            queue_owner: None,
        }
    }
    fn admitted(state: &mut State, bytes: usize) -> usize {
        match reserve_in(state, bytes) {
            Reservation::Active(index) => index,
            _ => panic!("expected admission"),
        }
    }
    #[test]
    fn allocator_refuses_before_call_and_failed_allocation_refunds() {
        let mut state = state(64);
        assert!(matches!(reserve_in(&mut state, 65), Reservation::Refused));
        assert_eq!(state.charged, 0);
        let index = admitted(&mut state, 64);
        assert_eq!(state.charged, 64);
        assert!(matches!(reserve_in(&mut state, 1), Reservation::Refused));
        assert_eq!(finish_in(&mut state, index, 0, REFUSED), REFUSED);
        assert_eq!((state.charged, state.refunds, state.live), (0, 64, 0));
        assert_eq!(state.entries[index].state, 0);
    }
    #[test]
    fn allocator_success_and_free_retain_charge_until_final_retirement() {
        let mut state = state(64);
        let index = admitted(&mut state, 64);
        assert_eq!(finish_in(&mut state, index, 0x1000, SUCCESS), SUCCESS);
        assert_eq!((state.charged, state.live), (64, 64));
        free_in(&mut state, 0x1000, SUCCESS);
        assert_eq!((state.charged, state.live), (64, 0));
        assert_eq!(state.entries[index].state, 3);
        assert!(matches!(reserve_in(&mut state, 1), Reservation::Refused));
        assert_eq!(snapshot(&state).live_allocations, 0);
    }
    #[test]
    fn allocator_unknown_free_and_duplicate_success_do_not_release_budget() {
        let mut state = state(128);
        let first = admitted(&mut state, 64);
        finish_in(&mut state, first, 0x1000, SUCCESS);
        free_in(&mut state, 0x2000, SUCCESS);
        assert_eq!((state.charged, state.live), (64, 64));
        let second = admitted(&mut state, 64);
        assert_ne!(finish_in(&mut state, second, 0x1000, SUCCESS), SUCCESS);
        assert_eq!(state.phase, Phase::Poisoned);
        assert_eq!(state.charged, 128);
        assert!(matches!(reserve_in(&mut state, 1), Reservation::Refused));
    }
    #[test]
    fn allocator_exhausted_tracking_and_zero_requests_fail_closed() {
        let mut state = state(u64::MAX);
        assert!(matches!(reserve_in(&mut state, 0), Reservation::Refused));
        for index in 0..SLOTS {
            let slot = admitted(&mut state, 1);
            finish_in(&mut state, slot, (index + 1) as u64, SUCCESS);
        }
        assert!(matches!(reserve_in(&mut state, 1), Reservation::Refused));
        assert_eq!(state.charged, SLOTS as u64);
    }
    #[test]
    fn glibc_resolvers_bootstrap_without_interposed_lookup() {
        assert!(!elf::glibc_symbol(b"dlsym").is_null());
        assert!(!elf::glibc_symbol(b"dlvsym").is_null());
        assert!(elf::glibc_symbol(b"not_an_owned_glibc_symbol").is_null());
    }
}

pub(crate) fn bind_original_context(device: i32, context: usize) -> Result<(), ServerError> {
    let mut state = STATE
        .lock()
        .map_err(|_| error("BLAS original context binding lock poisoned"))?;
    if state.phase != Phase::Cold || device < 0 || context == 0 {
        return Err(error(
            "BLAS original live primary context required before activation",
        ));
    }
    state.original_device = device;
    state.original_context = context;
    Ok(())
}
fn capture_driver(handle: *mut c_void, name: &[u8], actual: *mut c_void) {
    if name != b"cuDevicePrimaryCtxRetain"
        || actual.is_null()
        || handle.is_null()
        || handle == (-1_isize) as *mut c_void
    {
        return;
    }
    if let Ok(state) = STATE.lock() {
        if state.phase != Phase::Cold {
            return;
        }
        let _ = DRIVER_LIBRARY.compare_exchange(
            ptr::null_mut(),
            handle,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}
