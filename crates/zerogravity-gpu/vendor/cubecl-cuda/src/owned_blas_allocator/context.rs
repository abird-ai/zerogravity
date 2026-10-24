//! The actor's already-retained primary context, never a second context.
use super::*;
use cudarc::driver::sys::{CUcontext, CUdevice};
static RETAIN: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static RELEASE: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static RELEASE_OLD: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

pub(super) fn rewrite(name: &[u8], actual: *mut c_void, version: c_int) -> Option<*mut c_void> {
    let (slot, wrapper) = match name {
        b"cuDevicePrimaryCtxRetain" => (&RETAIN, retain as *const () as *mut c_void),
        b"cuDevicePrimaryCtxRelease_v2" => (&RELEASE, release as *const () as *mut c_void),
        b"cuDevicePrimaryCtxRelease" if version >= 11000 => {
            (&RELEASE, release as *const () as *mut c_void)
        }
        b"cuDevicePrimaryCtxRelease" => (&RELEASE_OLD, release_old as *const () as *mut c_void),
        _ => return None,
    };
    Some(record(slot, actual, wrapper))
}
unsafe extern "C" fn retain(output: *mut CUcontext, device: CUdevice) -> CUresult {
    if output.is_null() {
        return INVALID;
    }
    let actual = RETAIN.load(Ordering::Acquire);
    if actual.is_null() {
        return UNSUPPORTED;
    }
    let original = match begin(device, false, b"cuDevicePrimaryCtxRetain") {
        Ok(value) => value,
        Err(result) => return result,
    };
    let call: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult =
        unsafe { std::mem::transmute(actual) };
    let status = unsafe { call(output, device) };
    if let Some(expected) = original {
        let mut state = match STATE.lock() {
            Ok(state) => state,
            Err(_) => return CUresult::CUDA_ERROR_UNKNOWN,
        };
        state.context_inflight -= 1;
        if status != SUCCESS {
            state.primary_retains -= 1;
        }
        if status == SUCCESS {
            if unsafe { *output } as usize != expected {
                state.phase = Phase::Poisoned;
                return CUresult::CUDA_ERROR_INVALID_CONTEXT;
            }
            // Credit was reserved before the real retain call.
        }
    }
    status
}
fn begin(device: CUdevice, releasing: bool, name: &[u8]) -> Result<Option<usize>, CUresult> {
    let mut state = STATE.lock().map_err(|_| UNSUPPORTED)?;
    if state.phase == Phase::Cold {
        return Ok(None);
    }
    if state.phase != Phase::Active
        || state.original_context == 0
        || device != state.original_device
        || state.context_inflight == u64::MAX
        || (!releasing && state.primary_retains == u64::MAX)
        || (releasing && state.primary_retains == 0)
    {
        state.unsupported = state.unsupported.saturating_add(1);
        note_name(&mut state, name);
        state.events.record(name, events::EventKind::DeniedCall);
        if state.events.overflow {
            state.phase = Phase::Poisoned;
        }
        return Err(UNSUPPORTED);
    }
    if releasing {
        state.primary_retains -= 1;
    } else {
        state.primary_retains += 1;
    }
    state.context_inflight += 1;
    state.events.record(name, events::EventKind::ForwardedCall);
    if state.events.overflow {
        state.context_inflight -= 1;
        if releasing {
            state.primary_retains += 1;
        } else {
            state.primary_retains -= 1;
        }
        state.phase = Phase::Poisoned;
        return Err(UNSUPPORTED);
    }
    Ok(Some(state.original_context))
}
unsafe fn release_with(slot: &AtomicPtr<c_void>, device: CUdevice) -> CUresult {
    let actual = slot.load(Ordering::Acquire);
    if actual.is_null() {
        return UNSUPPORTED;
    }
    let active = match begin(device, true, b"cuDevicePrimaryCtxRelease") {
        Ok(value) => value.is_some(),
        Err(result) => return result,
    };
    let call: unsafe extern "C" fn(CUdevice) -> CUresult = unsafe { std::mem::transmute(actual) };
    let status = unsafe { call(device) };
    if active {
        if let Ok(mut state) = STATE.lock() {
            state.context_inflight -= 1;
            if status != SUCCESS {
                state.primary_retains = state.primary_retains.saturating_add(1);
                state.phase = Phase::Poisoned;
            }
        }
    }
    status
}
unsafe extern "C" fn release(device: CUdevice) -> CUresult {
    unsafe { release_with(&RELEASE, device) }
}
unsafe extern "C" fn release_old(device: CUdevice) -> CUresult {
    unsafe { release_with(&RELEASE_OLD, device) }
}
