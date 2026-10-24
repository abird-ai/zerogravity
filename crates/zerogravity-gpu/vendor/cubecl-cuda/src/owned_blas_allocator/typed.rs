//! Exact installed cudarc0.19.10 driver ABI declarations; refusal is at CALL.
//! Explicit aliases are seeded from the original driver before activation.
use super::*;
use cudarc::driver::sys::*;
use std::ffi::{c_uint, c_ulonglong};
macro_rules! gate {
 ($fun:ident,$slot:ident,$symbol:literal,($($arg:ident:$ty:ty),*)) => {
  static $slot: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
  #[allow(non_snake_case)]
  unsafe extern "C" fn $fun($($arg:$ty),*)->CUresult {
   if deny_named($symbol) {return UNSUPPORTED;}
   let actual=$slot.load(Ordering::Acquire);if actual.is_null(){return UNSUPPORTED;}
   let call:unsafe extern "C" fn($($ty),*)->CUresult=unsafe{std::mem::transmute(actual)};
   unsafe{call($($arg),*)}
  }
 };
}
gate!(call_0, SLOT_0, b"cuArray3DCreate_v2", (pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR));
gate!(call_1, SLOT_1, b"cuArrayCreate_v2", (pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR));
gate!(call_2, SLOT_2, b"cuCtxCreate_v2", (pctx: *mut CUcontext, flags: c_uint, dev: CUdevice));
gate!(call_3, SLOT_3, b"cuCtxCreate_v3", (pctx: *mut CUcontext, paramsArray: *mut CUexecAffinityParam, numParams: c_int, flags: c_uint, dev: CUdevice));
gate!(call_4, SLOT_4, b"cuCtxCreate_v4", (pctx: *mut CUcontext, ctxCreateParams: *mut CUctxCreateParams, flags: c_uint, dev: CUdevice));
gate!(call_5, SLOT_5, b"cuCtxDestroy_v2", (ctx: CUcontext));
gate!(call_6, SLOT_6, b"cuDevicePrimaryCtxReset_v2", (dev: CUdevice));
gate!(call_7, SLOT_7, b"cuDevicePrimaryCtxSetFlags_v2", (dev: CUdevice, flags: c_uint));
gate!(call_8, SLOT_8, b"cuExternalMemoryGetMappedBuffer", (devPtr: *mut CUdeviceptr, extMem: CUexternalMemory, bufferDesc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC));
gate!(call_9, SLOT_9, b"cuExternalMemoryGetMappedMipmappedArray", (mipmap: *mut CUmipmappedArray, extMem: CUexternalMemory, mipmapDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC));
static SLOT_10: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
unsafe extern "C" fn call_10(output: *mut *const c_void, uuid: *const CUuuid) -> CUresult {
    // Copy exactly the requested argument BEFORE refusing; never inspect the
    // result table or call the original function while the pool is active.
    if let Ok(mut state) = STATE.lock() {
        if state.phase != Phase::Cold {
            let value = if uuid.is_null() {
                None
            } else {
                Some(unsafe { (*uuid).bytes }.map(|byte| byte as u8))
            };
            state.export_queries.record(value);
            if state.export_queries.overflow {
                state.phase = Phase::Poisoned;
            }
        }
    }
    if deny_named(b"cuGetExportTable") {
        return UNSUPPORTED;
    }
    let actual = SLOT_10.load(Ordering::Acquire);
    if actual.is_null() {
        return UNSUPPORTED;
    }
    let call: unsafe extern "C" fn(*mut *const c_void, *const CUuuid) -> CUresult =
        unsafe { std::mem::transmute(actual) };
    unsafe { call(output, uuid) }
}
gate!(call_11, SLOT_11, b"cuGraphAddMemAllocNode", (phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *mut CUDA_MEM_ALLOC_NODE_PARAMS));
gate!(call_12, SLOT_12, b"cuGraphInstantiateWithFlags", (phGraphExec: *mut CUgraphExec, hGraph: CUgraph, flags: c_ulonglong));
gate!(call_13, SLOT_13, b"cuGraphInstantiateWithParams", (phGraphExec: *mut CUgraphExec, hGraph: CUgraph, instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS));
gate!(call_14, SLOT_14, b"cuGraphInstantiate_v2", (phGraphExec: *mut CUgraphExec, hGraph: CUgraph, phErrorNode: *mut CUgraphNode, logBuffer: *mut c_char, bufferSize: usize));
gate!(call_15, SLOT_15, b"cuImportExternalMemory", (extMem_out: *mut CUexternalMemory, memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC));
gate!(call_16, SLOT_16, b"cuMemAddressReserve", (ptr: *mut CUdeviceptr, size: usize, alignment: usize, addr: CUdeviceptr, flags: c_ulonglong));
gate!(call_17, SLOT_17, b"cuMemAllocManaged", (dptr: *mut CUdeviceptr, bytesize: usize, flags: c_uint));
gate!(call_18, SLOT_18, b"cuMemAllocPitch_v2", (dptr: *mut CUdeviceptr, pPitch: *mut usize, WidthInBytes: usize, Height: usize, ElementSizeBytes: c_uint));
gate!(call_19, SLOT_19, b"cuMemCreate", (handle: *mut CUmemGenericAllocationHandle, size: usize, prop: *const CUmemAllocationProp, flags: c_ulonglong));
gate!(call_20, SLOT_20, b"cuMemHostAlloc", (pp: *mut *mut c_void, bytesize: usize, Flags: c_uint));
gate!(call_21, SLOT_21, b"cuMemHostGetDevicePointer_v2", (pdptr: *mut CUdeviceptr, p: *mut c_void, Flags: c_uint));
gate!(call_22, SLOT_22, b"cuMemHostGetFlags", (pFlags: *mut c_uint, p: *mut c_void));
gate!(call_23, SLOT_23, b"cuMemHostRegister_v2", (p: *mut c_void, bytesize: usize, Flags: c_uint));
gate!(call_24, SLOT_24, b"cuMemHostUnregister", (p: *mut c_void));
gate!(call_25, SLOT_25, b"cuMemImportFromShareableHandle", (handle: *mut CUmemGenericAllocationHandle, osHandle: *mut c_void, shHandleType: CUmemAllocationHandleType));
gate!(call_26, SLOT_26, b"cuMemMap", (ptr: CUdeviceptr, size: usize, offset: usize, handle: CUmemGenericAllocationHandle, flags: c_ulonglong));
gate!(call_27, SLOT_27, b"cuMemMapArrayAsync", (mapInfoList: *mut CUarrayMapInfo, count: c_uint, hStream: CUstream));
gate!(call_28, SLOT_28, b"cuMemPoolCreate", (pool: *mut CUmemoryPool, poolProps: *const CUmemPoolProps));
gate!(call_29, SLOT_29, b"cuMemPoolDestroy", (pool: CUmemoryPool));
gate!(call_30, SLOT_30, b"cuMemPoolExportPointer", (shareData_out: *mut CUmemPoolPtrExportData, ptr: CUdeviceptr));
gate!(call_31, SLOT_31, b"cuMemPoolExportToShareableHandle", (handle_out: *mut c_void, pool: CUmemoryPool, handleType: CUmemAllocationHandleType, flags: c_ulonglong));
gate!(call_32, SLOT_32, b"cuMemPoolGetAccess", (flags: *mut CUmemAccess_flags, memPool: CUmemoryPool, location: *mut CUmemLocation));
gate!(call_33, SLOT_33, b"cuMemPoolGetAttribute", (pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut c_void));
gate!(call_34, SLOT_34, b"cuMemPoolImportFromShareableHandle", (pool_out: *mut CUmemoryPool, handle: *mut c_void, handleType: CUmemAllocationHandleType, flags: c_ulonglong));
gate!(call_35, SLOT_35, b"cuMemPoolImportPointer", (ptr_out: *mut CUdeviceptr, pool: CUmemoryPool, shareData: *mut CUmemPoolPtrExportData));
gate!(call_36, SLOT_36, b"cuMemPoolSetAccess", (pool: CUmemoryPool, map: *const CUmemAccessDesc, count: usize));
gate!(call_37, SLOT_37, b"cuMemPoolSetAttribute", (pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut c_void));
gate!(call_38, SLOT_38, b"cuMemPoolTrimTo", (pool: CUmemoryPool, minBytesToKeep: usize));
gate!(call_39, SLOT_39, b"cuMipmappedArrayCreate", (pHandle: *mut CUmipmappedArray, pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR, numMipmapLevels: c_uint));
gate!(call_40, SLOT_40, b"cuMulticastCreate", (mcHandle: *mut CUmemGenericAllocationHandle, prop: *const CUmulticastObjectProp));
gate!(call_map_ptsz, SLOT_MAP_PTSZ, b"cuMemMapArrayAsync_ptsz", (mapInfoList:*mut CUarrayMapInfo,count:c_uint,hStream:CUstream));
gate!(call_graph_params_ptsz, SLOT_GRAPH_PARAMS_PTSZ, b"cuGraphInstantiateWithParams_ptsz", (graph_exec:*mut CUgraphExec,graph:CUgraph,params:*mut CUDA_GRAPH_INSTANTIATE_PARAMS));
struct Api {
    name: &'static [u8],
    slot: &'static AtomicPtr<c_void>,
    wrapper: fn() -> *mut c_void,
}
static APIS: [Api; 43] = [
    Api {
        name: b"cuGraphInstantiateWithParams_ptsz",
        slot: &SLOT_GRAPH_PARAMS_PTSZ,
        wrapper: || call_graph_params_ptsz as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemMapArrayAsync_ptsz",
        slot: &SLOT_MAP_PTSZ,
        wrapper: || call_map_ptsz as *const () as *mut c_void,
    },
    Api {
        name: b"cuArray3DCreate_v2",
        slot: &SLOT_0,
        wrapper: || call_0 as *const () as *mut c_void,
    },
    Api {
        name: b"cuArrayCreate_v2",
        slot: &SLOT_1,
        wrapper: || call_1 as *const () as *mut c_void,
    },
    Api {
        name: b"cuCtxCreate_v2",
        slot: &SLOT_2,
        wrapper: || call_2 as *const () as *mut c_void,
    },
    Api {
        name: b"cuCtxCreate_v3",
        slot: &SLOT_3,
        wrapper: || call_3 as *const () as *mut c_void,
    },
    Api {
        name: b"cuCtxCreate_v4",
        slot: &SLOT_4,
        wrapper: || call_4 as *const () as *mut c_void,
    },
    Api {
        name: b"cuCtxDestroy_v2",
        slot: &SLOT_5,
        wrapper: || call_5 as *const () as *mut c_void,
    },
    Api {
        name: b"cuDevicePrimaryCtxReset_v2",
        slot: &SLOT_6,
        wrapper: || call_6 as *const () as *mut c_void,
    },
    Api {
        name: b"cuDevicePrimaryCtxSetFlags_v2",
        slot: &SLOT_7,
        wrapper: || call_7 as *const () as *mut c_void,
    },
    Api {
        name: b"cuExternalMemoryGetMappedBuffer",
        slot: &SLOT_8,
        wrapper: || call_8 as *const () as *mut c_void,
    },
    Api {
        name: b"cuExternalMemoryGetMappedMipmappedArray",
        slot: &SLOT_9,
        wrapper: || call_9 as *const () as *mut c_void,
    },
    Api {
        name: b"cuGetExportTable",
        slot: &SLOT_10,
        wrapper: || call_10 as *const () as *mut c_void,
    },
    Api {
        name: b"cuGraphAddMemAllocNode",
        slot: &SLOT_11,
        wrapper: || call_11 as *const () as *mut c_void,
    },
    Api {
        name: b"cuGraphInstantiateWithFlags",
        slot: &SLOT_12,
        wrapper: || call_12 as *const () as *mut c_void,
    },
    Api {
        name: b"cuGraphInstantiateWithParams",
        slot: &SLOT_13,
        wrapper: || call_13 as *const () as *mut c_void,
    },
    Api {
        name: b"cuGraphInstantiate_v2",
        slot: &SLOT_14,
        wrapper: || call_14 as *const () as *mut c_void,
    },
    Api {
        name: b"cuImportExternalMemory",
        slot: &SLOT_15,
        wrapper: || call_15 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemAddressReserve",
        slot: &SLOT_16,
        wrapper: || call_16 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemAllocManaged",
        slot: &SLOT_17,
        wrapper: || call_17 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemAllocPitch_v2",
        slot: &SLOT_18,
        wrapper: || call_18 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemCreate",
        slot: &SLOT_19,
        wrapper: || call_19 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemHostAlloc",
        slot: &SLOT_20,
        wrapper: || call_20 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemHostGetDevicePointer_v2",
        slot: &SLOT_21,
        wrapper: || call_21 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemHostGetFlags",
        slot: &SLOT_22,
        wrapper: || call_22 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemHostRegister_v2",
        slot: &SLOT_23,
        wrapper: || call_23 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemHostUnregister",
        slot: &SLOT_24,
        wrapper: || call_24 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemImportFromShareableHandle",
        slot: &SLOT_25,
        wrapper: || call_25 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemMap",
        slot: &SLOT_26,
        wrapper: || call_26 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemMapArrayAsync",
        slot: &SLOT_27,
        wrapper: || call_27 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolCreate",
        slot: &SLOT_28,
        wrapper: || call_28 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolDestroy",
        slot: &SLOT_29,
        wrapper: || call_29 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolExportPointer",
        slot: &SLOT_30,
        wrapper: || call_30 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolExportToShareableHandle",
        slot: &SLOT_31,
        wrapper: || call_31 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolGetAccess",
        slot: &SLOT_32,
        wrapper: || call_32 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolGetAttribute",
        slot: &SLOT_33,
        wrapper: || call_33 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolImportFromShareableHandle",
        slot: &SLOT_34,
        wrapper: || call_34 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolImportPointer",
        slot: &SLOT_35,
        wrapper: || call_35 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolSetAccess",
        slot: &SLOT_36,
        wrapper: || call_36 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolSetAttribute",
        slot: &SLOT_37,
        wrapper: || call_37 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMemPoolTrimTo",
        slot: &SLOT_38,
        wrapper: || call_38 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMipmappedArrayCreate",
        slot: &SLOT_39,
        wrapper: || call_39 as *const () as *mut c_void,
    },
    Api {
        name: b"cuMulticastCreate",
        slot: &SLOT_40,
        wrapper: || call_40 as *const () as *mut c_void,
    },
];
fn apis() -> &'static [Api] {
    &APIS
}
pub(super) unsafe fn seed(driver: *mut c_void) {
    for api in apis() {
        let mut symbol = [0_u8; 96];
        symbol[..api.name.len()].copy_from_slice(api.name);
        let actual = unsafe { real_dlsym(driver, symbol.as_ptr().cast()) };
        if !actual.is_null() {
            let _ = api.slot.compare_exchange(
                ptr::null_mut(),
                actual,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}
pub(super) fn rewrite(
    name: &[u8],
    actual: *mut c_void,
    version: c_int,
    flags: u64,
) -> Option<*mut c_void> {
    if actual.is_null() {
        return None;
    }
    let name = match (name, flags & 2 != 0) {
        (b"cuMemMapArrayAsync", true) => b"cuMemMapArrayAsync_ptsz".as_slice(),
        (b"cuGraphInstantiateWithParams", true) => b"cuGraphInstantiateWithParams_ptsz".as_slice(),
        _ => name,
    };
    if let Some(api) = apis().iter().find(|api| api.name == name) {
        return Some(record(api.slot, actual, (api.wrapper)()));
    }
    // GPA base names can select different ABI revisions. Compare actual seeded
    // entrypoint identity; never infer a function signature from a broad prefix.
    if version > 0 {
        let mut selected = None;
        for api in apis() {
            let base = api.name.rsplit(|&b| b == b'_').next();
            let family = if base.is_some_and(|suffix| {
                suffix.len() > 1 && suffix[0] == b'v' && suffix[1..].iter().all(u8::is_ascii_digit)
            }) {
                &api.name[..api.name.len() - base.unwrap().len() - 1]
            } else {
                api.name
            };
            let graph_flags_alias =
                name == b"cuGraphInstantiate" && api.name == b"cuGraphInstantiateWithFlags";
            if (family == name || graph_flags_alias) && api.slot.load(Ordering::Acquire) == actual {
                if selected.is_some() {
                    return None;
                }
                selected = Some(record(api.slot, actual, (api.wrapper)()));
            }
        }
        return selected;
    }
    None
}
