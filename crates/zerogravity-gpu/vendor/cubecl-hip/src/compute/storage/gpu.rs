use crate::compute::status::checked;
use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use crate::AMD_MAX_BINDINGS;

/// Buffer storage for AMD GPUs.
///
/// This struct manages memory resources for HIP kernels, allowing them to be used as bindings
/// for launching kernels.
pub struct GpuStorage {
    mem_alignment: usize,
    memory: HashMap<StorageId, (cubecl_hip_sys::hipDeviceptr_t, u64)>,
    /// The bytes `memory` holds.
    allocated: u64,
    deallocations: Vec<StorageId>,
    ptr_bindings: PtrBindings,
    free_unknown: Arc<OnceLock<cubecl_server::driver::DriverError>>,
}

/// A GPU memory resource allocated for HIP using [`GpuStorage`].
#[derive(new, Debug)]
pub struct GpuResource {
    /// The GPU memory pointer.
    pub ptr: cubecl_hip_sys::hipDeviceptr_t,
    /// The HIP binding pointer.
    pub binding: cubecl_hip_sys::hipDeviceptr_t,
    /// The size of the resource.
    pub size: u64,
    retirement: Arc<OnceLock<cubecl_server::driver::DriverError>>,
}
impl GpuResource {
    pub(crate) fn require_known_retirement(
        &self,
    ) -> Result<(), cubecl_server::driver::DriverError> {
        self.retirement.get().cloned().map_or(Ok(()), Err)
    }
}

impl GpuStorage {
    /// Creates a new [`GpuStorage`] instance.
    ///
    /// # Arguments
    ///
    /// * `mem_alignment` - The memory alignment requirement in bytes.
    pub fn new(mem_alignment: usize) -> Self {
        Self {
            mem_alignment,
            memory: HashMap::new(),
            allocated: 0,
            deallocations: Vec::new(),
            ptr_bindings: PtrBindings::new(),
            free_unknown: Arc::new(OnceLock::new()),
        }
    }

    /// Deallocates buffers marked for deallocation.
    ///
    /// This method processes all pending deallocations by freeing the associated GPU memory.
    pub fn perform_deallocations(&mut self) {
        self.retire_deallocations(|ptr| {
            // SAFETY: original allocation, never previously freed or retried.
            checked("hipFree", unsafe { cubecl_hip_sys::hipFree(ptr) })
        });
    }
    pub(crate) fn require_known_retirement(
        &self,
    ) -> Result<(), cubecl_server::driver::DriverError> {
        match self.free_unknown.get() {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }
    fn retire_deallocations(
        &mut self,
        mut free: impl FnMut(
            cubecl_hip_sys::hipDeviceptr_t,
        ) -> Result<(), cubecl_server::driver::DriverError>,
    ) {
        if self.free_unknown.get().is_some() {
            return;
        }
        let mut pending = std::mem::take(&mut self.deallocations).into_iter();
        while let Some(id) = pending.next() {
            if let Some((ptr, size)) = self.memory.get(&id).copied() {
                match free(ptr) {
                    Ok(()) => {
                        self.memory.remove(&id);
                        self.allocated -= size;
                    }
                    Err(error) => {
                        // Uncertain free may have succeeded. Never retry this ptr,
                        // reuse it or erase its original allocation/accounting.
                        let _ = self.free_unknown.set(error);
                        self.deallocations.extend(pending);
                        return;
                    }
                }
            }
        }
    }
}

/// Manages active HIP buffer bindings in a ring buffer.
///
/// This ensures that pointers remain valid during kernel execution, preventing use-after-free errors.
struct PtrBindings {
    slots: Vec<u64>,
    cursor: usize,
}

impl PtrBindings {
    /// Creates a new [`PtrBindings`] instance with a fixed-size ring buffer.
    fn new() -> Self {
        Self {
            slots: vec![0; AMD_MAX_BINDINGS as usize],
            cursor: 0,
        }
    }

    /// Registers a new pointer in the ring buffer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The HIP device pointer to register.
    ///
    /// # Returns
    ///
    /// A reference to the registered pointer.
    fn register(&mut self, ptr: u64) -> &u64 {
        self.slots[self.cursor] = ptr;
        let ptr_ref = self.slots.get(self.cursor).unwrap();

        self.cursor += 1;

        // Reset the cursor when the ring buffer is full.
        if self.cursor >= self.slots.len() {
            self.cursor = 0;
        }

        ptr_ref
    }
}

impl ComputeStorage for GpuStorage {
    type Resource = GpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError> {
        self.require_known_retirement()?;
        let (ptr, _) =
            *self
                .memory
                .get(&handle.id)
                .ok_or_else(|| IoError::StorageHandleNotFound {
                    reason: format!("{} in the HIP gpu storage", handle.id).into(),
                    backtrace: BackTrace::capture(),
                })?;
        let ptr = ptr as u64;

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        Ok(GpuResource::new(
            *ptr as cubecl_hip_sys::hipDeviceptr_t,
            std::ptr::from_ref(ptr) as *mut std::ffi::c_void,
            size,
            self.free_unknown.clone(),
        ))
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        self.require_known_retirement()?;
        let id = StorageId::new();
        // Plain (synchronous) `hipMalloc` on purpose: `hipMallocAsync` draws
        // from the driver's async mempool, which on ROCm never returns freed
        // memory to the OS (`hipFreeAsync` + `hipMemPoolTrimTo(0)` keep the
        // high-water mark forever — measured on gfx1151). The pools above
        // amortize allocations into large, rare pages, so the synchronous
        // call costs nothing where it matters — and freed pages really leave
        // the process.
        //
        // SAFETY: Calling HIP FFI to allocate device memory. The pointer is
        // stored in `self.memory` and freed via `hipFree` on deallocation.
        unsafe {
            let mut ptr: *mut ::std::os::raw::c_void = std::ptr::null_mut();
            let status = cubecl_hip_sys::hipMalloc(&mut ptr, size as usize);

            checked("hipMalloc", status)?;

            self.memory.insert(id, (ptr, size));
            self.allocated += size;
        };

        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    fn flush(&mut self) {
        self.perform_deallocations();
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated
    }
}

// SAFETY: `GpuStorage` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The raw HIP pointers it contains are never shared
// across threads without synchronization.
unsafe impl Send for GpuStorage {}
// SAFETY: `GpuResource` contains raw HIP device pointers that are safe to send between
// threads as long as proper stream synchronization is maintained by the caller.
unsafe impl Send for GpuResource {}

impl core::fmt::Debug for GpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GpuStorage".to_string().as_str())
    }
}

#[cfg(test)]
mod retirement_tests {
    use super::*;
    use cubecl_server::driver::DriverError;
    #[test]
    fn failed_free_keeps_pointer_charge_and_never_retries_or_reuses() {
        let mut storage = GpuStorage::new(256);
        let a = StorageId::new();
        let b = StorageId::new();
        storage.memory.insert(a, (std::ptr::dangling_mut(), 256));
        storage.memory.insert(b, (std::ptr::dangling_mut(), 512));
        storage.allocated = 768;
        storage.deallocations.extend([a, b]);
        let resolved = GpuResource::new(
            std::ptr::dangling_mut(),
            std::ptr::dangling_mut(),
            256,
            storage.free_unknown.clone(),
        );
        assert!(resolved.require_known_retirement().is_ok());
        let mut calls = 0;
        storage.retire_deallocations(|_| {
            calls += 1;
            Err(DriverError::new("test-unknown-free", 999))
        });
        assert_eq!(calls, 1);
        assert!(resolved.require_known_retirement().is_err());
        assert_eq!(storage.allocated, 768);
        assert_eq!(storage.memory.len(), 2);
        assert!(storage.require_known_retirement().is_err());
        storage.retire_deallocations(|_| panic!("unknown free must not retry"));
        assert!(storage.alloc(256).is_err());
        assert_eq!(storage.memory.len(), 2);
    }
    #[test]
    fn successful_frees_debit_only_the_actual_retired_pages() {
        let mut storage = GpuStorage::new(256);
        let a = StorageId::new();
        let b = StorageId::new();
        storage.memory.insert(a, (std::ptr::dangling_mut(), 256));
        storage.memory.insert(b, (std::ptr::dangling_mut(), 512));
        storage.allocated = 768;
        storage.deallocations.push(a);
        storage.retire_deallocations(|_| Ok(()));
        assert_eq!(storage.allocated, 512);
        assert_eq!(storage.memory.len(), 1);
        assert!(storage.memory.contains_key(&b));
        assert!(storage.require_known_retirement().is_ok());
    }
}
