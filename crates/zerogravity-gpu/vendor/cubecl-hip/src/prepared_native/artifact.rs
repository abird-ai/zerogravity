//! Source artifact inventory and retained admission, independent of a device.
use std::{any::Any, sync::Arc};

pub type NativeRetention = Arc<dyn Any + Send + Sync>;

/// Module unload does not destroy the primary HIP/RTC context. Until an actual
/// context-destruction boundary exists, retain its distinct original charge.
pub(crate) struct PersistentNativeOwner(NativeRetention);
impl PersistentNativeOwner {
    pub(crate) fn new(owner: NativeRetention) -> Self {
        Self(owner)
    }
    pub(crate) fn same(&self, owner: &NativeRetention) -> bool {
        Arc::ptr_eq(&self.0, owner)
    }
}
impl std::fmt::Debug for PersistentNativeOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentNativeOwner")
            .finish_non_exhaustive()
    }
}
impl Drop for PersistentNativeOwner {
    fn drop(&mut self) {
        std::mem::forget(self.0.clone());
    }
}

const MAX_CODE_BYTES: usize = 64 * 1024 * 1024;
const MODULE_METADATA_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub(crate) struct ArtifactError(&'static str);
impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
fn failure(message: &'static str) -> ArtifactError {
    ArtifactError(message)
}

/// Source-derived bounds, not measured driver allocation or an opaque allowance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HipNativeInventory {
    pub modules: usize,
    pub retained_host_bytes: usize,
    pub code_object_load_bytes: usize,
}
impl HipNativeInventory {
    pub(crate) fn add(self, other: Self) -> Result<Self, ArtifactError> {
        Ok(Self {
            modules: self
                .modules
                .checked_add(other.modules)
                .ok_or_else(|| failure("HIP module count overflow"))?,
            retained_host_bytes: self
                .retained_host_bytes
                .checked_add(other.retained_host_bytes)
                .ok_or_else(|| failure("HIP host inventory overflow"))?,
            code_object_load_bytes: self
                .code_object_load_bytes
                .checked_add(other.code_object_load_bytes)
                .ok_or_else(|| failure("HIP native inventory overflow"))?,
        })
    }
}

pub(crate) struct LoadedNativeOwner {
    _code: Option<Box<[i8]>>,
    _admission: Option<NativeRetention>,
    unloaded: bool,
}
impl LoadedNativeOwner {
    pub(crate) fn new(code: &[i8], owner: NativeRetention) -> Self {
        Self {
            _code: Some(code.into()),
            _admission: Some(owner),
            unloaded: false,
        }
    }
    pub(crate) fn mark_unloaded(&mut self) {
        self.unloaded = true;
    }
}
impl Drop for LoadedNativeOwner {
    fn drop(&mut self) {
        if !self.unloaded {
            // Context destruction is not physical completion or module unload.
            // Keep the original source/charge and bytes for the process lifetime.
            std::mem::forget(self._admission.take());
            std::mem::forget(self._code.take());
        }
    }
}
impl std::fmt::Debug for LoadedNativeOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedNativeOwner")
            .field("code_bytes", &self._code.as_ref().map(|code| code.len()))
            .finish_non_exhaustive()
    }
}

/// Bounded ELF64 AMDGPU load segments. Heap size is not native module size.
/// Includes per-segment alignment slack; overlapping segments are conservative.
pub(crate) fn code_inventory(code: &[i8]) -> Result<HipNativeInventory, ArtifactError> {
    if code.len() < 64 || code.len() > MAX_CODE_BYTES {
        return Err(failure("HIP code object size"));
    }
    let byte = |at: usize| -> Result<u8, ArtifactError> {
        code.get(at)
            .map(|n| *n as u8)
            .ok_or_else(|| failure("HIP truncated code object"))
    };
    let uint = |at: usize, bytes: usize| -> Result<u64, ArtifactError> {
        let mut value = 0;
        for i in 0..bytes {
            value |= (byte(
                at.checked_add(i)
                    .ok_or_else(|| failure("HIP ELF offset overflow"))?,
            )? as u64)
                << (8 * i);
        }
        Ok(value)
    };
    if [byte(0)?, byte(1)?, byte(2)?, byte(3)?] != *b"\x7fELF"
        || byte(4)? != 2
        || byte(5)? != 1
        || byte(6)? != 1
        || uint(16, 2)? != 3
        || uint(18, 2)? != 224
        || uint(52, 2)? != 64
        || uint(54, 2)? != 56
    {
        return Err(failure(
            "HIP requires complete ELF64 little-endian AMDGPU ET_DYN",
        ));
    }
    let base = usize::try_from(uint(32, 8)?).map_err(|_| failure("HIP ELF table offset"))?;
    let count = usize::try_from(uint(56, 2)?).map_err(|_| failure("HIP ELF segment count"))?;
    let end = count
        .checked_mul(56)
        .and_then(|n| base.checked_add(n))
        .ok_or_else(|| failure("HIP ELF table overflow"))?;
    if count == 0 || count > 128 || end > code.len() {
        return Err(failure("HIP ELF complete segment table"));
    }
    let mut device = 0u64;
    for i in 0..count {
        let p = base + i * 56;
        if uint(p, 4)? != 1 {
            continue;
        }
        let offset = uint(p + 8, 8)?;
        let virtual_address = uint(p + 16, 8)?;
        let file = uint(p + 32, 8)?;
        let memory = uint(p + 40, 8)?;
        let alignment = uint(p + 48, 8)?;
        if memory == 0
            || file > memory
            || offset
                .checked_add(file)
                .is_none_or(|n| n > code.len() as u64)
            || virtual_address.checked_add(memory).is_none()
            || alignment == 0
            || !alignment.is_power_of_two()
            || offset % alignment != virtual_address % alignment
        {
            return Err(failure("HIP ELF load segment range/alignment"));
        }
        device = device
            .checked_add(memory)
            .and_then(|n| n.checked_add(alignment - 1))
            .ok_or_else(|| failure("HIP native segment sum overflow"))?;
    }
    let device = usize::try_from(device).map_err(|_| failure("HIP native extent overflow"))?;
    if device == 0 {
        return Err(failure("HIP ELF has no nonempty load segment"));
    }
    let host = code
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_add(MODULE_METADATA_BYTES))
        .ok_or_else(|| failure("HIP retained code capacity overflow"))?;
    Ok(HipNativeInventory {
        modules: 1,
        retained_host_bytes: host,
        code_object_load_bytes: device,
    })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
