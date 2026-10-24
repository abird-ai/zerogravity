//! Fixed zero prefixes; pointers and streams remain original actor resources.
use super::{CubinKernelManifest, failure};
use cubecl_core::server::ServerError;
use std::ops::Range;

/// Offset zero, value zero, exactly 32 or 64 bytes of a writable binding.
/// Metadata grants no buffer, pointer, stream or completion authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubinWorkspaceClear {
    binding: usize,
    bytes: usize,
}

impl CubinWorkspaceClear {
    pub fn zero32(binding: usize) -> Result<Self, ServerError> {
        Self::zero(binding, 32)
    }
    pub fn zero64(binding: usize) -> Result<Self, ServerError> {
        Self::zero(binding, 64)
    }
    fn zero(binding: usize, bytes: usize) -> Result<Self, ServerError> {
        if binding >= 32 || !matches!(bytes, 32 | 64) {
            return Err(failure("closed workspace clear binding/count"));
        }
        Ok(Self { binding, bytes })
    }
    pub(crate) fn binding(self) -> usize {
        self.binding
    }
    pub(crate) fn bytes(self) -> usize {
        self.bytes
    }
    pub(crate) fn validate(self, manifest: &CubinKernelManifest) -> Result<(), ServerError> {
        manifest.validate()?;
        let spec = manifest
            .bindings
            .get(self.binding)
            .ok_or_else(|| failure("workspace clear binding absent"))?;
        if !spec.io.writes() || spec.bytes < self.bytes as u64 {
            return Err(failure(
                "workspace clear requires bounded original write binding",
            ));
        }
        for fixup in manifest
            .fixups
            .iter()
            .filter(|f| f.binding == self.binding && f.io.reads())
        {
            if fixup
                .binding_offset
                .checked_add(fixup.extent)
                .is_none_or(|end| end > self.bytes as u64)
            {
                return Err(failure(
                    "workspace clear does not overwrite every kernel read",
                ));
            }
        }
        Ok(())
    }
    /// Exact overwritten regions only; successful writes leave tail failures intact.
    pub(crate) fn write_regions(
        self,
        manifest: &CubinKernelManifest,
    ) -> Result<Vec<Range<u64>>, ServerError> {
        self.validate(manifest)?;
        let mut regions = Vec::with_capacity(1 + manifest.fixups.len());
        regions.push(0..self.bytes as u64);
        for fixup in manifest
            .fixups
            .iter()
            .filter(|f| f.binding == self.binding && f.io.writes())
        {
            let end = fixup
                .binding_offset
                .checked_add(fixup.extent)
                .ok_or_else(|| failure("workspace write range overflow"))?;
            regions.push(fixup.binding_offset..end);
        }
        regions.sort_by_key(|r| r.start);
        let mut merged: Vec<Range<u64>> = Vec::with_capacity(regions.len());
        for range in regions {
            if let Some(last) = merged.last_mut().filter(|last| range.start <= last.end) {
                last.end = last.end.max(range.end);
            } else {
                merged.push(range);
            }
        }
        Ok(merged)
    }
    /// Actor-only, after original resource lookup. Canonical handle/service/
    /// alignment checks and full argument validation remain in original enqueue.
    pub(crate) fn checked_native_range(
        self,
        declared_bytes: u64,
        actual_bytes: u64,
        pointer: u64,
    ) -> Result<u64, ServerError> {
        if declared_bytes != actual_bytes
            || actual_bytes < self.bytes as u64
            || pointer == 0
            || pointer.checked_add(self.bytes as u64).is_none()
        {
            return Err(failure("workspace clear original native extent/range"));
        }
        Ok(pointer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_command::{
        CubinBinding, CubinIo, CubinLaunchAttributes, CubinParameter, CubinPointerFixup,
    };

    fn manifest(bytes: u64, io: CubinIo) -> CubinKernelManifest {
        CubinKernelManifest {
            entry: "workspace_scope_test".into(),
            binary_sm: 80,
            grid: [1, 1, 1],
            block: [32, 1, 1],
            dynamic_shared_bytes: 0,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![CubinBinding {
                bytes,
                alignment: 16,
                io,
            }],
            parameters: vec![CubinParameter {
                offset: 0,
                template: vec![0; 8],
            }],
            fixups: vec![CubinPointerFixup {
                parameter: 0,
                byte_offset: 0,
                binding: 0,
                binding_offset: 0,
                extent: 32,
                alignment: 16,
                io,
            }],
        }
    }

    #[test]
    fn only_fixed_prefixes_and_declared_write_bindings_are_available() {
        let small = CubinWorkspaceClear::zero32(0).unwrap();
        let large = CubinWorkspaceClear::zero64(0).unwrap();
        assert_eq!(small.binding(), 0);
        assert_eq!((small.bytes(), large.bytes()), (32, 64));
        assert!(CubinWorkspaceClear::zero32(32).is_err());
        let m = manifest(64, CubinIo::ReadWrite);
        assert!(
            CubinWorkspaceClear::zero32(1)
                .unwrap()
                .validate(&m)
                .is_err()
        );
        assert!(small.validate(&manifest(32, CubinIo::Read)).is_err());
        assert!(small.validate(&manifest(31, CubinIo::Write)).is_err());
        assert!(small.validate(&manifest(32, CubinIo::Write)).is_ok());
        assert!(large.validate(&manifest(32, CubinIo::Write)).is_err());
        assert!(large.validate(&m).is_ok());
    }

    #[test]
    fn every_historical_read_must_be_covered_and_write_gaps_remain_undefined() {
        let clear = CubinWorkspaceClear::zero32(0).unwrap();
        let mut m = manifest(128, CubinIo::ReadWrite);
        m.fixups[0].extent = 33;
        assert!(clear.validate(&m).is_err());
        m.fixups[0].extent = 32;
        m.fixups[0].binding_offset = 16;
        assert!(clear.validate(&m).is_err());
        m.fixups[0].binding_offset = 0;
        m.parameters[0].template.resize(24, 0);
        for (field, offset) in [(8, 64), (16, 80)] {
            m.fixups.push(CubinPointerFixup {
                parameter: 0,
                byte_offset: field,
                binding: 0,
                binding_offset: offset,
                extent: 16,
                alignment: 16,
                io: CubinIo::Write,
            });
        }
        assert_eq!(clear.write_regions(&m).unwrap(), vec![0..32, 64..96]);
    }

    #[test]
    fn actual_actor_extent_and_checked_address_must_cover_the_prefix() {
        let clear = CubinWorkspaceClear::zero64(0).unwrap();
        assert_eq!(clear.checked_native_range(64, 64, 4096).unwrap(), 4096);
        for (declared, actual, pointer) in [
            (64, 63, 4096),
            (63, 63, 4096),
            (64, 65, 4096),
            (64, 64, 0),
            (64, 64, u64::MAX - 31),
        ] {
            assert!(
                clear
                    .checked_native_range(declared, actual, pointer)
                    .is_err()
            );
        }
    }
}
