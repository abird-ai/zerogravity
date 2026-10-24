//! Checked cumulative known-byte limits; no driver/model/source authority.
/// Fixed finite qualification policy, not a universal NVRTC/driver allocator bound.
pub struct CudaCompilerContextPolicy;
impl CudaCompilerContextPolicy {
    pub const HOST_BYTES: u64 = 384 * 1024 * 1024;
    pub const DEVICE_BYTES: u64 = 512 * 1024 * 1024;
    pub const MAX_LOADS: usize = 2048;
    pub const MAX_SOURCE_BYTES: usize = 2 * 1024 * 1024;
    pub const MAX_KEY_SOURCE_BYTES: usize = 32 * 1024 * 1024 - 64 * 1024;
    // The remaining64KiB prices fixed context/SDK-path/lease metadata.
    pub const MAX_PTX_BYTES: usize = 16 * 1024 * 1024;
    pub const MAX_PTX_TOTAL_BYTES: usize = 128 * 1024 * 1024;
    pub const MAX_LOG_BYTES: usize = 1024 * 1024;
    pub const MAX_BINDINGS: usize = 32;
    /// Optional retained access analysis is not a physical ABI declaration.
    /// Precompiled source has no compiler analysis; launch preserves caller IO
    /// declarations, conservatively ReadWrite when absent. Factories bound ABI.
    pub fn validate_compiled_metadata(
        entry_bytes: usize,
        retained_io_bindings: Option<usize>,
    ) -> Result<(), &'static str> {
        if entry_bytes > 256 || retained_io_bindings.is_some_and(|n| n > Self::MAX_BINDINGS) {
            Err("admitted CUDA entry/retained IO metadata bound exceeded")
        } else {
            Ok(())
        }
    }
    // HOST:32MiB key/source/map/IO +128MiB cumulative PTX +64MiB known
    // compiler source/output/log staging +160MiB opaque compiler/library work.
    // DEVICE:512MiB opaque code/context. Physical source qualification required.
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CompilerContextUsage {
    pub attempts: usize,
    pub key_source_bytes: usize,
    pub ptx_bytes: usize,
}
impl CompilerContextUsage {
    pub(crate) fn begin(&mut self) -> Result<(), &'static str> {
        self.attempts = self
            .attempts
            .checked_add(1)
            .filter(|n| *n <= CudaCompilerContextPolicy::MAX_LOADS)
            .ok_or("admitted CUDA cumulative compilation count exceeded")?;
        Ok(())
    }
    pub(crate) fn source(&mut self, source: usize) -> Result<(), &'static str> {
        if source == 0 || source > CudaCompilerContextPolicy::MAX_SOURCE_BYTES {
            return Err("admitted CUDA emitted source extent exceeded");
        }
        // Closed factory compilation arguments/maps fit4KiB per entry. Full
        // source/key owner is charged without relying on digest/debug size.
        self.key_source_bytes = self
            .key_source_bytes
            .checked_add(source)
            .and_then(|n| n.checked_add(4096))
            .filter(|n| *n <= CudaCompilerContextPolicy::MAX_KEY_SOURCE_BYTES)
            .ok_or("admitted CUDA cumulative source/key metadata exceeded")?;
        Ok(())
    }
    pub(crate) fn ptx(&mut self, bytes: usize) -> Result<(), &'static str> {
        if bytes == 0 || bytes > CudaCompilerContextPolicy::MAX_PTX_BYTES {
            return Err("admitted CUDA PTX entry extent exceeded");
        }
        self.ptx_bytes = self
            .ptx_bytes
            .checked_add(bytes)
            .filter(|n| *n <= CudaCompilerContextPolicy::MAX_PTX_TOTAL_BYTES)
            .ok_or("admitted CUDA cumulative PTX extent exceeded")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_access_analysis_is_conservative_and_oversize_metadata_still_refuses() {
        for io in [None, Some(0), Some(CudaCompilerContextPolicy::MAX_BINDINGS)] {
            assert!(CudaCompilerContextPolicy::validate_compiled_metadata(256, io).is_ok());
        }
        for (entry, io) in [
            (257, None),
            (1, Some(33)),
            (usize::MAX, None),
            (1, Some(usize::MAX)),
        ] {
            assert!(CudaCompilerContextPolicy::validate_compiled_metadata(entry, io).is_err());
        }
        assert_eq!(CudaCompilerContextPolicy::MAX_BINDINGS, 32);
    }
    #[test]
    fn zero_oversize_and_overflow_refuse_without_wrapping_or_credit() {
        let mut u = CompilerContextUsage::default();
        for n in [
            0,
            CudaCompilerContextPolicy::MAX_SOURCE_BYTES + 1,
            usize::MAX,
        ] {
            assert!(u.source(n).is_err());
        }
        for n in [0, CudaCompilerContextPolicy::MAX_PTX_BYTES + 1, usize::MAX] {
            assert!(u.ptx(n).is_err());
        }
        assert_eq!((u.key_source_bytes, u.ptx_bytes), (0, 0));
        u.source(1024).unwrap();
        let paid = u.key_source_bytes;
        assert!(u.ptx(usize::MAX).is_err());
        assert_eq!(u.key_source_bytes, paid); // failed later effect gives no refund
        u.key_source_bytes = usize::MAX;
        assert!(u.source(1).is_err());
        assert_eq!(u.key_source_bytes, usize::MAX);
    }
    #[test]
    fn cumulative_load_source_and_ptx_limits_are_independent() {
        let mut u = CompilerContextUsage::default();
        for _ in 0..CudaCompilerContextPolicy::MAX_LOADS {
            u.begin().unwrap();
        }
        assert!(u.begin().is_err());
        assert_eq!(u.attempts, CudaCompilerContextPolicy::MAX_LOADS);
        for _ in 0..8 {
            u.ptx(CudaCompilerContextPolicy::MAX_PTX_BYTES).unwrap();
        }
        assert!(u.ptx(1).is_err());
        assert_eq!(u.ptx_bytes, CudaCompilerContextPolicy::MAX_PTX_TOTAL_BYTES);
        for _ in 0..15 {
            u.source(CudaCompilerContextPolicy::MAX_SOURCE_BYTES)
                .unwrap();
        }
        assert!(
            u.source(CudaCompilerContextPolicy::MAX_SOURCE_BYTES)
                .is_err()
        );
        assert_eq!(
            u.key_source_bytes,
            15 * (CudaCompilerContextPolicy::MAX_SOURCE_BYTES + 4096)
        );
    }
    #[test]
    fn host_policy_prices_all_known_regions_and_opaque_work_without_cap_raise() {
        const MIB: u64 = 1024 * 1024;
        assert_eq!(
            CudaCompilerContextPolicy::HOST_BYTES,
            32 * MIB + 128 * MIB + 64 * MIB + 160 * MIB
        );
        assert_eq!(
            4096 * MIB + CudaCompilerContextPolicy::DEVICE_BYTES,
            4608 * MIB
        );
        assert_eq!(
            4608 * MIB + CudaCompilerContextPolicy::DEVICE_BYTES,
            5120 * MIB
        );
    }
}
