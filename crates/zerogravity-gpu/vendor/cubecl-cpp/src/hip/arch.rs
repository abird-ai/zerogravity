use crate::shared::Architecture;

pub use cubecl_core::ir::amd::{AMDArchitecture, AmdWmma};

impl Architecture for AMDArchitecture {
    fn warp_size(&self) -> u32 {
        // Zero for an architecture the table does not know, which the HIP runtime checks the
        // driver's own answer against before anything is compiled for it.
        self.plane_dim().unwrap_or(0)
    }

    fn is_wmma_capable(&self) -> bool {
        matches!(
            self,
            AMDArchitecture::GFX103 | AMDArchitecture::GFX11 | AMDArchitecture::GFX12
        )
    }

    fn is_mfma_capable(&self) -> bool {
        matches!(
            self,
            AMDArchitecture::GFX908 | AMDArchitecture::GFX90A | AMDArchitecture::GFX94
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx90c_uses_wave64_without_matrix_instructions() {
        let arch = AMDArchitecture::GFX90C;
        assert_eq!(arch.warp_size(), 64);
        assert_eq!(arch.wmma_generation(), None);
        assert!(!arch.is_wmma_capable());
        assert!(!arch.is_mfma_capable());
    }

    #[test]
    fn unknown_architecture_has_no_usable_wave_or_matrix_capability() {
        let arch = AMDArchitecture::Other;
        assert_eq!(arch.warp_size(), 0);
        assert!(!arch.is_wmma_capable());
        assert!(!arch.is_mfma_capable());
    }
}
