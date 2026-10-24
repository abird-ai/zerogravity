//! Explicit argument ABI; captured addresses are never inferred or rebased.
use super::failure;
use cubecl_core::server::ServerError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubinIo {
    Read,
    /// No dependency on prior contents; same-command reads remain physical.
    Write,
    ReadWrite,
}
impl CubinIo {
    pub(crate) fn reads(self) -> bool {
        self != Self::Write
    }
    pub(crate) fn writes(self) -> bool {
        self != Self::Read
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CubinBinding {
    pub bytes: u64,
    pub alignment: u64,
    pub io: CubinIo,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CubinParameter {
    /// Exact device-side offset returned by cuFuncGetParamInfo.
    pub offset: usize,
    /// Pinned scalar/padding bytes; every declared pointer field must be zero.
    pub template: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CubinPointerFixup {
    pub parameter: usize,
    pub byte_offset: usize,
    pub binding: usize,
    pub binding_offset: u64,
    pub extent: u64,
    pub alignment: u64,
    pub io: CubinIo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubinLaunchAttributes {
    /// Ordinary cuLaunchKernel, kernelParams, no extra/extended attributes.
    None,
    /// Reserved refusal, never converted to ordinary launch semantics.
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CubinKernelManifest {
    pub entry: String,
    /// Only the explicitly captured SM80/SM86 forms are supported.
    pub binary_sm: u8,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_shared_bytes: u32,
    pub attributes: CubinLaunchAttributes,
    pub bindings: Vec<CubinBinding>,
    pub parameters: Vec<CubinParameter>,
    pub fixups: Vec<CubinPointerFixup>,
}

fn alignment(value: u64) -> bool {
    value.is_power_of_two() && value <= 4096
}

impl CubinKernelManifest {
    pub(crate) fn validate(&self) -> Result<(), ServerError> {
        if self.entry.is_empty()
            || self.entry.len() > 4096
            || !self.entry.bytes().all(|v| v.is_ascii_graphic())
            || !matches!(self.binary_sm, 80 | 86)
            || self.attributes != CubinLaunchAttributes::None
            || self.grid.contains(&0)
            || self.block.contains(&0)
            || self
                .block
                .into_iter()
                .try_fold(1u32, |n, x| n.checked_mul(x))
                .is_none_or(|n| n > 1024)
            || self.dynamic_shared_bytes > 96 * 1024
            || self.bindings.is_empty()
            || self.bindings.len() > 32
            || self.parameters.is_empty()
            || self.parameters.len() > 64
            || self.fixups.is_empty()
            || self.fixups.len() > 128
            || self
                .bindings
                .iter()
                .any(|b| b.bytes == 0 || !alignment(b.alignment))
        {
            return Err(failure("closed cubin geometry/attributes/bindings"));
        }
        let mut ranges = Vec::with_capacity(self.parameters.len());
        for p in &self.parameters {
            let end = p
                .offset
                .checked_add(p.template.len())
                .filter(|end| *end <= 4096 && !p.template.is_empty())
                .ok_or_else(|| failure("closed cubin parameter extent"))?;
            if ranges
                .iter()
                .any(|&(start, stop)| p.offset < stop && start < end)
            {
                return Err(failure("overlapping cubin device parameter layout"));
            }
            ranges.push((p.offset, end));
        }
        let mut fields: Vec<(usize, usize, usize)> = Vec::with_capacity(self.fixups.len());
        let mut access = vec![(false, false); self.bindings.len()];
        for f in &self.fixups {
            let p = self
                .parameters
                .get(f.parameter)
                .ok_or_else(|| failure("cubin pointer parameter index"))?;
            let b = self
                .bindings
                .get(f.binding)
                .ok_or_else(|| failure("cubin pointer binding index"))?;
            let end = f
                .byte_offset
                .checked_add(8)
                .filter(|end| *end <= p.template.len() && f.byte_offset.is_multiple_of(8))
                .ok_or_else(|| failure("cubin pointer field extent/alignment"))?;
            if p.template[f.byte_offset..end].iter().any(|v| *v != 0)
                || fields.iter().any(|&(param, start, stop)| {
                    param == f.parameter && f.byte_offset < stop && start < end
                })
                || f.extent == 0
                || !alignment(f.alignment)
                || !f.binding_offset.is_multiple_of(f.alignment)
                || f.binding_offset
                    .checked_add(f.extent)
                    .is_none_or(|end| end > b.bytes)
            {
                return Err(failure("cubin pointer bytes/overlap/range"));
            }
            fields.push((f.parameter, f.byte_offset, end));
            access[f.binding].0 |= f.io.reads();
            access[f.binding].1 |= f.io.writes();
        }
        if self
            .bindings
            .iter()
            .zip(access)
            .any(|(b, (r, w))| b.io.reads() != r || b.io.writes() != w)
        {
            return Err(failure("cubin binding IO differs from explicit fixups"));
        }
        Ok(())
    }

    pub(crate) fn arguments(&self, addresses: &[u64]) -> Result<Vec<Vec<u8>>, ServerError> {
        self.validate()?;
        if addresses.len() != self.bindings.len() {
            return Err(failure("cubin original address count"));
        }
        let mut extents = Vec::with_capacity(addresses.len());
        for (&address, binding) in addresses.iter().zip(&self.bindings) {
            let end = address
                .checked_add(binding.bytes)
                .filter(|_| address != 0 && address.is_multiple_of(binding.alignment))
                .ok_or_else(|| failure("cubin original address extent/alignment"))?;
            if extents
                .iter()
                .any(|&(start, stop)| address < stop && start < end)
            {
                return Err(failure("cubin distinct original bindings alias"));
            }
            extents.push((address, end));
        }
        let mut args: Vec<Vec<u8>> = self.parameters.iter().map(|p| p.template.clone()).collect();
        for f in &self.fixups {
            let pointer = addresses[f.binding]
                .checked_add(f.binding_offset)
                .filter(|v| v.is_multiple_of(f.alignment))
                .ok_or_else(|| failure("cubin pointer addition/alignment"))?;
            args[f.parameter][f.byte_offset..f.byte_offset + 8]
                .copy_from_slice(&pointer.to_le_bytes());
        }
        Ok(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest() -> CubinKernelManifest {
        CubinKernelManifest {
            entry: "observed_kernel".into(),
            binary_sm: 80,
            grid: [1, 1, 1],
            block: [32, 1, 1],
            dynamic_shared_bytes: 0,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 64,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 32,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![
                CubinParameter {
                    offset: 0,
                    template: vec![0; 16],
                },
                CubinParameter {
                    offset: 16,
                    template: 7u32.to_le_bytes().to_vec(),
                },
            ],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 16,
                    extent: 48,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 8,
                    binding: 1,
                    binding_offset: 0,
                    extent: 32,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        }
    }
    #[test]
    fn explicit_fixups_keep_scalars_and_template_immutable() {
        let m = manifest();
        let args = m.arguments(&[0x1000, 0x2000]).unwrap();
        assert_eq!(
            args[0],
            [0x1010u64.to_le_bytes(), 0x2000u64.to_le_bytes()].concat()
        );
        assert_eq!(args[1], 7u32.to_le_bytes());
        assert!(m.parameters[0].template.iter().all(|v| *v == 0));
    }
    #[test]
    fn malformed_templates_refuse_before_native_resolution() {
        let original = manifest();
        for kind in 0..10 {
            let mut m = original.clone();
            match kind {
                0 => m.parameters[0].template[0] = 1,
                1 => m.fixups[1].byte_offset = 0,
                2 => m.fixups[0].binding_offset = u64::MAX,
                3 => m.fixups[0].extent = 49,
                4 => m.parameters[1].offset = 8,
                5 => m.fixups[0].binding = 9,
                6 => m.bindings[0].io = CubinIo::ReadWrite,
                7 => m.attributes = CubinLaunchAttributes::Unsupported,
                8 => m.block = [u32::MAX, 2, 1],
                _ => m.binary_sm = 90,
            }
            assert!(m.validate().is_err(), "case {kind}");
        }
        for addresses in [
            [0x1000, 0x1020],
            [u64::MAX - 15, 0x2000],
            [0, 0x2000],
            [0x1001, 0x2000],
        ] {
            assert!(original.arguments(&addresses).is_err());
        }
    }
}
