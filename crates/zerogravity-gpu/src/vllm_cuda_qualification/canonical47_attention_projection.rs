//! Exact Initial47/call0/layer3 QKV packets; no QK/KV/FA/model publication.
use super::canonical47_model::{Canonical47CallKind, Canonical47ProjectionBinding};
use super::canonical47_model_native::{Binding, Packet};
use super::cubin_manifests::ProjectionImage;
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup,
};
use zerogravity_core::accelerator::program::{ValueId, VllmBf16Invocation};
use zerogravity_core::{Error, Result};

// ROOT materializes only after independent selected memory/resource review.
const SELECTED_MEMORY_RECIPE_REVIEWED: bool = true;

pub(super) fn require_site(
    request: &VllmBf16Invocation,
    layer: usize,
    node: usize,
    inputs: &[ValueId],
    outputs: &[ValueId],
) -> Result<()> {
    if Canonical47CallKind::of(request)? != Canonical47CallKind::Prefill
        || layer != 3
        || node != 32
        || inputs != [ValueId(39)]
        || outputs != [ValueId(41)]
    {
        return Err(Error::Unsupported(
            "canonical47 actual call0/layer3/node32 QKV site only".into(),
        ));
    }
    Ok(())
}

pub(crate) fn model_packets(binding: &Canonical47ProjectionBinding) -> Result<Vec<Packet>> {
    binding.require_initial47_attention_qkv()?;
    if !SELECTED_MEMORY_RECIPE_REVIEWED {
        return Err(Error::Unsupported(
            "canonical47 QKV selected memory recipe pending ROOT review".into(),
        ));
    }
    Ok(raw_packets())
}

// Actual current packet25 SHA e399a515c3453fcd391248368ae341bf2dbe2d19b27bcc7d915cf1deeeb108ba.
// Only W/X/workspace16+120/deadC168 pointer bytes are replaced by owned fixups.
const MAIN: [u8; 192] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 2, 0, 0, 0, 0, 0, 64, 0, 250, 255, 255, 255, 255, 255, 64, 0, 254, 255, 255, 255, 255,
    255, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 172, 3, 0, 0, 0, 0, 0, 0, 172, 3, 0, 0,
    0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 20, 0, 0, 0, 20, 0, 0, 0, 20, 0, 0, 47, 0, 0, 0, 0, 4, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
];
// Actual reflected parameter0 is60B; gap60..63 is unobserved and NOT a parameter.
const REDUCER: [u8; 60] = [
    0, 0, 128, 63, 0, 0, 0, 0, 0, 20, 0, 0, 47, 0, 0, 0, 0, 20, 0, 0, 0, 20, 0, 0, 0, 20, 0, 0, 0,
    20, 0, 0, 1, 0, 0, 0, 0, 172, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
];

fn main_packet() -> Packet {
    Packet {
        image: Some(ProjectionImage::AttentionAmpere256x128Sm80),
        roles: vec![Binding::W, Binding::X, Binding::Y, Binding::Workspace],
        manifest: CubinKernelManifest {
            entry: "ampere_s16816gemm_bf16_256x128_ldg8_stages_32x3_tn".into(),
            binary_sm: 80,
            grid: [20, 1, 2],
            block: [256, 1, 1],
            dynamic_shared_bytes: 24576,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 10485760,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 96256,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 481280,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 8519680,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: MAIN.to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 10485760,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 8,
                    binding: 1,
                    binding_offset: 0,
                    extent: 96256,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 16,
                    binding: 3,
                    binding_offset: 0,
                    extent: 1925120,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 120,
                    binding: 3,
                    binding_offset: 0,
                    extent: 1925120,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 481280,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        },
    }
}

fn reducer_packet() -> Packet {
    let mut parameters = vec![CubinParameter {
        offset: 0,
        template: REDUCER.to_vec(),
    }];
    // Exactly20 reflected pointers; mode0 makes original host scalar96/104 unread.
    parameters.extend((64..=216).step_by(8).map(|offset| CubinParameter {
        offset,
        template: vec![0; 8],
    }));
    Packet {
        image: Some(ProjectionImage::Bf16SplitKReduceSm86),
        roles: vec![Binding::Y, Binding::Workspace],
        manifest: CubinKernelManifest {
            entry: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Eif13__nv_bfloat16fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(),
            binary_sm: 86, grid: [160, 3, 1], block: [32, 16, 1],
            dynamic_shared_bytes: 0, attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding { bytes: 481280, alignment: 16, io: CubinIo::Write },
                CubinBinding { bytes: 8519680, alignment: 16, io: CubinIo::Read },
            ],
            parameters,
            fixups: vec![
                CubinPointerFixup { parameter: 1, byte_offset: 0, binding: 1, binding_offset: 0, extent: 1925120, alignment: 16, io: CubinIo::Read },
                CubinPointerFixup { parameter: 2, byte_offset: 0, binding: 0, binding_offset: 0, extent: 481280, alignment: 16, io: CubinIo::Write },
                CubinPointerFixup { parameter: 3, byte_offset: 0, binding: 0, binding_offset: 0, extent: 481280, alignment: 16, io: CubinIo::Write },
                CubinPointerFixup { parameter: 4, byte_offset: 0, binding: 0, binding_offset: 0, extent: 481280, alignment: 16, io: CubinIo::Write },
            ],
        },
    }
}

fn raw_packets() -> Vec<Packet> {
    vec![main_packet(), reducer_packet()]
}

#[cfg(test)]
mod tests;

// Metadata-only inventory of the exact payload used after the genuine binding
// guard. This cfg(test) bridge creates no source/state/native authority.
#[cfg(test)]
pub(super) fn packets_for_root_inventory() -> Vec<Packet> {
    raw_packets()
}
