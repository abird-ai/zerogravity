//! Fixed original47 first QKVZ and BA+reducer. Source-selected packets only.
//! BA440: peer selected-branch/address worksheet; no generic440 parameter mapping.
use super::*;
// Finite memory recipe reviewed by ROOT; see canonical47 projection run,
// memory-recipe-acceptance-55.json. Arithmetic still requires native comparison.
const SELECTED_PROJECTION_MEMORY_SEMANTICS_CLOSED: bool = true;
pub(super) fn require_memory_semantics() -> Result<()> {
    if !SELECTED_PROJECTION_MEMORY_SEMANTICS_CLOSED {
        return Err(Error::Unsupported(
            "canonical47 selected projection memory semantics remain source-only".into(),
        ));
    }
    Ok(())
}
fn launch_0() -> Launch {
    Launch {
        image: ProjectionImage::AmpereBf16Gemm128x64Sm80,
        roles: vec![Binding::W, Binding::X, Binding::Y],
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn".into(),
            binary_sm: ProjectionImage::AmpereBf16Gemm128x64Sm80.binary_sm(),
            grid: [64, 1, 1],
            block: [64, 1, 1],
            dynamic_shared_bytes: 0,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 16777216,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 96256,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 770048,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128,
                    0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 64, 128, 252, 255, 255, 255, 255,
                    255, 64, 128, 254, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4,
                    0, 0, 0, 32, 0, 0, 0, 32, 0, 0, 0, 32, 0, 0, 47, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ],
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 16777216,
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
                    binding: 2,
                    binding_offset: 0,
                    extent: 770048,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 770048,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        },
    }
}
fn launch_1() -> Launch {
    Launch {
        image: ProjectionImage::Canonical47CutlassBf16Sm80,
        roles: vec![Binding::W,Binding::X,Binding::Y,Binding::Workspace],
        manifest: CubinKernelManifest {
            entry: "_ZN7cutlass7Kernel2I66cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8EEvNT_6ParamsE".into(),
            binary_sm: ProjectionImage::Canonical47CutlassBf16Sm80.binary_sm(),
            grid: [1, 1, 16], block: [128, 1, 1],
            dynamic_shared_bytes: 49152, attributes: CubinLaunchAttributes::None,
            bindings: vec![CubinBinding { bytes: 65536, alignment: 16, io: CubinIo::Read },CubinBinding { bytes: 96256, alignment: 16, io: CubinIo::Read },CubinBinding { bytes: 3008, alignment: 16, io: CubinIo::Write },CubinBinding { bytes: 8519680, alignment: 16, io: CubinIo::Write }],
            parameters: vec![CubinParameter { offset: 0, template: vec![47,0,0,0,32,0,0,0,0,4,0,0,1,0,0,0,1,0,0,0,16,0,0,0,0,0,0,0,1,0,0,0,16,0,0,0,64,0,0,0,224,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,64,192,255,255,255,255,255,255,64,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,64,192,255,255,255,255,255,255,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,4,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,127,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] }],
            fixups: vec![CubinPointerFixup { parameter: 0, byte_offset: 344, binding: 1, binding_offset: 0, extent: 96256, alignment: 16, io: CubinIo::Read },CubinPointerFixup { parameter: 0, byte_offset: 352, binding: 0, binding_offset: 0, extent: 65536, alignment: 16, io: CubinIo::Read },CubinPointerFixup { parameter: 0, byte_offset: 360, binding: 2, binding_offset: 0, extent: 3008, alignment: 16, io: CubinIo::Write },CubinPointerFixup { parameter: 0, byte_offset: 368, binding: 3, binding_offset: 0, extent: 48128, alignment: 16, io: CubinIo::Write }],
        },
    }
}
fn launch_2() -> Launch {
    Launch {
        image: ProjectionImage::Bf16SplitKReduceSm86,
        roles: vec![Binding::Y,Binding::Workspace],
        manifest: CubinKernelManifest {
            entry: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(),
            binary_sm: ProjectionImage::Bf16SplitKReduceSm86.binary_sm(),
            grid: [1, 3, 1], block: [32, 16, 1],
            dynamic_shared_bytes: 0, attributes: CubinLaunchAttributes::None,
            bindings: vec![CubinBinding { bytes: 3008, alignment: 16, io: CubinIo::Write },CubinBinding { bytes: 8519680, alignment: 16, io: CubinIo::Read }],
            parameters: vec![CubinParameter { offset: 0, template: vec![0,0,128,63,0,0,0,0,32,0,0,0,47,0,0,0,32,0,0,0,32,0,0,0,32,0,0,0,32,0,0,0,1,0,0,0,224,5,0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0] },CubinParameter { offset: 64, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 72, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 80, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 88, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 96, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 104, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 112, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 120, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 128, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 136, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 144, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 152, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 160, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 168, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 176, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 184, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 192, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 200, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 208, template: vec![0,0,0,0,0,0,0,0] },CubinParameter { offset: 216, template: vec![0,0,0,0,0,0,0,0] }],
            fixups: vec![CubinPointerFixup { parameter: 1, byte_offset: 0, binding: 1, binding_offset: 0, extent: 48128, alignment: 16, io: CubinIo::Read },CubinPointerFixup { parameter: 2, byte_offset: 0, binding: 0, binding_offset: 0, extent: 3008, alignment: 16, io: CubinIo::Write },CubinPointerFixup { parameter: 3, byte_offset: 0, binding: 0, binding_offset: 0, extent: 3008, alignment: 16, io: CubinIo::Write },CubinPointerFixup { parameter: 4, byte_offset: 0, binding: 0, binding_offset: 0, extent: 3008, alignment: 16, io: CubinIo::Write }],
        },
    }
}
pub(super) fn launches(rows: usize, role: Role) -> Result<Vec<Launch>> {
    match (rows, role) {
        (47, Role::Qkvz) => Ok(vec![launch_0()]),
        (47, Role::Ba) => Ok(vec![launch_1(), launch_2()]),
        _ => Err(Error::Unsupported(
            "closed original47 learned-projection leaf".into(),
        )),
    }
}
