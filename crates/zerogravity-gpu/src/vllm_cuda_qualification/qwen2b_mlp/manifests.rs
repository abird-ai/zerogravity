//! Finite original2B GateUp/Down and D6144 ACT; no old phase template retag.
use super::*;
mod packets;

// Finite source/SASS/resource proofs plus independently matched complete14 ABI records.
const MEMORY_PERMISSION: Option<()> = Some(());
pub(super) fn require_memory_semantics() -> Result<()> {
    MEMORY_PERMISSION.ok_or_else(|| {
        Error::Unsupported("2B MLP finite memory/resource/native join not materialized".into())
    })
}
fn bind(bytes: u64, io: CubinIo) -> CubinBinding {
    CubinBinding {
        bytes,
        alignment: 16,
        io,
    }
}
fn fix(
    parameter: usize,
    byte_offset: usize,
    binding: usize,
    extent: u64,
    io: CubinIo,
) -> CubinPointerFixup {
    CubinPointerFixup {
        parameter,
        byte_offset,
        binding,
        binding_offset: 0,
        extent,
        alignment: 16,
        io,
    }
}
fn launch_0() -> Launch {
    Launch {
        image: ProjectionImage::Bf16Gemv2TSm86,
        roles: vec![Binding::W, Binding::X, Binding::Y],
        manifest: CubinKernelManifest {
            entry: "_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi4ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_".into(),
            binary_sm: ProjectionImage::Bf16Gemv2TSm86.binary_sm(),
            grid: [1536,1,1], block: [128,1,1], dynamic_shared_bytes: 0,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![bind(50331648, CubinIo::Read), bind(4096, CubinIo::Read), bind(24576, CubinIo::Write)],
            parameters: vec![
                CubinParameter { offset: 0, template: packets::P0[0..152].to_vec() },
                CubinParameter { offset: 152, template: packets::P0[152..156].to_vec() },
                CubinParameter { offset: 156, template: packets::P0[156..160].to_vec() },
            ],
            fixups: vec![fix(0, 0, 0, 50331648, CubinIo::Read), fix(0, 16, 1, 4096, CubinIo::Read), fix(0, 32, 2, 24576, CubinIo::Write), fix(0, 48, 2, 24576, CubinIo::Write)],
        },
    }
}
fn launch_1() -> Launch {
    Launch {
        image: ProjectionImage::CutlassWmmaSm80,
        roles: vec![Binding::W, Binding::X, Binding::Y, Binding::Workspace],
        manifest: CubinKernelManifest {
            entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(),
            binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
            grid: [8,16,3], block: [32,1,1], dynamic_shared_bytes: 16896,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![bind(25165824, CubinIo::Read), bind(12288, CubinIo::Read), bind(4096, CubinIo::Write), bind(8519680, CubinIo::Write)],
            parameters: vec![
                CubinParameter { offset: 0, template: packets::P1[0..360].to_vec() },
            ],
            fixups: vec![fix(0, 280, 1, 12288, CubinIo::Read), fix(0, 288, 0, 25165824, CubinIo::Read), fix(0, 296, 2, 4096, CubinIo::Write), fix(0, 304, 3, 12288, CubinIo::Write)],
        },
    }
}
fn launch_2() -> Launch {
    Launch {
        image: ProjectionImage::Bf16SplitKReduceSm86,
        roles: vec![Binding::Y, Binding::Workspace],
        manifest: CubinKernelManifest {
            entry: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(),
            binary_sm: ProjectionImage::Bf16SplitKReduceSm86.binary_sm(),
            grid: [64,1,1], block: [32,16,1], dynamic_shared_bytes: 0,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![bind(4096, CubinIo::Write), bind(8519680, CubinIo::Read)],
            parameters: vec![
                CubinParameter { offset: 0, template: packets::P2[0..60].to_vec() },
                CubinParameter { offset: 64, template: packets::P2[64..72].to_vec() },
                CubinParameter { offset: 72, template: packets::P2[72..80].to_vec() },
                CubinParameter { offset: 80, template: packets::P2[80..88].to_vec() },
                CubinParameter { offset: 88, template: packets::P2[88..96].to_vec() },
                CubinParameter { offset: 96, template: packets::P2[96..104].to_vec() },
                CubinParameter { offset: 104, template: packets::P2[104..112].to_vec() },
                CubinParameter { offset: 112, template: packets::P2[112..120].to_vec() },
                CubinParameter { offset: 120, template: packets::P2[120..128].to_vec() },
                CubinParameter { offset: 128, template: packets::P2[128..136].to_vec() },
                CubinParameter { offset: 136, template: packets::P2[136..144].to_vec() },
                CubinParameter { offset: 144, template: packets::P2[144..152].to_vec() },
                CubinParameter { offset: 152, template: packets::P2[152..160].to_vec() },
                CubinParameter { offset: 160, template: packets::P2[160..168].to_vec() },
                CubinParameter { offset: 168, template: packets::P2[168..176].to_vec() },
                CubinParameter { offset: 176, template: packets::P2[176..184].to_vec() },
                CubinParameter { offset: 184, template: packets::P2[184..192].to_vec() },
                CubinParameter { offset: 192, template: packets::P2[192..200].to_vec() },
                CubinParameter { offset: 200, template: packets::P2[200..208].to_vec() },
                CubinParameter { offset: 208, template: packets::P2[208..216].to_vec() },
                CubinParameter { offset: 216, template: packets::P2[216..224].to_vec() },
            ],
            fixups: vec![fix(1, 0, 1, 12288, CubinIo::Read), fix(2, 0, 0, 4096, CubinIo::Write), fix(3, 0, 0, 4096, CubinIo::Write), fix(4, 0, 0, 4096, CubinIo::Write)],
        },
    }
}
fn launch_3() -> Launch {
    Launch {
        image: ProjectionImage::Canonical47CutlassBf16Sm80,
        roles: vec![Binding::W, Binding::X, Binding::Y],
        manifest: CubinKernelManifest {
            entry: "_ZN7cutlass7Kernel2I67cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8EEvNT_6ParamsE".into(),
            binary_sm: ProjectionImage::Canonical47CutlassBf16Sm80.binary_sm(),
            grid: [8,6,1], block: [128,1,1], dynamic_shared_bytes: 81920,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![bind(50331648, CubinIo::Read), bind(262144, CubinIo::Read), bind(1572864, CubinIo::Write)],
            parameters: vec![
                CubinParameter { offset: 0, template: packets::P3[0..440].to_vec() },
            ],
            fixups: vec![fix(0, 344, 1, 262144, CubinIo::Read), fix(0, 352, 0, 50331648, CubinIo::Read), fix(0, 360, 2, 1572864, CubinIo::Write), fix(0, 368, 2, 1572864, CubinIo::Write)],
        },
    }
}
fn launch_4() -> Launch {
    Launch {
        image: ProjectionImage::AmpereBf16Gemm128x64Sm80,
        roles: vec![Binding::W, Binding::X, Binding::Y],
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn".into(),
            binary_sm: ProjectionImage::AmpereBf16Gemm128x64Sm80.binary_sm(),
            grid: [16, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 24576,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                bind(25165824, CubinIo::Read),
                bind(786432, CubinIo::Read),
                bind(262144, CubinIo::Write),
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: packets::P4[0..192].to_vec(),
            }],
            fixups: vec![
                fix(0, 0, 0, 25165824, CubinIo::Read),
                fix(0, 8, 1, 786432, CubinIo::Read),
                fix(0, 16, 2, 262144, CubinIo::Write),
                fix(0, 168, 2, 262144, CubinIo::Write),
            ],
        },
    }
}
pub(super) fn launches(rows: usize, index: usize) -> Result<Vec<Launch>> {
    Ok(match (rows, index) {
        (1, 8) => vec![launch_0()],
        (1, 10) => vec![launch_1(), launch_2()],
        (64, 8) => vec![launch_3()],
        (64, 10) => vec![launch_4()],
        _ => {
            return Err(Error::Unsupported(
                "closed2B MLP initial1/64 nodes8/10 only".into(),
            ));
        }
    })
}
pub(super) fn activation(rows: usize) -> Result<CubinKernelManifest> {
    if !matches!(rows, 1 | 64) {
        return Err(Error::Unsupported("2B ACT initial1/64 only".into()));
    }
    // Separate selected D6144 template; existing D3584 factories remain unchanged.
    let template: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x18, 0, 0, 0, 0, 0, 0, 0, 0, 0x80,
        0x3f, 0, 0, 0, 0,
    ];
    Ok(CubinKernelManifest {
        entry: "_ZN4vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_bfloat162XadL_ZNS_11silu_kernelIS2_EET_RKS5_fEEXadL_ZNS_18packed_silu_kernelIS3_EES5_S7_fEELb1ELb1ELb0ELb0EEEvPS5_PS6_ifff".into(),
        binary_sm: 86, grid: [rows as u32,1,1], block: [768,1,1],
        dynamic_shared_bytes: 0, attributes: CubinLaunchAttributes::None,
        bindings: vec![bind((rows * 24_576) as u64,CubinIo::Read),
            bind((rows * 12_288) as u64,CubinIo::Write)],
        parameters: [(0,8),(8,8),(16,4),(20,4),(24,4),(28,4)].into_iter()
            .map(|(offset,n)| CubinParameter { offset, template: template[offset..offset+n].to_vec() }).collect(),
        fixups: vec![fix(0,0,1,(rows*12_288) as u64,CubinIo::Write),
            fix(1,0,0,(rows*24_576) as u64,CubinIo::Read)],
    })
}
#[cfg(test)]
mod tests;
