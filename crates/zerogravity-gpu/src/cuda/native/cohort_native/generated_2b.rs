//! Generated fixed_promptB4 ABI; physical execution/math not qualified.
use super::super::cubin_manifests::ProjectionImage;
use super::packets::{Definition, Packet, Role};
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup,
};
use zerogravity_core::{Error, Result};
// Whole compiler input SHA256: 13f9aff53f2cedcdbf35d18d0602df5fe1e76b796ecd6ea9f3fdd2cc7c5a587c
pub(super) const METADATA_BOUND: usize = 13542960;
pub(super) const DEFINITION_COUNT: usize = 105;
#[inline(never)]
fn packet_0() -> Definition {
    Definition{image:Some(ProjectionImage::CutlassWmmaSm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8,64,1],block:[32,1,1],dynamic_shared_bytes:8704,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:33554432,alignment:16,io:CubinIo::Read},CubinBinding{bytes:65536,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[4,0,0,0,0,32,0,0,0,8,0,0,1,0,0,0,0,2,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,33,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,33,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,64,254,255,255,255,255,255,0,64,254,255,255,255,255,255,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,64,254,255,255,255,255,255,0,64,254,255,255,255,255,255,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,128,63,0,0,0,0,255,255,255,255,197,112,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:280,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:288,extent:33554432,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:296,extent:65536,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:304,extent:65536,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_1() -> Definition {
    Definition{image:Some(ProjectionImage::CutlassWmmaSm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[2,1,1],block:[32,1,1],dynamic_shared_bytes:16896,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:131072,alignment:16,io:CubinIo::Read},CubinBinding{bytes:256,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[4,0,0,0,32,0,0,0,0,8,0,0,1,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,33,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,33,255,255,255,255,255,255,0,1,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,64,254,255,255,255,255,255,255,64,254,255,255,255,255,255,255,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,64,254,255,255,255,255,255,255,64,254,255,255,255,255,255,255,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,128,63,0,0,0,0,255,255,255,255,197,112,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:280,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:288,extent:131072,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:296,extent:256,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:304,extent:256,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_2() -> Definition {
    Definition {
        image: Some(ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80),
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn".into(),
            binary_sm: 80,
            grid: [32, 1, 2],
            block: [128, 1, 1],
            dynamic_shared_bytes: 32768,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 8388608,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 16384,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 16384,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 8519680,
                    alignment: 16,
                    io: CubinIo::ReadWrite,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 128, 0, 253, 255, 255, 255, 255, 255,
                    128, 0, 253, 255, 255, 255, 255, 255, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0,
                    0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
                .to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    alignment: 16,
                    binding: 0,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 8388608,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 8,
                    extent: 16384,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 16,
                    extent: 16384,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 168,
                    extent: 16384,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 3,
                    binding_offset: 0,
                    byte_offset: 120,
                    extent: 128,
                    io: CubinIo::ReadWrite,
                    parameter: 0,
                },
            ],
        },
    }
}
#[inline(never)]
fn packet_3() -> Definition {
    Definition{image:Some(ProjectionImage::FixedPromptCutlassBf16Sm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I67cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x10_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8,24,1],block:[128,1,1],dynamic_shared_bytes:81920,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:50331648,alignment:16,io:CubinIo::Read},CubinBinding{bytes:98304,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[4,0,0,0,0,48,0,0,0,8,0,0,1,0,0,0,192,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,0,0,0,0,0,0,64,128,255,255,255,255,255,255,64,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,0,0,0,0,0,0,64,128,255,255,255,255,255,255,64,0,0,0,0,0,0,0,0,96,0,0,0,0,0,0,0,128,1,0,0,0,0,0,0,96,0,0,0,0,0,0,0,96,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,12,0,0,0,0,0,0,0,24,0,0,0,0,0,0,0,6,0,0,0,0,0,0,96,0,0,0,0,0,0,0,128,1,0,0,0,0,0,0,96,0,0,0,0,0,0,0,96,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,12,0,0,0,0,0,0,0,24,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,127,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:344,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:352,extent:50331648,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:360,extent:98304,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:368,extent:98304,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_4() -> Definition {
    Definition {
        image: Some(ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80),
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn".into(),
            binary_sm: 80,
            grid: [32, 1, 2],
            block: [128, 1, 1],
            dynamic_shared_bytes: 32768,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 25165824,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 49152,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 16384,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 8519680,
                    alignment: 16,
                    io: CubinIo::ReadWrite,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 128, 0, 247, 255, 255, 255, 255, 255,
                    128, 0, 247, 255, 255, 255, 255, 255, 0, 12, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 24, 0,
                    0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 4, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
                .to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    alignment: 16,
                    binding: 0,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 25165824,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 8,
                    extent: 49152,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 16,
                    extent: 16384,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 168,
                    extent: 16384,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 3,
                    binding_offset: 0,
                    byte_offset: 120,
                    extent: 128,
                    io: CubinIo::ReadWrite,
                    parameter: 0,
                },
            ],
        },
    }
}
#[inline(never)]
fn packet_5() -> Definition {
    Definition {
        image: Some(ProjectionImage::Cohort2BAmpere64SlicedSm80),
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn".into(),
            binary_sm: 80,
            grid: [80, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 49152,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 20971520,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 16384,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 40960,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 128, 0, 253, 255, 255, 255, 255, 255,
                    128, 0, 253, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0,
                    0, 20, 0, 0, 0, 20, 0, 0, 0, 20, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
                .to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    alignment: 16,
                    binding: 0,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 20971520,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 8,
                    extent: 16384,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 16,
                    extent: 40960,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 168,
                    extent: 40960,
                    io: CubinIo::Write,
                    parameter: 0,
                },
            ],
        },
    }
}
#[inline(never)]
fn packet_6() -> Definition {
    Definition {
        image: Some(ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80),
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn".into(),
            binary_sm: 80,
            grid: [3880, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 49152,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 1017118720,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 16384,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 1986560,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 128, 0, 253, 255, 255, 255, 255, 255,
                    128, 0, 253, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0,
                    0, 202, 3, 0, 0, 202, 3, 0, 0, 202, 3, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
                .to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    alignment: 16,
                    binding: 0,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 1017118720,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 8,
                    extent: 16384,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 16,
                    extent: 1986560,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 168,
                    extent: 1986560,
                    io: CubinIo::Write,
                    parameter: 0,
                },
            ],
        },
    }
}
#[inline(never)]
fn packet_7() -> Definition {
    Definition{image:None,manifest:CubinKernelManifest{entry:"_ZN4vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_bfloat162XadL_ZNS_11silu_kernelIS2_EET_RKS5_fEEXadL_ZNS_18packed_silu_kernelIS3_EES5_S7_fEELb1ELb1ELb0ELb0EEEvPS5_PS6_ifff".into(),binary_sm:86,grid:[4,1,1],block:[768,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:49152,alignment:16,io:CubinIo::Write},CubinBinding{bytes:98304,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:16,template:[0,24,0,0].to_vec()},CubinParameter{offset:20,template:[0,0,0,0].to_vec()},CubinParameter{offset:24,template:[0,0,128,63].to_vec()},CubinParameter{offset:28,template:[0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:49152,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:0,extent:98304,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_8() -> Definition {
    Definition {
        image: Some(ProjectionImage::AttentionQkSm86),
        manifest: CubinKernelManifest {
            entry: "_fused_qk_rmsnorm_rope_gate_kernel".into(),
            binary_sm: 86,
            grid: [4, 10, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 16,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 40960,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 16384,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 4096,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 16384,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 10112,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 12312,
                    alignment: 16,
                    io: CubinIo::Read,
                },
            ],
            parameters: vec![
                CubinParameter {
                    offset: 0,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 8,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 16,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 24,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 32,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 40,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 48,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 56,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 64,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 72,
                    template: [0, 20, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 76,
                    template: [0, 20, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 80,
                    template: [0, 8, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 84,
                    template: [0, 2, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 88,
                    template: [0, 8, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 92,
                    template: [64, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 96,
                    template: [1, 2, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 104,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
                CubinParameter {
                    offset: 112,
                    template: [0, 0, 0, 0, 0, 0, 0, 0].to_vec(),
                },
            ],
            fixups: vec![
                CubinPointerFixup {
                    alignment: 16,
                    binding: 0,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 38912,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 0,
                    binding_offset: 8192,
                    byte_offset: 0,
                    extent: 31744,
                    io: CubinIo::Read,
                    parameter: 1,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 16384,
                    io: CubinIo::Write,
                    parameter: 2,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 4096,
                    io: CubinIo::Write,
                    parameter: 3,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 3,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 16384,
                    io: CubinIo::Write,
                    parameter: 4,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 4,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 512,
                    io: CubinIo::Read,
                    parameter: 5,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 5,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 512,
                    io: CubinIo::Read,
                    parameter: 6,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 6,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 6912,
                    io: CubinIo::Read,
                    parameter: 7,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 7,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 12312,
                    io: CubinIo::Read,
                    parameter: 8,
                },
            ],
        },
    }
}
#[inline(never)]
fn packet_9() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionKvSm86),manifest:CubinKernelManifest{entry:"_ZN4vllm30reshape_and_cache_flash_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKlllllliiiPKfSB_i".into(),binary_sm:86,grid:[4,1,1],block:[512,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:40960,alignment:16,io:CubinIo::Read},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::ReadWrite},CubinBinding{bytes:32,alignment:8,io:CubinIo::Read},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:16,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:24,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:32,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:40,template:[0,128,8,0,0,0,0,0].to_vec()},CubinParameter{offset:48,template:[0,4,0,0,0,0,0,0].to_vec()},CubinParameter{offset:56,template:[0,2,0,0,0,0,0,0].to_vec()},CubinParameter{offset:64,template:[0,2,0,0,0,0,0,0].to_vec()},CubinParameter{offset:72,template:[0,20,0,0,0,0,0,0].to_vec()},CubinParameter{offset:80,template:[2,0,0,0].to_vec()},CubinParameter{offset:84,template:[0,1,0,0].to_vec()},CubinParameter{offset:88,template:[32,2,0,0].to_vec()},CubinParameter{offset:96,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:104,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:112,template:[0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:4096,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:9216,byte_offset:0,extent:31744,io:CubinIo::Read,parameter:1},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:0,extent:18939904,io:CubinIo::ReadWrite,parameter:2},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:0,extent:18939392,io:CubinIo::ReadWrite,parameter:3},CubinPointerFixup{alignment:8,binding:3,binding_offset:0,byte_offset:0,extent:32,io:CubinIo::Read,parameter:4},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:0,extent:4,io:CubinIo::Read,parameter:13},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:0,extent:4,io:CubinIo::Read,parameter:14}]}}
}
#[inline(never)]
fn packet_10() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,8,0,0,0,4,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_11() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,48,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,169,3,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_12() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,253,127,0,0,174,62,249,68,1,0,0,0,232,214,245,212,253,127,0,0,96,79,154,16,0,0,0,0,16,85,86,11,0,0,0,0,160,217,245,212,253,127,0,0,0,128,1,0,0,0,0,0,127,215,245,212,253,127,0,0,176,215,245,212,253,127,0,0,3,58,162,11,197,112,0,0,0,0,0,0,0,0,0,0,208,121,45,16,0,0,0,0,96,79,154,16,0,0,0,0,112,182,34,16,0,0,0,0,112,182,34,16,0,0,0,0,40,250,45,16,0,0,0,0,32,216,245,212,253,127,0,0,61,67,162,11,197,112,0,0,0,0,0,0,48,228,0,0,196,237,86,11,0,0,0,0,240,83,50,16,0,0,0,0,16,85,86,11,0,0,0,0,0,0,0,0,0,0,0,0,0,56,212,16,0,0,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,1,0,217,245,212,253,127,0,0,192,237,59,72,0,0,0,0,48,23,172,60,0,0,0,0,0,102,129,148,105,125,248,63,80,50,212,16,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,0,0,0,0,0,0,0,148,123,72,0,2,0,0,0,96,79,154,16,0,0,0,0,174,62,249,68,197,112,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,112,83,49,16,0,0,0,0,192,94,54,72,0,0,0,0,0,201,0,0,0,0,0,0,192,220,245,212,253,127,0,0,231,90,189,11,197,112,0,0,96,79,154,16,0,0,0,0,0,120,31,0,0,0,0,0,208,187,166,69,0,0,0,0,240,228,218,16,0,0,0,0,144,187,166,69,0,0,0,0,255,0,0,0,0,0,0,0,32,181,80,69,0,0,0,0,0,136,0,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_13() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,27,73,15,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,78,158,192,16,197,112,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,83,76,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,137,44,187,11,197,112,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,197,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_14() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionSigmoidSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_19sigmoid_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE0_clEvENKUlvE2_clEvEUlN3c108BFloat16EE_St5arrayIPcLm2EEEEviT0_T1_".into(),binary_sm:86,grid:[8,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:4,template:[0].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Write,parameter:2},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:8,extent:16384,io:CubinIo::Read,parameter:2}]}}
}
#[inline(never)]
fn packet_15() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionMultiplySm86),manifest:CubinKernelManifest{entry:"_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIN3c108BFloat16ES4_S4_NS0_15binary_internal10MulFunctorIfEEEESt5arrayIPcLm3EEEEviT0_T1_".into(),binary_sm:86,grid:[8,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:4,template:[15].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Write,parameter:2},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:8,extent:16384,io:CubinIo::Read,parameter:2},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:16,extent:16384,io:CubinIo::Read,parameter:2}]}}
}
#[inline(never)]
fn packet_16() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,144,53,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,96,54,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,92,32,75,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_17() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,133,85,15,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,230,84,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_18() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,32,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,240,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,156,165,77,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_19() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,238,97,15,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,120,93,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_20() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,48,55,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,0,56,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,220,42,80,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_21() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,87,110,15,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,10,102,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_22() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,192,52,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,144,53,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,176,207,65,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_23() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,192,122,15,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,156,110,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_24() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,80,50,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,32,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,240,84,68,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_25() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,41,135,15,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,46,119,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_26() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,49,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,169,3,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_27() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,80,50,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,32,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,24,117,71,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_28() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,118,88,16,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,47,129,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_29() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,96,54,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,48,55,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,88,250,73,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_30() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,223,100,16,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,193,137,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,170,190,72,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_31() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,8,0,0,0,4,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_32() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,240,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,192,52,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,152,127,76,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_33() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,72,113,16,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,83,146,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_34() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,0,56,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,80,50,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,216,4,79,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_35() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,177,125,16,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,229,154,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_36() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,144,53,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,96,54,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,208,227,65,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_37() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,26,138,16,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,119,163,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_38() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,32,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,240,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,16,105,68,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_39() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,131,150,16,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,9,172,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_40() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,50,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,169,3,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_41() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,32,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,240,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,56,137,71,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_42() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,175,103,17,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,10,182,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_43() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,48,55,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,0,56,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,120,14,74,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_44() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,24,116,17,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,156,190,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_45() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,192,52,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,144,53,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,184,147,76,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_46() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,129,128,17,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,46,199,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_47() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,80,50,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,32,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,248,24,79,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_48() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,234,140,17,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,192,207,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_49() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,96,54,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,48,55,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,64,254,65,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_50() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,83,153,17,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,82,216,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_51() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,240,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,192,52,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,128,131,68,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_52() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,188,165,17,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,228,224,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_53() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,51,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,169,3,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_54() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,240,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,192,52,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,168,163,71,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_55() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,230,118,18,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,229,234,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_56() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,0,56,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,80,50,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,232,40,74,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_57() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,79,131,18,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,119,243,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_58() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,144,53,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,96,54,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,40,174,76,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_59() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,184,143,18,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,9,252,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_60() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,32,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,240,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,104,51,79,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_61() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,33,156,18,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,155,4,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_62() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,48,55,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,0,56,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,8,27,66,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_63() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,138,168,18,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,45,13,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_64() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,192,52,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,144,53,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,72,160,68,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_65() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,243,180,18,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,191,21,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_66() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,52,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,169,3,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_67() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,192,52,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,144,53,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,112,192,71,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_68() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,31,134,19,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,192,31,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_69() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,80,50,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,32,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,176,69,74,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_70() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,136,146,19,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,82,40,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_71() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,96,54,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,48,55,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,240,202,76,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_72() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,241,158,19,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,228,48,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_73() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,240,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,192,52,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,48,80,79,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_74() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,90,171,19,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,118,57,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_75() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,0,56,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,80,50,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,28,47,66,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_76() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,195,183,19,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,8,66,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_77() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,144,53,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,96,54,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,92,180,68,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_78() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,44,196,19,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,154,74,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_79() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,53,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,169,3,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_80() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,144,53,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,96,54,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,132,212,71,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_81() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,86,149,20,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,155,84,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_82() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,32,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,240,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,196,89,74,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_83() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,191,161,20,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,45,93,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_84() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,48,55,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,0,56,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,4,223,76,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_85() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,40,174,20,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,191,101,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_86() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,192,52,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,144,53,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,68,100,79,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_87() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,145,186,20,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,81,110,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_88() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,80,50,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,32,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,156,68,66,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_89() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,250,198,20,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,227,118,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_90() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,96,54,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,48,55,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,220,201,68,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_91() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,99,211,20,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,117,127,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_92() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,54,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,169,3,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_93() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,96,54,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,48,55,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,4,234,71,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_94() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,143,164,21,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,118,137,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_95() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,240,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,192,52,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,68,111,74,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_96() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,248,176,21,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,8,146,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_97() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,0,56,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,80,50,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,132,244,76,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_98() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,97,189,21,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,154,154,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_99() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,144,53,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,96,54,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,196,121,79,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_100() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,202,201,21,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,44,163,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_101() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,32,51,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,240,51,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,236,94,66,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_102() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,51,214,21,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,190,171,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_103() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,232,214,245,212,253,127,0,0,200,214,245,212,253,127,0,0,109,202,146,16,197,112,0,0,224,214,245,212,253,127,0,0,112,182,34,16,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,204,214,245,212,253,127,0,0,0,0,0,0,0,0,0,0,128,141,228,2,0,0,0,0,80,28,237,2,0,0,0,0,16,217,245,212,253,127,0,0,160,215,245,212,253,127,0,0,89,194,136,67,197,112,0,0,80,28,237,2,0,0,0,0,103,156,130,67,197,112,0,0,240,215,245,212,253,127,0,0,246,18,129,67,197,112,0,0,0,0,0,0,0,0,0,0,152,28,237,2,0,0,0,0,0,0,0,0,0,0,0,0,48,55,212,16,0,0,0,0,16,217,245,212,253,127,0,0,0,0,0,0,0,0,0,0,0,216,245,212,253,127,0,0,229,131,192,67,197,112,0,0,48,23,172,60,0,0,0,0,0,18,133,67,197,112,0,0,0,56,212,16,0,0,0,0,0,0,0,0,197,112,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,188,223,141,0,0,0,0,44,228,68,0,2,0,0,0,96,79,154,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,38,16,0,0,0,0,64,217,245,212,91,1,0,0,189,229,111,38,196,112,0,0,184,221,111,38,196,112,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_104() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,16,197,112,0,0,96,208,245,212,253,127,0,0,0,208,245,212,253,127,0,0,115,212,104,34,0,0,0,0,112,182,34,16,0,0,0,0,1,0,0,0,253,127,0,0,24,208,245,212,253,127,0,0,156,226,21,0,253,127,0,0,9,118,155,11,197,112,0,0,160,84,31,16,0,0,0,0,51,1,0,0,0,0,0,0,160,208,245,212,253,127,0,0,176,133,110,7,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,224,207,245,212,253,127,0,0,6,254,58,253,196,112,0,0,192,165,10,35,0,0,0,0,160,84,31,16,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,253,196,112,0,0,48,193,55,16,0,0,0,0,64,129,26,16,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,224,192,146,16,197,112,0,0,160,208,245,212,253,127,0,0,192,165,10,35,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,96,211,245,212,253,127,0,0,249,44,184,11,197,112,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,112,182,34,16,0,0,0,0,48,193,55,16,0,0,0,0,80,180,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,165,10,35,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,216,245,212,253,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,37,170,71,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
pub(super) fn definitions() -> Vec<Definition> {
    vec![
        packet_0(),
        packet_1(),
        packet_2(),
        packet_3(),
        packet_4(),
        packet_5(),
        packet_6(),
        packet_7(),
        packet_8(),
        packet_9(),
        packet_10(),
        packet_11(),
        packet_12(),
        packet_13(),
        packet_14(),
        packet_15(),
        packet_16(),
        packet_17(),
        packet_18(),
        packet_19(),
        packet_20(),
        packet_21(),
        packet_22(),
        packet_23(),
        packet_24(),
        packet_25(),
        packet_26(),
        packet_27(),
        packet_28(),
        packet_29(),
        packet_30(),
        packet_31(),
        packet_32(),
        packet_33(),
        packet_34(),
        packet_35(),
        packet_36(),
        packet_37(),
        packet_38(),
        packet_39(),
        packet_40(),
        packet_41(),
        packet_42(),
        packet_43(),
        packet_44(),
        packet_45(),
        packet_46(),
        packet_47(),
        packet_48(),
        packet_49(),
        packet_50(),
        packet_51(),
        packet_52(),
        packet_53(),
        packet_54(),
        packet_55(),
        packet_56(),
        packet_57(),
        packet_58(),
        packet_59(),
        packet_60(),
        packet_61(),
        packet_62(),
        packet_63(),
        packet_64(),
        packet_65(),
        packet_66(),
        packet_67(),
        packet_68(),
        packet_69(),
        packet_70(),
        packet_71(),
        packet_72(),
        packet_73(),
        packet_74(),
        packet_75(),
        packet_76(),
        packet_77(),
        packet_78(),
        packet_79(),
        packet_80(),
        packet_81(),
        packet_82(),
        packet_83(),
        packet_84(),
        packet_85(),
        packet_86(),
        packet_87(),
        packet_88(),
        packet_89(),
        packet_90(),
        packet_91(),
        packet_92(),
        packet_93(),
        packet_94(),
        packet_95(),
        packet_96(),
        packet_97(),
        packet_98(),
        packet_99(),
        packet_100(),
        packet_101(),
        packet_102(),
        packet_103(),
        packet_104(),
    ]
}
pub(super) fn site(call: usize, node: usize) -> Result<Vec<Packet>> {
    Ok(match (call, node) {
        (1, 2) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 3) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 6) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 8) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 9) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 10) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 12) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 13) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 16) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 18) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 19) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 20) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 22) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 23) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 26) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 28) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 29) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 30) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 32) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (1, 33) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (1, 34) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (1, 35) => vec![
            Packet {
                definition: 10,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 11,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 12,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 13,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (1, 36) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 37) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 38) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 40) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 41) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 42) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 44) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 45) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 48) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 50) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 51) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 52) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 54) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 55) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 58) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 60) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 61) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 62) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 64) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 65) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 68) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 70) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 71) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 72) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 74) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (1, 75) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (1, 76) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (1, 77) => vec![
            Packet {
                definition: 10,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 11,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 16,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 17,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (1, 78) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 79) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 80) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 82) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 83) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 84) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 86) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 87) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 90) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 92) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 93) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 94) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 96) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 97) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 100) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 102) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 103) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 104) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 106) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 107) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 110) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 112) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 113) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 114) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 116) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (1, 117) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (1, 118) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (1, 119) => vec![
            Packet {
                definition: 10,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 11,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 18,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 19,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (1, 120) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 121) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 122) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 124) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 125) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 126) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 128) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 129) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 132) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 134) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 135) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 136) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 138) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 139) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 142) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 144) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 145) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 146) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 148) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 149) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 152) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 154) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 155) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 156) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 158) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (1, 159) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (1, 160) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (1, 161) => vec![
            Packet {
                definition: 10,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 11,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 20,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 21,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (1, 162) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 163) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 164) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 166) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 167) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 168) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 170) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 171) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 174) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 176) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 177) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 178) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 180) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 181) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 184) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 186) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 187) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 188) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 190) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 191) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 194) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 196) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 197) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 198) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 200) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (1, 201) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (1, 202) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (1, 203) => vec![
            Packet {
                definition: 10,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 11,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 22,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 23,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (1, 204) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 205) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 206) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 208) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 209) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 210) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 212) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 213) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 216) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 218) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 219) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 220) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 222) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 223) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 226) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 228) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 229) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 230) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 232) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 233) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 236) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 238) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 239) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 240) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 242) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (1, 243) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (1, 244) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (1, 245) => vec![
            Packet {
                definition: 10,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 11,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 24,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 25,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (1, 246) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 247) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 248) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 250) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 251) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 252) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (1, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (2, 2) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 3) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 6) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 8) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 9) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 10) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 12) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 13) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 16) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 18) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 19) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 20) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 22) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 23) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 26) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 28) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 29) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 30) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 32) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (2, 33) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (2, 34) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (2, 35) => vec![
            Packet {
                definition: 10,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 26,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 27,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 28,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (2, 36) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 37) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 38) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 40) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 41) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 42) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 44) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 45) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 48) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 50) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 51) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 52) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 54) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 55) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 58) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 60) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 61) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 62) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 64) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 65) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 68) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 70) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 71) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 72) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 74) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (2, 75) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (2, 76) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (2, 77) => vec![
            Packet {
                definition: 10,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 26,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 29,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 30,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (2, 78) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 79) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 80) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 82) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 83) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 84) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 86) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 87) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 90) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 92) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 93) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 94) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 96) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 97) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 100) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 102) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 103) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 104) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 106) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 107) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 110) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 112) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 113) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 114) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 116) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (2, 117) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (2, 118) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (2, 119) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 26,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 32,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 33,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (2, 120) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 121) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 122) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 124) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 125) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 126) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 128) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 129) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 132) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 134) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 135) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 136) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 138) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 139) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 142) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 144) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 145) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 146) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 148) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 149) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 152) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 154) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 155) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 156) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 158) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (2, 159) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (2, 160) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (2, 161) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 26,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 34,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 35,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (2, 162) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 163) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 164) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 166) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 167) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 168) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 170) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 171) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 174) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 176) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 177) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 178) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 180) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 181) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 184) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 186) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 187) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 188) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 190) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 191) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 194) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 196) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 197) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 198) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 200) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (2, 201) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (2, 202) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (2, 203) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 26,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 36,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 37,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (2, 204) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 205) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 206) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 208) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 209) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 210) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 212) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 213) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 216) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 218) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 219) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 220) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 222) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 223) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 226) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 228) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 229) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 230) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 232) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 233) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 236) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 238) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 239) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 240) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 242) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (2, 243) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (2, 244) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (2, 245) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 26,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 38,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 39,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (2, 246) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 247) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 248) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 250) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 251) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 252) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (2, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (3, 2) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 3) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 6) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 8) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 9) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 10) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 12) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 13) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 16) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 18) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 19) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 20) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 22) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 23) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 26) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 28) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 29) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 30) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 32) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (3, 33) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (3, 34) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (3, 35) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 40,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 41,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 42,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (3, 36) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 37) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 38) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 40) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 41) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 42) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 44) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 45) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 48) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 50) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 51) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 52) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 54) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 55) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 58) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 60) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 61) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 62) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 64) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 65) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 68) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 70) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 71) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 72) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 74) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (3, 75) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (3, 76) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (3, 77) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 40,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 43,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 44,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (3, 78) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 79) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 80) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 82) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 83) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 84) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 86) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 87) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 90) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 92) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 93) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 94) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 96) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 97) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 100) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 102) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 103) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 104) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 106) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 107) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 110) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 112) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 113) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 114) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 116) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (3, 117) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (3, 118) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (3, 119) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 40,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 45,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 46,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (3, 120) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 121) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 122) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 124) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 125) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 126) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 128) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 129) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 132) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 134) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 135) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 136) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 138) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 139) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 142) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 144) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 145) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 146) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 148) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 149) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 152) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 154) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 155) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 156) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 158) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (3, 159) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (3, 160) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (3, 161) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 40,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 47,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 48,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (3, 162) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 163) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 164) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 166) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 167) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 168) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 170) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 171) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 174) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 176) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 177) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 178) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 180) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 181) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 184) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 186) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 187) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 188) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 190) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 191) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 194) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 196) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 197) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 198) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 200) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (3, 201) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (3, 202) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (3, 203) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 40,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 49,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 50,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (3, 204) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 205) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 206) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 208) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 209) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 210) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 212) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 213) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 216) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 218) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 219) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 220) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 222) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 223) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 226) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 228) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 229) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 230) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 232) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 233) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 236) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 238) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 239) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 240) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 242) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (3, 243) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (3, 244) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (3, 245) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 40,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 51,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 52,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (3, 246) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 247) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 248) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 250) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 251) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 252) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (3, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (4, 2) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 3) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 6) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 8) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 9) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 10) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 12) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 13) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 16) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 18) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 19) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 20) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 22) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 23) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 26) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 28) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 29) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 30) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 32) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (4, 33) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (4, 34) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (4, 35) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 53,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 54,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 55,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (4, 36) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 37) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 38) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 40) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 41) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 42) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 44) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 45) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 48) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 50) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 51) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 52) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 54) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 55) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 58) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 60) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 61) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 62) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 64) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 65) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 68) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 70) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 71) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 72) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 74) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (4, 75) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (4, 76) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (4, 77) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 53,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 56,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 57,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (4, 78) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 79) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 80) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 82) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 83) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 84) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 86) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 87) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 90) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 92) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 93) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 94) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 96) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 97) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 100) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 102) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 103) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 104) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 106) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 107) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 110) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 112) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 113) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 114) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 116) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (4, 117) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (4, 118) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (4, 119) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 53,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 58,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 59,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (4, 120) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 121) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 122) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 124) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 125) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 126) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 128) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 129) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 132) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 134) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 135) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 136) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 138) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 139) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 142) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 144) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 145) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 146) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 148) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 149) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 152) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 154) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 155) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 156) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 158) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (4, 159) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (4, 160) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (4, 161) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 53,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 60,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 61,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (4, 162) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 163) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 164) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 166) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 167) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 168) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 170) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 171) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 174) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 176) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 177) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 178) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 180) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 181) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 184) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 186) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 187) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 188) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 190) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 191) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 194) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 196) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 197) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 198) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 200) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (4, 201) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (4, 202) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (4, 203) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 53,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 62,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 63,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (4, 204) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 205) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 206) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 208) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 209) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 210) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 212) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 213) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 216) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 218) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 219) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 220) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 222) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 223) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 226) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 228) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 229) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 230) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 232) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 233) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 236) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 238) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 239) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 240) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 242) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (4, 243) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (4, 244) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (4, 245) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 53,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 64,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 65,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (4, 246) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 247) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 248) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 250) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 251) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 252) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (4, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (5, 2) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 3) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 6) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 8) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 9) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 10) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 12) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 13) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 16) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 18) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 19) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 20) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 22) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 23) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 26) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 28) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 29) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 30) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 32) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (5, 33) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (5, 34) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (5, 35) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 66,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 67,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 68,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (5, 36) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 37) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 38) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 40) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 41) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 42) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 44) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 45) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 48) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 50) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 51) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 52) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 54) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 55) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 58) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 60) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 61) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 62) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 64) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 65) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 68) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 70) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 71) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 72) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 74) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (5, 75) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (5, 76) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (5, 77) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 66,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 69,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 70,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (5, 78) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 79) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 80) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 82) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 83) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 84) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 86) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 87) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 90) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 92) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 93) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 94) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 96) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 97) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 100) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 102) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 103) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 104) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 106) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 107) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 110) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 112) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 113) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 114) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 116) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (5, 117) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (5, 118) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (5, 119) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 66,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 71,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 72,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (5, 120) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 121) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 122) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 124) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 125) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 126) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 128) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 129) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 132) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 134) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 135) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 136) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 138) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 139) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 142) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 144) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 145) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 146) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 148) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 149) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 152) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 154) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 155) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 156) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 158) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (5, 159) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (5, 160) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (5, 161) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 66,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 73,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 74,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (5, 162) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 163) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 164) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 166) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 167) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 168) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 170) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 171) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 174) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 176) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 177) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 178) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 180) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 181) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 184) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 186) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 187) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 188) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 190) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 191) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 194) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 196) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 197) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 198) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 200) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (5, 201) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (5, 202) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (5, 203) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 66,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 75,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 76,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (5, 204) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 205) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 206) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 208) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 209) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 210) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 212) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 213) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 216) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 218) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 219) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 220) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 222) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 223) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 226) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 228) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 229) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 230) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 232) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 233) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 236) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 238) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 239) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 240) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 242) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (5, 243) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (5, 244) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (5, 245) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 66,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 77,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 78,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (5, 246) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 247) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 248) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 250) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 251) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 252) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (5, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (6, 2) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 3) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 6) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 8) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 9) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 10) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 12) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 13) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 16) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 18) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 19) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 20) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 22) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 23) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 26) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 28) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 29) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 30) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 32) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (6, 33) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (6, 34) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (6, 35) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 79,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 80,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 81,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (6, 36) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 37) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 38) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 40) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 41) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 42) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 44) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 45) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 48) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 50) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 51) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 52) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 54) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 55) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 58) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 60) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 61) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 62) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 64) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 65) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 68) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 70) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 71) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 72) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 74) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (6, 75) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (6, 76) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (6, 77) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 79,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 82,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 83,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (6, 78) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 79) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 80) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 82) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 83) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 84) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 86) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 87) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 90) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 92) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 93) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 94) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 96) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 97) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 100) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 102) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 103) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 104) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 106) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 107) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 110) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 112) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 113) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 114) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 116) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (6, 117) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (6, 118) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (6, 119) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 79,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 84,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 85,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (6, 120) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 121) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 122) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 124) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 125) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 126) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 128) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 129) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 132) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 134) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 135) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 136) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 138) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 139) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 142) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 144) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 145) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 146) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 148) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 149) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 152) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 154) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 155) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 156) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 158) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (6, 159) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (6, 160) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (6, 161) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 79,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 86,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 87,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (6, 162) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 163) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 164) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 166) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 167) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 168) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 170) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 171) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 174) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 176) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 177) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 178) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 180) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 181) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 184) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 186) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 187) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 188) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 190) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 191) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 194) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 196) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 197) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 198) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 200) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (6, 201) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (6, 202) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (6, 203) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 79,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 88,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 89,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (6, 204) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 205) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 206) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 208) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 209) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 210) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 212) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 213) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 216) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 218) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 219) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 220) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 222) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 223) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 226) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 228) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 229) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 230) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 232) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 233) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 236) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 238) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 239) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 240) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 242) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (6, 243) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (6, 244) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (6, 245) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 79,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 90,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 91,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (6, 246) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 247) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 248) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 250) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 251) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 252) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (6, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (7, 2) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 3) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 6) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 8) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 9) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 10) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 12) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 13) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 16) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 18) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 19) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 20) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 22) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 23) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 26) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 28) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 29) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 30) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 32) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (7, 33) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (7, 34) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (7, 35) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 92,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 93,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 94,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 36) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 37) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 38) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 40) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 41) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 42) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 44) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 45) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 48) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 50) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 51) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 52) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 54) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 55) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 58) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 60) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 61) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 62) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 64) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 65) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 68) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 70) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 71) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 72) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 74) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (7, 75) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (7, 76) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (7, 77) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 92,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 95,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 96,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 78) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 79) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 80) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 82) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 83) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 84) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 86) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 87) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 90) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 92) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 93) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 94) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 96) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 97) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 100) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 102) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 103) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 104) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 106) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 107) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 110) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 112) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 113) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 114) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 116) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (7, 117) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (7, 118) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (7, 119) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 92,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 97,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 98,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 120) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 121) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 122) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 124) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 125) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 126) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 128) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 129) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 132) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 134) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 135) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 136) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 138) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 139) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 142) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 144) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 145) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 146) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 148) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 149) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 152) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 154) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 155) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 156) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 158) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (7, 159) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (7, 160) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (7, 161) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 92,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 99,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 100,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 162) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 163) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 164) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 166) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 167) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 168) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 170) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 171) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 174) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 176) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 177) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 178) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 180) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 181) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 184) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 186) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 187) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 188) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 190) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 191) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 194) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 196) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 197) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 198) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 200) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (7, 201) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (7, 202) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (7, 203) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 92,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 101,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 102,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 204) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 205) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 206) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 208) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 209) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 210) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 212) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 213) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 216) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 218) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 219) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 220) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 222) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 223) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 226) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 228) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 229) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 230) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 232) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 233) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 236) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 238) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 239) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 240) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 242) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (7, 243) => vec![Packet {
            definition: 8,
            roles: vec![
                Role::Packed,
                Role::Q,
                Role::K,
                Role::Gate,
                Role::QGamma,
                Role::KGamma,
                Role::Rotary,
                Role::Positions,
            ],
            clears: vec![],
        }],
        (7, 244) => vec![Packet {
            definition: 9,
            roles: vec![
                Role::K,
                Role::Packed,
                Role::Cache,
                Role::Slots,
                Role::KScale,
                Role::VScale,
            ],
            clears: vec![],
        }],
        (7, 245) => vec![
            Packet {
                definition: 31,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 92,
                roles: vec![
                    Role::PrivateQ,
                    Role::PrivateO,
                    Role::Cache,
                    Role::Lse,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ],
                clears: vec![],
            },
            Packet {
                definition: 103,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 104,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 246) => vec![Packet {
            definition: 14,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 247) => vec![Packet {
            definition: 15,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 248) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 250) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 251) => vec![Packet {
            definition: 7,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 252) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 128)],
        }],
        (7, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        _ => {
            return Err(Error::Unsupported(
                "actual fixed_promptB4 source packet site absent".into(),
            ));
        }
    })
}
