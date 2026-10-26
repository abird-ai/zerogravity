//! Generated fixed_promptB4 ABI; physical execution/math not qualified.
use super::super::cubin_manifests::ProjectionImage;
use super::packets::{Definition, Packet, Role};
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup,
};
use zerogravity_core::{Error, Result};
// Whole compiler input SHA256: d2b241e5e4feed9a8d16b871693e69169efa40c1b32572d08bb7cc342da80415
pub(super) const METADATA_BOUND: usize = 15371168;
pub(super) const DEFINITION_COUNT: usize = 116;
#[inline(never)]
fn packet_0() -> Definition {
    Definition{image:Some(ProjectionImage::FixedPromptCutlassBf16Sm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[16,8,1],block:[128,1,1],dynamic_shared_bytes:81920,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:385024,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16777216,alignment:16,io:CubinIo::Read},CubinBinding{bytes:3080192,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[188,0,0,0,0,32,0,0,0,4,0,0,2,0,0,0,64,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,64,64,255,255,255,255,255,255,64,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,64,64,255,255,255,255,255,255,64,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,192,255,255,255,255,255,255,0,192,255,255,255,255,255,255,0,0,2,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,4,0,0,0,0,0,0,64,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,192,255,255,255,255,255,255,0,192,255,255,255,255,255,255,0,0,2,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,127,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:344,extent:385024,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:352,extent:16777216,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:360,extent:3080192,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:368,extent:3080192,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_1() -> Definition {
    Definition{image:Some(ProjectionImage::CutlassWmmaSm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[6,1,3],block:[128,1,1],dynamic_shared_bytes:33792,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:385024,alignment:16,io:CubinIo::Read},CubinBinding{bytes:65536,alignment:16,io:CubinIo::Read},CubinBinding{bytes:12032,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8519680,alignment:16,io:CubinIo::ReadWrite}],parameters:vec![CubinParameter{offset:0,template:[188,0,0,0,32,0,0,0,0,4,0,0,6,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,128,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,65,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,65,255,255,255,255,255,255,0,1,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,64,255,255,255,255,255,255,255,64,255,255,255,255,255,255,255,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,8,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,64,255,255,255,255,255,255,255,64,255,255,255,255,255,255,255,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,128,63,0,0,0,0,255,255,255,255,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:280,extent:385024,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:288,extent:65536,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:296,extent:12032,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:304,extent:12032,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:3,binding_offset:0,byte_offset:48,extent:24,io:CubinIo::ReadWrite,parameter:0}]}}
}
#[inline(never)]
fn packet_2() -> Definition {
    Definition {
        image: Some(ProjectionImage::AmpereBf16Gemm128x64Sm80),
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn".into(),
            binary_sm: 80,
            grid: [8, 3, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 24576,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 4194304,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 770048,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 385024,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 64, 0, 250, 255, 255, 255, 255, 255,
                    64, 0, 254, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0,
                    0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 188, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
                    extent: 4194304,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 8,
                    extent: 770048,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 16,
                    extent: 385024,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 168,
                    extent: 385024,
                    io: CubinIo::Write,
                    parameter: 0,
                },
            ],
        },
    }
}
#[inline(never)]
fn packet_3() -> Definition {
    Definition{image:Some(ProjectionImage::FixedPromptCutlassBf16Sm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8,7,1],block:[256,1,1],dynamic_shared_bytes:73728,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:385024,alignment:16,io:CubinIo::Read},CubinBinding{bytes:14680064,alignment:16,io:CubinIo::Read},CubinBinding{bytes:2695168,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[188,0,0,0,0,28,0,0,0,4,0,0,1,0,0,0,56,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,64,64,255,255,255,255,255,255,64,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,64,192,255,255,255,255,255,255,64,0,0,0,0,0,0,0,0,56,0,0,0,0,0,0,0,112,0,0,0,0,0,0,0,200,255,255,255,255,255,255,0,200,255,255,255,255,255,255,0,192,1,0,0,0,0,0,0,0,42,0,0,0,0,0,0,0,56,0,0,0,0,0,0,0,7,0,0,0,0,0,0,56,0,0,0,0,0,0,0,112,0,0,0,0,0,0,0,200,255,255,255,255,255,255,0,200,255,255,255,255,255,255,0,192,1,0,0,0,0,0,0,0,42,0,0,0,0,0,0,0,56,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,127,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:344,extent:385024,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:352,extent:14680064,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:360,extent:2695168,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:368,extent:2695168,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_4() -> Definition {
    Definition {
        image: Some(ProjectionImage::AmpereBf16Gemm128x64Sm80),
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn".into(),
            binary_sm: 80,
            grid: [8, 3, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 24576,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 7340032,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 1347584,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 385024,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128,
                    3, 0, 0, 0, 0, 0, 0, 128, 3, 0, 0, 0, 0, 0, 64, 128, 245, 255, 255, 255, 255,
                    255, 64, 128, 252, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 14,
                    0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 188, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
                .to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    alignment: 16,
                    binding: 0,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 7340032,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 8,
                    extent: 1347584,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 16,
                    extent: 385024,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 168,
                    extent: 385024,
                    io: CubinIo::Write,
                    parameter: 0,
                },
            ],
        },
    }
}
#[inline(never)]
fn packet_5() -> Definition {
    Definition {
        image: Some(ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80),
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn".into(),
            binary_sm: 80,
            grid: [20, 2, 2],
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
                    bytes: 385024,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 1925120,
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
                    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 64, 0, 250, 255, 255, 255, 255, 255,
                    64, 0, 254, 255, 255, 255, 255, 255, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0,
                    0, 20, 0, 0, 0, 20, 0, 0, 0, 20, 0, 0, 188, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 12, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
                    extent: 10485760,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 8,
                    extent: 385024,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 16,
                    extent: 1925120,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 168,
                    extent: 1925120,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 3,
                    binding_offset: 0,
                    byte_offset: 120,
                    extent: 160,
                    io: CubinIo::ReadWrite,
                    parameter: 0,
                },
            ],
        },
    }
}
#[inline(never)]
fn packet_6() -> Definition {
    Definition{image:Some(ProjectionImage::FixedPromptCutlassBf16Sm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I67cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x10_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8,485,1],block:[128,1,1],dynamic_shared_bytes:81920,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},CubinBinding{bytes:508559360,alignment:16,io:CubinIo::Read},CubinBinding{bytes:1986560,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[4,0,0,0,0,202,3,0,0,4,0,0,1,0,0,0,40,15,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,64,192,255,255,255,255,255,255,64,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,64,0,0,0,0,0,0,64,192,255,255,255,255,255,255,64,0,0,0,0,0,0,0,0,148,7,0,0,0,0,0,0,80,30,0,0,0,0,0,0,148,7,0,0,0,0,0,0,148,7,0,0,0,0,0,0,160,60,0,0,0,0,0,0,128,242,0,0,0,0,0,0,0,229,1,0,0,0,0,0,64,121,0,0,0,0,0,0,148,7,0,0,0,0,0,0,80,30,0,0,0,0,0,0,148,7,0,0,0,0,0,0,148,7,0,0,0,0,0,0,160,60,0,0,0,0,0,0,128,242,0,0,0,0,0,0,0,229,1,0,0,0,0,0,64,121,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,127,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:344,extent:8192,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:352,extent:508559360,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:360,extent:1986560,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:368,extent:1986560,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_7() -> Definition {
    Definition{image:Some(ProjectionImage::CutlassWmmaSm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8,64,1],block:[32,1,1],dynamic_shared_bytes:8704,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16777216,alignment:16,io:CubinIo::Read},CubinBinding{bytes:65536,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[4,0,0,0,0,32,0,0,0,4,0,0,1,0,0,0,0,2,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,145,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,145,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,64,254,255,255,255,255,255,0,64,254,255,255,255,255,255,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,64,254,255,255,255,255,255,0,64,254,255,255,255,255,255,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,128,63,0,0,0,0,255,255,255,255,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:280,extent:8192,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:288,extent:16777216,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:296,extent:65536,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:304,extent:65536,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_8() -> Definition {
    Definition{image:Some(ProjectionImage::CutlassWmmaSm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[2,1,1],block:[32,1,1],dynamic_shared_bytes:16896,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},CubinBinding{bytes:65536,alignment:16,io:CubinIo::Read},CubinBinding{bytes:256,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[4,0,0,0,32,0,0,0,0,4,0,0,1,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,145,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,145,255,255,255,255,255,255,0,1,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,64,254,255,255,255,255,255,255,64,254,255,255,255,255,255,255,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,64,254,255,255,255,255,255,255,64,254,255,255,255,255,255,255,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,128,63,0,0,0,0,255,255,255,255,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:280,extent:8192,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:288,extent:65536,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:296,extent:256,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:304,extent:256,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_9() -> Definition {
    Definition{image:Some(ProjectionImage::CutlassWmmaSm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8,4,1],block:[128,1,1],dynamic_shared_bytes:33792,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4194304,alignment:16,io:CubinIo::Read},CubinBinding{bytes:8192,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[4,0,0,0,0,4,0,0,0,8,0,0,1,0,0,0,32,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,129,254,255,255,255,255,255,0,1,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,129,254,255,255,255,255,255,0,1,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,232,255,255,255,255,255,255,0,232,255,255,255,255,255,255,0,128,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,8,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,232,255,255,255,255,255,255,0,232,255,255,255,255,255,255,0,128,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,128,63,0,0,0,0,255,255,255,255,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:280,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:288,extent:4194304,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:296,extent:8192,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:304,extent:8192,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_10() -> Definition {
    Definition{image:Some(ProjectionImage::CutlassWmmaSm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8,56,1],block:[32,1,1],dynamic_shared_bytes:16896,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},CubinBinding{bytes:14680064,alignment:16,io:CubinIo::Read},CubinBinding{bytes:57344,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[4,0,0,0,0,28,0,0,0,4,0,0,1,0,0,0,192,1,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,145,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,145,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,56,0,0,0,0,0,0,0,192,1,0,0,0,0,0,0,120,254,255,255,255,255,255,0,120,254,255,255,255,255,255,0,128,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,3,0,0,0,0,0,0,128,3,0,0,0,0,0,0,56,0,0,0,0,0,0,0,192,1,0,0,0,0,0,0,120,254,255,255,255,255,255,0,120,254,255,255,255,255,255,0,128,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,3,0,0,0,0,0,0,128,3,0,0,0,0,0,0,0,128,63,0,0,0,0,255,255,255,255,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:280,extent:8192,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:288,extent:14680064,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:296,extent:57344,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:304,extent:57344,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_11() -> Definition {
    Definition {
        image: Some(ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80),
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn".into(),
            binary_sm: 80,
            grid: [16, 1, 2],
            block: [128, 1, 1],
            dynamic_shared_bytes: 49152,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 7340032,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 28672,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 8192,
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
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 192,
                    1, 0, 0, 0, 0, 0, 0, 192, 1, 0, 0, 0, 0, 0, 128, 192, 250, 255, 255, 255, 255,
                    255, 128, 192, 250, 255, 255, 255, 255, 255, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0,
                    14, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 12, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
                .to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    alignment: 16,
                    binding: 0,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 7340032,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 8,
                    extent: 28672,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 16,
                    extent: 8192,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 168,
                    extent: 8192,
                    io: CubinIo::Write,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 3,
                    binding_offset: 0,
                    byte_offset: 120,
                    extent: 64,
                    io: CubinIo::ReadWrite,
                    parameter: 0,
                },
            ],
        },
    }
}
#[inline(never)]
fn packet_12() -> Definition {
    Definition{image:Some(ProjectionImage::CutlassWmmaSm80),manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8,40,1],block:[32,1,1],dynamic_shared_bytes:16896,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},CubinBinding{bytes:10485760,alignment:16,io:CubinIo::Read},CubinBinding{bytes:40960,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:[4,0,0,0,0,20,0,0,0,4,0,0,1,0,0,0,64,1,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,145,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,145,255,255,255,255,255,255,0,1,0,0,0,0,0,0,0,40,0,0,0,0,0,0,0,64,1,0,0,0,0,0,0,232,254,255,255,255,255,255,0,232,254,255,255,255,255,255,0,128,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,2,0,0,0,0,0,0,128,2,0,0,0,0,0,0,40,0,0,0,0,0,0,0,64,1,0,0,0,0,0,0,232,254,255,255,255,255,255,0,232,254,255,255,255,255,255,0,128,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,2,0,0,0,0,0,0,128,2,0,0,0,0,0,0,0,128,63,0,0,0,0,255,255,255,255,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:280,extent:8192,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:288,extent:10485760,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:296,extent:40960,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:304,extent:40960,io:CubinIo::Write,parameter:0}]}}
}
#[inline(never)]
fn packet_13() -> Definition {
    Definition{image:None,manifest:CubinKernelManifest{entry:"_ZN4vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_bfloat162XadL_ZNS_11silu_kernelIS2_EET_RKS5_fEEXadL_ZNS_18packed_silu_kernelIS3_EES5_S7_fEELb1ELb1ELb0ELb0EEEvPS5_PS6_ifff".into(),binary_sm:86,grid:[188,1,1],block:[448,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:1347584,alignment:16,io:CubinIo::Write},CubinBinding{bytes:2695168,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:16,template:[0,14,0,0].to_vec()},CubinParameter{offset:20,template:[0,0,0,0].to_vec()},CubinParameter{offset:24,template:[0,0,128,63].to_vec()},CubinParameter{offset:28,template:[0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:1347584,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:0,extent:2695168,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_14() -> Definition {
    Definition {
        image: Some(ProjectionImage::AttentionQkSm86),
        manifest: CubinKernelManifest {
            entry: "_fused_qk_rmsnorm_rope_gate_kernel".into(),
            binary_sm: 86,
            grid: [188, 10, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 16,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 1925120,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 770048,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 192512,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 770048,
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
                    extent: 1923072,
                    io: CubinIo::Read,
                    parameter: 0,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 0,
                    binding_offset: 8192,
                    byte_offset: 0,
                    extent: 1915904,
                    io: CubinIo::Read,
                    parameter: 1,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 1,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 770048,
                    io: CubinIo::Write,
                    parameter: 2,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 2,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 192512,
                    io: CubinIo::Write,
                    parameter: 3,
                },
                CubinPointerFixup {
                    alignment: 16,
                    binding: 3,
                    binding_offset: 0,
                    byte_offset: 0,
                    extent: 770048,
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
fn packet_15() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionKvSm86),manifest:CubinKernelManifest{entry:"_ZN4vllm30reshape_and_cache_flash_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKlllllliiiPKfSB_i".into(),binary_sm:86,grid:[188,1,1],block:[512,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:192512,alignment:16,io:CubinIo::Read},CubinBinding{bytes:1925120,alignment:16,io:CubinIo::Read},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::ReadWrite},CubinBinding{bytes:1504,alignment:8,io:CubinIo::Read},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:16,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:24,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:32,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:40,template:[0,128,8,0,0,0,0,0].to_vec()},CubinParameter{offset:48,template:[0,4,0,0,0,0,0,0].to_vec()},CubinParameter{offset:56,template:[0,2,0,0,0,0,0,0].to_vec()},CubinParameter{offset:64,template:[0,2,0,0,0,0,0,0].to_vec()},CubinParameter{offset:72,template:[0,20,0,0,0,0,0,0].to_vec()},CubinParameter{offset:80,template:[2,0,0,0].to_vec()},CubinParameter{offset:84,template:[0,1,0,0].to_vec()},CubinParameter{offset:88,template:[32,2,0,0].to_vec()},CubinParameter{offset:96,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:104,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:112,template:[0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:192512,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:9216,byte_offset:0,extent:1915904,io:CubinIo::Read,parameter:1},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:0,extent:18939904,io:CubinIo::ReadWrite,parameter:2},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:0,extent:18939392,io:CubinIo::ReadWrite,parameter:3},CubinPointerFixup{alignment:8,binding:3,binding_offset:0,byte_offset:0,extent:1504,io:CubinIo::Read,parameter:4},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:0,extent:4,io:CubinIo::Read,parameter:13},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:0,extent:4,io:CubinIo::Read,parameter:14}]}}
}
#[inline(never)]
fn packet_16() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaCausalCombineSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb1ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,8],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:770048,alignment:16,io:CubinIo::Read},CubinBinding{bytes:770048,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:6016,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,8,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,8,0,0,0,2,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,47,0,0,0,47,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,188,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,199,27,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:770048,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:770048,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:6016,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:224,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:7,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_17() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionSigmoidSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_19sigmoid_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE0_clEvENKUlvE2_clEvEUlN3c108BFloat16EE_St5arrayIPcLm2EEEEviT0_T1_".into(),binary_sm:86,grid:[376,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:770048,alignment:16,io:CubinIo::Write},CubinBinding{bytes:770048,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,224,5,0].to_vec()},CubinParameter{offset:4,template:[0].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:770048,io:CubinIo::Write,parameter:2},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:8,extent:770048,io:CubinIo::Read,parameter:2}]}}
}
#[inline(never)]
fn packet_18() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionMultiplySm86),manifest:CubinKernelManifest{entry:"_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIN3c108BFloat16ES4_S4_NS0_15binary_internal10MulFunctorIfEEEESt5arrayIPcLm3EEEEviT0_T1_".into(),binary_sm:86,grid:[376,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:770048,alignment:16,io:CubinIo::Write},CubinBinding{bytes:770048,alignment:16,io:CubinIo::Read},CubinBinding{bytes:770048,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,224,5,0].to_vec()},CubinParameter{offset:4,template:[15].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:770048,io:CubinIo::Write,parameter:2},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:8,extent:770048,io:CubinIo::Read,parameter:2},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:16,extent:770048,io:CubinIo::Read,parameter:2}]}}
}
#[inline(never)]
fn packet_19() -> Definition {
    Definition{image:None,manifest:CubinKernelManifest{entry:"_ZN4vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_bfloat162XadL_ZNS_11silu_kernelIS2_EET_RKS5_fEEXadL_ZNS_18packed_silu_kernelIS3_EES5_S7_fEELb1ELb1ELb0ELb0EEEvPS5_PS6_ifff".into(),binary_sm:86,grid:[4,1,1],block:[448,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:28672,alignment:16,io:CubinIo::Write},CubinBinding{bytes:57344,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:16,template:[0,14,0,0].to_vec()},CubinParameter{offset:20,template:[0,0,0,0].to_vec()},CubinParameter{offset:24,template:[0,0,128,63].to_vec()},CubinParameter{offset:28,template:[0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:28672,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:0,extent:57344,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_20() -> Definition {
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
fn packet_21() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionKvSm86),manifest:CubinKernelManifest{entry:"_ZN4vllm30reshape_and_cache_flash_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKlllllliiiPKfSB_i".into(),binary_sm:86,grid:[4,1,1],block:[512,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:40960,alignment:16,io:CubinIo::Read},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::ReadWrite},CubinBinding{bytes:32,alignment:8,io:CubinIo::Read},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:16,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:24,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:32,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:40,template:[0,128,8,0,0,0,0,0].to_vec()},CubinParameter{offset:48,template:[0,4,0,0,0,0,0,0].to_vec()},CubinParameter{offset:56,template:[0,2,0,0,0,0,0,0].to_vec()},CubinParameter{offset:64,template:[0,2,0,0,0,0,0,0].to_vec()},CubinParameter{offset:72,template:[0,20,0,0,0,0,0,0].to_vec()},CubinParameter{offset:80,template:[2,0,0,0].to_vec()},CubinParameter{offset:84,template:[0,1,0,0].to_vec()},CubinParameter{offset:88,template:[32,2,0,0].to_vec()},CubinParameter{offset:96,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:104,template:[0,0,0,0,0,0,0,0].to_vec()},CubinParameter{offset:112,template:[0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:4096,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:9216,byte_offset:0,extent:31744,io:CubinIo::Read,parameter:1},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:0,extent:18939904,io:CubinIo::ReadWrite,parameter:2},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:0,extent:18939392,io:CubinIo::ReadWrite,parameter:3},CubinPointerFixup{alignment:8,binding:3,binding_offset:0,byte_offset:0,extent:32,io:CubinIo::Read,parameter:4},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:0,extent:4,io:CubinIo::Read,parameter:13},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:0,extent:4,io:CubinIo::Read,parameter:14}]}}
}
#[inline(never)]
fn packet_22() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,8,0,0,0,4,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_23() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,48,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,199,27,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_24() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,252,127,0,0,174,78,16,77,1,0,0,0,56,23,195,124,252,127,0,0,128,228,183,40,0,0,0,0,32,126,139,35,0,0,0,0,240,25,195,124,252,127,0,0,0,128,1,0,0,0,0,0,207,23,195,124,252,127,0,0,0,24,195,124,252,127,0,0,3,58,162,19,32,126,0,0,0,0,0,0,0,0,0,0,128,71,83,40,0,0,0,0,128,228,183,40,0,0,0,0,128,223,241,40,0,0,0,0,160,232,68,40,0,0,0,0,216,199,83,40,0,0,0,0,0,24,195,124,252,127,0,0,61,67,162,19,32,126,0,0,80,137,7,87,0,0,0,0,0,22,140,35,0,0,0,0,80,224,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,56,118,184,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,240,25,195,124,0,0,0,0,156,221,78,0,2,0,0,0,128,228,183,40,0,0,0,0,0,98,158,63,96,210,163,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,208,24,195,124,252,127,0,0,126,89,16,77,32,126,0,0,4,0,0,0,0,0,0,0,12,0,0,0,0,0,1,0,2,10,0,12,0,10,2,10,208,100,46,40,0,0,0,0,32,108,10,112,0,0,0,0,0,55,31,0,0,0,0,0,0,201,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,29,195,124,252,127,0,0,177,95,189,19,32,126,0,0,128,228,183,40,0,0,0,0,3,0,0,0,0,0,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,89,80,40,0,0,0,0,112,25,195,124,252,127,0,0,58,76,197,19,32,126,0,0,0,141,248,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_25() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,17,147,13,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,50,76,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,137,44,187,19,32,126,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,38,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_26() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionSigmoidSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_19sigmoid_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE0_clEvENKUlvE2_clEvEUlN3c108BFloat16EE_St5arrayIPcLm2EEEEviT0_T1_".into(),binary_sm:86,grid:[8,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:4,template:[0].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Write,parameter:2},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:8,extent:16384,io:CubinIo::Read,parameter:2}]}}
}
#[inline(never)]
fn packet_27() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionMultiplySm86),manifest:CubinKernelManifest{entry:"_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIN3c108BFloat16ES4_S4_NS0_15binary_internal10MulFunctorIfEEEESt5arrayIPcLm3EEEEviT0_T1_".into(),binary_sm:86,grid:[8,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:4,template:[15].to_vec()},CubinParameter{offset:8,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Write,parameter:2},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:8,extent:16384,io:CubinIo::Read,parameter:2},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:16,extent:16384,io:CubinIo::Read,parameter:2}]}}
}
#[inline(never)]
fn packet_28() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,16,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,224,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,192,196,64,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_29() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,143,159,13,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,217,84,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_30() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,32,225,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,112,219,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,24,75,67,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_31() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,12,172,13,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,127,93,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_32() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,176,222,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,128,223,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,112,209,69,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_33() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,137,184,13,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,37,102,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_34() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,64,220,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,16,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,200,87,72,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_35() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,6,197,13,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,203,110,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_36() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,80,224,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,32,225,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,32,222,74,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_37() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,131,209,13,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,113,119,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_38() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,49,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,199,27,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_39() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,80,224,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,32,225,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,88,0,78,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,230,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_40() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,123,165,14,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,140,129,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_41() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,224,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,176,222,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,236,123,64,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,228,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_42() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,248,177,14,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,50,138,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_43() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,112,219,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,64,220,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,68,2,67,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,230,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_44() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,117,190,14,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,216,146,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_45() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,128,223,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,80,224,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,156,136,69,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,228,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_46() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,242,202,14,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,126,155,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_47() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,16,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,224,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,244,14,72,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,230,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_48() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,111,215,14,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,36,164,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_49() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,32,225,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,112,219,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,76,149,74,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,228,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_50() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,236,227,14,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,202,172,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_51() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,50,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,199,27,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_52() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,32,225,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,112,219,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,132,183,77,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,236,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_53() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,211,184,15,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,229,182,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_54() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,176,222,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,128,223,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,80,56,64,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,234,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_55() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,80,197,15,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,139,191,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_56() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,64,220,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,16,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,168,190,66,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,234,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_57() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,205,209,15,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,49,200,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_58() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,80,224,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,32,225,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,0,69,69,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,234,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_59() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,74,222,15,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,215,208,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_60() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,224,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,176,222,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,88,203,71,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,234,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_61() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,199,234,15,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,125,217,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_62() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,112,219,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,64,220,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,176,81,74,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,234,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_63() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,68,247,15,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,35,226,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_64() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,51,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,199,27,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_65() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,112,219,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,64,220,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,232,115,77,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_66() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,51,204,16,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,62,236,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_67() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,128,223,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,80,224,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,64,250,79,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_68() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,176,216,16,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,228,244,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_69() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,16,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,224,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,0,109,66,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_70() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,45,229,16,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,138,253,0,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_71() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,32,225,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,112,219,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,88,243,68,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_72() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,170,241,16,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,48,6,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_73() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,176,222,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,128,223,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,176,121,71,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_74() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,39,254,16,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,214,14,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_75() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,64,220,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,16,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,8,0,74,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,232,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_76() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,164,10,17,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,124,23,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_77() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,52,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,199,27,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_78() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,64,220,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,16,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,64,34,77,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,226,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_79() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,156,223,17,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,151,33,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_80() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,80,224,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,32,225,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,152,168,79,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,226,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_81() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,25,236,17,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,61,42,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_82() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,224,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,176,222,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,100,41,66,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,226,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_83() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,150,248,17,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,227,50,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_84() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,112,219,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,64,220,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,188,175,68,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,226,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_85() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,19,5,18,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,137,59,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_86() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,128,223,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,80,224,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,20,54,71,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,226,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_87() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,144,17,18,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,47,68,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_88() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,16,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,224,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,108,188,73,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,226,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_89() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,13,30,18,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,213,76,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_90() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,53,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,199,27,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_91() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,16,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,224,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,164,222,76,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_92() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,13,243,18,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,240,86,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_93() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,32,225,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,112,219,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,252,100,79,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_94() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,138,255,18,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,150,95,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_95() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,176,222,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,128,223,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,180,221,65,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_96() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,7,12,19,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,60,104,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_97() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,64,220,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,16,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,12,100,68,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_98() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,132,24,19,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,226,112,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_99() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,80,224,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,32,225,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,100,234,70,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_100() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,1,37,19,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,136,121,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_101() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,224,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,176,222,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,188,112,73,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_102() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,126,49,19,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,46,130,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_103() -> Definition {
    Definition{image:Some(ProjectionImage::AttentionFaSplitSm80),manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1,4,2],block:[128,1,1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:18939904,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:20,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read},CubinBinding{bytes:64,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,128,8,0,0,0,0,0,0,128,8,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,4,0,0,0,54,0,0,0,0,0,0,0,0,1,0,0,128,0,0,0,128,0,0,0,0,1,0,0,0,0,0,0,16,0,0,0,0,0,128,61,59,170,184,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,32,2,0,0,0,0,128,63,255,0,0,0,0,0,128,63,0,0,128,61,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,199,27,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:0,extent:16384,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:112,extent:16384,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:0,byte_offset:8,extent:18939904,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:16,binding:2,binding_offset:512,byte_offset:16,extent:18939392,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:3,binding_offset:0,byte_offset:160,extent:128,io:CubinIo::Write,parameter:0},CubinPointerFixup{alignment:4,binding:4,binding_offset:0,byte_offset:232,extent:20,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:5,binding_offset:0,byte_offset:248,extent:16,io:CubinIo::Read,parameter:0},CubinPointerFixup{alignment:4,binding:6,binding_offset:0,byte_offset:352,extent:64,io:CubinIo::Read,parameter:0}]}}
}
#[inline(never)]
fn packet_104() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,224,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,176,222,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,244,146,76,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_105() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,128,6,20,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,73,140,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_106() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,112,219,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,64,220,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,76,25,79,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_107() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,253,18,20,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,239,148,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_108() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,128,223,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,80,224,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,40,148,65,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_109() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,122,31,20,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,149,157,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_110() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,16,221,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,224,221,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,128,26,68,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_111() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,247,43,20,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,59,166,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_112() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,32,225,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,112,219,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,216,160,70,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_113() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,116,56,20,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,225,174,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_114() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE10_clEvEUlN3c108BFloat16EE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[16,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[0,32,0,0].to_vec()},CubinParameter{offset:8,template:[4,0,0,0,0,1,0,0,1,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,56,23,195,124,252,127,0,0,24,23,195,124,252,127,0,0,109,202,146,24,32,126,0,0,48,23,195,124,252,127,0,0,160,232,68,40,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,28,23,195,124,252,127,0,0,0,0,0,0,0,0,0,0,112,95,4,27,0,0,0,0,16,142,6,27,0,0,0,0,176,222,241,40,0,0,0,0,240,23,195,124,252,127,0,0,89,194,136,75,32,126,0,0,0,142,6,27,0,0,0,0,103,156,130,75,32,126,0,0,80,137,7,87,0,0,0,0,0,18,129,75,32,126,0,0,128,223,241,40,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,208,84,72,0,1,0,0,0,160,232,68,40,0,0,0,0,160,232,68,40,0,0,0,0,64,25,195,124,0,0,0,0,48,39,73,0,2,0,0,0,128,228,183,40,0,0,0,0,88,18,133,75,32,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,4,0,0,0,8,0,0,0,2,0,0,0,16,0,0,0,16,0,0,0,12,95,61,29,126,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,24,195,124,252,127,0,0,41,15,129,75,32,126,0,0,1,0,0,0,0,0,0,0,0,0,28,40,0,0,0,0,144,25,195,124,91,1,0,0,189,229,111,46,31,126,0,0,184,221,111,46,31,126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:16,binding:0,binding_offset:0,byte_offset:504,extent:16384,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:16,binding:1,binding_offset:0,byte_offset:512,extent:16384,io:CubinIo::Read,parameter:1}]}}
}
#[inline(never)]
fn packet_115() -> Definition {
    Definition{image:Some(ProjectionImage::RotaryBf16CastSm86),manifest:CubinKernelManifest{entry:"_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_".into(),binary_sm:86,grid:[1,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:128,alignment:4,io:CubinIo::Write},CubinBinding{bytes:128,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:[32,0,0,0].to_vec()},CubinParameter{offset:8,template:[3,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,224,192,146,24,32,126,0,0,176,16,195,124,252,127,0,0,80,16,195,124,252,127,0,0,115,42,132,58,0,0,0,0,160,232,68,40,0,0,0,0,1,0,0,0,252,127,0,0,104,16,195,124,252,127,0,0,241,68,20,0,252,127,0,0,9,118,155,19,32,126,0,0,32,74,80,40,0,0,0,0,51,1,0,0,0,0,0,0,240,16,195,124,252,127,0,0,64,3,235,39,0,0,0,0,6,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,48,16,195,124,252,127,0,0,6,254,58,5,32,126,0,0,240,251,37,59,0,0,0,0,32,74,80,40,0,0,0,0,0,0,0,0,0,0,0,0,192,219,58,5,32,126,0,0,144,95,154,39,0,0,0,0,64,145,93,40,0,0,0,0,0,0,0,0,0,0,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,224,192,146,24,32,126,0,0,240,16,195,124,252,127,0,0,240,251,37,59,0,0,0,0,1,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,176,19,195,124,252,127,0,0,249,44,184,19,32,126,0,0,4,0,0,0,16,0,0,0,16,0,0,0,4,0,0,0,64,0,0,0,64,0,0,0,160,232,68,40,0,0,0,0,144,95,154,39,0,0,0,0,135,183,1,0,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,251,37,59,0,0,0,0,16,0,0,0,1,0,0,0,1,0,0,0,128,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,25,195,124,252,127,0,0,0,0,0,0,0,0,0,0,104,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,52,240,87,0,0,0,0].to_vec()}],fixups:vec![CubinPointerFixup{alignment:4,binding:0,binding_offset:0,byte_offset:504,extent:128,io:CubinIo::Write,parameter:1},CubinPointerFixup{alignment:4,binding:1,binding_offset:0,byte_offset:512,extent:128,io:CubinIo::Read,parameter:1}]}}
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
        packet_105(),
        packet_106(),
        packet_107(),
        packet_108(),
        packet_109(),
        packet_110(),
        packet_111(),
        packet_112(),
        packet_113(),
        packet_114(),
        packet_115(),
    ]
}
pub(super) fn site(call: usize, node: usize) -> Result<Vec<Packet>> {
    Ok(match (call, node) {
        (0, 2) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 3) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 6) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 8) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 9) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 10) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 12) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 13) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 16) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 18) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 19) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 20) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 22) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 23) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 26) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 28) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 29) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 30) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 32) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 160)],
        }],
        (0, 33) => vec![Packet {
            definition: 14,
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
        (0, 34) => vec![Packet {
            definition: 15,
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
        (0, 35) => vec![Packet {
            definition: 16,
            roles: vec![
                Role::Q,
                Role::O,
                Role::Cache,
                Role::Lse,
                Role::CuQ,
                Role::DummyCuK,
                Role::SeqUsed,
                Role::Table,
            ],
            clears: vec![],
        }],
        (0, 36) => vec![Packet {
            definition: 17,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 37) => vec![Packet {
            definition: 18,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (0, 38) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 40) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 41) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 42) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 44) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 45) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 48) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 50) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 51) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 52) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 54) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 55) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 58) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 60) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 61) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 62) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 64) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 65) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 68) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 70) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 71) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 72) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 74) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 160)],
        }],
        (0, 75) => vec![Packet {
            definition: 14,
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
        (0, 76) => vec![Packet {
            definition: 15,
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
        (0, 77) => vec![Packet {
            definition: 16,
            roles: vec![
                Role::Q,
                Role::O,
                Role::Cache,
                Role::Lse,
                Role::CuQ,
                Role::DummyCuK,
                Role::SeqUsed,
                Role::Table,
            ],
            clears: vec![],
        }],
        (0, 78) => vec![Packet {
            definition: 17,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 79) => vec![Packet {
            definition: 18,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (0, 80) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 82) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 83) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 84) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 86) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 87) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 90) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 92) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 93) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 94) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 96) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 97) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 100) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 102) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 103) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 104) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 106) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 107) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 110) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 112) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 113) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 114) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 116) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 160)],
        }],
        (0, 117) => vec![Packet {
            definition: 14,
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
        (0, 118) => vec![Packet {
            definition: 15,
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
        (0, 119) => vec![Packet {
            definition: 16,
            roles: vec![
                Role::Q,
                Role::O,
                Role::Cache,
                Role::Lse,
                Role::CuQ,
                Role::DummyCuK,
                Role::SeqUsed,
                Role::Table,
            ],
            clears: vec![],
        }],
        (0, 120) => vec![Packet {
            definition: 17,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 121) => vec![Packet {
            definition: 18,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (0, 122) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 124) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 125) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 126) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 128) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 129) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 132) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 134) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 135) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 136) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 138) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 139) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 142) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 144) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 145) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 146) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 148) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 149) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 152) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 154) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 155) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 156) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 158) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 160)],
        }],
        (0, 159) => vec![Packet {
            definition: 14,
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
        (0, 160) => vec![Packet {
            definition: 15,
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
        (0, 161) => vec![Packet {
            definition: 16,
            roles: vec![
                Role::Q,
                Role::O,
                Role::Cache,
                Role::Lse,
                Role::CuQ,
                Role::DummyCuK,
                Role::SeqUsed,
                Role::Table,
            ],
            clears: vec![],
        }],
        (0, 162) => vec![Packet {
            definition: 17,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 163) => vec![Packet {
            definition: 18,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (0, 164) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 166) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 167) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 168) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 170) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 171) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 174) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 176) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 177) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 178) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 180) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 181) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 184) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 186) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 187) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 188) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 190) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 191) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 194) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 196) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 197) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 198) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 200) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 160)],
        }],
        (0, 201) => vec![Packet {
            definition: 14,
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
        (0, 202) => vec![Packet {
            definition: 15,
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
        (0, 203) => vec![Packet {
            definition: 16,
            roles: vec![
                Role::Q,
                Role::O,
                Role::Cache,
                Role::Lse,
                Role::CuQ,
                Role::DummyCuK,
                Role::SeqUsed,
                Role::Table,
            ],
            clears: vec![],
        }],
        (0, 204) => vec![Packet {
            definition: 17,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 205) => vec![Packet {
            definition: 18,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (0, 206) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 208) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 209) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 210) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 212) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 213) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 216) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 218) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 219) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 220) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 222) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 223) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 226) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 228) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 229) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 230) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 232) => vec![Packet {
            definition: 0,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 233) => vec![Packet {
            definition: 1,
            roles: vec![Role::X, Role::W, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 24)],
        }],
        (0, 236) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 238) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 239) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 240) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 242) => vec![Packet {
            definition: 5,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 160)],
        }],
        (0, 243) => vec![Packet {
            definition: 14,
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
        (0, 244) => vec![Packet {
            definition: 15,
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
        (0, 245) => vec![Packet {
            definition: 16,
            roles: vec![
                Role::Q,
                Role::O,
                Role::Cache,
                Role::Lse,
                Role::CuQ,
                Role::DummyCuK,
                Role::SeqUsed,
                Role::Table,
            ],
            clears: vec![],
        }],
        (0, 246) => vec![Packet {
            definition: 17,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 247) => vec![Packet {
            definition: 18,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (0, 248) => vec![Packet {
            definition: 2,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 250) => vec![Packet {
            definition: 3,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (0, 251) => vec![Packet {
            definition: 13,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (0, 252) => vec![Packet {
            definition: 4,
            roles: vec![Role::W, Role::X, Role::Y],
            clears: vec![],
        }],
        (0, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 2) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 3) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 6) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 8) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 9) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 10) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 12) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 13) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 16) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 18) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 19) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 20) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 22) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 23) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 26) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 28) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 29) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 30) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 32) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 33) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 23,
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
        (1, 36) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 37) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 38) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 40) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 41) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 42) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 44) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 45) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 48) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 50) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 51) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 52) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 54) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 55) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 58) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 60) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 61) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 62) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 64) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 65) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 68) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 70) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 71) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 72) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 74) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 75) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 23,
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
                definition: 28,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 29,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (1, 78) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 79) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 80) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 82) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 83) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 84) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 86) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 87) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 90) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 92) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 93) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 94) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 96) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 97) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 100) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 102) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 103) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 104) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 106) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 107) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 110) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 112) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 113) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 114) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 116) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 117) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 23,
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
                definition: 30,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 31,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (1, 120) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 121) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 122) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 124) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 125) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 126) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 128) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 129) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 132) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 134) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 135) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 136) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 138) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 139) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 142) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 144) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 145) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 146) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 148) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 149) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 152) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 154) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 155) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 156) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 158) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 159) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 23,
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
        (1, 162) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 163) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 164) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 166) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 167) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 168) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 170) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 171) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 174) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 176) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 177) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 178) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 180) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 181) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 184) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 186) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 187) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 188) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 190) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 191) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 194) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 196) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 197) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 198) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 200) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 201) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 23,
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
        (1, 204) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 205) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 206) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 208) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 209) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 210) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 212) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 213) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 216) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 218) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 219) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 220) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 222) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 223) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 226) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 228) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 229) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 230) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 232) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 233) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 236) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 238) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 239) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 240) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 242) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 243) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 23,
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
        (1, 246) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 247) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (1, 248) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 250) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (1, 251) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (1, 252) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (1, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 2) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 3) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 6) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 8) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 9) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 10) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 12) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 13) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 16) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 18) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 19) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 20) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 22) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 23) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 26) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 28) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 29) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 30) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 32) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 33) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 38,
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
                definition: 39,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 40,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (2, 36) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 37) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 38) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 40) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 41) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 42) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 44) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 45) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 48) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 50) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 51) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 52) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 54) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 55) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 58) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 60) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 61) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 62) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 64) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 65) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 68) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 70) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 71) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 72) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 74) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 75) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 38,
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
        (2, 78) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 79) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 80) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 82) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 83) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 84) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 86) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 87) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 90) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 92) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 93) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 94) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 96) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 97) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 100) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 102) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 103) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 104) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 106) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 107) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 110) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 112) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 113) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 114) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 116) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 117) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 38,
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
        (2, 120) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 121) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 122) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 124) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 125) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 126) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 128) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 129) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 132) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 134) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 135) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 136) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 138) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 139) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 142) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 144) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 145) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 146) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 148) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 149) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 152) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 154) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 155) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 156) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 158) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 159) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 38,
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
        (2, 162) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 163) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 164) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 166) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 167) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 168) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 170) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 171) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 174) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 176) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 177) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 178) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 180) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 181) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 184) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 186) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 187) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 188) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 190) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 191) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 194) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 196) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 197) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 198) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 200) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 201) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 38,
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
        (2, 204) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 205) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 206) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 208) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 209) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 210) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 212) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 213) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 216) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 218) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 219) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 220) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 222) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 223) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 226) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 228) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 229) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 230) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 232) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 233) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 236) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 238) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 239) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 240) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 242) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 243) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 38,
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
        (2, 246) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 247) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (2, 248) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 250) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (2, 251) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (2, 252) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (2, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 2) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 3) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 6) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 8) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 9) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 10) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 12) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 13) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 16) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 18) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 19) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 20) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 22) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 23) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 26) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 28) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 29) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 30) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 32) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 33) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 51,
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
                definition: 52,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 53,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (3, 36) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 37) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 38) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 40) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 41) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 42) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 44) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 45) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 48) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 50) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 51) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 52) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 54) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 55) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 58) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 60) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 61) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 62) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 64) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 65) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 68) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 70) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 71) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 72) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 74) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 75) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 51,
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
        (3, 78) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 79) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 80) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 82) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 83) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 84) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 86) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 87) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 90) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 92) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 93) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 94) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 96) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 97) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 100) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 102) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 103) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 104) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 106) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 107) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 110) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 112) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 113) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 114) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 116) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 117) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 51,
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
        (3, 120) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 121) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 122) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 124) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 125) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 126) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 128) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 129) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 132) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 134) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 135) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 136) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 138) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 139) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 142) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 144) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 145) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 146) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 148) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 149) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 152) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 154) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 155) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 156) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 158) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 159) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 51,
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
        (3, 162) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 163) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 164) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 166) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 167) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 168) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 170) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 171) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 174) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 176) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 177) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 178) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 180) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 181) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 184) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 186) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 187) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 188) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 190) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 191) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 194) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 196) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 197) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 198) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 200) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 201) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 51,
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
        (3, 204) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 205) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 206) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 208) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 209) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 210) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 212) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 213) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 216) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 218) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 219) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 220) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 222) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 223) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 226) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 228) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 229) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 230) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 232) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 233) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 236) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 238) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 239) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 240) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 242) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 243) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 51,
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
        (3, 246) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 247) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (3, 248) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 250) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (3, 251) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (3, 252) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (3, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 2) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 3) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 6) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 8) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 9) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 10) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 12) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 13) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 16) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 18) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 19) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 20) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 22) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 23) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 26) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 28) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 29) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 30) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 32) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 33) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 64,
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
                definition: 65,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 66,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (4, 36) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 37) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 38) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 40) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 41) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 42) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 44) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 45) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 48) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 50) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 51) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 52) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 54) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 55) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 58) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 60) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 61) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 62) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 64) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 65) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 68) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 70) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 71) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 72) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 74) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 75) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 64,
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
        (4, 78) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 79) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 80) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 82) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 83) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 84) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 86) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 87) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 90) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 92) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 93) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 94) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 96) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 97) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 100) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 102) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 103) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 104) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 106) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 107) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 110) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 112) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 113) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 114) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 116) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 117) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 64,
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
        (4, 120) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 121) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 122) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 124) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 125) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 126) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 128) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 129) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 132) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 134) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 135) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 136) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 138) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 139) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 142) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 144) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 145) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 146) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 148) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 149) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 152) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 154) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 155) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 156) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 158) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 159) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 64,
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
        (4, 162) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 163) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 164) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 166) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 167) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 168) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 170) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 171) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 174) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 176) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 177) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 178) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 180) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 181) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 184) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 186) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 187) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 188) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 190) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 191) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 194) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 196) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 197) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 198) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 200) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 201) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 64,
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
        (4, 204) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 205) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 206) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 208) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 209) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 210) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 212) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 213) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 216) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 218) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 219) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 220) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 222) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 223) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 226) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 228) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 229) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 230) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 232) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 233) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 236) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 238) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 239) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 240) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 242) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 243) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 64,
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
        (4, 246) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 247) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (4, 248) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 250) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (4, 251) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (4, 252) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (4, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 2) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 3) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 6) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 8) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 9) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 10) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 12) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 13) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 16) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 18) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 19) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 20) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 22) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 23) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 26) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 28) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 29) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 30) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 32) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 33) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 77,
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
                definition: 78,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 79,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (5, 36) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 37) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 38) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 40) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 41) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 42) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 44) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 45) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 48) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 50) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 51) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 52) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 54) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 55) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 58) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 60) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 61) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 62) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 64) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 65) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 68) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 70) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 71) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 72) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 74) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 75) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 77,
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
        (5, 78) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 79) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 80) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 82) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 83) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 84) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 86) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 87) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 90) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 92) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 93) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 94) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 96) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 97) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 100) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 102) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 103) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 104) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 106) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 107) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 110) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 112) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 113) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 114) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 116) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 117) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 77,
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
        (5, 120) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 121) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 122) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 124) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 125) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 126) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 128) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 129) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 132) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 134) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 135) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 136) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 138) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 139) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 142) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 144) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 145) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 146) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 148) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 149) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 152) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 154) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 155) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 156) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 158) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 159) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 77,
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
        (5, 162) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 163) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 164) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 166) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 167) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 168) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 170) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 171) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 174) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 176) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 177) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 178) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 180) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 181) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 184) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 186) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 187) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 188) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 190) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 191) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 194) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 196) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 197) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 198) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 200) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 201) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 77,
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
        (5, 204) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 205) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 206) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 208) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 209) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 210) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 212) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 213) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 216) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 218) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 219) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 220) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 222) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 223) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 226) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 228) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 229) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 230) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 232) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 233) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 236) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 238) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 239) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 240) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 242) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 243) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 77,
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
        (5, 246) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 247) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (5, 248) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 250) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (5, 251) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (5, 252) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (5, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 2) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 3) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 6) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 8) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 9) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 10) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 12) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 13) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 16) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 18) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 19) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 20) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 22) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 23) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 26) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 28) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 29) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 30) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 32) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 33) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 90,
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
                definition: 91,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 92,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (6, 36) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 37) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 38) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 40) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 41) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 42) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 44) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 45) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 48) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 50) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 51) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 52) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 54) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 55) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 58) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 60) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 61) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 62) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 64) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 65) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 68) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 70) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 71) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 72) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 74) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 75) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 90,
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
        (6, 78) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 79) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 80) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 82) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 83) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 84) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 86) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 87) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 90) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 92) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 93) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 94) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 96) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 97) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 100) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 102) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 103) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 104) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 106) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 107) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 110) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 112) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 113) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 114) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 116) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 117) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 90,
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
        (6, 120) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 121) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 122) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 124) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 125) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 126) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 128) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 129) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 132) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 134) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 135) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 136) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 138) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 139) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 142) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 144) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 145) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 146) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 148) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 149) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 152) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 154) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 155) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 156) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 158) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 159) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 90,
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
        (6, 162) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 163) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 164) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 166) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 167) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 168) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 170) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 171) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 174) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 176) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 177) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 178) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 180) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 181) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 184) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 186) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 187) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 188) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 190) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 191) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 194) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 196) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 197) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 198) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 200) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 201) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 90,
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
        (6, 204) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 205) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 206) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 208) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 209) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 210) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 212) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 213) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 216) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 218) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 219) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 220) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 222) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 223) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 226) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 228) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 229) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 230) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 232) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 233) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 236) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 238) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 239) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 240) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 242) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 243) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 90,
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
        (6, 246) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 247) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (6, 248) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 250) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (6, 251) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (6, 252) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (6, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 2) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 3) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 6) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 8) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 9) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 10) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 12) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 13) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 16) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 18) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 19) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 20) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 22) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 23) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 26) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 28) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 29) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 30) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 32) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 33) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 103,
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
                definition: 104,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 105,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 36) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 37) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 38) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 40) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 41) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 42) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 44) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 45) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 48) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 50) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 51) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 52) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 54) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 55) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 58) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 60) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 61) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 62) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 64) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 65) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 68) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 70) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 71) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 72) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 74) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 75) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 103,
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
                definition: 106,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 107,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 78) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 79) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 80) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 82) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 83) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 84) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 86) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 87) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 90) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 92) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 93) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 94) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 96) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 97) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 100) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 102) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 103) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 104) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 106) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 107) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 110) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 112) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 113) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 114) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 116) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 117) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 103,
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
                definition: 108,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 109,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 120) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 121) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 122) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 124) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 125) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 126) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 128) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 129) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 132) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 134) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 135) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 136) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 138) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 139) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 142) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 144) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 145) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 146) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 148) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 149) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 152) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 154) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 155) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 156) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 158) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 159) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 103,
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
                definition: 110,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 111,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 162) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 163) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 164) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 166) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 167) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 168) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 170) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 171) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 174) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 176) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 177) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 178) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 180) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 181) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 184) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 186) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 187) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 188) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 190) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 191) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 194) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 196) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 197) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 198) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 200) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 201) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 103,
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
                definition: 112,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 113,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 204) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 205) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 206) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 208) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 209) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 210) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 212) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 213) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 216) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 218) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 219) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 220) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 222) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 223) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 226) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 228) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 229) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 230) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 232) => vec![Packet {
            definition: 7,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 233) => vec![Packet {
            definition: 8,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 236) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 238) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 239) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 240) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 242) => vec![Packet {
            definition: 12,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 243) => vec![Packet {
            definition: 20,
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
            definition: 21,
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
                definition: 22,
                roles: vec![Role::PrivateQ, Role::Q],
                clears: vec![],
            },
            Packet {
                definition: 103,
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
                definition: 114,
                roles: vec![Role::O, Role::PrivateO],
                clears: vec![],
            },
            Packet {
                definition: 115,
                roles: vec![Role::PublicLse, Role::Lse],
                clears: vec![],
            },
        ],
        (7, 246) => vec![Packet {
            definition: 26,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 247) => vec![Packet {
            definition: 27,
            roles: vec![Role::Y, Role::X, Role::Gate],
            clears: vec![],
        }],
        (7, 248) => vec![Packet {
            definition: 9,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 250) => vec![Packet {
            definition: 10,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        (7, 251) => vec![Packet {
            definition: 19,
            roles: vec![Role::Y, Role::X],
            clears: vec![],
        }],
        (7, 252) => vec![Packet {
            definition: 11,
            roles: vec![Role::W, Role::X, Role::Y, Role::Workspace],
            clears: vec![(Role::Workspace, 0, 64)],
        }],
        (7, 254) => vec![Packet {
            definition: 6,
            roles: vec![Role::X, Role::W, Role::Y],
            clears: vec![],
        }],
        _ => {
            return Err(Error::Unsupported(
                "actual fixed_promptB4 source packet site absent".into(),
            ));
        }
    })
}
