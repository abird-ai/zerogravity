//! Fixed original layer3 source fields. No runtime manifest/shape selector.
use super::super::cubin_manifests::ProjectionImage;
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup,
};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttentionCase {
    InitialOne,
    CachedOneAt1,
    Initial64,
    ContinuedTwoAt64,
    CachedOneAt66,
}
impl AttentionCase {
    pub(super) const ALL: [Self; 5] = [
        Self::InitialOne,
        Self::CachedOneAt1,
        Self::Initial64,
        Self::ContinuedTwoAt64,
        Self::CachedOneAt66,
    ];
    pub(super) const fn rows(self) -> usize {
        match self {
            Self::Initial64 => 64,
            Self::ContinuedTwoAt64 => 2,
            _ => 1,
        }
    }
    pub(super) const fn base(self) -> usize {
        match self {
            Self::CachedOneAt1 => 1,
            Self::ContinuedTwoAt64 => 64,
            Self::CachedOneAt66 => 66,
            _ => 0,
        }
    }
    pub(super) const fn block(self) -> usize {
        match self {
            Self::InitialOne | Self::CachedOneAt1 => 4,
            _ => 1,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum AttentionStage {
    Qkv,
    Qk,
    Kv,
    Fa,
    Sigmoid,
    Multiply,
    Out,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum AttentionRole {
    AttentionOutput,
    BlockTableI32,
    CosSinSourcePrefix67,
    CuQI32,
    DummyCuKI32,
    Gate,
    K,
    KGamma,
    KScale,
    LseF32,
    OriginalKvState,
    OutInput,
    OutW,
    PackedQkv,
    PartialLseF32,
    PartialOF32,
    PositionsI64Pitch65,
    PrivateO,
    Projected,
    Q,
    QGamma,
    QkvInput,
    QkvW,
    SigmoidGate,
    SeqUsedI32,
    SlotsI64,
    VScale,
    Workspace,
}

pub(super) struct AttentionLaunch {
    pub(super) image: ProjectionImage,
    pub(super) roles: &'static [AttentionRole],
    pub(super) manifest: CubinKernelManifest,
}
const fn literal<const N: usize>(s: &str) -> [u8; N] {
    const fn nibble(x: u8) -> u8 {
        match x {
            b'0'..=b'9' => x - b'0',
            b'a'..=b'f' => x - b'a' + 10,
            _ => panic!("invalid source literal"),
        }
    }
    assert!(s.len() == N * 2);
    let b = s.as_bytes();
    let mut out = [0; N];
    let mut i = 0;
    while i < N {
        out[i] = nibble(b[i * 2]) * 16 + nibble(b[i * 2 + 1]);
        i += 1;
    }
    out
}
// Qkv1_0; immutable image 94fc9c63d0562b88395c3d5001397aa1a3a3f572450a00db627750b817c7de56; ABI SHA e2f3cd4ce6b76cc18354618b706543a49cf9af2b2d2bee2acad7c04e9394145f.
const T_0_0: [u8; 152] = literal(
    "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000040000001400000004000001000000010000000100000001000000000000000000000000000000000000000000000000000000000000000000803f00000000000000000000000000000000000000000000000000000000",
);
const T_0_1: [u8; 4] = literal("0000803f");
const T_0_2: [u8; 4] = literal("00000000");
fn launch_0() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::Bf16Gemv2TSm86,roles:&[AttentionRole::QkvW,AttentionRole::QkvInput,AttentionRole::PackedQkv],manifest:CubinKernelManifest{entry:"_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi4ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_".into(),binary_sm:86,grid:[640, 1, 1],block:[128, 1, 1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:10485760,alignment:16,io:CubinIo::Read},CubinBinding{bytes:2048,alignment:16,io:CubinIo::Read},CubinBinding{bytes:10240,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:T_0_0.to_vec()},CubinParameter{offset:152,template:T_0_1.to_vec()},CubinParameter{offset:156,template:T_0_2.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:10485760,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:32,binding:2,binding_offset:0,extent:10240,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:48,binding:2,binding_offset:0,extent:10240,alignment:16,io:CubinIo::Write}]}}
}
// Out1_0; immutable image b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d; ABI SHA bc4a70439ee9f39307c59846c75decd14dd9814bc2c4177061b190a698fc6404.
const T_1_0: [u8; 360] = literal(
    "010000000004000000080000010000004000000003000000030000000100000003000000c002000000040000000000000000000000000000000800000000000000200000000000000021ffffffffffff0001000000000000000800000000000000200000000000000021ffffffffffff00010000000000000008000000000000004000000000000000c8ffffffffffff00c8ffffffffffff00800000000000000000000000000000008000000000000000800000000000000008000000000000004000000000000000c8ffffffffffff00c8ffffffffffff00800000000000000000000000000000008000000000000000800000000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_1() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::CutlassWmmaSm80,roles:&[AttentionRole::OutW,AttentionRole::OutInput,AttentionRole::Projected,AttentionRole::Workspace],manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8, 8, 3],block:[32, 1, 1],dynamic_shared_bytes:16896,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:4194304,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:2048,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8519680,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:T_1_0.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:4194304,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:304,binding:3,binding_offset:0,extent:6144,alignment:16,io:CubinIo::Write}]}}
}
// Out1_1; immutable image 074807d671d9156ccc56b5a8cbe82f1ba8fae21c8169e674b31b4b8603ba42ec; ABI SHA c688c3c56911c6313280fceb5481b09238f277812ae2c2d194487a85f13796df.
const T_2_0: [u8; 60] = literal(
    "0000803f0000000000040000010000000004000000040000000400000004000001000000000400000300000000000000000000000000000001000000",
);
const T_2_1: [u8; 8] = literal("0000000000000000");
const T_2_2: [u8; 8] = literal("0000000000000000");
const T_2_3: [u8; 8] = literal("0000000000000000");
const T_2_4: [u8; 8] = literal("0000000000000000");
const T_2_5: [u8; 8] = literal("0000000000000000");
const T_2_6: [u8; 8] = literal("0000000000000000");
const T_2_7: [u8; 8] = literal("0000000000000000");
const T_2_8: [u8; 8] = literal("0000000000000000");
const T_2_9: [u8; 8] = literal("0000000000000000");
const T_2_10: [u8; 8] = literal("0000000000000000");
const T_2_11: [u8; 8] = literal("0000000000000000");
const T_2_12: [u8; 8] = literal("0000000000000000");
const T_2_13: [u8; 8] = literal("0000000000000000");
const T_2_14: [u8; 8] = literal("0000000000000000");
const T_2_15: [u8; 8] = literal("0000000000000000");
const T_2_16: [u8; 8] = literal("0000000000000000");
const T_2_17: [u8; 8] = literal("0000000000000000");
const T_2_18: [u8; 8] = literal("0000000000000000");
const T_2_19: [u8; 8] = literal("0000000000000000");
const T_2_20: [u8; 8] = literal("0000000000000000");
fn launch_2() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::Bf16SplitKReduceSm86,roles:&[AttentionRole::Projected,AttentionRole::Workspace],manifest:CubinKernelManifest{entry:"_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(),binary_sm:86,grid:[32, 1, 1],block:[32, 16, 1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:2048,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8519680,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_2_0.to_vec()},CubinParameter{offset:64,template:T_2_1.to_vec()},CubinParameter{offset:72,template:T_2_2.to_vec()},CubinParameter{offset:80,template:T_2_3.to_vec()},CubinParameter{offset:88,template:T_2_4.to_vec()},CubinParameter{offset:96,template:T_2_5.to_vec()},CubinParameter{offset:104,template:T_2_6.to_vec()},CubinParameter{offset:112,template:T_2_7.to_vec()},CubinParameter{offset:120,template:T_2_8.to_vec()},CubinParameter{offset:128,template:T_2_9.to_vec()},CubinParameter{offset:136,template:T_2_10.to_vec()},CubinParameter{offset:144,template:T_2_11.to_vec()},CubinParameter{offset:152,template:T_2_12.to_vec()},CubinParameter{offset:160,template:T_2_13.to_vec()},CubinParameter{offset:168,template:T_2_14.to_vec()},CubinParameter{offset:176,template:T_2_15.to_vec()},CubinParameter{offset:184,template:T_2_16.to_vec()},CubinParameter{offset:192,template:T_2_17.to_vec()},CubinParameter{offset:200,template:T_2_18.to_vec()},CubinParameter{offset:208,template:T_2_19.to_vec()},CubinParameter{offset:216,template:T_2_20.to_vec()}],fixups:vec![CubinPointerFixup{parameter:1,byte_offset:0,binding:1,binding_offset:0,extent:6144,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:2,byte_offset:0,binding:0,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:3,byte_offset:0,binding:0,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:4,byte_offset:0,binding:0,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Write}]}}
}
// Qkv64_0; immutable image 6ab9468c2d21b9b24c66fbb4380efca47a9057ddcc6adb4eb83997460986a83e; ABI SHA 957b6002a6730bad4d41a681d281a28e2755d1271ae8cc9b97e9ed4218675b0d.
const T_3_0: [u8; 192] = literal(
    "000000000000000000000000000000000000000000000000000002000000000000000200000000004000faffffffffff4000feffffffffff0002000000000000000200000000000000000500000000000000050000000000000400000004000000140000001400000014000040000000000400000000000000000000000000000002000024000000000000000000000000000000000000000000803f000000000000000000000000000000000000000000000010000000100000000000000000",
);
fn launch_3() -> AttentionLaunch {
    AttentionLaunch {
        image: ProjectionImage::AttentionAmpere256x128Sm80,
        roles: &[
            AttentionRole::QkvW,
            AttentionRole::QkvInput,
            AttentionRole::PackedQkv,
            AttentionRole::Workspace,
        ],
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
                    bytes: 131072,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 655360,
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
                template: T_3_0.to_vec(),
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
                    extent: 131072,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 16,
                    binding: 3,
                    binding_offset: 0,
                    extent: 2621440,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 120,
                    binding: 3,
                    binding_offset: 0,
                    extent: 2621440,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 655360,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        },
    }
}
// Qkv64_1; immutable image 074807d671d9156ccc56b5a8cbe82f1ba8fae21c8169e674b31b4b8603ba42ec; ABI SHA d3b4cc9d41e830fdec228740fce0c519c4348b8251e86ac1042c1c307018def0.
const T_4_0: [u8; 60] = literal(
    "0000803f0000000000140000400000000014000000140000001400000014000001000000000005000200000000000000000000000000000001000000",
);
const T_4_1: [u8; 8] = literal("0000000000000000");
const T_4_2: [u8; 8] = literal("0000000000000000");
const T_4_3: [u8; 8] = literal("0000000000000000");
const T_4_4: [u8; 8] = literal("0000000000000000");
const T_4_5: [u8; 8] = literal("0000000000000000");
const T_4_6: [u8; 8] = literal("0000000000000000");
const T_4_7: [u8; 8] = literal("0000000000000000");
const T_4_8: [u8; 8] = literal("0000000000000000");
const T_4_9: [u8; 8] = literal("0000000000000000");
const T_4_10: [u8; 8] = literal("0000000000000000");
const T_4_11: [u8; 8] = literal("0000000000000000");
const T_4_12: [u8; 8] = literal("0000000000000000");
const T_4_13: [u8; 8] = literal("0000000000000000");
const T_4_14: [u8; 8] = literal("0000000000000000");
const T_4_15: [u8; 8] = literal("0000000000000000");
const T_4_16: [u8; 8] = literal("0000000000000000");
const T_4_17: [u8; 8] = literal("0000000000000000");
const T_4_18: [u8; 8] = literal("0000000000000000");
const T_4_19: [u8; 8] = literal("0000000000000000");
const T_4_20: [u8; 8] = literal("0000000000000000");
fn launch_4() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::Bf16SplitKReduceSm86,roles:&[AttentionRole::PackedQkv,AttentionRole::Workspace],manifest:CubinKernelManifest{entry:"_ZN8cublasLt19splitKreduce_kernelILi32ELi16Eif13__nv_bfloat16fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(),binary_sm:86,grid:[160, 4, 1],block:[32, 16, 1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:655360,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8519680,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_4_0.to_vec()},CubinParameter{offset:64,template:T_4_1.to_vec()},CubinParameter{offset:72,template:T_4_2.to_vec()},CubinParameter{offset:80,template:T_4_3.to_vec()},CubinParameter{offset:88,template:T_4_4.to_vec()},CubinParameter{offset:96,template:T_4_5.to_vec()},CubinParameter{offset:104,template:T_4_6.to_vec()},CubinParameter{offset:112,template:T_4_7.to_vec()},CubinParameter{offset:120,template:T_4_8.to_vec()},CubinParameter{offset:128,template:T_4_9.to_vec()},CubinParameter{offset:136,template:T_4_10.to_vec()},CubinParameter{offset:144,template:T_4_11.to_vec()},CubinParameter{offset:152,template:T_4_12.to_vec()},CubinParameter{offset:160,template:T_4_13.to_vec()},CubinParameter{offset:168,template:T_4_14.to_vec()},CubinParameter{offset:176,template:T_4_15.to_vec()},CubinParameter{offset:184,template:T_4_16.to_vec()},CubinParameter{offset:192,template:T_4_17.to_vec()},CubinParameter{offset:200,template:T_4_18.to_vec()},CubinParameter{offset:208,template:T_4_19.to_vec()},CubinParameter{offset:216,template:T_4_20.to_vec()}],fixups:vec![CubinPointerFixup{parameter:1,byte_offset:0,binding:1,binding_offset:0,extent:2621440,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:2,byte_offset:0,binding:0,binding_offset:0,extent:655360,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:3,byte_offset:0,binding:0,binding_offset:0,extent:655360,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:4,byte_offset:0,binding:0,binding_offset:0,extent:655360,alignment:16,io:CubinIo::Write}]}}
}
// Out64_0; immutable image c3361a22db27ce017fec90a46f66c508cad5d668d606070b214b0351b3cf9588; ABI SHA 982a7da961ec64834b09b8a645a3231967f9cf1d4fbbbd82f150ffda447996b5.
const T_5_0: [u8; 192] = literal(
    "000000000000000000000000000000000000000000000000000001000000000000000100000000008000fdffffffffff8000fdffffffffff0000000000000000000000000000000000000000000000000000000000000000000800000008000000040000000400000004000040000000000800000000000000000000000000000008000000000000000000000000000000000000000000000000803f000000000000000000000000000000000000000000000010000000100000000000000000",
);
fn launch_5() -> AttentionLaunch {
    AttentionLaunch {
        image: ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
        roles: &[
            AttentionRole::OutW,
            AttentionRole::OutInput,
            AttentionRole::Projected,
        ],
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn".into(),
            binary_sm: 80,
            grid: [16, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 32768,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 4194304,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 262144,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 131072,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: T_5_0.to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 4194304,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 8,
                    binding: 1,
                    binding_offset: 0,
                    extent: 262144,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 16,
                    binding: 2,
                    binding_offset: 0,
                    extent: 131072,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 131072,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        },
    }
}
// Qkv2_0; immutable image b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d; ABI SHA a2b1593b03e9ce548e9f359a2e7b896a038a590d93b829363dc0987181bb9411.
const T_6_0: [u8; 360] = literal(
    "0200000000140000000400000100000040010000010000000300000000000000010000000004000000000000000000000000000000000000000400000000000000100000000000000091ffffffffffff0001000000000000000400000000000000100000000000000091ffffffffffff00010000000000000028000000000000004001000000000000e8feffffffffff00e8feffffffffff00800200000000000000000000000000008002000000000000800200000000000028000000000000004001000000000000e8feffffffffff00e8feffffffffff00800200000000000000000000000000008002000000000000800200000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_6() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::CutlassWmmaSm80,roles:&[AttentionRole::QkvW,AttentionRole::QkvInput,AttentionRole::PackedQkv],manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8, 40, 1],block:[32, 1, 1],dynamic_shared_bytes:16896,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:10485760,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:20480,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:T_6_0.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:10485760,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:20480,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:304,binding:2,binding_offset:0,extent:20480,alignment:16,io:CubinIo::Write}]}}
}
// Out2_0; immutable image b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d; ABI SHA d8d2c86a12c5b43abd8dee203c07a66383d07d8fb464012461cf43edead47d5d.
const T_7_0: [u8; 360] = literal(
    "0200000000040000000800000100000020000000010000000300000000000000010000000008000000000000000000000000000000000000000800000000000000800000000000000081feffffffffff0001000000000000000800000000000000800000000000000081feffffffffff00010000000000000008000000000000002000000000000000e8ffffffffffff00e8ffffffffffff00800000000000000080000000000000000001000000000000000100000000000008000000000000002000000000000000e8ffffffffffff00e8ffffffffffff00800000000000000080000000000000000001000000000000000100000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_7() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::CutlassWmmaSm80,roles:&[AttentionRole::OutW,AttentionRole::OutInput,AttentionRole::Projected],manifest:CubinKernelManifest{entry:"_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE".into(),binary_sm:80,grid:[8, 4, 1],block:[128, 1, 1],dynamic_shared_bytes:33792,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:4194304,alignment:16,io:CubinIo::Read},CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:T_7_0.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:8192,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:4194304,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:304,binding:2,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write}]}}
}
// QK_InitialOne; immutable image 9049ad21c964126fe32a394f6e5335e9ec0901cb031559db3cab0e63d93b4664; ABI SHA be91b95609fa619300decdc4b90338bf8c0d03a01b01e004cd1fa4b9ec1e9359.
const T_8_0: [u8; 8] = literal("0000000000000000");
const T_8_1: [u8; 8] = literal("0000000000000000");
const T_8_2: [u8; 8] = literal("0000000000000000");
const T_8_3: [u8; 8] = literal("0000000000000000");
const T_8_4: [u8; 8] = literal("0000000000000000");
const T_8_5: [u8; 8] = literal("0000000000000000");
const T_8_6: [u8; 8] = literal("0000000000000000");
const T_8_7: [u8; 8] = literal("0000000000000000");
const T_8_8: [u8; 8] = literal("0000000000000000");
const T_8_9: [u8; 4] = literal("00140000");
const T_8_10: [u8; 4] = literal("00140000");
const T_8_11: [u8; 4] = literal("00080000");
const T_8_12: [u8; 4] = literal("00020000");
const T_8_13: [u8; 4] = literal("00080000");
const T_8_14: [u8; 4] = literal("40000000");
const T_8_15: [u8; 4] = literal("41000000");
const T_8_16: [u8; 8] = literal("0000000000000000");
const T_8_17: [u8; 8] = literal("0000000000000000");
fn launch_8() -> AttentionLaunch {
    AttentionLaunch {
        image: ProjectionImage::AttentionQkSm86,
        roles: &[
            AttentionRole::PackedQkv,
            AttentionRole::Q,
            AttentionRole::K,
            AttentionRole::Gate,
            AttentionRole::QGamma,
            AttentionRole::KGamma,
            AttentionRole::CosSinSourcePrefix67,
            AttentionRole::PositionsI64Pitch65,
        ],
        manifest: CubinKernelManifest {
            entry: "_fused_qk_rmsnorm_rope_gate_kernel".into(),
            binary_sm: 86,
            grid: [1, 10, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 16,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 10240,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 4096,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 1024,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 4096,
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
                    bytes: 8576,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 1560,
                    alignment: 8,
                    io: CubinIo::Read,
                },
            ],
            parameters: vec![
                CubinParameter {
                    offset: 0,
                    template: T_8_0.to_vec(),
                },
                CubinParameter {
                    offset: 8,
                    template: T_8_1.to_vec(),
                },
                CubinParameter {
                    offset: 16,
                    template: T_8_2.to_vec(),
                },
                CubinParameter {
                    offset: 24,
                    template: T_8_3.to_vec(),
                },
                CubinParameter {
                    offset: 32,
                    template: T_8_4.to_vec(),
                },
                CubinParameter {
                    offset: 40,
                    template: T_8_5.to_vec(),
                },
                CubinParameter {
                    offset: 48,
                    template: T_8_6.to_vec(),
                },
                CubinParameter {
                    offset: 56,
                    template: T_8_7.to_vec(),
                },
                CubinParameter {
                    offset: 64,
                    template: T_8_8.to_vec(),
                },
                CubinParameter {
                    offset: 72,
                    template: T_8_9.to_vec(),
                },
                CubinParameter {
                    offset: 76,
                    template: T_8_10.to_vec(),
                },
                CubinParameter {
                    offset: 80,
                    template: T_8_11.to_vec(),
                },
                CubinParameter {
                    offset: 84,
                    template: T_8_12.to_vec(),
                },
                CubinParameter {
                    offset: 88,
                    template: T_8_13.to_vec(),
                },
                CubinParameter {
                    offset: 92,
                    template: T_8_14.to_vec(),
                },
                CubinParameter {
                    offset: 96,
                    template: T_8_15.to_vec(),
                },
                CubinParameter {
                    offset: 104,
                    template: T_8_16.to_vec(),
                },
                CubinParameter {
                    offset: 112,
                    template: T_8_17.to_vec(),
                },
            ],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 8192,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 1,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 8192,
                    extent: 1024,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 2,
                    byte_offset: 0,
                    binding: 1,
                    binding_offset: 0,
                    extent: 4096,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 3,
                    byte_offset: 0,
                    binding: 2,
                    binding_offset: 0,
                    extent: 1024,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 4,
                    byte_offset: 0,
                    binding: 3,
                    binding_offset: 0,
                    extent: 4096,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 5,
                    byte_offset: 0,
                    binding: 4,
                    binding_offset: 0,
                    extent: 512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 6,
                    byte_offset: 0,
                    binding: 5,
                    binding_offset: 0,
                    extent: 512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 7,
                    byte_offset: 0,
                    binding: 6,
                    binding_offset: 0,
                    extent: 8576,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 8,
                    byte_offset: 0,
                    binding: 7,
                    binding_offset: 0,
                    extent: 1048,
                    alignment: 8,
                    io: CubinIo::Read,
                },
            ],
        },
    }
}
// KV_InitialOne; immutable image 0781fbbcae3a3f26cf97205af806e6e9001413827e66e298d4a0743cde067035; ABI SHA f15fab025664a6bf9d9d3283ce0d4587f951db238eb266345c615f23d7798459.
const T_9_0: [u8; 8] = literal("0000000000000000");
const T_9_1: [u8; 8] = literal("0000000000000000");
const T_9_2: [u8; 8] = literal("0000000000000000");
const T_9_3: [u8; 8] = literal("0000000000000000");
const T_9_4: [u8; 8] = literal("0000000000000000");
const T_9_5: [u8; 8] = literal("0080080000000000");
const T_9_6: [u8; 8] = literal("0004000000000000");
const T_9_7: [u8; 8] = literal("0002000000000000");
const T_9_8: [u8; 8] = literal("0002000000000000");
const T_9_9: [u8; 8] = literal("0002000000000000");
const T_9_10: [u8; 4] = literal("02000000");
const T_9_11: [u8; 4] = literal("00010000");
const T_9_12: [u8; 4] = literal("20020000");
const T_9_13: [u8; 8] = literal("0000000000000000");
const T_9_14: [u8; 8] = literal("0000000000000000");
const T_9_15: [u8; 4] = literal("00000000");
fn launch_9() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::AttentionKvSm86,roles:&[AttentionRole::K,AttentionRole::PackedQkv,AttentionRole::OriginalKvState,AttentionRole::SlotsI64,AttentionRole::KScale,AttentionRole::VScale],manifest:CubinKernelManifest{entry:"_ZN4vllm30reshape_and_cache_flash_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKlllllliiiPKfSB_i".into(),binary_sm:86,grid:[1, 1, 1],block:[512, 1, 1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:1024,alignment:16,io:CubinIo::Read},CubinBinding{bytes:10240,alignment:16,io:CubinIo::Read},CubinBinding{bytes:5570560,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8,alignment:8,io:CubinIo::Read},CubinBinding{bytes:4,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_9_0.to_vec()},CubinParameter{offset:8,template:T_9_1.to_vec()},CubinParameter{offset:16,template:T_9_2.to_vec()},CubinParameter{offset:24,template:T_9_3.to_vec()},CubinParameter{offset:32,template:T_9_4.to_vec()},CubinParameter{offset:40,template:T_9_5.to_vec()},CubinParameter{offset:48,template:T_9_6.to_vec()},CubinParameter{offset:56,template:T_9_7.to_vec()},CubinParameter{offset:64,template:T_9_8.to_vec()},CubinParameter{offset:72,template:T_9_9.to_vec()},CubinParameter{offset:80,template:T_9_10.to_vec()},CubinParameter{offset:84,template:T_9_11.to_vec()},CubinParameter{offset:88,template:T_9_12.to_vec()},CubinParameter{offset:96,template:T_9_13.to_vec()},CubinParameter{offset:104,template:T_9_14.to_vec()},CubinParameter{offset:112,template:T_9_15.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:1024,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:1,byte_offset:0,binding:1,binding_offset:9216,extent:1024,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:2,byte_offset:0,binding:2,binding_offset:0,extent:5570560,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:3,byte_offset:0,binding:2,binding_offset:512,extent:5570048,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:4,byte_offset:0,binding:3,binding_offset:0,extent:8,alignment:8,io:CubinIo::Read},CubinPointerFixup{parameter:13,byte_offset:0,binding:4,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:14,byte_offset:0,binding:5,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read}]}}
}
// FA_InitialOne; immutable image 915016a9eaee3a391e29bbfcf00ee8de82e4da53c4865d080c04505d0d736345; ABI SHA 352ac51e5a5ca605a72435b685fb86344d52703060c6aba8138969fda7d72f2e.
const T_10_0: [u8; 472] = literal(
    "0000000000000000000000000000000000000000000000000004000000000000008008000000000000800800000000000001000000000000000400000000000000040000000000000004000000000000000200000000000000020000000000000200000002000000010000000000000000000000000000000000000000000000000400000000000000010000000000000004000000000000000000000000000000000000000000000000000000000000010000000400000001000000000000000001000080000000800000000001000000000000040000000000803d3baab83d000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000200200000000803fff0000000000803f0000803dffffffffffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100010001000000000000000000000000000000000000000101000000000000",
);
fn launch_10() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::AttentionFaSplitSm80,roles:&[AttentionRole::Q,AttentionRole::OriginalKvState,AttentionRole::LseF32,AttentionRole::DummyCuKI32,AttentionRole::PrivateO,AttentionRole::SeqUsedI32,AttentionRole::BlockTableI32],manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1, 1, 2],block:[128, 1, 1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:5570560,alignment:16,io:CubinIo::Read},CubinBinding{bytes:32,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8,alignment:4,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_10_0.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:8,binding:1,binding_offset:0,extent:5570560,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:512,extent:5570048,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:112,binding:4,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:160,binding:2,binding_offset:0,extent:32,alignment:4,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:232,binding:3,binding_offset:0,extent:8,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:248,binding:5,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:352,binding:6,binding_offset:0,extent:16,alignment:4,io:CubinIo::Read}]}}
}
// FA_CachedOneAt1; immutable image 915016a9eaee3a391e29bbfcf00ee8de82e4da53c4865d080c04505d0d736345; ABI SHA 997aa113efbaeb94f0610799ec72d7df0c8d5cef94b0636bdbdbea679490b840.
const T_11_0: [u8; 472] = literal(
    "0000000000000000000000000000000000000000000000000004000000000000008008000000000000800800000000000001000000000000000400000000000000040000000000000004000000000000000200000000000000020000000000000200000002000000010000000000000000000000000000000000000000000000000400000000000000010000000000000004000000000000000000000000000000000000000000000000000000000000010000000400000002000000000000000001000080000000800000000001000000000000040000000000803d3baab83d000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000200200000000803fff0000000000803f0000803dffffffffffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100010001000000000000000000000000000000000000000101000000000000",
);
fn launch_11() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::AttentionFaSplitSm80,roles:&[AttentionRole::Q,AttentionRole::OriginalKvState,AttentionRole::LseF32,AttentionRole::DummyCuKI32,AttentionRole::PrivateO,AttentionRole::SeqUsedI32,AttentionRole::BlockTableI32],manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1, 1, 2],block:[128, 1, 1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:5570560,alignment:16,io:CubinIo::Read},CubinBinding{bytes:32,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8,alignment:4,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_11_0.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:8,binding:1,binding_offset:0,extent:5570560,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:512,extent:5570048,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:112,binding:4,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:160,binding:2,binding_offset:0,extent:32,alignment:4,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:232,binding:3,binding_offset:0,extent:8,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:248,binding:5,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:352,binding:6,binding_offset:0,extent:16,alignment:4,io:CubinIo::Read}]}}
}
// QK_Initial64; immutable image 9049ad21c964126fe32a394f6e5335e9ec0901cb031559db3cab0e63d93b4664; ABI SHA 72c0085e1fd0b55c3f7e526aef09129d6842624cedce3cca12a2f098052920c1.
const T_12_0: [u8; 8] = literal("0000000000000000");
const T_12_1: [u8; 8] = literal("0000000000000000");
const T_12_2: [u8; 8] = literal("0000000000000000");
const T_12_3: [u8; 8] = literal("0000000000000000");
const T_12_4: [u8; 8] = literal("0000000000000000");
const T_12_5: [u8; 8] = literal("0000000000000000");
const T_12_6: [u8; 8] = literal("0000000000000000");
const T_12_7: [u8; 8] = literal("0000000000000000");
const T_12_8: [u8; 8] = literal("0000000000000000");
const T_12_9: [u8; 4] = literal("00140000");
const T_12_10: [u8; 4] = literal("00140000");
const T_12_11: [u8; 4] = literal("00080000");
const T_12_12: [u8; 4] = literal("00020000");
const T_12_13: [u8; 4] = literal("00080000");
const T_12_14: [u8; 4] = literal("40000000");
const T_12_15: [u8; 4] = literal("41000000");
const T_12_16: [u8; 8] = literal("0000000000000000");
const T_12_17: [u8; 8] = literal("0000000000000000");
pub(super) fn launch_12() -> AttentionLaunch {
    AttentionLaunch {
        image: ProjectionImage::AttentionQkSm86,
        roles: &[
            AttentionRole::PackedQkv,
            AttentionRole::Q,
            AttentionRole::K,
            AttentionRole::Gate,
            AttentionRole::QGamma,
            AttentionRole::KGamma,
            AttentionRole::CosSinSourcePrefix67,
            AttentionRole::PositionsI64Pitch65,
        ],
        manifest: CubinKernelManifest {
            entry: "_fused_qk_rmsnorm_rope_gate_kernel".into(),
            binary_sm: 86,
            grid: [64, 10, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 16,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 655360,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 262144,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 65536,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 262144,
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
                    bytes: 8576,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 1560,
                    alignment: 8,
                    io: CubinIo::Read,
                },
            ],
            parameters: vec![
                CubinParameter {
                    offset: 0,
                    template: T_12_0.to_vec(),
                },
                CubinParameter {
                    offset: 8,
                    template: T_12_1.to_vec(),
                },
                CubinParameter {
                    offset: 16,
                    template: T_12_2.to_vec(),
                },
                CubinParameter {
                    offset: 24,
                    template: T_12_3.to_vec(),
                },
                CubinParameter {
                    offset: 32,
                    template: T_12_4.to_vec(),
                },
                CubinParameter {
                    offset: 40,
                    template: T_12_5.to_vec(),
                },
                CubinParameter {
                    offset: 48,
                    template: T_12_6.to_vec(),
                },
                CubinParameter {
                    offset: 56,
                    template: T_12_7.to_vec(),
                },
                CubinParameter {
                    offset: 64,
                    template: T_12_8.to_vec(),
                },
                CubinParameter {
                    offset: 72,
                    template: T_12_9.to_vec(),
                },
                CubinParameter {
                    offset: 76,
                    template: T_12_10.to_vec(),
                },
                CubinParameter {
                    offset: 80,
                    template: T_12_11.to_vec(),
                },
                CubinParameter {
                    offset: 84,
                    template: T_12_12.to_vec(),
                },
                CubinParameter {
                    offset: 88,
                    template: T_12_13.to_vec(),
                },
                CubinParameter {
                    offset: 92,
                    template: T_12_14.to_vec(),
                },
                CubinParameter {
                    offset: 96,
                    template: T_12_15.to_vec(),
                },
                CubinParameter {
                    offset: 104,
                    template: T_12_16.to_vec(),
                },
                CubinParameter {
                    offset: 112,
                    template: T_12_17.to_vec(),
                },
            ],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 653312,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 1,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 8192,
                    extent: 646144,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 2,
                    byte_offset: 0,
                    binding: 1,
                    binding_offset: 0,
                    extent: 262144,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 3,
                    byte_offset: 0,
                    binding: 2,
                    binding_offset: 0,
                    extent: 65536,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 4,
                    byte_offset: 0,
                    binding: 3,
                    binding_offset: 0,
                    extent: 262144,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 5,
                    byte_offset: 0,
                    binding: 4,
                    binding_offset: 0,
                    extent: 512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 6,
                    byte_offset: 0,
                    binding: 5,
                    binding_offset: 0,
                    extent: 512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 7,
                    byte_offset: 0,
                    binding: 6,
                    binding_offset: 0,
                    extent: 8576,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 8,
                    byte_offset: 0,
                    binding: 7,
                    binding_offset: 0,
                    extent: 1552,
                    alignment: 8,
                    io: CubinIo::Read,
                },
            ],
        },
    }
}
// KV_Initial64; immutable image 0781fbbcae3a3f26cf97205af806e6e9001413827e66e298d4a0743cde067035; ABI SHA 3539646b0364b22d21b04c6c953aa157d68fdfe1e6402c1e174d7cfae6cf3c77.
const T_13_0: [u8; 8] = literal("0000000000000000");
const T_13_1: [u8; 8] = literal("0000000000000000");
const T_13_2: [u8; 8] = literal("0000000000000000");
const T_13_3: [u8; 8] = literal("0000000000000000");
const T_13_4: [u8; 8] = literal("0000000000000000");
const T_13_5: [u8; 8] = literal("0080080000000000");
const T_13_6: [u8; 8] = literal("0004000000000000");
const T_13_7: [u8; 8] = literal("0002000000000000");
const T_13_8: [u8; 8] = literal("0002000000000000");
const T_13_9: [u8; 8] = literal("0014000000000000");
const T_13_10: [u8; 4] = literal("02000000");
const T_13_11: [u8; 4] = literal("00010000");
const T_13_12: [u8; 4] = literal("20020000");
const T_13_13: [u8; 8] = literal("0000000000000000");
const T_13_14: [u8; 8] = literal("0000000000000000");
const T_13_15: [u8; 4] = literal("00000000");
pub(super) fn launch_13() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::AttentionKvSm86,roles:&[AttentionRole::K,AttentionRole::PackedQkv,AttentionRole::OriginalKvState,AttentionRole::SlotsI64,AttentionRole::KScale,AttentionRole::VScale],manifest:CubinKernelManifest{entry:"_ZN4vllm30reshape_and_cache_flash_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKlllllliiiPKfSB_i".into(),binary_sm:86,grid:[64, 1, 1],block:[512, 1, 1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:65536,alignment:16,io:CubinIo::Read},CubinBinding{bytes:655360,alignment:16,io:CubinIo::Read},CubinBinding{bytes:5570560,alignment:16,io:CubinIo::Write},CubinBinding{bytes:512,alignment:8,io:CubinIo::Read},CubinBinding{bytes:4,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_13_0.to_vec()},CubinParameter{offset:8,template:T_13_1.to_vec()},CubinParameter{offset:16,template:T_13_2.to_vec()},CubinParameter{offset:24,template:T_13_3.to_vec()},CubinParameter{offset:32,template:T_13_4.to_vec()},CubinParameter{offset:40,template:T_13_5.to_vec()},CubinParameter{offset:48,template:T_13_6.to_vec()},CubinParameter{offset:56,template:T_13_7.to_vec()},CubinParameter{offset:64,template:T_13_8.to_vec()},CubinParameter{offset:72,template:T_13_9.to_vec()},CubinParameter{offset:80,template:T_13_10.to_vec()},CubinParameter{offset:84,template:T_13_11.to_vec()},CubinParameter{offset:88,template:T_13_12.to_vec()},CubinParameter{offset:96,template:T_13_13.to_vec()},CubinParameter{offset:104,template:T_13_14.to_vec()},CubinParameter{offset:112,template:T_13_15.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:65536,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:1,byte_offset:0,binding:1,binding_offset:9216,extent:646144,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:2,byte_offset:0,binding:2,binding_offset:0,extent:5570560,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:3,byte_offset:0,binding:2,binding_offset:512,extent:5570048,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:4,byte_offset:0,binding:3,binding_offset:0,extent:512,alignment:8,io:CubinIo::Read},CubinPointerFixup{parameter:13,byte_offset:0,binding:4,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:14,byte_offset:0,binding:5,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read}]}}
}
// FA_Initial64; immutable image b4c2cb3cdcf9d456d5a0c905e468d9fe01a79df46544c8f447b0471ae2b4f6d1; ABI SHA 2545736a332e8dbf2ab3c1efbea435de12de90956965bc7d1cd6491a85936b6f.
const T_14_0: [u8; 472] = literal(
    "0000000000000000000000000000000000000000000000000000000000000000008008000000000000800800000000000008000000000000000400000000000000040000000000000001000000000000000200000000000000020000000000000800000002000000040000000000000000000000000000000000000000000000000000000000000000080000000000000001000000000000000000000000000000000000000000000000000000000000010000004000000040000000000000000001000080000000800000000001000000000000400000000000803d3baab83d000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000200200000000803fff0000000000803f0000803d400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000101010000000000000000000000000000000000000000000100000000000000",
);
fn launch_14() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::AttentionFaCausalCombineSm80,roles:&[AttentionRole::Q,AttentionRole::OriginalKvState,AttentionRole::AttentionOutput,AttentionRole::LseF32,AttentionRole::DummyCuKI32,AttentionRole::CuQI32,AttentionRole::SeqUsedI32,AttentionRole::BlockTableI32],manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb1ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1, 1, 8],block:[128, 1, 1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:262144,alignment:16,io:CubinIo::Read},CubinBinding{bytes:5570560,alignment:16,io:CubinIo::Read},CubinBinding{bytes:262144,alignment:16,io:CubinIo::Write},CubinBinding{bytes:2048,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8,alignment:4,io:CubinIo::Read},CubinBinding{bytes:8,alignment:4,io:CubinIo::Read},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_14_0.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:262144,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:8,binding:1,binding_offset:0,extent:5570560,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:512,extent:5570048,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:112,binding:2,binding_offset:0,extent:262144,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:160,binding:3,binding_offset:0,extent:2048,alignment:4,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:232,binding:4,binding_offset:0,extent:8,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:224,binding:5,binding_offset:0,extent:8,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:248,binding:6,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:352,binding:7,binding_offset:0,extent:16,alignment:4,io:CubinIo::Read}]}}
}
// QK_ContinuedTwoAt64; immutable image 9049ad21c964126fe32a394f6e5335e9ec0901cb031559db3cab0e63d93b4664; ABI SHA ae37066796152a16d8fe7a9c05baba8f4601703e639bc8cc62fb69f0b3f1c17d.
const T_15_0: [u8; 8] = literal("0000000000000000");
const T_15_1: [u8; 8] = literal("0000000000000000");
const T_15_2: [u8; 8] = literal("0000000000000000");
const T_15_3: [u8; 8] = literal("0000000000000000");
const T_15_4: [u8; 8] = literal("0000000000000000");
const T_15_5: [u8; 8] = literal("0000000000000000");
const T_15_6: [u8; 8] = literal("0000000000000000");
const T_15_7: [u8; 8] = literal("0000000000000000");
const T_15_8: [u8; 8] = literal("0000000000000000");
const T_15_9: [u8; 4] = literal("00140000");
const T_15_10: [u8; 4] = literal("00140000");
const T_15_11: [u8; 4] = literal("00080000");
const T_15_12: [u8; 4] = literal("00020000");
const T_15_13: [u8; 4] = literal("00080000");
const T_15_14: [u8; 4] = literal("40000000");
const T_15_15: [u8; 4] = literal("41000000");
const T_15_16: [u8; 8] = literal("0000000000000000");
const T_15_17: [u8; 8] = literal("0000000000000000");
fn launch_15() -> AttentionLaunch {
    AttentionLaunch {
        image: ProjectionImage::AttentionQkSm86,
        roles: &[
            AttentionRole::PackedQkv,
            AttentionRole::Q,
            AttentionRole::K,
            AttentionRole::Gate,
            AttentionRole::QGamma,
            AttentionRole::KGamma,
            AttentionRole::CosSinSourcePrefix67,
            AttentionRole::PositionsI64Pitch65,
        ],
        manifest: CubinKernelManifest {
            entry: "_fused_qk_rmsnorm_rope_gate_kernel".into(),
            binary_sm: 86,
            grid: [2, 10, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 16,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 20480,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 8192,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 2048,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 8192,
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
                    bytes: 8576,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 1560,
                    alignment: 8,
                    io: CubinIo::Read,
                },
            ],
            parameters: vec![
                CubinParameter {
                    offset: 0,
                    template: T_15_0.to_vec(),
                },
                CubinParameter {
                    offset: 8,
                    template: T_15_1.to_vec(),
                },
                CubinParameter {
                    offset: 16,
                    template: T_15_2.to_vec(),
                },
                CubinParameter {
                    offset: 24,
                    template: T_15_3.to_vec(),
                },
                CubinParameter {
                    offset: 32,
                    template: T_15_4.to_vec(),
                },
                CubinParameter {
                    offset: 40,
                    template: T_15_5.to_vec(),
                },
                CubinParameter {
                    offset: 48,
                    template: T_15_6.to_vec(),
                },
                CubinParameter {
                    offset: 56,
                    template: T_15_7.to_vec(),
                },
                CubinParameter {
                    offset: 64,
                    template: T_15_8.to_vec(),
                },
                CubinParameter {
                    offset: 72,
                    template: T_15_9.to_vec(),
                },
                CubinParameter {
                    offset: 76,
                    template: T_15_10.to_vec(),
                },
                CubinParameter {
                    offset: 80,
                    template: T_15_11.to_vec(),
                },
                CubinParameter {
                    offset: 84,
                    template: T_15_12.to_vec(),
                },
                CubinParameter {
                    offset: 88,
                    template: T_15_13.to_vec(),
                },
                CubinParameter {
                    offset: 92,
                    template: T_15_14.to_vec(),
                },
                CubinParameter {
                    offset: 96,
                    template: T_15_15.to_vec(),
                },
                CubinParameter {
                    offset: 104,
                    template: T_15_16.to_vec(),
                },
                CubinParameter {
                    offset: 112,
                    template: T_15_17.to_vec(),
                },
            ],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 18432,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 1,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 8192,
                    extent: 11264,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 2,
                    byte_offset: 0,
                    binding: 1,
                    binding_offset: 0,
                    extent: 8192,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 3,
                    byte_offset: 0,
                    binding: 2,
                    binding_offset: 0,
                    extent: 2048,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 4,
                    byte_offset: 0,
                    binding: 3,
                    binding_offset: 0,
                    extent: 8192,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 5,
                    byte_offset: 0,
                    binding: 4,
                    binding_offset: 0,
                    extent: 512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 6,
                    byte_offset: 0,
                    binding: 5,
                    binding_offset: 0,
                    extent: 512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 7,
                    byte_offset: 0,
                    binding: 6,
                    binding_offset: 0,
                    extent: 8576,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 8,
                    byte_offset: 0,
                    binding: 7,
                    binding_offset: 0,
                    extent: 1056,
                    alignment: 8,
                    io: CubinIo::Read,
                },
            ],
        },
    }
}
// KV_ContinuedTwoAt64; immutable image 0781fbbcae3a3f26cf97205af806e6e9001413827e66e298d4a0743cde067035; ABI SHA 040f80ec3e2fe76c1f892119da6d3bc1aaae6f41919a282f47055d6f10e26f4d.
const T_16_0: [u8; 8] = literal("0000000000000000");
const T_16_1: [u8; 8] = literal("0000000000000000");
const T_16_2: [u8; 8] = literal("0000000000000000");
const T_16_3: [u8; 8] = literal("0000000000000000");
const T_16_4: [u8; 8] = literal("0000000000000000");
const T_16_5: [u8; 8] = literal("0080080000000000");
const T_16_6: [u8; 8] = literal("0004000000000000");
const T_16_7: [u8; 8] = literal("0002000000000000");
const T_16_8: [u8; 8] = literal("0002000000000000");
const T_16_9: [u8; 8] = literal("0014000000000000");
const T_16_10: [u8; 4] = literal("02000000");
const T_16_11: [u8; 4] = literal("00010000");
const T_16_12: [u8; 4] = literal("20020000");
const T_16_13: [u8; 8] = literal("0000000000000000");
const T_16_14: [u8; 8] = literal("0000000000000000");
const T_16_15: [u8; 4] = literal("00000000");
fn launch_16() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::AttentionKvSm86,roles:&[AttentionRole::K,AttentionRole::PackedQkv,AttentionRole::OriginalKvState,AttentionRole::SlotsI64,AttentionRole::KScale,AttentionRole::VScale],manifest:CubinKernelManifest{entry:"_ZN4vllm30reshape_and_cache_flash_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKlllllliiiPKfSB_i".into(),binary_sm:86,grid:[2, 1, 1],block:[512, 1, 1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:2048,alignment:16,io:CubinIo::Read},CubinBinding{bytes:20480,alignment:16,io:CubinIo::Read},CubinBinding{bytes:5570560,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16,alignment:8,io:CubinIo::Read},CubinBinding{bytes:4,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4,alignment:16,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_16_0.to_vec()},CubinParameter{offset:8,template:T_16_1.to_vec()},CubinParameter{offset:16,template:T_16_2.to_vec()},CubinParameter{offset:24,template:T_16_3.to_vec()},CubinParameter{offset:32,template:T_16_4.to_vec()},CubinParameter{offset:40,template:T_16_5.to_vec()},CubinParameter{offset:48,template:T_16_6.to_vec()},CubinParameter{offset:56,template:T_16_7.to_vec()},CubinParameter{offset:64,template:T_16_8.to_vec()},CubinParameter{offset:72,template:T_16_9.to_vec()},CubinParameter{offset:80,template:T_16_10.to_vec()},CubinParameter{offset:84,template:T_16_11.to_vec()},CubinParameter{offset:88,template:T_16_12.to_vec()},CubinParameter{offset:96,template:T_16_13.to_vec()},CubinParameter{offset:104,template:T_16_14.to_vec()},CubinParameter{offset:112,template:T_16_15.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:1,byte_offset:0,binding:1,binding_offset:9216,extent:11264,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:2,byte_offset:0,binding:2,binding_offset:0,extent:5570560,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:3,byte_offset:0,binding:2,binding_offset:512,extent:5570048,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:4,byte_offset:0,binding:3,binding_offset:0,extent:16,alignment:8,io:CubinIo::Read},CubinPointerFixup{parameter:13,byte_offset:0,binding:4,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:14,byte_offset:0,binding:5,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read}]}}
}
// FA_ContinuedTwoAt64; immutable image b4c2cb3cdcf9d456d5a0c905e468d9fe01a79df46544c8f447b0471ae2b4f6d1; ABI SHA 59122529c11beb1943b5abc69eaedaa4bfe9f5f8826d7b2876189d83dac2173b.
const T_17_0: [u8; 472] = literal(
    "0000000000000000000000000000000000000000000000000000000000000000008008000000000000800800000000000008000000000000000400000000000000040000000000000001000000000000000200000000000000020000000000000800000002000000040000000000000000000000000000000000000000000000000000000000000000080000000000000001000000000000000000000000000000000000000000000000000000000000010000000200000042000000000000000001000080000000800000000001000000000000020000000000803d3baab83d000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000200200000000803fff0000000000803f0000803d420000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000101010000000000000000000000000000000000000000000100000000000000",
);
fn launch_17() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::AttentionFaCausalCombineSm80,roles:&[AttentionRole::Q,AttentionRole::OriginalKvState,AttentionRole::AttentionOutput,AttentionRole::LseF32,AttentionRole::DummyCuKI32,AttentionRole::CuQI32,AttentionRole::SeqUsedI32,AttentionRole::BlockTableI32],manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb1ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1, 1, 8],block:[128, 1, 1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},CubinBinding{bytes:5570560,alignment:16,io:CubinIo::Read},CubinBinding{bytes:8192,alignment:16,io:CubinIo::Write},CubinBinding{bytes:64,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8,alignment:4,io:CubinIo::Read},CubinBinding{bytes:8,alignment:4,io:CubinIo::Read},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_17_0.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:8192,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:8,binding:1,binding_offset:0,extent:5570560,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:512,extent:5570048,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:112,binding:2,binding_offset:0,extent:8192,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:160,binding:3,binding_offset:0,extent:64,alignment:4,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:232,binding:4,binding_offset:0,extent:8,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:224,binding:5,binding_offset:0,extent:8,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:248,binding:6,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:352,binding:7,binding_offset:0,extent:16,alignment:4,io:CubinIo::Read}]}}
}
// FA_CachedOneAt66; immutable image 915016a9eaee3a391e29bbfcf00ee8de82e4da53c4865d080c04505d0d736345; ABI SHA de28f1a77576fdd70c2d3032d5c93c450f0319d8adaea5b6042318391be8ab9a.
const T_18_0: [u8; 472] = literal(
    "0000000000000000000000000000000000000000000000000004000000000000008008000000000000800800000000000001000000000000000400000000000000040000000000000004000000000000000200000000000000020000000000000200000002000000010000000000000000000000000000000000000000000000000400000000000000010000000000000004000000000000000000000000000000000000000000000000000000000000010000000400000043000000000000000001000080000000800000000001000000000000040000000000803d3baab83d000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000200200000000803fff0000000000803f0000803dffffffffffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100010002000000000000000000000000000000000000000101000000000000",
);
fn launch_18() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::AttentionFaSplitSm80,roles:&[AttentionRole::Q,AttentionRole::OriginalKvState,AttentionRole::LseF32,AttentionRole::DummyCuKI32,AttentionRole::PrivateO,AttentionRole::PartialOF32,AttentionRole::PartialLseF32,AttentionRole::SeqUsedI32,AttentionRole::BlockTableI32],manifest:CubinKernelManifest{entry:"_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb1ELb0EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[1, 2, 2],block:[128, 1, 1],dynamic_shared_bytes:98304,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:5570560,alignment:16,io:CubinIo::Read},CubinBinding{bytes:32,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8,alignment:4,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},CubinBinding{bytes:64,alignment:16,io:CubinIo::Write},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_18_0.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:8,binding:1,binding_offset:0,extent:5570560,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:512,extent:5570048,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:112,binding:4,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:160,binding:2,binding_offset:0,extent:32,alignment:4,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:232,binding:3,binding_offset:0,extent:8,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:120,binding:5,binding_offset:0,extent:16384,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:168,binding:6,binding_offset:0,extent:64,alignment:4,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:248,binding:7,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:352,binding:8,binding_offset:0,extent:16,alignment:4,io:CubinIo::Read}]}}
}
// FACombine_CachedOneAt66; immutable image b4c2cb3cdcf9d456d5a0c905e468d9fe01a79df46544c8f447b0471ae2b4f6d1; ABI SHA 52e3f25b6fba7762163bfcdc564b22da771fc73c5bfa2bc477f9741e76cbad29.
const T_19_0: [u8; 472] = literal(
    "0000000000000000000000000000000000000000000000000004000000000000008008000000000000800800000000000001000000000000000400000000000000040000000000000004000000000000000200000000000000020000000000000200000002000000010000000000000000000000000000000000000000000000000400000000000000010000000000000004000000000000000000000000000000000000000000000000000000000000010000000400000043000000000000000001000080000000800000000001000000000000040000000000803d3baab83d000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000200200000000803fff0000000000803f0000803dffffffffffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100010002000000000000000000000000000000000000000101000000000000",
);
fn launch_19() -> AttentionLaunch {
    AttentionLaunch{image:ProjectionImage::AttentionFaCausalCombineSm80,roles:&[AttentionRole::Q,AttentionRole::OriginalKvState,AttentionRole::LseF32,AttentionRole::DummyCuKI32,AttentionRole::PrivateO,AttentionRole::PartialOF32,AttentionRole::PartialLseF32,AttentionRole::SeqUsedI32,AttentionRole::BlockTableI32],manifest:CubinKernelManifest{entry:"_ZN5flash32flash_fwd_splitkv_combine_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELi4ELi1ELb1EEEvNS_16Flash_fwd_paramsE".into(),binary_sm:80,grid:[2, 1, 1],block:[128, 1, 1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,bindings:vec![CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:5570560,alignment:16,io:CubinIo::Read},CubinBinding{bytes:32,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8,alignment:4,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Read},CubinBinding{bytes:64,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4,alignment:4,io:CubinIo::Read},CubinBinding{bytes:16,alignment:4,io:CubinIo::Read}],parameters:vec![CubinParameter{offset:0,template:T_19_0.to_vec()}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:8,binding:1,binding_offset:0,extent:5570560,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:512,extent:5570048,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:112,binding:4,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:160,binding:2,binding_offset:0,extent:32,alignment:4,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:232,binding:3,binding_offset:0,extent:8,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:120,binding:5,binding_offset:0,extent:16384,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:168,binding:6,binding_offset:0,extent:64,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:248,binding:7,binding_offset:0,extent:4,alignment:4,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:352,binding:8,binding_offset:0,extent:16,alignment:4,io:CubinIo::Read}]}}
}
pub(super) fn launches(case: AttentionCase, stage: AttentionStage) -> Vec<AttentionLaunch> {
    match (case, stage) {
        (_, AttentionStage::Sigmoid) => vec![super::gate_manifests::sigmoid(case)],
        (_, AttentionStage::Multiply) => vec![super::gate_manifests::multiply(case)],
        (AttentionCase::InitialOne, AttentionStage::Qkv) => vec![launch_0()],
        (AttentionCase::InitialOne, AttentionStage::Qk) => vec![launch_8()],
        (AttentionCase::InitialOne, AttentionStage::Kv) => vec![launch_9()],
        (AttentionCase::InitialOne, AttentionStage::Fa) => vec![launch_10()],
        (AttentionCase::InitialOne, AttentionStage::Out) => vec![launch_1(), launch_2()],
        (AttentionCase::CachedOneAt1, AttentionStage::Qkv) => vec![launch_0()],
        (AttentionCase::CachedOneAt1, AttentionStage::Qk) => vec![launch_8()],
        (AttentionCase::CachedOneAt1, AttentionStage::Kv) => vec![launch_9()],
        (AttentionCase::CachedOneAt1, AttentionStage::Fa) => vec![launch_11()],
        (AttentionCase::CachedOneAt1, AttentionStage::Out) => vec![launch_1(), launch_2()],
        (AttentionCase::Initial64, AttentionStage::Qkv) => vec![launch_3(), launch_4()],
        (AttentionCase::Initial64, AttentionStage::Qk) => vec![launch_12()],
        (AttentionCase::Initial64, AttentionStage::Kv) => vec![launch_13()],
        (AttentionCase::Initial64, AttentionStage::Fa) => vec![launch_14()],
        (AttentionCase::Initial64, AttentionStage::Out) => vec![launch_5()],
        (AttentionCase::ContinuedTwoAt64, AttentionStage::Qkv) => vec![launch_6()],
        (AttentionCase::ContinuedTwoAt64, AttentionStage::Qk) => vec![launch_15()],
        (AttentionCase::ContinuedTwoAt64, AttentionStage::Kv) => vec![launch_16()],
        (AttentionCase::ContinuedTwoAt64, AttentionStage::Fa) => vec![launch_17()],
        (AttentionCase::ContinuedTwoAt64, AttentionStage::Out) => vec![launch_7()],
        (AttentionCase::CachedOneAt66, AttentionStage::Qkv) => vec![launch_0()],
        (AttentionCase::CachedOneAt66, AttentionStage::Qk) => vec![launch_8()],
        (AttentionCase::CachedOneAt66, AttentionStage::Kv) => vec![launch_9()],
        (AttentionCase::CachedOneAt66, AttentionStage::Fa) => vec![launch_18(), launch_19()],
        (AttentionCase::CachedOneAt66, AttentionStage::Out) => vec![launch_1(), launch_2()],
    }
}
pub(super) fn required_images() -> std::collections::BTreeSet<ProjectionImage> {
    std::collections::BTreeSet::from([
        ProjectionImage::Bf16SplitKReduceSm86,
        ProjectionImage::AttentionKvSm86,
        ProjectionImage::AttentionAmpere256x128Sm80,
        ProjectionImage::AttentionQkSm86,
        ProjectionImage::AttentionFaSplitSm80,
        ProjectionImage::Bf16Gemv2TSm86,
        ProjectionImage::CutlassWmmaSm80,
        ProjectionImage::AttentionFaCausalCombineSm80,
        ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
    ])
}
pub(super) fn require_closed_cases(cases: &[AttentionCase]) -> Result<()> {
    if cases != AttentionCase::ALL {
        return Err(Error::Unsupported(
            "exact five captured attention cases".into(),
        ));
    }
    Ok(())
}
