//! Literal original2B packets; numerical/native qualification remains pending.
use super::*;
fn launch_0() -> Launch {
    Launch { image: ProjectionImage::Bf16GemvxSm86, roles: vec![Binding::W,Binding::X,Binding::Y,], manifest: CubinKernelManifest {
entry: "_ZN8internal5gemvx6kernelIii13__nv_bfloat16S2_S2_fLb0ELb1ELb1ELb0ELi7ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS2_ES6_S4_IS2_EfEEENSt9enable_ifIXntT5_EvE4typeET11_".into(), binary_sm: ProjectionImage::Bf16GemvxSm86.binary_sm(),
grid:[8, 1, 1],block:[32, 4, 1],dynamic_shared_bytes:528,attributes:CubinLaunchAttributes::None,
bindings:vec![CubinBinding{bytes:131072,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:64,alignment:16,io:CubinIo::Write},],parameters:vec![CubinParameter{offset:0,template:vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 32, 0, 0, 0, 0, 8, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]},],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:131072,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:32,binding:2,binding_offset:0,extent:64,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:48,binding:2,binding_offset:0,extent:64,alignment:16,io:CubinIo::Write},]}}
}
fn launch_1() -> Launch {
    Launch { image: ProjectionImage::CutlassWmmaSm80, roles: vec![Binding::W,Binding::X,Binding::Y,], manifest: CubinKernelManifest {
entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
grid:[8, 64, 1],block:[32, 1, 1],dynamic_shared_bytes:8704,attributes:CubinLaunchAttributes::None,
bindings:vec![CubinBinding{bytes:33554432,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},],parameters:vec![CubinParameter{offset:0,template:vec![1, 0, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 33, 255, 255, 255, 255, 255, 255, 0, 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 33, 255, 255, 255, 255, 255, 255, 0, 1, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 64, 254, 255, 255, 255, 255, 255, 0, 64, 254, 255, 255, 255, 255, 255, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 64, 254, 255, 255, 255, 255, 255, 0, 64, 254, 255, 255, 255, 255, 255, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:33554432,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:16384,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:304,binding:2,binding_offset:0,extent:16384,alignment:16,io:CubinIo::Write},]}}
}
fn launch_2() -> Launch {
    Launch { image: ProjectionImage::CutlassWmmaSm80, roles: vec![Binding::W,Binding::X,Binding::Y,], manifest: CubinKernelManifest {
entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
grid:[2, 1, 1],block:[32, 1, 1],dynamic_shared_bytes:16896,attributes:CubinLaunchAttributes::None,
bindings:vec![CubinBinding{bytes:131072,alignment:16,io:CubinIo::Read},CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},CubinBinding{bytes:128,alignment:16,io:CubinIo::Write},],parameters:vec![CubinParameter{offset:0,template:vec![2, 0, 0, 0, 32, 0, 0, 0, 0, 8, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 33, 255, 255, 255, 255, 255, 255, 0, 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 33, 255, 255, 255, 255, 255, 255, 0, 1, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 64, 254, 255, 255, 255, 255, 255, 255, 64, 254, 255, 255, 255, 255, 255, 255, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 64, 254, 255, 255, 255, 255, 255, 255, 64, 254, 255, 255, 255, 255, 255, 255, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:8192,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:131072,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:128,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:304,binding:2,binding_offset:0,extent:128,alignment:16,io:CubinIo::Write},]}}
}
fn launch_3() -> Launch {
    Launch { image: ProjectionImage::CutlassWmmaSm80, roles: vec![Binding::W,Binding::X,Binding::Y,], manifest: CubinKernelManifest {
entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
grid:[8, 64, 1],block:[32, 1, 1],dynamic_shared_bytes:8704,attributes:CubinLaunchAttributes::None,
bindings:vec![CubinBinding{bytes:33554432,alignment:16,io:CubinIo::Read},CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},CubinBinding{bytes:32768,alignment:16,io:CubinIo::Write},],parameters:vec![CubinParameter{offset:0,template:vec![2, 0, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 33, 255, 255, 255, 255, 255, 255, 0, 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 33, 255, 255, 255, 255, 255, 255, 0, 1, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 64, 254, 255, 255, 255, 255, 255, 0, 64, 254, 255, 255, 255, 255, 255, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 64, 254, 255, 255, 255, 255, 255, 0, 64, 254, 255, 255, 255, 255, 255, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:8192,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:33554432,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:32768,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:304,binding:2,binding_offset:0,extent:32768,alignment:16,io:CubinIo::Write},]}}
}
fn launch_4() -> Launch {
    Launch { image: ProjectionImage::CutlassWmmaSm80, roles: vec![Binding::W,Binding::X,Binding::Y,Binding::Workspace,], manifest: CubinKernelManifest {
entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
grid:[2, 1, 16],block:[128, 1, 1],dynamic_shared_bytes:33792,attributes:CubinLaunchAttributes::None,
bindings:vec![CubinBinding{bytes:131072,alignment:16,io:CubinIo::Read},CubinBinding{bytes:262144,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8519680,alignment:16,io:CubinIo::Write},],parameters:vec![CubinParameter{offset:0,template:vec![64, 0, 0, 0, 32, 0, 0, 0, 0, 8, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 129, 254, 255, 255, 255, 255, 255, 0, 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 129, 254, 255, 255, 255, 255, 255, 0, 1, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 64, 255, 255, 255, 255, 255, 255, 255, 64, 255, 255, 255, 255, 255, 255, 255, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 64, 255, 255, 255, 255, 255, 255, 255, 64, 255, 255, 255, 255, 255, 255, 255, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:262144,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:131072,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:304,binding:3,binding_offset:0,extent:65536,alignment:16,io:CubinIo::Write},]}}
}
fn launch_5() -> Launch {
    Launch { image: ProjectionImage::Bf16SplitKReduceSm86, roles: vec![Binding::Y,Binding::Workspace,], manifest: CubinKernelManifest {
entry: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(), binary_sm: ProjectionImage::Bf16SplitKReduceSm86.binary_sm(),
grid:[1, 4, 1],block:[32, 16, 1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,
bindings:vec![CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write},CubinBinding{bytes:8519680,alignment:16,io:CubinIo::Read},],parameters:vec![CubinParameter{offset:0,template:vec![0, 0, 128, 63, 0, 0, 0, 0, 32, 0, 0, 0, 64, 0, 0, 0, 32, 0, 0, 0, 32, 0, 0, 0, 32, 0, 0, 0, 32, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]},CubinParameter{offset:64,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:72,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:80,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:88,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:96,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:104,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:112,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:120,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:128,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:136,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:144,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:152,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:160,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:168,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:176,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:184,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:192,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:200,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:208,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},CubinParameter{offset:216,template:vec![0, 0, 0, 0, 0, 0, 0, 0]},],fixups:vec![CubinPointerFixup{parameter:1,byte_offset:0,binding:1,binding_offset:0,extent:65536,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:2,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:3,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:4,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},]}}
}
fn launch_6() -> Launch {
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
                    bytes: 33554432,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 262144,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 1048576,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 64, 0, 249, 255, 255, 255, 255, 255,
                    64, 0, 253, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0,
                    0, 32, 0, 0, 0, 32, 0, 0, 0, 32, 0, 0, 64, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0,
                ],
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 33554432,
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
                    extent: 1048576,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 1048576,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        },
    }
}
pub(super) fn launches(rows: usize, role: Role) -> Result<Vec<Launch>> {
    Ok(match (rows, role) {
        (1, Role::Qkvz) => vec![launch_1()],
        (1, Role::Ba) => vec![launch_0()],
        (2, Role::Qkvz) => vec![launch_3()],
        (2, Role::Ba) => vec![launch_2()],
        (64, Role::Qkvz) => vec![launch_6()],
        (64, Role::Ba) => vec![launch_4(), launch_5()],
        _ => {
            return Err(Error::Unsupported(
                "closed2B projection row coverage".into(),
            ));
        }
    })
}

/// Finite original MLP union packets; no legacy packet canonicalization claim.
pub(super) fn mlp_launches(rows: usize, role: Role) -> Result<Vec<Launch>> {
    if !matches!(rows, 1 | 64) {
        return Err(Error::Unsupported("2B MLP packet initial1/64 only".into()));
    }
    let mut packets = launches(rows, role)?;
    let (entry, grid, block, shared) = match (rows, role) {
        (1, Role::Qkvz) => (
            "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8EEvNT_6ParamsE",
            [8, 64, 1],
            [32, 1, 1],
            8704,
        ),
        (64, Role::Ba) => (
            "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE",
            [2, 1, 16],
            [128, 1, 1],
            33792,
        ),
        _ => return Ok(packets),
    };
    let main = &mut packets[0];
    let manifest = &mut main.manifest;
    if main.image != ProjectionImage::CutlassWmmaSm80
        || manifest.entry != entry
        || manifest.grid != grid
        || manifest.block != block
        || manifest.dynamic_shared_bytes != shared
    {
        return Err(Error::State("2B MLP original WMMA packet identity".into()));
    }
    let [parameter] = manifest.parameters.as_mut_slice() else {
        return Err(Error::State("2B MLP original WMMA parameter count".into()));
    };
    if parameter.offset != 0
        || parameter.template.len() != 360
        || parameter.template[260..264] != [0; 4]
    {
        return Err(Error::State("2B MLP original WMMA word baseline".into()));
    }
    // Original events18/312 contain these four packet bytes; no pointer fixup.
    parameter.template[260..264].copy_from_slice(&[0x42, 0x79, 0, 0]);
    Ok(packets)
}

#[cfg(test)]
mod mlp_packet_tests {
    use super::*;

    #[test]
    fn captured_mlp_wmma_words_preserve_all_other_packet_fields() {
        for rows in [1, 64] {
            for role in [Role::Qkvz, Role::Ba] {
                let original = launches(rows, role).unwrap();
                let mut mlp = mlp_launches(rows, role).unwrap();
                assert_eq!(original.len(), mlp.len());
                let captured_word = matches!((rows, role), (1, Role::Qkvz) | (64, Role::Ba));
                for (index, (old, current)) in original.iter().zip(&mut mlp).enumerate() {
                    assert_eq!(old.image, current.image);
                    if index == 0 && captured_word {
                        assert_eq!(old.manifest.parameters[0].template[260..264], [0; 4]);
                        assert_eq!(
                            current.manifest.parameters[0].template[260..264],
                            [0x42, 0x79, 0, 0]
                        );
                        current.manifest.parameters[0].template[260..264].copy_from_slice(&[0; 4]);
                    }
                    assert_eq!(old.manifest, current.manifest);
                }
            }
        }
    }

    #[test]
    fn mlp_scope_refuses_legacy_two_rows_without_changing_original_scope() {
        for role in [Role::Qkvz, Role::Ba] {
            assert!(mlp_launches(2, role).is_err());
            assert_eq!(
                launches(2, role).unwrap()[0].manifest.parameters[0].template[260..264],
                [0; 4]
            );
        }
    }
}
