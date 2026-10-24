//! Three exact original normalized packets. Memory proof is separately closed.
use super::*;
const MEMORY_SEMANTICS_CLOSED: bool = true;
pub(super) fn require_memory_semantics() -> Result<()> {
    if !MEMORY_SEMANTICS_CLOSED {
        return Err(Error::Unsupported(
            "GdnOut finite memory semantics remain source-only".into(),
        ));
    }
    Ok(())
}
// Full image SHA 94fc9c63d0562b88395c3d5001397aa1a3a3f572450a00db627750b817c7de56; packet SHA 26cb170a690317eadecffd48ab2669de0244447ce4e0e914065cf70707f0c6ae.
fn launch_0() -> (ProjectionImage, CubinKernelManifest) {
    (ProjectionImage::Bf16Gemv2TSm86,CubinKernelManifest {
 entry:"_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi2ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_".into(),binary_sm:ProjectionImage::Bf16Gemv2TSm86.binary_sm(),
 grid:[256, 1, 1],block:[128, 1, 1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,
 bindings:vec![CubinBinding{bytes:8388608,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write}],parameters:vec![CubinParameter{offset:0,template:vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,8,0,0,0,8,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},CubinParameter{offset:152,template:vec![0,0,128,63]},CubinParameter{offset:156,template:vec![0,0,0,0]}],fixups:vec![CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:8388608,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},CubinPointerFixup{parameter:0,byte_offset:32,binding:2,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},CubinPointerFixup{parameter:0,byte_offset:48,binding:2,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write}]
 })
}
// Full image SHA c1ca8db71001178d04f1b83d7c55883107fa122927e10d41020672678fff2c86; packet SHA b8586b7c1aa824d6964a84e4cd8c178fcc41ba39f1a00465f0c96b495324efef.
fn launch_1() -> (ProjectionImage, CubinKernelManifest) {
    (
        ProjectionImage::AmpereBf16Gemm128x64Sm80,
        CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn".into(),
            binary_sm: ProjectionImage::AmpereBf16Gemm128x64Sm80.binary_sm(),
            grid: [16, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 24576,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 8388608,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 262144,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 262144,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 64, 0, 250, 255, 255, 255, 255, 255,
                    64, 0, 254, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0,
                    0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 64, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 8388608,
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
                    extent: 262144,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 262144,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        },
    )
}
// Full image SHA c3361a22db27ce017fec90a46f66c508cad5d668d606070b214b0351b3cf9588; packet SHA e6c0d749f4fce32e1cd0525c79af81c4f7aec73ae82136f90ef52d08c0acccf1.
fn launch_2() -> (ProjectionImage, CubinKernelManifest) {
    (
        ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
        CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn".into(),
            binary_sm: ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80.binary_sm(),
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
                    bytes: 192512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 96256,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 128, 0, 253, 255, 255, 255, 255, 255,
                    128, 0, 253, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0,
                    0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 47, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
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
                    extent: 192512,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 16,
                    binding: 2,
                    binding_offset: 0,
                    extent: 96256,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 96256,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        },
    )
}
pub(super) fn launch(
    kind: ComponentKind,
    rows: usize,
) -> Result<(ProjectionImage, CubinKernelManifest)> {
    match (kind, rows) {
        (ComponentKind::Qwen2b, 1) => Ok(launch_0()),
        (ComponentKind::Qwen2b, 64) => Ok(launch_1()),
        (ComponentKind::Canonical47(a), 47) => {
            a.require_owned()?;
            Ok(launch_2())
        }
        _ => Err(Error::Unsupported(
            "finite source GdnOut initial1/64/canonical47".into(),
        )),
    }
}
/// Same existing selected47 packet, independent of component metadata.
pub(super) fn model_canonical47_packet() -> (ProjectionImage, CubinKernelManifest) {
    launch_2()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_packets_keep_distinct_extents_and_launches() {
        let cases = [
            (launch_0(), [8388608, 4096, 4096], 160),
            (launch_1(), [8388608, 262144, 262144], 192),
            (launch_2(), [4194304, 192512, 96256], 192),
        ];
        for ((_, m), sizes, n) in cases {
            assert_eq!(
                m.parameters.iter().map(|p| p.template.len()).sum::<usize>(),
                n
            );
            assert_eq!(m.fixups.len(), 4);
            assert_eq!(
                m.bindings.iter().map(|b| b.bytes).collect::<Vec<_>>(),
                sizes
            );
            assert_eq!(m.fixups[2].binding, 2);
            assert_eq!(m.fixups[3].binding, 2);
            for f in &m.fixups {
                assert_eq!(
                    &m.parameters[0].template[f.byte_offset..f.byte_offset + 8],
                    &[0; 8]
                );
            }
        }
    }
    #[test]
    fn gemv_retains_actual_three_parameter_reflection() {
        let (_, m) = launch_0();
        assert_eq!(
            m.parameters
                .iter()
                .map(|p| (p.offset, p.template.len()))
                .collect::<Vec<_>>(),
            [(0, 152), (152, 4), (156, 4)]
        );
        assert_eq!(m.parameters[1].template, [0, 0, 128, 63]);
        assert_eq!(m.parameters[2].template, [0; 4]);
        assert!(m.fixups.iter().all(|f| f.parameter == 0));
        for (_, m) in [launch_1(), launch_2()] {
            assert_eq!(m.parameters.len(), 1);
            assert_eq!(m.parameters[0].template.len(), 192);
        }
    }
    #[test]
    fn finite_packet_geometry_cannot_retag_a_different_request() {
        use crate::vllm_cuda_gdn::Canonical47GdnArithmetic as A;
        for n in [0, 2, 47, 65] {
            assert!(launch(ComponentKind::Qwen2b, n).is_err());
        }
        assert!(launch(ComponentKind::Canonical47(A::OriginalPlainSolve2), 47).is_err());
        assert!(launch(ComponentKind::Canonical47(A::OwnedCanonical47Solve4), 64).is_err());
    }
}
