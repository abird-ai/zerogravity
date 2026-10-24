//! Exact original ATen BF16 stores remain two distinct native kernels.
//! f-byte4 is compiled unread; only it is canonicalized to0 (see SOURCE.md).
use super::super::cubin_manifests::ProjectionImage;
use super::manifests::{AttentionCase, AttentionLaunch, AttentionRole};
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup,
};
const SIGMOID: &str = "_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_19sigmoid_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE0_clEvENKUlvE2_clEvEUlN3c108BFloat16EE_St5arrayIPcLm2EEEEviT0_T1_";
const MULTIPLY: &str = "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIN3c108BFloat16ES4_S4_NS0_15binary_internal10MulFunctorIfEEEESt5arrayIPcLm3EEEEviT0_T1_";

pub(super) fn sigmoid(case: AttentionCase) -> AttentionLaunch {
    make(case, false)
}
pub(super) fn multiply(case: AttentionCase) -> AttentionLaunch {
    make(case, true)
}
fn make(case: AttentionCase, multiply: bool) -> AttentionLaunch {
    let rows = case.rows();
    let bytes = (rows * 4096) as u64;
    let (image, entry, roles): (_, _, &'static [AttentionRole]) = if multiply {
        (
            ProjectionImage::AttentionMultiplySm86,
            MULTIPLY,
            &[
                AttentionRole::OutInput,
                AttentionRole::AttentionOutput,
                AttentionRole::SigmoidGate,
            ],
        )
    } else {
        (
            ProjectionImage::AttentionSigmoidSm86,
            SIGMOID,
            &[AttentionRole::SigmoidGate, AttentionRole::Gate],
        )
    };
    let bindings = roles
        .iter()
        .enumerate()
        .map(|(i, _)| CubinBinding {
            bytes,
            alignment: 8,
            io: if i == 0 {
                CubinIo::Write
            } else {
                CubinIo::Read
            },
        })
        .collect();
    let fixups = roles
        .iter()
        .enumerate()
        .map(|(i, _)| CubinPointerFixup {
            parameter: 2,
            byte_offset: i * 8,
            binding: i,
            binding_offset: 0,
            extent: bytes,
            alignment: 8,
            io: if i == 0 {
                CubinIo::Write
            } else {
                CubinIo::Read
            },
        })
        .collect();
    AttentionLaunch {
        image,
        roles,
        manifest: CubinKernelManifest {
            entry: entry.into(),
            binary_sm: 86,
            grid: [(rows * 2) as u32, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            attributes: CubinLaunchAttributes::None,
            bindings,
            parameters: vec![
                CubinParameter {
                    offset: 0,
                    template: ((rows * 2048) as i32).to_le_bytes().to_vec(),
                },
                CubinParameter {
                    offset: 4,
                    template: vec![0],
                },
                CubinParameter {
                    offset: 8,
                    template: vec![0; roles.len() * 8],
                },
            ],
            fixups,
        },
    }
}
