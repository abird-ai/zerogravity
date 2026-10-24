//! Generated actual canonical47B4 source instructions; unqualified until physical replay.
use super::gdn_code::{Role, StageSource};
use cubecl_runtime::kernel::BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
// Whole source-unit manifest SHA256: 7507eef1a77a5c9d61b8d59c75f9e29980cc0a618d5893da7fa153aed9e25192
pub(super) static PREFILL: &[StageSource] = &[
    StageSource {
        name: "_causal_conv1d_fwd_kernel",
        source: include_str!(
            "gdn_assets/61fbd137aa75d6b8bc1c4c124c0049479019e47c008c087769ce30b79deae9fe.cu"
        ),
        entry: "zg_vllm_a905e193ffad4e19",
        threads: 128,
        grid: [24, 24, 1],
        bindings: &[
            (Role::Packed, R),
            (Role::ConvWeight, R),
            (Role::CacheConv, RW),
            (Role::StateIndices, R),
            (Role::HasInitial, R),
            (Role::CuSeq, R),
            (Role::ConvBatch, R),
            (Role::ConvOffset, R),
            (Role::Conv, W),
        ],
    },
    StageSource {
        name: "_fused_post_conv_kernel",
        source: include_str!(
            "gdn_assets/6e43dcc3a239f8a085a28d9b685fbf623272c13b1ed6465883924b4e356518fb.cu"
        ),
        entry: "zg_vllm_78ff2ff19f5099b9",
        threads: 128,
        grid: [12, 32, 1],
        bindings: &[
            (Role::Conv, R),
            (Role::A, R),
            (Role::B, R),
            (Role::ALog, R),
            (Role::DtBias, R),
            (Role::Q, W),
            (Role::K, W),
            (Role::V, W),
            (Role::G, W),
            (Role::Beta, W),
        ],
    },
    StageSource {
        name: "chunk_local_cumsum_scalar_kernel",
        source: include_str!(
            "gdn_assets/05415f4a695ec9676911bdc0d9d78f161d632d2f4a11ff9228cac3d9c9341ed1.cu"
        ),
        entry: "zg_vllm_a2ad2ccfa6e953f5",
        threads: 128,
        grid: [4, 16, 1],
        bindings: &[
            (Role::G, R),
            (Role::GCum, W),
            (Role::CuSeq, R),
            (Role::ChunkIndices, R),
        ],
    },
    StageSource {
        name: "chunk_scaled_dot_kkt_fwd_kernel",
        source: include_str!(
            "gdn_assets/fe41abf35df7268a8d64fc286fd485d2690785a596a172b4021ed6702c8a6a38.cu"
        ),
        entry: "zg_vllm_68bc016c3bcb6095",
        threads: 256,
        grid: [4, 16, 1],
        bindings: &[
            (Role::K, R),
            (Role::Beta, R),
            (Role::GCum, R),
            (Role::MatrixA, W),
            (Role::CuSeq, R),
            (Role::ChunkIndices, R),
        ],
    },
    StageSource {
        name: "merge_16x16_to_64x64_inverse_kernel",
        source: include_str!(
            "gdn_assets/3876259b4b5833a772a8e800901a71a83087146084ee5a7c9bf0f60178a3a496.cu"
        ),
        entry: "zg_vllm_64d5095f294cb1c1",
        threads: 128,
        grid: [4, 16, 1],
        bindings: &[
            (Role::MatrixA, R),
            (Role::MatrixInv, W),
            (Role::CuSeq, R),
            (Role::ChunkIndices, R),
        ],
    },
    StageSource {
        name: "recompute_w_u_fwd_kernel",
        source: include_str!(
            "gdn_assets/27c3c52d69d5cb5053c508b5820be8a8a31b7681759921e3a4d2d60aa837b841.cu"
        ),
        entry: "zg_vllm_eded6623852de87d",
        threads: 256,
        grid: [4, 16, 1],
        bindings: &[
            (Role::K, R),
            (Role::V, R),
            (Role::Beta, R),
            (Role::W, W),
            (Role::U, W),
            (Role::MatrixInv, R),
            (Role::GCum, R),
            (Role::CuSeq, R),
            (Role::ChunkIndices, R),
        ],
    },
    StageSource {
        name: "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        source: include_str!(
            "gdn_assets/3cb1cb2217e4205adb4ab26e0dacdf3d7255034fc74dcc4a282c51600b139e00.cu"
        ),
        entry: "zg_vllm_444aa643ee60a35a",
        threads: 128,
        grid: [2, 64, 1],
        bindings: &[
            (Role::K, R),
            (Role::U, R),
            (Role::W, R),
            (Role::VNew, W),
            (Role::GCum, R),
            (Role::History, W),
            (Role::H0, R),
            (Role::Ht, W),
            (Role::CuSeq, R),
            (Role::ChunkOffsets, R),
        ],
    },
    StageSource {
        name: "chunk_fwd_kernel_o",
        source: include_str!(
            "gdn_assets/89f346a5e71e2eb50308be4c13622d81c63264eb4f578407da13ea572490cdfa.cu"
        ),
        entry: "zg_vllm_a3d74997c0201bf1",
        threads: 128,
        grid: [4, 4, 16],
        bindings: &[
            (Role::Q, R),
            (Role::K, R),
            (Role::VNew, R),
            (Role::History, R),
            (Role::GCum, R),
            (Role::RawO, W),
            (Role::CuSeq, R),
            (Role::ChunkIndices, R),
        ],
    },
    StageSource {
        name: "layer_norm_fwd_kernel",
        source: include_str!(
            "gdn_assets/ecfb6212f2f84fe230b58d151e9b8832d673aea915e5e4b7614e56c5ac492697.cu"
        ),
        entry: "zg_vllm_820c1a61c99ce579",
        threads: 32,
        grid: [752, 1, 1],
        bindings: &[
            (Role::NormInput, R),
            (Role::Gated, W),
            (Role::Gamma, R),
            (Role::Z, R),
            (Role::Rstd, W),
        ],
    },
];
pub(super) static DECODE: &[StageSource] = &[
    StageSource {
        name: "_causal_conv1d_update_kernel",
        source: include_str!(
            "gdn_assets/c0d673a9524c4ed1215bfcadfc0e04a35e97aa1703f2f88377d4b4a886ee1cc1.cu"
        ),
        entry: "zg_vllm_40b19b76a867d4a4",
        threads: 128,
        grid: [4, 24, 1],
        bindings: &[
            (Role::Pitched, R),
            (Role::ConvWeight, R),
            (Role::CacheConv, RW),
            (Role::StateIndices, R),
            (Role::Pitched, W),
        ],
    },
    StageSource {
        name: "fused_recurrent_gated_delta_rule_packed_decode_kernel",
        source: include_str!(
            "gdn_assets/ac3ca66be4a8d5212a9a7d8bcac56c11cbf00c1e09c343f612049d69bd925f40.cu"
        ),
        entry: "zg_vllm_f846a25f8d6edb98",
        threads: 32,
        grid: [4, 64, 1],
        bindings: &[
            (Role::Pitched, R),
            (Role::A, R),
            (Role::B, R),
            (Role::ALog, R),
            (Role::DtBias, R),
            (Role::RawO, W),
            (Role::CacheH, RW),
            (Role::CacheH, RW),
            (Role::StateIndices, R),
        ],
    },
    StageSource {
        name: "layer_norm_fwd_kernel",
        source: include_str!(
            "gdn_assets/b10ffc50f111f20337cf61c4d0131b13c4d755b57f3f79e13c7540cd776a2a79.cu"
        ),
        entry: "zg_vllm_7d94731c2c340a3d",
        threads: 32,
        grid: [32, 1, 1],
        bindings: &[
            (Role::NormInput, R),
            (Role::Gated, W),
            (Role::Gamma, R),
            (Role::Z, R),
            (Role::Rstd, W),
        ],
    },
];
