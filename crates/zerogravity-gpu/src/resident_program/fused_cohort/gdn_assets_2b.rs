//! Generated actual canonical47B4 source instructions; unqualified until physical replay.
use super::gdn_code::{Role, StageSource};
use cubecl_runtime::kernel::BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
// Whole source-unit manifest SHA256: fec1cd9573c6ae5411249b3f4b5b019315549c280eb6f8291bd51b67f7c89daf
pub(super) static DECODE: &[StageSource] = &[
    StageSource {
        name: "_causal_conv1d_update_kernel",
        source: include_str!(
            "gdn_assets/d28e3cd22782c84483fc59a57d8db9a8437e4577b790bbff69d10d4596e2bff7.cu"
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
