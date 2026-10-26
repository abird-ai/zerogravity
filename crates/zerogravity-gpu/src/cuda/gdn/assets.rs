//! Generated static choices from the validated original source; no runtime trace.
use super::*;

#[path = "fixed_prompt_assets.rs"]
mod fixed_prompt;
#[cfg(any(test, target_os = "linux"))]
#[path = "qwen2b_assets.rs"]
mod qwen2b;
#[cfg(any(test, target_os = "linux"))]
pub(super) fn select_qwen2b(phase: GdnPhase, stage: GdnStage) -> Result<Asset> {
    qwen2b::select(phase, stage)
}
pub(super) struct Asset {
    pub entry: &'static str,
    pub source: &'static str,
    pub threads: u32,
    pub grid: [u32; 3],
    pub shared: u32,
    pub ptx_sha256: &'static str,
}
pub(super) fn select(phase: GdnPhase, stage: GdnStage) -> Result<Asset> {
    if matches!(phase, GdnPhase::TextPrefill(_)) {
        return Err(Error::Unsupported(
            "text code requires genuine source-owned selector".into(),
        ));
    }
    if phase == GdnPhase::FixedPrompt {
        return fixed_prompt::select(stage);
    }
    use GdnPhase::*;
    use GdnStage::*;
    let value = match (phase, stage) {
        (InitialOne, Conv) => Asset {
            entry: "zg_vllm_a905e193ffad4e19",
            source: include_str!("conv_prefill_one.cu"),
            threads: 128,
            grid: [1, 24, 1],
            shared: 0,
            ptx_sha256: "7e9600b09ee757c1d786657d7edfa5b3629cb5b1b00480de1fdb5a761e4a0a5c",
        },
        (InitialOne, Prep) => Asset {
            entry: "zg_vllm_6914e331960248b3",
            source: include_str!("prep_one.cu"),
            threads: 128,
            grid: [1, 32, 1],
            shared: 0,
            ptx_sha256: "ca887254afcfa3b02b6a604ac3670ef4cca3cc72e05826c1fc9e1256cd85909e",
        },
        (InitialOne, Cumsum) => Asset {
            entry: "zg_vllm_a2ad2ccfa6e953f5",
            source: include_str!("cumsum.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 8,
            ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
        },
        (InitialOne, Kkt) => Asset {
            entry: "zg_vllm_79e779d50e66d224",
            source: include_str!("kkt.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 8192,
            ptx_sha256: "e8ba065941c5dbf148d98ee28eba369a987d176c264ce8e05e9cb697498bad44",
        },
        (InitialOne, Solve) => Asset {
            entry: "zg_vllm_116d2ffe705b6643",
            source: include_str!("solve.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 10240,
            ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
        },
        (InitialOne, Recompute) => Asset {
            entry: "zg_vllm_a73872840c0a9697",
            source: include_str!("recompute.cu"),
            threads: 256,
            grid: [1, 16, 1],
            shared: 40960,
            ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
        },
        (InitialOne, H) => Asset {
            entry: "zg_vllm_444aa643ee60a35a",
            source: include_str!("state_h.cu"),
            threads: 128,
            grid: [2, 16, 1],
            shared: 49412,
            ptx_sha256: "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
        },
        (InitialOne, Output) => Asset {
            entry: "zg_vllm_81c9eabf9b9a78e0",
            source: include_str!("output.cu"),
            threads: 128,
            grid: [4, 1, 16],
            shared: 12288,
            ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
        },
        (InitialOne, Gated) => Asset {
            entry: "zg_vllm_c15586587b4a279e",
            source: include_str!("gated_one.cu"),
            threads: 32,
            grid: [16, 1, 1],
            shared: 0,
            ptx_sha256: "a97d127b124f14d226c2b24cff907e8c7aa260545e41743f84581acf6d510bbb",
        },
        (Initial64, Conv) => Asset {
            entry: "zg_vllm_a905e193ffad4e19",
            source: include_str!("conv_prefill_many.cu"),
            threads: 128,
            grid: [8, 24, 1],
            shared: 0,
            ptx_sha256: "7e9600b09ee757c1d786657d7edfa5b3629cb5b1b00480de1fdb5a761e4a0a5c",
        },
        (Initial64, Prep) => Asset {
            entry: "zg_vllm_2671a92b80f9dc47",
            source: include_str!("prep_64.cu"),
            threads: 128,
            grid: [4, 32, 1],
            shared: 0,
            ptx_sha256: "26138ddbd8249c06511453384e6877aef58ad39d23eb9d8733b3eb8369e2467a",
        },
        (Initial64, Cumsum) => Asset {
            entry: "zg_vllm_a2ad2ccfa6e953f5",
            source: include_str!("cumsum.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 8,
            ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
        },
        (Initial64, Kkt) => Asset {
            entry: "zg_vllm_79e779d50e66d224",
            source: include_str!("kkt.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 8192,
            ptx_sha256: "e8ba065941c5dbf148d98ee28eba369a987d176c264ce8e05e9cb697498bad44",
        },
        (Initial64, Solve) => Asset {
            entry: "zg_vllm_116d2ffe705b6643",
            source: include_str!("solve.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 10240,
            ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
        },
        (Initial64, Recompute) => Asset {
            entry: "zg_vllm_a73872840c0a9697",
            source: include_str!("recompute.cu"),
            threads: 256,
            grid: [1, 16, 1],
            shared: 40960,
            ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
        },
        (Initial64, H) => Asset {
            entry: "zg_vllm_444aa643ee60a35a",
            source: include_str!("state_h.cu"),
            threads: 128,
            grid: [2, 16, 1],
            shared: 49412,
            ptx_sha256: "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
        },
        (Initial64, Output) => Asset {
            entry: "zg_vllm_81c9eabf9b9a78e0",
            source: include_str!("output.cu"),
            threads: 128,
            grid: [4, 1, 16],
            shared: 12288,
            ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
        },
        (Initial64, Gated) => Asset {
            entry: "zg_vllm_820c1a61c99ce579",
            source: include_str!("gated_64.cu"),
            threads: 32,
            grid: [256, 1, 1],
            shared: 0,
            ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
        },
        (ContinuedTwo, Conv) => Asset {
            entry: "zg_vllm_a905e193ffad4e19",
            source: include_str!("conv_prefill_many.cu"),
            threads: 128,
            grid: [1, 24, 1],
            shared: 0,
            ptx_sha256: "7e9600b09ee757c1d786657d7edfa5b3629cb5b1b00480de1fdb5a761e4a0a5c",
        },
        (ContinuedTwo, Prep) => Asset {
            entry: "zg_vllm_78ff2ff19f5099b9",
            source: include_str!("prep_two.cu"),
            threads: 128,
            grid: [1, 32, 1],
            shared: 0,
            ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
        },
        (ContinuedTwo, Cumsum) => Asset {
            entry: "zg_vllm_a2ad2ccfa6e953f5",
            source: include_str!("cumsum.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 8,
            ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
        },
        (ContinuedTwo, Kkt) => Asset {
            entry: "zg_vllm_79e779d50e66d224",
            source: include_str!("kkt.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 8192,
            ptx_sha256: "e8ba065941c5dbf148d98ee28eba369a987d176c264ce8e05e9cb697498bad44",
        },
        (ContinuedTwo, Solve) => Asset {
            entry: "zg_vllm_116d2ffe705b6643",
            source: include_str!("solve.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 10240,
            ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
        },
        (ContinuedTwo, Recompute) => Asset {
            entry: "zg_vllm_a73872840c0a9697",
            source: include_str!("recompute.cu"),
            threads: 256,
            grid: [1, 16, 1],
            shared: 40960,
            ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
        },
        (ContinuedTwo, H) => Asset {
            entry: "zg_vllm_444aa643ee60a35a",
            source: include_str!("state_h.cu"),
            threads: 128,
            grid: [2, 16, 1],
            shared: 49412,
            ptx_sha256: "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
        },
        (ContinuedTwo, Output) => Asset {
            entry: "zg_vllm_81c9eabf9b9a78e0",
            source: include_str!("output.cu"),
            threads: 128,
            grid: [4, 1, 16],
            shared: 12288,
            ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
        },
        (ContinuedTwo, Gated) => Asset {
            entry: "zg_vllm_c15586587b4a279e",
            source: include_str!("gated_two.cu"),
            threads: 32,
            grid: [32, 1, 1],
            shared: 0,
            ptx_sha256: "a97d127b124f14d226c2b24cff907e8c7aa260545e41743f84581acf6d510bbb",
        },
        (CachedDecode, Conv) => Asset {
            entry: "zg_vllm_c8f85203ea21f242",
            source: include_str!("conv_decode.cu"),
            threads: 128,
            grid: [1, 24, 1],
            shared: 0,
            ptx_sha256: "a1e81ca0dfff5c4f241061a3baddcc25e7407ad42a2f99f039703199fa5469aa",
        },
        (CachedDecode, Decode) => Asset {
            entry: "zg_vllm_ebe262b1b494d65e",
            source: include_str!("decode.cu"),
            threads: 32,
            grid: [4, 16, 1],
            shared: 64,
            ptx_sha256: "e15170544e78ee3ab6b19670b4cc80e9a74db549314bb57f8a897ac855283fd5",
        },
        (CachedDecode, Gated) => Asset {
            entry: "zg_vllm_c15586587b4a279e",
            source: include_str!("gated_one.cu"),
            threads: 32,
            grid: [16, 1, 1],
            shared: 0,
            ptx_sha256: "a97d127b124f14d226c2b24cff907e8c7aa260545e41743f84581acf6d510bbb",
        },
        _ => return Err(Error::Unsupported("closed vLLM GDN phase/stage".into())),
    };
    Ok(value)
}
