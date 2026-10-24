//! Actual original selections; execution/numerics remain unqualified.
use super::*;
pub(super) fn select(stage: GdnStage) -> Result<assets::Asset> {
    use GdnStage::*;
    Ok(match stage {
        Conv => assets::Asset {
            entry: "zg_vllm_c8f85203ea21f242",
            source: include_str!("conv.cu"),
            threads: 128,
            grid: [1, 24, 1],
            shared: 0,
            ptx_sha256: "a1e81ca0dfff5c4f241061a3baddcc25e7407ad42a2f99f039703199fa5469aa",
        },
        Decode => assets::Asset {
            entry: "zg_vllm_ebe262b1b494d65e",
            source: include_str!("decode.cu"),
            threads: 32,
            grid: [4, 16, 1],
            shared: 64,
            ptx_sha256: "e15170544e78ee3ab6b19670b4cc80e9a74db549314bb57f8a897ac855283fd5",
        },
        Gated => assets::Asset {
            entry: "zg_vllm_c15586587b4a279e",
            source: include_str!("gated.cu"),
            threads: 32,
            grid: [16, 1, 1],
            shared: 0,
            ptx_sha256: "a97d127b124f14d226c2b24cff907e8c7aa260545e41743f84581acf6d510bbb",
        },
        _ => return Err(Error::Unsupported("original prefill stage".into())),
    })
}
