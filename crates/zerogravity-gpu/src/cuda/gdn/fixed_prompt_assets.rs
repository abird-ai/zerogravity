//! FixedPrompt source choices only; original typed call/root admission remains closed.
use super::*;

pub(super) fn select(stage: GdnStage) -> Result<Asset> {
    use GdnStage::*;
    Ok(match stage {
        Conv => Asset {
            entry: "zg_vllm_a905e193ffad4e19",
            source: include_str!("fixed_prompt_conv.cu"),
            threads: 128,
            grid: [6, 24, 1],
            shared: 0,
            ptx_sha256: "7e9600b09ee757c1d786657d7edfa5b3629cb5b1b00480de1fdb5a761e4a0a5c",
        },
        Prep => Asset {
            entry: "zg_vllm_78ff2ff19f5099b9",
            source: include_str!("fixed_prompt_prep.cu"),
            threads: 128,
            grid: [3, 32, 1],
            shared: 0,
            ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
        },
        Cumsum => Asset {
            entry: "zg_vllm_82a622be2630db81",
            source: include_str!("fixed_prompt_cumsum.cu"),
            threads: 64,
            grid: [1, 16, 1],
            shared: 8,
            ptx_sha256: "b4c97f73509b35ae7f3281b5621b44ca354e08b65a496575e95670537a0783ab",
        },
        Kkt => Asset {
            entry: "zg_vllm_6d24f41cda7920aa",
            source: include_str!("fixed_prompt_kkt.cu"),
            threads: 256,
            grid: [1, 16, 1],
            shared: 16384,
            ptx_sha256: "d65e28bad710a3e78a38583d14ec26a8c838eb75e8becc0da40a9a6c17da7639",
        },
        Solve => Asset {
            entry: "zg_vllm_116d2ffe705b6643",
            source: include_str!("solve.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 10240,
            ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
        },
        Recompute => Asset {
            entry: "zg_vllm_a73872840c0a9697",
            source: include_str!("recompute.cu"),
            threads: 256,
            grid: [1, 16, 1],
            shared: 40960,
            ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
        },
        H => Asset {
            entry: "zg_vllm_9f97b7023b619918",
            source: include_str!("fixed_prompt_h.cu"),
            threads: 128,
            grid: [4, 16, 1],
            shared: 41220,
            ptx_sha256: "2c05861ae7051b9f267b683d7a411686edb18d8ab34598c6f9dac9c8e39dcb38",
        },
        Output => Asset {
            entry: "zg_vllm_81c9eabf9b9a78e0",
            source: include_str!("output.cu"),
            threads: 128,
            grid: [4, 1, 16],
            shared: 12288,
            ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
        },
        Gated => Asset {
            entry: "zg_vllm_820c1a61c99ce579",
            source: include_str!("fixed_prompt_gated.cu"),
            threads: 32,
            grid: [188, 1, 1],
            shared: 0,
            ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
        },
        Decode => return Err(Error::Unsupported("fixed_prompt initial GDN stage".into())),
    })
}
