//! Actual original selections; execution/numerics remain unqualified.
use super::*;
pub(super) fn select(stage: GdnStage) -> Result<assets::Asset> {
    use GdnStage::*;
    Ok(match stage {
        Conv => assets::Asset {
            entry: "zg_vllm_a905e193ffad4e19",
            source: include_str!("conv.cu"),
            threads: 128,
            grid: [8, 24, 1],
            shared: 0,
            ptx_sha256: "7e9600b09ee757c1d786657d7edfa5b3629cb5b1b00480de1fdb5a761e4a0a5c",
        },
        Prep => assets::Asset {
            entry: "zg_vllm_78ff2ff19f5099b9",
            source: include_str!("prep.cu"),
            threads: 128,
            grid: [4, 32, 1],
            shared: 0,
            ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
        },
        Cumsum => assets::Asset {
            entry: "zg_vllm_a2ad2ccfa6e953f5",
            source: include_str!("cumsum.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 8,
            ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
        },
        Kkt => assets::Asset {
            entry: "zg_vllm_68bc016c3bcb6095",
            source: include_str!("kkt.cu"),
            threads: 256,
            grid: [1, 16, 1],
            shared: 16384,
            ptx_sha256: "973a87c31601182c3bc13f95891fbc2b460098c63019245d31939e55014d54f6",
        },
        Solve => assets::Asset {
            entry: "zg_vllm_116d2ffe705b6643",
            source: include_str!("solve.cu"),
            threads: 128,
            grid: [1, 16, 1],
            shared: 10240,
            ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
        },
        Recompute => assets::Asset {
            entry: "zg_vllm_a73872840c0a9697",
            source: include_str!("recompute.cu"),
            threads: 256,
            grid: [1, 16, 1],
            shared: 40960,
            ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
        },
        H => assets::Asset {
            entry: "zg_vllm_9f97b7023b619918",
            source: include_str!("h.cu"),
            threads: 128,
            grid: [4, 16, 1],
            shared: 41220,
            ptx_sha256: "2c05861ae7051b9f267b683d7a411686edb18d8ab34598c6f9dac9c8e39dcb38",
        },
        Output => assets::Asset {
            entry: "zg_vllm_81c9eabf9b9a78e0",
            source: include_str!("output.cu"),
            threads: 128,
            grid: [4, 1, 16],
            shared: 12288,
            ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
        },
        Gated => assets::Asset {
            entry: "zg_vllm_820c1a61c99ce579",
            source: include_str!("gated.cu"),
            threads: 32,
            grid: [248, 1, 1],
            shared: 0,
            ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
        },
        _ => return Err(Error::Unsupported("original prefill stage".into())),
    })
}
