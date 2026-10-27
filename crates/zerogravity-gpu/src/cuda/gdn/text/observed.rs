//! Actual original per-cell source tuples; whole-model numerics remain to be executed.
use super::*;

/// Historical source-cell selector used only by source regression tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ObservedCell {
    cell: usize,
}
struct Capture {
    input_tokens: usize,
    is_decode: bool,
}
const CAPTURES: &[Capture] = &[
    Capture {
        input_tokens: 47,
        is_decode: false,
    },
    Capture {
        input_tokens: 57,
        is_decode: false,
    },
    Capture {
        input_tokens: 62,
        is_decode: false,
    },
    Capture {
        input_tokens: 67,
        is_decode: false,
    },
    Capture {
        input_tokens: 72,
        is_decode: false,
    },
    Capture {
        input_tokens: 1,
        is_decode: true,
    },
    Capture {
        input_tokens: 47,
        is_decode: false,
    },
    Capture {
        input_tokens: 57,
        is_decode: false,
    },
    Capture {
        input_tokens: 62,
        is_decode: false,
    },
    Capture {
        input_tokens: 67,
        is_decode: false,
    },
    Capture {
        input_tokens: 72,
        is_decode: false,
    },
    Capture {
        input_tokens: 1,
        is_decode: true,
    },
];
impl ObservedCell {
    pub(super) fn fixture(catalogue: usize, input_tokens: usize, is_decode: bool) -> Result<Self> {
        let cell = catalogue
            .checked_mul(6)
            .and_then(|base| {
                CAPTURES
                    .get(base..base + 6)?
                    .iter()
                    .position(|c| c.input_tokens == input_tokens && c.is_decode == is_decode)
                    .map(|i| base + i)
            })
            .ok_or_else(|| Error::Unsupported("historical text source cell absent".into()))?;
        Ok(Self { cell })
    }
    fn require_phase(self, phase: GdnPhase) -> Result<()> {
        let capture = &CAPTURES[self.cell];
        if phase.tokens() != capture.input_tokens
            || !matches!(
                (phase, capture.is_decode),
                (GdnPhase::TextPrefill(_), false) | (GdnPhase::CachedDecode, true)
            )
        {
            return Err(Error::Unsupported(
                "historical GDN source cell/phase/count".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn select(
    cell: ObservedCell,
    phase: GdnPhase,
    stage: GdnStage,
) -> Result<assets::Asset> {
    cell.require_phase(phase)?;
    if stage == GdnStage::Conv && matches!(phase, GdnPhase::TextPrefill(_)) {
        return Err(Error::Unsupported(
            "legacy specialized text Conv is qualification-only".into(),
        ));
    }
    match (cell.cell, phase) {
        (0, GdnPhase::TextPrefill(47)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen08_47/prep.cu"),
                    threads: 128,
                    grid: [3, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen08_47/cumsum.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_68bc016c3bcb6095",
                    source: include_str!("qwen08_47/kkt.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 16384,
                    ptx_sha256: "973a87c31601182c3bc13f95891fbc2b460098c63019245d31939e55014d54f6",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_116d2ffe705b6643",
                    source: include_str!("qwen08_47/solve.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_a73872840c0a9697",
                    source: include_str!("qwen08_47/recompute.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 40960,
                    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                },
                H => assets::Asset {
                    entry: "zg_vllm_444aa643ee60a35a",
                    source: include_str!("qwen08_47/h.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 49412,
                    ptx_sha256: "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_81c9eabf9b9a78e0",
                    source: include_str!("qwen08_47/output.cu"),
                    threads: 128,
                    grid: [4, 1, 16],
                    shared: 12288,
                    ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen08_47/gated.cu"),
                    threads: 32,
                    grid: [188, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (1, GdnPhase::TextPrefill(57)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen08_57/prep.cu"),
                    threads: 128,
                    grid: [4, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen08_57/cumsum.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_68bc016c3bcb6095",
                    source: include_str!("qwen08_57/kkt.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 16384,
                    ptx_sha256: "973a87c31601182c3bc13f95891fbc2b460098c63019245d31939e55014d54f6",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_b58e94d8cfdcf47d",
                    source: include_str!("qwen08_57/solve.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_a73872840c0a9697",
                    source: include_str!("qwen08_57/recompute.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 40960,
                    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                },
                H => assets::Asset {
                    entry: "zg_vllm_444aa643ee60a35a",
                    source: include_str!("qwen08_57/h.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 49412,
                    ptx_sha256: "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_a3d74997c0201bf1",
                    source: include_str!("qwen08_57/output.cu"),
                    threads: 128,
                    grid: [4, 1, 16],
                    shared: 20480,
                    ptx_sha256: "b0f269381218e2a3fa2e8148d5b83122e9fcf4d628e741164c074f8c53c93efc",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen08_57/gated.cu"),
                    threads: 32,
                    grid: [228, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (2, GdnPhase::TextPrefill(62)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen08_62/prep.cu"),
                    threads: 128,
                    grid: [4, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen08_62/cumsum.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_68bc016c3bcb6095",
                    source: include_str!("qwen08_62/kkt.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 16384,
                    ptx_sha256: "973a87c31601182c3bc13f95891fbc2b460098c63019245d31939e55014d54f6",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_116d2ffe705b6643",
                    source: include_str!("qwen08_62/solve.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_a73872840c0a9697",
                    source: include_str!("qwen08_62/recompute.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 40960,
                    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                },
                H => assets::Asset {
                    entry: "zg_vllm_444aa643ee60a35a",
                    source: include_str!("qwen08_62/h.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 49412,
                    ptx_sha256: "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_a3d74997c0201bf1",
                    source: include_str!("qwen08_62/output.cu"),
                    threads: 128,
                    grid: [4, 1, 16],
                    shared: 20480,
                    ptx_sha256: "b0f269381218e2a3fa2e8148d5b83122e9fcf4d628e741164c074f8c53c93efc",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen08_62/gated.cu"),
                    threads: 32,
                    grid: [248, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (3, GdnPhase::TextPrefill(67)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen08_67/prep.cu"),
                    threads: 128,
                    grid: [5, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen08_67/cumsum.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_2456e5b08ad199dc",
                    source: include_str!("qwen08_67/kkt.cu"),
                    threads: 256,
                    grid: [2, 16, 1],
                    shared: 24576,
                    ptx_sha256: "72c6eb7934c9cf44203d2f4c9e358248719475e93b3ec5432a501366cf771e73",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_64d5095f294cb1c1",
                    source: include_str!("qwen08_67/solve.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_eded6623852de87d",
                    source: include_str!("qwen08_67/recompute.cu"),
                    threads: 256,
                    grid: [2, 16, 1],
                    shared: 32768,
                    ptx_sha256: "d9c9fff7acb0fc04ad9b7ef177f8889ead5c5ce2df8b5d5a81a85958ae191763",
                },
                H => assets::Asset {
                    entry: "zg_vllm_9f97b7023b619918",
                    source: include_str!("qwen08_67/h.cu"),
                    threads: 128,
                    grid: [4, 16, 1],
                    shared: 41220,
                    ptx_sha256: "2c05861ae7051b9f267b683d7a411686edb18d8ab34598c6f9dac9c8e39dcb38",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_a3d74997c0201bf1",
                    source: include_str!("qwen08_67/output.cu"),
                    threads: 128,
                    grid: [4, 2, 16],
                    shared: 20480,
                    ptx_sha256: "b0f269381218e2a3fa2e8148d5b83122e9fcf4d628e741164c074f8c53c93efc",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen08_67/gated.cu"),
                    threads: 32,
                    grid: [268, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (4, GdnPhase::TextPrefill(72)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen08_72/prep.cu"),
                    threads: 128,
                    grid: [5, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen08_72/cumsum.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_68bc016c3bcb6095",
                    source: include_str!("qwen08_72/kkt.cu"),
                    threads: 256,
                    grid: [2, 16, 1],
                    shared: 16384,
                    ptx_sha256: "973a87c31601182c3bc13f95891fbc2b460098c63019245d31939e55014d54f6",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_b58e94d8cfdcf47d",
                    source: include_str!("qwen08_72/solve.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_a73872840c0a9697",
                    source: include_str!("qwen08_72/recompute.cu"),
                    threads: 256,
                    grid: [2, 16, 1],
                    shared: 40960,
                    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                },
                H => assets::Asset {
                    entry: "zg_vllm_9f97b7023b619918",
                    source: include_str!("qwen08_72/h.cu"),
                    threads: 128,
                    grid: [4, 16, 1],
                    shared: 41220,
                    ptx_sha256: "2c05861ae7051b9f267b683d7a411686edb18d8ab34598c6f9dac9c8e39dcb38",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_81c9eabf9b9a78e0",
                    source: include_str!("qwen08_72/output.cu"),
                    threads: 128,
                    grid: [4, 2, 16],
                    shared: 12288,
                    ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen08_72/gated.cu"),
                    threads: 32,
                    grid: [288, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (5, GdnPhase::CachedDecode) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => assets::Asset {
                    entry: "zg_vllm_c8f85203ea21f242",
                    source: include_str!("qwen08_cached/conv.cu"),
                    threads: 128,
                    grid: [1, 24, 1],
                    shared: 0,
                    ptx_sha256: "a1e81ca0dfff5c4f241061a3baddcc25e7407ad42a2f99f039703199fa5469aa",
                },
                Decode => assets::Asset {
                    entry: "zg_vllm_ebe262b1b494d65e",
                    source: include_str!("qwen08_cached/decode.cu"),
                    threads: 32,
                    grid: [4, 16, 1],
                    shared: 64,
                    ptx_sha256: "e15170544e78ee3ab6b19670b4cc80e9a74db549314bb57f8a897ac855283fd5",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_c15586587b4a279e",
                    source: include_str!("qwen08_cached/gated.cu"),
                    threads: 32,
                    grid: [16, 1, 1],
                    shared: 0,
                    ptx_sha256: "a97d127b124f14d226c2b24cff907e8c7aa260545e41743f84581acf6d510bbb",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (6, GdnPhase::TextPrefill(47)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen2b_47/prep.cu"),
                    threads: 128,
                    grid: [3, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen2b_47/cumsum.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_79e779d50e66d224",
                    source: include_str!("qwen2b_47/kkt.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 8192,
                    ptx_sha256: "e8ba065941c5dbf148d98ee28eba369a987d176c264ce8e05e9cb697498bad44",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_b58e94d8cfdcf47d",
                    source: include_str!("qwen2b_47/solve.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_a73872840c0a9697",
                    source: include_str!("qwen2b_47/recompute.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 40960,
                    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                },
                H => assets::Asset {
                    entry: "zg_vllm_444aa643ee60a35a",
                    source: include_str!("qwen2b_47/h.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 49412,
                    ptx_sha256: "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_a3d74997c0201bf1",
                    source: include_str!("qwen2b_47/output.cu"),
                    threads: 128,
                    grid: [4, 1, 16],
                    shared: 20480,
                    ptx_sha256: "b0f269381218e2a3fa2e8148d5b83122e9fcf4d628e741164c074f8c53c93efc",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen2b_47/gated.cu"),
                    threads: 32,
                    grid: [188, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (7, GdnPhase::TextPrefill(57)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen2b_57/prep.cu"),
                    threads: 128,
                    grid: [4, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen2b_57/cumsum.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_68bc016c3bcb6095",
                    source: include_str!("qwen2b_57/kkt.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 16384,
                    ptx_sha256: "973a87c31601182c3bc13f95891fbc2b460098c63019245d31939e55014d54f6",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_116d2ffe705b6643",
                    source: include_str!("qwen2b_57/solve.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_a73872840c0a9697",
                    source: include_str!("qwen2b_57/recompute.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 40960,
                    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                },
                H => assets::Asset {
                    entry: "zg_vllm_444aa643ee60a35a",
                    source: include_str!("qwen2b_57/h.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 49412,
                    ptx_sha256: "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_81c9eabf9b9a78e0",
                    source: include_str!("qwen2b_57/output.cu"),
                    threads: 128,
                    grid: [4, 1, 16],
                    shared: 12288,
                    ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen2b_57/gated.cu"),
                    threads: 32,
                    grid: [228, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (8, GdnPhase::TextPrefill(62)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen2b_62/prep.cu"),
                    threads: 128,
                    grid: [4, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen2b_62/cumsum.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_68bc016c3bcb6095",
                    source: include_str!("qwen2b_62/kkt.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 16384,
                    ptx_sha256: "973a87c31601182c3bc13f95891fbc2b460098c63019245d31939e55014d54f6",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_116d2ffe705b6643",
                    source: include_str!("qwen2b_62/solve.cu"),
                    threads: 128,
                    grid: [1, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_a73872840c0a9697",
                    source: include_str!("qwen2b_62/recompute.cu"),
                    threads: 256,
                    grid: [1, 16, 1],
                    shared: 40960,
                    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                },
                H => assets::Asset {
                    entry: "zg_vllm_9f97b7023b619918",
                    source: include_str!("qwen2b_62/h.cu"),
                    threads: 128,
                    grid: [4, 16, 1],
                    shared: 41220,
                    ptx_sha256: "2c05861ae7051b9f267b683d7a411686edb18d8ab34598c6f9dac9c8e39dcb38",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_81c9eabf9b9a78e0",
                    source: include_str!("qwen2b_62/output.cu"),
                    threads: 128,
                    grid: [4, 1, 16],
                    shared: 12288,
                    ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen2b_62/gated.cu"),
                    threads: 32,
                    grid: [248, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (9, GdnPhase::TextPrefill(67)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen2b_67/prep.cu"),
                    threads: 128,
                    grid: [5, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen2b_67/cumsum.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_68bc016c3bcb6095",
                    source: include_str!("qwen2b_67/kkt.cu"),
                    threads: 256,
                    grid: [2, 16, 1],
                    shared: 16384,
                    ptx_sha256: "973a87c31601182c3bc13f95891fbc2b460098c63019245d31939e55014d54f6",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_b58e94d8cfdcf47d",
                    source: include_str!("qwen2b_67/solve.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_a73872840c0a9697",
                    source: include_str!("qwen2b_67/recompute.cu"),
                    threads: 256,
                    grid: [2, 16, 1],
                    shared: 40960,
                    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                },
                H => assets::Asset {
                    entry: "zg_vllm_9f97b7023b619918",
                    source: include_str!("qwen2b_67/h.cu"),
                    threads: 128,
                    grid: [4, 16, 1],
                    shared: 41220,
                    ptx_sha256: "2c05861ae7051b9f267b683d7a411686edb18d8ab34598c6f9dac9c8e39dcb38",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_81c9eabf9b9a78e0",
                    source: include_str!("qwen2b_67/output.cu"),
                    threads: 128,
                    grid: [4, 2, 16],
                    shared: 12288,
                    ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen2b_67/gated.cu"),
                    threads: 32,
                    grid: [268, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (10, GdnPhase::TextPrefill(72)) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => unreachable!("prefill Conv intercepted above"),
                Prep => assets::Asset {
                    entry: "zg_vllm_78ff2ff19f5099b9",
                    source: include_str!("qwen2b_72/prep.cu"),
                    threads: 128,
                    grid: [5, 32, 1],
                    shared: 0,
                    ptx_sha256: "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                },
                Cumsum => assets::Asset {
                    entry: "zg_vllm_a2ad2ccfa6e953f5",
                    source: include_str!("qwen2b_72/cumsum.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 8,
                    ptx_sha256: "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7",
                },
                Kkt => assets::Asset {
                    entry: "zg_vllm_79e779d50e66d224",
                    source: include_str!("qwen2b_72/kkt.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 8192,
                    ptx_sha256: "e8ba065941c5dbf148d98ee28eba369a987d176c264ce8e05e9cb697498bad44",
                },
                Solve => assets::Asset {
                    entry: "zg_vllm_116d2ffe705b6643",
                    source: include_str!("qwen2b_72/solve.cu"),
                    threads: 128,
                    grid: [2, 16, 1],
                    shared: 10240,
                    ptx_sha256: "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                },
                Recompute => assets::Asset {
                    entry: "zg_vllm_a73872840c0a9697",
                    source: include_str!("qwen2b_72/recompute.cu"),
                    threads: 256,
                    grid: [2, 16, 1],
                    shared: 40960,
                    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                },
                H => assets::Asset {
                    entry: "zg_vllm_9f97b7023b619918",
                    source: include_str!("qwen2b_72/h.cu"),
                    threads: 128,
                    grid: [4, 16, 1],
                    shared: 41220,
                    ptx_sha256: "2c05861ae7051b9f267b683d7a411686edb18d8ab34598c6f9dac9c8e39dcb38",
                },
                Output => assets::Asset {
                    entry: "zg_vllm_a3d74997c0201bf1",
                    source: include_str!("qwen2b_72/output.cu"),
                    threads: 128,
                    grid: [4, 2, 16],
                    shared: 20480,
                    ptx_sha256: "b0f269381218e2a3fa2e8148d5b83122e9fcf4d628e741164c074f8c53c93efc",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_820c1a61c99ce579",
                    source: include_str!("qwen2b_72/gated.cu"),
                    threads: 32,
                    grid: [288, 1, 1],
                    shared: 0,
                    ptx_sha256: "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        (11, GdnPhase::CachedDecode) => {
            use GdnStage::*;
            Ok(match stage {
                Conv => assets::Asset {
                    entry: "zg_vllm_c8f85203ea21f242",
                    source: include_str!("qwen2b_cached/conv.cu"),
                    threads: 128,
                    grid: [1, 24, 1],
                    shared: 0,
                    ptx_sha256: "a1e81ca0dfff5c4f241061a3baddcc25e7407ad42a2f99f039703199fa5469aa",
                },
                Decode => assets::Asset {
                    entry: "zg_vllm_ebe262b1b494d65e",
                    source: include_str!("qwen2b_cached/decode.cu"),
                    threads: 32,
                    grid: [4, 16, 1],
                    shared: 64,
                    ptx_sha256: "e15170544e78ee3ab6b19670b4cc80e9a74db549314bb57f8a897ac855283fd5",
                },
                Gated => assets::Asset {
                    entry: "zg_vllm_c15586587b4a279e",
                    source: include_str!("qwen2b_cached/gated.cu"),
                    threads: 32,
                    grid: [16, 1, 1],
                    shared: 0,
                    ptx_sha256: "a97d127b124f14d226c2b24cff907e8c7aa260545e41743f84581acf6d510bbb",
                },
                _ => return Err(Error::Unsupported("original prefill stage".into())),
            })
        }
        _ => Err(Error::Unsupported(
            "actual original text source tuple not present".into(),
        )),
    }
}

#[cfg(test)]
mod tests;
