//! Backend-owned KKT/Recompute/H/Output prefill implementations.
use super::*;
use crate::cuda::kernel::{
    StaticScalarBlock, StaticScalarSchema, StaticScalarType, StaticScalarValue,
};

const OUTPUT_SCALE_BITS: u32 = 1_035_273_459;

struct Variant {
    entry: &'static str,
    source: &'static str,
    threads: u32,
    shared: u32,
    ptx_sha256: &'static str,
}

const KKT_BK32_W4_S3: Variant = Variant {
    entry: "zg_gdn_kkt_prefill_bk32_w4_s3_v1",
    source: include_str!("kkt_bk32_w4_s3.cu"),
    threads: 128,
    shared: 8192,
    ptx_sha256: "e8ba065941c5dbf148d98ee28eba369a987d176c264ce8e05e9cb697498bad44",
};
const KKT_BK32_W8_S3: Variant = Variant {
    entry: "zg_gdn_kkt_prefill_bk32_w8_s3_v1",
    source: include_str!("kkt_bk32_w8_s3.cu"),
    threads: 256,
    shared: 16384,
    ptx_sha256: "973a87c31601182c3bc13f95891fbc2b460098c63019245d31939e55014d54f6",
};
const KKT_BK64_W8_S4: Variant = Variant {
    entry: "zg_gdn_kkt_prefill_bk64_w8_s4_v1",
    source: include_str!("kkt_bk64_w8_s4.cu"),
    threads: 256,
    shared: 24576,
    ptx_sha256: "72c6eb7934c9cf44203d2f4c9e358248719475e93b3ec5432a501366cf771e73",
};
const RECOMPUTE_S3: Variant = Variant {
    entry: "zg_gdn_recompute_prefill_bk64_bv64_w8_s3_v1",
    source: include_str!("recompute_bk64_bv64_w8_s3.cu"),
    threads: 256,
    shared: 32768,
    ptx_sha256: "d9c9fff7acb0fc04ad9b7ef177f8889ead5c5ce2df8b5d5a81a85958ae191763",
};
const RECOMPUTE_S4: Variant = Variant {
    entry: "zg_gdn_recompute_prefill_bk64_bv64_w8_s4_v1",
    source: include_str!("recompute_bk64_bv64_w8_s4.cu"),
    threads: 256,
    shared: 40960,
    ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
};
const H_BV32: Variant = Variant {
    entry: "zg_gdn_h_prefill_bv32_w4_s2_v1",
    source: include_str!("h_bv32_w4_s2.cu"),
    threads: 128,
    shared: 41220,
    ptx_sha256: "2c05861ae7051b9f267b683d7a411686edb18d8ab34598c6f9dac9c8e39dcb38",
};
const H_BV64: Variant = Variant {
    entry: "zg_gdn_h_prefill_bv64_w4_s2_v1",
    source: include_str!("h_bv64_w4_s2.cu"),
    threads: 128,
    shared: 49412,
    ptx_sha256: "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
};
const OUTPUT_S2: Variant = Variant {
    entry: "zg_gdn_output_prefill_bk32_bv32_w4_s2_v1",
    source: include_str!("output_bk32_bv32_w4_s2.cu"),
    threads: 128,
    shared: 12288,
    ptx_sha256: "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
};
const OUTPUT_S3: Variant = Variant {
    entry: "zg_gdn_output_prefill_bk32_bv32_w4_s3_v1",
    source: include_str!("output_bk32_bv32_w4_s3.cu"),
    threads: 128,
    shared: 20480,
    ptx_sha256: "b0f269381218e2a3fa2e8148d5b83122e9fcf4d628e741164c074f8c53c93efc",
};

fn tokens(phase: GdnPhase, slot: GdnCacheSlot) -> Result<usize> {
    let GdnPhase::TextPrefill(tokens) = phase else {
        return Err(Error::Unsupported(
            "configured GDN implementation is text prefill only".into(),
        ));
    };
    slot.validate(phase)?;
    if tokens == 0 || i32::try_from(tokens).is_err() {
        return Err(Error::Limit(
            "configured GDN text/control i32 extent".into(),
        ));
    }
    Ok(tokens)
}

pub(super) fn prepare(
    config: GdnKernelConfig,
    phase: GdnPhase,
    slot: GdnCacheSlot,
    stage: GdnStage,
) -> Result<parameterized_prefill::Definition> {
    use GdnKktKernel::*;
    use GdnOutputKernel::*;
    use GdnRecomputeKernel::*;
    use GdnStateKernel::*;

    let tokens = tokens(phase, slot)?;
    let chunks = u32::try_from(tokens.div_ceil(64))
        .map_err(|_| Error::Limit("configured GDN chunk grid u32 overflow".into()))?;
    let (variant, grid) = match stage {
        GdnStage::Kkt => {
            let variant = match config.kkt() {
                Bk32W4S3 => &KKT_BK32_W4_S3,
                Bk32W8S3 => &KKT_BK32_W8_S3,
                Bk64W8S4 => &KKT_BK64_W8_S4,
            };
            (variant, [chunks, 16, 1])
        }
        GdnStage::Recompute => (
            match config.recompute() {
                Bk64Bv64W8S3 => &RECOMPUTE_S3,
                Bk64Bv64W8S4 => &RECOMPUTE_S4,
            },
            [chunks, 16, 1],
        ),
        GdnStage::H => match config.state() {
            Bv32W4S2 => (&H_BV32, [4, 16, 1]),
            Bv64W4S2 => (&H_BV64, [2, 16, 1]),
        },
        GdnStage::Output => (
            match config.output() {
                Bk32Bv32W4S2 => &OUTPUT_S2,
                Bk32Bv32W4S3 => &OUTPUT_S3,
            },
            [4, chunks, 16],
        ),
        _ => {
            return Err(Error::Unsupported(
                "configured GDN KKT/Recompute/H/Output stage".into(),
            ));
        }
    };
    let scalars = if stage == GdnStage::Output {
        StaticScalarBlock::new(
            StaticScalarSchema::new(&[(StaticScalarType::F32Bits, 0)], 8, 8)?,
            &[StaticScalarValue::F32Bits(OUTPUT_SCALE_BITS)],
        )?
    } else {
        StaticScalarBlock::empty()
    };
    Ok(parameterized_prefill::Definition {
        asset: assets::Asset {
            entry: variant.entry,
            source: variant.source,
            threads: variant.threads,
            grid,
            shared: variant.shared,
            ptx_sha256: variant.ptx_sha256,
        },
        scalars,
    })
}
