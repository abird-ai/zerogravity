//! One source-numeric policy shared by all CubeCL GEMV codecs.
//!
//! Native SPIR-V f32 FmaOp produced non-source FP32 bits on AMD RADV for
//! original Qwen3.5 Q4_0 and Q8_0 matrices. The conservative FP64-emulated
//! operation produces the correct source result on the measured trained
//! checkpoints and is required for normal GPU providers until broader source
//! qualification is complete. NativeF32 is ONLY a source-bit-checked diagnostic.
//! Neither strategy changes the original CPU reference mathematics.
use cubecl::{
    client::Client,
    ir::{ElemType, FloatKind},
    prelude::*,
};
use zerogravity_core::Error;
use zerogravity_core::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceAccumulation {
    WideF64,
    #[cfg(feature = "qualification")]
    NativeF32,
}

pub(crate) fn ensure_capability(client: &Client, strategy: SourceAccumulation) -> Result<()> {
    if strategy == SourceAccumulation::WideF64
        && !client
            .properties()
            .features
            .supports_type(ElemType::Float(FloatKind::F64))
    {
        return Err(Error::Unsupported(
            "strict source GEMV accumulation requires attested shader Float64; refusing non-source NativeF32"
                .into(),
        ));
    }
    Ok(())
}

/// Preserve one GGML source f32 mul_add at a time. f64 has enough exact
/// mantissa to multiply two finite f32 values before adding the existing f32
/// partial sum; rounding back to f32 then reproduces the source FMA on the
/// physically tested Qwen checkpoints. Universal f32 edge-case equivalence
/// is NOT presumed: source-real physical audits remain a release gate.
#[cube]
pub(crate) fn source_mul_add(a: f32, b: f32, accumulator: f32, #[comptime] wide: bool) -> f32 {
    if wide {
        f32::cast_from(f64::cast_from(a) * f64::cast_from(b) + f64::cast_from(accumulator))
    } else {
        fma(a, b, accumulator)
    }
}
