//! Explicit original standalone-exp arithmetic; no provider/model permit.
//!
//! Normal profile derived from the actual selected original48-lane HA/FTZ-off
//! dispatch (pinned holder SHA1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab,
//! ELF0xb4185b0). Scalar lane uses ONLY separate RNE F32 operations, no FMA.
//! Range reduction n=round(x*128/ln2), x=n*ln2/128+r; table split hi+lo for
//! 2^(j/128), then (hi*(r+0.5*r*r)+lo)+hi and binary exponent scaling.
//! Constants/table are mathematical values, not observed output corrections:
//! hi_ln=round((ln2/128)*2^16)/2^16=355/65536; low=hi_ln-ln2/128.
//! Table hi=RNE_F32(2^(j/128)), lo=RNE_F32(exact-hi), independently generated
//! with180-digit Decimal before comparing any authentic source output.
//! Finite normal-result domain |x|<=87 only; remaining original special/range
//! paths refuse explicitly. No scalar-libm/SLEEF substitution outside it.
use super::*;

pub use zerogravity_core::accelerator::program::StandaloneExpProfile;
use zerogravity_core::accelerator::program::standalone_exp::MKL_EXHA_SSE2_EXP2_128 as EXP2_128;

/// Borrowed source-bound contiguous vector; caller admits output/scratch and
/// retains original source. Profile selects arithmetic, never model eligibility.
pub fn exp_f32_contiguous16_profiled(
    source: SourceIdentity,
    input: &[f32],
    profile: StandaloneExpProfile,
) -> Result<Vec<f32>> {
    if !source.is_bound()
        || input.is_empty()
        || !input.len().is_multiple_of(16)
        || input.len() > MAX_ELEMENTS
    {
        return Err(Error::Malformed(
            "standalone exp source/contiguous16 extent".into(),
        ));
    }
    match profile {
        StandaloneExpProfile::ScalarLibmDiagnostic => Ok(input.iter().map(|x| x.exp()).collect()),
        StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1 => {
            if input.iter().any(|x| !x.is_finite() || x.abs() > 87.0) {
                return Err(Error::Unsupported(
                    "original standalone-exp normal profile range".into(),
                ));
            }
            Ok(input.iter().map(|&x| normal_lane(x)).collect())
        }
    }
}
fn normal_lane(x: f32) -> f32 {
    let scaled = x * f32::from_bits(0x4338aa3b); // RNE_F32(128/ln2)
    let rounded = scaled + 12582912.0; // 3*2^22; RNE integer in mantissa
    let n = rounded - 12582912.0;
    let index = ((rounded.to_bits() << 3) & 0x3f8) as usize / 8;
    let high_reduction = n * (355.0 / 65536.0);
    let r = x - high_reduction;
    let low_reduction = n * f32::from_bits(0x35de8083); // RNE_F32(355/65536-ln2/128)
    let r = r + low_reduction;
    let square = r * r;
    let quadratic = square * 0.5;
    let correction = quadratic + r;
    let hi = f32::from_bits(EXP2_128[index][0]);
    let lo = f32::from_bits(EXP2_128[index][1]);
    let corrected = correction * hi;
    let corrected = corrected + lo;
    let unscaled = corrected + hi;
    let scale = f32::from_bits((rounded.to_bits() << 16).wrapping_add(0x3f800000) & 0x7f800000);
    unscaled * scale
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_ha_rounding_counterexamples_and_refusals() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut input = vec![0.0; 48];
        // Existing independently captured raw F32 words. Neither input nor
        // expected data participates in runtime range/table construction.
        input[2] = f32::from_bits(0xbcecdf1d);
        input[38] = f32::from_bits(0xbf2461c6);
        let out = exp_f32_contiguous16_profiled(
            source,
            &input,
            StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1,
        )
        .unwrap();
        assert_eq!(out[2].to_bits(), 0x3f78b42a);
        assert_eq!(out[38].to_bits(), 0x3f06b38f);
        assert!(
            out.iter()
                .enumerate()
                .all(|(i, x)| i == 2 || i == 38 || x.to_bits() == 1.0f32.to_bits())
        );
        assert!(
            exp_f32_contiguous16_profiled(
                source,
                &input[..17],
                StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1
            )
            .is_err()
        );
        for bad in [f32::NAN, f32::INFINITY, 87.001, -87.001] {
            let invalid = vec![bad; 16];
            assert!(
                exp_f32_contiguous16_profiled(
                    source,
                    &invalid,
                    StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1
                )
                .is_err()
            );
        }
    }
}
