//! Nonpositive original MKL HA lanes, including gradual underflow and -inf masks.
//! Normal packed-F32 lane and rare FP64 reduction are distinct source paths.
//! Original rare path: libtorch_cpu vsexp_cout_rare, ELF0xb41a130. No FMA/libm.
use super::*;
use zerogravity_core::accelerator::program::standalone_exp::MKL_EXHA_EXP2_64_F64;

/// Chunk-decay exp, including original causal -inf masks. NaN/+inf refuse.
/// Does not widen the existing normal-profile contract.
/// Caller admits storage; no native or full-model numerical eligibility.
pub fn exp_nonpositive_f32_contiguous16(source: SourceIdentity, input: &[f32]) -> Result<Vec<f32>> {
    if !source.is_bound()
        || input.is_empty()
        || !input.len().is_multiple_of(16)
        || input.len() > MAX_ELEMENTS
        || input.iter().any(|x| x.is_nan() || *x > 0.0)
    {
        return Err(Error::Malformed(
            "original chunk exp nonpositive/mask/contiguous16 extent".into(),
        ));
    }
    Ok(input.iter().map(|&x| lane(x)).collect())
}

fn lane(x: f32) -> f32 {
    if x >= -f32::from_bits(0x42ae_ac4f) {
        return super::normal_lane(x);
    }
    if x < f32::from_bits(0xc2cf_f1b4) {
        return 0.0;
    }
    // Source FP64 range reduction: round(x*64/ln2), split -ln2/64.
    let x = f64::from(x);
    let scaled = x * f64::from_bits(0x4057_1547_652b_82fe);
    let rounded = scaled + 6_755_399_441_055_744.0;
    let n = rounded - 6_755_399_441_055_744.0;
    let high = n * f64::from_bits(0xbf86_2e42_fefa_0000);
    let low = n * f64::from_bits(0xbd1c_f79a_bc9e_3b3a);
    let r = (high + x) + low;
    // Original HA minimax coefficients; separate FP64 multiply/add boundaries.
    let p = f64::from_bits(0x3f56_c16a_1c2a_3ffd) * r;
    let p = p + f64::from_bits(0x3f81_1112_3aaf_20d3);
    let p = p * r;
    let p = p + f64::from_bits(0x3fa5_5555_5555_8fcc);
    let p = p * r;
    let p = p + f64::from_bits(0x3fc5_5555_5555_48f8);
    let p = p * r;
    let p = p + 0.5;
    let p = p * r;
    let p = p * r;
    let p = p + r;
    let bits = rounded.to_bits() as u32;
    let pair = MKL_EXHA_EXP2_64_F64[(bits & 63) as usize];
    let correction = p + f64::from_bits(pair[1]);
    let hi = f64::from_bits(pair[0]);
    let correction = correction * hi;
    let exponent = ((bits >> 6).wrapping_add(0x43b)) & 0x7ff;
    let scale = f64::from_bits(u64::from(exponent) << 52);
    let correction = correction * scale;
    let hi = hi * scale;
    let sum = hi + correction;
    let residual = (hi - sum) + correction;
    // Original compensated scaling by2^-60; preserve its FP64 rounding points.
    let shifted = sum * 6_442_450_944.0;
    let split = (sum + shifted) - shifted;
    let tail = (sum - split) + residual;
    let scale_down = f64::from_bits(0x3c30_0000_0000_0000);
    let high = split * scale_down;
    let low = tail * scale_down;
    (high + low) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_domain_keeps_zero_sign_semantics_and_refuses_other_profiles() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut input = vec![0.0; 16];
        input[0] = -0.0;
        input[1] = -f32::MAX;
        input[2] = -104.0;
        input[3] = f32::NEG_INFINITY;
        let out = exp_nonpositive_f32_contiguous16(source, &input).unwrap();
        assert_eq!(out[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(out[1].to_bits(), 0);
        assert_eq!(out[2].to_bits(), 0);
        assert_eq!(out[3].to_bits(), 0);
        assert!(out[4..].iter().all(|v| *v == 1.0));
        for bad in [f32::NAN, f32::INFINITY, 0.01] {
            input[0] = bad;
            assert!(exp_nonpositive_f32_contiguous16(source, &input).is_err());
        }
        assert!(exp_nonpositive_f32_contiguous16(source, &[]).is_err());
        assert!(exp_nonpositive_f32_contiguous16(source, &[0.0; 17]).is_err());
        assert!(exp_nonpositive_f32_contiguous16(SourceIdentity::unbound(), &[0.0; 16]).is_err());
        assert!(
            exp_f32_contiguous16_profiled(
                source,
                &[-88.0; 16],
                StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1
            )
            .is_err()
        );
    }
}
