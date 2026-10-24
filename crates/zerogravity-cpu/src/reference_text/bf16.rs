//! Candidate eager BF16 cast schedule; not a claim about Torch GEMM/BMM/libm parity.
use super::*;
pub(super) fn round(x: f32) -> Result<f32> {
    crate::original_hf::norm::round_bf16(x)
}
pub(super) fn rms(x: &[f32], gamma: &[f32], eps: f32) -> Result<Vec<f32>> {
    if !matches!(gamma.len(), 256 | 512 | 1536)
        || !x.len().is_multiple_of(gamma.len())
        || eps.to_bits() != 1e-6f32.to_bits()
        || gamma
            .iter()
            .any(|g| !g.is_finite() || g.to_bits() & 0xffff != 0)
    {
        return Err(Error::Unsupported(
            "candidate direct-gamma BF16 RMS widths/epsilon/source".into(),
        ));
    }
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(gamma.len()) {
        let squares = row.iter().map(|v| v * v).collect::<Vec<_>>();
        // Reuse existing CPU cascade sum; add neither1 to gamma nor an intermediate BF16 cast.
        let mean = crate::original_hf::norm::sum_row(&squares) / gamma.len() as f32;
        let inverse = crate::original_hf::norm::reciprocal_sqrt_f32(mean + eps);
        for (v, g) in row.iter().zip(gamma) {
            out.push((v * inverse) * g);
        }
    }
    Ok(out)
}
pub(super) fn softcap(x: f32, cap: f32) -> f32 {
    // Three original ATen results, each stored BF16. Nonfinite values are rejected by produce.
    let divide = round(x / cap).unwrap_or(f32::NAN);
    let tanh = round(crate::original_hf::transcendentals::tanh_finite_lane(
        divide,
    ))
    .unwrap_or(f32::NAN);
    tanh * cap
}
pub(super) fn dot(x: &[f32], y: &[f32]) -> Result<f32> {
    if x.len() != y.len()
        || x.is_empty()
        || x.len() > 512
        || x.iter()
            .chain(y)
            .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed(
            "candidate BF16 attention dot source/extent".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(y.len() * 2);
    for v in y {
        bytes.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    Ok(crate::weights::dot_bf16_cpu_avx2(&bytes, x))
}
pub(super) fn rotary(x: &[f32], position: usize, g: TextRotary) -> Result<Vec<f32>> {
    g.validate(x.len())?;
    if !matches!(g.head_width, 256 | 512) || position >= 64 {
        return Err(Error::Unsupported(
            "candidate BF16 split-half rotary extent".into(),
        ));
    }
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(g.head_width) {
        let mut rotated = vec![0.0; g.head_width];
        for i in 0..g.head_width / 2 {
            let angle = if i < g.rotated_pairs {
                (1.0 / g.theta.powf((2 * i) as f32 / g.head_width as f32)) * position as f32
            } else {
                0.0
            };
            let cos = round(angle.cos())?;
            let sin = round(angle.sin())?;
            let a = row[i];
            let b = row[g.head_width / 2 + i];
            rotated[i] = round(round(a * cos)? + round(-b * sin)?)?;
            rotated[g.head_width / 2 + i] = round(round(b * cos)? + round(a * sin)?)?;
        }
        out.extend(rotated);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn casts_preserve_rne_ties_and_refuse_nonfinite_overflow() {
        assert_eq!(
            round(f32::from_bits(0x3f808000)).unwrap().to_bits(),
            0x3f800000
        );
        assert_eq!(
            round(f32::from_bits(0x3f818000)).unwrap().to_bits(),
            0x3f820000
        );
        assert_eq!(round(-0.0).unwrap().to_bits(), (-0.0f32).to_bits());
        assert!(round(f32::MAX).is_err());
        assert!(round(f32::NAN).is_err());
    }
    #[test]
    fn direct_gamma_rms_is_not_qwen_one_plus_gamma() {
        let x = vec![1.0; 256];
        let mut gamma = vec![2.0; 256];
        gamma[0] = 0.0;
        let result = rms(&x, &gamma, 1e-6).unwrap();
        assert_eq!(result[0].to_bits(), 0);
        assert_eq!(round(result[1]).unwrap(), 2.0);
        assert_ne!(result[1].to_bits(), 2.0f32.to_bits()); // FP32 weighted intermediate, before final BF16.
        assert!(rms(&x, &gamma, 1e-5).is_err());
        assert!(rms(&[1.0; 128], &[1.0; 128], 1e-6).is_err());
    }
    #[test]
    fn proportional_zero_frequency_pairs_still_execute_both_source_products() {
        let mut x = vec![0.0; 256];
        x[1] = -0.0;
        x[129] = -1.0;
        let result = rotary(
            &x,
            3,
            TextRotary {
                head_width: 256,
                rotated_pairs: 1,
                theta: 10000.0,
            },
        )
        .unwrap();
        assert_eq!(result[1].to_bits(), 0); // -0*1 + 1*0 is +0; copying the inactive pair would retain -0.
        assert_eq!(result[129], -1.0);
        assert!(
            rotary(
                &x,
                64,
                TextRotary {
                    head_width: 256,
                    rotated_pairs: 1,
                    theta: 10000.0
                }
            )
            .is_err()
        );
    }
}
