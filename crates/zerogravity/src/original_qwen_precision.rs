//! Qwen original-source numerical dtype boundaries, source-free and CPU-only.
//!
//! This module DOES NOT infer BF16 reference equivalence from a checkpoint
//! dtype. It implements observable PyTorch/FLA *dtype* casts and preserves
//! FP32 recurrent state. Original trained HF/FLA numerical parity needs
//! independent checkpoint/framework oracle tests.

use crate::accelerator::SourceIdentity;
use crate::{Error, Result};

/// Choice of numerical *diagnostic*, never a model family registration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OriginalNumericalDType {
    /// Existing analytically qualified CPU reference.
    F32,
    /// Explicit BF16 dtype-boundary probe, NOT trained-HF parity.
    Bf16Diagnostic,
}
impl OriginalNumericalDType {
    /// Convert finite f32 to IEEE BF16 using round-to-nearest, ties-to-even.
    /// The returned f32 contains exactly representable BF16 bits.
    ///
    /// This is the 16-bit roundtrip; it is NOT the instruction-by-instruction
    /// rounding policy of a CUDA matmul or the FLA kernel's accumulation.
    pub fn round(self, value: f32) -> Result<f32> {
        if !value.is_finite() {
            return Err(Error::Malformed("nonfinite Qwen source dtype input".into()));
        }
        if self == Self::F32 {
            return Ok(value);
        }
        let bits = value.to_bits();
        let bias = 0x7fffu32 + ((bits >> 16) & 1);
        let rounded = f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000);
        if !rounded.is_finite() {
            return Err(Error::Limit("original BF16 round overflow".into()));
        }
        Ok(rounded)
    }
    pub fn values(self, items: &[f32]) -> Result<Vec<f32>> {
        const MAX_ELEMENTS: usize = 1_048_576;
        if items.len() > MAX_ELEMENTS {
            return Err(Error::Limit("original dtype stage maximum elements".into()));
        }
        items.iter().map(|&v| self.round(v)).collect()
    }
    /// Source RMSNorm: x.float() -> FP32 norm -> multiply (1+w.float())
    /// -> .to(input.dtype). No BF16 cast BEFORE gamma multiplication.
    pub fn zero_centered_rms(self, input: &[f32], raw_gamma: &[f32], eps: f32) -> Result<Vec<f32>> {
        if input.len() != raw_gamma.len()
            || input.is_empty()
            || !eps.is_finite()
            || eps <= 0.0
            || raw_gamma
                .iter()
                .any(|&x| !x.is_finite() || !(1.0 + x).is_finite())
        {
            return Err(Error::Malformed(
                "original source zero-centered norm parameters".into(),
            ));
        }
        let input = self.values(input)?;
        let gamma = raw_gamma.iter().map(|&x| 1.0 + x).collect::<Vec<_>>();
        let output = crate::qwen35::source_rms_normalize(&input, &gamma, eps)?;
        self.values(&output)
    }
    /// Same headwise scalar source operator: independent rows,
    /// with cast only after the FP32 normalized * gamma result.
    pub fn zero_centered_head_rms(
        self,
        input: &[f32],
        raw_gamma: &[f32],
        heads: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if raw_gamma.is_empty()
            || raw_gamma.len() > 8192
            || heads == 0
            || heads.checked_mul(raw_gamma.len()) != Some(input.len())
            || !eps.is_finite()
            || eps <= 0.0
            || raw_gamma
                .iter()
                .any(|&x| !x.is_finite() || !(1.0 + x).is_finite())
        {
            return Err(Error::Malformed(
                "original Qwen head RMS shape/dtype".into(),
            ));
        }
        let input = self.values(input)?;
        let gamma = raw_gamma.iter().map(|&x| 1.0 + x).collect::<Vec<_>>();
        let output = crate::qwen35::source_grouped_rms_normalize(&input, &gamma, heads, eps)?;
        self.values(&output)
    }
    /// GDN GatedRMSNorm is *not* zero-centered: normalize FP32; cast
    /// core to its original BF16 dtype; multiply original raw learned gamma;
    /// then multiply SiLU(gate.float()) and cast to original dtype.
    /// This is distinct from ordinary RMSNorm and directly matches the
    /// published eager operator order, but not unqualified FLA kernels.
    pub fn gated_rms(
        self,
        source: SourceIdentity,
        core: &[f32],
        gate: &[f32],
        raw_gamma: &[f32],
        heads: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if !source.is_bound()
            || core.is_empty()
            || core.len() != gate.len()
            || raw_gamma.is_empty()
            || heads == 0
            || heads.checked_mul(raw_gamma.len()) != Some(core.len())
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(Error::Malformed(
                "original gated RMS shape/owner mismatch".into(),
            ));
        }
        let identity = vec![1.0; raw_gamma.len()];
        let input = self.values(core)?;
        let norm = crate::qwen35::source_grouped_rms_normalize(&input, &identity, heads, eps)?;
        // First original cast occurs BEFORE multiplication by raw gamma.
        let norm = self.values(&norm)?;
        let mut weighted = Vec::with_capacity(norm.len());
        for (i, &v) in norm.iter().enumerate() {
            let gamma = raw_gamma[i % raw_gamma.len()];
            weighted.push(self.round(v * gamma)?);
        }
        let gate = self.values(gate)?;
        let mut final_values = Vec::with_capacity(weighted.len());
        for (&n, &g) in weighted.iter().zip(&gate) {
            let activated = crate::math::silu(g);
            final_values.push(self.round(n * activated)?);
        }
        Ok(final_values)
    }
    /// FP32 GDN recurrent core stays FP32. Only returned core values cast
    /// back to input dtype after updating the private FP32 recurrent state.
    pub fn recurrent_output(self, core: &[f32]) -> Result<Vec<f32>> {
        self.values(core)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ties_to_even_and_signed_zeros_match_independent_ieee_bf16_bits() {
        let d = OriginalNumericalDType::Bf16Diagnostic;
        // Adjacent BF16 numbers: 1.0 (0x3f80), 1.0078125 (0x3f81),
        // 1.015625 (0x3f82). First exact midpoint ties down (even),
        // second midpoint ties up (even); negatives preserve sign.
        for (source_bits, expected_bits) in [
            (0x3f80_8000u32, 0x3f80_0000),
            (0x3f81_8000, 0x3f82_0000),
            (0xbf80_8000, 0xbf80_0000),
            (0xbf81_8000, 0xbf82_0000),
            (0x0000_0000, 0x0000_0000),
            (0x8000_0000, 0x8000_0000),
            (0x0000_8000, 0x0000_0000),
            (0x0001_8000, 0x0002_0000),
            (0x3f81_7fff, 0x3f81_0000),
            (0x3f81_8001, 0x3f82_0000),
        ] {
            let actual = d.round(f32::from_bits(source_bits)).unwrap();
            assert_eq!(actual.to_bits(), expected_bits, "input=0x{source_bits:08x}");
        }
        assert!(d.round(f32::INFINITY).is_err());
        assert!(d.round(f32::NAN).is_err());
        assert!(d.round(f32::from_bits(0x7f7f_ffff)).is_err());
        assert_eq!(
            OriginalNumericalDType::F32
                .round(f32::from_bits(0x3f80_8000))
                .unwrap(),
            f32::from_bits(0x3f80_8000)
        );
    }
    #[test]
    fn independent_tiny_rms_reference_places_one_bf16_cast_after_full_gamma() {
        let d = OriginalNumericalDType::Bf16Diagnostic;
        // Two identical 1s: normalized≈1, raw zero-centered gamma 0.1015625
        // -> 1.1015625, exactly halfway between BF16 1.09375 and 1.109375.
        // The source operator computes FP32 gamma first; BF16 casting the
        // gamma prematurely would not generally match that policy.
        let x = [1.0, 1.0];
        let weight = [0.1015625, 0.1015625];
        let result = d.zero_centered_rms(&x, &weight, 1e-6).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|v| v.to_bits() & 0xffff == 0));
        let raw = OriginalNumericalDType::F32
            .zero_centered_rms(&x, &weight, 1e-6)
            .unwrap();
        assert_eq!(result, d.values(&raw).unwrap());
        assert!(result[0] != raw[0]);
        assert!(d.zero_centered_rms(&x, &weight[..1], 1e-6).is_err());
        assert!(d.zero_centered_rms(&x, &weight, 0.0).is_err());
    }
    #[test]
    fn gated_rms_keeps_original_separate_normalize_gamma_and_gate_casts() {
        let d = OriginalNumericalDType::Bf16Diagnostic;
        let source = SourceIdentity::for_new_allocation().unwrap();
        let normalized = d
            .gated_rms(
                source,
                &[1.0, 1.0],
                [1.0, 1.0].as_slice(),
                &[0.375, 0.375],
                1,
                1e-6,
            )
            .unwrap();
        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().all(|v| v.to_bits() & 0xffff == 0));
        let f = OriginalNumericalDType::F32
            .gated_rms(source, &[1.0, 1.0], &[1.0, 1.0], &[0.375, 0.375], 1, 1e-6)
            .unwrap();
        assert!(f.iter().all(|v| v.is_finite()));
        assert!(
            d.gated_rms(SourceIdentity::unbound(), &[1.0], &[1.0], &[1.0], 1, 1e-6)
                .is_err()
        );
        assert!(
            d.gated_rms(source, &[1.0], &[1.0, 2.0], &[1.0], 1, 1e-6)
                .is_err()
        );
    }
    #[test]
    fn independent_numpy_scalar_reference_rms_casts_match_exact_bf16_vectors() {
        // Offline Python NumPy float32 scalar oracle, separately authored
        // from the Rust implementation; expected BF16 bits are immutable.
        // Source: Qwen3_5RMSNorm.forward, FP32 norm and gamma, cast once.
        let dtype = OriginalNumericalDType::Bf16Diagnostic;
        let x = [0.10123456, f32::from_bits(0x3f80_8000), -0.55, 1.25];
        let gamma = [0.3, -0.1, 0.75, -0.25];
        let value = dtype.zero_centered_rms(&x, &gamma, 1e-6).unwrap();
        let gold = [0x3e1f_0000, 0x3f88_0000, 0xbf91_0000, 0x3f8e_0000];
        for (i, v) in value.iter().enumerate() {
            assert_eq!(v.to_bits(), gold[i], "original RMS BF16 lane {i}");
        }
    }
    #[test]
    fn independent_numpy_scalar_reference_gated_rms_casts_match_exact_bf16_vectors() {
        // GatedRMSNorm: each head normalizes in FP32, casts to BF16,
        // multiplies raw learned BF16 gamma (round to BF16), then
        // SiLU(gate.float()) and output cast to BF16.
        let dtype = OriginalNumericalDType::Bf16Diagnostic;
        let source = SourceIdentity::for_new_allocation().unwrap();
        let core = [0.10123456, f32::from_bits(0x3f80_8000), -0.55, 1.25];
        let gate = [0.515, 0.1211, -0.9, 1.2345];
        let weights = [0.375, 0.125];
        let result = dtype
            .gated_rms(source, &core, &gate, &weights, 2, 1e-6)
            .unwrap();
        let gold = [0x3c8d_0000, 0x3c39_0000, 0x3d64_0000, 0x3e1f_0000];
        for (i, v) in result.iter().enumerate() {
            assert_eq!(v.to_bits(), gold[i], "original gated BF16 lane {i}");
        }
    }
    #[test]
    fn independent_two_step_fla_fp32_state_bf16_output_oracle() {
        // Independently evaluated NumPy eager FLA one-token recurrence:
        // S'=gain*S + k * (v - dot(k,gain*S))*beta;
        // o=dot(q,S')/sqrt(key_dim); o.bfloat16(), STATE stays FP32.
        // Historical GGUF/FLA FMA reduction order may differ a few ULPs:
        // assert exact BF16 returned value, FP32 state within 2e-7.
        use crate::accelerator::GdnCoefficients;
        let dtype = OriginalNumericalDType::Bf16Diagnostic;
        let coeff = GdnCoefficients {
            gain: 0.5,
            beta: 0.5,
            scale: 1.0 / (2.0f32).sqrt(),
        };
        let mut state = vec![0.0; 4];
        let mut core = vec![0.0; 2];
        let pairs = [
            (
                [0.6, 0.8],
                [0.8, 0.6],
                [0.5, -0.125],
                [0x3e2e_0000u32, 0xbd2e_0000],
                [0x3e4c_cccd, 0x3e19_999a, 0xbd4c_cccd, 0xbd19_999a],
            ),
            (
                [0.8, 0.6],
                [0.6, 0.8],
                [0.75, 0.375],
                [0x3e9b_0000u32, 0x3dec_0000],
                [0x3e93_f7cf, 0x3ea7_6c8c, 0x3dc5_a1cc, 0x3e12_b021],
            ),
        ];
        for (q, k, v, expected_core, expected_state) in pairs {
            let v = dtype.values(&v).unwrap();
            crate::qwen35::source_gdn_recurrent_step(&mut state, &q, &k, &v, coeff, &mut core)
                .unwrap();
            let returned = dtype.recurrent_output(&core).unwrap();
            for (actual, expected) in returned.iter().zip(expected_core) {
                assert_eq!(
                    actual.to_bits(),
                    expected,
                    "reference original BF16 recurrence return"
                );
            }
            for (actual, expected) in state.iter().zip(expected_state) {
                let exact = f32::from_bits(expected);
                assert!(
                    (actual - exact).abs() <= 2e-7,
                    "FP32 state discrepancy: actual={actual}, expected={exact}"
                );
            }
        }
        assert!(state.iter().any(|v| v.to_bits() & 0xffff != 0));
    }
    #[test]
    fn bf16_recurrent_output_rounds_but_does_not_rewrite_fp32_state() {
        let d = OriginalNumericalDType::Bf16Diagnostic;
        let state = vec![f32::from_bits(0x3f80_8000), 0.101_234_56];
        let bf16 = d.recurrent_output(&state).unwrap();
        assert_ne!(bf16, state);
        assert_eq!(state[0].to_bits(), f32::from_bits(0x3f80_8000).to_bits());
        assert_eq!(bf16[0], 1.0);
        assert_eq!(
            OriginalNumericalDType::F32
                .recurrent_output(&state)
                .unwrap(),
            state
        );
    }
}
