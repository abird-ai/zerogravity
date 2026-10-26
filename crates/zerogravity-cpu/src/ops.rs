//! Deterministic CPU reference operations used to validate compiled regions.
//!
//! These are mathematical building blocks, NOT full Qwen GDN or Gemma MoE layer
//! implementations. Model recipes will register exact higher-level semantics.

mod gdn;
pub use gdn::gdn_conv_step;

use crate::{Error, Result};

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Malformed(message.into()))
    }
}

/// Reference row-major matrix vector product, using diagnostic f64 accumulation.
pub fn matvec(matrix: &[f32], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    check(
        rows.checked_mul(cols) == Some(matrix.len()),
        "matvec matrix shape",
    )?;
    check(input.len() == cols, "matvec input size")?;
    check(rows <= 16_000_000, "matvec output too large")?;
    let mut out = Vec::new();
    out.try_reserve_exact(rows)
        .map_err(|_| Error::Limit("matvec output allocation failed".into()))?;
    for row in matrix.chunks_exact(cols.max(1)).take(rows) {
        let acc = row
            .iter()
            .zip(input)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>();
        out.push(acc as f32);
    }
    if cols == 0 {
        out.resize(rows, 0.0);
    }
    Ok(out)
}

/// RMS norm reference, without residual addition or model-specific postscaling.
pub fn rms_norm(values: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    check(
        values.len() == weight.len() && !values.is_empty(),
        "rms norm shape",
    )?;
    check(epsilon.is_finite() && epsilon > 0.0, "rms norm epsilon")?;
    let sum: f64 = values.iter().map(|v| f64::from(*v).powi(2)).sum();
    let inv = (sum / values.len() as f64 + f64::from(epsilon))
        .sqrt()
        .recip();
    Ok(values
        .iter()
        .zip(weight)
        .map(|(x, w)| (*x as f64 * inv * f64::from(*w)) as f32)
        .collect())
}

/// Stable probability normalization. Reject NaN and infinities instead of masking
/// an unknown model's attention or router semantics.
pub fn softmax(values: &[f32]) -> Result<Vec<f32>> {
    check(!values.is_empty(), "softmax empty")?;
    check(
        values.iter().all(|x| x.is_finite()),
        "softmax requires finite logits",
    )?;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = values.iter().map(|v| f64::from(*v - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    check(sum > 0.0 && sum.is_finite(), "softmax unstable")?;
    Ok(weights.into_iter().map(|v| (v / sum) as f32).collect())
}

/// Router top-k selection is separated from model-specific scoring,
/// renormalization, and shared-expert accumulation rules.
pub fn top_k(logits: &[f32], k: usize) -> Result<Vec<usize>> {
    check(k > 0 && k <= logits.len(), "top-k size")?;
    check(
        logits.iter().all(|x| x.is_finite()),
        "top-k requires finite logits",
    )?;
    let mut rank: Vec<usize> = (0..logits.len()).collect();
    rank.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    rank.truncate(k);
    Ok(rank)
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn gelu_tanh(x: f32) -> f32 {
    let a = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (a * (x + 0.044715 * x * x * x)).tanh())
}

/// One causal attention query over keys/values at positions 0..T.
/// `window=None` is full attention; `window=Some(n)` keeps the last n
/// positions inclusive of query position. This is diagnostic CPU math.
pub fn attention_one(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    query_position: usize,
    window: Option<usize>,
) -> Result<Vec<f32>> {
    let d = query.len();
    check(
        d > 0 && keys.len() == values.len() && keys.len().is_multiple_of(d),
        "attention head geometry",
    )?;
    let tokens = keys.len() / d;
    check(
        tokens > 0 && query_position < tokens,
        "attention query position",
    )?;
    if let Some(n) = window {
        check(n > 0, "sliding attention window")?;
    }
    let start = window.map_or(0, |n| (query_position + 1).saturating_sub(n));
    let scale = 1.0 / (d as f64).sqrt();
    let mut scores = Vec::new();
    for t in start..=query_position {
        let v = keys[t * d..(t + 1) * d]
            .iter()
            .zip(query)
            .map(|(&k, &q)| f64::from(k) * f64::from(q))
            .sum::<f64>();
        scores.push((v * scale) as f32);
    }
    let probs = softmax(&scores)?;
    let mut out = vec![0.0; d];
    for (i, p) in probs.iter().enumerate() {
        let v = &values[(start + i) * d..(start + i + 1) * d];
        for (dst, &x) in out.iter_mut().zip(v) {
            *dst += x * p;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_and_norm() {
        assert_eq!(
            matvec(&[1., 2., 3., 4.], 2, 2, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        assert!(matvec(&[1., 2.], 2, 2, &[1., 2.]).is_err());
        assert_eq!(matvec(&[], 3, 0, &[]).unwrap(), [0., 0., 0.]);
        let norm = rms_norm(&[3., 4.], &[1., 1.], 1e-6).unwrap();
        assert!((norm[0] - 0.848528).abs() < 1e-5);
        assert!((norm[1] - 1.131_37).abs() < 1e-5);
    }

    #[test]
    fn stable_softmax_and_tie_routing() {
        let probs = softmax(&[1000., 1000., 999.]).unwrap();
        assert!((probs.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert_eq!(top_k(&[3., 4., 4., 3.], 2).unwrap(), [1, 2]);
        assert!(top_k(&[f32::NAN, 1.], 1).is_err());
        assert!(softmax(&[f32::NEG_INFINITY]).is_err());
        assert!((silu(0.0)).abs() < 1e-9);
        assert!((gelu_tanh(0.0)).abs() < 1e-9);
    }

    #[test]
    fn sliding_vs_global_attention_and_future_mask() {
        let q = [1.];
        let k = [0., 0., 0., 0.];
        let v = [1., 2., 3., 100.];
        let global = attention_one(&q, &k, &v, 2, None).unwrap();
        let sliding = attention_one(&q, &k, &v, 2, Some(2)).unwrap();
        assert!((global[0] - 2.0).abs() < 1e-6);
        assert!((sliding[0] - 2.5).abs() < 1e-6);
        assert!(attention_one(&q, &k, &v, 1, Some(0)).is_err());
    }
}

/// Deterministic unfiltered logits/log-probability audit. These are produced
/// *before* temperature, penalties, truncation or grammar; a sampler's
/// post-filter probabilities are not an equivalent numerical oracle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogitRank {
    pub id: u32,
    pub delta: f32,
    pub logprob: f64,
}

pub fn top_logit_ranks(logits: &[f32], n: usize) -> Result<Vec<LogitRank>> {
    if logits.is_empty() || n == 0 || n > 128 || logits.len() > u32::MAX as usize {
        return Err(Error::Malformed("invalid logits audit width/rank".into()));
    }
    if !logits.iter().all(|x| x.is_finite()) {
        return Err(Error::Malformed("nonfinite logits in audit".into()));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|x| f64::from(*x - max).exp())
        .sum::<f64>();
    let ln_sum = sum.ln();
    if !ln_sum.is_finite() {
        return Err(Error::Malformed("nonfinite logits normalization".into()));
    }
    let mut indices: Vec<_> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&i, &j| logits[j].total_cmp(&logits[i]).then_with(|| i.cmp(&j)));
    Ok(indices
        .into_iter()
        .take(n)
        .map(|id| {
            let delta = logits[id] - max;
            LogitRank {
                id: id as u32,
                delta,
                logprob: f64::from(delta) - ln_sum,
            }
        })
        .collect())
}

#[cfg(test)]
mod logit_audit_tests {
    use super::*;
    #[test]
    fn test_raw_log_probs_pre_sampler() {
        let ranks = top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 3).unwrap();
        assert_eq!(ranks.iter().map(|r| r.id).collect::<Vec<_>>(), [1, 2, 0]);
        assert_eq!(ranks[0].delta, 0.0);
        assert_eq!(ranks[2].delta, -2.0);
        assert!(
            (ranks.iter().map(|r| r.logprob.exp()).sum::<f64>()
                + top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 4).unwrap()[3]
                    .logprob
                    .exp()
                - 1.0)
                .abs()
                < 1e-10
        );
        assert!(top_logit_ranks(&[0.0, f32::NAN], 2).is_err());
    }
}

#[cfg(test)]
mod long_context_reference_tests {
    use super::*;

    #[test]
    fn thirty_two_thousand_causal_attention_positions_vs_independent_double_oracle() {
        const CONTEXT: usize = 32_768;
        const WIDTH: usize = 4;
        let query = [0.4, -0.7, 0.2, 0.1];
        let keys = (0..=CONTEXT)
            .flat_map(|t| (0..WIDTH).map(move |i| (((t * 7 + i * 37) % 113) as f32 - 56.0) / 97.0))
            .collect::<Vec<_>>();
        let mut values = (0..=CONTEXT)
            .flat_map(|t| {
                (0..WIDTH).map(move |i| (((t * 13 + i * 17) % 199) as f32 - 99.0) / 201.0)
            })
            .collect::<Vec<_>>();
        // Future token must never contribute; sentinel catches mask bugs.
        values[CONTEXT * WIDTH..].fill(1_000_000.0);
        let actual = attention_one(&query, &keys, &values, CONTEXT - 1, None).unwrap();
        let window = attention_one(&query, &keys, &values, CONTEXT - 1, Some(1024)).unwrap();
        let independently_evaluate = |start: usize| {
            let mut scores = Vec::new();
            for t in start..CONTEXT {
                let score = query
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| f64::from(q) * f64::from(keys[t * WIDTH + i]))
                    .sum::<f64>()
                    / (WIDTH as f64).sqrt();
                scores.push(score);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            std::array::from_fn::<_, WIDTH, _>(|i| {
                (weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * f64::from(values[(start + j) * WIDTH + i]))
                    .sum::<f64>()
                    / sum) as f32
            })
        };
        let global_oracle = independently_evaluate(0);
        let sliding_oracle = independently_evaluate(CONTEXT - 1024);
        for i in 0..WIDTH {
            assert!(
                (actual[i] - global_oracle[i]).abs() < 2e-6,
                "full attention reference mismatch in channel {i}"
            );
            assert!(
                (window[i] - sliding_oracle[i]).abs() < 2e-6,
                "sliding attention reference mismatch in channel {i}"
            );
        }
    }
}

/// GGML Haswell/AVX2 scalar-equivalent approximate exponential used in the
/// source execution mode's SIMD SiLU implementation. This is *not* the
/// mathematical exp reference: its FMA polynomial and float-bit exponent
/// construction are part of the requested source-numerical specialization.
///
/// The common finite model gate range is handled by the exact polynomial
/// branch. For other finite ranges, follow the original SIMD scaling branches;
/// NaN/Inf values are rejected by the model state boundary before this stage.
pub fn ggml_source_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return x.exp();
    }
    let r = f32::from_bits(0x4b40_0000);
    let z = x.mul_add(f32::from_bits(0x3fb8_aa3b), r);
    let n = z - r;
    let b = (-n).mul_add(
        f32::from_bits(0x35bf_be8e),
        (-n).mul_add(f32::from_bits(0x3f31_7200), x),
    );
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = f32::from_bits(0x3c07_2010)
        .mul_add(b, f32::from_bits(0x3d2b_9f17))
        .mul_add(
            u,
            f32::from_bits(0x3e2a_af33).mul_add(b, f32::from_bits(0x3eff_fedb)),
        )
        .mul_add(u, f32::from_bits(0x3f7f_fff6) * b);
    if n.abs() <= 126.0 {
        return j.mul_add(k, k);
    }
    let g = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    if n.abs() > 192.0 {
        s1 * s1
    } else {
        s2.mul_add(j, s2) * s1
    }
}

/// Opt-in, model-recipe-specific source-compatible SiLU on f32. This matches
/// GGML's vector approximation, rather than silently redefining generic SiLU.
pub fn ggml_source_silu(x: f32) -> f32 {
    x / (1.0 + ggml_source_exp(0.0 - x))
}

#[cfg(test)]
mod ggml_source_silu_tests {
    use super::*;
    #[test]
    fn source_approx_is_bounded_and_retains_general_silu_behavior() {
        for integer in -100..=100 {
            let x = integer as f32 / 10.0;
            let y = ggml_source_silu(x);
            assert!(y.is_finite());
            assert!(
                (y - silu(x)).abs() <= 1e-5,
                "GGML source approximation unexpectedly divergent at {x}"
            );
        }
        assert_eq!(ggml_source_silu(0.0), 0.0);
    }
}
