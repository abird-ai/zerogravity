//! Noncausal F32 encoder primitives. Inputs are independent unpadded
//! sequences; no decoder state, implicit mask, or model-specific weights.
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

fn shape(message: &'static str) -> Error {
    Error::InvalidInput(message.into())
}
fn finite(values: &[f32]) -> bool {
    values.iter().all(|x| x.is_finite())
}
pub fn check_cancel(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("encoder cancelled".into()));
    }
    Ok(())
}

/// Per-row affine LayerNorm, not RMSNorm. F64 reference reductions;
/// the output is the model's F32 activation.
pub fn layer_norm(row: &mut [f32], weight: &[f32], bias: &[f32], epsilon: f32) -> Result<()> {
    if row.is_empty()
        || row.len() != weight.len()
        || row.len() != bias.len()
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || !finite(row)
        || !finite(weight)
        || !finite(bias)
    {
        return Err(shape("affine LayerNorm geometry/values"));
    }
    let n = row.len() as f64;
    let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let variance = row
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / n;
    let inv = (variance + f64::from(epsilon)).sqrt().recip();
    for ((value, &w), &b) in row.iter_mut().zip(weight).zip(bias) {
        *value = ((f64::from(*value) - mean) * inv * f64::from(w) + f64::from(b)) as f32;
    }
    if !finite(row) {
        return Err(shape("nonfinite LayerNorm output"));
    }
    Ok(())
}

/// Exact-erf GELU (BERT), deliberately distinct from the tanh approximation.
pub fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

/// Scaled dot-product self-attention over all positions, with heads
/// interleaved within each token row. No quadratic attention matrix: one
/// reusable score row and the output backing. Caller admits this scratch.
pub fn bidirectional_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    hidden: usize,
    heads: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<f32>> {
    if hidden == 0
        || heads == 0
        || !hidden.is_multiple_of(heads)
        || queries.is_empty()
        || !queries.len().is_multiple_of(hidden)
        || keys.len() != queries.len()
        || values.len() != queries.len()
        || !finite(queries)
        || !finite(keys)
        || !finite(values)
    {
        return Err(shape("bidirectional attention geometry/values"));
    }
    check_cancel(cancelled)?;
    let tokens = queries.len() / hidden;
    let head = hidden / heads;
    let scale = (head as f64).sqrt().recip();
    let mut scores = vec![0.0f64; tokens];
    let mut output = vec![0.0f32; queries.len()];
    for (query, result) in queries
        .chunks_exact(hidden)
        .zip(output.chunks_exact_mut(hidden))
    {
        check_cancel(cancelled)?;
        for (h, (q, out)) in query
            .chunks_exact(head)
            .zip(result.chunks_exact_mut(head))
            .enumerate()
        {
            let start = h * head;
            let end = start + head;
            for (score, k) in scores.iter_mut().zip(keys.chunks_exact(hidden)) {
                *score = q
                    .iter()
                    .zip(&k[start..end])
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
                    * scale;
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for score in &mut scores {
                *score = (*score - max).exp();
                total += *score;
            }
            for (channel, out) in out.iter_mut().enumerate() {
                *out = (scores
                    .iter()
                    .zip(values.chunks_exact(hidden))
                    .map(|(&p, v)| p * f64::from(v[start + channel]))
                    .sum::<f64>()
                    / total) as f32;
            }
        }
    }
    if !finite(&output) {
        return Err(shape("nonfinite attention output"));
    }
    check_cancel(cancelled)?;
    Ok(output)
}

/// Mean pooling across all real (unpadded) rows followed by L2 normalization.
/// Use the model's normalization epsilon; no arbitrary dimension reduction.
pub fn mean_l2_normalize(rows: &[f32], width: usize, epsilon: f64) -> Result<Vec<f32>> {
    if width == 0
        || rows.is_empty()
        || !rows.len().is_multiple_of(width)
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || !finite(rows)
    {
        return Err(shape("mean/L2 pooling geometry/values"));
    }
    let count = (rows.len() / width) as f64;
    let mut mean = vec![0.0f64; width];
    for row in rows.chunks_exact(width) {
        for (a, &b) in mean.iter_mut().zip(row) {
            *a += f64::from(b);
        }
    }
    for value in &mut mean {
        *value /= count;
    }
    let norm = mean.iter().map(|x| x * x).sum::<f64>().sqrt().max(epsilon);
    Ok(mean.into_iter().map(|v| (v / norm) as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn noncausal_attention_and_pooling() {
        // Different heads and a future value. With zero queries, every
        // position receives the same average for its own head.
        let q = [0.; 6];
        let values = [1., 10., 2., 20., 9., 90.];
        let out = bidirectional_attention(&q, &q, &values, 2, 2, None).unwrap();
        assert_eq!(out, [4., 40., 4., 40., 4., 40.]);
        let pooled = mean_l2_normalize(&[1., 2., 5., 6.], 2, 1e-12).unwrap();
        assert!((pooled[0] - 0.6).abs() < 1e-6 && (pooled[1] - 0.8).abs() < 1e-6);
        assert!(bidirectional_attention(&q, &q, &values, 2, 3, None).is_err());
        assert!(mean_l2_normalize(&[f32::NAN], 1, 1e-12).is_err());
        let cancel = AtomicBool::new(true);
        assert!(bidirectional_attention(&q, &q, &values, 2, 2, Some(&cancel)).is_err());
    }
    #[test]
    fn affine_layer_norm_and_erf_gelu() {
        let mut row = [1., 3.];
        layer_norm(&mut row, &[2., 3.], &[0.5, -0.5], 1e-12).unwrap();
        assert_eq!(row, [-1.5, 2.5]);
        assert!((gelu(1.) - 0.8413448).abs() < 1e-7);
        assert!((gelu(-1.) + 0.15865525).abs() < 1e-7);
    }
}
