//! Full-call causal attention over already rotated BF16 source rows.
//! Caller owns KV append, admission and publication; masked columns remain in
//! the original softmax/PV contraction. No repeated single-token approximation.
use super::{check, rounded, vector};
use crate::{Error, Result, accelerator::SourceIdentity};
use std::sync::atomic::AtomicBool;
use zerogravity_core::numerics::ActivationPrecision;

pub struct Inputs<'a> {
    pub source: SourceIdentity,
    /// Token-major `[row,query_head,width]`.
    pub query: &'a [f32],
    /// Complete time-major KV, including every current-call row.
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub rows: usize,
    pub prior_tokens: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub width: usize,
}

pub struct Completed {
    /// Head-major `[query_head,row,full_call_contraction]`.
    pub dots: Vec<f32>,
    pub scores: Vec<f32>,
    pub probabilities: Vec<f32>,
    /// Token-major `[row,query_head,width]`.
    pub context: Vec<f32>,
}

pub fn causal_bf16(input: Inputs<'_>, cancelled: Option<&AtomicBool>) -> Result<Completed> {
    check(cancelled)?;
    let Inputs {
        source,
        query,
        keys,
        values,
        rows,
        prior_tokens,
        query_heads,
        kv_heads,
        width,
    } = input;
    let tokens = prior_tokens
        .checked_add(rows)
        .filter(|n| *n <= 32_768)
        .ok_or_else(|| Error::Limit("source attention complete contraction".into()))?;
    if !source.is_bound()
        || !(1..=64).contains(&rows)
        || !(1..=24).contains(&query_heads)
        || !(1..=4).contains(&kv_heads)
        || !query_heads.is_multiple_of(kv_heads)
        || width != 256
        || query.len() != rows * query_heads * width
        || keys.len() != tokens * kv_heads * width
        || values.len() != keys.len()
    {
        return Err(Error::Malformed(
            "source attention complete row/head/KV geometry".into(),
        ));
    }
    for values in [query, keys, values] {
        ActivationPrecision::Bf16WidenedF32.validate_values(values)?;
    }
    let score_count = query_heads * rows * tokens;
    let mut dots = vector(score_count)?;
    let mut scores = vector(score_count)?;
    let mut probabilities = vector(score_count)?;
    let mut context = vector(query.len())?;
    context.resize(query.len(), 0.0);
    let mut packed = vec![0u8; width * 2];
    let mut column = vec![0.0f32; tokens];
    for head in 0..query_heads {
        let group = head / (query_heads / kv_heads);
        for row in 0..rows {
            check(cancelled)?;
            let start = scores.len();
            let q =
                &query[(row * query_heads + head) * width..(row * query_heads + head + 1) * width];
            for token in 0..tokens {
                let begin = (token * kv_heads + group) * width;
                for (dst, value) in packed.chunks_exact_mut(2).zip(&keys[begin..begin + width]) {
                    dst.copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                }
                let dot = rounded(super::super::projection::dot_bf16_cpu_avx2(&packed, q))?;
                dots.push(dot);
                scores.push(if token <= prior_tokens + row {
                    rounded(dot * 0.0625)?
                } else {
                    f32::NEG_INFINITY
                });
            }
            let mut p = super::super::transcendentals::softmax_f32_lastdim_avx2(
                source,
                &scores[start..],
                tokens,
            )?;
            for value in &mut p {
                *value = rounded(*value)?;
            }
            let geometry = super::super::pv::Bf16PvGeometry {
                query_rows: rows,
                contraction: tokens,
                output_width: width,
            };
            for d in 0..width {
                if d.is_multiple_of(32) {
                    check(cancelled)?;
                }
                for (t, v) in column.iter_mut().enumerate() {
                    *v = values[(t * kv_heads + group) * width + d];
                }
                context[(row * query_heads + head) * width + d] = rounded(
                    super::super::pv::bf16_pv_f32_accumulator(source, &p, &column, geometry)?,
                )?;
            }
            probabilities.extend(p);
        }
    }
    check(cancelled)?;
    Ok(Completed {
        dots,
        scores,
        probabilities,
        context,
    })
}
