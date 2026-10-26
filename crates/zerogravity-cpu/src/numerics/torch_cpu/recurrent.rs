//! Pinned Torch CPU AVX2 cached GDN core, no model/session/initial-chunk path.
//!
//! Transformers modeling_qwen3_5.py SHA256 762feb6c7426a7f15b5bf830df54c0743
//! 8bf9e7c27b8cdb23179045920412c3b, torch_recurrent_gated_delta_rule, with
//! Torch cf30153c4c131c8164ee7798e5022d810682e2cb CPU reductions/division.
//! Decay is an explicit already-computed exp(g) boundary; coefficients and
//! transcendental parity remain separate. GGUF/FLA/default math is unchanged.

use super::reduction::{l2_norm_128, sum_key_128};
use crate::accelerator::SourceIdentity;
use crate::{Error, Result};

const WIDTH: usize = 128;
const MAX_ELEMENTS: usize = 1_048_576;

// Fallible collect has no exact lower size hint. Keep the caller's admitted
// output extent for every head count, including non-power-of-two H48.
pub(super) fn collect_exact_output(
    count: usize,
    values: impl IntoIterator<Item = Result<f32>>,
) -> Result<Vec<f32>> {
    if count == 0 || count > MAX_ELEMENTS {
        return Err(Error::Limit("CPU GDN output element bound".into()));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| Error::Limit("CPU GDN admitted output allocation".into()))?;
    if out.capacity() != count {
        return Err(Error::Limit("CPU GDN exact output capacity".into()));
    }
    for value in values {
        if out.len() == count {
            return Err(Error::Malformed(
                "CPU GDN output exceeds declared extent".into(),
            ));
        }
        out.push(value?);
    }
    if out.len() != count {
        return Err(Error::Malformed("CPU GDN incomplete output extent".into()));
    }
    Ok(out)
}

/// BF16 inputs widened exactly to F32; one cached token, batch1, repeated heads.
/// Query/key/value are `[head,128]`, beta/decay `[head]`, and previous state is
/// FP32 `[head,key128,value128]`. No implicit head repeat/layout conversion.
#[derive(Clone, Copy)]
pub struct CachedGdn128Inputs<'a> {
    pub source: SourceIdentity,
    pub heads: usize,
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub beta: &'a [f32],
    /// FP32 exp(g), supplied from the independently qualified coefficient path.
    pub decay: &'a [f32],
    pub previous_state: &'a [f32],
}

pub struct CachedGdn128Completed {
    /// FP32 `[head,key128,value128]`, not the canonical value-major transport.
    pub state: Vec<f32>,
    pub core_f32: Vec<f32>,
    /// Final BF16 values carried as exactly widened F32 words.
    pub output_bf16: Vec<f32>,
}

fn finite(input: &[f32]) -> Result<()> {
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite pinned CPU cached GDN value".into(),
        ));
    }
    Ok(())
}
fn bf16_input(input: &[f32]) -> Result<()> {
    finite(input)?;
    if input.iter().any(|value| value.to_bits() & 0xffff != 0) {
        return Err(Error::Malformed(
            "cached GDN input is not exact widened BF16".into(),
        ));
    }
    Ok(())
}
fn validate(input: &CachedGdn128Inputs<'_>) -> Result<()> {
    let state = input
        .heads
        .checked_mul(WIDTH)
        .and_then(|n| n.checked_mul(WIDTH))
        .filter(|&n| input.heads > 0 && n <= MAX_ELEMENTS)
        .ok_or_else(|| Error::Limit("pinned CPU cached GDN head/element bound".into()))?;
    if !input.source.is_bound()
        || input.query.len() != input.heads * WIDTH
        || input.key.len() != input.query.len()
        || input.value.len() != input.query.len()
        || input.beta.len() != input.heads
        || input.decay.len() != input.heads
        || input.previous_state.len() != state
    {
        return Err(Error::Malformed(
            "pinned CPU cached GDN source/layout".into(),
        ));
    }
    for values in [input.query, input.key, input.value, input.beta] {
        bf16_input(values)?;
    }
    finite(input.previous_state)?;
    finite(input.decay)?;
    if input
        .beta
        .iter()
        .chain(input.decay)
        .any(|v| !(0.0..=1.0).contains(v))
    {
        return Err(Error::Malformed(
            "pinned CPU cached GDN beta/decay range".into(),
        ));
    }
    Ok(())
}

/// Cached source uses direct F32 division, not chunk's precomputed multiply.
/// This difference is observable in783/2048 actual layer0 second-token words.
pub fn scale_cached_queries_128(source: SourceIdentity, normalized: &[f32]) -> Result<Vec<f32>> {
    if !source.is_bound() || normalized.is_empty() || !normalized.len().is_multiple_of(WIDTH) {
        return Err(Error::Malformed(
            "pinned CPU cached query source/layout".into(),
        ));
    }
    if normalized.len() > MAX_ELEMENTS / WIDTH {
        return Err(Error::Limit("pinned CPU cached query head bound".into()));
    }
    finite(normalized)?;
    // Python sqrt(128) scalar becomes F32 for TensorIterator F32 div_true.
    let divisor = (WIDTH as f32).sqrt();
    let scaled = normalized.iter().map(|&q| q / divisor).collect::<Vec<_>>();
    finite(&scaled)?;
    Ok(scaled)
}

fn bf16_output(value: f32) -> Result<f32> {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    let output = f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000);
    if !output.is_finite() {
        return Err(Error::Limit("pinned CPU cached GDN BF16 overflow".into()));
    }
    Ok(output)
}

/// One pure cached recurrence with private outputs. A zero previous-state
/// tensor still selects cached arithmetic; it never qualifies initial chunk64.
/// The caller retains the immutable source, phase and publication authority.
pub fn cached_step_128(input: CachedGdn128Inputs<'_>) -> Result<CachedGdn128Completed> {
    validate(&input)?;
    let q = l2_norm_128(input.source, input.query, 1e-6)?;
    let q = scale_cached_queries_128(input.source, &q)?;
    let k = l2_norm_128(input.source, input.key, 1e-6)?;
    let decayed = input
        .previous_state
        .chunks_exact(WIDTH * WIDTH)
        .zip(input.decay)
        .flat_map(|(state, &decay)| state.iter().map(move |&value| value * decay))
        .collect::<Vec<_>>();
    finite(&decayed)?;
    // Materialize every product before the source key-axis reduction.
    let products = decayed
        .chunks_exact(WIDTH * WIDTH)
        .zip(k.chunks_exact(WIDTH))
        .flat_map(|(state, key)| {
            state
                .chunks_exact(WIDTH)
                .zip(key)
                .flat_map(|(values, &key)| values.iter().map(move |&value| value * key))
        })
        .collect::<Vec<_>>();
    let memory = sum_key_128(input.source, &products, input.heads)?;
    drop(products);
    // Source subtraction, beta multiplication, update multiplication and add
    // are separate F32 tensors; never substitute the GGUF FMA update.
    let difference = input
        .value
        .iter()
        .zip(&memory)
        .map(|(&v, &m)| v - m)
        .collect::<Vec<_>>();
    finite(&difference)?;
    let delta = difference
        .chunks_exact(WIDTH)
        .zip(input.beta)
        .flat_map(|(row, &beta)| row.iter().map(move |&value| value * beta))
        .collect::<Vec<_>>();
    finite(&delta)?;
    let update = k
        .chunks_exact(WIDTH)
        .zip(delta.chunks_exact(WIDTH))
        .flat_map(|(key, delta)| {
            key.iter()
                .flat_map(move |&key| delta.iter().map(move |&d| key * d))
        })
        .collect::<Vec<_>>();
    finite(&update)?;
    let state = decayed
        .iter()
        .zip(&update)
        .map(|(&s, &u)| s + u)
        .collect::<Vec<_>>();
    finite(&state)?;
    drop(decayed);
    drop(update);
    let products = state
        .chunks_exact(WIDTH * WIDTH)
        .zip(q.chunks_exact(WIDTH))
        .flat_map(|(state, query)| {
            state
                .chunks_exact(WIDTH)
                .zip(query)
                .flat_map(|(values, &q)| values.iter().map(move |&value| value * q))
        })
        .collect::<Vec<_>>();
    let core_f32 = sum_key_128(input.source, &products, input.heads)?;
    let output_bf16 = collect_exact_output(
        input.heads * WIDTH,
        core_f32.iter().map(|&v| bf16_output(v)),
    )?;
    Ok(CachedGdn128Completed {
        state,
        core_f32,
        output_bf16,
    })
}

#[cfg(test)]
#[path = "recurrent_tests.rs"]
mod tests;
