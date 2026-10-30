//! Single-row eager BF16 attention using the existing CPU numerical primitives.
//! Pinned HF source: modeling_qwen3_5.py 762feb6c…12c3b, lines143–214/674–826.
//! QK and PV retain distinct original reductions. Operator checks do not grant
//! learned-model parity; libm rotary still needs its own source comparison.
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    ActivationPrecision, TorchCpuAttentionInputs, TorchCpuAttentionValues,
};
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

fn check(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("CPU eager attention cancelled".into()));
    }
    Ok(())
}
fn rounded(x: f32) -> Result<f32> {
    if !x.is_finite() {
        return Err(Error::Malformed(
            "nonfinite CPU attention intermediate".into(),
        ));
    }
    let bits = x.to_bits();
    let out = f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
    if !out.is_finite() {
        return Err(Error::Malformed("CPU attention BF16 overflow".into()));
    }
    Ok(out)
}
fn vector(count: usize) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::Limit("CPU attention bounded output".into()))?;
    if v.capacity() != count {
        return Err(Error::Limit("CPU attention output capacity".into()));
    }
    Ok(v)
}
fn encode(row: &[f32], bytes: &mut Vec<u8>) {
    bytes.clear();
    for x in row {
        bytes.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
    }
}

/// Time-major KV with genuine grouped-head repetition. All returned buffers are
/// private; the model adapter reserves their combined capacity before this call.
pub fn eager_bf16(
    source: SourceIdentity,
    input: TorchCpuAttentionInputs<'_, [f32]>,
    cancelled: Option<&AtomicBool>,
) -> Result<TorchCpuAttentionValues<Vec<f32>>> {
    check(cancelled)?;
    let g = input.geometry;
    let (q, kv, tokens) = g.validate(input.position)?;
    if !source.is_bound()
        || input.query.len() != q
        || input.gate.len() != q
        || input.key.len() != kv
        || input.value.len() != kv
        || input.prior.is_some() != (input.position > 0)
    {
        return Err(Error::State(
            "CPU attention exact source/current/prior geometry".into(),
        ));
    }
    for values in [input.query, input.key, input.value, input.gate] {
        ActivationPrecision::Bf16WidenedF32.validate_values(values)?;
    }
    if let Some((k, v)) = input.prior {
        if k.len() != input.position * kv || v.len() != k.len() {
            return Err(Error::State(
                "CPU attention complete immutable prior".into(),
            ));
        }
        ActivationPrecision::Bf16WidenedF32.validate_values(k)?;
        ActivationPrecision::Bf16WidenedF32.validate_values(v)?;
    }
    let rotate = |values: &[f32]| -> Result<Vec<f32>> {
        let mut out = vector(values.len())?;
        out.extend_from_slice(values);
        for row in out.chunks_exact_mut(g.width) {
            check(cancelled)?;
            for i in 0..g.rotary / 2 {
                let angle =
                    (1.0 / g.theta.powf((2 * i) as f32 / g.rotary as f32)) * input.position as f32;
                let c = rounded(angle.cos())?;
                let s = rounded(angle.sin())?;
                let a = row[i];
                let b = row[i + g.rotary / 2];
                row[i] = rounded(rounded(a * c)? + rounded(-b * s)?)?;
                row[i + g.rotary / 2] = rounded(rounded(b * c)? + rounded(a * s)?)?;
            }
        }
        Ok(out)
    };
    let rotated_query = rotate(input.query)?;
    let rotated_key = rotate(input.key)?;
    let mut keys = vector(tokens * kv)?;
    let mut values = vector(tokens * kv)?;
    if let Some((k, v)) = input.prior {
        keys.extend_from_slice(k);
        values.extend_from_slice(v);
    }
    keys.extend_from_slice(&rotated_key);
    values.extend_from_slice(input.value);
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(g.width.max(tokens) * 2)
        .map_err(|_| Error::Limit("CPU attention dot scratch".into()))?;
    if packed.capacity() != g.width.max(tokens) * 2 {
        return Err(Error::Limit("CPU dot scratch capacity".into()));
    }
    let mut dots = vector(g.query_heads * tokens)?;
    let mut scores = vector(g.query_heads * tokens)?;
    for h in 0..g.query_heads {
        check(cancelled)?;
        let group = h / (g.query_heads / g.kv_heads);
        for t in 0..tokens {
            let begin = t * kv + group * g.width;
            encode(&keys[begin..begin + g.width], &mut packed);
            let dot = rounded(crate::source::dot_bf16_cpu_avx2(
                &packed,
                &rotated_query[h * g.width..(h + 1) * g.width],
            ))?;
            dots.push(dot);
            scores.push(rounded(dot * 0.0625)?);
        }
    }
    let mut probabilities =
        super::transcendentals::softmax_f32_lastdim_avx2(source, &scores, tokens)?;
    for v in &mut probabilities {
        *v = rounded(*v)?;
    }
    let mut context = vector(q)?;
    let mut column = [0.0; 64];
    let pv = super::pv::Bf16PvGeometry {
        query_rows: 1,
        contraction: tokens,
        output_width: g.width,
    };
    for h in 0..g.query_heads {
        check(cancelled)?;
        let group = h / (g.query_heads / g.kv_heads);
        for d in 0..g.width {
            for (t, value) in column[..tokens].iter_mut().enumerate() {
                *value = values[t * kv + group * g.width + d];
            }
            context.push(rounded(super::pv::bf16_pv_f32_accumulator(
                source,
                &probabilities[h * tokens..(h + 1) * tokens],
                &column[..tokens],
                pv,
            )?)?);
        }
    }
    let mut gated =
        super::transcendentals::sigmoid_bf16_widened_f32_contiguous_span(source, input.gate)?;
    for (out, x) in gated.iter_mut().zip(&context) {
        *out = rounded(*out * x)?;
    }
    check(cancelled)?;
    Ok(TorchCpuAttentionValues {
        rotated_query,
        rotated_key,
        keys,
        values,
        dots,
        scores,
        probabilities,
        context,
        gated,
    })
}

#[cfg(test)]
mod tests;
