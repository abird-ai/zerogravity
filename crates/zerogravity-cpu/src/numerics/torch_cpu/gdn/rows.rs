//! Source-call GDN frontend over the shared Conv4, chunk and cached primitives.
//! Call phase is explicit. Returned cache/state/output stay private to the caller.
use super::*;
use crate::accelerator::program::InvocationPhase;

#[cfg(test)]
mod tests;

pub struct Inputs<'a> {
    pub source: SourceIdentity,
    pub mode: InvocationPhase,
    pub rows: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub qkv: &'a [f32],
    pub gate: &'a [f32],
    pub alpha: &'a [f32],
    pub beta: &'a [f32],
    pub previous_conv: Option<&'a [f32]>,
    pub previous_state: Option<&'a [f32]>,
}

pub struct Completed {
    pub convolution: Vec<f32>,
    pub activated: Vec<f32>,
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
    pub core: Vec<f32>,
    pub gated: Vec<f32>,
    pub beta: Vec<f32>,
    pub log_decay: Vec<f32>,
}

pub fn forward(
    input: Inputs<'_>,
    p: GdnParameters<'_>,
    exp_profile: transcendentals::StandaloneExpProfile,
) -> Result<Completed> {
    let Inputs {
        source,
        mode,
        rows,
        key_heads: kh,
        value_heads: vh,
        qkv,
        gate,
        alpha,
        beta,
        previous_conv,
        previous_state,
    } = input;
    if !source.is_bound()
        || !(1..=64).contains(&rows)
        || !(1..=16).contains(&kh)
        || !(1..=48).contains(&vh)
        || !vh.is_multiple_of(kh)
        || !vh.is_multiple_of(16)
        || previous_conv.is_some() != previous_state.is_some()
        || mode == InvocationPhase::Decode && (rows != 1 || previous_state.is_none())
        || mode == InvocationPhase::Prefill && rows == 1 && previous_state.is_some()
    {
        return Err(Error::Malformed(
            "source GDN call geometry/phase/prior".into(),
        ));
    }
    let channels = (2 * kh + vh) * 128;
    if qkv.len() != rows * channels
        || gate.len() != rows * vh * 128
        || alpha.len() != rows * vh
        || beta.len() != rows * vh
        || previous_conv.is_some_and(|v| v.len() != channels * 4)
        || previous_state
            .is_some_and(|v| v.len() != vh * 128 * 128 || v.iter().any(|v| !v.is_finite()))
    {
        return Err(Error::Malformed(
            "source GDN complete call/cache extents".into(),
        ));
    }
    for input in [qkv, gate, alpha, beta] {
        if input
            .iter()
            .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
        {
            return Err(Error::Malformed("source GDN exact BF16 call inputs".into()));
        }
    }
    let a = parameter(p.a_log, vh)?;
    let dt = parameter(p.dt_bias, vh)?;
    let norm = parameter(p.norm, 128)?;
    let conv = conv::rows::causal_bf16(conv::rows::Inputs {
        source,
        tensor_name: p.convolution_name,
        weight: p.convolution,
        activation: qkv,
        rows,
        previous: previous_conv,
    })?;
    let activated = transcendentals::silu_bf16_widened_f32_contiguous16(source, &conv.raw_bf16)?
        .into_iter()
        .map(round)
        .collect::<Result<Vec<_>>>()?;
    let biased = alpha
        .iter()
        .enumerate()
        .map(|(i, x)| x + dt[i % vh])
        .collect::<Vec<_>>();
    let softplus = transcendentals::softplus_beta1_threshold20_f32_contiguous16(source, &biased)?;
    let a_exp = transcendentals::exp_f32_contiguous16_profiled(source, &a, exp_profile)?;
    let log_decay = softplus
        .iter()
        .enumerate()
        .map(|(i, s)| -a_exp[i % vh] * s)
        .collect::<Vec<_>>();
    let beta = transcendentals::sigmoid_bf16_widened_f32_contiguous_span(source, beta)?;
    let grouped = |offset: usize| {
        let mut out = Vec::with_capacity(rows * vh * 128);
        for t in 0..rows {
            for h in 0..vh {
                let begin = t * channels + offset + (h / (vh / kh)) * 128;
                out.extend_from_slice(&activated[begin..begin + 128]);
            }
        }
        out
    };
    let query = grouped(0);
    let key = grouped(kh * 128);
    let value = activated
        .chunks_exact(channels)
        .flat_map(|row| row[kh * 256..].iter().copied())
        .collect::<Vec<_>>();
    let (recurrent_state, core) = match mode {
        InvocationPhase::Prefill => {
            let output = chunk::core(chunk::ChunkInputs {
                source,
                rows,
                heads: vh,
                query: &query,
                key: &key,
                value: &value,
                beta: &beta,
                log_decay: &log_decay,
                previous_state,
            })?;
            (output.state, output.output_bf16)
        }
        InvocationPhase::Decode => {
            let decay =
                transcendentals::exp_f32_contiguous16_profiled(source, &log_decay, exp_profile)?;
            let output = recurrent::cached_step_128(recurrent::CachedGdn128Inputs {
                source,
                heads: vh,
                query: &query,
                key: &key,
                value: &value,
                beta: &beta,
                decay: &decay,
                previous_state: previous_state
                    .ok_or_else(|| Error::State("source cached GDN prior absent".into()))?,
            })?;
            (output.state, output.output_bf16)
        }
    };
    let norm_bytes = norm
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let gates = transcendentals::silu_bf16_widened_f32_contiguous16(source, gate)?;
    let mut gated = Vec::with_capacity(core.len());
    for (core_row, gate_row) in core
        .chunks_exact(vh * 128)
        .zip(gates.chunks_exact(vh * 128))
    {
        let normalized = norm::head_rms_128(
            source,
            SourceTensorRef {
                kind: StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32),
                shape: &[128],
                packed_bytes: &norm_bytes,
            },
            core_row,
            vh,
        )?;
        for (n, g) in normalized.weighted_fp32().iter().zip(gate_row) {
            let n = if p.norm.kind
                == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16)
            {
                round(*n)?
            } else {
                *n
            };
            gated.push(round(n * g)?);
        }
    }
    Ok(Completed {
        convolution: conv.raw_bf16,
        activated,
        conv_state: conv.state,
        recurrent_state,
        core,
        gated,
        beta,
        log_decay,
    })
}
