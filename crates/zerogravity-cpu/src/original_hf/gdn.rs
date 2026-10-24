//! Connected diagnostic CPU GDN. Cached core reuses the pinned recurrence.
//! Initial-one contraction has bounded original evidence; standalone exp and
//! complete model/cached arithmetic remain unqualified.
use super::{conv, norm, recurrent, reduction, transcendentals};
use crate::{Error, Result};
use zerogravity_core::{
    accelerator::SourceIdentity, accelerator::program::TorchCpuGdnPhase, quant::QuantKind,
    source::SourceTensorRef,
};

pub struct GdnParameters<'a> {
    pub convolution: SourceTensorRef<'a>,
    pub convolution_name: &'a str,
    pub a_log: SourceTensorRef<'a>,
    pub dt_bias: SourceTensorRef<'a>,
    pub norm: SourceTensorRef<'a>,
}
pub struct GdnInputs<'a> {
    pub source: SourceIdentity,
    pub key_heads: usize,
    pub value_heads: usize,
    pub qkv: &'a [f32],
    pub gate: &'a [f32],
    pub alpha: &'a [f32],
    pub beta: &'a [f32],
    pub history: &'a [f32],
    pub state: &'a [f32],
    pub phase: TorchCpuGdnPhase,
}
pub struct GdnCompleted {
    pub convolution: Vec<f32>,
    pub activated: Vec<f32>,
    pub history: Vec<f32>,
    pub state: Vec<f32>,
    pub core: Vec<f32>,
    pub gated: Vec<f32>,
    pub beta: Vec<f32>,
    pub log_decay: Vec<f32>,
    pub decay: Vec<f32>,
}
/// Borrowed values from the existing initial-one computation, never operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialCoreStage {
    Convolution,
    Activated,
    Query,
    Key,
    NormalizedQuery,
    NormalizedKey,
    ScaledQuery,
    QueryKey,
    BetaValue,
    CoreF32,
}
pub type InitialCoreObserver<'a> = dyn FnMut(InitialCoreStage, &[f32]) -> Result<()> + 'a;
fn observe(
    observer: &mut Option<&mut InitialCoreObserver<'_>>,
    stage: InitialCoreStage,
    values: &[f32],
) -> Result<()> {
    if let Some(observer) = observer {
        observer(stage, values)?;
    }
    Ok(())
}
fn round(v: f32) -> Result<f32> {
    let b = v.to_bits();
    let out = f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000);
    if !v.is_finite() || !out.is_finite() {
        return Err(Error::Malformed("diagnostic GDN BF16 overflow".into()));
    }
    Ok(out)
}
fn parameter(t: SourceTensorRef<'_>, elements: usize) -> Result<Vec<f32>> {
    t.validate()?;
    if t.shape != [elements as u64] || !matches!(t.kind, QuantKind::Bf16 | QuantKind::F32) {
        return Err(Error::Malformed(
            "diagnostic GDN actual parameter shape/dtype".into(),
        ));
    }
    let values = match t.kind {
        QuantKind::Bf16 => t
            .packed_bytes
            .chunks_exact(2)
            .map(|w| f32::from_bits(u32::from(u16::from_le_bytes(w.try_into().unwrap())) << 16))
            .collect::<Vec<_>>(),
        _ => t
            .packed_bytes
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
            .collect(),
    };
    if values.len() != elements || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Malformed(
            "diagnostic GDN parameter extent/finite".into(),
        ));
    }
    Ok(values)
}
/// Pure private outputs; caller must admit aggregate storage and retain source.
/// An initial-one call is not a chunk64 call, even with the same zero state.
pub fn step(input: GdnInputs<'_>, p: GdnParameters<'_>) -> Result<GdnCompleted> {
    step_observed(input, p, None)
}
/// Same primitive; caller admits any copies made by the borrowed observer.
/// Cached diagnostics retain the unchanged standalone-exp refusal boundary.
pub fn step_observed(
    input: GdnInputs<'_>,
    p: GdnParameters<'_>,
    observer: Option<&mut InitialCoreObserver<'_>>,
) -> Result<GdnCompleted> {
    step_observed_with_exp_profile(
        input,
        p,
        transcendentals::StandaloneExpProfile::ScalarLibmDiagnostic,
        observer,
    )
}
/// Same connected source/state evaluator, explicit standalone-exp arithmetic.
/// A normal-range profile does not grant whole-model numerical eligibility.
pub fn step_observed_with_exp_profile(
    input: GdnInputs<'_>,
    p: GdnParameters<'_>,
    exp_profile: transcendentals::StandaloneExpProfile,
    mut observer: Option<&mut InitialCoreObserver<'_>>,
) -> Result<GdnCompleted> {
    if observer.is_some() && input.phase != TorchCpuGdnPhase::InitialOne {
        return Err(Error::Unsupported(
            "initial core observer requires actual empty initial phase".into(),
        ));
    }
    let kh = input.key_heads;
    let vh = input.value_heads;
    if !input.source.is_bound()
        || kh == 0
        || kh > 16
        || vh == 0
        || vh > 48
        || !vh.is_multiple_of(kh)
        || !vh.is_multiple_of(16)
        || input.qkv.len() != (2 * kh + vh) * 128
        || input.gate.len() != vh * 128
        || input.alpha.len() != vh
        || input.beta.len() != vh
        || input.history.len() != input.qkv.len() * 3
        || input.state.len() != vh * 128 * 128
    {
        return Err(Error::Malformed(
            "diagnostic GDN full grouped source/state/input extent".into(),
        ));
    }
    for values in [
        input.qkv,
        input.gate,
        input.alpha,
        input.beta,
        input.history,
    ] {
        if values
            .iter()
            .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
        {
            return Err(Error::Malformed("diagnostic GDN exact BF16 inputs".into()));
        }
    }
    if input.state.iter().any(|v| !v.is_finite())
        || (input.phase == TorchCpuGdnPhase::InitialOne
            && input
                .state
                .iter()
                .chain(input.history)
                .any(|v| v.to_bits() != 0))
    {
        return Err(Error::State(
            "diagnostic initial GDN requires genuine empty positive-zero state".into(),
        ));
    }
    let raw = conv::causal_step_bf16_4(conv::CausalConv4Inputs {
        source: input.source,
        tensor_name: p.convolution_name,
        weight: p.convolution,
        activation: input.qkv,
        history: input.history,
    })?;
    observe(&mut observer, InitialCoreStage::Convolution, &raw.raw_bf16)?;
    let activated = recurrent::collect_exact_output(
        input.qkv.len(),
        transcendentals::silu_bf16_widened_f32_contiguous16(input.source, &raw.raw_bf16)?
            .into_iter()
            .map(round),
    )?;
    observe(&mut observer, InitialCoreStage::Activated, &activated)?;
    let a = parameter(p.a_log, vh)?;
    let dt = parameter(p.dt_bias, vh)?;
    let biased = input
        .alpha
        .iter()
        .zip(dt)
        .map(|(x, b)| x + b)
        .collect::<Vec<_>>();
    let softplus =
        transcendentals::softplus_beta1_threshold20_f32_contiguous16(input.source, &biased)?;
    // BOTH original standalone calls use the selected arithmetic profile.
    // Source metadata, never head count/model name, selects that profile.
    let a_exp = transcendentals::exp_f32_contiguous16_profiled(input.source, &a, exp_profile)?;
    let log_decay = a_exp
        .iter()
        .zip(softplus)
        .map(|(a, s)| -a * s)
        .collect::<Vec<_>>();
    let decay =
        transcendentals::exp_f32_contiguous16_profiled(input.source, &log_decay, exp_profile)?;
    if decay
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err(Error::Malformed("diagnostic GDN decay overflow".into()));
    }
    let beta = transcendentals::sigmoid_bf16_widened_f32_contiguous_span(input.source, input.beta)?;
    let repeat = |offset: usize| {
        (0..vh)
            .flat_map(|h| {
                let start = offset + (h / (vh / kh)) * 128;
                activated[start..start + 128].iter().copied()
            })
            .collect::<Vec<_>>()
    };
    let q = repeat(0);
    let k = repeat(kh * 128);
    let v = &activated[kh * 256..];
    observe(&mut observer, InitialCoreStage::Query, &q)?;
    observe(&mut observer, InitialCoreStage::Key, &k)?;
    let (state, core) = if input.phase == TorchCpuGdnPhase::CachedOne {
        let done = recurrent::cached_step_128(recurrent::CachedGdn128Inputs {
            source: input.source,
            heads: vh,
            query: &q,
            key: &k,
            value: v,
            beta: &beta,
            decay: &decay,
            previous_state: input.state,
        })?;
        (done.state, done.output_bf16)
    } else {
        // Original pads time to64 before SGEMM. Its active Initial1 QK uses
        // FP32 FMA, not SumKernel over separately rounded products.
        let q = reduction::l2_norm_128(input.source, &q, 1e-6)?;
        let k = reduction::l2_norm_128(input.source, &k, 1e-6)?;
        observe(&mut observer, InitialCoreStage::NormalizedQuery, &q)?;
        observe(&mut observer, InitialCoreStage::NormalizedKey, &k)?;
        let scale = (128.0f64).powf(-0.5) as f32;
        let scaled = recurrent::collect_exact_output(vh * 128, q.iter().map(|q| Ok(q * scale)))?;
        observe(&mut observer, InitialCoreStage::ScaledQuery, &scaled)?;
        let dots = reduction::padded64_qk_fma128(input.source, &scaled, &k)?;
        observe(&mut observer, InitialCoreStage::QueryKey, &dots)?;
        let mut observed_core = if observer.is_some() {
            Some(vec![0.; vh * 128])
        } else {
            None
        };
        let mut observed_beta_v = if observer.is_some() {
            Some(vec![0.; vh * 128])
        } else {
            None
        };
        let mut state = vec![0.; vh * 128 * 128];
        let mut out = vec![0.; vh * 128];
        for h in 0..vh {
            for j in 0..128 {
                let vb = v[h * 128 + j] * beta[h];
                // Original adds the empty-state inter-chunk contribution.
                let raw = 0.0 + dots[h] * vb;
                if let Some(v) = observed_core.as_mut() {
                    v[h * 128 + j] = raw;
                }
                if let Some(v) = observed_beta_v.as_mut() {
                    v[h * 128 + j] = vb;
                }
                out[h * 128 + j] = round(raw)?;
                for i in 0..128 {
                    state[(h * 128 + i) * 128 + j] = k[h * 128 + i] * vb;
                }
            }
        }
        if let Some(v) = observed_beta_v {
            observe(&mut observer, InitialCoreStage::BetaValue, &v)?;
        }
        if let Some(v) = observed_core {
            observe(&mut observer, InitialCoreStage::CoreF32, &v)?;
        }
        (state, out)
    };
    let weights = parameter(p.norm, 128)?;
    let bytes = weights
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let normalized = norm::head_rms_128(
        input.source,
        SourceTensorRef {
            kind: QuantKind::F32,
            shape: &[128],
            packed_bytes: &bytes,
        },
        &core,
        vh,
    )?;
    let gate = transcendentals::silu_bf16_widened_f32_contiguous16(input.source, input.gate)?;
    let gated = recurrent::collect_exact_output(
        vh * 128,
        normalized.weighted_fp32().iter().zip(gate).map(|(x, g)| {
            let x = if p.norm.kind == QuantKind::Bf16 {
                round(*x)?
            } else {
                *x
            };
            round(x * g)
        }),
    )?;
    Ok(GdnCompleted {
        convolution: raw.raw_bf16,
        activated,
        history: raw.next_history,
        state,
        core,
        gated,
        beta,
        log_decay,
        decay,
    })
}

#[cfg(test)]
mod tests;
