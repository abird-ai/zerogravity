//! Source-owned USM Conformer semantics. The shared CPU engine supplies
//! learned projection, norm, softmax and temporal convolution.
use super::{
    config::Audio,
    source::{Binder, Linear, Norm},
};
use crate::Result;
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_cpu::ops::{audio::depthwise_causal, encoder::check_cancel, silu, softmax};

const RMS_EPSILON: f32 = 1e-6;
fn clamp(x: &mut [f32], limit: f32) {
    for v in x {
        *v = v.clamp(-limit, limit);
    }
}
fn residual(rows: &mut [f32], prior: Vec<f32>, scale: f32) {
    for (v, r) in rows.iter_mut().zip(prior) {
        *v = *v * scale + r;
    }
}

struct FeedForward {
    up: Linear,
    down: Linear,
    pre: Norm,
    post: Norm,
}
impl FeedForward {
    fn bind(b: &mut Binder<'_>, root: &str, c: &Audio) -> Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            up: b.linear(
                &format!("{root}.ffw_layer_1"),
                h,
                h * 4,
                c.use_clipped_linears,
            )?,
            down: b.linear(
                &format!("{root}.ffw_layer_2"),
                h * 4,
                h,
                c.use_clipped_linears,
            )?,
            pre: b.norm(&format!("{root}.pre_layer_norm"), h, RMS_EPSILON)?,
            post: b.norm(&format!("{root}.post_layer_norm"), h, RMS_EPSILON)?,
        })
    }
    fn run(
        &self,
        s: &VerifiedTensorSource,
        c: &Audio,
        input: Vec<f32>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let mut hidden = input.clone();
        clamp(&mut hidden, c.gradient_clipping);
        self.pre.apply(&mut hidden)?;
        let mut intermediate = self.up.rows(s, &hidden, cancelled)?;
        drop(hidden);
        for v in &mut intermediate {
            *v = silu(*v);
        }
        let mut output = self.down.rows(s, &intermediate, cancelled)?;
        drop(intermediate);
        clamp(&mut output, c.gradient_clipping);
        self.post.apply(&mut output)?;
        residual(&mut output, input, c.residual_weight);
        Ok(output)
    }
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    post: Linear,
    relative_key: Linear,
    query_scale: Vec<f32>,
}
impl Attention {
    fn bind(b: &mut Binder<'_>, root: &str, c: &Audio) -> Result<Self> {
        let h = c.hidden_size;
        let head = h / c.num_attention_heads;
        let q_scale = ((head as f64).sqrt().recip() / std::f64::consts::LN_2) as f32;
        let query_scale = b
            .decoded(&format!("{root}.per_dim_scale"), &[head])?
            .into_iter()
            .map(|v| q_scale * if v > 20. { v } else { v.exp().ln_1p() })
            .collect();
        Ok(Self {
            query: b.linear(&format!("{root}.q_proj"), h, h, c.use_clipped_linears)?,
            key: b.linear(&format!("{root}.k_proj"), h, h, c.use_clipped_linears)?,
            value: b.linear(&format!("{root}.v_proj"), h, h, c.use_clipped_linears)?,
            post: b.linear(&format!("{root}.post"), h, h, c.use_clipped_linears)?,
            relative_key: b.linear(&format!("{root}.relative_k_proj"), h, h, false)?,
            query_scale,
        })
    }
    fn run(
        &self,
        s: &VerifiedTensorSource,
        c: &Audio,
        input: &[f32],
        positions: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let h = c.hidden_size;
        let head = h / c.num_attention_heads;
        let horizon = c.attention_context_left - 1;
        let mut q = self.query.rows(s, input, cancelled)?;
        let mut k = self.key.rows(s, input, cancelled)?;
        let v = self.value.rows(s, input, cancelled)?;
        let relative = self.relative_key.rows(s, positions, cancelled)?;
        let k_scale = (std::f64::consts::E.ln_1p() / std::f64::consts::LN_2) as f32;
        for row in q.chunks_exact_mut(head) {
            for (x, scale) in row.iter_mut().zip(&self.query_scale) {
                *x *= scale;
            }
        }
        for x in &mut k {
            *x *= k_scale;
        }
        // For each valid (q,k) pair, the original blocked relative shift
        // selects the row for distance q-k. On the supported unpadded
        // causal profile, directly evaluate that finite window; never
        // allocate the padded block score matrix or attend future tokens.
        let mut result = vec![0.; input.len()];
        let mut logits = Vec::with_capacity(horizon);
        let dot = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>() as f32
        };
        for (t, row) in q.chunks_exact(h).enumerate() {
            check_cancel(cancelled)?;
            let first = t.saturating_sub(horizon - 1);
            for (head_index, query) in row.chunks_exact(head).enumerate() {
                let offset = head_index * head;
                logits.clear();
                for key_index in first..=t {
                    let distance = t - key_index;
                    let key = &k[key_index * h + offset..key_index * h + offset + head];
                    let rel = &relative[distance * h + offset..distance * h + offset + head];
                    let score = dot(query, key) + dot(query, rel);
                    logits.push((score / c.attention_logit_cap).tanh() * c.attention_logit_cap);
                }
                let probs = softmax(&logits)?;
                for d in 0..head {
                    result[t * h + offset + d] = probs
                        .iter()
                        .enumerate()
                        .map(|(i, &prob)| {
                            f64::from(prob) * f64::from(v[(first + i) * h + offset + d])
                        })
                        .sum::<f64>() as f32;
                }
            }
        }
        self.post.rows(s, &result, cancelled)
    }
}

struct LightConv {
    start: Linear,
    end: Linear,
    pre: Norm,
    norm: Norm,
    depthwise: Vec<f32>,
}
impl LightConv {
    fn bind(b: &mut Binder<'_>, root: &str, c: &Audio) -> Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            start: b.linear(
                &format!("{root}.linear_start"),
                h,
                h * 2,
                c.use_clipped_linears,
            )?,
            end: b.linear(&format!("{root}.linear_end"), h, h, c.use_clipped_linears)?,
            pre: b.norm(&format!("{root}.pre_layer_norm"), h, c.rms_norm_eps)?,
            norm: b.norm(&format!("{root}.conv_norm"), h, c.rms_norm_eps)?,
            depthwise: b.decoded(
                &format!("{root}.depthwise_conv1d.weight"),
                &[h, 1, c.conv_kernel_size],
            )?,
        })
    }
    fn run(
        &self,
        s: &VerifiedTensorSource,
        c: &Audio,
        input: Vec<f32>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let h = c.hidden_size;
        let mut hidden = input.clone();
        self.pre.apply(&mut hidden)?;
        let gate = self.start.rows(s, &hidden, cancelled)?;
        drop(hidden);
        let mut mixed = Vec::with_capacity(input.len());
        for row in gate.chunks_exact(2 * h) {
            let (a, b) = row.split_at(h);
            mixed.extend(a.iter().zip(b).map(|(&x, &g)| x / (1. + (-g).exp())));
        }
        drop(gate);
        let mut convolved =
            depthwise_causal(&mixed, h, &self.depthwise, c.conv_kernel_size, cancelled)?;
        drop(mixed);
        clamp(&mut convolved, c.gradient_clipping);
        self.norm.apply(&mut convolved)?;
        for v in &mut convolved {
            *v = silu(*v);
        }
        let mut out = self.end.rows(s, &convolved, cancelled)?;
        residual(&mut out, input, 1.);
        Ok(out)
    }
}

pub(super) struct Layer {
    first: FeedForward,
    second: FeedForward,
    attention: Attention,
    convolution: LightConv,
    pre_attention: Norm,
    post_attention: Norm,
    output: Norm,
}
impl Layer {
    pub fn bind(b: &mut Binder<'_>, index: usize, c: &Audio) -> Result<Self> {
        let root = format!("model.audio_tower.layers.{index}");
        let h = c.hidden_size;
        Ok(Self {
            first: FeedForward::bind(b, &format!("{root}.feed_forward1"), c)?,
            second: FeedForward::bind(b, &format!("{root}.feed_forward2"), c)?,
            attention: Attention::bind(b, &format!("{root}.self_attn"), c)?,
            convolution: LightConv::bind(b, &format!("{root}.lconv1d"), c)?,
            pre_attention: b.norm(&format!("{root}.norm_pre_attn"), h, RMS_EPSILON)?,
            post_attention: b.norm(&format!("{root}.norm_post_attn"), h, RMS_EPSILON)?,
            output: b.norm(&format!("{root}.norm_out"), h, RMS_EPSILON)?,
        })
    }
    pub fn run(
        &self,
        s: &VerifiedTensorSource,
        c: &Audio,
        input: Vec<f32>,
        positions: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let input = self.first.run(s, c, input, cancelled)?;
        let mut normalized = input.clone();
        clamp(&mut normalized, c.gradient_clipping);
        self.pre_attention.apply(&mut normalized)?;
        let mut attended = self
            .attention
            .run(s, c, &normalized, positions, cancelled)?;
        drop(normalized);
        clamp(&mut attended, c.gradient_clipping);
        self.post_attention.apply(&mut attended)?;
        residual(&mut attended, input, 1.);
        let output = self.convolution.run(s, c, attended, cancelled)?;
        let mut output = self.second.run(s, c, output, cancelled)?;
        clamp(&mut output, c.gradient_clipping);
        self.output.apply(&mut output)?;
        Ok(output)
    }
}

/// Original sinusoidal relative distances, in [sin,cos] rather than rotary
/// layout. Ascending distance directly indexes the unpadded causal window.
pub(super) fn relative_positions(c: &Audio) -> Vec<f32> {
    let half = c.hidden_size / 2;
    let increment = -(10000f64.ln() / (half - 1).max(1) as f64) as f32;
    let inv = (0..half)
        .map(|i| (i as f32 * increment).exp())
        .collect::<Vec<_>>();
    let mut positions = Vec::with_capacity(c.attention_context_left * c.hidden_size);
    for distance in 0..c.attention_context_left {
        positions.extend(inv.iter().map(|v| (distance as f32 * v).sin()));
        positions.extend(inv.iter().map(|v| (distance as f32 * v).cos()));
    }
    positions
}
