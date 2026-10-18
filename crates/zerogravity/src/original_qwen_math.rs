//! Original Qwen3.8 source-bound one-layer CPU *proposals*.
//!
//! This is a thin adapter to the existing Qwen3.5 CPU arithmetic,
//! attention/GDN regions, and verified Safetensors source ownership.
//! It does not create an independent token loop, model loader, mutable
//! Qwen session or StateRoot publication owner. Source framework BF16
//! intermediate rounding and trained full-token parity are NOT qualified.
//!
//! Upstream source-order distinctions: zero-centered ordinary RMS gamma,
//! per-head interleaved full-attention [query|gate], GDN 16-key to
//! 48-value head repeat_interleave, negative exp(A_log) decay, and
//! partial RoPE over the original rotary width.

use crate::accelerator::{
    AttentionQkRegion, AttentionSoftmaxRegion, AttentionValueRegion, BinaryOperation, BinaryRegion,
    GdnCoefficientCompleted, GdnCoefficientRegion, RopeKvRegion, SourceIdentity,
};
use crate::model::{LayerKind, qwen::GdnOutputGate};
use crate::original_qwen_cpu::QwenOriginalCpu;
use crate::{Error, Result};
use std::sync::Arc;

/// Model/layer-labelled immutable tentative GDN state. Only a bound
/// OriginalQwenLayerMath can mint the initial source-specific state; no
/// constructor accepts untrusted raw convolution/recurrent arrays.
#[derive(Clone)]
pub struct OriginalGdnPrefix {
    pub(crate) source: SourceIdentity,
    pub(crate) layer: usize,
    pub(crate) position: usize,
    pub(crate) convolution: Arc<Vec<f32>>,
    pub(crate) recurrent: Arc<Vec<f32>>,
}
impl OriginalGdnPrefix {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn convolution(&self) -> &[f32] {
        &self.convolution
    }
    pub fn recurrent(&self) -> &[f32] {
        &self.recurrent
    }
}

/// One private GDN value+state proposal. The owner must separately admit
/// and commit the accepted state through the existing SessionAuthority.
pub struct OriginalGdnProposal {
    pub output: Vec<f32>,
    pub next: OriginalGdnPrefix,
    pub key_heads: usize,
    pub value_heads: usize,
}

/// Immutable original-source/layer-labelled tentative causal prefix.
/// The only public constructor is on a bound OriginalQwenLayerMath, so
/// raw KV from a different model cannot silently become this model's
/// history. An Arc clone reuses immutable slabs, not source tensors.
#[derive(Clone)]
pub struct OriginalFullPrefix {
    pub(crate) source: SourceIdentity,
    pub(crate) layer: usize,
    pub(crate) keys: Vec<Arc<[f32]>>,
    pub(crate) values: Vec<Arc<[f32]>>,
}
impl OriginalFullPrefix {
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn last_key(&self) -> Option<&[f32]> {
        self.keys.last().map(AsRef::as_ref)
    }
    pub fn last_value(&self) -> Option<&[f32]> {
        self.values.last().map(AsRef::as_ref)
    }
}

/// An immutable proposed key/value slab plus candidate output. This is NOT
/// a session root and cannot publish a token. The next prefix remains tied
/// to its exact original source and layer.
pub struct OriginalFullAttentionProposal {
    pub output: Vec<f32>,
    pub prefix: OriginalFullPrefix,
    pub gates: Vec<f32>,
    pub position: usize,
    pub source: SourceIdentity,
}

/// Split original HF query projection by HEAD, not by two flat halves.
/// Shape is [query_heads, 2 * head_dim] with [Q,gate] per head.
pub fn split_interleaved_query_gate(
    raw: &[f32],
    query_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let width = head_dim
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("query+gate width overflow".into()))?;
    if query_heads == 0
        || head_dim == 0
        || query_heads.checked_mul(width) != Some(raw.len())
        || raw.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Malformed(
            "original per-head Q/gate projection geometry".into(),
        ));
    }
    if query_heads * head_dim > 1_048_576 {
        return Err(Error::Limit("original Qwen query/gate stage budget".into()));
    }
    let mut query = Vec::with_capacity(query_heads * head_dim);
    let mut gate = Vec::with_capacity(query_heads * head_dim);
    for head in raw.chunks_exact(width) {
        query.extend_from_slice(&head[..head_dim]);
        gate.extend_from_slice(&head[head_dim..]);
    }
    Ok((query, gate))
}

/// Repeat original Q and K per key head into the wider *value* head domain.
/// For real Qwen3.8, 16 original keys map to 48 value heads: 0,0,0,
/// 1,1,1 ... 15,15,15. Never reshape by flat chunk concatenation.
pub fn repeat_qk_to_value_heads(
    keys: &[f32],
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if key_heads == 0
        || value_heads == 0
        || head_dim == 0
        || !value_heads.is_multiple_of(key_heads)
        || key_heads.checked_mul(head_dim) != Some(keys.len())
        || value_heads.checked_mul(head_dim).is_none()
        || keys.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Malformed(
            "original grouped GDN key/value head mismatch".into(),
        ));
    }
    let total = value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Limit("Qwen repeated GDN width".into()))?;
    if total > 1_048_576 {
        return Err(Error::Limit("original Qwen GDN repeat stage budget".into()));
    }
    let ratio = value_heads / key_heads;
    let mut repeated = Vec::with_capacity(total);
    for key in keys.chunks_exact(head_dim) {
        for _ in 0..ratio {
            repeated.extend_from_slice(key);
        }
    }
    Ok(repeated)
}

/// Convert original HF A_log into the preexisting single authoritative
/// source GDN coefficient operation: exp(g) where
/// g = -exp(A_log) * softplus(alpha+dt_bias), beta = sigmoid(b).
/// Source BF16 rounding between projections remains a later gate.
pub fn original_gdn_coefficients(
    source: SourceIdentity,
    alpha: &[f32],
    beta: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    key_dim: usize,
) -> Result<GdnCoefficientCompleted> {
    if !source.is_bound()
        || key_dim == 0
        || key_dim > 8192
        || alpha.is_empty()
        || alpha.len() != beta.len()
        || alpha.len() != dt_bias.len()
        || alpha.len() != a_log.len()
        || alpha.len() > 4096
        || a_log.iter().any(|x| !x.is_finite())
    {
        return Err(Error::Malformed(
            "original source GDN coefficient head/width mismatch".into(),
        ));
    }
    let decay: Vec<f32> = a_log.iter().map(|&a| -a.exp()).collect();
    if decay.iter().any(|x| !x.is_finite()) {
        return Err(Error::Malformed(
            "original source A_log exponential overflow".into(),
        ));
    }
    crate::qwen35::source_gdn_coefficients(GdnCoefficientRegion {
        source_identity: source,
        operation: "qwen.original.gdn.coefficients",
        alpha,
        beta,
        dt: dt_bias,
        a: &decay,
        scale: 1.0 / (key_dim as f32).sqrt(),
    })
}

/// Borrowed original Qwen source and its shared CPU mathematical operators.
/// All functions work on private values and remain outside SessionAuthority.
pub struct OriginalQwenLayerMath<'borrow, 'source> {
    cpu: &'borrow QwenOriginalCpu<'source>,
}
impl<'borrow, 'source> OriginalQwenLayerMath<'borrow, 'source> {
    pub fn new(cpu: &'borrow QwenOriginalCpu<'source>) -> Self {
        Self { cpu }
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.cpu.binding().source_identity()
    }
    /// Existing source-qualified original model config, never mutable.
    pub fn geometry(&self) -> &crate::model::ModelGeometry {
        self.cpu.binding().config().geometry()
    }
    pub fn hidden_width(&self) -> Result<usize> {
        usize::try_from(self.geometry().hidden_size)
            .map_err(|_| Error::Limit("Qwen original hidden width exceeds host".into()))
    }
    fn prefix(i: usize) -> String {
        format!("model.language_model.layers.{i}")
    }
    fn layer(&self, i: usize, kind: LayerKind) -> Result<String> {
        if self.cpu.binding().config().geometry().layers.get(i) != Some(&kind) {
            return Err(Error::Unsupported(
                "Qwen source layer role is incompatible with requested operator".into(),
            ));
        }
        Ok(Self::prefix(i))
    }
    fn eps(&self) -> Result<f32> {
        let value = self.cpu.binding().config().rms_norm_eps();
        let eps = value as f32;
        if !eps.is_finite() || eps <= 0.0 {
            return Err(Error::Malformed("original Qwen source RMS epsilon".into()));
        }
        Ok(eps)
    }
    fn hidden(&self, input: &[f32]) -> Result<()> {
        let expected = usize::try_from(self.cpu.binding().config().geometry().hidden_size)
            .map_err(|_| Error::Limit("Qwen original hidden dimension".into()))?;
        if input.len() != expected || input.iter().any(|v| !v.is_finite()) {
            return Err(Error::Malformed(
                "Qwen original hidden vector length/nonfinite input".into(),
            ));
        }
        Ok(())
    }
    /// Ordinary source RMSNorm has gamma = (1 + original checkpoint weight),
    /// unlike the separate GDN gated RMSNorm, which stores gamma directly.
    /// Uses the existing Qwen CPU norm arithmetic, not another RMS kernel.
    pub fn ordinary_norm(&self, input: &[f32], raw_gamma: &[f32]) -> Result<Vec<f32>> {
        if input.len() != raw_gamma.len()
            || raw_gamma
                .iter()
                .any(|x| !x.is_finite() || !(1.0 + x).is_finite())
        {
            return Err(Error::Malformed("Qwen original zero-centered norm".into()));
        }
        let gamma: Vec<f32> = raw_gamma.iter().map(|x| 1.0 + x).collect();
        crate::qwen35::source_rms_normalize(input, &gamma, self.eps()?)
    }
    fn residual_norm(&self, i: usize, kind: LayerKind, x: &[f32]) -> Result<Vec<f32>> {
        self.hidden(x)?;
        let prefix = self.layer(i, kind)?;
        self.ordinary_norm(
            x,
            &self
                .cpu
                .vector(&format!("{prefix}.input_layernorm.weight"))?,
        )
    }
    /// Norm and ordinary SwiGLU FFN of one original layer. This is a
    /// single-layer FP32 diagnostic proposal, NOT a multi-layer forward.
    pub fn propose_ffn(&self, layer: usize, post_attention_residual: &[f32]) -> Result<Vec<f32>> {
        self.hidden(post_attention_residual)?;
        let prefix = Self::prefix(layer);
        if layer >= self.cpu.binding().config().geometry().layers.len() {
            return Err(Error::Malformed(
                "Qwen FFN layer index outside source".into(),
            ));
        }
        let norm = self.ordinary_norm(
            post_attention_residual,
            &self
                .cpu
                .vector(&format!("{prefix}.post_attention_layernorm.weight"))?,
        )?;
        let gate = self
            .cpu
            .project(&format!("{prefix}.mlp.gate_proj.weight"), &norm, None)?;
        let up = self
            .cpu
            .project(&format!("{prefix}.mlp.up_proj.weight"), &norm, None)?;
        if gate.len() != up.len()
            || gate.len() != self.cpu.binding().config().intermediate_size() as usize
        {
            return Err(Error::Malformed("Qwen FFN gate/up geometry".into()));
        }
        let fused: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| crate::math::silu(g) * u)
            .collect();
        if fused.iter().any(|v| !v.is_finite()) {
            return Err(Error::Malformed("nonfinite original FFN SwiGLU".into()));
        }
        let down = self
            .cpu
            .project(&format!("{prefix}.mlp.down_proj.weight"), &fused, None)?;
        if down.len() != post_attention_residual.len() {
            return Err(Error::Malformed("Qwen original FFN residual width".into()));
        }
        let combined: Vec<f32> = post_attention_residual
            .iter()
            .zip(&down)
            .map(|(&r, &v)| r + v)
            .collect();
        if combined.iter().any(|v| !v.is_finite()) {
            return Err(Error::Malformed("nonfinite original FFN residual".into()));
        }
        Ok(combined)
    }

    /// Create an empty source- and layer-labelled *private* prefix.
    /// This is a numerical value container, not another SessionAuthority.
    pub fn empty_full_prefix(&self, layer: usize) -> Result<OriginalFullPrefix> {
        self.layer(layer, LayerKind::FullAttention)?;
        Ok(OriginalFullPrefix {
            source: self.source_identity(),
            layer,
            keys: Vec::new(),
            values: Vec::new(),
        })
    }

    /// Fully source-bound single-token full-attention proposal using
    /// *immutable* prior KV handles. This invokes the existing Qwen RoPE,
    /// QK, softmax, value and sigmoid gate CPU regions in source order.
    /// It never appends to a caller's session KV, nor publishes StateRoot.
    pub fn propose_full_attention(
        &self,
        layer: usize,
        input: &[f32],
        previous: &OriginalFullPrefix,
    ) -> Result<OriginalFullAttentionProposal> {
        let prefix = self.layer(layer, LayerKind::FullAttention)?;
        self.hidden(input)?;
        // No foreign model, stale layer, caller-injected raw KV, or
        // unbound prefix can be accepted as an implicit current session.
        if previous.source != self.source_identity()
            || previous.layer != layer
            || previous.keys.len() != previous.values.len()
        {
            return Err(Error::State("foreign Qwen original KV source/layer".into()));
        }
        let position = previous.len();
        let cfg = self.cpu.binding().config();
        let geo = cfg.geometry();
        let qheads = usize::try_from(geo.num_attention_heads)
            .map_err(|_| Error::Limit("Qwen query head count".into()))?;
        let kvheads = usize::try_from(geo.num_key_value_heads)
            .map_err(|_| Error::Limit("Qwen KV head count".into()))?;
        let d = usize::try_from(geo.head_dim)
            .map_err(|_| Error::Limit("Qwen source head width".into()))?;
        if position >= geo.max_position_embeddings as usize
            || position >= 8192 // diagnostic gate; longer trained windows not qualified yet
            || kvheads == 0
            || !qheads.is_multiple_of(kvheads)
        {
            return Err(Error::Malformed(
                "Qwen original attention KV/position mismatch".into(),
            ));
        }
        let normalized = self.residual_norm(layer, LayerKind::FullAttention, input)?;
        let p = format!("{prefix}.self_attn");
        let raw_q = self
            .cpu
            .project(&format!("{p}.q_proj.weight"), &normalized, None)?;
        let (query, gates) = if cfg.attention_output_gate() {
            split_interleaved_query_gate(&raw_q, qheads, d)?
        } else if raw_q.len() == qheads * d {
            (raw_q, Vec::new())
        } else {
            return Err(Error::Malformed("Qwen plain query shape".into()));
        };
        let key = self
            .cpu
            .project(&format!("{p}.k_proj.weight"), &normalized, None)?;
        let value = self
            .cpu
            .project(&format!("{p}.v_proj.weight"), &normalized, None)?;
        let norm_query = self.ordinary_head_norm(&query, &format!("{p}.q_norm.weight"), qheads)?;
        let norm_key = self.ordinary_head_norm(&key, &format!("{p}.k_norm.weight"), kvheads)?;
        let theta = cfg.rope_theta() as f32;
        let rotary = cfg.rotary_dimensions() as usize;
        if !theta.is_finite() || theta <= 0.0 {
            return Err(Error::Malformed("Qwen original RoPE theta".into()));
        }
        let rotated = crate::qwen35::source_rope_kv(RopeKvRegion {
            source_identity: self.source_identity(),
            operation: "qwen.original.full.rope",
            query: &norm_query,
            key: &norm_key,
            value: &value,
            head_width: d,
            query_heads: qheads,
            kv_heads: kvheads,
            rotate: rotary,
            position,
            theta_scale: theta.powf(-2.0 / rotary as f32),
        })?;
        let mut keys = previous.keys.clone();
        let mut values = previous.values.clone();
        keys.push(Arc::from(rotated.key.clone()));
        values.push(Arc::from(rotated.value.clone()));
        let qk = crate::qwen35::source_attention_qk(AttentionQkRegion {
            source_identity: self.source_identity(),
            operation: "qwen.original.full.qk",
            query: &rotated.query,
            keys: &keys,
            head_width: d,
            query_heads: qheads,
            kv_heads: kvheads,
            scale: 1.0 / (d as f32).sqrt(),
        })?;
        let prob = crate::qwen35::source_attention_softmax(AttentionSoftmaxRegion {
            source_identity: self.source_identity(),
            operation: "qwen.original.full.softmax",
            scores: &qk.scores,
            heads: qheads,
            width: keys.len(),
        })?;
        let padded = keys
            .len()
            .checked_add(31)
            .ok_or_else(|| Error::Limit("Qwen padded KV prefix overflow".into()))?
            / 32
            * 32;
        let attended = crate::qwen35::source_attention_value(AttentionValueRegion {
            source_identity: self.source_identity(),
            operation: "qwen.original.full.value",
            probabilities: &prob.probabilities,
            values: &values,
            head_width: d,
            query_heads: qheads,
            kv_heads: kvheads,
            padded_tokens: padded,
        })?;
        let gated = if cfg.attention_output_gate() {
            crate::qwen35::source_binary(BinaryRegion {
                source_identity: self.source_identity(),
                operation: "qwen.original.full.gated",
                kind: BinaryOperation::SigmoidGate,
                left: &attended,
                right: &gates,
            })?
        } else {
            attended
        };
        let output = self
            .cpu
            .project(&format!("{p}.o_proj.weight"), &gated, None)?;
        if output.len() != input.len() || output.iter().any(|x| !x.is_finite()) {
            return Err(Error::Malformed(
                "Qwen full-attention projected output".into(),
            ));
        }
        Ok(OriginalFullAttentionProposal {
            output,
            prefix: OriginalFullPrefix {
                source: self.source_identity(),
                layer,
                keys,
                values,
            },
            gates,
            position,
            source: self.source_identity(),
        })
    }
    fn ordinary_head_norm(&self, data: &[f32], name: &str, heads: usize) -> Result<Vec<f32>> {
        let raw = self.cpu.vector(name)?;
        if raw.iter().any(|x| !x.is_finite() || !(1.0 + x).is_finite()) {
            return Err(Error::Malformed("Qwen original head RMS weight".into()));
        }
        let gamma: Vec<f32> = raw.iter().map(|v| 1.0 + v).collect();
        crate::qwen35::source_grouped_rms_normalize(data, &gamma, heads, self.eps()?)
    }
    /// Mint an isolated zero-initialized original GDN proposal for one
    /// specific bound model source/layer; NEVER a committed SessionRoot.
    pub fn empty_gdn_prefix(&self, layer: usize) -> Result<OriginalGdnPrefix> {
        self.layer(layer, LayerKind::LinearAttention)?;
        let cfg = self.cpu.binding().config();
        let g = cfg.geometry();
        let nk = usize::try_from(g.recurrent_key_heads.unwrap_or(0))
            .map_err(|_| Error::Limit("original GDN key heads".into()))?;
        let nv = usize::try_from(g.recurrent_value_heads.unwrap_or(0))
            .map_err(|_| Error::Limit("original GDN value heads".into()))?;
        let kd = cfg.linear_key_head_dim() as usize;
        let vd = cfg.linear_value_head_dim() as usize;
        if nk == 0
            || nv == 0
            || kd == 0
            || kd != vd
            || !nv.is_multiple_of(nk)
            || cfg.linear_conv_kernel_dim() != 4
        {
            return Err(Error::Unsupported(
                "unsupported original GDN source state shape".into(),
            ));
        }
        let qkv = nk
            .checked_mul(kd)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| nv.checked_mul(vd).and_then(|v| n.checked_add(v)))
            .ok_or_else(|| Error::Limit("original GDN conv capacity".into()))?;
        let recurrent = nv
            .checked_mul(kd)
            .and_then(|n| n.checked_mul(vd))
            .ok_or_else(|| Error::Limit("original GDN recurrent capacity".into()))?;
        if recurrent
            .checked_mul(4)
            .is_none_or(|n| n > 64 * 1024 * 1024)
            || qkv.checked_mul(3).is_none_or(|n| n > 16 * 1024 * 1024)
        {
            return Err(Error::Limit(
                "original GDN diagnostic state capacity".into(),
            ));
        }
        Ok(OriginalGdnPrefix {
            source: self.source_identity(),
            layer,
            position: 0,
            convolution: Arc::new(vec![0.0; qkv * 3]),
            recurrent: Arc::new(vec![0.0; recurrent]),
        })
    }

    /// GDN preliminary key replication and causal conv are completed
    /// using the same original GGUF Qwen CPU primitives. The separate
    /// recurrent/normalization proposal is returned, never published.
    pub fn propose_gdn(
        &self,
        layer: usize,
        input: &[f32],
        previous: &OriginalGdnPrefix,
    ) -> Result<OriginalGdnProposal> {
        let prefix = self.layer(layer, LayerKind::LinearAttention)?;
        self.hidden(input)?;
        if previous.source != self.source_identity()
            || previous.layer != layer
            || previous.position >= 8192
            || previous.position
                >= self
                    .cpu
                    .binding()
                    .config()
                    .geometry()
                    .max_position_embeddings as usize
        {
            return Err(Error::State(
                "foreign or exhausted original GDN state prefix".into(),
            ));
        }
        let previous_conv = previous.convolution();
        let previous_recurrent = previous.recurrent();
        let cfg = self.cpu.binding().config();
        let g = cfg.geometry();
        let nk = usize::try_from(g.recurrent_key_heads.unwrap_or(0))
            .map_err(|_| Error::Limit("Qwen recurrent key heads".into()))?;
        let nv = usize::try_from(g.recurrent_value_heads.unwrap_or(0))
            .map_err(|_| Error::Limit("Qwen recurrent value heads".into()))?;
        let kd = cfg.linear_key_head_dim() as usize;
        let vd = cfg.linear_value_head_dim() as usize;
        let kernel = cfg.linear_conv_kernel_dim() as usize;
        // The presently qualified original scalar recurrence assumes equal
        // per-head K/V widths and a four-element depthwise source conv.
        // Other source variants must add a qualified specialized operator.
        if nk == 0 || nv == 0 || kd == 0 || !nv.is_multiple_of(nk) || kd != vd || kernel != 4 {
            return Err(Error::Unsupported(
                "Qwen original GDN requires qualified 4-tap equal K/V width".into(),
            ));
        }
        let ktotal = nk
            .checked_mul(kd)
            .ok_or_else(|| Error::Limit("GDN K dim".into()))?;
        let vtotal = nv
            .checked_mul(vd)
            .ok_or_else(|| Error::Limit("GDN V dim".into()))?;
        let qkv = ktotal
            .checked_mul(2)
            .and_then(|x| x.checked_add(vtotal))
            .ok_or_else(|| Error::Limit("GDN projection width".into()))?;
        let expected_conv = qkv
            .checked_mul(3)
            .ok_or_else(|| Error::Limit("GDN conv history".into()))?;
        let expected_recurrent = nv
            .checked_mul(kd)
            .and_then(|x| x.checked_mul(vd))
            .ok_or_else(|| Error::Limit("GDN recurrent count".into()))?;
        if expected_recurrent
            .checked_mul(std::mem::size_of::<f32>())
            .is_none_or(|bytes| bytes > 64 * 1024 * 1024)
            || previous_conv.len() != expected_conv
            || previous_recurrent.len() != expected_recurrent
            || previous_conv
                .iter()
                .chain(previous_recurrent)
                .any(|x| !x.is_finite())
        {
            return Err(Error::Malformed(
                "GDN private proposed state geometry".into(),
            ));
        }
        let normalized = self.residual_norm(layer, LayerKind::LinearAttention, input)?;
        let p = format!("{prefix}.linear_attn");
        let raw = self
            .cpu
            .project(&format!("{p}.in_proj_qkv.weight"), &normalized, None)?;
        let z = self
            .cpu
            .project(&format!("{p}.in_proj_z.weight"), &normalized, None)?;
        let alpha = self
            .cpu
            .project(&format!("{p}.in_proj_a.weight"), &normalized, None)?;
        let beta = self
            .cpu
            .project(&format!("{p}.in_proj_b.weight"), &normalized, None)?;
        if raw.len() != qkv || z.len() != vtotal || alpha.len() != nv || beta.len() != nv {
            return Err(Error::Malformed(
                "GDN original source projection width".into(),
            ));
        }
        let weights = self
            .cpu
            .binding()
            .text_tensor(&format!("{p}.conv1d.weight"))?;
        let conv_weight_len = qkv
            .checked_mul(kernel)
            .ok_or_else(|| Error::Limit("Qwen GDN conv source size".into()))?;
        let conv_weights =
            crate::quant::dequantize(weights.kind, weights.packed_bytes, conv_weight_len)?;
        let mut next_conv = previous_conv.to_vec();
        let mut raw_conv = vec![0.0; qkv];
        crate::qwen35::source_gdn_conv_step(&raw, &conv_weights, &mut next_conv, &mut raw_conv)?;
        let mixed: Vec<f32> = raw_conv.into_iter().map(crate::math::silu).collect();
        if mixed.iter().any(|v| !v.is_finite()) {
            return Err(Error::Malformed(
                "nonfinite original GDN convolution activation".into(),
            ));
        }
        let q = repeat_qk_to_value_heads(&mixed[..ktotal], nk, nv, kd)?;
        let key = repeat_qk_to_value_heads(&mixed[ktotal..2 * ktotal], nk, nv, kd)?;
        let v = &mixed[2 * ktotal..];
        let l2_gamma = vec![1.0 / (kd as f32).sqrt(); kd];
        let q = crate::qwen35::source_grouped_rms_normalize(&q, &l2_gamma, nv, 1e-6 / kd as f32)?;
        let key =
            crate::qwen35::source_grouped_rms_normalize(&key, &l2_gamma, nv, 1e-6 / kd as f32)?;
        let a_log = self.cpu.vector(&format!("{p}.A_log"))?;
        let dt = self.cpu.vector(&format!("{p}.dt_bias"))?;
        let ready =
            original_gdn_coefficients(self.source_identity(), &alpha, &beta, &dt, &a_log, kd)?;
        let mut next_state = previous_recurrent.to_vec();
        let mut core = vec![0.0; vtotal];
        for h in 0..nv {
            let off = h * kd * vd;
            let start = h * vd;
            crate::qwen35::source_gdn_recurrent_step(
                &mut next_state[off..off + kd * vd],
                &q[h * kd..(h + 1) * kd],
                &key[h * kd..(h + 1) * kd],
                &v[start..start + vd],
                ready.coefficients[h],
                &mut core[start..start + vd],
            )?;
        }
        let weights = self.cpu.vector(&format!("{p}.norm.weight"))?;
        let normalized =
            crate::qwen35::source_grouped_rms_normalize(&core, &weights, nv, self.eps()?)?;
        let output_gate: Vec<f32> = match cfg.gdn_output_gate() {
            GdnOutputGate::Silu => z.iter().map(|x| crate::math::silu(*x)).collect(),
            GdnOutputGate::Sigmoid => z.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
        };
        let fused: Vec<f32> = normalized
            .iter()
            .zip(output_gate)
            .map(|(&v, z)| v * z)
            .collect();
        if fused.iter().any(|x| !x.is_finite()) {
            return Err(Error::Malformed(
                "nonfinite Qwen original GDN gated normalization".into(),
            ));
        }
        let output = self
            .cpu
            .project(&format!("{p}.out_proj.weight"), &fused, None)?;
        if output.len() != input.len() || output.iter().any(|x| !x.is_finite()) {
            return Err(Error::Malformed(
                "invalid Qwen original GDN projection output".into(),
            ));
        }
        Ok(OriginalGdnProposal {
            output,
            next: OriginalGdnPrefix {
                source: self.source_identity(),
                layer,
                position: previous.position + 1,
                convolution: Arc::new(next_conv),
                recurrent: Arc::new(next_state),
            },
            key_heads: nk,
            value_heads: nv,
        })
    }
}
