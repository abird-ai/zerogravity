//! Closed projections/QK norms over an explicit private input, never a layer.
use super::OperatorSliceSource;
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    ActivationPrecision, MatrixAxes, OrdinaryRmsContract, PlanBuilder, PlanValue,
    ProgramNumericalProfile, SourceParameter, SourceTensor,
};
use crate::source::binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, LoadedQwenModel, Result};
use zerogravity_core::quant::QuantKind;

pub(crate) struct AttentionRoles {
    pub(crate) hidden: usize,
    pub(crate) query_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) width: usize,
    pub(crate) query: SourceTensor,
    pub(crate) key: SourceTensor,
    pub(crate) value: SourceTensor,
    pub(crate) query_norm: SourceParameter,
    pub(crate) key_norm: SourceParameter,
}

/// Check every role before recording or importing a private activation.
pub(crate) fn attention_roles(
    source: &impl OperatorSliceSource,
    layer: usize,
) -> Result<AttentionRoles> {
    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || source.rms_epsilon().to_bits() != 1e-6f32.to_bits()
        || !source.source_identity().is_bound()
    {
        return Err(Error::Unsupported(
            "closed attention slice profile/source/epsilon".into(),
        ));
    }
    let geometry = source.geometry();
    if geometry.layers.get(layer) != Some(&crate::config::LayerKind::FullAttention) {
        return Err(Error::Unsupported(
            "attention slice requires actual FullAttention layer".into(),
        ));
    }
    let size = |value| {
        usize::try_from(value).map_err(|_| Error::Limit("original attention geometry".into()))
    };
    let hidden = size(geometry.hidden_size)?;
    let query_heads = size(geometry.num_attention_heads)?;
    let kv_heads = size(geometry.num_key_value_heads)?;
    let width = size(geometry.head_dim)?;
    if !matches!(hidden, 256 | 1024 | 2048 | 5120)
        || width != 256
        || query_heads == 0
        || kv_heads == 0
        || !query_heads.is_multiple_of(kv_heads)
    {
        return Err(Error::Unsupported(
            "closed attention slice head/hidden geometry".into(),
        ));
    }
    let query_rows = query_heads
        .checked_mul(width)
        .and_then(|n| n.checked_mul(2))
        .filter(|n| *n <= 16384)
        .ok_or_else(|| Error::Limit("attention query/gate rows".into()))?;
    let kv_rows = kv_heads
        .checked_mul(width)
        .filter(|n| *n <= 16384)
        .ok_or_else(|| Error::Limit("attention key/value rows".into()))?;
    let query = source.matrix_descriptor(QwenMatrixRole::Query(layer))?;
    let key = source.matrix_descriptor(QwenMatrixRole::Key(layer))?;
    let value = source.matrix_descriptor(QwenMatrixRole::Value(layer))?;
    let namespace = format!("model.language_model.layers.{layer}.self_attn");
    for (matrix, rows, suffix) in [
        (&query, query_rows, "q_proj"),
        (&key, kv_rows, "k_proj"),
        (&value, kv_rows, "v_proj"),
    ] {
        if matrix.name != format!("{namespace}.{suffix}.weight")
            || matrix.source != source.source_identity()
            || matrix.kind != QuantKind::Bf16
            || matrix.axes != MatrixAxes::OutputInput
            || matrix.columns != hidden
            || matrix.rows != rows
        {
            return Err(Error::State(
                "attention original full matrix role/axes/shape".into(),
            ));
        }
    }
    let query_norm = source.parameter_descriptor(QwenParameterRole::QueryNorm(layer))?;
    let key_norm = source.parameter_descriptor(QwenParameterRole::KeyNorm(layer))?;
    for (parameter, elements, suffix) in [
        (&query_norm, query_rows / 2, "q_norm"),
        (&key_norm, kv_rows, "k_norm"),
    ] {
        if parameter.name != format!("{namespace}.{suffix}.weight")
            || parameter.source != source.source_identity()
            || OrdinaryRmsContract::TorchCpuBf16Avx2.validate(parameter, elements)? != width
        {
            return Err(Error::State(
                "attention original Q/K norm role/width".into(),
            ));
        }
    }
    Ok(AttentionRoles {
        hidden,
        query_heads,
        kv_heads,
        width,
        query,
        key,
        value,
        query_norm,
        key_norm,
    })
}

/// Existing typed values only. The input's producer remains the caller's.
/// No state domains, rotary, cache effects, token schedule or logits are added.
pub struct AttentionProjectionValues<V> {
    pub(crate) query_gate_projection: V,
    pub(crate) key_projection: V,
    pub(crate) value_projection: V,
    pub(crate) query: V,
    pub(crate) gate: V,
    pub(crate) normalized_query: V,
    pub(crate) normalized_key: V,
}
impl<V> AttentionProjectionValues<V> {
    pub fn query_gate_projection(&self) -> &V {
        &self.query_gate_projection
    }
    pub fn key_projection(&self) -> &V {
        &self.key_projection
    }
    pub fn value_projection(&self) -> &V {
        &self.value_projection
    }
    pub fn query(&self) -> &V {
        &self.query
    }
    pub fn gate(&self) -> &V {
        &self.gate
    }
    pub fn normalized_query(&self) -> &V {
        &self.normalized_query
    }
    pub fn normalized_key(&self) -> &V {
        &self.normalized_key
    }
}
impl LoadedQwenModel {
    /// Append to the caller's same-builder, same-source typed input.
    /// An embedding fixture is transport evidence, not a trained layer input.
    pub fn append_attention_projection_norm_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        input: &PlanValue,
    ) -> Result<AttentionProjectionValues<PlanValue>> {
        self.require_operator_slice()?;
        crate::forward::attention_projection_norm_slice(self, builder, layer, input)
    }
}

/// Qualifier-supplied raw BF16 bytes. The caller authenticates their external
/// source/archive provenance; this contract checks only binding/representation.
/// Import never asserts a model producer, StateRoot or token frontier.
#[derive(Clone, Copy)]
pub struct AttentionSliceInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    bytes: &'a [u8],
}
impl<'a> AttentionSliceInput<'a> {
    pub fn from_bf16_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        bytes: &'a [u8],
    ) -> Result<Self> {
        let value = Self {
            source,
            profile,
            bytes,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn precision(self) -> ActivationPrecision {
        ActivationPrecision::Bf16WidenedF32
    }
    pub fn bytes(self) -> &'a [u8] {
        self.bytes
    }
    pub(crate) fn validate(self) -> Result<()> {
        if !self.source.is_bound()
            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::State(
                "attention private input source/profile".into(),
            ));
        }
        if self.bytes.is_empty()
            || self.bytes.len() > 2 * 5120
            || !self.bytes.len().is_multiple_of(2)
            || self
                .bytes
                .chunks_exact(2)
                .any(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(Error::Malformed(
                "attention private finite BF16 input bytes".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "cpu")]
pub struct AttentionProjectionNormOutput {
    source: SourceIdentity,
    layer: usize,
    input: crate::host_math::HostValue,
    values: AttentionProjectionValues<crate::host_math::HostValue>,
}
#[cfg(feature = "cpu")]
impl AttentionProjectionNormOutput {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn precision(&self) -> ActivationPrecision {
        ActivationPrecision::Bf16WidenedF32
    }
    pub fn input(&self) -> Result<&[f32]> {
        self.input.as_slice()
    }
    pub fn query_gate_projection(&self) -> Result<&[f32]> {
        self.values.query_gate_projection.as_slice()
    }
    pub fn key_projection(&self) -> Result<&[f32]> {
        self.values.key_projection.as_slice()
    }
    pub fn value_projection(&self) -> Result<&[f32]> {
        self.values.value_projection.as_slice()
    }
    pub fn query(&self) -> Result<&[f32]> {
        self.values.query.as_slice()
    }
    pub fn gate(&self) -> Result<&[f32]> {
        self.values.gate.as_slice()
    }
    pub fn normalized_query(&self) -> Result<&[f32]> {
        self.values.normalized_query.as_slice()
    }
    pub fn normalized_key(&self) -> Result<&[f32]> {
        self.values.normalized_key.as_slice()
    }
}
#[cfg(feature = "cpu")]
impl LoadedQwenModel {
    /// Pure numerical fragment on explicit observed operator input, no session.
    pub fn execute_attention_projection_norm_slice(
        &self,
        layer: usize,
        input: AttentionSliceInput<'_>,
    ) -> Result<AttentionProjectionNormOutput> {
        self.require_operator_slice()?;
        let roles = attention_roles(self, layer)?;
        let mut math = crate::host_math::HostMath::for_operator_slice(self, None);
        let input = math.import_attention_input(input, roles.hidden)?;
        let values =
            crate::forward::attention_projection_norm_slice(self, &mut math, layer, &input)?;
        Ok(AttentionProjectionNormOutput {
            source: self.source_identity(),
            layer,
            input,
            values,
        })
    }
}
#[cfg(all(feature = "source-binding", target_os = "linux"))]
impl crate::source::binding::OriginalWindowOperators {
    /// No external-input node is invented; caller retains the source/metadata.
    pub fn append_attention_projection_norm_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        input: &PlanValue,
    ) -> Result<AttentionProjectionValues<PlanValue>> {
        crate::forward::attention_projection_norm_slice(self, builder, layer, input)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_attention_projection_norm_slice(
        &self,
        layer: usize,
        input: AttentionSliceInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<AttentionProjectionNormOutput> {
        let roles = attention_roles(self, layer)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let input = math.import_attention_input(input, roles.hidden)?;
        let values =
            crate::forward::attention_projection_norm_slice(self, &mut math, layer, &input)?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State("window attention slice cancelled".into()));
        }
        Ok(AttentionProjectionNormOutput {
            source: self.source_identity(),
            layer,
            input,
            values,
        })
    }
}
