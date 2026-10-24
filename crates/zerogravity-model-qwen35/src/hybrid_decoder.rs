//! One Qwen-family composition; adapters retain exact source arithmetic/SSA.
//! No backend, root authority, storage, native selector or second instruction IR.
use crate::{Error, Result, model::LayerKind};

/// A packed vLLM projection and separate source-Torch projections have distinct
/// value representations. The family ordering is shared; adapters choose only
/// their existing arithmetic and original state operations.
pub(crate) trait HybridDecoder {
    type Value: Clone;
    type GdnInputs;
    type AttentionInputs;
    type FfnInputs;
    fn embedding(&mut self) -> Result<Self::Value>;
    fn input_norm(
        &mut self,
        layer: usize,
        branch: Self::Value,
        residual: Option<Self::Value>,
    ) -> Result<(Self::Value, Self::Value)>;
    fn gdn_inputs(&mut self, layer: usize, normal: &Self::Value) -> Result<Self::GdnInputs>;
    fn gdn_evolve(&mut self, layer: usize, inputs: Self::GdnInputs) -> Result<Self::Value>;
    fn attention_inputs(
        &mut self,
        layer: usize,
        normal: &Self::Value,
    ) -> Result<Self::AttentionInputs>;
    fn attention_evolve(
        &mut self,
        layer: usize,
        inputs: Self::AttentionInputs,
    ) -> Result<Self::Value>;
    fn attention_output(
        &mut self,
        layer: usize,
        kind: LayerKind,
        value: Self::Value,
    ) -> Result<Self::Value>;
    fn post_attention_norm(
        &mut self,
        layer: usize,
        branch: Self::Value,
        carried: Self::Value,
    ) -> Result<(Self::Value, Self::Value)>;
    fn ffn_inputs(&mut self, layer: usize, normal: &Self::Value) -> Result<Self::FfnInputs>;
    fn ffn_activate(&mut self, layer: usize, inputs: Self::FfnInputs) -> Result<Self::Value>;
    fn ffn_output(&mut self, layer: usize, value: Self::Value) -> Result<Self::Value>;
    fn finish_layer(
        &mut self,
        layer: usize,
        branch: Self::Value,
        carried: Self::Value,
    ) -> Result<(Self::Value, Option<Self::Value>)>;
    fn finish_model(
        &mut self,
        branch: Self::Value,
        residual: Option<Self::Value>,
    ) -> Result<Self::Value>;
}
pub(crate) fn compose<D: HybridDecoder>(layers: &[LayerKind], math: &mut D) -> Result<D::Value> {
    let embedding = math.embedding()?;
    let (branch, residual) = crate::run_layer_sequence(
        layers,
        (embedding, None),
        |layer, kind, (branch, residual)| {
            let (normal, carried) = math.input_norm(layer, branch, residual)?;
            let attended = match kind {
                LayerKind::LinearAttention => {
                    let inputs = math.gdn_inputs(layer, &normal)?;
                    math.gdn_evolve(layer, inputs)?
                }
                LayerKind::FullAttention => {
                    let inputs = math.attention_inputs(layer, &normal)?;
                    math.attention_evolve(layer, inputs)?
                }
                LayerKind::SlidingAttention => {
                    return Err(Error::Unsupported(
                        "Qwen hybrid decoder has no sliding layer".into(),
                    ));
                }
            };
            let branch = math.attention_output(layer, kind, attended)?;
            let (normal, carried) = math.post_attention_norm(layer, branch, carried)?;
            let inputs = math.ffn_inputs(layer, &normal)?;
            let activated = math.ffn_activate(layer, inputs)?;
            let branch = math.ffn_output(layer, activated)?;
            math.finish_layer(layer, branch, carried)
        },
    )?;
    math.finish_model(branch, residual)
}

/// Family source roles. A group does not select arithmetic, layout or a kernel.
#[derive(Clone, Copy)]
pub enum ProjectionGroup {
    GdnInput,
    GdnCoefficients,
    GdnOutput,
    AttentionInput,
    AttentionOutput,
    FfnInput,
    FfnOutput,
    Head,
}
impl ProjectionGroup {
    pub fn roles(self, layer: usize) -> Vec<crate::original_binding::QwenMatrixRole> {
        use crate::original_binding::QwenMatrixRole as M;
        match self {
            Self::GdnInput => vec![M::GdnQkv(layer), M::GdnGate(layer)],
            Self::GdnCoefficients => vec![M::GdnBeta(layer), M::GdnAlpha(layer)],
            Self::GdnOutput => vec![M::GdnOutput(layer)],
            Self::AttentionInput => vec![M::Query(layer), M::Key(layer), M::Value(layer)],
            Self::AttentionOutput => vec![M::AttentionOutput(layer)],
            Self::FfnInput => vec![M::FfnGate(layer), M::FfnUp(layer)],
            Self::FfnOutput => vec![M::FfnDown(layer)],
            Self::Head => vec![M::Output],
        }
    }
}
#[derive(Clone, Copy)]
pub enum ParameterGroup {
    InputNorm,
    FfnNorm,
    FinalNorm,
    GdnConvolution,
    GdnRecurrent,
    GdnAll,
    AttentionNorm,
}
impl ParameterGroup {
    pub fn roles(self, layer: usize) -> Vec<crate::original_binding::QwenParameterRole> {
        use crate::original_binding::QwenParameterRole as P;
        match self {
            Self::InputNorm => vec![P::InputNorm(layer)],
            Self::FfnNorm => vec![P::FfnNorm(layer)],
            Self::FinalNorm => vec![P::FinalNorm],
            Self::GdnConvolution => vec![P::GdnConvolution(layer)],
            Self::GdnRecurrent => vec![P::GdnA(layer), P::GdnDt(layer), P::GdnNorm(layer)],
            Self::GdnAll => vec![
                P::GdnConvolution(layer),
                P::GdnA(layer),
                P::GdnDt(layer),
                P::GdnNorm(layer),
            ],
            Self::AttentionNorm => vec![P::QueryNorm(layer), P::KeyNorm(layer)],
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ConvolutionRetention {
    OperatorHistory,
    #[cfg(feature = "cpu")]
    FullSourceCache,
}
/// Same original .8B/2B source schema with an explicit retained-cache choice.
pub(crate) fn state_specs(
    geometry: &crate::model::ModelGeometry,
    g: crate::original_binding::OriginalGdnGeometry,
    retention: ConvolutionRetention,
) -> Result<Vec<crate::StateSpec>> {
    let overflow = || Error::Limit("Qwen family state geometry overflow".into());
    let key = g.key_heads.checked_mul(g.key_width).ok_or_else(overflow)?;
    let value = g
        .value_heads
        .checked_mul(g.value_width)
        .ok_or_else(overflow)?;
    let channels = key
        .checked_mul(2)
        .and_then(|n| n.checked_add(value))
        .ok_or_else(overflow)?;
    let frames = match retention {
        ConvolutionRetention::OperatorHistory => {
            g.convolution_kernel.checked_sub(1).ok_or_else(overflow)?
        }
        #[cfg(feature = "cpu")]
        ConvolutionRetention::FullSourceCache => g.convolution_kernel,
    };
    if frames == 0 {
        return Err(overflow());
    }
    let conv = channels.checked_mul(frames).ok_or_else(overflow)?;
    let state = g
        .value_heads
        .checked_mul(g.value_width)
        .and_then(|n| n.checked_mul(g.key_width))
        .ok_or_else(overflow)?;
    let kv_width = usize::try_from(
        geometry
            .num_key_value_heads
            .checked_mul(geometry.head_dim)
            .ok_or_else(overflow)?,
    )
    .map_err(|_| overflow())?;
    geometry
        .layers
        .iter()
        .map(|kind| match kind {
            LayerKind::LinearAttention => Ok(crate::StateSpec::Delta { conv, state }),
            LayerKind::FullAttention => Ok(crate::StateSpec::Full { kv_width }),
            LayerKind::SlidingAttention => {
                Err(Error::Unsupported("Qwen family sliding state".into()))
            }
        })
        .collect()
}
#[cfg(feature = "cpu")]
pub(crate) fn state_domains(
    geometry: &crate::model::ModelGeometry,
) -> Result<Vec<crate::accelerator::program::StateSlot>> {
    let kinds = geometry
        .layers
        .iter()
        .map(|kind| match kind {
            LayerKind::LinearAttention => Ok(false),
            LayerKind::FullAttention => Ok(true),
            LayerKind::SlidingAttention => {
                Err(Error::Unsupported("Qwen family sliding domain".into()))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    crate::accelerator::program::hybrid_state_domains(kinds)
}
