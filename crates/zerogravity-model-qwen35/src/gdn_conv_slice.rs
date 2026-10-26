//! Pure original causal convolution + SiLU on explicit observed operands.
//! No cache owner, producer/frontier assertion or full-model eligibility.
use super::OperatorSliceSource;
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    ActivationPrecision, CausalConvSiluContract, CausalConvValues, PlanBuilder, PlanValue,
    ProgramNumericalProfile, SourceParameter,
};
use crate::source::binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, LoadedQwenModel, Result};

pub(crate) fn convolution_role(
    source: &impl OperatorSliceSource,
    layer: usize,
) -> Result<(SourceParameter, usize)> {
    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || !source.source_identity().is_bound()
        || source.geometry().layers.get(layer) != Some(&crate::config::LayerKind::LinearAttention)
    {
        return Err(Error::Unsupported(
            "closed original convolution profile/layer".into(),
        ));
    }
    let parameter = source.parameter_descriptor(QwenParameterRole::GdnConvolution(layer))?;
    let channels = source
        .matrix_descriptor(QwenMatrixRole::GdnQkv(layer))?
        .rows;
    if parameter.source != source.source_identity()
        || parameter.name
            != format!("model.language_model.layers.{layer}.linear_attn.conv1d.weight")
    {
        return Err(Error::State("original convolution raw role/source".into()));
    }
    CausalConvSiluContract::TorchCpuBf16Avx2.validate(
        &parameter,
        channels,
        channels
            .checked_mul(3)
            .ok_or_else(|| Error::Limit("convolution history width".into()))?,
    )?;
    Ok((parameter, channels))
}

/// Observed operand phase only; never a committed-session binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvInputPhase {
    /// Initial operator's implicit left padding, supplied as authenticated +0.
    InitialPadding,
    /// Last three entries of the original incoming `[channel,4]` cache.
    CachedTail,
}

/// Qualifier authenticates external source/archive provenance. This type checks
/// representation only, and never asserts a model producer or StateRoot.
#[derive(Clone, Copy)]
pub struct GdnConvSliceInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    phase: ConvInputPhase,
    activation: &'a [u8],
    history: &'a [u8],
}
impl<'a> GdnConvSliceInput<'a> {
    pub fn from_bf16_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        phase: ConvInputPhase,
        activation: &'a [u8],
        history: &'a [u8],
    ) -> Result<Self> {
        let input = Self {
            source,
            profile,
            phase,
            activation,
            history,
        };
        input.validate()?;
        Ok(input)
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn phase(self) -> ConvInputPhase {
        self.phase
    }
    pub fn precision(self) -> ActivationPrecision {
        ActivationPrecision::Bf16WidenedF32
    }
    pub fn activation_bytes(self) -> &'a [u8] {
        self.activation
    }
    pub fn history_bytes(self) -> &'a [u8] {
        self.history
    }
    pub(crate) fn validate(self) -> Result<usize> {
        let channels = self.activation.len() / 2;
        if !self.source.is_bound()
            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || channels == 0
            || channels > 16384
            || !channels.is_multiple_of(32)
            || self.activation.len() != channels * 2
            || self.history.len() != channels * 6
            || self
                .activation
                .chunks_exact(2)
                .chain(self.history.chunks_exact(2))
                .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
            || (self.phase == ConvInputPhase::InitialPadding
                && self.history.iter().any(|v| *v != 0))
        {
            return Err(Error::Malformed(
                "private convolution phase/source/finite BF16 operands".into(),
            ));
        }
        Ok(channels)
    }
}

impl LoadedQwenModel {
    /// Append only, to existing same-builder BF16 values; no execution admission.
    pub fn append_gdn_convolution_silu_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        input: &PlanValue,
        history: &PlanValue,
    ) -> Result<CausalConvValues<PlanValue>> {
        self.require_operator_slice()?;
        crate::forward::gdn_convolution_silu_slice(self, builder, layer, input, history)
    }
}

#[cfg(all(feature = "cpu", feature = "source-binding", target_os = "linux"))]
pub struct GdnConvolutionSiluOutput {
    source: SourceIdentity,
    layer: usize,
    phase: ConvInputPhase,
    input: crate::host_math::HostValue,
    prior: crate::host_math::HostValue,
    values: CausalConvValues<crate::host_math::HostValue>,
}
#[cfg(all(feature = "cpu", feature = "source-binding", target_os = "linux"))]
impl GdnConvolutionSiluOutput {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn precision(&self) -> ActivationPrecision {
        ActivationPrecision::Bf16WidenedF32
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn phase(&self) -> ConvInputPhase {
        self.phase
    }
    pub fn input(&self) -> Result<&[f32]> {
        self.input.as_slice()
    }
    pub fn prior_history(&self) -> Result<&[f32]> {
        self.prior.as_slice()
    }
    pub fn raw(&self) -> Result<&[f32]> {
        self.values.raw.as_slice()
    }
    pub fn activated(&self) -> Result<&[f32]> {
        self.values.activated.as_slice()
    }
    pub fn next_history(&self) -> Result<&[f32]> {
        self.values.next_history.as_slice()
    }
}
#[cfg(all(feature = "source-binding", target_os = "linux"))]
impl crate::source::binding::OriginalWindowOperators {
    pub fn append_gdn_convolution_silu_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        input: &PlanValue,
        history: &PlanValue,
    ) -> Result<CausalConvValues<PlanValue>> {
        crate::forward::gdn_convolution_silu_slice(self, builder, layer, input, history)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_gdn_convolution_silu_slice(
        &self,
        layer: usize,
        input: GdnConvSliceInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<GdnConvolutionSiluOutput> {
        let (_, channels) = convolution_role(self, layer)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let (activation, prior) = math.import_gdn_conv_input(input, channels)?;
        let values = crate::forward::gdn_convolution_silu_slice(
            self,
            &mut math,
            layer,
            &activation,
            &prior,
        )?;
        if cancelled.is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State("window convolution slice cancelled".into()));
        }
        Ok(GdnConvolutionSiluOutput {
            source: self.source_identity(),
            layer,
            phase: input.phase(),
            input: activation,
            prior,
            values,
        })
    }
}
