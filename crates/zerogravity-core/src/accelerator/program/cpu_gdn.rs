//! Explicit diagnostic original CPU GDN; no full-model or native permit.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TorchCpuGdnPhase {
    InitialOne,
    CachedOne,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TorchCpuGdnContract {
    pub standalone_exp: StandaloneExpProfile,
    pub convolution: SourceParameter,
    pub a_log: SourceParameter,
    pub dt_bias: SourceParameter,
    pub norm: SourceParameter,
    pub key_heads: usize,
    pub value_heads: usize,
}
impl TorchCpuGdnContract {
    pub const fn arithmetic_identity(&self) -> (ProgramNumericalProfile, u64) {
        (
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            self.standalone_exp.arithmetic_revision(),
        )
    }

    pub fn validate(&self, source: SourceIdentity) -> Result<(usize, usize, usize)> {
        if !source.is_bound()
            || self.key_heads == 0
            || self.key_heads > 16
            || self.value_heads == 0
            || self.value_heads > 48
            || !self.value_heads.is_multiple_of(self.key_heads)
            || !self.value_heads.is_multiple_of(16)
        {
            return Err(Error::Unsupported(
                "CPU GDN authentic grouped-head geometry".into(),
            ));
        }
        let channels = (self.key_heads * 2 + self.value_heads) * 128;
        let value = self.value_heads * 128;
        for (p, shape, kind) in [
            (
                &self.convolution,
                vec![channels as u64, 1, 4],
                QuantKind::Bf16,
            ),
            (&self.a_log, vec![self.value_heads as u64], self.a_log.kind),
            (
                &self.dt_bias,
                vec![self.value_heads as u64],
                self.dt_bias.kind,
            ),
            (&self.norm, vec![128], self.norm.kind),
        ] {
            if p.source != source
                || p.name.is_empty()
                || p.shape != shape
                || !matches!(kind, QuantKind::Bf16 | QuantKind::F32)
                || p.kind != kind
                || p.bytes as u64 != kind.byte_len(shape.iter().product())?
            {
                return Err(Error::State(
                    "CPU GDN original raw parameter contract".into(),
                ));
            }
        }
        Ok((channels, value, value * 128))
    }
}
pub struct TorchCpuGdnInputs<'a, V> {
    pub qkv: &'a V,
    pub gate: &'a V,
    pub alpha: &'a V,
    pub beta: &'a V,
    pub history: &'a V,
    /// Original key-major FP32, never the legacy value-major state.
    pub state: &'a V,
    pub phase: TorchCpuGdnPhase,
}
pub struct TorchCpuGdnValues<V> {
    pub convolution: V,
    pub activated: V,
    pub history: V,
    pub state: V,
    pub core: V,
    pub gated: V,
}

pub(super) fn result_precision(
    builder: &PlanBuilder,
    contract: &TorchCpuGdnContract,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    let (channels, value, state) = contract.validate(builder.source)?;
    let expected = [
        channels,
        value,
        contract.value_heads,
        contract.value_heads,
        channels * 3,
        state,
    ];
    if builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || builder.standalone_exp != Some(contract.standalone_exp)
        || inputs.len() != expected.len()
        || inputs.iter().zip(expected).enumerate().any(|(i, (v, n))| {
            v.elements != n
                || v.precision
                    != if i == 5 {
                        ActivationPrecision::F32
                    } else {
                        ActivationPrecision::Bf16WidenedF32
                    }
        })
    {
        return Err(Error::State(
            "source GDN profile/explicit precision/extent".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn full_convolution_precision(
    builder: &PlanBuilder,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    let [prior, minimal, input] = inputs else {
        return Err(Error::State(
            "full convolution actual producer arity".into(),
        ));
    };
    if builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || input.elements == 0
        || input.elements > 16384
        || input.elements.checked_mul(4) != Some(prior.elements)
        || input.elements.checked_mul(3) != Some(minimal.elements)
        || inputs
            .iter()
            .any(|v| v.precision != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State(
            "full convolution source geometry/precision".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn record(
    builder: &mut PlanBuilder,
    contract: &TorchCpuGdnContract,
    input: TorchCpuGdnInputs<'_, PlanValue>,
) -> Result<TorchCpuGdnValues<PlanValue>> {
    let (channels, value, state) = contract.validate(builder.source)?;
    let values = builder.record(
        &contract.convolution.name,
        PlanOperation::TypedTorchGdn {
            contract: Box::new(contract.clone()),
            phase: input.phase,
        },
        &[
            *input.qkv,
            *input.gate,
            *input.alpha,
            *input.beta,
            *input.history,
            *input.state,
        ],
        &[channels, channels, channels * 3, state, value, value].map(ValueShape::Vector),
        None,
    )?;
    let [convolution, activated, history, state, core, gated] = values
        .try_into()
        .map_err(|_| Error::State("source GDN output arity".into()))?;
    Ok(TorchCpuGdnValues {
        convolution,
        activated,
        history,
        state,
        core,
        gated,
    })
}
