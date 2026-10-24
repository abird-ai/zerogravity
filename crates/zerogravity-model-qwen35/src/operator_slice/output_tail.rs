//! Computed final FFN/residual/norm/full-head tail from one explicit boundary.
//! The boundary is not a trained model producer, StateRoot or emission permit.
use super::OperatorSliceSource;
#[cfg(feature = "cpu")]
use super::pure::PureSliceInput;
#[cfg(feature = "cpu")]
use crate::accelerator::program::ProgramNumericalProfile;
use crate::accelerator::program::{PlanBuilder, PlanValue};
use crate::original_binding::{OriginalWindowOperators, QwenMatrixRole, QwenParameterRole};
use crate::{Error, Result};

/// Same backend values in execution order; no independent evaluator or copied weights.
pub struct OutputTailValues<V> {
    pub normalized_ffn: V,
    pub gate: V,
    pub up: V,
    pub gated: V,
    pub branch: V,
    pub residual: V,
    pub normalized: V,
    pub logits: V,
}
impl OriginalWindowOperators {
    pub(crate) fn output_tail_layer(&self) -> Result<usize> {
        let geometry = self.geometry();
        let layer = geometry
            .layers
            .len()
            .checked_sub(1)
            .ok_or_else(|| Error::State("output tail requires final source layer".into()))?;
        let hidden = usize::try_from(geometry.hidden_size)
            .map_err(|_| Error::Limit("output tail hidden width".into()))?;
        if layer >= 128 || !matches!(hidden, 256 | 1024 | 2048 | 5120) {
            return Err(Error::Unsupported("closed CPU output tail geometry".into()));
        }
        let gate = self.projection_role(QwenMatrixRole::FfnGate(layer))?;
        let up = self.projection_role(QwenMatrixRole::FfnUp(layer))?;
        let down = self.projection_role(QwenMatrixRole::FfnDown(layer))?;
        let head = self.projection_role(QwenMatrixRole::Output)?;
        self.ordinary_hidden_role(QwenParameterRole::FfnNorm(layer))?;
        self.ordinary_hidden_role(QwenParameterRole::FinalNorm)?;
        if gate.columns != hidden
            || up.columns != hidden
            || gate.rows != up.rows
            || !gate.rows.is_multiple_of(16)
            || gate.rows > 17408
            || down.columns != gate.rows
            || down.rows != hidden
            || head.columns != hidden
            || head.rows as u64 != geometry.vocab_size
        {
            return Err(Error::State(
                "output tail complete original FFN/head roles".into(),
            ));
        }
        Ok(layer)
    }
    /// Append to a genuine same-builder producer; no external-input node is invented.
    /// Caller retains the source issuer/metadata through recorded-plan use.
    pub fn append_output_tail(
        &self,
        builder: &mut PlanBuilder,
        post_attention_residual: &PlanValue,
    ) -> Result<OutputTailValues<PlanValue>> {
        crate::forward::output_tail_slice(self, builder, post_attention_residual)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_output_tail(
        &self,
        post_attention_residual: PureSliceInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<WindowOutputTail> {
        if cancelled.is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State("original CPU output tail cancelled".into()));
        }
        let layer = self.output_tail_layer()?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let input = math.import_pure_input(
            post_attention_residual,
            self.geometry().hidden_size as usize,
        )?;
        let values = crate::forward::output_tail_slice(self, &mut math, &input)?;
        if cancelled.is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "original CPU output tail cancelled before return".into(),
            ));
        }
        Ok(WindowOutputTail {
            source: self.source_identity(),
            layer,
            input,
            values,
        })
    }
}

/// Borrowed complete outputs retain admitted backing and the original source owner.
#[cfg(feature = "cpu")]
pub struct WindowOutputTail {
    source: crate::accelerator::SourceIdentity,
    layer: usize,
    input: crate::host_math::HostValue,
    values: OutputTailValues<crate::host_math::HostValue>,
}
#[cfg(feature = "cpu")]
impl WindowOutputTail {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn input(&self) -> Result<&[f32]> {
        self.input.as_slice()
    }
    pub fn normalized_ffn(&self) -> Result<&[f32]> {
        self.values.normalized_ffn.as_slice()
    }
    pub fn gate(&self) -> Result<&[f32]> {
        self.values.gate.as_slice()
    }
    pub fn up(&self) -> Result<&[f32]> {
        self.values.up.as_slice()
    }
    pub fn gated(&self) -> Result<&[f32]> {
        self.values.gated.as_slice()
    }
    pub fn branch(&self) -> Result<&[f32]> {
        self.values.branch.as_slice()
    }
    pub fn residual(&self) -> Result<&[f32]> {
        self.values.residual.as_slice()
    }
    pub fn normalized(&self) -> Result<&[f32]> {
        self.values.normalized.as_slice()
    }
    pub fn full_vocabulary(&self) -> Result<&[f32]> {
        self.values.logits.as_slice()
    }
}
