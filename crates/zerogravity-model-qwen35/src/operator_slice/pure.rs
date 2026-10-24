//! One pure projection or ordinary norm on explicit private BF16 input.
//! No producer, session, activation nonlinearities or model execution is asserted.
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
use crate::accelerator::program::{PlanBuilder, PlanValue};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, Result};

/// External raw transport only. Caller independently binds archive/source and
/// producer provenance; this type validates representation and source/profile.
#[derive(Clone, Copy)]
pub struct PureSliceInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    bytes: &'a [u8],
}
impl<'a> PureSliceInput<'a> {
    pub fn from_bf16_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        bytes: &'a [u8],
    ) -> Result<Self> {
        let input = Self {
            source,
            profile,
            bytes,
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
            return Err(Error::State("pure input source/profile".into()));
        }
        if self.bytes.is_empty()
            || self.bytes.len() > 2 * 17408
            || !self.bytes.len().is_multiple_of(2)
            || self
                .bytes
                .chunks_exact(2)
                .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(Error::Malformed(
                "pure input bounded finite BF16 bytes".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PureSliceRole {
    Projection(QwenMatrixRole),
    OrdinaryRms(QwenParameterRole),
}

#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl crate::original_binding::OriginalWindowOperators {
    /// Caller retains its existing same-builder producer and source/metadata.
    /// No external-input node, root or complete model plan is manufactured.
    pub fn append_projection_slice(
        &self,
        builder: &mut PlanBuilder,
        role: QwenMatrixRole,
        input: &PlanValue,
    ) -> Result<PlanValue> {
        self.projection_role(role)?;
        crate::forward::pure_projection_slice(self, builder, role, input)
    }
    pub fn append_ordinary_rms_slice(
        &self,
        builder: &mut PlanBuilder,
        role: QwenParameterRole,
        input: &PlanValue,
    ) -> Result<PlanValue> {
        self.ordinary_hidden_role(role)?;
        crate::forward::pure_ordinary_rms_slice(self, builder, role, input)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_projection_slice(
        &self,
        role: QwenMatrixRole,
        input: PureSliceInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<PureSliceOutput> {
        let matrix = self.projection_role(role)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let input = math.import_pure_input(input, matrix.columns)?;
        let output = crate::forward::pure_projection_slice(self, &mut math, role, &input)?;
        finish_pure_output(
            PureSliceOutput {
                source: self.source().source_identity(),
                role: PureSliceRole::Projection(role),
                input,
                output,
            },
            cancelled,
        )
    }
    #[cfg(feature = "cpu")]
    pub fn execute_ordinary_rms_slice(
        &self,
        role: QwenParameterRole,
        input: PureSliceInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<PureSliceOutput> {
        use crate::operator_slice::OperatorSliceSource;
        self.ordinary_hidden_role(role)?;
        let hidden = usize::try_from(self.geometry().hidden_size)
            .map_err(|_| Error::Limit("ordinary pure input width".into()))?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let input = math.import_pure_input(input, hidden)?;
        let output = crate::forward::pure_ordinary_rms_slice(self, &mut math, role, &input)?;
        finish_pure_output(
            PureSliceOutput {
                source: self.source().source_identity(),
                role: PureSliceRole::OrdinaryRms(role),
                input,
                output,
            },
            cancelled,
        )
    }
}

/// Retained original-source input/output backing, borrowed access only.
#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
pub struct PureSliceOutput {
    source: SourceIdentity,
    role: PureSliceRole,
    input: crate::host_math::HostValue,
    output: crate::host_math::HostValue,
}
#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
impl PureSliceOutput {
    pub fn role(&self) -> PureSliceRole {
        self.role
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
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
    pub fn output(&self) -> Result<&[f32]> {
        self.output.as_slice()
    }
}

/// Boundary after all synchronous math/RNE/validation, before private return.
#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
pub(crate) fn finish_pure_output(
    output: PureSliceOutput,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<PureSliceOutput> {
    if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
        return Err(Error::State(
            "pure window output cancelled before return".into(),
        ));
    }
    Ok(output)
}
