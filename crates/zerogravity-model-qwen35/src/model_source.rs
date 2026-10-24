//! Truthful shared model metadata. Never a partial TensorSource or model permit.
use crate::accelerator::program::{ProgramNumericalProfile, SourceParameter, SourceTensor};
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
use crate::original_binding::OriginalWindowModelSource;
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{LoadedQwenModel, Result, StateSpec};

/// Borrowed view used by the SAME forward header/traversal. Immutable source
/// payloads remain owned by their real model allocation or verified FD owner.
#[derive(Clone, Copy)]
pub enum QwenModelMetadata<'a> {
    Owned(&'a LoadedQwenModel),
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    VerifiedWindows(&'a OriginalWindowModelSource),
}
impl<'a> From<&'a LoadedQwenModel> for QwenModelMetadata<'a> {
    fn from(model: &'a LoadedQwenModel) -> Self {
        Self::Owned(model)
    }
}
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl<'a> From<&'a OriginalWindowModelSource> for QwenModelMetadata<'a> {
    fn from(model: &'a OriginalWindowModelSource) -> Self {
        Self::VerifiedWindows(model)
    }
}
impl<'a> QwenModelMetadata<'a> {
    pub fn source_identity(self) -> crate::accelerator::SourceIdentity {
        match self {
            Self::Owned(model) => model.source_identity(),
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::VerifiedWindows(model) => model.source_identity(),
        }
    }
    pub fn geometry(self) -> &'a crate::model::ModelGeometry {
        match self {
            Self::Owned(model) => model.geometry(),
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::VerifiedWindows(model) => model.geometry(),
        }
    }
    pub fn state_specs(self) -> &'a [StateSpec] {
        match self {
            Self::Owned(model) => model.state_specs(),
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::VerifiedWindows(model) => model.state_specs(),
        }
    }
    pub fn program_numerical_profile(self) -> ProgramNumericalProfile {
        match self {
            Self::Owned(model) => model.program_numerical_profile(),
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::VerifiedWindows(model) => model.program_numerical_profile(),
        }
    }
    pub fn matrix_descriptor(self, role: QwenMatrixRole) -> Result<SourceTensor> {
        match self {
            Self::Owned(model) => model.matrix_descriptor(role),
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::VerifiedWindows(model) => model.matrix_descriptor(role),
        }
    }
    pub fn parameter_descriptor(self, role: QwenParameterRole) -> Result<SourceParameter> {
        match self {
            Self::Owned(model) => model.parameter_descriptor(role),
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::VerifiedWindows(model) => model.parameter_descriptor(role),
        }
    }
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    pub fn original_window_owner(
        self,
    ) -> Option<&'a std::sync::Arc<zerogravity_artifacts::model::qwen_owned::QwenOriginalWindowText>>
    {
        match self {
            Self::Owned(_) => None,
            Self::VerifiedWindows(model) => Some(model.owner()),
        }
    }
    pub(crate) fn require_execution_profile(self) -> Result<()> {
        match self {
            Self::Owned(model) => model.require_execution_profile(),
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::VerifiedWindows(model) => model.require_execution_profile(),
        }
    }
    pub(crate) fn legacy_loaded(self) -> Result<&'a LoadedQwenModel> {
        match self {
            Self::Owned(model) => Ok(model),
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::VerifiedWindows(_) => Err(crate::Error::Unsupported(
                "verified windows have no legacy packed-source/cache adapter".into(),
            )),
        }
    }
}
