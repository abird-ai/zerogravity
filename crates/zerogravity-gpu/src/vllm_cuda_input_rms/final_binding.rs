//! Closed original final-norm gamma/config owner; no phase/root authority.
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::{
    OriginalWindowModelSource, QwenParameterDescriptor, QwenParameterRole,
};

struct FinalRmsOwner {
    model: Arc<OriginalWindowModelSource>,
    parameter: QwenParameterDescriptor,
    _metadata: WindowWorkspace,
}
#[derive(Clone)]
pub(crate) struct BoundFinalRmsSource(Arc<FinalRmsOwner>);
impl BoundFinalRmsSource {
    pub(crate) fn bind(model: Arc<OriginalWindowModelSource>) -> Result<Self> {
        let metadata = model.source().reserve_workspace(64 * 1024)?;
        let parameter = model.parameter_descriptor(QwenParameterRole::FinalNorm)?;
        if model.geometry().hidden_size != 1024
            || model.geometry().vocab_size != 248_320
            || model.geometry().layers.len() != 24
            || model.owner().config().rms_norm_eps() != 1e-6
            || parameter.name != "model.language_model.norm.weight"
            || parameter.kind != QuantKind::Bf16
            || parameter.shape != [1024]
            || parameter.bytes != 2048
        {
            return Err(Error::Unsupported(
                "closed original .8B final RMS source/config".into(),
            ));
        }
        model.validate_parameter(QwenParameterRole::FinalNorm, &parameter)?;
        Ok(Self(Arc::new(FinalRmsOwner {
            model,
            parameter,
            _metadata: metadata,
        })))
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        self.0.model.source()
    }
    pub(crate) fn gamma_name(&self) -> &str {
        &self.0.parameter.name
    }
    pub(crate) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if !source.same_issuer(self.source()) {
            return Err(Error::State(
                "final RMS original source issuer differs".into(),
            ));
        }
        self.0
            .model
            .validate_parameter(QwenParameterRole::FinalNorm, &self.0.parameter)
    }
}
