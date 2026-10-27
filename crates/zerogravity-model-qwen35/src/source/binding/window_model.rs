//! Immutable model metadata over ONE verified window owner; no model permit.
use super::*;
use crate::source::owned::QwenOriginalWindowText;
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};

/// Same indexed model/source/state geometry as the owned-source route. Holds
/// no learned cache, cursor, physical buffer, SessionAuthority or StateRoot.
pub struct OriginalWindowModelSource {
    roles: ResolvedQwenRoles,
    state_specs: Vec<crate::StateSpec>,
    owner: Arc<QwenOriginalWindowText>,
    // Actual resolved role/state payloads drop before original admission.
    _metadata: WindowWorkspace,
}
impl OriginalWindowModelSource {
    /// Reuse the caller's already-authenticated config/index/source issuer.
    /// No reopen, config reparse, source-ID mint, SHA-based retag or byte copy.
    pub fn from_owner(owner: Arc<QwenOriginalWindowText>) -> Result<Arc<Self>> {
        let metadata = owner.source().reserve_workspace(4 * 1024 * 1024)?;
        let roles = resolve_qwen_roles(
            owner.config(),
            |name| {
                owner.text_tensor(name)?;
                let actual = owner.source().matrix(name, MatrixAxes::OutputInput)?;
                Ok(SourceTensor {
                    source: owner.source_identity(),
                    name: name.into(),
                    kind: actual.tensor().kind,
                    axes: actual.axes(),
                    rows: actual.rows(),
                    columns: actual.columns(),
                })
            },
            |name| {
                let raw = owner.text_tensor(name)?;
                Ok(QwenParameterDescriptor {
                    source: owner.source_identity(),
                    name: name.into(),
                    kind: raw.kind,
                    shape: raw.shape.clone(),
                    bytes: usize::try_from(raw.byte_len)
                        .map_err(|_| Error::Limit("original parameter bytes".into()))?,
                })
            },
        )?;
        let state_specs = original_state_specs(owner.config().geometry(), roles.gdn)?;
        Ok(Arc::new(Self {
            roles,
            state_specs,
            owner,
            _metadata: metadata,
        }))
    }
    pub fn owner(&self) -> &Arc<QwenOriginalWindowText> {
        &self.owner
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        self.owner.source()
    }
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source().source_identity()
    }
    pub fn geometry(&self) -> &crate::config::ModelGeometry {
        self.owner.config().geometry()
    }
    pub fn state_specs(&self) -> &[crate::StateSpec] {
        &self.state_specs
    }
    pub fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
        self.roles.matrices.get(&role).cloned().ok_or_else(|| {
            Error::Unsupported("matrix role absent from original window model".into())
        })
    }
    /// Family semantics only; residency and transport belong to the engine.
    pub fn tied_output_descriptor(&self) -> Result<SourceTensor> {
        let embedding = self.matrix_descriptor(QwenMatrixRole::Embedding)?;
        let output = self.matrix_descriptor(QwenMatrixRole::Output)?;
        if !self.owner.config().tie_word_embeddings()
            || embedding != output
            || embedding.rows as u64 != self.geometry().vocab_size
            || embedding.columns as u64 != self.geometry().hidden_size
        {
            return Err(Error::Unsupported(
                "original Qwen tied embedding/output roles and configured geometry".into(),
            ));
        }
        self.validate_matrix(QwenMatrixRole::Embedding, &embedding)?;
        self.validate_matrix(QwenMatrixRole::Output, &embedding)?;
        Ok(embedding)
    }
    pub fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<QwenParameterDescriptor> {
        self.roles.parameters.get(&role).cloned().ok_or_else(|| {
            Error::Unsupported("parameter role absent from original window model".into())
        })
    }
    pub fn validate_matrix(&self, role: QwenMatrixRole, descriptor: &SourceTensor) -> Result<()> {
        if descriptor.source != self.source_identity()
            || self.roles.matrices.get(&role) != Some(descriptor)
        {
            return Err(Error::State(
                "original window model matrix role/issuer".into(),
            ));
        }
        let actual = self.source().matrix(&descriptor.name, descriptor.axes)?;
        if descriptor.kind != actual.tensor().kind
            || descriptor.rows != actual.rows()
            || descriptor.columns != actual.columns()
        {
            return Err(Error::State(
                "original window model matrix raw geometry".into(),
            ));
        }
        Ok(())
    }
    pub fn validate_parameter(
        &self,
        role: QwenParameterRole,
        descriptor: &QwenParameterDescriptor,
    ) -> Result<()> {
        if descriptor.source != self.source_identity()
            || self.roles.parameters.get(&role) != Some(descriptor)
        {
            return Err(Error::State(
                "original window model parameter role/issuer".into(),
            ));
        }
        let actual = self.owner.text_tensor(&descriptor.name)?;
        if descriptor.kind != actual.kind
            || descriptor.shape != actual.shape
            || descriptor.bytes as u64 != actual.byte_len
        {
            return Err(Error::State(
                "original window model parameter raw geometry".into(),
            ));
        }
        Ok(())
    }
    pub fn program_numerical_profile(
        &self,
    ) -> crate::accelerator::program::ProgramNumericalProfile {
        crate::accelerator::program::ProgramNumericalProfile::SourceCudaSm86Bf16V1
    }
    pub(crate) fn require_execution_profile(&self) -> Result<()> {
        Err(Error::Unsupported(
            "original window model bound; complete CUDA plan/state operators remain unqualified"
                .into(),
        ))
    }
}

mod gdn_fragment;

mod full_model_source;
mod ordinary_cuda;
mod source_plan;
pub use full_model_source::{BoundFullModelSource, BoundPromptCall};
pub use ordinary_cuda::OrdinaryCudaSourceSeal;
mod prepare;
pub use prepare::{
    ComponentImplementation, ComponentKernelCapture, ComponentOutput, PreparedComponent,
    PreparedFinalRmsSource, PreparedGdnParameters, PreparedPlan, PreparedQualificationSource,
    PreparedRmsSource, ReferenceKernelCapture,
};
