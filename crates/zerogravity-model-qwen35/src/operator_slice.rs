//! State-free operators through the existing schedule/backend, never a token.
use crate::accelerator::program::{
    EmbeddingContract, PlanBuilder, ProgramNumericalProfile, ProjectionContract,
    RecordedOperatorSlice,
};
use crate::original_binding::{QwenMatrixRole, QwenNumericalProfile};
use crate::{Error, LoadedQwenModel, Result};

impl LoadedQwenModel {
    /// Numerical contract only; the full execution/session guard is separate.
    pub fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        match self.numerical_profile() {
            QwenNumericalProfile::GgufSourceV1 => ProgramNumericalProfile::GgufSourceV1,
            QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2 => {
                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            }
        }
    }
    pub(crate) fn embedding_contract(&self) -> EmbeddingContract {
        match self.numerical_profile() {
            QwenNumericalProfile::GgufSourceV1 => EmbeddingContract::Source,
            QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2 => {
                EmbeddingContract::Bf16OutputInput
            }
        }
    }
    pub(crate) fn projection_contract(&self) -> ProjectionContract {
        match self.numerical_profile() {
            QwenNumericalProfile::GgufSourceV1 => ProjectionContract::Source,
            QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2 => {
                ProjectionContract::TorchCpuBf16Avx2
            }
        }
    }
    fn require_operator_slice(&self) -> Result<()> {
        if self.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported(
                "closed original embedding/projection profile".into(),
            ));
        }
        Ok(())
    }
    /// Record a pure source embedding→projection fragment, without state domains,
    /// token frontier advance, logits publication or a complete model claim.
    pub fn embedding_projection_slice(
        &self,
        role: QwenMatrixRole,
    ) -> Result<RecordedOperatorSlice> {
        self.require_operator_slice()?;
        let mut builder = PlanBuilder::new_operator_slice(
            self.source_identity(),
            self.program_numerical_profile(),
        )?;
        let (_, output) = crate::forward::embedding_projection_slice(
            self,
            &mut builder,
            PlanBuilder::inputs().token,
            role,
        )?;
        builder.finish_operator_slice(output)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_embedding_projection_slice(
        &self,
        token: u32,
        role: QwenMatrixRole,
    ) -> Result<EmbeddingProjectionOutput> {
        self.require_operator_slice()?;
        let mut math = crate::host_math::HostMath::for_operator_slice(self, None);
        let (embedding, projected) =
            crate::forward::embedding_projection_slice(self, &mut math, token as usize, role)?;
        Ok(EmbeddingProjectionOutput {
            source: self.source_identity(),
            profile: self.program_numerical_profile(),
            embedding: embedding.into_vec()?,
            projected: projected.into_vec()?,
        })
    }
    /// Record embedding→raw InputNorm(0)→GdnQkv(0), without any state effects.
    pub fn embedding_input_norm_projection_slice(&self) -> Result<RecordedOperatorSlice> {
        self.require_operator_slice()?;
        let mut builder = PlanBuilder::new_operator_slice(
            self.source_identity(),
            self.program_numerical_profile(),
        )?;
        let (_, _, output) = crate::forward::embedding_input_norm_projection_slice(
            self,
            &mut builder,
            PlanBuilder::inputs().token,
        )?;
        builder.finish_operator_slice(output)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_embedding_input_norm_projection_slice(
        &self,
        token: u32,
    ) -> Result<EmbeddingInputNormProjectionOutput> {
        self.require_operator_slice()?;
        let mut math = crate::host_math::HostMath::for_operator_slice(self, None);
        let (embedding, normalized, projected) =
            crate::forward::embedding_input_norm_projection_slice(self, &mut math, token as usize)?;
        Ok(EmbeddingInputNormProjectionOutput {
            source: self.source_identity(),
            profile: self.program_numerical_profile(),
            embedding: embedding.into_vec()?,
            normalized: normalized.into_vec()?,
            projected: projected.into_vec()?,
        })
    }
}

/// Private source-bound prefix values; no session, StateRoot or logits receipt.
#[cfg(feature = "cpu")]
pub struct EmbeddingInputNormProjectionOutput {
    source: crate::accelerator::SourceIdentity,
    profile: ProgramNumericalProfile,
    embedding: Vec<f32>,
    normalized: Vec<f32>,
    projected: Vec<f32>,
}
#[cfg(feature = "cpu")]
impl EmbeddingInputNormProjectionOutput {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn precision(&self) -> crate::accelerator::program::ActivationPrecision {
        crate::accelerator::program::ActivationPrecision::Bf16WidenedF32
    }
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }
    pub fn normalized(&self) -> &[f32] {
        &self.normalized
    }
    pub fn projected(&self) -> &[f32] {
        &self.projected
    }
}

/// Private source-bound operator results; no SessionAuthority or root receipt.
#[cfg(feature = "cpu")]
pub struct EmbeddingProjectionOutput {
    source: crate::accelerator::SourceIdentity,
    profile: ProgramNumericalProfile,
    embedding: Vec<f32>,
    projected: Vec<f32>,
}
#[cfg(feature = "cpu")]
impl EmbeddingProjectionOutput {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn precision(&self) -> crate::accelerator::program::ActivationPrecision {
        crate::accelerator::program::ActivationPrecision::Bf16WidenedF32
    }
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }
    pub fn projected(&self) -> &[f32] {
        &self.projected
    }
}

#[cfg(all(
    test,
    feature = "cpu",
    feature = "original-hf-binding",
    target_os = "linux"
))]
#[path = "operator_slice_tests.rs"]
mod tests;
