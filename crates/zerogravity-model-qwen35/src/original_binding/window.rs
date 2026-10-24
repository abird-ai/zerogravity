//! Pure original operators over one sealed source, never a model/session.
use super::*;
use crate::accelerator::program::{
    EmbeddingContract, PlanBuilder, ProgramNumericalProfile, ProjectionContract,
    RecordedOperatorSlice, SourceParameter,
};
use crate::operator_slice::OperatorSliceSource;
#[cfg(feature = "cpu")]
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::model::qwen_owned::QwenOriginalWindowText;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};

pub struct OriginalWindowOperators {
    checked: QwenOriginalWindowText,
    binding: OriginalQwenBinding,
    _metadata: WindowWorkspace,
}
impl OriginalWindowOperators {
    pub fn bind(
        source: VerifiedTensorSource,
        config_json: &[u8],
        config_sha256: &str,
        index_sha256: &str,
        profile: QwenNumericalProfile,
    ) -> Result<Self> {
        if profile != QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2 {
            return Err(Error::Unsupported(
                "window operators require the explicit pinned CPU profile".into(),
            ));
        }
        let checked =
            QwenOriginalWindowText::bind(source, config_json, config_sha256, index_sha256)?;
        let metadata = checked.source().reserve_workspace(4 * 1024 * 1024)?;
        let binding = OriginalQwenBinding::bind_resolved(
            config_json,
            BindingPins {
                config: checked.config_sha256(),
                index: checked.verified_index_sha256(),
                bundle: checked.source_bundle_sha256(),
                text: checked.text_tensor_count(),
                other: checked.other_tensor_count(),
            },
            |name| {
                checked.text_tensor(name)?;
                let matrix = checked.source().matrix(name, MatrixAxes::OutputInput)?;
                Ok(SourceTensor {
                    source: checked.source_identity(),
                    name: name.into(),
                    kind: matrix.tensor().kind,
                    columns: matrix.columns(),
                    rows: matrix.rows(),
                    axes: matrix.axes(),
                })
            },
            |name| {
                let raw = checked.text_tensor(name)?;
                Ok(SourceParameter {
                    source: checked.source_identity(),
                    name: name.into(),
                    kind: raw.kind,
                    shape: raw.shape.clone(),
                    bytes: usize::try_from(raw.byte_len)
                        .map_err(|_| Error::Limit("original parameter bytes".into()))?,
                })
            },
        )?;
        let qkv = binding
            .matrices
            .get(&QwenMatrixRole::GdnQkv(0))
            .ok_or_else(|| {
                Error::Unsupported("window slice requires original layer0 GDN".into())
            })?;
        if !matches!(qkv.columns, 256 | 1024 | 2048 | 5120)
            || qkv.rows > 16384
            || (binding.config.rms_norm_eps() as f32).to_bits() != 1e-6f32.to_bits()
        {
            return Err(Error::Unsupported(
                "closed window operator geometry/epsilon".into(),
            ));
        }
        Ok(Self {
            checked,
            binding,
            _metadata: metadata,
        })
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        self.checked.source()
    }
    pub fn binding(&self) -> &OriginalQwenBinding {
        &self.binding
    }
    pub fn record(&self) -> Result<WindowOperatorPlan> {
        let metadata = self.source().reserve_workspace(1024 * 1024)?;
        let mut builder = PlanBuilder::new_operator_slice(
            self.source_identity(),
            self.program_numerical_profile(),
        )?;
        let (_, _, output) = crate::forward::embedding_input_norm_projection_slice(
            self,
            &mut builder,
            PlanBuilder::inputs().token,
        )?;
        let recorded = builder.finish_operator_slice(output)?;
        Ok(WindowOperatorPlan {
            recorded,
            _metadata: metadata,
        })
    }
    pub(crate) fn validate_matrix(&self, tensor: &SourceTensor) -> Result<()> {
        if !self
            .binding
            .matrices
            .values()
            .any(|actual| actual == tensor)
        {
            return Err(Error::State(
                "window matrix differs from original role descriptor".into(),
            ));
        }
        Ok(())
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn workspace(
        &self,
        output_elements: usize,
        scratch_elements: usize,
    ) -> Result<WindowWorkspace> {
        let bytes = |n: usize| {
            n.checked_mul(std::mem::size_of::<f32>())
                .filter(|&n| n > 0)
                .ok_or_else(|| Error::Limit("window operator buffer bytes".into()))
        };
        self.source().reserve_operator_buffers(
            bytes(output_elements)?.max(64 * 1024),
            bytes(scratch_elements)?.max(960 * 1024),
        )
    }
    #[cfg(feature = "cpu")]
    pub fn execute(
        &self,
        token: u32,
        cancelled: Option<&AtomicBool>,
    ) -> Result<WindowOperatorOutput> {
        check_cancel(cancelled)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let (embedding, normalized, projected) =
            crate::forward::embedding_input_norm_projection_slice(self, &mut math, token as usize)?;
        check_cancel(cancelled)?;
        Ok(WindowOperatorOutput {
            source: self.source_identity(),
            embedding,
            normalized,
            projected,
        })
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn embed(
        &self,
        tensor: &SourceTensor,
        token: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        self.validate_matrix(tensor)?;
        if *tensor != self.matrix_descriptor(QwenMatrixRole::Embedding)? {
            return Err(Error::Unsupported("window embedding role".into()));
        }
        check_cancel(cancelled)?;
        let matrix = self
            .source()
            .matrix(&tensor.name, MatrixAxes::OutputInput)?;
        let window = match cancelled {
            Some(flag) => self.source().read_rows_cancellable(
                &matrix,
                token
                    ..token
                        .checked_add(1)
                        .ok_or_else(|| Error::Limit("window embedding token".into()))?,
                flag,
            )?,
            None => self.source().read_rows(
                &matrix,
                token
                    ..token
                        .checked_add(1)
                        .ok_or_else(|| Error::Limit("window embedding token".into()))?,
            )?,
        };
        zerogravity_cpu::original_hf::window_embedding::embedding_bf16_window(
            tensor,
            u32::try_from(token)
                .map_err(|_| Error::Limit("window embedding token domain".into()))?,
            &window,
            None,
            cancelled,
        )
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn normalize(
        &self,
        parameter: &SourceParameter,
        input: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        self.validate_ordinary_parameter(parameter, input.len())?;
        check_cancel(cancelled)?;
        let raw = self.source().parameter(&parameter.name)?;
        raw.validate_source_parameter(parameter)?;
        let window = match cancelled {
            Some(flag) => {
                self.source()
                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
            }
            None => self.source().read_parameter(&raw, 0..raw.elements())?,
        };
        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
            return Err(Error::State(
                "window ordinary RMS incomplete original parameter".into(),
            ));
        }
        let tensor = zerogravity_core::source::SourceTensorRef {
            kind: raw.tensor().kind,
            shape: &raw.tensor().shape,
            packed_bytes: window.as_slice(),
        };
        let output = zerogravity_cpu::original_hf::norm::ordinary_rms(
            self.source_identity(),
            tensor,
            input,
            raw.elements(),
        )?;
        check_cancel(cancelled)?;
        Ok(output.output_bf16().to_vec())
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn project(
        &self,
        tensor: &SourceTensor,
        input: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        self.validate_projection_matrix(tensor)?;
        check_cancel(cancelled)?;
        if input.len() != tensor.columns {
            return Err(Error::Malformed(
                "window projection original full input width".into(),
            ));
        }
        let matrix = self
            .source()
            .matrix(&tensor.name, MatrixAxes::OutputInput)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(tensor.rows)
            .map_err(|_| Error::Limit("private window complete output".into()))?;
        for start in (0..tensor.rows).step_by(128) {
            check_cancel(cancelled)?;
            let rows = start..(start + 128).min(tensor.rows);
            let window = match cancelled {
                Some(flag) => self
                    .source()
                    .read_rows_cancellable(&matrix, rows.clone(), flag)?,
                None => self.source().read_rows(&matrix, rows.clone())?,
            };
            let result = zerogravity_cpu::original_hf::window_projection::project_bf16_window(
                tensor,
                rows.clone(),
                &window,
                input,
                None,
                cancelled,
            )?;
            if result.identity().descriptor() != tensor
                || result.identity().row_range() != rows
                || output.len() != start
                || result.accumulations().len() != rows.len()
            {
                return Err(Error::State("window projection complete row cover".into()));
            }
            output.extend_from_slice(result.accumulations());
        }
        check_cancel(cancelled)?;
        if output.len() != tensor.rows {
            return Err(Error::State("window projection missing rows".into()));
        }
        Ok(output)
    }
}
impl OperatorSliceSource for OriginalWindowOperators {
    fn geometry(&self) -> &crate::model::ModelGeometry {
        self.binding.config.geometry()
    }
    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.checked.source_identity()
    }
    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    fn rms_epsilon(&self) -> f32 {
        self.binding.config.rms_norm_eps() as f32
    }
    fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
        self.binding
            .matrices
            .get(&role)
            .cloned()
            .ok_or_else(|| Error::Unsupported("absent original window matrix role".into()))
    }
    fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<SourceParameter> {
        self.binding
            .parameters
            .get(&role)
            .cloned()
            .ok_or_else(|| Error::Unsupported("absent original window parameter role".into()))
    }
    fn embedding_contract(&self) -> EmbeddingContract {
        EmbeddingContract::Bf16OutputInput
    }
    fn projection_contract(&self) -> ProjectionContract {
        ProjectionContract::TorchCpuBf16Avx2
    }
}
#[cfg(feature = "cpu")]
fn check_cancel(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Err(Error::State("original window operator cancelled".into()))
    } else {
        Ok(())
    }
}

/// Same recorded pure slice, with retained bounded metadata/source ownership.
pub struct WindowOperatorPlan {
    recorded: RecordedOperatorSlice,
    _metadata: WindowWorkspace,
}
impl WindowOperatorPlan {
    pub fn plan(&self) -> &crate::accelerator::program::ForwardPlan {
        self.recorded.plan()
    }
    pub fn recorded(&self) -> &RecordedOperatorSlice {
        &self.recorded
    }
}

#[cfg(feature = "cpu")]
pub struct WindowOperatorOutput {
    source: crate::accelerator::SourceIdentity,
    embedding: crate::host_math::HostValue,
    normalized: crate::host_math::HostValue,
    projected: crate::host_math::HostValue,
}
#[cfg(feature = "cpu")]
impl WindowOperatorOutput {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn embedding(&self) -> Result<&[f32]> {
        self.embedding.as_slice()
    }
    pub fn normalized(&self) -> Result<&[f32]> {
        self.normalized.as_slice()
    }
    pub fn projected(&self) -> Result<&[f32]> {
        self.projected.as_slice()
    }
}

#[cfg(feature = "cpu")]
mod gdn_conv;

#[cfg(feature = "cpu")]
pub(crate) mod full_cpu;
mod pure;

#[cfg(feature = "cpu")]
mod record_full;
#[cfg(feature = "cpu")]
pub use record_full::RecordedWindowToken;
