//! Config/header-bound roles only; source bytes and CPU arithmetic stay unchanged.
use super::*;
use crate::accelerator::program::{TorchCpuGdnContract, ZeroCenteredRmsContract};
use zerogravity_core::encoding::StoredTensorEncoding;

const MAX_INPUT: usize = 17408;
const MAX_OUTPUT: usize = 248320;

impl OriginalWindowOperators {
    pub(crate) fn projection_role(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
        if role == QwenMatrixRole::Embedding {
            return Err(Error::Unsupported(
                "embedding is not a projection role".into(),
            ));
        }
        let tensor = self.matrix_descriptor(role)?;
        self.validate_projection_matrix(&tensor)?;
        Ok(tensor)
    }
    pub(crate) fn validate_projection_matrix(&self, tensor: &SourceTensor) -> Result<()> {
        self.validate_matrix(tensor)?;
        // Complete config/index/header closure issued every original role.
        // Never rename a tensor or present partial geometry as its full source.
        if !self
            .binding
            .matrices
            .iter()
            .any(|(role, actual)| *role != QwenMatrixRole::Embedding && actual == tensor)
            || tensor.source != self.source_identity()
            || tensor.kind
                != StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16)
            || tensor.axes != MatrixAxes::OutputInput
            || tensor.columns == 0
            || tensor.columns > MAX_INPUT
            || tensor.rows == 0
            || tensor.rows > MAX_OUTPUT
        {
            return Err(Error::Unsupported(
                "closed original window pure matrix role/geometry".into(),
            ));
        }
        tensor
            .kind
            .byte_len(tensor.columns as u64)?
            .checked_mul(tensor.rows as u64)
            .ok_or_else(|| Error::Limit("pure original full matrix bytes".into()))?;
        Ok(())
    }
    pub(crate) fn zero_centered_hidden_role(
        &self,
        role: QwenParameterRole,
    ) -> Result<SourceParameter> {
        if !matches!(
            role,
            QwenParameterRole::InputNorm(_)
                | QwenParameterRole::FfnNorm(_)
                | QwenParameterRole::FinalNorm
        ) {
            return Err(Error::Unsupported(
                "pure zero-centered hidden norm role".into(),
            ));
        }
        let parameter = self.parameter_descriptor(role)?;
        let hidden = usize::try_from(self.geometry().hidden_size)
            .map_err(|_| Error::Limit("zero-centered hidden norm width".into()))?;
        if parameter.source != self.source_identity()
            || ZeroCenteredRmsContract::TorchCpuBf16Avx2.validate(&parameter, hidden)? != hidden
        {
            return Err(Error::State(
                "zero-centered hidden norm full raw parameter".into(),
            ));
        }
        Ok(parameter)
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn validate_zero_centered_parameter(
        &self,
        parameter: &SourceParameter,
        elements: usize,
    ) -> Result<()> {
        let hidden = usize::try_from(self.geometry().hidden_size)
            .map_err(|_| Error::Limit("zero-centered hidden norm width".into()))?;
        if elements == hidden
            && self.binding.parameters.iter().any(|(role, actual)| {
                matches!(
                    role,
                    QwenParameterRole::InputNorm(_)
                        | QwenParameterRole::FfnNorm(_)
                        | QwenParameterRole::FinalNorm
                ) && actual == parameter
            })
        {
            if parameter.source != self.source_identity()
                || ZeroCenteredRmsContract::TorchCpuBf16Avx2.validate(parameter, elements)?
                    != hidden
            {
                return Err(Error::State(
                    "zero-centered hidden norm exact source/width".into(),
                ));
            }
            return Ok(());
        }
        // Existing qualified attention Q/K heads retain their original guards.
        for (layer, kind) in self.geometry().layers.iter().enumerate() {
            if *kind != crate::config::LayerKind::FullAttention {
                continue;
            }
            let roles = crate::operator_slice::attention::attention_roles(self, layer)?;
            let expected = if *parameter == roles.query_norm {
                Some(roles.query_heads * roles.width)
            } else if *parameter == roles.key_norm {
                Some(roles.kv_heads * roles.width)
            } else {
                None
            };
            if expected == Some(elements) {
                return Ok(());
            }
        }
        Err(Error::State(
            "window zero-centered RMS original role/heads".into(),
        ))
    }
}

impl OriginalWindowOperators {
    pub(crate) fn gdn_contract(&self, layer: usize) -> Result<TorchCpuGdnContract> {
        if self.geometry().layers.get(layer) != Some(&crate::config::LayerKind::LinearAttention) {
            return Err(Error::Unsupported("diagnostic GDN source layer".into()));
        }
        let g = self.binding.gdn_geometry();
        if g.key_width != 128 || g.value_width != 128 || g.convolution_kernel != 4 {
            return Err(Error::Unsupported(
                "diagnostic GDN128/four-tap source geometry".into(),
            ));
        }
        let roles = crate::hybrid_decoder::ParameterGroup::GdnAll.roles(layer);
        let [convolution, a_log, dt_bias, norm]: [crate::accelerator::program::SourceParameter; 4] =
            roles
                .into_iter()
                .map(|role| self.parameter_descriptor(role))
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| Error::State("family GDN parameter group arity".into()))?;
        let c = TorchCpuGdnContract {
            standalone_exp:
                crate::accelerator::program::StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1,
            convolution,
            a_log,
            dt_bias,
            norm,
            key_heads: g.key_heads,
            value_heads: g.value_heads,
        };
        let (channels, value, _) = c.validate(self.source_identity())?;
        for (role, rows) in [
            (QwenMatrixRole::GdnQkv(layer), channels),
            (QwenMatrixRole::GdnGate(layer), value),
            (QwenMatrixRole::GdnAlpha(layer), g.value_heads),
            (QwenMatrixRole::GdnBeta(layer), g.value_heads),
        ] {
            let d = self.projection_role(role)?;
            if d.rows != rows || d.columns as u64 != self.geometry().hidden_size {
                return Err(Error::State("diagnostic GDN actual projection role".into()));
            }
        }
        Ok(c)
    }
}
