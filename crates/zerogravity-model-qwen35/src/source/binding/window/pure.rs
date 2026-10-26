//! Config/header-bound roles only; source bytes and CPU arithmetic stay unchanged.
use super::*;
use crate::accelerator::program::OrdinaryRmsContract;
use zerogravity_core::quant::QuantKind;

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
            || tensor.kind != QuantKind::Bf16
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
    pub(crate) fn ordinary_hidden_role(&self, role: QwenParameterRole) -> Result<SourceParameter> {
        if !matches!(
            role,
            QwenParameterRole::InputNorm(_)
                | QwenParameterRole::FfnNorm(_)
                | QwenParameterRole::FinalNorm
        ) {
            return Err(Error::Unsupported("pure ordinary hidden norm role".into()));
        }
        let parameter = self.parameter_descriptor(role)?;
        let hidden = usize::try_from(self.geometry().hidden_size)
            .map_err(|_| Error::Limit("ordinary hidden norm width".into()))?;
        if parameter.source != self.source_identity()
            || OrdinaryRmsContract::TorchCpuBf16Avx2.validate(&parameter, hidden)? != hidden
        {
            return Err(Error::State(
                "ordinary hidden norm full raw parameter".into(),
            ));
        }
        Ok(parameter)
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn validate_ordinary_parameter(
        &self,
        parameter: &SourceParameter,
        elements: usize,
    ) -> Result<()> {
        let hidden = usize::try_from(self.geometry().hidden_size)
            .map_err(|_| Error::Limit("ordinary hidden norm width".into()))?;
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
                || OrdinaryRmsContract::TorchCpuBf16Avx2.validate(parameter, elements)? != hidden
            {
                return Err(Error::State(
                    "ordinary hidden norm exact source/width".into(),
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
            "window ordinary RMS original role/heads".into(),
        ))
    }
}
