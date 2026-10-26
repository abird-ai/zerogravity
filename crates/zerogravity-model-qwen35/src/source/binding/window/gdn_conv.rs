//! Original full rank3 stage; arithmetic remains in accepted CPU helpers.
use super::*;
impl OriginalWindowOperators {
    pub(crate) fn convolve(
        &self,
        parameter: &SourceParameter,
        input: &[f32],
        history: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<zerogravity_cpu::numerics::torch_cpu::conv::CausalConv4Completed> {
        let mut found = false;
        for (layer, kind) in self.geometry().layers.iter().enumerate() {
            if *kind != crate::config::LayerKind::LinearAttention {
                continue;
            }
            let (actual, channels) =
                crate::operator_slice::gdn_conv::convolution_role(self, layer)?;
            if actual == *parameter && input.len() == channels && history.len() == channels * 3 {
                found = true;
                break;
            }
        }
        if !found {
            return Err(Error::State(
                "window convolution original role/geometry".into(),
            ));
        }
        check_cancel(cancelled)?;
        let raw = self.source().conv4_parameter(&parameter.name)?;
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
                "window convolution incomplete raw parameter".into(),
            ));
        }
        let output = zerogravity_cpu::numerics::torch_cpu::conv::causal_step_bf16_4(
            zerogravity_cpu::numerics::torch_cpu::conv::CausalConv4Inputs {
                source: self.source_identity(),
                tensor_name: &parameter.name,
                weight: zerogravity_core::source::SourceTensorRef {
                    kind: raw.tensor().kind,
                    shape: &raw.tensor().shape,
                    packed_bytes: window.as_slice(),
                },
                activation: input,
                history,
            },
        )?;
        check_cancel(cancelled)?;
        Ok(output)
    }
}
