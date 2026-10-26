//! Staged BF16 CPU pointwise math in the existing source-bound backend.
use super::*;
use crate::operator_slice::pure::PureSliceInput;
use std::sync::atomic::Ordering;

impl HostMath<'_, '_, '_> {
    pub(super) fn check_torch_cancel(&self) -> Result<()> {
        if self
            .cancelled
            .is_some_and(|flag| flag.load(Ordering::Acquire))
        {
            return Err(Error::State("original CPU output tail cancelled".into()));
        }
        Ok(())
    }
    pub(crate) fn import_pure_input(
        &self,
        input: PureSliceInput<'_>,
        elements: usize,
    ) -> Result<HostValue> {
        self.check_torch_cancel()?;
        input.validate()?;
        if self.region_kernel.is_some()
            || input.source_identity() != self.source.source_identity()
            || input.numerical_profile() != self.source.program_numerical_profile()
            || input.bytes().len()
                != elements
                    .checked_mul(2)
                    .ok_or_else(|| Error::Limit("pure input bytes".into()))?
        {
            return Err(Error::State(
                "pure input provider/source/profile/full width".into(),
            ));
        }
        let workspace = self.prepare_output(None, elements, 1)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| Error::Limit("private pure input transport".into()))?;
        if values.capacity() != elements {
            return Err(Error::Limit("pure input capacity exceeds admission".into()));
        }
        values.extend(
            input
                .bytes()
                .chunks_exact(2)
                .map(|w| f32::from_bits(u32::from(u16::from_le_bytes([w[0], w[1]])) << 16)),
        );
        self.check_torch_cancel()?;
        HostValue::typed(
            values,
            input.precision(),
            input.source_identity(),
            input.numerical_profile(),
        )?
        .retain_workspace(workspace)
    }
    fn require_torch_values(&self, inputs: [&HostValue; 2], silu: bool) -> Result<()> {
        self.check_torch_cancel()?;
        if self.region_kernel.is_some()
            || self.source.program_numerical_profile()
                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || inputs[0].elements == 0
            || inputs[0].elements > 17408
            || inputs[0].elements != inputs[1].elements
            || (silu && !inputs[0].elements.is_multiple_of(16))
        {
            return Err(Error::Unsupported(
                "closed Torch CPU pointwise profile/shape/provider".into(),
            ));
        }
        for value in inputs {
            if value.precision != ActivationPrecision::Bf16WidenedF32
                || value.source != Some(self.source.source_identity())
                || value.profile != Some(self.source.program_numerical_profile())
            {
                return Err(Error::State(
                    "Torch CPU pointwise input source/precision".into(),
                ));
            }
            ActivationPrecision::Bf16WidenedF32.validate_values(value.as_slice()?)?;
            if let HostSource::Windows(source) = &self.source
                && value
                    .data
                    .workspace
                    .as_ref()
                    .is_none_or(|lease| !source.source().same_issuer(lease.source()))
            {
                return Err(Error::State(
                    "Torch CPU pointwise source lease missing".into(),
                ));
            }
        }
        Ok(())
    }
    pub(super) fn execute_torch_pointwise(
        &self,
        left: &HostValue,
        right: &HostValue,
        silu: bool,
    ) -> Result<HostValue> {
        self.require_torch_values([left, right], silu)?;
        let count = left.elements;
        let workspace = self.prepare_output(Some(left), count, count)?;
        let mut output = if silu {
            zerogravity_cpu::numerics::torch_cpu::transcendentals::silu_bf16_widened_f32_contiguous16(
                self.source.source_identity(),
                left.as_slice()?,
            )?
        } else {
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| Error::Limit("CPU residual output".into()))?;
            values.extend_from_slice(left.as_slice()?);
            values
        };
        if output.len() != count || output.capacity() != count {
            return Err(Error::Limit(
                "Torch CPU pointwise complete output capacity".into(),
            ));
        }
        for (index, (value, right)) in output.iter_mut().zip(right.as_slice()?).enumerate() {
            if index.is_multiple_of(256) {
                self.check_torch_cancel()?;
            }
            *value = if silu {
                // Public BF16 SiLU materializes before the separate BF16 multiply.
                round_bf16(round_bf16(*value)? * right)?
            } else {
                round_bf16(*value + right)?
            };
        }
        self.check_torch_cancel()?;
        HostValue::typed(
            output,
            ActivationPrecision::Bf16WidenedF32,
            self.source.source_identity(),
            self.source.program_numerical_profile(),
        )?
        .retain_workspace(workspace)
    }
}
