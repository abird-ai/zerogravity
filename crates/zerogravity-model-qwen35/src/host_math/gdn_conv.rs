//! Aggregate admitted private inputs/results for the pure original region.
use super::*;
use crate::accelerator::program::{CausalConvSiluContract, CausalConvValues};
use crate::operator_slice::gdn_conv::GdnConvSliceInput;

const SCRATCH: usize = 960 * 1024;
impl HostMath<'_, '_, '_> {
    fn conv_windows(&self) -> Result<&crate::original_binding::OriginalWindowOperators> {
        if self.region_kernel.is_some()
            || self.engine.is_some()
            || self.state.is_some()
            || self.source.program_numerical_profile()
                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported(
                "convolution requires private pinned CPU window contract".into(),
            ));
        }
        match &self.source {
            HostSource::Windows(source) => Ok(source),
            HostSource::Loaded(_) => Err(Error::Unsupported(
                "convolution requires admitted window backing".into(),
            )),
        }
    }
    fn check_conv_cancel(&self) -> Result<()> {
        if self
            .cancelled
            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
        {
            return Err(Error::State("window convolution cancelled".into()));
        }
        Ok(())
    }
    pub(crate) fn import_gdn_conv_input(
        &self,
        input: GdnConvSliceInput<'_>,
        channels: usize,
    ) -> Result<(HostValue, HostValue)> {
        let source = self.conv_windows()?;
        if input.validate()? != channels
            || input.source_identity() != source.source_identity()
            || input.numerical_profile() != self.source.program_numerical_profile()
        {
            return Err(Error::State(
                "convolution input full width/source/profile".into(),
            ));
        }
        self.check_conv_cancel()?;
        // Both allocations are charged once, before either Vec allocation.
        let workspace = Arc::new(
            source.source().reserve_operator_buffers(
                channels
                    .checked_mul(16)
                    .ok_or_else(|| Error::Limit("convolution input byte count".into()))?,
                SCRATCH,
            )?,
        );
        let make = |bytes: &[u8]| -> Result<HostValue> {
            let values = bytes
                .chunks_exact(2)
                .map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16))
                .collect();
            HostValue::typed(
                values,
                ActivationPrecision::Bf16WidenedF32,
                source.source_identity(),
                self.source.program_numerical_profile(),
            )?
            .retain_workspace(Some(workspace.clone()))
        };
        let result = (
            make(input.activation_bytes())?,
            make(input.history_bytes())?,
        );
        self.check_conv_cancel()?;
        Ok(result)
    }
    pub(super) fn execute_causal_conv_silu(
        &self,
        parameter: SourceParameter,
        input: &HostValue,
        history: &HostValue,
        contract: CausalConvSiluContract,
    ) -> Result<CausalConvValues<HostValue>> {
        let source = self.conv_windows()?;
        let channels = contract.validate(&parameter, input.elements, history.elements)?;
        if parameter.source != source.source_identity() {
            return Err(Error::State("convolution parameter source".into()));
        }
        for value in [input, history] {
            if value.source != Some(source.source_identity())
                || value.profile != Some(self.source.program_numerical_profile())
                || value.precision != ActivationPrecision::Bf16WidenedF32
            {
                return Err(Error::State(
                    "convolution activation source/profile/precision".into(),
                ));
            }
            let workspace = value.data.workspace.as_ref().ok_or_else(|| {
                Error::State("convolution activation missing physical backing".into())
            })?;
            if !source.source().same_issuer(workspace.source()) {
                return Err(Error::State(
                    "convolution activation foreign source issuer".into(),
                ));
            }
        }
        // Contiguity and source membership checked before allocation or math.
        let activation = input.as_slice()?;
        let prior = history.as_slice()?;
        ActivationPrecision::Bf16WidenedF32.validate_values(activation)?;
        ActivationPrecision::Bf16WidenedF32.validate_values(prior)?;
        self.check_conv_cancel()?;
        // Three retained results: C raw + C activated + 3C history, F32 backing.
        // Scratch conservatively covers raw FP32 + SiLU FP32 + rounding Vec.
        let workspace = Arc::new(
            source.source().reserve_operator_buffers(
                channels
                    .checked_mul(20)
                    .ok_or_else(|| Error::Limit("convolution output byte count".into()))?,
                SCRATCH,
            )?,
        );
        let completed = source.convolve(&parameter, activation, prior, self.cancelled)?;
        let silu =
            zerogravity_cpu::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16(
                source.source_identity(),
                &completed.raw_bf16,
            )?;
        let activated = silu
            .into_iter()
            .map(round_bf16)
            .collect::<Result<Vec<_>>>()?;
        let make = |values| -> Result<HostValue> {
            HostValue::typed(
                values,
                ActivationPrecision::Bf16WidenedF32,
                source.source_identity(),
                self.source.program_numerical_profile(),
            )?
            .retain_workspace(Some(workspace.clone()))
        };
        let result = CausalConvValues {
            raw: make(completed.raw_bf16)?,
            activated: make(activated)?,
            next_history: make(completed.next_history)?,
        };
        self.check_conv_cancel()?;
        Ok(result)
    }
}
