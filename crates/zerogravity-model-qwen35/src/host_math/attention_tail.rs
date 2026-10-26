//! Source-window ownership/admission for the same eager CPU backend operation.
use super::*;
use crate::accelerator::program::{TorchCpuAttentionInputs, TorchCpuAttentionValues};

impl HostMath<'_, '_, '_> {
    pub(super) fn execute_cpu_attention(
        &self,
        input: TorchCpuAttentionInputs<'_, HostValue>,
    ) -> Result<TorchCpuAttentionValues<HostValue>> {
        self.check_torch_cancel()?;
        let HostSource::Windows(source) = &self.source else {
            return Err(Error::Unsupported(
                "bounded eager attention requires original source windows".into(),
            ));
        };
        if self.region_kernel.is_some()
            || self.source.program_numerical_profile()
                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported(
                "eager CPU attention provider/profile".into(),
            ));
        }
        let (q, kv, tokens) = input.geometry.validate(input.position)?;
        let mut inputs = vec![input.query, input.key, input.value, input.gate];
        if let Some((k, v)) = input.prior {
            inputs.extend([k, v]);
        }
        for value in inputs {
            if value.precision != ActivationPrecision::Bf16WidenedF32
                || value.source != Some(self.source.source_identity())
                || value.profile != Some(self.source.program_numerical_profile())
                || value
                    .data
                    .workspace
                    .as_ref()
                    .is_none_or(|ticket| !source.source().same_issuer(ticket.source()))
            {
                return Err(Error::State(
                    "CPU attention same source/precision/admitted backing".into(),
                ));
            }
        }
        let count = 3 * q + kv + 2 * tokens * kv + 3 * input.geometry.query_heads * tokens;
        let workspace = self.prepare_output(Some(input.query), count, 8192)?;
        let values = zerogravity_cpu::numerics::torch_cpu::attention::eager_bf16(
            self.source.source_identity(),
            TorchCpuAttentionInputs {
                query: input.query.as_slice()?,
                key: input.key.as_slice()?,
                value: input.value.as_slice()?,
                gate: input.gate.as_slice()?,
                prior: input
                    .prior
                    .map(|(k, v)| -> Result<_> { Ok((k.as_slice()?, v.as_slice()?)) })
                    .transpose()?,
                geometry: input.geometry,
                position: input.position,
            },
            self.cancelled,
        )?;
        values.try_map(|values| {
            if values.capacity() != values.len() {
                return Err(Error::Limit(
                    "CPU attention combined output capacity".into(),
                ));
            }
            HostValue::typed(
                values,
                ActivationPrecision::Bf16WidenedF32,
                self.source.source_identity(),
                self.source.program_numerical_profile(),
            )?
            .retain_workspace(workspace.clone())
        })
    }
}
