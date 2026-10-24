//! Private source-admitted GDN/state owners in the existing CPU backend.
use super::*;
use crate::accelerator::program::{TorchCpuGdnContract, TorchCpuGdnInputs, TorchCpuGdnValues};
impl HostMath<'_, '_, '_> {
    pub(crate) fn window_zeros(
        &self,
        n: usize,
        precision: ActivationPrecision,
    ) -> Result<HostValue> {
        self.check_torch_cancel()?;
        let HostSource::Windows(source) = &self.source else {
            return Err(Error::Unsupported(
                "window zeros require original source".into(),
            ));
        };
        let bytes = n
            .checked_mul(4)
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::Limit("window zero extent".into()))?;
        let (charge, _) = source
            .source()
            .reserve_operator_buffers_split(bytes.max(64 * 1024), 0)?;
        let charge = Some(Arc::new(charge));
        let mut v = Vec::new();
        v.try_reserve_exact(n)
            .map_err(|_| Error::Limit("window zero allocation".into()))?;
        if v.capacity() != n {
            return Err(Error::Limit("window zero capacity".into()));
        }
        v.resize(n, 0.);
        HostValue::typed(
            v,
            precision,
            self.source.source_identity(),
            self.source.program_numerical_profile(),
        )?
        .retain_workspace(charge)
    }
    /// State transport only: three real prior frames plus the current producer.
    pub(crate) fn window_full_convolution_cache(
        &self,
        prior: &HostValue,
        minimal: &HostValue,
        input: &HostValue,
    ) -> Result<HostValue> {
        self.check_torch_cancel()?;
        let HostSource::Windows(source) = &self.source else {
            return Err(Error::Unsupported(
                "full convolution state requires source windows".into(),
            ));
        };
        let channels = input.elements;
        let four = channels
            .checked_mul(4)
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::Limit("full convolution extent".into()))?;
        if self.source.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || prior.elements != four
            || minimal.elements != channels * 3
        {
            return Err(Error::State(
                "full convolution state geometry/profile".into(),
            ));
        }
        for value in [prior, minimal, input] {
            if value.precision != ActivationPrecision::Bf16WidenedF32
                || value.source != Some(source.source_identity())
                || value.profile != Some(self.source.program_numerical_profile())
                || value
                    .data
                    .workspace
                    .as_ref()
                    .is_none_or(|w| !source.source().same_issuer(w.source()))
            {
                return Err(Error::State(
                    "full convolution state source/precision/issuer".into(),
                ));
            }
            value.precision.validate_values(value.as_slice()?)?;
        }
        let bytes = four
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("full convolution bytes".into()))?;
        let (output, _) = source
            .source()
            .reserve_operator_buffers_split(bytes.max(64 * 1024), 0)?;
        let mut next = Vec::new();
        next.try_reserve_exact(four)
            .map_err(|_| Error::Limit("full convolution allocation".into()))?;
        if next.capacity() != four {
            return Err(Error::Limit("full convolution capacity".into()));
        }
        full_convolution_frames(
            prior.as_slice()?,
            minimal.as_slice()?,
            input.as_slice()?,
            &mut next,
        )?;
        self.check_torch_cancel()?;
        HostValue::typed(
            next,
            ActivationPrecision::Bf16WidenedF32,
            source.source_identity(),
            self.source.program_numerical_profile(),
        )?
        .retain_workspace(Some(Arc::new(output)))
    }
    pub(super) fn execute_cpu_gdn(
        &self,
        c: &TorchCpuGdnContract,
        input: TorchCpuGdnInputs<'_, HostValue>,
    ) -> Result<TorchCpuGdnValues<HostValue>> {
        self.check_torch_cancel()?;
        let HostSource::Windows(source) = &self.source else {
            return Err(Error::Unsupported(
                "diagnostic GDN requires source windows".into(),
            ));
        };
        if self.engine.is_some()
            || self.region_kernel.is_some()
            || self.state.is_some()
            || self.source.program_numerical_profile()
                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported("diagnostic GDN profile/provider".into()));
        }
        let (channels, value, state) = c.validate(source.source_identity())?;
        source.require_gdn_contract(c)?;
        for (v, n, precision) in [
            (input.qkv, channels, ActivationPrecision::Bf16WidenedF32),
            (input.gate, value, ActivationPrecision::Bf16WidenedF32),
            (
                input.alpha,
                c.value_heads,
                ActivationPrecision::Bf16WidenedF32,
            ),
            (
                input.beta,
                c.value_heads,
                ActivationPrecision::Bf16WidenedF32,
            ),
            (
                input.history,
                channels * 3,
                ActivationPrecision::Bf16WidenedF32,
            ),
            (input.state, state, ActivationPrecision::F32),
        ] {
            if v.elements != n
                || v.precision != precision
                || v.source != Some(source.source_identity())
                || v.profile != Some(self.source.program_numerical_profile())
                || v.data
                    .workspace
                    .as_ref()
                    .is_none_or(|w| !source.source().same_issuer(w.source()))
            {
                return Err(Error::State(
                    "diagnostic GDN actual source/value/state admission".into(),
                ));
            }
            precision.validate_values(v.as_slice()?)?;
        }
        // All six returned vectors share only their output envelope. Scratch
        // remains locally admitted until this synchronous operation completes.
        let output_bytes = (channels * 5 + state + value * 2) * 4;
        let scratch_bytes = (state * 5 + channels * 4 + value * 16 + 8192) * 4;
        let (output, _scratch) = source.source().reserve_operator_buffers_split(
            output_bytes.max(64 * 1024),
            scratch_bytes.max(960 * 1024),
        )?;
        let charge = Some(Arc::new(output));
        let raw = source.gdn_parameter_windows(c, self.cancelled)?;
        let tensor = |i: usize| zerogravity_core::source::SourceTensorRef {
            kind: raw[i].parameter().tensor().kind,
            shape: &raw[i].parameter().tensor().shape,
            packed_bytes: raw[i].as_slice(),
        };
        let selected = self.initial_core_tape.as_ref().filter(|_| {
            c.convolution.name == "model.language_model.layers.9.linear_attn.conv1d.weight"
        });
        let mut observe = |stage, values: &[f32]| -> Result<()> {
            let tape =
                selected.ok_or_else(|| Error::State("initial core exact source layer9".into()))?;
            let mut tape = tape
                .lock()
                .map_err(|_| Error::State("initial core observer poisoned".into()))?;
            tape.require_source(source)?;
            tape.capture(stage, values)
        };
        let done = zerogravity_cpu::original_hf::gdn::step_observed_with_exp_profile(
            zerogravity_cpu::original_hf::gdn::GdnInputs {
                source: source.source_identity(),
                key_heads: c.key_heads,
                value_heads: c.value_heads,
                qkv: input.qkv.as_slice()?,
                gate: input.gate.as_slice()?,
                alpha: input.alpha.as_slice()?,
                beta: input.beta.as_slice()?,
                history: input.history.as_slice()?,
                state: input.state.as_slice()?,
                phase: input.phase,
            },
            zerogravity_cpu::original_hf::gdn::GdnParameters {
                convolution: tensor(0),
                convolution_name: &c.convolution.name,
                a_log: tensor(1),
                dt_bias: tensor(2),
                norm: tensor(3),
            },
            // This branch already requires the explicit original mixed-source
            // Torch CPU arithmetic profile; no model/head-count selector.
            c.standalone_exp,
            if selected.is_some() {
                Some(&mut observe)
            } else {
                None
            },
        )?;
        self.check_torch_cancel()?;
        let make = |v: Vec<f32>, p| -> Result<HostValue> {
            if v.capacity() != v.len() {
                return Err(Error::Limit("diagnostic GDN output capacity".into()));
            }
            HostValue::typed(
                v,
                p,
                source.source_identity(),
                self.source.program_numerical_profile(),
            )?
            .retain_workspace(charge.clone())
        };
        Ok(TorchCpuGdnValues {
            convolution: make(done.convolution, ActivationPrecision::Bf16WidenedF32)?,
            activated: make(done.activated, ActivationPrecision::Bf16WidenedF32)?,
            history: make(done.history, ActivationPrecision::Bf16WidenedF32)?,
            state: make(done.state, ActivationPrecision::F32)?,
            core: make(done.core, ActivationPrecision::Bf16WidenedF32)?,
            gated: make(done.gated, ActivationPrecision::Bf16WidenedF32)?,
        })
    }
}

fn full_convolution_frames(
    prior: &[f32],
    minimal: &[f32],
    input: &[f32],
    next: &mut Vec<f32>,
) -> Result<()> {
    let four = input
        .len()
        .checked_mul(4)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("full convolution transport extent".into()))?;
    if prior.len() != four
        || minimal.len() != input.len() * 3
        || !next.is_empty()
        || next.capacity() != four
    {
        return Err(Error::State(
            "full convolution exact transport geometry/capacity".into(),
        ));
    }
    for (frames, previous) in prior.chunks_exact(4).zip(minimal.chunks_exact(3)) {
        if frames[1..]
            .iter()
            .zip(previous)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(Error::State(
                "full convolution exact prior/last-three join".into(),
            ));
        }
    }
    for (frames, input) in prior.chunks_exact(4).zip(input) {
        next.extend_from_slice(&frames[1..]);
        next.push(*input);
    }
    Ok(())
}

#[cfg(test)]
mod full_convolution_tests {
    use super::*;
    #[test]
    fn actual_prior_shift_preserves_oldest_frame_and_signed_zero_bits() {
        let prior = [1., 2., -0., 4., 5., 6., 7., 8.];
        let minimal = [2., -0., 4., 6., 7., 8.];
        let mut next = Vec::with_capacity(8);
        full_convolution_frames(&prior, &minimal, &[9., -0.], &mut next).unwrap();
        assert_eq!(
            next.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            [2.0f32, -0., 4., 9., 6., 7., 8., -0.].map(f32::to_bits)
        );
        assert_eq!(prior, [1., 2., -0., 4., 5., 6., 7., 8.]);
    }
    #[test]
    fn stale_last_three_or_output_capacity_never_produces_partial_cache() {
        let mut out = Vec::with_capacity(4);
        assert!(
            full_convolution_frames(&[1., -0., 3., 4.], &[0., 3., 4.], &[5.], &mut out).is_err()
        );
        assert!(out.is_empty());
        assert!(full_convolution_frames(&[1., 2., 3., 4.], &[2., 3.], &[5.], &mut out).is_err());
        let mut out = Vec::with_capacity(8);
        assert!(
            full_convolution_frames(&[1., 2., 3., 4.], &[2., 3., 4.], &[5.], &mut out).is_err()
        );
        assert!(out.is_empty());
    }
}
