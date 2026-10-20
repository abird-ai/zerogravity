//! Closed pinned CPU region; three private BF16 outputs, no state effects.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalConvSiluContract {
    TorchCpuBf16Avx2,
}
impl CausalConvSiluContract {
    /// Original raw depthwise `[channel,1,4]`, never an adjusted vector.
    pub fn validate(
        self,
        parameter: &SourceParameter,
        input: usize,
        history: usize,
    ) -> Result<usize> {
        let channels = input;
        if !parameter.source.is_bound() || parameter.name.is_empty() {
            return Err(Error::State("causal convolution source binding".into()));
        }
        if channels == 0
            || channels > 16384
            || !channels.is_multiple_of(32)
            || parameter.kind != QuantKind::Bf16
            || parameter.shape.as_slice() != [channels as u64, 1, 4]
            || channels.checked_mul(8) != Some(parameter.bytes)
            || channels.checked_mul(3) != Some(history)
        {
            return Err(Error::Unsupported(
                "closed causal convolution raw geometry/dtype".into(),
            ));
        }
        Ok(channels)
    }
}

#[derive(Clone)]
pub struct CausalConvValues<V> {
    pub raw: V,
    pub activated: V,
    /// Canonical `[channel,3]`, oldest to newest. Caller owns publication.
    pub next_history: V,
}

pub(super) fn result_precision(
    builder: &PlanBuilder,
    parameter: &SourceParameter,
    contract: CausalConvSiluContract,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    if !builder.operator_slice
        || builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || parameter.source != builder.source
        || inputs.len() != 2
        || inputs
            .iter()
            .any(|v| v.precision != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::Unsupported(
            "causal convolution profile/source/precision/slice".into(),
        ));
    }
    contract.validate(parameter, inputs[0].elements, inputs[1].elements)?;
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn record(
    builder: &mut PlanBuilder,
    parameter: SourceParameter,
    input: &PlanValue,
    history: &PlanValue,
    contract: CausalConvSiluContract,
) -> Result<CausalConvValues<PlanValue>> {
    let channels = contract.validate(&parameter, input.elements, history.elements)?;
    let name = parameter.name.clone();
    let values = builder.record(
        &name,
        PlanOperation::TypedCausalConvSilu {
            parameter,
            contract,
        },
        &[*input, *history],
        &[
            ValueShape::Vector(channels),
            ValueShape::Vector(channels),
            ValueShape::Vector(channels * 3),
        ],
        None,
    )?;
    let [raw, activated, next_history]: [PlanValue; 3] = values
        .try_into()
        .map_err(|_| Error::State("causal convolution output arity".into()))?;
    Ok(CausalConvValues {
        raw,
        activated,
        next_history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (PlanBuilder, SourceParameter) {
        let source = SourceIdentity::for_new_allocation().unwrap();
        (
            PlanBuilder::new_operator_slice(
                source,
                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            )
            .unwrap(),
            SourceParameter {
                source,
                name: "original.conv.weight".into(),
                kind: QuantKind::Bf16,
                shape: vec![32, 1, 4],
                bytes: 256,
            },
        )
    }
    // Synthetic embedding metadata only; no original model producer claim.
    fn input(builder: &mut PlanBuilder, width: usize) -> PlanValue {
        builder
            .embed_profiled(
                SourceTensor {
                    source: builder.source,
                    name: format!("synthetic.{width}"),
                    kind: QuantKind::Bf16,
                    axes: MatrixAxes::OutputInput,
                    columns: width,
                    rows: 2,
                },
                PlanBuilder::inputs().token,
                EmbeddingContract::Bf16OutputInput,
            )
            .unwrap()
    }
    #[test]
    fn pure_three_outputs_have_bf16_precision_and_no_state_effect() {
        let (mut b, p) = fixture();
        let x = input(&mut b, 32);
        let h = input(&mut b, 96);
        let output = b
            .causal_conv_silu_profiled(p, &x, &h, CausalConvSiluContract::TorchCpuBf16Avx2)
            .unwrap();
        for (v, n) in [
            (&output.raw, 32),
            (&output.activated, 32),
            (&output.next_history, 96),
        ] {
            assert_eq!(b.elements(v), n);
            assert_eq!(b.precision(v), ActivationPrecision::Bf16WidenedF32);
        }
        let recorded = b.finish_operator_slice(output.activated).unwrap();
        assert_eq!(recorded.plan().nodes().len(), 3);
        assert!(recorded.plan().required_domains().is_empty());
        assert!(recorded.plan().proposed_domains().is_empty());
        assert_eq!(recorded.plan().frontier_delta(), 0);
    }
    #[test]
    fn raw_shape_dtype_length_source_and_history_are_closed() {
        let (mut b, p) = fixture();
        let x = input(&mut b, 32);
        let h = input(&mut b, 96);
        let contract = CausalConvSiluContract::TorchCpuBf16Avx2;
        for wrong in [
            SourceParameter {
                source: SourceIdentity::for_new_allocation().unwrap(),
                ..p.clone()
            },
            SourceParameter {
                shape: vec![128],
                ..p.clone()
            },
            SourceParameter {
                shape: vec![32, 4, 1],
                ..p.clone()
            },
            SourceParameter {
                kind: QuantKind::F32,
                ..p.clone()
            },
            SourceParameter {
                bytes: 255,
                ..p.clone()
            },
        ] {
            assert!(
                b.causal_conv_silu_profiled(wrong, &x, &h, contract)
                    .is_err()
            );
        }
        let short = input(&mut b, 95);
        assert!(
            b.causal_conv_silu_profiled(p.clone(), &x, &short, contract)
                .is_err()
        );
        let mut foreign = PlanBuilder::new_operator_slice(b.source, b.profile).unwrap();
        let other = input(&mut foreign, 32);
        assert!(
            b.causal_conv_silu_profiled(p, &other, &h, contract)
                .is_err()
        );
    }
    #[test]
    fn valid_geometry_strided_values_require_explicit_pack() {
        let (mut b, p) = fixture();
        let base = input(&mut b, 64);
        let h = input(&mut b, 96);
        let strided = b
            .view(
                &base,
                TensorView {
                    offset: 0,
                    rows: 32,
                    width: 1,
                    row_stride: 2,
                },
            )
            .unwrap();
        let contract = CausalConvSiluContract::TorchCpuBf16Avx2;
        let error = b
            .causal_conv_silu_profiled(p.clone(), &strided, &h, contract)
            .err()
            .unwrap();
        assert!(error.to_string().contains("contiguous"));
        let packed = b.contiguous(&strided).unwrap();
        b.causal_conv_silu_profiled(p, &packed, &h, contract)
            .unwrap();
    }
}
