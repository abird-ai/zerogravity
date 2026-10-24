//! Explicit source-state components in the original SSA and StateEffect chain.
//! No allocation, source authentication, completion or publication authority.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceStateBinding {
    slot: StateSlot,
    components: Vec<usize>,
}
impl SourceStateBinding {
    /// Metadata only. The family lowerer validates its genuine source geometry.
    pub fn new(slot: StateSlot, components: Vec<usize>) -> Result<Self> {
        let expected = if matches!(slot, StateSlot::Kv(_)) {
            2
        } else {
            1
        };
        if components.len() != expected || components.iter().any(|n| *n == 0 || *n > 4_194_304) {
            return Err(Error::Limit(
                "source state positive finite components".into(),
            ));
        }
        Ok(Self { slot, components })
    }
    pub fn slot(&self) -> StateSlot {
        self.slot
    }
    pub fn components(&self) -> &[usize] {
        &self.components
    }
    pub fn precision(&self) -> ActivationPrecision {
        if matches!(self.slot, StateSlot::Recurrent(_)) {
            ActivationPrecision::F32
        } else {
            ActivationPrecision::Bf16WidenedF32
        }
    }
}
fn require_profile(builder: &PlanBuilder, binding: &SourceStateBinding) -> Result<()> {
    if builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || builder.standalone_exp.is_none()
        || builder.operator_slice
        || !builder.required.contains(&binding.slot)
    {
        return Err(Error::State(
            "source state profile/arithmetic/domain".into(),
        ));
    }
    Ok(())
}
pub(super) fn read_precision(
    builder: &PlanBuilder,
    binding: &SourceStateBinding,
    state: StateVersion,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    require_profile(builder, binding)?;
    if !inputs.is_empty() || state != builder.current(binding.slot)? || state.ordinal != 0 {
        return Err(Error::State("source state original read version".into()));
    }
    Ok(binding.precision())
}
pub(super) fn write_precision(
    builder: &PlanBuilder,
    binding: &SourceStateBinding,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    require_profile(builder, binding)?;
    if inputs.len() != binding.components.len()
        || inputs
            .iter()
            .zip(&binding.components)
            .any(|(v, n)| v.elements != *n || v.precision != binding.precision())
    {
        return Err(Error::State(
            "source state proposed component/precision".into(),
        ));
    }
    Ok(binding.precision())
}
impl PlanBuilder {
    pub fn source_state_read(&mut self, binding: SourceStateBinding) -> Result<Vec<PlanValue>> {
        let state = self.current(binding.slot)?;
        if self.nodes.iter().any(|n| {
            matches!(&n.operation,
            PlanOperation::TypedSourceStateRead { binding: old, .. } if old.slot == binding.slot)
        }) {
            return Err(Error::State("source state duplicate original read".into()));
        }
        let shapes = binding
            .components
            .iter()
            .map(|n| ValueShape::Vector(*n))
            .collect::<Vec<_>>();
        self.record(
            "source.state.read",
            PlanOperation::TypedSourceStateRead {
                binding: Box::new(binding),
                state,
            },
            &[],
            &shapes,
            None,
        )
    }
    pub fn source_state_write(
        &mut self,
        binding: SourceStateBinding,
        inputs: &[PlanValue],
    ) -> Result<()> {
        let effect = self.effect(binding.slot)?;
        self.record(
            "source.state.propose",
            PlanOperation::TypedSourceStateWrite {
                binding: Box::new(binding),
            },
            inputs,
            &[],
            Some(effect),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn builder(exp: StandaloneExpProfile) -> PlanBuilder {
        PlanBuilder::new_torch_source(
            SourceIdentity::for_new_allocation().unwrap(),
            2,
            vec![
                StateSlot::Conv(ConvDomain(0)),
                StateSlot::Recurrent(RecurrentDomain(0)),
            ],
            true,
            exp,
        )
        .unwrap()
    }
    fn input(b: &mut PlanBuilder, n: usize) -> PlanValue {
        b.embed_profiled(
            SourceTensor {
                source: b.source,
                name: format!("actual-{n}"),
                kind: QuantKind::Bf16,
                columns: n,
                rows: 512,
                axes: MatrixAxes::OutputInput,
            },
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap()
    }
    fn contract(source: SourceIdentity, exp: StandaloneExpProfile) -> TorchCpuGdnContract {
        let p = |name: &str, shape: Vec<u64>, kind: QuantKind| SourceParameter {
            source,
            name: name.into(),
            bytes: kind.byte_len(shape.iter().product()).unwrap() as usize,
            kind,
            shape,
        };
        TorchCpuGdnContract {
            standalone_exp: exp,
            convolution: p("conv", vec![10240, 1, 4], QuantKind::Bf16),
            a_log: p("a", vec![48], QuantKind::Bf16),
            dt_bias: p("dt", vec![48], QuantKind::F32),
            norm: p("gamma", vec![128], QuantKind::F32),
            key_heads: 16,
            value_heads: 48,
        }
    }
    #[test]
    fn source_full4_and_mixed_precision_use_original_effect_chain() {
        let exp = StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1;
        let mut b = builder(exp);
        let conv = SourceStateBinding::new(StateSlot::Conv(ConvDomain(0)), vec![40960]).unwrap();
        let recurrent =
            SourceStateBinding::new(StateSlot::Recurrent(RecurrentDomain(0)), vec![786432])
                .unwrap();
        let full = b.source_state_read(conv.clone()).unwrap()[0];
        let state = b.source_state_read(recurrent.clone()).unwrap()[0];
        let view = b
            .view(
                &full,
                TensorView {
                    offset: 1,
                    rows: 10240,
                    width: 3,
                    row_stride: 4,
                },
            )
            .unwrap();
        let history = b.contiguous(&view).unwrap();
        let qkv = input(&mut b, 10240);
        let gate = input(&mut b, 6144);
        let alpha = input(&mut b, 48);
        let c = contract(b.source, exp);
        let full_next = b
            .torch_cpu_full_convolution_cache(&full, &history, &qkv)
            .unwrap();
        let gdn = b
            .torch_cpu_bf16_gdn(
                &c,
                TorchCpuGdnInputs {
                    qkv: &qkv,
                    gate: &gate,
                    alpha: &alpha,
                    beta: &alpha,
                    history: &history,
                    state: &state,
                    phase: TorchCpuGdnPhase::InitialOne,
                },
            )
            .unwrap();
        assert_eq!(gdn.state.precision(), ActivationPrecision::F32);
        assert_eq!(gdn.gated.precision(), ActivationPrecision::Bf16WidenedF32);
        b.source_state_write(conv.clone(), &[full_next]).unwrap();
        assert!(b.source_state_write(conv, &[full_next]).is_err());
        b.source_state_write(recurrent, &[gdn.state]).unwrap();
        b.advance_frontier().unwrap();
        let plan = b.finish(Some(gdn.gated)).unwrap();
        assert_eq!(plan.proposed_domains().len(), 2);
        assert!(plan.proposed_domains().iter().all(|v| v.ordinal == 1));
        assert_eq!(plan.standalone_exp_profile(), Some(exp));
    }
    #[test]
    fn explicit_arithmetic_mismatch_refuses_before_record_mutation() {
        let mut b = builder(StandaloneExpProfile::ScalarLibmDiagnostic);
        let conv = SourceStateBinding::new(StateSlot::Conv(ConvDomain(0)), vec![30720]).unwrap();
        let history = b.source_state_read(conv).unwrap()[0];
        let state = b
            .source_state_read(
                SourceStateBinding::new(StateSlot::Recurrent(RecurrentDomain(0)), vec![786432])
                    .unwrap(),
            )
            .unwrap()[0];
        let qkv = input(&mut b, 10240);
        let gate = input(&mut b, 6144);
        let alpha = input(&mut b, 48);
        let c = contract(b.source, StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1);
        let before = (b.nodes.len(), b.values.len(), b.domains.clone());
        assert!(
            b.torch_cpu_bf16_gdn(
                &c,
                TorchCpuGdnInputs {
                    qkv: &qkv,
                    gate: &gate,
                    alpha: &alpha,
                    beta: &alpha,
                    history: &history,
                    state: &state,
                    phase: TorchCpuGdnPhase::CachedOne
                }
            )
            .is_err()
        );
        assert_eq!((b.nodes.len(), b.values.len(), b.domains.clone()), before);
        assert_ne!(
            StandaloneExpProfile::ScalarLibmDiagnostic.arithmetic_revision(),
            c.standalone_exp.arithmetic_revision()
        );
    }
}
