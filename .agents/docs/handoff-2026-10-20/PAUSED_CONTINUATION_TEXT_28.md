# Paused continuation text — volume 28

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program/cached_gdn.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 9795; SHA256 `83eea3ec759971840ab4cc0582ed13bc49bb7d9c033bad7c49ecd9041c49f206`.

````text
//! Pure cached-token core; explicit decay boundary, never initial chunk64.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CachedGdnContract {
    TorchCpuBf16Avx2Width128,
}
impl CachedGdnContract {
    pub fn validate(self, heads: usize, lengths: &[usize; 6]) -> Result<(usize, usize)> {
        let vector = heads.checked_mul(128);
        let state = vector.and_then(|n| n.checked_mul(128));
        let (Some(vector), Some(state)) = (vector, state) else {
            return Err(Error::Limit("cached GDN geometry overflow".into()));
        };
        if heads == 0 || state > 1_048_576 {
            return Err(Error::Limit("cached GDN head/element bound".into()));
        }
        if *lengths != [vector, vector, vector, heads, heads, state] {
            return Err(Error::Malformed(
                "cached GDN explicit operand geometry".into(),
            ));
        }
        Ok((vector, state))
    }
}

/// Q/K are already repeated to value heads. No implicit head/layout conversion.
#[derive(Clone)]
pub struct CachedGdnOperands<V> {
    pub query: V,
    pub key: V,
    pub value: V,
    pub beta: V,
    /// Explicit finite FP32 exp(g); this type grants no coefficient eligibility.
    pub decay: V,
    /// FP32 [head,key128,value128], distinct from canonical value-major state.
    pub previous_state: V,
}
#[derive(Clone)]
pub struct CachedGdnValues<V> {
    pub core_f32: V,
    pub output_bf16: V,
    pub next_state: V,
}

pub(super) fn result_precision(
    builder: &PlanBuilder,
    source: SourceIdentity,
    heads: usize,
    contract: CachedGdnContract,
    inputs: &[PlanValue],
    bf16_return: bool,
) -> Result<ActivationPrecision> {
    if !builder.operator_slice
        || builder.source != source
        || builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    {
        return Err(Error::Unsupported(
            "cached GDN source/profile/private slice".into(),
        ));
    }
    if bf16_return {
        let vector = heads
            .checked_mul(128)
            .filter(|_| heads > 0 && heads <= 64)
            .ok_or_else(|| Error::Limit("cached GDN return head bound".into()))?;
        if inputs.len() != 1
            || inputs[0].elements != vector
            || inputs[0].precision != ActivationPrecision::F32
        {
            return Err(Error::Malformed("cached GDN return F32 core".into()));
        }
        Ok(ActivationPrecision::Bf16WidenedF32)
    } else {
        let lengths: [usize; 6] = inputs
            .iter()
            .map(|v| v.elements)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| Error::Malformed("cached GDN operand arity".into()))?;
        contract.validate(heads, &lengths)?;
        if inputs[..4]
            .iter()
            .any(|v| v.precision != ActivationPrecision::Bf16WidenedF32)
            || inputs[4..]
                .iter()
                .any(|v| v.precision != ActivationPrecision::F32)
        {
            return Err(Error::Unsupported(
                "cached GDN BF16/F32 operand precision".into(),
            ));
        }
        Ok(ActivationPrecision::F32)
    }
}

pub(super) fn record(
    builder: &mut PlanBuilder,
    source: SourceIdentity,
    heads: usize,
    input: &CachedGdnOperands<PlanValue>,
    contract: CachedGdnContract,
) -> Result<CachedGdnValues<PlanValue>> {
    let inputs = [
        input.query,
        input.key,
        input.value,
        input.beta,
        input.decay,
        input.previous_state,
    ];
    let (vector, state) = contract.validate(heads, &inputs.map(|v| v.elements))?;
    // Existing record validates ALL original builder/value/stride memberships
    // before recording either pure node. No record precision policy changes.
    let values = builder.record(
        "original.cached_gdn.core",
        PlanOperation::TypedCachedGdnCore {
            source,
            heads,
            contract,
        },
        &inputs,
        &[ValueShape::Vector(vector), ValueShape::Vector(state)],
        None,
    )?;
    let [core_f32, next_state]: [PlanValue; 2] = values
        .try_into()
        .map_err(|_| Error::State("cached GDN core output arity".into()))?;
    let output_bf16 = builder.one(
        "original.cached_gdn.return_bf16",
        PlanOperation::TypedCachedGdnBf16Return {
            source,
            heads,
            contract,
        },
        &[core_f32],
        ValueShape::Vector(vector),
        None,
    )?;
    Ok(CachedGdnValues {
        core_f32,
        output_bf16,
        next_state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input(builder: &mut PlanBuilder, n: usize, bf16: bool) -> PlanValue {
        if bf16 {
            builder
                .embed_profiled(
                    SourceTensor {
                        source: builder.source,
                        name: format!("synthetic.operand.{n}"),
                        kind: QuantKind::Bf16,
                        axes: MatrixAxes::OutputInput,
                        columns: n,
                        rows: 1,
                    },
                    PlanBuilder::inputs().token,
                    EmbeddingContract::Bf16OutputInput,
                )
                .unwrap()
        } else {
            // Test-only F32 operand metadata; no executable source/model claim.
            builder
                .one(
                    "synthetic.fp32_operand",
                    PlanOperation::Observe {
                        layer: 0,
                        stage: "synthetic_fp32_operand",
                    },
                    &[],
                    ValueShape::Vector(n),
                    None,
                )
                .unwrap()
        }
    }
    fn fixture(b: &mut PlanBuilder) -> CachedGdnOperands<PlanValue> {
        CachedGdnOperands {
            query: input(b, 128, true),
            key: input(b, 128, true),
            value: input(b, 128, true),
            beta: input(b, 1, true),
            decay: input(b, 1, false),
            previous_state: input(b, 16384, false),
        }
    }
    fn builder() -> PlanBuilder {
        PlanBuilder::new_operator_slice(
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap()
    }
    #[test]
    fn cached_core_and_return_have_separate_precisions_without_state_effects() {
        let mut b = builder();
        let operands = fixture(&mut b);
        let source = b.source;
        let result = b
            .cached_gdn_profiled(
                source,
                1,
                &operands,
                CachedGdnContract::TorchCpuBf16Avx2Width128,
            )
            .unwrap();
        assert_eq!(result.core_f32.precision(), ActivationPrecision::F32);
        assert_eq!(result.next_state.precision(), ActivationPrecision::F32);
        assert_eq!(
            result.output_bf16.precision(),
            ActivationPrecision::Bf16WidenedF32
        );
        let recorded = b.finish_operator_slice(result.output_bf16).unwrap();
        assert_eq!(recorded.plan().nodes().len(), 8);
        assert!(recorded.plan().required_domains().is_empty());
        assert!(recorded.plan().proposed_domains().is_empty());
        assert_eq!(recorded.plan().frontier_delta(), 0);
    }
    #[test]
    fn foreign_inputs_and_valid_geometry_strides_refuse_before_first_core_node() {
        let mut b = builder();
        let mut operands = fixture(&mut b);
        let source = b.source;
        let mut other = builder();
        operands.decay = input(&mut other, 1, false);
        assert!(
            b.cached_gdn_profiled(
                source,
                1,
                &operands,
                CachedGdnContract::TorchCpuBf16Avx2Width128
            )
            .is_err()
        );
        assert_eq!(b.nodes.len(), 6);
        operands.decay = input(&mut b, 1, false);
        let base = input(&mut b, 256, true);
        operands.query = b
            .view(
                &base,
                TensorView {
                    offset: 0,
                    rows: 128,
                    width: 1,
                    row_stride: 2,
                },
            )
            .unwrap();
        let error = b
            .cached_gdn_profiled(
                source,
                1,
                &operands,
                CachedGdnContract::TorchCpuBf16Avx2Width128,
            )
            .err()
            .unwrap();
        assert!(error.to_string().contains("contiguous"));
        let nodes = b.nodes.len();
        operands.query = b.contiguous(&operands.query).unwrap();
        let decay = operands.decay;
        operands.decay = operands.beta;
        assert!(
            b.cached_gdn_profiled(
                source,
                1,
                &operands,
                CachedGdnContract::TorchCpuBf16Avx2Width128
            )
            .is_err()
        );
        assert_eq!(b.nodes.len(), nodes + 1);
        operands.decay = decay;
        b.cached_gdn_profiled(
            source,
            1,
            &operands,
            CachedGdnContract::TorchCpuBf16Avx2Width128,
        )
        .unwrap();
    }
    #[test]
    fn closed_bounds_cover48heads_without_head_padding() {
        let c = CachedGdnContract::TorchCpuBf16Avx2Width128;
        assert_eq!(
            c.validate(48, &[6144, 6144, 6144, 48, 48, 786432]).unwrap(),
            (6144, 786432)
        );
        assert!(c.validate(0, &[0; 6]).is_err());
        assert!(
            c.validate(65, &[8320, 8320, 8320, 65, 65, 1064960])
                .is_err()
        );
        assert!(c.validate(usize::MAX, &[0; 6]).is_err());
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program/coefficient_preparation.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5773; SHA256 `3d5cce563610fffd501d607d1c131e89f63162aa463f9560e4ee0056b38834a5`.

````text
//! State-free original coefficient preparation; no exp/g/decay capability.
use super::{QuantKind, SourceParameter};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GdnCoefficientPreparationContract {
    TorchCpuBf16Avx2,
}
impl GdnCoefficientPreparationContract {
    pub fn validate(self, dt: &SourceParameter, alpha: usize, beta: usize) -> Result<usize> {
        if !dt.source.is_bound() || dt.name.is_empty() {
            return Err(Error::State(
                "GDN preparation raw parameter source/name".into(),
            ));
        }
        if !matches!(alpha, 16 | 48) || beta != alpha || dt.shape != [alpha as u64] {
            return Err(Error::Malformed(
                "GDN preparation one-call head/span".into(),
            ));
        }
        let bytes = match dt.kind {
            QuantKind::Bf16 => 2,
            QuantKind::F32 => 4,
            _ => {
                return Err(Error::Unsupported(
                    "GDN preparation raw dt_bias dtype".into(),
                ));
            }
        };
        if dt.bytes != alpha * bytes {
            return Err(Error::Malformed(
                "GDN preparation raw dt_bias byte span".into(),
            ));
        }
        Ok(alpha)
    }
}

pub struct GdnCoefficientPreparationValues<V> {
    pub biased_alpha: V,
    pub softplus: V,
    pub beta: V,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::SourceIdentity;
    use crate::accelerator::program::{
        ActivationPrecision, EmbeddingContract, MathematicalBackend, MatrixAxes, PlanBuilder,
        ProgramNumericalProfile, SourceTensor, ValueShape,
    };
    fn dt(source: SourceIdentity) -> SourceParameter {
        SourceParameter {
            source,
            name: "synthetic.transport.dt_bias".into(),
            kind: QuantKind::Bf16,
            shape: vec![16],
            bytes: 32,
        }
    }
    fn embed(builder: &mut PlanBuilder, source: SourceIdentity) -> super::super::PlanValue {
        builder
            .embed_profiled(
                SourceTensor {
                    source,
                    name: "synthetic.transport.embed".into(),
                    kind: QuantKind::Bf16,
                    rows: 1,
                    columns: 16,
                    axes: MatrixAxes::OutputInput,
                },
                PlanBuilder::inputs().token,
                EmbeddingContract::Bf16OutputInput,
            )
            .unwrap()
    }
    #[test]
    fn synthetic_builder_transport_keeps_distinct_precisions_and_no_state() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut builder = PlanBuilder::new_operator_slice(
            source,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        let a = embed(&mut builder, source);
        let b = embed(&mut builder, source);
        let values = builder
            .coefficient_preparation_profiled(
                dt(source),
                &a,
                &b,
                GdnCoefficientPreparationContract::TorchCpuBf16Avx2,
            )
            .unwrap();
        assert_eq!(values.beta.precision(), ActivationPrecision::Bf16WidenedF32);
        assert_eq!(values.softplus.precision(), ActivationPrecision::F32);
        assert_eq!(values.biased_alpha.precision(), ActivationPrecision::F32);
        let plan = builder.finish_operator_slice(values.softplus).unwrap();
        assert_eq!(plan.plan().nodes().len(), 4);
        assert!(plan.plan().required_domains().is_empty());
        assert!(plan.plan().proposed_domains().is_empty());
        assert_eq!(plan.plan().frontier_delta(), 0);
        assert!(plan.plan().logits().is_none());
        assert!(
            plan.plan()
                .values()
                .iter()
                .all(|v| matches!(v, ValueShape::Vector(16)))
        );
    }
    #[test]
    fn foreign_builder_input_refuses_before_recording_either_preparation_node() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut first = PlanBuilder::new_operator_slice(
            source,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        let mut other = PlanBuilder::new_operator_slice(
            source,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        let a = embed(&mut first, source);
        let b = embed(&mut other, source);
        assert!(
            first
                .coefficient_preparation_profiled(
                    dt(source),
                    &a,
                    &b,
                    GdnCoefficientPreparationContract::TorchCpuBf16Avx2
                )
                .is_err()
        );
        assert_eq!(
            first.finish_operator_slice(a).unwrap().plan().nodes().len(),
            1
        );
    }
    #[test]
    fn raw_parameter_dtype_bytes_and_span_are_checked() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let contract = GdnCoefficientPreparationContract::TorchCpuBf16Avx2;
        let valid = dt(source);
        assert_eq!(contract.validate(&valid, 16, 16).unwrap(), 16);
        let mut wrong = valid.clone();
        wrong.shape = vec![1, 16];
        assert!(contract.validate(&wrong, 16, 16).is_err());
        wrong = valid.clone();
        wrong.kind = QuantKind::F16;
        assert!(contract.validate(&wrong, 16, 16).is_err());
        wrong = valid.clone();
        wrong.bytes -= 1;
        assert!(contract.validate(&wrong, 16, 16).is_err());
        assert!(contract.validate(&valid, 32, 32).is_err());
        assert!(contract.validate(&valid, 16, 48).is_err());
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program/gated_rms.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7572; SHA256 `7b8f589f7d2854de07d304d7f2153350851a3c33d9bae820309a761c7ae7c0f7`.

````text
//! Explicit BF16 source gamma product rounded before the FP32 SiLU gate.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedRmsGammaBf16Contract {
    TorchCpuBf16Avx2H48Width128,
}
impl GatedRmsGammaBf16Contract {
    pub fn validate(self, parameter: &SourceParameter, core: usize, gate: usize) -> Result<usize> {
        if !parameter.source.is_bound() || parameter.name.is_empty() {
            return Err(Error::State("gated RMS original source parameter".into()));
        }
        if parameter.kind != QuantKind::Bf16
            || parameter.shape != [128]
            || parameter.bytes != 256
            || core != 48 * 128
            || gate != core
        {
            return Err(Error::Unsupported(
                "gated RMS BF16 gamma H48 width128 contract".into(),
            ));
        }
        Ok(core)
    }
}

pub(super) fn result_precision(
    builder: &PlanBuilder,
    parameter: &SourceParameter,
    contract: GatedRmsGammaBf16Contract,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    if !builder.operator_slice
        || builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || parameter.source != builder.source
        || inputs.len() != 2
        || inputs
            .iter()
            .any(|v| !v.contiguous || v.precision != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::Unsupported(
            "gated RMS BF16 gamma pure profile/precision/source".into(),
        ));
    }
    contract.validate(parameter, inputs[0].elements, inputs[1].elements)?;
    Ok(ActivationPrecision::Bf16WidenedF32)
}

pub(super) fn record(
    builder: &mut PlanBuilder,
    parameter: SourceParameter,
    core: &PlanValue,
    gate: &PlanValue,
    contract: GatedRmsGammaBf16Contract,
) -> Result<PlanValue> {
    let elements = contract.validate(&parameter, core.elements, gate.elements)?;
    let name = parameter.name.clone();
    builder.one(
        &name,
        PlanOperation::TypedGatedRmsGammaBf16 {
            parameter,
            contract,
        },
        &[*core, *gate],
        ValueShape::Vector(elements),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn producer(builder: &mut PlanBuilder, width: usize) -> PlanValue {
        builder
            .embed_profiled(
                SourceTensor {
                    source: builder.source,
                    name: format!("fixture.{width}"),
                    kind: QuantKind::Bf16,
                    columns: width,
                    rows: 2,
                    axes: MatrixAxes::OutputInput,
                },
                PlanBuilder::inputs().token,
                EmbeddingContract::Bf16OutputInput,
            )
            .unwrap()
    }
    #[test]
    fn closed_bf16_gamma_node_retains_one_pure_bf16_value() {
        let id = SourceIdentity::for_new_allocation().unwrap();
        let mut builder = PlanBuilder::new_operator_slice(
            id,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        let core = producer(&mut builder, 6144);
        let gate = producer(&mut builder, 6144);
        let parameter = SourceParameter {
            source: id,
            name: "original.gamma".into(),
            kind: QuantKind::Bf16,
            shape: vec![128],
            bytes: 256,
        };
        let value = builder
            .gated_rms_gamma_bf16_profiled(
                parameter,
                &core,
                &gate,
                GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128,
            )
            .unwrap();
        assert_eq!(
            builder.precision(&value),
            ActivationPrecision::Bf16WidenedF32
        );
        let plan = builder.finish_operator_slice(value).unwrap();
        assert_eq!(plan.plan().nodes().len(), 3);
        assert!(plan.plan().required_domains().is_empty());
        assert!(plan.plan().proposed_domains().is_empty());
        assert_eq!(plan.plan().frontier_delta(), 0);
    }
    #[test]
    fn f32_gamma_wrong_extent_profile_and_width_are_not_implicitly_admitted() {
        let id = SourceIdentity::for_new_allocation().unwrap();
        let mut builder = PlanBuilder::new_operator_slice(
            id,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        let core = producer(&mut builder, 6144);
        let gate = producer(&mut builder, 6144);
        let contract = GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128;
        let good = SourceParameter {
            source: id,
            name: "original.gamma".into(),
            kind: QuantKind::Bf16,
            shape: vec![128],
            bytes: 256,
        };
        for bad in [
            SourceParameter {
                kind: QuantKind::F32,
                bytes: 512,
                ..good.clone()
            },
            SourceParameter {
                bytes: 512,
                ..good.clone()
            },
            SourceParameter {
                source: SourceIdentity::for_new_allocation().unwrap(),
                ..good.clone()
            },
        ] {
            assert!(
                builder
                    .gated_rms_gamma_bf16_profiled(bad, &core, &gate, contract)
                    .is_err()
            );
        }
        builder.profile = ProgramNumericalProfile::GgufSourceV1;
        assert!(
            builder
                .gated_rms_gamma_bf16_profiled(good.clone(), &core, &gate, contract)
                .is_err()
        );
        builder.profile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
        let short = producer(&mut builder, 128);
        assert!(
            builder
                .gated_rms_gamma_bf16_profiled(good, &core, &short, contract)
                .is_err()
        );
        let plan = builder.finish_operator_slice(core).unwrap();
        assert_eq!(plan.plan().nodes().len(), 3);
    }
    #[test]
    fn strided_valid_h48_inputs_require_explicit_pack_before_recording() {
        let id = SourceIdentity::for_new_allocation().unwrap();
        let mut builder = PlanBuilder::new_operator_slice(
            id,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        let backing = producer(&mut builder, 6145);
        let core = builder
            .view(
                &backing,
                TensorView {
                    offset: 0,
                    rows: 2,
                    width: 3072,
                    row_stride: 3073,
                },
            )
            .unwrap();
        let gate = producer(&mut builder, 6144);
        let gamma = SourceParameter {
            source: id,
            name: "original.gamma".into(),
            kind: QuantKind::Bf16,
            shape: vec![128],
            bytes: 256,
        };
        let contract = GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128;
        assert!(
            builder
                .gated_rms_gamma_bf16_profiled(gamma.clone(), &core, &gate, contract)
                .is_err()
        );
        assert_eq!(builder.nodes.len(), 3);
        let packed = builder.contiguous(&core).unwrap();
        let output = builder
            .gated_rms_gamma_bf16_profiled(gamma, &packed, &gate, contract)
            .unwrap();
        let plan = builder.finish_operator_slice(output).unwrap();
        assert_eq!(plan.plan().nodes().len(), 5);
        assert!(plan.plan().proposed_domains().is_empty());
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7234; SHA256 `a3032c27074f9a49545052f3d96062a02539616f45c1d0d8e3c9d578ed75e572`.

````text
//! Pinned original GDN coefficient PREPARATION only; no A_log exp/g/decay.
//! Source modeling762feb6c lines617/619, using accepted sigmoid and softplus.
use super::transcendentals::{
    sigmoid_bf16_widened_f32_contiguous_span, softplus_beta1_threshold20_f32_contiguous16,
};
use crate::accelerator::SourceIdentity;
use crate::quant::QuantKind;
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

/// Private stages only. The caller retains source/admission through this call.
pub struct GdnCoefficientPreparation {
    source: SourceIdentity,
    biased_alpha: Vec<f32>,
    softplus: Vec<f32>,
    beta: Vec<f32>,
}
impl GdnCoefficientPreparation {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn biased_alpha(&self) -> &[f32] {
        &self.biased_alpha
    }
    pub fn softplus(&self) -> &[f32] {
        &self.softplus
    }
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }
    /// Private buffers, with no publication or admission claim.
    pub fn into_parts(self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (self.biased_alpha, self.softplus, self.beta)
    }
}

/// One original [1,1,heads] contiguous a/b call, heads16 or48. Do not flatten
/// independent calls: BF16 sigmoid's vector32 prefix/scalar tail is observable.
/// Raw rank1 dt_bias is widened privately; no learned-source mutation/cache.
/// Physical callers admit the complete private storage before entering.
pub fn prepare_gdn_coefficients(
    source: SourceIdentity,
    dt_bias: SourceTensorRef<'_>,
    alpha_bf16: &[f32],
    beta_input_bf16: &[f32],
) -> Result<GdnCoefficientPreparation> {
    let heads = alpha_bf16.len();
    if !source.is_bound()
        || !matches!(heads, 16 | 48)
        || beta_input_bf16.len() != heads
        || dt_bias.shape != [heads as u64]
    {
        return Err(Error::Malformed(
            "original GDN preparation source/head/span".into(),
        ));
    }
    if !matches!(dt_bias.kind, QuantKind::Bf16 | QuantKind::F32) {
        return Err(Error::Unsupported(
            "original GDN preparation raw dt_bias dtype".into(),
        ));
    }
    dt_bias.validate()?;
    if alpha_bf16
        .iter()
        .chain(beta_input_bf16)
        .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed(
            "original GDN preparation finite BF16 a/b".into(),
        ));
    }
    let dt_f32 = match dt_bias.kind {
        QuantKind::Bf16 => dt_bias
            .packed_bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect::<Vec<_>>(),
        QuantKind::F32 => dt_bias
            .packed_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("validated raw F32 dt_bias")))
            .collect(),
        _ => unreachable!("validated dt_bias dtype"),
    };
    if dt_f32.iter().any(|v| !v.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite original GDN raw dt_bias".into(),
        ));
    }
    // Original line617 executes beta first; source line619 widens a before
    // addition to raw dt_bias, then materializes FP32 softplus separately.
    let beta = sigmoid_bf16_widened_f32_contiguous_span(source, beta_input_bf16)?;
    let biased_alpha = alpha_bf16
        .iter()
        .zip(dt_f32)
        .map(|(a, dt)| a + dt)
        .collect::<Vec<_>>();
    if biased_alpha.iter().any(|v| !v.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite original GDN biased alpha".into(),
        ));
    }
    let softplus = softplus_beta1_threshold20_f32_contiguous16(source, &biased_alpha)?;
    Ok(GdnCoefficientPreparation {
        source,
        biased_alpha,
        softplus,
        beta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    #[test]
    fn linear_softplus_and_zero_beta_are_independent_exact_stages() {
        for heads in [16, 48] {
            let raw = vec![0u8; heads * 2];
            let a = vec![21.0; heads];
            let b = vec![0.0; heads];
            let saved = raw.clone();
            let out = prepare_gdn_coefficients(
                source(),
                SourceTensorRef {
                    kind: QuantKind::Bf16,
                    shape: &[heads as u64],
                    packed_bytes: &raw,
                },
                &a,
                &b,
            )
            .unwrap();
            assert!(
                out.biased_alpha()
                    .iter()
                    .all(|v| v.to_bits() == 21.0f32.to_bits())
            );
            assert!(
                out.softplus()
                    .iter()
                    .all(|v| v.to_bits() == 21.0f32.to_bits())
            );
            assert!(out.beta().iter().all(|v| v.to_bits() == 0.5f32.to_bits()));
            assert_eq!(raw, saved);
        }
    }
    #[test]
    fn raw_f32_dt_is_not_bf16_rounded_or_adjusted() {
        let word = 0x3380_0001u32;
        let raw = [word.to_le_bytes(); 16].concat();
        let zero = [0.0; 16];
        let out = prepare_gdn_coefficients(
            source(),
            SourceTensorRef {
                kind: QuantKind::F32,
                shape: &[16],
                packed_bytes: &raw,
            },
            &zero,
            &zero,
        )
        .unwrap();
        assert!(out.biased_alpha().iter().all(|v| v.to_bits() == word));
        assert_eq!(raw, [word.to_le_bytes(); 16].concat());
    }
    #[test]
    fn source_raw_dtype_span_and_nonfinite_refuse() {
        let raw = vec![0u8; 32];
        let zero = [0.0; 16];
        let dt = SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &[16],
            packed_bytes: &raw,
        };
        assert!(prepare_gdn_coefficients(SourceIdentity::unbound(), dt, &zero, &zero).is_err());
        assert!(prepare_gdn_coefficients(source(), dt, &zero[..15], &zero[..15]).is_err());
        assert!(
            prepare_gdn_coefficients(
                source(),
                SourceTensorRef {
                    shape: &[1, 16],
                    ..dt
                },
                &zero,
                &zero
            )
            .is_err()
        );
        assert!(
            prepare_gdn_coefficients(
                source(),
                SourceTensorRef {
                    kind: QuantKind::F16,
                    ..dt
                },
                &zero,
                &zero
            )
            .is_err()
        );
        let mut bad = zero;
        bad[1] = f32::from_bits(0x3f80_0001);
        assert!(prepare_gdn_coefficients(source(), dt, &bad, &zero).is_err());
        let mut bad_raw = raw.clone();
        bad_raw[..2].copy_from_slice(&0x7fc1u16.to_le_bytes());
        assert!(
            prepare_gdn_coefficients(
                source(),
                SourceTensorRef {
                    packed_bytes: &bad_raw,
                    ..dt
                },
                &zero,
                &zero
            )
            .is_err()
        );
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5021; SHA256 `ee45b5047bab51388b9fd9f3e60ed9446798fa919cfbd9c379d1c19772700611`.

````text
//! Distinct original27B BF16 gamma rounding, never the F32 gamma prefix.
use super::{EPSILON, WIDTH, round_bf16, sum_inner_128};
use crate::accelerator::SourceIdentity;
use crate::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16;
use crate::quant::QuantKind;
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

const HEADS: usize = 48;
const ELEMENTS: usize = HEADS * WIDTH;

/// Private values only. Original source/state/publication authority stays outside.
#[derive(Debug)]
pub struct Bf16GammaGatedRms128 {
    source: SourceIdentity,
    mean: Vec<f32>,
    inverse: Vec<f32>,
    normalized_bf16: Vec<f32>,
    weighted_bf16: Vec<f32>,
    silu_fp32: Vec<f32>,
    product_fp32: Vec<f32>,
    output_bf16: Vec<f32>,
}
impl Bf16GammaGatedRms128 {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }
    pub fn inverse(&self) -> &[f32] {
        &self.inverse
    }
    pub fn normalized_bf16(&self) -> &[f32] {
        &self.normalized_bf16
    }
    pub fn weighted_bf16(&self) -> &[f32] {
        &self.weighted_bf16
    }
    pub fn silu_fp32(&self) -> &[f32] {
        &self.silu_fp32
    }
    pub fn product_fp32(&self) -> &[f32] {
        &self.product_fp32
    }
    pub fn output_bf16(&self) -> &[f32] {
        &self.output_bf16
    }
}

fn exact_bf16(input: &[f32]) -> bool {
    input
        .iter()
        .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
}
fn validate(
    source: SourceIdentity,
    gamma: SourceTensorRef<'_>,
    core: &[f32],
    gate: &[f32],
) -> Result<[f32; WIDTH]> {
    if !source.is_bound() {
        return Err(Error::State("gated RMS BF16 gamma source".into()));
    }
    if gamma.kind != QuantKind::Bf16 {
        return Err(Error::Unsupported(
            "gated RMS requires original BF16 gamma".into(),
        ));
    }
    if gamma.shape != [128]
        || gamma.packed_bytes.len() != 256
        || core.len() != ELEMENTS
        || gate.len() != ELEMENTS
    {
        return Err(Error::Malformed(
            "gated RMS BF16 gamma H48 width128 extent".into(),
        ));
    }
    gamma.validate()?;
    if !exact_bf16(core) || !exact_bf16(gate) {
        return Err(Error::Malformed(
            "gated RMS finite exact BF16 core/gate".into(),
        ));
    }
    let weights = std::array::from_fn(|i| {
        f32::from_bits(
            u32::from(u16::from_le_bytes([
                gamma.packed_bytes[2 * i],
                gamma.packed_bytes[2 * i + 1],
            ])) << 16,
        )
    });
    if !exact_bf16(&weights) {
        return Err(Error::Malformed("gated RMS finite original gamma".into()));
    }
    Ok(weights)
}
fn gamma_product(normalized: f32, weight: f32) -> Result<f32> {
    // Original BF16 Tensor multiply rounds HERE, before FP32 SiLU product.
    round_bf16(weight * normalized)
}

/// Genuine raw BF16 gamma128 plus `[48,128]` BF16 core/z, epsilon1e-6.
/// Same accepted mean/RNE/SLEEF helpers; no exp/A_log/initial/cache/model engine.
pub fn gated_rms_128_bf16_gamma_h48(
    source: SourceIdentity,
    gamma: SourceTensorRef<'_>,
    core: &[f32],
    gate: &[f32],
) -> Result<Bf16GammaGatedRms128> {
    // Every source/codec/extent/precision guard precedes private allocations/math.
    let gamma = validate(source, gamma, core, gate)?;
    let squares = core.iter().map(|&v| v * v).collect::<Vec<_>>();
    let sums = sum_inner_128(source, &squares)?;
    let mut mean = Vec::with_capacity(HEADS);
    let mut inverse = Vec::with_capacity(HEADS);
    let mut normalized_bf16 = Vec::with_capacity(ELEMENTS);
    let mut weighted_bf16 = Vec::with_capacity(ELEMENTS);
    for (row, sum) in core.chunks_exact(WIDTH).zip(sums) {
        let m = sum / WIDTH as f32;
        let denominator = m + EPSILON;
        let inv = 1.0f32 / denominator.sqrt();
        if !denominator.is_finite() || !inv.is_finite() {
            return Err(Error::Malformed("nonfinite gated RMS inverse".into()));
        }
        mean.push(m);
        inverse.push(inv);
        for (&v, &w) in row.iter().zip(&gamma) {
            let normalized = round_bf16(v * inv)?;
            normalized_bf16.push(normalized);
            weighted_bf16.push(gamma_product(normalized, w)?);
        }
    }
    let silu_fp32 = silu_bf16_widened_f32_contiguous16(source, gate)?;
    let mut product_fp32 = Vec::with_capacity(ELEMENTS);
    let mut output_bf16 = Vec::with_capacity(ELEMENTS);
    for (&w, &s) in weighted_bf16.iter().zip(&silu_fp32) {
        let product = w * s;
        if !product.is_finite() {
            return Err(Error::Malformed("nonfinite gated RMS gate product".into()));
        }
        product_fp32.push(product);
        output_bf16.push(round_bf16(product)?);
    }
    Ok(Bf16GammaGatedRms128 {
        source,
        mean,
        inverse,
        normalized_bf16,
        weighted_bf16,
        silu_fp32,
        product_fp32,
        output_bf16,
    })
}

#[cfg(test)]
mod tests;

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 2499; SHA256 `acddd75ad40bc7e0bce36e88cbaeb5aa5168570185e66e4a9e82140f6d162383`.

````text
use super::*;

fn source() -> SourceIdentity {
    SourceIdentity::for_new_allocation().unwrap()
}
fn tensor(raw: &[u8]) -> SourceTensorRef<'_> {
    SourceTensorRef {
        kind: QuantKind::Bf16,
        shape: &[128],
        packed_bytes: raw,
    }
}

#[test]
fn bf16_gamma_product_has_its_own_rounding_boundary_before_gate() {
    // Exact dyadic product253/256 *129/128 =32637/32768 =F32 3f7efa00.
    // Closest BF16 is255/256 =3f7f; the former F32 product cannot pass through.
    let n = f32::from_bits(0x3f7d_0000);
    let g = f32::from_bits(0x3f81_0000);
    assert_eq!((g * n).to_bits(), 0x3f7e_fa00);
    assert_eq!(gamma_product(n, g).unwrap().to_bits(), 0x3f7f_0000);
    assert_ne!(gamma_product(n, g).unwrap().to_bits(), (g * n).to_bits());
}

#[test]
fn h48_signed_zero_and_source_identity_survive_private_outputs() {
    let raw = 0x3f80u16.to_le_bytes().repeat(128);
    let mut core = vec![0.0; ELEMENTS];
    core[0] = -0.0;
    let gate = vec![1.0; ELEMENTS];
    let id = source();
    let out = gated_rms_128_bf16_gamma_h48(id, tensor(&raw), &core, &gate).unwrap();
    assert_eq!(out.source_identity(), id);
    assert_eq!(out.mean().len(), 48);
    assert_eq!(out.inverse().len(), 48);
    assert_eq!(out.weighted_bf16().len(), 6144);
    assert_eq!(out.output_bf16().len(), 6144);
    assert_eq!(out.weighted_bf16()[0].to_bits(), 0x8000_0000);
    assert_eq!(out.output_bf16()[0].to_bits(), 0x8000_0000);
    assert!(out.product_fp32().iter().all(|v| v.is_finite()));
}

#[test]
fn source_codec_extent_headcount_and_precision_refuse_without_outputs() {
    let raw = 0x3f80u16.to_le_bytes().repeat(128);
    let core = vec![1.0; ELEMENTS];
    let gate = core.clone();
    let id = source();
    assert!(
        gated_rms_128_bf16_gamma_h48(SourceIdentity::unbound(), tensor(&raw), &core, &gate)
            .is_err()
    );
    for bad in [
        SourceTensorRef {
            kind: QuantKind::F32,
            ..tensor(&raw)
        },
        SourceTensorRef {
            shape: &[1, 128],
            ..tensor(&raw)
        },
        tensor(&raw[..255]),
    ] {
        assert!(gated_rms_128_bf16_gamma_h48(id, bad, &core, &gate).is_err());
    }
    assert!(gated_rms_128_bf16_gamma_h48(id, tensor(&raw), &core[..128], &gate[..128]).is_err());
    for invalid in [f32::NAN, f32::INFINITY, 1.0001] {
        let mut bad = gate.clone();
        bad[13] = invalid;
        assert!(gated_rms_128_bf16_gamma_h48(id, tensor(&raw), &core, &bad).is_err());
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 10274; SHA256 `8a70bd9f52f299ac1cbac9df3e420f1096edd0a5acd77779bcae4d5aac99dcbc`.

````text
//! Full-header synthetic H48 codec/owner checks; no trained arithmetic evidence.
use super::*;
use crate::accelerator::program::{
    EmbeddingContract, ProjectionContract, SourceParameter, SourceTensor,
};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use zerogravity_core::quant::QuantKind;

fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    let text = &mut cfg["text_config"];
    text["num_hidden_layers"] = json!(2);
    text["full_attention_interval"] = json!(2);
    text["layer_types"] = json!(["linear_attention", "full_attention"]);
    text["hidden_size"] = json!(256);
    text["linear_num_key_heads"] = json!(16);
    text["linear_num_value_heads"] = json!(48);
    text["linear_key_head_dim"] = json!(128);
    text["linear_value_head_dim"] = json!(128);
    let cfg = serde_json::to_vec(&cfg).unwrap();
    let specs = QwenTextConfig::from_json(&cfg)
        .unwrap()
        .tensor_specs()
        .unwrap();
    let mut header = BTreeMap::new();
    let mut map = BTreeMap::new();
    let mut payload = Vec::new();
    for spec in specs {
        let gamma = spec.name.ends_with("linear_attn.norm.weight");
        let bf16 = gamma || spec.shape.len() != 1 || spec.name.ends_with("dt_bias");
        let begin = payload.len();
        for _ in 0..spec.shape.iter().product::<u64>() {
            if bf16 {
                payload.extend_from_slice(&0x3e80u16.to_le_bytes());
            } else {
                payload.extend_from_slice(&0.25f32.to_le_bytes());
            }
        }
        header.insert(
            spec.name.clone(),
            json!({"dtype":if bf16 {"BF16"} else {"F32"},
            "shape":spec.shape,"data_offsets":[begin,payload.len()]}),
        );
        map.insert(spec.name, "model.safetensors");
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(&payload);
    assert!(bytes.len() < 16 * 1024 * 1024);
    let path = fixture.0.join("model.safetensors");
    fs::write(&path, &bytes).unwrap();
    let index =
        serde_json::to_vec(&json!({"metadata":{"total_size":payload.len()},"weight_map":map}))
            .unwrap();
    let index_sha = digest(&index);
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "cached-h48-source-test").unwrap();
    arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
    let host = TransferAdmission {
        arbiter,
        resource: key,
        owner: ResourceOwner::new("synthetic", "cached-h48").unwrap(),
    };
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
    let source = SourceWeights::prepare_verified_windows(
        &index,
        &index_sha,
        vec![
            ExpectedSourceShard::new(
                "model.safetensors".into(),
                artifact,
                bytes.len() as u64,
                digest(&bytes),
            )
            .unwrap(),
        ],
        WindowOpenLimits {
            max_source_bytes: 16 * 1024 * 1024,
            max_metadata_bytes: 32 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 200,
            scan_chunk_bytes: 64 * 1024,
        },
        &host,
    )
    .unwrap()
    .open()
    .unwrap();
    (source, host, cfg, index_sha)
}

fn h48_raw() -> [Vec<u8>; 6] {
    [
        0x3f00u16.to_le_bytes().repeat(48 * 128),
        0xbe80u16.to_le_bytes().repeat(48 * 128),
        0x3f80u16.to_le_bytes().repeat(48 * 128),
        0x3f00u16.to_le_bytes().repeat(48),
        0.75f32.to_le_bytes().repeat(48),
        (0..48 * 128 * 128)
            .flat_map(|i| if i % 11 == 0 { -0.0f32 } else { 0.125 }.to_le_bytes())
            .collect(),
    ]
}

struct ParameterFault<'a> {
    source: &'a OriginalWindowOperators,
    norm: SourceParameter,
    profile: ProgramNumericalProfile,
}
impl OperatorSliceSource for ParameterFault<'_> {
    fn geometry(&self) -> &crate::model::ModelGeometry {
        self.source.geometry()
    }
    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source.source_identity()
    }
    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    fn rms_epsilon(&self) -> f32 {
        self.source.rms_epsilon()
    }
    fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
        self.source.matrix_descriptor(role)
    }
    fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<SourceParameter> {
        assert_eq!(role, QwenParameterRole::GdnNorm(0));
        Ok(self.norm.clone())
    }
    fn embedding_contract(&self) -> EmbeddingContract {
        self.source.embedding_contract()
    }
    fn projection_contract(&self) -> ProjectionContract {
        self.source.projection_contract()
    }
}

#[test]
fn actual_f32_and_bf16_source_metadata_allow_only_the_unused_gamma_extent() {
    let old = Fixture::new();
    let loaded = cached_source(&old);
    let (source, _) = windows(&old, &loaded);
    let old_ops = bind_windows(source, &loaded);
    assert_eq!(
        old_ops
            .parameter_descriptor(QwenParameterRole::GdnNorm(0))
            .unwrap()
            .kind,
        QuantKind::F32
    );
    assert_eq!(
        crate::operator_slice::cached_gdn::cached_heads(&old_ops, 0).unwrap(),
        2
    );

    let fixture = Fixture::new();
    let (source, host, cfg, index) = h48_source(&fixture);
    let ops = OriginalWindowOperators::bind(
        source.clone(),
        &cfg,
        &digest(&cfg),
        &index,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    let norm = ops
        .parameter_descriptor(QwenParameterRole::GdnNorm(0))
        .unwrap();
    assert_eq!(
        (norm.kind, norm.shape.clone(), norm.bytes),
        (QuantKind::Bf16, vec![128], 256)
    );
    assert_eq!(
        crate::operator_slice::cached_gdn::cached_heads(&ops, 0).unwrap(),
        48
    );
    let baseline = reserved(&host);
    for (kind, bytes, shape, foreign, profile) in [
        (
            QuantKind::F16,
            256,
            vec![128],
            false,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            512,
            vec![128],
            false,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::F32,
            256,
            vec![128],
            false,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            256,
            vec![127],
            false,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            256,
            vec![128],
            true,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            256,
            vec![128],
            false,
            ProgramNumericalProfile::GgufSourceV1,
        ),
    ] {
        let mut bad = norm.clone();
        bad.kind = kind;
        bad.bytes = bytes;
        bad.shape = shape;
        if foreign {
            bad.source = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
        }
        let fault = ParameterFault {
            source: &ops,
            norm: bad,
            profile,
        };
        assert!(crate::operator_slice::cached_gdn::cached_heads(&fault, 0).is_err());
        assert_eq!(reserved(&host), baseline);
    }
    drop(ops);
    drop(source);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn bf16_h48_cached_fragment_retains_real_owners_and_obeys_cancel_and_pool_pressure() {
    let fixture = Fixture::new();
    let (source, host, cfg, index) = h48_source(&fixture);
    let ops = OriginalWindowOperators::bind(
        source.clone(),
        &cfg,
        &digest(&cfg),
        &index,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    let raw = h48_raw();
    let input = CachedGdnSliceInput::from_bytes(
        source.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        48,
        raw.each_ref().map(Vec::as_slice),
    )
    .unwrap();
    let baseline = reserved(&host);
    assert!(
        ops.execute_cached_gdn_slice(0, input, Some(&AtomicBool::new(true)))
            .is_err()
    );
    assert_eq!(reserved(&host), baseline);
    let needed = (48 * (16384 + 3 * 128 + 2) * 4
        + 48 * (16384 + 2 * 128) * 4
        + crate::host_math::cached_gdn::CACHED_INPUT_SCRATCH
        + crate::host_math::cached_gdn::cached_core_scratch(48).unwrap()) as u64;
    let pressure = host
        .arbiter
        .admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 128 * 1024 * 1024 - baseline - needed + 1,
            }],
        )
        .unwrap();
    let pressured = reserved(&host);
    assert!(ops.execute_cached_gdn_slice(0, input, None).is_err());
    assert_eq!(reserved(&host), pressured);
    drop(pressure);
    let result = ops.execute_cached_gdn_slice(0, input, None).unwrap();
    assert_eq!(result.heads(), 48);
    assert_eq!(result.core_f32().unwrap().len(), 6144);
    assert_eq!(result.output_bf16().unwrap().len(), 6144);
    assert_eq!(result.next_state_key_major().unwrap().len(), 786432);
    assert_eq!(reserved(&host), baseline + needed);
    let expected = result
        .next_state_key_major()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    drop(ops);
    drop(source);
    assert_eq!(
        result
            .next_state_key_major()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    drop(result);
    assert_eq!(reserved(&host), 0);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11198; SHA256 `c2a0115da9308799aacf450c9c1c12f7b05141b9f57d32479aef2e581866a0d4`.

````text
//! Synthetic ownership/schedule checks; genuine archived replay is separate.
#[path = "cached_gdn_h48_tests.rs"]
mod source_gamma;

use super::*;
use crate::accelerator::program::CachedGdnContract;
use crate::operator_slice::cached_gdn::CachedGdnSliceInput;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

fn cached_source(fixture: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["linear_key_head_dim"] = json!(128);
    cfg["text_config"]["linear_value_head_dim"] = json!(128);
    source_with_config(
        fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |_, _, v| v,
        |_| {},
    )
}
fn raw() -> [Vec<u8>; 6] {
    let bf16 = |values: Vec<f32>| {
        values
            .into_iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let fp32 = |values: Vec<f32>| values.into_iter().flat_map(f32::to_le_bytes).collect();
    [
        bf16(
            (0..256)
                .map(|i| if i % 3 == 0 { -0.0 } else { 0.5 })
                .collect(),
        ),
        bf16(
            (0..256)
                .map(|i| if i % 2 == 0 { 0.25 } else { -0.5 })
                .collect(),
        ),
        bf16((0..256).map(|i| if i < 128 { 1.0 } else { -2.0 }).collect()),
        bf16(vec![0.5, 0.25]),
        fp32(vec![0.75, 0.5]),
        fp32(
            (0..32768)
                .map(|i| {
                    if i % 11 == 0 {
                        -0.0
                    } else {
                        (i % 9) as f32 * 0.125
                    }
                })
                .collect(),
        ),
    ]
}
fn input<'a>(
    source: crate::accelerator::SourceIdentity,
    raw: &'a [Vec<u8>; 6],
) -> CachedGdnSliceInput<'a> {
    CachedGdnSliceInput::from_bytes(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        2,
        raw.each_ref().map(|v| v.as_slice()),
    )
    .unwrap()
}
fn imported_bytes() -> u64 {
    2 * (16384 + 3 * 128 + 2) * 4
}
fn result_bytes() -> u64 {
    2 * (16384 + 2 * 128) * 4
}

#[test]
fn cached_fragment_reuses_closed_core_keeps_signed_bits_and_retains_all_outputs() {
    let f = Fixture::new();
    let loaded = cached_source(&f);
    let (source, admission) = windows(&f, &loaded);
    let ops = bind_windows(source.clone(), &loaded);
    let baseline = reserved(&admission);
    let raw = raw();
    let actual = ops
        .execute_cached_gdn_slice(0, input(source.source_identity(), &raw), None)
        .unwrap();
    let originals = actual.original_inputs().unwrap();
    for (index, (raw, values)) in raw.iter().zip(originals).enumerate() {
        if index < 4 {
            assert_eq!(raw.len(), values.len() * 2);
            for (bytes, value) in raw.chunks_exact(2).zip(values) {
                assert_eq!(
                    value.to_bits(),
                    u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16
                );
            }
        } else {
            assert_eq!(raw.len(), values.len() * 4);
            for (bytes, value) in raw.chunks_exact(4).zip(values) {
                assert_eq!(
                    value.to_bits(),
                    u32::from_le_bytes(bytes.try_into().unwrap())
                );
            }
        }
    }

    let expected = zerogravity_cpu::original_hf::recurrent::cached_step_128(
        zerogravity_cpu::original_hf::recurrent::CachedGdn128Inputs {
            source: source.source_identity(),
            heads: 2,
            query: originals[0],
            key: originals[1],
            value: originals[2],
            beta: originals[3],
            decay: originals[4],
            previous_state: originals[5],
        },
    )
    .unwrap();
    same(&expected.core_f32, actual.core_f32().unwrap());
    same(&expected.output_bf16, actual.output_bf16().unwrap());
    same(&expected.state, actual.next_state_key_major().unwrap());
    assert_eq!(actual.layer(), 0);
    assert_eq!(actual.heads(), 2);
    let scratch = crate::host_math::cached_gdn::CACHED_INPUT_SCRATCH as u64
        + crate::host_math::cached_gdn::cached_core_scratch(2).unwrap() as u64;
    assert_eq!(
        reserved(&admission),
        baseline + imported_bytes() + result_bytes() + scratch
    );
    let snapshot = admission.arbiter.snapshot(&admission.resource).unwrap();
    assert_eq!(
        snapshot.per_class[&AllocationClass::Output],
        imported_bytes() + result_bytes()
    );
    assert_eq!(
        snapshot
            .per_class
            .get(&AllocationClass::TransferStaging)
            .copied()
            .unwrap_or(0),
        0
    );
    let (model, _, _) = bind(loaded);
    assert!(model.new_session().is_err());
    assert!(model.forward_plan(8, true, false).is_err());
    drop(ops);
    drop(source);
    same(&expected.state, actual.next_state_key_major().unwrap());
    same(&expected.output_bf16, actual.output_bf16().unwrap());
    assert!(reserved(&admission) > 0);
    drop(actual);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn bad_raw_phase_representation_and_finite_boundaries_refuse() {
    let id = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
    let original = raw();
    input(id, &original);
    for (index, bad) in [
        (0, vec![0; 511]),
        (3, vec![0, 0x40, 0, 0x3f]),
        (4, f32::NAN.to_le_bytes().repeat(2)),
        (4, 2f32.to_le_bytes().repeat(2)),
        (5, f32::INFINITY.to_le_bytes().repeat(32768)),
    ] {
        let mut changed = original.clone();
        changed[index] = bad;
        assert!(
            CachedGdnSliceInput::from_bytes(
                id,
                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
                2,
                changed.each_ref().map(|v| v.as_slice())
            )
            .is_err()
        );
    }
    // Zero state is legal cached input; it never becomes initial chunk64.
    let mut zero = original;
    zero[5] = vec![0; 32768 * 4];
    input(id, &zero);
}
#[test]
fn real_pressure_pre_cancel_and_after_import_cancel_release_only_owned_leases() {
    let f = Fixture::new();
    let loaded = cached_source(&f);
    let (source, admission) = windows(&f, &loaded);
    let ops = bind_windows(source.clone(), &loaded);
    let baseline = reserved(&admission);
    let raw = raw();
    let value = input(source.source_identity(), &raw);
    let cancelled = AtomicBool::new(true);
    assert!(
        ops.execute_cached_gdn_slice(0, value, Some(&cancelled))
            .is_err()
    );
    assert_eq!(reserved(&admission), baseline);
    let core_scratch = crate::host_math::cached_gdn::cached_core_scratch(2).unwrap() as u64;
    let needed = imported_bytes()
        + result_bytes()
        + core_scratch
        + crate::host_math::cached_gdn::CACHED_INPUT_SCRATCH as u64;
    let pressure = admission
        .arbiter
        .admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 128 * 1024 * 1024 - baseline - needed + 1,
            }],
        )
        .unwrap();
    let before = reserved(&admission);
    assert!(ops.execute_cached_gdn_slice(0, value, None).is_err());
    assert_eq!(reserved(&admission), before);
    drop(pressure);
    cancelled.store(false, Ordering::Release);
    let mut math =
        crate::host_math::HostMath::for_window_operator_slice(&ops, Some(&cancelled), None);
    let operands = math.import_cached_gdn_input(value, 2).unwrap();
    let imported = reserved(&admission);
    cancelled.store(true, Ordering::Release);
    assert!(
        math.cached_gdn_profiled(
            source.source_identity(),
            2,
            &operands,
            CachedGdnContract::TorchCpuBf16Avx2Width128
        )
        .is_err()
    );
    assert_eq!(reserved(&admission), imported);
    cancelled.store(false, Ordering::Release);
    let results = math
        .cached_gdn_profiled(
            source.source_identity(),
            2,
            &operands,
            CachedGdnContract::TorchCpuBf16Avx2Width128,
        )
        .unwrap();
    let clone = results.next_state.clone();
    assert!(clone.clone().into_vec().is_err());
    drop(results);
    assert_eq!(clone.as_slice().unwrap().len(), 32768);
    drop(clone);
    assert_eq!(reserved(&admission), imported);
    drop(operands);
    assert_eq!(reserved(&admission), baseline);
    drop(ops);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn foreign_issuer_strided_inputs_unaware_provider_and_wrong_layer_refuse() {
    struct Unaware;
    impl crate::accelerator::MatVecKernel for Unaware {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            panic!("cached GDN must refuse before unaware provider callback")
        }
    }
    let f = Fixture::new();
    let loaded = cached_source(&f);
    let (source, admission) = windows(&f, &loaded);
    let ops = bind_windows(source.clone(), &loaded);
    let baseline = reserved(&admission);
    let raw = raw();
    let value = input(source.source_identity(), &raw);
    assert!(ops.execute_cached_gdn_slice(3, value, None).is_err());
    let provider = Unaware;
    let unaware =
        crate::host_math::HostMath::for_window_operator_slice(&ops, None, Some(&provider));
    assert!(unaware.import_cached_gdn_input(value, 2).is_err());
    assert_eq!(reserved(&admission), baseline);
    let mut math = crate::host_math::HostMath::for_window_operator_slice(&ops, None, None);
    let operands = math.import_cached_gdn_input(value, 2).unwrap();
    let before = reserved(&admission);
    let strided = math
        .view(
            &operands.previous_state,
            TensorView {
                offset: 0,
                rows: 2,
                width: 1,
                row_stride: 2,
            },
        )
        .unwrap();
    let mut bad = operands.clone();
    bad.decay = strided;
    let error = math
        .cached_gdn_profiled(
            source.source_identity(),
            2,
            &bad,
            CachedGdnContract::TorchCpuBf16Avx2Width128,
        )
        .err()
        .unwrap();
    assert!(error.to_string().contains("contiguous"));
    assert_eq!(reserved(&admission), before);
    let foreign_source = windows(&f, &loaded).0;
    let foreign_ops = bind_windows(foreign_source, &loaded);
    let mut foreign =
        crate::host_math::HostMath::for_window_operator_slice(&foreign_ops, None, None);
    assert!(
        foreign
            .cached_gdn_profiled(
                source.source_identity(),
                2,
                &operands,
                CachedGdnContract::TorchCpuBf16Avx2Width128
            )
            .is_err()
    );
    drop(foreign_ops);
    drop(bad);
    drop(operands);
    assert_eq!(reserved(&admission), baseline);
    drop(ops);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 8890; SHA256 `74b8a311402e42c04cd3060dd34a8c4dfa05dd2d061f4d2ae89f896eecb7fec8`.

````text
//! Synthetic source/transport/lease guards only; never trained coefficients.
use super::*;
use crate::accelerator::program::{
    GdnCoefficientPreparationContract, PlanBuilder, PlanOperation, ProgramNumericalProfile,
};
use crate::operator_slice::coefficient_preparation::GdnCoefficientPreparationInput;
fn coefficient_source(fixture: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["linear_num_value_heads"] = json!(16);
    source_with_config(
        fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |name, _, fallback| {
            if name.ends_with(".dt_bias") {
                0
            } else {
                fallback
            }
        },
        |_| {},
    )
}
fn raw(value: u16) -> Vec<u8> {
    [value.to_le_bytes(); 16].concat()
}
fn input<'a>(
    source: crate::accelerator::SourceIdentity,
    a: &'a [u8],
    b: &'a [u8],
) -> GdnCoefficientPreparationInput<'a> {
    GdnCoefficientPreparationInput::from_bf16_bytes(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        a,
        b,
    )
    .unwrap()
}
#[test]
fn synthetic_coefficient_stages_retain_two_aggregate_backings_and_source() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let a = raw(0x41a8);
    let b = raw(0); // a21, b0: public linear softplus and sigmoid0.
    let before = reserved(&admission);
    let output = operators
        .execute_gdn_coefficient_preparation_slice(0, input(source.source_identity(), &a, &b), None)
        .unwrap();
    assert!(
        output
            .biased_alpha_f32()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 21.0f32.to_bits())
    );
    assert!(
        output
            .softplus_f32()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 21.0f32.to_bits())
    );
    assert!(
        output
            .beta_bf16()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 0.5f32.to_bits())
    );
    assert_eq!(
        output.raw_dt_bias_parameter().kind,
        zerogravity_core::quant::QuantKind::Bf16
    );
    assert_eq!(
        output.unevaluated_raw_a_log_parameter().kind,
        zerogravity_core::quant::QuantKind::F32
    );
    assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 34 * 1024 * 1024);
    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 16]);
    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 16]);
    drop(output);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn wrong_layer_foreign_input_cancel_and_pressure_refuse_without_leak() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let zero = raw(0);
    let before = reserved(&admission);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                3,
                input(source.source_identity(), &zero, &zero),
                None
            )
            .is_err()
    );
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(
                    crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
                    &zero,
                    &zero
                ),
                None
            )
            .is_err()
    );
    let cancelled = AtomicBool::new(true);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(source.source_identity(), &zero, &zero),
                Some(&cancelled)
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    let mut pressure = Vec::new();
    loop {
        let remaining = admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .remaining;
        if remaining == 0 {
            break;
        }
        pressure.push(
            source
                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    let pressured = reserved(&admission);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(source.source_identity(), &zero, &zero),
                None
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), pressured);
    drop(pressure);
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn raw_a_log_alias_and_unaware_provider_refuse_before_callback_or_output() {
    struct Provider(AtomicBool);
    impl crate::accelerator::MatVecKernel for Provider {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            panic!("unaware coefficient provider invoked");
        }
    }
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let zero = raw(0);
    let (a, b) = {
        let math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
        math.import_coefficient_preparation_input(input(source.source_identity(), &zero, &zero), 16)
            .unwrap()
    };
    let before = reserved(&admission);
    let provider = Provider(AtomicBool::new(false));
    {
        let mut math = crate::host_math::HostMath::for_window_operator_slice(
            &operators,
            None,
            Some(&provider),
        );
        assert!(
            math.coefficient_preparation_profiled(
                operators
                    .parameter_descriptor(QwenParameterRole::GdnDt(0))
                    .unwrap(),
                &a,
                &b,
                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
            )
            .is_err()
        );
    }
    {
        let mut math =
            crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
        assert!(
            math.coefficient_preparation_profiled(
                operators
                    .parameter_descriptor(QwenParameterRole::GdnA(0))
                    .unwrap(),
                &a,
                &b,
                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
            )
            .is_err()
        );
    }
    assert!(!provider.0.load(Ordering::Relaxed));
    assert_eq!(reserved(&admission), before);
    drop(a);
    drop(b);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn plan_appends_preparation_without_exp_state_token_or_producer_invention() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (model, _, _) = bind(loaded);
    let mut builder =
        PlanBuilder::new_operator_slice(model.source_identity(), model.program_numerical_profile())
            .unwrap();
    // Synthetic same-builder views are transport ONLY, never actual trained a/b.
    let embed = builder
        .embed_profiled(
            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let a = builder.view(&embed, TensorView::vector(0, 16)).unwrap();
    let b = builder.view(&embed, TensorView::vector(16, 16)).unwrap();
    let values = model
        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
        .unwrap();
    let plan = builder.finish_operator_slice(values.softplus).unwrap();
    assert_eq!(plan.plan().nodes().len(), 5);
    assert_eq!(
        plan.plan()
            .nodes()
            .iter()
            .filter(|n| matches!(
                n.operation,
                PlanOperation::TypedGdnAlphaPreparation { .. }
                    | PlanOperation::TypedGdnBetaPreparation { .. }
            ))
            .count(),
        2
    );
    assert!(plan.plan().required_domains().is_empty());
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(plan.plan().logits().is_none());
    assert_eq!(plan.plan().frontier_delta(), 0);
    assert!(model.new_session().is_err());
    assert!(model.forward_plan(8, true, true).is_err());
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 13508; SHA256 `b6f7fcb6d8828051b6004b90f10c7b549303d94c9c1bf358ca701d3fda524b40`.

````text
//! Synthetic full-source H48/codec/lease tests; no trained arithmetic proof.
use super::*;
use crate::accelerator::program::{
    EmbeddingContract, ProjectionContract, SourceParameter, SourceTensor,
};
use crate::operator_slice::gated_rms::{GatedRmsSliceInput, gated_rms_role};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

fn h48_source(
    fixture: &Fixture,
    bf16_gamma: bool,
) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    let text = &mut cfg["text_config"];
    text["num_hidden_layers"] = json!(2);
    text["full_attention_interval"] = json!(2);
    text["layer_types"] = json!(["linear_attention", "full_attention"]);
    text["hidden_size"] = json!(256);
    text["linear_num_key_heads"] = json!(16);
    text["linear_num_value_heads"] = json!(48);
    text["linear_key_head_dim"] = json!(128);
    text["linear_value_head_dim"] = json!(128);
    let cfg = serde_json::to_vec(&cfg).unwrap();
    let specs = QwenTextConfig::from_json(&cfg)
        .unwrap()
        .tensor_specs()
        .unwrap();
    let mut header = BTreeMap::new();
    let mut map = BTreeMap::new();
    let mut payload = Vec::new();
    for spec in specs {
        let gamma = spec.name.ends_with("linear_attn.norm.weight");
        let bf16 = (gamma && bf16_gamma) || spec.shape.len() != 1 || spec.name.ends_with("dt_bias");
        let begin = payload.len();
        for _ in 0..spec.shape.iter().product::<u64>() {
            if bf16 {
                payload.extend_from_slice(&0x3e80u16.to_le_bytes());
            } else {
                payload.extend_from_slice(&0.25f32.to_le_bytes());
            }
        }
        header.insert(
            spec.name.clone(),
            json!({"dtype":if bf16 {"BF16"} else {"F32"},
            "shape":spec.shape,"data_offsets":[begin,payload.len()]}),
        );
        map.insert(spec.name, "model.safetensors");
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(&payload);
    assert!(bytes.len() < 16 * 1024 * 1024);
    let path = fixture.0.join("model.safetensors");
    fs::write(&path, &bytes).unwrap();
    let index =
        serde_json::to_vec(&json!({"metadata":{"total_size":payload.len()},"weight_map":map}))
            .unwrap();
    let index_sha = digest(&index);
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "gated-rms-h48-source-test").unwrap();
    arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
    let host = TransferAdmission {
        arbiter,
        resource: key,
        owner: ResourceOwner::new("synthetic", "gated-rms-h48").unwrap(),
    };
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
    let source = SourceWeights::prepare_verified_windows(
        &index,
        &index_sha,
        vec![
            ExpectedSourceShard::new(
                "model.safetensors".into(),
                artifact,
                bytes.len() as u64,
                digest(&bytes),
            )
            .unwrap(),
        ],
        WindowOpenLimits {
            max_source_bytes: 16 * 1024 * 1024,
            max_metadata_bytes: 32 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 200,
            scan_chunk_bytes: 64 * 1024,
        },
        &host,
    )
    .unwrap()
    .open()
    .unwrap();
    (source, host, cfg, index_sha)
}

fn operators(fixture: &Fixture, bf16_gamma: bool) -> (OriginalWindowOperators, TransferAdmission) {
    let (source, host, cfg, index) = h48_source(fixture, bf16_gamma);
    let ops = OriginalWindowOperators::bind(
        source,
        &cfg,
        &digest(&cfg),
        &index,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    (ops, host)
}
fn raw() -> [Vec<u8>; 2] {
    [
        0x3f7du16.to_le_bytes().repeat(6144),
        0xbf00u16.to_le_bytes().repeat(6144),
    ]
}
fn input<'a>(ops: &OriginalWindowOperators, raw: &'a [Vec<u8>; 2]) -> GatedRmsSliceInput<'a> {
    GatedRmsSliceInput::from_bf16_bytes(
        ops.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        &raw[0],
        &raw[1],
    )
    .unwrap()
}
struct ParameterFault<'a> {
    source: &'a OriginalWindowOperators,
    norm: SourceParameter,
    profile: ProgramNumericalProfile,
}
impl OperatorSliceSource for ParameterFault<'_> {
    fn geometry(&self) -> &crate::model::ModelGeometry {
        self.source.geometry()
    }
    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source.source_identity()
    }
    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    fn rms_epsilon(&self) -> f32 {
        self.source.rms_epsilon()
    }
    fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
        self.source.matrix_descriptor(role)
    }
    fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<SourceParameter> {
        assert_eq!(role, QwenParameterRole::GdnNorm(0));
        Ok(self.norm.clone())
    }
    fn embedding_contract(&self) -> EmbeddingContract {
        self.source.embedding_contract()
    }
    fn projection_contract(&self) -> ProjectionContract {
        self.source.projection_contract()
    }
}
#[test]
fn exact_original_bf16_gamma_role_codec_extent_source_and_profile_are_required() {
    let fixture = Fixture::new();
    let (ops, host) = operators(&fixture, true);
    let good = gated_rms_role(&ops, 0).unwrap();
    let before = reserved(&host);
    assert_eq!(
        (good.kind, good.shape.clone(), good.bytes),
        (QuantKind::Bf16, vec![128], 256)
    );
    for mut bad in [
        SourceParameter {
            kind: QuantKind::F32,
            bytes: 512,
            ..good.clone()
        },
        SourceParameter {
            bytes: 512,
            ..good.clone()
        },
        SourceParameter {
            shape: vec![127],
            ..good.clone()
        },
        SourceParameter {
            name: "model.language_model.layers.1.linear_attn.norm.weight".into(),
            ..good.clone()
        },
        SourceParameter {
            source: crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
            ..good.clone()
        },
    ] {
        let fault = ParameterFault {
            source: &ops,
            norm: bad.clone(),
            profile: ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        };
        assert!(gated_rms_role(&fault, 0).is_err());
        assert_eq!(reserved(&host), before);
        // Also test a different unsupported raw codec without changing geometry.
        bad.kind = QuantKind::F16;
        bad.bytes = 256;
        assert!(
            gated_rms_role(
                &ParameterFault {
                    source: &ops,
                    norm: bad,
                    profile: ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
                },
                0
            )
            .is_err()
        );
    }
    assert!(
        gated_rms_role(
            &ParameterFault {
                source: &ops,
                norm: good,
                profile: ProgramNumericalProfile::GgufSourceV1
            },
            0
        )
        .is_err()
    );
    assert!(gated_rms_role(&ops, 1).is_err());
    let f32_fixture = Fixture::new();
    let (f32_ops, f32_host) = operators(&f32_fixture, false);
    assert_eq!(
        f32_ops
            .parameter_descriptor(QwenParameterRole::GdnNorm(0))
            .unwrap()
            .kind,
        QuantKind::F32
    );
    assert!(gated_rms_role(&f32_ops, 0).is_err());
    drop(f32_ops);
    assert_eq!(reserved(&f32_host), 0);
    drop(ops);
    assert_eq!(reserved(&host), 0);
}
#[test]
fn aggregate_cancel_pressure_retained_inputs_and_result_own_their_actual_source() {
    let fixture = Fixture::new();
    let (ops, host) = operators(&fixture, true);
    let before = reserved(&host);
    let raw = raw();
    let input = input(&ops, &raw);
    let cancelled = AtomicBool::new(true);
    assert!(
        ops.execute_gated_rms_bf16_gamma_slice(0, input, Some(&cancelled))
            .is_err()
    );
    assert_eq!(reserved(&host), before);
    // Independent declared H48 payloads and conservative scratch floors.
    let input_need = (2 * 48 * 128 * 4 + 64 * 1024) as u64;
    let output_need = (48 * 128 * 4 + 960 * 1024) as u64;
    let pressure = host
        .arbiter
        .admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 128 * 1024 * 1024 - before - input_need - output_need + 1,
            }],
        )
        .unwrap();
    let pressured = reserved(&host);
    assert!(
        ops.execute_gated_rms_bf16_gamma_slice(0, input, None)
            .is_err()
    );
    assert_eq!(reserved(&host), pressured);
    drop(pressure);
    assert_eq!(reserved(&host), before);
    cancelled.store(false, Ordering::Release);
    let mut math =
        crate::host_math::HostMath::for_window_operator_slice(&ops, Some(&cancelled), None);
    let (core, gate) = math.import_gated_rms_input(input).unwrap();
    assert_eq!(reserved(&host), before + input_need);
    cancelled.store(true, Ordering::Release);
    assert!(crate::forward::gated_rms_bf16_gamma_slice(&ops, &mut math, 0, &core, &gate).is_err());
    assert_eq!(reserved(&host), before + input_need);
    drop(core);
    drop(gate);
    assert_eq!(reserved(&host), before);
    let result = ops
        .execute_gated_rms_bf16_gamma_slice(0, input, None)
        .unwrap();
    assert_eq!(result.core_bf16().unwrap().len(), 6144);
    assert_eq!(result.gate_bf16().unwrap().len(), 6144);
    assert_eq!(result.output_bf16().unwrap().len(), 6144);
    assert_eq!(reserved(&host), before + input_need + output_need);
    let snapshot = host.arbiter.snapshot(&host.resource).unwrap();
    assert_eq!(
        snapshot.per_class.get(&AllocationClass::Output),
        Some(&(3 * 6144 * 4))
    );
    assert_eq!(
        snapshot
            .per_class
            .get(&AllocationClass::TransferStaging)
            .copied()
            .unwrap_or(0),
        0
    );
    // This is shared-helper wiring evidence, not an independent math oracle.
    let parameter = ops
        .source()
        .parameter(&gated_rms_role(&ops, 0).unwrap().name)
        .unwrap();
    let gamma = ops.source().read_parameter(&parameter, 0..128).unwrap();
    let expected = zerogravity_cpu::original_hf::norm::gated_rms_128_bf16_gamma_h48(
        ops.source_identity(),
        zerogravity_core::source::SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &[128],
            packed_bytes: gamma.as_slice(),
        },
        result.core_bf16().unwrap(),
        result.gate_bf16().unwrap(),
    )
    .unwrap();
    same(expected.output_bf16(), result.output_bf16().unwrap());
    drop(expected);
    drop(gamma);
    drop(parameter);
    let bits = result
        .output_bf16()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    drop(ops);
    assert_eq!(
        result
            .output_bf16()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        bits
    );
    assert!(reserved(&host) > 0);
    drop(result);
    assert_eq!(reserved(&host), 0);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
}
#[test]
fn external_bad_bindings_and_optional_provider_refuse_before_callback_or_new_lease() {
    struct Provider(std::sync::atomic::AtomicBool);
    impl crate::accelerator::MatVecKernel for Provider {
        fn try_execute(
            &self,
            _region: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, Ordering::Release);
            panic!("must refuse before provider")
        }
    }
    let fixture = Fixture::new();
    let (ops, host) = operators(&fixture, true);
    let before = reserved(&host);
    let raw = raw();
    for (id, profile) in [
        (
            crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (ops.source_identity(), ProgramNumericalProfile::GgufSourceV1),
    ] {
        let bad = GatedRmsSliceInput::from_bf16_bytes(id, profile, &raw[0], &raw[1]);
        match bad {
            Ok(input) => assert!(
                ops.execute_gated_rms_bf16_gamma_slice(0, input, None)
                    .is_err()
            ),
            Err(_) => assert_eq!(profile, ProgramNumericalProfile::GgufSourceV1),
        }
        assert_eq!(reserved(&host), before);
    }
    let provider = Provider(std::sync::atomic::AtomicBool::new(false));
    let math = crate::host_math::HostMath::for_window_operator_slice(&ops, None, Some(&provider));
    assert!(math.import_gated_rms_input(input(&ops, &raw)).is_err());
    assert!(!provider.0.load(Ordering::Acquire));
    assert_eq!(reserved(&host), before);
    drop(ops);
    assert_eq!(reserved(&host), 0);
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gdn_cached_slice.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 8136; SHA256 `e80ebb297e18e597a6177c38780b282b29e89fe294e6858d8c97bfa6b686eb7b`.

````text
//! Observed cached-token operands only; no coefficient producer/cache owner.
use super::OperatorSliceSource;
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    CachedGdnContract, CachedGdnOperands, CachedGdnValues, PlanBuilder, PlanValue,
    ProgramNumericalProfile,
};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, LoadedQwenModel, Result};

pub(crate) fn cached_heads(source: &impl OperatorSliceSource, layer: usize) -> Result<usize> {
    if !source.source_identity().is_bound()
        || source.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
    {
        return Err(Error::Unsupported(
            "cached GDN original profile/layer".into(),
        ));
    }
    let heads = usize::try_from(source.geometry().recurrent_value_heads.unwrap_or(0))
        .map_err(|_| Error::Limit("cached GDN value heads".into()))?;
    let keys = usize::try_from(source.geometry().recurrent_key_heads.unwrap_or(0))
        .map_err(|_| Error::Limit("cached GDN key heads".into()))?;
    if keys == 0 || !heads.is_multiple_of(keys) || heads == 0 || heads > 64 {
        return Err(Error::Unsupported(
            "cached GDN repeated-head geometry".into(),
        ));
    }
    let qkv = source.matrix_descriptor(QwenMatrixRole::GdnQkv(layer))?;
    let output = source.matrix_descriptor(QwenMatrixRole::GdnOutput(layer))?;
    let norm = source.parameter_descriptor(QwenParameterRole::GdnNorm(layer))?;
    if qkv.source != source.source_identity()
        || output.source != source.source_identity()
        || qkv.rows != (keys * 2 + heads) * 128
        || output.columns != heads * 128
        || norm.source != source.source_identity()
        || norm.shape != [128]
        // Gamma is source metadata only here; cached recurrence never reads it.
        || !matches!(
            (norm.kind, norm.bytes),
            (zerogravity_core::quant::QuantKind::F32, 512)
                | (zerogravity_core::quant::QuantKind::Bf16, 256)
        )
    {
        return Err(Error::Unsupported(
            "cached GDN original width128/source geometry".into(),
        ));
    }
    Ok(heads)
}

/// Raw source-function inputs AFTER Q/K repeat; qualifier authenticates archive
/// provenance. Finite FP32 decay is explicitly supplied, never computed here.
#[derive(Clone, Copy)]
pub struct CachedGdnSliceInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    heads: usize,
    words: [&'a [u8]; 6],
}
impl<'a> CachedGdnSliceInput<'a> {
    /// q/k/v/beta are BF16 LE; decay and key-major prior state are FP32 LE.
    /// A zero prior still selects CACHED arithmetic, never initial chunk64.
    pub fn from_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        heads: usize,
        words: [&'a [u8]; 6],
    ) -> Result<Self> {
        let result = Self {
            source,
            profile,
            heads,
            words,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn heads(self) -> usize {
        self.heads
    }
    pub fn raw_operands(self) -> [&'a [u8]; 6] {
        self.words
    }
    pub(crate) fn validate(self) -> Result<()> {
        if !self.source.is_bound()
            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported("cached GDN input source/profile".into()));
        }
        let mut lengths = [0; 6];
        for (index, bytes) in self.words.iter().enumerate() {
            let size = if index < 4 { 2 } else { 4 };
            if bytes.len() % size != 0 {
                return Err(Error::Malformed("cached GDN partial raw word".into()));
            }
            lengths[index] = bytes.len() / size;
        }
        CachedGdnContract::TorchCpuBf16Avx2Width128.validate(self.heads, &lengths)?;
        for (index, bytes) in self.words.iter().enumerate() {
            if index < 4 {
                for raw in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([raw[0], raw[1]]);
                    let v = f32::from_bits(u32::from(bits) << 16);
                    if !v.is_finite() || index == 3 && !(0.0..=1.0).contains(&v) {
                        return Err(Error::Malformed("cached GDN finite BF16/beta".into()));
                    }
                }
            } else {
                for raw in bytes.chunks_exact(4) {
                    let v = f32::from_bits(u32::from_le_bytes(raw.try_into().unwrap()));
                    if !v.is_finite() || index == 4 && !(0.0..=1.0).contains(&v) {
                        return Err(Error::Malformed("cached GDN finite FP32/decay".into()));
                    }
                }
            }
        }
        Ok(())
    }
}

impl LoadedQwenModel {
    pub fn append_cached_gdn_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        input: &CachedGdnOperands<PlanValue>,
    ) -> Result<CachedGdnValues<PlanValue>> {
        self.require_operator_slice()?;
        crate::forward::cached_gdn_slice(self, builder, layer, input)
    }
}

#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
pub struct CachedGdnSliceOutput {
    source: SourceIdentity,
    layer: usize,
    heads: usize,
    input: CachedGdnOperands<crate::host_math::HostValue>,
    values: CachedGdnValues<crate::host_math::HostValue>,
}
#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
impl CachedGdnSliceOutput {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn heads(&self) -> usize {
        self.heads
    }
    pub fn original_inputs(&self) -> Result<[&[f32]; 6]> {
        Ok([
            self.input.query.as_slice()?,
            self.input.key.as_slice()?,
            self.input.value.as_slice()?,
            self.input.beta.as_slice()?,
            self.input.decay.as_slice()?,
            self.input.previous_state.as_slice()?,
        ])
    }
    pub fn core_f32(&self) -> Result<&[f32]> {
        self.values.core_f32.as_slice()
    }
    pub fn output_bf16(&self) -> Result<&[f32]> {
        self.values.output_bf16.as_slice()
    }
    pub fn next_state_key_major(&self) -> Result<&[f32]> {
        self.values.next_state.as_slice()
    }
}
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl crate::original_binding::OriginalWindowOperators {
    pub fn append_cached_gdn_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        input: &CachedGdnOperands<PlanValue>,
    ) -> Result<CachedGdnValues<PlanValue>> {
        crate::forward::cached_gdn_slice(self, builder, layer, input)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_cached_gdn_slice(
        &self,
        layer: usize,
        input: CachedGdnSliceInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<CachedGdnSliceOutput> {
        let heads = cached_heads(self, layer)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let operands = math.import_cached_gdn_input(input, heads)?;
        let values = crate::forward::cached_gdn_slice(self, &mut math, layer, &operands)?;
        if cancelled.is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State("cached GDN fragment cancelled".into()));
        }
        Ok(CachedGdnSliceOutput {
            source: self.source_identity(),
            layer,
            heads,
            input: operands,
            values,
        })
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gdn_gated_rms_slice.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6519; SHA256 `c0ee66540859eda49f5f2ff17f1ba26adf28864445eca01ec5380fc10d7a7ad2`.

````text
//! One pure original BF16 gamma gated RMS call on explicit observed operands.
use super::OperatorSliceSource;
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    GatedRmsGammaBf16Contract, MatrixAxes, PlanBuilder, PlanValue, ProgramNumericalProfile,
    SourceParameter,
};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, LoadedQwenModel, Result};
use zerogravity_core::quant::QuantKind;

pub(crate) fn gated_rms_role(
    source: &impl OperatorSliceSource,
    layer: usize,
) -> Result<SourceParameter> {
    if layer != 0
        || !source.source_identity().is_bound()
        || source.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
        || source.geometry().recurrent_key_heads != Some(16)
        || source.geometry().recurrent_value_heads != Some(48)
        || source.rms_epsilon().to_bits() != 1e-6f32.to_bits()
    {
        return Err(Error::Unsupported(
            "original gated RMS BF16 gamma layer0/H48 profile".into(),
        ));
    }
    let parameter = source.parameter_descriptor(QwenParameterRole::GdnNorm(layer))?;
    let qkv = source.matrix_descriptor(QwenMatrixRole::GdnQkv(layer))?;
    let gate = source.matrix_descriptor(QwenMatrixRole::GdnGate(layer))?;
    let output = source.matrix_descriptor(QwenMatrixRole::GdnOutput(layer))?;
    let hidden = usize::try_from(source.geometry().hidden_size)
        .map_err(|_| Error::Limit("gated RMS original hidden width".into()))?;
    if parameter.source != source.source_identity()
        || parameter.name != "model.language_model.layers.0.linear_attn.norm.weight"
        || qkv.source != source.source_identity()
        || qkv.rows != 10240
        || gate.source != source.source_identity()
        || gate.rows != 6144
        || output.source != source.source_identity()
        || output.columns != 6144
        || qkv.columns != hidden
        || gate.columns != hidden
        || output.rows != hidden
        || [&qkv, &gate, &output]
            .iter()
            .any(|matrix| matrix.kind != QuantKind::Bf16 || matrix.axes != MatrixAxes::OutputInput)
    {
        return Err(Error::State(
            "original gated RMS raw role/width/source".into(),
        ));
    }
    GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128.validate(&parameter, 6144, 6144)?;
    Ok(parameter)
}

/// Representation only. The qualifier authenticates actual source call provenance.
#[derive(Clone, Copy)]
pub struct GatedRmsSliceInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    core: &'a [u8],
    gate: &'a [u8],
}
impl<'a> GatedRmsSliceInput<'a> {
    pub fn from_bf16_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        core: &'a [u8],
        gate: &'a [u8],
    ) -> Result<Self> {
        let value = Self {
            source,
            profile,
            core,
            gate,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn raw_inputs(self) -> [&'a [u8]; 2] {
        [self.core, self.gate]
    }
    pub(crate) fn validate(self) -> Result<()> {
        if !self.source.is_bound()
            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || self.core.len() != 6144 * 2
            || self.gate.len() != 6144 * 2
            || [self.core, self.gate].iter().any(|raw| {
                raw.chunks_exact(2)
                    .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
            })
        {
            return Err(Error::Malformed(
                "original gated RMS finite H48 BF16 operands/source/profile".into(),
            ));
        }
        Ok(())
    }
}
impl LoadedQwenModel {
    pub fn append_gated_rms_bf16_gamma_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        core: &PlanValue,
        gate: &PlanValue,
    ) -> Result<PlanValue> {
        self.require_operator_slice()?;
        crate::forward::gated_rms_bf16_gamma_slice(self, builder, layer, core, gate)
    }
}

#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
pub struct GatedRmsSliceOutput {
    source: SourceIdentity,
    core: crate::host_math::HostValue,
    gate: crate::host_math::HostValue,
    output: crate::host_math::HostValue,
}
#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
impl GatedRmsSliceOutput {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn core_bf16(&self) -> Result<&[f32]> {
        self.core.as_slice()
    }
    pub fn gate_bf16(&self) -> Result<&[f32]> {
        self.gate.as_slice()
    }
    pub fn output_bf16(&self) -> Result<&[f32]> {
        self.output.as_slice()
    }
}
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl crate::original_binding::OriginalWindowOperators {
    pub fn append_gated_rms_bf16_gamma_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        core: &PlanValue,
        gate: &PlanValue,
    ) -> Result<PlanValue> {
        crate::forward::gated_rms_bf16_gamma_slice(self, builder, layer, core, gate)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_gated_rms_bf16_gamma_slice(
        &self,
        layer: usize,
        input: GatedRmsSliceInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<GatedRmsSliceOutput> {
        gated_rms_role(self, layer)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let (core, gate) = math.import_gated_rms_input(input)?;
        let output =
            crate::forward::gated_rms_bf16_gamma_slice(self, &mut math, layer, &core, &gate)?;
        if cancelled.is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State("gated RMS private fragment cancelled".into()));
        }
        Ok(GatedRmsSliceOutput {
            source: self.source_identity(),
            core,
            gate,
            output,
        })
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math/cached_gdn.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7735; SHA256 `cfb8884d0e28e48639400ef1c5108f6927deeda4d2d5f216f547cebe09583832`.

````text
//! Admitted private transport/core; no provider exp, state owner or publication.
use super::*;
use crate::accelerator::program::{CachedGdnContract, CachedGdnOperands, CachedGdnValues};
use crate::operator_slice::cached_gdn::CachedGdnSliceInput;

pub(crate) const CACHED_INPUT_SCRATCH: usize = 64 * 1024;
pub(crate) fn cached_core_scratch(heads: usize) -> Result<usize> {
    heads
        .checked_mul(128 * 128 * 4 * 6 + 128 * 4 * 16)
        .and_then(|n| n.checked_add(256 * 1024))
        .ok_or_else(|| Error::Limit("cached GDN conservative scratch bound".into()))
}
impl HostMath<'_, '_, '_> {
    fn cached_windows(&self) -> Result<&crate::original_binding::OriginalWindowOperators> {
        if self.region_kernel.is_some()
            || self.engine.is_some()
            || self.state.is_some()
            || self.source.program_numerical_profile()
                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported(
                "cached GDN private pinned CPU window contract".into(),
            ));
        }
        match &self.source {
            HostSource::Windows(source) => Ok(source),
            HostSource::Loaded(_) => Err(Error::Unsupported(
                "cached GDN requires admitted window backing".into(),
            )),
        }
    }
    fn check_cached_cancel(&self) -> Result<()> {
        if self
            .cancelled
            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
        {
            return Err(Error::State("cached GDN private fragment cancelled".into()));
        }
        Ok(())
    }
    pub(crate) fn import_cached_gdn_input(
        &self,
        input: CachedGdnSliceInput<'_>,
        heads: usize,
    ) -> Result<CachedGdnOperands<HostValue>> {
        let source = self.cached_windows()?;
        input.validate()?;
        if input.source_identity() != source.source_identity()
            || input.heads() != heads
            || input.numerical_profile() != self.source.program_numerical_profile()
        {
            return Err(Error::State(
                "cached GDN input source/profile/head count".into(),
            ));
        }
        self.check_cached_cancel()?;
        let words = heads
            .checked_mul(128 * 128 + 3 * 128 + 2)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Limit("cached GDN aggregate input bytes".into()))?;
        // ALL six retained payloads admitted before the first private Vec.
        let workspace = Arc::new(
            source
                .source()
                .reserve_operator_buffers(words, CACHED_INPUT_SCRATCH)?,
        );
        let mut imported = Vec::with_capacity(6);
        for (index, raw) in input.raw_operands().iter().enumerate() {
            let values = if index < 4 {
                raw.chunks_exact(2)
                    .map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16))
                    .collect()
            } else {
                raw.chunks_exact(4)
                    .map(|v| f32::from_bits(u32::from_le_bytes(v.try_into().unwrap())))
                    .collect()
            };
            let precision = if index < 4 {
                ActivationPrecision::Bf16WidenedF32
            } else {
                ActivationPrecision::F32
            };
            imported.push(
                HostValue::typed(
                    values,
                    precision,
                    source.source_identity(),
                    self.source.program_numerical_profile(),
                )?
                .retain_workspace(Some(workspace.clone()))?,
            );
        }
        let [query, key, value, beta, decay, previous_state]: [HostValue; 6] = imported
            .try_into()
            .map_err(|_| Error::State("cached GDN imported operand arity".into()))?;
        self.check_cached_cancel()?;
        Ok(CachedGdnOperands {
            query,
            key,
            value,
            beta,
            decay,
            previous_state,
        })
    }
    pub(super) fn execute_cached_gdn(
        &self,
        identity: crate::accelerator::SourceIdentity,
        heads: usize,
        input: &CachedGdnOperands<HostValue>,
        contract: CachedGdnContract,
    ) -> Result<CachedGdnValues<HostValue>> {
        let source = self.cached_windows()?;
        let operands = [
            &input.query,
            &input.key,
            &input.value,
            &input.beta,
            &input.decay,
            &input.previous_state,
        ];
        let lengths = operands.map(|v| v.elements);
        let (vector, state) = contract.validate(heads, &lengths)?;
        if identity != source.source_identity()
            || self.source.metadata().geometry().recurrent_value_heads != Some(heads as u64)
        {
            return Err(Error::State(
                "cached GDN source/configured head count".into(),
            ));
        }
        for (index, value) in operands.iter().enumerate() {
            let precision = if index < 4 {
                ActivationPrecision::Bf16WidenedF32
            } else {
                ActivationPrecision::F32
            };
            if value.source != Some(identity)
                || value.profile != Some(self.source.program_numerical_profile())
                || value.precision != precision
            {
                return Err(Error::State(
                    "cached GDN operand source/profile/precision".into(),
                ));
            }
            let workspace = value.data.workspace.as_ref().ok_or_else(|| {
                Error::State("cached GDN operand missing physical backing".into())
            })?;
            if !source.source().same_issuer(workspace.source()) {
                return Err(Error::State("cached GDN operand foreign issuer".into()));
            }
            precision.validate_values(value.as_slice()?)?;
        }
        self.check_cached_cancel()?;
        let bytes = state
            .checked_add(vector * 2)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Limit("cached GDN aggregate output bytes".into()))?;
        // Covers every temporary materialized state/product and small reducer
        // arrays of the existing helper. No native opaque allocator is called.
        let workspace = Arc::new(
            source
                .source()
                .reserve_operator_buffers(bytes, cached_core_scratch(heads)?)?,
        );
        let completed = zerogravity_cpu::original_hf::recurrent::cached_step_128(
            zerogravity_cpu::original_hf::recurrent::CachedGdn128Inputs {
                source: identity,
                heads,
                query: input.query.as_slice()?,
                key: input.key.as_slice()?,
                value: input.value.as_slice()?,
                beta: input.beta.as_slice()?,
                decay: input.decay.as_slice()?,
                previous_state: input.previous_state.as_slice()?,
            },
        )?;
        let make = |values, precision| -> Result<HostValue> {
            HostValue::typed(
                values,
                precision,
                identity,
                self.source.program_numerical_profile(),
            )?
            .retain_workspace(Some(workspace.clone()))
        };
        let result = CachedGdnValues {
            core_f32: make(completed.core_f32, ActivationPrecision::F32)?,
            output_bf16: make(completed.output_bf16, ActivationPrecision::Bf16WidenedF32)?,
            next_state: make(completed.state, ActivationPrecision::F32)?,
        };
        self.check_cached_cancel()?;
        Ok(result)
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6351; SHA256 `c4cbb9c57cb1b39db15aae4533fa3baf3724974f41b74a9b842abb572aca8d60`.

````text
//! One existing HostMath adapter; source/leases retained, no new engine.
use super::*;
use crate::accelerator::program::{
    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues,
};
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
use crate::operator_slice::coefficient_preparation::GdnCoefficientPreparationInput;
use crate::operator_slice::coefficient_preparation::coefficient_roles;

impl HostMath<'_, '_, '_> {
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    pub(crate) fn import_coefficient_preparation_input(
        &self,
        input: GdnCoefficientPreparationInput<'_>,
        heads: usize,
    ) -> Result<(HostValue, HostValue)> {
        if input.validate()? != heads
            || input.source_identity() != self.source.source_identity()
            || input.numerical_profile() != self.source.program_numerical_profile()
        {
            return Err(Error::State(
                "coefficient preparation imported source/profile/heads".into(),
            ));
        }
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "coefficient preparation unaware input provider".into(),
            ));
        }
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let workspace = self.prepare_output(None, 2 * heads, 1)?; // BEFORE both private copies.
        let decode = |raw: &[u8]| {
            raw.chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect::<Vec<_>>()
        };
        let mut values = Vec::with_capacity(2);
        for raw in [input.alpha_bytes(), input.beta_bytes()] {
            let value = HostValue::typed(
                decode(raw),
                ActivationPrecision::Bf16WidenedF32,
                input.source_identity(),
                input.numerical_profile(),
            )?;
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            let value = value.retain_workspace(workspace.clone())?;
            values.push(value);
        }
        Ok((values.remove(0), values.remove(0)))
    }
    pub(super) fn coefficient_preparation_closed(
        &self,
        dt: SourceParameter,
        alpha: &HostValue,
        beta: &HostValue,
        contract: GdnCoefficientPreparationContract,
    ) -> Result<GdnCoefficientPreparationValues<HostValue>> {
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "physical provider lacks original coefficient preparation contract".into(),
            ));
        }
        let source = self.source.source_identity();
        let profile = self.source.program_numerical_profile();
        if profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || [alpha, beta].iter().any(|v| {
                v.precision != ActivationPrecision::Bf16WidenedF32
                    || v.source != Some(source)
                    || v.profile != Some(profile)
            })
        {
            return Err(Error::Unsupported(
                "host coefficient preparation precision/source/profile".into(),
            ));
        }
        let heads = contract.validate(&dt, alpha.elements, beta.elements)?;
        let mut matching_role = false;
        for (layer, kind) in self.source.metadata().geometry().layers.iter().enumerate() {
            if *kind != crate::model::LayerKind::LinearAttention {
                continue;
            }
            let roles = coefficient_roles(self.source.metadata(), layer)?;
            if roles.dt == dt && roles.heads == heads {
                matching_role = true;
                break;
            }
        }
        if !matching_role {
            return Err(Error::State(
                "host coefficient preparation original dt_bias role".into(),
            ));
        }
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let workspace = {
            if let HostSource::Windows(owner) = &self.source {
                let beta_workspace = beta.data.workspace.as_ref().ok_or_else(|| {
                    Error::State("coefficient beta has no original physical backing".into())
                })?;
                if !owner.source().same_issuer(beta_workspace.source()) {
                    return Err(Error::State(
                        "coefficient beta source issuer changed".into(),
                    ));
                }
            }
            self.prepare_output(Some(alpha), 3 * heads, 1)? // Shared aggregate for ALL3 outputs.
        };
        let prepared = match &self.source {
            HostSource::Loaded(owner) => {
                dt.validate_source(*owner)?;
                let raw = zerogravity_core::source::TensorSource::tensor(*owner, &dt.name)?;
                zerogravity_cpu::original_hf::coefficient_preparation::prepare_gdn_coefficients(
                    source,
                    raw,
                    alpha.as_slice()?,
                    beta.as_slice()?,
                )?
            }
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            HostSource::Windows(owner) => owner.prepare_coefficient_stages(
                &dt,
                alpha.as_slice()?,
                beta.as_slice()?,
                self.cancelled,
            )?,
        };
        if prepared.source_identity() != source {
            return Err(Error::State("coefficient preparation result source".into()));
        }
        let (biased, softplus, beta) = prepared.into_parts();
        let mut outputs = Vec::with_capacity(3);
        for (values, precision) in [
            (biased, ActivationPrecision::F32),
            (softplus, ActivationPrecision::F32),
            (beta, ActivationPrecision::Bf16WidenedF32),
        ] {
            let value = HostValue::typed(values, precision, source, profile)?;
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            let value = value.retain_workspace(workspace.clone())?;
            outputs.push(value);
        }
        Ok(GdnCoefficientPreparationValues {
            biased_alpha: outputs.remove(0),
            softplus: outputs.remove(0),
            beta: outputs.remove(0),
        })
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math/gated_rms.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5649; SHA256 `0420e4e1c94fe94875f2dce2f10d0d7f4b6ea6627ca1ded9a53e492567509df7`.

````text
//! Same-source aggregate leases; no provider fallback or state owner.
use super::*;
use crate::accelerator::program::GatedRmsGammaBf16Contract;
use crate::operator_slice::gated_rms::GatedRmsSliceInput;

pub(crate) const INPUT_BYTES: usize = 2 * 6144 * 4;
pub(crate) const INPUT_SCRATCH: usize = 64 * 1024;
pub(crate) const OUTPUT_BYTES: usize = 6144 * 4;
// Seven helper arrays +squares/sums/gamma and final copy fit within this old floor.
pub(crate) const OUTPUT_SCRATCH: usize = 960 * 1024;
impl HostMath<'_, '_, '_> {
    fn gated_rms_windows(&self) -> Result<&crate::original_binding::OriginalWindowOperators> {
        if self.region_kernel.is_some()
            || self.engine.is_some()
            || self.state.is_some()
            || self.source.program_numerical_profile()
                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported(
                "gated RMS private pinned CPU window contract".into(),
            ));
        }
        match &self.source {
            HostSource::Windows(source) => Ok(source),
            HostSource::Loaded(_) => Err(Error::Unsupported(
                "gated RMS requires admitted original windows".into(),
            )),
        }
    }
    fn gated_rms_cancel(&self) -> Result<()> {
        if self
            .cancelled
            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
        {
            return Err(Error::State("gated RMS private fragment cancelled".into()));
        }
        Ok(())
    }
    pub(crate) fn import_gated_rms_input(
        &self,
        input: GatedRmsSliceInput<'_>,
    ) -> Result<(HostValue, HostValue)> {
        let source = self.gated_rms_windows()?;
        input.validate()?;
        if input.source_identity() != source.source_identity()
            || input.numerical_profile() != self.source.program_numerical_profile()
        {
            return Err(Error::State("gated RMS input source/profile".into()));
        }
        self.gated_rms_cancel()?;
        let workspace = Arc::new(
            source
                .source()
                .reserve_operator_buffers(INPUT_BYTES, INPUT_SCRATCH)?,
        );
        let make = |raw: &[u8]| -> Result<HostValue> {
            let mut values = Vec::with_capacity(6144);
            if values.capacity() != 6144 {
                return Err(Error::Limit("gated RMS input capacity".into()));
            }
            values.extend(
                raw.chunks_exact(2)
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)),
            );
            HostValue::typed(
                values,
                ActivationPrecision::Bf16WidenedF32,
                source.source_identity(),
                self.source.program_numerical_profile(),
            )?
            .retain_workspace(Some(workspace.clone()))
        };
        let [core, gate] = input.raw_inputs();
        let result = (make(core)?, make(gate)?);
        self.gated_rms_cancel()?;
        Ok(result)
    }
    pub(super) fn execute_gated_rms_bf16_gamma(
        &self,
        parameter: SourceParameter,
        core: &HostValue,
        gate: &HostValue,
        contract: GatedRmsGammaBf16Contract,
    ) -> Result<HostValue> {
        let source = self.gated_rms_windows()?;
        contract.validate(&parameter, core.elements, gate.elements)?;
        if parameter.source != source.source_identity() {
            return Err(Error::State("gated RMS parameter source".into()));
        }
        for input in [core, gate] {
            if input.source != Some(source.source_identity())
                || input.profile != Some(self.source.program_numerical_profile())
                || input.precision != ActivationPrecision::Bf16WidenedF32
            {
                return Err(Error::State(
                    "gated RMS activation source/profile/precision".into(),
                ));
            }
            let owner = input
                .data
                .workspace
                .as_ref()
                .ok_or_else(|| Error::State("gated RMS missing physical backing".into()))?;
            if !source.source().same_issuer(owner.source()) {
                return Err(Error::State("gated RMS foreign physical issuer".into()));
            }
        }
        if crate::operator_slice::gated_rms::gated_rms_role(source, 0)? != parameter {
            return Err(Error::State(
                "gated RMS exact original parameter role".into(),
            ));
        }
        let core = core.as_slice()?;
        let gate = gate.as_slice()?;
        ActivationPrecision::Bf16WidenedF32.validate_values(core)?;
        ActivationPrecision::Bf16WidenedF32.validate_values(gate)?;
        self.gated_rms_cancel()?;
        let workspace = Arc::new(
            source
                .source()
                .reserve_operator_buffers(OUTPUT_BYTES, OUTPUT_SCRATCH)?,
        );
        let completed = source.gated_rms_bf16_gamma(&parameter, core, gate, self.cancelled)?;
        let mut values = Vec::with_capacity(6144);
        if values.capacity() != 6144 || completed.output_bf16().len() != 6144 {
            return Err(Error::Limit("gated RMS output capacity/length".into()));
        }
        values.extend_from_slice(completed.output_bf16());
        let output = HostValue::typed(
            values,
            ActivationPrecision::Bf16WidenedF32,
            source.source_identity(),
            self.source.program_numerical_profile(),
        )?
        .retain_workspace(Some(workspace))?;
        self.gated_rms_cancel()?;
        Ok(output)
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 10372; SHA256 `27d7f314a00c599e6d7a90a4e0c6519a820c7c46ec31f3ae44788b89735363ce`.

````text
//! Source-bound coefficient PREPARATION on caller-owned a/b; no g/decay/exp.
use super::OperatorSliceSource;
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues, MatrixAxes, PlanBuilder,
    PlanValue, ProgramNumericalProfile, SourceParameter,
};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, LoadedQwenModel, Result};
use zerogravity_core::quant::QuantKind;

pub(crate) struct CoefficientPreparationRoles {
    pub(crate) heads: usize,
    pub(crate) dt: SourceParameter,
    pub(crate) a_log: SourceParameter,
}
/// Immutable complete role descriptors; A_log is not evaluated by this slice.
pub(crate) fn coefficient_roles(
    source: &(impl OperatorSliceSource + ?Sized),
    layer: usize,
) -> Result<CoefficientPreparationRoles> {
    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || !source.source_identity().is_bound()
        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
    {
        return Err(Error::Unsupported(
            "coefficient preparation original LinearAttention source/profile".into(),
        ));
    }
    let heads = usize::try_from(source.geometry().recurrent_value_heads.unwrap_or(0))
        .map_err(|_| Error::Limit("coefficient heads".into()))?;
    let hidden = usize::try_from(source.geometry().hidden_size)
        .map_err(|_| Error::Limit("coefficient hidden".into()))?;
    if !matches!(heads, 16 | 48) || !matches!(hidden, 256 | 1024 | 2048 | 5120) {
        return Err(Error::Unsupported(
            "coefficient preparation closed source geometry".into(),
        ));
    }
    let namespace = format!("model.language_model.layers.{layer}.linear_attn");
    for (role, suffix) in [
        (QwenMatrixRole::GdnAlpha(layer), "in_proj_a"),
        (QwenMatrixRole::GdnBeta(layer), "in_proj_b"),
    ] {
        let tensor = source.matrix_descriptor(role)?;
        if tensor.source != source.source_identity()
            || tensor.name != format!("{namespace}.{suffix}.weight")
            || tensor.kind != QuantKind::Bf16
            || tensor.axes != MatrixAxes::OutputInput
            || tensor.rows != heads
            || tensor.columns != hidden
        {
            return Err(Error::State(
                "coefficient preparation original a/b full matrix role".into(),
            ));
        }
    }
    let dt = source.parameter_descriptor(QwenParameterRole::GdnDt(layer))?;
    let a_log = source.parameter_descriptor(QwenParameterRole::GdnA(layer))?;
    if dt.source != source.source_identity()
        || dt.name != format!("{namespace}.dt_bias")
        || a_log.source != source.source_identity()
        || a_log.name != format!("{namespace}.A_log")
        || a_log.kind != QuantKind::F32
        || a_log.shape != [heads as u64]
        || a_log.bytes != heads * 4
    {
        return Err(Error::State(
            "coefficient preparation original raw A_log/dt_bias roles".into(),
        ));
    }
    GdnCoefficientPreparationContract::TorchCpuBf16Avx2.validate(&dt, heads, heads)?;
    Ok(CoefficientPreparationRoles { heads, dt, a_log })
}

impl LoadedQwenModel {
    /// Append only beta/biased-alpha/softplus to caller's original plan values.
    pub fn append_gdn_coefficient_preparation_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        a: &PlanValue,
        b: &PlanValue,
    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
        self.require_operator_slice()?;
        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
    }
}

/// External actual source/operator a/b bytes; caller authenticates provenance.
/// No hidden state, matrix producer, token or frontier is asserted by import.
#[derive(Clone, Copy)]
pub struct GdnCoefficientPreparationInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    alpha: &'a [u8],
    beta: &'a [u8],
}
impl<'a> GdnCoefficientPreparationInput<'a> {
    pub fn from_bf16_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        alpha: &'a [u8],
        beta: &'a [u8],
    ) -> Result<Self> {
        let input = Self {
            source,
            profile,
            alpha,
            beta,
        };
        input.validate()?;
        Ok(input)
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn alpha_bytes(self) -> &'a [u8] {
        self.alpha
    }
    pub fn beta_bytes(self) -> &'a [u8] {
        self.beta
    }
    pub(crate) fn validate(self) -> Result<usize> {
        if !self.source.is_bound()
            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::State(
                "coefficient preparation private input source/profile".into(),
            ));
        }
        if !matches!(self.alpha.len(), 32 | 96)
            || self.beta.len() != self.alpha.len()
            || self
                .alpha
                .chunks_exact(2)
                .chain(self.beta.chunks_exact(2))
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(Error::Malformed(
                "coefficient preparation one finite BF16 a/b call".into(),
            ));
        }
        Ok(self.alpha.len() / 2)
    }
}

#[cfg(feature = "cpu")]
pub struct GdnCoefficientPreparationOutput {
    source: SourceIdentity,
    layer: usize,
    dt: SourceParameter,
    a_log: SourceParameter,
    alpha: crate::host_math::HostValue,
    beta_input: crate::host_math::HostValue,
    values: GdnCoefficientPreparationValues<crate::host_math::HostValue>,
}
#[cfg(feature = "cpu")]
impl GdnCoefficientPreparationOutput {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn raw_dt_bias_parameter(&self) -> &SourceParameter {
        &self.dt
    }
    pub fn unevaluated_raw_a_log_parameter(&self) -> &SourceParameter {
        &self.a_log
    }
    pub fn alpha_bf16(&self) -> Result<&[f32]> {
        self.alpha.as_slice()
    }
    pub fn beta_input_bf16(&self) -> Result<&[f32]> {
        self.beta_input.as_slice()
    }
    pub fn biased_alpha_f32(&self) -> Result<&[f32]> {
        self.values.biased_alpha.as_slice()
    }
    pub fn softplus_f32(&self) -> Result<&[f32]> {
        self.values.softplus.as_slice()
    }
    pub fn beta_bf16(&self) -> Result<&[f32]> {
        self.values.beta.as_slice()
    }
}

#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl crate::original_binding::OriginalWindowOperators {
    pub fn append_gdn_coefficient_preparation_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        a: &PlanValue,
        b: &PlanValue,
    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_gdn_coefficient_preparation_slice(
        &self,
        layer: usize,
        input: GdnCoefficientPreparationInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<GdnCoefficientPreparationOutput> {
        let roles = coefficient_roles(self, layer)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let (alpha, beta_input) = math.import_coefficient_preparation_input(input, roles.heads)?;
        let values = crate::forward::gdn_coefficient_preparation_slice(
            self,
            &mut math,
            layer,
            &alpha,
            &beta_input,
        )?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient preparation cancelled".into(),
            ));
        }
        Ok(GdnCoefficientPreparationOutput {
            source: self.source_identity(),
            layer,
            dt: roles.dt,
            a_log: roles.a_log,
            alpha,
            beta_input,
            values,
        })
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn prepare_coefficient_stages(
        &self,
        dt: &SourceParameter,
        alpha: &[f32],
        beta: &[f32],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<zerogravity_cpu::original_hf::coefficient_preparation::GdnCoefficientPreparation>
    {
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient raw parameter cancelled".into(),
            ));
        }
        let raw = self.source().parameter(&dt.name)?;
        raw.validate_source_parameter(dt)?;
        let window = match cancelled {
            Some(flag) => {
                self.source()
                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
            }
            None => self.source().read_parameter(&raw, 0..raw.elements())?,
        };
        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
            return Err(Error::State(
                "window coefficient complete raw dt_bias".into(),
            ));
        }
        let result =
            zerogravity_cpu::original_hf::coefficient_preparation::prepare_gdn_coefficients(
                self.source_identity(),
                zerogravity_core::source::SourceTensorRef {
                    kind: raw.tensor().kind,
                    shape: &raw.tensor().shape,
                    packed_bytes: window.as_slice(),
                },
                alpha,
                beta,
            )?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient cancelled after synchronous math".into(),
            ));
        }
        Ok(result)
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice/pure.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6526; SHA256 `9a2f8d7130f77e3244fe6aedc2b04cb09cfd9474800267e7dd7af22295ad5e4d`.

````text
//! One pure projection or ordinary norm on explicit private BF16 input.
//! No producer, session, activation nonlinearities or model execution is asserted.
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
use crate::accelerator::program::{PlanBuilder, PlanValue};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, Result};

/// External raw transport only. Caller independently binds archive/source and
/// producer provenance; this type validates representation and source/profile.
#[derive(Clone, Copy)]
pub struct PureSliceInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    bytes: &'a [u8],
}
impl<'a> PureSliceInput<'a> {
    pub fn from_bf16_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        bytes: &'a [u8],
    ) -> Result<Self> {
        let input = Self {
            source,
            profile,
            bytes,
        };
        input.validate()?;
        Ok(input)
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn precision(self) -> ActivationPrecision {
        ActivationPrecision::Bf16WidenedF32
    }
    pub fn bytes(self) -> &'a [u8] {
        self.bytes
    }
    pub(crate) fn validate(self) -> Result<()> {
        if !self.source.is_bound()
            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::State("pure input source/profile".into()));
        }
        if self.bytes.is_empty()
            || self.bytes.len() > 2 * 17408
            || !self.bytes.len().is_multiple_of(2)
            || self
                .bytes
                .chunks_exact(2)
                .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(Error::Malformed(
                "pure input bounded finite BF16 bytes".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PureSliceRole {
    Projection(QwenMatrixRole),
    OrdinaryRms(QwenParameterRole),
}

#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl crate::original_binding::OriginalWindowOperators {
    /// Caller retains its existing same-builder producer and source/metadata.
    /// No external-input node, root or complete model plan is manufactured.
    pub fn append_projection_slice(
        &self,
        builder: &mut PlanBuilder,
        role: QwenMatrixRole,
        input: &PlanValue,
    ) -> Result<PlanValue> {
        self.projection_role(role)?;
        crate::forward::pure_projection_slice(self, builder, role, input)
    }
    pub fn append_ordinary_rms_slice(
        &self,
        builder: &mut PlanBuilder,
        role: QwenParameterRole,
        input: &PlanValue,
    ) -> Result<PlanValue> {
        self.ordinary_hidden_role(role)?;
        crate::forward::pure_ordinary_rms_slice(self, builder, role, input)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_projection_slice(
        &self,
        role: QwenMatrixRole,
        input: PureSliceInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<PureSliceOutput> {
        let matrix = self.projection_role(role)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let input = math.import_pure_input(input, matrix.columns)?;
        let output = crate::forward::pure_projection_slice(self, &mut math, role, &input)?;
        finish_pure_output(
            PureSliceOutput {
                source: self.source().source_identity(),
                role: PureSliceRole::Projection(role),
                input,
                output,
            },
            cancelled,
        )
    }
    #[cfg(feature = "cpu")]
    pub fn execute_ordinary_rms_slice(
        &self,
        role: QwenParameterRole,
        input: PureSliceInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<PureSliceOutput> {
        use crate::operator_slice::OperatorSliceSource;
        self.ordinary_hidden_role(role)?;
        let hidden = usize::try_from(self.geometry().hidden_size)
            .map_err(|_| Error::Limit("ordinary pure input width".into()))?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let input = math.import_pure_input(input, hidden)?;
        let output = crate::forward::pure_ordinary_rms_slice(self, &mut math, role, &input)?;
        finish_pure_output(
            PureSliceOutput {
                source: self.source().source_identity(),
                role: PureSliceRole::OrdinaryRms(role),
                input,
                output,
            },
            cancelled,
        )
    }
}

/// Retained original-source input/output backing, borrowed access only.
#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
pub struct PureSliceOutput {
    source: SourceIdentity,
    role: PureSliceRole,
    input: crate::host_math::HostValue,
    output: crate::host_math::HostValue,
}
#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
impl PureSliceOutput {
    pub fn role(&self) -> PureSliceRole {
        self.role
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn precision(&self) -> ActivationPrecision {
        ActivationPrecision::Bf16WidenedF32
    }
    pub fn input(&self) -> Result<&[f32]> {
        self.input.as_slice()
    }
    pub fn output(&self) -> Result<&[f32]> {
        self.output.as_slice()
    }
}

/// Boundary after all synchronous math/RNE/validation, before private return.
#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
pub(crate) fn finish_pure_output(
    output: PureSliceOutput,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<PureSliceOutput> {
    if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
        return Err(Error::State(
            "pure window output cancelled before return".into(),
        ));
    }
    Ok(output)
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 16480; SHA256 `9769cf2902bf6fda572a4a52010cfbedb5bbc06e9869f3f6306043290389df8f`.

````text
//! Source-free full-header fixtures; no trained27B/operator independence claim.
use super::*;
use crate::accelerator::program::ProgramNumericalProfile;
use crate::operator_slice::pure::PureSliceInput;
use zerogravity_core::resource_arbiter::AllocationClass;

fn input<'a>(source: &OriginalWindowOperators, bytes: &'a [u8]) -> PureSliceInput<'a> {
    PureSliceInput::from_bf16_bytes(
        source.source().source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        bytes,
    )
    .unwrap()
}

#[test]
fn all_bound_projection_roles_use_complete_actual_source_and_retained_results() {
    for tied in [false, true] {
        let fixture = Fixture::new();
        let loaded = source_with_options(&fixture, tied, 256, true, |_| {});
        let (source, admission) = windows(&fixture, &loaded);
        let operators = bind_windows(source.clone(), &loaded);
        for role in [
            QwenMatrixRole::GdnQkv(0),
            QwenMatrixRole::GdnGate(0),
            QwenMatrixRole::GdnAlpha(0),
            QwenMatrixRole::GdnBeta(0),
            QwenMatrixRole::GdnOutput(0),
            QwenMatrixRole::Query(3),
            QwenMatrixRole::Key(3),
            QwenMatrixRole::Value(3),
            QwenMatrixRole::AttentionOutput(3),
            QwenMatrixRole::FfnGate(0),
            QwenMatrixRole::FfnUp(0),
            QwenMatrixRole::FfnDown(0),
            QwenMatrixRole::Output,
        ] {
            let tensor = operators.matrix_descriptor(role).unwrap();
            let bytes = 0x3f80u16.to_le_bytes().repeat(tensor.columns);
            let before = reserved(&admission);
            let result = operators
                .execute_projection_slice(role, input(&operators, &bytes), None)
                .unwrap();
            let sign = if role == QwenMatrixRole::Output && !tied {
                -1.0
            } else {
                1.0
            };
            assert_eq!(result.input().unwrap(), vec![1.0; tensor.columns]);
            assert_eq!(
                result.output().unwrap(),
                vec![sign * tensor.columns as f32; tensor.rows]
            );
            assert_eq!(result.source_identity(), source.source_identity());
            assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
            drop(result);
            assert_eq!(reserved(&admission), before);
        }
        drop(operators);
        drop(source);
        assert_eq!(reserved(&admission), 0);
    }
}

#[test]
fn ordinary_hidden_roles_preserve_raw_gamma_and_complete_source_membership() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    for role in [
        QwenParameterRole::InputNorm(0),
        QwenParameterRole::InputNorm(2),
        QwenParameterRole::FfnNorm(0),
        QwenParameterRole::FinalNorm,
    ] {
        let result = operators
            .execute_ordinary_rms_slice(role, input(&operators, &bytes), None)
            .unwrap();
        let expected = if matches!(role, QwenParameterRole::InputNorm(_)) {
            2.0
        } else {
            1.25
        };
        assert_eq!(result.output().unwrap(), vec![expected; 256]);
    }
    let before = reserved(&admission);
    for role in [
        QwenParameterRole::GdnNorm(0),
        QwenParameterRole::GdnA(0),
        QwenParameterRole::GdnDt(0),
        QwenParameterRole::InputNorm(99),
    ] {
        assert!(
            operators
                .execute_ordinary_rms_slice(role, input(&operators, &bytes), None)
                .is_err()
        );
        assert_eq!(reserved(&admission), before);
    }
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn same_plan_projection_and_norm_nodes_do_not_create_state_or_external_producers() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let mut builder = PlanBuilder::new_operator_slice(
        source.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    let embedding = builder
        .embed_profiled(
            operators
                .matrix_descriptor(QwenMatrixRole::Embedding)
                .unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let normalized = operators
        .append_ordinary_rms_slice(&mut builder, QwenParameterRole::FfnNorm(0), &embedding)
        .unwrap();
    let up = operators
        .append_projection_slice(&mut builder, QwenMatrixRole::FfnUp(0), &normalized)
        .unwrap();
    let down = operators
        .append_projection_slice(&mut builder, QwenMatrixRole::FfnDown(0), &up)
        .unwrap();
    let head = operators
        .append_projection_slice(&mut builder, QwenMatrixRole::Output, &down)
        .unwrap();
    let plan = builder.finish_operator_slice(head).unwrap();
    assert_eq!(plan.plan().nodes().len(), 5);
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(matches!(
        plan.plan().nodes()[4].operation,
        PlanOperation::TypedProject { .. }
    ));
    drop(operators);
    drop(source);
    drop(plan);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn checked_actual_vocabulary_output_and_wide_input_charge_grow_before_allocation() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let before = admission.arbiter.snapshot(&admission.resource).unwrap();
    let head = operators.workspace(248320, 248448).unwrap();
    let current = admission.arbiter.snapshot(&admission.resource).unwrap();
    assert_eq!(
        current.per_class[&AllocationClass::Output]
            - before
                .per_class
                .get(&AllocationClass::Output)
                .copied()
                .unwrap_or(0),
        248320 * 4
    );
    assert_eq!(head.bytes(), (248320 + 248448) * 4);
    drop(head);
    let wide = operators.workspace(17408, 1).unwrap();
    assert_eq!(wide.bytes(), 17408 * 4 + 960 * 1024);
    drop(wide);
    assert!(operators.workspace(usize::MAX, 1).is_err());
    assert!(operators.workspace(1, usize::MAX).is_err());
    assert_eq!(reserved(&admission), before.reserved);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn invalid_input_role_capacity_and_cancel_refuse_without_leaking_or_fencing() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let before = reserved(&admission);
    for role in [
        QwenMatrixRole::Embedding,
        QwenMatrixRole::GdnGate(3),
        QwenMatrixRole::Query(0),
    ] {
        assert!(
            operators
                .execute_projection_slice(role, input(&operators, &bytes), None)
                .is_err()
        );
    }
    assert!(
        operators
            .execute_projection_slice(
                QwenMatrixRole::FfnUp(0),
                input(&operators, &bytes[..2]),
                None
            )
            .is_err()
    );
    assert!(
        operators
            .execute_projection_slice(
                QwenMatrixRole::FfnUp(0),
                input(&operators, &bytes),
                Some(&AtomicBool::new(true))
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    for bad in [
        vec![0],
        0x7fc1u16.to_le_bytes().to_vec(),
        vec![0; 2 * 17409],
    ] {
        assert!(
            PureSliceInput::from_bf16_bytes(
                source.source_identity(),
                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
                &bad
            )
            .is_err()
        );
    }
    let pressure = source.reserve_workspace(32 * 1024 * 1024).unwrap();
    let pressure2 = source.reserve_workspace(32 * 1024 * 1024).unwrap();
    let remaining = admission
        .arbiter
        .snapshot(&admission.resource)
        .unwrap()
        .remaining;
    let pressure3 = source
        .reserve_workspace((remaining - (2 * 1024 * 1024 - 1)) as usize)
        .unwrap();
    let held = reserved(&admission);
    assert!(
        operators
            .execute_projection_slice(QwenMatrixRole::Output, input(&operators, &bytes), None)
            .is_err()
    );
    assert_eq!(reserved(&admission), held);
    assert!(
        !admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .fenced
    );
    drop(pressure3);
    drop(pressure2);
    drop(pressure);
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn returned_borrowed_results_keep_source_and_original_tickets_after_callers_drop() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let source_metadata = source.metadata_reserved_bytes();
    let result = operators
        .execute_projection_slice(QwenMatrixRole::Output, input(&operators, &bytes), None)
        .unwrap();
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), source_metadata + 2 * 1024 * 1024);
    assert_eq!(result.output().unwrap(), vec![-256.0; 16]);
    assert_eq!(result.input().unwrap(), vec![1.0; 256]);
    drop(result);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn unaware_provider_refuses_new_roles_before_any_callback_or_admission() {
    struct Unaware(AtomicBool);
    impl crate::accelerator::MatVecKernel for Unaware {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            panic!("pure original CPU role cannot call an unaware provider");
        }
    }
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let provider = Unaware(AtomicBool::new(false));
    let plain = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
    let activation = plain
        .import_pure_input(input(&operators, &bytes), 256)
        .unwrap();
    let before = reserved(&admission);
    let mut unaware =
        crate::host_math::HostMath::for_window_operator_slice(&operators, None, Some(&provider));
    assert!(
        unaware
            .project_profiled(
                operators.matrix_descriptor(QwenMatrixRole::Output).unwrap(),
                &activation,
                crate::accelerator::program::ProjectionContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    assert!(!provider.0.load(std::sync::atomic::Ordering::Relaxed));
    assert_eq!(reserved(&admission), before);
    drop(activation);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn separately_opened_same_content_source_cannot_rebind_a_retained_input() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source_a, admission_a) = windows(&fixture, &loaded);
    let (source_b, admission_b) = windows(&fixture, &loaded);
    assert_ne!(source_a.source_identity(), source_b.source_identity());
    assert!(!source_a.same_issuer(&source_b));
    let operators_a = bind_windows(source_a.clone(), &loaded);
    let operators_b = bind_windows(source_b.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let math_a = crate::host_math::HostMath::for_window_operator_slice(&operators_a, None, None);
    let activation = math_a
        .import_pure_input(input(&operators_a, &bytes), 256)
        .unwrap();
    let before_b = reserved(&admission_b);
    let mut math_b =
        crate::host_math::HostMath::for_window_operator_slice(&operators_b, None, None);
    assert!(
        math_b
            .project_profiled(
                operators_b
                    .matrix_descriptor(QwenMatrixRole::Output)
                    .unwrap(),
                &activation,
                crate::accelerator::program::ProjectionContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    assert_eq!(reserved(&admission_b), before_b);
    drop(activation);
    drop(operators_a);
    drop(operators_b);
    drop(source_a);
    drop(source_b);
    assert_eq!(reserved(&admission_a), 0);
    assert_eq!(reserved(&admission_b), 0);
}

#[test]
fn valid_two_hidden_qkv_producer_cannot_record_one_hidden_norm_or_partial_node() {
    let fixture = Fixture::new();
    let mut cfg: Value = serde_json::from_slice(&config(false)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["linear_key_head_dim"] = json!(128);
    cfg["text_config"]["linear_value_head_dim"] = json!(128);
    let loaded = source_with_config(
        &fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |_, _, bits| bits,
        |_| {},
    );
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let matrix = operators
        .matrix_descriptor(QwenMatrixRole::GdnQkv(0))
        .unwrap();
    assert_eq!(matrix.rows, 512);
    assert_eq!(matrix.columns, 256);
    let mut builder = PlanBuilder::new_operator_slice(
        source.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    let embedding = builder
        .embed_profiled(
            operators
                .matrix_descriptor(QwenMatrixRole::Embedding)
                .unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let qkv = operators
        .append_projection_slice(&mut builder, QwenMatrixRole::GdnQkv(0), &embedding)
        .unwrap();
    assert!(matches!(operators.append_ordinary_rms_slice(&mut builder,
        QwenParameterRole::InputNorm(0), &qkv),
        Err(Error::Malformed(message)) if message.contains("exact hidden input")));
    let recorded = builder.finish_operator_slice(qkv).unwrap();
    assert_eq!(recorded.plan().nodes().len(), 2);
    assert!(recorded.plan().proposed_domains().is_empty());
    drop(recorded);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn completed_private_output_cancellation_drops_actual_source_backing_before_return() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let before = reserved(&admission);
    let output = operators
        .execute_projection_slice(QwenMatrixRole::Output, input(&operators, &bytes), None)
        .unwrap();
    assert_eq!(output.output().unwrap(), vec![-256.0; 16]);
    assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
    let flag = AtomicBool::new(true);
    assert!(
        matches!(crate::operator_slice::pure::finish_pure_output(output, Some(&flag)),
        Err(Error::State(message)) if message.contains("cancelled before return"))
    );
    assert_eq!(reserved(&admission), before);
    assert!(
        !admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .fenced
    );
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding/window/gated_rms.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1665; SHA256 `0d9f1b1e599cf7104e73817fa89067a2837b564a7af71bcd304acec94d2a6bf1`.

````text
//! Real raw gamma window on the existing immutable owner, never a learned cache.
use super::*;
impl OriginalWindowOperators {
    pub(crate) fn gated_rms_bf16_gamma(
        &self,
        parameter: &SourceParameter,
        core: &[f32],
        gate: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<zerogravity_cpu::original_hf::norm::Bf16GammaGatedRms128> {
        let actual = crate::operator_slice::gated_rms::gated_rms_role(self, 0)?;
        if actual != *parameter || core.len() != 6144 || gate.len() != 6144 {
            return Err(Error::State("gated RMS original raw role/geometry".into()));
        }
        check_cancel(cancelled)?;
        let raw = self.source().parameter(&parameter.name)?;
        raw.validate_source_parameter(parameter)?;
        let window = match cancelled {
            Some(flag) => self
                .source()
                .read_parameter_cancellable(&raw, 0..128, flag)?,
            None => self.source().read_parameter(&raw, 0..128)?,
        };
        if !window.is_complete() || window.element_range() != (0..128) {
            return Err(Error::State(
                "gated RMS incomplete original gamma extent".into(),
            ));
        }
        let completed = zerogravity_cpu::original_hf::norm::gated_rms_128_bf16_gamma_h48(
            self.source_identity(),
            zerogravity_core::source::SourceTensorRef {
                kind: raw.tensor().kind,
                shape: &raw.tensor().shape,
                packed_bytes: window.as_slice(),
            },
            core,
            gate,
        )?;
        check_cancel(cancelled)?;
        Ok(completed)
    }
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding/window/pure.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4613; SHA256 `8ac493e822e7c7e804526ea75231d8d0c361e5bed490a06c3255a0c9f4290077`.

````text
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
            if *kind != crate::model::LayerKind::FullAttention {
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

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-default-bulk-20261020 :: tracked source diff against fb68d8fa165de16e21d1bd37afef3790b80da971

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 15396; SHA256 `217e35d81018d911f0f5ec287efd2f20ef2ebf88fecf0aa06f72006628af3c20`.

````text
diff --git a/crates/zerogravity-core/src/accelerator/program.rs b/crates/zerogravity-core/src/accelerator/program.rs
index 416dfd1..5b82c55 100644
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -20,6 +20,12 @@ pub use typed::{
     ProjectionContract, SourceParameter,
 };
 
+mod bulk_f32;
+pub use bulk_f32::{
+    BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Origin, BulkF32Side,
+    TeacherSpan,
+};
+
 static NEXT_PLAN_BUILDER: AtomicU64 = AtomicU64::new(1);
 
 #[derive(Clone, Copy, Debug, PartialEq, Eq)]
@@ -294,6 +300,21 @@ pub trait MathematicalBackend {
             "backend does not implement explicit causal convolution SiLU contract".into(),
         ))
     }
+    fn bulk_f32_input(&mut self, _operand: &BulkF32Operand) -> Result<Self::Value> {
+        Err(Error::Unsupported(
+            "backend has no closed bulk F32 input contract".into(),
+        ))
+    }
+    fn bulk_f32_matmul(
+        &mut self,
+        _matrix: &BulkF32Matrix,
+        _left: &Self::Value,
+        _right: &Self::Value,
+    ) -> Result<Self::Value> {
+        Err(Error::Unsupported(
+            "backend has no closed Haswell bulk F32 matrix contract".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -458,6 +479,12 @@ pub struct StateEffect {
 /// Lowering may add explicit Pack/Copy instructions for qualified kernel ABIs.
 #[derive(Clone, Debug)]
 pub enum PlanOperation {
+    TypedBulkF32Input {
+        operand: BulkF32Operand,
+    },
+    TypedBulkF32Matmul {
+        matrix: BulkF32Matrix,
+    },
     TypedEmbed {
         tensor: SourceTensor,
         token: ParameterRef,
@@ -774,6 +801,12 @@ impl PlanBuilder {
         inputs: &[PlanValue],
     ) -> Result<ActivationPrecision> {
         match operation {
+            PlanOperation::TypedBulkF32Input { operand } => {
+                bulk_f32::input_precision(self, operand)
+            }
+            PlanOperation::TypedBulkF32Matmul { matrix } => {
+                bulk_f32::matrix_precision(self, matrix, inputs)
+            }
             PlanOperation::TypedEmbed {
                 tensor,
                 contract: EmbeddingContract::Bf16OutputInput,
@@ -835,7 +868,9 @@ impl PlanBuilder {
                     .all(|input| input.precision == ActivationPrecision::F32)
                 && !matches!(
                     operation,
-                    PlanOperation::TypedEmbed { .. }
+                    PlanOperation::TypedBulkF32Input { .. }
+                        | PlanOperation::TypedBulkF32Matmul { .. }
+                        | PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
@@ -856,6 +891,22 @@ impl PlanBuilder {
         shapes: &[ValueShape],
         state: Option<StateEffect>,
     ) -> Result<Vec<PlanValue>> {
+        if !matches!(operation, PlanOperation::TypedBulkF32Matmul { .. })
+            && inputs.iter().any(|input| {
+                self.nodes.iter().any(|node| {
+                    node.outputs.contains(&input.id)
+                        && matches!(
+                            node.operation,
+                            PlanOperation::TypedBulkF32Input { .. }
+                                | PlanOperation::TypedBulkF32Matmul { .. }
+                        )
+                })
+            })
+        {
+            return Err(Error::Unsupported(
+                "bulk typed value cannot enter a legacy operation".into(),
+            ));
+        }
         for input in inputs {
             if input.owner != self.owner
                 || self.values.get(input.id.0 as usize) != Some(&input.shape)
@@ -1033,6 +1084,17 @@ impl PlanBuilder {
 impl MathematicalBackend for PlanBuilder {
     type Value = PlanValue;
     type Index = ParameterRef;
+    fn bulk_f32_input(&mut self, operand: &BulkF32Operand) -> Result<PlanValue> {
+        bulk_f32::record_input(self, operand)
+    }
+    fn bulk_f32_matmul(
+        &mut self,
+        matrix: &BulkF32Matrix,
+        left: &PlanValue,
+        right: &PlanValue,
+    ) -> Result<PlanValue> {
+        bulk_f32::record_matrix(self, matrix, left, right)
+    }
     fn elements(&self, value: &PlanValue) -> usize {
         value.elements
     }
diff --git a/crates/zerogravity-cpu/src/math.rs b/crates/zerogravity-cpu/src/math.rs
index 9c1b08b..97f6e3f 100644
--- a/crates/zerogravity-cpu/src/math.rs
+++ b/crates/zerogravity-cpu/src/math.rs
@@ -368,3 +368,6 @@ mod ggml_source_silu_tests {
         assert_eq!(ggml_source_silu(0.0), 0.0);
     }
 }
+
+/// Explicit pinned GGML bulk-F32 diagnostic; default source math is unchanged.
+pub mod haswell_bulk_f32;
diff --git a/crates/zerogravity-gpu/src/resident_program.rs b/crates/zerogravity-gpu/src/resident_program.rs
index eb772ba..9ff8759 100644
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -345,7 +345,9 @@ impl ResidentProgram {
             || plan.nodes().iter().any(|node| {
                 matches!(
                     node.operation,
-                    PlanOperation::TypedEmbed { .. }
+                    PlanOperation::TypedBulkF32Input { .. }
+                        | PlanOperation::TypedBulkF32Matmul { .. }
+                        | PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
@@ -1428,7 +1430,9 @@ impl ResidentProgram {
         // bound every offset/count. Parent retains the complete original cohort.
         unsafe {
             match &node.operation {
-                PlanOperation::TypedEmbed { .. }
+                PlanOperation::TypedBulkF32Input { .. }
+                | PlanOperation::TypedBulkF32Matmul { .. }
+                | PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
                 | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
diff --git a/crates/zerogravity-gpu/src/resident_program/batch.rs b/crates/zerogravity-gpu/src/resident_program/batch.rs
index 628b1d6..888f151 100644
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -928,7 +928,9 @@ impl BatchProgram {
         let scratch = |i: usize| &self.inner.scratch[i];
         unsafe {
             match &node.operation {
-                PlanOperation::TypedEmbed { .. }
+                PlanOperation::TypedBulkF32Input { .. }
+                | PlanOperation::TypedBulkF32Matmul { .. }
+                | PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
                 | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
diff --git a/crates/zerogravity-model-qwen35/src/forward.rs b/crates/zerogravity-model-qwen35/src/forward.rs
index 8c81064..a8f1c17 100644
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -650,3 +650,9 @@ pub(crate) fn gdn_convolution_silu_slice<B: MathematicalBackend>(
     }
     Ok(values)
 }
+
+#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
+#[path = "forward/bulk_f32.rs"]
+mod bulk_f32;
+#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
+pub(crate) use bulk_f32::bulk_f32_slice;
diff --git a/crates/zerogravity-model-qwen35/src/host_math.rs b/crates/zerogravity-model-qwen35/src/host_math.rs
index e55a93c..f00e8d7 100644
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -22,6 +22,8 @@ use zerogravity_artifacts::source_weights::WindowWorkspace;
 
 struct HostStorage {
     values: Vec<f32>,
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    bulk: Option<bulk_f32::BulkBacking>,
     // Payload drops before its actual same-source host reservation.
     #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
     workspace: Option<Arc<WindowWorkspace>>,
@@ -89,6 +91,8 @@ impl HostValue {
         Self {
             data: Arc::new(HostStorage {
                 values,
+                #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+                bulk: None,
                 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
                 workspace: None,
             }),
@@ -128,11 +132,31 @@ impl HostValue {
                 "host operator requires explicit contiguous input".into(),
             ));
         }
-        self.data
-            .get(self.view.offset..self.view.offset + self.elements)
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        let values = match &self.data.bulk {
+            Some(backing) => backing.values()?,
+            None => &self.data.values,
+        };
+        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
+        let values = &self.data.values;
+        values
+            .get(
+                self.view.offset
+                    ..self
+                        .view
+                        .offset
+                        .checked_add(self.elements)
+                        .ok_or_else(|| Error::Limit("host view end overflow".into()))?,
+            )
             .ok_or_else(|| Error::State("invalid private host tensor view".into()))
     }
     pub fn into_vec(self) -> Result<Vec<f32>> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if self.data.bulk.is_some() {
+            return Err(Error::Unsupported(
+                "leased bulk result has no Vec escape".into(),
+            ));
+        }
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         if self.data.workspace.is_some() {
             return Err(Error::Unsupported(
@@ -151,6 +175,8 @@ impl HostValue {
 
 pub(super) struct HostMath<'engine, 'state, 'trace> {
     source: HostSource<'engine>,
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    bulk_call: Option<bulk_f32::BulkCall<'engine>>,
     engine: Option<&'engine QwenDenseCpu>,
     state: Option<&'state mut QwenState>,
     region_kernel: Option<&'engine dyn crate::accelerator::MatVecKernel>,
@@ -168,6 +194,8 @@ impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
         let position = state.position;
         Self {
             source: HostSource::Loaded(engine.model.as_ref()),
+            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+            bulk_call: None,
             engine: Some(engine),
             state: Some(state),
             region_kernel: engine.kernel.as_deref(),
@@ -183,6 +211,8 @@ impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
     ) -> Self {
         Self {
             source: HostSource::Loaded(source),
+            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+            bulk_call: None,
             engine: None,
             state: None,
             region_kernel: kernel,
@@ -200,6 +230,8 @@ impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
     ) -> Self {
         Self {
             source: HostSource::Windows(source),
+            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+            bulk_call: None,
             engine: None,
             state: None,
             region_kernel: kernel,
@@ -302,6 +334,12 @@ impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
             .ok_or_else(|| Error::State("operator slice has no session state".into()))
     }
     fn legacy_precision(&self, values: &[&HostValue]) -> Result<()> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if values.iter().any(|v| v.data.bulk.is_some()) {
+            return Err(Error::Unsupported(
+                "bulk F32 values require explicit matrix contract".into(),
+            ));
+        }
         if self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
             || values
                 .iter()
@@ -342,6 +380,22 @@ impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
 impl MathematicalBackend for HostMath<'_, '_, '_> {
     type Value = HostValue;
     type Index = usize;
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    fn bulk_f32_input(
+        &mut self,
+        operand: &crate::accelerator::program::BulkF32Operand,
+    ) -> Result<HostValue> {
+        self.import_bulk_f32(operand)
+    }
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    fn bulk_f32_matmul(
+        &mut self,
+        matrix: &crate::accelerator::program::BulkF32Matrix,
+        left: &HostValue,
+        right: &HostValue,
+    ) -> Result<HostValue> {
+        self.execute_bulk_f32(matrix, left, right)
+    }
     fn elements(&self, value: &HostValue) -> usize {
         value.elements
     }
@@ -545,6 +599,12 @@ impl MathematicalBackend for HostMath<'_, '_, '_> {
         ))
     }
     fn view(&mut self, input: &HostValue, view: TensorView) -> Result<HostValue> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if input.data.bulk.is_some() {
+            return Err(Error::Unsupported(
+                "bulk geometry cannot be reinterpreted by legacy view".into(),
+            ));
+        }
         // A view of an already strided view must first use explicit packing.
         if !input.view.is_contiguous() {
             return Err(Error::State(
@@ -568,6 +628,12 @@ impl MathematicalBackend for HostMath<'_, '_, '_> {
         })
     }
     fn contiguous(&mut self, input: &HostValue) -> Result<HostValue> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if input.data.bulk.is_some() && input.view != TensorView::vector(0, input.elements) {
+            return Err(Error::Unsupported(
+                "bulk geometry cannot be implicitly packed".into(),
+            ));
+        }
         if input.view.is_contiguous() {
             return Ok(input.clone());
         }
@@ -1053,3 +1119,6 @@ mod typed_value_tests {
 
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 mod gdn_conv;
+
+#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+mod bulk_f32;
diff --git a/crates/zerogravity-model-qwen35/src/operator_slice.rs b/crates/zerogravity-model-qwen35/src/operator_slice.rs
index b212b81..9dca5b0 100644
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -234,3 +234,8 @@ pub mod attention;
 
 #[path = "gdn_conv_slice.rs"]
 pub mod gdn_conv;
+
+/// Explicit external/source F32 temporal matrix, no model/state eligibility.
+#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
+#[path = "bulk_f32_slice.rs"]
+pub mod bulk_f32;

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-default-bulk-20261020/crates/zerogravity-core/src/accelerator/program/bulk_f32.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 12667; SHA256 `57805fe39fdee6bee7ece57e0b8fc2163ea73a1c85d78ca6c11b716d82ae110f`.

````text
//! Closed temporal matrix metadata; no session, model route or producer permit.
use super::*;

/// ONE consecutive teacher sequence. Clones share one immutable issuer; equal
/// text/IDs from another constructor do not silently become the same scope.
#[derive(Clone, Debug)]
pub struct TeacherSpan(Arc<TeacherSpanData>);
#[derive(Debug)]
struct TeacherSpanData {
    source: SourceIdentity,
    base: usize,
    capacity: usize,
    ids: Arc<[u32]>,
}
impl PartialEq for TeacherSpan {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for TeacherSpan {}
impl TeacherSpan {
    /// Metadata only; these IDs are not a current SessionAuthority/StateRoot.
    pub fn new(
        source: SourceIdentity,
        base: usize,
        capacity: usize,
        vocabulary: u64,
        ids: &[u32],
    ) -> Result<Self> {
        if !source.is_bound()
            || !(2..=64).contains(&ids.len())
            || capacity == 0
            || capacity > 8448
            || base.checked_add(ids.len()).is_none_or(|end| end > capacity)
            || vocabulary == 0
            || vocabulary > u64::from(u32::MAX) + 1
            || ids.iter().any(|&id| u64::from(id) >= vocabulary)
        {
            return Err(Error::Malformed(
                "closed teacher span source/geometry/IDs".into(),
            ));
        }
        Ok(Self(Arc::new(TeacherSpanData {
            source,
            base,
            capacity,
            ids: Arc::from(ids),
        })))
    }
    pub fn source(&self) -> SourceIdentity {
        self.0.source
    }
    pub fn base(&self) -> usize {
        self.0.base
    }
    pub fn capacity(&self) -> usize {
        self.0.capacity
    }
    pub fn ids(&self) -> &[u32] {
        &self.0.ids
    }
    pub fn len(&self) -> usize {
        self.0.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Actual F32 storage axes `[full_K,M_or_N,head,sequence]`, word strides.
/// Offset/backing describe THIS allocation; original capture provenance stays
/// separate if transport reconstructed gaps. Full K is never split/reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BulkF32Layout {
    shape: [usize; 4],
    strides: [usize; 4],
    offset: usize,
    backing: usize,
}
impl BulkF32Layout {
    pub fn new(
        shape: [usize; 4],
        strides: [usize; 4],
        offset: usize,
        backing: usize,
    ) -> Result<Self> {
        let value = Self {
            shape,
            strides,
            offset,
            backing,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn dense(shape: [usize; 4]) -> Result<Self> {
        let mut strides = [1usize; 4];
        for i in 1..4 {
            strides[i] = strides[i - 1]
                .checked_mul(shape[i - 1])
                .ok_or_else(|| Error::Limit("bulk dense stride".into()))?;
        }
        let backing = strides[3]
            .checked_mul(shape[3])
            .ok_or_else(|| Error::Limit("bulk dense words".into()))?;
        Self::new(shape, strides, 0, backing)
    }
    pub fn shape(self) -> [usize; 4] {
        self.shape
    }
    pub fn strides(self) -> [usize; 4] {
        self.strides
    }
    pub fn offset(self) -> usize {
        self.offset
    }
    pub fn backing_words(self) -> usize {
        self.backing
    }
    pub fn logical_words(self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |n, &x| {
            n.checked_mul(x)
                .ok_or_else(|| Error::Limit("bulk logical words".into()))
        })
    }
    pub fn validate(self) -> Result<()> {
        if self.shape.contains(&0)
            || self.shape[0] > 8192
            || self.shape[1] > 4096
            || self.shape[2] > 64
            || self.shape[3] != 1
            || self.strides[0] != 1
            || self.backing == 0
            || self.backing > 8 * 1024 * 1024
        {
            return Err(Error::Unsupported("closed bulk F32 axes/backing".into()));
        }
        let mut previous = self.shape[0];
        for (&stride, &size) in self.strides.iter().zip(&self.shape).skip(1) {
            if stride < previous {
                return Err(Error::Malformed("bulk overlapping rows/axes".into()));
            }
            previous = stride
                .checked_mul(size)
                .ok_or_else(|| Error::Limit("bulk stride product".into()))?;
        }
        let end = self
            .shape
            .iter()
            .zip(self.strides)
            .try_fold(self.offset, |n, (&size, stride)| {
                (size - 1)
                    .checked_mul(stride)
                    .and_then(|x| n.checked_add(x))
                    .ok_or_else(|| Error::Limit("bulk full span".into()))
            })?
            .checked_add(1)
            .ok_or_else(|| Error::Limit("bulk span end".into()))?;
        if end > self.backing || self.logical_words()? > 1024 * 1024 {
            return Err(Error::Limit("bulk operand span/logical bound".into()));
        }
        Ok(())
    }
    pub fn is_dense(self) -> bool {
        Self::dense(self.shape).is_ok_and(|dense| dense == self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulkF32Side {
    Left,
    Right,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BulkF32Origin {
    /// Caller-supplied pure operator input, explicitly NOT a model/state producer.
    ExternalObserved { name: String },
    /// Exact whole original raw learned matrix, no dequantization/retag/cache.
    SourceMatrix(SourceTensor),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BulkF32Operand {
    span: TeacherSpan,
    side: BulkF32Side,
    layout: BulkF32Layout,
    origin: BulkF32Origin,
}
impl BulkF32Operand {
    pub fn external(
        span: TeacherSpan,
        side: BulkF32Side,
        name: &str,
        layout: BulkF32Layout,
    ) -> Result<Self> {
        if name.is_empty() || name.len() > 256 || name.contains('\0') {
            return Err(Error::Malformed("bulk external input label bound".into()));
        }
        let value = Self {
            span,
            side,
            layout,
            origin: BulkF32Origin::ExternalObserved { name: name.into() },
        };
        value.validate()?;
        Ok(value)
    }
    pub fn source_matrix(span: TeacherSpan, source: &dyn TensorSource, name: &str) -> Result<Self> {
        if name.is_empty() || name.len() > 256 || name.contains('\0') {
            return Err(Error::Malformed("bulk learned input label bound".into()));
        }
        let tensor = SourceTensor::from_source(source, name, MatrixAxes::InputOutput)?;
        if tensor.kind != QuantKind::F32 || tensor.source != span.source() {
            return Err(Error::Unsupported(
                "bulk original F32 source matrix only".into(),
            ));
        }
        let layout = BulkF32Layout::dense([tensor.columns, tensor.rows, 1, 1])?;
        let value = Self {
            span,
            side: BulkF32Side::Left,
            layout,
            origin: BulkF32Origin::SourceMatrix(tensor),
        };
        value.validate()?;
        Ok(value)
    }
    pub fn span(&self) -> &TeacherSpan {
        &self.span
    }
    pub fn side(&self) -> BulkF32Side {
        self.side
    }
    pub fn layout(&self) -> BulkF32Layout {
        self.layout
    }
    pub fn origin(&self) -> &BulkF32Origin {
        &self.origin
    }
    pub fn name(&self) -> &str {
        match &self.origin {
            BulkF32Origin::ExternalObserved { name } => name,
            BulkF32Origin::SourceMatrix(tensor) => &tensor.name,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        if self.side == BulkF32Side::Right
            && (self.layout.shape[1] != self.span.len() || !self.layout.is_dense())
        {
            return Err(Error::Unsupported(
                "actual teacher N and original contiguous RHS required".into(),
            ));
        }
        if let BulkF32Origin::SourceMatrix(tensor) = &self.origin {
            if self.side != BulkF32Side::Left
                || tensor.source != self.span.source()
                || tensor.kind != QuantKind::F32
                || tensor.axes != MatrixAxes::InputOutput
                || self.layout != BulkF32Layout::dense([tensor.columns, tensor.rows, 1, 1])?
            {
                return Err(Error::State("bulk whole learned descriptor/axes".into()));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulkF32Contract {
    HaswellTinyBlasF32V1,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BulkF32Matrix {
    left: BulkF32Operand,
    right: BulkF32Operand,
    contract: BulkF32Contract,
    output: BulkF32Layout,
}
impl BulkF32Matrix {
    pub fn new(
        left: BulkF32Operand,
        right: BulkF32Operand,
        contract: BulkF32Contract,
    ) -> Result<Self> {
        left.validate()?;
        right.validate()?;
        let a = left.layout.shape;
        let b = right.layout.shape;
        if left.side != BulkF32Side::Left
            || right.side != BulkF32Side::Right
            || left.span != right.span
            || a[0] != b[0]
            || !a[1].is_multiple_of(4)
            || !b[2].is_multiple_of(a[2])
        {
            return Err(Error::Unsupported(
                "closed temporal matrix issuer/fullK/head broadcast".into(),
            ));
        }
        let output = BulkF32Layout::dense([a[1], b[1], b[2], 1])?;
        if output
            .logical_words()?
            .checked_mul(a[0])
            .is_none_or(|n| n > 128 * 1024 * 1024)
        {
            return Err(Error::Limit("bulk complete K work bound".into()));
        }
        Ok(Self {
            left,
            right,
            contract,
            output,
        })
    }
    pub fn left(&self) -> &BulkF32Operand {
        &self.left
    }
    pub fn right(&self) -> &BulkF32Operand {
        &self.right
    }
    pub fn contract(&self) -> BulkF32Contract {
        self.contract
    }
    pub fn output(&self) -> BulkF32Layout {
        self.output
    }
    pub fn source(&self) -> SourceIdentity {
        self.left.span.source()
    }
}

fn check_builder(builder: &PlanBuilder, source: SourceIdentity) -> Result<()> {
    if !builder.operator_slice
        || builder.profile != ProgramNumericalProfile::GgufSourceV1
        || builder.source != source
    {
        return Err(Error::Unsupported(
            "bulk F32 requires explicit state-free GGUF operator slice".into(),
        ));
    }
    Ok(())
}
pub(super) fn input_precision(
    builder: &PlanBuilder,
    operand: &BulkF32Operand,
) -> Result<ActivationPrecision> {
    check_builder(builder, operand.span.source())?;
    operand.validate()?;
    Ok(ActivationPrecision::F32)
}
pub(super) fn matrix_precision(
    builder: &PlanBuilder,
    matrix: &BulkF32Matrix,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    check_builder(builder, matrix.source())?;
    if inputs.len() != 2 {
        return Err(Error::Malformed("bulk input arity".into()));
    }
    for (input, expected) in inputs.iter().zip([matrix.left(), matrix.right()]) {
        let producer = builder
            .nodes
            .iter()
            .find(|node| node.outputs.contains(&input.id));
        if input.precision != ActivationPrecision::F32
            || input.elements != expected.layout.backing_words()
            || !matches!(producer.map(|node|&node.operation),Some(PlanOperation::TypedBulkF32Input{operand})if operand==expected)
        {
            return Err(Error::State(
                "bulk exact input producer/layout/span membership".into(),
            ));
        }
    }
    Ok(ActivationPrecision::F32)
}
pub(super) fn record_input(
    builder: &mut PlanBuilder,
    operand: &BulkF32Operand,
) -> Result<PlanValue> {
    input_precision(builder, operand)?;
    builder.one(
        operand.name(),
        PlanOperation::TypedBulkF32Input {
            operand: operand.clone(),
        },
        &[],
        ValueShape::Vector(operand.layout.backing_words()),
        None,
    )
}
pub(super) fn record_matrix(
    builder: &mut PlanBuilder,
    matrix: &BulkF32Matrix,
    left: &PlanValue,
    right: &PlanValue,
) -> Result<PlanValue> {
    matrix_precision(builder, matrix, &[*left, *right])?;
    builder.one(
        "closed-bulk-f32",
        PlanOperation::TypedBulkF32Matmul {
            matrix: matrix.clone(),
        },
        &[*left, *right],
        ValueShape::Vector(matrix.output.backing_words()),
        None,
    )
}

#[cfg(test)]
mod tests;

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-default-bulk-20261020/crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6711; SHA256 `fbbd8b45b26c201e558d555dbc41335010bf391e5578ceb0e50c9cc8a028c1fc`.

````text
use super::*;
fn span() -> TeacherSpan {
    TeacherSpan::new(
        SourceIdentity::for_new_allocation().unwrap(),
        0,
        8448,
        248320,
        &(0..64).collect::<Vec<_>>(),
    )
    .unwrap()
}
fn matrix(span: &TeacherSpan) -> BulkF32Matrix {
    let left = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Left,
        "actual external V",
        BulkF32Layout::new([256, 256, 2, 1], [1, 8448, 2162688, 4325376], 0, 4317184).unwrap(),
    )
    .unwrap();
    let right = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "actual external P",
        BulkF32Layout::dense([256, 64, 8, 1]).unwrap(),
    )
    .unwrap();
    BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1).unwrap()
}
#[test]
fn actual_temporal_n_is_not_cohort_width_and_full_k_is_preserved() {
    let span = span();
    let operation = matrix(&span);
    assert_eq!(operation.left().layout().shape(), [256, 256, 2, 1]);
    assert_eq!(operation.right().layout().shape(), [256, 64, 8, 1]);
    assert_eq!(operation.output().shape(), [256, 64, 8, 1]);
    assert_eq!(operation.output().backing_words(), 131072);
    // Real N2 is a different declared teacher span, never a B2 request label.
    let wrong = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "P",
        BulkF32Layout::dense([256, 2, 8, 1]).unwrap(),
    );
    assert!(wrong.is_err());
    assert!(TeacherSpan::new(span.source(), 0, 8448, 248320, &[11]).is_err());
    assert!(TeacherSpan::new(span.source(), 8447, 8448, 248320, &[11, 353]).is_err());
}
#[test]
fn same_ids_source_names_do_not_merge_issuers_and_wrong_full_k_refuses() {
    let span = span();
    let operation = matrix(&span);
    let foreign = TeacherSpan::new(
        span.source(),
        span.base(),
        span.capacity(),
        248320,
        span.ids(),
    )
    .unwrap();
    let right = BulkF32Operand::external(
        foreign,
        BulkF32Side::Right,
        "actual external P",
        operation.right().layout(),
    )
    .unwrap();
    assert!(
        BulkF32Matrix::new(
            operation.left().clone(),
            right,
            BulkF32Contract::HaswellTinyBlasF32V1
        )
        .is_err()
    );
    let right = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "P",
        BulkF32Layout::dense([128, 64, 8, 1]).unwrap(),
    )
    .unwrap();
    assert!(
        BulkF32Matrix::new(
            operation.left().clone(),
            right,
            BulkF32Contract::HaswellTinyBlasF32V1
        )
        .is_err()
    );
}
#[test]
fn real_nonzero_offsets_and_gaps_need_full_checked_storage() {
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 47).is_ok());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 46).is_err());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 7, 28, 28], 0, 32).is_err());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], usize::MAX, 64).is_err());
    assert!(BulkF32Layout::dense([8192, 4096, 8, 1]).is_err());
    let span = span();
    assert!(
        BulkF32Operand::external(
            span,
            BulkF32Side::Right,
            "strided P",
            BulkF32Layout::new([8, 64, 1, 1], [1, 12, 768, 768], 0, 764).unwrap()
        )
        .is_err()
    );
}
#[test]
fn same_builder_three_nodes_stay_pure_and_legacy_use_or_foreign_values_refuse() {
    let span = span();
    let operation = matrix(&span);
    let mut b =
        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let left = b.bulk_f32_input(operation.left()).unwrap();
    let right = b.bulk_f32_input(operation.right()).unwrap();
    let mut other =
        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let foreign = other.bulk_f32_input(operation.right()).unwrap();
    assert!(b.bulk_f32_matmul(&operation, &left, &foreign).is_err());
    assert!(b.silu("legacy", &right, None).is_err());
    let output = b.bulk_f32_matmul(&operation, &left, &right).unwrap();
    let slice = b.finish_operator_slice(output).unwrap();
    assert_eq!(slice.plan().nodes().len(), 3);
    assert!(slice.plan().required_domains().is_empty());
    assert!(slice.plan().proposed_domains().is_empty());
    assert_eq!(slice.plan().frontier_delta(), 0);
    assert!(slice.plan().logits().is_none());
    assert!(slice.plan().is_operator_slice());
    let slot = StateSlot::Conv(ConvDomain(0));
    let mut full = PlanBuilder::new(span.source(), 8448, vec![slot], false).unwrap();
    assert!(full.bulk_f32_input(operation.left()).is_err());
}

struct Original {
    identity: SourceIdentity,
    kind: QuantKind,
    shape: [u64; 2],
    bytes: Vec<u8>,
}
impl TensorSource for Original {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        if name != "original.weight" {
            return Err(Error::Malformed("unknown original name".into()));
        }
        Ok(SourceTensorRef {
            kind: self.kind,
            shape: &self.shape,
            packed_bytes: &self.bytes,
        })
    }
}
#[test]
fn whole_raw_source_matrix_keeps_axes_dtype_extent_and_exact_issuer() {
    let original = Original {
        identity: SourceIdentity::for_new_allocation().unwrap(),
        kind: QuantKind::F32,
        shape: [8, 4],
        bytes: vec![0; 128],
    };
    let span = TeacherSpan::new(original.identity, 0, 8, 248320, &[11, 353]).unwrap();
    let left = BulkF32Operand::source_matrix(span.clone(), &original, "original.weight").unwrap();
    assert_eq!(left.layout().shape(), [8, 4, 1, 1]);
    let BulkF32Origin::SourceMatrix(raw) = left.origin() else {
        panic!("original raw source expected")
    };
    assert_eq!(raw.axes, MatrixAxes::InputOutput);
    raw.validate_source(&original).unwrap();
    let foreign = Original {
        identity: SourceIdentity::for_new_allocation().unwrap(),
        ..original
    };
    assert!(BulkF32Operand::source_matrix(span.clone(), &foreign, "original.weight").is_err());
    let bf16 = Original {
        identity: span.source(),
        kind: QuantKind::Bf16,
        shape: [8, 4],
        bytes: vec![0; 64],
    };
    assert!(BulkF32Operand::source_matrix(span.clone(), &bf16, "original.weight").is_err());
    let short = Original {
        identity: span.source(),
        kind: QuantKind::F32,
        shape: [8, 4],
        bytes: vec![0; 124],
    };
    assert!(BulkF32Operand::source_matrix(span, &short, "original.weight").is_err());
}

````

## Source: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-default-bulk-20261020/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 14301; SHA256 `7557b599baf43d2f6e8b598484aee4fe95200b87972288df5c396a60c38331a5`.

````text
//! Closed GGML d812350 Haswell tinyBLAS F32 schedule, no model/plan selection.
//!
//! One Vec8 FMA chain per output and the original even/odd horizontal tree.
//! Default GGUF four-Vec8 projections/attention and source profiles are unchanged.
//! Source: ggml/src/ggml-cpu/llamafile/sgemm.cpp:252-270,559-730,3908-3934.

// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use crate::accelerator::SourceIdentity;
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};

const MAX_SPAN_WORDS: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_WORDS: usize = 1024 * 1024;
const MAX_FMAS: usize = 128 * 1024 * 1024;
const SCRATCH_BYTES: u64 = 256;

#[cfg(test)]
std::thread_local! {
    static AFTER_FIRST_OUTPUT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Distinct closed arithmetic contract; no executable model/profile capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BulkF32Profile {
    GgmlD812350HaswellTinyBlas,
}

/// Original F32 axes `[inner,rows,head,sequence]`; strides/offset are in words.
/// Borrowed raw activations/weights retain their caller's owner and admission.
/// This descriptor validates a view, not its model-file SHA or source issuer.
#[derive(Clone, Copy)]
pub struct F32View<'a> {
    pub source: SourceIdentity,
    pub name: &'a str,
    pub values: &'a [f32],
    pub offset: usize,
    pub shape: [usize; 4],
    pub strides: [usize; 4],
}

/// Same source for both operands. B is the original contiguous RHS; A may
/// retain a strided V-cache view. All broadcast heads are independently read.
#[derive(Clone, Copy)]
pub struct BulkF32Request<'a> {
    pub profile: BulkF32Profile,
    pub source: SourceIdentity,
    pub left: F32View<'a>,
    pub right: F32View<'a>,
}

/// Existing host ledger, not a new pool or a claim about all host RAM.
pub struct BulkF32Admission<'a> {
    pub arbiter: &'a ResourceArbiter,
    pub resource: &'a PhysicalDeviceKey,
    pub owner: &'a ResourceOwner,
}

/// Payload drops before its original output/scratch ticket. No unleased Vec
/// escape or state publication; synchronous borrowed inputs outlive execution.
pub struct BulkF32Completed {
    values: Vec<f32>,
    source: SourceIdentity,
    shape: [usize; 4],
    _lease: GlobalResourceLease,
}
impl BulkF32Completed {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
    pub fn profile(&self) -> BulkF32Profile {
        BulkF32Profile::GgmlD812350HaswellTinyBlas
    }
}

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("Haswell bulk F32 cancelled".into()));
    }
    Ok(())
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &next| {
        acc.checked_mul(next)
            .ok_or_else(|| Error::Limit("Haswell bulk F32 geometry overflow".into()))
    })
}

impl F32View<'_> {
    fn validate(self, source: SourceIdentity, cancelled: Option<&AtomicBool>) -> Result<()> {
        not_cancelled(cancelled)?;
        if !source.is_bound()
            || self.source != source
            || self.name.is_empty()
            || self.name.len() > 256
            || self.name.contains('\0')
            || self.shape.contains(&0)
            || self.shape[0] > 8192
            || self.shape[1] > 4096
            || self.shape[2] > 64
            || self.shape[3] > 64
            || self.strides[0] != 1
        {
            return Err(Error::Malformed(
                "Haswell bulk F32 source/view geometry".into(),
            ));
        }
        // Original matrix rows do not overlap. Axis3's declared stride is
        // checked even when that dimension has size1, preserving raw geometry.
        let mut previous = self.shape[0];
        for (&stride, &size) in self.strides.iter().zip(&self.shape).skip(1) {
            if stride < previous {
                return Err(Error::Malformed(
                    "Haswell bulk F32 overlapping strides".into(),
                ));
            }
            previous = stride
                .checked_mul(size)
                .ok_or_else(|| Error::Limit("Haswell bulk F32 stride overflow".into()))?;
        }
        let end = self
            .shape
            .iter()
            .zip(self.strides)
            .try_fold(self.offset, |at, (&size, stride)| {
                (size - 1)
                    .checked_mul(stride)
                    .and_then(|n| at.checked_add(n))
                    .ok_or_else(|| Error::Limit("Haswell bulk F32 view span overflow".into()))
            })?
            .checked_add(1)
            .ok_or_else(|| Error::Limit("Haswell bulk F32 view end overflow".into()))?;
        if end > self.values.len() || self.values.len() > MAX_SPAN_WORDS {
            return Err(Error::Limit(
                "Haswell bulk F32 view exceeds backing/bound".into(),
            ));
        }
        let words = checked_product(&self.shape)?;
        if words > MAX_OUTPUT_WORDS {
            return Err(Error::Limit(
                "Haswell bulk F32 logical operand bound".into(),
            ));
        }
        // Only actual logical words are read; storage gaps/canaries remain
        // outside the arithmetic and may contain arbitrary IEEE754 words.
        for sequence in 0..self.shape[3] {
            for head in 0..self.shape[2] {
                for row in 0..self.shape[1] {
                    not_cancelled(cancelled)?;
                    let start = self.at(0, row, head, sequence);
                    if self.values[start..start + self.shape[0]]
                        .iter()
                        .any(|x| !x.is_finite())
                    {
                        return Err(Error::Malformed(
                            "nonfinite Haswell bulk F32 operand".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    fn at(self, inner: usize, row: usize, head: usize, sequence: usize) -> usize {
        // Whole view's final index and products were checked before access.
        self.offset
            + inner
            + row * self.strides[1]
            + head * self.strides[2]
            + sequence * self.strides[3]
    }
}

fn validate(
    request: BulkF32Request<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<([usize; 4], usize)> {
    not_cancelled(cancelled)?;
    let BulkF32Profile::GgmlD812350HaswellTinyBlas = request.profile;
    let a = request.left;
    let b = request.right;
    a.validate(request.source, cancelled)?;
    b.validate(request.source, cancelled)?;
    if a.shape[0] != b.shape[0]
        || !(2..=64).contains(&b.shape[1])
        || !a.shape[1].is_multiple_of(4)
        || !b.shape[2].is_multiple_of(a.shape[2])
        || !b.shape[3].is_multiple_of(a.shape[3])
    {
        // n<2 selects the original vec-dot path, never this bulk schedule.
        return Err(Error::Unsupported(
            "unsupported Haswell tinyBLAS F32 shape".into(),
        ));
    }
    let mut stride = 1usize;
    for (&actual, &size) in b.strides.iter().zip(&b.shape) {
        if actual != stride {
            return Err(Error::Malformed(
                "Haswell tinyBLAS requires original contiguous RHS".into(),
            ));
        }
        stride = stride
            .checked_mul(size)
            .ok_or_else(|| Error::Limit("Haswell bulk F32 contiguous span overflow".into()))?;
    }
    let shape = [a.shape[1], b.shape[1], b.shape[2], b.shape[3]];
    let count = checked_product(&shape)?;
    if count > MAX_OUTPUT_WORDS || count.checked_mul(a.shape[0]).is_none_or(|n| n > MAX_FMAS) {
        return Err(Error::Limit("Haswell bulk F32 output/work bound".into()));
    }
    not_cancelled(cancelled)?;
    Ok((shape, count))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let full = a.len() & !7usize;
    for base in (0..full).step_by(8) {
        for (lane, sum) in lanes.iter_mut().enumerate() {
            *sum = a[base + lane].mul_add(b[base + lane], *sum);
        }
    }
    if full != a.len() {
        // tinyBLAS load_partial zero-fills BOTH remaining Vec8 operands.
        // Execute those +0*+0 FMAs too; skipping them changes zero signs.
        for (lane, sum) in lanes.iter_mut().enumerate() {
            let at = full + lane;
            let (x, y) = if at < a.len() {
                (a[at], b[at])
            } else {
                (0.0, 0.0)
            };
            *sum = x.mul_add(y, *sum);
        }
    }
    let low = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    // Actual hsum(__m128): add(movehl), then add_ss(movehdup).
    // This is (0+2)+(1+3), not vec_dot's (0+1)+(2+3).
    (low[0] + low[2]) + (low[1] + low[3])
}

/// Pure source-derived primitive. Provider/model/session/checkpoint eligibility
/// is unchanged. Caller authenticates operands and retains input reservations;
/// this function admits only its own private result and bounded stack scratch.
pub fn execute(
    request: BulkF32Request<'_>,
    admission: BulkF32Admission<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<BulkF32Completed> {
    let (shape, count) = validate(request, cancelled)?;
    if admission.resource.backend != "host" {
        return Err(Error::Unsupported(
            "Haswell bulk F32 requires existing host admission".into(),
        ));
    }
    let output_bytes = count
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("Haswell bulk F32 output bytes overflow".into()))?;
    let lease = admission.arbiter.admit(
        admission.resource,
        admission.owner.clone(),
        &[
            AllocationRequest {
                kind: AllocationClass::Output,
                bytes: output_bytes as u64,
            },
            AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: SCRATCH_BYTES,
            },
        ],
    )?;
    not_cancelled(cancelled)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("Haswell bulk F32 output allocation".into()))?;
    if values.capacity() != count {
        return Err(Error::Limit(
            "Haswell bulk F32 unexpected output capacity".into(),
        ));
    }
    let a = request.left;
    let b = request.right;
    let ratio2 = b.shape[2] / a.shape[2];
    let ratio3 = b.shape[3] / a.shape[3];
    for sequence in 0..shape[3] {
        for head in 0..shape[2] {
            for column in 0..shape[1] {
                not_cancelled(cancelled)?;
                let br = b.at(0, column, head, sequence);
                for row in 0..shape[0] {
                    not_cancelled(cancelled)?;
                    let ar = a.at(0, row, head / ratio2, sequence / ratio3);
                    let value = dot(
                        &a.values[ar..ar + a.shape[0]],
                        &b.values[br..br + b.shape[0]],
                    );
                    if !value.is_finite() {
                        return Err(Error::Limit("Haswell bulk F32 nonfinite result".into()));
                    }
                    values.push(value);
                    #[cfg(test)]
                    if values.len() == 1 {
                        AFTER_FIRST_OUTPUT.with(|hook| {
                            if let Some(hook) = hook.borrow_mut().take() {
                                hook();
                            }
                        });
                    }
                }
            }
        }
    }
    not_cancelled(cancelled)?;
    Ok(BulkF32Completed {
        values,
        source: request.source,
        shape,
        _lease: lease,
    })
}

#[cfg(test)]
mod tests;

````

