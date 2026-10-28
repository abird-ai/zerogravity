//! Qualification-only lowering audit for the integrated ordinary CUDA path.
//! This retains the exact model invocation and checks all reviewed primitive
//! contracts. Projection, source-exact attention, GDN, and physical
//! queue/flight/completion/StateRoot ownership are now integrated under the
//! existing resident owner; this module grants no production routing.

use super::*;
use crate::gpu::cuda::ordinary_text::{
    Access, BufferRole, DefinitionContract, Geometry, PlanPrimitive, Program, Step,
};
use crate::qwen35::plan::{OrdinaryQwenInvocation, OrdinaryQwenPlan, QwenSourceProfile};
use zerogravity_core::accelerator::program::{
    PlanOperation, SourceGamma, SourceNumericalContract, SourceOperatorKind, TensorView,
};

const EPSILON: f32 = 1e-6;

struct LoweringAudit {
    // Retains the exact plan/source seal through the complete audit/refusal.
    _invocation: OrdinaryQwenInvocation,
    contracts: Vec<DefinitionContract>,
}

pub(super) fn require_binding(
    plan: &OrdinaryQwenPlan,
    invocation: &OrdinaryQwenInvocation,
) -> Result<()> {
    let audit = audit(plan, invocation)?;
    let mapped = audit.contracts.len();
    Err(Error::Unsupported(format!(
        "ordinary Qwen CUDA is integrated but qualification-only: audited {mapped} checked primitive definitions; current-source real-model parity, fault/cancellation/retirement/shutdown qualification, and context-fit gates remain"
    )))
}

fn audit(plan: &OrdinaryQwenPlan, invocation: &OrdinaryQwenInvocation) -> Result<LoweringAudit> {
    if !std::sync::Arc::ptr_eq(plan.plan(), invocation.plan())
        || plan.model_profile() != invocation.model_profile()
        || plan.context_capacity_tokens() != invocation.plan().prefix_capacity()
        || plan.source().source_identity() != invocation.source().source_identity()
        || !plan.source().same_issuer(invocation.source())
    {
        return Err(Error::State(
            "ordinary Qwen plan/invocation/source owner differs".into(),
        ));
    }
    let geometry = match plan.model_profile() {
        QwenSourceProfile::Qwen08 => Geometry::qwen35_08(),
        QwenSourceProfile::Qwen2B => Geometry::qwen35_2b(),
    };
    let program = Program::new(geometry)?;
    let step = Step::new(
        invocation.rows(),
        invocation.base(),
        plan.context_capacity_tokens(),
    )?;
    let mut contracts = vec![program.contract(PlanPrimitive::Positions, step)?];
    for node in invocation.active_nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            return Err(Error::State(
                "ordinary Qwen invocation contains a non-source-template node".into(),
            ));
        };
        if descriptor.source() != plan.plan().source()
            || descriptor.max_rows() != plan.max_chunk_rows()
            || descriptor.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
        {
            return Err(Error::State(
                "ordinary Qwen source template identity/contract differs".into(),
            ));
        }
        for operation in plan_primitives(descriptor.kind(), geometry, invocation.rows())? {
            contracts.push(program.contract(operation, step)?);
        }
    }
    if contracts.iter().any(|contract| {
        contract.bindings.last().is_none_or(|binding| {
            binding.role != BufferRole::Status
                || binding.access != Access::ReadWrite
                || binding.bytes != 4
        })
    }) {
        return Err(Error::State(
            "ordinary Qwen primitive status binding contract differs".into(),
        ));
    }
    Ok(LoweringAudit {
        _invocation: invocation.clone(),
        contracts,
    })
}

fn plan_primitives(
    kind: SourceOperatorKind,
    geometry: Geometry,
    rows: usize,
) -> Result<Vec<PlanPrimitive>> {
    let operations = match kind {
        SourceOperatorKind::EmbeddingRows => vec![PlanPrimitive::Embedding],
        SourceOperatorKind::Rms(g) => {
            require_rms(g.width, g.epsilon, g.gamma, geometry.hidden_width)?;
            vec![PlanPrimitive::Rms { epsilon: g.epsilon }]
        }
        SourceOperatorKind::ResidualRms(g) => {
            require_rms(g.width, g.epsilon, g.gamma, geometry.hidden_width)?;
            vec![PlanPrimitive::ResidualRms { epsilon: g.epsilon }]
        }
        SourceOperatorKind::QkRmsRotaryGated(g) => {
            if g.attention.width != geometry.head_width
                || g.attention.query_heads != geometry.query_heads
                || g.attention.kv_heads != geometry.kv_heads
                || g.rotary_dimensions != geometry.rotary_width
                || g.theta.to_bits() != 10_000_000.0f32.to_bits()
                || g.epsilon.to_bits() != EPSILON.to_bits()
            {
                return Err(Error::State(
                    "ordinary Qwen packed Q/K/RoPE geometry differs".into(),
                ));
            }
            vec![
                PlanPrimitive::QueryHeadRms { epsilon: g.epsilon },
                PlanPrimitive::KeyHeadRms { epsilon: g.epsilon },
                PlanPrimitive::Rope,
            ]
        }
        SourceOperatorKind::KvAppend {
            key_width,
            packed_value,
            ..
        } => {
            let query_width = geometry.query_width()?;
            let kv_width = geometry.kv_width()?;
            let value_offset = query_width
                .checked_mul(2)
                .and_then(|width| width.checked_add(kv_width))
                .ok_or_else(|| Error::Limit("ordinary Qwen packed value offset overflow".into()))?;
            let row_stride = value_offset.checked_add(kv_width).ok_or_else(|| {
                Error::Limit("ordinary Qwen packed attention stride overflow".into())
            })?;
            let expected = TensorView {
                offset: value_offset,
                rows,
                width: kv_width,
                row_stride,
            };
            if key_width != kv_width || packed_value != expected {
                return Err(Error::State(
                    "ordinary Qwen packed KV append view differs".into(),
                ));
            }
            vec![PlanPrimitive::KvAppend]
        }
        SourceOperatorKind::Sigmoid { width } => {
            if width != geometry.query_width()? {
                return Err(Error::State("ordinary Qwen gate width differs".into()));
            }
            vec![PlanPrimitive::Sigmoid]
        }
        SourceOperatorKind::Multiply { width } => {
            if width != geometry.query_width()? {
                return Err(Error::State(
                    "ordinary Qwen attention multiply width differs".into(),
                ));
            }
            vec![PlanPrimitive::Multiply]
        }
        SourceOperatorKind::SiluAndMul { width } => {
            if width != geometry.intermediate_width {
                return Err(Error::State(
                    "ordinary Qwen packed GateUp width differs".into(),
                ));
            }
            vec![PlanPrimitive::SiluAndMul]
        }
        // These execute through the integrated resident owner rather than the
        // ordinary-text primitive ABI audited here.
        SourceOperatorKind::Projection { .. }
        | SourceOperatorKind::CausalConvSplit { .. }
        | SourceOperatorKind::GatedDelta { .. }
        | SourceOperatorKind::FlashAttention { .. } => Vec::new(),
        SourceOperatorKind::RetainedRows { .. }
        | SourceOperatorKind::CausalConvSiluZero { .. }
        | SourceOperatorKind::GroupedQkNormalize(_)
        | SourceOperatorKind::StateCarry { .. } => {
            return Err(Error::Unsupported(
                "ordinary Qwen plan contains an unsupported source operator".into(),
            ));
        }
    };
    Ok(operations)
}

fn require_rms(width: usize, epsilon: f32, gamma: SourceGamma, hidden: usize) -> Result<()> {
    if width != hidden || epsilon.to_bits() != EPSILON.to_bits() || gamma != SourceGamma::OnePlus {
        return Err(Error::State("ordinary Qwen RMS contract differs".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::program::{
        AttentionGeometry, KvDomain, SourceQkRotaryGeometry, SourceRmsGeometry,
    };

    #[test]
    fn every_implemented_core_operation_maps_to_checked_primitives_for_both_profiles() -> Result<()>
    {
        for geometry in [Geometry::qwen35_08(), Geometry::qwen35_2b()] {
            let rms = SourceRmsGeometry {
                width: geometry.hidden_width,
                epsilon: EPSILON,
                gamma: SourceGamma::OnePlus,
            };
            let qk = SourceQkRotaryGeometry {
                attention: AttentionGeometry {
                    width: 256,
                    query_heads: 8,
                    kv_heads: 2,
                },
                rotary_dimensions: 64,
                theta: 10_000_000.0,
                epsilon: EPSILON,
            };
            let cases = [
                (
                    SourceOperatorKind::EmbeddingRows,
                    vec![PlanPrimitive::Embedding],
                ),
                (
                    SourceOperatorKind::Rms(rms),
                    vec![PlanPrimitive::Rms { epsilon: EPSILON }],
                ),
                (
                    SourceOperatorKind::ResidualRms(rms),
                    vec![PlanPrimitive::ResidualRms { epsilon: EPSILON }],
                ),
                (
                    SourceOperatorKind::QkRmsRotaryGated(qk),
                    vec![
                        PlanPrimitive::QueryHeadRms { epsilon: EPSILON },
                        PlanPrimitive::KeyHeadRms { epsilon: EPSILON },
                        PlanPrimitive::Rope,
                    ],
                ),
                (
                    SourceOperatorKind::KvAppend {
                        domain: KvDomain(3),
                        key_width: 512,
                        packed_value: TensorView {
                            offset: 4608,
                            rows: 17,
                            width: 512,
                            row_stride: 5120,
                        },
                    },
                    vec![PlanPrimitive::KvAppend],
                ),
                (
                    SourceOperatorKind::Sigmoid { width: 2048 },
                    vec![PlanPrimitive::Sigmoid],
                ),
                (
                    SourceOperatorKind::Multiply { width: 2048 },
                    vec![PlanPrimitive::Multiply],
                ),
                (
                    SourceOperatorKind::SiluAndMul {
                        width: geometry.intermediate_width,
                    },
                    vec![PlanPrimitive::SiluAndMul],
                ),
            ];
            for (kind, expected) in cases {
                assert_eq!(plan_primitives(kind, geometry, 17)?, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn missing_runtime_operations_and_layout_substitutions_refuse() -> Result<()> {
        let geometry = Geometry::qwen35_08();
        assert!(
            plan_primitives(
                SourceOperatorKind::Projection { selected_row: None },
                geometry,
                64,
            )?
            .is_empty()
        );
        assert!(
            plan_primitives(SourceOperatorKind::SiluAndMul { width: 6144 }, geometry, 64).is_err()
        );
        assert!(
            plan_primitives(
                SourceOperatorKind::KvAppend {
                    domain: KvDomain(3),
                    key_width: 512,
                    packed_value: TensorView {
                        offset: 4608,
                        rows: 64,
                        width: 512,
                        row_stride: 5121,
                    },
                },
                geometry,
                64,
            )
            .is_err()
        );

        let wrong_theta = SourceQkRotaryGeometry {
            attention: AttentionGeometry {
                width: 256,
                query_heads: 8,
                kv_heads: 2,
            },
            rotary_dimensions: 64,
            theta: 1_000_000.0,
            epsilon: EPSILON,
        };
        assert!(
            plan_primitives(
                SourceOperatorKind::QkRmsRotaryGated(wrong_theta),
                geometry,
                64,
            )
            .is_err()
        );
        Ok(())
    }
}
