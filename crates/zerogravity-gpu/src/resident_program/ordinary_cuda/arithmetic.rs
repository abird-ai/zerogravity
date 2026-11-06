//! Source-contract arithmetic lowering, not a cuBLAS heuristic clone.
//! Roles come from typed SSA edges, never prompts, node ordinals or captures.
use crate::cuda::bf16_gemm::{Bf16GemmReduction, Bf16GemmShape};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{ForwardPlan, PlanOperation, SourceOperatorKind, SourceRowSelection},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProjectionUse {
    ConvPacked,
    ConvCoefficients,
    AttentionPacked,
    Expansion,
    Contraction,
    Output,
    LastRow,
}

pub(super) fn projection_use(plan: &ForwardPlan, index: usize) -> Result<ProjectionUse> {
    let node = plan
        .nodes()
        .get(index)
        .ok_or_else(|| Error::State("ordinary projection node absent".into()))?;
    let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
        return Err(Error::State(
            "ordinary projection requires a source template".into(),
        ));
    };
    if !matches!(descriptor.kind(), SourceOperatorKind::Projection { .. })
        || node.inputs.len() != 1
        || node.outputs.len() != 1
    {
        return Err(Error::State(
            "ordinary projection arithmetic SSA differs".into(),
        ));
    }
    if descriptor.row_selection() == SourceRowSelection::LastActive {
        return Ok(ProjectionUse::LastRow);
    }
    let mut role = None;
    let mut declare = |candidate| -> Result<()> {
        if role.is_some_and(|previous| previous != candidate) {
            return Err(Error::Unsupported(
                "ordinary projection arithmetic uses conflict".into(),
            ));
        }
        role = Some(candidate);
        Ok(())
    };
    for other in plan.nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &other.operation else {
            continue;
        };
        if other.outputs.contains(&node.inputs[0]) {
            match descriptor.kind() {
                SourceOperatorKind::SiluAndMul { .. } => declare(ProjectionUse::Contraction)?,
                SourceOperatorKind::GatedDelta { .. } | SourceOperatorKind::Multiply { .. } => {
                    declare(ProjectionUse::Output)?
                }
                _ => {}
            }
        }
        match descriptor.kind() {
            SourceOperatorKind::CausalConvSplit { .. } => {
                if other.inputs.first() == node.outputs.first() {
                    declare(ProjectionUse::ConvPacked)?;
                }
                if other.inputs.get(1) == node.outputs.first() {
                    declare(ProjectionUse::ConvCoefficients)?;
                }
            }
            SourceOperatorKind::QkRmsRotaryGated(_) if other.inputs.contains(&node.outputs[0]) => {
                declare(ProjectionUse::AttentionPacked)?
            }
            SourceOperatorKind::SiluAndMul { .. } if other.inputs.contains(&node.outputs[0]) => {
                declare(ProjectionUse::Expansion)?
            }
            _ => {}
        }
    }
    role.ok_or_else(|| {
        Error::Unsupported("ordinary projection arithmetic use is unimplemented".into())
    })
}

impl ProjectionUse {
    pub(super) fn reduction(self, shape: Bf16GemmShape) -> Bf16GemmReduction {
        use Bf16GemmReduction::*;
        let vector = shape.rows == 1;
        let lane_sum = |terms| LaneStrided { lanes: 16, terms };
        let rounded = |parts: usize| RoundedSegments {
            terms: shape.inner.div_ceil(parts).div_ceil(64) * 64,
        };
        match self {
            Self::LastRow => lane_sum(512),
            Self::ConvCoefficients if vector => LaneBlocked {
                lanes: 32,
                terms: shape.inner.div_ceil(32),
            },
            Self::ConvCoefficients => RoundedSegments {
                terms: shape.inner.div_ceil(16).div_ceil(32) * 32,
            },
            Self::ConvPacked if vector && shape.inner < 2048 => lane_sum(512),
            Self::ConvPacked => Ordered,
            Self::AttentionPacked if vector => lane_sum(512),
            Self::AttentionPacked => InterleavedPair {
                terms: shape.inner.div_ceil(2).div_ceil(32) * 32,
            },
            Self::Expansion if vector => lane_sum(512),
            Self::Expansion => Ordered,
            Self::Contraction if vector => rounded(3),
            Self::Contraction => rounded(shape.inner.div_ceil(2048)),
            Self::Output if vector && shape.columns < shape.inner => rounded(3),
            Self::Output if vector => lane_sum(256),
            // Two K slices below three-quarter occupancy of a source 64-row tile.
            Self::Output if shape.columns < shape.inner && shape.rows < 64 * 3 / 4 => {
                InterleavedPair { terms: 32 }
            }
            Self::Output => Ordered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_reductions_preserve_partial_precision_and_k_grouping() -> Result<()> {
        use Bf16GemmReduction::*;
        for (role, columns, rows, inner, expected) in [
            (ProjectionUse::ConvPacked, 8192, 47, 1024, Ordered),
            (
                ProjectionUse::ConvCoefficients,
                32,
                47,
                1024,
                RoundedSegments { terms: 64 },
            ),
            (
                ProjectionUse::Output,
                1024,
                47,
                2048,
                InterleavedPair { terms: 32 },
            ),
            (ProjectionUse::Output, 1024, 48, 2048, Ordered),
            (ProjectionUse::Output, 1024, 49, 2048, Ordered),
            (ProjectionUse::Output, 1024, 50, 2048, Ordered),
            (ProjectionUse::Output, 1024, 64, 2048, Ordered),
            (ProjectionUse::Output, 2048, 47, 2048, Ordered),
            (
                ProjectionUse::ConvPacked,
                8192,
                1,
                1024,
                LaneStrided {
                    lanes: 16,
                    terms: 512,
                },
            ),
            (
                ProjectionUse::ConvCoefficients,
                32,
                1,
                1024,
                LaneBlocked {
                    lanes: 32,
                    terms: 32,
                },
            ),
            (
                ProjectionUse::Output,
                1024,
                1,
                2048,
                RoundedSegments { terms: 704 },
            ),
            (
                ProjectionUse::Contraction,
                1024,
                47,
                3584,
                RoundedSegments { terms: 1792 },
            ),
            (
                ProjectionUse::Contraction,
                1024,
                1,
                3584,
                RoundedSegments { terms: 1216 },
            ),
            (
                ProjectionUse::Contraction,
                2048,
                47,
                6144,
                RoundedSegments { terms: 2048 },
            ),
            (
                ProjectionUse::AttentionPacked,
                5120,
                47,
                1024,
                InterleavedPair { terms: 512 },
            ),
            (
                ProjectionUse::ConvCoefficients,
                24,
                19,
                3072,
                RoundedSegments { terms: 192 },
            ),
        ] {
            assert_eq!(
                role.reduction(Bf16GemmShape::new(columns, rows, inner)?),
                expected
            );
        }
        Ok(())
    }
}
