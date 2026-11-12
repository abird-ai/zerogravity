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

/// Structural compatibility classes for sharing all-row prefill projections.
///
/// Classes are derived from the authenticated static plan's projection roles,
/// matrix geometry and exact reduction schedules. They contain no model-family
/// thresholds, scheduler policy, session/root identity or launch authority.
#[derive(Clone, Debug)]
pub(crate) struct PrefillProjectionClasses {
    classes: Box<[usize]>,
}

impl PrefillProjectionClasses {
    pub(crate) fn from_plan(plan: &ForwardPlan) -> Result<Self> {
        let geometry = plan.source_plan_geometry().ok_or_else(|| {
            Error::State("prefill projection classes require source template".into())
        })?;
        let max_rows = geometry.max_rows();
        if max_rows == 0 {
            return Err(Error::Limit("prefill projection class row bound".into()));
        }
        let mut signatures: Vec<Vec<Bf16GemmReduction>> = Vec::new();
        let mut classes = vec![0usize; max_rows + 1];
        for (rows, slot) in classes.iter_mut().enumerate().take(max_rows + 1).skip(1) {
            let signature = prefill_reduction_signature(plan, rows)?;
            let class = if let Some(index) = signatures.iter().position(|known| known == &signature)
            {
                index
            } else {
                signatures.push(signature);
                signatures.len() - 1
            };
            *slot = class + 1;
        }
        Ok(Self {
            classes: classes.into_boxed_slice(),
        })
    }

    pub(crate) fn compatible(&self, left: usize, right: usize) -> bool {
        self.classes
            .get(left)
            .copied()
            .is_some_and(|class| class != 0 && self.classes.get(right).copied() == Some(class))
    }

    #[cfg(test)]
    pub(crate) fn class(&self, rows: usize) -> Option<usize> {
        self.classes.get(rows).copied().filter(|class| *class != 0)
    }
}

fn prefill_reduction_signature(plan: &ForwardPlan, rows: usize) -> Result<Vec<Bf16GemmReduction>> {
    let geometry = plan.source_plan_geometry().ok_or_else(|| {
        Error::State("prefill reduction signature requires source template".into())
    })?;
    if rows == 0 || rows > geometry.max_rows() {
        return Err(Error::Malformed(
            "prefill reduction signature row extent".into(),
        ));
    }
    let mut signature = Vec::new();
    for (index, node) in plan.nodes().iter().enumerate() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            continue;
        };
        if !matches!(descriptor.kind(), SourceOperatorKind::Projection { .. })
            || descriptor.row_selection() != SourceRowSelection::AllActive
        {
            continue;
        }
        let matrices = descriptor.matrices();
        let first = matrices
            .first()
            .ok_or_else(|| Error::State("prefill projection class matrix set absent".into()))?;
        let inner = first.columns;
        let columns = matrices.iter().try_fold(0usize, |sum, matrix| {
            if matrix.columns != inner {
                return Err(Error::State(
                    "prefill projection class inner geometry differs".into(),
                ));
            }
            sum.checked_add(matrix.rows)
                .ok_or_else(|| Error::Limit("prefill projection class rows overflow".into()))
        })?;
        let shape = Bf16GemmShape::new(columns, rows, inner)?;
        signature.push(projection_use(plan, index)?.reduction(shape));
    }
    Ok(signature)
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
    use zerogravity_model_qwen35::plan::{QwenSourceProfile, diagnostic_ordinary_plan};

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

    #[test]
    fn prefill_projection_classes_follow_complete_static_reduction_signatures() -> Result<()> {
        for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
            let plan = diagnostic_ordinary_plan(profile, 1024)?;
            let classes = PrefillProjectionClasses::from_plan(&plan)?;
            assert!(classes.compatible(48, 49));
            assert!(classes.compatible(63, 64));
            assert!(!classes.compatible(0, 1));
            assert!(!classes.compatible(64, 65));
            assert_eq!(
                classes.compatible(47, 48),
                prefill_reduction_signature(&plan, 47)? == prefill_reduction_signature(&plan, 48)?
            );
            assert!(classes.class(48).is_some());
        }

        // Vector rows are a genuine arithmetic class of their own in the
        // current 2B plan. The classifier discovers that boundary from the
        // complete plan; importantly, it does *not* invent a 47/48 boundary
        // from an isolated projection geometry that is absent here.
        let plan = diagnostic_ordinary_plan(QwenSourceProfile::Qwen2B, 1024)?;
        assert_ne!(
            prefill_reduction_signature(&plan, 1)?,
            prefill_reduction_signature(&plan, 2)?
        );
        let classes = PrefillProjectionClasses::from_plan(&plan)?;
        assert!(!classes.compatible(1, 2));
        assert!(classes.compatible(47, 48));
        Ok(())
    }
}
