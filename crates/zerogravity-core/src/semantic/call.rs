//! Implicit text inputs made explicit without a second value graph.
use crate::{
    Error, Result,
    accelerator::program::{PlanNode, PlanOperation, SourceOperatorKind},
};

/// Invocation row count and phase are always explicit in BoundTextCall.
/// These additional dependencies identify actual logical token/position
/// inputs, not a mandatory host upload or a new physical buffer.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InvocationPorts {
    pub token_ids: bool,
    /// Contiguous absolute text positions, including prior/frontier/capacity.
    pub positions: bool,
}

impl InvocationPorts {
    pub(super) fn for_operator(kind: SourceOperatorKind) -> Result<Self> {
        use SourceOperatorKind::*;
        Ok(match kind {
            EmbeddingRows => Self {
                token_ids: true,
                positions: false,
            },
            QkRmsRotaryGated(_)
            | CausalConvSplit { .. }
            | GatedDelta { .. }
            | KvAppend { .. }
            | Attention { .. } => Self {
                token_ids: false,
                positions: true,
            },
            Projection { selected_row: None }
            | Rms(_)
            | ResidualRms(_)
            | Sigmoid { .. }
            | Multiply { .. }
            | SiluAndMul { .. } => Self::default(),
            _ => {
                return Err(Error::Unsupported(
                    "region operation lacks a supported input/effect contract".into(),
                ));
            }
        })
    }

    pub(super) fn include(&mut self, other: Self) {
        self.token_ids |= other.token_ids;
        self.positions |= other.positions;
    }

    pub(crate) fn for_nodes(nodes: &[PlanNode]) -> Result<Self> {
        let mut ports = Self::default();
        for node in nodes {
            let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
                return Err(Error::Unsupported(
                    "text invocation ports require a source template".into(),
                ));
            };
            ports.include(Self::for_operator(descriptor.kind())?);
        }
        Ok(ports)
    }
}
