use super::*;
use zerogravity_core::accelerator::program::{
    ActivationPrecision, EmbeddingContract, MatrixAxes, OrdinaryRmsContract, ParameterSlot,
    ProgramNumericalProfile, ProjectionContract, SourceParameter, SourceTensor,
};

pub(super) const PROFILE: ProgramNumericalProfile =
    ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
#[derive(Clone)]
pub(super) struct Shape {
    pub embedding: SourceTensor,
    pub gamma: SourceParameter,
    pub weights: SourceTensor,
    pub width: usize,
    pub rows: usize,
}
impl Shape {
    pub fn bind(
        slice: &RecordedOperatorSlice,
        source: &VerifiedTensorSource,
        token: u32,
    ) -> Result<Self> {
        let p = slice.plan();
        if !p.is_operator_slice()
            || p.source() != source.source_identity()
            || p.numerical_profile() != PROFILE
            || p.frontier_delta() != 0
            || !p.required_domains().is_empty()
            || !p.proposed_domains().is_empty()
            || !p.state_nodes().is_empty()
            || p.logits().is_some()
            || !p.constants().is_empty()
            || p.nodes().len() != 3
            || p.values().len() != 3
            || p.precisions() != [ActivationPrecision::Bf16WidenedF32; 3]
        {
            return Err(Error::Unsupported(
                "portable exact three-node state-free CPU BF16 slice".into(),
            ));
        }
        let [a, b, c] = p.nodes() else { unreachable!() };
        let PlanOperation::TypedEmbed {
            tensor: embedding,
            token: slot,
            contract: EmbeddingContract::Bf16OutputInput,
        } = &a.operation
        else {
            return Err(Error::Unsupported(
                "portable typed embedding contract".into(),
            ));
        };
        let PlanOperation::TypedOrdinaryRms {
            parameter: gamma,
            contract: OrdinaryRmsContract::TorchCpuBf16Avx2,
        } = &b.operation
        else {
            return Err(Error::Unsupported(
                "portable typed ordinary RMS contract".into(),
            ));
        };
        let PlanOperation::TypedProject {
            tensor: weights,
            contract: ProjectionContract::TorchCpuBf16Avx2,
        } = &c.operation
        else {
            return Err(Error::Unsupported(
                "portable typed projection contract".into(),
            ));
        };
        let ([x], [n], [y]) = (
            a.outputs.as_slice(),
            b.outputs.as_slice(),
            c.outputs.as_slice(),
        ) else {
            return Err(Error::State("portable typed output arity".into()));
        };
        if !a.inputs.is_empty()
            || b.inputs != [*x]
            || c.inputs != [*n]
            || [a, b, c].iter().any(|n| n.state.is_some())
            || slot.0 != ParameterSlot::Token
            || slice.output() != *y
            || [x.0, n.0, y.0] != [0, 1, 2]
            || embedding.columns != weights.columns
            || !matches!(weights.columns, 1024 | 2048)
            || weights.rows == 0
            || weights.rows > 16384
            || weights
                .rows
                .checked_mul(weights.columns)
                .and_then(|n| n.checked_mul(2))
                .is_none_or(|n| n > 32 * 1024 * 1024)
            || token as usize >= embedding.rows
            || gamma.kind != QuantKind::Bf16
            || gamma.source != p.source()
            || gamma.shape != [weights.columns as u64]
            || gamma.bytes != weights.columns * 2
        {
            return Err(Error::Unsupported(
                "portable exact source/SSA/geometry/precision bound".into(),
            ));
        }
        for t in [embedding, weights] {
            let actual = source.matrix(&t.name, MatrixAxes::OutputInput)?;
            if t.source != p.source()
                || t.kind != QuantKind::Bf16
                || t.axes != MatrixAxes::OutputInput
                || t.rows != actual.rows()
                || t.columns != actual.columns()
                || actual.tensor().kind != QuantKind::Bf16
            {
                return Err(Error::State(
                    "portable whole original matrix descriptor".into(),
                ));
            }
        }
        source
            .parameter(&gamma.name)?
            .validate_source_parameter(gamma)?;
        for (id, elements) in [
            (*x, weights.columns),
            (*n, weights.columns),
            (*y, weights.rows),
        ] {
            if p.values()[id.0 as usize].capacity_elements()? != elements {
                return Err(Error::State("portable typed value extent".into()));
            }
        }
        Ok(Self {
            embedding: embedding.clone(),
            gamma: gamma.clone(),
            weights: weights.clone(),
            width: weights.columns,
            rows: weights.rows,
        })
    }
    pub fn layout(&self) -> Vec<(Buffer, usize)> {
        use Buffer::*;
        let h = self.width;
        let r = self.rows;
        vec![
            (Embedding, 2 * h),
            (GammaRaw, 2 * h),
            (Zero, 4),
            (Input, 4 * h),
            (Gamma, 4 * h),
            (RuntimeGamma, 4 * h),
            (Squares, 4 * h),
            (Sum, 4),
            (Mean, 4),
            (Denominator, 4),
            (Sqrt, 4),
            (Inverse, 4),
            (Normalized, 4 * h),
            (Weighted, 4 * h),
            (Rms, 4 * h),
            (PackedRms, 2 * h),
            (Accumulation, 4 * r),
            (Projection, 4 * r),
            (ProjectionStatus, 4 * r),
            (Status, 4),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Buffer {
    Embedding,
    GammaRaw,
    Zero,
    Input,
    Gamma,
    RuntimeGamma,
    Squares,
    Sum,
    Mean,
    Denominator,
    Sqrt,
    Inverse,
    Normalized,
    Weighted,
    Rms,
    PackedRms,
    Accumulation,
    Projection,
    ProjectionStatus,
    Status,
}

pub(super) fn same_node(a: &PlanNode, b: &PlanNode) -> bool {
    if a.inputs != b.inputs || a.outputs != b.outputs || a.state != b.state {
        return false;
    }
    match (&a.operation, &b.operation) {
        (
            PlanOperation::TypedEmbed {
                tensor: a,
                token: ta,
                contract: ca,
            },
            PlanOperation::TypedEmbed {
                tensor: b,
                token: tb,
                contract: cb,
            },
        ) => a == b && ta == tb && ca == cb,
        (
            PlanOperation::TypedOrdinaryRms {
                parameter: a,
                contract: ca,
            },
            PlanOperation::TypedOrdinaryRms {
                parameter: b,
                contract: cb,
            },
        ) => a == b && ca == cb,
        (
            PlanOperation::TypedProject {
                tensor: a,
                contract: ca,
            },
            PlanOperation::TypedProject {
                tensor: b,
                contract: cb,
            },
        ) => a == b && ca == cb,
        _ => false,
    }
}
