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
    vocabulary: u64,
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
            vocabulary,
            ids: Arc::from(ids),
        })))
    }
    pub fn source(&self) -> SourceIdentity {
        self.0.source
    }
    pub fn base(&self) -> usize {
        self.0.base
    }
    pub fn vocabulary(&self) -> u64 {
        self.0.vocabulary
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
/// Mathematical retained RMS origin only; runtime backing stays with its source owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedRmsRows {
    span: TeacherSpan,
    layout: BulkF32Layout,
    norm: SourceParameter,
    epsilon_bits: u32,
    contract: SourceNumericalContract,
}
impl RetainedRmsRows {
    pub fn new(
        span: TeacherSpan,
        layout: BulkF32Layout,
        norm: SourceParameter,
        epsilon: f32,
        contract: SourceNumericalContract,
    ) -> Result<Self> {
        let value = Self {
            span,
            layout,
            norm,
            epsilon_bits: epsilon.to_bits(),
            contract,
        };
        value.validate()?;
        Ok(value)
    }
    /// Allocation-free join to the model owner's retained mathematical metadata.
    pub fn matches_rms(
        &self,
        span: &TeacherSpan,
        layout: BulkF32Layout,
        norm: &SourceParameter,
        epsilon: f32,
        contract: SourceNumericalContract,
    ) -> bool {
        self.span == *span
            && self.layout == layout
            && self.norm == *norm
            && self.epsilon_bits == epsilon.to_bits()
            && self.contract == contract
    }
    pub fn span(&self) -> &TeacherSpan {
        &self.span
    }
    pub fn layout(&self) -> BulkF32Layout {
        self.layout
    }
    pub fn validate(&self) -> Result<()> {
        let shape = self.layout.shape();
        let width = shape[0];
        let epsilon = f32::from_bits(self.epsilon_bits);
        if self.contract != SourceNumericalContract::GgmlSourceF32V1
            || !self.layout.is_dense()
            || shape != [width, self.span.len(), 1, 1]
            || self.norm.source != self.span.source()
            || self.norm.name.is_empty()
            || self.norm.kind != QuantKind::F32
            || self.norm.shape != [width as u64]
            || width.checked_mul(4) != Some(self.norm.bytes)
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(Error::State(
                "retained RMS original source/span/layout/math contract".into(),
            ));
        }
        self.layout.validate()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BulkF32Origin {
    /// Caller-supplied pure operator input, explicitly NOT a model/state producer.
    ExternalObserved { name: String },
    /// Exact whole original raw learned matrix, no dequantization/retag/cache.
    SourceMatrix(SourceTensor),
    /// Actual backing must retain this same normalized prefix issuer.
    RetainedNormalized { binding: RetainedRmsRows },
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
    pub fn retained_normalized(binding: RetainedRmsRows) -> Result<Self> {
        binding.validate()?;
        let value = Self {
            span: binding.span().clone(),
            side: BulkF32Side::Right,
            layout: binding.layout(),
            origin: BulkF32Origin::RetainedNormalized { binding },
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
            BulkF32Origin::RetainedNormalized { .. } => "source.normalized_rows",
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
        if let BulkF32Origin::SourceMatrix(tensor) = &self.origin
            && (self.side != BulkF32Side::Left
                || tensor.source != self.span.source()
                || tensor.kind != QuantKind::F32
                || tensor.axes != MatrixAxes::InputOutput
                || self.layout != BulkF32Layout::dense([tensor.columns, tensor.rows, 1, 1])?)
        {
            return Err(Error::State("bulk whole learned descriptor/axes".into()));
        }
        if let BulkF32Origin::RetainedNormalized { binding } = &self.origin {
            binding.validate()?;
            if self.side != BulkF32Side::Right
                || binding.span() != &self.span
                || binding.layout() != self.layout
            {
                return Err(Error::State(
                    "bulk retained normalized issuer/layout".into(),
                ));
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
            matrix: Box::new(matrix.clone()),
        },
        &[*left, *right],
        ValueShape::Vector(matrix.output.backing_words()),
        None,
    )
}

#[cfg(test)]
mod tests;
