//! Reserved source-row embedding and dual BF16 RMS recording only.
//! No backend, model eligibility, physical allocation or StateRoot authority.
use super::*;

const WIDTH: usize = 1024;
const VOCABULARY: usize = 248_320;
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const POST_NORM: &str = "model.language_model.layers.0.post_attention_layernorm.weight";
const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualRmsContract {
    Vllm031CudaAtenBf16H1024Rows1_2_64,
    Vllm031CudaGdnPrefix(GdnPrefixRmsBinding),
}
impl ResidualRmsContract {
    /// Raw source gamma. Actual backing must be revalidated at consumption.
    pub fn validate(self, parameter: &SourceParameter, elements: usize) -> Result<usize> {
        if let Self::Vllm031CudaGdnPrefix(binding) = self {
            return binding.validate(parameter, elements);
        }
        if !parameter.source.is_bound() || parameter.name != POST_NORM {
            return Err(Error::State(
                "residual RMS original postnorm source/name".into(),
            ));
        }
        let bytes = match parameter.kind {
            QuantKind::Bf16 => WIDTH * 2,
            QuantKind::F32 => WIDTH * 4,
            _ => return Err(Error::Unsupported("residual RMS raw BF16/F32 gamma".into())),
        };
        if parameter.shape.as_slice() != [WIDTH as u64] || parameter.bytes != bytes {
            return Err(Error::Malformed("residual RMS raw gamma1024 extent".into()));
        }
        if !elements.is_multiple_of(WIDTH) || !matches!(elements / WIDTH, 1 | 2 | 64) {
            return Err(Error::Unsupported(
                "residual RMS fixed dense rows1/2/64".into(),
            ));
        }
        Ok(elements / WIDTH)
    }
}

#[derive(Clone)]
pub struct ResidualRmsValues<V> {
    pub normalized: V,
    pub residual: V,
}

/// New-profile metadata only. Existing TeacherSpan remains unchanged for2/64;
/// one teacher has its own immutable issuer, not a legacy-bulk capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VllmBf16Teachers(TeacherRows);
#[derive(Clone, Debug)]
enum TeacherRows {
    One(Arc<SingleTeacherRow>),
    Many(TeacherSpan),
    ShortHf(ShortHfSpan),
}
#[derive(Debug)]
struct SingleTeacherRow {
    source: SourceIdentity,
    base: usize,
    capacity: usize,
    ids: [u32; 1],
}
impl PartialEq for TeacherRows {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::One(a), Self::One(b)) => Arc::ptr_eq(a, b),
            (Self::Many(a), Self::Many(b)) => a == b,
            (Self::ShortHf(a), Self::ShortHf(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for TeacherRows {}
impl VllmBf16Teachers {
    pub(super) fn from_short_hf_span(span: ShortHfSpan) -> Self {
        Self(TeacherRows::ShortHf(span))
    }
    /// Separate semantic issuer; never a legacy/native phase capability.
    pub fn short_hf_span(&self) -> Option<&ShortHfSpan> {
        match &self.0 {
            TeacherRows::ShortHf(span) => Some(span),
            _ => None,
        }
    }
    /// Adopt the SAME bounded issuer; no new native phase or current-root permit.
    pub fn from_span(span: TeacherSpan) -> Result<Self> {
        if !span.source().is_bound()
            || !(2..=64).contains(&span.len())
            || span.capacity() == 0
            || span.capacity() > 8448
            || span
                .base()
                .checked_add(span.len())
                .is_none_or(|end| end > span.capacity())
            || span.ids().iter().any(|&id| id as usize >= VOCABULARY)
        {
            return Err(Error::Malformed(
                "vLLM adopted teacher span/source/IDs".into(),
            ));
        }
        Ok(Self(TeacherRows::Many(span)))
    }
    pub fn new(source: SourceIdentity, base: usize, capacity: usize, ids: &[u32]) -> Result<Self> {
        let rows = match ids {
            &[id]
                if source.is_bound()
                    && capacity > 0
                    && capacity <= 8448
                    && base.checked_add(1).is_some_and(|end| end <= capacity)
                    && (id as usize) < VOCABULARY =>
            {
                TeacherRows::One(Arc::new(SingleTeacherRow {
                    source,
                    base,
                    capacity,
                    ids: [id],
                }))
            }
            _ if matches!(ids.len(), 2 | 64) => TeacherRows::Many(TeacherSpan::new(
                source,
                base,
                capacity,
                VOCABULARY as u64,
                ids,
            )?),
            _ => return Err(Error::Malformed("closed vLLM teacher rows1/2/64".into())),
        };
        Ok(Self(rows))
    }
    pub fn source(&self) -> SourceIdentity {
        match &self.0 {
            TeacherRows::One(row) => row.source,
            TeacherRows::Many(span) => span.source(),
            TeacherRows::ShortHf(span) => span.source(),
        }
    }
    pub fn base(&self) -> usize {
        match &self.0 {
            TeacherRows::One(row) => row.base,
            TeacherRows::Many(span) => span.base(),
            TeacherRows::ShortHf(span) => span.base(),
        }
    }
    pub fn capacity(&self) -> usize {
        match &self.0 {
            TeacherRows::One(row) => row.capacity,
            TeacherRows::Many(span) => span.capacity(),
            TeacherRows::ShortHf(span) => span.capacity(),
        }
    }
    pub fn ids(&self) -> &[u32] {
        match &self.0 {
            TeacherRows::One(row) => &row.ids,
            TeacherRows::Many(span) => span.ids(),
            TeacherRows::ShortHf(span) => span.ids(),
        }
    }
    pub fn len(&self) -> usize {
        self.ids().len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Full original embedding descriptor and immutable teacher issuer.
/// It contains neither captured activations nor an allocation/route permit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VllmBf16Embedding {
    tensor: Arc<SourceTensor>,
    teachers: VllmBf16Teachers,
    model_profile: FullModelProfile,
}
impl VllmBf16Embedding {
    pub(super) fn bind_short_hf_teachers(&mut self, teachers: VllmBf16Teachers) {
        self.teachers = teachers;
    }
    pub fn new(tensor: SourceTensor, teachers: VllmBf16Teachers) -> Result<Self> {
        Self::for_model(FullModelProfile::Qwen08, tensor, teachers)
    }
    /// Full-model metadata only; recording requires a matching full-model call.
    pub fn for_model(
        model_profile: FullModelProfile,
        tensor: SourceTensor,
        teachers: VllmBf16Teachers,
    ) -> Result<Self> {
        if teachers.short_hf_span().is_some() || !matches!(teachers.len(), 1 | 2 | 64) {
            return Err(Error::Unsupported(
                "legacy source-row embedding rows1/2/64".into(),
            ));
        }
        let value = Self {
            tensor: Arc::new(tensor),
            teachers,
            model_profile,
        };
        value.validate()?;
        Ok(value)
    }
    /// Full-model semantic metadata; builder must join this exact call issuer.
    pub fn for_call(tensor: SourceTensor, call: &Vllm08Call) -> Result<Self> {
        call.validate()?;
        let value = Self {
            tensor: Arc::new(tensor),
            teachers: call.request().teachers().clone(),
            model_profile: call.model_profile(),
        };
        value.validate()?;
        if value.tensor.source != call.source() {
            return Err(Error::State(
                "source-row embedding differs from call source".into(),
            ));
        }
        Ok(value)
    }
    pub fn model_profile(&self) -> FullModelProfile {
        self.model_profile
    }
    pub fn tensor(&self) -> &SourceTensor {
        &self.tensor
    }
    pub fn teachers(&self) -> &VllmBf16Teachers {
        &self.teachers
    }
    pub fn rows(&self) -> usize {
        self.teachers.len()
    }
    pub fn validate(&self) -> Result<()> {
        if !self.tensor.source.is_bound()
            || self.tensor.source != self.teachers.source()
            || self.tensor.name != EMBEDDING
            || self.tensor.kind != QuantKind::Bf16
            || self.tensor.axes != MatrixAxes::OutputInput
            || self.tensor.rows != VOCABULARY
            || self.tensor.columns != self.model_profile.hidden_size()
            || if let Some(span) = self.teachers.short_hf_span() {
                span.validate().is_err()
            } else {
                !(1..=64).contains(&self.rows())
            }
            || self
                .teachers
                .ids()
                .iter()
                .any(|&id| id as usize >= VOCABULARY)
        {
            return Err(Error::Unsupported(
                "closed original BF16 source-row embedding".into(),
            ));
        }
        Ok(())
    }
    /// Descriptor metadata alone cannot authenticate the immutable backing.
    pub fn validate_source(&self, source: &dyn TensorSource) -> Result<()> {
        self.validate()?;
        self.tensor.validate_source(source)
    }
    fn shape(&self) -> ValueShape {
        ValueShape::Vector(self.rows() * self.model_profile.hidden_size())
    }
}

fn check_builder(builder: &PlanBuilder) -> Result<()> {
    if !(builder.operator_slice || builder.vllm_fragment.is_some()) || builder.profile != PROFILE {
        return Err(Error::Unsupported(
            "reserved vLLM contract requires operator slice".into(),
        ));
    }
    Ok(())
}
pub(super) fn embedding_precision(
    builder: &PlanBuilder,
    descriptor: &VllmBf16Embedding,
) -> Result<ActivationPrecision> {
    check_builder(builder)?;
    descriptor.validate()?;
    if descriptor.teachers.short_hf_span().is_some() && builder.vllm_model.is_none() {
        return Err(Error::Unsupported(
            "short-HF embedding requires its exact full-model call".into(),
        ));
    }
    if builder.vllm_model.as_ref().is_some_and(|call| {
        call.request().teachers() != descriptor.teachers()
            || call.source() != descriptor.teachers().source()
    }) {
        return Err(Error::State(
            "full-model embedding teacher issuer differs from builder".into(),
        ));
    }
    if builder
        .vllm_model
        .as_ref()
        .is_some_and(|call| call.model_profile() != descriptor.model_profile)
        || descriptor.model_profile == FullModelProfile::Qwen2B && builder.vllm_model.is_none()
    {
        return Err(Error::Unsupported(
            "source embedding/full-model geometry profile".into(),
        ));
    }
    if descriptor.tensor.source != builder.source {
        return Err(Error::State(
            "vLLM source-row embedding builder issuer".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn result_precision(
    builder: &PlanBuilder,
    parameter: &SourceParameter,
    contract: ResidualRmsContract,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    check_builder(builder)?;
    let [branch, residual] = inputs else {
        return Err(Error::Malformed("residual RMS two inputs".into()));
    };
    if parameter.source != builder.source
        || branch.id == residual.id
        || branch.precision != ActivationPrecision::Bf16WidenedF32
        || residual.precision != ActivationPrecision::Bf16WidenedF32
        || branch.shape != ValueShape::Vector(branch.elements)
        || residual.shape != branch.shape
        || residual.elements != branch.elements
    {
        return Err(Error::Unsupported(
            "residual RMS source/distinct BF16 dense inputs".into(),
        ));
    }
    contract.validate(parameter, branch.elements)?;
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn record_embedding(
    builder: &mut PlanBuilder,
    descriptor: &VllmBf16Embedding,
) -> Result<PlanValue> {
    embedding_precision(builder, descriptor)?;
    builder.source_tensor(descriptor.tensor())?;
    builder.one(
        "vllm.original.embedding.rows",
        PlanOperation::TypedVllmBf16Embedding {
            descriptor: Box::new(descriptor.clone()),
        },
        &[],
        descriptor.shape(),
        None,
    )
}
pub(super) fn record(
    builder: &mut PlanBuilder,
    parameter: SourceParameter,
    branch: &PlanValue,
    residual: &PlanValue,
    contract: ResidualRmsContract,
) -> Result<ResidualRmsValues<PlanValue>> {
    contract.validate(&parameter, branch.elements)?;
    let name = parameter.name.clone();
    let values = builder.record(
        &name,
        PlanOperation::TypedResidualRms {
            parameter,
            contract,
        },
        &[*branch, *residual],
        &[branch.shape, branch.shape],
        None,
    )?;
    let [normalized, residual]: [PlanValue; 2] = values
        .try_into()
        .map_err(|_| Error::State("residual RMS output arity".into()))?;
    Ok(ResidualRmsValues {
        normalized,
        residual,
    })
}

#[cfg(test)]
#[path = "residual_rms/tests.rs"]
mod tests;
