//! Reserved source-row embedding and dual BF16 RMS recording only.
//! No backend, model eligibility, physical allocation or StateRoot authority.
use super::*;

const WIDTH: usize = 1024;
const VOCABULARY: usize = 248_320;
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const POST_NORM: &str = "model.language_model.layers.0.post_attention_layernorm.weight";
const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::SourceCudaSm86Bf16V1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenResidualRmsContract {
    ReferenceAtenBf16H1024Rows1_2_64,
    GdnPrefix(QwenPrefixRmsBinding),
}
impl QwenResidualRmsContract {
    /// Raw source gamma. Actual backing must be revalidated at consumption.
    pub fn validate(self, parameter: &SourceParameter, elements: usize) -> Result<usize> {
        if let Self::GdnPrefix(binding) = self {
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
pub struct QwenInputRows {
    issuer: TeacherRows,
    rows: SourceRows,
}
#[derive(Clone, Debug)]
enum TeacherRows {
    One(Arc<SingleTeacherRow>),
    Many(TeacherSpan),
    Prompt(PromptSpan),
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
            (Self::Prompt(a), Self::Prompt(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for TeacherRows {}
impl QwenInputRows {
    pub(super) fn from_prompt_span(span: PromptSpan) -> Result<Self> {
        Self::from_issuer(TeacherRows::Prompt(span))
    }
    fn from_issuer(issuer: TeacherRows) -> Result<Self> {
        let (source, base, capacity, ids): (_, _, _, &[u32]) = match &issuer {
            TeacherRows::One(row) => (row.source, row.base, row.capacity, &row.ids),
            TeacherRows::Many(span) => (span.source(), span.base(), span.capacity(), span.ids()),
            TeacherRows::Prompt(span) => (span.source(), span.base(), span.capacity(), span.ids()),
        };
        let rows = match &issuer {
            TeacherRows::Many(span) => SourceRows::from_teacher_span(span.clone(), VOCABULARY)?,
            TeacherRows::Prompt(span) => span.source_rows(),
            TeacherRows::One(_) => SourceRows::new(
                TextCallGeometry::new(source, VOCABULARY, capacity)?,
                base,
                Arc::from(ids),
            )?,
        };
        Ok(Self { issuer, rows })
    }
    pub fn source_call(&self, mode: SourceCallMode) -> Result<SourceCall> {
        SourceCall::new(self.rows.clone(), mode)
    }
    /// Separate semantic issuer; never a legacy/native phase capability.
    pub fn prompt_span(&self) -> Option<&PromptSpan> {
        match &self.issuer {
            TeacherRows::Prompt(span) => Some(span),
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
                "reference CUDA adopted teacher span/source/IDs".into(),
            ));
        }
        Self::from_issuer(TeacherRows::Many(span))
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
            _ => {
                return Err(Error::Malformed(
                    "closed reference CUDA teacher rows1/2/64".into(),
                ));
            }
        };
        Self::from_issuer(rows)
    }
    pub fn source(&self) -> SourceIdentity {
        match &self.issuer {
            TeacherRows::One(row) => row.source,
            TeacherRows::Many(span) => span.source(),
            TeacherRows::Prompt(span) => span.source(),
        }
    }
    pub fn base(&self) -> usize {
        match &self.issuer {
            TeacherRows::One(row) => row.base,
            TeacherRows::Many(span) => span.base(),
            TeacherRows::Prompt(span) => span.base(),
        }
    }
    pub fn capacity(&self) -> usize {
        match &self.issuer {
            TeacherRows::One(row) => row.capacity,
            TeacherRows::Many(span) => span.capacity(),
            TeacherRows::Prompt(span) => span.capacity(),
        }
    }
    pub fn ids(&self) -> &[u32] {
        match &self.issuer {
            TeacherRows::One(row) => &row.ids,
            TeacherRows::Many(span) => span.ids(),
            TeacherRows::Prompt(span) => span.ids(),
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
pub struct QwenEmbedding {
    tensor: Arc<SourceTensor>,
    teachers: QwenInputRows,
    model_profile: QwenSourceProfile,
}
impl QwenEmbedding {
    pub fn new(tensor: SourceTensor, teachers: QwenInputRows) -> Result<Self> {
        Self::for_model(QwenSourceProfile::Qwen08, tensor, teachers)
    }
    /// Full-model metadata only; recording requires a matching full-model call.
    pub fn for_model(
        model_profile: QwenSourceProfile,
        tensor: SourceTensor,
        teachers: QwenInputRows,
    ) -> Result<Self> {
        if teachers.prompt_span().is_some() || !matches!(teachers.len(), 1 | 2 | 64) {
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
    pub fn for_call(tensor: SourceTensor, call: &QwenSourceCall) -> Result<Self> {
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
    pub fn model_profile(&self) -> QwenSourceProfile {
        self.model_profile
    }
    pub fn tensor(&self) -> &SourceTensor {
        &self.tensor
    }
    pub fn teachers(&self) -> &QwenInputRows {
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
            || if let Some(span) = self.teachers.prompt_span() {
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
}

fn check_builder(builder: &QwenPlanBuilder) -> Result<()> {
    if !(builder.operator_slice || builder.fragment_call.is_some()) || builder.profile != PROFILE {
        return Err(Error::Unsupported(
            "reserved reference CUDA contract requires operator slice".into(),
        ));
    }
    Ok(())
}
pub(super) fn embedding_precision(
    builder: &QwenPlanBuilder,
    descriptor: &QwenEmbedding,
) -> Result<ActivationPrecision> {
    check_builder(builder)?;
    descriptor.validate()?;
    if descriptor.teachers.prompt_span().is_some() && builder.model_call.is_none() {
        return Err(Error::Unsupported(
            "bounded prompt embedding requires its exact full-model call".into(),
        ));
    }
    if builder.model_call.as_ref().is_some_and(|call| {
        call.request().teachers() != descriptor.teachers()
            || call.source() != descriptor.teachers().source()
    }) {
        return Err(Error::State(
            "full-model embedding teacher issuer differs from builder".into(),
        ));
    }
    if builder
        .fragment_call
        .as_ref()
        .is_some_and(|call| call.teachers() != descriptor.teachers())
    {
        return Err(Error::State(
            "fragment embedding teacher issuer differs from builder".into(),
        ));
    }
    if builder
        .model_call
        .as_ref()
        .is_some_and(|call| call.model_profile() != descriptor.model_profile)
        || descriptor.model_profile == QwenSourceProfile::Qwen2B && builder.model_call.is_none()
    {
        return Err(Error::Unsupported(
            "source embedding/full-model geometry profile".into(),
        ));
    }
    if descriptor.tensor.source != builder.source {
        return Err(Error::State(
            "reference CUDA source-row embedding builder issuer".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn result_precision(
    builder: &QwenPlanBuilder,
    parameter: &SourceParameter,
    contract: QwenResidualRmsContract,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    check_builder(builder)?;
    let [branch, residual] = inputs else {
        return Err(Error::Malformed("residual RMS two inputs".into()));
    };
    for input in inputs {
        builder.core.validate_value(input)?;
        if !input.is_contiguous() {
            return Err(Error::State(
                "plan operator requires explicit contiguous input".into(),
            ));
        }
    }
    if parameter.source != builder.source
        || branch.id == residual.id
        || branch.precision() != ActivationPrecision::Bf16WidenedF32
        || residual.precision() != ActivationPrecision::Bf16WidenedF32
        || branch.shape != ValueShape::Vector(branch.elements())
        || residual.shape != branch.shape
        || residual.elements() != branch.elements()
    {
        return Err(Error::Unsupported(
            "residual RMS source/distinct BF16 dense inputs".into(),
        ));
    }
    contract.validate(parameter, branch.elements())?;
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn record_embedding(
    builder: &mut QwenPlanBuilder,
    descriptor: &QwenEmbedding,
) -> Result<PlanValue> {
    embedding_precision(builder, descriptor)?;
    let tag =
        gdn_projection::QwenValueProvenance::embedding(builder, descriptor.teachers().clone())?;
    QwenPlanBuilder::one_output(builder.record_source(
        "qwen.source.embedding.rows",
        builder.embedding_operator(descriptor)?,
        &[],
        vec![tag],
    )?)
}
pub(super) fn record(
    builder: &mut QwenPlanBuilder,
    parameter: SourceParameter,
    branch: &PlanValue,
    residual: &PlanValue,
    contract: QwenResidualRmsContract,
) -> Result<ResidualRmsValues<PlanValue>> {
    let inputs = [*branch, *residual];
    result_precision(builder, &parameter, contract, &inputs)?;
    let tag = gdn_projection::same_input_provenance(builder, &inputs)?;
    let origins = match contract {
        QwenResidualRmsContract::GdnPrefix(binding) => [
            QwenValueOrigin::ResidualNormalized { binding },
            QwenValueOrigin::CarriedResidual { binding },
        ],
        _ => [
            QwenValueOrigin::LegacyResidualNormalized,
            QwenValueOrigin::LegacyCarriedResidual,
        ],
    };
    let tags = origins
        .into_iter()
        .map(|origin| tag.clone().with_origin(origin))
        .collect();
    let name = parameter.name.clone();
    let values = builder.record_source(
        &name,
        builder.residual_operator(&parameter, &inputs)?,
        &inputs,
        tags,
    )?;
    let [normalized, residual] = values
        .try_into()
        .map_err(|_| Error::State("Qwen dual residual RMS outputs".into()))?;
    Ok(ResidualRmsValues {
        normalized,
        residual,
    })
}

#[cfg(test)]
#[path = "rms/tests.rs"]
mod tests;
