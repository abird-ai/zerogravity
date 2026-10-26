//! Closed .8B full-layer metadata inside the original mathematical engine.
//! No allocation, runtime issuer, frontier advance or RootReady capability.
use super::*;

/// Original request issuer; selecting one output row never replaces this call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenSourceCall {
    request: QwenGdnCall,
    model_profile: QwenSourceProfile,
}
impl QwenSourceCall {
    pub fn new(request: QwenGdnCall) -> Result<Self> {
        Self::for_profile(QwenSourceProfile::Qwen08, request)
    }
    /// Geometry metadata only; the original source owner must bind the profile.
    pub fn for_profile(model_profile: QwenSourceProfile, request: QwenGdnCall) -> Result<Self> {
        request.reference_phase()?;
        Self::from_request(model_profile, request)
    }
    /// Semantic .8B span recipe only; no finite native/runtime coverage.
    pub fn from_span(request: QwenGdnCall) -> Result<Self> {
        if request.teachers().prompt_span().is_some() {
            return Err(Error::Unsupported(
                "bounded prompt issuer requires its source-profile call".into(),
            ));
        }
        Self::from_request(QwenSourceProfile::Qwen08, request)
    }
    pub(super) fn from_request(
        model_profile: QwenSourceProfile,
        request: QwenGdnCall,
    ) -> Result<Self> {
        request.validate()?;
        if request.layer().index() != 0 || request.teachers().capacity() != 512 {
            return Err(Error::Unsupported(
                "closed .8B full-model request metadata".into(),
            ));
        }
        Ok(Self {
            request,
            model_profile,
        })
    }
    pub fn model_profile(&self) -> QwenSourceProfile {
        self.model_profile
    }
    pub fn request(&self) -> &QwenGdnCall {
        &self.request
    }
    pub fn source(&self) -> SourceIdentity {
        self.request.layer().source()
    }
    pub fn rows(&self) -> usize {
        self.request.rows()
    }
    pub fn layer(&self, index: usize) -> Result<QwenLayer> {
        if index >= 24 {
            return Err(Error::Unsupported("closed original24 layer index".into()));
        }
        Ok(QwenLayer {
            source: self.source(),
            index: index as u8,
        })
    }
    pub fn validate(&self) -> Result<()> {
        self.request.validate()?;
        if self.request.layer().index() != 0 || self.request.teachers().capacity() != 512 {
            return Err(Error::State("original full-model request changed".into()));
        }
        Ok(())
    }
}

/// Source layer index, never the ordinal among GDN or attention blocks.
/// Existing QwenPrefixLayer::new remains closed to0..2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenLayer {
    source: SourceIdentity,
    index: u8,
}
impl QwenLayer {
    pub fn source(self) -> SourceIdentity {
        self.source
    }
    pub fn index(self) -> usize {
        self.index as usize
    }
    pub fn is_attention(self) -> bool {
        self.index % 4 == 3
    }
    fn name(self, suffix: &str) -> String {
        format!("model.language_model.layers.{}.{suffix}", self.index)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenProjection {
    Qkvz,
    Ba,
    GdnOut,
    AttentionQkv,
    AttentionOut,
    GateUp,
    Down,
    Head,
}
impl QwenProjection {
    pub fn input_width(self) -> usize {
        match self {
            Self::GdnOut | Self::AttentionOut => 2048,
            Self::Down => 3584,
            _ => 1024,
        }
    }
    pub fn output_width(self) -> usize {
        match self {
            Self::Qkvz => 8192,
            Self::Ba => 32,
            Self::AttentionQkv => 5120,
            Self::GateUp => 7168,
            Self::Head => 248320,
            _ => 1024,
        }
    }
    fn members_for_profile(self, profile: QwenSourceProfile) -> &'static [(&'static str, usize)] {
        if profile == QwenSourceProfile::Qwen08 {
            return self.members();
        }
        match self {
            Self::GdnOut => &[("linear_attn.out_proj.weight", 2048)],
            Self::AttentionOut => &[("self_attn.o_proj.weight", 2048)],
            Self::GateUp => &[("mlp.gate_proj.weight", 6144), ("mlp.up_proj.weight", 6144)],
            Self::Down => &[("mlp.down_proj.weight", 2048)],
            _ => self.members(),
        }
    }
    fn members(self) -> &'static [(&'static str, usize)] {
        match self {
            Self::Qkvz => &[
                ("linear_attn.in_proj_qkv.weight", 6144),
                ("linear_attn.in_proj_z.weight", 2048),
            ],
            Self::Ba => &[
                ("linear_attn.in_proj_b.weight", 16),
                ("linear_attn.in_proj_a.weight", 16),
            ],
            Self::GdnOut => &[("linear_attn.out_proj.weight", 1024)],
            Self::AttentionQkv => &[
                ("self_attn.q_proj.weight", 4096),
                ("self_attn.k_proj.weight", 512),
                ("self_attn.v_proj.weight", 512),
            ],
            Self::AttentionOut => &[("self_attn.o_proj.weight", 1024)],
            Self::GateUp => &[("mlp.gate_proj.weight", 3584), ("mlp.up_proj.weight", 3584)],
            Self::Down => &[("mlp.down_proj.weight", 1024)],
            Self::Head => &[("model.language_model.embed_tokens.weight", 248320)],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenRmsSite {
    Input,
    PostAttention,
    Final,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenOperation {
    InputRms,
    ResidualRms(QwenRmsSite),
    Projection(QwenProjection),
    GdnConv,
    GdnRecurrent,
    AttentionQk,
    KvAppend,
    FlashAttention,
    Sigmoid,
    Multiply,
    SiluAndMul,
}

/// Complete ordered original source members. No caller supplied packed name,
/// source activation, cache bytes, cosSin payload or arbitrary native manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenOperator {
    call: QwenSourceCall,
    semantics: Arc<QwenOperatorSemantics>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenOperatorSemantics {
    layer: Option<QwenLayer>,
    kind: QwenOperation,
    matrices: Vec<SourceTensor>,
    parameters: Vec<SourceParameter>,
}
impl std::ops::Deref for QwenOperator {
    type Target = QwenOperatorSemantics;
    fn deref(&self) -> &Self::Target {
        &self.semantics
    }
}
#[cfg(test)]
impl std::ops::DerefMut for QwenOperator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.semantics)
    }
}
impl QwenOperator {
    pub fn new(
        call: QwenSourceCall,
        layer: Option<QwenLayer>,
        kind: QwenOperation,
        matrices: Vec<SourceTensor>,
        parameters: Vec<SourceParameter>,
    ) -> Result<Self> {
        let value = Self {
            call,
            semantics: Arc::new(QwenOperatorSemantics {
                layer,
                kind,
                matrices,
                parameters,
            }),
        };
        value.validate()?;
        Ok(value)
    }
    pub fn call(&self) -> &QwenSourceCall {
        &self.call
    }
    pub fn layer(&self) -> Option<QwenLayer> {
        self.layer
    }
    pub fn kind(&self) -> QwenOperation {
        self.kind
    }
    pub fn matrices(&self) -> &[SourceTensor] {
        &self.matrices
    }
    pub fn parameters(&self) -> &[SourceParameter] {
        &self.parameters
    }
    pub fn validate(&self) -> Result<()> {
        self.call.validate()?;
        let final_site = matches!(
            self.kind,
            QwenOperation::ResidualRms(QwenRmsSite::Final)
                | QwenOperation::Projection(QwenProjection::Head)
        );
        if self.layer.is_none() != final_site
            || self
                .layer
                .is_some_and(|layer| layer.source != self.call.source() || layer.index >= 24)
        {
            return Err(Error::State("full-model exact source/layer/site".into()));
        }
        if self.matrices.capacity() > 3 || self.parameters.capacity() > 3 {
            return Err(Error::Limit("full-model bounded original members".into()));
        }
        let attention = self.layer.is_some_and(QwenLayer::is_attention);
        match self.kind {
            QwenOperation::InputRms if self.layer.map(QwenLayer::index) != Some(0) => {
                return Err(Error::Unsupported(
                    "ordinary input RMS only original layer0".into(),
                ));
            }
            QwenOperation::ResidualRms(QwenRmsSite::Input)
                if self.layer.map(QwenLayer::index) == Some(0) =>
            {
                return Err(Error::Unsupported("layer0 has no pending residual".into()));
            }
            QwenOperation::GdnConv | QwenOperation::GdnRecurrent if attention => {
                return Err(Error::State(
                    "GDN source operator at attention layer".into(),
                ));
            }
            QwenOperation::AttentionQk
            | QwenOperation::KvAppend
            | QwenOperation::FlashAttention
            | QwenOperation::Sigmoid
            | QwenOperation::Multiply
                if !attention =>
            {
                return Err(Error::State(
                    "attention source operator at GDN layer".into(),
                ));
            }
            QwenOperation::Projection(p)
                if match p {
                    QwenProjection::Qkvz | QwenProjection::Ba | QwenProjection::GdnOut => attention,
                    QwenProjection::AttentionQkv | QwenProjection::AttentionOut => !attention,
                    _ => false,
                } =>
            {
                return Err(Error::State("projection original layer kind".into()));
            }
            _ => {}
        }
        let expected = match self.kind {
            QwenOperation::Projection(p) => p.members_for_profile(self.call.model_profile),
            _ => &[],
        };
        if self.matrices.len() != expected.len() {
            return Err(Error::Malformed(
                "full-model exact ordered matrix count".into(),
            ));
        }
        for (matrix, &(suffix, rows)) in self.matrices.iter().zip(expected.iter()) {
            let QwenOperation::Projection(p) = self.kind else {
                unreachable!()
            };
            let name = if p == QwenProjection::Head {
                suffix.into()
            } else {
                self.layer.unwrap().name(suffix)
            };
            if matrix.source != self.call.source()
                || matrix.name != name
                || matrix.axes != MatrixAxes::OutputInput
                || matrix.kind != QuantKind::Bf16
                || matrix.columns != self.call.model_profile.projection_input(p)
                || matrix.rows != rows
            {
                return Err(Error::State(
                    "full-model original BF16 member/source/order/axes".into(),
                ));
            }
            matrix.packed_bytes()?;
        }
        let expected: Vec<(String, QuantKind, Vec<u64>, usize)> = match self.kind {
            QwenOperation::InputRms => vec![(
                self.layer.unwrap().name("input_layernorm.weight"),
                QuantKind::Bf16,
                vec![self.call.model_profile.hidden_size() as u64],
                self.call.model_profile.hidden_size() * 2,
            )],
            QwenOperation::ResidualRms(site) => vec![(
                match site {
                    QwenRmsSite::Input => self.layer.unwrap().name("input_layernorm.weight"),
                    QwenRmsSite::PostAttention => {
                        self.layer.unwrap().name("post_attention_layernorm.weight")
                    }
                    QwenRmsSite::Final => "model.language_model.norm.weight".into(),
                },
                QuantKind::Bf16,
                vec![self.call.model_profile.hidden_size() as u64],
                self.call.model_profile.hidden_size() * 2,
            )],
            QwenOperation::GdnConv => vec![(
                self.layer.unwrap().name("linear_attn.conv1d.weight"),
                QuantKind::Bf16,
                vec![6144, 1, 4],
                49152,
            )],
            QwenOperation::GdnRecurrent => [
                ("linear_attn.A_log", QuantKind::F32, 16, 64),
                ("linear_attn.dt_bias", QuantKind::Bf16, 16, 32),
                ("linear_attn.norm.weight", QuantKind::F32, 128, 512),
            ]
            .into_iter()
            .map(|(name, kind, n, bytes)| (self.layer.unwrap().name(name), kind, vec![n], bytes))
            .collect(),
            QwenOperation::AttentionQk => ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
                .into_iter()
                .map(|name| {
                    (
                        self.layer.unwrap().name(name),
                        QuantKind::Bf16,
                        vec![256],
                        512,
                    )
                })
                .collect(),
            _ => Vec::new(),
        };
        if self.parameters.len() != expected.len() {
            return Err(Error::Malformed(
                "full-model exact ordered parameter count".into(),
            ));
        }
        for (parameter, (name, kind, shape, bytes)) in self.parameters.iter().zip(expected) {
            if parameter.source != self.call.source()
                || parameter.name != name
                || parameter.kind != kind
                || parameter.shape != shape
                || parameter.bytes != bytes
            {
                return Err(Error::State(
                    "full-model exact original parameter/source/extent".into(),
                ));
            }
        }
        Ok(())
    }
    pub fn input_elements(&self) -> Vec<usize> {
        let n = self.call.rows();
        let profile = self.call.model_profile;
        match self.kind {
            QwenOperation::InputRms => vec![n * profile.hidden_size()],
            QwenOperation::ResidualRms(_) => {
                vec![n * profile.hidden_size(), n * profile.hidden_size()]
            }
            QwenOperation::Projection(p) => vec![n * profile.projection_input(p)],
            QwenOperation::GdnConv => vec![n * 8192, n * 32],
            QwenOperation::GdnRecurrent => vec![n * 6144, n * 2048, n * 32],
            QwenOperation::AttentionQk => vec![n * 5120],
            // V is the original packed projection's strided tail; fused QK
            // writes only Q/K/gate. Never mint an unwritten dense V producer.
            QwenOperation::KvAppend => vec![n * 512, n * 5120],
            QwenOperation::FlashAttention => vec![n * 2048],
            QwenOperation::Sigmoid => vec![n * 2048],
            QwenOperation::Multiply => vec![n * 2048, n * 2048],
            QwenOperation::SiluAndMul => vec![n * profile.intermediate_size() * 2],
        }
    }
    pub fn output_elements(&self) -> Vec<usize> {
        let n = self.call.rows();
        let profile = self.call.model_profile;
        match self.kind {
            QwenOperation::InputRms => vec![n * profile.hidden_size()],
            QwenOperation::ResidualRms(_) => {
                vec![n * profile.hidden_size(), n * profile.hidden_size()]
            }
            QwenOperation::Projection(p) => vec![if p == QwenProjection::Head {
                248320
            } else {
                n * profile.projection_output(p)
            }],
            QwenOperation::GdnConv => vec![n * 6144, n * 2048],
            QwenOperation::GdnRecurrent
            | QwenOperation::FlashAttention
            | QwenOperation::Sigmoid
            | QwenOperation::Multiply => vec![n * 2048],
            QwenOperation::AttentionQk => vec![n * 2048, n * 512, n * 2048],
            QwenOperation::KvAppend => vec![],
            QwenOperation::SiluAndMul => vec![n * profile.intermediate_size()],
        }
    }
    /// Data-only view of the actual full-phase normalizer; no new teacher.
    pub fn selected_row(&self) -> Option<usize> {
        (self.kind == QwenOperation::Projection(QwenProjection::Head))
            .then_some(self.call.rows() - 1)
    }
}

/// Core-minted result identity. Generic views/ops cannot manufacture this tag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenProducer {
    call: QwenSourceCall,
    layer: Option<QwenLayer>,
    kind: QwenOperation,
    node_index: usize,
    output_index: usize,
    inputs: Arc<[ValueId]>,
}
impl QwenProducer {
    pub(super) fn bind_call(&mut self, call: QwenSourceCall) {
        self.call = call;
    }
    pub fn call(&self) -> &QwenSourceCall {
        &self.call
    }
    pub fn layer(&self) -> Option<QwenLayer> {
        self.layer
    }
    pub fn kind(&self) -> QwenOperation {
        self.kind
    }
    pub fn node_index(&self) -> usize {
        self.node_index
    }
    pub fn output_index(&self) -> usize {
        self.output_index
    }
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }
}

pub(super) fn precision(
    builder: &QwenPlanBuilder,
    d: &QwenOperator,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    d.validate()?;
    if builder.model_call.as_ref() != Some(&d.call)
        || builder.operator_slice
        || builder.profile != ProgramNumericalProfile::SourceCudaSm86Bf16V1
        || builder.source != d.call.source()
        || inputs.len() != d.input_elements().len()
    {
        return Err(Error::Unsupported(
            "closed full-model source/profile/call/arity".into(),
        ));
    }
    for (input, elements) in inputs.iter().zip(d.input_elements()) {
        if input.precision() != ActivationPrecision::Bf16WidenedF32
            || input.shape != ValueShape::Vector(elements)
            || input.elements() != elements
        {
            return Err(Error::State("full-model original dense BF16 input".into()));
        }
    }
    let tag = gdn_projection::require_invocation(builder, d.call.request(), inputs)?;
    // Stateful/cohort checks use immutable per-value tags, never a graph walk.
    let origin = |i: usize| -> Result<&QwenProducer> {
        match builder
            .value_provenance(&inputs[i])?
            .map(QwenValueProvenance::origin)
        {
            Some(QwenValueOrigin::FullModel(origin))
                if origin.call == d.call && origin.layer == d.layer =>
            {
                Ok(origin)
            }
            _ => Err(Error::State(
                "full-model exact same-call/layer producer".into(),
            )),
        }
    };
    match d.kind {
        QwenOperation::InputRms if tag.origin() != &QwenValueOrigin::SourceRows => {
            return Err(Error::State(
                "full-model input RMS requires source rows".into(),
            ));
        }
        QwenOperation::GdnConv => {
            for (i, kind) in [QwenProjection::Qkvz, QwenProjection::Ba]
                .into_iter()
                .enumerate()
            {
                if origin(i)?.kind != QwenOperation::Projection(kind) {
                    return Err(Error::State("full-model real QKVZ/BA producers".into()));
                }
            }
        }
        QwenOperation::GdnRecurrent => {
            let (conv, z, ba) = (origin(0)?, origin(1)?, origin(2)?);
            if conv.kind != QwenOperation::GdnConv
                || z.kind != QwenOperation::GdnConv
                || conv.output_index != 0
                || z.output_index != 1
                || conv.node_index != z.node_index
                || conv.inputs != z.inputs
                || conv.inputs.get(1) != Some(&inputs[2].id)
                || ba.kind != QwenOperation::Projection(QwenProjection::Ba)
            {
                return Err(Error::State(
                    "full-model real Conv/Z/original BA cohort".into(),
                ));
            }
        }
        QwenOperation::KvAppend => {
            let (k, packed) = (origin(0)?, origin(1)?);
            if k.kind != QwenOperation::AttentionQk
                || k.output_index != 1
                || packed.kind != QwenOperation::Projection(QwenProjection::AttentionQkv)
                || k.inputs.as_ref() != [inputs[1].id]
            {
                return Err(Error::State(
                    "full-model K and original packed-V producer cohort".into(),
                ));
            }
        }
        QwenOperation::FlashAttention => {
            if origin(0)?.kind != QwenOperation::AttentionQk || origin(0)?.output_index != 0 {
                return Err(Error::State(
                    "full-model real rotated query producer".into(),
                ));
            }
            if builder.kv_cohorts.get(&d.layer.unwrap().index()) != Some(&origin(0)?.node_index) {
                return Err(Error::State(
                    "FA query must match same-plan K/V append cohort".into(),
                ));
            }
        }
        QwenOperation::Projection(QwenProjection::Head)
            if origin(0)?.kind != QwenOperation::ResidualRms(QwenRmsSite::Final)
                || origin(0)?.output_index != 0 =>
        {
            return Err(Error::State(
                "head requires actual full-phase final normalization".into(),
            ));
        }
        _ => {}
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}

pub(super) fn provenance(
    builder: &QwenPlanBuilder,
    d: &QwenOperator,
    inputs: &[PlanValue],
    output_index: usize,
) -> Result<QwenValueProvenance> {
    if output_index >= d.output_elements().len() {
        return Err(Error::State("full-model output provenance index".into()));
    }
    let tag = gdn_projection::require_invocation(builder, d.call.request(), inputs)?;
    Ok(tag.with_origin(QwenValueOrigin::FullModel(QwenProducer {
        call: d.call.clone(),
        layer: d.layer,
        kind: d.kind,
        node_index: builder.core.nodes().len(),
        output_index,
        inputs: inputs.iter().map(|value| value.id).collect(),
    })))
}

pub(super) fn record(
    builder: &mut QwenPlanBuilder,
    d: &QwenOperator,
    inputs: &[PlanValue],
) -> Result<Vec<PlanValue>> {
    precision(builder, d, inputs)?;
    let tags = (0..d.output_elements().len())
        .map(|index| provenance(builder, d, inputs, index))
        .collect::<Result<Vec<_>>>()?;
    let values = builder.record_source(
        "qwen.source.full_model",
        d.mathematical_operator()?,
        inputs,
        tags,
    )?;
    if d.kind == QwenOperation::KvAppend {
        let Some(QwenValueOrigin::FullModel(origin)) = builder
            .value_provenance(&inputs[0])?
            .map(QwenValueProvenance::origin)
        else {
            return Err(Error::State("Qwen validated KV producer absent".into()));
        };
        let node = origin.node_index;
        builder.kv_cohorts.insert(d.layer.unwrap().index(), node);
    }
    Ok(values)
}

#[derive(Clone, Debug)]
pub struct QwenPlan {
    plan: ForwardPlan,
    provenance: Vec<Option<QwenValueProvenance>>,
    call: QwenSourceCall,
    boundaries: Arc<[[ValueId; 2]]>,
    normalized: ValueId,
    logits: ValueId,
}
impl QwenPlan {
    pub const PREPARED_METADATA_BYTES: usize = 4 * 1024 * 1024;
    /// Retain the original host charge with every shared invariant allocation.
    /// This is lifetime metadata, never source authentication or execution authority.
    pub fn retain_prepared_metadata(
        mut self,
        metadata: zerogravity_core::resource_arbiter::HostWorkspace,
    ) -> Result<Self> {
        if metadata.source_identity() != self.plan.source()
            || metadata.bytes() < Self::PREPARED_METADATA_BYTES
        {
            return Err(Error::State(
                "prepared Qwen original host metadata charge".into(),
            ));
        }
        self.plan = self.plan.retain_prepared_metadata(metadata)?;
        Ok(self)
    }
    /// Invocation-local wrappers only; shared source/topology owners are priced at preparation.
    pub const BINDING_METADATA_BYTES: usize = 512 * 1024;

    /// Rebind only the existing sealed bounded prompt recipe, never a native/root permit.
    /// Every node and producer receives this call's authentic input issuer.
    pub fn bind_prompt_call(&self, call: &PromptCall) -> Result<Self> {
        self.require_prompt_shape(call)?;
        let mut bound = Self {
            plan: self
                .plan
                .rebind_source_call(call.request().source_call()?)?,
            provenance: self.provenance.clone(),
            call: call.as_source_call().clone(),
            boundaries: self.boundaries.clone(),
            normalized: self.normalized,
            logits: self.logits,
        };
        for tag in &mut bound.provenance {
            tag.as_mut()
                .ok_or_else(|| Error::State("prepared Qwen producer coverage".into()))?
                .bind_prompt_call(&bound.call)?;
        }
        bound.require_binding_metadata(Self::BINDING_METADATA_BYTES)?;
        Ok(bound)
    }
    /// Local record clone payload; shared Arc data and request/root owners are separately charged.
    pub fn binding_metadata_bytes(&self) -> Result<usize> {
        let overflow = || Error::Limit("prepared Qwen binding metadata extent".into());
        let producers = self
            .provenance
            .capacity()
            .checked_mul(std::mem::size_of::<Option<QwenValueProvenance>>())
            .ok_or_else(overflow)?;
        [
            self.plan.source_binding_metadata_bytes()?,
            producers,
            2 * std::mem::size_of::<usize>(),
        ]
        .into_iter()
        .try_fold(std::mem::size_of::<Self>(), |bytes, n| {
            bytes.checked_add(n).ok_or_else(overflow)
        })
    }
    fn require_binding_metadata(&self, available: usize) -> Result<usize> {
        let bytes = self.binding_metadata_bytes()?;
        if bytes > available {
            return Err(Error::Limit(
                "prepared full-model binding metadata bound".into(),
            ));
        }
        Ok(bytes)
    }
    /// Exact immutable semantics identity, separate from invocation equality.
    /// Different calls must still pass their original per-row runtime/root checks.
    pub fn shares_prepared_semantics(&self, other: &Self) -> bool {
        self.plan.shares_semantics(&other.plan) && Arc::ptr_eq(&self.boundaries, &other.boundaries)
    }
    fn require_prompt_shape(&self, call: &PromptCall) -> Result<()> {
        call.validate()?;
        let prepared = self
            .call
            .request()
            .teachers()
            .prompt_span()
            .ok_or_else(|| {
                Error::Unsupported("prepared binding requires a bounded prompt recipe".into())
            })?;
        let actual = call.span();
        if self.call.source() != call.as_source_call().source()
            || self.call.model_profile() != call.model_profile()
            || prepared.workload() != actual.workload()
            || prepared.mode() != actual.mode()
            || prepared.base() != actual.base()
            || prepared.len() != actual.len()
            || prepared.capacity() != actual.capacity()
        {
            return Err(Error::State(
                "prepared bounded prompt source/profile/workload/phase/layout differs".into(),
            ));
        }
        Ok(())
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    /// Consume family metadata without cloning the mathematical graph.
    pub fn into_plan(self) -> ForwardPlan {
        self.plan
    }
    /// Cold adapter metadata only; reconstruct and validate the actual common
    /// source operation. No allocation-free hot-path or native permit is implied.
    pub fn operator(&self, node_index: usize) -> Result<Option<QwenOperator>> {
        let node = self
            .plan
            .nodes()
            .get(node_index)
            .ok_or_else(|| Error::State("Qwen operator node outside actual plan".into()))?;
        let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
            return Err(Error::State(
                "Qwen operator contains foreign mathematical node".into(),
            ));
        };
        let call = self.call.request().source_call()?;
        if descriptor.call() != &call {
            return Err(Error::State(
                "Qwen operator original call issuer differs".into(),
            ));
        }
        let (layer, kind) = if let Some(id) = node.outputs.first() {
            let tag = self
                .value_provenance(*id)?
                .ok_or_else(|| Error::State("Qwen operator producer metadata absent".into()))?;
            tag.require_invocation(self.call.request())?;
            match tag.origin() {
                QwenValueOrigin::SourceRows if node_index == 0 => {
                    let [matrix] = descriptor.matrices() else {
                        return Err(Error::State("Qwen original embedding member arity".into()));
                    };
                    let expected = QwenEmbedding::for_call(matrix.clone(), &self.call)?;
                    if !matches!(descriptor.kind(), SourceOperatorKind::EmbeddingRows)
                        || !node.inputs.is_empty()
                        || node.outputs.len() != 1
                        || node.state.is_some()
                    {
                        return Err(Error::State(
                            "Qwen original embedding producer differs".into(),
                        ));
                    }
                    expected.validate()?;
                    return Ok(None);
                }
                QwenValueOrigin::FullModel(origin)
                    if origin.node_index == node_index
                        && origin.output_index == 0
                        && origin.inputs.as_ref() == node.inputs.as_slice()
                        && origin.call == self.call =>
                {
                    (origin.layer, origin.kind)
                }
                _ => {
                    return Err(Error::State(
                        "Qwen actual operator producer/index/inputs differ".into(),
                    ));
                }
            }
        } else if let SourceOperatorKind::KvAppend { domain, .. } = descriptor.kind() {
            let layer = self.call.layer(domain.0)?;
            let Some(effect) = node.state else {
                return Err(Error::State(
                    "Qwen KV append lacks actual state effect".into(),
                ));
            };
            if effect.read
                != (StateVersion {
                    slot: StateSlot::Kv(domain),
                    ordinal: 0,
                })
                || effect.write
                    != (StateVersion {
                        slot: StateSlot::Kv(domain),
                        ordinal: 1,
                    })
            {
                return Err(Error::State(
                    "Qwen KV append original state version differs".into(),
                ));
            }
            (Some(layer), QwenOperation::KvAppend)
        } else {
            return Err(Error::State(
                "Qwen empty-output operation is not its actual KV append".into(),
            ));
        };
        let family = QwenOperator::new(
            self.call.clone(),
            layer,
            kind,
            descriptor.matrices().to_vec(),
            descriptor.parameters().to_vec(),
        )?;
        if family.mathematical_operator()? != **descriptor {
            return Err(Error::State(
                "Qwen operator mathematical source/geometry differs".into(),
            ));
        }
        Ok(Some(family))
    }
    pub fn value_provenance(&self, id: ValueId) -> Result<Option<&QwenValueProvenance>> {
        self.plan.source_value_provenance(id)?;
        Ok(self.provenance.get(id.0 as usize).and_then(Option::as_ref))
    }
    pub fn call(&self) -> &QwenSourceCall {
        &self.call
    }
    pub fn boundaries(&self) -> &[[ValueId; 2]] {
        &self.boundaries
    }
    pub fn normalized(&self) -> ValueId {
        self.normalized
    }
    pub fn selected_row(&self) -> usize {
        self.call.rows() - 1
    }
    pub fn logits(&self) -> ValueId {
        self.logits
    }
}

fn all_domains() -> Result<Vec<StateSlot>> {
    hybrid_state_domains((0..24).map(|index| index % 4 == 3))
}
impl QwenPlanBuilder {
    pub fn new_small_model(call: QwenSourceCall) -> Result<Self> {
        if call.model_profile() != QwenSourceProfile::Qwen08 {
            return Err(Error::Unsupported(
                "original .8B builder constructor profile".into(),
            ));
        }
        Self::new_model(call)
    }
    pub fn new_model(call: QwenSourceCall) -> Result<Self> {
        call.validate()?;
        let mut builder = Self::from_core(CorePlanBuilder::new_source_program(
            call.request().source_call()?,
            all_domains()?,
        )?);
        builder.fragment_call = Some(call.request.clone());
        builder.model_call = Some(call);
        Ok(builder)
    }
    /// Whole arithmetic record remains unpublished: no logits authority/frontier.
    pub fn finish_small_model(
        self,
        boundaries: Vec<[PlanValue; 2]>,
        normalized: PlanValue,
        logits: PlanValue,
    ) -> Result<QwenPlan> {
        if self
            .model_call
            .as_ref()
            .is_some_and(|call| call.model_profile() != QwenSourceProfile::Qwen08)
        {
            return Err(Error::Unsupported(
                "original .8B finish constructor profile".into(),
            ));
        }
        self.finish_model(boundaries, normalized, logits)
    }
    pub fn finish_model(
        self,
        boundaries: Vec<[PlanValue; 2]>,
        normalized: PlanValue,
        logits: PlanValue,
    ) -> Result<QwenPlan> {
        let call = self
            .model_call
            .clone()
            .ok_or_else(|| Error::State("not full-model builder".into()))?;
        if self.core.has_advanced()
            || self.operator_slice
            || self.core.required_domains().len() != 42
            || *self.core.required_domains() != all_domains()?.into_iter().collect()
            || self.core.proposed_domains().len() != 42
            || boundaries.len() != 24
            || boundaries.capacity() > 24
        {
            return Err(Error::State(
                "full24 record requires all42 domains/24 boundaries/no frontier".into(),
            ));
        }
        for slot in self.core.required_domains() {
            if self.core.current_state(*slot)?.ordinal != 1 {
                return Err(Error::State(
                    "one actual transition per original indexed domain".into(),
                ));
            }
        }
        let require = |value: &PlanValue,
                       layer: Option<QwenLayer>,
                       kind: QwenOperation,
                       out: usize|
         -> Result<()> {
            match self
                .value_provenance(value)?
                .map(QwenValueProvenance::origin)
            {
                Some(QwenValueOrigin::FullModel(origin))
                    if origin.call == call
                        && origin.layer == layer
                        && origin.kind == kind
                        && origin.output_index == out =>
                {
                    Ok(())
                }
                _ => Err(Error::State("full-model final producer role/issuer".into())),
            }
        };
        for (index, [branch, residual]) in boundaries.iter().enumerate() {
            if branch.id == residual.id {
                return Err(Error::State(
                    "dual source boundaries must be distinct".into(),
                ));
            }
            let layer = Some(call.layer(index)?);
            require(
                branch,
                layer,
                QwenOperation::Projection(QwenProjection::Down),
                0,
            )?;
            require(
                residual,
                layer,
                QwenOperation::ResidualRms(QwenRmsSite::PostAttention),
                1,
            )?;
        }
        require(
            &normalized,
            None,
            QwenOperation::ResidualRms(QwenRmsSite::Final),
            0,
        )?;
        require(
            &logits,
            None,
            QwenOperation::Projection(QwenProjection::Head),
            0,
        )?;
        verify_recipe(&self, &call, &boundaries, normalized.id, logits.id)?;
        Ok(QwenPlan {
            call,
            boundaries: boundaries.into_iter().map(|v| [v[0].id, v[1].id]).collect(),
            normalized: normalized.id,
            logits: logits.id,
            provenance: self.provenance,
            plan: self.core.finish_source_program()?,
        })
    }
}

/// Linear closed-recipe check, not a provenance graph search. It ties every
/// named boundary to the actual earlier node's ValueIds before any lowerer.
fn verify_recipe(
    builder: &QwenPlanBuilder,
    call: &QwenSourceCall,
    boundaries: &[[PlanValue; 2]],
    normalized: ValueId,
    logits: ValueId,
) -> Result<()> {
    let embedding = builder
        .core
        .nodes()
        .first()
        .ok_or_else(|| Error::State("source rows node absent".into()))?;
    let PlanOperation::SourceOperator {
        descriptor,
        state: None,
    } = &embedding.operation
    else {
        return Err(Error::State(
            "Qwen plan must begin with original source rows".into(),
        ));
    };
    if !matches!(descriptor.kind(), SourceOperatorKind::EmbeddingRows)
        || descriptor.call() != &call.request().source_call()?
        || descriptor.matrices().len() != 1
        || !embedding.inputs.is_empty()
        || embedding.outputs.len() != 1
        || embedding.state.is_some()
    {
        return Err(Error::State(
            "Qwen actual original source/teacher producer".into(),
        ));
    }
    let expected = QwenEmbedding::for_call(descriptor.matrices()[0].clone(), call)?;
    expected.validate()?;
    let raw = embedding.outputs[0];
    let mut cursor = 1usize;
    let mut take = |layer: Option<QwenLayer>,
                    kind: QwenOperation,
                    inputs: &[ValueId]|
     -> Result<Vec<ValueId>> {
        let node = builder
            .core
            .nodes()
            .get(cursor)
            .ok_or_else(|| Error::State("full24 recipe node absent".into()))?;
        let PlanOperation::SourceOperator { descriptor, state } = &node.operation else {
            return Err(Error::State("Qwen exact typed source recipe".into()));
        };
        let expected = QwenOperator::new(
            call.clone(),
            layer,
            kind,
            descriptor.matrices().to_vec(),
            descriptor.parameters().to_vec(),
        )?;
        if **descriptor != expected.mathematical_operator()?
            || node.inputs != inputs
            || node.outputs.len() != expected.output_elements().len()
        {
            return Err(Error::State(
                "Qwen actual node source/operator/SSA recipe".into(),
            ));
        }
        let slot = match kind {
            QwenOperation::GdnConv => Some(StateSlot::Conv(ConvDomain(layer.unwrap().index()))),
            QwenOperation::GdnRecurrent => Some(StateSlot::Recurrent(RecurrentDomain(
                layer.unwrap().index(),
            ))),
            QwenOperation::KvAppend => Some(StateSlot::Kv(KvDomain(layer.unwrap().index()))),
            _ => None,
        };
        match (node.state, slot) {
            (Some(effect), Some(slot))
                if effect.read.slot == slot
                    && effect.read.ordinal == 0
                    && effect.write.slot == slot
                    && effect.write.ordinal == 1 => {}
            (None, None) => {}
            _ => {
                return Err(Error::State(
                    "full24 one real effect at actual source layer".into(),
                ));
            }
        }
        match (kind, state) {
            (QwenOperation::FlashAttention, Some(version))
                if version.slot == StateSlot::Kv(KvDomain(layer.unwrap().index()))
                    && version.ordinal == 1 => {}
            (QwenOperation::FlashAttention, _) => {
                return Err(Error::State(
                    "FA must read its actual private appended version".into(),
                ));
            }
            (_, None) => {}
            _ => {
                return Err(Error::State(
                    "non-FA operation cannot import a state read".into(),
                ));
            }
        }
        cursor += 1;
        Ok(node.outputs.clone())
    };
    let mut branch = raw;
    let mut residual = None;
    for (index, boundary) in boundaries.iter().enumerate() {
        let layer = call.layer(index)?;
        let (normal, carried) = if index == 0 {
            (
                take(Some(layer), QwenOperation::InputRms, &[branch])?[0],
                branch,
            )
        } else {
            let previous =
                residual.ok_or_else(|| Error::State("full24 pending carried residual".into()))?;
            let dual = take(
                Some(layer),
                QwenOperation::ResidualRms(QwenRmsSite::Input),
                &[branch, previous],
            )?;
            (dual[0], dual[1])
        };
        let projected = if layer.is_attention() {
            let packed = take(
                Some(layer),
                QwenOperation::Projection(QwenProjection::AttentionQkv),
                &[normal],
            )?[0];
            let qk = take(Some(layer), QwenOperation::AttentionQk, &[packed])?;
            take(Some(layer), QwenOperation::KvAppend, &[qk[1], packed])?;
            let fa = take(Some(layer), QwenOperation::FlashAttention, &[qk[0]])?[0];
            let sigmoid = take(Some(layer), QwenOperation::Sigmoid, &[qk[2]])?[0];
            let gated = take(Some(layer), QwenOperation::Multiply, &[fa, sigmoid])?[0];
            take(
                Some(layer),
                QwenOperation::Projection(QwenProjection::AttentionOut),
                &[gated],
            )?[0]
        } else {
            let packed = take(
                Some(layer),
                QwenOperation::Projection(QwenProjection::Qkvz),
                &[normal],
            )?[0];
            let ba = take(
                Some(layer),
                QwenOperation::Projection(QwenProjection::Ba),
                &[normal],
            )?[0];
            let conv = take(Some(layer), QwenOperation::GdnConv, &[packed, ba])?;
            let gated = take(
                Some(layer),
                QwenOperation::GdnRecurrent,
                &[conv[0], conv[1], ba],
            )?[0];
            take(
                Some(layer),
                QwenOperation::Projection(QwenProjection::GdnOut),
                &[gated],
            )?[0]
        };
        let post = take(
            Some(layer),
            QwenOperation::ResidualRms(QwenRmsSite::PostAttention),
            &[projected, carried],
        )?;
        let gate_up = take(
            Some(layer),
            QwenOperation::Projection(QwenProjection::GateUp),
            &[post[0]],
        )?[0];
        let activated = take(Some(layer), QwenOperation::SiluAndMul, &[gate_up])?[0];
        branch = take(
            Some(layer),
            QwenOperation::Projection(QwenProjection::Down),
            &[activated],
        )?[0];
        residual = Some(post[1]);
        if [branch, post[1]] != [boundary[0].id, boundary[1].id] {
            return Err(Error::State(
                "all24 expected boundary must be actual branch/carried producer".into(),
            ));
        }
    }
    let final_norm = take(
        None,
        QwenOperation::ResidualRms(QwenRmsSite::Final),
        &[branch, residual.unwrap()],
    )?;
    let head = take(
        None,
        QwenOperation::Projection(QwenProjection::Head),
        &[final_norm[0]],
    )?;
    if final_norm[0] != normalized || head[0] != logits || cursor != builder.core.nodes().len() {
        return Err(Error::State(
            "full24 finalnorm/fullV and exact no-extra recipe".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "model/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "model/profile_tests.rs"]
mod profile_tests;

#[cfg(test)]
#[path = "model/span_tests.rs"]
mod span_tests;

#[cfg(test)]
#[path = "model/prepared_tests.rs"]
mod prepared_tests;
