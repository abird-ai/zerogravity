//! Closed .8B full-layer metadata inside the original mathematical engine.
//! No allocation, runtime issuer, frontier advance or RootReady capability.
use super::*;

/// Original request issuer; selecting one output row never replaces this call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vllm08Call {
    request: VllmBf16Invocation,
    model_profile: FullModelProfile,
}
impl Vllm08Call {
    pub fn new(request: VllmBf16Invocation) -> Result<Self> {
        Self::for_profile(FullModelProfile::Qwen08, request)
    }
    /// Geometry metadata only; the original source owner must bind the profile.
    pub fn for_profile(
        model_profile: FullModelProfile,
        request: VllmBf16Invocation,
    ) -> Result<Self> {
        request.legacy_phase()?;
        Self::from_request(model_profile, request)
    }
    /// Semantic .8B span recipe only; no finite native/runtime coverage.
    pub fn from_span(request: VllmBf16Invocation) -> Result<Self> {
        if request.teachers().short_hf_span().is_some() {
            return Err(Error::Unsupported(
                "short-HF issuer requires its source-profile call".into(),
            ));
        }
        Self::from_request(FullModelProfile::Qwen08, request)
    }
    pub(super) fn from_request(
        model_profile: FullModelProfile,
        request: VllmBf16Invocation,
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
    pub fn model_profile(&self) -> FullModelProfile {
        self.model_profile
    }
    pub fn request(&self) -> &VllmBf16Invocation {
        &self.request
    }
    pub fn source(&self) -> SourceIdentity {
        self.request.layer().source()
    }
    pub fn rows(&self) -> usize {
        self.request.rows()
    }
    pub fn layer(&self, index: usize) -> Result<Vllm08Layer> {
        if index >= 24 {
            return Err(Error::Unsupported("closed original24 layer index".into()));
        }
        Ok(Vllm08Layer {
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
/// Existing GdnPrefixLayer::new remains closed to0..2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vllm08Layer {
    source: SourceIdentity,
    index: u8,
}
impl Vllm08Layer {
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
pub enum Vllm08Projection {
    Qkvz,
    Ba,
    GdnOut,
    AttentionQkv,
    AttentionOut,
    GateUp,
    Down,
    Head,
}
impl Vllm08Projection {
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
    fn members_for_profile(self, profile: FullModelProfile) -> &'static [(&'static str, usize)] {
        if profile == FullModelProfile::Qwen08 {
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
pub enum Vllm08RmsSite {
    Input,
    PostAttention,
    Final,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vllm08Kind {
    InputRms,
    ResidualRms(Vllm08RmsSite),
    Projection(Vllm08Projection),
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
pub struct Vllm08Operator {
    call: Vllm08Call,
    semantics: Arc<Vllm08OperatorSemantics>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vllm08OperatorSemantics {
    layer: Option<Vllm08Layer>,
    kind: Vllm08Kind,
    matrices: Vec<SourceTensor>,
    parameters: Vec<SourceParameter>,
}
impl std::ops::Deref for Vllm08Operator {
    type Target = Vllm08OperatorSemantics;
    fn deref(&self) -> &Self::Target {
        &self.semantics
    }
}
#[cfg(test)]
impl std::ops::DerefMut for Vllm08Operator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.semantics)
    }
}
impl Vllm08Operator {
    pub fn new(
        call: Vllm08Call,
        layer: Option<Vllm08Layer>,
        kind: Vllm08Kind,
        matrices: Vec<SourceTensor>,
        parameters: Vec<SourceParameter>,
    ) -> Result<Self> {
        let value = Self {
            call,
            semantics: Arc::new(Vllm08OperatorSemantics {
                layer,
                kind,
                matrices,
                parameters,
            }),
        };
        value.validate()?;
        Ok(value)
    }
    pub fn call(&self) -> &Vllm08Call {
        &self.call
    }
    pub fn layer(&self) -> Option<Vllm08Layer> {
        self.layer
    }
    pub fn kind(&self) -> Vllm08Kind {
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
            Vllm08Kind::ResidualRms(Vllm08RmsSite::Final)
                | Vllm08Kind::Projection(Vllm08Projection::Head)
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
        let attention = self.layer.is_some_and(Vllm08Layer::is_attention);
        match self.kind {
            Vllm08Kind::InputRms if self.layer.map(Vllm08Layer::index) != Some(0) => {
                return Err(Error::Unsupported(
                    "ordinary input RMS only original layer0".into(),
                ));
            }
            Vllm08Kind::ResidualRms(Vllm08RmsSite::Input)
                if self.layer.map(Vllm08Layer::index) == Some(0) =>
            {
                return Err(Error::Unsupported("layer0 has no pending residual".into()));
            }
            Vllm08Kind::GdnConv | Vllm08Kind::GdnRecurrent if attention => {
                return Err(Error::State(
                    "GDN source operator at attention layer".into(),
                ));
            }
            Vllm08Kind::AttentionQk
            | Vllm08Kind::KvAppend
            | Vllm08Kind::FlashAttention
            | Vllm08Kind::Sigmoid
            | Vllm08Kind::Multiply
                if !attention =>
            {
                return Err(Error::State(
                    "attention source operator at GDN layer".into(),
                ));
            }
            Vllm08Kind::Projection(p)
                if match p {
                    Vllm08Projection::Qkvz | Vllm08Projection::Ba | Vllm08Projection::GdnOut => {
                        attention
                    }
                    Vllm08Projection::AttentionQkv | Vllm08Projection::AttentionOut => !attention,
                    _ => false,
                } =>
            {
                return Err(Error::State("projection original layer kind".into()));
            }
            _ => {}
        }
        let expected = match self.kind {
            Vllm08Kind::Projection(p) => p.members_for_profile(self.call.model_profile),
            _ => &[],
        };
        if self.matrices.len() != expected.len() {
            return Err(Error::Malformed(
                "full-model exact ordered matrix count".into(),
            ));
        }
        for (matrix, &(suffix, rows)) in self.matrices.iter().zip(expected.iter()) {
            let Vllm08Kind::Projection(p) = self.kind else {
                unreachable!()
            };
            let name = if p == Vllm08Projection::Head {
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
            Vllm08Kind::InputRms => vec![(
                self.layer.unwrap().name("input_layernorm.weight"),
                QuantKind::Bf16,
                vec![self.call.model_profile.hidden_size() as u64],
                self.call.model_profile.hidden_size() * 2,
            )],
            Vllm08Kind::ResidualRms(site) => vec![(
                match site {
                    Vllm08RmsSite::Input => self.layer.unwrap().name("input_layernorm.weight"),
                    Vllm08RmsSite::PostAttention => {
                        self.layer.unwrap().name("post_attention_layernorm.weight")
                    }
                    Vllm08RmsSite::Final => "model.language_model.norm.weight".into(),
                },
                QuantKind::Bf16,
                vec![self.call.model_profile.hidden_size() as u64],
                self.call.model_profile.hidden_size() * 2,
            )],
            Vllm08Kind::GdnConv => vec![(
                self.layer.unwrap().name("linear_attn.conv1d.weight"),
                QuantKind::Bf16,
                vec![6144, 1, 4],
                49152,
            )],
            Vllm08Kind::GdnRecurrent => [
                ("linear_attn.A_log", QuantKind::F32, 16, 64),
                ("linear_attn.dt_bias", QuantKind::Bf16, 16, 32),
                ("linear_attn.norm.weight", QuantKind::F32, 128, 512),
            ]
            .into_iter()
            .map(|(name, kind, n, bytes)| (self.layer.unwrap().name(name), kind, vec![n], bytes))
            .collect(),
            Vllm08Kind::AttentionQk => ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
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
            Vllm08Kind::InputRms => vec![n * profile.hidden_size()],
            Vllm08Kind::ResidualRms(_) => {
                vec![n * profile.hidden_size(), n * profile.hidden_size()]
            }
            Vllm08Kind::Projection(p) => vec![n * profile.projection_input(p)],
            Vllm08Kind::GdnConv => vec![n * 8192, n * 32],
            Vllm08Kind::GdnRecurrent => vec![n * 6144, n * 2048, n * 32],
            Vllm08Kind::AttentionQk => vec![n * 5120],
            // V is the original packed projection's strided tail; fused QK
            // writes only Q/K/gate. Never mint an unwritten dense V producer.
            Vllm08Kind::KvAppend => vec![n * 512, n * 5120],
            Vllm08Kind::FlashAttention => vec![n * 2048],
            Vllm08Kind::Sigmoid => vec![n * 2048],
            Vllm08Kind::Multiply => vec![n * 2048, n * 2048],
            Vllm08Kind::SiluAndMul => vec![n * profile.intermediate_size() * 2],
        }
    }
    pub fn output_elements(&self) -> Vec<usize> {
        let n = self.call.rows();
        let profile = self.call.model_profile;
        match self.kind {
            Vllm08Kind::InputRms => vec![n * profile.hidden_size()],
            Vllm08Kind::ResidualRms(_) => {
                vec![n * profile.hidden_size(), n * profile.hidden_size()]
            }
            Vllm08Kind::Projection(p) => vec![if p == Vllm08Projection::Head {
                248320
            } else {
                n * profile.projection_output(p)
            }],
            Vllm08Kind::GdnConv => vec![n * 6144, n * 2048],
            Vllm08Kind::GdnRecurrent
            | Vllm08Kind::FlashAttention
            | Vllm08Kind::Sigmoid
            | Vllm08Kind::Multiply => vec![n * 2048],
            Vllm08Kind::AttentionQk => vec![n * 2048, n * 512, n * 2048],
            Vllm08Kind::KvAppend => vec![],
            Vllm08Kind::SiluAndMul => vec![n * profile.intermediate_size()],
        }
    }
    /// Data-only view of the actual full-phase normalizer; no new teacher.
    pub fn selected_row(&self) -> Option<usize> {
        (self.kind == Vllm08Kind::Projection(Vllm08Projection::Head))
            .then_some(self.call.rows() - 1)
    }
}

/// Core-minted result identity. Generic views/ops cannot manufacture this tag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vllm08Origin {
    call: Vllm08Call,
    layer: Option<Vllm08Layer>,
    kind: Vllm08Kind,
    node_index: usize,
    output_index: usize,
    inputs: Arc<[ValueId]>,
}
impl Vllm08Origin {
    pub(super) fn bind_call(&mut self, call: Vllm08Call) {
        self.call = call;
    }
    pub fn call(&self) -> &Vllm08Call {
        &self.call
    }
    pub fn layer(&self) -> Option<Vllm08Layer> {
        self.layer
    }
    pub fn kind(&self) -> Vllm08Kind {
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
    builder: &PlanBuilder,
    d: &Vllm08Operator,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    d.validate()?;
    if builder.vllm_model.as_ref() != Some(&d.call)
        || builder.operator_slice
        || builder.profile != ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1
        || builder.source != d.call.source()
        || inputs.len() != d.input_elements().len()
    {
        return Err(Error::Unsupported(
            "closed full-model source/profile/call/arity".into(),
        ));
    }
    for (input, elements) in inputs.iter().zip(d.input_elements()) {
        if input.precision != ActivationPrecision::Bf16WidenedF32
            || input.shape != ValueShape::Vector(elements)
            || input.elements != elements
        {
            return Err(Error::State("full-model original dense BF16 input".into()));
        }
    }
    let tag = vllm_native::require_invocation(builder, d.call.request(), inputs)?;
    // Stateful/cohort checks use immutable per-value tags, never a graph walk.
    let origin = |i: usize| -> Result<&Vllm08Origin> {
        match builder
            .vllm_value_provenance(&inputs[i])?
            .map(VllmValueProvenance::origin)
        {
            Some(VllmValueOrigin::FullModel(origin))
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
        Vllm08Kind::InputRms if tag.origin() != &VllmValueOrigin::SourceRows => {
            return Err(Error::State(
                "full-model input RMS requires source rows".into(),
            ));
        }
        Vllm08Kind::GdnConv => {
            for (i, kind) in [Vllm08Projection::Qkvz, Vllm08Projection::Ba]
                .into_iter()
                .enumerate()
            {
                if origin(i)?.kind != Vllm08Kind::Projection(kind) {
                    return Err(Error::State("full-model real QKVZ/BA producers".into()));
                }
            }
        }
        Vllm08Kind::GdnRecurrent => {
            let (conv, z, ba) = (origin(0)?, origin(1)?, origin(2)?);
            if conv.kind != Vllm08Kind::GdnConv
                || z.kind != Vllm08Kind::GdnConv
                || conv.output_index != 0
                || z.output_index != 1
                || conv.node_index != z.node_index
                || conv.inputs != z.inputs
                || conv.inputs.get(1) != Some(&inputs[2].id)
                || ba.kind != Vllm08Kind::Projection(Vllm08Projection::Ba)
            {
                return Err(Error::State(
                    "full-model real Conv/Z/original BA cohort".into(),
                ));
            }
        }
        Vllm08Kind::KvAppend => {
            let (k, packed) = (origin(0)?, origin(1)?);
            if k.kind != Vllm08Kind::AttentionQk
                || k.output_index != 1
                || packed.kind != Vllm08Kind::Projection(Vllm08Projection::AttentionQkv)
                || k.inputs.as_ref() != [inputs[1].id]
            {
                return Err(Error::State(
                    "full-model K and original packed-V producer cohort".into(),
                ));
            }
        }
        Vllm08Kind::FlashAttention => {
            if origin(0)?.kind != Vllm08Kind::AttentionQk || origin(0)?.output_index != 0 {
                return Err(Error::State(
                    "full-model real rotated query producer".into(),
                ));
            }
            if builder.vllm08_kv_cohorts.get(&d.layer.unwrap().index())
                != Some(&origin(0)?.node_index)
            {
                return Err(Error::State(
                    "FA query must match same-plan K/V append cohort".into(),
                ));
            }
        }
        Vllm08Kind::Projection(Vllm08Projection::Head)
            if origin(0)?.kind != Vllm08Kind::ResidualRms(Vllm08RmsSite::Final)
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
    builder: &PlanBuilder,
    d: &Vllm08Operator,
    inputs: &[PlanValue],
    output_index: usize,
) -> Result<VllmValueProvenance> {
    if output_index >= d.output_elements().len() {
        return Err(Error::State("full-model output provenance index".into()));
    }
    let tag = vllm_native::require_invocation(builder, d.call.request(), inputs)?;
    Ok(tag.with_origin(VllmValueOrigin::FullModel(Vllm08Origin {
        call: d.call.clone(),
        layer: d.layer,
        kind: d.kind,
        node_index: builder.nodes.len(),
        output_index,
        inputs: inputs.iter().map(|value| value.id).collect(),
    })))
}

pub(super) fn record(
    builder: &mut PlanBuilder,
    d: &Vllm08Operator,
    inputs: &[PlanValue],
) -> Result<Vec<PlanValue>> {
    // Validate before requesting a state effect; failed recording is atomic.
    precision(builder, d, inputs)?;
    for matrix in &d.matrices {
        builder.source_tensor(matrix)?;
    }
    let effect = match d.kind {
        Vllm08Kind::GdnConv => {
            Some(builder.effect(StateSlot::Conv(ConvDomain(d.layer.unwrap().index())))?)
        }
        Vllm08Kind::GdnRecurrent => Some(builder.effect(StateSlot::Recurrent(RecurrentDomain(
            d.layer.unwrap().index(),
        )))?),
        Vllm08Kind::KvAppend => {
            Some(builder.effect(StateSlot::Kv(KvDomain(d.layer.unwrap().index())))?)
        }
        _ => None,
    };
    let state = if d.kind == Vllm08Kind::FlashAttention {
        let current = builder.current(StateSlot::Kv(KvDomain(d.layer.unwrap().index())))?;
        if current.ordinal != 1 {
            return Err(Error::State(
                "FA requires same-plan private appended KV version".into(),
            ));
        }
        Some(current)
    } else {
        None
    };
    let values = builder.record(
        "vllm.original.full24",
        PlanOperation::TypedVllm08 {
            descriptor: Box::new(d.clone()),
            state,
        },
        inputs,
        &d.output_elements()
            .into_iter()
            .map(ValueShape::Vector)
            .collect::<Vec<_>>(),
        effect,
    )?;
    if d.kind == Vllm08Kind::KvAppend {
        let Some(VllmValueOrigin::FullModel(origin)) = builder
            .vllm_value_provenance(&inputs[0])?
            .map(VllmValueProvenance::origin)
        else {
            unreachable!()
        };
        let qk_node = origin.node_index;
        builder
            .vllm08_kv_cohorts
            .insert(d.layer.unwrap().index(), qk_node);
    }
    Ok(values)
}

#[derive(Clone, Debug)]
pub struct RecordedVllm08Model {
    plan: ForwardPlan,
    call: Vllm08Call,
    boundaries: Arc<[[ValueId; 2]]>,
    normalized: ValueId,
    logits: ValueId,
}
impl RecordedVllm08Model {
    pub const PREPARED_METADATA_BYTES: usize = 4 * 1024 * 1024;
    /// Retain the original host charge with every shared invariant allocation.
    /// This is lifetime metadata, never source authentication or execution authority.
    pub fn retain_prepared_metadata(
        mut self,
        metadata: crate::resource_arbiter::HostWorkspace,
    ) -> Result<Self> {
        if metadata.source_identity() != self.plan.source()
            || metadata.bytes() < Self::PREPARED_METADATA_BYTES
        {
            return Err(Error::State(
                "prepared record original host metadata charge".into(),
            ));
        }
        let semantics = Arc::get_mut(&mut self.plan.semantics)
            .ok_or_else(|| Error::State("prepared record metadata must precede sharing".into()))?;
        if semantics.metadata.is_some() {
            return Err(Error::State(
                "prepared record metadata already retained".into(),
            ));
        }
        semantics.metadata = Some(Arc::new(PreparedPlanMetadata(metadata)));
        Ok(self)
    }
    /// Invocation-local wrappers only; shared source/topology owners are priced at preparation.
    pub const BINDING_METADATA_BYTES: usize = 512 * 1024;

    /// Rebind only the existing sealed short-HF recipe, never a native/root permit.
    /// Every node and producer receives this call's authentic input issuer.
    pub fn bind_short_hf_call(&self, call: &ShortHfCall) -> Result<Self> {
        self.require_short_hf_shape(call)?;
        let mut bound = self.clone();
        bound.call = call.as_vllm_call().clone();
        bound.plan.vllm_fragment = Some(call.request().clone());
        for node in &mut bound.plan.nodes {
            match &mut node.operation {
                PlanOperation::TypedVllm08 { descriptor, .. } => {
                    descriptor.call = bound.call.clone();
                }
                PlanOperation::TypedVllmBf16Embedding { descriptor } => {
                    descriptor.bind_short_hf_teachers(call.request().teachers().clone());
                }
                _ => {
                    return Err(Error::State(
                        "prepared full-model contains a foreign operation".into(),
                    ));
                }
            }
        }
        for tag in &mut bound.plan.vllm_provenance {
            tag.as_mut()
                .ok_or_else(|| Error::State("prepared full-model producer coverage".into()))?
                .bind_short_hf_call(&bound.call)?;
        }
        bound.require_binding_metadata(Self::BINDING_METADATA_BYTES)?;
        Ok(bound)
    }
    /// Local record clone payload; shared Arc data and request/root owners are separately charged.
    pub fn binding_metadata_bytes(&self) -> Result<usize> {
        let overflow = || Error::Limit("prepared full-model binding metadata extent".into());
        let boxes = self.plan.nodes.iter().try_fold(0usize, |bytes, node| {
            let payload = match &node.operation {
                PlanOperation::TypedVllm08 { descriptor, .. } => {
                    std::mem::size_of_val(&**descriptor)
                }
                PlanOperation::TypedVllmBf16Embedding { descriptor } => {
                    std::mem::size_of_val(&**descriptor)
                }
                _ => {
                    return Err(Error::State(
                        "prepared full-model binding operation differs".into(),
                    ));
                }
            };
            bytes.checked_add(payload).ok_or_else(overflow)
        })?;
        // FullModel(Vllm08Origin) is inline in each producer tag; its SSA inputs,
        // teachers/call, node topology and source members are Arc-backed.
        // Only the two local Vecs and each actual typed descriptor Box allocate.
        let nodes = self
            .plan
            .nodes
            .capacity()
            .checked_mul(std::mem::size_of::<PlanNode>())
            .ok_or_else(overflow)?;
        let producers = self
            .plan
            .vllm_provenance
            .capacity()
            .checked_mul(std::mem::size_of::<Option<VllmValueProvenance>>())
            .ok_or_else(overflow)?;
        // Includes the caller's Arc<Record> header; request/phase/root owners have
        // separate existing allowances in the complete call inventory.
        [nodes, producers, boxes, 2 * std::mem::size_of::<usize>()]
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
        Arc::ptr_eq(&self.plan.semantics, &other.plan.semantics)
            && Arc::ptr_eq(&self.boundaries, &other.boundaries)
    }
    fn require_short_hf_shape(&self, call: &ShortHfCall) -> Result<()> {
        call.validate()?;
        let prepared = self
            .call
            .request()
            .teachers()
            .short_hf_span()
            .ok_or_else(|| {
                Error::Unsupported("prepared binding requires a short-HF recipe".into())
            })?;
        let actual = call.span();
        if self.call.source() != call.as_vllm_call().source()
            || self.call.model_profile() != call.model_profile()
            || prepared.workload() != actual.workload()
            || prepared.mode() != actual.mode()
            || prepared.base() != actual.base()
            || prepared.len() != actual.len()
            || prepared.capacity() != actual.capacity()
        {
            return Err(Error::State(
                "prepared short-HF source/profile/workload/phase/layout differs".into(),
            ));
        }
        Ok(())
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn call(&self) -> &Vllm08Call {
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
impl PlanBuilder {
    pub fn new_vllm08_model(call: Vllm08Call) -> Result<Self> {
        if call.model_profile() != FullModelProfile::Qwen08 {
            return Err(Error::Unsupported(
                "original .8B builder constructor profile".into(),
            ));
        }
        Self::new_full_model(call)
    }
    pub fn new_full_model(call: Vllm08Call) -> Result<Self> {
        call.validate()?;
        let mut builder = Self::new_inner(
            call.source(),
            512,
            all_domains()?,
            false,
            ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
            false,
        )?;
        builder.vllm_fragment = Some(call.request.clone());
        builder.vllm_model = Some(call);
        Ok(builder)
    }
    /// Whole arithmetic record remains unpublished: no logits authority/frontier.
    pub fn finish_vllm08_model(
        self,
        boundaries: Vec<[PlanValue; 2]>,
        normalized: PlanValue,
        logits: PlanValue,
    ) -> Result<RecordedVllm08Model> {
        if self
            .vllm_model
            .as_ref()
            .is_some_and(|call| call.model_profile() != FullModelProfile::Qwen08)
        {
            return Err(Error::Unsupported(
                "original .8B finish constructor profile".into(),
            ));
        }
        self.finish_full_model(boundaries, normalized, logits)
    }
    pub fn finish_full_model(
        self,
        boundaries: Vec<[PlanValue; 2]>,
        normalized: PlanValue,
        logits: PlanValue,
    ) -> Result<RecordedVllm08Model> {
        let call = self
            .vllm_model
            .clone()
            .ok_or_else(|| Error::State("not full-model builder".into()))?;
        if self.advanced.is_some()
            || self.operator_slice
            || self.required.len() != 42
            || self.required != all_domains()?.into_iter().collect()
            || self.domains.len() != 42
            || boundaries.len() != 24
            || boundaries.capacity() > 24
        {
            return Err(Error::State(
                "full24 record requires all42 domains/24 boundaries/no frontier".into(),
            ));
        }
        for slot in &self.required {
            if self.current(*slot)?.ordinal != 1 {
                return Err(Error::State(
                    "one actual transition per original indexed domain".into(),
                ));
            }
        }
        let require = |value: &PlanValue,
                       layer: Option<Vllm08Layer>,
                       kind: Vllm08Kind,
                       out: usize|
         -> Result<()> {
            match self
                .vllm_value_provenance(value)?
                .map(VllmValueProvenance::origin)
            {
                Some(VllmValueOrigin::FullModel(origin))
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
                Vllm08Kind::Projection(Vllm08Projection::Down),
                0,
            )?;
            require(
                residual,
                layer,
                Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention),
                1,
            )?;
        }
        require(
            &normalized,
            None,
            Vllm08Kind::ResidualRms(Vllm08RmsSite::Final),
            0,
        )?;
        require(
            &logits,
            None,
            Vllm08Kind::Projection(Vllm08Projection::Head),
            0,
        )?;
        verify_recipe(&self, &call, &boundaries, normalized.id, logits.id)?;
        let state_node_count = self.nodes.len();
        Ok(RecordedVllm08Model {
            call,
            boundaries: boundaries.into_iter().map(|v| [v[0].id, v[1].id]).collect(),
            normalized: normalized.id,
            logits: logits.id,
            plan: ForwardPlan {
                vllm_fragment: self.vllm_fragment,
                vllm_provenance: self.vllm_provenance,
                nodes: self.nodes,
                semantics: Arc::new(ForwardPlanSemantics {
                    metadata: None,
                    source: self.source,
                    profile: self.profile,
                    standalone_exp: None,
                    operator_slice: false,
                    precisions: self.precisions,
                    prefix_capacity: self.prefix_capacity,
                    values: self.values,
                    constants: self.constants,
                    required_domains: self.required.into_iter().collect(),
                    proposed_domains: self.domains.into_values().collect(),
                    logits: None,
                    frontier_delta: 0,
                    state_node_count,
                }),
            },
        })
    }
}

/// Linear closed-recipe check, not a provenance graph search. It ties every
/// named boundary to the actual earlier node's ValueIds before any lowerer.
fn verify_recipe(
    builder: &PlanBuilder,
    call: &Vllm08Call,
    boundaries: &[[PlanValue; 2]],
    normalized: ValueId,
    logits: ValueId,
) -> Result<()> {
    let embedding = builder
        .nodes
        .first()
        .ok_or_else(|| Error::State("source rows node absent".into()))?;
    let PlanOperation::TypedVllmBf16Embedding { descriptor } = &embedding.operation else {
        return Err(Error::State(
            "full24 must begin with original source rows".into(),
        ));
    };
    descriptor.validate()?;
    if descriptor.model_profile() != call.model_profile()
        || descriptor.teachers() != call.request().teachers()
        || !embedding.inputs.is_empty()
        || embedding.outputs.len() != 1
        || embedding.state.is_some()
    {
        return Err(Error::State(
            "full24 raw original source/teacher producer".into(),
        ));
    }
    let raw = embedding.outputs[0];
    let mut cursor = 1usize;
    let mut take = |layer: Option<Vllm08Layer>,
                    kind: Vllm08Kind,
                    inputs: &[ValueId]|
     -> Result<Vec<ValueId>> {
        let node = builder
            .nodes
            .get(cursor)
            .ok_or_else(|| Error::State("full24 recipe node absent".into()))?;
        let PlanOperation::TypedVllm08 { descriptor, state } = &node.operation else {
            return Err(Error::State("full24 exact typed source recipe".into()));
        };
        descriptor.validate()?;
        if descriptor.call() != call
            || descriptor.layer() != layer
            || descriptor.kind() != kind
            || node.inputs != inputs
            || node.outputs.len() != descriptor.output_elements().len()
        {
            return Err(Error::State(
                "full24 actual node call/layer/source/SSA recipe".into(),
            ));
        }
        let slot = match kind {
            Vllm08Kind::GdnConv => Some(StateSlot::Conv(ConvDomain(layer.unwrap().index()))),
            Vllm08Kind::GdnRecurrent => Some(StateSlot::Recurrent(RecurrentDomain(
                layer.unwrap().index(),
            ))),
            Vllm08Kind::KvAppend => Some(StateSlot::Kv(KvDomain(layer.unwrap().index()))),
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
            (Vllm08Kind::FlashAttention, Some(version))
                if version.slot == StateSlot::Kv(KvDomain(layer.unwrap().index()))
                    && version.ordinal == 1 => {}
            (Vllm08Kind::FlashAttention, _) => {
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
                take(Some(layer), Vllm08Kind::InputRms, &[branch])?[0],
                branch,
            )
        } else {
            let previous =
                residual.ok_or_else(|| Error::State("full24 pending carried residual".into()))?;
            let dual = take(
                Some(layer),
                Vllm08Kind::ResidualRms(Vllm08RmsSite::Input),
                &[branch, previous],
            )?;
            (dual[0], dual[1])
        };
        let projected = if layer.is_attention() {
            let packed = take(
                Some(layer),
                Vllm08Kind::Projection(Vllm08Projection::AttentionQkv),
                &[normal],
            )?[0];
            let qk = take(Some(layer), Vllm08Kind::AttentionQk, &[packed])?;
            take(Some(layer), Vllm08Kind::KvAppend, &[qk[1], packed])?;
            let fa = take(Some(layer), Vllm08Kind::FlashAttention, &[qk[0]])?[0];
            let sigmoid = take(Some(layer), Vllm08Kind::Sigmoid, &[qk[2]])?[0];
            let gated = take(Some(layer), Vllm08Kind::Multiply, &[fa, sigmoid])?[0];
            take(
                Some(layer),
                Vllm08Kind::Projection(Vllm08Projection::AttentionOut),
                &[gated],
            )?[0]
        } else {
            let packed = take(
                Some(layer),
                Vllm08Kind::Projection(Vllm08Projection::Qkvz),
                &[normal],
            )?[0];
            let ba = take(
                Some(layer),
                Vllm08Kind::Projection(Vllm08Projection::Ba),
                &[normal],
            )?[0];
            let conv = take(Some(layer), Vllm08Kind::GdnConv, &[packed, ba])?;
            let gated = take(
                Some(layer),
                Vllm08Kind::GdnRecurrent,
                &[conv[0], conv[1], ba],
            )?[0];
            take(
                Some(layer),
                Vllm08Kind::Projection(Vllm08Projection::GdnOut),
                &[gated],
            )?[0]
        };
        let post = take(
            Some(layer),
            Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention),
            &[projected, carried],
        )?;
        let gate_up = take(
            Some(layer),
            Vllm08Kind::Projection(Vllm08Projection::GateUp),
            &[post[0]],
        )?[0];
        let activated = take(Some(layer), Vllm08Kind::SiluAndMul, &[gate_up])?[0];
        branch = take(
            Some(layer),
            Vllm08Kind::Projection(Vllm08Projection::Down),
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
        Vllm08Kind::ResidualRms(Vllm08RmsSite::Final),
        &[branch, residual.unwrap()],
    )?;
    let head = take(
        None,
        Vllm08Kind::Projection(Vllm08Projection::Head),
        &[final_norm[0]],
    )?;
    if final_norm[0] != normalized || head[0] != logits || cursor != builder.nodes.len() {
        return Err(Error::State(
            "full24 finalnorm/fullV and exact no-extra recipe".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "vllm_model/profile_tests.rs"]
mod profile_tests;

#[cfg(test)]
#[path = "vllm_model/span_tests.rs"]
mod span_tests;

#[cfg(test)]
mod prepared_tests;
