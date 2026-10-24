//! Cold source binding and bounded CPU reference driver. No tokenizer/serving grant.
mod portable;
mod resources;
use super::*;
pub use portable::{
    Gemma4Bf16Inventory, Gemma4Bf16NormSource, Gemma4Bf16Observation, Gemma4Bf16TextPlan,
    Gemma4SourceTile, Gemma4SourceTiles,
};
pub use resources::Gemma4CpuInventory;
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::{
    quant::QuantKind,
    session_root::{RootBinding, SessionAuthority},
};
use zerogravity_cpu::reference_text::{
    ReferenceTextArithmetic, ReferenceTextMath, ReferenceTextObserver, ReferenceTextParameter,
    ReferenceTextSource, ReferenceTextState, ReferenceTextValue, WindowReferenceTextSource,
};

pub const E2B_REPOSITORY: &str = "google/gemma-4-E2B-it";
pub const E2B_REVISION: &str = "3e22461f65e89153144f8adb70e3b8c2cc9845a7";
/// Independent caller authority over an already verified immutable source.
/// An unsharded checkpoint needs an authenticated header-derived index closure;
/// this is not a claim that E2B publishes an official HF index.
pub struct Gemma4SourcePins<'a> {
    pub repository: &'a str,
    pub revision: &'a str,
    pub config_sha256: &'a str,
    pub source_index_sha256: &'a str,
    pub source_bundle_sha256: &'a str,
}
pub struct OriginalGemma4Text {
    geometry: Geometry,
    source: Arc<WindowReferenceTextSource>,
    matrices: BTreeMap<Gemma4MatrixRole, SourceTensor>,
    parameters: BTreeMap<Gemma4ParameterRole, ReferenceTextParameter>,
    config_sha256: String,
    experts: BTreeMap<Gemma4ExpertRole, TextExpertTensor>,
    _metadata: WindowWorkspace,
}
impl OriginalGemma4Text {
    pub fn bind(
        source: VerifiedTensorSource,
        config_json: &[u8],
        pins: Gemma4SourcePins<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        cancelled_check(cancelled)?;
        if config_json.len() > 1024 * 1024 {
            return Err(Error::Limit("Gemma original config exceeds1MiB".into()));
        }
        for hash in [
            pins.config_sha256,
            pins.source_index_sha256,
            pins.source_bundle_sha256,
        ] {
            if hash.len() != 64
                || !hash
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            {
                return Err(Error::Malformed(
                    "complete original Gemma source SHA256 pins".into(),
                ));
            }
        }
        let config = Gemma4Config::from_json(config_json)?;
        let (repository, revision) = match config.variant() {
            crate::Gemma4Variant::E2b => (E2B_REPOSITORY, E2B_REVISION),
            crate::Gemma4Variant::Moe26bA4b => (
                crate::variant_26b::INSTRUCTION_MODEL_ID,
                crate::variant_26b::INSTRUCTION_REVISION,
            ),
            crate::Gemma4Variant::Dense31b => (
                crate::variant_31b::INSTRUCTION_MODEL_ID,
                crate::variant_31b::INSTRUCTION_REVISION,
            ),
        };
        if pins.repository != repository
            || pins.revision != revision
            || format!("{:x}", Sha256::digest(config_json)) != pins.config_sha256
            || source.verified_shard_index_sha256() != pins.source_index_sha256
            || source.verified_bundle_sha256() != pins.source_bundle_sha256
        {
            return Err(Error::State(
                "Gemma original repository/revision/config/index/bundle identity".into(),
            ));
        }
        let geometry = Geometry::from_config(&config)?;
        let metadata = source.reserve_workspace(2 * 1024 * 1024)?;
        let specs = config.tensor_specs()?;
        let names = specs
            .iter()
            .map(|s| s.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for name in source.tensor_names() {
            if (name.starts_with("model.language_model.") || name == "lm_head.weight")
                && !names.contains(name)
            {
                return Err(Error::Unsupported(format!(
                    "unregistered original Gemma text member {name}"
                )));
            }
        }
        for spec in &specs {
            match source.tensor_info(&spec.name) {
                Ok(raw) => {
                    let shape = raw
                        .shape
                        .iter()
                        .map(|x| {
                            usize::try_from(*x)
                                .map_err(|_| Error::Limit("Gemma source tensor shape".into()))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    spec.validate_shape(&shape)?;
                    if !matches!(raw.kind, QuantKind::Bf16 | QuantKind::F32) {
                        return Err(Error::Unsupported(format!(
                            "original Gemma BF16/F32 member {}",
                            spec.name
                        )));
                    }
                }
                Err(error) if spec.required => return Err(error),
                Err(_) => {}
            }
        }
        let source = Arc::new(WindowReferenceTextSource::new(source));
        let matrices = matrix_names(&geometry)
            .into_iter()
            .map(|(role, name)| Ok((role, source.matrix(&name)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let experts = expert_names(&geometry)
            .into_iter()
            .map(|(role, name)| Ok((role, source.expert_tensor(&name)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let parameters = parameter_names(&geometry)
            .into_iter()
            .map(|(role, name)| Ok((role, source.parameter(&name, cancelled)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        cancelled_check(cancelled)?;
        Ok(Self {
            geometry,
            source,
            matrices,
            parameters,
            config_sha256: pins.config_sha256.into(),
            experts,
            _metadata: metadata,
        })
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source.source_identity()
    }
    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        self.source.source()
    }
    pub fn matrix_descriptor(&self, role: Gemma4MatrixRole) -> Result<SourceTensor> {
        self.matrix(role)
    }
    pub fn parameter_descriptor(&self, role: Gemma4ParameterRole) -> Result<&SourceParameter> {
        Ok(&self
            .parameters
            .get(&role)
            .ok_or_else(|| Error::Unsupported("absent original Gemma parameter role".into()))?
            .descriptor)
    }
    pub fn record_reference(self: &Arc<Self>, capacity: usize) -> Result<Gemma4ReferencePlan> {
        if capacity == 0 || capacity > 8192 || capacity > self.geometry.max_position {
            return Err(Error::Limit("bounded Gemma reference plan capacity".into()));
        }
        let metadata = self.source.source().reserve_workspace(4 * 1024 * 1024)?;
        let mut math = PlanBuilder::new_with_profile(
            self.source_identity(),
            capacity,
            self.geometry
                .domains()
                .into_iter()
                .map(StateSlot::Kv)
                .collect(),
            true,
            ProgramNumericalProfile::AnalyticF32TextV1,
        )?;
        let logits = forward(self.as_ref(), &mut math, PlanBuilder::inputs(), true)?;
        let plan = math.finish(logits)?;
        Ok(Gemma4ReferencePlan {
            plan,
            _source: self.clone(),
            _metadata: metadata,
        })
    }
    pub fn trained_model_qualified(&self) -> bool {
        false
    }
}
impl Weights for OriginalGemma4Text {
    fn source_identity(&self) -> SourceIdentity {
        self.source_identity()
    }
    fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    fn matrix(&self, role: Gemma4MatrixRole) -> Result<SourceTensor> {
        self.matrices.get(&role).cloned().ok_or_else(|| {
            Error::Unsupported(
                "absent original Gemma matrix role/shared KV has no projection".into(),
            )
        })
    }
    fn expert(&self, role: Gemma4ExpertRole) -> Result<TextExpertTensor> {
        self.experts
            .get(&role)
            .cloned()
            .ok_or_else(|| Error::Unsupported("absent original sparse source role".into()))
    }
    fn parameter(&self, role: Gemma4ParameterRole) -> Result<SourceVector<'_>> {
        let parameter = self.parameters.get(&role).ok_or_else(|| {
            Error::Unsupported("absent original Gemma parameter role/shared KV has no norm".into())
        })?;
        Ok(SourceVector {
            source: parameter.descriptor.source,
            name: &parameter.descriptor.name,
            values: &parameter.values,
        })
    }
}
pub struct Gemma4ReferencePlan {
    plan: ForwardPlan,
    _source: Arc<OriginalGemma4Text>,
    _metadata: WindowWorkspace,
}
impl Gemma4ReferencePlan {
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
}

/// Borrowed completed proposal, before original SessionAuthority publication.
/// Expected data can reject it; it cannot supply operands or alter KV.
pub struct Gemma4ReferenceReadback<'a> {
    before: RootBinding,
    state: &'a ReferenceTextState,
    logits: Option<&'a ReferenceTextValue>,
}
impl Gemma4ReferenceReadback<'_> {
    pub fn root_before(&self) -> RootBinding {
        self.before
    }
    pub fn frontier(&self) -> usize {
        self.state.frontier()
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.state.source_identity()
    }
    pub fn logits(&self) -> Result<Option<&[f32]>> {
        self.logits.map(ReferenceTextValue::as_slice).transpose()
    }
    pub fn kv_row(&self, domain: KvDomain, token: usize) -> Result<(&[f32], &[f32])> {
        self.state.domain_row(domain, token)
    }
    pub fn domain_tokens(&self, domain: KvDomain) -> Result<usize> {
        self.state.domain_tokens(domain)
    }
}
pub struct Gemma4ReferenceSession {
    model: Arc<OriginalGemma4Text>,
    state: ReferenceTextState,
    authority: SessionAuthority,
    arithmetic: ReferenceTextArithmetic,
}
impl Gemma4ReferenceSession {
    pub fn new(model: Arc<OriginalGemma4Text>, capacity: usize) -> Result<Self> {
        Self::new_arithmetic(model, capacity, ReferenceTextArithmetic::AnalyticF32)
    }
    /// Whole E2B cast-schedule candidate only. No trained/recorded/native eligibility.
    pub fn new_bf16_candidate(model: Arc<OriginalGemma4Text>, capacity: usize) -> Result<Self> {
        let g = &model.geometry;
        if g.hidden != 1536
            || g.vocab != 262144
            || g.layers.len() != 35
            || g.ple != 256
            || g.moe.is_some()
            || model.matrices.values().any(|x| x.kind != QuantKind::Bf16)
            || model
                .parameters
                .values()
                .any(|x| x.descriptor.kind != QuantKind::Bf16)
        {
            return Err(Error::Unsupported(
                "E2B candidate requires authentic whole BF16 source geometry".into(),
            ));
        }
        Self::new_arithmetic(model, capacity, ReferenceTextArithmetic::Bf16EagerCandidate)
    }
    fn new_arithmetic(
        model: Arc<OriginalGemma4Text>,
        capacity: usize,
        arithmetic: ReferenceTextArithmetic,
    ) -> Result<Self> {
        if capacity > model.geometry.max_position {
            return Err(Error::Limit("Gemma source context limit".into()));
        }
        let state = ReferenceTextState::new_with_arithmetic(
            model.source.as_ref(),
            &model.geometry.domains(),
            capacity,
            arithmetic,
        )?;
        let authority = SessionAuthority::new(model.source_identity())?;
        Ok(Self {
            model,
            state,
            authority,
            arithmetic,
        })
    }
    pub fn binding(&self) -> RootBinding {
        self.authority.binding()
    }
    pub fn forward_reference(
        &mut self,
        tokens: &[u32],
        need_logits: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Option<ReferenceTextValue>> {
        if self.arithmetic == ReferenceTextArithmetic::AnalyticF32 {
            evaluate(
                self.model.as_ref(),
                self.model.source.as_ref(),
                &mut self.state,
                &mut self.authority,
                tokens,
                need_logits,
                cancelled,
            )
        } else {
            self.forward_reference_with(tokens, need_logits, cancelled, |_| Ok(()))
        }
    }
    /// The same evaluator and original publication; callback failure/cancel rolls back.
    pub fn forward_reference_with(
        &mut self,
        tokens: &[u32],
        need_logits: bool,
        cancelled: Option<&AtomicBool>,
        validate: impl FnMut(Gemma4ReferenceReadback<'_>) -> Result<()>,
    ) -> Result<Option<ReferenceTextValue>> {
        if self.arithmetic == ReferenceTextArithmetic::AnalyticF32 {
            evaluate_with(
                self.model.as_ref(),
                self.model.source.as_ref(),
                &mut self.state,
                &mut self.authority,
                tokens,
                need_logits,
                cancelled,
                validate,
            )
        } else {
            self.forward_reference_observed(tokens, need_logits, cancelled, None, validate)
        }
    }
    pub fn forward_reference_observed(
        &mut self,
        tokens: &[u32],
        need_logits: bool,
        cancelled: Option<&AtomicBool>,
        observer: Option<&dyn ReferenceTextObserver>,
        validate: impl FnMut(Gemma4ReferenceReadback<'_>) -> Result<()>,
    ) -> Result<Option<ReferenceTextValue>> {
        evaluate_configured(
            self.model.as_ref(),
            self.model.source.as_ref(),
            &mut self.state,
            &mut self.authority,
            tokens,
            need_logits,
            EvaluationOptions {
                cancelled,
                observer,
                arithmetic: self.arithmetic,
            },
            validate,
        )
    }
    pub fn arithmetic(&self) -> ReferenceTextArithmetic {
        self.arithmetic
    }
    pub fn kv_row(&self, domain: KvDomain, token: usize) -> Result<(&[f32], &[f32])> {
        self.state.domain_row(domain, token)
    }
    pub fn trained_model_qualified(&self) -> bool {
        false
    }
}
fn cancelled_check(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|x| x.load(Ordering::Acquire)) {
        Err(Error::State("reference text cancelled".into()))
    } else {
        Ok(())
    }
}
pub(super) fn evaluate(
    weights: &impl Weights,
    source: &dyn ReferenceTextSource,
    state: &mut ReferenceTextState,
    authority: &mut SessionAuthority,
    tokens: &[u32],
    need_logits: bool,
    cancelled: Option<&AtomicBool>,
) -> Result<Option<ReferenceTextValue>> {
    evaluate_with(
        weights,
        source,
        state,
        authority,
        tokens,
        need_logits,
        cancelled,
        |_| Ok(()),
    )
}
#[expect(clippy::too_many_arguments)]
pub(super) fn evaluate_with(
    weights: &impl Weights,
    source: &dyn ReferenceTextSource,
    state: &mut ReferenceTextState,
    authority: &mut SessionAuthority,
    tokens: &[u32],
    need_logits: bool,
    cancelled: Option<&AtomicBool>,
    validate: impl FnMut(Gemma4ReferenceReadback<'_>) -> Result<()>,
) -> Result<Option<ReferenceTextValue>> {
    evaluate_configured(
        weights,
        source,
        state,
        authority,
        tokens,
        need_logits,
        EvaluationOptions {
            cancelled,
            observer: None,
            arithmetic: ReferenceTextArithmetic::AnalyticF32,
        },
        validate,
    )
}
pub(super) struct EvaluationOptions<'a> {
    pub cancelled: Option<&'a AtomicBool>,
    pub observer: Option<&'a dyn ReferenceTextObserver>,
    pub arithmetic: ReferenceTextArithmetic,
}
struct ArithmeticWeights<'a, W: Weights> {
    base: &'a W,
    arithmetic: ReferenceTextArithmetic,
}
impl<W: Weights> Weights for ArithmeticWeights<'_, W> {
    fn source_identity(&self) -> SourceIdentity {
        self.base.source_identity()
    }
    fn geometry(&self) -> &Geometry {
        self.base.geometry()
    }
    fn activation_precision(&self) -> ActivationPrecision {
        self.arithmetic.precision()
    }
    fn matrix(&self, role: Gemma4MatrixRole) -> Result<SourceTensor> {
        self.base.matrix(role)
    }
    fn expert(&self, role: Gemma4ExpertRole) -> Result<TextExpertTensor> {
        self.base.expert(role)
    }
    fn parameter(&self, role: Gemma4ParameterRole) -> Result<SourceVector<'_>> {
        self.base.parameter(role)
    }
}
#[expect(clippy::too_many_arguments)]
pub(super) fn evaluate_configured(
    weights: &impl Weights,
    source: &dyn ReferenceTextSource,
    state: &mut ReferenceTextState,
    authority: &mut SessionAuthority,
    tokens: &[u32],
    need_logits: bool,
    options: EvaluationOptions<'_>,
    mut validate: impl FnMut(Gemma4ReferenceReadback<'_>) -> Result<()>,
) -> Result<Option<ReferenceTextValue>> {
    let cancelled = options.cancelled;
    cancelled_check(cancelled)?;
    if tokens.is_empty()
        || tokens.len() > 8192
        || tokens
            .iter()
            .any(|id| *id as usize >= weights.geometry().vocab)
        || source.source_identity() != weights.source_identity()
        || state.source_identity() != weights.source_identity()
        || authority.binding().source() != weights.source_identity()
        || authority.binding().frontier() != state.frontier() as u64
    {
        return Err(Error::State(
            "Gemma reference source/session/tokens/frontier".into(),
        ));
    }
    let attempt = authority.begin_advance(tokens.len() as u64)?;
    if attempt.frontier() > weights.geometry().max_position as u64 {
        return Err(Error::Limit("Gemma reference source positions".into()));
    }
    let mut proposed = state.fork(source)?;
    let mut logits = None;
    for (index, token) in tokens.iter().enumerate() {
        cancelled_check(cancelled)?;
        let position = proposed.frontier();
        let mut math = ReferenceTextMath::new_observed(
            source,
            &mut proposed,
            cancelled,
            options.arithmetic,
            options.observer,
        )?;
        if options.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
            math.bind_bf16_attention_call(
                tokens.len(),
                usize::try_from(attempt.frontier())
                    .map_err(|_| Error::Limit("candidate attention call frontier".into()))?,
            )?;
        }
        let weights = ArithmeticWeights {
            base: weights,
            arithmetic: options.arithmetic,
        };
        logits = forward(
            &weights,
            &mut math,
            ForwardInputs {
                token: *token as usize,
                position,
                valid_tokens: position + 1,
            },
            need_logits && index + 1 == tokens.len(),
        )?;
    }
    proposed.validate_complete()?;
    cancelled_check(cancelled)?;
    if proposed.frontier() as u64 != attempt.frontier() {
        return Err(Error::State(
            "Gemma complete source-private frontier".into(),
        ));
    }
    validate(Gemma4ReferenceReadback {
        before: authority.binding(),
        state: &proposed,
        logits: logits.as_ref(),
    })?;
    cancelled_check(cancelled)?;
    let publication = authority.prepare_publish(attempt)?;
    *state = proposed;
    publication.commit();
    Ok(logits)
}
