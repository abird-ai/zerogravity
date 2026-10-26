//! Actual source calls and original current-root/committed-packet ownership.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule, PrefixPhaseEnqueue};
use super::*;
use crate::cuda::gdn::GdnKernelRecipe;
use crate::cuda::native::text_model::{TextModelBinding, TextModelBindings};
use crate::cuda::native::text_native::TextKernelCapture;
use crate::resident_program::source_bindings::SourceBindings;
use std::sync::Weak;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{SourceCall, SourceCallMode};

// Previous allowance: QwenPlan::BINDING_METADATA_BYTES.
const CALL_METADATA_BYTES: usize = 512 * 1024;

enum Origin {
    Prompt,
    Feedback(Arc<CommittedTextPacket>),
}

// No public constructor accepts ResidentToken.id or caller-selected sampled IDs.
pub(crate) struct CommittedTextPacket {
    prompt_tokens: usize,
    source: VerifiedTensorSource,
    model: Weak<ResidentModel>,
    token: ResidentToken,
    root: RootBinding,
    packet_bytes: [u8; 12],
    allocation_generation: u64,
    service_generation: u64,
}
impl CommittedTextPacket {
    fn require_current(
        &self,
        session: &ResidentSession,
        source: &VerifiedTensorSource,
    ) -> Result<u32> {
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("text feedback original model retired".into()))?;
        let id = u32::from_le_bytes(self.packet_bytes[..4].try_into().unwrap());
        let bits = u32::from_le_bytes(self.packet_bytes[4..8].try_into().unwrap());
        if !Arc::ptr_eq(&model, &session.model)
            || !self.source.same_issuer(source)
            || self.root != session.binding()
            || self.token.producer != Some(self.root)
            || id != self.token.id
            || bits != self.token.max_bits
            || u64::from(id) >= session.model.vocab
            || bits & 0xffff != 0
            || bits & 0x7f80_0000 == 0x7f80_0000
            || self.packet_bytes[8..] != [0; 4]
            || !Arc::ptr_eq(self.token.lease.service(), &session.model.service)
            || self.token.lease.role() != BufferRole::Output
            || self.token.lease.bytes() != 12
            || !self.token.lease.is_initialized()
            || self.token.lease.allocation_generation() != self.allocation_generation
            || self.token.lease.service_generation() != self.service_generation
        {
            return Err(Error::State(
                "text feedback is stale/foreign or not the original committed packet".into(),
            ));
        }
        session.model.service.ensure_usable()?;
        Ok(id)
    }
}

/// Read-only model-preparation data borrowed from the actual committed packet.
/// Holding it adds no packet/root owner; consumption rechecks the physical join.
pub struct TextFeedbackRequest<'a> {
    session: &'a TextSession,
    packet: &'a Arc<CommittedTextPacket>,
    base: usize,
}
impl TextFeedbackRequest<'_> {
    pub fn selected_id(&self) -> u32 {
        self.packet.token.id
    }
    pub fn base(&self) -> usize {
        self.base
    }
    pub fn prompt_tokens(&self) -> usize {
        self.packet.prompt_tokens
    }
    fn require(
        &self,
        session: &TextSession,
        source: &VerifiedTensorSource,
    ) -> Result<&Arc<CommittedTextPacket>> {
        if !std::ptr::eq(self.session, session)
            || !session
                .feedback
                .as_ref()
                .is_some_and(|p| Arc::ptr_eq(p, self.packet))
            || self.base as u64 != session.binding().frontier()
        {
            return Err(Error::State(
                "text feedback request belongs to another current session".into(),
            ));
        }
        self.packet.require_current(&session.original, source)?;
        Ok(self.packet)
    }
}

fn require_feedback_coordinates(
    call: &SourceCall,
    root: RootBinding,
    prompt_tokens: usize,
    selected_id: u32,
) -> Result<()> {
    if call.mode() != SourceCallMode::Decode
        || call.source() != root.source()
        || call.rows().base() as u64 != root.frontier()
        || call.rows().len() != 1
        || call.rows().ids() != [selected_id]
        || call.rows().base() < prompt_tokens
    {
        return Err(Error::State(
            "text bound call differs from actual committed feedback".into(),
        ));
    }
    Ok(())
}

fn embedding_member(
    record: &SourceBindings,
) -> Result<&zerogravity_core::accelerator::program::SourceTensor> {
    use zerogravity_core::accelerator::program::SourceOperatorKind;
    let descriptor = record
        .operator(0)?
        .ok_or_else(|| Error::State("text actual embedding descriptor absent".into()))?;
    let [matrix] = descriptor.matrices() else {
        return Err(Error::State("text embedding exact source matrix".into()));
    };
    if descriptor.kind() != SourceOperatorKind::EmbeddingRows {
        return Err(Error::State("text actual initial embedding differs".into()));
    }
    Ok(matrix)
}
fn hidden_width(record: &SourceBindings) -> Result<usize> {
    Ok(embedding_member(record)?.columns)
}

fn head_member(
    record: &SourceBindings,
) -> Result<&zerogravity_core::accelerator::program::SourceTensor> {
    use zerogravity_core::accelerator::program::SourceOperatorKind;
    let index = record
        .plan()
        .source_value_provenance(record.selection().output())?
        .ok_or_else(|| Error::State("text actual head producer absent".into()))?
        .node_index();
    let descriptor = record
        .operator(index)?
        .ok_or_else(|| Error::State("text actual head descriptor absent".into()))?;
    let [matrix] = descriptor.matrices() else {
        return Err(Error::State("text head exact source matrix".into()));
    };
    if !matches!(
        descriptor.kind(),
        SourceOperatorKind::Projection {
            selected_row: Some(_)
        }
    ) {
        return Err(Error::State("text actual selected head differs".into()));
    }
    Ok(matrix)
}

// No strong Model reference: native/current-flight owners cannot form a Body cycle.
pub(crate) struct TextCallInput {
    model: Weak<ResidentModel>,
    source: VerifiedTensorSource,
    service: Arc<ResidentService>,
    attempt: RootAttempt,
    scope: Arc<TextModelBinding>,
    schedule: Arc<GdnAttemptSchedule>,
    phase: Arc<AttemptPhaseBinding>,
    origin: Origin,
    prompt_tokens: usize,
    _metadata: WindowWorkspace,
}
impl TextCallInput {
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn scope(&self) -> &Arc<TextModelBinding> {
        &self.scope
    }
    pub(crate) fn record(&self) -> &Arc<SourceBindings> {
        self.scope.record()
    }
    pub(crate) fn call(&self) -> &SourceCall {
        self.scope.call()
    }
    /// Original immutable committed packet; checked again before cohort permits.
    pub(crate) fn feedback_lease(&self) -> Result<Option<&ResidentLease>> {
        match &self.origin {
            Origin::Prompt => Ok(None),
            Origin::Feedback(packet)
                if packet.root == self.attempt.base()
                    && packet.token.producer == Some(packet.root) =>
            {
                Ok(Some(&packet.token.lease))
            }
            Origin::Feedback(_) => {
                Err(Error::State("cohort feedback producer/root differs".into()))
            }
        }
    }
    pub(crate) fn ids(&self) -> &[u32] {
        self.call().rows().ids()
    }
    pub(crate) fn attempt(&self) -> &RootAttempt {
        &self.attempt
    }
    pub(in crate::resident_inference) fn phase(&self) -> &Arc<AttemptPhaseBinding> {
        &self.phase
    }
    pub(crate) fn base(&self) -> usize {
        self.call().rows().base()
    }
    pub(crate) fn rows(&self) -> usize {
        self.call().rows().len()
    }
    pub(crate) fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }
    pub(crate) fn call_ordinal(&self) -> usize {
        if self.call().mode() == SourceCallMode::Prefill {
            0
        } else {
            self.base() - self.prompt_tokens + 1
        }
    }
    pub(crate) fn require_session(&self, session: &ResidentSession) -> Result<()> {
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("text source model retired".into()))?;
        if !Arc::ptr_eq(&model, &session.model)
            || !Arc::ptr_eq(&self.service, &session.model.service)
            || !self.source.same_issuer(self.scope.source())
            || self.service.source_identity() != self.call().source()
            || self.source.source_identity() != session.binding().source()
            || session.model.capacity != self.call().rows().capacity()
            || Some(session.model.vocab) != u64::try_from(self.call().rows().vocabulary()).ok()
        {
            return Err(Error::State(
                "text SAME source/model/service/current call".into(),
            ));
        }
        self.phase
            .require_text_binding(session, &self.attempt, self.call())?;
        match (&self.origin, self.call().mode()) {
            (Origin::Prompt, SourceCallMode::Prefill) => {
                if self.base() != 0 || self.prompt_tokens != self.rows() {
                    return Err(Error::State("text actual prompt coordinates differ".into()));
                }
            }
            (Origin::Feedback(packet), SourceCallMode::Decode) => {
                if packet.prompt_tokens != self.prompt_tokens
                    || self.base() < self.prompt_tokens
                    || self.ids() != [packet.require_current(session, &self.source)?]
                {
                    return Err(Error::State(
                        "text actual committed greedy feedback/prompt differs".into(),
                    ));
                }
            }
            _ => return Err(Error::State("text prompt/feedback origin differs".into())),
        }
        Ok(())
    }
    pub(crate) fn require_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
    ) -> Result<()> {
        self.phase.require_text_permit(permit, service, self.call())
    }
    pub(crate) fn permit<'a>(
        &'a self,
        session: &'a ResidentSession,
    ) -> Result<PrefixPhaseEnqueue<'a>> {
        self.require_session(session)?;
        PrefixPhaseEnqueue::new(session, &self.schedule, 0, None)
    }
}
impl ResidentSession {
    pub(crate) fn bind_text_prompt(
        &self,
        prepared: Arc<TextModelBindings>,
        record: Arc<SourceBindings>,
        recipe: GdnKernelRecipe,
        capture: TextKernelCapture,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<TextCallInput>> {
        if self.binding().frontier() != 0
            || record.call().mode() != SourceCallMode::Prefill
            || record.call().rows().base() != 0
        {
            return Err(Error::State(
                "text prompt requires the original empty root".into(),
            ));
        }
        let prompt_tokens = record.call().rows().len();
        self.bind_text_input(
            prepared,
            record,
            recipe,
            capture,
            prompt_tokens,
            Origin::Prompt,
            cancelled,
        )
    }
    pub(crate) fn bind_text_feedback(
        &self,
        prepared: Arc<TextModelBindings>,
        record: Arc<SourceBindings>,
        recipe: GdnKernelRecipe,
        capture: TextKernelCapture,
        packet: Arc<CommittedTextPacket>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<TextCallInput>> {
        not_cancelled(cancelled)?;
        let id = packet.require_current(self, prepared.source())?;
        require_feedback_coordinates(record.call(), self.binding(), packet.prompt_tokens, id)?;
        let prompt_tokens = packet.prompt_tokens;
        self.bind_text_input(
            prepared,
            record,
            recipe,
            capture,
            prompt_tokens,
            Origin::Feedback(packet),
            cancelled,
        )
    }
    fn bind_text_input(
        &self,
        prepared: Arc<TextModelBindings>,
        record: Arc<SourceBindings>,
        recipe: GdnKernelRecipe,
        capture: TextKernelCapture,
        prompt_tokens: usize,
        origin: Origin,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<TextCallInput>> {
        not_cancelled(cancelled)?;
        let source = prepared.source().clone();
        if !source.same_issuer(record.source())
            || source.source_identity() != self.binding().source()
            || self.model.capacity != record.call().rows().capacity()
            || Some(self.model.vocab) != u64::try_from(record.call().rows().vocabulary()).ok()
        {
            return Err(Error::State("text original source/session differs".into()));
        }
        let metadata = source.reserve_workspace(CALL_METADATA_BYTES)?;
        let scope = prepared.bind(record, recipe, capture)?;
        let attempt = self
            .authority
            .begin_advance(scope.call().rows().len() as u64)?;
        let schedule = GdnAttemptSchedule::new_text(&scope, attempt.clone())?;
        let phase = AttemptPhaseBinding::new(schedule.clone(), 0, None)?;
        let input = Arc::new(TextCallInput {
            model: Arc::downgrade(&self.model),
            source,
            service: self.model.service.clone(),
            attempt,
            scope,
            schedule,
            phase,
            origin,
            prompt_tokens,
            _metadata: metadata,
        });
        input.require_session(self)?;
        Ok(input)
    }
}

mod state;
pub(crate) use state::TextCallState;
mod readback;
pub use readback::TextReadback;

mod loaded;
pub use loaded::{
    TextCallReceipt, TextCloseReceipt, TextCohortFault, TextCohortFaultOutcome,
    TextCohortFaultRetention, TextCohortPreparation, TextCohortReadback, TextCohortReceipt,
    TextCohortRowReceipt, TextEmptySnapshot, TextFault, TextFaultOutcome, TextLoaded,
    TextLoadedObservation, TextLoadedSnapshot, TextPending, TextPendingCohort,
    TextQualificationInputs, TextResourceInventory, TextSession, prepare_text, prepare_text_b4,
};
mod completion;
type TextValidation<'a> = &'a mut dyn for<'b> FnMut(TextReadback<'b>) -> Result<()>;

#[cfg(test)]
mod tests;
