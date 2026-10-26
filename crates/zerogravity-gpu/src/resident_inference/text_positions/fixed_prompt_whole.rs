//! Same exact position integer producer with a whole24 original attempt seal.
use super::*;
use crate::cuda::native::fixed_prompt_model::FixedPromptCallKind;
use crate::cuda::native::fixed_prompt_model_prefix::FixedPromptPrefixCut;
impl ResidentSession {
    pub(in crate::resident_inference) fn bind_fixed_prompt_whole_text_positions(
        &self,
        cut: &Arc<FixedPromptPrefixCut>,
        attempt: &RootAttempt,
        phase: Arc<AttemptPhaseBinding>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<BoundTextPositions>> {
        not_cancelled(cancelled)?;
        let scope = cut.scope();
        let source = scope.source().clone();
        let call = scope.record().call();
        if !cut.is_whole() || scope.kind() != FixedPromptCallKind::Prefill {
            return Err(Error::Unsupported(
                "closed canonical47 QK/KV position cut".into(),
            ));
        }
        phase.require_fixed_prompt_whole_binding(self, attempt, call)?;
        let base = self.binding();
        if base != attempt.base()
            || base.frontier() != 0
            || attempt.frontier() != 47
            || call.rows().base() != 0
            || call.rows().len() != 47
            || call.rows().capacity() != 512
            || self.model.capacity != 512
            || source.source_identity() != base.source()
            || self.model.service.source_identity() != base.source()
        {
            return Err(Error::State(
                "canonical47 source positions original attempt/root/service".into(),
            ));
        }
        self.model.service.ensure_usable()?;
        self.model.validate_domains(&self.domains, true)?;
        let metadata = source.reserve_workspace(METADATA_BYTES)?;
        let span = PositionSpan { base: 0, rows: 47 };
        span.bytes_fixed_prompt()?;
        Ok(Arc::new(BoundTextPositions {
            source,
            _record: scope.record().clone(),
            service: self.model.service.clone(),
            invocation: call.clone(),
            base,
            attempt: attempt.clone(),
            phase: Some(phase),
            fixed_prompt: Some(scope.clone()),
            span,
            service_generation: self.model.service.generation(),
            _metadata: metadata,
        }))
    }
}
