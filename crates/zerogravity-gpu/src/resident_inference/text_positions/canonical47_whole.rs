//! Same exact position integer producer with a whole24 original attempt seal.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model::Canonical47CallKind;
use crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut;
impl ResidentSession {
    pub(in crate::resident_inference) fn bind_canonical47_whole_text_positions(
        &self,
        cut: &Arc<Canonical47PrefixCut>,
        attempt: &RootAttempt,
        phase: Arc<AttemptPhaseBinding>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<BoundTextPositions>> {
        not_cancelled(cancelled)?;
        let scope = cut.scope();
        let source = scope.source().source().clone();
        let call = scope.record().call();
        if !cut.is_whole() || scope.kind() != Canonical47CallKind::Prefill {
            return Err(Error::Unsupported(
                "closed canonical47 QK/KV position cut".into(),
            ));
        }
        require_position_source(&source)?;
        phase.require_canonical47_whole_binding(self, attempt, call)?;
        let base = self.binding();
        if base != attempt.base()
            || base.frontier() != 0
            || attempt.frontier() != 47
            || call.request().teachers().base() != 0
            || call.rows() != 47
            || call.request().teachers().capacity() != 512
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
        let metadata = source.source().reserve_workspace(METADATA_BYTES)?;
        let span = PositionSpan { base: 0, rows: 47 };
        span.bytes_canonical47()?;
        Ok(Arc::new(BoundTextPositions {
            source,
            service: self.model.service.clone(),
            invocation: call.request().clone(),
            base,
            attempt: attempt.clone(),
            phase: Some(phase),
            canonical47: Some(scope.clone()),
            span,
            service_generation: self.model.service.generation(),
            _metadata: metadata,
        }))
    }
}
