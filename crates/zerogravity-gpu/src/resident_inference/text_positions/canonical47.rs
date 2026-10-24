//! Exact original Prefill47 position issuer; legacy five spans remain closed.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model::Canonical47CallKind;
use crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut;

impl ResidentSession {
    pub(in crate::resident_inference) fn bind_canonical47_text_positions(
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
        if !cut.is_attention_cut() || scope.kind() != Canonical47CallKind::Prefill {
            return Err(Error::Unsupported(
                "closed canonical47 QK/KV position cut".into(),
            ));
        }
        require_position_source(&source)?;
        phase.require_canonical47_model_prefix_binding(self, attempt, call)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical47_active_three_axis_words_and_defined_padding_are_separate_from_legacy() {
        let span = PositionSpan { base: 0, rows: 47 };
        assert!(span.bytes().is_err());
        let raw = span.bytes_canonical47().unwrap();
        for axis in 0..3 {
            for row in 0..65 {
                let start = (65 * axis + row) * 8;
                let actual = i64::from_le_bytes(raw[start..start + 8].try_into().unwrap());
                assert_eq!(actual, if row < 47 { row as i64 } else { 0 });
            }
        }
        for (base, rows) in [(1, 47), (0, 46), (0, 48), (0, 64), (0, 1)] {
            assert!(PositionSpan { base, rows }.bytes_canonical47().is_err());
        }
    }
}
