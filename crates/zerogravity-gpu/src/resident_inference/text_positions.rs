//! Original source/request position control; no Root, model or KV authority.
use super::*;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_inference::native_fragment::continuation::AttemptPhaseBinding;
use crate::resident_program::source_bindings::SourceBindings;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{SourceCall, SourceCallMode};

pub(crate) const POSITION_PITCH: usize = 65;
pub(crate) const POSITION_WORDS: usize = 3 * POSITION_PITCH;
pub(crate) const POSITION_BYTES: usize = POSITION_WORDS * 8;
const METADATA_BYTES: usize = 64 * 1024;

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("source positions cancelled".into()));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PositionSpan {
    base: usize,
    rows: usize,
}
impl PositionSpan {
    fn new(invocation: &SourceCall) -> Result<Self> {
        let value = Self {
            base: invocation.rows().base(),
            rows: invocation.rows().len(),
        };
        if invocation.rows().capacity() != 512
            || invocation.rows().vocabulary() != 248320
            || !matches!(
                (invocation.mode(), value.base, value.rows),
                (SourceCallMode::Prefill, 0, 1 | 64)
                    | (SourceCallMode::Prefill, 64, 2)
                    | (SourceCallMode::Decode, 1 | 66, 1)
            )
        {
            return Err(Error::State(
                "positions use exact original request issuer, not an attention-layer retag".into(),
            ));
        }
        value.validate()?;
        Ok(value)
    }
    fn validate(self) -> Result<()> {
        if !matches!(
            (self.base, self.rows),
            (0, 1) | (1, 1) | (0, 64) | (64, 2) | (66, 1)
        ) || self.rows >= POSITION_PITCH
            || self.base.checked_add(self.rows).is_none_or(|end| end > 512)
            || i64::try_from(self.base).is_err()
        {
            return Err(Error::Unsupported(
                "closed original five text-only position spans".into(),
            ));
        }
        Ok(())
    }
    fn bytes(self) -> Result<[u8; POSITION_BYTES]> {
        self.validate()?;
        self.encode()
    }
    fn bytes_fixed_prompt(self) -> Result<[u8; POSITION_BYTES]> {
        if self != (Self { base: 0, rows: 47 }) {
            return Err(Error::Unsupported(
                "closed canonical47 source position span".into(),
            ));
        }
        self.encode()
    }
    fn encode(self) -> Result<[u8; POSITION_BYTES]> {
        let mut bytes = [0; POSITION_BYTES];
        for axis in 0..3 {
            for row in 0..self.rows {
                let position = self
                    .base
                    .checked_add(row)
                    .and_then(|v| i64::try_from(v).ok())
                    .ok_or_else(|| Error::Limit("text position integer overflow".into()))?;
                let offset = (axis * POSITION_PITCH + row) * 8;
                bytes[offset..offset + 8].copy_from_slice(&position.to_le_bytes());
            }
        }
        // Private defined padding. No claim about original inactive GPU lanes.
        Ok(bytes)
    }
}
fn require_published_span(
    current: RootBinding,
    attempt: &RootAttempt,
    invocation: &SourceCall,
) -> Result<PositionSpan> {
    let span = PositionSpan::new(invocation)?;
    if current != attempt.base()
        || current.source() != invocation.source()
        || u64::try_from(span.base).ok() != Some(current.frontier())
        || u64::try_from(span.base + span.rows).ok() != Some(attempt.frontier())
    {
        return Err(Error::State(
            "original published-root position attempt/span differs".into(),
        ));
    }
    Ok(span)
}

/// Immutable source/request binding only. Original session stays sole authority.
pub(crate) struct BoundTextPositions {
    source: VerifiedTensorSource,
    _record: Arc<SourceBindings>,
    service: Arc<ResidentService>,
    invocation: SourceCall,
    base: RootBinding,
    attempt: RootAttempt,
    phase: Option<Arc<AttemptPhaseBinding>>,
    fixed_prompt: Option<Arc<crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding>>,
    span: PositionSpan,
    service_generation: u64,
    _metadata: WindowWorkspace,
}
impl ResidentSession {
    /// Existing fragment continuation supplies its real private phase binding.
    /// A complete-model caller uses the actual published root/advance instead.
    pub(super) fn bind_text_positions(
        &self,
        record: Arc<SourceBindings>,
        attempt: &RootAttempt,
        phase: Option<Arc<AttemptPhaseBinding>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<BoundTextPositions>> {
        not_cancelled(cancelled)?;
        let source = record.source().clone();
        let invocation = record.call().clone();
        let metadata = source.reserve_workspace(METADATA_BYTES)?;
        let base = self.binding();
        if source.source_identity() != base.source()
            || source.source_identity() != self.model.service.source_identity()
            || invocation.source() != base.source()
            || invocation.rows().capacity() != self.model.capacity
        {
            return Err(Error::State(
                "same original position model/service/session/source".into(),
            ));
        }
        let span = if let Some(binding) = &phase {
            binding.require_binding(self, attempt, &invocation, 0)?;
            PositionSpan::new(&invocation)?
        } else {
            require_published_span(base, attempt, &invocation)?
        };
        self.model.service.ensure_usable()?;
        self.model.validate_domains(&self.domains, true)?;
        Ok(Arc::new(BoundTextPositions {
            source,
            _record: record,
            service: self.model.service.clone(),
            invocation: invocation.clone(),
            base,
            attempt: attempt.clone(),
            phase,
            fixed_prompt: None,
            span,
            service_generation: self.model.service.generation(),
            _metadata: metadata,
        }))
    }
}
impl BoundTextPositions {
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.service
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn bytes(&self) -> Result<[u8; POSITION_BYTES]> {
        match &self.fixed_prompt {
            Some(scope) => {
                if scope.record().call() != &self.invocation {
                    return Err(Error::State(
                        "canonical47 positions changed original call issuer".into(),
                    ));
                }
                self.span.bytes_fixed_prompt()
            }
            None => self.span.bytes(),
        }
    }
    pub(crate) fn invocation(&self) -> &SourceCall {
        &self.invocation
    }
    /// Fresh borrowed original session + prior completion permit, before QK/RoPE.
    pub(crate) fn require_enqueue(
        &self,
        session: &ResidentSession,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        if session.binding() != self.base
            || !Arc::ptr_eq(&session.model.service, &self.service)
            || self.service.generation() != self.service_generation
            || self.source.source_identity() != self.base.source()
        {
            return Err(Error::State(
                "source position root/epoch/service revoked".into(),
            ));
        }
        if let Some(scope) = &self.fixed_prompt {
            let binding = self
                .phase
                .as_ref()
                .ok_or_else(|| Error::State("canonical47 original position phase absent".into()))?;
            let permit = permit.ok_or_else(|| {
                Error::State("canonical47 original position permit absent".into())
            })?;
            if scope.record().call() != &self.invocation
                || !scope.source().same_issuer(&self.source)
            {
                return Err(Error::State(
                    "canonical47 position source/call owner changed".into(),
                ));
            }
            if binding.fixed_prompt_whole_position_scope() {
                binding.require_fixed_prompt_whole_binding(
                    session,
                    &self.attempt,
                    scope.record().call(),
                )?;
                return binding.require_fixed_prompt_whole_permit(
                    permit,
                    &self.service,
                    scope.record().call(),
                );
            }
            binding.require_fixed_prompt_model_prefix_binding(
                session,
                &self.attempt,
                scope.record().call(),
            )?;
            return binding.require_fixed_prompt_model_prefix_permit(
                permit,
                &self.service,
                scope.record().call(),
            );
        }
        match (&self.phase, permit) {
            (Some(binding), Some(permit)) => {
                binding.require_binding(session, &self.attempt, &self.invocation, 0)?;
                binding.require_permit(permit, &self.service, &self.invocation, 0)
            }
            (None, None) => {
                require_published_span(session.binding(), &self.attempt, &self.invocation)
                    .map(|_| ())
            }
            _ => Err(Error::State(
                "position original continuation permit absent/foreign".into(),
            )),
        }
    }
}

#[cfg(test)]
#[path = "text_positions/tests.rs"]
mod tests;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod fixed_prompt;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[path = "text_positions/fixed_prompt_whole.rs"]
mod fixed_prompt_whole;
