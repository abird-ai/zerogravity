//! One defined native arena for each actual whole47 source layer.
use super::*;
use crate::cuda::native::fixed_prompt_model_prefix::FixedPromptPrefixCut;
impl ModelCache {
    pub(crate) async fn prepare_fixed_prompt_whole(
        cut: &Arc<FixedPromptPrefixCut>,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        cut.require_whole()?;
        initial.require_fixed_prompt_whole(cut)?;
        if attempt.base().frontier() != 0 || attempt.frontier() != 47 {
            return Err(Error::Unsupported("whole cache initial47 only".into()));
        }
        Self::prepare_checked(
            cut.scope().source().clone(),
            service,
            attempt,
            initial,
            History::FixedPromptWhole,
            Some(cut.scope().clone()),
            cancelled,
        )
        .await
    }
    pub(crate) fn retained_whole_binding(
        &self,
        cut: &Arc<FixedPromptPrefixCut>,
        binding: &BoundModelState,
    ) -> Result<bool> {
        binding.require_fixed_prompt_whole(cut)?;
        Ok(self.history == History::FixedPromptWhole
            && binding.same_request(&self.initial)
            && self.base == binding.base()
            && self.layer == binding.layer()
            && self.source.same_issuer(cut.scope().source())
            && self.lease.is_initialized()
            && self.lease.bytes() == BYTES
            && self.lease.source_identity() == binding.call().source()
            && self.lease.role() == zerogravity_core::device_events::BufferRole::RecurrentState
            && binding.leases().iter().all(|(lease, _)| {
                self.lease.id() != lease.id()
                    && Arc::ptr_eq(self.lease.service(), lease.service())
                    && self.lease.service_generation() == lease.service_generation()
            }))
    }
}
