//! One defined native arena for each actual whole47 source layer.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut;
impl ModelCache {
    pub(crate) async fn prepare_canonical47_whole(
        cut: &Arc<Canonical47PrefixCut>,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        cut.require_whole()?;
        initial.require_canonical47_whole(cut)?;
        if attempt.base().frontier() != 0 || attempt.frontier() != 47 {
            return Err(Error::Unsupported("whole cache initial47 only".into()));
        }
        Self::prepare_checked(
            cut.scope().source().source().clone(),
            service,
            attempt,
            initial,
            History::Canonical47Whole,
            Some(cut.scope().clone()),
            cancelled,
        )
        .await
    }
    pub(crate) fn retained_whole_binding(
        &self,
        cut: &Arc<Canonical47PrefixCut>,
        binding: &BoundModelState,
    ) -> Result<bool> {
        binding.require_canonical47_whole(cut)?;
        Ok(self.history == History::Canonical47Whole
            && binding.same_request(&self.initial)
            && self.base == binding.base()
            && self.layer == binding.layer()
            && Arc::ptr_eq(&self.source, cut.scope().source().source())
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
