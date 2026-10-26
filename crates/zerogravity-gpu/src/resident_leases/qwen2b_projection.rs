//! Exact original pending upload; no completion/readiness/lease authority.
use super::*;
impl ResidentInitialization<'_> {
    pub(crate) fn require_qwen2b_projection_upload(&self, lease: &ResidentLease) -> Result<()> {
        self.require_pending_original_projection_upload(lease)
    }
    pub(crate) fn require_fixed_prompt_projection_upload(
        &self,
        lease: &ResidentLease,
    ) -> Result<()> {
        self.require_pending_original_projection_upload(lease)
    }
    fn require_pending_original_projection_upload(&self, lease: &ResidentLease) -> Result<()> {
        self.service.ensure_usable()?;
        self.service.check_lease(lease)?;
        if !self.armed
            || self.work.is_none()
            || self.events.is_empty()
            || lease.role() != BufferRole::ImmutableWeights
            || lease.is_initialized()
            || !self.uploaded.contains(&lease.id())
            || self
                .leases
                .iter()
                .filter(|actual| actual.id() == lease.id())
                .count()
                != 1
        {
            return Err(Error::State(
                "source2B W must belong to this SAME actual pending H2D initializer".into(),
            ));
        }
        Ok(())
    }
}
