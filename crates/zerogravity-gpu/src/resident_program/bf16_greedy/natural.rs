//! Actual new Head/status seal, existing original reducer/packet storage.
use super::*;
use crate::resident_program::natural_binding::{same_lease, same_recipe};
impl PreparedBf16Greedy {
    pub(crate) fn bind_natural(self: &Arc<Self>, values: Arc<ModelValues>) -> Result<Arc<Self>> {
        same_recipe(self.owner.values.record(), values.record())?;
        require_head(values.record())?;
        same_lease(
            &self.logits,
            values.lease(values.record().selection().output())?,
        )?;
        same_lease(&self.status, values.status())?;
        let metadata = self.owner.source.reserve_workspace(METADATA_BYTES)?;
        let call = values.record().call().clone();
        let owner = Arc::new(HostOwner {
            values,
            source: self.owner.source.clone(),
            _metadata: metadata,
        });
        let result = Arc::new(Self {
            service: self.service.clone(),
            logits: self.logits.clone(),
            partials: self.partials.clone(),
            packet: self.packet.clone(),
            status: self.status.clone(),
            accesses: self.accesses.clone(),
            outputs: self.outputs.clone(),
            call,
            logits_value: self.logits_value,
            owner,
        });
        result.validate_bindings()?;
        Ok(result)
    }
}
