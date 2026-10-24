//! Actual body/actor/storage join; no current-attempt authority is minted here.
use super::*;
use crate::vllm_cuda_input_rms::FullModelRmsScratch;
impl FullModelProgram {
    pub(crate) fn require_rms_scratch(
        &self,
        scope: &Arc<FullModelRmsScratch>,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        if !Arc::ptr_eq(self.record(), scope.record())
            || self.fused.len() != 48
            || !Arc::ptr_eq(self.values().service(), scope.service())
            || self
                .record()
                .call()
                .request()
                .teachers()
                .short_hf_span()
                .is_none()
        {
            return Err(Error::State(
                "pooled RMS SAME source/full body/record/service".into(),
            ));
        }
        self.rms.require_rms_scratch(scope, 1)?;
        let mut accesses = self.rms.accesses().to_vec();
        for (node, actor) in &self.fused {
            actor.require_rms_scratch(scope, *node)?;
            accesses.extend(actor.accesses().iter().cloned());
        }
        scope.merge_accesses(accesses)
    }
}
