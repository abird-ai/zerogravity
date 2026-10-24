//! Install private scratch ownership on the SAME original armed submission.
use super::*;
use crate::vllm_cuda_input_rms::FullModelRmsScratch;
impl ResidentSubmission<'_> {
    pub(crate) fn require_rms_actor_accesses(
        &self,
        required: &[(ResidentLease, DeviceAccess)],
    ) -> Result<()> {
        self.require_accesses(required.iter().map(|(lease, access)| (lease, *access)))
    }

    pub(crate) fn require_rms_scratch_scope(&self, scope: &Arc<FullModelRmsScratch>) -> Result<()> {
        if !self.armed || self.work.is_none() {
            return Err(Error::State(
                "RMS pool needs original armed physical flight".into(),
            ));
        }
        scope.require_submitted(self.service, self)
    }
}
