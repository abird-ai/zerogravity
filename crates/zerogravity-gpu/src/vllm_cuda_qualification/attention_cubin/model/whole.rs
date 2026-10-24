//! Six exact source gamma pairs; no layer3 parameter reuse.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut;
impl ModelAttentionParameters {
    pub(crate) fn require_whole_qk(
        &self,
        cut: &Arc<Canonical47PrefixCut>,
        values: &Arc<ModelValues>,
        layer: usize,
    ) -> Result<[ResidentLease; 2]> {
        cut.require_whole()?;
        cut.scope().require_values(values)?;
        let site = cut.attention_site(layer)?;
        let node = &cut.scope().record().plan().nodes()[site.nodes()[1]];
        let PlanOperation::TypedVllm08 {
            descriptor,
            state: None,
        } = &node.operation
        else {
            return Err(Error::State("whole source QK operator".into()));
        };
        if descriptor.kind() != Vllm08Kind::AttentionQk
            || descriptor.call() != values.record().call()
            || descriptor.layer() != Some(self.layer)
            || self.layer != site.layer()
            || !Arc::ptr_eq(&self.source, cut.scope().source().source())
            || descriptor.parameters().len() != 2
            || !descriptor.matrices().is_empty()
        {
            return Err(Error::State("whole QK actual source gamma owner".into()));
        }
        for (i, role) in [
            QwenParameterRole::QueryNorm(layer),
            QwenParameterRole::KeyNorm(layer),
        ]
        .into_iter()
        .enumerate()
        {
            self.source
                .validate_parameter(role, &descriptor.parameters()[i])?;
            let lease = &self.leases[i];
            if self.payload.windows[i].parameter().source_parameter() != descriptor.parameters()[i]
                || lease.bytes() != 512
                || lease.role() != BufferRole::ImmutableWeights
                || !lease.is_initialized()
                || !Arc::ptr_eq(lease.service(), values.service())
                || lease.service_generation() != values.service().generation()
            {
                return Err(Error::State("whole QK source/generation/extent".into()));
            }
        }
        Ok(self.leases.clone())
    }
}
