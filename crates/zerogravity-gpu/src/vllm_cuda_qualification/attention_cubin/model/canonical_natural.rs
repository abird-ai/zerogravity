//! Exact current-call source gamma binding; no legacy span conversion.
use super::*;
use crate::resident_inference::native_fragment::canonical47_natural::CanonicalCallInput;
impl ModelAttentionParameters {
    pub(crate) fn require_canonical_call(
        &self,
        input: &CanonicalCallInput,
        values: &Arc<ModelValues>,
        node: usize,
    ) -> Result<[ResidentLease; 2]> {
        input.scope().require_values(values)?;
        let PlanOperation::TypedVllm08 {
            descriptor,
            state: None,
        } = &input
            .record()
            .plan()
            .nodes()
            .get(node)
            .ok_or_else(|| Error::State("current QK node absent".into()))?
            .operation
        else {
            return Err(Error::State("current QK descriptor absent".into()));
        };
        if descriptor.kind() != Vllm08Kind::AttentionQk
            || descriptor.layer() != Some(self.layer)
            || descriptor.call() != input.call()
            || !Arc::ptr_eq(&self.source, input.source().source())
            || descriptor.parameters().len() != 2
            || !descriptor.matrices().is_empty()
        {
            return Err(Error::State("current source QK parameter seal".into()));
        }
        for (i, role) in [
            QwenParameterRole::QueryNorm(self.layer.index()),
            QwenParameterRole::KeyNorm(self.layer.index()),
        ]
        .into_iter()
        .enumerate()
        {
            self.source
                .validate_parameter(role, &descriptor.parameters()[i])?;
            let l = &self.leases[i];
            if self.payload.windows[i].parameter().source_parameter() != descriptor.parameters()[i]
                || l.bytes() != 512
                || l.role() != BufferRole::ImmutableWeights
                || !l.is_initialized()
                || !Arc::ptr_eq(l.service(), values.service())
                || l.service_generation() != values.service().generation()
            {
                return Err(Error::State("current QK source/generation/extent".into()));
            }
        }
        Ok(self.leases.clone())
    }
}
