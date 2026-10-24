//! Exact raw original Q/K gamma binding for the computed canonical47 node33.
use super::super::super::canonical47_model_prefix::Canonical47PrefixCut;
use super::*;
impl ModelAttentionParameters {
    pub(crate) fn require_canonical47_qk(
        &self,
        cut: &Arc<Canonical47PrefixCut>,
        values: &Arc<ModelValues>,
    ) -> Result<[ResidentLease; 2]> {
        cut.scope().require_values(values)?;
        let node = cut
            .scope()
            .record()
            .plan()
            .nodes()
            .get(33)
            .ok_or_else(|| Error::State("canonical47 actual QK gamma node absent".into()))?;
        let PlanOperation::TypedVllm08 {
            descriptor,
            state: None,
        } = &node.operation
        else {
            return Err(Error::State(
                "canonical47 QK gamma stateless operator".into(),
            ));
        };
        if !cut.includes_kv()
            || descriptor.kind() != Vllm08Kind::AttentionQk
            || descriptor.call() != values.record().call()
            || descriptor.layer() != Some(self.layer)
            || self.layer.index() != 3
            || !self.layer.is_attention()
            || !Arc::ptr_eq(&self.source, cut.scope().source().source())
            || descriptor.parameters().len() != 2
            || !descriptor.matrices().is_empty()
        {
            return Err(Error::State(
                "canonical47 actual original QK parameter/source/call owners".into(),
            ));
        }
        for (index, role) in [
            QwenParameterRole::QueryNorm(3),
            QwenParameterRole::KeyNorm(3),
        ]
        .into_iter()
        .enumerate()
        {
            self.source
                .validate_parameter(role, &descriptor.parameters()[index])?;
            if self.payload.windows[index].parameter().source_parameter()
                != descriptor.parameters()[index]
                || self.leases[index].bytes() != 512
                || self.leases[index].role() != BufferRole::ImmutableWeights
                || !Arc::ptr_eq(self.leases[index].service(), values.service())
                || self.leases[index].service_generation() != values.service().generation()
                || !self.leases[index].is_initialized()
            {
                return Err(Error::State(
                    "canonical47 actual initialized raw QK gamma lease/header".into(),
                ));
            }
        }
        Ok(self.leases.clone())
    }
}
