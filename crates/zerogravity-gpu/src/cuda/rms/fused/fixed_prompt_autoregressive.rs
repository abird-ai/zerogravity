//! Fresh canonical call/operator seal over identical source gamma and RMS storage.
use super::*;
use crate::cuda::native::fixed_prompt_model::FixedPromptInvocationKind;
use crate::resident_inference::native_fragment::fixed_prompt_autoregressive::FixedPromptInvocationInput;

impl FusedRmsPrepared {
    pub(crate) fn bind_fixed_prompt_call(
        self: &Arc<Self>,
        input: &Arc<FixedPromptInvocationInput>,
        node: usize,
        branch: &ResidentLease,
        residual: &ResidentLease,
    ) -> Result<Arc<Self>> {
        if self.pooled.is_some() {
            return Err(Error::Unsupported(
                "pooled RMS requires fresh text epoch binding".into(),
            ));
        }
        let binding = input.scope().rms(node)?.clone();
        let old = self
            .model_binding
            .as_ref()
            .or(self.owner._model_binding.as_ref())
            .ok_or_else(|| {
                Error::State("canonical cold RMS needs actual model gamma source".into())
            })?;
        let (a, b) = (old.operator(), binding.operator());
        binding.require_source(&self.owner.source)?;
        let old_kind = FixedPromptInvocationKind::of(a.invocation())?;
        let new_kind = FixedPromptInvocationKind::of(b.invocation())?;
        let same_geometry = matches!(
            (old_kind, new_kind),
            (
                FixedPromptInvocationKind::Prefill,
                FixedPromptInvocationKind::Prefill
            ) | (
                FixedPromptInvocationKind::Decode { .. },
                FixedPromptInvocationKind::Decode { .. }
            )
        );
        if !same_geometry
            || b.invocation() != input.invocation()
            || a.kind() != b.kind()
            || a.matrices() != b.matrices()
            || a.parameters() != b.parameters()
            || a.invocation().source() != b.invocation().source()
            || a.numerical_contract() != b.numerical_contract()
            || a.invocation().tokens().capacity() != b.invocation().tokens().capacity()
            || a.invocation().tokens().vocabulary() != b.invocation().tokens().vocabulary()
            || a.invocation().tokens().len() != b.invocation().tokens().len()
        {
            return Err(Error::State(
                "canonical RMS source/site/storage geometry differs".into(),
            ));
        }
        crate::resident_program::autoregressive_binding::same_lease(
            self.lease(Buffer::Attention),
            branch,
        )?;
        crate::resident_program::autoregressive_binding::same_lease(
            self.lease(Buffer::PriorResidual),
            residual,
        )?;
        let result = Arc::new(Self {
            service: self.service.clone(),
            leases: self.leases.clone(),
            stages: self.stages.clone(),
            accesses: self.accesses.clone(),
            owner: self.owner.clone(),
            model_binding: Some(binding),
            pooled: None,
            #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
            _text_binding: None,
        });
        result.require_full_model_operator(input.scope().rms(node)?.operator())?;
        Ok(result)
    }
}
