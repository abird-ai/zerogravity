//! Fresh canonical call/operator seal over identical source gamma and RMS storage.
use super::*;
use crate::resident_inference::native_fragment::canonical47_natural::CanonicalCallInput;
use crate::vllm_cuda_qualification::canonical47_model::Canonical47CallKind;

impl FusedRmsPrepared {
    pub(crate) fn bind_canonical_call(
        self: &Arc<Self>,
        input: &Arc<CanonicalCallInput>,
        node: usize,
        branch: &ResidentLease,
        residual: &ResidentLease,
    ) -> Result<Arc<Self>> {
        if self.pooled.is_some() {
            return Err(Error::Unsupported(
                "pooled RMS requires fresh short-HF epoch binding".into(),
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
        let old_kind = Canonical47CallKind::of(a.call().request())?;
        let new_kind = Canonical47CallKind::of(b.call().request())?;
        let same_geometry = matches!(
            (old_kind, new_kind),
            (Canonical47CallKind::Prefill, Canonical47CallKind::Prefill)
                | (
                    Canonical47CallKind::Decode { .. },
                    Canonical47CallKind::Decode { .. }
                )
        );
        if !same_geometry
            || b.call() != input.call()
            || a.kind() != b.kind()
            || a.layer() != b.layer()
            || a.matrices() != b.matrices()
            || a.parameters() != b.parameters()
            || a.call().source() != b.call().source()
            || a.call().model_profile() != b.call().model_profile()
            || a.call().rows() != b.call().rows()
        {
            return Err(Error::State(
                "canonical RMS source/site/storage geometry differs".into(),
            ));
        }
        crate::resident_program::natural_binding::same_lease(
            self.lease(Buffer::Attention),
            branch,
        )?;
        crate::resident_program::natural_binding::same_lease(
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
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            _short_hf_binding: None,
        });
        result.require_full_model_operator(input.scope().rms(node)?.operator())?;
        Ok(result)
    }
}
