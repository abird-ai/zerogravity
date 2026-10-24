//! Same original47 source-row/RMS producer, with the full-record SSA seal.
use super::*;
use crate::resident_leases::ResidentService;
use zerogravity_core::accelerator::program::{
    FullModelProfile, PlanOperation, RecordedVllm08Model, Vllm08Kind, VllmBf16Invocation,
    VllmBf16Mode,
};
use zerogravity_model_qwen35::original_binding::BoundFullModelSource;

fn require_component_request(request: &VllmBf16Invocation) -> Result<()> {
    request.validate()?;
    if request.teachers().short_hf_span().is_some()
        || request.mode() != VllmBf16Mode::Prefill
        || request.teachers().base() != 0
        || request.teachers().capacity() != 512
        || request.rows() != COUNT
        || request.teachers().ids() != PROMPT
        || request.legacy_phase().is_ok()
    {
        return Err(Error::Unsupported(
            "source-owned canonical47 InputRms request only".into(),
        ));
    }
    Ok(())
}
impl RmsPrepared {
    pub(crate) fn validate_canonical47_component_source(
        bound: &BoundFullModelSource,
    ) -> Result<()> {
        if bound.model_profile() != FullModelProfile::Qwen08 {
            return Err(Error::Unsupported("original47 .8B source family".into()));
        }
        Canonical47RmsSource::bind(bound.source().clone()).map(|_| ())
    }
    pub(crate) async fn prepare_canonical47_source_component(
        bound: &BoundFullModelSource,
        record: &RecordedVllm08Model,
        service: Arc<ResidentService>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let call = record.call();
        let request = call.request();
        require_component_request(request)?;
        Self::validate_canonical47_component_source(bound)?;
        if call.model_profile() != FullModelProfile::Qwen08
            || call.source() != bound.source().source_identity()
            || service.source_identity() != call.source()
            || record.plan().nodes().len() != 255
        {
            return Err(Error::Unsupported(
                "source-owned canonical47 InputRms only".into(),
            ));
        }
        let descriptor = match &record.plan().nodes()[1].operation {
            PlanOperation::TypedVllm08 {
                descriptor,
                state: None,
            } if descriptor.kind() == Vllm08Kind::InputRms
                && descriptor.call() == call
                && descriptor.parameters().len() == 1 =>
            {
                descriptor
            }
            _ => return Err(Error::State("actual47 InputRms descriptor absent".into())),
        };
        bound
            .source()
            .validate_parameter(QwenParameterRole::InputNorm(0), &descriptor.parameters()[0])?;
        let actual = Canonical47RmsSource::bind(bound.source().clone())?;
        Self::prepare_canonical47(service, actual, &PROMPT, 0, cancelled).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::{
        SourceIdentity,
        program::{
            GdnPrefixLayer, ShortHfCall, ShortHfSpan, ShortHfWorkload, TeacherSpan,
            VllmBf16Teachers,
        },
    };

    #[test]
    fn original47_rms_request_refuses_short_hf_issuer_with_the_same_prompt() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let old = VllmBf16Invocation::from_span(
            GdnPrefixLayer::new(source, 0).unwrap(),
            VllmBf16Mode::Prefill,
            VllmBf16Teachers::from_span(TeacherSpan::new(source, 0, 512, 248320, &PROMPT).unwrap())
                .unwrap(),
        )
        .unwrap();
        require_component_request(&old).unwrap();
        let short = ShortHfCall::new(
            FullModelProfile::Qwen08,
            ShortHfSpan::prefill(source, ShortHfWorkload::Prompt47, &PROMPT).unwrap(),
        )
        .unwrap();
        assert_eq!(short.request().teachers().ids(), old.teachers().ids());
        assert!(short.request().legacy_phase().is_err());
        assert!(matches!(
            require_component_request(short.request()),
            Err(Error::Unsupported(_))
        ));
        for base in 47..=53 {
            let short = ShortHfCall::new(
                FullModelProfile::Qwen08,
                ShortHfSpan::cached(source, ShortHfWorkload::Prompt47, base, 11).unwrap(),
            )
            .unwrap();
            assert!(matches!(
                require_component_request(short.request()),
                Err(Error::Unsupported(_))
            ));
        }
    }
}
