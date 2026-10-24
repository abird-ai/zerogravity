//! Same original2048 source-row/gamma producer for the checked component.
use super::*;
use crate::resident_leases::ResidentService;
use zerogravity_core::accelerator::program::{
    FullModelProfile, RecordedVllm08Model, VllmBf16Phase,
};
use zerogravity_model_qwen35::original_binding::{BoundFullModelSource, QwenParameterRole};
impl RmsPrepared {
    pub(crate) fn validate_qwen2b_component_source(bound: &BoundFullModelSource) -> Result<()> {
        if bound.model_profile() != FullModelProfile::Qwen2B {
            return Err(Error::Unsupported("original2B source family".into()));
        }
        Qwen2bRmsSource::bind(bound.source().clone()).map(|_| ())
    }
    pub(crate) async fn prepare_qwen2b_source_component(
        bound: &BoundFullModelSource,
        record: &RecordedVllm08Model,
        service: Arc<ResidentService>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let source = bound.source();
        let call = record.call();
        if bound.model_profile() != FullModelProfile::Qwen2B
            || call.model_profile() != FullModelProfile::Qwen2B
            || call.source() != source.source_identity()
            || service.source_identity() != source.source_identity()
            || !matches!(
                call.request().legacy_phase()?,
                VllmBf16Phase::InitialOne | VllmBf16Phase::Initial64
            )
            || call.request().teachers().base() != 0
            || record.plan().nodes().len() != 255
        {
            return Err(Error::Unsupported(
                "source-owned initial2B RMS component only".into(),
            ));
        }
        let descriptor = match &record.plan().nodes()[1].operation {
            zerogravity_core::accelerator::program::PlanOperation::TypedVllm08 {
                descriptor,
                state: None,
            } if descriptor.kind()
                == zerogravity_core::accelerator::program::Vllm08Kind::InputRms
                && descriptor.call() == call
                && descriptor.parameters().len() == 1 =>
            {
                descriptor
            }
            _ => {
                return Err(Error::State(
                    "actual2B source InputRms descriptor absent".into(),
                ));
            }
        };
        source.validate_parameter(QwenParameterRole::InputNorm(0), &descriptor.parameters()[0])?;
        // Exact config/index/epsilon/2048 raw gamma+embedding source guards.
        let actual = Qwen2bRmsSource::bind(source.clone())?;
        Self::prepare_2b(
            service,
            actual,
            call.request().teachers().ids(),
            0,
            cancelled,
        )
        .await
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) async fn prepare_qwen2b_mlp_source_component(
        recipe: Arc<crate::resident_program::qwen2b_component::ComponentRecipe>,
        service: Arc<ResidentService>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        if recipe.cut() != crate::resident_program::qwen2b_component::ComponentCut::Qwen2bMlp
            || !recipe.qwen2b_mlp_record_admission_retained()
        {
            return Err(Error::State(
                "same actual2B MLP source record admission before RMS".into(),
            ));
        }
        let bound = recipe.source();
        let record = recipe.record();
        let source = bound.source();
        let call = record.call();
        if bound.model_profile() != FullModelProfile::Qwen2B
            || call.model_profile() != FullModelProfile::Qwen2B
            || call.source() != source.source_identity()
            || service.source_identity() != source.source_identity()
            || !matches!(
                call.request().legacy_phase()?,
                VllmBf16Phase::InitialOne | VllmBf16Phase::Initial64
            )
            || call.request().teachers().base() != 0
            || record.plan().nodes().len() != 255
        {
            return Err(Error::Unsupported(
                "source-owned initial2B RMS component only".into(),
            ));
        }
        let descriptor = match &record.plan().nodes()[1].operation {
            zerogravity_core::accelerator::program::PlanOperation::TypedVllm08 {
                descriptor,
                state: None,
            } if descriptor.kind()
                == zerogravity_core::accelerator::program::Vllm08Kind::InputRms
                && descriptor.call() == call
                && descriptor.parameters().len() == 1 =>
            {
                descriptor
            }
            _ => {
                return Err(Error::State(
                    "actual2B source InputRms descriptor absent".into(),
                ));
            }
        };
        source.validate_parameter(QwenParameterRole::InputNorm(0), &descriptor.parameters()[0])?;
        // Exact config/index/epsilon/2048 raw gamma+embedding source guards.
        let actual = Qwen2bRmsSource::bind(source.clone())?;
        Self::prepare_2b_mlp(
            service,
            actual,
            recipe.clone(),
            call.request().teachers().ids(),
            0,
            cancelled,
        )
        .await
    }
}
