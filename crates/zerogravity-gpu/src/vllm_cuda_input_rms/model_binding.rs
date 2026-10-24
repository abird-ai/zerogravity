//! Source binding only; existing RMS stages retain their exact instructions.
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    SourceParameter, Vllm08Kind, Vllm08Operator, Vllm08RmsSite,
};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::{OriginalWindowModelSource, QwenParameterRole};

struct ModelRmsOwner {
    operator: Vllm08Operator,
    role: QwenParameterRole,
    model: Arc<OriginalWindowModelSource>,
    _component: Option<zerogravity_model_qwen35::original_binding::BoundFullModelSource>,
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    _qwen2b_mlp: Option<Arc<crate::resident_program::qwen2b_component::ComponentRecipe>>,
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    short_hf: Option<zerogravity_model_qwen35::original_binding::BoundShortHfCall>,
    _metadata: WindowWorkspace,
}
#[derive(Clone)]
pub(crate) struct BoundModelRmsSource(Arc<ModelRmsOwner>);
impl BoundModelRmsSource {
    /// Call before device/service creation as part of cold source preflight.
    pub(crate) fn bind(
        model: Arc<OriginalWindowModelSource>,
        operator: Vllm08Operator,
    ) -> Result<Self> {
        model.require_vllm08_full_source()?;
        Self::bind_checked(model, operator, None)
    }
    fn bind_checked(
        model: Arc<OriginalWindowModelSource>,
        operator: Vllm08Operator,
        component: Option<zerogravity_model_qwen35::original_binding::BoundFullModelSource>,
    ) -> Result<Self> {
        operator.validate()?;
        let role = match operator.kind() {
            Vllm08Kind::InputRms | Vllm08Kind::ResidualRms(Vllm08RmsSite::Input) => {
                QwenParameterRole::InputNorm(
                    operator
                        .layer()
                        .ok_or_else(|| Error::State("input norm layer absent".into()))?
                        .index(),
                )
            }
            Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention) => QwenParameterRole::FfnNorm(
                operator
                    .layer()
                    .ok_or_else(|| Error::State("postnorm layer absent".into()))?
                    .index(),
            ),
            Vllm08Kind::ResidualRms(Vllm08RmsSite::Final) => QwenParameterRole::FinalNorm,
            _ => {
                return Err(Error::Unsupported(
                    "full-model original RMS binding only".into(),
                ));
            }
        };
        if operator.call().source() != model.source_identity() || operator.parameters().len() != 1 {
            return Err(Error::State("full-model original RMS source/arity".into()));
        }
        model.validate_parameter(role, &operator.parameters()[0])?;
        let metadata = model.source().reserve_workspace(64 * 1024)?;
        Ok(Self(Arc::new(ModelRmsOwner {
            operator,
            role,
            model,
            _component: component,
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            _qwen2b_mlp: None,
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            short_hf: None,
            _metadata: metadata,
        })))
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn bind_post_attention_component(
        recipe: &crate::resident_program::qwen2b_component::ComponentRecipe,
    ) -> Result<Self> {
        use crate::resident_program::qwen2b_component::ComponentCut;
        if recipe.cut() != ComponentCut::PostAttention {
            return Err(Error::Unsupported(
                "post-attention component source cut required".into(),
            ));
        }
        Self::bind_component_post_attention(recipe)
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn bind_mlp_component_post_attention(
        recipe: &crate::resident_program::qwen2b_component::ComponentRecipe,
    ) -> Result<Self> {
        use crate::resident_program::qwen2b_component::{ComponentCut, ComponentKind};
        if recipe.cut() != ComponentCut::Mlp
            || !matches!(recipe.kind(), ComponentKind::Canonical47(_))
        {
            return Err(Error::Unsupported(
                "canonical47 MLP source cut required".into(),
            ));
        }
        Self::bind_component_post_attention(recipe)
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn bind_qwen2b_mlp_component_post_attention(
        recipe: &Arc<crate::resident_program::qwen2b_component::ComponentRecipe>,
    ) -> Result<Self> {
        use crate::resident_program::qwen2b_component::{ComponentCut, ComponentKind};
        if recipe.cut() != ComponentCut::Qwen2bMlp
            || recipe.kind() != ComponentKind::Qwen2b
            || recipe.qwen2b_arithmetic()
                != Some(crate::vllm_cuda_gdn::Qwen2bGdnArithmetic::OwnedSolve4Recompute3)
        {
            return Err(Error::Unsupported(
                "explicit owned2B eleven-node dual source required".into(),
            ));
        }
        let mut binding = Self::bind_component_post_attention(recipe)?;
        Arc::get_mut(&mut binding.0)
            .ok_or_else(|| Error::State("fresh2B MLP gamma binding".into()))?
            ._qwen2b_mlp = Some(recipe.clone());
        Ok(binding)
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    fn bind_component_post_attention(
        recipe: &crate::resident_program::qwen2b_component::ComponentRecipe,
    ) -> Result<Self> {
        recipe
            .kind()
            .require_request(recipe.record().call().request())?;
        let operator = recipe.operator(7)?.clone();
        if operator.call().model_profile() != recipe.source().model_profile()
            || operator.kind() != Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention)
            || operator.layer().map(|x| x.index()) != Some(0)
            || recipe.source().source().geometry().hidden_size != recipe.kind().columns() as u64
            || recipe.source().source().owner().config().rms_norm_eps() != 1e-6
        {
            return Err(Error::State(
                "exact source-family component post-attention gamma/config".into(),
            ));
        }
        Self::bind_checked(
            recipe.source().source().clone(),
            operator,
            Some(recipe.source().clone()),
        )
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn bind_short_hf(
        source: &zerogravity_model_qwen35::original_binding::BoundShortHfCall,
        operator: Vllm08Operator,
    ) -> Result<Self> {
        source.call().validate()?;
        if operator.call() != source.call().as_vllm_call()
            || operator.call().source() != source.source().source().source_identity()
            || operator.call().model_profile() != source.source().model_profile()
        {
            return Err(Error::State("RMS exact source-bound short-HF call".into()));
        }
        let mut binding = Self::bind_checked(source.source().source().clone(), operator, None)?;
        Arc::get_mut(&mut binding.0)
            .ok_or_else(|| Error::State("fresh short-HF RMS binding".into()))?
            .short_hf = Some(source.clone());
        Ok(binding)
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(super) fn short_hf_profile(
        &self,
    ) -> Option<zerogravity_core::accelerator::program::FullModelProfile> {
        self.0
            .short_hf
            .as_ref()
            .map(|source| source.call().model_profile())
    }
    pub(crate) fn component_profile(
        &self,
    ) -> Option<zerogravity_core::accelerator::program::FullModelProfile> {
        self.0._component.as_ref().map(|s| s.model_profile())
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        self.0.model.source()
    }
    pub(crate) fn parameter(&self) -> &SourceParameter {
        &self.0.operator.parameters()[0]
    }
    pub(crate) fn gamma_name(&self) -> &str {
        &self.parameter().name
    }
    pub(crate) fn operator(&self) -> &Vllm08Operator {
        &self.0.operator
    }
    pub(crate) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if !self.source().same_issuer(source)
            || source.source_identity() != self.0.operator.call().source()
        {
            return Err(Error::State(
                "full-model RMS actual immutable source owner differs".into(),
            ));
        }
        self.0
            .model
            .validate_parameter(self.0.role, self.parameter())
    }
}
