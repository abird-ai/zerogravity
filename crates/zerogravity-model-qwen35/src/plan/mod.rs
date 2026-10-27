//! Source-owned Qwen recording policy over the one shared mathematical builder.
//! Provenance: formerly core Vllm08/ShortHf/vllm_gdn descriptors; no execution permit.
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::sync::Arc;
use zerogravity_core::accelerator::program::PlanBuilder as CorePlanBuilder;
use zerogravity_core::accelerator::{SourceIdentity, program::*};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::source::TensorSource;
mod source_profile;
pub use source_profile::QwenSourceProfile;
mod ordinary;
pub(crate) use ordinary::build_ordinary_plan;
pub use ordinary::{
    ORDINARY_CUDA_MAX_CHUNK_ROWS, ORDINARY_CUDA_MAX_CONTEXT, ORDINARY_CUDA_MIN_CONTEXT,
    OrdinaryQwenInvocation, OrdinaryQwenPlan,
};
mod prompt;
pub use prompt::{PromptCall, PromptSpan, PromptWorkload};
mod prefix;
pub use prefix::{QwenPrefixLayer, QwenPrefixRmsBinding, QwenPrefixRmsSite};
mod model;
pub use model::{
    QwenLayer, QwenOperation, QwenOperator, QwenPlan, QwenProducer, QwenProjection, QwenRmsSite,
    QwenSourceCall,
};
mod gdn_projection;
pub use gdn_projection::{
    GdnProjection, QwenGdnCall, QwenGdnProjection, QwenProjectionContract, QwenSiluAndMul,
    QwenSiluContract, QwenValueOrigin, QwenValueProvenance, ReferencePhase,
};
mod gdn;
pub use gdn::{
    GdnConvValues, QwenFragmentMetadata, QwenFragmentPlan, QwenFragmentView, QwenGdnConv,
    QwenGdnRecurrent, QwenInputRms,
};
mod rms;
pub use rms::{QwenEmbedding, QwenInputRows, QwenResidualRmsContract, ResidualRmsValues};
mod lower;

/// Family validation adapter; all nodes, SSA ownership and effects belong to core.
pub struct QwenPlanBuilder {
    core: CorePlanBuilder,
    model_call: Option<QwenSourceCall>,
    kv_cohorts: BTreeMap<usize, usize>,
    fragment_call: Option<QwenGdnCall>,
    provenance: Vec<Option<QwenValueProvenance>>,
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    operator_slice: bool,
}
impl std::ops::Deref for QwenPlanBuilder {
    type Target = CorePlanBuilder;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}
impl std::ops::DerefMut for QwenPlanBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.core
    }
}
impl QwenPlanBuilder {
    fn from_core(core: CorePlanBuilder) -> Self {
        Self {
            source: core.source(),
            profile: core.numerical_profile(),
            operator_slice: core.is_operator_slice(),
            core,
            model_call: None,
            kv_cohorts: BTreeMap::new(),
            fragment_call: None,
            provenance: Vec::new(),
        }
    }
    pub fn new_operator_slice(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
    ) -> Result<Self> {
        Ok(Self::from_core(CorePlanBuilder::new_operator_slice(
            source, profile,
        )?))
    }
    pub fn new_with_profile(
        source: SourceIdentity,
        capacity: usize,
        required: Vec<StateSlot>,
        capture: bool,
        profile: ProgramNumericalProfile,
    ) -> Result<Self> {
        Ok(Self::from_core(CorePlanBuilder::new_with_profile(
            source, capacity, required, capture, profile,
        )?))
    }
    pub fn inputs() -> ForwardInputs<ParameterRef> {
        CorePlanBuilder::inputs()
    }
    pub fn value_provenance(&self, value: &PlanValue) -> Result<Option<&QwenValueProvenance>> {
        self.core.validate_value(value)?;
        Ok(self
            .provenance
            .get(value.id.0 as usize)
            .and_then(Option::as_ref))
    }
    fn record_source(
        &mut self,
        name: &str,
        descriptor: SourceOperator,
        inputs: &[PlanValue],
        tags: Vec<QwenValueProvenance>,
    ) -> Result<Vec<PlanValue>> {
        if tags.len() != descriptor.output_elements()?.len() {
            return Err(Error::State("Qwen exact source producer arity".into()));
        }
        let base = self.core.values().len();
        let output = self
            .core
            .record_source_operator(name, &descriptor, inputs)?;
        self.provenance.resize(base, None);
        self.provenance.extend(tags.into_iter().map(Some));
        Ok(output)
    }
    pub fn source_embedding(&mut self, d: &QwenEmbedding) -> Result<PlanValue> {
        rms::record_embedding(self, d)
    }
    pub fn source_operator(
        &mut self,
        d: &QwenOperator,
        inputs: &[PlanValue],
    ) -> Result<Vec<PlanValue>> {
        model::record(self, d, inputs)
    }
    pub fn gdn_project(&mut self, d: &QwenGdnProjection, input: &PlanValue) -> Result<PlanValue> {
        gdn_projection::record_projection(self, d, input)
    }
    pub fn silu_and_mul(&mut self, d: &QwenSiluAndMul, input: &PlanValue) -> Result<PlanValue> {
        gdn_projection::record_activation(self, d, input)
    }
    pub fn input_rms(&mut self, d: &QwenInputRms, input: &PlanValue) -> Result<PlanValue> {
        gdn::record_input_rms(self, d, input)
    }
    pub fn gdn_conv(
        &mut self,
        d: &QwenGdnConv,
        qkvz: &PlanValue,
        ba: &PlanValue,
    ) -> Result<GdnConvValues<PlanValue>> {
        gdn::record_conv(self, d, qkvz, ba)
    }
    pub fn gdn_recurrent(
        &mut self,
        d: &QwenGdnRecurrent,
        activated: &PlanValue,
        z: &PlanValue,
        ba: &PlanValue,
    ) -> Result<PlanValue> {
        gdn::record_recurrent(self, d, activated, z, ba)
    }
    pub fn residual_rms_profiled(
        &mut self,
        parameter: SourceParameter,
        branch: &PlanValue,
        residual: &PlanValue,
        contract: QwenResidualRmsContract,
    ) -> Result<ResidualRmsValues<PlanValue>> {
        rms::record(self, parameter, branch, residual, contract)
    }
    pub fn finish_operator_slice(self, output: PlanValue) -> Result<RecordedOperatorSlice> {
        self.core.finish_operator_slice(output)
    }
    fn one_output(values: Vec<PlanValue>) -> Result<PlanValue> {
        let [value] = values
            .try_into()
            .map_err(|_| Error::State("Qwen single source output".into()))?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests_support;
