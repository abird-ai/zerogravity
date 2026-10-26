//! Original learned members; bounded uploads and completed whole-W verification.
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::{ModelWeight, ResidentReadback};
use cubecl::bytes::Bytes;
use std::ops::Range;

pub(crate) struct FixedPromptMlpWeights {
    recipe: Arc<ComponentRecipe>,
    gate_up: Arc<ModelWeight>,
    down: Arc<ModelWeight>,
}

pub(super) fn require_recipe(recipe: &ComponentRecipe) -> Result<()> {
    let ComponentKind::FixedPrompt(arithmetic) = recipe.kind() else {
        return Err(Error::Unsupported(
            "fixed_prompt MLP source family required".into(),
        ));
    };
    arithmetic.require_owned()?;
    recipe.kind().require_call(recipe.bindings().call())?;
    if recipe.cut() != ComponentCut::Mlp
        || recipe.nodes().len() != 11
        || recipe.bindings().call().rows().len() != 47
    {
        return Err(Error::Unsupported(
            "fixed_prompt eleven-node MLP cut required".into(),
        ));
    }
    for (index, arity) in [(8, 2), (10, 1)] {
        let op = recipe.operator(index)?;
        if op.call()!=recipe.bindings().call() || op.kind()!=(zerogravity_core::accelerator::program::SourceOperatorKind::Projection{selected_row:None}) || op.numerical_contract()!=zerogravity_core::accelerator::program::SourceNumericalContract::ReferenceCudaSm86Bf16V1 || !op.parameters().is_empty() || op.matrices().len()!=arity{return Err(Error::State("MLP actual source operator/member arity differs".into()));}
        for member in op.matrices() {
            super::super::post_attention_projection::require_matrix_source(
                recipe.source(),
                member,
            )?;
        }
    }
    Ok(())
}

impl FixedPromptMlpWeights {
    pub(crate) async fn upload(
        recipe: Arc<ComponentRecipe>,
        service: Arc<ResidentService>,
        completion_owner: CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        require_recipe(&recipe)?;
        if service.source_identity() != recipe.bindings().call().source() {
            return Err(Error::State(
                "fixed_prompt MLP W original service/source".into(),
            ));
        }
        let mut init = service.begin_initialization(0)?;
        // Unknown upload retains the existing graph before the first W write.
        // The successful result keeps no graph backedge.
        init.retain_completion_owner(completion_owner)?;
        let gate = init
            .upload_model_weight(
                recipe.source().clone(),
                recipe.operator(8)?.clone(),
                cubecl::cuda::SourceWeightLayout::MlpGateUp,
                cancelled,
            )
            .await?;
        let down = init
            .upload_model_weight(
                recipe.source().clone(),
                recipe.operator(10)?.clone(),
                cubecl::cuda::SourceWeightLayout::MlpDown,
                cancelled,
            )
            .await?;
        drop(init.complete_sync().await?);
        let weights = Arc::new(Self {
            recipe,
            gate_up: gate.bind_initialized()?,
            down: down.bind_initialized()?,
        });
        weights.require_initialized()?;
        Ok(weights)
    }
    pub(super) fn require_initialized(&self) -> Result<()> {
        require_recipe(&self.recipe)?;
        for (index, weight, bytes, windows) in [
            (8, &self.gate_up, 14_680_064, 14),
            (10, &self.down, 7_340_032, 8),
        ] {
            weight.require_source_operator(self.recipe.source(), self.recipe.operator(index)?)?;
            if weight.lease().bytes() != bytes || weight.copy_counts()? != (windows, windows) {
                return Err(Error::State(
                    "fixed_prompt complete ordered MLP W upload".into(),
                ));
            }
        }
        Ok(())
    }
    pub(super) fn require_values(&self, values: &Arc<ComponentValues>) -> Result<()> {
        self.require_initialized()?;
        if !Arc::ptr_eq(&self.recipe, values.recipe()) || !self.retained_values(values) {
            return Err(Error::State(
                "fixed_prompt SAME MLP W and value owner".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn retained_values(&self, values: &Arc<ComponentValues>) -> bool {
        Arc::ptr_eq(&self.recipe, values.recipe())
            && [&self.gate_up, &self.down].iter().all(|weight| {
                Arc::ptr_eq(weight.lease().service(), values.service())
                    && weight.lease().source_identity() == values.service().source_identity()
                    && weight.lease().service_generation() == values.service().generation()
            })
    }
    pub(super) fn lease(&self, index: usize) -> Result<&ResidentLease> {
        match index {
            8 => Ok(self.gate_up.lease()),
            10 => Ok(self.down.lease()),
            _ => Err(Error::State("fixed_prompt MLP learned node absent".into())),
        }
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        // Retained metadata remains inspectable after an uncertain completion.
        [&self.gate_up, &self.down]
            .iter()
            .map(|weight| (weight.lease().clone(), DeviceAccess::Read))
            .collect()
    }
    pub(crate) fn append_readbacks(
        &self,
        reads: &mut Vec<ResidentReadback>,
    ) -> Result<Range<usize>> {
        self.require_initialized()?;
        let start = reads.len();
        for weight in [&self.gate_up, &self.down] {
            reads.push(weight.lease().readback(0, weight.lease().bytes())?);
        }
        Ok(start..reads.len())
    }
    pub(crate) fn validate_completed(
        &self,
        raw: &[Bytes],
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.require_initialized()?;
        if raw.len() != 2 {
            return Err(Error::State(
                "fixed_prompt complete two-W D2H cohort required".into(),
            ));
        }
        let mut count = 0;
        for (weight, bytes) in [&self.gate_up, &self.down].into_iter().zip(raw) {
            count += weight.verify_completed_readback(bytes.as_ref(), cancelled)?;
        }
        if count != 22 {
            return Err(Error::State(
                "fixed_prompt complete twenty-two source W windows".into(),
            ));
        }
        Ok(count)
    }
    pub(crate) fn immutable_bytes(&self) -> usize {
        self.gate_up.lease().bytes() + self.down.lease().bytes()
    }
}
