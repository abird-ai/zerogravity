//! All finite learned groups shared across request phases; no packed host W.
use crate::resident_leases::{
    ModelWeight, PendingModelWeight, ResidentInitialization, ResidentLease,
};
use crate::resident_program::source_bindings::SourceBindings;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{SourceOperator, SourceOperatorKind};
use zerogravity_core::device_events::DeviceAccess;
use zerogravity_core::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ModelWeightKey(String);
impl std::borrow::Borrow<str> for ModelWeightKey {
    fn borrow(&self) -> &str {
        &self.0
    }
}
impl ModelWeightKey {
    fn name(operator: &SourceOperator) -> Result<&str> {
        if !matches!(
            operator.kind(),
            SourceOperatorKind::Projection { selected_row: None }
        ) || operator.matrices().is_empty()
            || !operator.parameters().is_empty()
        {
            return Err(Error::State(
                "learned W key requires full-row source projection".into(),
            ));
        }
        Ok(operator.matrices()[0].name.as_str())
    }
    pub(crate) fn of(operator: &SourceOperator) -> Result<Self> {
        Ok(Self(Self::name(operator)?.to_owned()))
    }
}
pub(crate) struct PendingModelWeights {
    weights: BTreeMap<ModelWeightKey, PendingModelWeight>,
    source: VerifiedTensorSource,
    expected_groups: usize,
    _metadata: Arc<WindowWorkspace>,
    fixed_prompt_prefix: Option<Arc<super::fixed_prompt_model_prefix::FixedPromptPrefixCut>>,
    text: Option<Arc<super::text_model::TextModelBinding>>,
}
pub(crate) struct ModelWeights {
    weights: BTreeMap<ModelWeightKey, Arc<ModelWeight>>,
    source: VerifiedTensorSource,
    expected_groups: usize,
    _metadata: Arc<WindowWorkspace>,
    fixed_prompt_prefix: Option<Arc<super::fixed_prompt_model_prefix::FixedPromptPrefixCut>>,
    text: Option<Arc<super::text_model::TextModelBinding>>,
}

impl PendingModelWeights {
    /// The same original initializer owns every checkpoint and pending window.
    /// One genuine full-plan recipe determines all114 groups; no caller names.
    pub(crate) async fn upload(
        record: &Arc<SourceBindings>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::upload_impl(record, init, None, cancelled).await
    }
    pub(crate) async fn upload_fixed_prompt_prefix(
        cut: &Arc<super::fixed_prompt_model_prefix::FixedPromptPrefixCut>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::upload_impl(cut.scope().record(), init, Some(cut.clone()), cancelled).await
    }
    async fn upload_impl(
        record: &Arc<SourceBindings>,
        init: &mut ResidentInitialization<'_>,
        fixed_prompt_prefix: Option<Arc<super::fixed_prompt_model_prefix::FixedPromptPrefixCut>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        let source = record.source().clone();
        let metadata = Arc::new(source.reserve_workspace(512 * 1024)?);
        let end = fixed_prompt_prefix
            .as_ref()
            .map_or(record.plan().nodes().len(), |c| c.nodes().len());
        if end > record.plan().nodes().len() {
            return Err(Error::State("learned prefix node extent".into()));
        }
        let head_node = record
            .plan()
            .source_value_provenance(record.selection().output())?
            .map(|p| p.node_index());
        let mut descriptors = BTreeMap::<ModelWeightKey, SourceOperator>::new();
        for index in 0..end {
            let Some(descriptor) = record.operator(index)? else {
                continue;
            };
            if matches!(
                descriptor.kind(),
                SourceOperatorKind::Projection { selected_row: None }
            ) && Some(index) != head_node
            {
                if descriptor.call() != record.call() {
                    return Err(Error::State("learned actual request issuer differs".into()));
                }
                if descriptors
                    .insert(ModelWeightKey::of(descriptor)?, descriptor.clone())
                    .is_some()
                {
                    return Err(Error::State(
                        "learned member key unexpectedly repeated".into(),
                    ));
                }
            }
        }
        if descriptors.is_empty()
            || fixed_prompt_prefix
                .as_ref()
                .is_some_and(|c| descriptors.len() != c.weight_groups())
        {
            return Err(Error::State(
                "learned exact selected group inventory".into(),
            ));
        }
        let expected_groups = descriptors.len();
        let mut weights = BTreeMap::new();
        for (key, descriptor) in descriptors {
            if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Err(Error::State("original source upload cancelled".into()));
            }
            let layout = ModelWeight::packing_layout(&descriptor)?;
            let pending = init
                .upload_model_weight(source.clone(), descriptor, layout, cancelled)
                .await?;
            weights.insert(key, pending);
        }
        Ok(Self {
            weights,
            source,
            expected_groups,
            _metadata: metadata,
            fixed_prompt_prefix,
            text: None,
        })
    }
    /// Only after original complete_sync marked every exact W lease initialized.
    pub(crate) fn bind_initialized(self) -> Result<Arc<ModelWeights>> {
        let weights = self
            .weights
            .into_iter()
            .map(|(key, pending)| Ok((key, pending.bind_initialized()?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Arc::new(ModelWeights {
            weights,
            source: self.source,
            expected_groups: self.expected_groups,
            _metadata: self._metadata,
            fixed_prompt_prefix: self.fixed_prompt_prefix,
            text: self.text,
        }))
    }
}
impl ModelWeights {
    pub(crate) fn require_fixed_prompt_prefix(
        &self,
        cut: &Arc<super::fixed_prompt_model_prefix::FixedPromptPrefixCut>,
    ) -> Result<()> {
        if !self
            .fixed_prompt_prefix
            .as_ref()
            .is_some_and(|own| Arc::ptr_eq(own, cut))
            || !self.source.same_issuer(cut.scope().source())
        {
            return Err(Error::State(
                "fixed_prompt exact15-group source/cut issuer differs".into(),
            ));
        }
        self.require_initialized()
    }
    pub(crate) fn require_operator(&self, operator: &SourceOperator) -> Result<&Arc<ModelWeight>> {
        let key = ModelWeightKey::name(operator)?;
        let weight = self
            .weights
            .get(key)
            .ok_or_else(|| Error::State("full24 actual learned W group absent".into()))?;
        weight.require_source_operator(&self.source, operator)?;
        Ok(weight)
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.weights
            .values()
            .map(|weight| (weight.lease().clone(), DeviceAccess::Read))
            .collect()
    }
    pub(crate) async fn verify_source_windows(
        &self,
        completion_owner: crate::physical_quarantine::CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(usize, usize, usize)> {
        self.require_initialized()?;
        let mut verified = 0usize;
        let mut copies = 0usize;
        let mut checkpoints = 0usize;
        for weight in self.weights.values() {
            let (c, p) = weight.copy_counts()?;
            copies += c;
            checkpoints += p;
            verified += weight
                .verify_source_windows(completion_owner.clone(), cancelled)
                .await?;
        }
        if copies != self.expected_source_windows()? || checkpoints != copies || verified != copies
        {
            return Err(Error::State(
                "full24 complete114 learned groups/window H2D/checkpoint/readback coverage".into(),
            ));
        }
        Ok((verified, copies, checkpoints))
    }
    pub(crate) fn require_initialized(&self) -> Result<()> {
        if self.weights.len() != self.expected_groups {
            return Err(Error::State("initialized W inventory changed".into()));
        }
        for weight in self.weights.values() {
            weight.require_initialized()?;
        }
        Ok(())
    }
}

mod text;
