//! All finite learned groups shared across request phases; no packed host W.
use crate::resident_leases::{
    ModelWeight, PendingModelWeight, ResidentInitialization, ResidentLease,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_core::accelerator::program::{
    PlanOperation, RecordedVllm08Model, Vllm08Kind, Vllm08Operator, Vllm08Projection,
};
use zerogravity_core::device_events::DeviceAccess;
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ModelWeightKey {
    layer: u8,
    role: u8,
}
impl ModelWeightKey {
    pub(crate) fn of(operator: &Vllm08Operator) -> Result<Self> {
        operator.validate()?;
        let Vllm08Kind::Projection(kind) = operator.kind() else {
            return Err(Error::State(
                "full24 W key requires learned projection".into(),
            ));
        };
        let role = match kind {
            Vllm08Projection::Qkvz => 0,
            Vllm08Projection::Ba => 1,
            Vllm08Projection::GdnOut => 2,
            Vllm08Projection::AttentionQkv => 3,
            Vllm08Projection::AttentionOut => 4,
            Vllm08Projection::GateUp => 5,
            Vllm08Projection::Down => 6,
            Vllm08Projection::Head => {
                return Err(Error::Unsupported("head shares original TiedWeight".into()));
            }
        };
        Ok(Self {
            layer: operator
                .layer()
                .ok_or_else(|| Error::State("full24 learned W layer absent".into()))?
                .index() as u8,
            role,
        })
    }
}
pub(crate) struct PendingModelWeights {
    weights: BTreeMap<ModelWeightKey, PendingModelWeight>,
    source: Arc<OriginalWindowModelSource>,
    canonical47_prefix: Option<Arc<super::canonical47_model_prefix::Canonical47PrefixCut>>,
    short_hf: Option<Arc<super::short_hf_model::ShortHfModelBinding>>,
}
pub(crate) struct ModelWeights {
    weights: BTreeMap<ModelWeightKey, Arc<ModelWeight>>,
    source: Arc<OriginalWindowModelSource>,
    canonical47_prefix: Option<Arc<super::canonical47_model_prefix::Canonical47PrefixCut>>,
    short_hf: Option<Arc<super::short_hf_model::ShortHfModelBinding>>,
}

impl PendingModelWeights {
    /// The same original initializer owns every checkpoint and pending window.
    /// One genuine full-plan recipe determines all114 groups; no caller names.
    pub(crate) async fn upload(
        source: Arc<OriginalWindowModelSource>,
        record: &RecordedVllm08Model,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::upload_impl(source, record, init, None, cancelled).await
    }
    pub(crate) async fn upload_canonical47_prefix(
        cut: &Arc<super::canonical47_model_prefix::Canonical47PrefixCut>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::upload_impl(
            cut.scope().source().source().clone(),
            cut.scope().record(),
            init,
            Some(cut.clone()),
            cancelled,
        )
        .await
    }
    async fn upload_impl(
        source: Arc<OriginalWindowModelSource>,
        record: &RecordedVllm08Model,
        init: &mut ResidentInitialization<'_>,
        canonical47_prefix: Option<Arc<super::canonical47_model_prefix::Canonical47PrefixCut>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        source.require_vllm08_full_source()?;
        if record.call().source() != source.source_identity() || record.plan().nodes().len() != 255
        {
            return Err(Error::State(
                "full24 original complete learned recipe/source".into(),
            ));
        }
        let mut descriptors = BTreeMap::<ModelWeightKey, Vllm08Operator>::new();
        for node in &record.plan().nodes()[..canonical47_prefix
            .as_ref()
            .map_or(record.plan().nodes().len(), |c| c.nodes().len())]
        {
            let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                continue;
            };
            if let Vllm08Kind::Projection(kind) = descriptor.kind()
                && kind != Vllm08Projection::Head
            {
                if descriptor.call() != record.call() {
                    return Err(Error::State(
                        "full24 learned recipe request issuer changed".into(),
                    ));
                }
                let key = ModelWeightKey::of(descriptor)?;
                if descriptors
                    .insert(key, descriptor.as_ref().clone())
                    .is_some()
                {
                    return Err(Error::State(
                        "full24 learned role unexpectedly repeated".into(),
                    ));
                }
            }
        }
        if descriptors.len()
            != canonical47_prefix
                .as_ref()
                .map_or(114, |c| c.weight_groups())
        {
            return Err(Error::State(
                "full24 exact114 ordered learned groups".into(),
            ));
        }
        let mut weights = BTreeMap::new();
        for (key, descriptor) in descriptors {
            if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Err(Error::State(
                    "full24 original source upload cancelled".into(),
                ));
            }
            let pending = init
                .upload_original_model_weight(source.clone(), descriptor, cancelled)
                .await?;
            weights.insert(key, pending);
        }
        Ok(Self {
            weights,
            source,
            canonical47_prefix,
            short_hf: None,
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
            canonical47_prefix: self.canonical47_prefix,
            short_hf: self.short_hf,
        }))
    }
}
impl ModelWeights {
    pub(crate) fn require_canonical47_prefix(
        &self,
        cut: &Arc<super::canonical47_model_prefix::Canonical47PrefixCut>,
    ) -> Result<()> {
        if !self
            .canonical47_prefix
            .as_ref()
            .is_some_and(|own| Arc::ptr_eq(own, cut))
            || !Arc::ptr_eq(&self.source, cut.scope().source().source())
        {
            return Err(Error::State(
                "canonical47 exact15-group source/cut issuer differs".into(),
            ));
        }
        self.require_initialized()
    }
    pub(crate) fn require_operator(&self, operator: &Vllm08Operator) -> Result<&Arc<ModelWeight>> {
        let key = ModelWeightKey::of(operator)?;
        let weight = self
            .weights
            .get(&key)
            .ok_or_else(|| Error::State("full24 actual learned W group absent".into()))?;
        weight.require_operator(&self.source, operator)?;
        Ok(weight)
    }
    pub(crate) fn source(&self) -> &Arc<OriginalWindowModelSource> {
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
        if copies != self.expected_source_windows() || checkpoints != copies || verified != copies {
            return Err(Error::State(
                "full24 complete114 learned groups/window H2D/checkpoint/readback coverage".into(),
            ));
        }
        Ok((verified, copies, checkpoints))
    }
    pub(crate) fn require_initialized(&self) -> Result<()> {
        if self.weights.len()
            != self
                .canonical47_prefix
                .as_ref()
                .map_or(114, |c| c.weight_groups())
        {
            return Err(Error::State("full24 initialized W count".into()));
        }
        for weight in self.weights.values() {
            weight.require_initialized()?;
        }
        Ok(())
    }
}

mod short_hf;
