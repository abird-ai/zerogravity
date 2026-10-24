//! Authentic all114 groups through the original bounded ordered H2D/checkpoints.
use super::super::short_hf_model::ShortHfModelBinding;
use super::*;
use zerogravity_core::accelerator::program::FullModelProfile;

impl PendingModelWeights {
    pub(crate) async fn upload_short_hf(
        scope: &Arc<ShortHfModelBinding>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        if scope.projection_nodes().len() != 114 {
            return Err(Error::State(
                "short-HF exact114 actual source groups before H2D".into(),
            ));
        }
        let mut weights = BTreeMap::new();
        for &node in scope.projection_nodes() {
            if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Err(Error::State(
                    "short-HF original learned upload cancelled".into(),
                ));
            }
            let PlanOperation::TypedVllm08 { descriptor, .. } =
                &scope.record().plan().nodes()[node].operation
            else {
                return Err(Error::State(
                    "short-HF actual learned source node absent".into(),
                ));
            };
            let key = ModelWeightKey::of(descriptor)?;
            if weights.contains_key(&key) {
                return Err(Error::State("short-HF duplicate layer/learned role".into()));
            }
            let pending = init
                .upload_short_hf_model_weight(scope, node, cancelled)
                .await?;
            weights.insert(key, pending);
        }
        Ok(Self {
            weights,
            source: scope.source().source().clone(),
            canonical47_prefix: None,
            short_hf: Some(scope.clone()),
        })
    }
}
impl ModelWeights {
    pub(crate) fn require_short_hf(&self, scope: &Arc<ShortHfModelBinding>) -> Result<()> {
        if self.canonical47_prefix.is_some()
            || self.short_hf.as_ref().is_none_or(|own| {
                !Arc::ptr_eq(own.source().source(), scope.source().source())
                    || own.call().model_profile() != scope.call().model_profile()
            })
            || !Arc::ptr_eq(&self.source, scope.source().source())
            || self.weights.len() != 114
        {
            return Err(Error::State(
                "short-HF exact source/family/shared114 immutable W groups".into(),
            ));
        }
        for &node in scope.projection_nodes() {
            let PlanOperation::TypedVllm08 { descriptor, .. } =
                &scope.record().plan().nodes()[node].operation
            else {
                return Err(Error::State(
                    "short-HF actual learned descriptor absent".into(),
                ));
            };
            self.require_operator(descriptor)?;
        }
        Ok(())
    }
    pub(super) fn expected_source_windows(&self) -> usize {
        match self
            .short_hf
            .as_ref()
            .map(|scope| scope.call().model_profile())
        {
            Some(FullModelProfile::Qwen08) => 1008,
            Some(FullModelProfile::Qwen2B) => 2676,
            None => self
                .canonical47_prefix
                .as_ref()
                .map_or(1008, |c| c.source_windows()),
        }
    }
}

impl PendingModelWeights {
    pub(crate) fn allocation_layout(scope: &ShortHfModelBinding) -> Result<Vec<usize>> {
        let mut keys = std::collections::BTreeSet::new();
        let mut result = Vec::with_capacity(114);
        for &node in scope.projection_nodes() {
            let PlanOperation::TypedVllm08 { descriptor, .. } =
                &scope.record().plan().nodes()[node].operation
            else {
                return Err(Error::State(
                    "short-HF actual source weight recipe absent".into(),
                ));
            };
            if !keys.insert(ModelWeightKey::of(descriptor)?) {
                return Err(Error::State("short-HF duplicate inventory W group".into()));
            }
            result.push(ModelWeight::allocation_bytes(descriptor)?);
        }
        if result.len() != 114 {
            return Err(Error::State("short-HF exact114 allocator W groups".into()));
        }
        Ok(result)
    }
}
