//! Authentic all114 groups through the original bounded ordered H2D/checkpoints.
use super::super::text_model::TextModelBinding;
use super::*;

impl PendingModelWeights {
    pub(crate) async fn upload_text(
        scope: &Arc<TextModelBinding>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        if scope.projection_nodes().is_empty() {
            return Err(Error::State(
                "text exact114 actual source groups before H2D".into(),
            ));
        }
        let metadata = Arc::new(scope.source().reserve_workspace(512 * 1024)?);
        let mut weights = BTreeMap::new();
        for &node in scope.projection_nodes() {
            if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Err(Error::State(
                    "text original learned upload cancelled".into(),
                ));
            }
            let Some(descriptor) = scope.record().operator(node)? else {
                return Err(Error::State(
                    "text actual learned source node absent".into(),
                ));
            };
            let key = ModelWeightKey::of(descriptor)?;
            if weights.contains_key(&key) {
                return Err(Error::State("text duplicate layer/learned role".into()));
            }
            let pending = init
                .upload_model_weight(
                    scope.source().clone(),
                    descriptor.clone(),
                    ModelWeight::packing_layout(descriptor)?,
                    cancelled,
                )
                .await?;
            weights.insert(key, pending);
        }
        Ok(Self {
            weights,
            source: scope.source().clone(),
            expected_groups: scope.projection_nodes().len(),
            _metadata: metadata,
            fixed_prompt_prefix: None,
            text: Some(scope.clone()),
        })
    }
}
impl ModelWeights {
    pub(crate) fn require_text(&self, scope: &Arc<TextModelBinding>) -> Result<()> {
        if self.fixed_prompt_prefix.is_some()
            || self.text.as_ref().is_none_or(|own| {
                !own.source().same_issuer(scope.source())
                    || own.projection_nodes().len() != scope.projection_nodes().len()
            })
            || !self.source.same_issuer(scope.source())
            || self.weights.len() != scope.projection_nodes().len()
        {
            return Err(Error::State(
                "text exact source/family/shared114 immutable W groups".into(),
            ));
        }
        for &node in scope.projection_nodes() {
            let Some(descriptor) = scope.record().operator(node)? else {
                return Err(Error::State("text actual learned descriptor absent".into()));
            };
            self.require_operator(descriptor)?;
        }
        Ok(())
    }
    pub(super) fn expected_source_windows(&self) -> Result<usize> {
        self.weights.values().try_fold(0usize, |n, w| {
            n.checked_add(w.expected_source_windows()?)
                .ok_or_else(|| Error::Limit("learned complete window count".into()))
        })
    }
}

impl PendingModelWeights {
    pub(crate) fn allocation_layout(scope: &TextModelBinding) -> Result<Vec<usize>> {
        let mut keys = std::collections::BTreeSet::new();
        let mut result = Vec::with_capacity(scope.projection_nodes().len());
        for &node in scope.projection_nodes() {
            let Some(descriptor) = scope.record().operator(node)? else {
                return Err(Error::State(
                    "text actual source weight recipe absent".into(),
                ));
            };
            if !keys.insert(ModelWeightKey::of(descriptor)?) {
                return Err(Error::State("text duplicate inventory W group".into()));
            }
            result.push(ModelWeight::allocation_bytes(descriptor)?);
        }
        if result.len() != scope.projection_nodes().len() {
            return Err(Error::State("text exact114 allocator W groups".into()));
        }
        Ok(result)
    }
}
