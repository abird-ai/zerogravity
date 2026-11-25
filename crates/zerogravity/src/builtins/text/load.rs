//! CPU sessions over the shared admitted local source loader.
use super::*;

pub(super) fn load_models(
    declared: super::super::local::GgufRoutes,
    host: &ServingHost,
    max_context: usize,
) -> Result<BTreeMap<String, Arc<dyn LogitModel>>, ServiceError> {
    declared.prepare(|id, source, target, _| {
        let super::super::local::GgufModel {
            weights,
            tokenizer,
            max_token_bytes,
        } = super::super::local::load(id, source, host, target, max_context)?;
        let max_session_bytes = session_budget(weights.model(), max_context)?;
        Ok(Arc::new(Model {
            weights,
            tokenizer,
            max_context,
            max_session_bytes,
            max_token_bytes,
        }) as Arc<dyn LogitModel>)
    })
}
