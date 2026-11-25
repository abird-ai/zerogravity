//! CPU sessions over the shared admitted local source loader.
use super::*;

pub(super) fn load_models(
    declared: super::super::local::LocalRoutes,
    host: &ServingHost,
    max_context: usize,
    targets: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, Arc<dyn LogitModel>>, ServiceError> {
    declared.prepare(|id, path, _| {
        let super::super::local::LocalModel {
            weights,
            tokenizer,
            max_token_bytes,
        } = super::super::local::load(id, path, host, targets.get(id).map(String::as_str))?;
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
