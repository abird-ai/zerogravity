//! Loaded catalog only; future target metadata never populates it.
use crate::{Result, Runtime, model_instance::ModelInstance};
pub struct ModelCatalog<'a> {
    runtime: &'a Runtime,
}
impl<'a> ModelCatalog<'a> {
    pub(crate) fn new(runtime: &'a Runtime) -> Self {
        Self { runtime }
    }
    pub fn list(&self) -> Result<Vec<ModelInstance>> {
        self.runtime.loaded_text_models()
    }
    pub fn resolve(&self, alias: &str) -> Result<ModelInstance> {
        self.runtime.resolve_text(alias)
    }
    /// All registered aliases of this exact prepared owner.
    /// Equal source IDs or an alias string do not establish ownership.
    pub fn aliases(&self, model: &ModelInstance) -> Result<Vec<String>> {
        self.runtime.text_model_aliases(model)
    }

    /// Retire this prepared owner and all its aliases, draining accepted work.
    /// Independent owners continue on the same workers if the adapter supports
    /// selective retirement. The last owner joins the workers, leaving an open
    /// runtime. External source/tokenizer readers retain their own ownership.
    pub fn unload(&self, model: &ModelInstance) -> Result<()> {
        self.runtime.unload_text_model(model)
    }
}
