//! Loaded catalog only; future target metadata never populates it.
use crate::{Error, Result, Runtime, model_instance::ModelInstance};
use zerogravity_core::run_policy::SourceSelector;

#[derive(Clone, Debug)]
pub struct ModelLoadRequest {
    pub source: SourceSelector,
}
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
    pub fn load(&self, _request: &ModelLoadRequest) -> Result<ModelInstance> {
        self.runtime.require_open()?;
        Err(Error::Unsupported(
            "source loading requires the facade's family adapter; use Runtime::load_text".into(),
        ))
    }
    pub fn unload(&self, _model: &ModelInstance) -> Result<()> {
        self.runtime.require_open()?;
        Err(Error::Unsupported(
            "per-model retirement is not implemented; runtime shutdown joins workers and retires its sources".into(),
        ))
    }
}
