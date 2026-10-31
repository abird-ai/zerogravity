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
        self.runtime.require_open()?;
        Ok(Vec::new())
    }
    pub fn load(&self, _request: &ModelLoadRequest) -> Result<ModelInstance> {
        self.runtime.require_open()?;
        Err(Error::Unsupported(
            "Fabric model loading awaits source-bound F0 adapters".into(),
        ))
    }
    pub fn unload(&self, _model: &ModelInstance) -> Result<()> {
        self.runtime.require_open()?;
        Err(Error::Unsupported(
            "Fabric model retirement awaits F0 adapters".into(),
        ))
    }
}
