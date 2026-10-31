//! Alias resolution must pin an actual loaded generation before execution.
use crate::{Error, Result, Runtime, model_instance::ModelInstance};
pub fn resolve(runtime: &Runtime, alias: &str) -> Result<ModelInstance> {
    runtime.require_open()?;
    zerogravity_core::run_policy::SourceSelector::parse(alias)?;
    Err(Error::Unsupported(
        "Fabric alias routing awaits loaded model catalog".into(),
    ))
}
