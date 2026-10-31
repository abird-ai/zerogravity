//! Existing typed validation is real; native pipeline execution remains unsupported.
pub use zerogravity_core::pipeline::{ComponentRegistry, Config, Pipeline, VerifiedShape};
pub fn check(config: &Config) -> crate::Result<VerifiedShape> {
    config.validate()
}
pub fn run(_config: &Config) -> crate::Result<()> {
    Err(crate::Error::Unsupported(
        "native task pipeline execution is not implemented".into(),
    ))
}
