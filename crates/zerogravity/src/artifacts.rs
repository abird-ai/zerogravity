//! Explicit source IO and existing cache/acquisition APIs. No constructor IO.
#[cfg(all(feature = "acquisition", target_os = "linux"))]
pub use zerogravity_artifacts::acquisition;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub use zerogravity_artifacts::cache;
pub use zerogravity_artifacts::{
    SourceWeights,
    artifact::{Artifact, ArtifactInfo},
    derived,
};

pub struct ArtifactService<'a> {
    runtime: &'a zerogravity_engine::Runtime,
}
impl<'a> ArtifactService<'a> {
    pub(crate) fn new(runtime: &'a zerogravity_engine::Runtime) -> Self {
        Self { runtime }
    }
    /// Bounded metadata parsing only; never model eligibility.
    pub fn inspect(&self, path: &std::path::Path) -> crate::Result<ArtifactInfo> {
        self.runtime.require_open()?;
        Ok(Artifact::open(path)?.info)
    }
}
