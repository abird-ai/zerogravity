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

#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod pull;
#[cfg(all(feature = "acquisition", target_os = "linux"))]
pub use pull::{HubPull, HubSelection, PreparedHubPull, PullMode};

/// Resolve a digest or human reference through the original verified cache.
#[cfg(all(feature = "cache", target_os = "linux"))]
pub fn cached(store: &cache::CacheStore, selector: &str) -> crate::Result<cache::SnapshotLease> {
    if selector.len() == 64
        && selector
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        store.open_snapshot(selector)
    } else {
        store.resolve_ref(selector)
    }
}

/// Original process-shared transfer pool, not host-RAM or inference admission.
/// An existing pool must retain its configured limit; no per-call fresh ledger.
#[cfg(all(feature = "cache", target_os = "linux"))]
pub fn transfer_admission(memory: u64) -> crate::Result<cache::TransferAdmission> {
    use crate::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "artifact-transfer-staging")?;
    arbiter.register_matching(resource.clone(), memory)?;
    Ok(cache::TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("artifact-acquisition", "sdk")?,
    })
}

#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod components;
#[cfg(all(feature = "acquisition", target_os = "linux"))]
pub use components::{ComponentClients, ComponentDocument};

/// Unverified selection only; resolution must retain the actual snapshot lease.
#[cfg(all(feature = "cache", target_os = "linux"))]
#[derive(Clone, Copy)]
pub struct SnapshotFile<'a> {
    pub snapshot: &'a str,
    pub path: &'a str,
}
