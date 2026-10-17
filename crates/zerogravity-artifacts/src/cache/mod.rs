//! Opt-in, offline artifact acquisition transactions. A published snapshot is
//! verified data, never model/device readiness or a session state authority.
//!
//! Linux-local-filesystem first. Quotas coordinate cooperating writers within
//! this cache only, not arbitrary filesystem users or all process/UMA memory.
//! Read-only descriptors do not protect against a malicious same-UID writer;
//! the later model-loader bridge must verify the bytes of its owned copy.

mod leases;
pub mod manifest;
pub mod paths;
mod store;

pub use leases::SnapshotLease;
pub use manifest::{
    CacheLimits, DigestProvenance, FileRole, Manifest, ManifestEntry, ManifestSource,
};
pub use store::{CacheStore, CacheTransaction, DiscoveredEntry, DiscoveredTransaction};

use crate::Error;
#[cfg(feature = "diagnostic-cache")]
use crate::Result;
use std::path::PathBuf;
use std::time::Duration;
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

#[derive(Clone, Debug)]
pub struct CacheOptions {
    pub root: Option<PathBuf>,
    pub disk_limit_bytes: u64,
    pub limits: CacheLimits,
    pub lock_timeout: Duration,
    pub buffer_bytes: usize,
}

impl CacheOptions {
    pub fn new(root: Option<PathBuf>, disk_limit_bytes: u64) -> Self {
        Self {
            root,
            disk_limit_bytes,
            limits: CacheLimits::default(),
            lock_timeout: Duration::from_secs(30),
            buffer_bytes: 64 * 1024,
        }
    }
}

/// Borrow an existing, caller-registered resource domain. The cache never
/// creates an arbiter, registers a physical key, or submits a GPU operation.
#[derive(Clone)]
pub struct TransferAdmission {
    pub arbiter: ResourceArbiter,
    pub resource: PhysicalDeviceKey,
    pub owner: ResourceOwner,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishedSnapshot {
    pub(crate) digest: String,
    pub(crate) reference: Option<String>,
}

impl PublishedSnapshot {
    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn reference(&self) -> Option<&str> {
        self.reference.as_deref()
    }
}

#[derive(Debug)]
pub enum CommitOutcome {
    Durable(PublishedSnapshot),
    /// The new complete binding is visible, but its durability was not proved.
    /// Never retry blindly or claim that the prior reference was rolled back.
    DurabilityUnknown {
        snapshot: PublishedSnapshot,
        reference: Option<String>,
        cause: Error,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilesystemProfile {
    pub block_bytes: u64,
    pub fragment_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskUsage {
    pub used_bytes: u64,
    pub reserved_bytes: u64,
    pub limit_bytes: u64,
}

#[cfg(feature = "diagnostic-cache")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOperation {
    DiscoveredManifestReady,
    ReservationFileSync,
    ReservationDirectorySync,
    StagingDirectorySync,
    ObjectFileSync,
    ObjectInstall,
    ObjectDirectorySync,
    ManifestFileSync,
    ManifestInstall,
    ManifestDirectorySync,
    SnapshotEntryInstall,
    SnapshotDirectorySync,
    SnapshotInstall,
    SnapshotParentSync,
    RefFileSync,
    ReferenceRecordInstall,
    ReferenceRecordDirectorySync,
    RefRename,
    RefDirectorySync,
}

#[cfg(feature = "diagnostic-cache")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheMoment {
    Before,
    After,
}

#[cfg(feature = "diagnostic-cache")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachePoint {
    pub operation: CacheOperation,
    pub moment: CacheMoment,
    pub object: Option<usize>,
}

/// Explicit diagnostic callback; production has no environment-controlled
/// failure injection. Tests may return an I/O error or terminate their child.
#[cfg(feature = "diagnostic-cache")]
pub trait CacheHook: Send + Sync {
    fn point(&self, point: CachePoint) -> Result<()>;
}
