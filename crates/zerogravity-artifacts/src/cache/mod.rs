//! Opt-in, offline artifact acquisition and explicit cache maintenance. A snapshot is
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
pub use store::{
    CacheStore, CacheTransaction, DiscoveredEntry, DiscoveredObject, DiscoveredTransaction,
    PartialObject,
};

/// An established length or an explicit reservation bound. A bound is never
/// an observed object length; published manifest entries always contain exact
/// actual counts after the caller's source-completion checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteLength {
    Exact(u64),
    AtMost(u64),
}

impl ByteLength {
    pub const fn exact(self) -> Option<u64> {
        match self {
            Self::Exact(bytes) => Some(bytes),
            Self::AtMost(_) => None,
        }
    }

    pub const fn maximum(self) -> u64 {
        match self {
            Self::Exact(bytes) | Self::AtMost(bytes) => bytes,
        }
    }
}

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

/// Exact-target maintenance is explicit and never an automatic eviction policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReclaimMode {
    DryRun,
    Apply,
}

/// Observed cooperative cache charge and namespace work, not filesystem free
/// space. Entry counts include hard links; they are not counts of freed inodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReclaimReport {
    pub(crate) snapshot: String,
    pub(crate) retired: bool,
    pub(crate) used_before_bytes: u64,
    pub(crate) used_after_bytes: Option<u64>,
    pub(crate) candidate_files: usize,
    pub(crate) candidate_directories: usize,
    pub(crate) removed_files: usize,
    pub(crate) removed_directories: usize,
    pub(crate) retained_blobs: usize,
}

impl ReclaimReport {
    pub fn snapshot(&self) -> &str {
        &self.snapshot
    }

    /// Whether this request observed or began retirement. This does not mean a
    /// retired name still exists after successful completion.
    pub fn retired(&self) -> bool {
        self.retired
    }

    pub fn used_before_bytes(&self) -> u64 {
        self.used_before_bytes
    }

    /// None means a post-mutation inventory failed; no current charge or
    /// released charge was established for that incomplete operation.
    pub fn used_after_bytes(&self) -> Option<u64> {
        self.used_after_bytes
    }

    pub fn candidate_files(&self) -> usize {
        self.candidate_files
    }

    pub fn candidate_directories(&self) -> usize {
        self.candidate_directories
    }

    pub fn removed_files(&self) -> usize {
        self.removed_files
    }

    pub fn removed_directories(&self) -> usize {
        self.removed_directories
    }

    pub fn retained_blobs(&self) -> usize {
        self.retained_blobs
    }
}

#[derive(Debug)]
pub enum ReclaimOutcome {
    DryRun(ReclaimReport),
    /// An activity or payload descriptor pin prevents this attempt. No target
    /// namespace entry has been changed.
    Busy,
    Referenced {
        references: usize,
    },
    Durable(ReclaimReport),
    /// Mutation began; retained names permit exact-target retry. Do not infer
    /// rollback or successful durability from a visible namespace change.
    Incomplete {
        report: ReclaimReport,
        cause: Error,
    },
}

#[cfg(feature = "diagnostic-cache")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOperation {
    PartialFileSync,
    PartialRecordSync,
    PartialRecordInstall,
    PartialDirectorySync,
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
    RetirementRename,
    ReclaimUnlink,
    ReclaimDirectorySync,
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
