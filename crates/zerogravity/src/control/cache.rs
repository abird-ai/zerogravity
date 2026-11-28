//! Cold cache observations and explicit actions on the original cache owners.
use super::{ControlEventKind, ControlPlane, ReclaimStatus};
use crate::{
    Error, Result,
    cache::{
        CacheMetadata, CacheOptions, CacheStore, Manifest, ReclaimMode, ReclaimOutcome,
        SnapshotMetadata,
    },
    run_policy::SourceSelector,
};
use std::{path::Path, sync::atomic::AtomicBool};

#[derive(Debug, serde::Serialize)]
pub struct CachedModelSummary {
    pub snapshot: String,
    pub provider: String,
    pub repository: String,
    pub revision: String,
    pub bytes: u64,
    pub files: usize,
    /// Listing never authenticates source payloads.
    pub payload_verified: bool,
    /// Manifest declares at least one weight payload; not a runnable guarantee.
    pub has_weights: bool,
    pub source_layout: crate::cache::SourceLayout,
}

impl From<SnapshotMetadata> for CachedModelSummary {
    fn from(value: SnapshotMetadata) -> Self {
        let source = value.manifest.source();
        let source_layout = value.manifest.source_layout();
        Self {
            snapshot: value.digest,
            provider: source.provider().into(),
            repository: source.repository().into(),
            revision: source.revision().into(),
            bytes: value.manifest.total_bytes(),
            files: value.manifest.entries().len(),
            payload_verified: false,
            has_weights: source_layout.weights != 0,
            source_layout,
        }
    }
}

/// Observation only: even verified content does not grant execution or retain
/// a source lease. The actual model loader must authenticate its own reader.
#[derive(Debug)]
pub struct CachedModelDetail {
    pub snapshot: String,
    pub manifest: Manifest,
    pub payload_verified: bool,
}

impl ControlPlane<'_> {
    pub fn cached_models(&self, root: &Path) -> Result<Vec<CachedModelSummary>> {
        self.runtime.engine.require_open()?;
        let metadata = match CacheMetadata::open(root) {
            Ok(metadata) => metadata,
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Vec::new());
            }
            Err(error) => return Err(error),
        };
        Ok(metadata.snapshots()?.into_iter().map(Into::into).collect())
    }

    /// Existing checkpoint records, not verified or guaranteed resumable.
    pub fn cached_partials(&self, root: &Path) -> Result<Vec<crate::cache::PartialSummary>> {
        self.runtime.engine.require_open()?;
        match CacheMetadata::open(root) {
            Ok(metadata) => metadata.partials(),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(error) => Err(error),
        }
    }

    pub fn inspect_cached(
        &self,
        root: &Path,
        selector: &SourceSelector,
        pin: Option<&str>,
    ) -> Result<CachedModelDetail> {
        self.runtime.engine.require_open()?;
        let selected = crate::models::select_cached(&CacheMetadata::open(root)?, selector, pin)?;
        Ok(CachedModelDetail {
            snapshot: selected.digest,
            manifest: selected.manifest,
            payload_verified: false,
        })
    }

    pub fn verify_cached(
        &self,
        root: &Path,
        selector: &SourceSelector,
        pin: Option<&str>,
        cancel: &AtomicBool,
    ) -> Result<CachedModelDetail> {
        self.verify_cached_observed(root, selector, pin, cancel, |_| {})
    }

    pub fn verify_cached_observed(
        &self,
        root: &Path,
        selector: &SourceSelector,
        pin: Option<&str>,
        cancel: &AtomicBool,
        observe: impl FnMut(crate::cache::VerificationProgress),
    ) -> Result<CachedModelDetail> {
        let result = (|| {
            self.runtime.engine.require_open()?;
            let metadata = CacheMetadata::open(root)?;
            let selected = crate::models::select_cached(&metadata, selector, pin)?;
            let store = CacheStore::open_existing(CacheOptions::new(
                Some(root.to_owned()),
                metadata.disk_limit_bytes()?,
            ))?;
            let lease = store.open_snapshot_observed(&selected.digest, cancel, observe)?;
            Ok(CachedModelDetail {
                snapshot: selected.digest,
                manifest: lease.manifest().clone(),
                payload_verified: true,
            })
        })();
        self.emit(ControlEventKind::CacheVerificationFinished {
            success: result.is_ok(),
        });
        result
    }

    /// Add a human alias for the exact snapshot. Never silently retarget an
    /// existing alias; the original publisher/projection owner validates.
    pub fn alias_cached(
        &self,
        root: &Path,
        digest: &str,
        name: &str,
    ) -> Result<crate::cache::ByNameUpdate> {
        let result = (|| {
            self.runtime.engine.require_open()?;
            crate::cache::paths::validate_digest(digest)?;
            let metadata = CacheMetadata::open(root)?;
            let store = CacheStore::open_existing(CacheOptions::new(
                Some(root.to_owned()),
                metadata.disk_limit_bytes()?,
            ))?;
            store.set_by_name_alias(name, digest, false)
        })();
        self.emit(ControlEventKind::CacheAliasFinished {
            success: result.is_ok(),
        });
        result
    }

    /// A mutation targets an exact digest selected/confirmed by the caller.
    /// Preserve all original Busy/Referenced/Incomplete/Durable outcomes.
    pub fn reclaim_cached(
        &self,
        root: &Path,
        digest: &str,
        mode: ReclaimMode,
        cancel: &AtomicBool,
    ) -> Result<ReclaimOutcome> {
        let result = (|| {
            self.runtime.engine.require_open()?;
            crate::cache::paths::validate_digest(digest)?;
            let metadata = CacheMetadata::open(root)?;
            let store = CacheStore::open_existing(CacheOptions::new(
                Some(root.to_owned()),
                metadata.disk_limit_bytes()?,
            ))?;
            store.reclaim_snapshot(digest, mode, cancel)
        })();
        let outcome = match &result {
            Ok(ReclaimOutcome::DryRun(_)) => ReclaimStatus::DryRun,
            Ok(ReclaimOutcome::Busy) => ReclaimStatus::Busy,
            Ok(ReclaimOutcome::Referenced { .. }) => ReclaimStatus::Referenced,
            Ok(ReclaimOutcome::Durable(_)) => ReclaimStatus::Durable,
            Ok(ReclaimOutcome::Incomplete { .. }) => ReclaimStatus::Incomplete,
            Err(_) => ReclaimStatus::Failed,
        };
        self.emit(ControlEventKind::CacheReclamationFinished { outcome });
        result
    }
}
