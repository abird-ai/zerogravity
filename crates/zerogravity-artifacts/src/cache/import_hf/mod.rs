//! Explicit, read-only import from one caller-selected HF repository cache.
//! Repository/commit labels describe a local assertion, not authenticated Hub
//! membership. No provider, implicit closure, executable source or ref is issued.

mod source;

use super::paths::{RootDir, validate_digest, validate_relative_path};
use super::{
    CacheLimits, CacheOptions, CacheStore, CommitOutcome, DigestProvenance, DiscoveredEntry,
    FileRole, ManifestSource, TransferAdmission,
};
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};

/// One exact logical path and explicitly assigned data role. An optional SHA256
/// is caller-authorized expected content; HF blob names never supply that trust.
#[derive(Clone, Debug)]
pub struct HfImportEntry {
    path: String,
    role: FileRole,
    sha256: Option<String>,
}

impl HfImportEntry {
    pub fn new(path: impl Into<String>, role: FileRole, sha256: Option<String>) -> Result<Self> {
        let path = path.into();
        validate_relative_path(&path)?;
        if let Some(hash) = &sha256 {
            validate_digest(hash)?;
        }
        Ok(Self { path, role, sha256 })
    }
}

/// Explicit source selection. No cache discovery, mutable refs, credentials or
/// model-closure inference is performed.
#[derive(Clone, Debug)]
pub struct HfImportRequest {
    root: PathBuf,
    repository: String,
    commit: String,
    entries: Vec<HfImportEntry>,
}

impl HfImportRequest {
    pub fn new(
        root: PathBuf,
        repository: impl Into<String>,
        commit: impl Into<String>,
        entries: Vec<HfImportEntry>,
    ) -> Result<Self> {
        let repository = repository.into();
        let commit = commit.into();
        if entries.is_empty() || entries.len() > CacheLimits::default().max_files {
            return Err(Error::Limit("HF import file count".into()));
        }
        let hard_limits = CacheLimits {
            max_manifest_bytes: 8 * 1024 * 1024,
            ..CacheLimits::default()
        };
        let selected: Vec<_> = entries.iter().map(|entry| entry.path.clone()).collect();
        source::validate_request(&root, &repository, &commit, &selected, &hard_limits)?;
        let mut paths = BTreeSet::new();
        for entry in &entries {
            if !paths.insert(entry.path.as_str()) {
                return Err(Error::Malformed("duplicate HF import path".into()));
            }
        }
        // Reuse the canonical manifest path-collision rules with placeholders
        // before source metadata work. These are not publishable identities.
        let request = Self {
            root,
            repository,
            commit,
            entries,
        };
        request.validate_paths(&hard_limits)?;
        Ok(request)
    }

    fn source(&self) -> Result<ManifestSource> {
        ManifestSource::new("local-hf-cache", &self.repository, &self.commit)
    }

    fn validate_paths(&self, limits: &CacheLimits) -> Result<()> {
        let entries = self
            .entries
            .iter()
            .map(|entry| super::ManifestEntry::new(&entry.path, "0".repeat(64), 0, entry.role))
            .collect::<Result<Vec<_>>>()?;
        super::Manifest::new(self.source()?, entries, limits)?;
        Ok(())
    }

    /// Metadata-only preparation, admitted before opening source descriptors.
    /// Its control lease and original descriptors remain owned through apply.
    pub fn prepare(
        self,
        limits: &CacheLimits,
        admission: &TransferAdmission,
        cancel: Arc<AtomicBool>,
    ) -> Result<HfImportPlan> {
        check_cancel(&cancel)?;
        limits.validate()?;
        let control_bytes = (limits.max_manifest_bytes as u64)
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_add(16 * 1024 * 1024))
            .ok_or_else(|| Error::Limit("HF import control envelope overflow".into()))?;
        let control = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: control_bytes,
            }],
        )?;
        self.validate_paths(limits)?;
        let selected: Vec<_> = self
            .entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect();
        let view = source::SourceView::open(
            &self.root,
            &self.repository,
            &self.commit,
            &selected,
            limits,
            &cancel,
        )?;
        let mut entries = Vec::with_capacity(self.entries.len());
        let mut total_bytes = 0u64;
        for selected in self.entries {
            let opened = view
                .entries()
                .iter()
                .find(|opened| opened.path() == selected.path)
                .ok_or_else(|| Error::State("HF source entry missing".into()))?;
            total_bytes = total_bytes
                .checked_add(opened.bytes())
                .ok_or_else(|| Error::Limit("HF import total overflow".into()))?;
            entries.push(HfImportPlannedEntry {
                path: selected.path,
                role: selected.role,
                bytes: opened.bytes(),
                sha256: selected.sha256,
            });
        }
        entries.sort_unstable_by(|left, right| left.path.cmp(&right.path));
        check_cancel(&cancel)?;
        Ok(HfImportPlan {
            source: ManifestSource::new("local-hf-cache", self.repository, self.commit)?,
            entries,
            total_bytes,
            view,
            admission: admission.clone(),
            cancel,
            _control: control,
        })
    }
}

#[derive(Clone, Debug)]
pub struct HfImportPlannedEntry {
    path: String,
    role: FileRole,
    bytes: u64,
    sha256: Option<String>,
}

impl HfImportPlannedEntry {
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn role(&self) -> FileRole {
        self.role
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn expected_sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }
}

/// Retains the read-only source view and the caller's admitted control owner.
pub struct HfImportPlan {
    source: ManifestSource,
    entries: Vec<HfImportPlannedEntry>,
    total_bytes: u64,
    view: source::SourceView,
    admission: TransferAdmission,
    cancel: Arc<AtomicBool>,
    // Original source handles close before their control reservation releases.
    _control: GlobalResourceLease,
}

impl HfImportPlan {
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }
    pub fn entries(&self) -> &[HfImportPlannedEntry] {
        &self.entries
    }
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    fn destination(&self, options: &CacheOptions) -> Result<RootDir> {
        check_cancel(&self.cancel)?;
        CacheStore::validate_options(options)?;
        let path = options.root.as_deref().ok_or_else(|| {
            Error::Malformed("HF import requires an explicit existing destination".into())
        })?;
        let root = RootDir::open(path, false)?;
        self.view.ensure_disjoint(root.as_fd(), &self.cancel)?;
        check_cancel(&self.cancel)?;
        Ok(root)
    }

    /// Read-only overlap proof. Missing destinations refuse without creation;
    /// a later apply repeats the proof rather than treating this as a permit.
    pub fn check_destination(&self, options: &CacheOptions) -> Result<()> {
        self.destination(options)?;
        Ok(())
    }

    /// Copy selected local data through the existing admitted A1 transaction.
    /// The existing destination may be empty. No source name/inode is written,
    /// locked or linked; no mutable reference is published.
    pub fn apply(self, options: CacheOptions) -> Result<HfImportReceipt> {
        self.apply_inner(
            options,
            #[cfg(feature = "diagnostic-cache")]
            None,
        )
    }

    /// Use the original cache fault schedule; no importer publication protocol
    /// or additional hook schedule is introduced. Absent in production builds.
    #[cfg(feature = "diagnostic-cache")]
    pub fn apply_with_hook(
        self,
        options: CacheOptions,
        hook: Arc<dyn super::CacheHook>,
    ) -> Result<HfImportReceipt> {
        self.apply_inner(options, Some(hook))
    }

    fn apply_inner(
        self,
        options: CacheOptions,
        #[cfg(feature = "diagnostic-cache")] hook: Option<Arc<dyn super::CacheHook>>,
    ) -> Result<HfImportReceipt> {
        let root = self.destination(&options)?;
        // This is the same preflighted capability, never a pathname reopen.
        let store = CacheStore::initialize_import_root(options, root)?;
        #[cfg(feature = "diagnostic-cache")]
        let store = if let Some(hook) = hook {
            store.with_hook(hook)
        } else {
            store
        };
        let discovered = self
            .entries
            .iter()
            .map(|entry| {
                DiscoveredEntry::new(&entry.path, entry.bytes, entry.role, entry.sha256.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut transaction = store.begin_discovered(
            self.source.clone(),
            discovered,
            &self.admission,
            self.cancel.clone(),
        )?;
        let mut entries = Vec::with_capacity(self.entries.len());
        for planned in &self.entries {
            check_cancel(&self.cancel)?;
            let opened = self
                .view
                .entries()
                .iter()
                .find(|entry| entry.path() == planned.path)
                .ok_or_else(|| Error::State("HF source entry missing".into()))?;
            opened.ensure_stable(&self.cancel)?;
            let mut reader =
                ObservedSource::new(opened.reader(&self.cancel)?, planned.bytes, &self.cancel);
            let result = transaction.import_object(&planned.path, &mut reader);
            check_cancel(&self.cancel)?;
            reader.check_lfs()?;
            let actual = result?;
            let copied_bytes = reader.bytes;
            // Known-digest dedup may have skipped this reader entirely. Still
            // hash the actual external bytes under the transaction admission.
            reader.verify(actual.digest())?;
            opened.ensure_stable(&self.cancel)?;
            entries.push(HfImportEntryReceipt {
                path: actual.path().to_owned(),
                role: actual.role(),
                bytes: actual.bytes(),
                digest: actual.digest().to_owned(),
                provenance: actual.digest_provenance(),
                copied_bytes,
            });
        }
        self.view.ensure_stable(&self.cancel)?;
        check_cancel(&self.cancel)?;
        let outcome = transaction.commit(None)?;
        Ok(HfImportReceipt {
            source: self.source.clone(),
            entries,
            outcome,
        })
    }
}

#[derive(Debug)]
pub struct HfImportEntryReceipt {
    path: String,
    role: FileRole,
    bytes: u64,
    digest: String,
    provenance: DigestProvenance,
    copied_bytes: u64,
}

impl HfImportEntryReceipt {
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn role(&self) -> FileRole {
        self.role
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub fn digest_provenance(&self) -> DigestProvenance {
        self.provenance
    }
    /// Bytes consumed by the A1 importer. A verified known-cache hit consumes
    /// zero; its external source body is nevertheless fully hashed before commit.
    pub fn copied_bytes(&self) -> u64 {
        self.copied_bytes
    }
}

#[derive(Debug)]
pub struct HfImportReceipt {
    source: ManifestSource,
    entries: Vec<HfImportEntryReceipt>,
    outcome: CommitOutcome,
}

impl HfImportReceipt {
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }
    pub fn entries(&self) -> &[HfImportEntryReceipt] {
        &self.entries
    }
    pub fn outcome(&self) -> &CommitOutcome {
        &self.outcome
    }
    pub fn into_outcome(self) -> CommitOutcome {
        self.outcome
    }
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::State("HF cache import canceled".into()))
    } else {
        Ok(())
    }
}

struct ObservedSource<'a, R> {
    inner: R,
    expected: u64,
    bytes: u64,
    ended: bool,
    prefix: [u8; 128],
    prefix_len: usize,
    cancel: &'a AtomicBool,
}

impl<'a, R: Read> ObservedSource<'a, R> {
    fn new(inner: R, expected: u64, cancel: &'a AtomicBool) -> Self {
        Self {
            inner,
            expected,
            bytes: 0,
            ended: false,
            prefix: [0; 128],
            prefix_len: 0,
            cancel,
        }
    }

    fn verify(mut self, digest: &str) -> Result<()> {
        check_cancel(self.cancel)?;
        self.check_lfs()?;
        // The existing A1 importer already hashed the actual copied bytes and
        // proved exact EOF. Keep one SHA pass for this normal route.
        if self.bytes == self.expected && self.ended {
            return Ok(());
        }
        if self.bytes != 0 || self.ended {
            return Err(Error::State(
                "HF importer only partially consumed source".into(),
            ));
        }
        // Its verified known-digest fast path consumes no source body. Hash
        // those actual external bytes independently before acknowledging reuse.
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        while !self.ended {
            let result = self.read(&mut buffer);
            check_cancel(self.cancel)?;
            self.check_lfs()?;
            let count = result?;
            hasher.update(&buffer[..count]);
        }
        check_cancel(self.cancel)?;
        if self.bytes != self.expected || format!("{:x}", hasher.finalize()) != digest {
            return Err(Error::Malformed("HF import source content mismatch".into()));
        }
        Ok(())
    }

    fn check_lfs(&self) -> Result<()> {
        if self.prefix[..self.prefix_len].starts_with(b"version https://git-lfs.github.com/spec/v1")
        {
            Err(Error::Unsupported(
                "local HF LFS pointer payload refused".into(),
            ))
        } else {
            Ok(())
        }
    }
}

impl<R: Read> Read for ObservedSource<'_, R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.cancel.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "HF import canceled",
            ));
        }
        let maximum = self.expected.saturating_sub(self.bytes).saturating_add(1);
        let count = output.len().min(maximum.min(usize::MAX as u64) as usize);
        let count = self.inner.read(&mut output[..count])?;
        if self.cancel.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "HF import canceled",
            ));
        }
        self.bytes = self.bytes.checked_add(count as u64).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "HF import source count overflow",
            )
        })?;
        if self.bytes > self.expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HF import source grew",
            ));
        }
        let prefix_count = count.min(self.prefix.len() - self.prefix_len);
        self.prefix[self.prefix_len..self.prefix_len + prefix_count]
            .copy_from_slice(&output[..prefix_count]);
        self.prefix_len += prefix_count;
        if self.check_lfs().is_err() {
            return Err(io::Error::other("local HF LFS pointer payload refused"));
        }
        self.ended = count == 0;
        Ok(count)
    }
}
