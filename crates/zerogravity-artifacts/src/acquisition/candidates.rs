//! Metadata-only file discovery. Choices are not a transfer plan, model closure,
//! verified tensor quantization or remote authorization proof for cached bytes.
use super::http::{HttpRequest, HttpResponse, HttpTransport, check_cancel};
use super::{Provider, PullLimits};
use crate::cache::{
    DigestProvenance, FileRole, Manifest, ManifestSource, SnapshotLease, TransferAdmission,
    paths::{validate_digest, validate_relative_path},
};
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

#[derive(Clone, Debug)]
pub struct CandidateRequest {
    provider: Provider,
    repository: String,
    revision: String,
    subfolder: Option<String>,
}
impl CandidateRequest {
    pub fn new(
        provider: Provider,
        repository: &str,
        revision: &str,
        subfolder: Option<&str>,
    ) -> Result<Self> {
        for (label, maximum) in [(repository, 4096), (revision, 256)] {
            if label.is_empty()
                || label.len() > maximum
                || label.chars().any(char::is_control)
                || label.contains(['\\', '@', '?', '#'])
                || label.contains("://")
            {
                return Err(Error::Malformed("invalid candidate source labels".into()));
            }
        }
        match provider {
            Provider::HuggingFace => super::huggingface::validate_repo(repository)?,
            Provider::GitHub => super::github::repository(repository)?,
        }
        if let Some(path) = subfolder {
            validate_relative_path(path)?;
        }
        Ok(Self {
            provider,
            repository: repository.to_owned(),
            revision: revision.to_owned(),
            subfolder: subfolder.map(str::to_owned),
        })
    }
    pub fn provider(&self) -> Provider {
        self.provider
    }
    pub fn repository(&self) -> &str {
        &self.repository
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
    pub fn subfolder(&self) -> Option<&str> {
        self.subfolder.as_deref()
    }
    pub(super) fn includes(&self, path: &str) -> bool {
        self.subfolder.as_ref().is_none_or(|folder| {
            path.strip_prefix(folder)
                .is_some_and(|rest| rest.starts_with('/'))
        })
    }
}

/// Coverage of this metadata result, not content/authorization strength.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateCoverage {
    PinnedRepositoryTree,
    VerifiedCacheSubset,
}

#[derive(Clone, Debug)]
pub struct CandidateFile {
    path: String,
    bytes: Option<u64>,
    role: FileRole,
    sha256: Option<String>,
}
impl CandidateFile {
    pub(super) fn from_metadata(
        path: String,
        bytes: Option<u64>,
        role: FileRole,
        sha256: Option<String>,
    ) -> Result<Self> {
        validate_relative_path(&path)?;
        if let Some(hash) = &sha256 {
            validate_digest(hash)?;
        }
        Ok(Self {
            path,
            bytes,
            role,
            sha256,
        })
    }
    /// Repository-relative path, preserving its original case and bytes.
    pub fn path(&self) -> &str {
        &self.path
    }
    /// Observed metadata only; None is unknown, never zero. Remote GitHub sizes
    /// describe stored Git blobs (possibly LFS pointers), not resolved model
    /// bytes. HF sizes may come from its tree/LFS metadata. Cached sizes are
    /// verified retained content lengths. None of these implies model readiness.
    pub fn bytes(&self) -> Option<u64> {
        self.bytes
    }
    /// Existing filename-role hint; not parsed model content or readiness.
    pub fn role(&self) -> FileRole {
        self.role
    }
    /// Provider/caller expected raw SHA only. Cached FirstSeen is not promoted.
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }
}

#[derive(Clone, Debug)]
pub struct RepositoryCandidates {
    source: ManifestSource,
    coverage: CandidateCoverage,
    files: Vec<CandidateFile>,
}
impl RepositoryCandidates {
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }
    pub fn coverage(&self) -> CandidateCoverage {
        self.coverage
    }
    pub fn files(&self) -> &[CandidateFile] {
        &self.files
    }
    fn new(
        source: ManifestSource,
        coverage: CandidateCoverage,
        mut files: Vec<CandidateFile>,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        if files.len() > limits.cache.max_files {
            return Err(bound());
        }
        files.sort_unstable_by(|a, b| a.path.cmp(&b.path));
        let mut control =
            source.provider().len() + source.repository().len() + source.revision().len();
        if control > limits.cache.max_manifest_bytes {
            return Err(bound());
        }
        for (index, file) in files.iter().enumerate() {
            check_cancel(cancel)?;
            if index > 0 && files[index - 1].path == file.path {
                return Err(Error::Malformed("duplicate candidate path".into()));
            }
            control = control
                .checked_add(file.path.len() + file.sha256.as_ref().map_or(0, String::len) + 128)
                .filter(|n| *n <= limits.cache.max_manifest_bytes)
                .ok_or_else(bound)?;
            for (offset, _) in file.path.match_indices('/') {
                if files
                    .binary_search_by(|candidate| candidate.path.as_str().cmp(&file.path[..offset]))
                    .is_ok()
                {
                    return Err(Error::Malformed(
                        "candidate file/directory collision".into(),
                    ));
                }
            }
        }
        check_cancel(cancel)?;
        Ok(Self {
            source,
            coverage,
            files,
        })
    }
}
fn bound() -> Error {
    Error::Limit("candidate metadata bounds exceeded".into())
}

/// Counts logical metadata calls (pin included). Original HTTPS owns bounded
/// physical retries/deadlines/TLS; this is not physical-wire telemetry.
struct MetadataCalls<'a, T> {
    inner: &'a T,
    calls: AtomicUsize,
    maximum: usize,
}
impl<T: HttpTransport> HttpTransport for MetadataCalls<'_, T> {
    fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        check_cancel(cancel)?;
        self.calls
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(1).filter(|next| *next <= self.maximum)
            })
            .map_err(|_| bound())?;
        let result = self.inner.get(request, cancel);
        check_cancel(cancel)?;
        result.map_err(|error| match error {
            Error::Limit(_) => bound(),
            Error::Unsupported(_) => {
                Error::Unsupported("candidate metadata request unsupported".into())
            }
            _ => Error::State("candidate metadata request failed".into()),
        })
    }
}

/// Acquire only pin/tree metadata. Unknown/large listed sizes are shown, never
/// reserved or silently selected. Subsequent acquisition enforces payload caps.
pub fn discover_candidates(
    transport: &impl HttpTransport,
    request: &CandidateRequest,
    limits: &PullLimits,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<RepositoryCandidates> {
    limits.validate()?;
    check_cancel(cancel)?;
    let bytes = (limits.max_metadata_bytes as u64)
        .checked_mul(8)
        .and_then(|v| v.checked_add(16 * 1024 * 1024))
        .ok_or_else(bound)?;
    let _control = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes,
        }],
    )?;
    let transport = MetadataCalls {
        inner: transport,
        calls: AtomicUsize::new(0),
        maximum: limits.max_pages,
    };
    let (source, files) = match request.provider {
        Provider::HuggingFace => {
            super::huggingface::discover_candidates(&transport, request, limits, cancel)?
        }
        Provider::GitHub => {
            super::github::discover_candidates(&transport, request, limits, cancel)?
        }
    };
    // Validate the complete regular inventory BEFORE projecting a subfolder:
    // a regular `q` must not disappear and hide its collision with `q/a.gguf`.
    let mut result = RepositoryCandidates::new(
        source,
        CandidateCoverage::PinnedRepositoryTree,
        files,
        limits,
        cancel,
    )?;
    result.files.retain(|file| request.includes(file.path()));
    check_cancel(cancel)?;
    Ok(result)
}

/// Inspect only the explicitly retained verified snapshot. No remote lookup,
/// token/client/admission input, full-tree/freshness or native-protocol claim.
pub fn inspect_cached_candidates(
    lease: &SnapshotLease,
    request: &CandidateRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
) -> Result<RepositoryCandidates> {
    limits.validate()?;
    check_cancel(cancel)?;
    super::components::validate_cached_file_source(
        lease,
        request.provider,
        request.repository(),
        request.revision(),
    )?;
    Manifest::new(
        lease.manifest().source().clone(),
        lease.manifest().entries().to_vec(),
        &limits.cache,
    )?;
    super::executor::reject_cached_pointers(lease, cancel)?;
    let mut files = Vec::new();
    for entry in lease.manifest().entries() {
        check_cancel(cancel)?;
        if request.includes(entry.path()) {
            let role = match request.provider {
                Provider::HuggingFace => super::huggingface::file_role(entry.path()),
                Provider::GitHub => super::github::role(entry.path()),
            };
            let expected = (entry.digest_provenance() == DigestProvenance::TrustedSha256)
                .then(|| entry.digest().to_string());
            files.push(CandidateFile::from_metadata(
                entry.path().to_string(),
                Some(entry.bytes()),
                role,
                expected,
            )?);
        }
    }
    RepositoryCandidates::new(
        lease.manifest().source().clone(),
        CandidateCoverage::VerifiedCacheSubset,
        files,
        limits,
        cancel,
    )
}
