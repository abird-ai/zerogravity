//! Explicit, bounded acquisition into the existing cache. Acquired bytes do not
//! establish executable model support, SourceIdentity or device readiness.

mod executor;
pub mod github;
pub mod http;
pub mod huggingface;
pub mod resume;

pub use crate::cache::manifest::DigestProvenance;
use crate::cache::paths::{validate_digest, validate_relative_path};
use crate::cache::{CacheLimits, FileRole, ManifestSource, TransferAdmission};
use crate::{Error, Result};
pub use executor::{AcquiredFile, PullReceipt, execute};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Provider {
    HuggingFace,
    GitHub,
}
impl Provider {
    pub fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::GitHub => "github",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Selection {
    Files {
        paths: Vec<String>,
        subfolder: Option<String>,
    },
    ReleaseAsset {
        name: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PullRequest {
    pub provider: Provider,
    pub repository: String,
    pub revision: String,
    pub selection: Selection,
}
impl PullRequest {
    pub fn validate(&self, limits: &PullLimits) -> Result<()> {
        limits.validate()?;
        ManifestSource::new(self.provider.label(), &self.repository, &self.revision)?;
        match &self.selection {
            Selection::Files { paths, subfolder } => {
                if paths.is_empty() || paths.len() > limits.cache.max_files {
                    return Err(Error::Limit("pull selection file count".into()));
                }
                if let Some(folder) = subfolder {
                    validate_relative_path(folder)?;
                }
                let mut seen = std::collections::BTreeSet::new();
                for path in paths {
                    validate_relative_path(path)?;
                    if !seen.insert(path) {
                        return Err(Error::Malformed("duplicate pull path".into()));
                    }
                    if let Some(folder) = subfolder {
                        validate_relative_path(&format!("{folder}/{path}"))?;
                    }
                }
            }
            Selection::ReleaseAsset { name } => {
                if self.provider != Provider::GitHub || name.contains('/') {
                    return Err(Error::Unsupported("release asset selection".into()));
                }
                validate_relative_path(name)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PullLimits {
    pub cache: CacheLimits,
    pub max_metadata_bytes: usize,
    pub max_pages: usize,
}
impl Default for PullLimits {
    fn default() -> Self {
        Self {
            cache: CacheLimits::default(),
            max_metadata_bytes: 8 * 1024 * 1024,
            max_pages: 64,
        }
    }
}
impl PullLimits {
    pub fn validate(&self) -> Result<()> {
        self.cache.validate()?;
        if self.max_metadata_bytes == 0
            || self.max_metadata_bytes > 8 * 1024 * 1024
            || self.max_pages == 0
            || self.max_pages > 256
        {
            return Err(Error::Limit("pull metadata/page bounds".into()));
        }
        Ok(())
    }
}

/// URL stays in memory only. Neither Debug nor the immutable cache manifest
/// contains redirect URLs, query credentials or authorization headers.
#[derive(Clone)]
pub struct ResolvedFile {
    path: String,
    url: String,
    role: FileRole,
    bytes: u64,
    sha256: Option<String>,
    identity: String,
}
impl ResolvedFile {
    pub fn new(
        path: impl Into<String>,
        url: impl Into<String>,
        role: FileRole,
        bytes: u64,
        sha256: Option<String>,
        identity: impl Into<String>,
    ) -> Result<Self> {
        let file = Self {
            path: path.into(),
            url: url.into(),
            role,
            bytes,
            sha256,
            identity: identity.into(),
        };
        validate_relative_path(&file.path)?;
        if let Some(hash) = &file.sha256 {
            validate_digest(hash)?;
        }
        if file.url.len() > 16 * 1024
            || file.url.chars().any(char::is_control)
            || !(file.url.starts_with("https://")
                || (cfg!(feature = "diagnostic-acquisition") && file.url.starts_with("http://")))
        {
            return Err(Error::Malformed("invalid resolved download URL".into()));
        }
        if file.identity.is_empty()
            || file.identity.len() > 4096
            || file.identity.chars().any(char::is_control)
            || file.identity.contains("://")
            || file.identity.contains(['?', '#', '@', '\\'])
        {
            return Err(Error::Malformed(
                "download identity must be bounded secret-free metadata".into(),
            ));
        }
        Ok(file)
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn url(&self) -> &str {
        &self.url
    }
    pub fn role(&self) -> FileRole {
        self.role
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn digest_provenance(&self) -> DigestProvenance {
        if self.sha256.is_some() {
            DigestProvenance::TrustedSha256
        } else {
            DigestProvenance::FirstSeen
        }
    }
}
impl fmt::Debug for ResolvedFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolvedFile")
            .field("path", &self.path)
            .field("role", &self.role)
            .field("bytes", &self.bytes)
            .field("sha256", &self.sha256)
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedPullPlan {
    source: ManifestSource,
    files: Vec<ResolvedFile>,
    total_bytes: u64,
    limits: PullLimits,
}
impl ResolvedPullPlan {
    pub fn new(
        source: ManifestSource,
        mut files: Vec<ResolvedFile>,
        limits: &PullLimits,
    ) -> Result<Self> {
        limits.validate()?;
        if files.is_empty() || files.len() > limits.cache.max_files {
            return Err(Error::Limit("resolved pull file count".into()));
        }
        files.sort_unstable_by(|a, b| a.path.cmp(&b.path));
        let mut total_bytes = 0u64;
        let mut paths = std::collections::BTreeSet::new();
        let mut metadata_bytes = 0usize;
        for file in &files {
            if !paths.insert(file.path.as_str()) {
                return Err(Error::Malformed("duplicate resolved pull path".into()));
            }
            if file.bytes > limits.cache.max_file_bytes {
                return Err(Error::Limit("resolved pull file length".into()));
            }
            total_bytes = total_bytes
                .checked_add(file.bytes)
                .filter(|v| *v <= limits.cache.max_snapshot_bytes)
                .ok_or_else(|| Error::Limit("resolved pull total length".into()))?;
            metadata_bytes = metadata_bytes
                .checked_add(file.path.len() + file.url.len() + file.identity.len() + 128)
                .filter(|v| *v <= limits.cache.max_manifest_bytes)
                .ok_or_else(|| Error::Limit("resolved pull metadata bound".into()))?;
        }
        for file in &files {
            for (i, _) in file.path.match_indices('/') {
                if paths.contains(&file.path[..i]) {
                    return Err(Error::Malformed("pull file/directory collision".into()));
                }
            }
        }
        Ok(Self {
            source,
            files,
            total_bytes,
            limits: *limits,
        })
    }
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }
    pub fn files(&self) -> &[ResolvedFile] {
        &self.files
    }
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
    pub fn limits(&self) -> &PullLimits {
        &self.limits
    }
}

/// Cold resolution admission covers bounded provider metadata/control work,
/// not all TLS/OS memory or returned caller-owned plan allocations indefinitely.
/// The caller must take its offline branch before creating client/credentials.
pub fn resolve(
    transport: &impl http::HttpTransport,
    request: &PullRequest,
    limits: &PullLimits,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    request.validate(limits)?;
    if cancel.load(Ordering::Acquire) {
        return Err(Error::State("pull resolution canceled".into()));
    }
    let bytes = (limits.max_metadata_bytes as u64)
        .checked_mul(8)
        .and_then(|v| v.checked_add(16 * 1024 * 1024))
        .ok_or_else(|| Error::Limit("pull metadata admission overflow".into()))?;
    let _lease = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes,
        }],
    )?;
    match request.provider {
        Provider::HuggingFace => huggingface::resolve(transport, request, limits, cancel),
        Provider::GitHub => github::resolve(transport, request, limits, cancel),
    }
}
