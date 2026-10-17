//! Explicit, bounded acquisition into the existing cache. Acquired bytes do not
//! establish executable model support, SourceIdentity or device readiness.

pub mod azure;
pub mod components;
mod executor;
pub mod gcs;
pub mod github;
pub mod http;
pub mod huggingface;
pub mod object;
pub mod oci;
pub mod resume;
pub mod s3;

pub use crate::cache::ByteLength;
pub use crate::cache::manifest::DigestProvenance;
use crate::cache::paths::{validate_digest, validate_relative_path};
use crate::cache::{CacheLimits, FileRole, ManifestSource, TransferAdmission};
use crate::{Error, Result};
pub use components::{
    ComponentKind, ComponentPullRequest, ComponentReceipt, ComponentRequest, ComponentTransport,
    ResolvedComponent, ResolvedComponentPlan, VerifiedComponent, VerifiedComponentSet,
    execute_components, execute_components_with_progress, open_components, resolve_components,
    validate_cached_selection,
};
pub use executor::{AcquiredFile, PullProgress, PullReceipt, execute, execute_with_progress};
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
    /// Explicit authorization for otherwise length-unavailable Auxiliary files.
    pub unknown_length_cap: Option<u64>,
}
impl Default for PullLimits {
    fn default() -> Self {
        Self {
            cache: CacheLimits::default(),
            max_metadata_bytes: 8 * 1024 * 1024,
            max_pages: 64,
            unknown_length_cap: None,
        }
    }
}
impl PullLimits {
    pub fn validate(&self) -> Result<()> {
        self.cache.validate()?;
        if self
            .unknown_length_cap
            .is_some_and(|cap| cap == 0 || cap > self.cache.max_file_bytes)
        {
            return Err(Error::Limit(
                "unknown object cap exceeds file bounds".into(),
            ));
        }
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
    length: ByteLength,
    sha256: Option<String>,
    identity: String,
    // Transient HEAD validator; never changes digest provenance or persistence.
    validator: Option<String>,
    response_binding: Option<ResponseBinding>,
    request_target: Option<http::HttpTarget>,
}

/// Provider protocol expectations, separate from strong ETag resume and raw
/// content SHA256. Only the owning resolver may construct these bindings.
#[derive(Clone, Copy, Debug)]
pub(super) enum ResponseBinding {
    AzureVersion {
        version_sha256: [u8; 32],
    },
    AzureSnapshot,
    S3Version {
        version_sha256: [u8; 32],
    },
    GcsVersion {
        generation: u64,
        metageneration: u64,
    },
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
        Self::new_with_length(path, url, role, ByteLength::Exact(bytes), sha256, identity)
    }
    /// A caller constructing an AtMost Auxiliary file explicitly authorizes its
    /// cap. PullLimits::unknown_length_cap controls provider discovery instead;
    /// resolved plans always enforce the cache per-file and aggregate maxima.
    pub fn new_with_length(
        path: impl Into<String>,
        url: impl Into<String>,
        role: FileRole,
        length: ByteLength,
        sha256: Option<String>,
        identity: impl Into<String>,
    ) -> Result<Self> {
        if matches!(length, ByteLength::AtMost(0)) {
            return Err(Error::Limit("unknown object cap is zero".into()));
        }
        if matches!(length, ByteLength::AtMost(_)) && role != FileRole::Auxiliary {
            return Err(Error::Unsupported(
                "capped acquisition requires an auxiliary file".into(),
            ));
        }
        let file = Self {
            path: path.into(),
            url: url.into(),
            role,
            length,
            sha256,
            identity: identity.into(),
            validator: None,
            response_binding: None,
            request_target: None,
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
    pub fn bytes(&self) -> Option<u64> {
        self.length.exact()
    }
    pub fn length(&self) -> ByteLength {
        self.length
    }
    pub fn maximum_bytes(&self) -> u64 {
        self.length.maximum()
    }
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub(super) fn with_validator(mut self, validator: Option<String>) -> Result<Self> {
        if let Some(value) = &validator {
            resume::strong_etag(value)?;
        }
        self.validator = validator;
        Ok(self)
    }
    pub(super) fn validator(&self) -> Option<&str> {
        self.validator.as_deref()
    }
    pub(super) fn with_request_target(mut self, target: http::HttpTarget) -> Self {
        self.request_target = Some(target);
        self
    }
    pub(super) fn request_target(&self) -> Option<&http::HttpTarget> {
        self.request_target.as_ref()
    }
    pub(super) fn with_response_binding(mut self, binding: ResponseBinding) -> Self {
        self.response_binding = Some(binding);
        self
    }
    pub(super) fn response_binding(&self) -> Option<ResponseBinding> {
        self.response_binding
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
            .field("length", &self.length)
            .field("sha256", &self.sha256)
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedPullPlan {
    source: ManifestSource,
    files: Vec<ResolvedFile>,
    total_bytes: Option<u64>,
    maximum_bytes: u64,
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
        let mut maximum_bytes = 0u64;
        let mut all_exact = true;
        let mut paths = std::collections::BTreeSet::new();
        let mut metadata_bytes = 0usize;
        for file in &files {
            if !paths.insert(file.path.as_str()) {
                return Err(Error::Malformed("duplicate resolved pull path".into()));
            }
            if file.maximum_bytes() > limits.cache.max_file_bytes {
                return Err(Error::Limit("resolved pull file length".into()));
            }
            maximum_bytes = maximum_bytes
                .checked_add(file.maximum_bytes())
                .filter(|v| *v <= limits.cache.max_snapshot_bytes)
                .ok_or_else(|| Error::Limit("resolved pull total length".into()))?;
            all_exact &= file.bytes().is_some();
            metadata_bytes = metadata_bytes
                .checked_add(
                    file.path.len()
                        + file.url.len()
                        + file.identity.len()
                        + file.validator.as_ref().map_or(0, String::len)
                        + file.response_binding.map_or(0, |_| 32)
                        + file
                            .request_target
                            .as_ref()
                            .map_or(0, http::HttpTarget::metadata_bytes)
                        + 128,
                )
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
            total_bytes: all_exact.then_some(maximum_bytes),
            maximum_bytes,
            limits: *limits,
        })
    }
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }
    pub fn files(&self) -> &[ResolvedFile] {
        &self.files
    }
    pub fn total_bytes(&self) -> Option<u64> {
        self.total_bytes
    }
    pub fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
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

/// An explicit component set can supply only these otherwise required sidecars
/// from a separately pinned source. Ordinary resolve always uses strict defaults.
#[derive(Clone, Copy, Default)]
pub(super) struct ExternalSidecars {
    config: bool,
    tokenizer: bool,
}
impl ExternalSidecars {
    fn allows_missing(self, name: &str) -> bool {
        match name {
            "config.json" => self.config,
            "tokenizer.json" => self.tokenizer,
            _ => false,
        }
    }
}
