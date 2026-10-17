//! Native ModelScope model repositories, explicit commits only. Completeness
//! relies on the official shallow-list protocol: any directory at its 3000-row
//! silent-truncation boundary refuses. This is not local Git/Merkle verification.
use super::{PullLimits, ResolvedFile, ResolvedPullPlan, http, huggingface, selection};
use crate::cache::paths::{validate_digest, validate_relative_path};
use crate::cache::{
    CacheStore, DigestProvenance, Manifest, ManifestSource, SnapshotLease, TransferAdmission,
};
use crate::{Error, Result};
use http::{HttpAccept, HttpRequest, HttpTarget, HttpTransport, check_cancel};
use serde::{
    Deserialize, Deserializer,
    de::{SeqAccess, Visitor},
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::os::unix::fs::FileExt;
use std::sync::atomic::AtomicBool;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
const DIRECTORY_CAP: usize = 3000;
const NATIVE: &str = "https://modelscope.cn";

#[derive(Clone, Debug)]
pub struct ModelScopeRequest {
    origin: String,
    repository: String,
    commit: String,
    paths: Vec<String>,
    subfolder: Option<String>,
    source: ManifestSource,
}
impl ModelScopeRequest {
    pub fn new(
        origin: &str,
        repository: &str,
        commit: &str,
        paths: Vec<String>,
        subfolder: Option<String>,
    ) -> Result<Self> {
        if origin.len() > 4096 || repository.len() > 201 {
            return Err(Error::Malformed(
                "invalid ModelScope origin, repository or full commit".into(),
            ));
        }
        if commit.len() != 40
            || !commit
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::Unsupported(
                "ModelScope requires a canonical full lowercase commit; aliases are unsupported"
                    .into(),
            ));
        }
        let mut parts = repository.split('/');
        for _ in 0..2 {
            let part = parts.next().ok_or_else(malformed)?;
            if part.is_empty()
                || part.len() > 100
                || part.starts_with('.')
                || !part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            {
                return Err(malformed());
            }
        }
        if parts.next().is_some() {
            return Err(malformed());
        }
        let parsed = http::parse_provider_origin(origin)?;
        let normalized = parsed.origin().ascii_serialization();
        if origin != normalized
            || parsed.path() != "/"
            || parsed.query().is_some()
            || !(origin == NATIVE
                || (cfg!(any(test, feature = "diagnostic-acquisition"))
                    && parsed.scheme() == "http"))
        {
            return Err(Error::Unsupported("unsupported ModelScope origin".into()));
        }
        if paths.is_empty() || paths.len() > 4096 {
            return Err(Error::Limit("ModelScope selector count".into()));
        }
        let mut path_bytes = 0usize;
        for path in &paths {
            validate_relative_path(path)?;
            path_bytes = path_bytes
                .checked_add(path.len())
                .filter(|n| *n <= 8 * 1024 * 1024)
                .ok_or_else(|| Error::Limit("ModelScope selector bytes".into()))?;
            if let Some(folder) = &subfolder {
                validate_relative_path(folder)?;
                if folder.len().saturating_add(path.len()).saturating_add(1) > 4096 {
                    return Err(Error::Limit("ModelScope combined path".into()));
                }
                validate_relative_path(&format!("{folder}/{path}"))?;
            }
        }
        let source = ManifestSource::new(
            "modelscope",
            format!("origin-{}/{repository}", hash(origin.as_bytes())),
            commit,
        )?;
        Ok(Self {
            origin: normalized,
            repository: repository.into(),
            commit: commit.into(),
            paths,
            subfolder,
            source,
        })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn repository(&self) -> &str {
        &self.repository
    }
    pub fn commit(&self) -> &str {
        &self.commit
    }
    pub fn paths(&self) -> &[String] {
        &self.paths
    }
    pub fn subfolder(&self) -> Option<&str> {
        self.subfolder.as_deref()
    }
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }

    pub fn resolve(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<ResolvedPullPlan> {
        limits.validate()?;
        check_cancel(cancel)?;
        // Bounded parsed metadata/maps and one response/canonical buffer. This
        // operation-local envelope is not a reservation of global host memory.
        let _control = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: (limits.max_metadata_bytes as u64) * 8 + 16 * 1024 * 1024,
            }],
        )?;
        let mut budget = Budget::new(limits);
        let mut pending = VecDeque::from([String::new()]);
        let mut all_paths = BTreeSet::new();
        let mut files = BTreeMap::new();
        let mut path_bytes = 0usize;
        while let Some(parent) = pending.pop_front() {
            check_cancel(cancel)?;
            let bytes = budget.read(transport, &self.tree_url(&parent), None, None, cancel)?;
            let rows = parse_rows(&bytes)?;
            for row in rows {
                validate_relative_path(&row.path)?;
                let (directory, leaf) = row.path.rsplit_once('/').unwrap_or(("", &row.path));
                if directory != parent
                    || row.name.as_deref().is_some_and(|name| name != leaf)
                    || !all_paths.insert(row.path.clone())
                {
                    return Err(malformed());
                }
                path_bytes = path_bytes
                    .checked_add(row.path.len())
                    .filter(|n| *n <= 8 * 1024 * 1024)
                    .ok_or_else(|| Error::Limit("ModelScope inventory path bytes".into()))?;
                if all_paths.len() > limits.cache.max_files.saturating_add(limits.max_pages) {
                    return Err(Error::Limit("ModelScope inventory node bound".into()));
                }
                match row.kind.as_str() {
                    "tree" => pending.push_back(row.path),
                    "blob" => {
                        let size = match row.size {
                            ListedSize::Missing => None,
                            ListedSize::Exact(bytes) => Some(bytes),
                            ListedSize::Null => return Err(malformed()),
                        };
                        let digest = row.sha256.filter(|s| !s.is_empty());
                        if let Some(digest) = &digest {
                            validate_digest(digest)?;
                        }
                        if files.len() >= limits.cache.max_files {
                            return Err(Error::Limit("ModelScope inventory file bound".into()));
                        }
                        files.insert(
                            row.path,
                            File {
                                bytes: size,
                                digest,
                                validator: None,
                            },
                        );
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "unsupported ModelScope tree entry kind".into(),
                        ));
                    }
                }
            }
        }
        let available = files.keys().map(String::as_str).collect::<Vec<_>>();
        let mut selected = huggingface::select_paths(&self.paths, self.subfolder(), &available)?;
        let entries = files.keys().map(|path| (path.clone(), true)).collect();
        selection::close_model_selection(
            &entries,
            &mut selected,
            limits,
            super::ExternalSidecars::default(),
            &mut |path| {
                let file = files.get_mut(path).ok_or_else(malformed)?;
                let url = self.raw_url(path);
                let size = file.resolve_length(transport, &url, &mut budget, limits, cancel)?;
                let bytes = budget.read(
                    transport,
                    &url,
                    Some(size),
                    file.validator.as_deref(),
                    cancel,
                )?;
                if file
                    .digest
                    .as_ref()
                    .is_some_and(|digest| hash(&bytes) != *digest)
                {
                    return Err(Error::Malformed(
                        "ModelScope index raw digest mismatch".into(),
                    ));
                }
                super::executor::reject_pointer(&bytes)?;
                Ok(bytes)
            },
        )?;
        let mut resolved = Vec::new();
        for path in selected {
            check_cancel(cancel)?;
            let file = files.get_mut(&path).ok_or_else(malformed)?;
            let url = self.raw_url(&path);
            let bytes = file.resolve_length(transport, &url, &mut budget, limits, cancel)?;
            let identity = hash(
                &serde_json::to_vec(&(
                    "modelscope.file.v1",
                    &self.origin,
                    &self.repository,
                    &self.commit,
                    &path,
                    bytes,
                    &file.digest,
                ))
                .map_err(|_| malformed())?,
            );
            let target = HttpTarget::direct(&url)?;
            resolved.push(
                ResolvedFile::new(
                    &path,
                    &url,
                    super::github::role(&path),
                    bytes,
                    file.digest.clone(),
                    format!("modelscope:{identity}"),
                )?
                .with_validator(file.validator.clone())?
                .with_request_target(target),
            );
        }
        check_cancel(cancel)?;
        ResolvedPullPlan::new(self.source.clone(), resolved, limits)
    }

    /// Exact cached data only, before any credential/client construction. This
    /// does not establish remote freshness or recover omitted repository rows.
    pub fn open_cached(
        &self,
        store: &CacheStore,
        snapshot: &str,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<SnapshotLease> {
        limits.validate()?;
        check_cancel(cancel)?;
        validate_digest(snapshot)?;
        let lease = store.open_snapshot_cancellable(snapshot, cancel)?;
        let manifest = lease.manifest();
        if manifest.source() != &self.source {
            return Err(Error::Malformed("cached ModelScope source mismatch".into()));
        }
        Manifest::from_json(&manifest.canonical_bytes()?, &limits.cache)?;
        let available = manifest
            .entries()
            .iter()
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        let mut selected = huggingface::select_paths(&self.paths, self.subfolder(), &available)?;
        let entries = available
            .iter()
            .map(|path| ((*path).to_owned(), true))
            .collect();
        let mut budget = Budget::new(limits);
        selection::close_model_selection(
            &entries,
            &mut selected,
            limits,
            super::ExternalSidecars::default(),
            &mut |path| {
                check_cancel(cancel)?;
                budget.take_request()?;
                let bytes = lease.read_verified(path, budget.remaining as u64)?;
                budget.remaining = budget
                    .remaining
                    .checked_sub(bytes.len())
                    .ok_or_else(|| Error::Limit("ModelScope offline index bytes".into()))?;
                super::executor::reject_pointer(&bytes)?;
                Ok(bytes)
            },
        )?;
        if selected.len() != manifest.entries().len()
            || manifest.entries().iter().any(|entry| {
                entry.role() != super::github::role(entry.path())
                    || !selected.contains(entry.path())
                    || !matches!(
                        entry.digest_provenance(),
                        DigestProvenance::FirstSeen | DigestProvenance::TrustedSha256
                    )
            })
        {
            return Err(Error::Malformed(
                "cached ModelScope selection, role or provenance mismatch".into(),
            ));
        }
        // Generic CacheStore imports are data-only and can carry pointer bytes
        // under the same declared source. Reapply native body eligibility on
        // the original managed descriptors; never reopen paths or hash again.
        for entry in manifest.entries() {
            check_cancel(cancel)?;
            let count = entry.bytes().min(128) as usize;
            let mut prefix = [0u8; 128];
            let mut done = 0;
            let file = lease.file(entry.path())?;
            while done < count {
                let read = file.read_at(&mut prefix[done..count], done as u64)?;
                check_cancel(cancel)?;
                if read == 0 {
                    return Err(Error::Malformed(
                        "cached ModelScope file changed during eligibility check".into(),
                    ));
                }
                done += read;
            }
            super::executor::reject_pointer(&prefix[..done])?;
        }
        check_cancel(cancel)?;
        Ok(lease)
    }
    fn tree_url(&self, parent: &str) -> String {
        let mut url = self.base_url("/files");
        url.query_pairs_mut()
            .append_pair("Revision", &self.commit)
            .append_pair("Recursive", "False");
        if !parent.is_empty() {
            url.query_pairs_mut().append_pair("Root", parent);
        }
        url.into()
    }
    fn raw_url(&self, path: &str) -> String {
        let mut url = self.base_url("");
        url.query_pairs_mut()
            .append_pair("Revision", &self.commit)
            .append_pair("FilePath", path);
        url.into()
    }
    fn base_url(&self, suffix: &str) -> url::Url {
        url::Url::parse(&format!(
            "{}/api/v1/models/{}/repo{suffix}",
            self.origin, self.repository
        ))
        .expect("validated provider origin and repository")
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn malformed() -> Error {
    Error::Malformed("invalid ModelScope metadata or selection".into())
}
struct File {
    bytes: Option<u64>,
    digest: Option<String>,
    validator: Option<String>,
}
impl File {
    fn resolve_length(
        &mut self,
        transport: &(impl HttpTransport + ?Sized),
        url: &str,
        budget: &mut Budget,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<u64> {
        if let Some(bytes) = self.bytes {
            if bytes > limits.cache.max_file_bytes || bytes > limits.cache.max_snapshot_bytes {
                return Err(Error::Limit("ModelScope selected file length".into()));
            }
            return Ok(bytes);
        }
        // The same retained full-commit selector and original Direct capability
        // own this HEAD. Never fall back to GET, a redirect, a cap or an alias.
        let response = budget.head(transport, url, limits.cache.max_file_bytes, cancel)?;
        if response.status != 200 {
            return Err(Error::Unsupported(
                "ModelScope exact HEAD requires HTTP200".into(),
            ));
        }
        if response.header("content-range").is_some()
            || response.header("transfer-encoding").is_some()
            || response
                .header("content-encoding")
                .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
        {
            return Err(Error::Unsupported(
                "ModelScope HEAD requires complete identity metadata".into(),
            ));
        }
        let bytes =
            http::parse_u64(response.header("content-length").ok_or_else(|| {
                Error::Unsupported("ModelScope exact HEAD length is absent".into())
            })?)?;
        if bytes > limits.cache.max_file_bytes || bytes > limits.cache.max_snapshot_bytes {
            return Err(Error::Limit("ModelScope observed file length".into()));
        }
        let validator = response
            .header("etag")
            .filter(|v| super::resume::strong_etag(v).is_ok())
            .map(str::to_owned);
        check_cancel(cancel)?;
        self.bytes = Some(bytes);
        self.validator = validator;
        Ok(bytes)
    }
}
struct Budget {
    remaining: usize,
    calls: usize,
}
impl Budget {
    fn new(limits: &PullLimits) -> Self {
        Self {
            remaining: limits.max_metadata_bytes,
            calls: limits.max_pages,
        }
    }
    fn take_request(&mut self) -> Result<()> {
        if self.remaining == 0 || self.calls == 0 {
            return Err(Error::Limit(
                "ModelScope shared metadata request/byte budget".into(),
            ));
        }
        self.calls -= 1;
        Ok(())
    }
    fn head(
        &mut self,
        transport: &(impl HttpTransport + ?Sized),
        url: &str,
        maximum: u64,
        cancel: &AtomicBool,
    ) -> Result<http::HttpResponse> {
        check_cancel(cancel)?;
        self.take_request()?;
        let target = HttpTarget::direct(url)?;
        let response =
            transport.head_target(&HttpRequest::new(target.url(), maximum), &target, cancel);
        check_cancel(cancel)?;
        let response = response.map_err(http::redact_head_error)?;
        let bytes = response
            .headers
            .iter()
            .try_fold(0usize, |sum, (name, value)| {
                sum.checked_add(name.len())
                    .and_then(|n| n.checked_add(value.len()))
                    .and_then(|n| n.checked_add(32))
            })
            .ok_or_else(|| Error::Limit("ModelScope HEAD metadata overflow".into()))?;
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or_else(|| Error::Limit("ModelScope shared HEAD header budget".into()))?;
        Ok(response)
    }
    fn read(
        &mut self,
        transport: &(impl HttpTransport + ?Sized),
        url: &str,
        exact: Option<u64>,
        validator: Option<&str>,
        cancel: &AtomicBool,
    ) -> Result<Vec<u8>> {
        check_cancel(cancel)?;
        self.take_request()?;
        if exact.is_some_and(|bytes| bytes > self.remaining as u64) {
            return Err(Error::Limit("ModelScope index metadata byte budget".into()));
        }
        let maximum = exact.unwrap_or(self.remaining as u64);
        let target = HttpTarget::direct(url)?;
        let mut request = HttpRequest::new(target.url(), maximum);
        request.accept = HttpAccept::Json;
        let response = transport.get_target(&request, &target, cancel);
        check_cancel(cancel)?;
        let response = response?;
        if response.status != 200 {
            return Err(Error::Unsupported(
                "ModelScope pinned metadata request failed".into(),
            ));
        }
        http::validate_validator(validator, &response)?;
        let bytes = response.read_bounded(maximum as usize, cancel)?;
        if exact.is_some_and(|size| size != bytes.len() as u64) {
            return Err(malformed());
        }
        self.remaining -= bytes.len();
        Ok(bytes)
    }
}
#[derive(Deserialize)]
struct Envelope<'a> {
    #[serde(rename = "Code")]
    code: u64,
    #[serde(rename = "Success")]
    success: bool,
    #[serde(rename = "Data", borrow)]
    data: &'a serde_json::value::RawValue,
}
#[derive(Deserialize)]
struct Row {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Name")]
    name: Option<String>,
    #[serde(rename = "Type")]
    kind: String,
    #[serde(rename = "Size", default, deserialize_with = "listed_size")]
    size: ListedSize,
    #[serde(rename = "Sha256")]
    sha256: Option<String>,
}
#[derive(Default)]
enum ListedSize {
    #[default]
    Missing,
    Exact(u64),
    Null,
}
fn listed_size<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<ListedSize, D::Error> {
    Option::<u64>::deserialize(deserializer)
        .map(|value| value.map_or(ListedSize::Null, ListedSize::Exact))
}
struct Rows(Vec<Row>);
impl<'de> Deserialize<'de> for Rows {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct Bounded;
        impl<'de> Visitor<'de> for Bounded {
            type Value = Rows;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a complete bounded directory")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Rows, A::Error> {
                let mut rows = Vec::new();
                while let Some(row) = seq.next_element()? {
                    if rows.len() >= DIRECTORY_CAP - 1 {
                        return Err(serde::de::Error::custom(
                            "directory reaches silent truncation boundary",
                        ));
                    }
                    rows.push(row);
                }
                Ok(Rows(rows))
            }
        }
        deserializer.deserialize_seq(Bounded)
    }
}
fn parse_rows(bytes: &[u8]) -> Result<Vec<Row>> {
    let envelope: Envelope<'_> = serde_json::from_slice(bytes).map_err(|_| malformed())?;
    if envelope.code != 200 || !envelope.success {
        return Err(Error::Unsupported(
            "ModelScope metadata application failure".into(),
        ));
    }
    let data = envelope.data.get().trim_start();
    let rows = if data.starts_with('[') {
        serde_json::from_str::<Rows>(data).map_err(|_| malformed())?
    } else {
        #[derive(Deserialize)]
        struct Directory {
            #[serde(rename = "Files")]
            files: Rows,
        }
        serde_json::from_str::<Directory>(data)
            .map_err(|_| malformed())?
            .files
    };
    Ok(rows.0)
}
