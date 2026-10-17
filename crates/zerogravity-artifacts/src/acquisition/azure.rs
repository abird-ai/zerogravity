//! Explicit native Azure BlockBlob acquisition. Metadata resolution returns
//! original independent plans, never an atomic container or model closure.
//! Credentials/SAS remain in the original HTTP owner, not these source pins.
use super::http::{AzureRequestKind, HttpRequest, HttpTarget, HttpTransport, check_cancel};
use super::{DigestProvenance, PullLimits, ResolvedFile, ResolvedPullPlan, ResponseBinding};
use crate::cache::paths::{validate_digest, validate_relative_path};
use crate::cache::{
    CacheStore, FileRole, ManifestEntry, ManifestSource, SnapshotLease, TransferAdmission,
};
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::os::unix::fs::FileExt;
use std::sync::atomic::AtomicBool;
use url::Url;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};
#[path = "azure_xml.rs"]
mod xml;

/// These opaque IDs inhabit different namespaces; neither is a timestamp to
/// normalize or compare with another object. Latest lookup is not implicit.
#[derive(Clone, Eq, PartialEq)]
pub struct AzureRevision {
    kind: RevisionKind,
    value: String,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
enum RevisionKind {
    Version,
    Snapshot,
}
impl AzureRevision {
    pub fn version(value: &str) -> Result<Self> {
        Self::new(RevisionKind::Version, value)
    }
    pub fn snapshot(value: &str) -> Result<Self> {
        Self::new(RevisionKind::Snapshot, value)
    }
    fn new(kind: RevisionKind, value: &str) -> Result<Self> {
        if value.is_empty()
            || value.len() > 128
            || !value.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err(invalid());
        }
        Ok(Self {
            kind,
            value: value.to_owned(),
        })
    }
    pub fn value(&self) -> &str {
        &self.value
    }
    pub fn is_snapshot(&self) -> bool {
        self.kind == RevisionKind::Snapshot
    }
    fn query_name(&self) -> &'static str {
        if self.is_snapshot() {
            "snapshot"
        } else {
            "versionid"
        }
    }
    fn target_kind(&self) -> AzureRequestKind {
        if self.is_snapshot() {
            AzureRequestKind::Snapshot
        } else {
            AzureRequestKind::Version
        }
    }
    fn binding(&self) -> ResponseBinding {
        if self.is_snapshot() {
            ResponseBinding::AzureSnapshot
        } else {
            ResponseBinding::AzureVersion {
                version_sha256: Sha256::digest(self.value.as_bytes()).into(),
            }
        }
    }
}
impl fmt::Debug for AzureRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureRevision")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

/// Pure bounded selectors; an optional SHA is independently caller-authorized.
/// Azure MD5/CRC64/ETag and encryption-key digests never become raw SHA authority.
#[derive(Clone)]
pub struct AzureObjectRequest {
    origin: String,
    container: String,
    blob: String,
    path: String,
    role: FileRole,
    revision: AzureRevision,
    sha256: Option<String>,
}
impl AzureObjectRequest {
    pub fn new(
        origin: &str,
        container: &str,
        blob: &str,
        path: &str,
        role: FileRole,
        revision: AzureRevision,
        sha256: Option<&str>,
    ) -> Result<Self> {
        let origin = origin_checked(origin)?;
        container_checked(container)?;
        blob_checked(blob)?;
        validate_relative_path(path)?;
        if let Some(hash) = sha256 {
            validate_digest(hash)?;
        }
        Ok(Self {
            origin,
            container: container.to_owned(),
            blob: blob.to_owned(),
            path: path.to_owned(),
            role,
            revision,
            sha256: sha256.map(str::to_owned),
        })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn container(&self) -> &str {
        &self.container
    }
    pub fn blob(&self) -> &str {
        &self.blob
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn role(&self) -> FileRole {
        self.role
    }
    pub fn revision(&self) -> &AzureRevision {
        &self.revision
    }
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }
    fn target(&self) -> Result<HttpTarget> {
        HttpTarget::azure(
            &self.origin,
            &format!(
                "/{}/{}?{}={}",
                self.container,
                encoded_blob(&self.blob),
                self.revision.query_name(),
                component(self.revision.value())
            ),
            self.revision.target_kind(),
        )
    }
    pub fn resolve(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<ResolvedPullPlan> {
        self.resolve_with_pin(transport, limits, admission, cancel)
            .map(|(_, plan)| plan)
    }
    /// Exact HEAD metadata only; no content is consumed and no mutable-latest
    /// request is issued. Version and snapshot queries survive every later retry.
    pub fn resolve_with_pin(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<(AzureObjectPin, ResolvedPullPlan)> {
        let _lease = admit(limits, admission, cancel)?;
        let target = self.target()?;
        let response = transport.head_target(
            &HttpRequest::new(target.url(), limits.cache.max_file_bytes),
            &target,
            cancel,
        );
        check_cancel(cancel)?;
        let response =
            response.map_err(|_| Error::State("Azure properties request failed".into()))?;
        if response.status != 200 {
            return Err(Error::State(format!(
                "Azure properties status {}",
                response.status
            )));
        }
        super::http::validate_response_binding(Some(self.revision.binding()), &response)?;
        if response.header("content-range").is_some()
            || response.header("transfer-encoding").is_some()
        {
            return Err(invalid());
        }
        if response
            .header("content-encoding")
            .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
        {
            return Err(Error::Unsupported(
                "Azure content requires identity encoding".into(),
            ));
        }
        let bytes = response
            .header("content-length")
            .ok_or_else(invalid)
            .and_then(super::http::parse_u64)?;
        let validator = response.header("etag").ok_or_else(invalid)?;
        super::resume::strong_etag(validator)?;
        let pin = AzureObjectPin::new(self, bytes)?;
        let plan = pin.plan(limits, Some(validator.to_owned()))?;
        Ok((pin, plan))
    }
}
impl fmt::Debug for AzureObjectRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureObjectRequest")
            .field("origin", &self.origin)
            .field("path", &self.path)
            .field("role", &self.role)
            .field("revision", &self.revision)
            .field("sha256", &self.sha256)
            .finish_non_exhaustive()
    }
}

/// Exact caller-authorized raw bytes at a mutable native BlockBlob locator.
/// No remote version/metadata lookup or ETag digest authority is inferred.
#[derive(Clone)]
pub struct AzureContentSpec {
    origin: String,
    container: String,
    blob: String,
    path: String,
    role: FileRole,
    bytes: u64,
    sha256: String,
    source: ManifestSource,
    identity: String,
}
impl AzureContentSpec {
    pub fn new(
        origin: &str,
        container: &str,
        blob: &str,
        path: &str,
        role: FileRole,
        bytes: u64,
        sha256: &str,
    ) -> Result<Self> {
        let origin = origin_checked(origin)?;
        container_checked(container)?;
        blob_checked(blob)?;
        validate_relative_path(path)?;
        validate_digest(sha256)?;
        let object = hashed(&(
            "zerogravity.azure-content.object.v1",
            &origin,
            container,
            blob,
        ))?;
        let contract = hashed(&(
            "zerogravity.azure-content.contract.v1",
            &origin,
            container,
            blob,
            path,
            role,
            bytes,
            sha256,
        ))?;
        let source = ManifestSource::new(
            "azure-content",
            format!("object-{object}"),
            format!("content-contract-{contract}"),
        )?;
        Ok(Self {
            origin,
            container: container.to_owned(),
            blob: blob.to_owned(),
            path: path.to_owned(),
            role,
            bytes,
            sha256: sha256.to_owned(),
            source,
            identity: format!("azure-content-{contract}"),
        })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn container(&self) -> &str {
        &self.container
    }
    pub fn blob(&self) -> &str {
        &self.blob
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn role(&self) -> FileRole {
        self.role
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }
    fn validate_limits(&self, limits: &PullLimits) -> Result<()> {
        limits.validate()?;
        if self.bytes > limits.cache.max_file_bytes || self.bytes > limits.cache.max_snapshot_bytes
        {
            return Err(Error::Limit(
                "Azure content contract exceeds caller byte bounds".into(),
            ));
        }
        Ok(())
    }
    fn plan(&self, limits: &PullLimits) -> Result<ResolvedPullPlan> {
        self.validate_limits(limits)?;
        let target = HttpTarget::azure(
            &self.origin,
            &format!("/{}/{}", self.container, encoded_blob(&self.blob)),
            AzureRequestKind::Content,
        )?;
        let file = ResolvedFile::new(
            &self.path,
            target.url(),
            self.role,
            self.bytes,
            Some(self.sha256.clone()),
            &self.identity,
        )?
        .with_request_target(target)
        .with_response_binding(ResponseBinding::AzureContent);
        ResolvedPullPlan::new(self.source.clone(), vec![file], limits)
    }
    /// Pure local plan preparation under a bounded operation-local control
    /// lease; actual payload admission/verification stays in the original owner.
    pub fn prepare(
        &self,
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<ResolvedPullPlan> {
        self.validate_limits(limits)?;
        check_cancel(cancel)?;
        let _control = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: super::object::PREPARE_CONTROL_BYTES,
            }],
        )?;
        let plan = self.plan(limits)?;
        check_cancel(cancel)?;
        Ok(plan)
    }
    pub fn open_cached(
        &self,
        store: &CacheStore,
        snapshot: &str,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<SnapshotLease> {
        validate_digest(snapshot)?;
        self.plan(limits)?;
        let expected = ManifestEntry::new(&self.path, &self.sha256, self.bytes, self.role)?
            .with_digest_provenance(DigestProvenance::TrustedSha256);
        super::object::open_cached_trusted(store, snapshot, limits, &self.source, &expected, cancel)
    }
}
impl fmt::Debug for AzureContentSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureContentSpec")
            .field("origin", &self.origin)
            .field("path", &self.path)
            .field("role", &self.role)
            .field("bytes", &self.bytes)
            .field("sha256", &self.sha256)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// Explicit current-version auxiliary prefix. It never creates snapshots,
/// discovers missing versions, reads historical versions or implies model closure.
#[derive(Clone)]
pub struct AzurePrefixRequest {
    origin: String,
    container: String,
    prefix: String,
}
impl AzurePrefixRequest {
    pub fn new(origin: &str, container: &str, prefix: &str) -> Result<Self> {
        let origin = origin_checked(origin)?;
        container_checked(container)?;
        let name = prefix.strip_suffix('/').ok_or_else(invalid)?;
        blob_checked(name)?;
        Ok(Self {
            origin,
            container: container.to_owned(),
            prefix: prefix.to_owned(),
        })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn container(&self) -> &str {
        &self.container
    }
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
    pub fn resolve(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<Vec<ResolvedPullPlan>> {
        self.resolve_with_pins(transport, limits, admission, cancel)
            .map(|objects| objects.into_iter().map(|(_, plan)| plan).collect())
    }
    pub fn resolve_with_pins(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<Vec<(AzureObjectPin, ResolvedPullPlan)>> {
        let _lease = admit(limits, admission, cancel)?;
        let mut remaining = limits.max_metadata_bytes;
        let mut requests = 0;
        let mut marker = None::<String>;
        let mut markers = BTreeSet::new();
        let mut paths = BTreeSet::new();
        let mut objects = Vec::new();
        let mut total = 0u64;
        let mut control = 0usize;
        loop {
            check_cancel(cancel)?;
            if requests == limits.max_pages || remaining == 0 {
                return Err(Error::Limit("Azure listing request budget".into()));
            }
            requests += 1;
            let mut path_query = format!(
                "/{}?restype=container&comp=list&include=copy&prefix={}&maxresults={}",
                self.container,
                component(&self.prefix),
                limits.cache.max_files.min(5000)
            );
            if let Some(marker) = &marker {
                path_query.push_str("&marker=");
                path_query.push_str(&component(marker));
            }
            let target = HttpTarget::azure(&self.origin, &path_query, AzureRequestKind::List)?;
            let response = transport.get_target(
                &HttpRequest::new(target.url(), remaining as u64),
                &target,
                cancel,
            );
            check_cancel(cancel)?;
            let response =
                response.map_err(|_| Error::State("Azure listing request failed".into()))?;
            if response.status != 200 || response.header("content-range").is_some() {
                return Err(Error::State("Azure listing response invalid".into()));
            }
            if response
                .header("content-encoding")
                .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
            {
                return Err(Error::Unsupported("Azure XML encoding".into()));
            }
            let declared = response
                .header("content-length")
                .map(super::http::parse_u64)
                .transpose()?;
            let bytes = response.read_bounded(remaining, cancel)?;
            if declared.is_some_and(|count| count != bytes.len() as u64) {
                return Err(invalid());
            }
            remaining -= bytes.len();
            let page = xml::parse(
                &bytes,
                limits.cache.max_files.saturating_sub(objects.len()),
                cancel,
            )?;
            if page.origin != format!("{}/", self.origin)
                || page.container != self.container
                || page.prefix != self.prefix
                || page.marker != marker.clone().unwrap_or_default()
            {
                return Err(Error::Malformed(
                    "Azure listing source or continuation mismatch".into(),
                ));
            }
            for row in page.rows {
                check_cancel(cancel)?;
                let logical = row
                    .name
                    .strip_prefix(self.prefix.as_str())
                    .ok_or_else(invalid)?;
                validate_relative_path(logical)?;
                if !paths.insert(logical.to_owned()) {
                    return Err(Error::Malformed("duplicate Azure selected blob".into()));
                }
                let request = AzureObjectRequest::new(
                    &self.origin,
                    &self.container,
                    &row.name,
                    logical,
                    FileRole::Auxiliary,
                    AzureRevision::version(&row.version)?,
                    None,
                )?;
                super::resume::strong_etag(&row.etag)?;
                let pin = AzureObjectPin::new(&request, row.bytes)?;
                total = total
                    .checked_add(row.bytes)
                    .filter(|bytes| *bytes <= limits.cache.max_snapshot_bytes)
                    .ok_or_else(|| Error::Limit("Azure selection total bytes".into()))?;
                let plan = pin.plan(limits, Some(row.etag))?;
                let file = &plan.files()[0];
                control = [
                    file.path().len(),
                    file.url().len(),
                    file.identity().len(),
                    plan.source().repository().len(),
                    plan.source().revision().len(),
                    file.request_target().map_or(0, HttpTarget::metadata_bytes),
                    file.validator().map_or(0, str::len),
                    256,
                ]
                .into_iter()
                .try_fold(control, usize::checked_add)
                .filter(|bytes| *bytes <= limits.cache.max_manifest_bytes)
                .ok_or_else(|| Error::Limit("Azure selection metadata bound".into()))?;
                objects.push((pin, plan));
            }
            if page.next.is_empty() {
                break;
            }
            if page.next.len() > 4096
                || page.next.chars().any(char::is_control)
                || !markers.insert(page.next.clone())
            {
                return Err(Error::Malformed(
                    "Azure listing marker invalid or cyclic".into(),
                ));
            }
            marker = Some(page.next);
        }
        if objects.is_empty() {
            return Err(Error::Malformed("Azure prefix selection is empty".into()));
        }
        for path in &paths {
            for (offset, _) in path.match_indices('/') {
                if paths.contains(&path[..offset]) {
                    return Err(Error::Malformed(
                        "Azure selected file/directory collision".into(),
                    ));
                }
            }
        }
        objects.sort_unstable_by(|a, b| a.1.files()[0].path().cmp(b.1.files()[0].path()));
        Ok(objects)
    }
}
impl fmt::Debug for AzurePrefixRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzurePrefixRequest")
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

/// Offline pin has no SAS, URL input or transient ETag. Remote freshness and
/// token revocation are not checked by verifying an original cached snapshot.
#[derive(Clone, Debug)]
pub struct AzureObjectPin {
    request: AzureObjectRequest,
    bytes: u64,
    source: ManifestSource,
    identity: String,
}
impl AzureObjectPin {
    pub fn new(request: &AzureObjectRequest, bytes: u64) -> Result<Self> {
        let object = hashed(&(
            "zerogravity.azure.object.v1",
            &request.origin,
            &request.container,
            &request.blob,
        ))?;
        let contract = hashed(&(
            "zerogravity.azure.contract.v1",
            &request.origin,
            &request.container,
            &request.blob,
            request.revision.kind,
            request.revision.value(),
            &request.path,
            request.role,
            bytes,
            &request.sha256,
        ))?;
        let revision = hashed(&(request.revision.kind, request.revision.value()))?;
        let source = ManifestSource::new(
            "azure",
            format!("object-{object}"),
            format!(
                "{}-{revision}-contract-{contract}",
                request.revision.query_name()
            ),
        )?;
        Ok(Self {
            request: request.clone(),
            bytes,
            source,
            identity: format!("azure-{contract}"),
        })
    }
    pub fn request(&self) -> &AzureObjectRequest {
        &self.request
    }
    pub fn revision(&self) -> &AzureRevision {
        self.request.revision()
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }
    fn plan(&self, limits: &PullLimits, validator: Option<String>) -> Result<ResolvedPullPlan> {
        let target = self.request.target()?;
        let file = ResolvedFile::new(
            &self.request.path,
            target.url(),
            self.request.role,
            self.bytes,
            self.request.sha256.clone(),
            &self.identity,
        )?
        .with_validator(validator)?
        .with_response_binding(self.request.revision.binding())
        .with_request_target(target);
        ResolvedPullPlan::new(self.source.clone(), vec![file], limits)
    }
    pub fn open_cached(
        &self,
        store: &CacheStore,
        snapshot: &str,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<SnapshotLease> {
        self.plan(limits, None)?;
        check_cancel(cancel)?;
        let lease = store.open_snapshot_cancellable(snapshot, cancel)?;
        if lease.manifest().source() != &self.source || lease.manifest().entries().len() != 1 {
            return Err(Error::Malformed(
                "Azure cached source or selection mismatch".into(),
            ));
        }
        let entry = &lease.manifest().entries()[0];
        let provenance = if self.request.sha256.is_some() {
            DigestProvenance::TrustedSha256
        } else {
            DigestProvenance::FirstSeen
        };
        if entry.path() != self.request.path
            || entry.role() != self.request.role
            || entry.bytes() != self.bytes
            || entry.digest_provenance() != provenance
            || self
                .request
                .sha256
                .as_ref()
                .is_some_and(|hash| hash != entry.digest())
        {
            return Err(Error::Malformed("Azure cached contract mismatch".into()));
        }
        let file = lease.file(entry.path())?;
        let mut prefix = [0u8; 128];
        let mut done = 0;
        while done < prefix.len() {
            check_cancel(cancel)?;
            let read = file.read_at(&mut prefix[done..], done as u64)?;
            check_cancel(cancel)?;
            if read == 0 {
                break;
            }
            done += read;
        }
        super::executor::reject_pointer(&prefix[..done])?;
        Ok(lease)
    }
}

fn admit(
    limits: &PullLimits,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<GlobalResourceLease> {
    limits.validate()?;
    check_cancel(cancel)?;
    let bytes = (limits.max_metadata_bytes as u64)
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(16 * 1024 * 1024))
        .ok_or_else(|| Error::Limit("Azure metadata admission overflow".into()))?;
    let lease = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes,
        }],
    )?;
    check_cancel(cancel)?;
    Ok(lease)
}
fn invalid() -> Error {
    Error::Malformed("invalid Azure metadata or selection".into())
}
fn origin_checked(origin: &str) -> Result<String> {
    if origin.is_empty()
        || origin.len() > 2048
        || origin.trim() != origin
        || origin.chars().any(char::is_control)
        || origin.contains(['@', '?', '#', '\\'])
    {
        return Err(invalid());
    }
    let url = Url::parse(origin).map_err(|_| invalid())?;
    let authority = origin.split_once("://").ok_or_else(invalid)?.1;
    if authority.contains('/')
        && authority
            .strip_suffix('/')
            .is_none_or(|authority| authority.contains('/'))
    {
        return Err(invalid());
    }
    if url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(invalid());
    }
    let normalized = url.origin().ascii_serialization();
    let diagnostic = cfg!(feature = "diagnostic-acquisition")
        && matches!(url.scheme(), "http" | "https")
        && (matches!(url.host(),Some(url::Host::Ipv4(ip)) if ip.is_loopback())
            || matches!(url.host(),Some(url::Host::Ipv6(ip)) if ip.is_loopback()));
    if !diagnostic {
        let account = url
            .host_str()
            .and_then(|host| host.strip_suffix(".blob.core.windows.net"))
            .ok_or_else(|| {
                Error::Unsupported("Azure requires a native public-cloud account origin".into())
            })?;
        if url.scheme() != "https"
            || url.port().is_some()
            || !(3..=24).contains(&account.len())
            || !account
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        {
            return Err(invalid());
        }
    }
    Ok(normalized)
}
fn container_checked(container: &str) -> Result<()> {
    if !(3..=63).contains(&container.len())
        || !container
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || container.starts_with('-')
        || container.ends_with('-')
        || container.contains("--")
    {
        return Err(invalid());
    }
    Ok(())
}
fn blob_checked(blob: &str) -> Result<()> {
    if blob.len() > 1024 {
        return Err(Error::Limit("Azure blob name limit".into()));
    }
    validate_relative_path(blob)
}
fn decimal(value: &str) -> Result<u64> {
    if value.is_empty() || value.len() > 20 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }
    value.parse().map_err(|_| invalid())
}
fn component(value: &str) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || b"-_.~".contains(&byte) {
            output.push(char::from(byte));
        } else {
            write!(&mut output, "%{byte:02X}").expect("writing to string");
        }
    }
    output
}
fn encoded_blob(blob: &str) -> String {
    blob.split('/').map(component).collect::<Vec<_>>().join("/")
}
fn hashed(value: &impl serde::Serialize) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).map_err(|_| invalid())?)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    const ORIGIN: &str = "https://fixture.blob.core.windows.net";
    fn request(revision: AzureRevision) -> AzureObjectRequest {
        AzureObjectRequest::new(
            ORIGIN,
            "container",
            "folder/a.bin",
            "a.bin",
            FileRole::Auxiliary,
            revision,
            None,
        )
        .unwrap()
    }
    #[test]
    fn origins_and_pure_selectors_are_bounded_before_io() {
        for origin in [
            "https://fixture.blob.core.windows.net/?sig=SECRET",
            "https://@fixture.blob.core.windows.net",
            "https://fixture.blob.core.windows.net/a/..",
            "https://fixture.blob.core.windows.net:444",
            "https://bad.account.blob.core.windows.net",
            "https://fixture-secondary.blob.core.windows.net",
            "https://fixture.blob.core.windows.net\\evil",
        ] {
            assert!(
                AzureObjectRequest::new(
                    origin,
                    "container",
                    "a.bin",
                    "a.bin",
                    FileRole::Auxiliary,
                    AzureRevision::version("opaque").unwrap(),
                    None
                )
                .is_err()
            );
        }
        assert_eq!(
            AzureObjectRequest::new(
                "HTTPS://FIXTURE.BLOB.CORE.WINDOWS.NET:443/",
                "container",
                "a.bin",
                "a.bin",
                FileRole::Auxiliary,
                AzureRevision::version("opaque").unwrap(),
                None
            )
            .unwrap()
            .origin(),
            ORIGIN
        );
        for value in ["", "bad value", "bad\nvalue"] {
            assert!(AzureRevision::version(value).is_err());
        }
        assert!(AzureRevision::version(&"x".repeat(129)).is_err());
        assert!(AzurePrefixRequest::new(ORIGIN, "container", "folder").is_err());
        assert!(
            AzureObjectRequest::new(
                ORIGIN,
                "container",
                "../bad",
                "a.bin",
                FileRole::Auxiliary,
                AzureRevision::version("opaque").unwrap(),
                None
            )
            .is_err()
        );
    }
    #[test]
    fn immutable_pin_binds_revision_namespace_bytes_and_contract_without_etag() {
        let version = request(AzureRevision::version("OPAQUE_VALUE").unwrap());
        let snapshot = request(AzureRevision::snapshot("OPAQUE_VALUE").unwrap());
        let pin = AzureObjectPin::new(&version, 3).unwrap();
        assert_ne!(
            pin.source(),
            AzureObjectPin::new(&snapshot, 3).unwrap().source()
        );
        assert_ne!(
            pin.source(),
            AzureObjectPin::new(&version, 4).unwrap().source()
        );
        let a = pin
            .plan(&PullLimits::default(), Some("\"one\"".into()))
            .unwrap();
        let b = pin
            .plan(&PullLimits::default(), Some("\"two\"".into()))
            .unwrap();
        let offline = pin.plan(&PullLimits::default(), None).unwrap();
        assert_eq!(a.source(), b.source());
        assert_eq!(b.source(), offline.source());
        assert_eq!(a.files()[0].identity(), b.files()[0].identity());
        assert!(!format!("{pin:?}").contains("OPAQUE_VALUE"));
        assert!(!pin.source().revision().contains("OPAQUE_VALUE"));
    }
}
