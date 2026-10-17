//! Native GCS JSONv1 metadata resolution, with generation and metageneration
//! preconditions on every media request. Each returned plan is one original
//! object, not an atomic bucket snapshot or a model closure. No SDK, credential
//! discovery, metadata-service request, refresh or provider mutation occurs.
//!
//! Public access and caller-supplied bearer credentials use the existing HTTPS
//! owner. CRC32C, MD5, ETag and custom metadata never authenticate raw SHA256.

use super::http::{HttpAccept, HttpRequest, HttpResponse, HttpTransport, check_cancel};
use super::{DigestProvenance, PullLimits, ResolvedFile, ResolvedPullPlan, ResponseBinding};
use crate::cache::paths::{validate_digest, validate_relative_path};
use crate::cache::{CacheStore, FileRole, ManifestSource, SnapshotLease, TransferAdmission};
use crate::{Error, Result};
use serde::Deserialize;
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::os::unix::fs::FileExt;
use std::sync::atomic::AtomicBool;
use url::Url;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};

const MAX_OBJECTS: usize = 4096;
const MAX_TOKEN: usize = 4096;
const NATIVE_ORIGIN: &str = "https://storage.googleapis.com";
const OBJECT_FIELDS: &str = "kind,bucket,name,generation,metageneration,size,contentEncoding,customerEncryption,softDeleteTime,restoreToken";

/// Explicit object selection. The URL and bearer token are not inputs to source
/// identity. `sha256` is an independently authorized raw digest, never metadata
/// inferred from an ETag, MD5, CRC32C or object name.
#[derive(Clone)]
pub struct GcsObjectRequest {
    origin: String,
    bucket: String,
    object: String,
    path: String,
    role: FileRole,
    generation: Option<u64>,
    sha256: Option<String>,
}
impl GcsObjectRequest {
    pub fn new(
        origin: &str,
        bucket: &str,
        object: &str,
        path: &str,
        role: FileRole,
        generation: Option<&str>,
        sha256: Option<&str>,
    ) -> Result<Self> {
        let origin = validate_origin(origin)?;
        validate_bucket(bucket)?;
        validate_object(object)?;
        validate_relative_path(path)?;
        let generation = generation.map(version).transpose()?;
        if let Some(hash) = sha256 {
            validate_digest(hash)?;
        }
        Ok(Self {
            origin,
            bucket: bucket.to_owned(),
            object: object.to_owned(),
            path: path.to_owned(),
            role,
            generation,
            sha256: sha256.map(str::to_owned),
        })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn bucket(&self) -> &str {
        &self.bucket
    }
    pub fn object(&self) -> &str {
        &self.object
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn role(&self) -> FileRole {
        self.role
    }
    pub fn requested_generation(&self) -> Option<u64> {
        self.generation
    }
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }

    /// Metadata only, admitted before the first request. Latest is resolved once
    /// to a generation. The returned original plan uses no mutable-latest URL.
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

    /// Return the observed exact versions/size alongside the original plan,
    /// without parsing an identity string or requesting metadata a second time.
    pub fn resolve_with_pin(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<(GcsObjectPin, ResolvedPullPlan)> {
        let _lease = admit(limits, admission, cancel)?;
        let mut url = object_url(&self.origin, &self.bucket, &self.object)?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("fields", OBJECT_FIELDS);
            if let Some(generation) = self.generation {
                query.append_pair("generation", &generation.to_string());
            }
        }
        let mut budget = MetadataBudget::new(limits);
        let bytes = budget.get(transport, url, cancel)?;
        let metadata: Object = serde_json::from_slice(&bytes).map_err(|_| malformed())?;
        let pin = checked_pin(self, metadata)?;
        let plan = pin.plan(limits)?;
        Ok((pin, plan))
    }
}
impl fmt::Debug for GcsObjectRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcsObjectRequest")
            .field("origin", &self.origin)
            .field("path", &self.path)
            .field("role", &self.role)
            .field("generation", &self.generation)
            .field("sha256", &self.sha256)
            .finish_non_exhaustive()
    }
}

/// Explicit auxiliary-only prefix selection. All pages are bounded and each
/// selected object retains its own generation/source. Listing is not atomic;
/// deleted generations can fail during later transfer. No closure is inferred.
#[derive(Clone)]
pub struct GcsPrefixRequest {
    origin: String,
    bucket: String,
    prefix: String,
}
impl GcsPrefixRequest {
    pub fn new(origin: &str, bucket: &str, prefix: &str) -> Result<Self> {
        let origin = validate_origin(origin)?;
        validate_bucket(bucket)?;
        validate_object(prefix)?;
        if !prefix.ends_with('/') {
            return Err(Error::Malformed("GCS prefix must end with slash".into()));
        }
        Ok(Self {
            origin,
            bucket: bucket.to_owned(),
            prefix: prefix.to_owned(),
        })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn bucket(&self) -> &str {
        &self.bucket
    }
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Return independent original plans after the whole bounded selection is
    /// validated. Prefix entries are FirstSeen Auxiliary data, not trusted model
    /// weights or a combined atomic publication.
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

    /// Expose each independently observed pin for reproducible offline inputs.
    /// The original plans and per-object publication boundaries are unchanged.
    pub fn resolve_with_pins(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<Vec<(GcsObjectPin, ResolvedPullPlan)>> {
        let _lease = admit(limits, admission, cancel)?;
        let mut budget = MetadataBudget::new(limits);
        let mut token = None::<String>;
        let mut tokens = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut paths = BTreeSet::new();
        let mut plans = Vec::new();
        let mut total = 0u64;
        let mut control = 0usize;
        loop {
            check_cancel(cancel)?;
            let mut url = Url::parse(&format!("{}/storage/v1/b/{}/o", self.origin, self.bucket))
                .map_err(|_| malformed())?;
            {
                let mut query = url.query_pairs_mut();
                query.append_pair("prefix", &self.prefix);
                query.append_pair("maxResults", &limits.cache.max_files.min(1000).to_string());
                query.append_pair("fields", &format!("nextPageToken,items({OBJECT_FIELDS})"));
                if let Some(token) = &token {
                    query.append_pair("pageToken", token);
                }
            }
            let bytes = budget.get(transport, url, cancel)?;
            let page: Page = serde_json::from_slice(&bytes).map_err(|_| malformed())?;
            for metadata in page.items.0 {
                check_cancel(cancel)?;
                if plans.len() == limits.cache.max_files {
                    return Err(Error::Limit("GCS selection file count".into()));
                }
                let path = metadata
                    .name
                    .strip_prefix(self.prefix.as_str())
                    .ok_or_else(malformed)?;
                validate_relative_path(path)?;
                if !names.insert(metadata.name.clone()) || !paths.insert(path.to_owned()) {
                    return Err(Error::Malformed("duplicate GCS selected object".into()));
                }
                let request = GcsObjectRequest::new(
                    &self.origin,
                    &self.bucket,
                    &metadata.name,
                    path,
                    FileRole::Auxiliary,
                    None,
                    None,
                )?;
                let pin = checked_pin(&request, metadata)?;
                total = total
                    .checked_add(pin.bytes)
                    .filter(|bytes| *bytes <= limits.cache.max_snapshot_bytes)
                    .ok_or_else(|| Error::Limit("GCS selection total bytes".into()))?;
                let plan = pin.plan(limits)?;
                let file = &plan.files()[0];
                control = control
                    .checked_add(
                        file.path().len()
                            + file.url().len()
                            + file.identity().len()
                            + plan.source().repository().len()
                            + plan.source().revision().len()
                            + 160,
                    )
                    .filter(|bytes| *bytes <= limits.cache.max_manifest_bytes)
                    .ok_or_else(|| Error::Limit("GCS selection metadata bound".into()))?;
                plans.push((pin, plan));
            }
            match page.next_page_token {
                None => break,
                Some(next) => {
                    if next.is_empty()
                        || next.len() > MAX_TOKEN
                        || next.chars().any(char::is_control)
                    {
                        return Err(malformed());
                    }
                    if !tokens.insert(next.clone()) {
                        return Err(Error::Malformed("GCS pagination cycle".into()));
                    }
                    token = Some(next);
                }
            }
        }
        if plans.is_empty() {
            return Err(Error::Malformed("GCS prefix selection is empty".into()));
        }
        for path in &paths {
            for (index, _) in path.match_indices('/') {
                if paths.contains(&path[..index]) {
                    return Err(Error::Malformed(
                        "GCS selection file/directory collision".into(),
                    ));
                }
            }
        }
        plans.sort_unstable_by(|a, b| a.1.files()[0].path().cmp(b.1.files()[0].path()));
        Ok(plans)
    }
}
impl fmt::Debug for GcsPrefixRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcsPrefixRequest")
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

/// Exact original object and metadata version. It can be reconstructed offline
/// from caller-authorized values; it does not prove present bucket membership.
#[derive(Clone, Debug)]
pub struct GcsObjectPin {
    request: GcsObjectRequest,
    generation: u64,
    metageneration: u64,
    bytes: u64,
    source: ManifestSource,
    identity: String,
}
impl GcsObjectPin {
    pub fn new(
        request: &GcsObjectRequest,
        generation: &str,
        metageneration: &str,
        bytes: u64,
    ) -> Result<Self> {
        let generation = version(generation)?;
        let metageneration = version(metageneration)?;
        if request
            .generation
            .is_some_and(|expected| expected != generation)
        {
            return Err(Error::State("GCS requested generation changed".into()));
        }
        let object_key = hash_tuple(&(
            "zerogravity.gcs.object.v1",
            &request.origin,
            &request.bucket,
            &request.object,
        ))?;
        let contract = hash_tuple(&(
            "zerogravity.gcs.contract.v1",
            &request.origin,
            &request.bucket,
            &request.object,
            generation,
            metageneration,
            &request.path,
            request.role,
            bytes,
            &request.sha256,
        ))?;
        let source = ManifestSource::new(
            "gcs",
            format!("object-{object_key}"),
            format!("generation-{generation}-metageneration-{metageneration}-contract-{contract}"),
        )?;
        Ok(Self {
            request: request.clone(),
            generation,
            metageneration,
            bytes,
            source,
            identity: format!("gcs-{contract}"),
        })
    }
    pub fn request(&self) -> &GcsObjectRequest {
        &self.request
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn metageneration(&self) -> u64 {
        self.metageneration
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }

    fn plan(&self, limits: &PullLimits) -> Result<ResolvedPullPlan> {
        let mut url = object_url(
            &self.request.origin,
            &self.request.bucket,
            &self.request.object,
        )?;
        url.query_pairs_mut()
            .append_pair("alt", "media")
            .append_pair("generation", &self.generation.to_string())
            .append_pair("ifMetagenerationMatch", &self.metageneration.to_string());
        let file = ResolvedFile::new(
            &self.request.path,
            url.as_str(),
            self.request.role,
            self.bytes,
            self.request.sha256.clone(),
            &self.identity,
        )?
        .with_response_binding(ResponseBinding::GcsVersion {
            generation: self.generation,
            metageneration: self.metageneration,
        });
        ResolvedPullPlan::new(self.source.clone(), vec![file], limits)
    }

    /// Full original cache verification without URL input, client or credentials.
    /// FirstSeen remains FirstSeen, and no remote freshness/revocation is checked.
    pub fn open_cached(
        &self,
        store: &CacheStore,
        snapshot: &str,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<SnapshotLease> {
        limits.validate()?;
        if self.bytes > limits.cache.max_file_bytes || self.bytes > limits.cache.max_snapshot_bytes
        {
            return Err(Error::Limit("GCS cached object length".into()));
        }
        // Check the caller's current path/control bounds before opening/hashing.
        self.plan(limits)?;
        let lease = store.open_snapshot_cancellable(snapshot, cancel)?;
        let manifest = lease.manifest();
        if manifest.source() != &self.source || manifest.entries().len() != 1 {
            return Err(Error::Malformed(
                "GCS cached source or selection mismatch".into(),
            ));
        }
        let entry = &manifest.entries()[0];
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
            return Err(Error::Malformed(
                "GCS cached object contract mismatch".into(),
            ));
        }
        let file = lease.file(entry.path())?;
        let mut prefix = [0u8; 128];
        let mut read = 0;
        while read < prefix.len() {
            check_cancel(cancel)?;
            let count = file.read_at(&mut prefix[read..], read as u64)?;
            check_cancel(cancel)?;
            if count == 0 {
                break;
            }
            read += count;
        }
        super::executor::reject_pointer(&prefix[..read])?;
        Ok(lease)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Object {
    kind: String,
    bucket: String,
    name: String,
    generation: String,
    metageneration: String,
    size: String,
    content_encoding: Option<String>,
    customer_encryption: Option<IgnoredAny>,
    soft_delete_time: Option<String>,
    restore_token: Option<String>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Page {
    #[serde(default)]
    items: Objects,
    next_page_token: Option<String>,
}
#[derive(Default)]
struct Objects(Vec<Object>);
impl<'de> Deserialize<'de> for Objects {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct BoundedObjects;
        impl<'de> Visitor<'de> for BoundedObjects {
            type Value = Objects;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a bounded GCS object list")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut array: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut objects = Vec::new();
                while let Some(object) = array.next_element()? {
                    if objects.len() == MAX_OBJECTS {
                        return Err(serde::de::Error::custom("GCS object count"));
                    }
                    objects.push(object);
                }
                Ok(Objects(objects))
            }
        }
        deserializer.deserialize_seq(BoundedObjects)
    }
}

fn checked_pin(request: &GcsObjectRequest, metadata: Object) -> Result<GcsObjectPin> {
    if metadata.kind != "storage#object"
        || metadata.bucket != request.bucket
        || metadata.name != request.object
    {
        return Err(Error::Malformed(
            "GCS object metadata identity mismatch".into(),
        ));
    }
    if metadata
        .content_encoding
        .as_ref()
        .is_some_and(|encoding| !encoding.eq_ignore_ascii_case("identity"))
    {
        return Err(Error::Unsupported(
            "GCS stored content requires identity encoding".into(),
        ));
    }
    if metadata.customer_encryption.is_some()
        || metadata.soft_delete_time.is_some()
        || metadata.restore_token.is_some()
    {
        return Err(Error::Unsupported(
            "GCS encrypted or soft-deleted object profile".into(),
        ));
    }
    let bytes = decimal(&metadata.size)?;
    GcsObjectPin::new(
        request,
        &metadata.generation,
        &metadata.metageneration,
        bytes,
    )
}

struct MetadataBudget {
    remaining: usize,
    requests: usize,
    maximum_requests: usize,
}
impl MetadataBudget {
    fn new(limits: &PullLimits) -> Self {
        Self {
            remaining: limits.max_metadata_bytes,
            requests: 0,
            maximum_requests: limits.max_pages,
        }
    }
    fn get(
        &mut self,
        transport: &(impl HttpTransport + ?Sized),
        url: Url,
        cancel: &AtomicBool,
    ) -> Result<Vec<u8>> {
        check_cancel(cancel)?;
        if self.remaining == 0
            || self.requests == self.maximum_requests
            || url.as_str().len() > 16 * 1024
        {
            return Err(Error::Limit("GCS metadata request budget".into()));
        }
        self.requests += 1;
        let response = transport.get(
            &HttpRequest {
                url: url.into(),
                accept: HttpAccept::Json,
                range: None,
                max_body_bytes: self.remaining as u64,
            },
            cancel,
        );
        check_cancel(cancel)?;
        let response = response.map_err(|_| Error::State("GCS metadata request failed".into()))?;
        validate_metadata_response(&response)?;
        let declared = response
            .header("content-length")
            .map(super::http::parse_u64)
            .transpose()?;
        let bytes = response.read_bounded(self.remaining, cancel)?;
        if declared.is_some_and(|size| size != bytes.len() as u64) {
            return Err(malformed());
        }
        self.remaining -= bytes.len();
        Ok(bytes)
    }
}
fn validate_metadata_response(response: &HttpResponse) -> Result<()> {
    if response.status != 200 {
        return Err(Error::State(format!(
            "GCS metadata status {}",
            response.status
        )));
    }
    if response
        .header("content-encoding")
        .is_some_and(|encoding| !encoding.eq_ignore_ascii_case("identity"))
    {
        return Err(Error::Unsupported("GCS metadata encoding".into()));
    }
    Ok(())
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
        .ok_or_else(|| Error::Limit("GCS metadata admission overflow".into()))?;
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
fn malformed() -> Error {
    Error::Malformed("invalid GCS metadata or selection".into())
}
fn decimal(value: &str) -> Result<u64> {
    if value.is_empty()
        || value.len() > 20
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(malformed());
    }
    value.parse().map_err(|_| malformed())
}
fn version(value: &str) -> Result<u64> {
    let value = decimal(value)?;
    if value == 0 || value > i64::MAX as u64 {
        return Err(malformed());
    }
    Ok(value)
}
fn validate_bucket(bucket: &str) -> Result<()> {
    if !(3..=222).contains(&bucket.len())
        || !bucket.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        })
        || !bucket.as_bytes()[0].is_ascii_alphanumeric()
        || !bucket.as_bytes()[bucket.len() - 1].is_ascii_alphanumeric()
        || bucket
            .split('.')
            .any(|part| part.is_empty() || part.len() > 63)
    {
        return Err(malformed());
    }
    Ok(())
}
fn validate_object(object: &str) -> Result<()> {
    if object.is_empty()
        || object.len() > 1024
        || matches!(object, "." | "..")
        || object.chars().any(char::is_control)
    {
        return Err(malformed());
    }
    Ok(())
}
fn validate_origin(origin: &str) -> Result<String> {
    if origin.is_empty()
        || origin.len() > 2048
        || origin.chars().any(char::is_control)
        || origin.trim() != origin
        || origin.contains(['@', '?', '#', '\\'])
    {
        return Err(malformed());
    }
    let parsed = Url::parse(origin).map_err(|_| malformed())?;
    let Some((_, authority)) = origin.split_once("://") else {
        return Err(malformed());
    };
    if authority.contains('/')
        && authority
            .strip_suffix('/')
            .is_none_or(|authority| authority.contains('/'))
    {
        return Err(malformed());
    }
    if parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(malformed());
    }
    let canonical = parsed.origin().ascii_serialization();
    if canonical != NATIVE_ORIGIN {
        let diagnostic = cfg!(feature = "diagnostic-acquisition")
            && matches!(parsed.scheme(), "http" | "https")
            && (matches!(parsed.host(), Some(url::Host::Ipv4(ip)) if ip.is_loopback())
                || matches!(parsed.host(), Some(url::Host::Ipv6(ip)) if ip.is_loopback()));
        if !diagnostic {
            return Err(Error::Unsupported(
                "GCS requires the native HTTPS origin".into(),
            ));
        }
    }
    Ok(canonical)
}
fn component(value: &str) -> String {
    use std::fmt::Write;
    let mut result = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || b"-_.~".contains(&byte) {
            result.push(char::from(byte));
        } else {
            write!(&mut result, "%{byte:02X}").expect("writing to string");
        }
    }
    result
}
fn object_url(origin: &str, bucket: &str, object: &str) -> Result<Url> {
    Url::parse(&format!(
        "{origin}/storage/v1/b/{bucket}/o/{}",
        component(object)
    ))
    .map_err(|_| malformed())
}
fn hash_tuple(value: &impl serde::Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(|_| malformed())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
