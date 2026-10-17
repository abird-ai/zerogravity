//! Explicit versioned S3 GET/HEAD and bounded auxiliary prefix selections.
//! Each object has its own immutable version; a list is not an atomic snapshot.
//! No SDK, ambient credentials, region discovery, write or archive restore.
use super::http::{self, HttpRequest, HttpResponse, HttpTarget, HttpTransport};
use super::{PullLimits, ResolvedFile, ResolvedPullPlan, ResponseBinding};
use crate::cache::paths::{validate_digest, validate_relative_path};
use crate::cache::{
    CacheStore, DigestProvenance, FileRole, ManifestSource, SnapshotLease, TransferAdmission,
};
use crate::{Error, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::os::unix::fs::FileExt;
use std::sync::atomic::AtomicBool;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};
#[path = "s3_xml.rs"]
mod xml;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum S3Addressing {
    PathStyle,
    VirtualHosted,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum Service {
    AwsStandard,
    Compatible,
}
#[derive(Clone, Debug, Serialize)]
pub struct S3Endpoint {
    origin: String,
    region: String,
    addressing: S3Addressing,
    service: Service,
}
impl S3Endpoint {
    /// Standard commercial AWS regional endpoint only. China/Gov/ISO, access
    /// points, directory buckets, FIPS/dualstack/accelerate are not inferred.
    pub fn aws(region: &str, addressing: S3Addressing) -> Result<Self> {
        if region.len() > 64 {
            return Err(malformed());
        }
        let parts = region.split('-').collect::<Vec<_>>();
        if parts.len() != 3
            || !matches!(
                parts[0],
                "us" | "eu" | "ap" | "ca" | "sa" | "me" | "af" | "il" | "mx"
            )
            || parts[1].is_empty()
            || !parts[1].bytes().all(|b| b.is_ascii_lowercase())
            || parts[2].is_empty()
            || !parts[2].bytes().all(|b| b.is_ascii_digit())
            || parts[2].starts_with('0')
        {
            return Err(Error::Unsupported(
                "unsupported AWS S3 region profile".into(),
            ));
        }
        Self::build(
            &format!("https://s3.{region}.amazonaws.com"),
            region,
            addressing,
            Service::AwsStandard,
        )
    }
    /// Explicit bare endpoint and signing region, including compatible 'auto'.
    /// This profile does not certify arbitrary vendor features or AWS parity.
    pub fn compatible(origin: &str, region: &str, addressing: S3Addressing) -> Result<Self> {
        Self::build(origin, region, addressing, Service::Compatible)
    }
    fn build(
        origin: &str,
        region: &str,
        addressing: S3Addressing,
        service: Service,
    ) -> Result<Self> {
        if origin.len() > 4096
            || origin.trim() != origin
            || origin.chars().any(char::is_control)
            || origin.contains('\\')
            || region.is_empty()
            || region.len() > 64
            || !region
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return Err(malformed());
        }
        let (_, tail) = origin.split_once("://").ok_or_else(malformed)?;
        let split = tail.find(['/', '?', '#']).unwrap_or(tail.len());
        if tail[..split].contains('@') || !matches!(&tail[split..], "" | "/") {
            return Err(malformed());
        }
        let url = http::parse_provider_origin(origin)?;
        Ok(Self {
            origin: url.origin().ascii_serialization(),
            region: region.to_owned(),
            addressing,
            service,
        })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn region(&self) -> &str {
        &self.region
    }
    pub fn addressing(&self) -> S3Addressing {
        self.addressing
    }
    pub fn credential_origin(&self, bucket: &str) -> Result<String> {
        bucket_name(bucket)?;
        if self.addressing == S3Addressing::PathStyle {
            return Ok(self.origin.clone());
        }
        if bucket.contains('.') {
            return Err(Error::Unsupported(
                "virtual-host S3 bucket dots require a separately qualified TLS profile".into(),
            ));
        }
        let mut url = http::parse_provider_origin(&self.origin)?;
        let host = url.host_str().ok_or_else(malformed)?;
        if host.parse::<std::net::IpAddr>().is_ok() {
            return Err(Error::Unsupported(
                "virtual-host S3 requires an explicit DNS endpoint".into(),
            ));
        }
        url.set_host(Some(&format!("{bucket}.{host}")))
            .map_err(|_| malformed())?;
        Ok(url.origin().ascii_serialization())
    }
    fn target(
        &self,
        bucket: &str,
        key: Option<&str>,
        query: &[(&str, &str)],
    ) -> Result<HttpTarget> {
        let origin = self.credential_origin(bucket)?;
        let mut path = if self.addressing == S3Addressing::PathStyle {
            format!("/{bucket}")
        } else {
            String::new()
        };
        path.push('/');
        if let Some(key) = key {
            path.push_str(&encode(key, false));
        }
        let mut pairs = query
            .iter()
            .map(|(k, v)| (encode(k, true), encode(v, true)))
            .collect::<Vec<_>>();
        pairs.sort_unstable();
        for (index, (name, value)) in pairs.into_iter().enumerate() {
            path.push(if index == 0 { '?' } else { '&' });
            path.push_str(&name);
            path.push('=');
            path.push_str(&value);
        }
        HttpTarget::s3(&origin, &self.region, &path)
    }
}

#[derive(Clone)]
pub struct S3ObjectRequest {
    endpoint: S3Endpoint,
    bucket: String,
    key: String,
    path: String,
    role: FileRole,
    requested_version: Option<String>,
    sha256: Option<String>,
}
impl fmt::Debug for S3ObjectRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3ObjectRequest")
            .field("path", &self.path)
            .field("role", &self.role)
            .finish_non_exhaustive()
    }
}
impl S3ObjectRequest {
    pub fn new(
        endpoint: &S3Endpoint,
        bucket: &str,
        key: &str,
        path: &str,
        role: FileRole,
        version: Option<&str>,
        sha256: Option<&str>,
    ) -> Result<Self> {
        bucket_name(bucket)?;
        object_key(key)?;
        validate_relative_path(path)?;
        if let Some(version) = version {
            valid_version(version)?;
        }
        if let Some(digest) = sha256 {
            validate_digest(digest)?;
        }
        endpoint.credential_origin(bucket)?;
        Ok(Self {
            endpoint: endpoint.clone(),
            bucket: bucket.to_owned(),
            key: key.to_owned(),
            path: path.to_owned(),
            role,
            requested_version: version.map(str::to_owned),
            sha256: sha256.map(str::to_owned),
        })
    }
    pub fn endpoint(&self) -> &S3Endpoint {
        &self.endpoint
    }
    pub fn bucket(&self) -> &str {
        &self.bucket
    }
    pub fn key(&self) -> &str {
        &self.key
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn role(&self) -> FileRole {
        self.role
    }
    pub fn requested_version(&self) -> Option<&str> {
        self.requested_version.as_deref()
    }
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
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
    pub fn resolve_with_pin(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<(S3ObjectPin, ResolvedPullPlan)> {
        let _lease = admit(limits, admission, cancel)?;
        let mut budget = Budget::new(limits);
        let (pin, etag) = self.head(transport, &mut budget, cancel)?;
        let plan = pin.plan_with_validator(limits, etag)?;
        Ok((pin, plan))
    }
    fn head(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        budget: &mut Budget<'_>,
        cancel: &AtomicBool,
    ) -> Result<(S3ObjectPin, Option<String>)> {
        let query = self
            .requested_version
            .as_deref()
            .map(|v| vec![("versionId", v)])
            .unwrap_or_default();
        let target = self
            .endpoint
            .target(&self.bucket, Some(&self.key), &query)?;
        let response = budget.head(transport, &target, cancel)?;
        if response.status != 200 {
            return Err(Error::State(format!(
                "S3 object HEAD status {}",
                response.status
            )));
        }
        if response.header("content-range").is_some()
            || response.header("transfer-encoding").is_some()
            || response
                .header("content-encoding")
                .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
        {
            return Err(Error::Unsupported(
                "S3 requires complete identity object metadata".into(),
            ));
        }
        let version = response.header("x-amz-version-id").ok_or_else(|| {
            Error::Unsupported("S3 requires a non-null immutable object version".into())
        })?;
        valid_version(version)?;
        let bytes = decimal(response.header("content-length").ok_or_else(malformed)?)?;
        let pin = S3ObjectPin::new(self, version, bytes)?;
        http::validate_response_binding(
            Some(ResponseBinding::S3Version {
                version_sha256: version_hash(version),
            }),
            &response,
        )?;
        if bytes > budget.limits.cache.max_file_bytes {
            return Err(Error::Limit("S3 object file length".into()));
        }
        let etag = response
            .header("etag")
            .map(|v| {
                super::resume::strong_etag(v)?;
                Ok::<String, Error>(v.to_owned())
            })
            .transpose()?;
        http::check_cancel(cancel)?;
        Ok((pin, etag))
    }
}

#[derive(Clone)]
pub struct S3ObjectPin {
    request: S3ObjectRequest,
    version: String,
    bytes: u64,
    source: ManifestSource,
    identity: String,
}
impl fmt::Debug for S3ObjectPin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3ObjectPin")
            .field("request", &self.request)
            .field("bytes", &self.bytes)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}
impl S3ObjectPin {
    pub fn new(request: &S3ObjectRequest, version: &str, bytes: u64) -> Result<Self> {
        valid_version(version)?;
        if request
            .requested_version
            .as_deref()
            .is_some_and(|v| v != version)
        {
            return Err(Error::State("S3 requested version changed".into()));
        }
        let object = hash_tuple(&(
            "zerogravity.s3.object.v1",
            &request.endpoint,
            &request.bucket,
            &request.key,
        ))?;
        let contract = hash_tuple(&(
            "zerogravity.s3.contract.v1",
            &request.endpoint,
            &request.bucket,
            &request.key,
            &request.path,
            request.role,
            &request.sha256,
            version,
            bytes,
        ))?;
        let source = ManifestSource::new(
            "s3",
            format!("object-{object}"),
            format!("version-contract-{contract}"),
        )?;
        Ok(Self {
            request: request.clone(),
            version: version.to_owned(),
            bytes,
            source,
            identity: format!("s3-{contract}"),
        })
    }
    pub fn request(&self) -> &S3ObjectRequest {
        &self.request
    }
    pub fn version(&self) -> &str {
        &self.version
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn source(&self) -> &ManifestSource {
        &self.source
    }
    pub fn plan(&self, limits: &PullLimits) -> Result<ResolvedPullPlan> {
        self.plan_with_validator(limits, None)
    }
    fn plan_with_validator(
        &self,
        limits: &PullLimits,
        validator: Option<String>,
    ) -> Result<ResolvedPullPlan> {
        let target = self.request.endpoint.target(
            &self.request.bucket,
            Some(&self.request.key),
            &[("versionId", &self.version)],
        )?;
        let file = ResolvedFile::new(
            &self.request.path,
            target.url(),
            self.request.role,
            self.bytes,
            self.request.sha256.clone(),
            &self.identity,
        )?
        .with_request_target(target)
        .with_response_binding(ResponseBinding::S3Version {
            version_sha256: version_hash(&self.version),
        })
        .with_validator(validator)?;
        ResolvedPullPlan::new(self.source.clone(), vec![file], limits)
    }
    pub fn open_cached(
        &self,
        store: &CacheStore,
        snapshot: &str,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<SnapshotLease> {
        self.plan(limits)?;
        let lease = store.open_snapshot_cancellable(snapshot, cancel)?;
        let manifest = lease.manifest();
        if manifest.source() != &self.source || manifest.entries().len() != 1 {
            return Err(malformed());
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
                .is_some_and(|v| v != entry.digest())
        {
            return Err(malformed());
        }
        let mut prefix = [0u8; 128];
        let count = self.bytes.min(128) as usize;
        http::check_cancel(cancel)?;
        lease
            .file(&self.request.path)?
            .read_exact_at(&mut prefix[..count], 0)?;
        http::check_cancel(cancel)?;
        super::executor::reject_pointer(&prefix[..count])?;
        Ok(lease)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum S3Listing {
    CurrentObjects,
    LatestVersions,
}
#[derive(Clone)]
pub struct S3PrefixRequest {
    endpoint: S3Endpoint,
    bucket: String,
    prefix: String,
    listing: S3Listing,
}
impl fmt::Debug for S3PrefixRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3PrefixRequest").finish_non_exhaustive()
    }
}
impl S3PrefixRequest {
    pub fn new(
        endpoint: &S3Endpoint,
        bucket: &str,
        prefix: &str,
        listing: S3Listing,
    ) -> Result<Self> {
        bucket_name(bucket)?;
        object_key(prefix)?;
        if !prefix.ends_with('/') {
            return Err(Error::Malformed(
                "S3 auxiliary prefix must end with slash".into(),
            ));
        }
        endpoint.credential_origin(bucket)?;
        Ok(Self {
            endpoint: endpoint.clone(),
            bucket: bucket.into(),
            prefix: prefix.into(),
            listing,
        })
    }
    pub fn resolve(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<Vec<(S3ObjectPin, ResolvedPullPlan)>> {
        let _lease = admit(limits, admission, cancel)?;
        let mut budget = Budget::new(limits);
        let mut marker = None::<(String, String)>;
        let mut markers = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let mut selected = BTreeSet::new();
        let mut results = Vec::new();
        let mut total = 0u64;
        let mut controls = 0usize;
        loop {
            http::check_cancel(cancel)?;
            let maximum = limits.cache.max_files.min(1000).to_string();
            let mut query = vec![
                ("encoding-type", "url"),
                ("prefix", self.prefix.as_str()),
                ("max-keys", maximum.as_str()),
            ];
            let versions = self.listing == S3Listing::LatestVersions;
            if versions {
                query.push(("versions", ""));
            } else {
                query.push(("list-type", "2"));
            }
            if let Some((key, version)) = &marker {
                if versions {
                    query.extend([
                        ("key-marker", key.as_str()),
                        ("version-id-marker", version.as_str()),
                    ]);
                } else {
                    query.push(("continuation-token", key.as_str()));
                }
            }
            let target = self.endpoint.target(&self.bucket, None, &query)?;
            let raw = budget.get(transport, &target, cancel)?;
            let page = xml::parse(&raw, versions, limits.cache.max_files.min(1000), cancel)?;
            if page.fields.get("Name") != Some(&self.bucket)
                || page
                    .fields
                    .get("Prefix")
                    .map(|v| decode_key(v))
                    .transpose()?
                    .as_deref()
                    != Some(self.prefix.as_str())
            {
                return Err(malformed());
            }
            let truncated = xml::boolean(page.fields.get("IsTruncated").ok_or_else(malformed)?)?;
            if let Some(actual) = page.fields.get("MaxKeys")
                && decimal(actual)? != limits.cache.max_files.min(1000) as u64
            {
                return Err(malformed());
            }
            if !versions {
                if page
                    .fields
                    .get("KeyCount")
                    .map(|v| decimal(v))
                    .transpose()?
                    .is_some_and(|n| n != page.rows.len() as u64)
                {
                    return Err(malformed());
                }
                if page.fields.get("ContinuationToken").is_some_and(|v| {
                    Some(v.as_str()) != marker.as_ref().map(|(token, _)| token.as_str())
                }) {
                    return Err(malformed());
                }
            } else {
                if page
                    .fields
                    .get("KeyMarker")
                    .map(|v| decode_key(v))
                    .transpose()?
                    .is_some_and(|v| v != marker.as_ref().map_or("", |(key, _)| key.as_str()))
                    || page.fields.get("VersionIdMarker").is_some_and(|v| {
                        v != marker.as_ref().map_or("", |(_, version)| version.as_str())
                    })
                {
                    return Err(malformed());
                }
            }
            for row in page.rows {
                http::check_cancel(cancel)?;
                object_key(&row.key)?;
                if !row.key.starts_with(&self.prefix)
                    || !seen.insert((row.key.clone(), row.version.clone()))
                {
                    return Err(malformed());
                }
                if versions && row.latest != Some(true) {
                    continue;
                }
                if !selected.insert(row.key.clone()) {
                    return Err(Error::Malformed("S3 duplicate latest selected key".into()));
                }
                if row.deleted {
                    continue;
                }
                let path = row.key.strip_prefix(&self.prefix).ok_or_else(malformed)?;
                validate_relative_path(path)?;
                if results.len() == limits.cache.max_files {
                    return Err(Error::Limit("S3 selection file count".into()));
                }
                let request = S3ObjectRequest::new(
                    &self.endpoint,
                    &self.bucket,
                    &row.key,
                    path,
                    FileRole::Auxiliary,
                    row.version.as_deref(),
                    None,
                )?;
                let (pin, etag) = request.head(transport, &mut budget, cancel)?;
                if row.size != Some(pin.bytes) || row.etag.as_ref() != etag.as_ref() {
                    return Err(Error::State("S3 listing and pinned metadata differ".into()));
                }
                total = total
                    .checked_add(pin.bytes)
                    .filter(|v| *v <= limits.cache.max_snapshot_bytes)
                    .ok_or_else(|| Error::Limit("S3 selection total bytes".into()))?;
                let plan = pin.plan_with_validator(limits, etag)?;
                controls = controls
                    .checked_add(
                        plan.files()[0].path().len()
                            + plan.files()[0].url().len() * 2
                            + plan.files()[0].identity().len()
                            + plan.source().repository().len()
                            + plan.source().revision().len()
                            + 256,
                    )
                    .filter(|v| *v <= limits.cache.max_manifest_bytes)
                    .ok_or_else(|| Error::Limit("S3 selection control bound".into()))?;
                results.push((pin, plan));
            }
            if !truncated {
                break;
            }
            let next = if versions {
                (
                    decode_key(page.fields.get("NextKeyMarker").ok_or_else(malformed)?)?,
                    page.fields
                        .get("NextVersionIdMarker")
                        .cloned()
                        .ok_or_else(malformed)?,
                )
            } else {
                (
                    page.fields
                        .get("NextContinuationToken")
                        .cloned()
                        .ok_or_else(malformed)?,
                    String::new(),
                )
            };
            if next.0.is_empty()
                || next.0.len() > 4096
                || next.1.len() > 1024
                || next.0.chars().chain(next.1.chars()).any(char::is_control)
                || !markers.insert(next.clone())
            {
                return Err(Error::Malformed(
                    "S3 pagination marker invalid or repeated".into(),
                ));
            }
            marker = Some(next);
        }
        if results.is_empty() {
            return Err(Error::Malformed("S3 auxiliary selection is empty".into()));
        }
        let paths = results
            .iter()
            .map(|(_, p)| p.files()[0].path())
            .collect::<BTreeSet<_>>();
        for path in &paths {
            for (index, _) in path.match_indices('/') {
                if paths.contains(&path[..index]) {
                    return Err(malformed());
                }
            }
        }
        results.sort_unstable_by(|(_, a), (_, b)| a.files()[0].path().cmp(b.files()[0].path()));
        http::check_cancel(cancel)?;
        Ok(results)
    }
}

struct Budget<'a> {
    limits: &'a PullLimits,
    remaining: usize,
    calls: usize,
}
impl<'a> Budget<'a> {
    fn new(limits: &'a PullLimits) -> Self {
        Self {
            limits,
            remaining: limits.max_metadata_bytes,
            calls: limits.max_pages,
        }
    }
    fn charge_call(&mut self, cancel: &AtomicBool) -> Result<()> {
        http::check_cancel(cancel)?;
        if self.calls == 0 {
            return Err(Error::Limit("S3 metadata request budget".into()));
        }
        self.calls -= 1;
        Ok(())
    }
    fn charge_headers(&mut self, response: &HttpResponse) -> Result<()> {
        let bytes = response
            .headers
            .iter()
            .try_fold(0usize, |sum, (key, value)| {
                sum.checked_add(key.len() + value.len() + 32)
            })
            .ok_or_else(malformed)?;
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or_else(|| Error::Limit("S3 aggregate metadata headers".into()))?;
        Ok(())
    }
    fn head(
        &mut self,
        transport: &(impl HttpTransport + ?Sized),
        target: &HttpTarget,
        cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        self.charge_call(cancel)?;
        let response = transport.head_target(
            &HttpRequest::new(target.url(), self.limits.cache.max_file_bytes),
            target,
            cancel,
        );
        http::check_cancel(cancel)?;
        let response = response.map_err(|_| Error::State("S3 metadata HEAD failed".into()))?;
        self.charge_headers(&response)?;
        Ok(response)
    }
    fn get(
        &mut self,
        transport: &(impl HttpTransport + ?Sized),
        target: &HttpTarget,
        cancel: &AtomicBool,
    ) -> Result<Vec<u8>> {
        self.charge_call(cancel)?;
        let response = transport.get_target(
            &HttpRequest::new(target.url(), self.remaining as u64),
            target,
            cancel,
        );
        http::check_cancel(cancel)?;
        let response = response.map_err(|_| Error::State("S3 listing request failed".into()))?;
        if response.status != 200
            || response.header("content-range").is_some()
            || response
                .header("content-encoding")
                .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
        {
            return Err(Error::State("S3 listing response refused".into()));
        }
        self.charge_headers(&response)?;
        let bytes = response.read_bounded(self.remaining, cancel)?;
        self.remaining -= bytes.len();
        Ok(bytes)
    }
}
fn admit(
    limits: &PullLimits,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<GlobalResourceLease> {
    limits.validate()?;
    http::check_cancel(cancel)?;
    let bytes = (limits.max_metadata_bytes as u64)
        .checked_mul(8)
        .and_then(|n| n.checked_add(16 * 1024 * 1024))
        .ok_or_else(malformed)?;
    admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes,
        }],
    )
}
fn bucket_name(bucket: &str) -> Result<()> {
    if !(3..=63).contains(&bucket.len())
        || !bucket
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b".-".contains(&b))
        || bucket
            .split('.')
            .any(|part| part.is_empty() || part.starts_with('-') || part.ends_with('-'))
        || bucket.parse::<std::net::Ipv4Addr>().is_ok()
        || bucket.ends_with("--x-s3")
        || bucket.ends_with("-s3alias")
        || bucket.ends_with("--ol-s3")
    {
        return Err(malformed());
    }
    Ok(())
}
fn object_key(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 1024 || value.chars().any(char::is_control) {
        return Err(malformed());
    }
    Ok(())
}
fn valid_version(value: &str) -> Result<()> {
    if value.is_empty()
        || value == "null"
        || value.len() > 1024
        || value.chars().any(char::is_control)
    {
        return Err(Error::Unsupported(
            "S3 requires a bounded non-null immutable version".into(),
        ));
    }
    Ok(())
}
pub(super) fn decimal(value: &str) -> Result<u64> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(malformed());
    }
    value.parse().map_err(|_| malformed())
}
fn version_hash(value: &str) -> [u8; 32] {
    Sha256::digest(value.as_bytes()).into()
}
fn hash_tuple(value: &impl Serialize) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).map_err(|_| malformed())?)
    ))
}
fn malformed() -> Error {
    Error::Malformed("invalid explicit S3 metadata or selection".into())
}
fn encode(value: &str, slash: bool) -> String {
    let mut output = String::new();
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) || (!slash && b == b'/') {
            output.push(b as char);
        } else {
            output.push_str(&format!("%{b:02X}"));
        }
    }
    output
}
pub(super) fn decode_key(value: &str) -> Result<String> {
    let mut output = Vec::new();
    let mut bytes = value.bytes();
    while let Some(b) = bytes.next() {
        if b == b'%' {
            let high = bytes
                .next()
                .and_then(|v| (v as char).to_digit(16))
                .ok_or_else(malformed)?;
            let low = bytes
                .next()
                .and_then(|v| (v as char).to_digit(16))
                .ok_or_else(malformed)?;
            output.push(((high << 4) | low) as u8);
        } else {
            output.push(b);
        }
    }
    String::from_utf8(output).map_err(|_| malformed())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_endpoint_regions_addressing_and_canonical_keys_preserve_source() {
        let aws = S3Endpoint::aws("us-east-1", S3Addressing::VirtualHosted).unwrap();
        assert_eq!(
            aws.credential_origin("bucket").unwrap(),
            "https://bucket.s3.us-east-1.amazonaws.com"
        );
        for region in [
            "cn-north-1",
            "us-gov-west-1",
            "us-iso-east-1",
            "unknown",
            "US-EAST-1",
        ] {
            assert!(S3Endpoint::aws(region, S3Addressing::PathStyle).is_err());
        }
        assert!(aws.credential_origin("has.dots").is_err());
        assert!(
            S3Endpoint::compatible("http://object.example", "auto", S3Addressing::PathStyle)
                .is_err()
        );
        let fixture =
            S3Endpoint::compatible("http://127.0.0.1:12345", "auto", S3Addressing::PathStyle)
                .unwrap();
        assert!(fixture.target("bucket", Some("key"), &[]).is_ok());
        let endpoint = S3Endpoint::compatible(
            "HTTPS://OBJECT.EXAMPLE:443/",
            "auto",
            S3Addressing::PathStyle,
        )
        .unwrap();
        let request = S3ObjectRequest::new(
            &endpoint,
            "has.dots",
            "a/../b//./%2E é",
            "model.gguf",
            FileRole::Weights,
            None,
            None,
        )
        .unwrap();
        let pin = S3ObjectPin::new(&request, "v+/=é", 0).unwrap();
        let plan = pin.plan(&PullLimits::default()).unwrap();
        assert!(
            plan.files()[0]
                .url()
                .ends_with("/has.dots/a/../b//./%252E%20%C3%A9?versionId=v%2B%2F%3D%C3%A9")
        );
        assert!(!format!("{request:?} {pin:?} {:?}", plan.files()[0]).contains("a/../b"));
        assert_eq!(pin.bytes(), 0);
        for version in ["", "null", "bad\nversion"] {
            assert!(S3ObjectPin::new(&request, version, 0).is_err());
        }
        assert!(
            S3Endpoint::compatible("https://object.example/..", "auto", S3Addressing::PathStyle)
                .is_err()
        );
    }
    #[test]
    fn exact_object_contract_binds_key_profile_version_role_size_and_checksum() {
        let endpoint = S3Endpoint::compatible(
            "https://object.example",
            "us-east-1",
            S3Addressing::PathStyle,
        )
        .unwrap();
        let request = S3ObjectRequest::new(
            &endpoint,
            "bucket",
            "secret-key",
            "file.bin",
            FileRole::Auxiliary,
            None,
            None,
        )
        .unwrap();
        let pin = S3ObjectPin::new(&request, "v1", 3).unwrap();
        for other in [
            S3ObjectPin::new(&request, "v2", 3).unwrap(),
            S3ObjectPin::new(&request, "v1", 4).unwrap(),
            S3ObjectPin::new(
                &S3ObjectRequest::new(
                    &endpoint,
                    "bucket",
                    "other-key",
                    "file.bin",
                    FileRole::Auxiliary,
                    None,
                    None,
                )
                .unwrap(),
                "v1",
                3,
            )
            .unwrap(),
            S3ObjectPin::new(
                &S3ObjectRequest::new(
                    &endpoint,
                    "bucket",
                    "secret-key",
                    "file.bin",
                    FileRole::Config,
                    None,
                    None,
                )
                .unwrap(),
                "v1",
                3,
            )
            .unwrap(),
            S3ObjectPin::new(
                &S3ObjectRequest::new(
                    &endpoint,
                    "bucket",
                    "secret-key",
                    "file.bin",
                    FileRole::Auxiliary,
                    None,
                    Some(&"a".repeat(64)),
                )
                .unwrap(),
                "v1",
                3,
            )
            .unwrap(),
        ] {
            assert_ne!(pin.source(), other.source());
        }
    }
}
