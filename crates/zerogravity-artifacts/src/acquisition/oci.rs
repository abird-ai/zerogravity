//! Bounded, explicitly selected raw GGUF layers from an OCI registry.
//! This is acquisition, never container execution or generic registry parity.
use super::http::{HttpAccept, HttpOptions, HttpRequest, HttpTransport, HttpsClient};
use super::{PullLimits, ResolvedFile, ResolvedPullPlan, http};
use crate::cache::paths::{validate_digest, validate_relative_path};
use crate::cache::{
    CacheStore, DigestProvenance, FileRole, Manifest, ManifestSource, SnapshotLease,
    TransferAdmission,
};
use crate::{Error, Result};
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};

const DOCKER_REGISTRY: &str = "https://registry-1.docker.io";
const DOCKER_BROKER: &str = "https://auth.docker.io";
const OCI_MANIFEST: &str = "application/vnd.oci.image.manifest.v1+json";
const DOCKER_MANIFEST: &str = "application/vnd.docker.distribution.manifest.v2+json";
const GGUF_LAYER: &str = "application/vnd.docker.ai.gguf.v3";
const MAX_DESCRIPTORS: usize = 4096;
const MAX_BROKER_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OciProfile {
    Anonymous,
    Bearer,
    DockerHubAnonymous,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OciReference {
    Tag(String),
    /// Canonical raw lowercase SHA256, without the algorithm prefix.
    ManifestSha256(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GgufLayer {
    /// Exactly one recognized raw GGUF layer must exist in the manifest.
    Unique,
    /// Select this canonical raw lowercase SHA256 among recognized GGUF layers.
    Sha256(String),
}

#[derive(Clone, Debug)]
pub struct OciRequest {
    origin: String,
    profile: OciProfile,
    repository: String,
    reference: OciReference,
    logical_path: String,
    layer: GgufLayer,
    contract: String,
}

impl OciRequest {
    pub fn new(
        origin: &str,
        profile: OciProfile,
        repository: &str,
        reference: OciReference,
        logical_path: &str,
        layer: GgufLayer,
    ) -> Result<Self> {
        let origin = normalize_origin(origin)?;
        validate_repository(repository)?;
        validate_relative_path(logical_path)?;
        match &reference {
            OciReference::Tag(tag)
                if tag.is_empty()
                    || tag.len() > 128
                    || !tag
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
                    || !tag.as_bytes()[0].is_ascii_alphanumeric() && tag.as_bytes()[0] != b'_' =>
            {
                return Err(Error::Malformed("invalid OCI tag".into()));
            }
            OciReference::ManifestSha256(digest) => validate_digest(digest)?,
            _ => {}
        }
        if let GgufLayer::Sha256(digest) = &layer {
            validate_digest(digest)?;
        }
        if profile == OciProfile::DockerHubAnonymous && origin != DOCKER_REGISTRY {
            return Err(Error::Unsupported(
                "Docker Hub profile requires its fixed registry origin".into(),
            ));
        }
        let contract = hash(
            &serde_json::to_vec(&(
                "zerogravity.oci.request.v1",
                &origin,
                profile,
                repository,
                &reference,
                logical_path,
                &layer,
            ))
            .map_err(|_| Error::Malformed("OCI request identity encoding failed".into()))?,
        );
        Ok(Self {
            origin,
            profile,
            repository: repository.to_owned(),
            reference,
            logical_path: logical_path.to_owned(),
            layer,
            contract,
        })
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn profile(&self) -> OciProfile {
        self.profile
    }
    pub fn repository(&self) -> &str {
        &self.repository
    }
    pub fn reference(&self) -> &OciReference {
        &self.reference
    }
    pub fn logical_path(&self) -> &str {
        &self.logical_path
    }
    pub fn layer(&self) -> &GgufLayer {
        &self.layer
    }

    fn source(&self, manifest: &str) -> Result<ManifestSource> {
        validate_digest(manifest)?;
        ManifestSource::new("oci", format!("request-{}", self.contract), manifest)
    }

    fn metadata_budget(&self, limits: &PullLimits) -> Result<(usize, usize)> {
        limits.validate()?;
        let broker = if self.profile == OciProfile::DockerHubAnonymous {
            if limits.max_pages < 2 {
                return Err(Error::Limit("OCI metadata request limit".into()));
            }
            (limits.max_metadata_bytes / 2).min(MAX_BROKER_BYTES)
        } else {
            0
        };
        let manifest = limits.max_metadata_bytes - broker;
        if manifest == 0 || self.profile == OciProfile::DockerHubAnonymous && broker == 0 {
            return Err(Error::Limit("OCI metadata byte limit".into()));
        }
        Ok((broker, manifest))
    }

    /// Metadata only. The supplied transport is also used for the returned
    /// plan's payloads; explicit bearer/broker credentials remain in that owner,
    /// never in the source identity or plan. Tag lookup occurs exactly once.
    pub fn resolve(
        &self,
        transport: &(impl HttpTransport + ?Sized),
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<ResolvedPullPlan> {
        let (_, maximum) = self.metadata_budget(limits)?;
        let _control = admit_metadata(limits, admission, cancel)?;
        let reference = match &self.reference {
            OciReference::Tag(tag) => tag.clone(),
            OciReference::ManifestSha256(digest) => format!("sha256:{digest}"),
        };
        let mut request = HttpRequest::new(
            format!(
                "{}/v2/{}/manifests/{reference}",
                self.origin, self.repository
            ),
            maximum as u64,
        );
        request.accept = HttpAccept::OciManifest;
        let response = transport.get(&request, cancel);
        check_cancel(cancel)?;
        let response = response.map_err(|_| Error::State("OCI manifest request failed".into()))?;
        if response.status != 200 {
            return Err(
                if self.profile == OciProfile::DockerHubAnonymous
                    && matches!(response.status, 401 | 403)
                {
                    Error::Unsupported("Docker Hub anonymous profile cannot read this private or denied repository".into())
                } else {
                    Error::State(format!("OCI manifest HTTP status {}", response.status))
                },
            );
        }
        if response
            .header("content-encoding")
            .is_some_and(|encoding| !encoding.eq_ignore_ascii_case("identity"))
            || response.header("content-range").is_some()
        {
            return Err(Error::Unsupported(
                "OCI manifest requires complete identity-encoded bytes".into(),
            ));
        }
        let content_type = response
            .header("content-type")
            .filter(|value| value.len() <= 256)
            .and_then(|value| value.split(';').next())
            .map(str::trim)
            .ok_or_else(|| {
                Error::Malformed("OCI manifest content type is missing or oversized".into())
            })?;
        if !matches!(content_type, OCI_MANIFEST | DOCKER_MANIFEST) {
            return Err(Error::Unsupported(
                "OCI index, platform list or unrecognized manifest type".into(),
            ));
        }
        let content_type = content_type.to_owned();
        let header_digest = response
            .header("docker-content-digest")
            .map(parse_digest)
            .transpose()?
            .map(str::to_owned);
        let declared = response
            .header("content-length")
            .map(parse_size)
            .transpose()?;
        if declared.is_some_and(|bytes| bytes > maximum as u64) {
            return Err(Error::Limit("OCI manifest declared byte limit".into()));
        }
        let raw = response.read_bounded(maximum, cancel)?;
        check_cancel(cancel)?;
        if declared.is_some_and(|bytes| bytes != raw.len() as u64) {
            return Err(Error::Malformed("OCI manifest length mismatch".into()));
        }
        // Hash the original received representation BEFORE parsing/selection.
        let manifest_digest = hash(&raw);
        if header_digest
            .as_ref()
            .is_some_and(|expected| expected != &manifest_digest)
            || matches!(&self.reference, OciReference::ManifestSha256(expected) if expected != &manifest_digest)
        {
            return Err(Error::Malformed("OCI manifest raw SHA256 mismatch".into()));
        }
        let manifest: ImageManifest = serde_json::from_slice(&raw)
            .map_err(|_| Error::Malformed("invalid bounded OCI image manifest".into()))?;
        if manifest.schema_version != 2 || manifest.media_type != content_type {
            return Err(Error::Malformed(
                "OCI manifest schema or media type mismatch".into(),
            ));
        }
        validate_descriptor(&manifest.config)?;
        let mut selected = None;
        for descriptor in &manifest.layers {
            check_cancel(cancel)?;
            let digest = validate_descriptor(descriptor)?;
            if descriptor.media_type != GGUF_LAYER {
                continue;
            }
            if let GgufLayer::Sha256(expected) = &self.layer
                && expected != digest
            {
                continue;
            }
            if selected.replace((descriptor, digest)).is_some() {
                return Err(Error::Unsupported(
                    "ambiguous OCI GGUF layer selection".into(),
                ));
            }
        }
        let (descriptor, digest) = selected
            .ok_or_else(|| Error::Unsupported("requested raw OCI GGUF layer is absent".into()))?;
        let file = ResolvedFile::new(
            &self.logical_path,
            format!(
                "{}/v2/{}/blobs/sha256:{digest}",
                self.origin, self.repository
            ),
            FileRole::Weights,
            descriptor.size,
            Some(digest.to_owned()),
            format!("oci:{}:{manifest_digest}:sha256:{digest}", self.contract),
        )?;
        let plan = ResolvedPullPlan::new(self.source(&manifest_digest)?, vec![file], limits)?;
        check_cancel(cancel)?;
        Ok(plan)
    }

    /// Verify cached data against this exact declared request and source labels.
    /// The original OCI JSON is not retained, so this does not independently
    /// authenticate descriptor membership, remote tag freshness or signatures.
    pub fn open_cached(
        &self,
        store: &CacheStore,
        snapshot: &str,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<SnapshotLease> {
        limits.validate()?;
        check_cancel(cancel)?;
        let lease = if validate_digest(snapshot).is_ok() {
            store.open_snapshot_cancellable(snapshot, cancel)?
        } else {
            store.resolve_ref_cancellable(snapshot, cancel)?
        };
        let manifest = lease.manifest();
        let revision = manifest.source().revision();
        if manifest.source() != &self.source(revision)?
            || manifest.entries().len() != 1
            || matches!(&self.reference, OciReference::ManifestSha256(expected) if expected != revision)
        {
            return Err(Error::Malformed(
                "cached OCI request/source mismatch".into(),
            ));
        }
        let entry = &manifest.entries()[0];
        if entry.path() != self.logical_path
            || entry.role() != FileRole::Weights
            || entry.digest_provenance() != DigestProvenance::TrustedSha256
            || matches!(&self.layer, GgufLayer::Sha256(expected) if expected != entry.digest())
        {
            return Err(Error::Malformed(
                "cached OCI layer contract mismatch".into(),
            ));
        }
        Manifest::from_json(&manifest.canonical_bytes()?, &limits.cache)?;
        let mut prefix = [0u8; 128];
        let count = entry.bytes().min(prefix.len() as u64) as usize;
        lease
            .file(&self.logical_path)?
            .read_exact_at(&mut prefix[..count], 0)?;
        check_cancel(cancel)?;
        super::executor::reject_pointer(&prefix[..count])?;
        Ok(lease)
    }
}

/// Obtain one anonymous pull token from the fixed Docker Hub broker, using a
/// credential-free broker transport. No challenge realm, Basic credentials,
/// environment mutation, refresh flow or private registry login is inferred.
/// Broker and manifest budgets are disjoint portions of the same caller limit.
pub fn docker_hub_client(
    request: &OciRequest,
    broker: &(impl HttpTransport + ?Sized),
    options: HttpOptions,
    limits: &PullLimits,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<HttpsClient> {
    if request.profile != OciProfile::DockerHubAnonymous || request.origin != DOCKER_REGISTRY {
        return Err(Error::Unsupported(
            "Docker Hub broker requires its explicit fixed profile".into(),
        ));
    }
    let (maximum, _) = request.metadata_budget(limits)?;
    let _control = admit_metadata(limits, admission, cancel)?;
    let mut url = http::parse_url(DOCKER_BROKER, false)?;
    url.set_path("/token");
    url.query_pairs_mut()
        .append_pair("service", "registry.docker.io")
        .append_pair("scope", &format!("repository:{}:pull", request.repository));
    let mut call = HttpRequest::new(url.as_str(), maximum as u64);
    call.accept = HttpAccept::Json;
    let response = broker.get(&call, cancel);
    check_cancel(cancel)?;
    let response = response.map_err(|_| Error::State("Docker Hub broker request failed".into()))?;
    if response.status != 200 {
        return Err(Error::State(
            "Docker Hub anonymous broker refused pull access".into(),
        ));
    }
    let bytes = response.read_bounded(maximum, cancel)?;
    let token: TokenResponse = serde_json::from_slice(&bytes)
        .map_err(|_| Error::Malformed("invalid bounded Docker Hub token response".into()))?;
    if token.expires_in == Some(0)
        || matches!((&token.token, &token.access_token), (Some(a), Some(b)) if a != b)
    {
        return Err(Error::Malformed(
            "inconsistent or expired Docker Hub token response".into(),
        ));
    }
    let token = token
        .token
        .or(token.access_token)
        .ok_or_else(|| Error::Malformed("Docker Hub token is missing".into()))?;
    check_cancel(cancel)?;
    HttpsClient::with_bearer(&request.origin, &token, options)
}

#[derive(Deserialize)]
struct TokenResponse {
    token: Option<String>,
    access_token: Option<String>,
    expires_in: Option<u64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImageManifest {
    schema_version: u64,
    media_type: String,
    config: Descriptor,
    #[serde(deserialize_with = "deserialize_layers")]
    layers: Vec<Descriptor>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Descriptor {
    media_type: String,
    digest: String,
    size: u64,
    #[serde(default, deserialize_with = "deserialize_external_urls")]
    urls: bool,
}

fn deserialize_external_urls<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<bool, D::Error> {
    struct Urls;
    impl<'de> Visitor<'de> for Urls {
        type Value = bool;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("bounded OCI descriptor URLs")
        }
        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> std::result::Result<bool, A::Error> {
            let mut count = 0usize;
            while sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
                count += 1;
                if count > MAX_DESCRIPTORS {
                    return Err(serde::de::Error::custom("OCI external URL count limit"));
                }
            }
            Ok(count != 0)
        }
    }
    deserializer.deserialize_seq(Urls)
}

fn deserialize_layers<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<Descriptor>, D::Error> {
    struct Layers;
    impl<'de> Visitor<'de> for Layers {
        type Value = Vec<Descriptor>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("bounded OCI layer descriptors")
        }
        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut layers = Vec::new();
            while let Some(layer) = sequence.next_element()? {
                if layers.len() == MAX_DESCRIPTORS {
                    return Err(serde::de::Error::custom("OCI descriptor count limit"));
                }
                layers.push(layer);
            }
            Ok(layers)
        }
    }
    deserializer.deserialize_seq(Layers)
}

fn validate_descriptor(descriptor: &Descriptor) -> Result<&str> {
    if descriptor.media_type.is_empty() || descriptor.media_type.len() > 256 {
        return Err(Error::Malformed("invalid OCI descriptor media type".into()));
    }
    if descriptor.urls {
        return Err(Error::Unsupported(
            "external OCI descriptor URLs are unsupported".into(),
        ));
    }
    parse_digest(&descriptor.digest)
}

fn parse_digest(value: &str) -> Result<&str> {
    let digest = value
        .strip_prefix("sha256:")
        .ok_or_else(|| Error::Unsupported("OCI requires SHA256 descriptors".into()))?;
    validate_digest(digest)?;
    Ok(digest)
}

fn parse_size(value: &str) -> Result<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(Error::Malformed("invalid OCI content length".into()));
    }
    value
        .parse()
        .map_err(|_| Error::Malformed("OCI content length overflow".into()))
}

fn normalize_origin(value: &str) -> Result<String> {
    if value.len() > 4096
        || value.trim() != value
        || value.chars().any(char::is_control)
        || value.contains('\\')
    {
        return Err(Error::Malformed("invalid OCI origin".into()));
    }
    let (_, remainder) = value
        .split_once("://")
        .ok_or_else(|| Error::Malformed("invalid OCI origin".into()))?;
    let authority_end = remainder.find(['/', '?', '#']).unwrap_or(remainder.len());
    if remainder[..authority_end].contains('@') || !matches!(&remainder[authority_end..], "" | "/")
    {
        return Err(Error::Malformed("OCI requires a strict bare origin".into()));
    }
    let diagnostic = cfg!(feature = "diagnostic-acquisition")
        && value
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http:"));
    let parsed = http::parse_url(value, diagnostic)?;
    Ok(parsed.origin().ascii_serialization())
}

fn validate_repository(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 255 {
        return Err(Error::Malformed("invalid OCI repository length".into()));
    }
    for component in value.split('/') {
        let bytes = component.as_bytes();
        let mut offset = 0;
        while offset < bytes.len() {
            let start = offset;
            while offset < bytes.len()
                && (bytes[offset].is_ascii_lowercase() || bytes[offset].is_ascii_digit())
            {
                offset += 1;
            }
            if offset == start {
                return Err(Error::Malformed("invalid OCI repository component".into()));
            }
            if offset == bytes.len() {
                break;
            }
            match bytes[offset] {
                b'.' => offset += 1,
                b'_' => {
                    offset += 1;
                    if bytes.get(offset) == Some(&b'_') {
                        offset += 1;
                    }
                }
                b'-' => {
                    while bytes.get(offset) == Some(&b'-') {
                        offset += 1;
                    }
                }
                _ => return Err(Error::Malformed("invalid OCI repository separator".into())),
            }
            if offset == bytes.len() {
                return Err(Error::Malformed("trailing OCI repository separator".into()));
            }
        }
        if bytes.is_empty() {
            return Err(Error::Malformed("empty OCI repository component".into()));
        }
    }
    Ok(())
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        return Err(Error::State("OCI acquisition canceled".into()));
    }
    Ok(())
}
fn admit_metadata(
    limits: &PullLimits,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<GlobalResourceLease> {
    limits.validate()?;
    check_cancel(cancel)?;
    let bytes = (limits.max_metadata_bytes as u64)
        .checked_mul(8)
        .and_then(|value| value.checked_add(16 * 1024 * 1024))
        .ok_or_else(|| Error::Limit("OCI metadata admission overflow".into()))?;
    admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes,
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(origin: &str, repository: &str) -> Result<OciRequest> {
        OciRequest::new(
            origin,
            OciProfile::Anonymous,
            repository,
            OciReference::Tag("stable".into()),
            "model.gguf",
            GgufLayer::Unique,
        )
    }

    #[test]
    fn bounded_repository_origin_reference_and_selector_are_pure() {
        for repo in ["library/model", "a/b__c", "a-b/c_d", "a.b/c---d", "a"] {
            assert!(
                request("HTTPS://REGISTRY.EXAMPLE:443/", repo).is_ok(),
                "{repo}"
            );
        }
        for repo in [
            "", "/a", "a/", "a//b", "A/b", "a..b", "a___b", "a.-b", "a-", ".",
        ] {
            assert!(request("https://registry.example", repo).is_err(), "{repo}");
        }
        for origin in [
            "https://user@registry.example",
            "https://@registry.example",
            "https://registry.example/.",
            "https://registry.example/..",
            "https://registry.example/path",
            "https://registry.example?secret=x",
            "https://registry.example#x",
            " https://registry.example",
            "ftp://registry.example",
            "https://registry.example\\path",
        ] {
            assert!(request(origin, "a/b").is_err(), "{origin}");
        }
        for tag in ["", ".bad", "-bad", "space value", "../bad", "a/b"] {
            assert!(
                OciRequest::new(
                    "https://registry.example",
                    OciProfile::Anonymous,
                    "a/b",
                    OciReference::Tag(tag.into()),
                    "model.gguf",
                    GgufLayer::Unique
                )
                .is_err()
            );
        }
        assert!(
            OciRequest::new(
                "https://registry.example",
                OciProfile::DockerHubAnonymous,
                "a/b",
                OciReference::Tag("stable".into()),
                "model.gguf",
                GgufLayer::Unique
            )
            .is_err()
        );
    }

    #[test]
    fn immutable_source_contract_binds_every_declared_selection_field() {
        let base = request("https://registry.example", "a/b").unwrap();
        assert_eq!(
            base.contract,
            request("HTTPS://REGISTRY.EXAMPLE:443/", "a/b")
                .unwrap()
                .contract
        );
        let manifest = "a".repeat(64);
        let variants = [
            OciRequest::new(
                "https://other.example",
                OciProfile::Anonymous,
                "a/b",
                OciReference::Tag("stable".into()),
                "model.gguf",
                GgufLayer::Unique,
            ),
            OciRequest::new(
                "https://registry.example",
                OciProfile::Bearer,
                "a/b",
                OciReference::Tag("stable".into()),
                "model.gguf",
                GgufLayer::Unique,
            ),
            OciRequest::new(
                "https://registry.example",
                OciProfile::Anonymous,
                "a/c",
                OciReference::Tag("stable".into()),
                "model.gguf",
                GgufLayer::Unique,
            ),
            OciRequest::new(
                "https://registry.example",
                OciProfile::Anonymous,
                "a/b",
                OciReference::Tag("other".into()),
                "model.gguf",
                GgufLayer::Unique,
            ),
            OciRequest::new(
                "https://registry.example",
                OciProfile::Anonymous,
                "a/b",
                OciReference::ManifestSha256(manifest.clone()),
                "model.gguf",
                GgufLayer::Unique,
            ),
            OciRequest::new(
                "https://registry.example",
                OciProfile::Anonymous,
                "a/b",
                OciReference::Tag("stable".into()),
                "other.gguf",
                GgufLayer::Unique,
            ),
            OciRequest::new(
                "https://registry.example",
                OciProfile::Anonymous,
                "a/b",
                OciReference::Tag("stable".into()),
                "model.gguf",
                GgufLayer::Sha256("b".repeat(64)),
            ),
        ];
        for variant in variants {
            assert_ne!(
                base.source(&manifest).unwrap(),
                variant.unwrap().source(&manifest).unwrap()
            );
        }
    }

    #[test]
    fn descriptor_json_enforces_exact_sizes_duplicates_counts_and_no_external_urls() {
        let descriptor = json!({"mediaType": GGUF_LAYER, "digest": format!("sha256:{}", "a".repeat(64)), "size": 1});
        let raw = json!({"schemaVersion":2,"mediaType":OCI_MANIFEST,"config":descriptor,"layers":[descriptor]}).to_string();
        assert!(serde_json::from_str::<ImageManifest>(&raw).is_ok());
        let fractional = raw.replace("\"size\":1", "\"size\":1.0");
        assert!(serde_json::from_str::<ImageManifest>(&fractional).is_err());
        let duplicate = raw.replace("\"size\":1", "\"size\":1,\"size\":1");
        assert!(serde_json::from_str::<ImageManifest>(&duplicate).is_err());
        let mut manifest: serde_json::Value = serde_json::from_str(&raw).unwrap();
        manifest["layers"] = json!(vec![descriptor; MAX_DESCRIPTORS + 1]);
        assert!(serde_json::from_value::<ImageManifest>(manifest).is_err());
        let mut external: Descriptor = serde_json::from_value(json!({"mediaType":GGUF_LAYER,
            "digest":format!("sha256:{}", "a".repeat(64)), "size":1,
            "urls":["https://untrusted.example/?secret=not_followed"]}))
        .unwrap();
        assert!(matches!(
            validate_descriptor(&external),
            Err(Error::Unsupported(_))
        ));
        external.urls = false;
        assert!(validate_descriptor(&external).is_ok());
    }

    #[test]
    fn docker_hub_broker_and_manifest_split_one_aggregate_metadata_budget() {
        let request = OciRequest::new(
            DOCKER_REGISTRY,
            OciProfile::DockerHubAnonymous,
            "ai/toy",
            OciReference::Tag("stable".into()),
            "model.gguf",
            GgufLayer::Unique,
        )
        .unwrap();
        for maximum in [2, 1000, 64 * 1024, 8 * 1024 * 1024] {
            let limits = PullLimits {
                max_metadata_bytes: maximum,
                ..Default::default()
            };
            let (broker, manifest) = request.metadata_budget(&limits).unwrap();
            assert!(broker > 0 && broker <= MAX_BROKER_BYTES && manifest > 0);
            assert_eq!(broker + manifest, maximum);
        }
        assert!(
            request
                .metadata_budget(&PullLimits {
                    max_pages: 1,
                    ..Default::default()
                })
                .is_err()
        );
    }
}
