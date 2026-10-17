//! One synchronous, verified HTTPS owner. Callers admit metadata/staging first.
//! Provider URLs and credentials are deliberately absent from diagnostic errors.

use crate::{Error, Result};
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use url::Url;

use super::resume::ResumeRequest;
#[path = "s3_signing.rs"]
mod s3_signing;
pub use s3_signing::S3Credentials;
#[path = "azure_auth.rs"]
mod azure_auth;
pub use azure_auth::AzureSharedKey;

/// Provider-constructed immutable request target. Public custom transports may
/// explicitly implement the capability; arbitrary raw text is never accepted
/// through a public constructor or a permissive HTTP switch.
#[derive(Clone)]
pub struct HttpTarget {
    origin: String,
    profile: TargetProfile,
    uri: ureq::http::Uri,
    condition: Option<ReadCondition>,
}
#[derive(Clone)]
enum ReadCondition {
    IfMatch(String),
}
#[derive(Clone)]
enum TargetProfile {
    Direct,
    S3 {
        region: String,
    },
    Azure {
        kind: AzureRequestKind,
    },
    XetHub {
        budget: Arc<super::xet::budget::Budget>,
        mode: XetHubMode,
    },
    XetCas {
        grant: Arc<XetGrant>,
        budget: Arc<super::xet::budget::Budget>,
    },
    XetRanges {
        ranges: Vec<super::xet::reconstruction::ByteInterval>,
        budget: Arc<super::xet::budget::Budget>,
    },
}
/// Ephemeral read grant owned by this HTTP module. In particular, there is no
/// secret getter for a custom transport: opaque profiles require this owner or
/// a wrapper delegating to it, otherwise get_target safely stays Unsupported.
pub(super) struct XetGrant {
    origin: String,
    token: String,
    expires: u64,
    repository: String,
    commit: String,
}
impl XetGrant {
    pub(super) fn new(
        origin: &str,
        token: String,
        expires: u64,
        repository: &str,
        commit: &str,
        diagnostic: bool,
    ) -> Result<Arc<Self>> {
        super::huggingface::validate_repo(repository)?;
        let parsed = parse_url(origin, diagnostic)?;
        validate_origin_url(&parsed)?;
        if parsed.origin().ascii_serialization() != origin
            || (!diagnostic && origin != "https://cas-server.xethub.hf.co")
            || token.is_empty()
            || token.len() > 64000
            || !token.bytes().all(|byte| byte.is_ascii_graphic())
            || expires == 0
            || commit.len() != 40
            || !commit
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::Malformed("invalid scoped Xet read grant".into()));
        }
        if diagnostic && !cfg!(any(test, feature = "diagnostic-acquisition")) {
            return Err(Error::Unsupported(
                "Xet diagnostic endpoint is unavailable".into(),
            ));
        }
        Ok(Arc::new(Self {
            origin: origin.to_owned(),
            token,
            expires,
            repository: repository.to_owned(),
            commit: commit.to_owned(),
        }))
    }
    pub(super) fn expires(&self) -> u64 {
        self.expires
    }
}
#[derive(Clone, Copy)]
pub(super) enum XetHubMode {
    Metadata,
    Discovery,
    OrdinaryPayload,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AzureRequestKind {
    Content,
    List,
    Version,
    Snapshot,
}
impl std::fmt::Debug for HttpTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpTarget").finish_non_exhaustive()
    }
}
impl HttpTarget {
    pub(super) fn xet_hub(
        url: &str,
        budget: Arc<super::xet::budget::Budget>,
        mode: XetHubMode,
    ) -> Result<Self> {
        let parsed = parse_provider_origin(url)?;
        budget.check_hub_origin(&parsed.origin().ascii_serialization())?;
        Ok(Self {
            condition: None,
            origin: parsed.origin().ascii_serialization(),
            profile: TargetProfile::XetHub { budget, mode },
            uri: parsed
                .as_str()
                .parse()
                .map_err(|_| Error::Malformed("invalid Xet Hub target".into()))?,
        })
    }
    pub(super) fn xet_cas(
        url: &str,
        grant: Arc<XetGrant>,
        budget: Arc<super::xet::budget::Budget>,
    ) -> Result<Self> {
        let parsed = parse_provider_origin(url)?;
        if parsed.origin().ascii_serialization() != grant.origin {
            return Err(Error::State("Xet CAS grant origin differs".into()));
        }
        Ok(Self {
            condition: None,
            origin: grant.origin.clone(),
            profile: TargetProfile::XetCas { grant, budget },
            uri: parsed
                .as_str()
                .parse()
                .map_err(|_| Error::Malformed("invalid Xet CAS target".into()))?,
        })
    }
    pub(super) fn xet_ranges(
        url: &str,
        ranges: Vec<super::xet::reconstruction::ByteInterval>,
        budget: Arc<super::xet::budget::Budget>,
    ) -> Result<Self> {
        let parsed = parse_provider_origin(url)?;
        if ranges.is_empty()
            || url.len() > 16 * 1024
            || !url
                .bytes()
                .all(|b| b.is_ascii_graphic() && !b"#\\".contains(&b))
        {
            return Err(Error::Malformed("invalid Xet signed target".into()));
        }
        Ok(Self {
            condition: None,
            origin: parsed.origin().ascii_serialization(),
            profile: TargetProfile::XetRanges { ranges, budget },
            uri: url
                .parse()
                .map_err(|_| Error::Malformed("invalid Xet signed URI".into()))?,
        })
    }
    fn xet_budget(&self) -> Option<&Arc<super::xet::budget::Budget>> {
        match &self.profile {
            TargetProfile::XetHub { budget, .. }
            | TargetProfile::XetCas { budget, .. }
            | TargetProfile::XetRanges { budget, .. } => Some(budget),
            _ => None,
        }
    }
    fn xet_metadata(&self) -> bool {
        matches!(
            self.profile,
            TargetProfile::XetHub {
                mode: XetHubMode::Metadata | XetHubMode::Discovery,
                ..
            } | TargetProfile::XetCas { .. }
        )
    }
    fn xet_isolated(&self) -> bool {
        matches!(
            self.profile,
            TargetProfile::XetCas { .. } | TargetProfile::XetRanges { .. }
        )
    }
    fn xet_observation(&self) -> bool {
        matches!(
            self.profile,
            TargetProfile::XetHub {
                mode: XetHubMode::Discovery,
                ..
            }
        )
    }
    fn xet_ordinary(&self) -> bool {
        matches!(
            self.profile,
            TargetProfile::XetHub {
                mode: XetHubMode::OrdinaryPayload,
                ..
            }
        )
    }
    /// Custom wrappers executing an opaque Xet target must call this immediately
    /// before EVERY actual attempt, including internal retries. This exposes no
    /// authentication capability or profile/secret getter. A wrapper delegating
    /// to HttpsClient leaves accounting to that owner, without double charging.
    pub fn begin_attempt(&self, cancel: &AtomicBool) -> Result<()> {
        self.xet_budget()
            .ok_or_else(|| Error::Unsupported("target has no Xet work capability".into()))?
            .begin_attempt(cancel)
    }
    /// Charge actual decoded HTTP body bytes, including discarded/refetched data.
    /// Metadata bodies also charge the separate cumulative control bound.
    pub fn charge_body(&self, bytes: u64) -> Result<()> {
        self.xet_budget()
            .ok_or_else(|| Error::Unsupported("target has no Xet work capability".into()))?
            .charge_body(bytes, self.xet_metadata())
    }
    /// Charge received response-header/control bytes, including retry responses.
    pub fn charge_metadata(&self, bytes: u64) -> Result<()> {
        self.xet_budget()
            .ok_or_else(|| Error::Unsupported("target has no Xet work capability".into()))?
            .charge_metadata(bytes)
    }
    /// Read allowance includes at most one overflow sentinel byte. The caller
    /// still charges actual returned bytes and refuses an exhausted budget.
    pub fn body_read_limit(&self, requested: usize) -> Result<usize> {
        self.xet_budget()
            .ok_or_else(|| Error::Unsupported("target has no Xet work capability".into()))?
            .read_limit(requested, self.xet_metadata())
    }
    // Only the pinned provider may construct this no-redirect target. Generic
    // callers cannot relabel arbitrary URLs with a privileged credential profile.
    pub(super) fn direct(url: &str) -> Result<Self> {
        let parsed = parse_provider_origin(url)?;
        let uri = parsed
            .as_str()
            .parse()
            .map_err(|_| Error::Malformed("invalid direct provider target".into()))?;
        Ok(Self {
            condition: None,
            origin: parsed.origin().ascii_serialization(),
            profile: TargetProfile::Direct,
            uri,
        })
    }
    pub(super) fn azure(
        origin: &str,
        path_and_query: &str,
        kind: AzureRequestKind,
    ) -> Result<Self> {
        let parsed = parse_provider_origin(origin)?;
        validate_origin_url(&parsed)?;
        if parsed.origin().ascii_serialization() != origin
            || !path_and_query.starts_with('/')
            || path_and_query.len() > 14 * 1024
            || !path_and_query
                .bytes()
                .all(|byte| byte.is_ascii_graphic() && !b"#\\".contains(&byte))
        {
            return Err(Error::Malformed("invalid canonical Azure target".into()));
        }
        let uri = format!("{origin}{path_and_query}")
            .parse()
            .map_err(|_| Error::Malformed("invalid canonical Azure URI".into()))?;
        Ok(Self {
            condition: None,
            origin: origin.to_owned(),
            profile: TargetProfile::Azure { kind },
            uri,
        })
    }
    fn azure_kind(&self) -> Option<AzureRequestKind> {
        match self.profile {
            TargetProfile::Azure { kind } => Some(kind),
            _ => None,
        }
    }
    pub(super) fn s3(origin: &str, region: &str, path_and_query: &str) -> Result<Self> {
        let parsed = parse_provider_origin(origin)?;
        validate_origin_url(&parsed)?;
        if parsed.origin().ascii_serialization() != origin
            || !path_and_query.starts_with('/')
            || path_and_query.len() > 14 * 1024
            || !path_and_query
                .bytes()
                .all(|b| b.is_ascii_graphic() && !b"#\\".contains(&b))
        {
            return Err(Error::Malformed("invalid canonical S3 target".into()));
        }
        let uri: ureq::http::Uri = format!("{origin}{path_and_query}")
            .parse()
            .map_err(|_| Error::Malformed("invalid canonical S3 URI".into()))?;
        Ok(Self {
            condition: None,
            origin: origin.to_owned(),
            profile: TargetProfile::S3 {
                region: region.to_owned(),
            },
            uri,
        })
    }
    fn s3_region(&self) -> Result<&str> {
        match &self.profile {
            TargetProfile::S3 { region } => Ok(region),
            _ => Err(Error::Unsupported("SigV4 requires an S3 target".into())),
        }
    }
    /// A single strong If-Match condition, if this provider target requires it.
    /// Explicit custom target transports must emit it on every request, including
    /// ranges/retries, and must never downgrade to an unconditional read.
    pub fn if_match(&self) -> Option<&str> {
        self.condition.as_ref().map(|condition| match condition {
            ReadCondition::IfMatch(value) => value.as_str(),
        })
    }
    pub(super) fn with_if_match(mut self, value: &str) -> Result<Self> {
        super::resume::strong_etag(value)?;
        if self.uri.query().is_some()
            || !matches!(
                self.profile,
                TargetProfile::S3 { .. }
                    | TargetProfile::Azure {
                        kind: AzureRequestKind::Content
                    }
            )
        {
            return Err(Error::Unsupported(
                "conditional reads require a current object target".into(),
            ));
        }
        self.condition = Some(ReadCondition::IfMatch(value.to_owned()));
        Ok(self)
    }
    pub(super) fn url(&self) -> String {
        self.uri.to_string()
    }
    pub(super) fn metadata_bytes(&self) -> usize {
        self.origin.len()
            + match &self.profile {
                TargetProfile::S3 { region } => region.len(),
                TargetProfile::Azure { .. } => 8,
                TargetProfile::Direct => 0,
                TargetProfile::XetCas { grant, .. } => {
                    grant.token.len()
                        + grant.origin.len()
                        + grant.repository.len()
                        + grant.commit.len()
                        + 256
                }
                TargetProfile::XetHub { .. } | TargetProfile::XetRanges { .. } => 256,
            }
            + self.uri.to_string().len()
            + self.if_match().map_or(0, str::len)
            + 128
    }
}

#[derive(Clone, Copy, Debug)]
pub enum HttpAccept {
    Json,
    GitHubJson,
    OciManifest,
    Binary,
}

pub struct HttpRequest {
    pub url: String,
    pub accept: HttpAccept,
    pub range: Option<ResumeRequest>,
    pub max_body_bytes: u64,
}

impl HttpRequest {
    pub fn new(url: impl Into<String>, max_body_bytes: u64) -> Self {
        Self {
            url: url.into(),
            accept: HttpAccept::Binary,
            range: None,
            max_body_bytes,
        }
    }
}

/// `body` yields decoded payload bytes. Custom transports must report framing
/// failures as read errors, never successful EOF. The pinned ureq decoder also
/// accepts a zero-chunk line followed by FIN without complete trailer framing;
/// payload completion here does not assert independent trailer validation, and
/// trailer metadata never supplies digest or source authority.
pub struct HttpResponse {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    pub body: Box<dyn Read + Send>,
}

impl HttpResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    /// Bounded metadata only; model payloads stream into the cache staging owner.
    pub fn read_bounded(mut self, max: usize, cancel: &AtomicBool) -> Result<Vec<u8>> {
        if !(200..300).contains(&self.status) {
            return Err(Error::State(format!(
                "HTTP metadata status {}",
                self.status
            )));
        }
        let mut bytes = Vec::new();
        let mut chunk = [0_u8; 8192];
        loop {
            check_cancel(cancel)?;
            let room = max.saturating_sub(bytes.len()).saturating_add(1);
            let read = self
                .body
                .read(&mut chunk[..room.min(8192)])
                .map_err(|_| Error::State("HTTP metadata read failed".into()))?;
            check_cancel(cancel)?;
            if read == 0 {
                return Ok(bytes);
            }
            if read > max.saturating_sub(bytes.len()) {
                return Err(Error::Limit("HTTP metadata byte limit".into()));
            }
            bytes
                .try_reserve(read)
                .map_err(|_| Error::Limit("HTTP metadata allocation refused".into()))?;
            bytes.extend_from_slice(&chunk[..read]);
        }
    }
}

pub trait HttpTransport: Send + Sync {
    fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse>;
    /// Preserve the provider-built target exactly, never follow redirects and
    /// retain framing/cancellation guarantees. Legacy transports safely refuse.
    fn get_target(
        &self,
        _request: &HttpRequest,
        _target: &HttpTarget,
        _cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        Err(Error::Unsupported(
            "canonical provider request capability is unavailable".into(),
        ))
    }
    /// Body-free counterpart using the same canonical target capability.
    fn head_target(
        &self,
        _request: &HttpRequest,
        _target: &HttpTarget,
        _cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        Err(Error::Unsupported(
            "canonical provider HEAD capability is unavailable".into(),
        ))
    }

    /// A version-precondition URL must reach the selected endpoint unchanged.
    /// Legacy custom transports explicitly refuse until they can guarantee that
    /// redirects are neither followed nor returned as a successful object body.
    fn get_without_redirects(
        &self,
        _request: &HttpRequest,
        _cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        Err(Error::Unsupported(
            "HTTP direct-response capability is unavailable".into(),
        ))
    }
    /// Metadata only. Older custom transports safely refuse probes until they
    /// implement HEAD; no GET fallback may transfer an unadmitted object body.
    fn head(&self, _request: &HttpRequest, _cancel: &AtomicBool) -> Result<HttpResponse> {
        Err(Error::Unsupported(
            "HTTP HEAD metadata is unavailable".into(),
        ))
    }
}

pub(super) fn probe_length_with_cap(
    transport: &(impl HttpTransport + ?Sized),
    url: String,
    maximum: u64,
    probes: &mut usize,
    cancel: &AtomicBool,
    expected_commit: Option<&str>,
    cap: Option<u64>,
) -> Result<(super::ByteLength, Option<String>)> {
    check_cancel(cancel)?;
    if *probes == 0 {
        return Err(Error::Limit("object length probe limit".into()));
    }
    *probes -= 1;
    let response = transport.head(&HttpRequest::new(url, maximum), cancel);
    check_cancel(cancel)?;
    let response = response.map_err(redact_head_error)?;
    let declared = response
        .header("content-length")
        .map(parse_u64)
        .transpose()?;
    let transfer = response.header("transfer-encoding");
    if response.header("content-range").is_some()
        || transfer.is_some_and(|value| {
            declared.is_some() || cap.is_none() || !value.eq_ignore_ascii_case("chunked")
        })
    {
        return Err(Error::Malformed(
            "ambiguous object length probe framing".into(),
        ));
    }
    if response
        .header("content-encoding")
        .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
    {
        return Err(Error::Unsupported(
            "length probe requires identity encoding".into(),
        ));
    }
    let linked = response
        .header("x-linked-size")
        .map(parse_u64)
        .transpose()?;
    if response.status == 200 && linked.is_some() && linked != declared {
        return Err(Error::Malformed("object length metadata conflicts".into()));
    }
    if let (Some(expected), Some(actual)) = (expected_commit, response.header("x-repo-commit"))
        && !actual.eq_ignore_ascii_case(expected)
    {
        return Err(Error::State(
            "object length probe contradicts pinned commit".into(),
        ));
    }
    let validator = response
        .header("etag")
        .filter(|value| super::resume::strong_etag(value).is_ok())
        .map(str::to_string);
    // A405/501 describes an error representation. Its length and ETag are not
    // authority for the pinned payload. All other status failures still refuse.
    let length = match (response.status, declared) {
        (200, Some(bytes)) => {
            if bytes > maximum {
                return Err(Error::Limit(
                    "object length exceeds planned file limit".into(),
                ));
            }
            super::ByteLength::Exact(bytes)
        }
        (200, None) | (405 | 501, _) if cap.is_some() && linked.is_none() => {
            let cap = cap
                .filter(|cap| *cap != 0 && *cap <= maximum)
                .ok_or_else(|| Error::Limit("invalid unknown object cap".into()))?;
            super::ByteLength::AtMost(cap)
        }
        (200, None) => return Err(Error::Unsupported("exact object length is required".into())),
        _ => {
            return Err(Error::Unsupported(
                "object length probe requires HTTP200".into(),
            ));
        }
    };
    // In particular, never inspect response.body: dry-run remains payload-free.
    Ok((
        length,
        if response.status == 200 {
            validator
        } else {
            None
        },
    ))
}

pub(super) fn redact_head_error(error: Error) -> Error {
    match error {
        Error::Limit(_) => Error::Limit("object length probe exceeded a bound".into()),
        Error::Malformed(_) => Error::Malformed("invalid object length probe metadata".into()),
        Error::State(_) | Error::Io(_) => {
            Error::State("object length probe failed or canceled".into())
        }
        _ => Error::Unsupported("exact object length metadata is unavailable".into()),
    }
}

/// Body-free validation for one explicitly opted-in current-object observation.
pub(super) fn conditional_metadata(
    response: &HttpResponse,
    limits: &super::PullLimits,
    binding: super::ResponseBinding,
) -> Result<(u64, String)> {
    if response.status != 200 {
        return Err(Error::State(format!(
            "conditional HEAD status {}",
            response.status
        )));
    }
    let header_bytes = response
        .headers
        .iter()
        .try_fold(0usize, |sum, (key, value)| {
            sum.checked_add(key.len())
                .and_then(|n| n.checked_add(value.len()))
                .and_then(|n| n.checked_add(32))
        })
        .ok_or_else(|| Error::Limit("conditional metadata overflow".into()))?;
    if header_bytes > limits.max_metadata_bytes {
        return Err(Error::Limit("conditional metadata header bound".into()));
    }
    validate_response_binding(Some(binding), response)?;
    if response.header("content-range").is_some()
        || response.header("transfer-encoding").is_some()
        || response
            .header("content-encoding")
            .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        return Err(Error::Unsupported(
            "conditional HEAD requires complete identity metadata".into(),
        ));
    }
    let bytes = parse_u64(
        response
            .header("content-length")
            .ok_or_else(|| Error::Malformed("conditional HEAD length is absent".into()))?,
    )?;
    if bytes > limits.cache.max_file_bytes || bytes > limits.cache.max_snapshot_bytes {
        return Err(Error::Limit("conditional object byte bound".into()));
    }
    let etag = response
        .header("etag")
        .ok_or_else(|| Error::Malformed("conditional HEAD validator is absent".into()))?;
    super::resume::strong_etag(etag)?;
    Ok((bytes, etag.to_owned()))
}

pub(super) fn validate_validator(expected: Option<&str>, response: &HttpResponse) -> Result<()> {
    if expected.is_some_and(|validator| response.header("etag") != Some(validator)) {
        return Err(Error::State(
            "HTTP object validator changed after length probe".into(),
        ));
    }
    Ok(())
}

/// JSON GCS generation/precondition queries bind the request. Its version
/// response headers are optional in the documented JSON protocol (and official
/// client); contradictory or noncanonical present headers always refuse.
pub(super) fn validate_response_binding(
    binding: Option<super::ResponseBinding>,
    response: &HttpResponse,
) -> Result<()> {
    use sha2::Digest;
    match binding {
        None => Ok(()),
        Some(
            super::ResponseBinding::AzureVersion { .. }
            | super::ResponseBinding::AzureSnapshot
            | super::ResponseBinding::AzureContent,
        ) => {
            if response.header("x-ms-blob-type") != Some("BlockBlob")
                || response.header("x-ms-structured-body").is_some()
                || response.header("x-ms-encryption-key-sha256").is_some()
                || response
                    .header("x-ms-copy-status")
                    .is_some_and(|value| value != "success")
                || response
                    .header("content-encoding")
                    .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
            {
                return Err(Error::Unsupported(
                    "Azure requires a complete identity-encoded BlockBlob".into(),
                ));
            }
            if let Some(super::ResponseBinding::AzureVersion { version_sha256 }) = binding
                && let Some(actual) = response.header("x-ms-version-id")
                && (actual.is_empty()
                    || actual.len() > 128
                    || !actual.bytes().all(|byte| byte.is_ascii_graphic())
                    || <[u8; 32]>::from(sha2::Sha256::digest(actual.as_bytes())) != version_sha256)
            {
                return Err(Error::State("Azure object version binding changed".into()));
            }
            Ok(())
        }
        Some(super::ResponseBinding::S3Content) => {
            if response
                .header("x-amz-delete-marker")
                .is_some_and(|value| value != "false")
            {
                return Err(Error::State("S3 content target is deleted".into()));
            }
            Ok(())
        }
        Some(super::ResponseBinding::S3Version { version_sha256 }) => {
            let version = response
                .header("x-amz-version-id")
                .ok_or_else(|| Error::Malformed("S3 response version is absent".into()))?;
            if version.is_empty()
                || version == "null"
                || version.len() > 1024
                || version.chars().any(char::is_control)
                || <[u8; 32]>::from(sha2::Sha256::digest(version.as_bytes())) != version_sha256
                || response
                    .header("x-amz-delete-marker")
                    .is_some_and(|value| value != "false")
            {
                return Err(Error::State("S3 object version binding changed".into()));
            }
            Ok(())
        }
        Some(super::ResponseBinding::GcsVersion {
            generation,
            metageneration,
        }) => {
            for (name, expected) in [
                ("x-goog-generation", generation),
                ("x-goog-metageneration", metageneration),
            ] {
                if response
                    .header(name)
                    .is_some_and(|actual| actual != expected.to_string())
                {
                    return Err(Error::State("HTTP object version binding changed".into()));
                }
            }
            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
pub struct HttpOptions {
    pub max_redirects: usize,
    pub max_retries: usize,
    pub connect_timeout: Duration,
    /// Budget for response headers; this is not a body idle timeout.
    pub response_header_timeout: Duration,
    /// Whole request/body deadline, also bounding a blocked synchronous read.
    /// Cancellation is checked between reads and after that deadline returns.
    pub total_timeout: Duration,
    pub backoff: Duration,
    pub max_header_bytes: usize,
}

impl Default for HttpOptions {
    fn default() -> Self {
        Self {
            max_redirects: 5,
            max_retries: 2,
            connect_timeout: Duration::from_secs(10),
            response_header_timeout: Duration::from_secs(10),
            total_timeout: Duration::from_secs(120),
            backoff: Duration::from_millis(100),
            max_header_bytes: 16 * 1024,
        }
    }
}

#[derive(Clone)]
pub struct HttpsClient {
    agent: ureq::Agent,
    credential_origins: Vec<String>,
    token: Option<String>,
    options: HttpOptions,
    loopback: bool,
    s3_signing: Option<s3_signing::Signing>,
    azure_auth: Option<azure_auth::Auth>,
    modelscope_token: bool,
}

impl HttpsClient {
    /// Explicit preissued official token, scoped to one native ModelScope
    /// origin and Direct targets only. No cookie jar, login or SDK state.
    pub fn with_modelscope_token(origin: &str, token: &str, options: HttpOptions) -> Result<Self> {
        if origin != "https://modelscope.cn" {
            return Err(Error::Unsupported(
                "unsupported ModelScope credential origin".into(),
            ));
        }
        Self::modelscope_client(origin, token, options, false)
    }
    fn modelscope_client(
        origin: &str,
        token: &str,
        options: HttpOptions,
        loopback: bool,
    ) -> Result<Self> {
        if token.is_empty()
            || token.len() > 8192
            || !token
                .bytes()
                .all(|b| matches!(b, 0x21 | 0x23..=0x2b | 0x2d..=0x3a | 0x3c..=0x5b | 0x5d..=0x7e))
        {
            return Err(Error::Malformed("invalid ModelScope official token".into()));
        }
        let mut client = Self::build(origin, Some(token.to_owned()), options, loopback, None)?;
        client.modelscope_token = true;
        Ok(client)
    }
    #[cfg(any(test, feature = "diagnostic-acquisition"))]
    pub fn diagnostic_modelscope_loopback(
        origin: &str,
        token: &str,
        options: HttpOptions,
    ) -> Result<Self> {
        let parsed = parse_url(origin, true)?;
        if parsed.scheme() != "http" {
            return Err(Error::Malformed(
                "ModelScope fixture requires literal HTTP loopback".into(),
            ));
        }
        Self::modelscope_client(origin, token, options, true)
    }
    pub fn with_azure_shared_key(
        origin: &str,
        credentials: AzureSharedKey,
        options: HttpOptions,
    ) -> Result<Self> {
        azure_auth::check_origin(origin, Some(credentials.account()), false)?;
        let mut client = Self::build(origin, None, options, false, None)?;
        client.azure_auth = Some(azure_auth::Auth::SharedKey(credentials));
        Ok(client)
    }
    pub fn with_azure_sas(origin: &str, query: &str, options: HttpOptions) -> Result<Self> {
        azure_auth::check_origin(origin, None, false)?;
        let auth = azure_auth::Auth::sas(query)?;
        let mut client = Self::build(origin, None, options, false, None)?;
        client.azure_auth = Some(auth);
        Ok(client)
    }
    #[cfg(any(test, feature = "diagnostic-acquisition"))]
    pub fn diagnostic_azure_shared_key(
        origin: &str,
        credentials: AzureSharedKey,
        options: HttpOptions,
    ) -> Result<Self> {
        azure_auth::check_origin(origin, Some(credentials.account()), true)?;
        let mut client = Self::build(origin, None, options, true, None)?;
        client.azure_auth = Some(azure_auth::Auth::SharedKey(credentials));
        Ok(client)
    }
    #[cfg(any(test, feature = "diagnostic-acquisition"))]
    pub fn diagnostic_azure_sas(origin: &str, query: &str, options: HttpOptions) -> Result<Self> {
        azure_auth::check_origin(origin, None, true)?;
        let auth = azure_auth::Auth::sas(query)?;
        let mut client = Self::build(origin, None, options, true, None)?;
        client.azure_auth = Some(auth);
        Ok(client)
    }
    /// Explicit, single-origin SigV4 credentials. Only opaque provider-created
    /// S3 targets may use this client; no ambient credential chain is consulted.
    pub fn with_s3_credentials(
        origin: &str,
        region: &str,
        credentials: S3Credentials,
        options: HttpOptions,
    ) -> Result<Self> {
        let signing = s3_signing::Signing::new(region, credentials)?;
        let mut client = Self::build(origin, None, options, false, None)?;
        client.s3_signing = Some(signing);
        Ok(client)
    }
    #[cfg(any(test, feature = "diagnostic-acquisition"))]
    pub fn diagnostic_s3_loopback(
        origin: &str,
        region: &str,
        credentials: S3Credentials,
        options: HttpOptions,
    ) -> Result<Self> {
        let signing = s3_signing::Signing::new(region, credentials)?;
        let mut client = Self::build(origin, None, options, true, None)?;
        client.s3_signing = Some(signing);
        Ok(client)
    }

    pub fn new(origin: &str, options: HttpOptions) -> Result<Self> {
        Self::build(origin, None, options, false, None)
    }

    /// Explicit in-memory bearer credential for one declared origin. This
    /// performs no environment lookup or token exchange; provider code owns
    /// those cold steps before using this same HTTPS/redirect owner.
    pub fn with_bearer(origin: &str, token: &str, options: HttpOptions) -> Result<Self> {
        if token.is_empty()
            || token.len() > 8192
            || !token.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err(Error::Malformed("invalid bearer credential".into()));
        }
        Self::build(origin, Some(token.to_owned()), options, false, None)
    }

    /// This constructor reads only the explicitly selected environment variable.
    /// The public offline entry point must branch before calling it.
    pub fn from_env(
        origin: &str,
        profile: &str,
        token_env: Option<&str>,
        options: HttpOptions,
    ) -> Result<Self> {
        if profile.is_empty()
            || profile.len() > 64
            || !profile
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
        {
            return Err(Error::Malformed("invalid credential profile".into()));
        }
        let token = match token_env {
            Some(name)
                if !name.is_empty()
                    && name.len() <= 128
                    && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') =>
            {
                match std::env::var(name) {
                    Ok(value) if !value.is_empty() => Some(value),
                    Ok(_) | Err(std::env::VarError::NotPresent) => None,
                    Err(_) => {
                        return Err(Error::Malformed(
                            "credential environment is not UTF-8".into(),
                        ));
                    }
                }
            }
            Some(_) => {
                return Err(Error::Malformed(
                    "invalid credential environment name".into(),
                ));
            }
            None => None,
        };
        Self::build(origin, token, options, false, None)
    }

    /// Explicit additional initial-request origin, never redirect authorization.
    pub fn with_credential_origin(mut self, origin: &str) -> Result<Self> {
        if self.s3_signing.is_some() || self.azure_auth.is_some() || self.modelscope_token {
            return Err(Error::Unsupported(
                "provider credentials have one exact origin".into(),
            ));
        }
        let parsed = parse_url(origin, self.loopback)?;
        validate_origin_url(&parsed)?;
        if self.credential_origins.len() >= 8 {
            return Err(Error::Limit("credential origin limit".into()));
        }
        let normalized = parsed.origin().ascii_serialization();
        if !self.credential_origins.contains(&normalized) {
            self.credential_origins.push(normalized);
        }
        Ok(self)
    }

    #[cfg(any(test, feature = "diagnostic-acquisition"))]
    pub fn diagnostic_loopback(
        origin: &str,
        token: Option<String>,
        options: HttpOptions,
    ) -> Result<Self> {
        Self::build(origin, token, options, true, None)
    }

    #[cfg(any(test, feature = "diagnostic-acquisition"))]
    pub fn diagnostic_tls(
        origin: &str,
        certificate_pem: &[u8],
        options: HttpOptions,
    ) -> Result<Self> {
        let url = parse_url(origin, true)?;
        if url.scheme() != "https" {
            return Err(Error::Malformed("TLS fixture requires HTTPS".into()));
        }
        let certificate = ureq::tls::Certificate::from_pem(certificate_pem)
            .map_err(|_| Error::Malformed("invalid TLS fixture certificate".into()))?;
        let roots = ureq::tls::RootCerts::Specific(std::sync::Arc::new(vec![certificate]));
        Self::build(origin, None, options, true, Some(roots))
    }

    fn build(
        origin: &str,
        token: Option<String>,
        options: HttpOptions,
        loopback: bool,
        roots: Option<ureq::tls::RootCerts>,
    ) -> Result<Self> {
        let parsed = parse_url(origin, loopback)?;
        validate_origin_url(&parsed)?;
        if options.max_redirects > 10
            || options.max_retries > 8
            || options.max_header_bytes == 0
            || options.max_header_bytes > 64 * 1024
            || options.connect_timeout.is_zero()
            || options.response_header_timeout.is_zero()
            || options.total_timeout.is_zero()
            || options.total_timeout > Duration::from_secs(86400)
            || options.backoff > Duration::from_secs(2)
        {
            return Err(Error::Limit("invalid bounded HTTP options".into()));
        }
        if token
            .as_ref()
            .is_some_and(|v| v.len() > 8192 || !v.bytes().all(|b| b.is_ascii_graphic()))
        {
            return Err(Error::Malformed("invalid bearer credential".into()));
        }
        // Automatic redirects, decompression features and inherited proxies are
        // not part of this owner. TLS validation stays at rustls defaults.
        let mut config = ureq::Agent::config_builder()
            .https_only(!loopback)
            .proxy(None)
            .max_redirects(0)
            .max_redirects_will_error(false)
            .http_status_as_error(false)
            .max_response_header_size(options.max_header_bytes)
            .timeout_resolve(Some(options.connect_timeout))
            .timeout_connect(Some(options.connect_timeout))
            .timeout_recv_response(Some(options.response_header_timeout))
            .timeout_recv_body(Some(options.total_timeout))
            .user_agent("zerogravity-acquisition/0.1");
        if let Some(roots) = roots {
            config = config.tls_config(ureq::tls::TlsConfig::builder().root_certs(roots).build());
        }
        let config = config.build();
        Ok(Self {
            agent: config.into(),
            credential_origins: vec![parsed.origin().ascii_serialization()],
            token,
            options,
            loopback,
            s3_signing: None,
            azure_auth: None,
            modelscope_token: false,
        })
    }
}

impl HttpTransport for HttpsClient {
    fn get_target(
        &self,
        request: &HttpRequest,
        target: &HttpTarget,
        cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        self.request_target(request, target, cancel, false)
    }
    fn head_target(
        &self,
        request: &HttpRequest,
        target: &HttpTarget,
        cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        if request.range.is_some() {
            return Err(Error::Malformed("S3 HEAD cannot request a range".into()));
        }
        self.request_target(request, target, cancel, true)
    }

    fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        self.request(request, cancel, false, true, None)
    }
    fn get_without_redirects(
        &self,
        request: &HttpRequest,
        cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        self.request(request, cancel, false, false, None)
    }
    fn head(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        if request.range.is_some() {
            return Err(Error::Malformed(
                "HEAD probe cannot request a byte range".into(),
            ));
        }
        self.request(request, cancel, true, true, None)
    }
}
impl HttpsClient {
    fn request_target(
        &self,
        request: &HttpRequest,
        target: &HttpTarget,
        cancel: &AtomicBool,
        head: bool,
    ) -> Result<HttpResponse> {
        if request.url != target.url()
            || (!target.xet_isolated() && !self.credential_origins.contains(&target.origin))
        {
            return Err(Error::State(
                "provider target and client origin differ".into(),
            ));
        }
        self.request(request, cancel, head, target.xet_ordinary(), Some(target))
    }
    fn request(
        &self,
        request: &HttpRequest,
        cancel: &AtomicBool,
        head: bool,
        follow_redirects: bool,
        target: Option<&HttpTarget>,
    ) -> Result<HttpResponse> {
        let isolated = target.is_some_and(HttpTarget::xet_isolated);
        let observation = target.is_some_and(HttpTarget::xet_observation);
        if !isolated
            && self.modelscope_token
            && !target.is_some_and(|target| matches!(target.profile, TargetProfile::Direct))
        {
            return Err(Error::Unsupported(
                "ModelScope credentials require a direct provider target".into(),
            ));
        }
        if target.is_some_and(|target| matches!(target.profile, TargetProfile::Direct))
            && self.s3_signing.is_some()
        {
            return Err(Error::Unsupported(
                "SigV4 cannot authorize a direct provider target".into(),
            ));
        }
        if !isolated && target.is_none() && self.s3_signing.is_some() {
            return Err(Error::Unsupported(
                "SigV4 requires a canonical S3 target".into(),
            ));
        }
        if !isolated
            && self.azure_auth.is_some()
            && target.is_none_or(|target| target.azure_kind().is_none())
        {
            return Err(Error::Unsupported(
                "Azure credentials require a canonical Azure target".into(),
            ));
        }
        check_cancel(cancel)?;
        let initial = parse_url(&request.url, self.loopback)?;
        if let Some(range) = &request.range {
            range.validate()?;
            if target
                .and_then(HttpTarget::if_match)
                .is_some_and(|expected| expected != range.validator)
            {
                return Err(Error::State(
                    "range validator differs from conditional target".into(),
                ));
            }
            if target.is_some_and(|target| target.xet_budget().is_some() && !target.xet_ordinary())
            {
                return Err(Error::Unsupported(
                    "Xet requests cannot use HTTP suffix resume".into(),
                ));
            }
        }
        let deadline = Instant::now()
            .checked_add(self.options.total_timeout)
            .ok_or_else(|| Error::Limit("HTTP deadline overflow".into()))?;
        let mut attempt = 0_usize;
        'retry: loop {
            let mut current = initial.clone();
            let mut authorization = self
                .credential_origins
                .contains(&initial.origin().ascii_serialization())
                && !isolated;
            for redirects in 0..=self.options.max_redirects {
                check_cancel(cancel)?;
                let remaining = deadline
                    .checked_duration_since(Instant::now())
                    .filter(|d| !d.is_zero())
                    .ok_or_else(|| Error::Limit("HTTP deadline exceeded".into()))?;
                let accept = match request.accept {
                    HttpAccept::Json => "application/json",
                    HttpAccept::GitHubJson => "application/vnd.github+json",
                    HttpAccept::OciManifest => {
                        "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
                    }
                    HttpAccept::Binary => "application/octet-stream",
                };
                let prepared = target
                    .filter(|target| target.azure_kind().is_some())
                    .map(|target| {
                        azure_auth::prepare(
                            self.azure_auth.as_ref(),
                            target,
                            request,
                            head,
                            std::time::SystemTime::now(),
                        )
                    })
                    .transpose()?;
                let wire = prepared.as_ref().map_or_else(
                    || {
                        target
                            .filter(|target| !target.xet_ordinary())
                            .map_or_else(|| current.as_str().to_owned(), HttpTarget::url)
                    },
                    |prepared| prepared.uri.clone(),
                );
                let call = if head {
                    self.agent.head(wire.as_str())
                } else {
                    self.agent.get(wire.as_str())
                };
                let mut call = call
                    .config()
                    .timeout_global(Some(remaining))
                    .build()
                    .header("Accept-Encoding", "identity")
                    .header("Accept", accept);
                if authorization && let Some(token) = &self.token {
                    call = call.header("Authorization", format!("Bearer {token}"));
                    if self.modelscope_token {
                        call = call.header("Cookie", format!("m_session_id={token}"));
                    }
                }
                if let Some(prepared) = prepared {
                    for (name, value) in prepared.headers {
                        call = call.header(name, value);
                    }
                }
                if !isolated && let (Some(target), Some(signing)) = (target, &self.s3_signing) {
                    for (name, value) in
                        signing.headers(target, request, head, std::time::SystemTime::now())?
                    {
                        call = call.header(name, value);
                    }
                }
                if let Some(value) = target.and_then(HttpTarget::if_match) {
                    call = call.header("If-Match", value);
                }
                if let Some(range) = &request.range {
                    call = call
                        .header("Range", format!("bytes={}-", range.start))
                        .header("If-Range", &range.validator);
                }
                if let Some(target) = target {
                    match &target.profile {
                        TargetProfile::XetCas { grant, .. } => {
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_err(|_| {
                                    Error::State("invalid Xet authorization clock".into())
                                })?
                                .as_secs();
                            if now >= grant.expires {
                                return Err(Error::State("Xet read grant expired".into()));
                            }
                            call = call.header("Authorization", format!("Bearer {}", grant.token));
                        }
                        TargetProfile::XetRanges { ranges, .. } => {
                            use std::fmt::Write;
                            let mut header = String::from("bytes=");
                            for (index, interval) in ranges.iter().enumerate() {
                                if index != 0 {
                                    header.push(',');
                                }
                                write!(header, "{}-{}", interval.start(), interval.end()).map_err(
                                    |_| Error::Malformed("invalid Xet range capability".into()),
                                )?;
                            }
                            call = call.header("Range", header);
                        }
                        _ => {}
                    }
                    if target.xet_budget().is_some() {
                        target.begin_attempt(cancel)?;
                    }
                }
                let response = match call.call() {
                    Ok(response) => response,
                    Err(error) => {
                        let retryable = matches!(
                            error,
                            ureq::Error::Io(_)
                                | ureq::Error::Timeout(_)
                                | ureq::Error::ConnectionFailed
                                | ureq::Error::HostNotFound
                        );
                        if retryable && attempt < self.options.max_retries {
                            attempt += 1;
                            pause(
                                self.options.backoff.saturating_mul(attempt as u32),
                                deadline,
                                cancel,
                            )?;
                            continue 'retry;
                        }
                        // Never format upstream errors: they may contain signed
                        // query strings, credentials or untrusted server text.
                        return Err(Error::State("HTTPS request failed".into()));
                    }
                };
                let status = response.status().as_u16();
                if let Some(target) = target.filter(|target| target.xet_budget().is_some()) {
                    let header_bytes =
                        response
                            .headers()
                            .iter()
                            .try_fold(32u64, |total, (name, value)| {
                                total
                                    .checked_add(
                                        name.as_str().len() as u64
                                            + value.as_bytes().len() as u64
                                            + 4,
                                    )
                                    .ok_or_else(|| {
                                        Error::Limit(
                                            "Xet response header accounting overflow".into(),
                                        )
                                    })
                            })?;
                    target.charge_metadata(header_bytes)?;
                }
                if matches!(status, 301 | 302 | 303 | 307 | 308) && !observation {
                    if !follow_redirects {
                        return Err(Error::Unsupported(
                            "version-pinned HTTP redirects are unsupported".into(),
                        ));
                    }
                    if redirects == self.options.max_redirects {
                        return Err(Error::Limit("HTTP redirect limit".into()));
                    }
                    let location = response
                        .headers()
                        .get("location")
                        .and_then(|v| v.to_str().ok())
                        .ok_or_else(|| Error::Malformed("invalid HTTP redirect".into()))?;
                    let next = current
                        .join(location)
                        .map_err(|_| Error::Malformed("invalid HTTP redirect".into()))?;
                    let next = parse_url(next.as_str(), self.loopback)?;
                    if current.scheme() == "https" && next.scheme() != "https" {
                        return Err(Error::Unsupported(
                            "HTTPS redirect downgrade refused".into(),
                        ));
                    }
                    if current.origin() != next.origin() {
                        authorization = false;
                    }
                    current = next;
                    continue;
                }
                if matches!(status, 429 | 500 | 502 | 503 | 504)
                    && attempt < self.options.max_retries
                {
                    attempt += 1;
                    // Integer Retry-After only; all delays stay under the same
                    // deadline and the bounded two-second backoff policy.
                    let server_delay = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        .map(Duration::from_secs)
                        .unwrap_or_default();
                    let delay = server_delay
                        .max(self.options.backoff.saturating_mul(attempt as u32))
                        .min(Duration::from_secs(2));
                    pause(delay, deadline, cancel)?;
                    continue 'retry;
                }
                let mut headers = BTreeMap::new();
                for name in [
                    "content-length",
                    "content-range",
                    "content-encoding",
                    "content-type",
                    "docker-content-digest",
                    "transfer-encoding",
                    "etag",
                    "last-modified",
                    "link",
                    "x-repo-commit",
                    "x-linked-etag",
                    "x-linked-size",
                    "x-xet-hash",
                    "x-amz-version-id",
                    "x-amz-delete-marker",
                    "x-goog-generation",
                    "x-goog-metageneration",
                    "x-ms-version-id",
                    "x-ms-blob-type",
                    "x-ms-structured-body",
                    "x-ms-encryption-key-sha256",
                    "x-ms-copy-status",
                ] {
                    let values: Vec<_> = response.headers().get_all(name).iter().collect();
                    if values.len() > 1 {
                        return Err(Error::Malformed("duplicate HTTP metadata header".into()));
                    }
                    if let Some(value) = values.first() {
                        let value = value
                            .to_str()
                            .map_err(|_| Error::Malformed("invalid HTTP metadata header".into()))?;
                        headers.insert(name.to_string(), value.to_string());
                    }
                }
                if headers
                    .get("content-encoding")
                    .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
                {
                    return Err(Error::Unsupported(
                        "non-identity HTTP content encoding".into(),
                    ));
                }
                let declared = headers
                    .get("content-length")
                    .map(|s| parse_u64(s))
                    .transpose()?;
                if declared.is_some_and(|n| n > request.max_body_bytes)
                    && !(head && matches!(status, 405 | 501))
                    && !observation
                {
                    return Err(Error::Limit("HTTP declared body limit".into()));
                }
                let body: Box<dyn Read + Send> = if head || observation {
                    // Drop the HTTP response without reading a body, including
                    // when a nonconforming server sends bytes after HEAD.
                    Box::new(io::empty())
                } else {
                    Box::new(BoundedBody {
                        inner: response.into_body().into_reader(),
                        remaining: request.max_body_bytes,
                        deadline,
                        xet: target
                            .filter(|target| target.xet_budget().is_some())
                            .cloned(),
                    })
                };
                let response = HttpResponse {
                    status,
                    headers,
                    body,
                };
                if let Some(range) = &request.range {
                    super::resume::validate_range_response(range, &response)?;
                }
                return Ok(response);
            }
            unreachable!("bounded redirect loop returns or retries")
        }
    }
}

struct BoundedBody<R> {
    inner: R,
    remaining: u64,
    deadline: Instant,
    xet: Option<HttpTarget>,
}
impl<R: Read> Read for BoundedBody<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if Instant::now() >= self.deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "HTTP body deadline",
            ));
        }
        let mut count = output
            .len()
            .min(usize::try_from(self.remaining).unwrap_or(usize::MAX))
            .max(1);
        if let Some(target) = &self.xet {
            count = target
                .body_read_limit(count)
                .map_err(|_| io::Error::other("Xet body work budget exhausted"))?;
        }
        let read = self
            .inner
            .read(&mut output[..count])
            .map_err(|_| io::Error::other("HTTP body read failed"))?;
        if let Some(target) = &self.xet {
            target
                .charge_body(read as u64)
                .map_err(|_| io::Error::other("Xet body work budget exhausted"))?;
        }
        if Instant::now() >= self.deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "HTTP body deadline",
            ));
        }
        if read as u64 > self.remaining {
            return Err(io::Error::other("HTTP body byte limit"));
        }
        self.remaining -= read as u64;
        Ok(read)
    }
}

pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::State("acquisition canceled".into()))
    } else {
        Ok(())
    }
}

fn pause(delay: Duration, deadline: Instant, cancel: &AtomicBool) -> Result<()> {
    let until = Instant::now()
        .checked_add(delay)
        .ok_or_else(|| Error::Limit("HTTP retry delay".into()))?
        .min(deadline);
    loop {
        check_cancel(cancel)?;
        let now = Instant::now();
        if now >= deadline {
            return Err(Error::Limit("HTTP deadline exceeded".into()));
        }
        if now >= until {
            return Ok(());
        }
        std::thread::sleep(until.duration_since(now).min(Duration::from_millis(10)));
    }
}

pub(crate) fn parse_u64(value: &str) -> Result<u64> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Error::Malformed("invalid HTTP byte count".into()));
    }
    value
        .parse()
        .map_err(|_| Error::Limit("HTTP byte count overflow".into()))
}

/// Native provider origins retain ordinary HTTPS validation in every build.
/// Only diagnostic literal-loopback HTTP origins use the fixture parser.
pub(super) fn parse_provider_origin(value: &str) -> Result<Url> {
    let diagnostic_http = cfg!(any(test, feature = "diagnostic-acquisition"))
        && value
            .get(..5)
            .is_some_and(|scheme| scheme.eq_ignore_ascii_case("http:"));
    parse_url(value, diagnostic_http)
}

pub(super) fn parse_url(value: &str, loopback: bool) -> Result<Url> {
    if value.len() > 16 * 1024 {
        return Err(Error::Limit("HTTP URL limit".into()));
    }
    let parsed = Url::parse(value).map_err(|_| Error::Malformed("invalid HTTPS URL".into()))?;
    if !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
        || parsed.host_str().is_none()
    {
        return Err(Error::Malformed("invalid HTTPS authority".into()));
    }
    if loopback {
        if !matches!(parsed.scheme(), "http" | "https")
            || !matches!(parsed.host(), Some(url::Host::Ipv4(ip)) if ip.is_loopback())
                && !matches!(parsed.host(), Some(url::Host::Ipv6(ip)) if ip.is_loopback())
        {
            return Err(Error::Unsupported(
                "diagnostic HTTP requires literal loopback".into(),
            ));
        }
    } else if parsed.scheme() != "https" {
        return Err(Error::Unsupported("HTTPS required".into()));
    }
    Ok(parsed)
}

fn validate_origin_url(url: &Url) -> Result<()> {
    if url.path() != "/" || url.query().is_some() {
        return Err(Error::Malformed(
            "credential origin must contain no path or query".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    #[test]
    fn provider_origins_keep_native_https_in_diagnostic_builds() {
        for origin in [
            "https://s3.us-east-1.amazonaws.com",
            "HTTPS://OBJECT.EXAMPLE:443",
        ] {
            assert_eq!(parse_provider_origin(origin).unwrap().scheme(), "https");
        }
        assert!(parse_provider_origin("http://object.example").is_err());
        assert!(parse_provider_origin("http://localhost").is_err());
        assert_eq!(
            parse_provider_origin("http://127.0.0.1:12345")
                .unwrap()
                .host_str(),
            Some("127.0.0.1")
        );
    }

    fn accept_bounded(listener: &TcpListener) -> std::net::TcpStream {
        listener.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match listener.accept() {
                Ok((stream, _)) => return stream,
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(2))
                }
                Err(error) => panic!("bounded fixture accept failed: {error}"),
            }
        }
    }

    fn serve(
        replies: Vec<String>,
    ) -> (String, Arc<Mutex<Vec<String>>>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let captured = Arc::new(Mutex::new(Vec::new()));
        let copies = captured.clone();
        let worker = std::thread::spawn(move || {
            for reply in replies {
                let mut stream = accept_bounded(&listener);
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut request = Vec::new();
                let mut byte = [0_u8; 1];
                while !request.ends_with(b"\r\n\r\n") {
                    assert!(request.len() < 16384);
                    assert_eq!(stream.read(&mut byte).unwrap(), 1);
                    request.push(byte[0]);
                }
                copies
                    .lock()
                    .unwrap()
                    .push(String::from_utf8(request).unwrap());
                stream.write_all(reply.as_bytes()).unwrap();
            }
        });
        (origin, captured, worker)
    }
    fn reply(status: &str, headers: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\n{headers}\r\n{body}",
            body.len()
        )
    }
    fn client(origin: &str) -> HttpsClient {
        HttpsClient::diagnostic_loopback(
            origin,
            Some("secret_fixture_token".into()),
            HttpOptions {
                backoff: Duration::ZERO,
                ..HttpOptions::default()
            },
        )
        .unwrap()
    }

    fn xet_budget(origin: &str, maximum: usize) -> Arc<super::super::xet::budget::Budget> {
        let mut options = super::super::xet::XetOptions::default()
            .diagnostic_endpoints(origin, origin)
            .unwrap();
        options.max_requests = maximum;
        Arc::new(super::super::xet::budget::Budget::for_test(&options, 1024 * 1024).unwrap())
    }

    #[test]
    fn xet_discovery_observes_headers_without_following_or_opening_compatibility_body() {
        let (origin, captured, worker) = serve(vec![reply(
            "302 Found",
            "Location: http://127.0.0.1:1/NEVER_FOLLOW?secret=NEVER_ECHO\r\nX-Xet-Hash: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\n",
            "NEVER_READ_COMPATIBILITY_PAYLOAD",
        )]);
        let request = HttpRequest::new(format!("{origin}/owner/repo/resolve/pin/a.bin"), 1);
        let target =
            HttpTarget::xet_hub(&request.url, xet_budget(&origin, 1), XetHubMode::Discovery)
                .unwrap();
        let mut response = client(&origin)
            .get_target(&request, &target, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(response.status, 302);
        assert_eq!(response.header("x-xet-hash").unwrap().len(), 64);
        assert!(response.header("location").is_none());
        assert_eq!(response.body.read(&mut [0u8; 1]).unwrap(), 0);
        worker.join().unwrap();
        assert_eq!(captured.lock().unwrap().len(), 1);
    }

    #[test]
    fn xet_attempt_budget_is_enforced_inside_original_transient_retry_loop() {
        let (origin, captured, worker) = serve(vec![reply("503 Service Unavailable", "", "")]);
        let request = HttpRequest::new(format!("{origin}/metadata"), 1024);
        let target =
            HttpTarget::xet_hub(&request.url, xet_budget(&origin, 1), XetHubMode::Metadata)
                .unwrap();
        let error = client(&origin)
            .get_target(&request, &target, &AtomicBool::new(false))
            .err()
            .unwrap();
        assert!(matches!(error, Error::Limit(_)));
        worker.join().unwrap();
        assert_eq!(captured.lock().unwrap().len(), 1);
    }

    #[test]
    fn xet_observed_overflow_charges_one_sentinel_and_prevents_further_reads_or_attempts() {
        use super::super::xet::{XetOptions, budget::Budget, reconstruction::ByteInterval};
        let (origin, captured, worker) = serve(vec![reply(
            "206 Partial Content",
            "Content-Range: bytes 0-3/4\r\n",
            "abcd",
        )]);
        let mut options = XetOptions::default()
            .diagnostic_endpoints(&origin, &origin)
            .unwrap();
        options.max_wire_bytes = 3;
        options.max_requests = 3;
        let budget = Arc::new(Budget::for_test(&options, 1024 * 1024).unwrap());
        let interval: ByteInterval = serde_json::from_str(r#"{"start":0,"end":3}"#).unwrap();
        let target = HttpTarget::xet_ranges(
            &format!("{origin}/xorb?X-Xet-Signed-Range=bytes%3D0-3"),
            vec![interval],
            budget.clone(),
        )
        .unwrap();
        let request = HttpRequest::new(target.url(), 64);
        let owner = client(&origin);
        let mut response = owner
            .get_target(&request, &target, &AtomicBool::new(false))
            .unwrap();
        loop {
            match response.body.read(&mut [0u8; 64]) {
                Ok(0) => panic!("overlong bounded body cannot report successful EOF"),
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert_eq!(budget.observed_wire_bytes(), 4);
        assert!(response.body.read(&mut [0u8; 64]).is_err());
        assert_eq!(budget.observed_wire_bytes(), 4);
        assert!(matches!(
            owner.get_target(&request, &target, &AtomicBool::new(false)),
            Err(Error::Limit(_))
        ));
        worker.join().unwrap();
        assert_eq!(captured.lock().unwrap().len(), 1);
    }

    #[test]
    fn xet_exact_wire_budget_still_allows_the_terminal_zero_byte_probe() {
        use super::super::xet::{XetOptions, budget::Budget, reconstruction::ByteInterval};
        let (origin, captured, worker) = serve(vec![reply(
            "206 Partial Content",
            "Content-Range: bytes 0-2/3\r\n",
            "abc",
        )]);
        let mut options = XetOptions::default()
            .diagnostic_endpoints(&origin, &origin)
            .unwrap();
        options.max_wire_bytes = 3;
        let budget = Arc::new(Budget::for_test(&options, 1024 * 1024).unwrap());
        let interval: ByteInterval = serde_json::from_str(r#"{"start":0,"end":2}"#).unwrap();
        let target = HttpTarget::xet_ranges(
            &format!("{origin}/xorb?X-Xet-Signed-Range=bytes%3D0-2"),
            vec![interval],
            budget.clone(),
        )
        .unwrap();
        let request = HttpRequest::new(target.url(), 3);
        // read_bounded must make the extra read and observe actual zero-byte EOF.
        assert_eq!(
            client(&origin)
                .get_target(&request, &target, &AtomicBool::new(false))
                .unwrap()
                .read_bounded(3, &AtomicBool::new(false))
                .unwrap(),
            b"abc"
        );
        assert_eq!(budget.observed_wire_bytes(), 3);
        worker.join().unwrap();
        assert_eq!(captured.lock().unwrap().len(), 1);
    }

    #[test]
    fn xet_cas_and_signed_ranges_isolate_credentials_even_at_same_client_origin() {
        use super::super::xet::reconstruction::ByteInterval;
        let (origin, captured, worker) = serve(vec![
            reply("200 OK", "", "abc"),
            reply(
                "206 Partial Content",
                "Content-Range: bytes 0-2/3\r\n",
                "abc",
            ),
        ]);
        let budget = xet_budget(&origin, 2);
        let expires = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;
        let grant = XetGrant::new(
            &origin,
            "CAS_SECRET".into(),
            expires,
            "owner/repo",
            &"a".repeat(40),
            true,
        )
        .unwrap();
        let owner = client(&origin);
        let cas = HttpRequest::new(format!("{origin}/v2/reconstructions/file"), 3);
        let target = HttpTarget::xet_cas(&cas.url, grant, budget.clone()).unwrap();
        assert_eq!(
            owner
                .get_target(&cas, &target, &AtomicBool::new(false))
                .unwrap()
                .read_bounded(3, &AtomicBool::new(false))
                .unwrap(),
            b"abc"
        );
        let interval: ByteInterval = serde_json::from_str(r#"{"start":0,"end":2}"#).unwrap();
        let target = HttpTarget::xet_ranges(
            &format!("{origin}/xorb?X-Xet-Signed-Range=bytes%3D0-2&Signature=HIDE"),
            vec![interval],
            budget,
        )
        .unwrap();
        let request = HttpRequest::new(target.url(), 3);
        assert_eq!(
            owner
                .get_target(&request, &target, &AtomicBool::new(false))
                .unwrap()
                .read_bounded(3, &AtomicBool::new(false))
                .unwrap(),
            b"abc"
        );
        worker.join().unwrap();
        let wire = captured.lock().unwrap();
        assert!(
            wire[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer cas_secret")
        );
        assert!(!wire[0].contains("secret_fixture_token"));
        let range = wire[1].to_ascii_lowercase();
        assert!(range.contains("range: bytes=0-2"));
        for name in [
            "authorization:",
            "cookie:",
            "x-amz-security-token:",
            "x-ms-date:",
        ] {
            assert!(!range.contains(name));
        }
    }

    #[test]
    fn xet_ordinary_payload_uses_redirected_wire_target_and_strips_cross_origin_credentials() {
        let (destination, final_wire, destination_worker) = serve(vec![reply("200 OK", "", "abc")]);
        let (origin, first_wire, first_worker) = serve(vec![reply(
            "302 Found",
            &format!("Location: {destination}/changed-path?opaque=HIDE\r\n"),
            "",
        )]);
        let request = HttpRequest::new(format!("{origin}/ordinary-git-file"), 3);
        let target = HttpTarget::xet_hub(
            &request.url,
            xet_budget(&origin, 2),
            XetHubMode::OrdinaryPayload,
        )
        .unwrap();
        assert_eq!(
            client(&origin)
                .get_target(&request, &target, &AtomicBool::new(false))
                .unwrap()
                .read_bounded(3, &AtomicBool::new(false))
                .unwrap(),
            b"abc"
        );
        first_worker.join().unwrap();
        destination_worker.join().unwrap();
        assert!(
            first_wire.lock().unwrap()[0]
                .to_ascii_lowercase()
                .contains("authorization:")
        );
        let wire = final_wire.lock().unwrap();
        assert!(wire[0].starts_with("GET /changed-path?opaque=HIDE "));
        assert!(!wire[0].to_ascii_lowercase().contains("authorization:"));
    }

    #[test]
    fn xet_ordinary_suffix_resume_preserves_original_wire_checks_and_actual_retry_budget() {
        for maximum in [1, 2] {
            let mut replies = vec![reply("503 Service Unavailable", "", "")];
            if maximum == 2 {
                replies.push(reply(
                    "206 Partial Content",
                    "ETag: \"stable\"\r\nContent-Range: bytes 2-4/5\r\n",
                    "cde",
                ));
            }
            let (origin, captured, worker) = serve(replies);
            let mut request = HttpRequest::new(format!("{origin}/ordinary.txt"), 3);
            request.range = Some(ResumeRequest {
                start: 2,
                total: 5,
                validator: "\"stable\"".into(),
            });
            let target = HttpTarget::xet_hub(
                &request.url,
                xet_budget(&origin, maximum),
                XetHubMode::OrdinaryPayload,
            )
            .unwrap();
            let result = client(&origin).get_target(&request, &target, &AtomicBool::new(false));
            if maximum == 1 {
                assert!(matches!(result, Err(Error::Limit(_))));
            } else {
                assert_eq!(
                    result
                        .unwrap()
                        .read_bounded(3, &AtomicBool::new(false))
                        .unwrap(),
                    b"cde"
                );
            }
            worker.join().unwrap();
            let wire = captured.lock().unwrap();
            assert_eq!(wire.len(), maximum);
            for request in wire.iter() {
                let lower = request.to_ascii_lowercase();
                assert!(lower.contains("range: bytes=2-"));
                assert!(lower.contains("if-range: \"stable\""));
                assert!(lower.contains("authorization: bearer secret_fixture_token"));
            }
        }
    }

    struct CaptureLogger(Mutex<Vec<String>>);
    impl log::Log for CaptureLogger {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            self.0
                .lock()
                .unwrap()
                .push(format!("{} {}", record.target(), record.args()));
        }
        fn flush(&self) {}
    }
    static LOGGER: CaptureLogger = CaptureLogger(Mutex::new(Vec::new()));

    #[test]
    fn actual_library_logs_redact_bearer_signed_query_and_location_at_trace_request() {
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Trace);
        assert_eq!(log::max_level(), log::LevelFilter::Trace);
        assert_eq!(log::STATIC_MAX_LEVEL, log::LevelFilter::Debug);
        let secret_query = "signed_query_credential_fixture_28491";
        let (other, _, second) = serve(vec![reply("200 OK", "", "ok")]);
        let (origin, captured, first) = serve(vec![reply(
            "302 Found",
            &format!("Location: {other}/private?signature={secret_query}\r\n"),
            "",
        )]);
        let transport = client(&origin);
        let response = transport
            .get(
                &HttpRequest::new(format!("{origin}/private?signature={secret_query}"), 2),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(
            response.read_bounded(2, &AtomicBool::new(false)).unwrap(),
            b"ok"
        );
        first.join().unwrap();
        second.join().unwrap();
        assert!(captured.lock().unwrap()[0].contains("secret_fixture_token"));
        let error = transport
            .get(
                &HttpRequest::new(format!("{origin}/#signature={secret_query}"), 1),
                &AtomicBool::new(false),
            )
            .err()
            .unwrap();
        let errors = format!("{error:?} {error}");
        let logs = LOGGER.0.lock().unwrap();
        assert!(
            logs.iter().any(|line| line.starts_with("ureq")),
            "actual library logging exercised"
        );
        for secret in [secret_query, "secret_fixture_token"] {
            assert!(!errors.contains(secret));
            assert!(
                logs.iter().all(|line| !line.contains(secret)),
                "library logs exposed fixture credential"
            );
        }
    }

    #[test]
    fn s3_actual_wire_preserves_raw_dot_segments_and_never_redirects_signed_credentials() {
        let (origin, captured, worker) = serve(vec![
            reply("200 OK", "x-amz-version-id: v1\r\n", "ok"),
            reply("200 OK", "x-amz-version-id: v1\r\n", "")
                .replace("Content-Length: 0", "Content-Length: 2"),
            reply(
                "206 Partial Content",
                "x-amz-version-id: v1\r\nETag: \"e\"\r\nContent-Range: bytes 1-1/2\r\n",
                "k",
            ),
        ]);
        let target = HttpTarget::s3(
            &origin,
            "us-east-1",
            "/bucket/a/../b//./%252E%20%C3%A9?versionId=v%2B%2F",
        )
        .unwrap();
        assert_ne!(
            parse_url(&target.url(), true).unwrap().path(),
            target.uri.path()
        );
        let mut client =
            HttpsClient::diagnostic_loopback(&origin, None, HttpOptions::default()).unwrap();
        client.s3_signing = Some(
            s3_signing::Signing::new(
                "us-east-1",
                S3Credentials::new("fixture_id", "SECRET_KEY", Some("SESSION_TOKEN")).unwrap(),
            )
            .unwrap(),
        );
        let response = client
            .get_target(
                &HttpRequest::new(target.url(), 2),
                &target,
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(
            response.read_bounded(2, &AtomicBool::new(false)).unwrap(),
            b"ok"
        );
        let head = client
            .head_target(
                &HttpRequest::new(target.url(), 2),
                &target,
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(head.header("content-length"), Some("2"));
        assert!(
            head.read_bounded(0, &AtomicBool::new(false))
                .unwrap()
                .is_empty()
        );
        let mut range = HttpRequest::new(target.url(), 1);
        range.range = Some(ResumeRequest {
            start: 1,
            total: 2,
            validator: "\"e\"".into(),
        });
        let response = client
            .get_target(&range, &target, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(
            response.read_bounded(1, &AtomicBool::new(false)).unwrap(),
            b"k"
        );
        worker.join().unwrap();
        let wire = captured.lock().unwrap()[0].clone();
        assert!(
            wire.starts_with("GET /bucket/a/../b//./%252E%20%C3%A9?versionId=v%2B%2F HTTP/1.1\r\n"),
            "{wire}"
        );
        assert!(wire.contains("AWS4-HMAC-SHA256 Credential=fixture_id/"));
        assert!(
            wire.to_ascii_lowercase()
                .contains("x-amz-security-token: session_token")
        );
        assert!(!wire.contains("SECRET_KEY"));
        for wire in captured.lock().unwrap().iter() {
            let mut first = wire.lines().next().unwrap().split_ascii_whitespace();
            let method = first.next().unwrap();
            let uri: ureq::http::Uri = format!("{origin}{}", first.next().unwrap())
                .parse()
                .unwrap();
            let headers = wire
                .lines()
                .skip(1)
                .filter_map(|line| line.split_once(':'))
                .map(|(name, value)| (name.to_ascii_lowercase(), value.trim().to_owned()))
                .collect::<BTreeMap<_, _>>();
            let authorization = &headers["authorization"];
            let signed = authorization
                .split("SignedHeaders=")
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap();
            let canonical = signed
                .split(';')
                .map(|name| (name, headers[name].clone()))
                .collect();
            let expected = client
                .s3_signing
                .as_ref()
                .unwrap()
                .authorization(method, &uri, &canonical, &headers["x-amz-date"])
                .unwrap();
            assert_eq!(
                authorization, &expected,
                "actual method/target/Host/Range/session headers must match the signature"
            );
        }
        let wires = captured.lock().unwrap();
        assert!(wires[1].starts_with("HEAD "));
        assert!(
            wires[2]
                .to_ascii_lowercase()
                .contains("range: bytes=1-\r\n")
        );
        assert!(
            wires[2]
                .to_ascii_lowercase()
                .contains("if-range: \"e\"\r\n")
        );
        drop(wires);
        assert!(
            client
                .get(&HttpRequest::new(target.url(), 2), &AtomicBool::new(false))
                .is_err()
        );
        assert!(
            client
                .clone()
                .with_credential_origin("https://other.example")
                .is_err()
        );

        let destination = TcpListener::bind("127.0.0.1:0").unwrap();
        destination.set_nonblocking(true).unwrap();
        let next = format!("http://{}/unrequested", destination.local_addr().unwrap());
        let (origin, _, worker) = serve(vec![reply(
            "307 Temporary Redirect",
            &format!("Location: {next}\r\n"),
            "",
        )]);
        let target = HttpTarget::s3(&origin, "us-east-1", "/bucket/key?versionId=v1").unwrap();
        let mut client =
            HttpsClient::diagnostic_loopback(&origin, None, HttpOptions::default()).unwrap();
        client.s3_signing = Some(
            s3_signing::Signing::new(
                "us-east-1",
                S3Credentials::new("fixture_id", "SECRET_KEY", None).unwrap(),
            )
            .unwrap(),
        );
        assert!(
            client
                .get_target(
                    &HttpRequest::new(target.url(), 2),
                    &target,
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        worker.join().unwrap();
        assert!(
            matches!(destination.accept(), Err(error) if error.kind()==io::ErrorKind::WouldBlock)
        );
    }

    #[test]
    fn in_memory_bearer_is_validated_and_scoped_to_the_declared_origin() {
        let client = HttpsClient::with_bearer(
            "HTTPS://REGISTRY.EXAMPLE:443",
            "fixture_bearer",
            HttpOptions::default(),
        )
        .unwrap();
        assert_eq!(client.token.as_deref(), Some("fixture_bearer"));
        assert_eq!(client.credential_origins, ["https://registry.example"]);
        for token in ["", "has space", "line\nfeed", "unicode-é"] {
            assert!(
                HttpsClient::with_bearer("https://registry.example", token, HttpOptions::default())
                    .is_err()
            );
        }
        assert!(
            HttpsClient::with_bearer(
                "https://registry.example",
                &"x".repeat(8193),
                HttpOptions::default()
            )
            .is_err()
        );
    }

    #[test]
    fn production_refuses_non_https_and_secret_errors_are_static() {
        assert!(HttpsClient::new("http://127.0.0.1:1", HttpOptions::default()).is_err());
        let client = HttpsClient::new("https://example.invalid", HttpOptions::default()).unwrap();
        for url in [
            "http://example.invalid/?token=secret_fixture_token",
            "https://user:secret_fixture_token@example.invalid/",
            "https://example.invalid/#secret_fixture_token",
        ] {
            let error = client
                .get(&HttpRequest::new(url, 1), &AtomicBool::new(false))
                .err()
                .unwrap();
            assert!(!format!("{error:?} {error}").contains("secret_fixture_token"));
        }
        assert!(
            client
                .get(
                    &HttpRequest::new("https://example.invalid", 1),
                    &AtomicBool::new(true)
                )
                .is_err()
        );
    }

    #[test]
    fn real_http_cross_origin_redirect_drops_auth_and_preserves_identity() {
        let (other, second, worker2) = serve(vec![reply("200 OK", "", "ok")]);
        let (origin, first, worker1) = serve(vec![reply(
            "302 Found",
            &format!("Location: {other}/signed?token=secret_fixture_token\r\n"),
            "",
        )]);
        let response = client(&origin)
            .get(
                &HttpRequest::new(format!("{origin}/start"), 2),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(
            response.read_bounded(2, &AtomicBool::new(false)).unwrap(),
            b"ok"
        );
        worker1.join().unwrap();
        worker2.join().unwrap();
        let first = first.lock().unwrap()[0].to_ascii_lowercase();
        let second = second.lock().unwrap()[0].to_ascii_lowercase();
        assert!(first.contains("authorization: bearer secret_fixture_token"));
        assert!(first.contains("accept-encoding: identity"));
        assert!(!second.contains("authorization:"));
    }

    #[test]
    fn real_http_encoding_header_and_stream_bounds_refuse() {
        let (origin, _, worker) = serve(vec![reply("200 OK", "Content-Encoding: gzip\r\n", "x")]);
        assert!(
            client(&origin)
                .get(
                    &HttpRequest::new(format!("{origin}/"), 1),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        worker.join().unwrap();
        let (origin, _, worker) = serve(vec![reply("200 OK", "", "toolong")]);
        assert!(
            client(&origin)
                .get(
                    &HttpRequest::new(format!("{origin}/"), 2),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        worker.join().unwrap();
        let (origin,_,worker)=serve(vec!["HTTP/1.1 200 OK\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n".into()]);
        let response = client(&origin)
            .get(
                &HttpRequest::new(format!("{origin}/"), 2),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert!(response.read_bounded(2, &AtomicBool::new(false)).is_err());
        worker.join().unwrap();
    }

    #[test]
    fn real_http_retry_and_strong_range_are_bounded() {
        let (origin, seen, worker) = serve(vec![
            reply("503 Service Unavailable", "Retry-After: 0\r\n", ""),
            reply(
                "206 Partial Content",
                "ETag: \"v1\"\r\nContent-Range: bytes 2-4/5\r\n",
                "abc",
            ),
        ]);
        let mut request = HttpRequest::new(format!("{origin}/file"), 3);
        request.range = Some(ResumeRequest {
            start: 2,
            total: 5,
            validator: "\"v1\"".into(),
        });
        let response = client(&origin)
            .get(&request, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(
            response.read_bounded(3, &AtomicBool::new(false)).unwrap(),
            b"abc"
        );
        worker.join().unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);
        for request in seen.lock().unwrap().iter() {
            assert!(request.to_ascii_lowercase().contains("if-range: \"v1\""));
        }
    }

    #[test]
    fn retry_redirect_limits_and_backoff_cancellation_are_actual() {
        for redirect in [false, true] {
            let response = if redirect {
                reply("302 Found", "Location: /loop\r\n", "")
            } else {
                reply("503 Service Unavailable", "", "")
            };
            let (origin, seen, worker) = serve(vec![response.clone(), response]);
            let client = HttpsClient::diagnostic_loopback(
                &origin,
                None,
                HttpOptions {
                    max_redirects: 1,
                    max_retries: 1,
                    backoff: Duration::ZERO,
                    ..HttpOptions::default()
                },
            )
            .unwrap();
            let result = client.get(
                &HttpRequest::new(format!("{origin}/"), 1),
                &AtomicBool::new(false),
            );
            if redirect {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap().status, 503);
            }
            worker.join().unwrap();
            assert_eq!(seen.lock().unwrap().len(), 2);
        }
        let (origin, _, worker) = serve(vec![reply(
            "503 Service Unavailable",
            "Retry-After: 1\r\n",
            "",
        )]);
        let client = client(&origin);
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            flag.store(true, Ordering::Release);
        });
        let started = Instant::now();
        assert!(
            client
                .get(&HttpRequest::new(format!("{origin}/"), 1), &cancel)
                .is_err()
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        canceller.join().unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn actual_tls_handshake_failure_does_not_expose_signed_query() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let mut stream = accept_bounded(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut hello = [0_u8; 512];
            let _ = stream.read(&mut hello);
            let _ = stream.shutdown(std::net::Shutdown::Both);
        });
        let options = HttpOptions {
            max_retries: 0,
            connect_timeout: Duration::from_secs(2),
            response_header_timeout: Duration::from_secs(2),
            total_timeout: Duration::from_secs(3),
            ..HttpOptions::default()
        };
        let client = HttpsClient::new(&format!("https://{address}"), options).unwrap();
        let error = client
            .get(
                &HttpRequest::new(
                    format!("https://{address}/?signature=secret_fixture_token"),
                    10,
                ),
                &AtomicBool::new(false),
            )
            .err()
            .unwrap();
        assert!(!format!("{error:?} {error}").contains("secret_fixture_token"));
        worker.join().unwrap();
    }

    #[test]
    fn real_http_same_origin_auth_and_bad_framing() {
        let (origin, seen, worker) = serve(vec![
            reply("302 Found", "Location: /next\r\n", ""),
            reply("200 OK", "", "x"),
        ]);
        client(&origin)
            .get(
                &HttpRequest::new(format!("{origin}/"), 1),
                &AtomicBool::new(false),
            )
            .unwrap()
            .read_bounded(1, &AtomicBool::new(false))
            .unwrap();
        worker.join().unwrap();
        assert!(seen.lock().unwrap().iter().all(|r| {
            r.to_ascii_lowercase()
                .contains("authorization: bearer secret_fixture_token")
        }));
        for headers in [
            "Content-Length: 2\r\nContent-Length: 3\r\n",
            "Content-Encoding: gzip\r\nContent-Encoding: identity\r\n",
        ] {
            let raw = format!("HTTP/1.1 200 OK\r\nConnection: close\r\n{headers}\r\nxx");
            let (origin, _, worker) = serve(vec![raw]);
            let client = HttpsClient::diagnostic_loopback(
                &origin,
                None,
                HttpOptions {
                    max_retries: 0,
                    ..HttpOptions::default()
                },
            )
            .unwrap();
            let result = client.get(
                &HttpRequest::new(format!("{origin}/"), 10),
                &AtomicBool::new(false),
            );
            assert!(result.is_err());
            worker.join().unwrap();
        }
    }

    #[test]
    fn metadata_cancellation_after_read_does_not_return_success() {
        struct CancelRead {
            flag: Arc<AtomicBool>,
        }
        impl Read for CancelRead {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                self.flag.store(true, Ordering::Release);
                bytes[0] = b'x';
                Ok(1)
            }
        }
        let flag = Arc::new(AtomicBool::new(false));
        let response = HttpResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: Box::new(CancelRead { flag: flag.clone() }),
        };
        assert!(response.read_bounded(10, &flag).is_err());
    }

    fn slow_stream(delay: Duration) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            let mut stream = accept_bounded(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = Vec::new();
            let mut byte = [0_u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                assert!(request.len() < 16384);
                assert_eq!(stream.read(&mut byte).unwrap(), 1);
                request.push(byte[0]);
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\na")
                .unwrap();
            for b in [b'b', b'c'] {
                std::thread::sleep(delay);
                if stream.write_all(&[b]).is_err() {
                    break;
                }
            }
        });
        (origin, worker)
    }

    #[test]
    fn progressing_body_uses_whole_deadline_not_header_budget() {
        let (origin, worker) = slow_stream(Duration::from_millis(100));
        let options = HttpOptions {
            max_retries: 0,
            response_header_timeout: Duration::from_millis(50),
            total_timeout: Duration::from_secs(2),
            ..HttpOptions::default()
        };
        let response = HttpsClient::diagnostic_loopback(&origin, None, options)
            .unwrap()
            .get(
                &HttpRequest::new(format!("{origin}/"), 3),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(
            response.read_bounded(3, &AtomicBool::new(false)).unwrap(),
            b"abc"
        );
        worker.join().unwrap();
        let (origin, worker) = slow_stream(Duration::from_millis(100));
        let options = HttpOptions {
            max_retries: 0,
            total_timeout: Duration::from_millis(50),
            ..HttpOptions::default()
        };
        let response = HttpsClient::diagnostic_loopback(&origin, None, options)
            .unwrap()
            .get(
                &HttpRequest::new(format!("{origin}/"), 3),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert!(response.read_bounded(3, &AtomicBool::new(false)).is_err());
        worker.join().unwrap();
    }

    #[test]
    fn cancellation_after_blocked_read_refuses_completion() {
        let (origin, worker) = slow_stream(Duration::from_millis(100));
        let client = HttpsClient::diagnostic_loopback(
            &origin,
            None,
            HttpOptions {
                max_retries: 0,
                total_timeout: Duration::from_secs(2),
                ..HttpOptions::default()
            },
        )
        .unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let response = client
            .get(&HttpRequest::new(format!("{origin}/"), 3), &cancel)
            .unwrap();
        let flag = cancel.clone();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            flag.store(true, Ordering::Release);
        });
        assert!(response.read_bounded(3, &cancel).is_err());
        canceller.join().unwrap();
        worker.join().unwrap();
    }

    struct PythonTls(std::process::Child);
    impl Drop for PythonTls {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    fn tls_server(host: &str, redirect: bool) -> (String, PythonTls) {
        use std::io::BufRead;
        use std::process::{Command, Stdio};
        let fixtures =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/acquisition");
        let script = r#"
import socket, ssl, sys, signal
signal.alarm(8)
context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(sys.argv[1],sys.argv[2])
with socket.socket() as listener:
    listener.settimeout(5)
    listener.bind((sys.argv[3],0)); listener.listen(1)
    print(listener.getsockname()[1],flush=True)
    raw,_=listener.accept(); raw.settimeout(5)
    try:
        with context.wrap_socket(raw,server_side=True) as stream:
            request=b''
            while not request.endswith(b'\r\n\r\n'):
                part=stream.recv(1)
                if not part: break
                request+=part
                if len(request)>16384: raise ValueError('header bound')
            if sys.argv[4]=='redirect':
                stream.sendall(b'HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/downgrade\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            else:
                stream.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok')
    except ssl.SSLError:
        raw.close()
"#;
        let child = Command::new("python3")
            .arg("-u")
            .arg("-c")
            .arg(script)
            .arg(fixtures.join("server.pem"))
            .arg(fixtures.join("server.key"))
            .arg(host)
            .arg(if redirect { "redirect" } else { "body" })
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut child = PythonTls(child);
        let stdout = child.0.stdout.take().unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        let readiness = std::thread::spawn(move || {
            let mut port = String::new();
            let result = std::io::BufReader::new(stdout)
                .read_line(&mut port)
                .map(|_| port);
            let _ = send.send(result);
        });
        let port = match receive.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(port)) => port,
            _ => {
                drop(child);
                readiness.join().unwrap();
                panic!("TLS fixture readiness deadline");
            }
        };
        readiness.join().unwrap();
        let port: u16 = port
            .trim()
            .parse()
            .expect("TLS fixture server did not start");
        (format!("https://{host}:{port}"), child)
    }

    #[test]
    fn real_tls_valid_root_untrusted_wrong_host_and_downgrade() {
        let ca = include_bytes!("../../tests/fixtures/acquisition/ca.pem");
        let options = HttpOptions {
            max_retries: 0,
            connect_timeout: Duration::from_secs(2),
            response_header_timeout: Duration::from_secs(2),
            total_timeout: Duration::from_secs(4),
            ..HttpOptions::default()
        };
        let (origin, mut server) = tls_server("127.0.0.1", false);
        let client = HttpsClient::diagnostic_tls(&origin, ca, options.clone()).unwrap();
        let body = client
            .get(
                &HttpRequest::new(format!("{origin}/"), 2),
                &AtomicBool::new(false),
            )
            .unwrap()
            .read_bounded(2, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(body, b"ok");
        assert!(server.0.wait().unwrap().success());
        let (origin, mut server) = tls_server("127.0.0.1", false);
        let client = HttpsClient::new(&origin, options.clone()).unwrap();
        assert!(
            client
                .get(
                    &HttpRequest::new(format!("{origin}/?secret=fixture"), 2),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(server.0.wait().unwrap().success());
        let (origin, mut server) = tls_server("127.0.0.2", false);
        let client = HttpsClient::diagnostic_tls(&origin, ca, options.clone()).unwrap();
        assert!(
            client
                .get(
                    &HttpRequest::new(format!("{origin}/"), 2),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(server.0.wait().unwrap().success());
        let (origin, mut server) = tls_server("127.0.0.1", true);
        let client = HttpsClient::diagnostic_tls(&origin, ca, options).unwrap();
        assert!(
            client
                .get(
                    &HttpRequest::new(format!("{origin}/"), 2),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(server.0.wait().unwrap().success());
    }
    #[test]
    fn modelscope_token_is_strict_cookie_safe_and_exact_origin_only() {
        for token in ["", "a b", "a;b", "a,b", "a\"b", "a\\b", "a\nb", "é"] {
            assert!(
                HttpsClient::with_modelscope_token(
                    "https://modelscope.cn",
                    token,
                    HttpOptions::default()
                )
                .is_err()
            );
        }
        assert!(
            HttpsClient::with_modelscope_token(
                "https://other.example",
                "SECRET",
                HttpOptions::default()
            )
            .is_err()
        );
        let client = HttpsClient::with_modelscope_token(
            "https://modelscope.cn",
            "valid_token-123",
            HttpOptions::default(),
        )
        .unwrap();
        assert!(
            client
                .clone()
                .with_credential_origin("https://other.example")
                .is_err()
        );
        assert!(
            client
                .get(
                    &HttpRequest::new("https://modelscope.cn/api", 1),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        let target = HttpTarget::direct("https://modelscope.cn/api").unwrap();
        let s3 = HttpsClient::with_s3_credentials(
            "https://modelscope.cn",
            "us-east-1",
            S3Credentials::new("ID", "SECRET", None).unwrap(),
            HttpOptions::default(),
        )
        .unwrap();
        assert!(
            s3.get_target(
                &HttpRequest::new(target.url(), 1),
                &target,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        let s3target = HttpTarget::s3("https://modelscope.cn", "us-east-1", "/api").unwrap();
        assert!(
            client
                .get_target(
                    &HttpRequest::new(s3target.url(), 1),
                    &s3target,
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        // These are cold profile refusals, before any DNS/socket operation.
        let azure_origin = "https://fixtureaccount.blob.core.windows.net";
        let azure_direct = HttpTarget::direct(&format!("{azure_origin}/container/a")).unwrap();
        let azure_key = AzureSharedKey::new("fixtureaccount", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==").unwrap();
        let shared =
            HttpsClient::with_azure_shared_key(azure_origin, azure_key, HttpOptions::default())
                .unwrap();
        let sas = HttpsClient::with_azure_sas(azure_origin, "sv=2023-11-03&sr=bv&sp=r&se=2030-01-01T00%3a00%3A00Z&spr=https&sig=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA%3d", HttpOptions::default()).unwrap();
        for azure_client in [shared, sas] {
            assert!(
                azure_client
                    .get_target(
                        &HttpRequest::new(azure_direct.url(), 1),
                        &azure_direct,
                        &AtomicBool::new(false)
                    )
                    .is_err()
            );
        }
        let azure_target = HttpTarget::azure(
            "https://modelscope.cn",
            "/container/a?versionid=v",
            AzureRequestKind::Version,
        )
        .unwrap();
        assert!(
            client
                .get_target(
                    &HttpRequest::new(azure_target.url(), 1),
                    &azure_target,
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(!format!("{target:?}").contains("api"));
    }
    #[cfg(feature = "diagnostic-acquisition")]
    fn modelscope_admission() -> crate::cache::TransferAdmission {
        use zerogravity_core::resource_arbiter::{
            PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
        };
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("fixture-host", "modelscope").unwrap();
        arbiter
            .register(resource.clone(), 128 * 1024 * 1024)
            .unwrap();
        crate::cache::TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("fixture", "modelscope").unwrap(),
        }
    }
    #[cfg(feature = "diagnostic-acquisition")]
    #[test]
    fn modelscope_official_token_actual_tree_and_raw_wire_use_one_direct_owner() {
        use super::super::modelscope::ModelScopeRequest;
        let tree = r#"{"Code":200,"Success":true,"Data":{"Files":[{"Path":"a.txt","Name":"a.txt","Type":"blob","Size":2}]}}"#;
        let (origin, captured, worker) =
            serve(vec![reply("200 OK", "", tree), reply("200 OK", "", "ok")]);
        let client = HttpsClient::diagnostic_modelscope_loopback(
            &origin,
            "TOKEN_FIXTURE",
            HttpOptions::default(),
        )
        .unwrap();
        let request = ModelScopeRequest::new(
            &origin,
            "owner/repo",
            "0123456789abcdef0123456789abcdef01234567",
            vec!["a.txt".into()],
            None,
        )
        .unwrap();
        let admission = modelscope_admission();
        let plan = request
            .resolve(
                &client,
                &super::super::PullLimits::default(),
                &admission,
                &AtomicBool::new(false),
            )
            .unwrap();
        let file = &plan.files()[0];
        let body = client
            .get_target(
                &HttpRequest::new(file.url(), 2),
                file.request_target().unwrap(),
                &AtomicBool::new(false),
            )
            .unwrap()
            .read_bounded(2, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(body, b"ok");
        worker.join().unwrap();
        let captured = captured.lock().unwrap();
        assert_eq!(captured.len(), 2);
        for wire in captured.iter() {
            let lower = wire.to_ascii_lowercase();
            assert!(lower.contains("authorization: bearer token_fixture\r\n"));
            assert!(lower.contains("cookie: m_session_id=token_fixture\r\n"));
            assert!(wire.contains("Revision=0123456789abcdef0123456789abcdef01234567"));
        }
        assert!(captured[0].contains("Recursive=False"));
        assert!(captured[1].contains("FilePath=a.txt"));
        assert_eq!(
            admission
                .arbiter
                .snapshot(&admission.resource)
                .unwrap()
                .reserved,
            0
        );
    }
    #[cfg(feature = "diagnostic-acquisition")]
    #[test]
    fn modelscope_anonymous_and_pinned_metadata_raw_redirects_never_follow() {
        use super::super::modelscope::ModelScopeRequest;
        let tree =
            r#"{"Code":200,"Success":true,"Data":[{"Path":"a.txt","Type":"blob","Size":2}]}"#;
        for (metadata_redirect, authenticated, cross_origin) in [
            (true, false, false),
            (false, false, true),
            (true, true, true),
            (false, true, false),
        ] {
            let redirect = reply(
                "302 Found",
                if cross_origin {
                    "Location: https://other.invalid/?token=NEVER_ECHO\r\n"
                } else {
                    "Location: /other?token=NEVER_ECHO\r\n"
                },
                "SECRET_BODY",
            );
            let replies = if metadata_redirect {
                vec![redirect]
            } else {
                vec![reply("200 OK", "", tree), redirect]
            };
            let (origin, captured, worker) = serve(replies);
            let client = if authenticated {
                HttpsClient::diagnostic_modelscope_loopback(
                    &origin,
                    "TOKEN_FIXTURE",
                    HttpOptions::default(),
                )
                .unwrap()
            } else {
                HttpsClient::diagnostic_loopback(&origin, None, HttpOptions::default()).unwrap()
            };
            let request = ModelScopeRequest::new(
                &origin,
                "owner/repo",
                "0123456789abcdef0123456789abcdef01234567",
                vec!["a.txt".into()],
                None,
            )
            .unwrap();
            let result = request.resolve(
                &client,
                &super::super::PullLimits::default(),
                &modelscope_admission(),
                &AtomicBool::new(false),
            );
            let error = if metadata_redirect {
                result.unwrap_err()
            } else {
                let plan = result.unwrap();
                let file = &plan.files()[0];
                client
                    .get_target(
                        &HttpRequest::new(file.url(), 2),
                        file.request_target().unwrap(),
                        &AtomicBool::new(false),
                    )
                    .err()
                    .unwrap()
            };
            assert!(!error.to_string().contains("NEVER_ECHO"));
            assert!(!error.to_string().contains("SECRET_BODY"));
            worker.join().unwrap();
            for wire in captured.lock().unwrap().iter() {
                assert_eq!(
                    wire.to_ascii_lowercase().contains("authorization:"),
                    authenticated
                );
                assert_eq!(wire.to_ascii_lowercase().contains("cookie:"), authenticated);
            }
        }
    }
}
