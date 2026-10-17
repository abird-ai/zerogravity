//! One synchronous, verified HTTPS owner. Callers admit metadata/staging first.
//! Provider URLs and credentials are deliberately absent from diagnostic errors.

use crate::{Error, Result};
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use url::Url;

use super::resume::ResumeRequest;

#[derive(Clone, Copy, Debug)]
pub enum HttpAccept {
    Json,
    GitHubJson,
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

pub(super) fn validate_validator(expected: Option<&str>, response: &HttpResponse) -> Result<()> {
    if expected.is_some_and(|validator| response.header("etag") != Some(validator)) {
        return Err(Error::State(
            "HTTP object validator changed after length probe".into(),
        ));
    }
    Ok(())
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
}

impl HttpsClient {
    pub fn new(origin: &str, options: HttpOptions) -> Result<Self> {
        Self::build(origin, None, options, false, None)
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
        })
    }
}

impl HttpTransport for HttpsClient {
    fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        self.request(request, cancel, false)
    }
    fn head(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        if request.range.is_some() {
            return Err(Error::Malformed(
                "HEAD probe cannot request a byte range".into(),
            ));
        }
        self.request(request, cancel, true)
    }
}
impl HttpsClient {
    fn request(
        &self,
        request: &HttpRequest,
        cancel: &AtomicBool,
        head: bool,
    ) -> Result<HttpResponse> {
        check_cancel(cancel)?;
        let initial = parse_url(&request.url, self.loopback)?;
        if let Some(range) = &request.range {
            range.validate()?;
        }
        let deadline = Instant::now()
            .checked_add(self.options.total_timeout)
            .ok_or_else(|| Error::Limit("HTTP deadline overflow".into()))?;
        let mut attempt = 0_usize;
        'retry: loop {
            let mut current = initial.clone();
            let mut authorization = self
                .credential_origins
                .contains(&initial.origin().ascii_serialization());
            for redirects in 0..=self.options.max_redirects {
                check_cancel(cancel)?;
                let remaining = deadline
                    .checked_duration_since(Instant::now())
                    .filter(|d| !d.is_zero())
                    .ok_or_else(|| Error::Limit("HTTP deadline exceeded".into()))?;
                let accept = match request.accept {
                    HttpAccept::Json => "application/json",
                    HttpAccept::GitHubJson => "application/vnd.github+json",
                    HttpAccept::Binary => "application/octet-stream",
                };
                let call = if head {
                    self.agent.head(current.as_str())
                } else {
                    self.agent.get(current.as_str())
                };
                let mut call = call
                    .config()
                    .timeout_global(Some(remaining))
                    .build()
                    .header("Accept-Encoding", "identity")
                    .header("Accept", accept);
                if authorization && let Some(token) = &self.token {
                    call = call.header("Authorization", format!("Bearer {token}"));
                }
                if let Some(range) = &request.range {
                    call = call
                        .header("Range", format!("bytes={}-", range.start))
                        .header("If-Range", &range.validator);
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
                if matches!(status, 301 | 302 | 303 | 307 | 308) {
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
                    "transfer-encoding",
                    "etag",
                    "last-modified",
                    "link",
                    "x-repo-commit",
                    "x-linked-etag",
                    "x-linked-size",
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
                {
                    return Err(Error::Limit("HTTP declared body limit".into()));
                }
                let body: Box<dyn Read + Send> = if head {
                    // Drop the HTTP response without reading a body, including
                    // when a nonconforming server sends bytes after HEAD.
                    Box::new(io::empty())
                } else {
                    Box::new(BoundedBody {
                        inner: response.into_body().into_reader(),
                        remaining: request.max_body_bytes,
                        deadline,
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
        let count = output
            .len()
            .min(usize::try_from(self.remaining).unwrap_or(usize::MAX))
            .max(1);
        let read = self
            .inner
            .read(&mut output[..count])
            .map_err(|_| io::Error::other("HTTP body read failed"))?;
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

fn parse_url(value: &str, loopback: bool) -> Result<Url> {
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
}
