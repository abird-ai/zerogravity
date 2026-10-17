//! Resume metadata describes immutable source identity, never expiring URLs.
//! Persistence, owned prefix files and reservations belong to CacheStore.

use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::sync::atomic::AtomicBool;

use super::http::{HttpResponse, check_cancel, parse_u64};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResumeRequest {
    pub start: u64,
    pub total: u64,
    pub validator: String,
}

impl ResumeRequest {
    pub fn validate(&self) -> Result<()> {
        if self.start == 0 || self.start >= self.total {
            return Err(Error::Malformed("invalid resume byte range".into()));
        }
        strong_etag(&self.validator)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumeDisposition {
    Append,
    Restart,
}

pub fn strong_etag(value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    if bytes.len() < 2
        || bytes.len() > 1024
        || bytes[0] != b'"'
        || bytes[bytes.len() - 1] != b'"'
        || !bytes[1..bytes.len() - 1]
            .iter()
            .all(|b| *b >= 0x21 && *b != b'"' && *b < 0x7f)
    {
        return Err(Error::Malformed(
            "strong HTTP ETag required for resume".into(),
        ));
    }
    Ok(())
}

/// A caller may restart only its own admitted partial, never concatenate a200.
pub fn validate_range_response(
    request: &ResumeRequest,
    response: &HttpResponse,
) -> Result<ResumeDisposition> {
    request.validate()?;
    if response
        .header("content-encoding")
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        return Err(Error::Unsupported(
            "resume requires identity encoding".into(),
        ));
    }
    if response.status == 200 {
        return Ok(ResumeDisposition::Restart);
    }
    if response.status != 206 {
        return Err(Error::State(
            "HTTP resume requires206 or owned restart".into(),
        ));
    }
    if response.header("etag") != Some(request.validator.as_str()) {
        return Err(Error::State("HTTP resume validator changed".into()));
    }
    let value = response
        .header("content-range")
        .and_then(|v| v.strip_prefix("bytes "))
        .ok_or_else(|| Error::Malformed("missing resume Content-Range".into()))?;
    let (range, total) = value
        .split_once('/')
        .ok_or_else(|| Error::Malformed("invalid Content-Range".into()))?;
    let (start, end) = range
        .split_once('-')
        .ok_or_else(|| Error::Malformed("invalid Content-Range".into()))?;
    let (start, end, total) = (parse_u64(start)?, parse_u64(end)?, parse_u64(total)?);
    if start != request.start || total != request.total || end < start || end != total - 1 {
        return Err(Error::State("HTTP resume range or length changed".into()));
    }
    if let Some(length) = response.header("content-length")
        && parse_u64(length)? != total - start
    {
        return Err(Error::Malformed("resume response length mismatch".into()));
    }
    Ok(ResumeDisposition::Append)
}

/// Verify bytes from the already-open owned prefix before any resumptive append.
/// This helper creates no file and does not claim cross-invocation persistence.
pub fn verify_prefix(
    reader: &mut impl Read,
    bytes: u64,
    expected_sha256: &str,
    cancel: &AtomicBool,
) -> Result<()> {
    if expected_sha256.len() != 64 || !expected_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::Malformed("invalid resume prefix digest".into()));
    }
    let mut left = bytes;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 8192];
    while left != 0 {
        check_cancel(cancel)?;
        let size = left.min(buffer.len() as u64) as usize;
        let read = reader
            .read(&mut buffer[..size])
            .map_err(|_| Error::State("resume prefix read failed".into()))?;
        check_cancel(cancel)?;
        if read == 0 {
            return Err(Error::Malformed("resume prefix truncated".into()));
        }
        digest.update(&buffer[..read]);
        left -= read as u64;
    }
    let mut tail = [0_u8; 1];
    check_cancel(cancel)?;
    if reader
        .read(&mut tail)
        .map_err(|_| Error::State("resume prefix read failed".into()))?
        != 0
    {
        return Err(Error::Malformed("resume prefix length changed".into()));
    }
    check_cancel(cancel)?;
    if format!("{:x}", digest.finalize()) != expected_sha256.to_ascii_lowercase() {
        return Err(Error::Malformed("resume prefix digest mismatch".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::Cursor;

    fn response(status: u16, values: &[(&str, &str)]) -> HttpResponse {
        HttpResponse {
            status,
            headers: values
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            body: Box::new(Cursor::new(Vec::new())),
        }
    }

    #[test]
    fn exact_range_only_or_explicit_owned_restart() {
        let request = ResumeRequest {
            start: 2,
            total: 5,
            validator: "\"strong\"".into(),
        };
        let good = response(
            206,
            &[
                ("etag", "\"strong\""),
                ("content-range", "bytes 2-4/5"),
                ("content-length", "3"),
            ],
        );
        assert_eq!(
            validate_range_response(&request, &good).unwrap(),
            ResumeDisposition::Append
        );
        assert_eq!(
            validate_range_response(&request, &response(200, &[])).unwrap(),
            ResumeDisposition::Restart
        );
        for values in [
            vec![("etag", "W/\"strong\""), ("content-range", "bytes 2-4/5")],
            vec![("etag", "\"changed\""), ("content-range", "bytes 2-4/5")],
            vec![("etag", "\"strong\""), ("content-range", "bytes 1-4/5")],
            vec![("etag", "\"strong\""), ("content-range", "bytes 2-5/6")],
            vec![("etag", "\"strong\""), ("content-range", "bytes 2-3/5")],
            vec![
                ("etag", "\"strong\""),
                ("content-range", "bytes 2-4/5"),
                ("content-encoding", "gzip"),
            ],
            vec![
                ("etag", "\"strong\""),
                ("content-range", "bytes 2-4/5"),
                ("content-length", "4"),
            ],
        ] {
            assert!(validate_range_response(&request, &response(206, &values)).is_err());
        }
        assert!(validate_range_response(&request, &response(416, &[])).is_err());
        for value in ["W/\"weak\"", "unquoted", "\"line\r\nbreak\"", "\"a\"b\""] {
            assert!(strong_etag(value).is_err());
        }
    }

    #[test]
    fn owned_prefix_is_rehashed_and_exact_length() {
        let cancel = AtomicBool::new(false);
        let hash = format!("{:x}", Sha256::digest(b"abc"));
        assert!(verify_prefix(&mut Cursor::new(b"abc"), 3, &hash, &cancel).is_ok());
        for bytes in [&b"abd"[..], &b"ab"[..], &b"abcd"[..]] {
            assert!(verify_prefix(&mut Cursor::new(bytes), 3, &hash, &cancel).is_err());
        }
        let canceled = AtomicBool::new(true);
        assert!(verify_prefix(&mut Cursor::new(b"abc"), 3, &hash, &canceled).is_err());
    }
}
