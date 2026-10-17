//! Streaming range envelope. Binary data is delimited by its validated exact
//! length, never by searching compressed bytes for MIME boundary strings.
use super::budget::Budget;
use super::{check_cancel, reconstruction::Descriptor};
use crate::acquisition::http::HttpResponse;
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Read};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const MAX_HEADER_BYTES: usize = 8192;
const MAX_HEADERS: usize = 64;

fn invalid() -> Error {
    Error::Malformed("invalid Xet range response envelope".into())
}
fn transport_error() -> Error {
    Error::Io(io::Error::other("Xet range body interrupted"))
}
fn decimal(value: &str) -> Result<u64> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(invalid());
    }
    value.parse().map_err(|_| invalid())
}
fn content_range(value: &str) -> Result<(u64, u64, u64)> {
    let (range, total) = value
        .strip_prefix("bytes ")
        .ok_or_else(invalid)?
        .split_once('/')
        .ok_or_else(invalid)?;
    let (start, end) = range.split_once('-').ok_or_else(invalid)?;
    let (start, end, total) = (decimal(start)?, decimal(end)?, decimal(total)?);
    if start > end || end >= total || total > 64 * 1024 * 1024 {
        return Err(invalid());
    }
    Ok((start, end, total))
}

/// Bound before asking the original HTTP owner to make the range request.
pub(super) fn maximum_body(descriptors: &[Descriptor]) -> Result<u64> {
    descriptors.iter().try_fold(128u64, |total, descriptor| {
        total
            .checked_add(descriptor.bytes.bytes()?)
            .and_then(|v| v.checked_add(MAX_HEADER_BYTES as u64 + 256))
            .ok_or_else(|| Error::Limit("Xet range envelope bound".into()))
    })
}

struct Body<'a> {
    reader: Box<dyn Read + Send>,
    declared: Option<u64>,
    maximum: u64,
    count: u64,
    cancel: &'a AtomicBool,
}
impl Read for Body<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        check_cancel(self.cancel).map_err(|_| io::Error::other("Xet range canceled"))?;
        let bound = self.declared.unwrap_or(self.maximum).min(self.maximum);
        let allowed = output.len().min(
            bound
                .saturating_sub(self.count)
                .saturating_add(1)
                .min(usize::MAX as u64) as usize,
        );
        let bytes = self.reader.read(&mut output[..allowed])?;
        self.count = self
            .count
            .checked_add(bytes as u64)
            .ok_or_else(|| io::Error::other("Xet range length bound"))?;
        if self.count > bound || (bytes == 0 && self.declared.is_some_and(|n| n != self.count)) {
            return Err(io::Error::other("Xet range framing length changed"));
        }
        Ok(bytes)
    }
}

pub(super) struct RangeBody<'a> {
    body: BufReader<Body<'a>>,
    descriptors: Vec<Descriptor>,
    seen: Vec<bool>,
    boundary: Option<String>,
    total: Option<u64>,
    active: Option<usize>,
    remaining: u64,
    started: bool,
    closed: bool,
    budget: Arc<Budget>,
}
impl<'a> RangeBody<'a> {
    pub(super) fn new(
        response: HttpResponse,
        descriptors: &[Descriptor],
        cancel: &'a AtomicBool,
        budget: Arc<Budget>,
    ) -> Result<Self> {
        check_cancel(cancel)?;
        if response.status != 206
            || descriptors.is_empty()
            || response
                .header("content-encoding")
                .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
        {
            return Err(invalid());
        }
        let declared = response.header("content-length").map(decimal).transpose()?;
        if response
            .header("transfer-encoding")
            .is_some_and(|value| declared.is_some() || !value.eq_ignore_ascii_case("chunked"))
        {
            return Err(invalid());
        }
        let maximum = maximum_body(descriptors)?;
        if declared.is_some_and(|n| n > maximum) {
            return Err(Error::Limit("Xet range response length bound".into()));
        }
        let boundary = boundary(response.header("content-type"))?;
        let total = if boundary.is_none() {
            if descriptors.len() != 1 {
                return Err(invalid());
            }
            let (start, end, total) =
                content_range(response.header("content-range").ok_or_else(invalid)?)?;
            if start != descriptors[0].bytes.start
                || end != descriptors[0].bytes.end
                || declared.is_some_and(|n| n != end - start + 1)
            {
                return Err(invalid());
            }
            Some(total)
        } else {
            if response.header("content-range").is_some() {
                return Err(invalid());
            }
            None
        };
        Ok(Self {
            body: BufReader::with_capacity(
                8192,
                Body {
                    reader: response.body,
                    declared,
                    maximum,
                    count: 0,
                    cancel,
                },
            ),
            descriptors: descriptors.to_vec(),
            seen: vec![false; descriptors.len()],
            boundary,
            total,
            active: None,
            remaining: 0,
            started: false,
            closed: false,
            budget,
        })
    }
    fn line(&mut self, allowance: &mut usize) -> Result<String> {
        let mut result = Vec::new();
        loop {
            check_cancel(self.body.get_ref().cancel)?;
            let available = self.body.fill_buf().map_err(|_| transport_error())?;
            if available.is_empty() {
                return Err(invalid());
            }
            let count = available
                .iter()
                .position(|b| *b == b'\n')
                .map_or(available.len(), |n| n + 1);
            if count > *allowance {
                return Err(Error::Limit("Xet multipart header bound".into()));
            }
            *allowance -= count;
            self.budget.charge_metadata(count as u64)?;
            result.extend_from_slice(&available[..count]);
            self.body.consume(count);
            if result.last() == Some(&b'\n') {
                if !result.ends_with(b"\r\n") {
                    return Err(invalid());
                }
                result.truncate(result.len() - 2);
                if !result
                    .iter()
                    .all(|b| b.is_ascii_graphic() || *b == b' ' || *b == b'\t')
                {
                    return Err(invalid());
                }
                return String::from_utf8(result).map_err(|_| invalid());
            }
        }
    }
    fn eof(&mut self) -> Result<()> {
        check_cancel(self.body.get_ref().cancel)?;
        let mut extra = [0u8; 1];
        if self.body.read(&mut extra).map_err(|_| transport_error())? != 0
            || self.seen.iter().any(|seen| !seen)
        {
            return Err(invalid());
        }
        self.closed = true;
        Ok(())
    }
    /// The preceding part must be consumed exactly before advancing. Returning
    /// None proves all expected parts, final delimiter and HTTP body completion.
    pub(super) fn next_part(&mut self) -> Result<Option<usize>> {
        check_cancel(self.body.get_ref().cancel)?;
        if self.closed {
            return Ok(None);
        }
        if self.remaining != 0 {
            return Err(invalid());
        }
        let Some(boundary) = self.boundary.clone() else {
            if self.started {
                self.eof()?;
                return Ok(None);
            }
            self.started = true;
            self.seen[0] = true;
            self.active = Some(0);
            self.remaining = self.descriptors[0].bytes.bytes()?;
            return Ok(Some(0));
        };
        let mut allowance = MAX_HEADER_BYTES;
        if self.started && !self.line(&mut allowance)?.is_empty() {
            return Err(invalid());
        }
        let delimiter = self.line(&mut allowance)?;
        if delimiter == format!("--{boundary}--") {
            if !self.started {
                return Err(invalid());
            }
            self.eof()?;
            return Ok(None);
        }
        if delimiter != format!("--{boundary}") {
            return Err(invalid());
        }
        let mut headers = BTreeMap::new();
        loop {
            let line = self.line(&mut allowance)?;
            if line.is_empty() {
                break;
            }
            let (name, value) = line.split_once(':').ok_or_else(invalid)?;
            if name.is_empty()
                || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                || headers.len() == MAX_HEADERS
                || headers
                    .insert(name.to_ascii_lowercase(), value.trim().to_string())
                    .is_some()
            {
                return Err(invalid());
            }
        }
        if headers.contains_key("transfer-encoding")
            || headers
                .get("content-encoding")
                .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
        {
            return Err(invalid());
        }
        let (start, end, total) = content_range(headers.get("content-range").ok_or_else(invalid)?)?;
        if self.total.is_some_and(|expected| expected != total) {
            return Err(invalid());
        }
        self.total = Some(total);
        let index = self
            .descriptors
            .binary_search_by_key(&start, |d| d.bytes.start)
            .map_err(|_| invalid())?;
        if self.descriptors[index].bytes.end != end
            || self.seen[index]
            || headers
                .get("content-length")
                .map(|v| decimal(v))
                .transpose()?
                .is_some_and(|n| n != end - start + 1)
        {
            return Err(invalid());
        }
        self.started = true;
        self.seen[index] = true;
        self.active = Some(index);
        self.remaining = end - start + 1;
        Ok(Some(index))
    }
    pub(super) fn read_exact(&mut self, output: &mut [u8]) -> Result<()> {
        if self.active.is_none() || output.len() as u64 > self.remaining {
            return Err(invalid());
        }
        for chunk in output.chunks_mut(4096) {
            check_cancel(self.body.get_ref().cancel)?;
            self.body.read_exact(chunk).map_err(|_| transport_error())?;
            self.remaining -= chunk.len() as u64;
        }
        Ok(())
    }
    pub(super) fn remaining(&self) -> u64 {
        self.remaining
    }
}

fn boundary(value: Option<&str>) -> Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let mut pieces = value.split(';');
    if !pieces
        .next()
        .ok_or_else(invalid)?
        .trim()
        .eq_ignore_ascii_case("multipart/byteranges")
    {
        return Ok(None);
    }
    let mut boundary = None;
    for parameter in pieces {
        let (name, value) = parameter.trim().split_once('=').ok_or_else(invalid)?;
        if !name.eq_ignore_ascii_case("boundary") || boundary.is_some() {
            return Err(invalid());
        }
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(value);
        if value.is_empty()
            || value.len() > 70
            || !value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"'()+_,-./:=?".contains(&b))
        {
            return Err(invalid());
        }
        boundary = Some(value.to_string());
    }
    boundary.ok_or_else(invalid).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    fn budget() -> Arc<Budget> {
        Arc::new(
            Budget::new(
                &super::super::XetOptions::default(),
                1024 * 1024,
                Default::default(),
            )
            .unwrap(),
        )
    }
    fn descriptors() -> Vec<Descriptor> {
        serde_json::from_str(r#"[{"chunks":{"start":0,"end":1},"bytes":{"start":0,"end":2}},{"chunks":{"start":1,"end":2},"bytes":{"start":10,"end":13}}]"#).unwrap()
    }
    fn part(start: u64, end: u64, total: u64, bytes: &[u8]) -> Vec<u8> {
        let mut result =
            format!("--fixture\r\nContent-Range: bytes {start}-{end}/{total}\r\n\r\n").into_bytes();
        result.extend_from_slice(bytes);
        result.extend_from_slice(b"\r\n");
        result
    }
    fn response(body: Vec<u8>) -> HttpResponse {
        HttpResponse {
            status: 206,
            headers: BTreeMap::from([
                (
                    "content-type".into(),
                    "multipart/byteranges; boundary=fixture".into(),
                ),
                ("content-length".into(), body.len().to_string()),
            ]),
            body: Box::new(Cursor::new(body)),
        }
    }
    fn drain(mut reader: RangeBody<'_>) -> Result<Vec<(usize, Vec<u8>)>> {
        let mut result = Vec::new();
        while let Some(index) = reader.next_part()? {
            let mut bytes = vec![0; reader.remaining() as usize];
            reader.read_exact(&mut bytes)?;
            result.push((index, bytes));
        }
        Ok(result)
    }
    #[test]
    fn arbitrary_part_order_and_binary_payload_are_matched_by_exact_interval() {
        let mut bytes = part(10, 13, 20, b"\0\r\n\0");
        bytes.extend(part(0, 2, 20, b"a--"));
        bytes.extend_from_slice(b"--fixture--\r\n");
        assert_eq!(
            drain(
                RangeBody::new(
                    response(bytes),
                    &descriptors(),
                    &AtomicBool::new(false),
                    budget()
                )
                .unwrap()
            )
            .unwrap(),
            vec![(1, b"\0\r\n\0".to_vec()), (0, b"a--".to_vec())]
        );
    }
    #[test]
    fn missing_duplicate_extra_conflicting_or_trailing_parts_refuse() {
        let first = part(0, 2, 20, b"abc");
        let second = part(10, 13, 20, b"defg");
        for middle in [
            Vec::new(),
            first.clone(),
            part(11, 14, 20, b"defg"),
            part(10, 13, 21, b"defg"),
        ] {
            let mut bytes = first.clone();
            bytes.extend(middle);
            bytes.extend_from_slice(b"--fixture--\r\n");
            assert!(
                drain(
                    RangeBody::new(
                        response(bytes),
                        &descriptors(),
                        &AtomicBool::new(false),
                        budget()
                    )
                    .unwrap()
                )
                .is_err()
            );
        }
        let mut bytes = first;
        bytes.extend(second);
        bytes.extend_from_slice(b"--fixture--\r\nEXTRA");
        assert!(
            drain(
                RangeBody::new(
                    response(bytes),
                    &descriptors(),
                    &AtomicBool::new(false),
                    budget()
                )
                .unwrap()
            )
            .is_err()
        );
    }
    #[test]
    fn truncated_and_noncanonical_delimiters_or_part_headers_refuse() {
        for raw in [
            "--fixture\nContent-Range: bytes 0-2/20\r\n\r\nabc\r\n--fixture--\r\n",
            "--fixture\r\nContent-Range: bytes 0-2/20\r\nContent-Range: bytes 0-2/20\r\n\r\nabc\r\n--fixture--\r\n",
            "--fixture\r\nContent-Range: bytes 00-2/20\r\n\r\nabc\r\n--fixture--\r\n",
            "--fixture\r\nContent-Range: bytes 0-2/20\r\nContent-Encoding: gzip\r\n\r\nabc\r\n--fixture--\r\n",
            "--fixture\r\nContent-Range: bytes 0-2/20\r\n\r\nabc\r\n--fixture--",
        ] {
            assert!(
                drain(
                    RangeBody::new(
                        response(raw.as_bytes().to_vec()),
                        &descriptors()[..1],
                        &AtomicBool::new(false),
                        budget()
                    )
                    .unwrap()
                )
                .is_err()
            );
        }
    }
    #[test]
    fn single_range_requires_exact_identity_length_and_terminal_read() {
        for (range, declared, bytes, passes) in [
            ("bytes 0-2/3", "3", b"abc".as_slice(), true),
            ("bytes 1-3/4", "3", b"abc".as_slice(), false),
            ("bytes 0-2/3", "3", b"ab".as_slice(), false),
            ("bytes 0-2/3", "3", b"abcd".as_slice(), false),
            ("bytes 0-2/3", "4", b"abcd".as_slice(), false),
        ] {
            let response = HttpResponse {
                status: 206,
                headers: BTreeMap::from([
                    ("content-range".into(), range.into()),
                    ("content-length".into(), declared.into()),
                ]),
                body: Box::new(Cursor::new(bytes.to_vec())),
            };
            let result = RangeBody::new(
                response,
                &descriptors()[..1],
                &AtomicBool::new(false),
                budget(),
            )
            .and_then(drain);
            assert_eq!(result.is_ok(), passes);
        }
    }
    #[test]
    fn boundary_profiles_are_bounded_without_guessing_mime_syntax() {
        assert_eq!(
            boundary(Some("multipart/byteranges; boundary=\"ab.c-3\"")).unwrap(),
            Some("ab.c-3".into())
        );
        for value in [
            "multipart/byteranges",
            "multipart/byteranges; boundary=",
            "multipart/byteranges; boundary=a; boundary=b",
            "multipart/byteranges; charset=utf8; boundary=a",
            "multipart/byteranges; boundary=\"a\\b\"",
        ] {
            assert!(boundary(Some(value)).is_err());
        }
        assert!(
            boundary(Some(&format!(
                "multipart/byteranges; boundary={}",
                "a".repeat(71)
            )))
            .is_err()
        );
    }
}
