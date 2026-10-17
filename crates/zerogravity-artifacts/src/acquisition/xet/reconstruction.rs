//! Bounded native reconstruction metadata. Transient grants/URLs stay here in
//! the reader and never become immutable source labels or raw SHA authority.
use super::{XetOptions, check_cancel, hash::XetHash};
use crate::{Error, Result};
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct ChunkSpan {
    pub start: u32,
    pub end: u32,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ByteInterval {
    pub(super) start: u64,
    pub(super) end: u64,
}
impl ByteInterval {
    pub(crate) fn start(self) -> u64 {
        self.start
    }
    pub(crate) fn end(self) -> u64 {
        self.end
    }
    pub(crate) fn bytes(self) -> Result<u64> {
        self.end
            .checked_sub(self.start)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(invalid)
    }
}
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Descriptor {
    pub chunks: ChunkSpan,
    pub bytes: ByteInterval,
}
pub(super) struct Group {
    pub url: String,
    pub ranges: Vec<Descriptor>,
}
pub(super) struct Term {
    pub hash: XetHash,
    pub unpacked: u64,
    pub chunks: ChunkSpan,
}
struct Location {
    descriptor: Descriptor,
    group: usize,
    range: usize,
}
pub(super) struct Reconstruction {
    pub terms: Vec<Term>,
    pub groups: BTreeMap<[u8; 32], Vec<Group>>,
    locations: BTreeMap<[u8; 32], Vec<Location>>,
    pub contract: [u8; 32],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTerm {
    hash: String,
    unpacked_length: u32,
    range: ChunkSpan,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGroup {
    url: String,
    ranges: Vec<Descriptor>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawV2 {
    offset_into_first_range: u64,
    terms: Vec<RawTerm>,
    xorbs: UniqueMap<Vec<RawGroup>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFetch {
    range: ChunkSpan,
    url: String,
    url_range: ByteInterval,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawV1 {
    offset_into_first_range: u64,
    terms: Vec<RawTerm>,
    fetch_info: UniqueMap<Vec<RawFetch>>,
}
struct UniqueMap<V>(BTreeMap<String, V>);
impl<'de, V: Deserialize<'de>> Deserialize<'de> for UniqueMap<V> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct UniqueVisitor<V>(std::marker::PhantomData<V>);
        impl<'de, V: Deserialize<'de>> Visitor<'de> for UniqueVisitor<V> {
            type Value = UniqueMap<V>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("bounded unique Xet map")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut input: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut result = BTreeMap::new();
                while let Some((key, value)) = input.next_entry::<String, V>()? {
                    if result.len() == 65536 || result.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("invalid bounded unique Xet map"));
                    }
                }
                Ok(UniqueMap(result))
            }
        }
        deserializer.deserialize_map(UniqueVisitor(std::marker::PhantomData))
    }
}
fn invalid() -> Error {
    Error::Malformed("invalid bounded Xet reconstruction".into())
}
fn add(total: &mut usize, value: usize, maximum: usize) -> Result<()> {
    *total = total
        .checked_add(value)
        .filter(|v| *v <= maximum)
        .ok_or_else(|| Error::Limit("Xet reconstruction count bound".into()))?;
    Ok(())
}
fn span(span: ChunkSpan) -> Result<()> {
    if span.start >= span.end || span.end > 8192 {
        Err(invalid())
    } else {
        Ok(())
    }
}
impl Reconstruction {
    pub(super) fn parse(
        raw: &[u8],
        v1: bool,
        file: XetHash,
        bytes: u64,
        options: &XetOptions,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        check_cancel(cancel)?;
        // JSON body and parsing/canonical controls have already been admitted.
        if raw.len() > 8 * 1024 * 1024 {
            return Err(Error::Limit("Xet metadata body bound".into()));
        }
        let (offset, raw_terms, raw_groups) = if v1 {
            let response: RawV1 = serde_json::from_slice(raw).map_err(|_| invalid())?;
            let groups = response
                .fetch_info
                .0
                .into_iter()
                .map(|(key, fetches)| {
                    (
                        key,
                        fetches
                            .into_iter()
                            .map(|f| RawGroup {
                                url: f.url,
                                ranges: vec![Descriptor {
                                    chunks: f.range,
                                    bytes: f.url_range,
                                }],
                            })
                            .collect(),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            (response.offset_into_first_range, response.terms, groups)
        } else {
            let response: RawV2 = serde_json::from_slice(raw).map_err(|_| invalid())?;
            (
                response.offset_into_first_range,
                response.terms,
                response.xorbs.0,
            )
        };
        if offset != 0 || raw_terms.len() > options.max_terms {
            return Err(invalid());
        }
        if bytes == 0 && (!raw_terms.is_empty() || !raw_groups.is_empty() || file != XetHash::ZERO)
        {
            return Err(invalid());
        }
        let mut needed = BTreeMap::<[u8; 32], Vec<ChunkSpan>>::new();
        let mut terms = Vec::new();
        let mut total = 0u64;
        let mut leaves = 0usize;
        let mut contract = Sha256::new();
        contract.update(b"zerogravity.xet.full-reconstruction.v1\0");
        contract.update(file.raw());
        contract.update(bytes.to_le_bytes());
        contract.update((raw_terms.len() as u64).to_le_bytes());
        for term in raw_terms {
            check_cancel(cancel)?;
            span(term.range)?;
            let chunks = (term.range.end - term.range.start) as usize;
            add(&mut leaves, chunks, options.max_chunks)?;
            let maximum = (chunks as u64)
                .checked_mul(128 * 1024)
                .ok_or_else(invalid)?;
            if u64::from(term.unpacked_length) > maximum {
                return Err(invalid());
            }
            total = total
                .checked_add(u64::from(term.unpacked_length))
                .filter(|total| *total <= bytes)
                .ok_or_else(invalid)?;
            let hash = XetHash::from_text(&term.hash)?;
            needed.entry(*hash.raw()).or_default().push(term.range);
            contract.update(hash.raw());
            contract.update(term.range.start.to_le_bytes());
            contract.update(term.range.end.to_le_bytes());
            contract.update(term.unpacked_length.to_le_bytes());
            terms.push(Term {
                hash,
                unpacked: u64::from(term.unpacked_length),
                chunks: term.range,
            });
        }
        if total != bytes || (bytes != 0 && terms.is_empty()) {
            return Err(invalid());
        }
        let mut groups = BTreeMap::new();
        let mut locations = BTreeMap::new();
        let mut group_count = 0usize;
        let mut range_count = 0usize;
        for (key, fetches) in raw_groups {
            check_cancel(cancel)?;
            let hash = XetHash::from_text(&key)?;
            if !needed.contains_key(hash.raw()) || fetches.is_empty() {
                return Err(invalid());
            }
            add(&mut group_count, fetches.len(), options.max_fetch_groups)?;
            let mut all_descriptors = Vec::new();
            let mut parsed = Vec::new();
            for (group_index, group) in fetches.into_iter().enumerate() {
                if group.ranges.is_empty() {
                    return Err(invalid());
                }
                add(&mut range_count, group.ranges.len(), options.max_ranges)?;
                let mut previous = None::<Descriptor>;
                for (range_index, descriptor) in group.ranges.iter().enumerate() {
                    span(descriptor.chunks)?;
                    if descriptor.bytes.end >= 64 * 1024 * 1024 {
                        return Err(invalid());
                    }
                    descriptor.bytes.bytes()?;
                    if previous.is_some_and(|p| {
                        p.chunks.end > descriptor.chunks.start
                            || p.bytes.end >= descriptor.bytes.start
                    }) {
                        return Err(invalid());
                    }
                    previous = Some(*descriptor);
                    all_descriptors.push(Location {
                        descriptor: *descriptor,
                        group: group_index,
                        range: range_index,
                    });
                }
                validate_signed_url(&group.url, &group.ranges, options)?;
                parsed.push(Group {
                    url: group.url,
                    ranges: group.ranges,
                });
            }
            all_descriptors.sort_unstable_by_key(|location| location.descriptor.chunks.start);
            for descriptors in all_descriptors.windows(2) {
                if descriptors[0].descriptor.chunks.end > descriptors[1].descriptor.chunks.start
                    || descriptors[0].descriptor.bytes.end >= descriptors[1].descriptor.bytes.start
                {
                    return Err(invalid());
                }
            }
            for chunks in needed.get(hash.raw()).ok_or_else(invalid)? {
                let mut cursor = chunks.start;
                while cursor < chunks.end {
                    check_cancel(cancel)?;
                    let position = all_descriptors
                        .partition_point(|d| d.descriptor.chunks.start <= cursor)
                        .checked_sub(1)
                        .ok_or_else(invalid)?;
                    let descriptor = all_descriptors[position].descriptor;
                    if descriptor.chunks.end <= cursor {
                        return Err(invalid());
                    }
                    cursor = descriptor.chunks.end.min(chunks.end);
                }
            }
            groups.insert(*hash.raw(), parsed);
            locations.insert(*hash.raw(), all_descriptors);
        }
        if groups.len() != needed.len() {
            return Err(invalid());
        }
        check_cancel(cancel)?;
        Ok(Self {
            terms,
            groups,
            locations,
            contract: contract.finalize().into(),
        })
    }
    pub(super) fn locate(&self, hash: XetHash, chunk: u32) -> Result<(usize, usize)> {
        let locations = self.locations.get(hash.raw()).ok_or_else(invalid)?;
        let position = locations
            .partition_point(|location| location.descriptor.chunks.start <= chunk)
            .checked_sub(1)
            .ok_or_else(invalid)?;
        let location = &locations[position];
        if chunk >= location.descriptor.chunks.end {
            return Err(invalid());
        }
        Ok((location.group, location.range))
    }
}
fn validate_signed_url(value: &str, ranges: &[Descriptor], options: &XetOptions) -> Result<()> {
    if value.len() > 16 * 1024
        || !value.bytes().all(|byte| byte.is_ascii_graphic())
        || value.contains(['#', '\\'])
    {
        return Err(invalid());
    }
    let parsed = super::super::http::parse_url(value, options.endpoints.diagnostic)?;
    let uri: ureq::http::Uri = value.parse().map_err(|_| invalid())?;
    if uri.authority().is_none() || uri.path().is_empty() || parsed.query().is_none() {
        return Err(invalid());
    }
    let mut signed = None;
    for parameter in parsed.query().ok_or_else(invalid)?.split('&') {
        let (name, value) = parameter.split_once('=').ok_or_else(invalid)?;
        if decode_query(name)? == "X-Xet-Signed-Range" {
            if signed.is_some() {
                return Err(invalid());
            }
            signed = Some(decode_query(value)?);
        }
    }
    let header = range_header(ranges)?;
    if signed.as_deref() != Some(header.as_str()) {
        return Err(invalid());
    }
    Ok(())
}
pub(super) fn range_header(ranges: &[Descriptor]) -> Result<String> {
    use std::fmt::Write;
    let mut output = String::from("bytes=");
    for (index, range) in ranges.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{}-{}", range.bytes.start, range.bytes.end).map_err(|_| invalid())?;
        if output.len() > 16 * 1024 {
            return Err(Error::Limit("Xet signed Range header bound".into()));
        }
    }
    Ok(output)
}
fn decode_query(value: &str) -> Result<String> {
    let mut result = Vec::new();
    let mut input = value.bytes();
    while let Some(byte) = input.next() {
        result.push(if byte == b'%' {
            let high = input
                .next()
                .and_then(|b| (b as char).to_digit(16))
                .ok_or_else(invalid)?;
            let low = input
                .next()
                .and_then(|b| (b as char).to_digit(16))
                .ok_or_else(invalid)?;
            (high * 16 + low) as u8
        } else {
            byte
        });
    }
    String::from_utf8(result).map_err(|_| invalid())
}
