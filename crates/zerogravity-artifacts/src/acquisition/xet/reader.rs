//! One sequential native source reader. It emits selected logical chunks only,
//! while every decoded transport edge/refetch still consumes the work budget.
use super::budget::Budget;
use super::codec::{self, ChunkHeader};
use super::hash::{self, XetHash};
use super::multipart::RangeBody;
use super::reconstruction::{ChunkSpan, Group, Reconstruction};
use super::{XetOptions, check_cancel};
use crate::acquisition::http::HttpResponse;
use crate::{Error, Result};
use std::io::{self, Read};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn invalid() -> Error {
    Error::Malformed("invalid Xet logical reconstruction".into())
}

/// Private source boundary, implemented by the original HTTP owner adapter.
/// No grant, token, signed URL or arbitrary header getter is exposed publicly.
pub(super) trait RangeSource {
    fn fetch(&mut self, group: &Group) -> Result<HttpResponse>;
    fn refresh(&mut self) -> Result<Reconstruction>;
}

/// The caller must retain the existing admission lease BEFORE constructing
/// these vectors, and through the reader's final source EOF/drop.
pub(super) struct Scratch {
    compressed: Vec<u8>,
    raw: Vec<u8>,
    grouped: Vec<u8>,
    leaves: Vec<(XetHash, u64)>,
}
impl Scratch {
    pub(super) fn new(options: &XetOptions) -> Result<Self> {
        fn buffer(bytes: usize) -> Result<Vec<u8>> {
            let mut result = Vec::new();
            result
                .try_reserve_exact(bytes)
                .map_err(|_| Error::Limit("Xet scratch allocation refused".into()))?;
            result.resize(bytes, 0);
            Ok(result)
        }
        options.validate()?;
        let mut leaves = Vec::new();
        leaves
            .try_reserve_exact(options.max_chunks)
            .map_err(|_| Error::Limit("Xet leaf allocation refused".into()))?;
        Ok(Self {
            compressed: buffer(codec::MAX_COMPRESSED_BYTES)?,
            raw: buffer(codec::MAX_RAW_BYTES)?,
            grouped: buffer(codec::MAX_RAW_BYTES)?,
            leaves,
        })
    }
    pub(super) fn reservation(options: &XetOptions, metadata: usize) -> Result<u64> {
        options.validate()?;
        (options.max_chunks as u64)
            .checked_mul(40)
            .and_then(|v| v.checked_add(512 * 1024 + 64 * 1024 + 16 * 1024 * 1024))
            .and_then(|v| {
                (metadata as u64)
                    .checked_mul(8)
                    .and_then(|m| v.checked_add(m))
            })
            .ok_or_else(|| Error::Limit("Xet admitted scratch bound".into()))
    }
}

struct Pending {
    group: usize,
    range: usize,
    selected: ChunkSpan,
    retries: u8,
}
struct Fetch<'a> {
    body: RangeBody<'a>,
    part: Option<(usize, u32)>,
    selected_seen: bool,
    decoded_bytes: u64,
}
pub(in crate::acquisition) struct XetReader<'a> {
    source: Box<dyn RangeSource + Send + 'a>,
    reconstruction: Reconstruction,
    file: XetHash,
    expected: u64,
    budget: Arc<Budget>,
    scratch: &'a mut Scratch,
    cancel: &'a AtomicBool,
    term: usize,
    term_leaf_start: usize,
    cursor: u32,
    scan: u32,
    term_bytes: u64,
    file_bytes: u64,
    pending: Option<Pending>,
    fetch: Option<Fetch<'a>>,
    raw_position: usize,
    raw_length: usize,
    complete: bool,
    poisoned: bool,
}
impl<'a> XetReader<'a> {
    pub(super) fn new(
        source: Box<dyn RangeSource + Send + 'a>,
        reconstruction: Reconstruction,
        identity: (XetHash, u64),
        budget: Arc<Budget>,
        scratch: &'a mut Scratch,
        cancel: &'a AtomicBool,
    ) -> Result<Self> {
        check_cancel(cancel)?;
        scratch.leaves.clear();
        let first = reconstruction
            .terms
            .first()
            .map_or(0, |term| term.chunks.start);
        Ok(Self {
            source,
            reconstruction,
            file: identity.0,
            expected: identity.1,
            budget,
            scratch,
            cancel,
            term: 0,
            term_leaf_start: 0,
            cursor: first,
            scan: first,
            term_bytes: 0,
            file_bytes: 0,
            pending: None,
            fetch: None,
            raw_position: 0,
            raw_length: 0,
            complete: false,
            poisoned: false,
        })
    }
    fn reopen(&mut self) -> Result<()> {
        check_cancel(self.cancel)?;
        let pending = self.pending.as_ref().ok_or_else(invalid)?;
        let hash = self
            .reconstruction
            .terms
            .get(self.term)
            .ok_or_else(invalid)?
            .hash;
        let group = self
            .reconstruction
            .groups
            .get(hash.raw())
            .and_then(|v| v.get(pending.group))
            .ok_or_else(invalid)?;
        let response = self.source.fetch(group)?;
        if response.status == 403 {
            // The refresh owner counts this against the combined finite grant
            // reissue budget; no untrusted error response body is read.
            self.fetch = None;
            let refreshed = self.source.refresh()?;
            if refreshed.contract != self.reconstruction.contract {
                return Err(invalid());
            }
            self.reconstruction = refreshed;
            self.pending = None;
            self.scan = self.reconstruction.terms[self.term].chunks.start;
            return Ok(());
        }
        self.fetch = Some(Fetch {
            body: RangeBody::new(response, &group.ranges, self.cancel, self.budget.clone())?,
            part: None,
            selected_seen: false,
            decoded_bytes: 0,
        });
        Ok(())
    }
    fn begin(&mut self) -> Result<()> {
        let term = self
            .reconstruction
            .terms
            .get(self.term)
            .ok_or_else(invalid)?;
        let (group, range) = self.reconstruction.locate(term.hash, self.scan)?;
        let descriptor = self.reconstruction.groups[term.hash.raw()][group].ranges[range];
        self.pending = Some(Pending {
            group,
            range,
            selected: ChunkSpan {
                start: self.scan,
                end: descriptor.chunks.end.min(term.chunks.end),
            },
            retries: 0,
        });
        self.reopen()
    }
    fn advance_term(&mut self) -> Result<()> {
        let term = &self.reconstruction.terms[self.term];
        if self.scan != term.chunks.end
            || self.cursor != term.chunks.end
            || self.term_bytes != term.unpacked
        {
            return Err(invalid());
        }
        self.term += 1;
        self.term_bytes = 0;
        self.term_leaf_start = self.scratch.leaves.len();
        if let Some(term) = self.reconstruction.terms.get(self.term) {
            self.scan = term.chunks.start;
            self.cursor = term.chunks.start;
        }
        Ok(())
    }
    fn finish_source(&mut self) -> Result<()> {
        check_cancel(self.cancel)?;
        if self.file_bytes != self.expected
            || hash::file_hash(&mut self.scratch.leaves, self.cancel)? != self.file
        {
            return Err(invalid());
        }
        self.complete = true;
        Ok(())
    }
    /// Decode/drain until one complete new logical chunk is available. Already
    /// emitted chunks on a reissue are compared, without duplicate logical credit.
    fn fill_once(&mut self) -> Result<()> {
        loop {
            check_cancel(self.cancel)?;
            if self.term == self.reconstruction.terms.len() {
                return self.finish_source();
            }
            if self.fetch.is_none() {
                if self.pending.is_some() {
                    self.reopen()?;
                } else {
                    self.begin()?;
                }
                if self.fetch.is_none() {
                    continue;
                }
            }
            let pending = self.pending.as_ref().ok_or_else(invalid)?;
            let term = &self.reconstruction.terms[self.term];
            let group = &self.reconstruction.groups[term.hash.raw()][pending.group];
            let fetch = self.fetch.as_mut().ok_or_else(invalid)?;
            if fetch.part.is_none() {
                let Some(index) = fetch.body.next_part()? else {
                    if !fetch.selected_seen || self.cursor < pending.selected.end {
                        return Err(invalid());
                    }
                    self.scan = pending.selected.end;
                    self.fetch = None;
                    self.pending = None;
                    if self.scan == self.reconstruction.terms[self.term].chunks.end {
                        self.advance_term()?;
                    }
                    continue;
                };
                let descriptor = group.ranges.get(index).ok_or_else(invalid)?;
                fetch.selected_seen |= index == pending.range;
                fetch.part = Some((index, descriptor.chunks.start));
            }
            let (index, chunk) = fetch.part.ok_or_else(invalid)?;
            let descriptor = group.ranges.get(index).ok_or_else(invalid)?;
            if chunk == descriptor.chunks.end {
                if fetch.body.remaining() != 0 {
                    return Err(invalid());
                }
                fetch.part = None;
                continue;
            }
            let mut header = [0u8; codec::HEADER_BYTES];
            fetch.body.read_exact(&mut header)?;
            let header = ChunkHeader::parse(header)?;
            fetch
                .body
                .read_exact(&mut self.scratch.compressed[..header.compressed_bytes()])?;
            self.budget.decode_chunk(self.cancel)?;
            let bytes = codec::decode_chunk(
                header,
                &self.scratch.compressed[..header.compressed_bytes()],
                &mut self.scratch.raw,
                &mut self.scratch.grouped,
                self.cancel,
            )?;
            fetch.decoded_bytes = fetch
                .decoded_bytes
                .checked_add(bytes as u64)
                .filter(|n| *n <= 64 * 1024 * 1024)
                .ok_or_else(invalid)?;
            fetch.part = Some((index, chunk + 1));
            if index != pending.range
                || chunk < pending.selected.start
                || chunk >= pending.selected.end
            {
                continue;
            }
            let hash = hash::chunk_hash(&self.scratch.raw[..bytes], self.cancel)?;
            if chunk < self.cursor {
                let index = self.term_leaf_start + (chunk - term.chunks.start) as usize;
                if self.scratch.leaves.get(index) != Some(&(hash, bytes as u64)) {
                    return Err(invalid());
                }
                continue;
            }
            if chunk != self.cursor {
                return Err(invalid());
            }
            let term_bytes = self
                .term_bytes
                .checked_add(bytes as u64)
                .filter(|n| *n <= term.unpacked)
                .ok_or_else(invalid)?;
            let file_bytes = self
                .file_bytes
                .checked_add(bytes as u64)
                .filter(|n| *n <= self.expected)
                .ok_or_else(invalid)?;
            self.budget.file_leaf(self.cancel)?;
            if self.scratch.leaves.len() == self.scratch.leaves.capacity() {
                return Err(invalid());
            }
            self.scratch.leaves.push((hash, bytes as u64));
            self.cursor += 1;
            self.term_bytes = term_bytes;
            self.file_bytes = file_bytes;
            self.raw_position = 0;
            self.raw_length = bytes;
            if bytes != 0 {
                return Ok(());
            }
        }
    }
    fn fill(&mut self) -> Result<()> {
        loop {
            match self.fill_once() {
                Ok(()) => return Ok(()),
                Err(Error::Io(_)) => {
                    check_cancel(self.cancel)?;
                    self.budget.ensure_live()?;
                    let pending = self.pending.as_mut().ok_or_else(invalid)?;
                    if pending.retries == 2 {
                        return Err(Error::State("Xet range retry bound exhausted".into()));
                    }
                    pending.retries += 1;
                    self.fetch = None;
                    // The next complete group request repeats every authorized
                    // interval. All refetched chunks/bytes/attempts are charged.
                }
                Err(error) => return Err(error),
            }
        }
    }
    fn read_result(&mut self, output: &mut [u8]) -> Result<usize> {
        check_cancel(self.cancel)?;
        if output.is_empty() {
            return Ok(0);
        }
        if self.poisoned {
            return Err(Error::State("Xet source reader failed".into()));
        }
        if self.raw_position == self.raw_length && !self.complete {
            self.fill()?;
        }
        if self.complete {
            return Ok(0);
        }
        let count = output.len().min(self.raw_length - self.raw_position);
        for (destination, source) in output[..count]
            .chunks_mut(4096)
            .zip(self.scratch.raw[self.raw_position..self.raw_position + count].chunks(4096))
        {
            check_cancel(self.cancel)?;
            destination.copy_from_slice(source);
        }
        self.raw_position += count;
        Ok(count)
    }
}
impl Read for XetReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.read_result(output).map_err(|_| {
            self.poisoned = true;
            io::Error::other("Xet source verification or transfer failed")
        })
    }
}
