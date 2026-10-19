//! Verified immutable SSD extents. Not yet a model TensorSource.
//!
//! One pinned source file descriptor and SHA256 per chunk; one bounded,
//! contiguous positional read group and physical staging lease. No mmap,
//! untrusted cache, dynamic model fallback, GPU submission or device readiness.

use crate::cache::TransferAdmission;
use crate::{Error, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};

const SCHEMA: &str = "zerogravity.extents.v1";
const MAX_MANIFEST: usize = 16 * 1024 * 1024;
const MAX_CHUNKS: usize = 200_000;
const MAX_CHUNK: usize = 4 * 1024 * 1024;
const MAX_READ: usize = 32 * 1024 * 1024;
const MAX_READ_CHUNKS: usize = 32;
const MAX_FILE: u64 = 1024 * 1024 * 1024 * 1024;

fn digest_ok(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtentManifest {
    schema: String,
    file_bytes: u64,
    // Whole-file expected SHA is a manifest claim, not verified on open.
    whole_sha256: String,
    chunks: Vec<ExtentChunk>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtentChunk {
    offset: u64,
    bytes: usize,
    sha256: String,
}
pub struct ExtentSource {
    file: Arc<File>,
    file_bytes: u64,
    index_sha256: String,
    claimed_whole_sha256: String,
    chunks: Vec<ExtentChunk>,
}
pub struct StagedExtents {
    bytes: Vec<u8>,
    offsets: Vec<(usize, Range<usize>)>,
    // Hold until any consuming device has an actual completion receipt.
    _lease: GlobalResourceLease,
}
impl StagedExtents {
    pub fn bytes_for(&self, index: usize) -> Option<&[u8]> {
        let (_, range) = self.offsets.iter().find(|(i, _)| *i == index)?;
        self.bytes.get(range.clone())
    }
    pub fn len(&self) -> usize {
        self.bytes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
    pub fn count(&self) -> usize {
        self.offsets.len()
    }
}
/// Immutable, physical-admission-pinned tensor/window bytes.
pub struct StagedSlice {
    stage: StagedExtents,
    range: Range<usize>,
    source_offset: u64,
}
impl StagedSlice {
    pub fn as_slice(&self) -> &[u8] {
        &self.stage.bytes[self.range.clone()]
    }
    pub fn source_offset(&self) -> u64 {
        self.source_offset
    }
    pub fn charged_bytes(&self) -> usize {
        self.stage.len()
    }
}

impl ExtentSource {
    /// Manifest trust comes from caller-pinned independent SHA256.
    /// Requested ranges are verified before exposure, not at open time.
    pub fn from_open_file(
        file: File,
        manifest_json: &[u8],
        expected_manifest_sha256: &str,
    ) -> Result<Self> {
        if !digest_ok(expected_manifest_sha256)
            || manifest_json.is_empty()
            || manifest_json.len() > MAX_MANIFEST
            || format!("{:x}", Sha256::digest(manifest_json)) != expected_manifest_sha256
        {
            return Err(Error::Malformed(
                "SSD extent manifest authentication mismatch".into(),
            ));
        }
        let manifest: ExtentManifest = serde_json::from_slice(manifest_json)
            .map_err(|_| Error::Malformed("invalid strict immutable extent index".into()))?;
        if manifest.schema != SCHEMA {
            return Err(Error::Unsupported(
                "immutable source extent manifest version".into(),
            ));
        }
        if !digest_ok(&manifest.whole_sha256)
            || manifest.file_bytes == 0
            || manifest.file_bytes > MAX_FILE
            || manifest.chunks.is_empty()
            || manifest.chunks.len() > MAX_CHUNKS
        {
            return Err(Error::Limit("extent source geometry, SHA or count".into()));
        }
        if file.metadata()?.len() != manifest.file_bytes {
            return Err(Error::Malformed(
                "opened file length differs from pinned extent index".into(),
            ));
        }
        let mut next = 0u64;
        for chunk in &manifest.chunks {
            if chunk.offset != next
                || chunk.bytes == 0
                || chunk.bytes > MAX_CHUNK
                || !digest_ok(&chunk.sha256)
            {
                return Err(Error::Malformed(
                    "noncontiguous or unsafe immutable source extent".into(),
                ));
            }
            next = next
                .checked_add(chunk.bytes as u64)
                .ok_or_else(|| Error::Limit("source extent offset overflow".into()))?;
            if next > manifest.file_bytes {
                return Err(Error::Malformed("source extent exceeds file length".into()));
            }
        }
        if next != manifest.file_bytes {
            return Err(Error::Malformed("source extent index incomplete".into()));
        }
        Ok(Self {
            file: Arc::new(file),
            file_bytes: manifest.file_bytes,
            index_sha256: expected_manifest_sha256.into(),
            claimed_whole_sha256: manifest.whole_sha256,
            chunks: manifest.chunks,
        })
    }
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
    pub fn index_sha256(&self) -> &str {
        &self.index_sha256
    }
    pub fn claimed_whole_sha256(&self) -> &str {
        &self.claimed_whole_sha256
    }
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    /// Sorted unique source indices; adjacent chunks coalesce into one
    /// positional read. Stage memory charged before IO and allocation.
    /// Materialize only the source bytes needed by one exact tensor tile.
    /// Never returns a borrowed pointer to a mutable SSD inode. The
    /// returned StagedSlice retains its physical staging lease until Drop.
    /// Original trusted TensorInfo and source binding remain the
    /// higher-level model's responsibility (this is only a byte primitive).
    pub fn read_window(
        &self,
        offset: u64,
        bytes: usize,
        admission: &TransferAdmission,
    ) -> Result<StagedSlice> {
        let end = offset
            .checked_add(bytes as u64)
            .ok_or_else(|| Error::Limit("source tensor window byte offset overflow".into()))?;
        if bytes == 0 || bytes > MAX_READ || end > self.file_bytes {
            return Err(Error::Limit(
                "SSD tensor tile outside source or byte cap".into(),
            ));
        }
        let first = self
            .chunks
            .partition_point(|chunk| chunk.offset <= offset)
            .checked_sub(1)
            .ok_or_else(|| Error::Malformed("source first extent missing".into()))?;
        let last = self
            .chunks
            .partition_point(|chunk| chunk.offset < end)
            .checked_sub(1)
            .ok_or_else(|| Error::Malformed("source last extent missing".into()))?;
        if last < first || last - first + 1 > MAX_READ_CHUNKS {
            return Err(Error::Limit(
                "SSD tensor window crosses too many chunks".into(),
            ));
        }
        let indices = (first..=last).collect::<Vec<_>>();
        let stage = self.read_verified(&indices, admission)?;
        let start = usize::try_from(offset - self.chunks[first].offset)
            .map_err(|_| Error::Limit("source local tile offset overflow".into()))?;
        let tail = start
            .checked_add(bytes)
            .ok_or_else(|| Error::Limit("source tile length overflow".into()))?;
        if tail > stage.len() {
            return Err(Error::State("source tile staging range incomplete".into()));
        }
        Ok(StagedSlice {
            stage,
            range: start..tail,
            source_offset: offset,
        })
    }

    pub fn read_verified(
        &self,
        indices: &[usize],
        admission: &TransferAdmission,
    ) -> Result<StagedExtents> {
        if indices.is_empty()
            || indices.len() > MAX_READ_CHUNKS
            || indices.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(Error::Limit(
                "SSD request indices not strictly increasing or unbounded".into(),
            ));
        }
        let mut count = 0usize;
        for index in indices {
            let chunk = self
                .chunks
                .get(*index)
                .ok_or_else(|| Error::Malformed("unknown source extent index".into()))?;
            count = count
                .checked_add(chunk.bytes)
                .ok_or_else(|| Error::Limit("SSD staged bytes overflow".into()))?;
            if count > MAX_READ {
                return Err(Error::Limit(
                    "SSD staged read exceeds bounded capacity".into(),
                ));
            }
        }
        let lease = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: count as u64,
            }],
        )?;
        let mut bytes = vec![0u8; count];
        let mut ranges = Vec::with_capacity(indices.len());
        let mut offset = 0usize;
        for index in indices {
            let expected = &self.chunks[*index];
            let end = offset + expected.bytes;
            ranges.push((*index, offset..end));
            offset = end;
        }
        if self.file.metadata()?.len() != self.file_bytes {
            return Err(Error::Malformed(
                "SSD source descriptor size mutated".into(),
            ));
        }
        let mut start = 0usize;
        while start < indices.len() {
            let mut end = start + 1;
            while end < indices.len() && indices[end] == indices[end - 1] + 1 {
                end += 1;
            }
            let range = ranges[start].1.start..ranges[end - 1].1.end;
            self.file
                .read_exact_at(&mut bytes[range], self.chunks[indices[start]].offset)?;
            start = end;
        }
        for (index, range) in &ranges {
            let observed = format!("{:x}", Sha256::digest(&bytes[range.clone()]));
            if observed != self.chunks[*index].sha256 {
                return Err(Error::Malformed(
                    "SSD read changed expected source bytes".into(),
                ));
            }
        }
        Ok(StagedExtents {
            bytes,
            offsets: ranges,
            _lease: lease,
        })
    }
}
