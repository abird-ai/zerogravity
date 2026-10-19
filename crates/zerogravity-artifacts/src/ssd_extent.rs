//! Verified immutable SSD extents. Not yet a model TensorSource.
//!
//! One pinned source file descriptor and SHA256 per chunk; one bounded,
//! contiguous positional read group and physical staging lease. No mmap,
//! untrusted cache, dynamic model fallback, GPU submission or device readiness.

use crate::cache::TransferAdmission;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExtentManifest {
    schema: String,
    file_bytes: u64,
    // Whole-file expected SHA is a manifest claim, not verified on open.
    whole_sha256: String,
    chunks: Vec<ExtentChunk>,
}
#[derive(Deserialize, Serialize)]
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
    charged_bytes: usize,
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
    /// Owned data plus retained offset-vector storage, excluding allocator
    /// rounding and generic arbiter bookkeeping.
    pub fn charged_bytes(&self) -> usize {
        self.charged_bytes
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
        self.stage.charged_bytes()
    }
}

impl ExtentSource {
    pub(crate) fn scan_chunk_count(file_bytes: u64, chunk_bytes: usize) -> Result<usize> {
        if file_bytes == 0 || file_bytes > MAX_FILE || chunk_bytes == 0 || chunk_bytes > MAX_CHUNK {
            return Err(Error::Limit("verified source scan geometry".into()));
        }
        let count = file_bytes.div_ceil(chunk_bytes as u64);
        if count > MAX_CHUNKS as u64 {
            return Err(Error::Limit("verified source scan chunk count".into()));
        }
        Ok(count as usize)
    }

    /// Internal source factory: all chunk hashes come from the same bounded
    /// scan whose whole-file digest matches the independently pinned original.
    pub(crate) fn scan_verified(
        file: File,
        expected_bytes: u64,
        expected_sha256: &str,
        chunk_bytes: usize,
        admission: &TransferAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        let count = Self::scan_chunk_count(expected_bytes, chunk_bytes)?;
        if !digest_ok(expected_sha256) || file.metadata()?.len() != expected_bytes {
            return Err(Error::Malformed(
                "verified source descriptor length/digest".into(),
            ));
        }
        source_not_cancelled(cancelled)?;
        let buffer_bytes = (chunk_bytes as u64).min(expected_bytes) as usize;
        let _staging = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: buffer_bytes as u64,
            }],
        )?;
        let mut buffer = vec![0u8; buffer_bytes];
        let mut chunks = Vec::with_capacity(count);
        let mut whole = Sha256::new();
        let mut offset = 0u64;
        while offset < expected_bytes {
            source_not_cancelled(cancelled)?;
            let bytes = (buffer_bytes as u64).min(expected_bytes - offset) as usize;
            file.read_exact_at(&mut buffer[..bytes], offset)?;
            whole.update(&buffer[..bytes]);
            chunks.push(ExtentChunk {
                offset,
                bytes,
                sha256: format!("{:x}", Sha256::digest(&buffer[..bytes])),
            });
            offset += bytes as u64;
        }
        source_not_cancelled(cancelled)?;
        if file.metadata()?.len() != expected_bytes
            || format!("{:x}", whole.finalize()) != expected_sha256
        {
            return Err(Error::Malformed(
                "original source whole-file SHA256 mismatch".into(),
            ));
        }
        let manifest = ExtentManifest {
            schema: SCHEMA.into(),
            file_bytes: expected_bytes,
            whole_sha256: expected_sha256.into(),
            chunks,
        };
        let manifest_json = serde_json::to_vec(&manifest)
            .map_err(|_| Error::State("verified extent manifest serialization".into()))?;
        if manifest_json.len() > MAX_MANIFEST {
            return Err(Error::Limit("verified extent manifest byte cap".into()));
        }
        Ok(Self {
            file: Arc::new(file),
            file_bytes: expected_bytes,
            index_sha256: format!("{:x}", Sha256::digest(&manifest_json)),
            claimed_whole_sha256: manifest.whole_sha256,
            chunks: manifest.chunks,
        })
    }

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
        let selected = last - first + 1;
        let mut indices = [0usize; MAX_READ_CHUNKS];
        for (index, slot) in indices[..selected].iter_mut().enumerate() {
            *slot = first + index;
        }
        let stage = self.read_verified(&indices[..selected], admission)?;
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
        let offset_bytes = indices
            .len()
            .checked_mul(std::mem::size_of::<(usize, Range<usize>)>())
            .ok_or_else(|| Error::Limit("SSD staged offset metadata overflow".into()))?;
        let charged_bytes = count
            .checked_add(offset_bytes)
            .ok_or_else(|| Error::Limit("SSD staging total charge overflow".into()))?;
        let lease = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: charged_bytes as u64,
            }],
        )?;
        let mut bytes = vec![0u8; count];
        let mut ranges = Vec::with_capacity(indices.len());
        if bytes.capacity() != count || ranges.capacity() != indices.len() {
            return Err(Error::State(
                "SSD staging allocator capacity changed".into(),
            ));
        }
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
            charged_bytes,
            _lease: lease,
        })
    }
}

pub(crate) fn source_not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("source window operation cancelled".into()));
    }
    Ok(())
}
