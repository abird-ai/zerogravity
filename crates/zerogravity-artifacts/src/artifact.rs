//! Bounded source-artifact directory readers. Never loads whole checkpoints.
//!
//! Supported today: safetensors F32/F16/BF16 and selected GGUF v3 tensor storage
//! layouts. No tokenizer, model-specific tensor naming, trained math or complete
//! model-shard verification is implied by successful directory inspection.

use crate::quant::{QuantKind, dequantize};
use crate::{Error, Result};
use serde::Deserialize;
use serde::de::{MapAccess, Visitor};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;

const MAX_HEADER: u64 = 64 * 1024 * 1024;
const MAX_METADATA: u64 = 128 * 1024 * 1024;
const MAX_TENSORS: usize = 200_000;
const MAX_NAME: usize = 65_535;
const MAX_STRING: u64 = 8 * 1024 * 1024;
const MAX_ARRAY_ITEMS: u64 = 4_000_000;
const MAX_REFERENCE_READ: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactFormat {
    Safetensors,
    Gguf,
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Logical tensor shape in source-file dimension order.
    pub shape: Vec<u64>,
    pub kind: QuantKind,
    /// Absolute byte offset in this file (not a GGUF relative tensor offset).
    pub byte_offset: u64,
    pub byte_len: u64,
}
#[derive(Debug)]
pub struct ArtifactInfo {
    pub format: ArtifactFormat,
    pub file_len: u64,
    pub tensors: Vec<TensorInfo>,
    /// Only bounded high-value metadata is surfaced during Phase 1.
    pub architecture: Option<String>,
}

pub struct Artifact {
    file: File,
    pub info: ArtifactInfo,
}

// File::try_clone shares an OS cursor. Parse and materialize original opened
// cache descriptors through a local positioned view, never that shared cursor.
#[cfg(unix)]
struct PositionedFile<'a> {
    file: &'a File,
    position: u64,
}
#[cfg(unix)]
impl<'a> PositionedFile<'a> {
    fn new(file: &'a File) -> Self {
        Self { file, position: 0 }
    }
}
#[cfg(unix)]
impl Read for PositionedFile<'_> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let count = self.file.read_at(output, self.position)?;
        self.position = self
            .position
            .checked_add(count as u64)
            .ok_or_else(|| std::io::Error::other("artifact read position overflow"))?;
        Ok(count)
    }
}
#[cfg(unix)]
impl Seek for PositionedFile<'_> {
    fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
        self.position = match from {
            SeekFrom::Start(position) => Some(position),
            SeekFrom::Current(delta) => self.position.checked_add_signed(delta),
            SeekFrom::End(delta) => self.file.metadata()?.len().checked_add_signed(delta),
        }
        .ok_or_else(|| std::io::Error::other("invalid artifact seek position"))?;
        Ok(self.position)
    }
}

fn read_owned_bytes(reader: &mut impl Read, expected: u64, maximum: u64) -> Result<Vec<u8>> {
    if expected > maximum {
        return Err(Error::Limit(
            "source artifact exceeds CPU byte budget".into(),
        ));
    }
    let mut source = Vec::new();
    source
        .try_reserve_exact(
            usize::try_from(expected)
                .map_err(|_| Error::Limit("source allocation exceeds address space".into()))?,
        )
        .map_err(|_| Error::Limit("source allocation refused".into()))?;
    source.resize(expected as usize, 0);
    reader.read_exact(&mut source)?;
    if reader.read(&mut [0; 1])? != 0 {
        return Err(Error::Malformed(
            "source length changed while reading opened artifact".into(),
        ));
    }
    Ok(source)
}

#[cfg(all(feature = "cache", target_os = "linux"))]
fn verify_owned_bytes(source: &[u8], expected_sha256: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    if format!("{:x}", Sha256::digest(source)) != expected_sha256 {
        return Err(Error::Malformed(
            "owned cache source digest mismatch".into(),
        ));
    }
    Ok(())
}

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) fn copy_verified_file(
    file: &File,
    expected_bytes: u64,
    expected_sha256: &str,
    maximum: u64,
) -> Result<Vec<u8>> {
    crate::cache::paths::validate_digest(expected_sha256)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != expected_bytes {
        return Err(Error::Malformed(
            "cache source type or length changed".into(),
        ));
    }
    let source = read_owned_bytes(&mut PositionedFile::new(file), expected_bytes, maximum)?;
    verify_owned_bytes(&source, expected_sha256)?;
    Ok(source)
}

impl Artifact {
    #[cfg(all(feature = "cache", target_os = "linux"))]
    pub(crate) fn into_open_parts(self) -> (ArtifactInfo, File) {
        (self.info, self.file)
    }

    pub fn open(path: &Path) -> Result<Self> {
        Self::from_file(File::open(path)?, path)
    }

    /// Inspect this exact opened descriptor using its original logical name
    /// only as a checked format hint. No pathname is reopened by this method.
    pub fn from_file(file: File, logical_name: &Path) -> Result<Self> {
        #[cfg(not(unix))]
        let mut file = file;
        let len = file.metadata()?.len();
        if !file.metadata()?.is_file() {
            return Err(Error::Malformed(
                "artifact descriptor is not a regular file".into(),
            ));
        }
        if len < 8 {
            return Err(Error::Malformed("file too small".into()));
        }
        let mut magic = [0; 4];
        #[cfg(unix)]
        let mut reader = PositionedFile::new(&file);
        #[cfg(not(unix))]
        let mut reader = &mut file;
        reader.read_exact(&mut magic)?;
        reader.seek(SeekFrom::Start(0))?;
        let ext = logical_name
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();
        let info = match (ext, magic == *b"GGUF") {
            ("gguf", true) => parse_gguf(&mut reader, len)?,
            ("safetensors", false) => parse_safetensors(&mut reader, len)?,
            ("gguf", false) => return Err(Error::Malformed("GGUF magic mismatch".into())),
            ("safetensors", true) => {
                return Err(Error::Malformed("safetensors has GGUF magic".into()));
            }
            _ => {
                return Err(Error::Unsupported(
                    "expected a .safetensors or .gguf file".into(),
                ));
            }
        };
        Ok(Self { file, info })
    }

    /// Consume the EXACT file descriptor used for metadata inspection and
    /// re-parse its complete bounded in-memory byte snapshot. Reopening the
    /// pathname after inspection would allow same-length atomic file
    /// replacement to silently mix one artifact's tensor geometry with the
    /// bytes of another artifact. This method does not claim a simultaneous
    /// writer cannot mutate a file *during* the read; callers must provide
    /// immutable source checkpoints.
    pub fn into_bounded_source(self, maximum: u64) -> Result<(ArtifactInfo, Vec<u8>)> {
        #[cfg(unix)]
        let mut reader = PositionedFile::new(&self.file);
        #[cfg(not(unix))]
        let mut file = self.file;
        #[cfg(not(unix))]
        let mut reader = {
            file.seek(SeekFrom::Start(0))?;
            &mut file
        };
        let source = read_owned_bytes(&mut reader, self.info.file_len, maximum)?;
        Self::reparse_source(self.info.format, source)
    }

    /// Hash the actual immutable owned allocation, then reparse those exact
    /// bytes. Cache verification before this copy cannot establish its digest.
    #[cfg(all(feature = "cache", target_os = "linux"))]
    pub fn into_verified_bounded_source(
        self,
        maximum: u64,
        expected_bytes: u64,
        expected_sha256: &str,
    ) -> Result<(ArtifactInfo, Vec<u8>)> {
        if self.info.file_len != expected_bytes {
            return Err(Error::Malformed("cache artifact length mismatch".into()));
        }
        let source = copy_verified_file(&self.file, expected_bytes, expected_sha256, maximum)?;
        Self::reparse_source(self.info.format, source)
    }

    fn reparse_source(format: ArtifactFormat, source: Vec<u8>) -> Result<(ArtifactInfo, Vec<u8>)> {
        let mut snapshot = Cursor::new(source.as_slice());
        let info = match format {
            ArtifactFormat::Gguf => parse_gguf(&mut snapshot, source.len() as u64)?,
            ArtifactFormat::Safetensors => parse_safetensors(&mut snapshot, source.len() as u64)?,
        };
        Ok((info, source))
    }

    /// Bounded reference tensor materialization; for large weight matrices a
    /// streaming/mapped view and physical admission will replace this method.
    pub fn read_reference_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .info
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::Malformed(format!("tensor '{name}' absent")))?;
        if tensor.byte_len > MAX_REFERENCE_READ {
            return Err(Error::Limit("reference tensor read exceeds 64 MiB".into()));
        }
        let elements = tensor
            .shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::Limit("tensor element count overflow".into()))?;
        let count =
            usize::try_from(elements).map_err(|_| Error::Limit("tensor too large".into()))?;
        let n = usize::try_from(tensor.byte_len)
            .map_err(|_| Error::Limit("tensor too large".into()))?;
        let mut bytes = vec![0; n];
        #[cfg(unix)]
        let mut reader = PositionedFile::new(&self.file);
        #[cfg(not(unix))]
        let mut reader = &mut self.file;
        reader.seek(SeekFrom::Start(tensor.byte_offset))?;
        reader.read_exact(&mut bytes)?;
        dequantize(tensor.kind, &bytes, count)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SafeTensorEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Custom root visitor rejects duplicate tensor keys. Deriving a map directly
/// would silently replace a duplicate name and could hide aliased source bytes.
struct SafeHeader {
    tensors: BTreeMap<String, SafeTensorEntry>,
}
impl<'de> Deserialize<'de> for SafeHeader {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = SafeHeader;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("safetensors header object")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut tensors = BTreeMap::new();
                let mut saw_metadata = false;
                while let Some(key) = map.next_key::<String>()? {
                    if key == "__metadata__" {
                        if saw_metadata {
                            return Err(serde::de::Error::custom("duplicate __metadata__"));
                        }
                        saw_metadata = true;
                        let _strings: BTreeMap<String, String> = map.next_value()?;
                    } else {
                        if tensors.contains_key(&key) {
                            return Err(serde::de::Error::custom("duplicate tensor name"));
                        }
                        if tensors.len() >= MAX_TENSORS {
                            return Err(serde::de::Error::custom("too many tensors"));
                        }
                        tensors.insert(key, map.next_value()?);
                    }
                }
                Ok(SafeHeader { tensors })
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

pub fn parse_safetensors<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<ArtifactInfo> {
    reader.seek(SeekFrom::Start(0))?;
    if file_len < 10 {
        return Err(Error::Malformed("safetensors missing header".into()));
    }
    let mut nbuf = [0; 8];
    reader.read_exact(&mut nbuf)?;
    let n = u64::from_le_bytes(nbuf);
    let data_start = 8u64
        .checked_add(n)
        .ok_or_else(|| Error::Limit("header overflow".into()))?;
    if n == 0 || n > MAX_HEADER || data_start > file_len {
        return Err(Error::Malformed("invalid safetensors header length".into()));
    }
    let mut json = vec![0u8; n as usize];
    reader.read_exact(&mut json)?;
    if json.first() != Some(&b'{') {
        return Err(Error::Malformed(
            "safetensors header must begin with '{'".into(),
        ));
    }
    let header: SafeHeader = serde_json::from_slice(&json)
        .map_err(|e| Error::Malformed(format!("safetensors JSON: {e}")))?;
    let data_len = file_len - data_start;
    let mut tensors = Vec::with_capacity(header.tensors.len());
    let mut occupied = Vec::with_capacity(header.tensors.len());
    for (name, tensor) in header.tensors {
        if name.is_empty() || name.len() > MAX_NAME || tensor.shape.len() > 16 {
            return Err(Error::Malformed("invalid tensor name or rank".into()));
        }
        let kind = QuantKind::from_safetensors(&tensor.dtype)?;
        let elements = checked_elements(&tensor.shape)?;
        let len = kind.byte_len(elements)?;
        let [begin, end] = tensor.data_offsets;
        if begin > end || end > data_len || end - begin != len {
            return Err(Error::Malformed(format!(
                "tensor '{name}' offsets/size disagree with shape"
            )));
        }
        let byte_offset = data_start
            .checked_add(begin)
            .ok_or_else(|| Error::Limit("offset overflow".into()))?;
        occupied.push((begin, end));
        tensors.push(TensorInfo {
            name,
            shape: tensor.shape,
            kind,
            byte_offset,
            byte_len: len,
        });
    }
    // Safetensors requires full coverage of the data section, not just in-bounds
    // tensors. Zero-length scalars/tensors may share a boundary without overlap.
    occupied.sort_unstable();
    let mut cursor = 0u64;
    for (start, end) in occupied {
        if start != cursor {
            return Err(Error::Malformed(
                "safetensors overlapping or missing data ranges".into(),
            ));
        }
        cursor = end;
    }
    if cursor != data_len {
        return Err(Error::Malformed("safetensors unindexed data bytes".into()));
    }
    Ok(ArtifactInfo {
        format: ArtifactFormat::Safetensors,
        file_len,
        tensors,
        architecture: None,
    })
}

fn checked_elements(dims: &[u64]) -> Result<u64> {
    if dims.len() > 16 {
        return Err(Error::Malformed("tensor rank exceeds 16".into()));
    }
    dims.iter()
        .try_fold(1u64, |v, d| v.checked_mul(*d))
        .ok_or_else(|| Error::Limit("tensor element count overflow".into()))
}

struct GgufReader<'a, R: Read + Seek> {
    r: &'a mut R,
    len: u64,
}
impl<R: Read + Seek> GgufReader<'_, R> {
    fn pos(&mut self) -> Result<u64> {
        Ok(self.r.stream_position()?)
    }
    fn ensure(&mut self, n: u64) -> Result<()> {
        let p = self.pos()?;
        if p.checked_add(n).is_none_or(|v| v > self.len) {
            Err(Error::Malformed(
                "truncated GGUF metadata or tensor table".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.ensure(N as u64)?;
        let mut buf = [0; N];
        self.r.read_exact(&mut buf)?;
        Ok(buf)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take::<1>()?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn skip(&mut self, n: u64) -> Result<()> {
        self.ensure(n)?;
        let dest = self
            .pos()?
            .checked_add(n)
            .ok_or_else(|| Error::Limit("GGUF position overflow".into()))?;
        self.r.seek(SeekFrom::Start(dest))?;
        Ok(())
    }
    fn string(&mut self, max: u64) -> Result<String> {
        let n = self.u64()?;
        if n > max {
            return Err(Error::Limit(
                "GGUF string exceeds configured maximum".into(),
            ));
        }
        self.ensure(n)?;
        let mut buf = vec![0; n as usize];
        self.r.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|_| Error::Malformed("GGUF string invalid UTF-8".into()))
    }
    fn skip_string(&mut self) -> Result<()> {
        let n = self.u64()?;
        if n > MAX_STRING {
            return Err(Error::Limit("GGUF metadata string exceeds 8 MiB".into()));
        }
        self.skip(n)
    }
    fn metadata_value(&mut self, ty: u32, depth: u8, capture: bool) -> Result<Option<String>> {
        if depth > 4 {
            return Err(Error::Limit("GGUF nested arrays too deep".into()));
        }
        match ty {
            0 | 1 | 7 => {
                let n = self.u8()?;
                if ty == 7 && n > 1 {
                    return Err(Error::Malformed("invalid GGUF bool".into()));
                }
                Ok(capture.then(|| n.to_string()))
            }
            2 | 3 => {
                let v = u16::from_le_bytes(self.take()?);
                Ok(capture.then(|| v.to_string()))
            }
            4..=6 => {
                let v = self.u32()?;
                Ok(capture.then(|| v.to_string()))
            }
            10..=12 => {
                let v = self.u64()?;
                Ok(capture.then(|| v.to_string()))
            }
            8 if capture => Ok(Some(self.string(MAX_STRING)?)),
            8 => {
                self.skip_string()?;
                Ok(None)
            }
            9 => {
                let element_ty = self.u32()?;
                let count = self.u64()?;
                if count > MAX_ARRAY_ITEMS {
                    return Err(Error::Limit("GGUF array count exceeds 4M".into()));
                }
                if let Some(element_size) = match element_ty {
                    0 | 1 | 7 => Some(1u64),
                    2 | 3 => Some(2),
                    4..=6 => Some(4),
                    10..=12 => Some(8),
                    _ => None,
                } {
                    self.skip(
                        count
                            .checked_mul(element_size)
                            .ok_or_else(|| Error::Limit("GGUF array size overflow".into()))?,
                    )?;
                } else {
                    for _ in 0..count {
                        self.metadata_value(element_ty, depth + 1, false)?;
                    }
                }
                Ok(None)
            }
            _ => Err(Error::Unsupported(format!("GGUF metadata value type {ty}"))),
        }
    }
}

fn align_up(n: u64, alignment: u64) -> Result<u64> {
    n.checked_add(alignment - 1)
        .map(|v| (v / alignment) * alignment)
        .ok_or_else(|| Error::Limit("GGUF alignment overflow".into()))
}

pub fn parse_gguf<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<ArtifactInfo> {
    reader.seek(SeekFrom::Start(0))?;
    let mut r = GgufReader {
        r: reader,
        len: file_len,
    };
    if r.take::<4>()? != *b"GGUF" {
        return Err(Error::Malformed("invalid GGUF magic".into()));
    }
    if r.u32()? != 3 {
        return Err(Error::Unsupported(
            "only little-endian GGUF v3 is implemented".into(),
        ));
    }
    let n = r.u64()?;
    let m = r.u64()?;
    if n > MAX_TENSORS as u64 || m > MAX_TENSORS as u64 {
        return Err(Error::Limit("GGUF tensor/metadata count too large".into()));
    }
    let mut align = 32u64;
    let mut architecture = None;
    let mut keys = HashSet::new();
    for _ in 0..m {
        if r.pos()? > MAX_METADATA {
            return Err(Error::Limit("GGUF metadata exceeds 128 MiB".into()));
        }
        let key = r.string(MAX_NAME as u64)?;
        if key.is_empty() || !keys.insert(key.clone()) {
            return Err(Error::Malformed(
                "empty or duplicate GGUF metadata key".into(),
            ));
        }
        let ty = r.u32()?;
        let capture = key == "general.architecture" || key == "general.alignment";
        let value = r.metadata_value(ty, 0, capture)?;
        match key.as_str() {
            "general.architecture" => {
                if ty != 8 {
                    return Err(Error::Malformed("GGUF architecture is not string".into()));
                }
                architecture = value;
            }
            "general.alignment" => {
                if ty != 4 && ty != 10 {
                    return Err(Error::Malformed(
                        "GGUF alignment is not unsigned integer".into(),
                    ));
                }
                align = value
                    .ok_or_else(|| Error::Malformed("GGUF missing alignment".into()))?
                    .parse()
                    .map_err(|_| Error::Malformed("GGUF bad alignment".into()))?;
                if !(8..=4096).contains(&align) || !align.is_power_of_two() {
                    return Err(Error::Malformed(
                        "GGUF alignment must be power of two 8..4096".into(),
                    ));
                }
            }
            _ => {}
        }
    }
    let mut tensors = Vec::with_capacity(n as usize);
    let mut names = HashSet::new();
    for _ in 0..n {
        if r.pos()? > MAX_METADATA {
            return Err(Error::Limit("GGUF tensor table exceeds 128 MiB".into()));
        }
        let name = r.string(64)?;
        if name.is_empty() || !names.insert(name.clone()) {
            return Err(Error::Malformed("GGUF duplicate/empty tensor name".into()));
        }
        let rank = r.u32()?;
        if rank == 0 || rank > 8 {
            return Err(Error::Malformed("GGUF rank must be 1..8".into()));
        }
        let mut shape = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            shape.push(r.u64()?);
        }
        let kind = QuantKind::from_ggml(r.u32()?)?;
        let (block_elements, _) = kind.block();
        if !shape[0].is_multiple_of(block_elements) {
            return Err(Error::Malformed(
                "GGUF quantized row not aligned to full blocks".into(),
            ));
        }
        let relative = r.u64()?;
        if relative % align != 0 {
            return Err(Error::Malformed("unaligned GGUF tensor offset".into()));
        }
        let elements = checked_elements(&shape)?;
        let len = kind.byte_len(elements)?;
        tensors.push(TensorInfo {
            name,
            shape,
            kind,
            byte_offset: relative,
            byte_len: len,
        });
    }
    let data_start = align_up(r.pos()?, align)?;
    if data_start > file_len {
        return Err(Error::Malformed("GGUF table extends past EOF".into()));
    }
    let mut ranges = Vec::new();
    for tensor in &mut tensors {
        tensor.byte_offset = data_start
            .checked_add(tensor.byte_offset)
            .ok_or_else(|| Error::Limit("GGUF tensor absolute offset overflow".into()))?;
        let end = tensor
            .byte_offset
            .checked_add(tensor.byte_len)
            .ok_or_else(|| Error::Limit("GGUF tensor end overflow".into()))?;
        if end > file_len {
            return Err(Error::Malformed(format!(
                "GGUF tensor '{}' extends beyond EOF",
                tensor.name
            )));
        }
        ranges.push((tensor.byte_offset, end));
    }
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(Error::Malformed("GGUF tensor regions overlap".into()));
        }
    }
    Ok(ArtifactInfo {
        format: ArtifactFormat::Gguf,
        file_len,
        tensors,
        architecture,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn safe(tensor_json: &str, bytes: &[u8]) -> Vec<u8> {
        let mut v = (tensor_json.len() as u64).to_le_bytes().to_vec();
        v.extend(tensor_json.as_bytes());
        v.extend(bytes);
        v
    }
    fn check_safe(file: &[u8]) -> Result<ArtifactInfo> {
        parse_safetensors(&mut Cursor::new(file), file.len() as u64)
    }
    fn strfield(v: &mut Vec<u8>, s: &str) {
        v.extend((s.len() as u64).to_le_bytes());
        v.extend(s.as_bytes());
    }
    fn gguf_single() -> Vec<u8> {
        let mut v = b"GGUF".to_vec();
        v.extend(3u32.to_le_bytes()); // v3
        v.extend(1u64.to_le_bytes()); // one tensor
        v.extend(1u64.to_le_bytes()); // one metadata entry
        strfield(&mut v, "general.architecture");
        v.extend(8u32.to_le_bytes()); // string
        strfield(&mut v, "test");
        strfield(&mut v, "blk.0.attn_q.weight");
        v.extend(2u32.to_le_bytes()); // rank
        v.extend(2u64.to_le_bytes()); // dim[0]
        v.extend(2u64.to_le_bytes()); // dim[1]
        v.extend(0u32.to_le_bytes()); // F32
        v.extend(0u64.to_le_bytes()); // tensor relative offset
        while !v.len().is_multiple_of(32) {
            v.push(0);
        }
        for i in 0..4 {
            v.extend((i as f32).to_le_bytes());
        }
        v
    }

    #[test]
    fn safetensors_real_float_directory() {
        let bytes = safe(
            r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[0, 0, 128, 63, 0, 0, 0, 64],
        );
        let info = check_safe(&bytes).unwrap();
        assert_eq!(info.tensors[0].kind, QuantKind::F32);
        assert_eq!(info.tensors[0].shape, [2]);
        assert_eq!(info.tensors[0].byte_offset, 8 + (bytes.len() - 16) as u64);
    }
    #[test]
    fn opens_real_file_and_reads_verified_reference_tensor() {
        let bytes = safe(
            r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[0, 0, 128, 63, 0, 0, 0, 64],
        );
        let path = std::env::temp_dir().join(format!(
            "zerogravity-artifact-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        let mut artifact = Artifact::open(&path).unwrap();
        assert_eq!(artifact.read_reference_f32("w").unwrap(), vec![1., 2.]);
        assert!(artifact.read_reference_f32("missing").is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn safetensors_adversarial_rejections() {
        assert!(
            check_safe(&safe(
                r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
                &[0; 8]
            ))
            .is_err()
        );
        assert!(
            check_safe(&safe(
                r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
                &[0; 12]
            ))
            .is_err()
        );
        assert!(
            check_safe(&safe(
                r#"{"w":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,8]}}"#,
                &[0; 8]
            ))
            .is_err()
        );
        assert!(check_safe(&safe(r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,&[0;8])).is_err());
        assert!(
            check_safe(&safe(
                r#"{"w":{"dtype":"U99","shape":[1],"data_offsets":[0,1]}}"#,
                &[0; 1]
            ))
            .is_err()
        );
    }
    #[cfg(unix)]
    #[test]
    fn source_materialization_uses_open_inode_not_replaced_same_length_path() {
        let original = gguf_single();
        let mut replacement = original.clone();
        let last = replacement.len() - 1;
        replacement[last] ^= 0x7f;
        assert_eq!(original.len(), replacement.len());
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "zerogravity-review-open-source-{}.gguf",
            std::process::id()
        ));
        let updated = dir.join(format!(
            "zerogravity-review-updated-source-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &original).unwrap();
        let parsed = Artifact::open(&path).unwrap();
        std::fs::write(&updated, &replacement).unwrap();
        // Linux atomic pathname replacement must not change the *open*
        // file descriptor from which the model's original weights are read.
        std::fs::rename(&updated, &path).unwrap();
        let (info, bytes) = parsed.into_bounded_source(4096).unwrap();
        assert_eq!(info.format, ArtifactFormat::Gguf);
        assert_eq!(bytes, original);
        assert_ne!(bytes, replacement);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cpu_source_snapshot_rejects_too_small_cap_before_materialization() {
        let original = gguf_single();
        let path = std::env::temp_dir().join(format!(
            "zerogravity-review-source-cap-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &original).unwrap();
        let parsed = Artifact::open(&path).unwrap();
        assert!(matches!(
            parsed.into_bounded_source(original.len() as u64 - 1),
            Err(Error::Limit(_))
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn gguf_v3_directory_and_bounded_invalid_types() {
        let bytes = gguf_single();
        let info = parse_gguf(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
        assert_eq!(info.format, ArtifactFormat::Gguf);
        assert_eq!(info.architecture.as_deref(), Some("test"));
        assert_eq!(info.tensors.len(), 1);
        assert_eq!(info.tensors[0].byte_offset, bytes.len() as u64 - 16);
        let mut bad = bytes.clone();
        bad[4] = 8; // unsupported version
        assert!(parse_gguf(&mut Cursor::new(&bad), bad.len() as u64).is_err());
        let mut truncated = bytes.clone();
        truncated.pop();
        assert!(parse_gguf(&mut Cursor::new(&truncated), truncated.len() as u64).is_err());
    }
}
