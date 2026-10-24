//! Original reference data enters only the actual completion callback.
use super::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;

pub(super) struct FixedCell {
    root: PathBuf,
    pins: serde_json::Value,
}
impl FixedCell {
    pub(super) fn open(
        root: &Path,
        profile: zerogravity::accelerator::program::FullModelProfile,
        rows: usize,
    ) -> Result<Self> {
        use zerogravity::accelerator::program::FullModelProfile;
        let model = match profile {
            FullModelProfile::Qwen08 => "0.8b",
            FullModelProfile::Qwen2B => "2b",
        };
        let catalogue: serde_json::Value =
            serde_json::from_str(include_str!("oracle-pins.json"))
                .map_err(|e| fail(format!("compiled numerical reference identities: {e}")))?;
        ensure(
            catalogue["schema"] == "zg.short-hf-fixed-original-oracles.v1",
            "fixed original oracle schema",
        )?;
        let pins = catalogue["cells"][format!("{model}-{rows}")].clone();
        ensure(
            pins["model"] == model && pins["prompt_rows"].as_u64() == Some(rows as u64),
            "fixed model and prompt oracle",
        )?;
        Ok(Self {
            root: root.to_path_buf(),
            pins,
        })
    }
    pub(super) fn json(&self, name: &str, cap: usize) -> Result<serde_json::Value> {
        let pin = &self.pins["files"][name];
        let bytes = number(pin, "bytes")?;
        ensure(bytes <= cap, "fixed numerical metadata bound")?;
        let mut file = File::open(self.root.join(name))?;
        ensure(
            file.metadata()?.len() == bytes as u64,
            "fixed numerical metadata exact extent",
        )?;
        let mut raw = vec![0; bytes];
        file.read_exact(&mut raw)?;
        ensure(file.read(&mut [0])? == 0, "fixed numerical metadata EOF")?;
        hash(&raw, &pin["sha256"])?;
        serde_json::from_slice(&raw).map_err(|e| fail(format!("fixed reference JSON: {e}")))
    }
    pub(super) fn source_manifest(&self) -> &serde_json::Value {
        &self.pins["source_manifest_sha256"]
    }
}

pub(super) fn json(path: &Path, cap: usize) -> Result<serde_json::Value> {
    let mut file = File::open(path)?;
    let bytes = usize::try_from(file.metadata()?.len()).map_err(|_| fail("reference extent"))?;
    ensure(bytes > 0 && bytes <= cap, "bounded reference metadata")?;
    let mut raw = Vec::with_capacity(bytes);
    file.read_to_end(&mut raw)?;
    ensure(raw.len() == bytes, "complete reference metadata")?;
    serde_json::from_slice(&raw).map_err(|e| fail(format!("reference JSON: {e}")))
}
fn number(v: &serde_json::Value, key: &str) -> Result<usize> {
    v[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| fail(format!("reference {key}")))
}
fn hash(raw: &[u8], expected: &serde_json::Value) -> Result<()> {
    ensure(
        expected
            .as_str()
            .is_some_and(|s| format!("{:x}", Sha256::digest(raw)) == s),
        "complete reference SHA",
    )
}
pub(super) struct Archive {
    records: Vec<serde_json::Value>,
    raw: Vec<u8>,
}
impl Archive {
    pub(super) fn open(root: &Path, receipt: &serde_json::Value) -> Result<Self> {
        ensure(
            receipt["file"] == "raw.bin",
            "closed reference archive name",
        )?;
        let bytes = number(receipt, "bytes")?;
        ensure(
            bytes > 0 && bytes <= 32 * MIB,
            "bounded whole original archive",
        )?;
        let mut file = File::open(root.join("raw.bin"))?;
        ensure(
            file.metadata()?.len() == bytes as u64,
            "original archive exact extent",
        )?;
        let mut raw = vec![0; bytes];
        file.read_exact(&mut raw)?;
        ensure(file.read(&mut [0])? == 0, "whole original archive EOF")?;
        hash(&raw, &receipt["sha256"])?;
        let records = receipt["records"]
            .as_array()
            .ok_or_else(|| fail("reference records"))?
            .clone();
        let mut end = 0;
        for record in &records {
            let offset = number(record, "offset")?;
            let len = number(record, "bytes")?;
            ensure(offset == end && len > 0, "ordered complete archive records")?;
            end = offset
                .checked_add(len)
                .filter(|n| *n <= bytes)
                .ok_or_else(|| fail("reference range"))?;
            hash(&raw[offset..end], &record["sha256"])?;
        }
        ensure(end == bytes, "all original archive bytes accounted")?;
        Ok(Self { records, raw })
    }
    pub(super) fn get(&self, call: usize, label: &str) -> Result<&[u8]> {
        let mut rows = self
            .records
            .iter()
            .filter(|r| r["call"].as_u64() == Some(call as u64) && r["label"] == label);
        let row = rows
            .next()
            .ok_or_else(|| fail(format!("missing reference {call}/{label}")))?;
        ensure(rows.next().is_none(), "unique reference record")?;
        let first = number(row, "offset")?;
        let end = first + number(row, "bytes")?;
        self.raw
            .get(first..end)
            .ok_or_else(|| fail("reference record range"))
    }
}
pub(super) fn exact(actual: &[u8], expected: &[u8], label: &str) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    let first = actual.iter().zip(expected).position(|(a, b)| a != b);
    Err(fail(format!(
        "{label}: bytes {}/{}, first mismatch {first:?}",
        actual.len(),
        expected.len()
    )))
}
pub(super) fn widen(raw: &[u8]) -> Result<Vec<u8>> {
    ensure(raw.len().is_multiple_of(2), "complete BF16 words")?;
    let mut result = Vec::with_capacity(raw.len() * 2);
    for word in raw.chunks_exact(2) {
        let bits = (u16::from_le_bytes([word[0], word[1]]) as u32) << 16;
        ensure(bits & 0x7f800000 != 0x7f800000, "finite original BF16 word")?;
        result.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(result)
}
pub(super) fn sum(left: &[u8], right: &[u8]) -> Result<Vec<u8>> {
    ensure(
        left.len() == right.len(),
        "same original dual operand geometry",
    )?;
    let a = widen(left)?;
    let b = widen(right)?;
    Ok(a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .flat_map(|(a, b)| {
            (f32::from_le_bytes(a.try_into().unwrap()) + f32::from_le_bytes(b.try_into().unwrap()))
                .to_le_bytes()
        })
        .collect())
}
pub(super) fn convolution(raw: &[u8]) -> Result<Vec<u8>> {
    ensure(
        raw.len() == 3 * 6144 * 2,
        "original native convolution state",
    )?;
    let mut canonical = Vec::with_capacity(3 * 6144 * 4);
    for channel in 0..6144 {
        for frame in 0..3 {
            let index = (frame * 6144 + channel) * 2;
            canonical.extend(widen(&raw[index..index + 2])?);
        }
    }
    Ok(canonical)
}
pub(super) fn recurrent(raw: &[u8]) -> Result<Vec<u8>> {
    ensure(
        raw.len() == 16 * 128 * 128 * 4,
        "original native recurrent state",
    )?;
    let mut canonical = Vec::with_capacity(raw.len());
    for head in 0..16 {
        for major in 0..128 {
            for minor in 0..128 {
                let index = (head * 16384 + minor * 128 + major) * 4;
                canonical.extend_from_slice(&raw[index..index + 4]);
            }
        }
    }
    Ok(canonical)
}
pub(super) fn kv(
    actual: (&[u8], Option<&[u8]>),
    archive: &Archive,
    call: usize,
    label: &str,
    active: usize,
) -> Result<()> {
    let (k, v) = actual;
    let v = v.ok_or_else(|| fail("actual paired canonical KV"))?;
    ensure(
        k.len() == 512 * 512 * 4 && v.len() == k.len() && active <= 512,
        "canonical KV geometry",
    )?;
    for position in 0..active {
        let original = archive.get(call, &format!("{label}.position.{position}"))?;
        ensure(original.len() == 2048, "complete native two-head KV row")?;
        let start = position * 512 * 4;
        for (plane, offset) in [(k, 0usize), (v, 256)] {
            for head in 0..2 {
                let source = (head * 512 + offset) * 2;
                exact(
                    &plane[start + head * 1024..start + (head + 1) * 1024],
                    &widen(&original[source..source + 512])?,
                    label,
                )?;
            }
        }
    }
    ensure(
        k[active * 512 * 4..]
            .iter()
            .chain(&v[active * 512 * 4..])
            .all(|b| *b == 0),
        "canonical future KV remains positive zero",
    )
}
