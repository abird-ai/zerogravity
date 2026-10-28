//! Host comparison evidence for two-process ordinary CUDA qualification.
//!
//! This file is never an execution capability. The ordinary process independently
//! authenticates its source, chooses the fixed qualification context/prompt and
//! feeds back its own greedy tokens. Evidence is only consulted after outputs exist.

use super::{
    Cached47To50Reference, FRONTIERS, ParitySourceIdentity, ReferenceParityStep, VOCABULARY,
    expected_domain_slots, expected_domain_values, require_reference, require_reference_close,
};
use crate::{
    Error, Result,
    accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot},
    gpu::cuda::native::TextCloseReceipt,
};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    sync::Arc,
};

const MAGIC: &[u8; 16] = b"ZGQ35P47EVID0001";
const DIGEST_BYTES: u64 = 32;
const MAX_EVIDENCE_BYTES: u64 = 160 * 1024 * 1024;
const MAX_STRING_BYTES: usize = 512;

#[derive(Clone, Debug)]
pub(super) struct ReferenceEvidence {
    pub(super) identity: ParitySourceIdentity,
    pub(super) reference: Cached47To50Reference,
    pub(super) close: TextCloseReceipt,
}

struct HashingWriter<W> {
    inner: W,
    hash: Sha256,
}
impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hash: Sha256::new(),
        }
    }
    fn finish(mut self) -> Result<(W, [u8; 32])> {
        self.inner.flush()?;
        let digest: [u8; 32] = self.hash.finalize().into();
        Ok((self.inner, digest))
    }
}
impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.hash.update(&bytes[..written]);
        Ok(written)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

struct HashingReader<R> {
    inner: R,
    hash: Sha256,
    remaining: u64,
}
impl<R: Read> HashingReader<R> {
    fn new(inner: R, remaining: u64) -> Self {
        Self {
            inner,
            hash: Sha256::new(),
            remaining,
        }
    }
    fn finish(self) -> Result<(R, [u8; 32])> {
        if self.remaining != 0 {
            return Err(Error::Malformed(
                "ordinary parity evidence trailing/unknown payload bytes".into(),
            ));
        }
        Ok((self.inner, self.hash.finalize().into()))
    }
}
impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, target: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 || target.is_empty() {
            return Ok(0);
        }
        let limit = usize::try_from(self.remaining.min(target.len() as u64)).unwrap();
        let read = self.inner.read(&mut target[..limit])?;
        self.remaining -= read as u64;
        self.hash.update(&target[..read]);
        Ok(read)
    }
}

pub(super) fn write(path: &Path, evidence: &ReferenceEvidence) -> Result<[u8; 32]> {
    require_path(path)?;
    require_evidence(evidence)?;

    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let result = (|| -> Result<[u8; 32]> {
        let mut writer = HashingWriter::new(BufWriter::new(file));
        writer.write_all(MAGIC)?;
        write_string(&mut writer, &evidence.identity.family)?;
        write_string(&mut writer, &evidence.identity.repository)?;
        write_string(&mut writer, &evidence.identity.revision)?;
        write_string(&mut writer, &evidence.identity.index_sha256)?;
        write_u64(&mut writer, evidence.identity.hidden_width)?;
        write_u64(&mut writer, evidence.identity.capacity)?;

        for value in close_values(&evidence.close) {
            write_u64(&mut writer, value)?;
        }

        write_u32(&mut writer, evidence.reference.prompt.len())?;
        for &id in evidence.reference.prompt.iter() {
            write_u32_raw(&mut writer, id)?;
        }

        write_u32(&mut writer, evidence.reference.steps.len())?;
        for step in &evidence.reference.steps {
            write_u32(&mut writer, step.frontier)?;
            write_u32_raw(&mut writer, step.selected_id)?;
            write_u32_raw(&mut writer, step.max_bits)?;
            write_f32_vec(&mut writer, &step.logits)?;

            let mut domains = step.domains.iter().collect::<Vec<_>>();
            domains.sort_unstable_by_key(|domain| domain.0);
            write_u32(&mut writer, domains.len())?;
            for (slot, values) in domains {
                write_slot(&mut writer, *slot)?;
                write_f32_vec(&mut writer, values)?;
            }
        }

        let (writer, digest) = writer.finish()?;
        let mut file = writer
            .into_inner()
            .map_err(|error| Error::Io(error.into_error()))?;
        file.write_all(&digest)?;
        file.sync_all()?;
        Ok(digest)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(path);
    }
    result
}

pub(super) fn read(path: &Path) -> Result<(ReferenceEvidence, [u8; 32])> {
    require_path(path)?;
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(Error::Malformed(
            "ordinary parity evidence must be a regular non-symlink file".into(),
        ));
    }
    let bytes = metadata.len();
    if bytes <= DIGEST_BYTES || bytes > MAX_EVIDENCE_BYTES {
        return Err(Error::Limit(
            "ordinary parity evidence file extent outside finite bound".into(),
        ));
    }
    let payload = bytes - DIGEST_BYTES;
    let file = File::open(path)?;
    let mut reader = HashingReader::new(BufReader::new(file), payload);

    let mut magic = [0u8; 16];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::Malformed(
            "ordinary parity evidence schema/magic differs".into(),
        ));
    }

    let identity = ParitySourceIdentity {
        family: read_string(&mut reader)?,
        repository: read_string(&mut reader)?,
        revision: read_string(&mut reader)?,
        index_sha256: read_string(&mut reader)?,
        hidden_width: read_usize(&mut reader, "hidden width")?,
        capacity: read_usize(&mut reader, "context capacity")?,
    };

    let close = TextCloseReceipt {
        source_weight_windows: read_usize(&mut reader, "source windows")?,
        streamed_head_windows: read_usize(&mut reader, "streamed head windows")?,
        retirement_source_weight_windows: read_usize(&mut reader, "retirement source windows")?,
        retirement_streamed_head_windows: read_usize(
            &mut reader,
            "retirement streamed head windows",
        )?,
        resident_head_windows: read_usize(&mut reader, "resident head windows")?,
        retirement_resident_head_windows: read_usize(
            &mut reader,
            "retirement resident head windows",
        )?,
    };

    let prompt_len = read_usize32(&mut reader, "prompt length")?;
    if prompt_len != 47 {
        return Err(Error::Malformed(
            "ordinary parity evidence prompt length differs".into(),
        ));
    }
    let mut prompt = Vec::with_capacity(prompt_len);
    for _ in 0..prompt_len {
        prompt.push(read_u32(&mut reader)?);
    }

    let step_count = read_usize32(&mut reader, "step count")?;
    if step_count != FRONTIERS.len() {
        return Err(Error::Malformed(
            "ordinary parity evidence step count differs".into(),
        ));
    }
    let expected_slots = expected_domain_slots();
    let mut steps = Vec::with_capacity(step_count);
    for &expected_frontier in &FRONTIERS {
        let frontier = read_usize32(&mut reader, "frontier")?;
        if frontier != expected_frontier {
            return Err(Error::Malformed(
                "ordinary parity evidence frontier sequence differs".into(),
            ));
        }
        let selected_id = read_u32(&mut reader)?;
        let max_bits = read_u32(&mut reader)?;
        let logits = read_f32_vec(&mut reader, VOCABULARY, false, "logits")?;
        let domain_count = read_usize32(&mut reader, "domain count")?;
        if domain_count != expected_slots.len() {
            return Err(Error::Malformed(
                "ordinary parity evidence domain count differs".into(),
            ));
        }
        let mut domains = Vec::with_capacity(domain_count);
        for expected_slot in &expected_slots {
            let slot = read_slot(&mut reader)?;
            if &slot != expected_slot {
                return Err(Error::Malformed(
                    "ordinary parity evidence domain order/identity differs".into(),
                ));
            }
            let expected_values = expected_domain_values(slot, frontier)?;
            let require_bf16_widened = matches!(slot, StateSlot::Kv(_));
            let values = read_f32_vec(
                &mut reader,
                expected_values,
                require_bf16_widened,
                "state domain",
            )?;
            domains.push((slot, values));
        }
        steps.push(ReferenceParityStep {
            frontier,
            selected_id,
            max_bits,
            logits,
            domains,
        });
    }

    let (mut file, digest) = reader.finish()?;
    let mut recorded = [0u8; 32];
    file.read_exact(&mut recorded)?;
    let mut extra = [0u8; 1];
    if file.read(&mut extra)? != 0 {
        return Err(Error::Malformed(
            "ordinary parity evidence bytes follow digest".into(),
        ));
    }
    if recorded != digest {
        return Err(Error::Malformed(
            "ordinary parity evidence SHA256 differs".into(),
        ));
    }

    let evidence = ReferenceEvidence {
        identity,
        reference: Cached47To50Reference {
            capacity: 512,
            prompt: Arc::from(prompt),
            steps,
        },
        close,
    };
    require_evidence(&evidence)?;
    Ok((evidence, digest))
}

fn require_evidence(evidence: &ReferenceEvidence) -> Result<()> {
    if evidence.identity.capacity != 512
        || evidence.reference.capacity != 512
        || evidence.identity.family.is_empty()
        || evidence.identity.repository.is_empty()
        || evidence.identity.revision.len() != 40
        || evidence.identity.index_sha256.len() != 64
        || evidence.identity.hidden_width != 1024
    {
        return Err(Error::State(
            "ordinary parity evidence source identity/profile differs".into(),
        ));
    }
    require_reference_close(&evidence.close)?;
    require_reference(&evidence.reference)
}

fn require_path(path: &Path) -> Result<()> {
    if !path.is_absolute() || path.file_name().is_none() {
        return Err(Error::Malformed(
            "ordinary parity evidence requires an absolute file path".into(),
        ));
    }
    Ok(())
}

fn write_string(writer: &mut impl Write, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > MAX_STRING_BYTES {
        return Err(Error::Limit(
            "ordinary parity evidence metadata string extent".into(),
        ));
    }
    write_u32(writer, value.len())?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let len = read_usize32(reader, "metadata string length")?;
    if len == 0 || len > MAX_STRING_BYTES {
        return Err(Error::Limit(
            "ordinary parity evidence metadata string extent".into(),
        ));
    }
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|_| Error::Malformed("ordinary parity evidence metadata UTF-8".into()))
}

fn write_f32_vec(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    write_u64(writer, values.len())?;
    for &value in values {
        if !value.is_finite() {
            return Err(Error::State(
                "ordinary parity evidence contains nonfinite value".into(),
            ));
        }
        writer.write_all(&value.to_bits().to_le_bytes())?;
    }
    Ok(())
}

fn read_f32_vec(
    reader: &mut impl Read,
    expected: usize,
    require_bf16_widened: bool,
    label: &str,
) -> Result<Vec<f32>> {
    let len = read_usize(reader, label)?;
    if len != expected {
        return Err(Error::Malformed(format!(
            "ordinary parity evidence {label} element count differs"
        )));
    }
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let bits = read_u32(reader)?;
        let value = f32::from_bits(bits);
        if !value.is_finite() || (require_bf16_widened && bits & 0xffff != 0) {
            return Err(Error::Malformed(format!(
                "ordinary parity evidence {label} finite/BF16 contract differs"
            )));
        }
        values.push(value);
    }
    Ok(values)
}

fn write_slot(writer: &mut impl Write, slot: StateSlot) -> Result<()> {
    let (tag, layer) = match slot {
        StateSlot::Conv(ConvDomain(layer)) => (0u8, layer),
        StateSlot::Recurrent(RecurrentDomain(layer)) => (1u8, layer),
        StateSlot::Kv(KvDomain(layer)) => (2u8, layer),
    };
    writer.write_all(&[tag])?;
    write_u32(writer, layer)
}

fn read_slot(reader: &mut impl Read) -> Result<StateSlot> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    let layer = read_usize32(reader, "state layer")?;
    match tag[0] {
        0 => Ok(StateSlot::Conv(ConvDomain(layer))),
        1 => Ok(StateSlot::Recurrent(RecurrentDomain(layer))),
        2 => Ok(StateSlot::Kv(KvDomain(layer))),
        _ => Err(Error::Malformed(
            "ordinary parity evidence state slot tag differs".into(),
        )),
    }
}

fn close_values(close: &TextCloseReceipt) -> [usize; 6] {
    [
        close.source_weight_windows,
        close.streamed_head_windows,
        close.retirement_source_weight_windows,
        close.retirement_streamed_head_windows,
        close.resident_head_windows,
        close.retirement_resident_head_windows,
    ]
}

fn write_u32(writer: &mut impl Write, value: usize) -> Result<()> {
    let value = u32::try_from(value)
        .map_err(|_| Error::Limit("ordinary parity evidence u32 extent".into()))?;
    write_u32_raw(writer, value)
}

fn write_u32_raw(writer: &mut impl Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64(writer: &mut impl Write, value: usize) -> Result<()> {
    let value = u64::try_from(value)
        .map_err(|_| Error::Limit("ordinary parity evidence u64 extent".into()))?;
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_usize32(reader: &mut impl Read, label: &str) -> Result<usize> {
    usize::try_from(read_u32(reader)?)
        .map_err(|_| Error::Limit(format!("ordinary parity evidence {label}")))
}

fn read_usize(reader: &mut impl Read, label: &str) -> Result<usize> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    usize::try_from(u64::from_le_bytes(bytes))
        .map_err(|_| Error::Limit(format!("ordinary parity evidence {label}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn hashing_reader_rejects_unknown_payload_tail_and_detects_digest_change() {
        let payload = b"payload";
        let digest: [u8; 32] = Sha256::digest(payload).into();
        let mut bytes = payload.to_vec();
        bytes.extend_from_slice(&digest);

        let mut reader = HashingReader::new(Cursor::new(bytes.clone()), payload.len() as u64);
        let mut out = vec![0u8; payload.len()];
        reader.read_exact(&mut out).unwrap();
        let (mut cursor, actual) = reader.finish().unwrap();
        let mut expected = [0u8; 32];
        cursor.read_exact(&mut expected).unwrap();
        assert_eq!(actual, expected);

        bytes[0] ^= 1;
        let mut reader = HashingReader::new(Cursor::new(bytes), payload.len() as u64);
        reader.read_exact(&mut out).unwrap();
        let (mut cursor, actual) = reader.finish().unwrap();
        cursor.read_exact(&mut expected).unwrap();
        assert_ne!(actual, expected);

        let reader = HashingReader::new(Cursor::new(b"abc".to_vec()), 3);
        assert!(reader.finish().is_err());
    }

    #[test]
    fn slot_codec_is_exact_for_all_domain_kinds() {
        for slot in [
            StateSlot::Conv(ConvDomain(2)),
            StateSlot::Recurrent(RecurrentDomain(5)),
            StateSlot::Kv(KvDomain(7)),
        ] {
            let mut bytes = Vec::new();
            write_slot(&mut bytes, slot).unwrap();
            assert_eq!(read_slot(&mut Cursor::new(bytes)).unwrap(), slot);
        }
    }
}
