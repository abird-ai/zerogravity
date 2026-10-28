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
    qwen35_cuda::held_file::{HeldFile, open_absolute_directory},
};
use rustix::fs::{self, AtFlags, Mode, OFlags};
use sha2::{Digest, Sha256};
use std::{
    ffi::{OsStr, OsString},
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    sync::Arc,
};

const MAGIC: &[u8; 16] = b"ZGQ35P47EVID0003";
const DIGEST_BYTES: u64 = 32;
const MAX_STRING_BYTES: usize = 512;

const fn f32_vector_bytes(values: u64) -> u64 {
    8 + values * 4
}

const fn evidence_max_bytes() -> u64 {
    let mut bytes = 16; // magic
    bytes += 3 * (4 + MAX_STRING_BYTES as u64); // family/repository/revision
    bytes += 2 * 8; // hidden width + context capacity
    bytes += 6 * 8; // close receipt
    bytes += 4 + 47 * 4; // prompt length + ids
    bytes += 4; // step count

    let mut index = 0;
    while index < FRONTIERS.len() {
        let frontier = FRONTIERS[index] as u64;
        bytes += 3 * 4; // frontier + selected id + max bits
        bytes += f32_vector_bytes(VOCABULARY as u64);
        bytes += 4; // domain count

        // 18 convolution + 18 recurrent domains and 6 KV domains.
        bytes += 18 * (5 + f32_vector_bytes((6144 * 3) as u64));
        bytes += 18 * (5 + f32_vector_bytes((16 * 128 * 128) as u64));
        bytes += 6 * (5 + f32_vector_bytes(2 * 512 * frontier));
        index += 1;
    }
    bytes + DIGEST_BYTES
}

const MAX_EVIDENCE_BYTES: u64 = evidence_max_bytes();

#[derive(Clone, Debug)]
pub(super) struct ReferenceEvidence {
    pub(super) identity: ParitySourceIdentity,
    pub(super) reference: Cached47To50Reference,
    pub(super) close: TextCloseReceipt,
}

struct HashingWriter<W> {
    inner: W,
    hash: Sha256,
    written: u64,
}
impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hash: Sha256::new(),
            written: 0,
        }
    }
    fn finish(mut self) -> Result<(W, [u8; 32], u64)> {
        self.inner.flush()?;
        let digest: [u8; 32] = self.hash.finalize().into();
        Ok((self.inner, digest, self.written))
    }
}
impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.hash.update(&bytes[..written]);
        self.written = self
            .written
            .checked_add(written as u64)
            .ok_or_else(|| std::io::Error::other("ordinary parity evidence size overflow"))?;
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

struct EvidenceShape<'a> {
    prompt_len: usize,
    frontiers: &'a [usize],
    vocabulary: usize,
    slots: &'a [StateSlot],
    domain_values: fn(StateSlot, usize) -> Result<usize>,
}

fn write_payload(writer: &mut impl Write, evidence: &ReferenceEvidence) -> Result<()> {
    writer.write_all(MAGIC)?;
    write_string(writer, &evidence.identity.family)?;
    write_string(writer, &evidence.identity.repository)?;
    write_string(writer, &evidence.identity.revision)?;
    write_u64(writer, evidence.identity.hidden_width)?;
    write_u64(writer, evidence.identity.capacity)?;

    for value in close_values(&evidence.close) {
        write_u64(writer, value)?;
    }

    write_u32(writer, evidence.reference.prompt.len())?;
    for &id in evidence.reference.prompt.iter() {
        write_u32_raw(writer, id)?;
    }

    write_u32(writer, evidence.reference.steps.len())?;
    for step in &evidence.reference.steps {
        write_u32(writer, step.frontier)?;
        write_u32_raw(writer, step.selected_id)?;
        write_u32_raw(writer, step.max_bits)?;
        write_f32_vec(writer, &step.logits)?;

        let mut domains = step.domains.iter().collect::<Vec<_>>();
        domains.sort_unstable_by_key(|domain| domain.0);
        write_u32(writer, domains.len())?;
        for (slot, values) in domains {
            write_slot(writer, *slot)?;
            write_f32_vec(writer, values)?;
        }
    }
    Ok(())
}

fn read_payload(reader: &mut impl Read, shape: &EvidenceShape<'_>) -> Result<ReferenceEvidence> {
    let mut magic = [0u8; 16];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::Malformed(
            "ordinary parity evidence schema/magic differs".into(),
        ));
    }

    let identity = ParitySourceIdentity {
        family: read_string(reader)?,
        repository: read_string(reader)?,
        revision: read_string(reader)?,
        hidden_width: read_usize(reader, "hidden width")?,
        capacity: read_usize(reader, "context capacity")?,
    };

    let close = TextCloseReceipt {
        source_weight_windows: read_usize(reader, "source windows")?,
        streamed_head_windows: read_usize(reader, "streamed head windows")?,
        retirement_source_weight_windows: read_usize(reader, "retirement source windows")?,
        retirement_streamed_head_windows: read_usize(reader, "retirement streamed head windows")?,
        resident_head_windows: read_usize(reader, "resident head windows")?,
        retirement_resident_head_windows: read_usize(reader, "retirement resident head windows")?,
    };

    let prompt_len = read_usize32(reader, "prompt length")?;
    if prompt_len != shape.prompt_len {
        return Err(Error::Malformed(
            "ordinary parity evidence prompt length differs".into(),
        ));
    }
    let mut prompt = Vec::with_capacity(prompt_len);
    for _ in 0..prompt_len {
        prompt.push(read_u32(reader)?);
    }

    let step_count = read_usize32(reader, "step count")?;
    if step_count != shape.frontiers.len() {
        return Err(Error::Malformed(
            "ordinary parity evidence step count differs".into(),
        ));
    }
    let mut steps = Vec::with_capacity(step_count);
    for &expected_frontier in shape.frontiers {
        let frontier = read_usize32(reader, "frontier")?;
        if frontier != expected_frontier {
            return Err(Error::Malformed(
                "ordinary parity evidence frontier sequence differs".into(),
            ));
        }
        let selected_id = read_u32(reader)?;
        let max_bits = read_u32(reader)?;
        let logits = read_f32_vec(reader, shape.vocabulary, false, "logits")?;
        let domain_count = read_usize32(reader, "domain count")?;
        if domain_count != shape.slots.len() {
            return Err(Error::Malformed(
                "ordinary parity evidence domain count differs".into(),
            ));
        }
        let mut domains = Vec::with_capacity(domain_count);
        for &expected_slot in shape.slots {
            let slot = read_slot(reader)?;
            if slot != expected_slot {
                return Err(Error::Malformed(
                    "ordinary parity evidence domain order/identity differs".into(),
                ));
            }
            let expected_values = (shape.domain_values)(slot, frontier)?;
            let require_bf16_widened = matches!(slot, StateSlot::Kv(_));
            let values = read_f32_vec(
                reader,
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

    let capacity = identity.capacity;
    Ok(ReferenceEvidence {
        identity,
        reference: Cached47To50Reference {
            capacity,
            prompt: Arc::from(prompt),
            steps,
        },
        close,
    })
}

pub(super) fn write(path: &Path, evidence: &ReferenceEvidence) -> Result<[u8; 32]> {
    require_path(path)?;
    require_evidence(evidence)?;

    let temporary = create_sibling_temp(path)?;
    let EvidenceTemp {
        parent,
        final_name,
        temporary_name,
        file,
    } = temporary;
    let result = (|| -> Result<[u8; 32]> {
        let mut writer = HashingWriter::new(BufWriter::new(file));
        write_payload(&mut writer, evidence)?;

        let (writer, digest, payload_bytes) = writer.finish()?;
        if payload_bytes
            .checked_add(DIGEST_BYTES)
            .is_none_or(|bytes| bytes > MAX_EVIDENCE_BYTES)
        {
            return Err(Error::Limit(
                "ordinary parity evidence output exceeds finite bound".into(),
            ));
        }
        let mut file = writer
            .into_inner()
            .map_err(|error| Error::Io(error.into_error()))?;
        file.write_all(&digest)?;
        file.sync_all()?;
        publish_sibling_temp(&parent, &temporary_name, &final_name)?;
        Ok(digest)
    })();
    if result.is_err() {
        let _ = fs::unlinkat(&parent, &temporary_name, AtFlags::empty());
    }
    result
}

struct EvidenceTemp {
    parent: File,
    final_name: OsString,
    temporary_name: OsString,
    file: File,
}

fn os<T>(result: rustix::io::Result<T>) -> Result<T> {
    result.map_err(|error| std::io::Error::from(error).into())
}

fn create_sibling_temp(path: &Path) -> Result<EvidenceTemp> {
    let parent_path = path
        .parent()
        .ok_or_else(|| Error::Malformed("ordinary parity evidence parent absent".into()))?;
    let parent = open_absolute_directory(parent_path)?;
    let final_name = path
        .file_name()
        .ok_or_else(|| Error::Malformed("ordinary parity evidence filename absent".into()))?
        .to_owned();
    for attempt in 0..128u32 {
        let mut temporary_name = OsString::from(".");
        temporary_name.push(&final_name);
        temporary_name.push(format!(".zg-parity.tmp.{}.{}", std::process::id(), attempt));
        match fs::openat(
            &parent,
            &temporary_name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::RUSR | Mode::WUSR,
        ) {
            Ok(file) => {
                return Ok(EvidenceTemp {
                    parent,
                    final_name,
                    temporary_name,
                    file: File::from(file),
                });
            }
            Err(rustix::io::Errno::EXIST) => continue,
            Err(error) => return Err(std::io::Error::from(error).into()),
        }
    }
    Err(Error::State(
        "ordinary parity evidence temporary namespace exhausted".into(),
    ))
}

fn publish_sibling_temp(parent: &File, temporary: &OsStr, final_name: &OsStr) -> Result<()> {
    os(fs::linkat(
        parent,
        temporary,
        parent,
        final_name,
        AtFlags::empty(),
    ))?;
    parent.sync_all()?;
    os(fs::unlinkat(parent, temporary, AtFlags::empty()))?;
    parent.sync_all()?;
    Ok(())
}

pub(super) fn read(
    path: &Path,
    expected_sha256: [u8; 32],
) -> Result<(ReferenceEvidence, [u8; 32])> {
    require_path(path)?;
    let mut held = HeldFile::open_absolute(path)?;
    let bytes = held.len();
    if bytes <= DIGEST_BYTES || bytes > MAX_EVIDENCE_BYTES {
        return Err(Error::Limit(
            "ordinary parity evidence file extent outside finite bound".into(),
        ));
    }
    let payload = bytes - DIGEST_BYTES;
    let mut reader = HashingReader::new(BufReader::new(held.file_mut()), payload);

    let slots = expected_domain_slots();
    let shape = EvidenceShape {
        prompt_len: 47,
        frontiers: &FRONTIERS,
        vocabulary: VOCABULARY,
        slots: &slots,
        domain_values: expected_domain_values,
    };
    let evidence = read_payload(&mut reader, &shape)?;

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
            "ordinary parity evidence embedded SHA256 differs".into(),
        ));
    }
    drop(file);
    held.verify_unchanged()?;
    require_expected_digest(digest, expected_sha256)?;

    require_evidence(&evidence)?;
    Ok((evidence, digest))
}

fn require_expected_digest(actual: [u8; 32], expected: [u8; 32]) -> Result<()> {
    if actual != expected {
        return Err(Error::State(
            "ordinary parity evidence does not match trusted expected SHA256".into(),
        ));
    }
    Ok(())
}

fn require_evidence(evidence: &ReferenceEvidence) -> Result<()> {
    if evidence.identity.capacity != 512
        || evidence.reference.capacity != 512
        || evidence.identity.family.is_empty()
        || evidence.identity.repository.is_empty()
        || evidence.identity.revision.len() != 40
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
    fn trusted_expected_digest_is_independent_from_embedded_integrity() {
        let actual = [7u8; 32];
        assert!(require_expected_digest(actual, actual).is_ok());
        assert!(require_expected_digest(actual, [8u8; 32]).is_err());
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

    fn tiny_domain_values(slot: StateSlot, _frontier: usize) -> Result<usize> {
        match slot {
            StateSlot::Conv(ConvDomain(0)) => Ok(2),
            _ => Err(Error::State("unexpected tiny domain".into())),
        }
    }

    #[test]
    fn evidence_payload_schema_roundtrips_with_tiny_geometry() {
        let evidence = ReferenceEvidence {
            identity: ParitySourceIdentity {
                family: "0.8b".into(),
                repository: "Qwen/Qwen3.5-0.8B".into(),
                revision: "2fc06364715b967f1860aea9cf38778875588b17".into(),
                hidden_width: 1024,
                capacity: 8,
            },
            reference: Cached47To50Reference {
                capacity: 8,
                prompt: Arc::from([11u32, 12]),
                steps: vec![ReferenceParityStep {
                    frontier: 2,
                    selected_id: 1,
                    max_bits: 7,
                    logits: vec![1.0, 2.0, 3.0],
                    domains: vec![(StateSlot::Conv(ConvDomain(0)), vec![4.0, 5.0])],
                }],
            },
            close: TextCloseReceipt {
                source_weight_windows: 1,
                streamed_head_windows: 0,
                retirement_source_weight_windows: 1,
                retirement_streamed_head_windows: 0,
                resident_head_windows: 0,
                retirement_resident_head_windows: 0,
            },
        };
        let mut bytes = Vec::new();
        write_payload(&mut bytes, &evidence).unwrap();

        let frontiers = [2usize];
        let slots = [StateSlot::Conv(ConvDomain(0))];
        let shape = EvidenceShape {
            prompt_len: 2,
            frontiers: &frontiers,
            vocabulary: 3,
            slots: &slots,
            domain_values: tiny_domain_values,
        };
        let mut cursor = Cursor::new(bytes);
        let decoded = read_payload(&mut cursor, &shape).unwrap();
        assert_eq!(cursor.position(), cursor.get_ref().len() as u64);
        assert_eq!(decoded.identity, evidence.identity);
        assert_eq!(decoded.reference.capacity, 8);
        assert_eq!(decoded.reference.prompt.as_ref(), [11, 12]);
        assert_eq!(decoded.reference.steps.len(), 1);
        assert_eq!(decoded.reference.steps[0].frontier, 2);
        assert_eq!(decoded.reference.steps[0].selected_id, 1);
        assert_eq!(decoded.reference.steps[0].max_bits, 7);
        assert_eq!(decoded.reference.steps[0].logits, [1.0, 2.0, 3.0]);
        assert_eq!(
            decoded.reference.steps[0].domains,
            vec![(StateSlot::Conv(ConvDomain(0)), vec![4.0, 5.0])]
        );
        assert_eq!(close_values(&decoded.close), close_values(&evidence.close));
    }

    #[test]
    fn evidence_publish_is_create_new_and_final_name_is_atomic() {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(1);
        let root = std::env::temp_dir().join(format!(
            "zg-qwen35-parity-publish-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let final_path = root.join("evidence.bin");

        let temporary = create_sibling_temp(&final_path).unwrap();
        let EvidenceTemp {
            parent,
            final_name,
            temporary_name,
            mut file,
        } = temporary;
        file.write_all(b"complete-evidence").unwrap();
        file.sync_all().unwrap();
        assert!(!final_path.exists());
        publish_sibling_temp(&parent, &temporary_name, &final_name).unwrap();
        assert_eq!(std::fs::read(&final_path).unwrap(), b"complete-evidence");
        assert!(fs::statat(&parent, &temporary_name, AtFlags::SYMLINK_NOFOLLOW).is_err());

        let second = create_sibling_temp(&final_path).unwrap();
        let EvidenceTemp {
            parent,
            final_name,
            temporary_name,
            mut file,
        } = second;
        file.write_all(b"must-not-overwrite").unwrap();
        file.sync_all().unwrap();
        assert!(publish_sibling_temp(&parent, &temporary_name, &final_name).is_err());
        let _ = fs::unlinkat(&parent, &temporary_name, AtFlags::empty());
        assert_eq!(std::fs::read(&final_path).unwrap(), b"complete-evidence");

        let alias = root.with_extension("alias");
        std::os::unix::fs::symlink(&root, &alias).unwrap();
        assert!(create_sibling_temp(&alias.join("other.bin")).is_err());
        std::fs::remove_file(alias).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}
