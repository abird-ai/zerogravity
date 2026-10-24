//! Fixed original closed-cohort bytes are expected values, never runtime inputs.
use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{File, Metadata},
    os::unix::fs::{FileExt, MetadataExt},
};

fn number(v: &Value, key: &str) -> Result<usize> {
    v[key]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| fail(format!("original integer {key}")))
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| fail(format!("original string {key}")))
}
fn identity(m: &Metadata) -> (u64, u64, u64, i64, i64, i64, i64, u32, u64) {
    (
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
        m.mode(),
        m.nlink(),
    )
}
struct Held {
    path: PathBuf,
    file: File,
    before: Metadata,
    sha: String,
}
impl Held {
    fn open(path: PathBuf, bytes: usize, sha: &str) -> Result<Self> {
        ensure(
            std::fs::symlink_metadata(&path)?.is_file(),
            "original regular file without final symlink",
        )?;
        let file = File::open(&path)?;
        let before = file.metadata()?;
        ensure(
            before.is_file()
                && before.len() == bytes as u64
                && identity(&before) == identity(&std::fs::metadata(&path)?),
            "original held FD/path exact extent",
        )?;
        let held = Self {
            path,
            file,
            before,
            sha: sha.into(),
        };
        held.verify()?;
        Ok(held)
    }
    fn verify(&self) -> Result<()> {
        ensure(
            identity(&self.before) == identity(&self.file.metadata()?)
                && identity(&self.before) == identity(&std::fs::metadata(&self.path)?),
            "original held FD/path before-after identity",
        )?;
        let mut hash = Sha256::new();
        let mut offset = 0u64;
        let mut buffer = vec![0u8; MIB];
        while offset < self.before.len() {
            let len = buffer.len().min((self.before.len() - offset) as usize);
            self.file.read_exact_at(&mut buffer[..len], offset)?;
            hash.update(&buffer[..len]);
            offset += len as u64;
        }
        ensure(
            self.file.read_at(&mut [0u8; 1], offset)? == 0,
            "original exact held EOF",
        )?;
        ensure(
            format!("{:x}", hash.finalize()) == self.sha,
            "original whole-image/archive SHA",
        )?;
        ensure(
            identity(&self.before) == identity(&self.file.metadata()?)
                && identity(&self.before) == identity(&std::fs::metadata(&self.path)?),
            "original held source changed while hashing",
        )
    }
    fn bytes(&self, offset: usize, bytes: usize) -> Result<Vec<u8>> {
        ensure(
            bytes > 0
                && bytes <= 32 * MIB
                && offset
                    .checked_add(bytes)
                    .is_some_and(|n| n as u64 <= self.before.len()),
            "bounded original record range",
        )?;
        let mut raw = vec![0; bytes];
        self.file.read_exact_at(&mut raw, offset as u64)?;
        Ok(raw)
    }
}
pub(super) struct Original {
    _retained_metadata: zerogravity::resource_arbiter::GlobalResourceLease,
    metadata: Vec<Held>,
    members: Vec<Held>,
    records: Vec<Value>,
    steps: Vec<Value>,
    requests: Vec<Value>,
}
impl Original {
    pub(super) fn open(root: &Path, host: &TransferAdmission) -> Result<Self> {
        let charge = |bytes| {
            host.arbiter.admit(
                &host.resource,
                host.owner.clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes,
                }],
            )
        };
        let _opening = charge((256 * MIB) as u64)?;
        let retained_metadata = charge((16 * MIB) as u64)?;
        let pins = [
            (
                "cohort-validation.json",
                13233214,
                "a87d73bcc26742fe2a89241c3e29fa8fe43c0eb8f0710972f5a6738477a71854",
            ),
            (
                "plain/cohort/report.json",
                3540962,
                "a629fa5f4a591c035c6edf1163662f3188c2607efd9916bba49eff2e46657427",
            ),
            (
                "plain/result.json",
                4003,
                "ffd2023b3a47af69d9d19822b587c9c834bcfb469e9e3d0e0f5d9fef6c72a29d",
            ),
            (
                "source-closure.json",
                582,
                "07d5ce0b62c6dcf72b0cb9766a7c751a04648151140cd89c714c00e2e63654e5",
            ),
        ];
        let metadata = pins
            .into_iter()
            .map(|(name, bytes, sha)| Held::open(root.join(name), bytes, sha))
            .collect::<Result<Vec<_>>>()?;
        let json = |index: usize| -> Result<Value> {
            serde_json::from_slice(
                &metadata[index].bytes(0, metadata[index].before.len() as usize)?,
            )
            .map_err(|e| fail(format!("original fixed JSON: {e}")))
        };
        let validation = json(0)?;
        let report = json(1)?;
        let result = json(2)?;
        let closure = json(3)?;
        ensure(
            validation["status"] == "ACTUAL_ORIGINAL_COHORT_EXPECTED_ONLY"
                && validation["calls"] == 8
                && validation["width"] == 4
                && validation["boundaries"] == 24
                && validation["active_state_domains"] == 42
                && validation["mathematical_replay_qualified"] == false
                && validation["serving_qualified"] == false,
            "accepted original expected-only reference scope",
        )?;
        for v in [&validation, &report, &result, &closure] {
            ensure(
                v["prompt_profile"] == "canonical47-repeated-v1",
                "same explicit canonical47 profile",
            )?;
        }
        ensure(
            closure["sha256"] == "572a76b4016f23c8b716b5088b05bc5426a5f5c4f1efde637a99b1a6f3ea96ba",
            "actual original immutable source7 producer",
        )?;
        let array = |v: &Value, key: &str| {
            v[key]
                .as_array()
                .cloned()
                .ok_or_else(|| fail(format!("original {key} population")))
        };
        let requests = array(&result, "requests")?;
        let steps = array(&report, "steps")?;
        let records = array(&report["archive"], "records")?;
        ensure(
            requests.len() == 4
                && steps.len() == 8
                && records.len() == 4344
                && number(&report["archive"], "bytes")? == 1262868896,
            "actual original complete population",
        )?;
        let mut ids = BTreeSet::new();
        for (row, request) in requests.iter().enumerate() {
            ensure(
                number(request, "input_ordinal")? == row
                    && ids.insert(text(request, "request_id")?.to_owned())
                    && request["finish_reason"] == "length"
                    && request["prompt_ids"]
                        .as_array()
                        .is_some_and(|a| a.len() == 47)
                    && request["completion_ids"]
                        .as_array()
                        .is_some_and(|a| a.len() == 8),
                "original authoritative RequestState input-ordinal join",
            )?;
            for (call, step) in steps.iter().enumerate() {
                let matches = step["rows"]
                    .as_array()
                    .ok_or_else(|| fail("original scheduled rows"))?
                    .iter()
                    .filter(|v| v["request_id"] == request["request_id"])
                    .collect::<Vec<_>>();
                ensure(
                    number(step, "call")? == call
                        && matches.len() == 1
                        && number(matches[0], "row")? == row
                        && number(matches[0], "tokens")? == if call == 0 { 47 } else { 1 }
                        && number(matches[0], "base")? == if call == 0 { 0 } else { 46 + call },
                    "actual scheduled request identity and independent span",
                )?;
            }
        }
        let mut members = Vec::new();
        let mut cursor = 0usize;
        let mut full = Sha256::new();
        for (index, member) in array(&report["archive"], "members")?.iter().enumerate() {
            let name = format!("raw-{index:03}.bin");
            ensure(text(member, "file")? == name, "finite original member name")?;
            let bytes = number(member, "bytes")?;
            ensure(bytes > 0 && bytes <= 32 * MIB, "original member ceiling")?;
            let held = Held::open(
                root.join("plain/cohort").join(&name),
                bytes,
                text(member, "sha256")?,
            )?;
            let raw = held.bytes(0, bytes)?;
            full.update(&raw);
            let mut covered = 0usize;
            for record in records.iter().filter(|v| v["file"] == name) {
                let offset = number(record, "file_offset")?;
                let len = number(record, "bytes")?;
                ensure(
                    offset == covered
                        && number(record, "offset")? == cursor + covered
                        && offset.checked_add(len).is_some_and(|n| n <= bytes),
                    "all original record ranges cover member exactly",
                )?;
                ensure(
                    format!("{:x}", Sha256::digest(&raw[offset..offset + len]))
                        == text(record, "sha256")?,
                    "original record bit identity",
                )?;
                covered += len;
            }
            ensure(
                covered == bytes,
                "all member bytes belong to observed records",
            )?;
            cursor += bytes;
            members.push(held);
        }
        ensure(
            members.len() == 38
                && cursor == 1262868896
                && format!("{:x}", full.finalize())
                    == "fb8b19f41e5a8e2d886051cf2baf54975754065061537277c73adb44f4ed6509",
            "complete fixed original archive closure",
        )?;
        // Fixed4344-record retained tree plus eight witnessed schedules and
        // paths. Full native-validation JSON/member bytes remain opening-only.
        ensure(
            serde_json::to_vec(&records)
                .map_err(|e| fail(format!("original retained metadata: {e}")))?
                .len()
                * 4
                + MIB
                <= 16 * MIB,
            "bounded retained original index metadata",
        )?;
        Ok(Self {
            _retained_metadata: retained_metadata,
            metadata,
            members,
            records,
            steps,
            requests,
        })
    }
    pub(super) fn prompt(&self) -> Result<Vec<u32>> {
        self.ids(0, "prompt_ids")
    }
    fn ids(&self, row: usize, key: &str) -> Result<Vec<u32>> {
        self.requests
            .get(row)
            .and_then(|r| r[key].as_array())
            .ok_or_else(|| fail("original request token array"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .filter(|n| *n < 248320)
                    .map(|n| n as u32)
                    .ok_or_else(|| fail("original token range"))
            })
            .collect()
    }
    pub(super) fn input(&self, row: usize, call: usize) -> Result<Vec<u32>> {
        if call == 0 {
            self.ids(row, "prompt_ids")
        } else {
            Ok(vec![self.ids(row, "completion_ids")?[call - 1]])
        }
    }
    pub(super) fn selected(&self, row: usize, call: usize) -> Result<u32> {
        self.ids(row, "completion_ids")?
            .get(call)
            .copied()
            .ok_or_else(|| fail("original sampled call"))
    }
    pub(super) fn get(
        &self,
        row: usize,
        call: usize,
        layer: Option<usize>,
        label: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<Vec<u8>> {
        ensure(
            call < self.steps.len() && row < self.requests.len(),
            "actual original ordinal bounds",
        )?;
        let id = &self.requests[row]["request_id"];
        let found = self
            .records
            .iter()
            .filter(|v| {
                v["request_id"] == *id
                    && v["call"].as_u64() == Some(call as u64)
                    && v["label"] == label
                    && v["layer"].as_u64() == layer.map(|n| n as u64)
            })
            .collect::<Vec<_>>();
        ensure(
            found.len() == 1,
            "unique actual original row/call/layer/label",
        )?;
        let record = found[0];
        ensure(
            record["dtype"] == dtype && record["shape"] == serde_json::json!(shape),
            "typed original tensor extent",
        )?;
        let filename = text(record, "file")?;
        let index = filename
            .strip_prefix("raw-")
            .and_then(|s| s.strip_suffix(".bin"))
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| fail("closed member index"))?;
        let raw = self
            .members
            .get(index)
            .ok_or_else(|| fail("held original member"))?
            .bytes(number(record, "file_offset")?, number(record, "bytes")?)?;
        ensure(
            format!("{:x}", Sha256::digest(&raw)) == text(record, "sha256")?,
            "original selected record unchanged",
        )?;
        Ok(raw)
    }
    /// Audit the WHOLE consumer's typed reference surface before model/GPU load.
    pub(super) fn preflight(&self) -> Result<()> {
        let prompt = self.prompt()?;
        for row in 0..4 {
            ensure(
                self.ids(row, "prompt_ids")? == prompt,
                "closed canonical equal47 profile",
            )?;
            for call in 0..8 {
                let rows = if call == 0 { 47 } else { 1 };
                let base = if call == 0 { 0 } else { 46 + call };
                let input = self.get(row, call, None, "model.input_ids", "torch.int32", &[rows])?;
                let expected = self
                    .input(row, call)?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                exact(&input, &expected, "original actual scheduled input")?;
                let positions = self.get(
                    row,
                    call,
                    None,
                    "model.positions",
                    "torch.int64",
                    &[3, rows],
                )?;
                let expected = (0..3)
                    .flat_map(|_| (base..base + rows).flat_map(|v| (v as i64).to_le_bytes()))
                    .collect::<Vec<_>>();
                exact(
                    &positions,
                    &expected,
                    "original actual three-axis positions",
                )?;
                for layer in 0..24 {
                    for label in ["block.output.branch", "block.output.residual"] {
                        self.get(
                            row,
                            call,
                            Some(layer),
                            label,
                            "torch.bfloat16",
                            &[rows, 1024],
                        )?;
                    }
                    for after in [false, true] {
                        if !after && call == 0 {
                            continue;
                        }
                        let suffix = if after { "after" } else { "before" };
                        if layer % 4 == 3 {
                            let active = if after { base + rows } else { base };
                            self.get(
                                row,
                                call,
                                Some(layer),
                                &format!("state.kv.{suffix}"),
                                "torch.bfloat16",
                                &[2, active, 512],
                            )?;
                        } else {
                            self.get(
                                row,
                                call,
                                Some(layer),
                                &format!("state.conv.{suffix}"),
                                "torch.bfloat16",
                                &[3, 6144],
                            )?;
                            self.get(
                                row,
                                call,
                                Some(layer),
                                &format!("state.recurrent.{suffix}"),
                                "torch.float32",
                                &[16, 128, 128],
                            )?;
                        }
                    }
                }
                self.get(
                    row,
                    call,
                    None,
                    "final.normalized",
                    "torch.bfloat16",
                    &[rows, 1024],
                )?;
                self.get(row, call, None, "head.input", "torch.bfloat16", &[1, 1024])?;
                let logits = self.get(
                    row,
                    call,
                    None,
                    "logits.pre_sampler",
                    "torch.bfloat16",
                    &[1, 248320],
                )?;
                exact(
                    &logits,
                    &self.get(
                        row,
                        call,
                        None,
                        "sampler.input",
                        "torch.bfloat16",
                        &[1, 248320],
                    )?,
                    "full original head/sampler join",
                )?;
                ensure(
                    greedy(&logits)?.0 == self.selected(row, call)?,
                    "original independent natural greedy result",
                )?;
                exact(
                    &self.get(
                        row,
                        call,
                        None,
                        "sampler.sampled_ids",
                        "torch.int64",
                        &[1, 1],
                    )?,
                    &(self.selected(row, call)? as i64).to_le_bytes(),
                    "actual original sampled ID",
                )?;
                exact(
                    &self.get(row, call, None, "sampler.num_sampled", "torch.int32", &[1])?,
                    &1i32.to_le_bytes(),
                    "actual original sampled count",
                )?;
                exact(
                    &self.get(row, call, None, "sampler.num_rejected", "torch.int32", &[1])?,
                    &0i32.to_le_bytes(),
                    "actual original rejected count",
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn close(self) -> Result<()> {
        for file in self.metadata.iter().chain(&self.members) {
            file.verify()?;
        }
        Ok(())
    }
}
pub(super) fn exact(a: &[u8], b: &[u8], label: &str) -> Result<()> {
    ensure(
        a == b,
        &format!(
            "{label}: bytes {}/{}, first {:?}",
            a.len(),
            b.len(),
            a.iter().zip(b).position(|(a, b)| a != b)
        ),
    )
}
pub(super) fn widen(raw: &[u8]) -> Result<Vec<u8>> {
    ensure(raw.len().is_multiple_of(2), "whole BF16 words")?;
    raw.chunks_exact(2)
        .map(|v| {
            let bits = (u16::from_le_bytes([v[0], v[1]]) as u32) << 16;
            ensure(f32::from_bits(bits).is_finite(), "finite original BF16")?;
            Ok(bits.to_le_bytes())
        })
        .collect::<Result<Vec<_>>>()
        .map(|v| v.into_iter().flatten().collect())
}
pub(super) fn sum(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    ensure(a.len() == b.len(), "original dual sum operands")?;
    Ok(widen(a)?
        .chunks_exact(4)
        .zip(widen(b)?.chunks_exact(4))
        .flat_map(|(a, b)| {
            (f32::from_le_bytes(a.try_into().unwrap()) + f32::from_le_bytes(b.try_into().unwrap()))
                .to_le_bytes()
        })
        .collect())
}
pub(super) fn convolution(raw: &[u8]) -> Result<Vec<u8>> {
    ensure(raw.len() == 3 * 6144 * 2, "original Conv3 geometry")?;
    let raw = widen(raw)?;
    let mut result = Vec::with_capacity(raw.len());
    for channel in 0..6144 {
        for frame in 0..3 {
            let i = (frame * 6144 + channel) * 4;
            result.extend_from_slice(&raw[i..i + 4]);
        }
    }
    Ok(result)
}
pub(super) fn recurrent(raw: &[u8]) -> Result<Vec<u8>> {
    ensure(
        raw.len() == 16 * 128 * 128 * 4,
        "original recurrent geometry",
    )?;
    let mut result = Vec::with_capacity(raw.len());
    for head in 0..16 {
        for major in 0..128 {
            for minor in 0..128 {
                let i = (head * 16384 + minor * 128 + major) * 4;
                result.extend_from_slice(&raw[i..i + 4]);
            }
        }
    }
    Ok(result)
}
pub(super) fn greedy(raw: &[u8]) -> Result<(u32, u32)> {
    ensure(
        raw.len() == 248320 * 2,
        "complete independent original vocabulary",
    )?;
    let mut best = None;
    for (id, word) in raw.chunks_exact(2).enumerate() {
        let bits = (u16::from_le_bytes([word[0], word[1]]) as u32) << 16;
        let value = f32::from_bits(bits);
        ensure(value.is_finite(), "original logits finite")?;
        if best.is_none_or(|(_, old)| value > f32::from_bits(old)) {
            best = Some((id as u32, bits));
        }
    }
    best.ok_or_else(|| fail("original numeric argmax"))
}
