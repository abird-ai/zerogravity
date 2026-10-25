//! Test-only compositional transition archive; never a production input reader.
use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{DirBuilder, File, Metadata, OpenOptions},
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, FileExt, MetadataExt, OpenOptionsExt},
    },
};
use zerogravity::session_root::RootBinding;

const PARENT_SCOPE_SHA: &str = "7e377543f8383c52b7ae56989dedb7bc4fa4fbb746bd795dad1014c854d4a303";
const ROOT_README_SHA: &str = "6c00873ec0ac2c59bc73703b55a9bfa4d93703334a03fa16bcac2d204cda5b15";
const MEMBER_CAP: usize = 32 * MIB;
const ARCHIVE_CAP: usize = 320 * MIB;
const SECTION_CAP: usize = MIB;
const METADATA_CAP: usize = 4 * MIB;
fn source_inventory(family: &str) -> Result<(&'static str, &'static str)> {
    match family {
        "0.8b" => Ok((include_str!("../../src/short_hf/source08.json"), "Qwen08")),
        "2b" => Ok((include_str!("../../src/short_hf/source2b.json"), "Qwen2B")),
        _ => Err(fail("export unsupported source family")),
    }
}

#[derive(Clone, Copy)]
pub(super) struct History {
    ids: [u32; 8],
    scores: [u32; 8],
    roots: [Option<RootBinding>; 8],
    widths: [u8; 8],
    count: usize,
}
impl History {
    pub(super) const fn empty() -> Self {
        Self {
            ids: [0; 8],
            scores: [0; 8],
            roots: [None; 8],
            widths: [0; 8],
            count: 0,
        }
    }
    pub(super) fn commit(
        &mut self,
        root: RootBinding,
        id: u32,
        score: u32,
        width: u8,
    ) -> Result<()> {
        ensure(
            self.count < 8 && id < 248320 && matches!(width, 1 | 4),
            "bounded actual natural commit",
        )?;
        ensure(
            root.frontier() == 47 + self.count as u64,
            "natural history actual frontier",
        )?;
        if self.count > 0 {
            let prior = self.roots[self.count - 1].ok_or_else(|| fail("natural history root"))?;
            ensure(
                root.source() == prior.source()
                    && root.session() == prior.session()
                    && root.session_epoch() == prior.session_epoch()
                    && root.root() == prior.root() + 1,
                "natural history genuine current-root lineage",
            )?;
        }
        self.ids[self.count] = id;
        self.scores[self.count] = score;
        self.roots[self.count] = Some(root);
        self.widths[self.count] = width;
        self.count += 1;
        Ok(())
    }
    fn metadata(
        &self,
        root: RootBinding,
        prompt: &[u32],
        input: &[u32],
        logical_rows: usize,
    ) -> Result<Value> {
        ensure(
            prompt.len() == 47 && self.count <= 8,
            "bounded authentic prompt/history",
        )?;
        if logical_rows == 47 {
            ensure(
                root.frontier() == 0 && self.count == 0 && input == prompt,
                "actual public N1 prefill history",
            )?;
        } else {
            ensure(
                logical_rows == 1
                    && self.count > 0
                    && root.frontier() == 46 + self.count as u64
                    && self.roots[self.count - 1] == Some(root)
                    && input == [self.ids[self.count - 1]],
                "export authentic current-root natural feedback",
            )?;
        }
        let commits = (0..self.count).map(|i| {
            let root = self.roots[i].ok_or_else(||fail("actual history commit root"))?;
            Ok(json!({"id":self.ids[i],"score_bits":self.scores[i],"physical_width":self.widths[i],"root":root_metadata(root)}))
        }).collect::<Result<Vec<_>>>()?;
        let full_ids = prompt
            .iter()
            .chain(&self.ids[..self.count])
            .copied()
            .collect::<Vec<_>>();
        ensure(
            full_ids.len()
                == if logical_rows == 47 {
                    47
                } else {
                    root.frontier() as usize + 1
                },
            "genuine consumed sequence with current feedback",
        )?;
        Ok(
            json!({"prompt_ids":prompt,"natural_history":commits,"full_ids":full_ids,
            "feedback":if self.count==0 {Value::Null} else {json!({"id":self.ids[self.count-1],"score_bits":self.scores[self.count-1],"committed":true})},
            "root":root_metadata(root),"base":root.frontier(),"end":root.frontier()+logical_rows as u64}),
        )
    }
}
fn root_metadata(root: RootBinding) -> Value {
    json!({"source_debug":format!("{:?}",root.source()), "session":root.session().get(),
        "root":root.root(), "session_epoch":root.session_epoch(), "frontier":root.frontier()})
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
fn hash(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn protocol(prompt: &[u32], model_profile: &str) -> Result<Value> {
    ensure(
        prompt.len() == 47 && prompt.iter().all(|id| *id < 248320),
        "exact source canonical prompt IDs",
    )?;
    let mut raw = [0u8; 47 * 4];
    for (out, id) in raw.chunks_exact_mut(4).zip(prompt) {
        out.copy_from_slice(&id.to_le_bytes());
    }
    Ok(
        json!({"prompt_profile":"canonical47-repeated-v1","prompt_u32_le_sha256":hash(&raw),
        "native_definition_source_scope_sha256":if model_profile == "Qwen2B" {
            native::short_hf_cohort_source_scope_sha256(zerogravity::accelerator::program::FullModelProfile::Qwen2B)
        } else { PARENT_SCOPE_SHA },
        "original_source7_closure_sha256":"572a76b4016f23c8b716b5088b05bc5426a5f5c4f1efde637a99b1a6f3ea96ba",
        "canonical_state":"full512-F32-prior-and-candidate; native544-spare-excluded",
        "inference_support_granted":false}),
    )
}
fn validate_state(raw: &[u8], kind: &str, active: usize) -> Result<()> {
    let bytes = match kind {
        "conv" => 6144 * 3 * 4,
        "h" | "kv" => 512 * 512 * 4,
        _ => return Err(fail("closed canonical state codec")),
    };
    ensure(
        raw.len() == bytes && active <= 512,
        "canonical state full footprint before inspection",
    )?;
    ensure(
        raw.chunks_exact(4).all(|word| {
            let bits = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            bits & 0x7f800000 != 0x7f800000 && (kind == "h" || bits & 0xffff == 0)
        }),
        "finite canonical F32/BF16-widened state ABI",
    )?;
    if active == 0 {
        ensure(
            raw.iter().all(|v| *v == 0),
            "authentic empty root positive zero",
        )?;
    }
    if kind == "kv" {
        ensure(
            raw[active * 512 * 4..].iter().all(|v| *v == 0),
            "canonical future KV remains positive zero",
        )?;
    }
    Ok(())
}

pub(super) enum Completed<'a, 'b> {
    N1(&'a native::ShortHfReadback<'b>),
    B4(&'a native::ShortHfCohortReadback<'b>),
}
impl Completed<'_, '_> {
    fn width(&self) -> usize {
        match self {
            Self::N1(_) => 1,
            Self::B4(r) => r.records.len(),
        }
    }
    fn record(&self, row: usize) -> &zerogravity::accelerator::program::RecordedVllm08Model {
        match self {
            Self::N1(r) => r.record(),
            Self::B4(r) => &r.records[row],
        }
    }
    fn selected(&self, row: usize) -> (u32, u32) {
        match self {
            Self::N1(r) => r.greedy(),
            Self::B4(r) => r.selected[row],
        }
    }
    fn value(&self, row: usize, id: ValueId) -> Result<&[u8]> {
        match self {
            Self::N1(r) => r.value(id),
            Self::B4(r) => r.value(row, id),
        }
    }
    fn sum(&self, row: usize, node: usize) -> Result<&[u8]> {
        match self {
            Self::N1(r) => r.rms_sum(node),
            Self::B4(r) => r.rms_sum(row, node),
        }
    }
    fn state(
        &self,
        row: usize,
        slot: StateSlot,
        candidate: bool,
    ) -> Result<(&[u8], Option<&[u8]>)> {
        match self {
            Self::N1(r) => r.state(slot, candidate),
            Self::B4(r) => compare::state(r, row, slot, candidate),
        }
    }
}

struct Member {
    name: String,
    file: File,
    bytes: usize,
    hash: Sha256,
}
pub(super) struct Export {
    path: PathBuf,
    directory: File,
    directory_identity: (u64, u64),
    _charge: Option<zerogravity::resource_arbiter::GlobalResourceLease>,
    source: Value,
    native: Value,
    members: Vec<Member>,
    sections: Vec<Value>,
    bytes: usize,
    limit: usize,
    done: bool,
}
impl Export {
    fn configured(
        model: &short_hf::ShortHfModel,
        host: &TransferAdmission,
    ) -> Result<Option<Self>> {
        let host_limit = fixture_host::declared_limit(host)?;
        // Before JSON, hashes, file buffers or archive metadata; no payload clone.
        ensure(
            model
                .resident()
                .resources()
                .host_upper_bytes
                .checked_add(METADATA_CAP)
                .is_some_and(|n| n as u64 <= host_limit),
            "export metadata fits declared original HOST domain",
        )?;
        let charge = host.arbiter.admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: METADATA_CAP as u64,
            }],
        )?;
        let Some(path) = std::env::var_os("ZG_SHORT_HF_SAME_PRIOR_EXPORT") else {
            return Ok(None);
        };
        let path = PathBuf::from(path);
        let parent = path
            .parent()
            .ok_or_else(|| fail("export absolute parent"))?;
        ensure(
            path.is_absolute() && path.file_name().is_some() && parent.canonicalize()? == parent,
            "export canonical absolute fresh directory",
        )?;
        DirBuilder::new().mode(0o700).create(&path)?;
        let directory = File::open(&path)?;
        let meta = directory.metadata()?;
        ensure(
            meta.is_dir() && !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
            "held export directory",
        )?;
        let native_path = env("ZG_SHORT_HF_NATIVE")?.join("short-hf-native.json");
        ensure(
            std::fs::symlink_metadata(&native_path)?.is_file(),
            "native manifest regular file",
        )?;
        let mut native_file = File::open(&native_path)?;
        let before = native_file.metadata()?;
        ensure(
            before.is_file()
                && before.len() <= 64 * 1024
                && before.len() > 0
                && identity(&before) == identity(&std::fs::metadata(&native_path)?),
            "native manifest held exact FD",
        )?;
        let mut raw = Vec::with_capacity(before.len() as usize);
        (&mut native_file)
            .take(64 * 1024 + 1)
            .read_to_end(&mut raw)?;
        ensure(
            raw.len() as u64 == before.len()
                && identity(&before) == identity(&native_file.metadata()?)
                && identity(&before) == identity(&std::fs::metadata(&native_path)?),
            "native manifest immutable complete read",
        )?;
        let native = json!({"name":"short-hf-native.json", "bytes":raw.len(), "sha256":hash(&raw),
            "image_identity":"compiled source7 finite native image kinds independently verified by original loader"});
        let m = model.source_metadata();
        ensure(
            matches!((m.family, m.hidden_width), ("0.8b", 1024) | ("2b", 2048))
                && m.vocabulary_size == 248320,
            "export exact source family",
        )?;
        let (source_inventory, model_profile) = source_inventory(m.family)?;
        ensure(
            hash(source_inventory.as_bytes()) == m.manifest_sha256,
            "export same actual loaded tensor/shard inventory",
        )?;
        let inventory: Value = serde_json::from_str(source_inventory)
            .map_err(|e| fail(format!("pinned source inventory: {e}")))?;
        let source = json!({"family":m.family,"repository":m.repository,"revision":m.revision,
            "manifest_sha256":m.manifest_sha256,"index_sha256":m.index_sha256,
            "source_parent_scope_sha256":if model_profile == "Qwen2B" {
            "f1c7736767057eb5ca49f073bfdbd9666b6e3bf7fcab3e42325b5a806e4d9fc4"
        } else { PARENT_SCOPE_SHA },"root_readme_sha256":ROOT_README_SHA,
            "source_inventory":inventory,"model_profile":model_profile});
        Ok(Some(Self {
            path,
            directory,
            directory_identity: (meta.dev(), meta.ino()),
            _charge: Some(charge),
            source,
            native,
            members: Vec::with_capacity(10),
            sections: Vec::with_capacity(1024),
            bytes: 0,
            limit: ARCHIVE_CAP,
            done: false,
        }))
    }
    fn require_directory(&self) -> Result<()> {
        let fd = self.directory.metadata()?;
        let path = std::fs::symlink_metadata(&self.path)?;
        ensure(
            fd.is_dir()
                && path.is_dir()
                && (fd.dev(), fd.ino()) == self.directory_identity
                && (path.dev(), path.ino()) == self.directory_identity,
            "export held directory/path identity",
        )
    }
    fn fixed_path(&self, name: &str) -> PathBuf {
        // File names are generated constants; directory FD defeats ancestor races.
        PathBuf::from(format!(
            "/proc/self/fd/{}/{}",
            self.directory.as_raw_fd(),
            name
        ))
    }
    fn create(&self, name: &str) -> Result<File> {
        self.require_directory()?;
        Ok(OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(self.fixed_path(name))?)
    }
    fn write_json(&self, name: &str, value: &Value) -> Result<(usize, String)> {
        let mut writer = BoundedWriter {
            file: self.create(name)?,
            bytes: 0,
            hash: Sha256::new(),
        };
        serde_json::to_writer(&mut writer, value)
            .map_err(|e| fail(format!("bounded {name}: {e}")))?;
        writer.file.sync_all()?;
        let before = writer.file.metadata()?;
        let sha = format!("{:x}", writer.hash.finalize());
        ensure(
            before.is_file()
                && before.nlink() == 1
                && before.len() == writer.bytes as u64
                && identity(&before)
                    == identity(&std::fs::symlink_metadata(self.fixed_path(name))?),
            "held complete JSON extent",
        )?;
        let mut scratch = vec![0u8; 64 * 1024];
        let mut actual = Sha256::new();
        let mut offset = 0;
        while offset < writer.bytes {
            let bytes = scratch.len().min(writer.bytes - offset);
            writer
                .file
                .read_exact_at(&mut scratch[..bytes], offset as u64)?;
            actual.update(&scratch[..bytes]);
            offset += bytes;
        }
        ensure(
            writer.file.read_at(&mut [0u8; 1], offset as u64)? == 0
                && format!("{:x}", actual.finalize()) == sha
                && identity(&before) == identity(&writer.file.metadata()?)
                && identity(&before)
                    == identity(&std::fs::symlink_metadata(self.fixed_path(name))?),
            "JSON held FD hash/EOF/identity",
        )?;
        self.require_directory()?;
        self.directory.sync_all()?;
        Ok((writer.bytes, sha))
    }
    fn push(&mut self, mut section: Value, raw: &[u8]) -> Result<()> {
        ensure(
            !raw.is_empty()
                && raw.len() <= SECTION_CAP
                && self.sections.len() < 1024
                && self
                    .bytes
                    .checked_add(raw.len())
                    .is_some_and(|n| n <= self.limit),
            "bounded archive section before write",
        )?;
        ensure(
            matches!(section["dtype"].as_str(), Some("bf16-le" | "f32-le")),
            "archive closed scalar ABI",
        )?;
        let word = if section["dtype"] == "bf16-le" { 2 } else { 4 };
        let elements = section["shape"]
            .as_array()
            .ok_or_else(|| fail("archive shape"))?
            .iter()
            .try_fold(1usize, |n, v| n.checked_mul(v.as_u64()? as usize))
            .ok_or_else(|| fail("archive shape overflow"))?;
        ensure(
            elements.checked_mul(word) == Some(raw.len()),
            "exact archive dtype/shape/extent",
        )?;
        if self
            .members
            .last()
            .is_none_or(|m| m.bytes + raw.len() > MEMBER_CAP)
        {
            ensure(self.members.len() < 10, "archive bounded fixed members")?;
            let name = format!("raw-{:03}.bin", self.members.len());
            self.members.push(Member {
                file: self.create(&name)?,
                name,
                bytes: 0,
                hash: Sha256::new(),
            });
        }
        let member = self
            .members
            .last_mut()
            .ok_or_else(|| fail("archive member"))?;
        section["member"] = json!(member.name);
        section["file_offset"] = json!(member.bytes);
        section["global_offset"] = json!(self.bytes);
        section["bytes"] = json!(raw.len());
        section["sha256"] = json!(hash(raw));
        member.file.write_all(raw)?;
        member.hash.update(raw);
        member.bytes += raw.len();
        self.bytes += raw.len();
        self.sections.push(section);
        Ok(())
    }
    fn value(
        &mut self,
        read: &Completed<'_, '_>,
        row: usize,
        id: ValueId,
        role: &str,
        layer: Option<usize>,
    ) -> Result<()> {
        let raw = read.value(row, id)?;
        let logical_rows = read.record(row).call().rows();
        self.push(
            json!({"role":role,"kind":"producer","phase":if logical_rows==47 {"prefill"} else {"decode"},
            "codec":"bf16-dense-row-major-v1","row":row,"value_id":id.0,"layer":layer,
            "dtype":"bf16-le","shape":[if role=="logits" {1} else {read.record(row).call().rows()},raw.len()/2/if role=="logits" {1} else {read.record(row).call().rows()}]}),
            raw,
        )
    }
    pub(super) fn completed(
        &mut self,
        read: &Completed<'_, '_>,
        roots: &[RootBinding],
        prompt: &[u32],
        histories: &[History],
    ) -> Result<String> {
        ensure(
            !self.done
                && self.members.is_empty()
                && matches!(roots.len(), 1 | 4)
                && read.width() == roots.len()
                && histories.len() == roots.len(),
            "one-shot complete actual source transition",
        )?;
        self.done = true;
        let mut rows = Vec::with_capacity(roots.len());
        let logical_rows = read.record(0).call().rows();
        ensure(matches!(logical_rows, 1 | 47), "source transition rows1/47")?;
        for row in 0..roots.len() {
            let record = read.record(row);
            let span = record
                .call()
                .request()
                .teachers()
                .short_hf_span()
                .ok_or_else(|| fail("export original span"))?;
            ensure(
                span.len() == logical_rows
                    && span.base() as u64 == roots[row].frontier()
                    && record.call().source() == roots[row].source()
                    && record.boundaries().len() == 24,
                "export exact actual current source/root/span",
            )?;
            let mut row_meta =
                histories[row].metadata(roots[row], prompt, span.ids(), logical_rows)?;
            row_meta["ordinal"] = json!(row);
            row_meta["profile"] = json!(format!("{:?}", record.plan().numerical_profile()));
            row_meta["selected"] = json!({"id":read.selected(row).0,"score_bits":read.selected(row).1,"committed":false});
            rows.push(row_meta);
            for (layer, [branch, residual]) in record.boundaries().iter().enumerate() {
                self.value(read, row, *branch, "boundary.branch", Some(layer))?;
                self.value(read, row, *residual, "boundary.residual", Some(layer))?;
            }
            self.value(read, row, record.normalized(), "final.normalized", None)?;
            self.value(read, row, ValueId(325), "final.residual", None)?;
            self.value(read, row, record.logits(), "logits", None)?;
            for id in 0..10 {
                self.value(read, row, ValueId(id), "early", Some(0))?;
            }
            let nodes = sum_nodes(record);
            ensure(nodes.len() == 24, "export all24 dual sums")?;
            for node in nodes {
                self.push(json!({"role":"rms.sum","kind":"rms-unrounded-sum","phase":if logical_rows==47 {"prefill"} else {"decode"},"codec":"f32-dense-row-major-v1","row":row,"node":node,"dtype":"f32-le","shape":[logical_rows,record.call().model_profile().hidden_size()]}),read.sum(row,node)?)?;
            }
            ensure(
                record.plan().proposed_domains().len() == 42,
                "export all42 state domains",
            )?;
            for version in record.plan().proposed_domains() {
                let (kind, layer, shape) = match version.slot {
                    StateSlot::Conv(zerogravity::accelerator::program::ConvDomain(l)) => {
                        ("conv", l, vec![6144, 3])
                    }
                    StateSlot::Recurrent(zerogravity::accelerator::program::RecurrentDomain(l)) => {
                        ("h", l, vec![16, 128, 128])
                    }
                    StateSlot::Kv(zerogravity::accelerator::program::KvDomain(l)) => {
                        ("kv", l, vec![512, 512])
                    }
                };
                for candidate in [false, true] {
                    let state = read.state(row, version.slot, candidate)?;
                    ensure(
                        state.1.is_some() == (kind == "kv"),
                        "export exact canonical state planes",
                    )?;
                    for (plane, raw) in [Some(state.0), state.1].into_iter().flatten().enumerate() {
                        let active = span.base() + if candidate { logical_rows } else { 0 };
                        validate_state(raw, kind, active)?;
                        self.push(json!({"role":if candidate {"state.candidate"} else {"state.prior"},
                            "kind":"state","phase":if logical_rows==47 {"prefill"} else {"decode"},
                            "codec":match kind {"conv"=>"f32-canonical-conv-channel-frame-v1","h"=>"f32-canonical-h-native-last-two-transposed-v1",_=>"f32-canonical-kv-token-flat512-v1"},
                            "active":active,"canonical_capacity":512,"native_kv_capacity":544,
                            "row":row,"layer":layer,"plane":if kind=="kv" {if plane==0 {"k"} else {"v"}} else {kind},
                            "candidate":candidate,"dtype":"f32-le","shape":shape}),raw)?;
                    }
                }
            }
        }
        ensure(
            self.sections.len() == roots.len() * (48 + 3 + 10 + 24 + 96),
            "complete archive section census",
        )?;
        let mut members = Vec::with_capacity(self.members.len());
        let mut scratch = vec![0u8; 64 * 1024];
        for member in &self.members {
            member.file.sync_all()?;
            let before = member.file.metadata()?;
            ensure(
                before.is_file()
                    && before.nlink() == 1
                    && before.len() == member.bytes as u64
                    && identity(&before)
                        == identity(&std::fs::symlink_metadata(self.fixed_path(&member.name))?),
                "archive held exact regular member",
            )?;
            let expected = format!("{:x}", member.hash.clone().finalize());
            let mut actual = Sha256::new();
            let mut offset = 0;
            while offset < member.bytes {
                let bytes = scratch.len().min(member.bytes - offset);
                member
                    .file
                    .read_exact_at(&mut scratch[..bytes], offset as u64)?;
                actual.update(&scratch[..bytes]);
                offset += bytes;
            }
            ensure(
                member.file.read_at(&mut [0u8; 1], offset as u64)? == 0
                    && format!("{:x}", actual.finalize()) == expected
                    && identity(&before) == identity(&member.file.metadata()?)
                    && identity(&before)
                        == identity(&std::fs::symlink_metadata(self.fixed_path(&member.name))?),
                "archive FD hash/EOF/identity before manifest",
            )?;
            members.push(json!({"name":member.name,"bytes":member.bytes,"sha256":expected}));
        }
        let manifest = json!({"schema":"zg.short-hf-same-prior-transition.v1",
            "evidence":"labelled-compositional-same-prior-transition", "independent_public_trajectory":false,
            "candidate_committed":false,"physical_width":roots.len(),"logical_rows":logical_rows,"mode":if logical_rows==47 {"prefill"} else {"decode"},"source":self.source,"native":self.native,"rows":rows,
            "protocol":protocol(prompt,self.source["model_profile"].as_str().ok_or_else(||fail("actual exported model profile"))?)?,
            "archive":{"bytes":self.bytes,"members":members,"sections":self.sections},
            "production_inputs_imported":false,"mathematical_accepted":false,"performance_qualified":false});
        let (manifest_bytes, sha) = self.write_json("manifest.json", &manifest)?;
        println!(
            "SHORT_HF_SAME_PRIOR_EXPORT_COMPLETE schema=zg.short-hf-same-prior-transition.v1 bytes={} sections={} manifest_bytes={} manifest_sha256={} candidate_committed=false mathematical_accepted=false",
            self.bytes,
            self.sections.len(),
            manifest_bytes,
            sha
        );
        Ok(sha)
    }
}
struct BoundedWriter {
    file: File,
    bytes: usize,
    hash: Sha256,
}

const TRACE_CAP: usize = 2 * 1024 * MIB;
type PendingRows = Vec<(RootBinding, usize, (u32, u32))>;
pub(super) struct Trace {
    root: Export,
    events: Vec<Value>,
    transitions: usize,
    bytes: usize,
    pending: Option<PendingRows>,
}
impl Trace {
    pub(super) fn configured(
        model: &short_hf::ShortHfModel,
        host: &TransferAdmission,
    ) -> Result<Self> {
        let root = Export::configured(model, host)?.ok_or_else(|| {
            fail("diagnostic trajectory requires fresh ZG_SHORT_HF_SAME_PRIOR_EXPORT")
        })?;
        Ok(Self {
            root,
            events: Vec::with_capacity(64),
            transitions: 0,
            bytes: 0,
            pending: None,
        })
    }
    pub(super) fn event(&mut self, value: Value) -> Result<()> {
        ensure(
            self.events.len() < 128 && self.pending.is_none(),
            "bounded trace event outside unpublished transition",
        )?;
        self.events.push(value);
        Ok(())
    }
    pub(super) fn roots(roots: &[RootBinding]) -> Value {
        json!(roots.iter().copied().map(root_metadata).collect::<Vec<_>>())
    }
    pub(super) fn capture(
        &mut self,
        read: Completed<'_, '_>,
        roots: &[RootBinding],
        prompt: &[u32],
        histories: &[History],
    ) -> Result<()> {
        ensure(
            self.transitions < 16 && self.pending.is_none() && self.bytes < TRACE_CAP,
            "bounded sequential trace transition",
        )?;
        let name = format!("transition-{:03}", self.transitions);
        self.root.require_directory()?;
        DirBuilder::new()
            .mode(0o700)
            .create(self.root.fixed_path(&name))?;
        let directory = File::open(self.root.fixed_path(&name))?;
        let meta = directory.metadata()?;
        let mut archive = Export {
            path: self.root.path.join(&name),
            directory,
            directory_identity: (meta.dev(), meta.ino()),
            _charge: None,
            source: self.root.source.clone(),
            native: self.root.native.clone(),
            members: Vec::with_capacity(10),
            sections: Vec::with_capacity(1024),
            bytes: 0,
            limit: ARCHIVE_CAP.min(TRACE_CAP - self.bytes),
            done: false,
        };
        // Conservative remaining trace allowance is checked before every borrowed write.
        let remaining = TRACE_CAP - self.bytes;
        ensure(remaining > 0, "trace remainder before first write")?;
        let sha = archive.completed(&read, roots, prompt, histories)?;
        ensure(archive.bytes <= remaining, "complete trace disk ceiling")?;
        self.bytes += archive.bytes;
        let pending = (0..roots.len())
            .map(|row| {
                (
                    roots[row],
                    read.record(row).call().rows(),
                    read.selected(row),
                )
            })
            .collect::<Vec<_>>();
        self.event(json!({"kind":"transition","ordinal":self.transitions,"directory":name,"manifest_sha256":sha,
            "physical_width":read.width(),"logical_rows":read.record(0).call().rows(),"bytes":archive.bytes}))?;
        self.transitions += 1;
        self.pending = Some(pending);
        Ok(())
    }
    pub(super) fn committed(
        &mut self,
        roots: &[RootBinding],
        selected: &[(u32, u32)],
    ) -> Result<()> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| fail("trace commit without actual transition"))?;
        ensure(
            roots.len() == pending.len() && selected.len() == pending.len(),
            "trace commit exact row population",
        )?;
        for ((after, selected), (before, rows, expected)) in roots.iter().zip(selected).zip(pending)
        {
            ensure(
                after.source() == before.source()
                    && after.session() == before.session()
                    && after.session_epoch() == before.session_epoch()
                    && after.root() == before.root() + 1
                    && after.frontier() == before.frontier() + *rows as u64
                    && selected == expected,
                "trace genuine publication uses only its own completed candidate/feedback",
            )?;
        }
        self.pending = None;
        self.event(json!({"kind":"commit","transition":self.transitions-1,"roots":Self::roots(roots),
            "selected":selected.iter().map(|(id,bits)|json!({"id":id,"score_bits":bits})).collect::<Vec<_>>(),
            "diagnostic_unqualified":true}))
    }
    pub(super) fn finish(mut self) -> Result<()> {
        ensure(
            self.transitions == 15 && self.pending.is_none() && self.bytes <= TRACE_CAP,
            "complete bounded mixed/refill/partial trace",
        )?;
        let manifest = json!({"schema":"zg.short-hf-diagnostic-trajectory.v1",
            "status":"DIAGNOSTIC_UNQUALIFIED_TRAJECTORY_COMPLETE", "source":self.root.source,"native":self.root.native,
            "transitions":self.transitions,"bytes":self.bytes,"events":self.events,
            "original_n1_equivalence_test_unchanged":true,"independent_public_trajectory":false,
            "mathematical_accepted":false,"serving_qualified":false,"performance_qualified":false});
        let (_, sha) = self.root.write_json("trace.json", &manifest)?;
        println!(
            "SHORT_HF_DIAGNOSTIC_UNQUALIFIED_TRAJECTORY_COMPLETE transitions={} bytes={} trace_sha256={} mathematical_accepted=false serving_qualified=false performance_qualified=false",
            self.transitions, self.bytes, sha
        );
        self.root.done = true;
        Ok(())
    }
}

impl Write for BoundedWriter {
    fn write(&mut self, raw: &[u8]) -> std::io::Result<usize> {
        if self.bytes.checked_add(raw.len()).is_none_or(|n| n > MIB) {
            return Err(std::io::Error::other("export manifest exceeds1MiB"));
        }
        let bytes = self.file.write(raw)?;
        self.bytes += bytes;
        self.hash.update(&raw[..bytes]);
        Ok(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity::{accelerator::SourceIdentity, session_root::SessionAuthority};
    fn advance(s: &mut SessionAuthority, tokens: u64) -> RootBinding {
        let a = s.begin_advance(tokens).unwrap();
        s.prepare_publish(a).unwrap().commit()
    }
    #[test]
    fn authentic_history_rejects_foreign_or_stale_roots_and_wrong_feedback() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut s = SessionAuthority::new(source).unwrap();
        let mut h = History::empty();
        let prompt = [17; 47];
        assert!(h.metadata(s.binding(), &prompt, &prompt, 47).is_ok());
        let r = advance(&mut s, 47);
        h.commit(r, 23, 0x3f800000, 1).unwrap();
        assert!(h.metadata(r, &prompt, &[23], 1).is_ok());
        assert!(h.metadata(r, &prompt, &[24], 1).is_err());
        assert!(h.commit(r, 23, 0, 1).is_err());
        let mut other = SessionAuthority::new(source).unwrap();
        let foreign = advance(&mut other, 48);
        assert!(h.commit(foreign, 24, 0, 4).is_err());
        let next = advance(&mut s, 1);
        h.commit(next, 24, 0x40000000, 4).unwrap();
        let m = h.metadata(next, &prompt, &[24], 1).unwrap();
        assert_eq!(m["full_ids"].as_array().unwrap().len(), 49);
        assert_eq!(m["natural_history"][1]["physical_width"], 4);
        assert!(h.metadata(r, &prompt, &[23], 1).is_err());
    }
    #[test]
    fn bounded_history_cannot_grow_past_source_context_or_bad_token() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut s = SessionAuthority::new(source).unwrap();
        let mut h = History::empty();
        let first = advance(&mut s, 47);
        assert!(h.commit(first, 248320, 0, 1).is_err());
        h.commit(first, 1, 0, 1).unwrap();
        for _ in 1..8 {
            let r = advance(&mut s, 1);
            h.commit(r, 1, 0, 1).unwrap();
        }
        let extra = advance(&mut s, 1);
        assert!(h.commit(extra, 1, 0, 1).is_err());
    }
    #[test]
    fn canonical_state_rejects_wrong_extent_nonfinite_and_non_bf16_transport() {
        let mut conv = vec![0u8; 6144 * 3 * 4];
        assert!(validate_state(&conv, "conv", 0).is_ok());
        conv[..4].copy_from_slice(&0x3f800001u32.to_le_bytes());
        assert!(validate_state(&conv, "conv", 1).is_err());
        conv[..4].copy_from_slice(&0x7f800000u32.to_le_bytes());
        assert!(validate_state(&conv, "conv", 1).is_err());
        assert!(validate_state(&conv[..conv.len() - 4], "conv", 1).is_err());
        let mut h = vec![0u8; 512 * 512 * 4];
        h[..4].copy_from_slice(&0x3f800001u32.to_le_bytes());
        assert!(validate_state(&h, "h", 1).is_ok());
        assert!(validate_state(&h, "h", 0).is_err());
    }
    #[test]
    fn future_kv_and_initial_roots_require_exact_positive_zero() {
        let mut kv = vec![0u8; 512 * 512 * 4];
        kv[..4].copy_from_slice(&0x80000000u32.to_le_bytes());
        assert!(validate_state(&kv, "kv", 0).is_err());
        assert!(validate_state(&kv, "kv", 1).is_ok());
        kv[512 * 4..512 * 4 + 4].copy_from_slice(&0x3f800000u32.to_le_bytes());
        assert!(validate_state(&kv, "kv", 1).is_err());
        assert!(validate_state(&kv, "kv", 513).is_err());
        assert!(validate_state(&kv, "unknown", 1).is_err());
    }
}
