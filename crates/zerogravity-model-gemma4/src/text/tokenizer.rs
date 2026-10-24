//! Pinned E2B plain-text preparation; no chat, learned-math or serving grant.
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, BufReader, Read},
    os::unix::fs::{FileExt, MetadataExt},
    sync::atomic::{AtomicBool, Ordering},
};
use tokenizers::Tokenizer;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};
mod parse;
mod selected;
#[cfg(test)]
mod tests;

const MIB: u64 = 1024 * 1024;
pub const CONSTRUCTION_BYTES: u64 = 116 * MIB;
pub const RETAINED_BYTES: u64 = 64 * MIB;
pub const CONTROL_BYTES: u64 = 8 * MIB;
const OUTPUT_BYTES: u64 = 65536;
const CONFIG_SHA: &str = "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330";
const INDEX_SHA: &str = "832ca0ae39886d98a39483d2d644323ab1984f2c3724e6c56936578bf43d0de7";
const BUNDLE_SHA: &str = "9e9e0ad01573eff3b1da73f22c8064f5eb893d7f9f57909bc2247d97c21b28c7";
const PINS: [(u64, &str); 4] = [
    (
        32169626,
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    ),
    (
        3082,
        "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
    ),
    (
        208,
        "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
    ),
    (4954, CONFIG_SHA),
];
/// Already opened managed-HF artifacts, in explicit source roles. No new path
/// discovery, download or pathname reopening takes place inside the adapter.
pub struct Gemma4TokenizerFiles {
    pub tokenizer: File,
    pub tokenizer_config: File,
    pub generation_config: File,
    pub model_config: File,
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct Stamp([u64; 7]);
fn stamp(f: &File) -> Result<Stamp> {
    let m = f.metadata()?;
    if !m.is_file() {
        return Err(Error::Malformed("regular held tokenizer source".into()));
    }
    Ok(Stamp([
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime() as u64,
        m.mtime_nsec() as u64,
        m.ctime() as u64,
        m.ctime_nsec() as u64,
    ]))
}
fn cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|x| x.load(Ordering::Acquire)) {
        Err(Error::State("Gemma text preparation cancelled".into()))
    } else {
        Ok(())
    }
}
struct Held {
    file: File,
    stamp: Stamp,
    pin: (u64, &'static str),
}
impl Held {
    fn check(&self) -> Result<()> {
        if stamp(&self.file)? != self.stamp {
            return Err(Error::State("tokenizer held source changed".into()));
        }
        Ok(())
    }
    fn verify(&self, cancelled: Option<&AtomicBool>) -> Result<()> {
        self.check()?;
        let mut reader = Reader {
            source: self,
            at: 0,
            cancelled,
        };
        let mut hash = Sha256::new();
        let mut chunk = [0; 65536];
        loop {
            let n = reader.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            hash.update(&chunk[..n]);
        }
        if reader.at != self.pin.0 || format!("{:x}", hash.finalize()) != self.pin.1 {
            return Err(Error::State(
                "complete authentic tokenizer source SHA/EOF".into(),
            ));
        }
        self.check()
    }
}
struct Reader<'a> {
    source: &'a Held,
    at: u64,
    cancelled: Option<&'a AtomicBool>,
}
impl Read for Reader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        cancel(self.cancelled).map_err(|e| io::Error::other(e.to_string()))?;
        self.source
            .check()
            .map_err(|e| io::Error::other(e.to_string()))?;
        let n = self.source.file.read_at(out, self.at)?;
        self.source
            .check()
            .map_err(|e| io::Error::other(e.to_string()))?;
        self.at += n as u64;
        Ok(n)
    }
}
fn admit(
    host: &TransferAdmission,
    bytes: u64,
    kind: AllocationClass,
) -> Result<GlobalResourceLease> {
    host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest { kind, bytes }],
    )
}
struct Operation<'a>(&'a AtomicBool);
impl Drop for Operation<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
fn operation(busy: &AtomicBool) -> Result<Operation<'_>> {
    busy.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .map_err(|_| Error::State("Gemma text operation already active".into()))?;
    Ok(Operation(busy))
}
/// The upstream tokenizer stays private so no parsed owner can escape its lease.
/// One nonblocking operation at a time shares the admitted8MiB control scratch.
pub struct Gemma4TextTokenizer {
    inner: Tokenizer,
    busy: AtomicBool,
    files: Vec<Held>,
    host: TransferAdmission,
    _retained: GlobalResourceLease,
    _control: GlobalResourceLease,
}
pub struct Gemma4EncodedText {
    ids: Vec<u32>,
    _admission: GlobalResourceLease,
}
impl Gemma4EncodedText {
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}
pub struct Gemma4DecodedText {
    text: String,
    _admission: GlobalResourceLease,
}
impl Gemma4DecodedText {
    pub fn as_str(&self) -> &str {
        &self.text
    }
}
/// Only bounded computed IDs and immutable source provenance survive tokenizer
/// retirement. This is a user input, not a state/root/publication authority.
pub struct Gemma4PreparedText {
    files: Vec<Held>,
    busy: AtomicBool,
    host: TransferAdmission,
    encoded: Gemma4EncodedText,
}
impl Gemma4PreparedText {
    pub fn ids(&self) -> &[u32] {
        self.encoded.ids()
    }
    pub fn verify_source(&self, cancelled: Option<&AtomicBool>) -> Result<()> {
        let _operation = operation(&self.busy)?;
        self.files.iter().try_for_each(|f| f.verify(cancelled))
    }
    pub fn require_model(&self, model: &super::OriginalGemma4Text) -> Result<()> {
        let _operation = operation(&self.busy)?;
        self.files.iter().try_for_each(Held::check)?;
        if model.config_sha256() != CONFIG_SHA
            || model.source().verified_shard_index_sha256() != INDEX_SHA
            || model.source().verified_bundle_sha256() != BUNDLE_SHA
        {
            return Err(Error::State(
                "prepared text and model source closure differ".into(),
            ));
        }
        Ok(())
    }
    /// Whole output sequence decoded from authentic selected vocabulary rows.
    /// No BPE/merge graph is retained alongside the model. Same upstream decoder.
    pub fn decode_output(
        &self,
        ids: &[u32],
        skip_special: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4DecodedText> {
        let _operation = operation(&self.busy)?;
        cancel(cancelled)?;
        if ids.len() > 64 || ids.iter().any(|id| *id >= 262144) {
            return Err(Error::Limit("bounded Gemma output source IDs".into()));
        }
        // Source vocabulary strings<=93B, selected<=64, specials24; typed
        // streaming parse ignores merges, with64KiB reader/at most8MiB control.
        let _scratch = admit(&self.host, CONTROL_BYTES, AllocationClass::Scratch)?;
        let admission = admit(&self.host, OUTPUT_BYTES, AllocationClass::Output)?;
        self.files
            .iter()
            .try_for_each(|file| file.verify(cancelled))?;
        let text = selected::decode(&self.files[0], ids, skip_special, cancelled)?;
        if text.capacity() > 16384 {
            return Err(Error::Limit("bounded source output decoder".into()));
        }
        self.files.iter().try_for_each(Held::check)?;
        cancel(cancelled)?;
        Ok(Gemma4DecodedText {
            text,
            _admission: admission,
        })
    }
    /// Authentic generation stop IDs only; no sampler/default-math grant.
    pub fn eos_ids(&self) -> [u32; 3] {
        [1, 106, 50]
    }
}
impl Gemma4TextTokenizer {
    pub fn prepare(
        files: Gemma4TokenizerFiles,
        host: &TransferAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        cancel(cancelled)?;
        let control = admit(host, CONTROL_BYTES, AllocationClass::Scratch)?;
        let retained = admit(host, RETAINED_BYTES, AllocationClass::Other)?;
        let scratch = admit(
            host,
            CONSTRUCTION_BYTES - RETAINED_BYTES,
            AllocationClass::Scratch,
        )?;
        let files = [
            files.tokenizer,
            files.tokenizer_config,
            files.generation_config,
            files.model_config,
        ]
        .into_iter()
        .zip(PINS)
        .map(|(file, pin)| {
            let s = stamp(&file)?;
            if s.0[2] != pin.0 {
                return Err(Error::Malformed(
                    "exact tokenizer artifact extent/role".into(),
                ));
            }
            let held = Held {
                file,
                stamp: s,
                pin,
            };
            held.verify(cancelled)?;
            Ok(held)
        })
        .collect::<Result<Vec<_>>>()?;
        // Authenticate every byte before the only large parse. Each subsequent
        // reader refill checks inode/mtime/ctime before AND after reading, so
        // changed input cannot inject an unbounded string after authentication.
        let reader = BufReader::with_capacity(
            65536,
            Reader {
                source: &files[0],
                at: 0,
                cancelled,
            },
        );
        let mut de = serde_json::Deserializer::from_reader(reader);
        let parse::Parsed(inner) = serde::Deserialize::deserialize(&mut de)
            .map_err(|e| Error::Malformed(format!("source tokenizer: {e}")))?;
        de.end()
            .map_err(|e| Error::Malformed(format!("complete tokenizer JSON: {e}")))?;
        files.iter().try_for_each(Held::check)?;
        cancel(cancelled)?;
        drop(scratch);
        Ok(Self {
            inner,
            busy: AtomicBool::new(false),
            files,
            host: host.clone(),
            _retained: retained,
            _control: control,
        })
    }
    pub fn encode_plain(
        &self,
        text: &str,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4EncodedText> {
        let _operation = operation(&self.busy)?;
        cancel(cancelled)?;
        if text.len() > 4096 {
            return Err(Error::Limit(
                "Gemma plain text exceeds4096 UTF8 bytes".into(),
            ));
        }
        let admission = admit(&self.host, OUTPUT_BYTES, AllocationClass::Output)?;
        let encoded = self
            .inner
            .encode(text, true)
            .map_err(|e| Error::Malformed(format!("Gemma plain tokenizer: {e}")))?;
        if encoded.len() > 64 || encoded.get_ids().iter().any(|id| *id >= 262144) {
            return Err(Error::Limit(
                "Gemma diagnostic prompt exceeds64 source tokens".into(),
            ));
        }
        let ids = encoded.get_ids().to_vec();
        cancel(cancelled)?;
        Ok(Gemma4EncodedText {
            ids,
            _admission: admission,
        })
    }
    /// Source vocabulary lookup; no expected IDs or parsed owner escape.
    pub fn token_id(&self, piece: &str) -> Result<u32> {
        if piece.len() > 93 {
            return Err(Error::Limit("bounded Gemma source piece".into()));
        }
        self.inner
            .token_to_id(piece)
            .ok_or_else(|| Error::Malformed("piece absent from authentic Gemma vocabulary".into()))
    }
    pub fn decode(
        &self,
        ids: &[u32],
        skip_special: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4DecodedText> {
        let _operation = operation(&self.busy)?;
        cancel(cancelled)?;
        if ids.len() > 64 || ids.iter().any(|id| *id >= 262144) {
            return Err(Error::Limit("bounded Gemma decode source IDs".into()));
        }
        let admission = admit(&self.host, OUTPUT_BYTES, AllocationClass::Output)?;
        let text = self
            .inner
            .decode(ids, skip_special)
            .map_err(|e| Error::Malformed(format!("Gemma source decode: {e}")))?;
        if text.capacity() > 16384 {
            return Err(Error::Limit("bounded decoded source output".into()));
        }
        cancel(cancelled)?;
        Ok(Gemma4DecodedText {
            text,
            _admission: admission,
        })
    }
    pub fn verify_source(&self, cancelled: Option<&AtomicBool>) -> Result<()> {
        let _operation = operation(&self.busy)?;
        self.files.iter().try_for_each(|f| f.verify(cancelled))
    }
    pub fn into_plain_text(
        self,
        text: &str,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4PreparedText> {
        let encoded = self.encode_plain(text, cancelled)?;
        if encoded.ids.is_empty() {
            return Err(Error::Malformed(
                "empty text has no source tokens; no invented BOS".into(),
            ));
        }
        self.verify_source(cancelled)?;
        let Self {
            inner,
            busy,
            files,
            host,
            _retained,
            _control,
        } = self;
        drop((inner, _retained, _control));
        Ok(Gemma4PreparedText {
            files,
            busy,
            host,
            encoded,
        })
    }
}
