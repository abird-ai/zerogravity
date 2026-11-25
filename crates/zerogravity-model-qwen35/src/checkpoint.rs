//! Source-bound numerical checkpoint body shared by CPU and diagnostic resident owners.
//! The original v1 wire encoding is preserved. Only committed numerical state
//! and a frontier are portable: no local root, proposal, event, handle or lease.
//!
//! Model and raw-input SHA256 digests are caller-verified identities, stored and
//! compared byte for byte. Callers MUST independently verify those digests for
//! the immutable source and exact input stream. They are not calculated here;
//! the state fingerprint detects corruption, not malicious edits/authenticity.
use super::*;
#[cfg(feature = "diagnostic-cpu-checkpoint")]
use crate::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
#[cfg(feature = "diagnostic-cpu-checkpoint")]
use std::collections::BTreeMap;
#[cfg(feature = "diagnostic-cpu-checkpoint")]
use std::io::BufReader;
use std::io::{Read, Write};

#[cfg(all(feature = "state-checkpoint", feature = "cpu"))]
mod committed;

#[cfg(feature = "diagnostic-cpu-checkpoint")]
const MAGIC: [u8; 16] = *b"ZGQWENCPU_STATE1";
#[cfg(feature = "diagnostic-cpu-checkpoint")]
const VERSION: u32 = 1;

fn put_u32<W: Write + ?Sized>(w: &mut W, value: u32) -> Result<()> {
    w.write_all(&value.to_le_bytes())?;
    Ok(())
}
fn put_u64<W: Write + ?Sized>(w: &mut W, value: u64) -> Result<()> {
    w.write_all(&value.to_le_bytes())?;
    Ok(())
}
fn get_u32<R: Read + ?Sized>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn get_u64<R: Read + ?Sized>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn store_vector<W: Write + ?Sized>(w: &mut W, v: &[f32]) -> Result<()> {
    put_u32(
        w,
        u32::try_from(v.len()).map_err(|_| Error::Limit("checkpoint vector too large".into()))?,
    )?;
    let mut bytes = [0u8; 4096];
    for chunk in v.chunks(bytes.len() / 4) {
        for (word, target) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            target.copy_from_slice(&word.to_bits().to_le_bytes());
        }
        w.write_all(&bytes[..chunk.len() * 4])?;
    }
    Ok(())
}
fn load_vector<R: Read + ?Sized>(r: &mut R, expected: usize) -> Result<Vec<f32>> {
    if get_u32(r)? as usize != expected {
        return Err(Error::Malformed(
            "checkpoint tensor geometry mismatch".into(),
        ));
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(expected)
        .map_err(|_| Error::Limit("checkpoint tensor allocation refused".into()))?;
    let mut bytes = [0u8; 4096];
    while result.len() < expected {
        let words = (expected - result.len()).min(bytes.len() / 4);
        let chunk = &mut bytes[..words * 4];
        r.read_exact(chunk)?;
        for word in chunk.chunks_exact(4) {
            let value = f32::from_bits(u32::from_le_bytes(word.try_into().unwrap()));
            if !value.is_finite() {
                return Err(Error::Malformed("nonfinite stored CPU state".into()));
            }
            result.push(value);
        }
    }
    Ok(result)
}

/// Complete validated numerical state bound to one immutable loaded source.
/// It carries no process-local session/root or device handle in its wire form.
/// An in-memory view of admitted imported arrays retains their original lease.
/// Loading a verified v1 file rebinds its values to the receiving model owner.
#[cfg(feature = "diagnostic-cpu-checkpoint")]
#[derive(Clone)]
pub struct QwenCheckpoint {
    layers: Vec<LayerState>,
    position: usize,
    source: crate::accelerator::SourceIdentity,
    #[cfg(all(feature = "cpu", feature = "state-checkpoint"))]
    imported: Option<Arc<zerogravity_core::resource_arbiter::GlobalResourceLease>>,
}

#[cfg(feature = "diagnostic-cpu-checkpoint")]
impl QwenCheckpoint {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }

    /// Exact native flattened domains: all K slabs followed by all V slabs.
    /// Unused physical KV capacity is never part of this portable payload.
    pub fn domain_values(&self) -> Vec<(StateSlot, Vec<f32>)> {
        state_domain_values(&self.layers)
    }

    /// Atomically save original v1 bytes. Both SHA256 identities must already
    /// have been independently verified by the caller; no SHA is computed here.
    pub fn save(&self, path: &Path, model_sha256: [u8; 32], tokens_sha256: [u8; 32]) -> Result<()> {
        zerogravity_artifacts::checkpoint::atomic_write(path, |writer| {
            self.write_v1(writer, model_sha256, tokens_sha256)
        })
    }

    fn write_v1<W: Write + ?Sized>(
        &self,
        w: &mut W,
        model: [u8; 32],
        tokens: [u8; 32],
    ) -> Result<()> {
        w.write_all(&MAGIC)?;
        put_u32(w, VERSION)?;
        w.write_all(&model)?;
        w.write_all(&tokens)?;
        write_state(w, self.position, &self.layers)
    }
}

fn write_state<W: Write + ?Sized>(w: &mut W, position: usize, layers: &[LayerState]) -> Result<()> {
    put_u64(
        w,
        u64::try_from(position).map_err(|_| Error::Limit("checkpoint position overflow".into()))?,
    )?;
    put_u32(
        w,
        u32::try_from(layers.len())
            .map_err(|_| Error::Limit("checkpoint layer count overflow".into()))?,
    )?;
    put_u64(
        w,
        u64::try_from(state_bytes(layers))
            .map_err(|_| Error::Limit("checkpoint state bytes overflow".into()))?,
    )?;
    for layer in layers {
        match layer {
            LayerState::Delta { conv, state } => {
                w.write_all(&[1])?;
                store_vector(w, conv)?;
                store_vector(w, state)?;
            }
            LayerState::Full { keys, values } => {
                w.write_all(&[2])?;
                put_u32(
                    w,
                    u32::try_from(keys.len())
                        .map_err(|_| Error::Limit("KV sequence too large".into()))?,
                )?;
                for (k, v) in keys.iter().zip(values) {
                    store_vector(w, k)?;
                    store_vector(w, v)?;
                }
            }
        }
    }
    put_u64(w, state_fingerprint(position, layers))
}

fn checked_bytes(model: &LoadedQwenModel, position: usize) -> Result<usize> {
    model.require_execution_profile()?;
    if u64::try_from(position)
        .ok()
        .is_none_or(|p| p > model.geometry.max_position_embeddings)
    {
        return Err(Error::Limit("checkpoint exceeds model context".into()));
    }
    let kv = model
        .geometry
        .num_key_value_heads
        .checked_mul(model.geometry.head_dim)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::Limit("checkpoint KV width overflow".into()))?;
    let mut words = 0usize;
    for layer in &model.state_specs {
        let n = match layer {
            StateSpec::Delta { conv, state } => conv.checked_add(*state),
            StateSpec::Full { .. } => position.checked_mul(kv).and_then(|n| n.checked_mul(2)),
        }
        .ok_or_else(|| Error::Limit("checkpoint state dimensions overflow".into()))?;
        words = words
            .checked_add(n)
            .ok_or_else(|| Error::Limit("checkpoint state bytes overflow".into()))?;
    }
    words
        .checked_mul(4)
        .filter(|n| *n <= MAX_QWEN_STATE_BYTES)
        .ok_or_else(|| Error::Limit("checkpoint exceeds session state admission".into()))
}

#[cfg(feature = "diagnostic-cpu-checkpoint")]
fn layers_from_domains(
    initial: &[StateSpec],
    kv_width: usize,
    position: usize,
    domains: &[(StateSlot, Vec<f32>)],
) -> Result<Vec<LayerState>> {
    let mut by_slot = BTreeMap::new();
    for (slot, values) in domains {
        if by_slot.insert(*slot, values).is_some() {
            return Err(Error::State("duplicate checkpoint state domain".into()));
        }
    }
    let kv_words = position
        .checked_mul(kv_width)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| Error::Limit("checkpoint KV dimensions overflow".into()))?;
    let mut expected = Vec::new();
    for (i, layer) in initial.iter().enumerate() {
        match layer {
            StateSpec::Delta { conv, state } => {
                expected.push((StateSlot::Conv(ConvDomain(i)), *conv));
                expected.push((StateSlot::Recurrent(RecurrentDomain(i)), *state));
            }
            StateSpec::Full { .. } => expected.push((StateSlot::Kv(KvDomain(i)), kv_words)),
        }
    }
    if by_slot.len() != expected.len() {
        return Err(Error::State(
            "checkpoint state domain coverage mismatch".into(),
        ));
    }
    // Validate every numerical word and shape before cloning any tensor payload.
    for (slot, len) in &expected {
        let values = by_slot
            .get(slot)
            .ok_or_else(|| Error::State("missing checkpoint state domain".into()))?;
        if values.len() != *len || !values.iter().all(|v| v.is_finite()) {
            return Err(Error::State(
                "invalid checkpoint state domain shape/value".into(),
            ));
        }
    }
    let mut layers = Vec::with_capacity(initial.len());
    for (i, layer) in initial.iter().enumerate() {
        layers.push(match layer {
            StateSpec::Delta { .. } => LayerState::Delta {
                conv: Arc::new(by_slot[&StateSlot::Conv(ConvDomain(i))].clone()),
                state: Arc::new(by_slot[&StateSlot::Recurrent(RecurrentDomain(i))].clone()),
            },
            StateSpec::Full { .. } => {
                let words = by_slot[&StateSlot::Kv(KvDomain(i))];
                let (keys, values) = words.split_at(kv_words / 2);
                let rows = |data: &[f32]| data.chunks_exact(kv_width).map(Arc::from).collect();
                LayerState::Full {
                    keys: rows(keys),
                    values: rows(values),
                }
            }
        });
    }
    Ok(layers)
}

#[cfg(feature = "diagnostic-cpu-checkpoint")]
impl LoadedQwenModel {
    /// Build an immutable payload from complete committed numerical domains.
    /// The caller must establish physical completion and snapshot/source binding;
    /// this checks schema, every finite value, context and the CPU byte bound.
    pub fn checkpoint_from_domains(
        &self,
        position: usize,
        domains: &[(StateSlot, Vec<f32>)],
    ) -> Result<QwenCheckpoint> {
        let expected = checked_bytes(self, position)?;
        let kv_width = (self.geometry.num_key_value_heads * self.geometry.head_dim) as usize;
        let layers = layers_from_domains(&self.state_specs, kv_width, position, domains)?;
        self.validate_numerical_state(position, &layers)?;
        if state_bytes(&layers) != expected {
            return Err(Error::State(
                "checkpoint numerical byte geometry mismatch".into(),
            ));
        }
        Ok(QwenCheckpoint {
            layers,
            position,
            source: self.source_identity(),
            #[cfg(all(feature = "cpu", feature = "state-checkpoint"))]
            imported: None,
        })
    }

    /// Load original v1 bytes and bind the validated values to THIS source.
    /// The caller must independently verify both expected SHA256 identities.
    /// No session, local root or resource admission is created by decoding.
    pub fn load_checkpoint(
        &self,
        path: &Path,
        expected_model_sha256: [u8; 32],
        expected_tokens_sha256: [u8; 32],
    ) -> Result<QwenCheckpoint> {
        self.require_execution_profile()?;
        let file = zerogravity_artifacts::checkpoint::open_regular(path)?;
        if file.metadata()?.len() > MAX_QWEN_STATE_BYTES as u64 + 1_048_576 {
            return Err(Error::Limit("checkpoint exceeds CPU state budget".into()));
        }
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 16];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC || get_u32(&mut reader)? != VERSION {
            return Err(Error::Malformed(
                "unknown CPU checkpoint format version".into(),
            ));
        }
        let mut model_digest = [0u8; 32];
        let mut input_digest = [0u8; 32];
        reader.read_exact(&mut model_digest)?;
        reader.read_exact(&mut input_digest)?;
        if model_digest != expected_model_sha256 || input_digest != expected_tokens_sha256 {
            return Err(Error::State(
                "CPU checkpoint source/input identity mismatch".into(),
            ));
        }
        let position = usize::try_from(get_u64(&mut reader)?)
            .map_err(|_| Error::Limit("checkpoint position exceeds platform".into()))?;
        let expected_bytes = checked_bytes(self, position)?;
        let loaded = read_state(self, &mut reader, position, expected_bytes)?;
        let mut extra = [0u8; 1];
        if reader.read(&mut extra)? != 0 {
            return Err(Error::Malformed("checkpoint contains trailing data".into()));
        }
        Ok(QwenCheckpoint {
            layers: loaded,
            position,
            source: self.source_identity(),
            #[cfg(all(feature = "cpu", feature = "state-checkpoint"))]
            imported: None,
        })
    }
}

fn read_state<R: Read + ?Sized>(
    model: &LoadedQwenModel,
    reader: &mut R,
    position: usize,
    expected_bytes: usize,
) -> Result<Vec<LayerState>> {
    if get_u32(reader)? as usize != model.state_specs.len() {
        return Err(Error::Malformed("checkpoint wrong layer count".into()));
    }
    let stored_bytes = get_u64(reader)?;
    if stored_bytes != expected_bytes as u64 {
        return Err(Error::Malformed(
            "checkpoint state fingerprint/size mismatch".into(),
        ));
    }
    let kv_width = (model.geometry.num_key_value_heads * model.geometry.head_dim) as usize;
    let mut loaded = Vec::with_capacity(model.state_specs.len());
    for layer in &model.state_specs {
        let mut tag = [0u8; 1];
        reader.read_exact(&mut tag)?;
        match (layer, tag[0]) {
            (StateSpec::Delta { conv, state }, 1) => loaded.push(LayerState::Delta {
                conv: Arc::new(load_vector(reader, *conv)?),
                state: Arc::new(load_vector(reader, *state)?),
            }),
            (StateSpec::Full { .. }, 2) => {
                if get_u32(reader)? as usize != position {
                    return Err(Error::Malformed("checkpoint KV position mismatch".into()));
                }
                let mut keys = Vec::new();
                let mut values = Vec::new();
                keys.try_reserve_exact(position)
                    .map_err(|_| Error::Limit("checkpoint KV key allocation".into()))?;
                values
                    .try_reserve_exact(position)
                    .map_err(|_| Error::Limit("checkpoint KV value allocation".into()))?;
                for _ in 0..position {
                    keys.push(Arc::from(load_vector(reader, kv_width)?));
                    values.push(Arc::from(load_vector(reader, kv_width)?));
                }
                loaded.push(LayerState::Full { keys, values });
            }
            _ => {
                return Err(Error::Malformed(
                    "checkpoint wrong state domain type".into(),
                ));
            }
        }
    }
    let fingerprint = get_u64(reader)?;
    model.validate_numerical_state(position, &loaded)?;
    if state_bytes(&loaded) as u64 != stored_bytes
        || state_fingerprint(position, &loaded) != fingerprint
    {
        return Err(Error::Malformed(
            "checkpoint state fingerprint/size mismatch".into(),
        ));
    }
    Ok(loaded)
}

#[cfg(all(feature = "cpu", feature = "diagnostic-cpu-checkpoint"))]
impl QwenDenseCpu {
    /// Export only this session's committed numerical state, sharing its
    /// immutable CPU storage. No session/root metadata is made portable.
    pub fn checkpoint(&self) -> Result<QwenCheckpoint> {
        self.validate_state(&self.state)?;
        if self.state.attempt.is_some() || self.state.binding != self.authority.binding() {
            return Err(Error::State(
                "checkpoint requires a committed local root".into(),
            ));
        }
        checked_bytes(&self.model, self.state.position)?;
        Ok(QwenCheckpoint {
            layers: self.state.layers.clone(),
            position: self.state.position,
            source: self.model.source_identity(),
            #[cfg(feature = "state-checkpoint")]
            imported: self.state.imported.clone(),
        })
    }

    /// Replace this local session from a validated payload of the SAME loaded
    /// source. A verified disk load is required to rebind a different source
    /// allocation. Success advances root and epoch, invalidating old drafts.
    pub fn restore_checkpoint(&mut self, checkpoint: &QwenCheckpoint) -> Result<()> {
        self.authority.preflight_replacement()?;
        if checkpoint.source != self.model.source_identity() {
            return Err(Error::State(
                "checkpoint belongs to another loaded source".into(),
            ));
        }
        checked_bytes(&self.model, checkpoint.position)?;
        self.model
            .validate_numerical_state(checkpoint.position, &checkpoint.layers)?;
        let attempt = self.authority.begin_import(
            u64::try_from(checkpoint.position)
                .map_err(|_| Error::Limit("checkpoint position overflow".into()))?,
        )?;
        let candidate = QwenState {
            layers: checkpoint.layers.clone(),
            position: checkpoint.position,
            binding: self.authority.binding(),
            attempt: None,
            #[cfg(feature = "state-checkpoint")]
            imported: checkpoint.imported.clone(),
        };
        self.publish_state(candidate, attempt)
    }

    /// Original source/prompt-bound v1 CPU export API.
    pub fn save_cpu_checkpoint(
        &self,
        path: &Path,
        model_sha256: [u8; 32],
        tokens_sha256: [u8; 32],
    ) -> Result<()> {
        self.checkpoint()?.save(path, model_sha256, tokens_sha256)
    }

    /// Original source/prompt-bound v1 CPU import API; all decoding is private.
    pub fn restore_cpu_checkpoint(
        &mut self,
        path: &Path,
        model_sha256: [u8; 32],
        tokens_sha256: [u8; 32],
    ) -> Result<()> {
        self.authority.preflight_replacement()?;
        let checkpoint = self
            .model
            .load_checkpoint(path, model_sha256, tokens_sha256)?;
        self.restore_checkpoint(&checkpoint)
    }
}

#[cfg(all(test, feature = "diagnostic-cpu-checkpoint"))]
mod tests {
    use super::*;
    use std::fs;

    fn fixture() -> QwenCheckpoint {
        QwenCheckpoint {
            layers: vec![
                LayerState::Delta {
                    conv: Arc::new(vec![-0.0, 1.0]),
                    state: Arc::new(vec![
                        2.0,
                        -3.0,
                        f32::from_bits(1),
                        f32::from_bits(0x80000001),
                    ]),
                },
                LayerState::Full {
                    keys: vec![Arc::from([4.0, 5.0]), Arc::from([6.0, 7.0])],
                    values: vec![Arc::from([8.0, 9.0]), Arc::from([10.0, -0.0])],
                },
            ],
            position: 2,
            source: crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
            #[cfg(all(feature = "cpu", feature = "state-checkpoint"))]
            imported: None,
        }
    }

    #[test]
    fn portable_checkpoint_preserves_original_v1_golden_bytes() {
        // Frozen original v1 wire image: K/V rows interleave on disk while the
        // fingerprint visits all K before all V. Includes -0 and subnormals.
        let golden = "5a475157454e4350555f535441544531010000003232323232323232323232323232323232323232323232323232323232323232585858585858585858585858585858585858585858585858585858585858585802000000000000000200000038000000000000000102000000000000800000803f0400000000000040000040c00100000001000080020200000002000000000080400000a040020000000000004100001041020000000000c0400000e0400200000000002041000000801b9460c126bb894f";
        let expected = golden
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let mut bytes = Vec::new();
        let checkpoint = fixture();
        checkpoint
            .write_v1(&mut bytes, [0x32; 32], [0x58; 32])
            .unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(
            state_fingerprint(checkpoint.position, &checkpoint.layers),
            0x4f89bb26c160941b
        );
        assert_eq!(state_bytes(&checkpoint.layers), 56);
    }

    #[test]
    fn portable_domain_mapping_preserves_bits_and_rejects_incomplete_or_invalid_inputs() {
        let checkpoint = fixture();
        let specs = [
            StateSpec::Delta { conv: 2, state: 4 },
            StateSpec::Full { kv_width: 2 },
        ];
        let mut domains = checkpoint.domain_values();
        domains.reverse(); // Domain identity, not caller order, selects schema.
        let layers = layers_from_domains(&specs, 2, 2, &domains).unwrap();
        assert_eq!(state_fingerprint(2, &layers), 0x4f89bb26c160941b);
        assert_eq!(state_domain_values(&layers)[0].1[0].to_bits(), 0x80000000);
        assert!(layers_from_domains(&specs, 2, 2, &domains[..2]).is_err());
        let mut invalid = domains.clone();
        invalid.push(invalid[0].clone());
        assert!(layers_from_domains(&specs, 2, 2, &invalid).is_err());
        let mut invalid = domains.clone();
        invalid[0].0 = StateSlot::Kv(KvDomain(99));
        assert!(layers_from_domains(&specs, 2, 2, &invalid).is_err());
        let mut invalid = domains.clone();
        invalid[0].1.pop();
        assert!(layers_from_domains(&specs, 2, 2, &invalid).is_err());
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = domains.clone();
            invalid[0].1[0] = nonfinite;
            assert!(layers_from_domains(&specs, 2, 2, &invalid).is_err());
        }
        assert!(layers_from_domains(&specs, 2, usize::MAX, &domains).is_err());
    }

    #[test]
    #[cfg(feature = "cpu")]
    #[ignore = "requires both original local Qwen sources; portable CPU/domain/v1 bridge"]
    fn portable_checkpoint_shared_source_domains_wire_and_import_are_exact() {
        for name in ["qwen35-0.8b", "qwen35-2b"] {
            let folder =
                Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/models")).join(name);
            let model =
                LoadedQwenModel::open(&folder.join("model.gguf"), &folder.join("config.json"))
                    .unwrap();
            let mut original = model.new_session().unwrap();
            original.prefill_chunk_without_logits(&[11, 353]).unwrap();
            let checkpoint = original.checkpoint().unwrap();
            let from_domains = model
                .checkpoint_from_domains(checkpoint.position(), &checkpoint.domain_values())
                .unwrap();
            assert_eq!(checkpoint.source_identity(), model.source_identity());
            assert_eq!(checkpoint.position(), 2);
            let mut old = Vec::new();
            let mut packed = Vec::new();
            checkpoint
                .write_v1(&mut old, [0x32; 32], [0x58; 32])
                .unwrap();
            from_domains
                .write_v1(&mut packed, [0x32; 32], [0x58; 32])
                .unwrap();
            assert_eq!(old, packed, "{name} flattened domains changed v1 bytes");
            let path = std::env::temp_dir().join(format!(
                "zerogravity-portable-{}-{name}.bin",
                std::process::id()
            ));
            original
                .save_cpu_checkpoint(&path, [0x32; 32], [0x58; 32])
                .unwrap();
            assert_eq!(fs::read(&path).unwrap(), old);
            let loaded = model
                .load_checkpoint(&path, [0x32; 32], [0x58; 32])
                .unwrap();
            let mut sibling = model.new_session().unwrap();
            let (_, stale) = sibling.next_logits(17).unwrap();
            sibling.restore_checkpoint(&loaded).unwrap();
            assert!(sibling.commit(stale).is_err());
            assert_eq!(sibling.state_fingerprint(), original.state_fingerprint());
            assert_eq!(sibling.root_id(), 1);
            assert_eq!(sibling.authority.binding().session_epoch(), 1);
            assert_eq!(sibling.position(), 2);
            let before = sibling.authority.binding();
            let mut foreign = loaded.clone();
            foreign.source = crate::accelerator::SourceIdentity::unbound();
            assert!(sibling.restore_checkpoint(&foreign).is_err());
            assert_eq!(sibling.authority.binding(), before);
            assert_eq!(sibling.state_fingerprint(), original.state_fingerprint());
            assert!(
                model
                    .checkpoint_from_domains(
                        model.geometry.max_position_embeddings as usize + 1,
                        &checkpoint.domain_values()
                    )
                    .is_err()
            );
            fs::remove_file(path).unwrap();
        }
    }
}
