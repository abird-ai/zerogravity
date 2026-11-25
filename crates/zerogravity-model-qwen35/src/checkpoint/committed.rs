//! Opt-in persisted committed GGUF CPU numerical state. No sampler/history,
//! local root, provider handle, caller-asserted prompt digest or prefix grant.
//! The same loaded source's native CPU arithmetic/state layout is required.

use super::*;
use sha2::{Digest, Sha256};
use zerogravity_artifacts::checkpoint as io;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, TransferAdmission,
};

const MAGIC: [u8; 16] = *b"ZGQWENCPU_STATE2";
const VERSION: u32 = 2;
// The wire schema fixes this exact CPU reference policy. Change the schema
// version on an incompatible numerical/layout change, not the model label.
const PROFILE: &[u8] = b"qwen35/gguf/reference-cpu-v1";
const HEADER_BYTES: u64 = 16 + 4 + 32;
const MAX_PAYLOAD_BYTES: u64 = MAX_QWEN_STATE_BYTES as u64 + 16 * 1024 * 1024;

fn reserve(
    host: &TransferAdmission,
    kind: AllocationClass,
    bytes: u64,
) -> Result<GlobalResourceLease> {
    if host.resource.backend != "host" {
        return Err(Error::State(
            "committed checkpoint requires original host admission".into(),
        ));
    }
    host.arbiter.check_host_cold_allocation(bytes)?;
    host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest { kind, bytes }],
    )
}

impl LoadedQwenModel {
    /// Content + actual semantic geometry, not a caller/path-derived digest.
    /// Only the existing finite owned GGUF CPU model is eligible.
    fn committed_checkpoint_identity(&self) -> Result<[u8; 32]> {
        self.require_execution_profile()?;
        let g = &self.geometry;
        let mut hash = Sha256::new();
        hash.update(PROFILE);
        hash.update(self.weights.gguf_content_sha256()?);
        for value in [
            g.hidden_size,
            g.head_dim,
            g.num_attention_heads,
            g.num_key_value_heads,
            g.vocab_size,
            g.max_position_embeddings,
            g.recurrent_key_heads.unwrap_or(0),
            g.recurrent_value_heads.unwrap_or(0),
            g.layers.len() as u64,
        ] {
            hash.update(value.to_le_bytes());
        }
        hash.update(self.eps.to_bits().to_le_bytes());
        for spec in &self.state_specs {
            match *spec {
                StateSpec::Delta { conv, state } => {
                    hash.update([1]);
                    hash.update((conv as u64).to_le_bytes());
                    hash.update((state as u64).to_le_bytes());
                }
                StateSpec::Full { kv_width } => {
                    hash.update([2]);
                    hash.update((kv_width as u64).to_le_bytes());
                }
            }
        }
        Ok(hash.finalize().into())
    }

    fn committed_wire_bytes(&self, position: usize, payload: usize) -> Result<u64> {
        // Position, layer count, state bytes and original numerical fingerprint.
        let mut bytes = HEADER_BYTES + 8 + 4 + 8 + 8 + payload as u64;
        for spec in &self.state_specs {
            let extra = match spec {
                StateSpec::Delta { .. } => Some(1 + 4 + 4),
                StateSpec::Full { .. } => (position as u64)
                    .checked_mul(8)
                    .and_then(|n| n.checked_add(1 + 4)),
            }
            .ok_or_else(|| Error::Limit("checkpoint wire extent overflow".into()))?;
            bytes = bytes
                .checked_add(extra)
                .ok_or_else(|| Error::Limit("checkpoint wire extent overflow".into()))?;
        }
        if bytes > MAX_PAYLOAD_BYTES {
            return Err(Error::Limit("checkpoint wire extent exceeds bound".into()));
        }
        Ok(bytes)
    }

    fn committed_import_bytes(&self, position: usize, payload: usize) -> Result<u64> {
        // Retained array/Arc/row-vector metadata, not a second copy of KV.
        // Conservative cold allowance, not allocator RSS or a future-step cap.
        let rows = self.state_specs.iter().try_fold(0u64, |rows, spec| {
            let additional = match spec {
                StateSpec::Delta { .. } => 2,
                StateSpec::Full { .. } => (position as u64)
                    .checked_mul(2)
                    .ok_or_else(|| Error::Limit("checkpoint KV rows overflow".into()))?,
            };
            rows.checked_add(additional)
                .ok_or_else(|| Error::Limit("checkpoint row count overflow".into()))
        })?;
        rows.checked_mul(64)
            .and_then(|n| n.checked_add(payload as u64))
            .and_then(|n| n.checked_add(64 * 1024))
            .ok_or_else(|| Error::Limit("checkpoint retained storage overflow".into()))
    }
}

impl QwenDenseCpu {
    fn require_committed_cpu(&self) -> Result<()> {
        if self.kernel.is_some() {
            return Err(Error::Unsupported(
                "committed checkpoint requires native CPU without an external kernel".into(),
            ));
        }
        self.validate_state(&self.state)?;
        if self.state.attempt.is_some() || self.state.binding != self.authority.binding() {
            return Err(Error::State(
                "checkpoint requires a committed local root".into(),
            ));
        }
        Ok(())
    }

    /// Explicit local persistence of this session's committed numerical state.
    /// The source identity comes from the held model, not the supplied path.
    /// No extra state payload copy; an original host lease covers cold IO.
    pub fn save_committed_checkpoint(&self, path: &Path, host: &TransferAdmission) -> Result<()> {
        self.require_committed_cpu()?;
        let payload = checked_bytes(&self.model, self.state.position)?;
        let bytes = self
            .model
            .committed_wire_bytes(self.state.position, payload)?;
        let _scratch = reserve(host, AllocationClass::Scratch, io::IO_WORKSPACE_BYTES)?;
        let identity = self.model.committed_checkpoint_identity()?;
        io::write_checked(path, bytes, |writer| {
            writer.write_all(&MAGIC)?;
            put_u32(writer, VERSION)?;
            writer.write_all(&identity)?;
            write_state(writer, self.state.position, &self.state.layers)
        })
    }

    /// Restore from a trusted local file. All bytes, source/profile identity,
    /// extent, finite values and SHA256 integrity must pass before the original
    /// authority publishes. The receiving session gets its own root/epoch.
    /// No prompt history or sampler state is restored; never a prefix-cache
    /// credential. The original import charge remains with all shared readers.
    pub fn restore_committed_checkpoint(
        &mut self,
        path: &Path,
        host: &TransferAdmission,
    ) -> Result<()> {
        self.require_committed_cpu()?;
        self.authority.preflight_replacement()?;
        let max_row_bytes = self
            .model
            .state_specs
            .iter()
            .filter_map(|spec| match spec {
                StateSpec::Full { kv_width } => Some(*kv_width as u64 * 4),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        // Arc<[f32]> conversion can temporarily hold one decoded KV row twice.
        let scratch = io::IO_WORKSPACE_BYTES
            .checked_add(max_row_bytes)
            .ok_or_else(|| Error::Limit("checkpoint scratch extent overflow".into()))?;
        let _scratch = reserve(host, AllocationClass::Scratch, scratch)?;
        let identity = self.model.committed_checkpoint_identity()?;
        let candidate = io::read_checked(path, MAX_PAYLOAD_BYTES, |reader, file_bytes| {
            let mut magic = [0; 16];
            reader.read_exact(&mut magic)?;
            if magic != MAGIC || get_u32(reader)? != VERSION {
                return Err(Error::Unsupported(
                    "committed CPU checkpoint format/profile".into(),
                ));
            }
            let mut stored_identity = [0; 32];
            reader.read_exact(&mut stored_identity)?;
            if stored_identity != identity {
                return Err(Error::State(
                    "committed checkpoint source/geometry/numerical identity mismatch".into(),
                ));
            }
            let position = usize::try_from(get_u64(reader)?)
                .map_err(|_| Error::Limit("checkpoint frontier exceeds platform".into()))?;
            let payload = checked_bytes(&self.model, position)?;
            if file_bytes != self.model.committed_wire_bytes(position, payload)? {
                return Err(Error::Malformed(
                    "committed checkpoint file extent differs from source geometry".into(),
                ));
            }
            let lease = reserve(
                host,
                AllocationClass::ProviderEnvelope,
                self.model.committed_import_bytes(position, payload)?,
            )?;
            let layers = read_state(&self.model, reader, position, payload)?;
            Ok(QwenState {
                layers,
                position,
                binding: self.authority.binding(),
                attempt: None,
                imported: Some(Arc::new(lease)),
            })
        })?;
        let attempt = self
            .authority
            .begin_validated_replacement(candidate.position as u64)?;
        self.publish_state(candidate, attempt)
    }
}
