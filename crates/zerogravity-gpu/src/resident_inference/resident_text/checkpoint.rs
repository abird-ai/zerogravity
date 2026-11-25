//! Explicit cold committed-state persistence. Native contiguous F32 state
//! only; no prompt history, prefix authorization or layout conversion.
use super::*;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::Path,
};
use zerogravity_artifacts::checkpoint as file;

const MAGIC: [u8; 16] = *b"ZGRESTEXT_STATE1";
const HEADER_BYTES: u64 = 16 + 32 + 8;
const MAX_STATE_BYTES: usize = 1024 * 1024 * 1024;

impl ResidentTextModel {
    fn checkpoint_identity(&self) -> Result<[u8; 32]> {
        let identity = self.state_identity()?;
        // Keep eligibility distinct from the generic plane/file transport.
        if identity.selected()
            != (SelectedMode {
                provider: ProviderKind::Hip,
                mode: ExecutionMode::Native,
            })
            || identity.numerical_profile() != NumericalProfile::HipRocBlasBf16MixedV2
            || identity.state_policy() != ResidentStatePolicy::Contiguous
        {
            return Err(Error::Unsupported(
                "resident checkpoint requires native HIP V2 contiguous state".into(),
            ));
        }
        self.inner.service.ensure_usable()?;
        let plan = self.prepared.model().program();
        let revision = plan.checkpoint_revision().ok_or_else(|| {
            Error::Unsupported(
                "resident checkpoint requires declared source semantic revision".into(),
            )
        })?;
        let source = self.prepared.model().source();
        let geometry = identity.geometry();
        let mut hash = Sha256::new();
        // Independent of local allocation, root/session and service identity.
        // This version fixes the exact provider/numerical/state coding.
        hash.update(b"resident-text/hip-native-v2/f32-contiguous/v1");
        hash.update(revision.0);
        hash.update(source.verified_bundle_sha256().as_bytes());
        let properties = self.inner.service.client().properties();
        let fingerprint = &properties.identity.fingerprint;
        hash.update((fingerprint.len() as u64).to_le_bytes());
        hash.update(fingerprint.as_bytes());
        for extent in [
            geometry.vocabulary(),
            geometry.context_capacity(),
            geometry.max_rows(),
            self.inner.states.len(),
        ] {
            hash.update((extent as u64).to_le_bytes());
        }
        let mut total = 0usize;
        for requirement in &self.inner.states {
            let (planes, physical) = requirement.plane_layout()?;
            if physical.kind() != ResidentPhysicalType::F32 {
                return Err(Error::Unsupported(
                    "resident checkpoint requires exact F32 storage".into(),
                ));
            }
            let (tag, index) = match requirement.slot {
                StateSlot::Conv(domain) => (1u8, domain.0),
                StateSlot::Recurrent(domain) => (2, domain.0),
                StateSlot::Kv(domain) => (3, domain.0),
            };
            hash.update([tag]);
            hash.update((index as u64).to_le_bytes());
            match requirement.layout {
                StateLayout::Vector { elements } => {
                    hash.update([1]);
                    hash.update((elements as u64).to_le_bytes());
                }
                StateLayout::Kv { slab, capacity } => {
                    hash.update([2]);
                    hash.update((slab as u64).to_le_bytes());
                    hash.update((capacity as u64).to_le_bytes());
                }
            }
            hash.update((planes as u64).to_le_bytes());
            total = total
                .checked_add(requirement.bytes)
                .ok_or_else(|| Error::Limit("resident checkpoint state extent overflow".into()))?;
        }
        if total != self.inner.state_bytes || total == 0 || total > MAX_STATE_BYTES {
            return Err(Error::Limit("resident checkpoint physical extent".into()));
        }
        Ok(hash.finalize().into())
    }

    fn checkpoint_extent(&self, position: usize) -> Result<u64> {
        if position > self.inner.capacity {
            return Err(Error::Limit(
                "resident checkpoint exceeds prepared context".into(),
            ));
        }
        self.inner
            .states
            .iter()
            .try_fold(HEADER_BYTES, |bytes, requirement| {
                let (planes, _, active) = plane_extent(requirement, position)?;
                let add = active
                    .checked_mul(planes)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or_else(|| {
                        Error::Limit("resident checkpoint wire extent overflow".into())
                    })?;
                bytes
                    .checked_add(add as u64)
                    .filter(|n| *n <= HEADER_BYTES + MAX_STATE_BYTES as u64)
                    .ok_or_else(|| Error::Limit("resident checkpoint wire extent".into()))
            })
    }
}

/// The existing state requirement is the only shape/layout authority. A
/// complete KV domain stores active K then active V, omitting zero tail.
fn plane_extent(requirement: &StateRequirement, position: usize) -> Result<(usize, usize, usize)> {
    let (planes, layout) = requirement.plane_layout()?;
    if layout.kind() != ResidentPhysicalType::F32 {
        return Err(Error::Unsupported(
            "resident checkpoint requires F32 planes".into(),
        ));
    }
    let active = match requirement.layout {
        StateLayout::Vector { elements } => elements,
        StateLayout::Kv { slab, capacity } if position <= capacity => slab
            .checked_mul(position)
            .ok_or_else(|| Error::Limit("resident checkpoint active KV extent".into()))?,
        _ => return Err(Error::Limit("resident checkpoint KV frontier".into())),
    };
    if active > layout.elements() {
        return Err(Error::State(
            "resident checkpoint active plane differs".into(),
        ));
    }
    Ok((planes, layout.bytes()?, active))
}

fn write_words(writer: &mut dyn Write, values: &[f32]) -> Result<()> {
    let mut bytes = [0u8; 4096];
    for chunk in values.chunks(bytes.len() / 4) {
        for (value, word) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            if !value.is_finite() {
                return Err(Error::State("nonfinite resident checkpoint value".into()));
            }
            word.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 4])?;
    }
    Ok(())
}

/// Decode one native plane; no extra flattened f32 allocation or conversion.
/// Keep all validated bytes private until the file integrity check completes.
fn read_plane(reader: &mut dyn Read, full_bytes: usize, active: usize) -> Result<Vec<u8>> {
    let active_bytes = active
        .checked_mul(4)
        .filter(|n| *n <= full_bytes)
        .ok_or_else(|| Error::Limit("resident checkpoint active bytes".into()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(full_bytes)
        .map_err(|_| Error::Limit("resident checkpoint host plane allocation".into()))?;
    bytes.resize(full_bytes, 0);
    reader.read_exact(&mut bytes[..active_bytes])?;
    for word in bytes[..active_bytes].chunks_exact_mut(4) {
        let bits = u32::from_le_bytes(word.try_into().unwrap());
        if !f32::from_bits(bits).is_finite() {
            return Err(Error::Malformed(
                "nonfinite resident checkpoint state".into(),
            ));
        }
        word.copy_from_slice(&bits.to_ne_bytes());
    }
    Ok(bytes)
}

impl ResidentTextSnapshot {
    /// Persist this actually committed, retained root. Uses the original
    /// event-backed readback, not a proposed frontier or a raw GPU handle.
    pub async fn save_committed_checkpoint(&self, path: &Path) -> Result<()> {
        let model = &self.0.scope.model;
        let identity = model.checkpoint_identity()?;
        let position = usize::try_from(self.binding().frontier())
            .map_err(|_| Error::Limit("resident checkpoint frontier".into()))?;
        let extent = model.checkpoint_extent(position)?;
        model.inner.validate_domains(&self.0.inner.domains, true)?;
        let _io = model.reader_scope(file::IO_WORKSPACE_BYTES as usize)?;
        let values = self.read_domains().await?;
        if values.len() != model.inner.states.len() {
            return Err(Error::State("resident checkpoint domain coverage".into()));
        }
        file::write_checked(path, extent, |writer| {
            writer.write_all(&MAGIC)?;
            writer.write_all(&identity)?;
            writer.write_all(&(position as u64).to_le_bytes())?;
            for ((slot, data), requirement) in values.iter().zip(&model.inner.states) {
                let (planes, _, active) = plane_extent(requirement, position)?;
                if *slot != requirement.slot || active.checked_mul(planes) != Some(data.len()) {
                    return Err(Error::State(
                        "resident checkpoint domain extent/order".into(),
                    ));
                }
                write_words(writer, data)?;
            }
            Ok(())
        })
    }
}

impl ResidentTextSession {
    pub async fn save_committed_checkpoint(&self, path: &Path) -> Result<()> {
        self.snapshot()?.save_committed_checkpoint(path).await
    }

    /// Restore a complete compatible committed root, with a private candidate
    /// from the existing root pool and the original completion/authority.
    /// File IO is synchronous, bounded and explicit cold work. The file is
    /// trusted local data, integrity-checked, not an authentication credential.
    pub async fn restore_committed_checkpoint(&mut self, path: &Path) -> Result<()> {
        let model = self.model.clone();
        let identity = model.checkpoint_identity()?;
        self.inner.authority.preflight_replacement()?;
        // Full packed native planes and possible transfer backing; the source
        // workspace is retained even on uncertain completion.
        let host_bytes = model
            .inner
            .state_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(file::IO_WORKSPACE_BYTES as usize + CONTROL_BYTES))
            .ok_or_else(|| Error::Limit("resident checkpoint host peak".into()))?;
        let scope = model.reader_scope(host_bytes)?;
        let (position, packed) = file::read_checked(
            path,
            HEADER_BYTES + MAX_STATE_BYTES as u64,
            |reader, extent| {
                let mut magic = [0; 16];
                reader.read_exact(&mut magic)?;
                if magic != MAGIC {
                    return Err(Error::Unsupported("resident checkpoint format".into()));
                }
                let mut stored = [0; 32];
                reader.read_exact(&mut stored)?;
                if stored != identity {
                    return Err(Error::State(
                        "resident checkpoint source/profile/layout identity differs".into(),
                    ));
                }
                let mut position = [0; 8];
                reader.read_exact(&mut position)?;
                let position = usize::try_from(u64::from_le_bytes(position))
                    .map_err(|_| Error::Limit("resident checkpoint frontier".into()))?;
                if extent != model.checkpoint_extent(position)? {
                    return Err(Error::Malformed(
                        "resident checkpoint file extent differs".into(),
                    ));
                }
                let mut packed = Vec::new();
                for requirement in &model.inner.states {
                    let (planes, full_bytes, active) = plane_extent(requirement, position)?;
                    for _ in 0..planes {
                        packed.push(read_plane(reader, full_bytes, active)?);
                    }
                }
                Ok((position, packed))
            },
        )?;
        let attempt = self
            .inner
            .authority
            .begin_validated_replacement(position as u64)?;
        let _admission = model.inner.allocation.lock().await;
        // Checkout the originally admitted candidate backing, before queue or
        // submission. No independent allocator or extra unpooled root.
        let candidate = model.inner.allocate_uninitialized_state()?;
        let leases = candidate
            .iter()
            .flat_map(|domain| domain.leases.iter())
            .collect::<Vec<_>>();
        if leases.len() != packed.len()
            || leases
                .iter()
                .zip(&packed)
                .any(|(lease, data)| lease.bytes() != data.len())
        {
            return Err(Error::State(
                "resident checkpoint candidate plane coverage differs".into(),
            ));
        }
        let ticket = model.inner.service.queue().submit(
            model.inner.service.budget(),
            ResourceOwner::new(
                &model.inner.service.owner().model,
                &format!("checkpoint-{}", self.binding().session().get()),
            )?,
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: model.inner.state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut initialization = model
            .inner
            .service
            .begin_initialization(attempt.base().root())?;
        initialization.retain_completion_owner(scope)?;
        #[cfg(feature = "hardware-abort-probe")]
        let launched = leases.len();
        for (lease, data) in leases.into_iter().zip(packed) {
            initialization.upload_owned_into(lease, data)?;
        }
        #[cfg(feature = "hardware-abort-probe")]
        if model
            .inner
            .abort_after_launch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok_and(|remaining| remaining == 1)
        {
            // The existing fault probe pauses after real writes have been
            // submitted and the original guard owns their completion backing.
            model.inner.launch_seen.store(launched, Ordering::Release);
            std::future::pending::<()>().await;
        }
        let ready = initialization.complete_sync().await?;
        slot.finish()?;
        self.inner
            .publish_initialized_replacement(candidate, attempt, ready)
    }
}
