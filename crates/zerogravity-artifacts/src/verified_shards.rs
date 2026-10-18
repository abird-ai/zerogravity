//! Cold, verified, immutable original-index Safetensors shards.
//!
//! Source-only. No Qwen/Gemma learned tensor binding, tokenization, trained
//! execution, GPU kernels or SSD placement follows from this transaction.
//! Explicitly opened descriptors, expected whole-file SHA256 and an index
//! whose bytes are separately caller-authenticated are required.
//! All bytes remain permanently owned until the last SourceWeights Arc dies.
//! No memory maps of mutable filesystem inodes, second source issuer,
//! alternate cache, session owner or unverifiable "lazy tensor" borrows.

use crate::artifact::{Artifact, ArtifactFormat, TensorInfo};
use crate::cache::TransferAdmission;
use crate::sharded::SafetensorsIndex;
use crate::source_weights::SourceWeights;
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};

/// Explicit loader allocation policy. Defaults retain legacy 4GiB safety;
/// higher limits never bypass the original physical host-RAM resource ledger.
#[derive(Clone, Copy, Debug)]
pub struct SourceOpenLimits {
    pub max_owned_bytes: u64,
    pub max_metadata_bytes: u64,
    pub max_shards: usize,
    pub max_tensors: usize,
}
impl Default for SourceOpenLimits {
    fn default() -> Self {
        Self {
            max_owned_bytes: 4 * 1024 * 1024 * 1024,
            max_metadata_bytes: 128 * 1024 * 1024,
            max_shards: 64,
            max_tensors: 200_000,
        }
    }
}
/// One already-opened descriptor, bound to an exact index basename,
/// expected whole-file length and trusted SHA256. The original filename
/// is not reopened after this constructor.
pub struct ExpectedSourceShard {
    basename: String,
    opened: Artifact,
    expected_bytes: u64,
    expected_sha256: String,
}
impl ExpectedSourceShard {
    pub fn new(
        basename: String,
        opened: Artifact,
        expected_bytes: u64,
        expected_sha256: String,
    ) -> Result<Self> {
        crate::cache::paths::validate_digest(&expected_sha256)?;
        if opened.info.format != ArtifactFormat::Safetensors {
            return Err(Error::Unsupported(
                "indexed source requires original safetensors shards".into(),
            ));
        }
        if opened.info.file_len != expected_bytes {
            return Err(Error::Malformed("opened shard length mismatch".into()));
        }
        Ok(Self {
            basename,
            opened,
            expected_bytes,
            expected_sha256,
        })
    }
}
/// Fully admitted cold load plan. No published SourceIdentity until all
/// copied owned shard contents and their exact index/header binding pass.
#[must_use = "the source plan must be materialized or dropped"]
pub struct VerifiedShardPlan {
    index: SafetensorsIndex,
    index_sha256: String,
    shards: Vec<ExpectedSourceShard>,
    max_owned_bytes: u64,
    _metadata_charge: u64,
    memory: GlobalResourceLease,
}
impl VerifiedShardPlan {
    pub(crate) fn prepare(
        index_json: &[u8],
        expected_index_sha256: &str,
        shards: Vec<ExpectedSourceShard>,
        limits: SourceOpenLimits,
        admission: &TransferAdmission,
    ) -> Result<Self> {
        crate::cache::paths::validate_digest(expected_index_sha256)?;
        if format!("{:x}", Sha256::digest(index_json)) != expected_index_sha256 {
            return Err(Error::Malformed(
                "original safetensors index SHA256 mismatch".into(),
            ));
        }
        let index = SafetensorsIndex::from_json(index_json)?;
        if shards.is_empty()
            || shards.len() > limits.max_shards
            || index.tensors().len() > limits.max_tensors
            || limits.max_owned_bytes == 0
            || limits.max_metadata_bytes == 0
            || limits.max_shards == 0
            || limits.max_tensors == 0
        {
            return Err(Error::Limit(
                "original source shard/index/load limits".into(),
            ));
        }
        let expected_set = index.shards();
        let mut shard_names = BTreeSet::new();
        let mut indexed_tensors = BTreeMap::<&str, &str>::new();
        let mut owned_file_bytes = 0u64;
        let mut payload_bytes = 0u64;
        // Reserve the original index and headers, plus a conservative
        // finite bookkeeping/JSON-parsing envelope in the same host budget.
        let mut metadata_charge = (index_json.len() as u64)
            .checked_add(8 * 1024 * 1024)
            .ok_or_else(|| Error::Limit("index metadata overflow".into()))?;
        for shard in &shards {
            if !expected_set.contains(shard.basename.as_str())
                || !shard_names.insert(shard.basename.as_str())
                || shard.opened.info.format != ArtifactFormat::Safetensors
                || shard.opened.info.file_len != shard.expected_bytes
            {
                return Err(Error::Malformed(
                    "extra, duplicate or wrong source shard".into(),
                ));
            }
            crate::cache::paths::validate_digest(&shard.expected_sha256)?;
            owned_file_bytes = owned_file_bytes
                .checked_add(shard.expected_bytes)
                .ok_or_else(|| Error::Limit("whole-shard byte sum overflow".into()))?;
            if shard.opened.info.tensors.is_empty() {
                return Err(Error::Malformed(
                    "empty original safetensors shard header".into(),
                ));
            }
            let mut first_payload_offset = u64::MAX;
            for tensor in &shard.opened.info.tensors {
                if index.tensors().get(&tensor.name).map(String::as_str)
                    != Some(shard.basename.as_str())
                    || indexed_tensors
                        .insert(&tensor.name, &shard.basename)
                        .is_some()
                {
                    return Err(Error::Malformed(
                        "original index tensor missing, extra or assigned wrong shard".into(),
                    ));
                }
                first_payload_offset = first_payload_offset.min(tensor.byte_offset);
                payload_bytes = payload_bytes
                    .checked_add(tensor.byte_len)
                    .ok_or_else(|| Error::Limit("tensor payload byte overflow".into()))?;
                metadata_charge = metadata_charge
                    .checked_add(512 + tensor.name.len() as u64 + 8 * tensor.shape.len() as u64)
                    .ok_or_else(|| Error::Limit("metadata allocation envelope overflow".into()))?;
            }
            metadata_charge = metadata_charge
                .checked_add(first_payload_offset)
                .ok_or_else(|| Error::Limit("source headers allocation overflow".into()))?;
        }
        if shard_names != expected_set
            || indexed_tensors.len() != index.tensors().len()
            || payload_bytes != index.declared_total_size()
        {
            return Err(Error::Malformed(
                "incomplete original shard/index tensor closure".into(),
            ));
        }
        if owned_file_bytes > limits.max_owned_bytes
            || owned_file_bytes > usize::MAX as u64
            || metadata_charge > limits.max_metadata_bytes
        {
            return Err(Error::Limit(
                "source exceeds explicit owned-RAM/metadata budget".into(),
            ));
        }
        // Sole process-shared physical resource authority, caller-registered.
        // A descriptor header alone does not claim hardware capacity.
        let memory = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::ImmutableWeights,
                    bytes: owned_file_bytes,
                },
                AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: metadata_charge,
                },
            ],
        )?;
        Ok(Self {
            index,
            index_sha256: expected_index_sha256.to_owned(),
            shards,
            max_owned_bytes: limits.max_owned_bytes,
            _metadata_charge: metadata_charge,
            memory,
        })
    }

    pub fn declared_payload_bytes(&self) -> u64 {
        self.index.declared_total_size()
    }
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Reuse Artifact's existing same-descriptor exact-length, expected
    /// digest, in-memory reparse and checked source geometry. A concurrent
    /// write may reject but never change the accepted owned allocation.
    /// No trained model names/layouts are inferred from generic headers.
    pub fn materialize(self) -> Result<SourceWeights> {
        // A reference must identify the *weights*, not merely the index.
        // Canonical versioned domain-separated manifest, independent of
        // caller shard order. SHA256 index + sorted basename/len/whole hash.
        let mut digest = Sha256::new();
        digest.update(b"zerogravity.original.source-bundle.v1\0");
        digest.update(self.index_sha256.as_bytes());
        let mut entries = self.shards.iter().collect::<Vec<_>>();
        entries.sort_by(|a, b| a.basename.cmp(&b.basename));
        for shard in entries {
            let name = shard.basename.as_bytes();
            let len = u32::try_from(name.len())
                .map_err(|_| Error::Limit("source basename too long for manifest".into()))?;
            digest.update(len.to_le_bytes());
            digest.update(name);
            digest.update(shard.expected_bytes.to_le_bytes());
            digest.update(shard.expected_sha256.as_bytes());
        }
        let bundle_sha256 = format!("{:x}", digest.finalize());
        let mut allocations = Vec::new();
        allocations
            .try_reserve_exact(self.shards.len())
            .map_err(|_| Error::Limit("owned shard vector allocation refused".into()))?;
        let mut tensors: HashMap<String, TensorInfo> = HashMap::new();
        let mut bindings = HashMap::new();
        let mut owned_payload = 0u64;
        for (shard_index, shard) in self.shards.into_iter().enumerate() {
            // Refuse same-length source mutation even when an expected
            // digest happens to match the changed bytes: original header
            // geometry was used to calculate admission and tensor binding.
            let inspected = shard.opened.info.tensors.clone();
            let (info, bytes) = shard.opened.into_verified_bounded_source(
                self.max_owned_bytes,
                shard.expected_bytes,
                &shard.expected_sha256,
            )?;
            if info.format != ArtifactFormat::Safetensors
                || info.file_len != shard.expected_bytes
                || info.tensors.len() != inspected.len()
                || !info.tensors.iter().zip(&inspected).all(|(now, before)| {
                    now.name == before.name
                        && now.kind == before.kind
                        && now.shape == before.shape
                        && now.byte_offset == before.byte_offset
                        && now.byte_len == before.byte_len
                })
            {
                return Err(Error::Malformed(
                    "owned shard header changed after original preflight".into(),
                ));
            }
            for tensor in info.tensors {
                owned_payload = owned_payload
                    .checked_add(tensor.byte_len)
                    .ok_or_else(|| Error::Limit("owned payload byte sum overflow".into()))?;
                if self.index.tensors().get(&tensor.name).map(String::as_str)
                    != Some(shard.basename.as_str())
                    || tensors.contains_key(&tensor.name)
                {
                    return Err(Error::Malformed(
                        "owned source tensor/header/index disagreement".into(),
                    ));
                }
                bindings.insert(tensor.name.clone(), shard_index);
                tensors.insert(tensor.name.clone(), tensor);
            }
            allocations.push(bytes);
        }
        if tensors.len() != self.index.tensors().len()
            || owned_payload != self.index.declared_total_size()
        {
            return Err(Error::Malformed(
                "owned source tensor names/payload do not close original index".into(),
            ));
        }
        // Sole existing source issuer. Preserve logical source lifetime,
        // zero-copy borrowed tensors and one ResourceArbiter lease until drop.
        SourceWeights::from_verified_shard_storage(
            allocations,
            tensors,
            bindings,
            self.index_sha256,
            bundle_sha256,
            self.memory,
        )
    }
}
