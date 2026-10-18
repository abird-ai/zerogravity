//! One immutable original source allocation and bounded cold decoding.
use crate::artifact::{Artifact, ArtifactFormat, ArtifactInfo, TensorInfo};
use crate::quant::dequantize;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub use crate::verified_shards::{ExpectedSourceShard, SourceOpenLimits, VerifiedShardPlan};
use crate::{Error, Result};
use std::{collections::HashMap, path::Path};
use zerogravity_core::accelerator::{EmbeddingRegion, SourceIdentity};
use zerogravity_core::resource_arbiter::GlobalResourceLease;
#[cfg(all(feature = "cache", target_os = "linux"))]
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

const MAX_CPU_SOURCE: u64 = 4 * 1024 * 1024 * 1024;

/// One source identity, possibly backed by several independently verified
/// immutable allocations. GGUF keeps the original 4GiB policy.
enum SourceStorage {
    Gguf(Vec<u8>),
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Safetensors(Vec<Vec<u8>>),
}
impl SourceStorage {
    fn total_bytes(&self) -> usize {
        match self {
            Self::Gguf(source) => source.len(),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            Self::Safetensors(shards) => shards.iter().map(Vec::len).sum(),
        }
    }
    fn file(&self, shard: usize) -> Option<&[u8]> {
        match self {
            Self::Gguf(source) if shard == 0 => Some(source),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            Self::Safetensors(shards) => shards.get(shard).map(Vec::as_slice),
            _ => None,
        }
    }
}
pub struct SourceWeights {
    source_identity: SourceIdentity,
    source: SourceStorage,
    tensors: HashMap<String, TensorInfo>,
    /// Each tensor binds one exact immutable owned file allocation.
    shards: HashMap<String, usize>,
    architecture: String,
    #[cfg(all(feature = "cache", target_os = "linux"))]
    _memory: Option<GlobalResourceLease>,
}

impl std::fmt::Debug for SourceWeights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SourceWeights")
            .field("source_identity", &self.source_identity)
            .field("source_bytes", &self.source.total_bytes())
            .field("tensors", &self.tensors)
            .field("architecture", &self.architecture)
            .finish_non_exhaustive()
    }
}

impl SourceWeights {
    pub fn open(path: &Path) -> Result<Self> {
        let parsed = Artifact::open(path)?;
        Self::validate_artifact(&parsed)?;
        // Materialize through the SAME opened file descriptor and reparse
        // tensor metadata from those exact bytes.
        let (info, source) = parsed.into_bounded_source(MAX_CPU_SOURCE)?;
        Self::from_owned_source(info, source, None)
    }

    fn validate_artifact(parsed: &Artifact) -> Result<()> {
        if parsed.info.format != ArtifactFormat::Gguf {
            return Err(Error::Unsupported(
                "first CPU model runtime requires GGUF v3".into(),
            ));
        }
        if parsed.info.file_len > MAX_CPU_SOURCE {
            return Err(Error::Limit(
                "CPU source exceeds the 4 GiB initial safety budget".into(),
            ));
        }
        Ok(())
    }

    /// The cache digest authenticates the actual immutable owned allocation,
    /// before the same source issuer used by ordinary open can run. It grants
    /// neither trained-model eligibility nor a new runtime/source identity.
    /// The caller's existing resource domain retains exactly the raw source
    /// byte charge until this source drops; metadata/allocator/host-wide memory
    /// are outside that scoped reservation.
    #[cfg(all(feature = "cache", target_os = "linux"))]
    pub fn from_verified_artifact(
        parsed: Artifact,
        expected_bytes: u64,
        expected_sha256: &str,
        admission: &crate::cache::TransferAdmission,
    ) -> Result<Self> {
        Self::validate_artifact(&parsed)?;
        let memory = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::ImmutableWeights,
                bytes: expected_bytes,
            }],
        )?;
        let (info, source) =
            parsed.into_verified_bounded_source(MAX_CPU_SOURCE, expected_bytes, expected_sha256)?;
        Self::from_owned_source(info, source, Some(memory))
    }

    fn from_owned_source(
        info: ArtifactInfo,
        source: Vec<u8>,
        _memory: Option<GlobalResourceLease>,
    ) -> Result<Self> {
        let shards = info.tensors.iter().map(|t| (t.name.clone(), 0)).collect();
        let tensors = info
            .tensors
            .into_iter()
            .map(|t| (t.name.clone(), t))
            .collect();
        Ok(Self {
            source_identity: SourceIdentity::for_new_allocation()?,
            source: SourceStorage::Gguf(source),
            tensors,
            shards,
            architecture: info.architecture.unwrap_or_default(),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            _memory,
        })
    }

    /// The only materialization seam for indexed source shards.
    /// Verification and admission are owned by the original verified_shards
    /// module; this constructs no second issuer or model/runtime state.
    #[cfg(all(feature = "cache", target_os = "linux"))]
    pub(crate) fn from_verified_shard_storage(
        source: Vec<Vec<u8>>,
        tensors: HashMap<String, TensorInfo>,
        shards: HashMap<String, usize>,
        memory: GlobalResourceLease,
    ) -> Result<Self> {
        Ok(Self {
            source_identity: SourceIdentity::for_new_allocation()?,
            source: SourceStorage::Safetensors(source),
            tensors,
            shards,
            architecture: String::new(),
            _memory: Some(memory),
        })
    }

    /// Identity of this immutable loaded model allocation for safe GPU
    /// residency/cache lookup. Loading the same pathname again gets a new ID.
    pub fn source_identity(&self) -> SourceIdentity {
        self.source_identity
    }

    /// Actual immutable in-process GGUF source-byte allocation. Cheap to
    /// query, useful for asserting shared LoadedQwenModel lifetime/capacity.
    pub fn source_len(&self) -> usize {
        self.source.total_bytes()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn tensor(&self, key: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(key)
            .ok_or_else(|| Error::Malformed(format!("required source tensor '{key}' missing")))
    }

    /// Borrow original packed bytes from THIS immutable source allocation.
    /// Pair with tensor metadata/source identity; no expansion or copy occurs.
    pub fn tensor_bytes(&self, key: &str) -> Result<&[u8]> {
        self.bytes(self.tensor(key)?)
    }

    pub fn bytes<'a>(&'a self, t: &TensorInfo) -> Result<&'a [u8]> {
        let offset =
            usize::try_from(t.byte_offset).map_err(|_| Error::Limit("offset too large".into()))?;
        let len =
            usize::try_from(t.byte_len).map_err(|_| Error::Limit("tensor too large".into()))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::Limit("tensor byte range overflow".into()))?;
        let owned = self.tensor(&t.name)?;
        if owned.kind != t.kind
            || owned.shape != t.shape
            || owned.byte_offset != t.byte_offset
            || owned.byte_len != t.byte_len
        {
            return Err(Error::Malformed(
                "foreign tensor metadata for owned source".into(),
            ));
        }
        let shard = *self
            .shards
            .get(&t.name)
            .ok_or_else(|| Error::State("owned tensor missing shard binding".into()))?;
        self.source
            .file(shard)
            .and_then(|bytes| bytes.get(offset..end))
            .ok_or_else(|| Error::Malformed(format!("tensor '{}' extends beyond source", t.name)))
    }

    /// Source-only admission for original indexed safetensors. No model
    /// family gains trained numerical support by constructing a source.
    #[cfg(all(feature = "cache", target_os = "linux"))]
    pub fn prepare_verified_shards(
        index_json: &[u8],
        expected_index_sha256: &str,
        shards: Vec<ExpectedSourceShard>,
        limits: SourceOpenLimits,
        admission: &crate::cache::TransferAdmission,
    ) -> Result<VerifiedShardPlan> {
        VerifiedShardPlan::prepare(index_json, expected_index_sha256, shards, limits, admission)
    }

    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.shape.len() != 1 {
            return Err(Error::Malformed(format!("tensor {name} must be vector")));
        }
        let count =
            usize::try_from(t.shape[0]).map_err(|_| Error::Limit("vector too large".into()))?;
        if count > 1_000_000 {
            return Err(Error::Limit("reference vector too large".into()));
        }
        dequantize(t.kind, self.bytes(t)?, count)
    }

    pub fn embedding_region<'a>(
        &'a self,
        token: u32,
        name: &'a str,
    ) -> Result<EmbeddingRegion<'a>> {
        let t = self.tensor(name)?;
        if t.shape.len() != 2 {
            return Err(Error::Malformed("token embedding rank invalid".into()));
        }
        let r = EmbeddingRegion {
            source_identity: self.source_identity(),
            tensor_name: name,
            kind: t.kind,
            packed_weights: self.bytes(t)?,
            token,
            width: usize::try_from(t.shape[0])
                .map_err(|_| Error::Limit("embedding width".into()))?,
            vocabulary: usize::try_from(t.shape[1])
                .map_err(|_| Error::Limit("embedding vocabulary".into()))?,
        };
        r.validate()?;
        Ok(r)
    }

    /// Small dense convolution matrices only (not arbitrary model weights).
    pub fn matrix_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.shape.len() != 2 || t.shape.iter().product::<u64>() > 100_000 {
            return Err(Error::Limit(
                "matrix_f32 only accepts small 2D tensors".into(),
            ));
        }
        let count = (t.shape[0] * t.shape[1]) as usize;
        dequantize(t.kind, self.bytes(t)?, count)
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }
}

impl zerogravity_core::source::TensorSource for SourceWeights {
    fn source_identity(&self) -> SourceIdentity {
        self.source_identity()
    }
    fn tensor(&self, name: &str) -> Result<zerogravity_core::source::SourceTensorRef<'_>> {
        let info = self.tensor(name)?;
        let tensor = zerogravity_core::source::SourceTensorRef {
            kind: info.kind,
            shape: &info.shape,
            packed_bytes: self.tensor_bytes(name)?,
        };
        tensor.validate()?;
        Ok(tensor)
    }
}
