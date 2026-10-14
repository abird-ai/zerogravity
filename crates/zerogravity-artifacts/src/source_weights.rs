//! One immutable original source allocation and bounded cold decoding.
use crate::artifact::{Artifact, ArtifactFormat, TensorInfo};
use crate::quant::dequantize;
use crate::{Error, Result};
use std::{collections::HashMap, path::Path};
use zerogravity_core::accelerator::{EmbeddingRegion, SourceIdentity};

const MAX_CPU_SOURCE: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug)]
pub struct SourceWeights {
    source_identity: SourceIdentity,
    source: Vec<u8>,
    tensors: HashMap<String, TensorInfo>,
    architecture: String,
}

impl SourceWeights {
    pub fn open(path: &Path) -> Result<Self> {
        let parsed = Artifact::open(path)?;
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
        // Materialize through the SAME opened file descriptor and reparse
        // tensor metadata from those exact bytes. A replaced pathname (even
        // when both files have the same length) must never mix metadata from
        // one checkpoint with another checkpoint's source allocations.
        let (info, source) = parsed.into_bounded_source(MAX_CPU_SOURCE)?;
        let tensors = info
            .tensors
            .into_iter()
            .map(|t| (t.name.clone(), t))
            .collect();
        Ok(Self {
            source_identity: SourceIdentity::for_new_allocation()?,
            source,
            tensors,
            architecture: info.architecture.unwrap_or_default(),
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
        self.source.len()
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
        self.source
            .get(offset..end)
            .ok_or_else(|| Error::Malformed(format!("tensor '{}' extends beyond source", t.name)))
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
