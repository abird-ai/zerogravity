//! Authenticated complete source parameter -> immutable resident bytes.
//!
//! The caller supplies the exact SourceParameter descriptor and an explicit
//! storage policy chosen by numerical lowering. Tensor names never choose a
//! codec, conversion, model operation, or execution route.

use super::{ResidentInitialization, ResidentLease, ResidentService};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{ParameterWindow, VerifiedTensorSource};
use zerogravity_core::{
    Error, Result,
    accelerator::{SourceIdentity, program::SourceParameter},
    device_events::BufferRole,
    encoding::StoredTensorEncoding,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ForwardParameterStorage {
    /// Preserve the exact authenticated source bytes and codec.
    Original,
    /// Source is either BF16 already or finite F32 converted with IEEE BF16 RNE.
    Bf16Rne,
    /// Exact widening of BF16, or the original finite F32 values.
    F32Exact,
}

impl ForwardParameterStorage {
    pub(crate) fn resident_bytes(self, descriptor: &SourceParameter) -> Result<usize> {
        use zerogravity_core::encoding::ScalarDType::{Bf16, F32};
        if self == Self::Original {
            return Ok(descriptor.bytes);
        }
        if !matches!(descriptor.kind, StoredTensorEncoding::Scalar(Bf16 | F32)) {
            return Err(Error::Unsupported(
                "forward parameter conversion codec".into(),
            ));
        }
        descriptor.shape.iter().try_fold(
            if self == Self::Bf16Rne {
                2usize
            } else {
                4usize
            },
            |bytes, &n| {
                usize::try_from(n)
                    .ok()
                    .and_then(|n| bytes.checked_mul(n))
                    .ok_or_else(|| Error::Limit("forward parameter converted extent".into()))
            },
        )
    }
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(2) {
        return Err(Error::Malformed("forward BF16 parameter extent".into()));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(
            bytes
                .len()
                .checked_mul(2)
                .ok_or_else(|| Error::Limit("forward parameter staging extent".into()))?,
        )
        .map_err(|_| Error::Limit("forward parameter staging allocation".into()))?;
    for word in bytes.chunks_exact(2) {
        let bits = u32::from(u16::from_le_bytes([word[0], word[1]])) << 16;
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            return Err(Error::Malformed(
                "forward parameter contains nonfinite BF16".into(),
            ));
        }
        output.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(output)
}

struct ParameterOwner {
    source: VerifiedTensorSource,
    descriptor: SourceParameter,
    window: Arc<ParameterWindow>,
    _conversion: zerogravity_artifacts::source_weights::WindowWorkspace,
}

pub(crate) struct ForwardParameter {
    owner: Arc<ParameterOwner>,
    lease: ResidentLease,
    storage: ForwardParameterStorage,
    kind: StoredTensorEncoding,
    elements: usize,
}

impl ForwardParameter {
    pub(crate) fn source_identity(&self) -> SourceIdentity {
        self.owner.source.source_identity()
    }
    pub(crate) fn descriptor(&self) -> &SourceParameter {
        &self.owner.descriptor
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn storage(&self) -> ForwardParameterStorage {
        self.storage
    }
    pub(crate) fn kind(&self) -> StoredTensorEncoding {
        self.kind
    }
    pub(crate) fn elements(&self) -> usize {
        self.elements
    }
    pub(crate) fn require_descriptor(&self, descriptor: &SourceParameter) -> Result<()> {
        if descriptor != &self.owner.descriptor
            || descriptor.source != self.source_identity()
            || self.owner.window.parameter().source_parameter() != *descriptor
        {
            return Err(Error::State(
                "CUDA forward parameter authenticated descriptor changed".into(),
            ));
        }
        Ok(())
    }
}

fn f32_bytes_to_bf16_rne(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(Error::Malformed(
            "CUDA forward F32 parameter byte extent".into(),
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes.len() / 2)
        .map_err(|_| Error::Limit("CUDA forward BF16 parameter staging allocation".into()))?;
    for word in bytes.chunks_exact(4) {
        let bits = u32::from_le_bytes(word.try_into().expect("four-byte chunk"));
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            return Err(Error::Malformed(
                "CUDA forward parameter source contains nonfinite F32".into(),
            ));
        }
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        let bf16 = (rounded >> 16) as u16;
        if u32::from(bf16) & 0x7f80 == 0x7f80 {
            return Err(Error::Malformed(
                "CUDA forward finite F32 parameter overflows BF16".into(),
            ));
        }
        output.extend_from_slice(&bf16.to_le_bytes());
    }
    Ok(output)
}

impl ResidentInitialization<'_> {
    pub(crate) fn upload_forward_parameter(
        &mut self,
        service: &Arc<ResidentService>,
        source: VerifiedTensorSource,
        descriptor: &SourceParameter,
        storage: ForwardParameterStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ForwardParameter>> {
        self.require_service(service)?;
        if descriptor.source != source.source_identity()
            || service.source_identity() != source.source_identity()
        {
            return Err(Error::State(
                "CUDA forward parameter source/service differs".into(),
            ));
        }
        let window = Arc::new(source.read_declared_parameter(descriptor, cancelled)?);
        if window.parameter().source_identity() != descriptor.source
            || window.parameter().source_parameter() != *descriptor
            || !window.is_complete()
        {
            return Err(Error::State(
                "CUDA forward parameter complete source binding differs".into(),
            ));
        }

        let elements = window.parameter().elements();
        let conversion = source.reserve_workspace(
            storage
                .resident_bytes(descriptor)?
                .checked_mul(2)
                .ok_or_else(|| Error::Limit("forward parameter staging charge".into()))?,
        )?;
        let (payload, kind) = match storage {
            ForwardParameterStorage::F32Exact => match descriptor.kind {
                StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16) => (
                    bf16_bytes_to_f32(window.as_slice())?,
                    StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32),
                ),
                StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32) => {
                    if window.as_slice().chunks_exact(4).any(|word| {
                        u32::from_le_bytes(word.try_into().expect("F32 parameter word"))
                            & 0x7f80_0000
                            == 0x7f80_0000
                    }) {
                        return Err(Error::Malformed(
                            "forward parameter contains nonfinite F32".into(),
                        ));
                    }
                    (window.as_slice().to_vec(), descriptor.kind)
                }
                _ => return Err(Error::Unsupported("forward F32 parameter codec".into())),
            },
            ForwardParameterStorage::Original => (window.as_slice().to_vec(), descriptor.kind),
            ForwardParameterStorage::Bf16Rne => match descriptor.kind {
                StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16) => (
                    window.as_slice().to_vec(),
                    StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
                ),
                StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32) => (
                    f32_bytes_to_bf16_rne(window.as_slice())?,
                    StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
                ),
                _ => {
                    return Err(Error::Unsupported(
                        "CUDA forward BF16 parameter source codec".into(),
                    ));
                }
            },
        };
        let element_bytes = usize::try_from(kind.byte_len(1)?)
            .map_err(|_| Error::Limit("CUDA forward parameter element bytes".into()))?;
        if payload.len()
            != elements.checked_mul(element_bytes).ok_or_else(|| {
                Error::Limit("CUDA forward parameter resident extent overflow".into())
            })?
        {
            return Err(Error::State(
                "CUDA forward parameter converted byte extent differs".into(),
            ));
        }

        let owner = Arc::new(ParameterOwner {
            source,
            descriptor: descriptor.clone(),
            window,
            _conversion: conversion,
        });
        self.retain_completion_owner(owner.clone())?;
        let lease = self.upload_bytes(&payload, BufferRole::ImmutableWeights)?;
        let parameter = Arc::new(ForwardParameter {
            owner,
            lease,
            storage,
            kind,
            elements,
        });
        self.retain_completion_owner(parameter.clone())?;
        Ok(parameter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(bits: &[u32]) -> Result<Vec<u16>> {
        let raw = bits
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        Ok(f32_bytes_to_bf16_rne(&raw)?
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes([word[0], word[1]]))
            .collect())
    }

    #[test]
    fn bf16_widening_preserves_every_finite_word_exactly() -> Result<()> {
        let words = (0..=u16::MAX)
            .filter(|word| word & 0x7f80 != 0x7f80)
            .collect::<Vec<_>>();
        let source = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let output = bf16_bytes_to_f32(&source)?;
        for (word, bytes) in words.into_iter().zip(output.chunks_exact(4)) {
            assert_eq!(
                u32::from_le_bytes(bytes.try_into().unwrap()),
                u32::from(word) << 16
            );
        }
        for word in [0x7f80u16, 0xff80, 0x7fc0] {
            assert!(bf16_bytes_to_f32(&word.to_le_bytes()).is_err());
        }
        assert!(bf16_bytes_to_f32(&[0]).is_err());
        assert!(bf16_bytes_to_f32(&[]).is_err());
        Ok(())
    }

    #[test]
    fn finite_f32_to_bf16_is_rne_preserves_sign_and_refuses_overflow() {
        assert_eq!(
            convert(&[
                0x0000_0000,
                0x8000_0000,
                0x3f80_0000,
                0xbf80_0000,
                0x0001_0000,
                0x8001_0000,
                0x3f80_7fff,
                0x3f80_8000,
                0x3f81_8000,
            ])
            .unwrap(),
            [
                0x0000, 0x8000, 0x3f80, 0xbf80, 0x0001, 0x8001, 0x3f80, 0x3f80, 0x3f82
            ]
        );
        assert!(convert(&[0x7f7f_ffff]).is_err());
        assert!(convert(&[0x7f80_0000]).is_err());
        assert!(convert(&[0x7fc0_0000]).is_err());
        assert!(f32_bytes_to_bf16_rne(&[0, 0]).is_err());
    }
}
