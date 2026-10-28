#![allow(dead_code)] // Private unselected cold owner until ordinary body consumes it.
//! Authenticated complete source parameter -> immutable resident bytes.
//!
//! The caller supplies the exact SourceParameter descriptor and an explicit
//! storage policy chosen by numerical lowering. Tensor names never choose a
//! codec, conversion, model operation, or execution route.

use super::{ResidentInitialization, ResidentLease, ResidentService};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::{
    quant::QuantKind,
    source_weights::{ParameterWindow, VerifiedTensorSource},
};
use zerogravity_core::{
    Error, Result,
    accelerator::{SourceIdentity, program::SourceParameter},
    device_events::BufferRole,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrdinaryParameterStorage {
    /// Preserve the exact authenticated source bytes and codec.
    Original,
    /// Source is either BF16 already or finite F32 converted with IEEE BF16 RNE.
    Bf16Rne,
}

#[derive(Clone)]
struct ParameterOwner {
    source: VerifiedTensorSource,
    descriptor: SourceParameter,
    window: Arc<ParameterWindow>,
}

pub(crate) struct OrdinaryParameter {
    owner: Arc<ParameterOwner>,
    lease: ResidentLease,
    storage: OrdinaryParameterStorage,
    kind: QuantKind,
    elements: usize,
}

impl OrdinaryParameter {
    pub(crate) fn source_identity(&self) -> SourceIdentity {
        self.owner.source.source_identity()
    }
    pub(crate) fn descriptor(&self) -> &SourceParameter {
        &self.owner.descriptor
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn storage(&self) -> OrdinaryParameterStorage {
        self.storage
    }
    pub(crate) fn kind(&self) -> QuantKind {
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
                "ordinary CUDA parameter authenticated descriptor changed".into(),
            ));
        }
        Ok(())
    }
}

fn f32_bytes_to_bf16_rne(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(Error::Malformed(
            "ordinary CUDA F32 parameter byte extent".into(),
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes.len() / 2)
        .map_err(|_| Error::Limit("ordinary CUDA BF16 parameter staging allocation".into()))?;
    for word in bytes.chunks_exact(4) {
        let bits = u32::from_le_bytes(word.try_into().expect("four-byte chunk"));
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            return Err(Error::Malformed(
                "ordinary CUDA parameter source contains nonfinite F32".into(),
            ));
        }
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        let bf16 = (rounded >> 16) as u16;
        if u32::from(bf16) & 0x7f80 == 0x7f80 {
            return Err(Error::Malformed(
                "ordinary CUDA finite F32 parameter overflows BF16".into(),
            ));
        }
        output.extend_from_slice(&bf16.to_le_bytes());
    }
    Ok(output)
}

impl ResidentInitialization<'_> {
    pub(crate) fn upload_ordinary_parameter(
        &mut self,
        service: &Arc<ResidentService>,
        source: VerifiedTensorSource,
        descriptor: &SourceParameter,
        storage: OrdinaryParameterStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<OrdinaryParameter>> {
        self.require_service(service)?;
        if descriptor.source != source.source_identity()
            || service.source_identity() != source.source_identity()
        {
            return Err(Error::State(
                "ordinary CUDA parameter source/service differs".into(),
            ));
        }
        let window = Arc::new(source.read_declared_parameter(descriptor, cancelled)?);
        if window.parameter().source_identity() != descriptor.source
            || window.parameter().source_parameter() != *descriptor
            || !window.is_complete()
        {
            return Err(Error::State(
                "ordinary CUDA parameter complete source binding differs".into(),
            ));
        }

        let elements = window.parameter().elements();
        let (payload, kind) = match storage {
            OrdinaryParameterStorage::Original => (window.as_slice().to_vec(), descriptor.kind),
            OrdinaryParameterStorage::Bf16Rne => match descriptor.kind {
                QuantKind::Bf16 => (window.as_slice().to_vec(), QuantKind::Bf16),
                QuantKind::F32 => (f32_bytes_to_bf16_rne(window.as_slice())?, QuantKind::Bf16),
                _ => {
                    return Err(Error::Unsupported(
                        "ordinary CUDA BF16 parameter source codec".into(),
                    ));
                }
            },
        };
        let element_bytes = usize::try_from(kind.byte_len(1)?)
            .map_err(|_| Error::Limit("ordinary CUDA parameter element bytes".into()))?;
        if payload.len()
            != elements.checked_mul(element_bytes).ok_or_else(|| {
                Error::Limit("ordinary CUDA parameter resident extent overflow".into())
            })?
        {
            return Err(Error::State(
                "ordinary CUDA parameter converted byte extent differs".into(),
            ));
        }

        let owner = Arc::new(ParameterOwner {
            source,
            descriptor: descriptor.clone(),
            window,
        });
        self.retain_completion_owner(owner.clone())?;
        let lease = self.upload_bytes(&payload, BufferRole::ImmutableWeights)?;
        let parameter = Arc::new(OrdinaryParameter {
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
