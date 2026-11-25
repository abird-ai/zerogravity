//! Authenticated parameter materialization and initialization-owned H2D.
use super::{ResidentInitialization, ResidentLease};
use cubecl::bytes::Bytes;
#[cfg(feature = "hip-source")]
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::{
    ParameterWindow, VerifiedParameter, VerifiedTensorSource, WindowWorkspace,
};
use zerogravity_core::{
    Error, Result,
    accelerator::program::SourceParameter,
    device_events::BufferRole,
    encoding::{
        ScalarDType::{Bf16, F32},
        StoredTensorEncoding,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(feature = "cuda-cublas"),
    allow(
        dead_code,
        reason = "shared original and RNE policies used by CUDA forward lowering"
    )
)]
pub(crate) enum ParameterStorage {
    /// Preserve the exact authenticated source bytes and codec.
    Original,
    /// Source is either BF16 already or finite F32 converted with IEEE BF16 RNE.
    Bf16Rne,
    /// Exact widening of BF16, or the original finite F32 values.
    F32Exact,
}

impl ParameterStorage {
    pub(crate) fn kind(self, source: StoredTensorEncoding) -> Result<StoredTensorEncoding> {
        match self {
            Self::Original => Ok(source),
            Self::Bf16Rne | Self::F32Exact => {
                if !matches!(source, StoredTensorEncoding::Scalar(Bf16 | F32)) {
                    return Err(Error::Unsupported(
                        "forward parameter conversion codec".into(),
                    ));
                }
                Ok(StoredTensorEncoding::Scalar(if self == Self::Bf16Rne {
                    Bf16
                } else {
                    F32
                }))
            }
        }
    }
    pub(crate) fn resident_bytes(self, descriptor: &SourceParameter) -> Result<usize> {
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

fn aligned(n: usize) -> Result<usize> {
    n.checked_add(255)
        .map(|n| n & !255)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("source parameter aligned extent".into()))
}

fn require_finite_f32(raw: &[u8]) -> Result<()> {
    if raw.is_empty() || !raw.len().is_multiple_of(4) {
        return Err(Error::Malformed("source F32 parameter extent".into()));
    }
    if raw.chunks_exact(4).any(|word| {
        u32::from_le_bytes(word.try_into().expect("F32 parameter word")) & 0x7f80_0000
            == 0x7f80_0000
    }) {
        return Err(Error::Malformed(
            "source parameter contains nonfinite F32".into(),
        ));
    }
    Ok(())
}

/// Materialize the explicit storage representation, including zero padding
/// for the portable aligned F32 layout. No implicit numerical conversion.
fn materialize(
    raw: &[u8],
    kind: StoredTensorEncoding,
    storage: ParameterStorage,
    logical_bytes: usize,
    bytes: usize,
) -> Result<Vec<u8>> {
    let target = storage.kind(kind)?;
    if logical_bytes == 0 || bytes < logical_bytes {
        return Err(Error::State("source parameter materialized extent".into()));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(bytes)
        .map_err(|_| Error::Limit("source parameter staging allocation".into()))?;
    if out.capacity() != bytes {
        return Err(Error::Limit("source parameter staging capacity".into()));
    }
    if target == kind {
        if raw.len() != logical_bytes {
            return Err(Error::State("source parameter original byte extent".into()));
        }
        if storage == ParameterStorage::F32Exact {
            require_finite_f32(raw)?;
        }
        out.extend_from_slice(raw);
    } else if target == StoredTensorEncoding::Scalar(F32) {
        if !raw.len().is_multiple_of(2) || raw.len().checked_mul(2) != Some(logical_bytes) {
            return Err(Error::Malformed("source BF16 parameter extent".into()));
        }
        for word in raw.chunks_exact(2) {
            let bits = u32::from(u16::from_le_bytes([word[0], word[1]])) << 16;
            if bits & 0x7f80_0000 == 0x7f80_0000 {
                return Err(Error::Malformed(
                    "source parameter contains nonfinite BF16".into(),
                ));
            }
            out.extend_from_slice(&bits.to_le_bytes());
        }
    } else {
        if !raw.len().is_multiple_of(4) || raw.len() / 2 != logical_bytes {
            return Err(Error::Malformed("source F32 parameter extent".into()));
        }
        for word in raw.chunks_exact(4) {
            let bits = u32::from_le_bytes(word.try_into().expect("F32 parameter word"));
            if bits & 0x7f80_0000 == 0x7f80_0000 {
                return Err(Error::Malformed(
                    "source parameter contains nonfinite F32".into(),
                ));
            }
            let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
            let bf16 = (rounded >> 16) as u16;
            if bf16 & 0x7f80 == 0x7f80 {
                return Err(Error::Malformed(
                    "finite F32 parameter overflows BF16".into(),
                ));
            }
            out.extend_from_slice(&bf16.to_le_bytes());
        }
    }
    debug_assert_eq!(out.len(), logical_bytes);
    out.resize(bytes, 0);
    Ok(out)
}

/// The source window and actual payload remain alive until the original
/// initialization barrier, or in its quarantine on unknown completion.
struct ParameterUpload {
    _window: ParameterWindow,
    bytes: Bytes,
    _staging: WindowWorkspace,
}

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("source parameter upload cancelled".into()));
    }
    Ok(())
}

impl ResidentInitialization<'_> {
    /// A sealed binding is retained independently of transient IO and
    /// conversion backing. The original allocation Write covers the upload.
    pub(crate) fn upload_parameter(
        &mut self,
        source: &VerifiedTensorSource,
        descriptor: &SourceParameter,
        storage: ParameterStorage,
        bytes: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(ResidentLease, VerifiedParameter)> {
        self.service.ensure_usable()?;
        if self.service.source_identity() != source.source_identity() {
            return Err(Error::State("source parameter service differs".into()));
        }
        let logical_bytes = storage.resident_bytes(descriptor)?;
        if bytes != logical_bytes
            && (storage != ParameterStorage::F32Exact || bytes != aligned(logical_bytes)?)
        {
            return Err(Error::State(
                "source parameter resident layout differs".into(),
            ));
        }
        let window = source.read_declared_parameter(descriptor, cancelled)?;
        let binding = window.parameter().clone();
        let kind = storage.kind(descriptor.kind)?;
        let element_bytes = usize::try_from(kind.byte_len(1)?)
            .map_err(|_| Error::Limit("source parameter element bytes".into()))?;
        if binding.elements().checked_mul(element_bytes) != Some(logical_bytes) {
            return Err(Error::State(
                "source parameter resident byte extent differs".into(),
            ));
        }

        // Parameters may occupy an exact logical view of a padded physical
        // allocation. Use the original portable Write for both providers;
        // the shared backing removes the redundant payload clone without
        // relaxing the native full-allocation source-window contract.
        let staging = source.reserve_workspace(
            bytes
                .checked_mul(2)
                .ok_or_else(|| Error::Limit("source parameter staging charge".into()))?,
        )?;
        let payload = materialize(
            window.as_slice(),
            descriptor.kind,
            storage,
            logical_bytes,
            bytes,
        )?;
        let payload = Bytes::from_bytes_vec(payload);
        if payload.len() != bytes || payload.capacity() > bytes {
            return Err(Error::Limit(
                "source parameter aligned staging capacity".into(),
            ));
        }
        let owner = Arc::new(ParameterUpload {
            _window: window,
            bytes: payload.shared(),
            _staging: staging,
        });
        not_cancelled(cancelled)?;
        self.retain_completion_owner(owner.clone())?;
        let lease = self.allocate_empty(bytes, BufferRole::ImmutableWeights)?;
        self.service
            .client()
            .write(&lease.handle(), owner.bytes.clone());
        self.service.record_value_upload(bytes);
        self.uploaded.insert(lease.id());
        Ok((lease, binding))
    }
}

#[cfg(feature = "hip-source")]
pub(crate) fn upload_parameters(
    source: &VerifiedTensorSource,
    declared: &BTreeMap<String, SourceParameter>,
    init: &mut ResidentInitialization<'_>,
) -> Result<BTreeMap<String, ResidentLease>> {
    let mut parameters = BTreeMap::new();
    for (name, p) in declared {
        if !matches!(p.kind, StoredTensorEncoding::Scalar(F32 | Bf16)) {
            return Err(Error::State("source parameter physical dtype".into()));
        }
        let storage = ParameterStorage::F32Exact;
        let bytes = aligned(storage.resident_bytes(p)?)?;
        let (lease, _) = init.upload_parameter(source, p, storage, bytes, None)?;
        parameters.insert(name.clone(), lease);
    }
    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes_to_f32(raw: &[u8]) -> Result<Vec<u8>> {
        materialize(
            raw,
            StoredTensorEncoding::Scalar(Bf16),
            ParameterStorage::F32Exact,
            raw.len() * 2,
            raw.len() * 2,
        )
    }

    fn f32_bytes_to_bf16_rne(raw: &[u8]) -> Result<Vec<u8>> {
        materialize(
            raw,
            StoredTensorEncoding::Scalar(F32),
            ParameterStorage::Bf16Rne,
            raw.len() / 2,
            raw.len() / 2,
        )
    }

    #[test]
    fn exact_portable_padding_and_original_bytes() -> Result<()> {
        let raw = [0x8000u16.to_le_bytes(), 0x3f80u16.to_le_bytes()].concat();
        let widened = materialize(
            &raw,
            StoredTensorEncoding::Scalar(Bf16),
            ParameterStorage::F32Exact,
            8,
            256,
        )?;
        assert_eq!(&widened[..8], &[0, 0, 0, 0x80, 0, 0, 0x80, 0x3f]);
        assert!(widened[8..].iter().all(|&b| b == 0));
        let original = materialize(
            &widened[..8],
            StoredTensorEncoding::Scalar(F32),
            ParameterStorage::F32Exact,
            8,
            256,
        )?;
        assert_eq!(widened, original);
        assert!(
            materialize(
                &f32::INFINITY.to_le_bytes(),
                StoredTensorEncoding::Scalar(F32),
                ParameterStorage::F32Exact,
                4,
                256
            )
            .is_err()
        );
        assert_eq!(
            materialize(
                &raw,
                StoredTensorEncoding::Scalar(Bf16),
                ParameterStorage::Original,
                4,
                4
            )?,
            raw
        );
        Ok(())
    }

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
