//! Borrowed, immutable source binding used only during cold preparation.
use crate::{Error, Result, accelerator::SourceIdentity, quant::QuantKind};

#[derive(Clone, Copy, Debug)]
pub struct SourceTensorRef<'a> {
    pub kind: QuantKind,
    pub shape: &'a [u64],
    pub packed_bytes: &'a [u8],
}

pub trait TensorSource: Send + Sync {
    fn source_identity(&self) -> SourceIdentity;
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>>;
}

impl SourceTensorRef<'_> {
    pub fn validate(&self) -> Result<()> {
        let elements = self
            .shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::Limit("source tensor shape overflow".into()))?;
        if self.shape.is_empty()
            || self.shape.contains(&0)
            || self.kind.byte_len(elements)? != self.packed_bytes.len() as u64
        {
            return Err(Error::Malformed(
                "source tensor borrowed geometry mismatch".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_source_geometry_rejects_overflow_truncation_and_empty_axes() {
        let packed = [0u8; 8];
        let valid = SourceTensorRef {
            kind: QuantKind::F32,
            shape: &[2],
            packed_bytes: &packed,
        };
        valid.validate().unwrap();
        for shape in [&[][..], &[0][..], &[u64::MAX, 2][..], &[3][..]] {
            assert!(SourceTensorRef { shape, ..valid }.validate().is_err());
        }
        assert!(
            SourceTensorRef {
                kind: QuantKind::Q8_0,
                shape: &[1],
                packed_bytes: &packed
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn source_identity_issuer_is_shared_and_unique_across_threads() {
        let ids = std::thread::scope(|scope| {
            (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        (0..16)
                            .map(|_| SourceIdentity::for_new_allocation().unwrap())
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(ids.iter().all(|id| id.is_bound()));
        let unique = ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), ids.len());
    }
}
