//! Borrowed, immutable source binding used only during cold preparation.
use crate::{Error, Result, accelerator::SourceIdentity, encoding::StoredTensorEncoding};

#[derive(Clone, Copy, Debug)]
pub struct SourceTensorRef<'a> {
    pub kind: StoredTensorEncoding,
    pub shape: &'a [u64],
    pub packed_bytes: &'a [u8],
}

pub trait TensorSource: Send + Sync {
    fn source_identity(&self) -> SourceIdentity;
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>>;
}

/// Retained source binding for cold preparation. No byte-access, allocation or
/// execution permission: implementations keep their actual backing and access
/// mechanism. Equality means the original issuer, not just equal source IDs.
pub trait ProgramSource: Clone + Send + Sync {
    fn source_identity(&self) -> SourceIdentity;
    fn same_issuer(&self, other: &Self) -> bool;
}
impl ProgramSource for std::sync::Arc<dyn TensorSource> {
    fn source_identity(&self) -> SourceIdentity {
        self.as_ref().source_identity()
    }
    fn same_issuer(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(self, other)
    }
}

/// Cold operand access for direct native-semantic lowering. The original
/// source chooses its backing: borrowed encoded bytes or sealed file-window
/// descriptors. This neither materializes windows nor grants device access.
pub trait RegionSource: ProgramSource {
    type Matrix<'a>
    where
        Self: 'a;
    type Parameter<'a>
    where
        Self: 'a;

    fn bind_matrix(
        &self,
        descriptor: &crate::accelerator::program::SourceTensor,
    ) -> Result<Self::Matrix<'_>>;
    fn bind_parameter(
        &self,
        descriptor: &crate::accelerator::program::SourceParameter,
    ) -> Result<Self::Parameter<'_>>;
}
impl RegionSource for std::sync::Arc<dyn TensorSource> {
    type Matrix<'a> = SourceTensorRef<'a>;
    type Parameter<'a> = SourceTensorRef<'a>;

    fn bind_matrix(
        &self,
        descriptor: &crate::accelerator::program::SourceTensor,
    ) -> Result<Self::Matrix<'_>> {
        descriptor.bind_source(self.as_ref())
    }
    fn bind_parameter(
        &self,
        descriptor: &crate::accelerator::program::SourceParameter,
    ) -> Result<Self::Parameter<'_>> {
        descriptor.bind_source(self.as_ref())
    }
}

/// Exact immutable content equality; shared storage avoids rereading bytes.
/// Different allocations still compare every byte. This is no source permit.
pub fn exact_text_eq(left: &str, right: &str) -> bool {
    left.len() == right.len() && (std::ptr::eq(left.as_ptr(), right.as_ptr()) || left == right)
}
pub fn exact_text_slice_eq(left: &[&str], right: &[&str]) -> bool {
    left.len() == right.len()
        && (std::ptr::eq(left.as_ptr(), right.as_ptr())
            || left.iter().zip(right).all(|(a, b)| exact_text_eq(a, b)))
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
    fn exact_source_comparison_preserves_different_storage_and_boundaries() {
        let a = String::from("complete source");
        let b = a.clone();
        assert!(exact_text_eq(&a, &a));
        assert!(exact_text_eq(&a, &b));
        assert!(!exact_text_eq(&a, "complete sourcf"));
        assert!(!exact_text_eq(&a, "complete source!"));
        assert!(exact_text_slice_eq(&[&a], &[&b]));
        assert!(!exact_text_slice_eq(&["ab", "c"], &["a", "bc"]));
        assert!(!exact_text_slice_eq(&[&a], &[]));
    }

    #[test]
    fn borrowed_source_geometry_rejects_overflow_truncation_and_empty_axes() {
        let packed = [0u8; 8];
        let valid = SourceTensorRef {
            kind: StoredTensorEncoding::Scalar(crate::encoding::ScalarDType::F32),
            shape: &[2],
            packed_bytes: &packed,
        };
        valid.validate().unwrap();
        for shape in [&[][..], &[0][..], &[u64::MAX, 2][..], &[3][..]] {
            assert!(SourceTensorRef { shape, ..valid }.validate().is_err());
        }
        assert!(
            SourceTensorRef {
                kind: StoredTensorEncoding::Quantized(crate::encoding::QuantScheme::Q8_0),
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
