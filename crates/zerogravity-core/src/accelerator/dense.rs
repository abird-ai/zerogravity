//! Compact row-major dense operations, independent of provider ABI or ownership.
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transpose {
    None,
    Transposed,
}
impl Transpose {
    pub const fn is_transposed(self) -> bool {
        matches!(self, Self::Transposed)
    }
}

/// Storage precision; both operations accumulate in FP32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensePrecision {
    Bf16,
    F32,
}
impl DensePrecision {
    const fn bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

/// C[M,N] = op(A)[M,K] * op(B)[K,N]. Each batch is contiguous and independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gemm {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batches: usize,
    pub left: Transpose,
    pub right: Transpose,
    pub precision: DensePrecision,
}
impl Gemm {
    pub const fn bf16_projection(rows: usize, outputs: usize, inner: usize) -> Self {
        Self {
            m: rows,
            n: outputs,
            k: inner,
            batches: 1,
            left: Transpose::None,
            right: Transpose::Transposed,
            precision: DensePrecision::Bf16,
        }
    }

    /// Byte extents in mathematical order: left, right, output.
    pub fn operand_bytes(self) -> Result<[usize; 3]> {
        let extent = |a, b| extent(a, b, self.batches, self.precision.bytes());
        Ok([
            extent(self.m, self.k)?,
            extent(self.k, self.n)?,
            extent(self.m, self.n)?,
        ])
    }
}

/// In-place solution of L * X = B, with unit lower triangular L.
/// L's diagonal and upper triangle are not operands. FP32 storage/compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitLowerSolve {
    pub order: usize,
    pub rhs_columns: usize,
    pub batches: usize,
}
impl UnitLowerSolve {
    /// Byte extents in mathematical order: coefficient, read/write RHS.
    pub fn operand_bytes(self) -> Result<[usize; 2]> {
        Ok([
            extent(self.order, self.order, self.batches, 4)?,
            extent(self.order, self.rhs_columns, self.batches, 4)?,
        ])
    }
}

fn extent(a: usize, b: usize, batches: usize, bytes: usize) -> Result<usize> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(batches))
        .and_then(|n| n.checked_mul(bytes))
        .filter(|n| *n != 0)
        .ok_or_else(|| Error::Limit("dense compact operand extent".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_dense_operands() {
        let projection = Gemm::bf16_projection(7, 11, 13);
        assert_eq!(projection.operand_bytes().unwrap(), [182, 286, 154]);
        let f32 = Gemm {
            batches: 3,
            precision: DensePrecision::F32,
            ..projection
        };
        assert_eq!(f32.operand_bytes().unwrap(), [1092, 1716, 924]);
        let solve = UnitLowerSolve {
            order: 7,
            rhs_columns: 11,
            batches: 3,
        };
        assert_eq!(solve.operand_bytes().unwrap(), [588, 924]);
        assert!(
            Gemm {
                m: usize::MAX,
                ..projection
            }
            .operand_bytes()
            .is_err()
        );
        assert!(
            UnitLowerSolve {
                batches: 0,
                ..solve
            }
            .operand_bytes()
            .is_err()
        );
    }
}
