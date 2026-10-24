//! One closed matrix fragment via the existing generic mathematical engine.
use crate::Result;
use crate::accelerator::program::{BulkF32Matrix, MathematicalBackend};
pub(crate) fn bulk_f32_slice<M: MathematicalBackend>(
    math: &mut M,
    matrix: &BulkF32Matrix,
) -> Result<M::Value> {
    let left = math.bulk_f32_input(matrix.left())?;
    let right = math.bulk_f32_input(matrix.right())?;
    math.bulk_f32_matmul(matrix, &left, &right)
}
