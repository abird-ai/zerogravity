//! One actual retained producer through the common source operator seam.
use crate::Result;
use crate::accelerator::program::MathematicalBackend;
use crate::operator_slice::bulk_f32::source_bindings::{GdnOutputBinding, one};
pub(crate) fn zen_ssm_out_slice<M: MathematicalBackend>(
    math: &mut M,
    descriptor: &GdnOutputBinding,
) -> Result<M::Value> {
    let input = one(math, descriptor.input_operator(), &[])?;
    one(
        math,
        descriptor.projection_operator(),
        std::slice::from_ref(&input),
    )
}
