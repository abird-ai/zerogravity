//! Same generic mathematical seam; one retained producer and one projection.
use crate::Result;
use crate::accelerator::program::{MathematicalBackend, ZenSsmOutDescriptor};

pub(crate) fn zen_ssm_out_slice<M: MathematicalBackend>(
    math: &mut M,
    descriptor: &ZenSsmOutDescriptor,
) -> Result<M::Value> {
    let input = math.zen_ssm_out_input(descriptor)?;
    math.zen_ssm_out_project(descriptor, &input)
}
