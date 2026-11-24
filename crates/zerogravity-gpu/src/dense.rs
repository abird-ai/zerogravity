//! Borrowed dense commands. No allocation, submission or publication authority.
use cubecl::server::Handle;
pub(crate) use zerogravity_core::accelerator::dense::{
    DensePrecision, Gemm, Transpose, UnitLowerSolve,
};

pub(crate) enum Command<'a> {
    Gemm {
        geometry: Gemm,
        left: &'a Handle,
        right: &'a Handle,
        output: &'a Handle,
    },
    Solve {
        geometry: UnitLowerSolve,
        lower: &'a Handle,
        rhs: &'a Handle,
    },
}
