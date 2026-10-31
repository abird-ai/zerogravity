//! Exact source profiles remain separate from storage and physical layout.
pub use crate::accelerator::program::{
    ActivationPrecision, ProgramNumericalProfile, SourceNumericalContract, StandaloneExpProfile,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationEncoding {
    Scalar(ActivationPrecision),
    GgmlQ8_0,
    GgmlQ8_1,
    GgmlQ8K,
}

/// Description only; a provider must validate the whole source/operator contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericalContract {
    pub profile: ProgramNumericalProfile,
    pub activation: ActivationEncoding,
    pub standalone_exp: Option<StandaloneExpProfile>,
}
