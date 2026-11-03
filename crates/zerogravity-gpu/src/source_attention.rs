//! Shared explicit mathematical attention kernels; no backend eligibility.
pub(crate) mod kernels;

#[cfg(feature = "hip-source")]
pub(crate) mod rows;
#[cfg(feature = "qwen-source-qualification")]
pub(crate) mod single_row;
#[cfg(feature = "hip-source")]
pub(crate) mod transport;
