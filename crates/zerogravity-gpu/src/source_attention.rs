//! Shared explicit mathematical attention kernels; no backend eligibility.
pub(crate) mod kernels;

#[cfg(feature = "qwen-source-qualification")]
pub(crate) mod single_row;
#[cfg(feature = "qwen-source-qualification")]
pub(crate) mod transport;
