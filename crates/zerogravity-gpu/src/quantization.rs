//! Quantized activation coding and source-format GPU projections.
// Previous modules: activation_quant, activation_quant_fused,
// batch_quant_projection, batch_quant_qualification, q4_q8 and q4.
pub(crate) mod activation;
pub(crate) mod activation_fused;
pub(crate) mod batch_projection;
#[cfg(feature = "qualification")]
pub(crate) mod batch_qualification;
mod common;
pub(crate) mod projection;
pub(crate) mod q4_0;
pub(crate) mod q4_1;
pub(crate) mod q6_k;
pub(crate) mod q8_0;
