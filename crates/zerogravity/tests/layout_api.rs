//! Compatibility contracts across the concrete crate owners and facade.
#[cfg(feature = "cpu")]
#[test]
fn legacy_cpu_weights_ufcs_and_mutable_architecture_api_remain_available() {
    use std::path::Path;
    use zerogravity::{Result, weights::CpuWeights};
    // Deref forwarding alone would preserve method syntax but break these
    // existing associated-function/UFCS call sites after storage extraction.
    let _: fn(&Path) -> Result<CpuWeights> = CpuWeights::open;
    let _: fn(&CpuWeights) -> usize = CpuWeights::source_len;
    let _: fn(&CpuWeights, &str) -> bool = CpuWeights::contains;
    let _: for<'a> fn(&'a CpuWeights, &str) -> Result<&'a [u8]> = CpuWeights::tensor_bytes;
    let _: fn(&CpuWeights, &str) -> Result<Vec<f32>> = CpuWeights::vector;
    let _: fn(&CpuWeights, &str) -> Result<Vec<f32>> = CpuWeights::matrix_f32;
    let _: fn(&CpuWeights, u32, &str) -> Result<Vec<f32>> = CpuWeights::embed;
    let _: fn(&CpuWeights, &str, &[f32]) -> Result<Vec<f32>> = CpuWeights::matvec;
    let _: fn(&CpuWeights, &str, &[f32]) -> Result<Vec<f32>> = CpuWeights::matvec_fp32_reference;
    fn architecture_field(weights: &mut CpuWeights) -> &mut String {
        &mut weights.architecture
    }
    let _: fn(&mut CpuWeights) -> &mut String = architecture_field;
}

#[test]
fn facade_descriptor_and_error_types_are_the_same_concrete_core_types() {
    let encoding: zerogravity::quant::QuantKind = zerogravity_core::quant::QuantKind::Q6K;
    assert_eq!(encoding.byte_len(256).unwrap(), 210);
    let error: zerogravity::Error = zerogravity_core::Error::State("same authority".into());
    assert_eq!(error.to_string(), "invalid session state: same authority");
}
