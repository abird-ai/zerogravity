//! Explicit physical/source qualification entry point. No production frontend.

use cubecl::{Device, prelude::*};

#[cube(launch_unchecked)]
fn multiply_by_two(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
    }
}

/// Actual basic device dispatch; no model support claim.
pub fn qualify_dispatch(device: Device, name: &str) {
    let client = device.client();
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let src = client.create_from_slice(f32::as_bytes(&input));
    let dst = client.empty(input.len() * std::mem::size_of::<f32>());
    // SAFETY: the launch dimensions cover exactly four in-bounds f32 elements;
    // our kernel additionally checks the dynamic array length per lane.
    unsafe {
        multiply_by_two::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(src, input.len()),
            BufferArg::from_raw_parts(dst.clone(), input.len()),
        );
    }
    let bytes = client.read_one(dst).expect("GPU kernel execution failed");
    let values = f32::from_bytes(&bytes);
    assert_eq!(
        values,
        &[2.0, 4.0, 6.0, 8.0],
        "physical GPU output differs from the checked CPU oracle"
    );
    println!("PASS {name}: {} {values:?}", client.name());
}

pub use crate::batch_attention::qualify as qualify_batched_attention;
pub use crate::batch_elementwise::qualify as qualify_batched_elementwise;
pub use crate::batch_kv::qualify as qualify_batched_kv;
pub use crate::batch_projection::qualify as qualify_batched_projection;
pub use crate::batch_recurrent::qualify as qualify_batched_recurrent;
pub use crate::gpu_telemetry::{read as read_driver_memory, report as report_driver_memory};
#[cfg(feature = "hardware-abort-probe")]
pub mod physical_abort {
    pub use crate::physical_abort_probe::{
        AbortOperator, arm_physical_abort, physical_launch_returned, qualify_abort_operator,
        qualify_invalid_quant,
    };
}
pub use crate::quantization::batch_projection::qualify as qualify_quantized_projection_batch;
pub use crate::quantization::projection::CubeMixedProvider;
pub use crate::quantization::q4_0::{CubeQ4Q8Provider, qualification::run as qualify_q4_0_fixture};
pub use crate::resident_inference::diagnostics as resident;
#[cfg(feature = "cuda")]
pub use crate::rope_angles::qualification::qualify_angles_with_plan;
pub use crate::source_accumulation::SourceAccumulation;

/// ROOT-authenticated CPU-reference inputs/outputs; closed operator only.
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub async fn qualify_cpu_profile_bf16_window(
    device: cubecl::Device,
    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
    descriptor: zerogravity_core::accelerator::program::SourceTensor,
    rows: std::ops::Range<usize>,
    inputs: [Vec<u8>; 2],
    public_outputs: [Vec<u8>; 2],
    device_cap: u64,
) -> zerogravity_core::Result<()> {
    crate::cpu_bf16_projection::qualification::qualify_window(
        device,
        source,
        descriptor,
        rows,
        inputs,
        public_outputs,
        device_cap,
    )
    .await
}

/// Separate fresh process only. Actual unknown completion must stay fenced.
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub async fn qualify_cpu_profile_bf16_abandon(
    device: cubecl::Device,
    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
    descriptor: zerogravity_core::accelerator::program::SourceTensor,
    rows: std::ops::Range<usize>,
    input: Vec<u8>,
    device_cap: u64,
) -> zerogravity_core::Result<()> {
    crate::cpu_bf16_projection::qualification::qualify_abandon(
        device, source, descriptor, rows, input, device_cap,
    )
    .await
}

/// Explicitly SYNTHETIC generic operator fixtures; no trained source/model claim.
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use crate::cpu_bf16_projection::adversary::{
    ADVERSARY_WIDTHS as CPU_BF16_ADVERSARY_WIDTHS, AdversaryKind as CpuBf16AdversaryKind,
    SYNTHETIC_ROWS as CPU_BF16_SYNTHETIC_ROWS, bad_geometry_name as cpu_bf16_bad_geometry_name,
};

/// Root-only same-kernel guarded proof or ONE deliberate nonfinite status case.
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub async fn qualify_cpu_profile_bf16_adversaries(
    device: cubecl::Device,
    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
    device_cap: u64,
    fault: Option<(CpuBf16AdversaryKind, usize)>,
) -> zerogravity_core::Result<()> {
    crate::cpu_bf16_projection::adversary::qualify(device, source, device_cap, fault).await
}

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use crate::resident_inference::{
    Qwen2bGdnComponentFault, Qwen2bGdnComponentFaultOutcome, Qwen2bGdnComponentReadback,
    Qwen2bGdnComponentReceipt, qualify_qwen2b_gdn_component, qualify_qwen2b_gdn_component_fault,
};

#[cfg(target_os = "linux")]
pub use crate::resident_leases::resident_matrix::qualification::{
    MatrixTransportMode, MatrixTransportOutcome, qualify_bf16_matrix_cross_provider,
    qualify_bf16_matrix_transport,
};

#[cfg(target_os = "linux")]
pub use crate::resident_program::typed_slice::qualification::{
    CompletedSlice, SliceFault, SliceOutcome, qualify_portable_bf16_slice,
    qualify_portable_bf16_slice_cross_provider,
};
