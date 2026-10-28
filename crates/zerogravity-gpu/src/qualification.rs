//! Explicit physical/source qualification entry point. No production frontend.

use cubecl::{Device, prelude::*};

#[cube(launch_unchecked)]
fn multiply_by_two(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
    }
}

#[cfg(all(test, feature = "cuda-cublas"))]
#[cube(launch_unchecked)]
fn source_multiply_oracle(
    input: &[f32],
    gate: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let a = input[i];
        let b = gate[i];
        let y = crate::source_arithmetic::bf16(crate::source_arithmetic::mul(a, b));
        output[i] = y;
        if !crate::source_arithmetic::source_bf16(a)
            || !crate::source_arithmetic::source_bf16(b)
            || !crate::source_arithmetic::finite(y)
        {
            status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
        }
    }
}

#[cfg(all(test, feature = "cuda-cublas"))]
#[cube(launch_unchecked)]
fn source_residual_add_oracle(
    input: &[f32],
    residual: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let a = input[i];
        let b = residual[i];
        let y = crate::source_arithmetic::bf16(crate::source_arithmetic::add(a, b));
        output[i] = y;
        if !crate::source_arithmetic::source_bf16(a)
            || !crate::source_arithmetic::source_bf16(b)
            || !crate::source_arithmetic::finite(y)
        {
            status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
        }
    }
}

#[cfg(all(test, feature = "cuda-cublas"))]
#[cube(launch_unchecked)]
fn source_silu_mul_oracle(
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let g = gate[i];
        let u = up[i];
        let s = crate::source_arithmetic::bf16(crate::source_sleef::silu(g));
        let y = crate::source_arithmetic::bf16(crate::source_arithmetic::mul(s, u));
        output[i] = y;
        if !crate::source_arithmetic::source_bf16(g)
            || !crate::source_arithmetic::source_bf16(u)
            || !crate::source_arithmetic::finite(y)
        {
            status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
        }
    }
}

#[cfg(all(test, feature = "cuda-cublas"))]
#[cube(launch_unchecked)]
fn source_sigmoid_oracle(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let x = input[i];
        let e = crate::source_sleef::exp(crate::source_arithmetic::neg(x));
        let y = crate::source_arithmetic::bf16(crate::source_arithmetic::div(
            1.0f32,
            crate::source_arithmetic::add(1.0f32, e),
        ));
        output[i] = y;
        if !crate::source_arithmetic::source_bf16(x)
            || !crate::source_arithmetic::finite(y)
            || y < 0.0f32
            || y > 1.0f32
        {
            status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
        }
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

#[cfg(all(
    test,
    feature = "cuda-cublas",
    feature = "qwen-source-qualification",
    target_os = "linux"
))]
fn qualify_ordinary_rms_source_parity(device: Device) -> zerogravity_core::Result<()> {
    use crate::cuda::ordinary_text::{Access, BufferRole, Geometry, LaunchBinding, Program};
    use zerogravity_core::{Error, Result};

    const WIDTH: usize = 1024;
    const EPSILON: f32 = 1e-6;
    const VALUE_PATTERNS: [u16; 12] = [
        0x3d80, 0xbd80, 0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x4040,
        0xc040,
    ];
    const GAMMA_PATTERNS: [u16; 8] = [
        0x0000, 0x3c80, 0xbc80, 0x3d00, 0xbd00, 0x3d80, 0xbd80, 0x3e00,
    ];

    fn bf16_values(count: usize, salt: usize, patterns: &[u16]) -> Vec<u16> {
        (0..count)
            .map(|index| patterns[(index * 17 + salt * 13) % patterns.len()])
            .collect()
    }
    fn bf16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
    fn widen(values: &[u16]) -> Vec<f32> {
        values
            .iter()
            .map(|&value| f32::from_bits(u32::from(value) << 16))
            .collect()
    }
    fn parse_u16(bytes: &[u8], label: &str) -> Result<Vec<u16>> {
        if !bytes.len().is_multiple_of(2) {
            return Err(Error::State(format!("{label} BF16 readback alignment")));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes([word[0], word[1]]))
            .collect())
    }
    fn parse_u32(bytes: &[u8], label: &str) -> Result<u32> {
        let word: [u8; 4] = bytes
            .try_into()
            .map_err(|_| Error::State(format!("{label} status readback extent")))?;
        Ok(u32::from_le_bytes(word))
    }

    let client = device.client();
    let ordinary = Program::new(Geometry::qwen35_08())?;
    let gamma_bits = bf16_values(WIDTH, 5, &GAMMA_PATTERNS);
    let source_gamma = widen(&gamma_bits);

    for rows in [1usize, 2, 17, 64] {
        let count = rows * WIDTH;
        let input_bits = bf16_values(count, rows, &VALUE_PATTERNS);
        let source_input = widen(&input_bits);

        let ordinary_input = client.create_from_slice(&bf16_bytes(&input_bits));
        let ordinary_gamma = client.create_from_slice(&bf16_bytes(&gamma_bits));
        let ordinary_output = client.empty(count * 2);
        let ordinary_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        let definition = ordinary.rms(rows, EPSILON)?;
        let prepared = definition.prepare_owned(
            &client,
            vec![
                LaunchBinding::new(BufferRole::Input, Access::Read, ordinary_input),
                LaunchBinding::new(BufferRole::RawGamma, Access::Read, ordinary_gamma),
                LaunchBinding::new(BufferRole::Output, Access::Write, ordinary_output.clone()),
            ],
            LaunchBinding::new(
                BufferRole::Status,
                Access::ReadWrite,
                ordinary_status.clone(),
            ),
        )?;
        prepared.enqueue_owned(&client, &ordinary_status)?;

        let source_input_handle = client.create_from_slice(f32::as_bytes(&source_input));
        let source_gamma_handle = client.create_from_slice(f32::as_bytes(&source_gamma));
        let source_output = client.empty(count * 4);
        let source_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        let zero = client.create_from_slice(f32::as_bytes(&[0.0f32]));
        let extents = [count, rows, rows, rows, rows, rows, count, count];
        let scratch = extents
            .into_iter()
            .map(|elements| client.empty(elements * 4))
            .collect::<Vec<_>>();
        crate::source_rms::launch(
            &client,
            &scratch,
            &source_input_handle,
            Some(&source_gamma_handle),
            &source_output,
            &source_status,
            &zero,
            WIDTH,
            rows,
            crate::source_rms::Gamma::OnePlusRaw,
        )?;

        let ordinary_bits = parse_u16(
            &client
                .read_one(ordinary_output)
                .map_err(|error| Error::State(format!("ordinary RMS output readback: {error}")))?,
            "ordinary RMS",
        )?;
        let source_raw = client
            .read_one(source_output)
            .map_err(|error| Error::State(format!("source RMS output readback: {error}")))?;
        let source_values = f32::from_bytes(&source_raw);
        if ordinary_bits.len() != count || source_values.len() != count {
            return Err(Error::State(
                "ordinary/source RMS output extent differs".into(),
            ));
        }
        let ordinary_flags = parse_u32(
            &client
                .read_one(ordinary_status)
                .map_err(|error| Error::State(format!("ordinary RMS status readback: {error}")))?,
            "ordinary RMS",
        )?;
        let source_flags = parse_u32(
            &client
                .read_one(source_status)
                .map_err(|error| Error::State(format!("source RMS status readback: {error}")))?,
            "source RMS",
        )?;
        if ordinary_flags != 0 || source_flags != 0 {
            return Err(Error::State(format!(
                "ordinary/source RMS status differs rows={rows}: ordinary={ordinary_flags:#x} source={source_flags:#x}"
            )));
        }
        for (index, (&actual, &expected)) in ordinary_bits.iter().zip(source_values).enumerate() {
            if expected.to_bits() & 0xffff != 0 {
                return Err(Error::State(format!(
                    "source RMS output is not BF16-widened rows={rows} index={index} bits={:#010x}",
                    expected.to_bits()
                )));
            }
            let expected = (expected.to_bits() >> 16) as u16;
            if actual != expected {
                return Err(Error::State(format!(
                    "ordinary/source RMS bit mismatch rows={rows} index={index} ordinary={actual:#06x} source={expected:#06x}"
                )));
            }
        }
        println!("ordinary-source-rms rows={rows} PASS");
    }
    Ok(())
}

#[cfg(all(
    test,
    feature = "cuda-cublas",
    feature = "qwen-source-qualification",
    target_os = "linux"
))]
fn qualify_ordinary_residual_rms_source_parity(device: Device) -> zerogravity_core::Result<()> {
    use crate::cuda::ordinary_text::{Access, BufferRole, Geometry, LaunchBinding, Program};
    use zerogravity_core::{Error, Result};

    const WIDTH: usize = 1024;
    const EPSILON: f32 = 1e-6;
    const INPUTS: [u16; 12] = [
        0x3d80, 0xbd80, 0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x4040,
        0xc040,
    ];
    const RESIDUALS: [u16; 10] = [
        0x3c80, 0xbc80, 0x3d00, 0xbd00, 0x3e00, 0xbe00, 0x3f00, 0xbf00, 0x3f80, 0xbf80,
    ];
    const GAMMA: [u16; 8] = [
        0x0000, 0x3c80, 0xbc80, 0x3d00, 0xbd00, 0x3d80, 0xbd80, 0x3e00,
    ];

    fn bits(count: usize, salt: usize, patterns: &[u16]) -> Vec<u16> {
        (0..count)
            .map(|i| patterns[(i * 23 + salt * 5) % patterns.len()])
            .collect()
    }
    fn bytes(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn widen(values: &[u16]) -> Vec<f32> {
        values
            .iter()
            .map(|&v| f32::from_bits(u32::from(v) << 16))
            .collect()
    }
    fn parse_bf16(raw: &[u8], label: &str) -> Result<Vec<u16>> {
        if !raw.len().is_multiple_of(2) {
            return Err(Error::State(format!("{label} BF16 alignment")));
        }
        Ok(raw
            .chunks_exact(2)
            .map(|w| u16::from_le_bytes([w[0], w[1]]))
            .collect())
    }
    fn parse_status(raw: &[u8], label: &str) -> Result<u32> {
        let word: [u8; 4] = raw
            .try_into()
            .map_err(|_| Error::State(format!("{label} status extent")))?;
        Ok(u32::from_le_bytes(word))
    }
    fn grid(elements: usize) -> CubeCount {
        CubeCount::Static((elements as u32).div_ceil(32), 1, 1)
    }
    fn arg(handle: cubecl::server::Handle, elements: usize) -> BufferArg {
        // SAFETY: qualification fixtures allocate exact typed extents.
        unsafe { BufferArg::from_raw_parts(handle, elements) }
    }

    let client = device.client();
    let ordinary = Program::new(Geometry::qwen35_08())?;
    let gamma_bits = bits(WIDTH, 9, &GAMMA);
    let source_gamma = client.create_from_slice(f32::as_bytes(&widen(&gamma_bits)));

    for rows in [1usize, 2, 17, 64] {
        let count = rows * WIDTH;
        let input_bits = bits(count, rows, &INPUTS);
        let residual_bits = bits(count, rows + 7, &RESIDUALS);

        let ordinary_input = client.create_from_slice(&bytes(&input_bits));
        let ordinary_residual = client.create_from_slice(&bytes(&residual_bits));
        let ordinary_gamma = client.create_from_slice(&bytes(&gamma_bits));
        let ordinary_output = client.empty(count * 2);
        let ordinary_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        let definition = ordinary.residual_rms(rows, EPSILON)?;
        let prepared = definition.prepare_owned(
            &client,
            vec![
                LaunchBinding::new(BufferRole::Input, Access::Read, ordinary_input),
                LaunchBinding::new(
                    BufferRole::Residual,
                    Access::ReadWrite,
                    ordinary_residual.clone(),
                ),
                LaunchBinding::new(BufferRole::RawGamma, Access::Read, ordinary_gamma),
                LaunchBinding::new(BufferRole::Output, Access::Write, ordinary_output.clone()),
            ],
            LaunchBinding::new(
                BufferRole::Status,
                Access::ReadWrite,
                ordinary_status.clone(),
            ),
        )?;
        prepared.enqueue_owned(&client, &ordinary_status)?;

        let source_input = client.create_from_slice(f32::as_bytes(&widen(&input_bits)));
        let source_residual_in = client.create_from_slice(f32::as_bytes(&widen(&residual_bits)));
        let source_residual = client.empty(count * 4);
        let source_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        unsafe {
            source_residual_add_oracle::launch_unchecked(
                &client,
                grid(count),
                CubeDim::new_1d(32),
                arg(source_input, count),
                arg(source_residual_in, count),
                arg(source_residual.clone(), count),
                arg(source_status.clone(), 1),
                count,
            );
        }

        let source_output = client.empty(count * 4);
        let zero = client.create_from_slice(f32::as_bytes(&[0.0f32]));
        let extents = [count, rows, rows, rows, rows, rows, count, count];
        let scratch = extents
            .into_iter()
            .map(|elements| client.empty(elements * 4))
            .collect::<Vec<_>>();
        crate::source_rms::launch(
            &client,
            &scratch,
            &source_residual,
            Some(&source_gamma),
            &source_output,
            &source_status,
            &zero,
            WIDTH,
            rows,
            crate::source_rms::Gamma::OnePlusRaw,
        )?;

        let ordinary_flags = parse_status(
            &client
                .read_one(ordinary_status)
                .map_err(|error| Error::State(format!("ordinary residual RMS status: {error}")))?,
            "ordinary residual RMS",
        )?;
        let source_flags = parse_status(
            &client
                .read_one(source_status)
                .map_err(|error| Error::State(format!("source residual RMS status: {error}")))?,
            "source residual RMS",
        )?;
        if ordinary_flags != 0 || source_flags != 0 {
            return Err(Error::State(format!(
                "ordinary/source residual RMS status rows={rows}: ordinary={ordinary_flags:#x} source={source_flags:#x}"
            )));
        }

        let ordinary_residual_bits = parse_bf16(
            &client.read_one(ordinary_residual).map_err(|error| {
                Error::State(format!("ordinary residual update readback: {error}"))
            })?,
            "ordinary residual",
        )?;
        let ordinary_output_bits = parse_bf16(
            &client.read_one(ordinary_output).map_err(|error| {
                Error::State(format!("ordinary residual RMS output readback: {error}"))
            })?,
            "ordinary residual RMS",
        )?;
        let source_residual_raw = client
            .read_one(source_residual)
            .map_err(|error| Error::State(format!("source residual update readback: {error}")))?;
        let source_output_raw = client.read_one(source_output).map_err(|error| {
            Error::State(format!("source residual RMS output readback: {error}"))
        })?;
        for (label, ordinary_bits, source_raw) in [
            (
                "residual",
                ordinary_residual_bits.as_slice(),
                &source_residual_raw[..],
            ),
            (
                "output",
                ordinary_output_bits.as_slice(),
                &source_output_raw[..],
            ),
        ] {
            let source_values = f32::from_bytes(source_raw);
            if ordinary_bits.len() != count || source_values.len() != count {
                return Err(Error::State(format!(
                    "residual RMS {label} extent differs rows={rows}"
                )));
            }
            for (index, (&actual, &expected)) in ordinary_bits.iter().zip(source_values).enumerate()
            {
                if expected.to_bits() & 0xffff != 0 {
                    return Err(Error::State(format!(
                        "source residual RMS {label} not BF16-widened rows={rows} index={index}"
                    )));
                }
                let expected = (expected.to_bits() >> 16) as u16;
                if actual != expected {
                    return Err(Error::State(format!(
                        "ordinary/source residual RMS {label} mismatch rows={rows} index={index} ordinary={actual:#06x} source={expected:#06x}"
                    )));
                }
            }
        }
        println!("ordinary-source-residual-rms rows={rows} PASS");
    }
    Ok(())
}

#[cfg(all(
    test,
    feature = "cuda-cublas",
    feature = "qwen-source-qualification",
    target_os = "linux"
))]
#[test]
#[ignore = "actual NVIDIA ordinary/source Qwen residual-RMS numerical parity"]
fn ordinary_residual_rms_matches_source_reference_on_nvidia() -> zerogravity_core::Result<()> {
    qualify_ordinary_residual_rms_source_parity(Device::cuda(0).map_err(|error| {
        zerogravity_core::Error::State(format!("CUDA device 0 unavailable: {error}"))
    })?)
}

#[cfg(all(
    test,
    feature = "cuda-cublas",
    feature = "qwen-source-qualification",
    target_os = "linux"
))]
fn qualify_ordinary_head_rms_source_parity(device: Device) -> zerogravity_core::Result<()> {
    use crate::cuda::ordinary_text::{Access, BufferRole, Geometry, LaunchBinding, Program};
    use zerogravity_core::{Error, Result};

    const WIDTH: usize = 256;
    const ROW_STRIDE: usize = 5120;
    const QUERY_HEADS: usize = 8;
    const KEY_HEADS: usize = 2;
    const QUERY_HEAD_STRIDE: usize = 512;
    const KEY_BASE: usize = 4096;
    const EPSILON: f32 = 1e-6;
    const VALUES: [u16; 12] = [
        0x3d80, 0xbd80, 0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x4040,
        0xc040,
    ];
    const GAMMA: [u16; 8] = [
        0x0000, 0x3c80, 0xbc80, 0x3d00, 0xbd00, 0x3d80, 0xbd80, 0x3e00,
    ];

    fn bits(count: usize, salt: usize, patterns: &[u16]) -> Vec<u16> {
        (0..count)
            .map(|index| patterns[(index * 19 + salt * 11) % patterns.len()])
            .collect()
    }
    fn bytes(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn widen(values: &[u16]) -> Vec<f32> {
        values
            .iter()
            .map(|&v| f32::from_bits(u32::from(v) << 16))
            .collect()
    }
    fn parse_bf16(raw: &[u8], label: &str) -> Result<Vec<u16>> {
        if !raw.len().is_multiple_of(2) {
            return Err(Error::State(format!("{label} BF16 alignment")));
        }
        Ok(raw
            .chunks_exact(2)
            .map(|w| u16::from_le_bytes([w[0], w[1]]))
            .collect())
    }
    fn parse_status(raw: &[u8], label: &str) -> Result<u32> {
        let word: [u8; 4] = raw
            .try_into()
            .map_err(|_| Error::State(format!("{label} status extent")))?;
        Ok(u32::from_le_bytes(word))
    }
    fn source_expected(
        client: &cubecl::client::Client,
        input_bits: &[u16],
        gamma_bits: &[u16],
        groups: usize,
    ) -> Result<Vec<u16>> {
        let gamma = client.create_from_slice(f32::as_bytes(&widen(gamma_bits)));
        let zero = client.create_from_slice(f32::as_bytes(&[0.0f32]));
        let mut result = Vec::with_capacity(input_bits.len());
        let mut group_base = 0usize;
        while group_base < groups {
            let chunk_groups = (groups - group_base).min(64);
            let start = group_base * WIDTH;
            let end = start + chunk_groups * WIDTH;
            let input = client.create_from_slice(f32::as_bytes(&widen(&input_bits[start..end])));
            let output = client.empty((end - start) * 4);
            let status = client.create_from_slice(u32::as_bytes(&[0u32]));
            let count = chunk_groups * WIDTH;
            let extents = [
                count,
                chunk_groups,
                chunk_groups,
                chunk_groups,
                chunk_groups,
                chunk_groups,
                count,
                count,
            ];
            let scratch = extents
                .into_iter()
                .map(|elements| client.empty(elements * 4))
                .collect::<Vec<_>>();
            crate::source_rms::launch(
                client,
                &scratch,
                &input,
                Some(&gamma),
                &output,
                &status,
                &zero,
                WIDTH,
                chunk_groups,
                crate::source_rms::Gamma::OnePlusRaw,
            )?;
            let flags = parse_status(
                &client
                    .read_one(status)
                    .map_err(|error| Error::State(format!("source head RMS status: {error}")))?,
                "source head RMS",
            )?;
            if flags != 0 {
                return Err(Error::State(format!(
                    "source head RMS status nonzero: {flags:#x}"
                )));
            }
            let raw = client
                .read_one(output)
                .map_err(|error| Error::State(format!("source head RMS output: {error}")))?;
            let values = f32::from_bytes(&raw);
            for (index, &value) in values.iter().enumerate() {
                if value.to_bits() & 0xffff != 0 {
                    return Err(Error::State(format!(
                        "source head RMS not BF16-widened group_base={group_base} index={index}"
                    )));
                }
                result.push((value.to_bits() >> 16) as u16);
            }
            group_base += chunk_groups;
        }
        Ok(result)
    }

    let client = device.client();
    let ordinary = Program::new(Geometry::qwen35_08())?;
    let gamma_bits = bits(WIDTH, 7, &GAMMA);

    for rows in [1usize, 2, 17, 64] {
        let query_bits = bits(rows * QUERY_HEADS * WIDTH, rows, &VALUES);
        let key_bits = bits(rows * KEY_HEADS * WIDTH, rows + 3, &VALUES);
        let mut packed = vec![0u16; rows * ROW_STRIDE];
        for row in 0..rows {
            for head in 0..QUERY_HEADS {
                for lane in 0..WIDTH {
                    let src = row * QUERY_HEADS * WIDTH + head * WIDTH + lane;
                    let dst = row * ROW_STRIDE + head * QUERY_HEAD_STRIDE + lane;
                    packed[dst] = query_bits[src];
                }
            }
            for head in 0..KEY_HEADS {
                for lane in 0..WIDTH {
                    let src = row * KEY_HEADS * WIDTH + head * WIDTH + lane;
                    let dst = row * ROW_STRIDE + KEY_BASE + head * WIDTH + lane;
                    packed[dst] = key_bits[src];
                }
            }
        }
        let packed_handle = client.create_from_slice(&bytes(&packed));
        let gamma_handle = client.create_from_slice(&bytes(&gamma_bits));

        for (label, definition, role, expected) in [
            (
                "query",
                ordinary.query_head_rms(rows, EPSILON)?,
                BufferRole::QueryNormalized,
                source_expected(&client, &query_bits, &gamma_bits, rows * QUERY_HEADS)?,
            ),
            (
                "key",
                ordinary.key_head_rms(rows, EPSILON)?,
                BufferRole::KeyNormalized,
                source_expected(&client, &key_bits, &gamma_bits, rows * KEY_HEADS)?,
            ),
        ] {
            let output = client.empty(expected.len() * 2);
            let status = client.create_from_slice(u32::as_bytes(&[0u32]));
            let gamma_role = if label == "query" {
                BufferRole::QueryRawGamma
            } else {
                BufferRole::KeyRawGamma
            };
            let prepared = definition.prepare_owned(
                &client,
                vec![
                    LaunchBinding::new(
                        BufferRole::PackedAttention,
                        Access::Read,
                        packed_handle.clone(),
                    ),
                    LaunchBinding::new(gamma_role, Access::Read, gamma_handle.clone()),
                    LaunchBinding::new(role, Access::Write, output.clone()),
                ],
                LaunchBinding::new(BufferRole::Status, Access::ReadWrite, status.clone()),
            )?;
            prepared.enqueue_owned(&client, &status)?;
            let flags = parse_status(
                &client.read_one(status).map_err(|error| {
                    Error::State(format!("ordinary {label} head RMS status: {error}"))
                })?,
                label,
            )?;
            if flags != 0 {
                return Err(Error::State(format!(
                    "ordinary {label} head RMS status rows={rows}: {flags:#x}"
                )));
            }
            let actual = parse_bf16(
                &client.read_one(output).map_err(|error| {
                    Error::State(format!("ordinary {label} head RMS output: {error}"))
                })?,
                label,
            )?;
            if actual != expected {
                let index = actual
                    .iter()
                    .zip(&expected)
                    .position(|(a, b)| a != b)
                    .unwrap_or(usize::MAX);
                return Err(Error::State(format!(
                    "ordinary/source {label} head RMS mismatch rows={rows} index={index} ordinary={:#06x} source={:#06x}",
                    actual.get(index).copied().unwrap_or_default(),
                    expected.get(index).copied().unwrap_or_default()
                )));
            }
        }
        println!("ordinary-source-head-rms rows={rows} PASS");
    }
    Ok(())
}

#[cfg(all(
    test,
    feature = "cuda-cublas",
    feature = "qwen-source-qualification",
    target_os = "linux"
))]
#[test]
#[ignore = "actual NVIDIA ordinary/source Qwen query/key head RMS numerical parity"]
fn ordinary_head_rms_matches_source_reference_on_nvidia() -> zerogravity_core::Result<()> {
    qualify_ordinary_head_rms_source_parity(Device::cuda(0).map_err(|error| {
        zerogravity_core::Error::State(format!("CUDA device 0 unavailable: {error}"))
    })?)
}

#[cfg(all(
    test,
    feature = "cuda-cublas",
    feature = "qwen-source-qualification",
    target_os = "linux"
))]
#[test]
#[ignore = "actual NVIDIA ordinary/source Qwen RMS numerical parity"]
fn ordinary_rms_matches_source_reference_on_nvidia() -> zerogravity_core::Result<()> {
    qualify_ordinary_rms_source_parity(Device::cuda(0).map_err(|error| {
        zerogravity_core::Error::State(format!("CUDA device 0 unavailable: {error}"))
    })?)
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
fn qualify_ordinary_nonlinear_source_parity(device: Device) -> zerogravity_core::Result<()> {
    use crate::cuda::ordinary_text::{Access, BufferRole, Geometry, LaunchBinding, Program};
    use zerogravity_core::{Error, Result};

    const INTERMEDIATE: usize = 3584;
    const QUERY_WIDTH: usize = 2048;
    const ATTN_STRIDE: usize = 5120;
    const HEADS: usize = 8;
    const HEAD_STRIDE: usize = 512;
    const GATE_OFFSET: usize = 256;
    const PATTERNS: [u16; 13] = [
        0xc100, 0xc080, 0xc000, 0xbf80, 0xbf00, 0xbe80, 0x0000, 0x3e80, 0x3f00, 0x3f80, 0x4000,
        0x4080, 0x4100,
    ];
    const UP_PATTERNS: [u16; 8] = [
        0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x4000, 0xc000,
    ];

    fn bits(count: usize, salt: usize, patterns: &[u16]) -> Vec<u16> {
        (0..count)
            .map(|i| patterns[(i * 29 + salt * 7) % patterns.len()])
            .collect()
    }
    fn bytes(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn widen(values: &[u16]) -> Vec<f32> {
        values
            .iter()
            .map(|&v| f32::from_bits(u32::from(v) << 16))
            .collect()
    }
    fn parse_bf16(bytes: &[u8], label: &str) -> Result<Vec<u16>> {
        if !bytes.len().is_multiple_of(2) {
            return Err(Error::State(format!("{label} BF16 alignment")));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|w| u16::from_le_bytes([w[0], w[1]]))
            .collect())
    }
    fn parse_status(bytes: &[u8], label: &str) -> Result<u32> {
        let word: [u8; 4] = bytes
            .try_into()
            .map_err(|_| Error::State(format!("{label} status extent")))?;
        Ok(u32::from_le_bytes(word))
    }
    fn grid(elements: usize) -> CubeCount {
        CubeCount::Static((elements as u32).div_ceil(32), 1, 1)
    }
    fn arg(handle: cubecl::server::Handle, elements: usize) -> BufferArg {
        // SAFETY: qualification fixtures allocate exact typed extents.
        unsafe { BufferArg::from_raw_parts(handle, elements) }
    }

    let client = device.client();
    let program = Program::new(Geometry::qwen35_08())?;

    for rows in [1usize, 2, 17, 64] {
        // FFN SiLU*up.
        let elements = rows * INTERMEDIATE;
        let gate_bits = bits(elements, rows, &PATTERNS);
        let up_bits = bits(elements, rows + 3, &UP_PATTERNS);
        let mut packed = Vec::with_capacity(rows * INTERMEDIATE * 2);
        for row in 0..rows {
            packed.extend_from_slice(&gate_bits[row * INTERMEDIATE..(row + 1) * INTERMEDIATE]);
            packed.extend_from_slice(&up_bits[row * INTERMEDIATE..(row + 1) * INTERMEDIATE]);
        }
        let ordinary_packed = client.create_from_slice(&bytes(&packed));
        let ordinary_output = client.empty(elements * 2);
        let ordinary_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        let definition = program.silu_up(rows)?;
        let prepared = definition.prepare_owned(
            &client,
            vec![
                LaunchBinding::new(BufferRole::PackedGateUp, Access::Read, ordinary_packed),
                LaunchBinding::new(BufferRole::Output, Access::Write, ordinary_output.clone()),
            ],
            LaunchBinding::new(
                BufferRole::Status,
                Access::ReadWrite,
                ordinary_status.clone(),
            ),
        )?;
        prepared.enqueue_owned(&client, &ordinary_status)?;

        let source_gate = client.create_from_slice(f32::as_bytes(&widen(&gate_bits)));
        let source_up = client.create_from_slice(f32::as_bytes(&widen(&up_bits)));
        let source_output = client.empty(elements * 4);
        let source_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        unsafe {
            source_silu_mul_oracle::launch_unchecked(
                &client,
                grid(elements),
                CubeDim::new_1d(32),
                arg(source_gate, elements),
                arg(source_up, elements),
                arg(source_output.clone(), elements),
                arg(source_status.clone(), 1),
                elements,
            );
        }
        compare_bf16_source_outputs(
            &client,
            &ordinary_output,
            &ordinary_status,
            &source_output,
            &source_status,
            elements,
            &format!("SiLU rows={rows}"),
        )?;

        // Attention sigmoid gate.
        let gate_elements = rows * QUERY_WIDTH;
        let gate_bits = bits(gate_elements, rows + 11, &PATTERNS);
        let mut attention = vec![0u16; rows * ATTN_STRIDE];
        for row in 0..rows {
            for head in 0..HEADS {
                for lane in 0..256usize {
                    let src = row * QUERY_WIDTH + head * 256 + lane;
                    let dst = row * ATTN_STRIDE + head * HEAD_STRIDE + GATE_OFFSET + lane;
                    attention[dst] = gate_bits[src];
                }
            }
        }
        let ordinary_packed = client.create_from_slice(&bytes(&attention));
        let ordinary_output = client.empty(gate_elements * 2);
        let ordinary_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        let definition = program.sigmoid(rows)?;
        let prepared = definition.prepare_owned(
            &client,
            vec![
                LaunchBinding::new(BufferRole::PackedAttention, Access::Read, ordinary_packed),
                LaunchBinding::new(BufferRole::Output, Access::Write, ordinary_output.clone()),
            ],
            LaunchBinding::new(
                BufferRole::Status,
                Access::ReadWrite,
                ordinary_status.clone(),
            ),
        )?;
        prepared.enqueue_owned(&client, &ordinary_status)?;

        let source_input = client.create_from_slice(f32::as_bytes(&widen(&gate_bits)));
        let source_output = client.empty(gate_elements * 4);
        let source_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        unsafe {
            source_sigmoid_oracle::launch_unchecked(
                &client,
                grid(gate_elements),
                CubeDim::new_1d(32),
                arg(source_input, gate_elements),
                arg(source_output.clone(), gate_elements),
                arg(source_status.clone(), 1),
                gate_elements,
            );
        }
        compare_bf16_source_outputs(
            &client,
            &ordinary_output,
            &ordinary_status,
            &source_output,
            &source_status,
            gate_elements,
            &format!("sigmoid rows={rows}"),
        )?;
        println!("ordinary-source-nonlinear rows={rows} PASS");
    }

    fn compare_bf16_source_outputs(
        client: &cubecl::client::Client,
        ordinary_output: &cubecl::server::Handle,
        ordinary_status: &cubecl::server::Handle,
        source_output: &cubecl::server::Handle,
        source_status: &cubecl::server::Handle,
        elements: usize,
        label: &str,
    ) -> Result<()> {
        let ordinary = parse_bf16(
            &client.read_one(ordinary_output.clone()).map_err(|error| {
                Error::State(format!("{label} ordinary output readback: {error}"))
            })?,
            label,
        )?;
        let source_raw = client
            .read_one(source_output.clone())
            .map_err(|error| Error::State(format!("{label} source output readback: {error}")))?;
        let source = f32::from_bytes(&source_raw);
        if ordinary.len() != elements || source.len() != elements {
            return Err(Error::State(format!("{label} output extent differs")));
        }
        let ordinary_flags = parse_status(
            &client.read_one(ordinary_status.clone()).map_err(|error| {
                Error::State(format!("{label} ordinary status readback: {error}"))
            })?,
            label,
        )?;
        let source_flags = parse_status(
            &client.read_one(source_status.clone()).map_err(|error| {
                Error::State(format!("{label} source status readback: {error}"))
            })?,
            label,
        )?;
        if ordinary_flags != 0 || source_flags != 0 {
            return Err(Error::State(format!(
                "{label} status differs: ordinary={ordinary_flags:#x} source={source_flags:#x}"
            )));
        }
        for (index, (&actual, &expected)) in ordinary.iter().zip(source).enumerate() {
            if expected.to_bits() & 0xffff != 0 {
                return Err(Error::State(format!(
                    "{label} source output not BF16-widened index={index} bits={:#010x}",
                    expected.to_bits()
                )));
            }
            let expected = (expected.to_bits() >> 16) as u16;
            if actual != expected {
                return Err(Error::State(format!(
                    "{label} bit mismatch index={index} ordinary={actual:#06x} source={expected:#06x}"
                )));
            }
        }
        Ok(())
    }

    Ok(())
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
#[test]
#[ignore = "actual NVIDIA ordinary/source Qwen SiLU+sigmoid numerical parity"]
fn ordinary_nonlinear_matches_source_reference_on_nvidia() -> zerogravity_core::Result<()> {
    qualify_ordinary_nonlinear_source_parity(Device::cuda(0).map_err(|error| {
        zerogravity_core::Error::State(format!("CUDA device 0 unavailable: {error}"))
    })?)
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
fn qualify_ordinary_multiply_and_kv(device: Device) -> zerogravity_core::Result<()> {
    use crate::cuda::ordinary_text::{Access, BufferRole, Geometry, LaunchBinding, Program, Step};
    use zerogravity_core::{Error, Result};

    const QUERY_WIDTH: usize = 2048;
    const KV_WIDTH: usize = 512;
    const PACKED_WIDTH: usize = 5120;
    const VALUE_OFFSET: usize = 4608;
    const CAPACITY: usize = 512;
    const BASE: usize = 31;
    const SENTINEL: u16 = 0x3c00;
    const INPUTS: [u16; 10] = [
        0x3d80, 0xbd80, 0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x4000, 0xc000,
    ];
    const GATES: [u16; 8] = [
        0x3e00, 0xbe00, 0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80, 0xbf80,
    ];

    fn bits(count: usize, salt: usize, patterns: &[u16]) -> Vec<u16> {
        (0..count)
            .map(|i| patterns[(i * 31 + salt * 7) % patterns.len()])
            .collect()
    }
    fn bytes(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn widen(values: &[u16]) -> Vec<f32> {
        values
            .iter()
            .map(|&v| f32::from_bits(u32::from(v) << 16))
            .collect()
    }
    fn parse_bf16(raw: &[u8], label: &str) -> Result<Vec<u16>> {
        if !raw.len().is_multiple_of(2) {
            return Err(Error::State(format!("{label} BF16 alignment")));
        }
        Ok(raw
            .chunks_exact(2)
            .map(|w| u16::from_le_bytes([w[0], w[1]]))
            .collect())
    }
    fn parse_status(raw: &[u8], label: &str) -> Result<u32> {
        let word: [u8; 4] = raw
            .try_into()
            .map_err(|_| Error::State(format!("{label} status extent")))?;
        Ok(u32::from_le_bytes(word))
    }
    fn grid(elements: usize) -> CubeCount {
        CubeCount::Static((elements as u32).div_ceil(32), 1, 1)
    }
    fn arg(handle: cubecl::server::Handle, elements: usize) -> BufferArg {
        // SAFETY: qualification fixtures allocate exact typed extents.
        unsafe { BufferArg::from_raw_parts(handle, elements) }
    }

    let client = device.client();
    let ordinary = Program::new(Geometry::qwen35_08())?;

    for rows in [1usize, 2, 17, 64] {
        // Source-rounded multiply after separately rounded sigmoid.
        let elements = rows * QUERY_WIDTH;
        let input_bits = bits(elements, rows, &INPUTS);
        let gate_bits = bits(elements, rows + 5, &GATES);
        let ordinary_input = client.create_from_slice(&bytes(&input_bits));
        let ordinary_gate = client.create_from_slice(&bytes(&gate_bits));
        let ordinary_output = client.empty(elements * 2);
        let ordinary_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        let prepared = ordinary.multiply(rows)?.prepare_owned(
            &client,
            vec![
                LaunchBinding::new(BufferRole::Input, Access::Read, ordinary_input),
                LaunchBinding::new(BufferRole::Gate, Access::Read, ordinary_gate),
                LaunchBinding::new(BufferRole::Output, Access::Write, ordinary_output.clone()),
            ],
            LaunchBinding::new(
                BufferRole::Status,
                Access::ReadWrite,
                ordinary_status.clone(),
            ),
        )?;
        prepared.enqueue_owned(&client, &ordinary_status)?;

        let source_input = client.create_from_slice(f32::as_bytes(&widen(&input_bits)));
        let source_gate = client.create_from_slice(f32::as_bytes(&widen(&gate_bits)));
        let source_output = client.empty(elements * 4);
        let source_status = client.create_from_slice(u32::as_bytes(&[0u32]));
        unsafe {
            source_multiply_oracle::launch_unchecked(
                &client,
                grid(elements),
                CubeDim::new_1d(32),
                arg(source_input, elements),
                arg(source_gate, elements),
                arg(source_output.clone(), elements),
                arg(source_status.clone(), 1),
                elements,
            );
        }

        let actual = parse_bf16(
            &client
                .read_one(ordinary_output)
                .map_err(|error| Error::State(format!("ordinary multiply output: {error}")))?,
            "ordinary multiply",
        )?;
        let source_raw = client
            .read_one(source_output)
            .map_err(|error| Error::State(format!("source multiply output: {error}")))?;
        let source = f32::from_bytes(&source_raw);
        if actual.len() != elements || source.len() != elements {
            return Err(Error::State(
                "ordinary/source multiply extent differs".into(),
            ));
        }
        let ordinary_flags = parse_status(
            &client
                .read_one(ordinary_status)
                .map_err(|error| Error::State(format!("ordinary multiply status: {error}")))?,
            "ordinary multiply",
        )?;
        let source_flags = parse_status(
            &client
                .read_one(source_status)
                .map_err(|error| Error::State(format!("source multiply status: {error}")))?,
            "source multiply",
        )?;
        if ordinary_flags != 0 || source_flags != 0 {
            return Err(Error::State(format!(
                "ordinary/source multiply status rows={rows}: ordinary={ordinary_flags:#x} source={source_flags:#x}"
            )));
        }
        for (index, (&a, &b)) in actual.iter().zip(source).enumerate() {
            if b.to_bits() & 0xffff != 0 {
                return Err(Error::State(format!(
                    "source multiply not BF16-widened rows={rows} index={index}"
                )));
            }
            let expected = (b.to_bits() >> 16) as u16;
            if a != expected {
                return Err(Error::State(format!(
                    "ordinary/source multiply mismatch rows={rows} index={index} ordinary={a:#06x} source={expected:#06x}"
                )));
            }
        }

        // KV append must write only [BASE, BASE+rows), preserving all other words.
        let key_bits = bits(rows * KV_WIDTH, rows + 11, &INPUTS);
        let value_bits = bits(rows * KV_WIDTH, rows + 17, &GATES);
        let mut packed = vec![0u16; rows * PACKED_WIDTH];
        for row in 0..rows {
            let start = row * PACKED_WIDTH + VALUE_OFFSET;
            packed[start..start + KV_WIDTH]
                .copy_from_slice(&value_bits[row * KV_WIDTH..(row + 1) * KV_WIDTH]);
        }
        let initial_cache = vec![SENTINEL; CAPACITY * KV_WIDTH];
        let key = client.create_from_slice(&bytes(&key_bits));
        let packed = client.create_from_slice(&bytes(&packed));
        let key_cache = client.create_from_slice(&bytes(&initial_cache));
        let value_cache = client.create_from_slice(&bytes(&initial_cache));
        let status = client.create_from_slice(u32::as_bytes(&[0u32]));
        let prepared = ordinary
            .kv_append(Step::new(rows, BASE, CAPACITY)?)?
            .prepare_owned(
                &client,
                vec![
                    LaunchBinding::new(BufferRole::Key, Access::Read, key),
                    LaunchBinding::new(BufferRole::PackedAttention, Access::Read, packed),
                    LaunchBinding::new(BufferRole::KeyCache, Access::ReadWrite, key_cache.clone()),
                    LaunchBinding::new(
                        BufferRole::ValueCache,
                        Access::ReadWrite,
                        value_cache.clone(),
                    ),
                ],
                LaunchBinding::new(BufferRole::Status, Access::ReadWrite, status.clone()),
            )?;
        prepared.enqueue_owned(&client, &status)?;

        let key_actual = parse_bf16(
            &client
                .read_one(key_cache)
                .map_err(|error| Error::State(format!("ordinary key cache: {error}")))?,
            "ordinary key cache",
        )?;
        let value_actual = parse_bf16(
            &client
                .read_one(value_cache)
                .map_err(|error| Error::State(format!("ordinary value cache: {error}")))?,
            "ordinary value cache",
        )?;
        let flags = parse_status(
            &client
                .read_one(status)
                .map_err(|error| Error::State(format!("ordinary KV status: {error}")))?,
            "ordinary KV",
        )?;
        if flags != 0 {
            return Err(Error::State(format!(
                "ordinary KV status rows={rows}: {flags:#x}"
            )));
        }
        for token in 0..CAPACITY {
            let target = token >= BASE && token < BASE + rows;
            let row = token.saturating_sub(BASE);
            for lane in 0..KV_WIDTH {
                let index = token * KV_WIDTH + lane;
                let expected_key = if target {
                    key_bits[row * KV_WIDTH + lane]
                } else {
                    SENTINEL
                };
                let expected_value = if target {
                    value_bits[row * KV_WIDTH + lane]
                } else {
                    SENTINEL
                };
                if key_actual[index] != expected_key || value_actual[index] != expected_value {
                    return Err(Error::State(format!(
                        "ordinary KV append mismatch rows={rows} token={token} lane={lane} key={:#06x}/{expected_key:#06x} value={:#06x}/{expected_value:#06x}",
                        key_actual[index], value_actual[index]
                    )));
                }
            }
        }
        println!("ordinary-multiply-kv rows={rows} PASS");
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
#[test]
#[ignore = "actual NVIDIA ordinary Qwen multiply + KV append physical qualification"]
fn ordinary_multiply_and_kv_on_nvidia() -> zerogravity_core::Result<()> {
    qualify_ordinary_multiply_and_kv(Device::cuda(0).map_err(|error| {
        zerogravity_core::Error::State(format!("CUDA device 0 unavailable: {error}"))
    })?)
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
fn qualify_ordinary_embedding(device: Device) -> zerogravity_core::Result<()> {
    use crate::cuda::ordinary_text::{
        Access, BufferRole, Geometry, LaunchBinding, Program, STATUS_OUT_OF_RANGE,
    };
    use cubecl::bytes::Bytes;
    use zerogravity_core::{Error, Result};

    const HIDDEN: usize = 1024;
    const VOCABULARY: usize = 248_320;
    const ROW_BYTES: usize = HIDDEN * 2;
    const TABLE_BYTES: usize = VOCABULARY * ROW_BYTES;
    const TOKENS: [u32; 12] = [0, 1, 47, 48, 49, 50, 511, 1023, 4095, 8191, 16_383, 248_319];
    const PATTERNS: [u16; 12] = [
        0x0000, 0x3d80, 0xbd80, 0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x4000, 0xc000,
        0x4040,
    ];

    fn row_bits(token: u32) -> Vec<u16> {
        (0..HIDDEN)
            .map(|lane| {
                let index = (lane * 17 + token as usize * 13) % PATTERNS.len();
                PATTERNS[index]
            })
            .collect()
    }
    fn bf16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
    fn parse_bf16(raw: &[u8], label: &str) -> Result<Vec<u16>> {
        if !raw.len().is_multiple_of(2) {
            return Err(Error::State(format!("{label} BF16 alignment")));
        }
        Ok(raw
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes([word[0], word[1]]))
            .collect())
    }
    fn parse_status(raw: &[u8], label: &str) -> Result<u32> {
        let word: [u8; 4] = raw
            .try_into()
            .map_err(|_| Error::State(format!("{label} status extent")))?;
        Ok(u32::from_le_bytes(word))
    }
    fn exact_view(
        handle: &cubecl::server::Handle,
        offset: usize,
        bytes: usize,
    ) -> Result<cubecl::server::Handle> {
        let available = usize::try_from(handle.size_in_used())
            .map_err(|_| Error::Limit("embedding table handle extent".into()))?;
        let end = offset
            .checked_add(bytes)
            .filter(|end| *end <= available)
            .ok_or_else(|| Error::State("embedding table row view exceeds handle".into()))?;
        Ok(handle
            .clone()
            .offset_start(offset as u64)
            .offset_end((available - end) as u64))
    }

    let client = device.client();
    let program = Program::new(Geometry::qwen35_08())?;
    let table = client.empty(TABLE_BYTES);

    for &token in &TOKENS {
        let bytes = bf16_bytes(&row_bits(token));
        let offset = usize::try_from(token)
            .map_err(|_| Error::Limit("embedding token index".into()))?
            .checked_mul(ROW_BYTES)
            .ok_or_else(|| Error::Limit("embedding row offset".into()))?;
        let row = exact_view(&table, offset, ROW_BYTES)?;
        client.write(&row, Bytes::from_bytes_vec(bytes));
    }

    for rows in [1usize, 2, 17, 64] {
        let token_ids = (0..rows)
            .map(|row| TOKENS[(row * 5 + rows) % TOKENS.len()])
            .collect::<Vec<_>>();
        let expected = token_ids
            .iter()
            .flat_map(|&token| row_bits(token))
            .collect::<Vec<_>>();
        let token_handle = client.create_from_slice(u32::as_bytes(&token_ids));
        let output = client.empty(rows * ROW_BYTES);
        let status = client.create_from_slice(u32::as_bytes(&[0u32]));
        let prepared = program.embedding(rows)?.prepare_owned(
            &client,
            vec![
                LaunchBinding::new(BufferRole::TokenIds, Access::Read, token_handle),
                LaunchBinding::new(BufferRole::EmbeddingTable, Access::Read, table.clone()),
                LaunchBinding::new(BufferRole::Output, Access::Write, output.clone()),
            ],
            LaunchBinding::new(BufferRole::Status, Access::ReadWrite, status.clone()),
        )?;
        prepared.enqueue_owned(&client, &status)?;

        let actual = parse_bf16(
            &client
                .read_one(output)
                .map_err(|error| Error::State(format!("embedding output readback: {error}")))?,
            "embedding",
        )?;
        let flags = parse_status(
            &client
                .read_one(status)
                .map_err(|error| Error::State(format!("embedding status readback: {error}")))?,
            "embedding",
        )?;
        if flags != 0 || actual != expected {
            let mismatch = actual
                .iter()
                .zip(&expected)
                .position(|(actual, expected)| actual != expected);
            return Err(Error::State(format!(
                "ordinary embedding rows={rows} differs: status={flags:#x} mismatch={mismatch:?}"
            )));
        }
        println!("ordinary-embedding rows={rows} PASS");
    }

    let invalid_token = [VOCABULARY as u32];
    let token_handle = client.create_from_slice(u32::as_bytes(&invalid_token));
    let output = client.create_from_slice(&vec![0xffu8; ROW_BYTES]);
    let status = client.create_from_slice(u32::as_bytes(&[0u32]));
    let prepared = program.embedding(1)?.prepare_owned(
        &client,
        vec![
            LaunchBinding::new(BufferRole::TokenIds, Access::Read, token_handle),
            LaunchBinding::new(BufferRole::EmbeddingTable, Access::Read, table),
            LaunchBinding::new(BufferRole::Output, Access::Write, output.clone()),
        ],
        LaunchBinding::new(BufferRole::Status, Access::ReadWrite, status.clone()),
    )?;
    prepared.enqueue_owned(&client, &status)?;
    let invalid_output = parse_bf16(
        &client
            .read_one(output)
            .map_err(|error| Error::State(format!("invalid embedding output: {error}")))?,
        "invalid embedding",
    )?;
    let flags = parse_status(
        &client
            .read_one(status)
            .map_err(|error| Error::State(format!("invalid embedding status: {error}")))?,
        "invalid embedding",
    )?;
    if flags != STATUS_OUT_OF_RANGE || invalid_output.iter().any(|value| *value != 0) {
        return Err(Error::State(format!(
            "ordinary invalid embedding differs: status={flags:#x} nonzero_output={}",
            invalid_output.iter().any(|value| *value != 0)
        )));
    }
    println!("ordinary-embedding invalid-token PASS");
    Ok(())
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
#[test]
#[ignore = "actual NVIDIA ordinary Qwen embedding gather/bounds qualification"]
fn ordinary_embedding_on_nvidia() -> zerogravity_core::Result<()> {
    qualify_ordinary_embedding(Device::cuda(0).map_err(|error| {
        zerogravity_core::Error::State(format!("CUDA device 0 unavailable: {error}"))
    })?)
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
async fn qualify_ordinary_owned_blas_identity(
    device: Device,
    cublas: &std::path::Path,
    cublas_lt: &std::path::Path,
) -> zerogravity_core::Result<()> {
    use crate::{
        resident_leases::ResidentService,
        resident_program::ordinary_cuda::{OrdinaryCudaBlas, OrdinaryCudaProviderPaths},
    };
    use cubecl::cuda::Bf16GemmOperation;
    use zerogravity_core::{
        Error,
        accelerator::SourceIdentity,
        device_events::{BufferRole, DeviceAccess},
        resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner},
    };

    const MIB: usize = 1024 * 1024;
    const WIDTH: usize = 64;
    const MAX_ROWS: usize = 64;
    const BF16_ONE: u16 = 0x3f80;
    const VALUES: [u16; 4] = [0x3f00, 0x3f80, 0x4000, 0xbf80];

    let source = SourceIdentity::for_new_allocation()?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source,
        ResourceOwner::new("ordinary-blas-identity", "isolated-qualification")?,
        256 * MIB,
        (512 * MIB) as u64,
    )?;

    let mut weight_bytes = Vec::with_capacity(WIDTH * WIDTH * 2);
    for row in 0..WIDTH {
        for column in 0..WIDTH {
            weight_bytes
                .extend_from_slice(&(if row == column { BF16_ONE } else { 0u16 }).to_le_bytes());
        }
    }
    let mut input_bytes = Vec::with_capacity(MAX_ROWS * WIDTH * 2);
    for row in 0..MAX_ROWS {
        for column in 0..WIDTH {
            input_bytes
                .extend_from_slice(&VALUES[(row * WIDTH + column) % VALUES.len()].to_le_bytes());
        }
    }
    let output_bytes = input_bytes.len();
    let init_bytes = weight_bytes
        .len()
        .checked_add(input_bytes.len())
        .and_then(|bytes| bytes.checked_add(output_bytes))
        .ok_or_else(|| Error::Limit("ordinary BLAS identity initialization bytes".into()))?;

    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: init_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let weights = init.upload_bytes(&weight_bytes, BufferRole::ImmutableWeights)?;
    let input = init.upload_bytes(&input_bytes, BufferRole::Activation)?;
    let output = init.allocate_empty(output_bytes, BufferRole::Output)?;
    drop(init.complete_sync().await?);
    slot.finish()?;

    let paths = OrdinaryCudaProviderPaths::new(cublas, cublas_lt)?;
    let blas = OrdinaryCudaBlas::prepare(service.clone(), 0, &paths).await?;

    for rows in [1usize, 2, 17, 64] {
        let operation = Bf16GemmOperation::reference(WIDTH, rows, WIDTH)
            .map_err(|error| Error::State(format!("ordinary BLAS identity operation: {error}")))?;
        let expected_bytes = rows * WIDTH * 2;
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: expected_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut accesses = vec![
            (weights.clone(), DeviceAccess::Read),
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ];
        accesses.extend(blas.execution_accesses());
        let mut flight = service.submit(0, &accesses)?;
        blas.enqueue(&mut flight, operation, &weights, &input, 0, &output, 0)?;
        let expected = &input_bytes[..expected_bytes];
        let (ready, actual) = flight
            .complete_with_readback(
                std::slice::from_ref(&output),
                &[output.readback(0, expected_bytes)?],
                |read| {
                    if read.len() != 1 || &read[0][..] != expected {
                        return Err(Error::State(format!(
                            "ordinary owned cuBLAS identity GEMM differs for rows={rows}"
                        )));
                    }
                    Ok(())
                },
            )
            .await?;
        drop(ready);
        slot.finish()?;
        if actual.len() != 1 || &actual[0][..] != expected {
            return Err(Error::State(format!(
                "ordinary owned cuBLAS identity readback differs for rows={rows}"
            )));
        }
        println!("ordinary-owned-cublas-identity rows={rows} PASS");
    }

    blas.retire(0).await?;
    drop(blas);
    drop((weights, input, output));
    service
        .client()
        .memory_cleanup()
        .map_err(|error| Error::State(format!("ordinary BLAS identity cleanup: {error}")))?;
    if service.resident_bytes()? != 0 || service.model_events()? != 0 {
        return Err(Error::State(
            "ordinary BLAS identity left resident bytes/events".into(),
        ));
    }
    service.close_model_namespace()?;
    Ok(())
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
#[test]
#[ignore = "isolated NVIDIA ordinary owned-cuBLAS dynamic-row numerical/lifecycle qualification"]
fn ordinary_owned_blas_identity_dynamic_rows_on_nvidia() -> zerogravity_core::Result<()> {
    let cublas = std::env::var_os("ZG_QWEN35_CUBLAS")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| zerogravity_core::Error::State("ZG_QWEN35_CUBLAS required".into()))?;
    let cublas_lt = std::env::var_os("ZG_QWEN35_CUBLAS_LT")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| zerogravity_core::Error::State("ZG_QWEN35_CUBLAS_LT required".into()))?;
    let device = Device::cuda(0).map_err(|error| {
        zerogravity_core::Error::State(format!("CUDA device 0 unavailable: {error}"))
    })?;
    futures_lite::future::block_on(qualify_ordinary_owned_blas_identity(
        device, &cublas, &cublas_lt,
    ))
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
fn qualify_ordinary_rope_source_parity(device: Device) -> zerogravity_core::Result<()> {
    use crate::resident_program::ordinary_cuda::rotary::enqueue_source_exact_raw;
    use zerogravity_core::{Error, Result};

    const QUERY_ELEMENTS: usize = 8 * 256;
    const KEY_ELEMENTS: usize = 2 * 256;
    const ROTARY: usize = 64;
    const HALF: usize = ROTARY / 2;
    const POSITIONS: [usize; 14] = [
        0, 1, 47, 48, 49, 50, 511, 512, 1023, 2047, 4095, 8191, 16_383, 32_767,
    ];
    const BF16_PATTERNS: [u16; 16] = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080, 0x3d80, 0xbd80, 0x3e80, 0xbe80, 0x3f00,
        0xbf00, 0x3f80, 0xbf80, 0x4000, 0xc000,
    ];

    fn input_bits(elements: usize, salt: usize) -> Vec<u16> {
        let mut state = 0x9e37_79b9u32 ^ (salt as u32).wrapping_mul(0x85eb_ca6b);
        (0..elements)
            .map(|index| {
                if index < BF16_PATTERNS.len() {
                    return BF16_PATTERNS[(index + salt) % BF16_PATTERNS.len()];
                }
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let sign = ((state >> 31) as u16) << 15;
                // Moderate finite exponent range exercises nontrivial mantissas
                // without making the oracle fail merely from arithmetic overflow.
                let exponent = (0x70u16 + ((state >> 23) as u16 & 0x1f)) << 7;
                let mantissa = (state as u16) & 0x7f;
                sign | exponent | mantissa
            })
            .collect()
    }
    fn bf16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
    fn widen(values: &[u16]) -> Vec<f32> {
        values
            .iter()
            .map(|&value| f32::from_bits(u32::from(value) << 16))
            .collect()
    }
    fn parse_u16(bytes: &[u8], label: &str) -> Result<Vec<u16>> {
        if !bytes.len().is_multiple_of(2) {
            return Err(Error::State(format!("{label} BF16 readback alignment")));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes([word[0], word[1]]))
            .collect())
    }
    fn parse_u32(bytes: &[u8], label: &str) -> Result<u32> {
        let word: [u8; 4] = bytes
            .try_into()
            .map_err(|_| Error::State(format!("{label} status readback extent")))?;
        Ok(u32::from_le_bytes(word))
    }
    fn grid(elements: usize) -> CubeCount {
        CubeCount::Static((elements as u32).div_ceil(32), 1, 1)
    }
    fn arg(handle: cubecl::server::Handle, elements: usize) -> BufferArg {
        // SAFETY: every fixture below creates an exact typed byte allocation and
        // retains the original handle through synchronous readback.
        unsafe { BufferArg::from_raw_parts(handle, elements) }
    }

    let client = device.client();
    let query_bits = input_bits(QUERY_ELEMENTS, 3);
    let key_bits = input_bits(KEY_ELEMENTS, 11);
    let source_query = widen(&query_bits);
    let source_key = widen(&key_bits);

    let mut frequencies = Vec::with_capacity(HALF);
    let scale = 10_000_000f32.powf(-2.0 / ROTARY as f32);
    let mut frequency = 1.0f32;
    for _ in 0..HALF {
        frequencies.push(frequency);
        frequency *= scale;
    }

    for position in POSITIONS {
        let ordinary_query = client.create_from_slice(&bf16_bytes(&query_bits));
        let ordinary_key = client.create_from_slice(&bf16_bytes(&key_bits));
        let ordinary_frequency = client.create_from_slice(f32::as_bytes(&frequencies));
        let ordinary_query_f32 = client.empty(QUERY_ELEMENTS * 4);
        let ordinary_key_f32 = client.empty(KEY_ELEMENTS * 4);
        let ordinary_angles = client.empty(HALF * 4);
        let ordinary_trig = client.empty(HALF * 2 * 4);
        let ordinary_tables =
            client.create_from_slice(f64::as_bytes(&crate::source_trig::TRIG_TABLES));
        let ordinary_status = client.create_from_slice(u32::as_bytes(&[0u32]));

        enqueue_source_exact_raw(
            &client,
            &ordinary_query,
            &ordinary_key,
            &ordinary_frequency,
            &ordinary_query_f32,
            &ordinary_key_f32,
            &ordinary_angles,
            &ordinary_trig,
            &ordinary_tables,
            &ordinary_status,
            position,
            1,
        )?;

        let source_query_input = client.create_from_slice(f32::as_bytes(&source_query));
        let source_key_input = client.create_from_slice(f32::as_bytes(&source_key));
        // Explicit caller-owned pass-through for the non-rotary 192 lanes.
        let source_query_output = client.create_from_slice(f32::as_bytes(&source_query));
        let source_key_output = client.create_from_slice(f32::as_bytes(&source_key));
        let source_frequency = client.create_from_slice(f32::as_bytes(&frequencies));
        let source_angles = client.empty(HALF * 4);
        let source_trig = client.empty(HALF * 2 * 4);
        let trig_tables = client.create_from_slice(f64::as_bytes(&crate::source_trig::TRIG_TABLES));
        let source_status = client.create_from_slice(u32::as_bytes(&[0u32]));

        // SAFETY: all source-oracle handles have exact checked fixture extents.
        unsafe {
            crate::source_rotary::kernels::angles::launch_unchecked(
                &client,
                grid(HALF),
                CubeDim::new_1d(32),
                arg(source_frequency, HALF),
                arg(source_angles.clone(), HALF),
                position as u32,
                HALF,
            );
            crate::source_trig::sincos_angles::launch_unchecked(
                &client,
                grid(HALF),
                CubeDim::new_1d(32),
                arg(source_angles, HALF),
                arg(trig_tables, crate::source_trig::TRIG_TABLES.len()),
                arg(source_trig.clone(), HALF * 2),
                HALF,
            );
            crate::source_rotary::kernels::apply::launch_unchecked(
                &client,
                grid((QUERY_ELEMENTS / 256) * HALF),
                CubeDim::new_1d(32),
                arg(source_query_input, QUERY_ELEMENTS),
                arg(source_trig.clone(), HALF * 2),
                arg(source_query_output.clone(), QUERY_ELEMENTS),
                arg(source_status.clone(), 1),
                256usize,
                QUERY_ELEMENTS,
                ROTARY,
            );
            crate::source_rotary::kernels::apply::launch_unchecked(
                &client,
                grid((KEY_ELEMENTS / 256) * HALF),
                CubeDim::new_1d(32),
                arg(source_key_input, KEY_ELEMENTS),
                arg(source_trig, HALF * 2),
                arg(source_key_output.clone(), KEY_ELEMENTS),
                arg(source_status.clone(), 1),
                256usize,
                KEY_ELEMENTS,
                ROTARY,
            );
        }

        let ordinary_q = parse_u16(
            &client
                .read_one(ordinary_query)
                .map_err(|error| Error::State(format!("ordinary RoPE query readback: {error}")))?,
            "ordinary query",
        )?;
        let ordinary_k = parse_u16(
            &client
                .read_one(ordinary_key)
                .map_err(|error| Error::State(format!("ordinary RoPE key readback: {error}")))?,
            "ordinary key",
        )?;
        let source_q_raw = client
            .read_one(source_query_output)
            .map_err(|error| Error::State(format!("source RoPE query readback: {error}")))?;
        let source_k_raw = client
            .read_one(source_key_output)
            .map_err(|error| Error::State(format!("source RoPE key readback: {error}")))?;
        let source_q = f32::from_bytes(&source_q_raw);
        let source_k = f32::from_bytes(&source_k_raw);
        let ordinary_flags = parse_u32(
            &client
                .read_one(ordinary_status)
                .map_err(|error| Error::State(format!("ordinary RoPE status readback: {error}")))?,
            "ordinary",
        )?;
        let source_flags = parse_u32(
            &client
                .read_one(source_status)
                .map_err(|error| Error::State(format!("source RoPE status readback: {error}")))?,
            "source",
        )?;
        if ordinary_flags != 0 || source_flags != 0 {
            return Err(Error::State(format!(
                "RoPE status differs at position {position}: ordinary={ordinary_flags:#x} source={source_flags:#x}"
            )));
        }
        if ordinary_q.len() != QUERY_ELEMENTS
            || ordinary_k.len() != KEY_ELEMENTS
            || source_q.len() != QUERY_ELEMENTS
            || source_k.len() != KEY_ELEMENTS
        {
            return Err(Error::State("RoPE physical output extent differs".into()));
        }
        for (role, ordinary, source) in [
            ("query", ordinary_q.as_slice(), source_q),
            ("key", ordinary_k.as_slice(), source_k),
        ] {
            for (index, (&actual, &expected)) in ordinary.iter().zip(source).enumerate() {
                if expected.to_bits() & 0xffff != 0 {
                    return Err(Error::State(format!(
                        "source RoPE output is not BF16-widened: position={position} role={role} index={index} bits={:#010x}",
                        expected.to_bits()
                    )));
                }
                let expected = (expected.to_bits() >> 16) as u16;
                if actual != expected {
                    return Err(Error::State(format!(
                        "ordinary/source RoPE bit mismatch: position={position} role={role} index={index} ordinary={actual:#06x} source={expected:#06x}"
                    )));
                }
            }
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda-cublas", target_os = "linux"))]
#[test]
#[ignore = "actual NVIDIA source-independent ordinary/source RoPE parity"]
fn ordinary_source_exact_rotary_matches_source_reference_on_nvidia() -> zerogravity_core::Result<()>
{
    qualify_ordinary_rope_source_parity(Device::cuda(0).map_err(|error| {
        zerogravity_core::Error::State(format!("CUDA device 0 unavailable: {error}"))
    })?)
}
