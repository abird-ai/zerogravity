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
