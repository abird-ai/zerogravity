//! Explicit split-half arithmetic profiles; caller owns the non-rotary tail.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::*;
use cubecl::prelude::*;

// CUDA's fused profile rounds only the final FMA to BF16. The CPU-target
// `apply` below retains its separate BF16 product boundaries.
#[cube(launch_unchecked)]
pub(crate) fn apply_fp32_fma(
    input: &[f32],
    trig: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
    #[comptime] count: usize,
    #[comptime] rotation: usize,
) {
    let i = ABSOLUTE_POS;
    if i < (count / width) * rotation / 2usize {
        let head = i / (rotation / 2usize);
        let pair = i % (rotation / 2usize);
        let first = head * width + pair;
        let second = first + rotation / 2usize;
        let a = input[first];
        let b = input[second];
        let sin = bf16(trig[pair * 2usize]);
        let cos = bf16(trig[pair * 2usize + 1usize]);
        let x = bf16(fma(a, cos, neg(mul(b, sin))));
        let y = bf16(fma(b, cos, mul(a, sin)));
        output[first] = x;
        output[second] = y;
        if !source_bf16(a) || !source_bf16(b) || !finite(x) || !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

/// The same source-exact FP32-FMA rotary arithmetic over independent rows with
/// one trig row per logical batch row. Callers retain session/root authority;
/// positions may be equal today or heterogeneous in future cohorts.
#[cube(launch_unchecked)]
pub(crate) fn apply_fp32_fma_batched(
    input: &[f32],
    trig: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
    #[comptime] heads_per_row: usize,
    #[comptime] rows: usize,
    #[comptime] rotation: usize,
) {
    let i = ABSOLUTE_POS;
    let half = rotation / 2usize;
    let pairs_per_row = heads_per_row * half;
    if i < rows * pairs_per_row {
        let row = i / pairs_per_row;
        let local = i % pairs_per_row;
        let head = local / half;
        let pair = local % half;
        let first = (row * heads_per_row + head) * width + pair;
        let second = first + half;
        let trig_base = (row * half + pair) * 2usize;
        let a = input[first];
        let b = input[second];
        let sin = bf16(trig[trig_base]);
        let cos = bf16(trig[trig_base + 1usize]);
        let x = bf16(fma(a, cos, neg(mul(b, sin))));
        let y = bf16(fma(b, cos, mul(a, sin)));
        output[first] = x;
        output[second] = y;
        if !source_bf16(a) || !source_bf16(b) || !finite(x) || !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn angles(
    frequencies: &[f32],
    output: &mut [f32],
    position: u32,
    #[comptime] half: usize,
) {
    let i = ABSOLUTE_POS;
    if i < half {
        output[i] = mul(frequencies[i], f32::cast_from(position));
    }
}

/// Row-indexed angle generation for shared rotary execution. Frequencies are
/// immutable and repeated logically; only caller-supplied positions vary.
#[cube(launch_unchecked)]
pub(crate) fn angles_batched(
    frequencies: &[f32],
    positions: &[u32],
    output: &mut [f32],
    #[comptime] half: usize,
    #[comptime] rows: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows * half {
        let row = i / half;
        let lane = i % half;
        output[i] = mul(frequencies[lane], f32::cast_from(positions[row]));
    }
}

/// Equal-position specialization of the row-indexed batch form. It preserves
/// the exact scalar multiply while avoiding host/device position transport for
/// aligned decode cohorts.
#[cube(launch_unchecked)]
pub(crate) fn angles_batched_repeated(
    frequencies: &[f32],
    output: &mut [f32],
    position: u32,
    #[comptime] half: usize,
    #[comptime] rows: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows * half {
        let lane = i % half;
        output[i] = mul(frequencies[lane], f32::cast_from(position));
    }
}

#[cube(launch_unchecked)]
pub(crate) fn apply(
    input: &[f32],
    trig: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
    #[comptime] count: usize,
    #[comptime] rotation: usize,
) {
    let i = ABSOLUTE_POS;
    if i < (count / width) * rotation / 2usize {
        let head = i / (rotation / 2usize);
        let pair = i % (rotation / 2usize);
        let first = head * width + pair;
        let second = first + rotation / 2usize;
        let a = input[first];
        let b = input[second];
        let sin = bf16(trig[pair * 2usize]);
        let cos = bf16(trig[pair * 2usize + 1usize]);
        let x = bf16(add(bf16(mul(a, cos)), bf16(mul(neg(b), sin))));
        let y = bf16(add(bf16(mul(b, cos)), bf16(mul(a, sin))));
        output[first] = x;
        output[second] = y;
        if !source_bf16(a) || !source_bf16(b) || !finite(x) || !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cfg(all(
    test,
    feature = "qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
mod batch_qualification {
    use super::*;
    use cubecl::{client::Client, server::Handle};
    use zerogravity_core::{Error, Result};

    const ROWS: usize = 4;
    const HEADS: usize = 8;
    const WIDTH: usize = 256;
    const ROTATION: usize = 64;
    const HALF: usize = ROTATION / 2;

    fn grid(elements: usize) -> CubeCount {
        CubeCount::Static((elements as u32).div_ceil(32), 1, 1)
    }

    fn threads() -> CubeDim {
        CubeDim::new_1d(32)
    }

    fn arg(handle: &Handle, elements: usize) -> BufferArg {
        // SAFETY: qualification constructs exact, private physical buffers.
        unsafe { BufferArg::from_raw_parts(handle.clone(), elements) }
    }

    fn view(handle: &Handle, offset_bytes: usize, bytes: usize) -> Result<Handle> {
        let available = usize::try_from(handle.size_in_used())
            .map_err(|_| Error::Limit("rotary qualification handle extent".into()))?;
        let end = offset_bytes
            .checked_add(bytes)
            .filter(|end| *end <= available)
            .ok_or_else(|| Error::State("rotary qualification view extent".into()))?;
        Ok(handle
            .clone()
            .offset_start(offset_bytes as u64)
            .offset_end((available - end) as u64))
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    fn f64_bytes(values: &[f64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    fn run_case(client: &Client, positions: [u32; ROWS], repeated: bool) -> Result<()> {
        let frequencies = (0..HALF)
            .map(|i| 1.0f32 / (1.0 + i as f32 * 0.125))
            .collect::<Vec<_>>();
        let input = (0..ROWS * HEADS * WIDTH)
            .map(|i| {
                let bf16 = 0x3f00u32 + (i as u32 % 0x40);
                f32::from_bits(bf16 << 16)
            })
            .collect::<Vec<_>>();
        let input_bytes = f32_bytes(&input);
        let batch = client.create_from_slice(&input_bytes);
        let oracle = client.create_from_slice(&input_bytes);
        let frequency = client.create_from_slice(&f32_bytes(&frequencies));
        let position_handle = client.create_from_slice(&u32_bytes(&positions));
        let tables = client.create_from_slice(&f64_bytes(&crate::source_trig::TRIG_TABLES));
        let batch_angles = client.create_from_slice(&[0u8; ROWS * HALF * 4]);
        let batch_trig = client.create_from_slice(&[0u8; ROWS * HALF * 2 * 4]);
        let oracle_angles = client.create_from_slice(&[0u8; HALF * 4]);
        let oracle_trig = client.create_from_slice(&[0u8; HALF * 2 * 4]);
        let batch_status = client.create_from_slice(&0u32.to_ne_bytes());
        let oracle_status = client.create_from_slice(&0u32.to_ne_bytes());

        // SAFETY: every launch below uses exact service-local extents created
        // immediately above; input/output aliases intentionally match the
        // established source-exact in-place rotary contract.
        unsafe {
            if repeated {
                angles_batched_repeated::launch_unchecked(
                    client,
                    grid(ROWS * HALF),
                    threads(),
                    arg(&frequency, HALF),
                    arg(&batch_angles, ROWS * HALF),
                    positions[0],
                    HALF,
                    ROWS,
                );
            } else {
                angles_batched::launch_unchecked(
                    client,
                    grid(ROWS * HALF),
                    threads(),
                    arg(&frequency, HALF),
                    arg(&position_handle, ROWS),
                    arg(&batch_angles, ROWS * HALF),
                    HALF,
                    ROWS,
                );
            }
            crate::source_trig::sincos_angles::launch_unchecked(
                client,
                grid(ROWS * HALF),
                threads(),
                arg(&batch_angles, ROWS * HALF),
                arg(&tables, crate::source_trig::TRIG_TABLES.len()),
                arg(&batch_trig, ROWS * HALF * 2),
                ROWS * HALF,
            );
            apply_fp32_fma_batched::launch_unchecked(
                client,
                grid(ROWS * HEADS * HALF),
                threads(),
                arg(&batch, ROWS * HEADS * WIDTH),
                arg(&batch_trig, ROWS * HALF * 2),
                arg(&batch, ROWS * HEADS * WIDTH),
                arg(&batch_status, 1),
                WIDTH,
                HEADS,
                ROWS,
                ROTATION,
            );
        }

        for (row, &position) in positions.iter().enumerate() {
            let row_handle = view(&oracle, row * HEADS * WIDTH * 4, HEADS * WIDTH * 4)?;
            unsafe {
                angles::launch_unchecked(
                    client,
                    grid(HALF),
                    threads(),
                    arg(&frequency, HALF),
                    arg(&oracle_angles, HALF),
                    position,
                    HALF,
                );
                crate::source_trig::sincos_angles::launch_unchecked(
                    client,
                    grid(HALF),
                    threads(),
                    arg(&oracle_angles, HALF),
                    arg(&tables, crate::source_trig::TRIG_TABLES.len()),
                    arg(&oracle_trig, HALF * 2),
                    HALF,
                );
                apply_fp32_fma::launch_unchecked(
                    client,
                    grid(HEADS * HALF),
                    threads(),
                    arg(&row_handle, HEADS * WIDTH),
                    arg(&oracle_trig, HALF * 2),
                    arg(&row_handle, HEADS * WIDTH),
                    arg(&oracle_status, 1),
                    WIDTH,
                    HEADS * WIDTH,
                    ROTATION,
                );
            }
        }

        let read = |handle: Handle, role: &str| {
            client
                .read_one(handle)
                .map(|bytes| bytes.to_vec())
                .map_err(|e| Error::State(format!("source rotary batch {role} readback: {e}")))
        };
        if read(batch, "batch")? != read(oracle, "oracle")?
            || read(batch_status, "batch status")? != 0u32.to_ne_bytes()
            || read(oracle_status, "oracle status")? != 0u32.to_ne_bytes()
        {
            return Err(Error::State(format!(
                "source rotary batch differs positions={positions:?} repeated={repeated}"
            )));
        }
        eprintln!(
            "SOURCE_ROTARY_CORE_BATCH_PASS positions={positions:?} repeated={repeated} exact=true launches=3 oracle_launches={}",
            ROWS * 3
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires physical NVIDIA CUDA"]
    fn heterogeneous_and_repeated_batches_match_per_row_source_exact() -> Result<()> {
        let diagnostic =
            crate::cuda::test_support::CudaDiagnostic::new("source-rotary-core-batch")?;
        let client = diagnostic.service.client();
        run_case(client, [0, 7, 511, 1_023], false)?;
        run_case(client, [37; ROWS], true)?;
        Ok(())
    }
}
