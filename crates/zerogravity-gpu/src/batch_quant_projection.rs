//! Native P12 Q8_0 × dynamic Q8_0 B>1 source projection.
//!
//! One physical CUDA/GPU dispatch processes B independent activation
//! rows against ONE original immutable GGUF Q8_0 matrix. Each row has
//! its own dynamic input codes and composite Q8 scales. The accepted B1
//! kernel and this B>1 kernel call the SAME source dot operation.
//!
//! This is operator qualification only: no second model execution graph,
//! no session state publication or vLLM throughput claim.

use cubecl::prelude::*;
use zerogravity_core::{Error, Result};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 8192;
const MAX_COLUMNS: usize = 8192;
const MAX_WEIGHTS: usize = 32 * 1024 * 1024;

pub(crate) fn validate(
    rows: usize,
    cols: usize,
    batch: usize,
    packed_bytes: usize,
    scale_words: usize,
    input_words: usize,
    output_words: usize,
) -> Result<()> {
    if rows == 0
        || rows > MAX_ROWS
        || cols == 0
        || cols > MAX_COLUMNS
        || !cols.is_multiple_of(32)
        || !(1..=MAX_BATCH).contains(&batch)
    {
        return Err(Error::Limit(
            "native B>1 Q8 source projection geometry".into(),
        ));
    }
    let blocks = cols / 32;
    let packed = rows
        .checked_mul(blocks)
        .and_then(|v| v.checked_mul(34))
        .filter(|&n| n <= MAX_WEIGHTS)
        .ok_or_else(|| Error::Limit("Q8 source matrix budget".into()))?;
    let scale = batch
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(blocks))
        .ok_or_else(|| Error::Limit("Q8 batch scale geometry overflow".into()))?;
    let inputs = batch
        .checked_mul(cols)
        .ok_or_else(|| Error::Limit("Q8 source batch input overflow".into()))?;
    let outputs = batch
        .checked_mul(rows)
        .ok_or_else(|| Error::Limit("Q8 source batch output overflow".into()))?;
    if packed_bytes != packed
        || scale_words != scale
        || input_words != inputs
        || output_words != outputs
        || u32::try_from(outputs).is_err()
    {
        return Err(Error::Malformed(
            "Q8 source batch matrix/input/output shape".into(),
        ));
    }
    Ok(())
}

/// Same source f32 accumulation/reduction as the accepted original B1
/// Q8 × dynamic Q8 GPU kernel. Only the B-specific scale and input
/// row offsets differ; immutable quantized weight layout is unmodified.
#[cube(launch_unchecked)]
pub(crate) fn q8_source_batched_projection(
    packed: &[u8],
    scales: &[f32],
    inputs: &[i32],
    output: &mut [f32],
    #[comptime] blocks: usize,
    #[comptime] cols: usize,
    #[comptime] rows: usize,
    #[comptime] batch: usize,
    #[comptime] wide_accumulate: bool,
) {
    let lane_id = ABSOLUTE_POS;
    if lane_id < rows * batch {
        let request = lane_id / rows;
        let row = lane_id % rows;
        output[lane_id] = crate::mixed_ops::source_q8_0_dot(
            packed,
            scales,
            inputs,
            row,
            request * rows * blocks,
            request * cols,
            blocks,
            wide_accumulate,
        );
    }
}

/// One physical B>1 Q4_1 projection. Shares exact GGUF Q8_1 signed
/// zero-point and F32 8-lane dot with accepted single-session B1.
#[cube(launch_unchecked)]
pub(crate) fn q4_1_batched(
    packed: &[u8],
    scales: &[f32],
    offsets: &[f32],
    q8: &[i32],
    output: &mut [f32],
    #[comptime] blocks: usize,
    #[comptime] cols: usize,
    #[comptime] rows: usize,
    #[comptime] batch: usize,
    #[comptime] wide: bool,
) {
    let lane = ABSOLUTE_POS;
    if lane < rows * batch {
        let request = lane / rows;
        output[lane] = crate::mixed_ops::source_q4_1_dot(
            packed,
            scales,
            offsets,
            q8,
            lane % rows,
            request * rows * blocks,
            request * cols,
            blocks,
            wide,
        );
    }
}

/// One physical B>1 Q6_K signed 6-bit matrix × dynamic Q8_K.
/// Per-row scales/input are private, immutable source Q6_K shared.
#[cube(launch_unchecked)]
pub(crate) fn q6_k_batched(
    packed: &[u8],
    scales: &[f32],
    q8: &[i32],
    output: &mut [f32],
    #[comptime] blocks: usize,
    #[comptime] cols: usize,
    #[comptime] rows: usize,
    #[comptime] batch: usize,
    #[comptime] wide: bool,
) {
    let lane = ABSOLUTE_POS;
    if lane < rows * batch {
        let request = lane / rows;
        output[lane] = crate::mixed_ops::source_q6_k_dot(
            packed,
            scales,
            q8,
            lane % rows,
            request * rows * blocks,
            request * cols,
            blocks,
            wide,
        );
    }
}
/// Typed original GGUF codec admission without any GPU allocation.
/// Q4_0 packed nibble × dynamic Q8_0. Each native batch output
/// reuses the ORIGINAL B1 device dot and its ordered source reduction.
#[cube(launch_unchecked)]
pub(crate) fn q4_0_batched(
    packed: &[u8],
    scales: &[f32],
    q8: &[i32],
    output: &mut [f32],
    #[comptime] blocks: usize,
    #[comptime] cols: usize,
    #[comptime] rows: usize,
    #[comptime] batch: usize,
    #[comptime] wide: bool,
) {
    let lane = ABSOLUTE_POS;
    if lane < rows * batch {
        let req = lane / rows;
        output[lane] = crate::q4_q8::source_q4_q8_dot(
            packed,
            scales,
            q8,
            lane % rows,
            req * rows * blocks,
            req * cols,
            blocks,
            wide,
        );
    }
}
pub(crate) fn validate_q4_0(
    rows: usize,
    cols: usize,
    batch: usize,
    weights: usize,
    scales: usize,
    codes: usize,
    outputs: usize,
) -> Result<()> {
    if rows == 0
        || rows > MAX_ROWS
        || !rows.is_multiple_of(8)
        || cols == 0
        || cols > MAX_COLUMNS
        || !cols.is_multiple_of(32)
        || !(1..=MAX_BATCH).contains(&batch)
    {
        return Err(Error::Limit("native Q4_0 source batch geometry".into()));
    }
    let blocks = cols / 32;
    let packed = rows
        .checked_mul(blocks)
        .and_then(|v| v.checked_mul(16))
        .filter(|&n| n <= MAX_WEIGHTS)
        .ok_or_else(|| Error::Limit("Q4_0 source packed weight budget".into()))?;
    let nscales = batch
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(blocks))
        .ok_or_else(|| Error::Limit("Q4_0 source batch scales".into()))?;
    let ncodes = batch
        .checked_mul(cols)
        .ok_or_else(|| Error::Limit("Q4_0 source codes overflow".into()))?;
    let nout = batch
        .checked_mul(rows)
        .ok_or_else(|| Error::Limit("Q4_0 output overflow".into()))?;
    if packed != weights
        || nscales != scales
        || ncodes != codes
        || nout != outputs
        || u32::try_from(nout).is_err()
    {
        return Err(Error::Malformed("Q4_0 typed batch shape mismatch".into()));
    }
    Ok(())
}

/// All buffers for one original typed GGUF B>1 projection.
#[derive(Clone, Copy)]
pub(crate) struct MixedBatchStorage {
    pub packed_bytes: usize,
    pub scale_words: usize,
    pub offset_words: usize,
    pub code_words: usize,
    pub output_words: usize,
}
pub(crate) fn validate_mixed(
    kind: zerogravity_artifacts::quant::QuantKind,
    rows: usize,
    cols: usize,
    batch: usize,
    storage: MixedBatchStorage,
) -> Result<()> {
    use zerogravity_artifacts::quant::QuantKind;
    let (width, block_bytes, bias) = match kind {
        QuantKind::Q4_1 => (32usize, 20usize, true),
        QuantKind::Q6K => (256usize, 210usize, false),
        _ => {
            return Err(Error::Unsupported(
                "native batch codec must be Q4_1 or Q6K".into(),
            ));
        }
    };
    if rows == 0
        || rows > MAX_ROWS
        || cols == 0
        || cols > MAX_COLUMNS
        || !cols.is_multiple_of(width)
        || !(1..=MAX_BATCH).contains(&batch)
    {
        return Err(Error::Limit("mixed source batch geometry".into()));
    }
    let blocks = cols / width;
    let bytes = rows
        .checked_mul(blocks)
        .and_then(|v| v.checked_mul(block_bytes))
        .filter(|&n| n <= MAX_WEIGHTS)
        .ok_or_else(|| Error::Limit("mixed source weight budget".into()))?;
    let nscale = batch
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(blocks))
        .ok_or_else(|| Error::Limit("mixed batch scale words".into()))?;
    let ncodes = batch
        .checked_mul(cols)
        .ok_or_else(|| Error::Limit("mixed batch codes".into()))?;
    let nout = batch
        .checked_mul(rows)
        .ok_or_else(|| Error::Limit("mixed batch output".into()))?;
    if bytes != storage.packed_bytes
        || nscale != storage.scale_words
        || storage.offset_words != if bias { nscale } else { 0 }
        || storage.code_words != ncodes
        || storage.output_words != nout
        || u32::try_from(nout).is_err()
    {
        return Err(Error::Malformed(
            "mixed batch packed/source/input geometry".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "qualification")]
pub(crate) fn qualify(device: cubecl::Device) -> Result<()> {
    use crate::source_accumulation::{self, SourceAccumulation};
    use zerogravity_artifacts::quant::{QuantKind, f32_to_f16_rne};
    use zerogravity_core::accelerator::{MatVecRegion, SourceIdentity};
    const ROWS: usize = 64;
    const COLS: usize = 64;
    let blocks = COLS / 32;
    let mut packed = Vec::<u8>::with_capacity(ROWS * blocks * 34);
    for row in 0..ROWS {
        for block in 0..blocks {
            let scale = f32_to_f16_rne(0.5 + (row % 7) as f32 * 0.0625 + block as f32 * 0.125);
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..32 {
                let signed = ((i * 37 + row * 11 + block * 7) % 255) as i32 - 127;
                packed.push(signed as i8 as u8);
            }
        }
    }
    let source = SourceIdentity::for_new_allocation()?;
    let client = device.client();
    source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
    let matrix = client.create_from_slice(&packed);
    let mut proven = 0;
    for batch in [2usize, 4, 8] {
        let mut scales = Vec::<f32>::new();
        let mut codes = Vec::<i32>::new();
        let mut rows = Vec::new();
        for request in 0..batch {
            let x = (0..COLS)
                .map(|i| (((i * 47 + request * 13 + 19) % 107) as f32 - 53.0) / 31.0)
                .collect::<Vec<_>>();
            let prepared = crate::mixed_ops::prepare(MatVecRegion {
                source_identity: source,
                tensor_name: "p12.q8.source",
                kind: QuantKind::Q8_0,
                columns: COLS,
                rows: ROWS,
                packed_weights: &packed,
                activation: &x,
            })?;
            scales.extend(&prepared.scales);
            codes.extend(&prepared.activation);
            rows.push(prepared);
        }
        validate(
            ROWS,
            COLS,
            batch,
            packed.len(),
            scales.len(),
            codes.len(),
            ROWS * batch,
        )?;
        let scale_gpu = client.create_from_slice(f32::as_bytes(&scales));
        let codes_gpu = client.create_from_slice(i32::as_bytes(&codes));
        let out_gpu = client.empty(ROWS * batch * 4);
        // Exactly ONE B>1 launch: unique request-lane writer, one matrix.
        unsafe {
            q8_source_batched_projection::launch_unchecked(
                &client,
                CubeCount::Static(((ROWS * batch) as u32).div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(matrix.clone(), packed.len()),
                BufferArg::from_raw_parts(scale_gpu, scales.len()),
                BufferArg::from_raw_parts(codes_gpu, codes.len()),
                BufferArg::from_raw_parts(out_gpu.clone(), ROWS * batch),
                blocks,
                COLS,
                ROWS,
                batch,
                true,
            );
        }
        let result = client
            .read_one(out_gpu)
            .map_err(|e| Error::State(format!("B>1 Q8 physical readback failed: {e}")))?;
        let observed = f32::from_bytes(&result);
        if observed.len() != ROWS * batch {
            return Err(Error::State("Q8 batch output shape incomplete".into()));
        }
        for (request, prepared) in rows.iter().enumerate() {
            let one_scale = client.create_from_slice(f32::as_bytes(&prepared.scales));
            let one_codes = client.create_from_slice(i32::as_bytes(&prepared.activation));
            let one_out = client.empty(ROWS * 4);
            unsafe {
                crate::mixed_ops::q8_0_q8_0_gemv::launch_unchecked(
                    &client,
                    CubeCount::Static((ROWS as u32).div_ceil(32), 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(matrix.clone(), packed.len()),
                    BufferArg::from_raw_parts(one_scale, prepared.scales.len()),
                    BufferArg::from_raw_parts(one_codes, prepared.activation.len()),
                    BufferArg::from_raw_parts(one_out.clone(), ROWS),
                    blocks,
                    ROWS,
                    true,
                );
            }
            let raw = client
                .read_one(one_out)
                .map_err(|e| Error::State(format!("accepted B1 Q8 GPU readback failed: {e}")))?;
            let expected = f32::from_bytes(&raw);
            if expected.len() != ROWS {
                return Err(Error::State("B1 Q8 output truncated".into()));
            }
            for (row, (actual, want)) in observed[request * ROWS..(request + 1) * ROWS]
                .iter()
                .zip(expected)
                .enumerate()
            {
                if actual.to_bits() != want.to_bits() {
                    return Err(Error::State(format!(
                        "P12 B{batch} Q8 source mismatch request={request} row={row} \
                        actual={:08x} reference={:08x}",
                        actual.to_bits(),
                        want.to_bits()
                    )));
                }
            }
            proven += ROWS;
        }
        println!(
            "P12_Q8_BATCH_GPU_PASS B={batch} source_codec=Q8_0 \
             rows={ROWS} cols={COLS} exact_f32_bits={} \
             physical_fused_projection_launches=1 cpu_inference_fallback=0",
            ROWS * batch
        );
    }
    if proven != ROWS * (2 + 4 + 8) {
        return Err(Error::State("Q8 B2/4/8 qualification incomplete".into()));
    }
    crate::batch_quant_qualification::qualify(&device)?;
    crate::batch_quant_qualification::qualify_q4_0(&device)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_q4_and_q6_batch_geometry_refuses_foreign_shapes() {
        use zerogravity_artifacts::quant::QuantKind;
        let good_q4 = MixedBatchStorage {
            packed_bytes: 64 * 2 * 20,
            scale_words: 2 * 64 * 2,
            offset_words: 2 * 64 * 2,
            code_words: 2 * 64,
            output_words: 2 * 64,
        };
        let good_q6 = MixedBatchStorage {
            packed_bytes: 64 * 210,
            scale_words: 8 * 64,
            offset_words: 0,
            code_words: 8 * 256,
            output_words: 8 * 64,
        };
        assert!(validate_mixed(QuantKind::Q4_1, 64, 64, 2, good_q4).is_ok());
        assert!(validate_mixed(QuantKind::Q6K, 64, 256, 8, good_q6).is_ok());
        assert!(validate_mixed(QuantKind::Q4_1, 0, 64, 2, good_q4).is_err());
        assert!(validate_mixed(QuantKind::Q4_1, 64, 65, 2, good_q4).is_err());
        assert!(validate_mixed(QuantKind::Q6K, 64, 256, 9, good_q6).is_err());
        assert!(
            validate_mixed(
                QuantKind::Q4_1,
                64,
                64,
                2,
                MixedBatchStorage {
                    offset_words: 0,
                    ..good_q4
                }
            )
            .is_err()
        );
        assert!(
            validate_mixed(
                QuantKind::Q6K,
                64,
                256,
                8,
                MixedBatchStorage {
                    offset_words: 1,
                    ..good_q6
                }
            )
            .is_err()
        );
        assert!(
            validate_mixed(
                QuantKind::Q4_1,
                64,
                64,
                2,
                MixedBatchStorage {
                    packed_bytes: usize::MAX,
                    ..good_q4
                }
            )
            .is_err()
        );
        assert!(
            validate_mixed(
                QuantKind::Q6K,
                64,
                256,
                8,
                MixedBatchStorage {
                    code_words: 8 * 256 - 1,
                    ..good_q6
                }
            )
            .is_err()
        );
        assert!(validate_q4_0(64, 64, 2, 64 * 2 * 16, 2 * 64 * 2, 2 * 64, 2 * 64).is_ok());
        assert!(validate_q4_0(65, 64, 2, 65 * 2 * 16, 2 * 65 * 2, 2 * 64, 2 * 65).is_err());
        assert!(validate_q4_0(64, 64, 9, 64 * 2 * 16, 9 * 64 * 2, 9 * 64, 9 * 64).is_err());
        assert!(validate_q4_0(64, 64, 2, 64 * 2 * 16 - 1, 2 * 64 * 2, 2 * 64, 2 * 64).is_err());
    }
    #[test]
    fn finite_source_q8_batch_geometry_and_exact_scales() {
        assert!(validate(64, 64, 2, 64 * 2 * 34, 2 * 64 * 2, 2 * 64, 2 * 64).is_ok());
        assert!(validate(64, 64, 4, 64 * 2 * 34, 4 * 64 * 2, 4 * 64, 4 * 64).is_ok());
        for (rows, cols, batch, weights, scales, input, output) in [
            (0, 64, 2, 0, 0, 0, 0),
            (64, 65, 2, 64 * 2 * 34, 2 * 64 * 2, 2 * 65, 2 * 64),
            (64, 64, 0, 64 * 2 * 34, 0, 0, 0),
            (64, 64, 9, 64 * 2 * 34, 9 * 64 * 2, 9 * 64, 9 * 64),
            (64, 64, 2, 64 * 2 * 34 - 1, 2 * 64 * 2, 2 * 64, 2 * 64),
            (64, 64, 2, 64 * 2 * 34, 2 * 64 * 2 - 1, 2 * 64, 2 * 64),
            (64, 64, 2, 64 * 2 * 34, 2 * 64 * 2, 2 * 64 - 1, 2 * 64),
            (64, 64, 2, 64 * 2 * 34, 2 * 64 * 2, 2 * 64, 2 * 64 - 1),
            (
                8192,
                8192,
                8,
                8192 * 256 * 34,
                8 * 8192 * 256,
                8 * 8192,
                8 * 8192,
            ),
        ] {
            assert!(
                validate(rows, cols, batch, weights, scales, input, output).is_err(),
                "invalid B{batch} Q8 projection accepted"
            );
        }
    }
}
