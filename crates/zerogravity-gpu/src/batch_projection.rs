//! First native multi-row F32 source projection kernel.
//!
//! One CUDA/HIP/Vulkan CubeCL dispatch processes B independent activation
//! rows against the SAME immutable matrix. Only a numerical operator:
//! NOT a second tokenizer, model scheduler, SessionAuthority or full Qwen
//! forward. The typed caller must retain original model/source/device
//! buffers and separately commit per-row StateRoots on actual completion.

#[cfg(feature = "qualification")]
use crate::source_accumulation;
use cubecl::prelude::*;
#[cfg(any(feature = "qualification", test))]
use zerogravity_core::{Error, Result};

#[cfg(any(feature = "qualification", test))]
const MAX_ROWS: usize = 8192;
#[cfg(any(feature = "qualification", test))]
const MAX_COLS: usize = 8192;
#[cfg(any(feature = "qualification", test))]
const MAX_BATCH: usize = 8;
#[cfg(any(feature = "qualification", test))]
const MAX_WEIGHT_ELEMENTS: usize = 16 * 1024 * 1024;

/// Validate a contiguous f32 source matrix and explicit [B,C] input,
/// [B,R] output without allowing implicit broadcasts or aliasing.
/// The physical binder also independently validates handle ownership.
#[cfg(any(feature = "qualification", test))]
pub(crate) fn validate(
    rows: usize,
    cols: usize,
    batch: usize,
    weights: usize,
    inputs: usize,
    outputs: usize,
) -> Result<()> {
    if rows == 0
        || rows > MAX_ROWS
        || cols == 0
        || cols > MAX_COLS
        || !cols.is_multiple_of(32)
        || !(1..=MAX_BATCH).contains(&batch)
    {
        return Err(Error::Limit(
            "batched F32 source projection geometry".into(),
        ));
    }
    let weight_count = rows
        .checked_mul(cols)
        .filter(|v| *v <= MAX_WEIGHT_ELEMENTS)
        .ok_or_else(|| Error::Limit("batched F32 matrix byte budget".into()))?;
    let input_count = batch
        .checked_mul(cols)
        .ok_or_else(|| Error::Limit("batched F32 input count overflow".into()))?;
    let output_count = batch
        .checked_mul(rows)
        .ok_or_else(|| Error::Limit("batched F32 output count overflow".into()))?;
    if weights != weight_count
        || inputs != input_count
        || outputs != output_count
        || u32::try_from(output_count).is_err()
    {
        return Err(Error::Malformed(
            "batched F32 projection tensor shape".into(),
        ));
    }
    Ok(())
}

/// Same 32-lane F32 source dot and 4x8 horizontal reduction schedule as
/// the accepted single-row Qwen F32 kernel. Each output [B,R] lane has
/// one and only one writer; the shared matrix is immutable.
#[cube(launch_unchecked)]
pub(crate) fn f32_source_batched_projection(
    weights: &[f32],
    inputs: &[f32],
    output: &mut [f32],
    #[comptime] cols: usize,
    #[comptime] rows: usize,
    #[comptime] batch: usize,
    #[comptime] wide_accumulate: bool,
) {
    let lane_id = ABSOLUTE_POS;
    if lane_id < rows * batch {
        let request = lane_id / rows;
        let row = lane_id % rows;
        output[lane_id] = crate::mixed_ops::source_f32_ggml_dot(
            weights,
            inputs,
            row,
            request * cols,
            cols,
            wide_accumulate,
        );
    }
}

#[cfg(feature = "qualification")]
pub(crate) fn qualify(device: cubecl::Device) -> Result<()> {
    use crate::source_accumulation::SourceAccumulation;

    const ROWS: usize = 64;
    const COLS: usize = 64;
    let client = device.client();
    source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
    let weights = (0..ROWS * COLS)
        .map(|i| (((i * 73 + 13) % 101) as f32 - 50.0) / 37.0)
        .collect::<Vec<_>>();
    let matrix = client.create_from_slice(f32::as_bytes(&weights));
    let mut proven = 0;
    for batch in [2usize, 4usize, 8usize] {
        let input = (0..batch * COLS)
            .map(|i| (((i * 47 + 19) % 107) as f32 - 53.0) / 31.0)
            .collect::<Vec<_>>();
        validate(ROWS, COLS, batch, weights.len(), input.len(), batch * ROWS)?;
        let in_handle = client.create_from_slice(f32::as_bytes(&input));
        let out_handle = client.empty(batch * ROWS * 4);
        // One physical multi-row launch, NOT B separate f32 kernels.
        unsafe {
            f32_source_batched_projection::launch_unchecked(
                &client,
                CubeCount::Static(((batch * ROWS) as u32).div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(matrix.clone(), weights.len()),
                BufferArg::from_raw_parts(in_handle, input.len()),
                BufferArg::from_raw_parts(out_handle.clone(), batch * ROWS),
                COLS,
                ROWS,
                batch,
                true,
            );
        }
        let actual = client
            .read_one(out_handle)
            .map_err(|e| Error::State(format!("batched physical GPU readback: {e}")))?;
        let output = f32::from_bytes(&actual);
        assert_eq!(output.len(), batch * ROWS);
        for request in 0..batch {
            let row_input = client
                .create_from_slice(f32::as_bytes(&input[request * COLS..(request + 1) * COLS]));
            let oracle_handle = client.empty(ROWS * 4);
            unsafe {
                crate::mixed_ops::f32_ggml_dot_gemv::launch_unchecked(
                    &client,
                    CubeCount::Static((ROWS as u32).div_ceil(32), 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(matrix.clone(), weights.len()),
                    BufferArg::from_raw_parts(row_input, COLS),
                    BufferArg::from_raw_parts(oracle_handle.clone(), ROWS),
                    COLS,
                    ROWS,
                    true,
                );
            }
            let reference = client
                .read_one(oracle_handle)
                .map_err(|e| Error::State(format!("single-row source GPU oracle: {e}")))?;
            let one = f32::from_bytes(&reference);
            for (row, (&observed, &expected)) in output[request * ROWS..(request + 1) * ROWS]
                .iter()
                .zip(one)
                .enumerate()
            {
                if observed.to_bits() != expected.to_bits() {
                    return Err(Error::State(format!(
                        "batch B{batch} request {request} row {row} differs from accepted B1 F32 kernel: {:08x}!={:08x}",
                        observed.to_bits(),
                        expected.to_bits()
                    )));
                }
            }
            proven += ROWS;
        }
        println!(
            "P12_F32_BATCH_GPU_PASS B={batch} rows={ROWS} cols={COLS} exact_f32_bits={} fused_projection_launches=1",
            batch * ROWS
        );
    }
    if proven != 896 {
        return Err(Error::State(
            "native multirow projection proof incomplete".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_batch_geometry_refuses_malformed_or_large_rows() {
        assert!(validate(64, 64, 2, 64 * 64, 128, 128).is_ok());
        assert!(validate(64, 64, 4, 64 * 64, 256, 256).is_ok());
        assert!(validate(8192, 8192, 8, 8192 * 8192, 65536, 65536).is_err());
        for (r, c, b, w, x, y) in [
            (0, 64, 2, 0, 128, 0),
            (64, 0, 2, 0, 0, 128),
            (64, 65, 2, 64 * 65, 130, 128),
            (64, 64, 0, 4096, 0, 0),
            (64, 64, 9, 4096, 576, 576),
            (64, 64, 2, 4095, 128, 128),
            (64, 64, 2, 4096, 127, 128),
            (64, 64, 2, 4096, 128, 127),
        ] {
            assert!(
                validate(r, c, b, w, x, y).is_err(),
                "must reject matrix/row B={b}, R={r}, C={c}"
            );
        }
    }
}
