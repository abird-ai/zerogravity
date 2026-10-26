//! Direct Qwen3.5 Q4_0 projection probe with actual checkpoint tensor bytes.
//!
//! The first 16 output rows of the source model's initial gated projection
//! form a small, closed native quantized GEMV region. This exercises a true
//! trained source-weight computation on GPU, without pretending the whole
//! stateful Qwen model has been migrated from CPU to GPU.

use cubecl::{Device, prelude::*};
use std::path::Path;

#[cube(launch_unchecked)]
fn q4_gemv(
    input: &[f32],
    scales: &[f32],
    values: &[u8],
    output: &mut [f32],
    #[comptime] blocks_per_row: usize,
    #[comptime] rows: usize,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let mut sum = 0.0f32;
        for block in 0..blocks_per_row {
            let bid = row * blocks_per_row + block;
            let scale = scales[bid];
            for lane in 0..16 {
                let value = u32::cast_from(values[bid * 16 + lane]);
                let lo = f32::cast_from(value & 15u32) - 8.0f32;
                let hi = f32::cast_from(value >> 4u32) - 8.0f32;
                let col = block * 32 + lane;
                sum += (lo * scale) * input[col] + (hi * scale) * input[col + 16];
            }
        }
        output[row] = sum;
    }
}

fn parse_u32(bytes: &[u8]) -> Result<u32, String> {
    Ok(u32::from_le_bytes(
        bytes.try_into().map_err(|_| "short u32".to_string())?,
    ))
}

fn parse_f32_vector(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("f32 fixture length mismatch".into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for x in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(
            x.try_into().map_err(|_| "short f32".to_string())?,
        ));
    }
    if !out.iter().all(|x| x.is_finite()) {
        return Err("nonfinite f32 reference fixture".into());
    }
    Ok(out)
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], String> {
    let start = *cursor;
    let end = start.checked_add(len).ok_or("fixture length overflow")?;
    let chunk = bytes.get(start..end).ok_or("truncated fixture payload")?;
    *cursor = end;
    Ok(chunk)
}

pub fn run(device: &Device, file: &Path) -> Result<(), String> {
    let data = std::fs::read(file).map_err(|e| format!("fixture {}: {e}", file.display()))?;
    let mut cursor = 0;
    if take(&data, &mut cursor, 4)? != b"ZQ4P" {
        return Err("incorrect Q4_0 fixture magic".into());
    }
    let cols = parse_u32(take(&data, &mut cursor, 4)?)? as usize;
    let rows = parse_u32(take(&data, &mut cursor, 4)?)? as usize;
    if !matches!(cols, 1024 | 2048) || rows == 0 || rows > 32 {
        return Err("unsupported Qwen3.5 fixture dimensions".into());
    }
    let blocks = cols / 32;
    let scales = parse_f32_vector(take(&data, &mut cursor, rows * blocks * 4)?)?;
    let input = parse_f32_vector(take(&data, &mut cursor, cols * 4)?)?;
    let expected = parse_f32_vector(take(&data, &mut cursor, rows * 4)?)?;
    let quants = take(&data, &mut cursor, rows * blocks * 16)?;
    if cursor != data.len() {
        return Err("trailing fixture payload".into());
    }

    let client = device.client();
    let input_buf = client.create_from_slice(f32::as_bytes(&input));
    let scale_buf = client.create_from_slice(f32::as_bytes(&scales));
    let quant_buf = client.create_from_slice(quants);
    let output_buf = client.empty(rows * std::mem::size_of::<f32>());

    // SAFETY: shape and physical byte lengths validated against the pinned
    // checkpoint-derived format above. Each device lane writes a unique row.
    unsafe {
        q4_gemv::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(input_buf, input.len()),
            BufferArg::from_raw_parts(scale_buf, scales.len()),
            BufferArg::from_raw_parts(quant_buf, quants.len()),
            BufferArg::from_raw_parts(output_buf.clone(), rows),
            blocks,
            rows,
        );
    }

    let result_bytes = client
        .read_one(output_buf)
        .map_err(|e| format!("device/async readback failure: {e}"))?;
    let actual = parse_f32_vector(&result_bytes)?;
    if actual.len() != expected.len() {
        return Err("wrong GPU projection output length".into());
    }
    let mut max_abs_err = 0.0f32;
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        let error = (a - b).abs();
        max_abs_err = max_abs_err.max(error);
        // This source uses F16 block scales but f32 arithmetic and a different
        // legal reduction order. Do not require cross-backend bit-identical
        // accumulators; retain a tight explicit numeric validation bound.
        let tolerance = 2.0e-4 + 1.0e-3 * b.abs();
        if !a.is_finite() || error > tolerance {
            return Err(format!(
                "GPU quantized Q4_0 projection row {i} mismatch: gpu={a}, reference={b}, error={error}, tolerance={tolerance}"
            ));
        }
    }

    println!(
        "PASS Q4_0 real-model projection: device={} rows={rows} cols={cols} max_abs_err={max_abs_err:e}",
        client.name()
    );
    Ok(())
}
