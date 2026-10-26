//! Physical native GGUF Q4_1/Q6_K B2/4/8 CUDA qualification.
#![cfg(feature = "qualification")]

use cubecl::prelude::*;
use zerogravity_artifacts::quant::{QuantKind, f32_to_f16_rne};
use zerogravity_core::accelerator::{MatVecRegion, SourceIdentity};
use zerogravity_core::{Error, Result};

const ROWS: usize = 64;

fn packed_source(kind: QuantKind, columns: usize) -> Result<Vec<u8>> {
    let (width, block_bytes) = match kind {
        QuantKind::Q4_1 => (32, 20),
        QuantKind::Q6K => (256, 210),
        _ => return Err(Error::Unsupported("Q4_1/Q6K source-only qualifier".into())),
    };
    let mut source = Vec::with_capacity(ROWS * (columns / width) * block_bytes);
    for row in 0..ROWS {
        for block in 0..columns / width {
            match kind {
                QuantKind::Q4_1 => {
                    source.extend_from_slice(
                        &f32_to_f16_rne(0.5 + ((row + block) % 5) as f32 * 0.125).to_le_bytes(),
                    );
                    source.extend_from_slice(
                        &f32_to_f16_rne(-0.375 + ((row + block) % 3) as f32 * 0.125).to_le_bytes(),
                    );
                    for at in 0..16 {
                        source.push(((row * 17 + block * 11 + at * 37) % 256) as u8);
                    }
                }
                QuantKind::Q6K => {
                    for at in 0..192 {
                        source.push(((row * 17 + block * 19 + at * 71) % 256) as u8);
                    }
                    for at in 0..16 {
                        source.push((((row + block + at) % 11) as i32 - 5) as i8 as u8);
                    }
                    source.extend_from_slice(
                        &f32_to_f16_rne(0.5 + (row % 5) as f32 * 0.125).to_le_bytes(),
                    );
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(source)
}
pub(super) fn qualify(device: &cubecl::Device) -> Result<()> {
    use crate::source_accumulation::{self, SourceAccumulation};
    let client = device.client();
    source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
    for (kind, cols) in [(QuantKind::Q4_1, 64usize), (QuantKind::Q6K, 256usize)] {
        let packed = packed_source(kind, cols)?;
        let source = SourceIdentity::for_new_allocation()?;
        let blocks = cols / if kind == QuantKind::Q4_1 { 32 } else { 256 };
        let weight = client.create_from_slice(&packed);
        let mut proven = 0usize;
        for batch in [2usize, 4, 8] {
            let mut scales = Vec::<f32>::new();
            let mut offsets = Vec::<f32>::new();
            let mut codes = Vec::<i32>::new();
            let mut originals = Vec::new();
            for request in 0..batch {
                let input = (0..cols)
                    .map(|i| (((i * 47 + request * 13 + 19) % 107) as f32 - 53.0) / 31.0)
                    .collect::<Vec<_>>();
                let prepared = crate::quantization::projection::prepare(MatVecRegion {
                    source_identity: source,
                    tensor_name: "qwen.p12.codec.probe",
                    kind,
                    columns: cols,
                    rows: ROWS,
                    packed_weights: &packed,
                    activation: &input,
                })?;
                scales.extend(&prepared.scales);
                offsets.extend(&prepared.offsets);
                codes.extend(&prepared.activation);
                originals.push(prepared);
            }
            crate::quantization::batch_projection::validate_mixed(
                kind,
                ROWS,
                cols,
                batch,
                crate::quantization::batch_projection::MixedBatchStorage {
                    packed_bytes: packed.len(),
                    scale_words: scales.len(),
                    offset_words: offsets.len(),
                    code_words: codes.len(),
                    output_words: ROWS * batch,
                },
            )?;
            let scale = client.create_from_slice(f32::as_bytes(&scales));
            let offset = client.create_from_slice(f32::as_bytes(if offsets.is_empty() {
                &[0.0]
            } else {
                offsets.as_slice()
            }));
            let activation = client.create_from_slice(i32::as_bytes(&codes));
            let output = client.empty(ROWS * batch * 4);
            // This launch covers all B request lanes with ONE physical
            // source GGUF projection kernel and one immutable weight buffer.
            unsafe {
                match kind {
                    QuantKind::Q4_1 => {
                        crate::quantization::batch_projection::q4_1_batched::launch_unchecked(
                            &client,
                            CubeCount::Static(((ROWS * batch) as u32).div_ceil(32), 1, 1),
                            CubeDim::new_1d(32),
                            BufferArg::from_raw_parts(weight.clone(), packed.len()),
                            BufferArg::from_raw_parts(scale, scales.len()),
                            BufferArg::from_raw_parts(offset, offsets.len()),
                            BufferArg::from_raw_parts(activation, codes.len()),
                            BufferArg::from_raw_parts(output.clone(), ROWS * batch),
                            blocks,
                            cols,
                            ROWS,
                            batch,
                            true,
                        )
                    }
                    QuantKind::Q6K => {
                        crate::quantization::batch_projection::q6_k_batched::launch_unchecked(
                            &client,
                            CubeCount::Static(((ROWS * batch) as u32).div_ceil(32), 1, 1),
                            CubeDim::new_1d(32),
                            BufferArg::from_raw_parts(weight.clone(), packed.len()),
                            BufferArg::from_raw_parts(scale, scales.len()),
                            BufferArg::from_raw_parts(activation, codes.len()),
                            BufferArg::from_raw_parts(output.clone(), ROWS * batch),
                            blocks,
                            cols,
                            ROWS,
                            batch,
                            true,
                        )
                    }
                    _ => unreachable!(),
                }
            }
            let result = client
                .read_one(output)
                .map_err(|e| Error::State(format!("physical source B>1 {kind:?} readback: {e}")))?;
            let result = f32::from_bytes(&result);
            if result.len() != ROWS * batch {
                return Err(Error::State("source mixed batch output incomplete".into()));
            }
            for (request, original) in originals.iter().enumerate() {
                let scales = client.create_from_slice(f32::as_bytes(&original.scales));
                let offsets =
                    client.create_from_slice(f32::as_bytes(if original.offsets.is_empty() {
                        &[0.0]
                    } else {
                        original.offsets.as_slice()
                    }));
                let codes = client.create_from_slice(i32::as_bytes(&original.activation));
                let oracle = client.empty(ROWS * 4);
                unsafe {
                    match kind {
                        QuantKind::Q4_1 => {
                            crate::quantization::q4_1::q4_1_q8_1_gemv::launch_unchecked(
                                &client,
                                CubeCount::Static((ROWS as u32).div_ceil(32), 1, 1),
                                CubeDim::new_1d(32),
                                BufferArg::from_raw_parts(weight.clone(), packed.len()),
                                BufferArg::from_raw_parts(scales, original.scales.len()),
                                BufferArg::from_raw_parts(offsets, original.offsets.len()),
                                BufferArg::from_raw_parts(codes, original.activation.len()),
                                BufferArg::from_raw_parts(oracle.clone(), ROWS),
                                blocks,
                                ROWS,
                                true,
                            )
                        }
                        QuantKind::Q6K => {
                            crate::quantization::q6_k::q6_k_q8_k_gemv::launch_unchecked(
                                &client,
                                CubeCount::Static((ROWS as u32).div_ceil(32), 1, 1),
                                CubeDim::new_1d(32),
                                BufferArg::from_raw_parts(weight.clone(), packed.len()),
                                BufferArg::from_raw_parts(scales, original.scales.len()),
                                BufferArg::from_raw_parts(codes, original.activation.len()),
                                BufferArg::from_raw_parts(oracle.clone(), ROWS),
                                blocks,
                                ROWS,
                                true,
                            )
                        }
                        _ => unreachable!(),
                    }
                }
                let reference = client
                    .read_one(oracle)
                    .map_err(|e| Error::State(format!("accepted source B1 readback: {e}")))?;
                let reference = f32::from_bytes(&reference);
                if reference.len() != ROWS {
                    return Err(Error::State("accepted B1 source output incomplete".into()));
                }
                for (row, (&a, &b)) in result[request * ROWS..(request + 1) * ROWS]
                    .iter()
                    .zip(reference.iter())
                    .enumerate()
                {
                    if a.to_bits() != b.to_bits() {
                        return Err(Error::State(format!(
                            "native {kind:?} B{batch} request {request} row {row} mismatch: \
                            {:08x}!={:08x}",
                            a.to_bits(),
                            b.to_bits()
                        )));
                    }
                }
                proven += ROWS;
            }
            println!(
                "P12_MIXED_BATCH_GPU_PASS B={batch} codec={kind:?} outputs={} \
                 exact_f32_bits=1 fused_projection_launches=1",
                ROWS * batch
            );
        }
        if proven != ROWS * (2 + 4 + 8) {
            return Err(Error::State(
                "physical original mixed batch incomplete".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn qualify_q4_0(device: &cubecl::Device) -> Result<()> {
    use crate::source_accumulation::{self, SourceAccumulation};
    const COLS: usize = 64;
    let source = SourceIdentity::for_new_allocation()?;
    let blocks = COLS / 32;
    let mut original = Vec::with_capacity(ROWS * blocks * 18);
    for row in 0..ROWS {
        for block in 0..blocks {
            original.extend_from_slice(
                &f32_to_f16_rne(0.5 + ((row + block) % 7) as f32 * 0.125).to_le_bytes(),
            );
            for i in 0..16 {
                original.push(((row * 17 + block * 19 + i * 37) % 256) as u8);
            }
        }
    }
    let client = device.client();
    source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
    let mut proven = 0usize;
    for batch in [2usize, 4, 8] {
        let mut scales = Vec::<f32>::new();
        let mut codes = Vec::<i32>::new();
        let mut rows = Vec::new();
        for request in 0..batch {
            let input = (0..COLS)
                .map(|i| (((i * 47 + request * 13 + 19) % 107) as f32 - 53.0) / 31.0)
                .collect::<Vec<_>>();
            let prepared = crate::quantization::q4_0::prepare_source(MatVecRegion {
                source_identity: source,
                tensor_name: "p12.q4_0",
                kind: QuantKind::Q4_0,
                columns: COLS,
                rows: ROWS,
                packed_weights: &original,
                activation: &input,
            })?;
            scales.extend(&prepared.scales);
            codes.extend(&prepared.activations);
            rows.push(prepared);
        }
        let packed = &rows[0].weights;
        for row in &rows {
            if row.weights != *packed {
                return Err(Error::State(
                    "Q4_0 packed source differs across lanes".into(),
                ));
            }
        }
        crate::quantization::batch_projection::validate_q4_0(
            ROWS,
            COLS,
            batch,
            packed.len(),
            scales.len(),
            codes.len(),
            ROWS * batch,
        )?;
        let matrix = client.create_from_slice(packed);
        let scales_gpu = client.create_from_slice(f32::as_bytes(&scales));
        let codes_gpu = client.create_from_slice(i32::as_bytes(&codes));
        let output_gpu = client.empty(ROWS * batch * 4);
        unsafe {
            crate::quantization::batch_projection::q4_0_batched::launch_unchecked(
                &client,
                CubeCount::Static(((ROWS * batch) as u32).div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(matrix.clone(), packed.len()),
                BufferArg::from_raw_parts(scales_gpu, scales.len()),
                BufferArg::from_raw_parts(codes_gpu, codes.len()),
                BufferArg::from_raw_parts(output_gpu.clone(), ROWS * batch),
                blocks,
                COLS,
                ROWS,
                batch,
                true,
            );
        }
        let raw = client
            .read_one(output_gpu)
            .map_err(|e| Error::State(format!("Q4_0 physical B>1 readback: {e}")))?;
        let actual = f32::from_bytes(&raw);
        if actual.len() != ROWS * batch {
            return Err(Error::State("Q4_0 fused output shape".into()));
        }
        for (request, prepared) in rows.iter().enumerate() {
            let scales = client.create_from_slice(f32::as_bytes(&prepared.scales));
            let q8 = client.create_from_slice(i32::as_bytes(&prepared.activations));
            let expected = client.empty(ROWS * 4);
            unsafe {
                crate::quantization::q4_0::source_q4_q8_gemv::launch_unchecked(
                    &client,
                    CubeCount::Static((ROWS as u32).div_ceil(32), 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(matrix.clone(), packed.len()),
                    BufferArg::from_raw_parts(scales, prepared.scales.len()),
                    BufferArg::from_raw_parts(q8, prepared.activations.len()),
                    BufferArg::from_raw_parts(expected.clone(), ROWS),
                    blocks,
                    ROWS,
                    true,
                );
            }
            let reference = client
                .read_one(expected)
                .map_err(|e| Error::State(format!("original Q4_0 B1 readback: {e}")))?;
            let reference = f32::from_bytes(&reference);
            for (line, (&observed, &oracle)) in actual[request * ROWS..(request + 1) * ROWS]
                .iter()
                .zip(reference.iter())
                .enumerate()
            {
                if observed.to_bits() != oracle.to_bits() {
                    return Err(Error::State(format!(
                        "Q4_0 physical B{batch} request {request} row {line}: \
                         {:08x}!={:08x}",
                        observed.to_bits(),
                        oracle.to_bits()
                    )));
                }
            }
            proven += ROWS;
        }
        println!(
            "P12_MIXED_BATCH_GPU_PASS B={batch} codec=Q4_0 outputs={} \
             exact_f32_bits=1 fused_projection_launches=1",
            ROWS * batch
        );
    }
    if proven != ROWS * (2 + 4 + 8) {
        return Err(Error::State("Q4_0 batch coverage incomplete".into()));
    }
    Ok(())
}
