//! P07-A: source-correct GGUF Q4_0 × dynamic Q8_0 CubeCL GPU region.
//! This provider is opt-in. A region is completed only after device readback.
//! This is NOT persistent residency or a full-model GPU backend.

use crate::source_accumulation;
#[cfg(feature = "qualification")]
use crate::source_accumulation::SourceAccumulation;
#[cfg(feature = "qualification")]
use cubecl::Device;
use cubecl::prelude::*;
#[cfg(feature = "qualification")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(any(test, feature = "qualification"))]
use zerogravity_artifacts::quant::QuantKind;
#[cfg(any(test, feature = "qualification"))]
use zerogravity_artifacts::quant::{f16_to_f32, f32_to_f16_rne};
#[cfg(any(test, feature = "qualification"))]
use zerogravity_core::Error;
#[cfg(any(test, feature = "qualification"))]
use zerogravity_core::Result;
#[cfg(any(test, feature = "qualification"))]
use zerogravity_core::accelerator::MatVecRegion;
#[cfg(feature = "qualification")]
use zerogravity_core::accelerator::{MatVecKernel, RegionExecution};

pub(crate) struct PreparedQ4Q8 {
    pub(crate) rows: usize,
    pub(crate) blocks_per_row: usize,
    #[cfg(any(test, feature = "qualification"))]
    pub(crate) weights: Vec<u8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) activations: Vec<i32>,
}

/// CPU preparation implements source-exact Q8_0 activation block coding.
/// No output matvec is executed on CPU for a supported GPU region.
#[cfg(any(test, feature = "qualification"))]
pub(crate) fn prepare_source(region: MatVecRegion<'_>) -> Result<PreparedQ4Q8> {
    if region.kind != QuantKind::Q4_0
        || region.columns == 0
        || !region.columns.is_multiple_of(32)
        || region.rows == 0
        || !region.rows.is_multiple_of(8)
    {
        return Err(Error::Unsupported(
            "P07-A requires Q4_0 complete 8-row matrix groups".into(),
        ));
    }
    if region.activation.len() != region.columns || !region.activation.iter().all(|v| v.is_finite())
    {
        return Err(Error::Malformed("P07-A invalid activation".into()));
    }
    let blocks = region.columns / 32;
    let total = region
        .rows
        .checked_mul(blocks)
        .ok_or_else(|| Error::Limit("GPU Q4 block count overflow".into()))?;
    let expected = total
        .checked_mul(18)
        .ok_or_else(|| Error::Limit("GPU Q4 byte size overflow".into()))?;
    if region.packed_weights.len() != expected {
        return Err(Error::Malformed(
            "P07-A invalid packed source Q4 byte length".into(),
        ));
    }
    let mut scales = Vec::with_capacity(total);
    let mut weights = Vec::with_capacity(total * 16);
    for block in region.packed_weights.chunks_exact(18) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        if !scale.is_finite() {
            return Err(Error::Malformed("nonfinite Q4 block scale".into()));
        }
        scales.push(scale);
        weights.extend_from_slice(&block[2..]);
    }
    let mut activations = Vec::with_capacity(region.columns);
    let mut input_scales = Vec::with_capacity(blocks);
    for block in region.activation.chunks_exact(32) {
        let largest = block.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let inv = if largest == 0.0 { 0.0 } else { 127.0 / largest };
        let decoded_scale = f16_to_f32(f32_to_f16_rne(largest / 127.0));
        if !decoded_scale.is_finite() {
            return Err(Error::Malformed("nonfinite Q8_0 input scale".into()));
        }
        input_scales.push(decoded_scale);
        for &x in block {
            activations.push((x * inv).round_ties_even().clamp(-128.0, 127.0) as i32);
        }
    }
    for (i, scale) in scales.iter_mut().enumerate() {
        *scale *= input_scales[i % blocks];
    }
    Ok(PreparedQ4Q8 {
        rows: region.rows,
        blocks_per_row: blocks,
        weights,
        scales,
        activations,
    })
}

#[cube(launch_unchecked)]
pub(crate) fn source_q4_q8_gemv(
    weights: &[u8],
    scales: &[f32],
    q8: &[i32],
    output: &mut [f32],
    #[comptime] blocks_per_row: usize,
    #[comptime] rows: usize,
    #[comptime] wide_accumulate: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let mut total = 0.0f32;
        for block in 0..blocks_per_row {
            let id = row * blocks_per_row + block;
            let mut dot = 0i32;
            for lane in 0..16usize {
                let packed = u32::cast_from(weights[id * 16 + lane]);
                let low = i32::cast_from(packed & 15u32) - 8i32;
                let high = i32::cast_from(packed >> 4u32) - 8i32;
                let index = block * 32 + lane;
                dot += low * q8[index] + high * q8[index + 16];
            }
            total = source_accumulation::source_mul_add(
                f32::cast_from(dot),
                scales[id],
                total,
                wide_accumulate,
            );
        }
        output[row] = total;
    }
}

/// For the first real model integration, dispatch only the configured tensor.
/// All other model regions return Unsupported and retain verified CPU math.
#[cfg(feature = "qualification")]
pub struct CubeQ4Q8Provider {
    device: Device,
    selected_tensor: String,
    max_rows: usize,
    completed: AtomicUsize,
}
#[cfg(feature = "qualification")]
impl CubeQ4Q8Provider {
    pub fn new(device: Device, selected_tensor: impl Into<String>, max_rows: usize) -> Self {
        Self {
            device,
            selected_tensor: selected_tensor.into(),
            max_rows,
            completed: AtomicUsize::new(0),
        }
    }
    pub fn completed_regions(&self) -> usize {
        self.completed.load(Ordering::Relaxed)
    }
}
#[cfg(feature = "qualification")]
impl MatVecKernel for CubeQ4Q8Provider {
    fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
        if region.tensor_name != self.selected_tensor
            || region.kind != QuantKind::Q4_0
            || region.rows > self.max_rows
            || !region.rows.is_multiple_of(8)
        {
            return Ok(RegionExecution::Unsupported);
        }
        let prepared = prepare_source(region)?;
        let client = self.device.client();
        source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
        let weights = client.create_from_slice(prepared.weights.as_slice());
        let scales = client.create_from_slice(f32::as_bytes(&prepared.scales));
        let q8 = client.create_from_slice(i32::as_bytes(&prepared.activations));
        let output = client.empty(prepared.rows * 4);
        // SAFETY: original source block layout and device buffer lengths were
        // validated above. Each GPU thread owns exactly one output row.
        unsafe {
            source_q4_q8_gemv::launch_unchecked(
                &client,
                CubeCount::Static((prepared.rows as u32).div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(weights, prepared.weights.len()),
                BufferArg::from_raw_parts(scales, prepared.scales.len()),
                BufferArg::from_raw_parts(q8, prepared.activations.len()),
                BufferArg::from_raw_parts(output.clone(), prepared.rows),
                prepared.blocks_per_row,
                prepared.rows,
                true,
            );
        }
        // read_one waits for completed, host-visible GPU output.
        let raw = client
            .read_one(output)
            .map_err(|e| Error::State(format!("P07-A GPU completion/readback failed: {e}")))?;
        if raw.len() != prepared.rows * 4 {
            return Err(Error::State(
                "P07-A GPU returned an incomplete buffer".into(),
            ));
        }
        let values = raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("validated GPU byte size")))
            .collect::<Vec<_>>();
        if !values.iter().all(|v| v.is_finite()) {
            return Err(Error::State("P07-A GPU returned nonfinite output".into()));
        }
        self.completed.fetch_add(1, Ordering::Relaxed);
        Ok(RegionExecution::Completed(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Host interpretation of exact GPU integer/FMA schedule for offline
    // fixture testing only. Production supported regions execute on GPU.
    #[cfg(feature = "qualification")]
    fn emulate_device_kernel(prepared: &PreparedQ4Q8) -> Vec<f32> {
        let mut output = vec![0f32; prepared.rows];
        for (row, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for block in 0..prepared.blocks_per_row {
                let bid = row * prepared.blocks_per_row + block;
                let mut dot = 0i32;
                for lane in 0..16 {
                    let b = prepared.weights[bid * 16 + lane];
                    let x = block * 32 + lane;
                    dot += (i32::from(b & 15) - 8) * prepared.activations[x];
                    dot += (i32::from(b >> 4) - 8) * prepared.activations[x + 16];
                }
                sum = (dot as f32).mul_add(prepared.scales[bid], sum);
            }
            *out = sum;
        }
        output
    }

    #[test]
    #[cfg(feature = "qualification")]
    #[ignore = "requires locally pinned real Qwen3.5 GGUF checkpoints"]
    fn trained_gguf_q4q8_gpu_preparation_matches_source_cpu_bitwise() {
        use zerogravity_cpu::weights::CpuWeights;
        struct SourceFixtureProvider;
        impl MatVecKernel for SourceFixtureProvider {
            fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
                if region.tensor_name != "blk.0.attn_gate.weight" {
                    return Ok(RegionExecution::Unsupported);
                }
                let p = prepare_source(region)?;
                Ok(RegionExecution::Completed(emulate_device_kernel(&p)))
            }
        }
        for model in ["qwen35-0.8b", "qwen35-2b"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("target/models")
                .join(model)
                .join("model.gguf");
            let weights = CpuWeights::open(&path).expect("source GGUF model");
            let input = weights.embed(11, "token_embd.weight").unwrap();
            let name = "blk.0.attn_gate.weight";
            assert_eq!(weights.tensor(name).unwrap().kind, QuantKind::Q4_0);
            let expected = weights.matvec(name, &input).unwrap();
            let actual = weights
                .matvec_with(name, &input, Some(&SourceFixtureProvider))
                .unwrap();
            assert_eq!(
                actual, expected,
                "{model}: GPU-compatible original-source Q8 coding/reduction differs"
            );
            assert!(actual.len() > 128);
            println!(
                "TRAINED_Q4Q8_SOURCE_HOST_BIT_EXACT model={model} output_rows={}",
                actual.len()
            );
        }
    }

    #[test]
    #[cfg(feature = "qualification")]
    #[ignore = "requires SHA-pinned Qwen3.5 model GGUFs locally"]
    fn trained_q4_q8_completed_region_model_execution_is_state_and_logit_identical() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        use zerogravity_model_qwen35::QwenDenseCpu;

        struct CompleteSourceTestKernel(AtomicUsize);
        impl MatVecKernel for CompleteSourceTestKernel {
            fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
                if region.tensor_name != "blk.0.attn_gate.weight" {
                    return Ok(RegionExecution::Unsupported);
                }
                let prepared = prepare_source(region)?;
                self.0.fetch_add(1, Ordering::Relaxed);
                // An offline mathematical simulation of the CubeCL GPU
                // integer/FMA kernel. NEVER counted as a physical GPU run.
                Ok(RegionExecution::Completed(emulate_device_kernel(&prepared)))
            }
        }

        for model in ["qwen35-0.8b", "qwen35-2b"] {
            let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("target/models")
                .join(model);
            let gguf = directory.join("model.gguf");
            let config = directory.join("config.json");
            let ids = [11u32, 353u32, 1007u32];
            let mut cpu = QwenDenseCpu::open(&gguf, &config).unwrap();
            let expected = cpu.prefill_transaction(&ids).unwrap();

            let hook = Arc::new(CompleteSourceTestKernel(AtomicUsize::new(0)));
            let mut mixed = QwenDenseCpu::open_with_kernel(&gguf, &config, hook.clone()).unwrap();
            let actual = mixed.prefill_transaction(&ids).unwrap();
            assert_eq!(hook.0.load(Ordering::Relaxed), ids.len());
            assert_eq!(
                expected, actual,
                "{model}: host-modeled GPU completed region changed logits"
            );
            assert_eq!(mixed.state_fingerprint(), cpu.state_fingerprint());
            assert_eq!(mixed.position(), cpu.position());
            assert_eq!(mixed.root_id(), cpu.root_id());
            println!(
                "TRAINED_PARTIAL_MODEL_COMPLETED_REGION_PARITY={model} tokens={} gpu_modeled_regions={}",
                ids.len(),
                hook.0.load(Ordering::Relaxed)
            );
        }
    }

    #[test]
    fn source_q8_encoding_checks_layout_and_refuses_bad_inputs() {
        let columns = 64usize;
        let rows = 8usize;
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[2..].fill(0x89);
        let packed = block.repeat(rows * 2);
        let input = [1f32; 64];
        let original = MatVecRegion {
            source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
            tensor_name: "blk.0.attn_gate.weight",
            kind: QuantKind::Q4_0,
            columns,
            rows,
            packed_weights: &packed,
            activation: &input,
        };
        let encoding = prepare_source(original).unwrap();
        assert_eq!(encoding.blocks_per_row, 2);
        assert_eq!(encoding.weights.len(), rows * 2 * 16);
        assert!(encoding.weights.iter().all(|&v| v == 0x89));
        assert!(encoding.activations.iter().all(|&v| v == 127));
        let scale = f16_to_f32(f32_to_f16_rne(1.0 / 127.0));
        assert!(encoding.scales.iter().all(|&v| v == scale));
        assert!(((16 * 127) as f32 * scale * 2.0 - 32.0).abs() < 0.02);
        let mut wrong = original;
        wrong.rows = 9;
        assert!(prepare_source(wrong).is_err());
        wrong = original;
        wrong.kind = QuantKind::Q8_0;
        assert!(prepare_source(wrong).is_err());
        let not_finite = [f32::NAN; 64];
        wrong = original;
        wrong.activation = &not_finite;
        assert!(prepare_source(wrong).is_err());
    }
}
