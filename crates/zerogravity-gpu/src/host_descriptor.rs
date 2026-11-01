//! P07-D: source-weight decode-once host descriptor.
//! Original GGUF source weights are parsed into immutable scales/zero points
//! once per SourceIdentity cache miss. Later token preparation touches only
//! live activation plus retained small metadata, never walks/copied weights.
//! Numeric order is source-compatible with P07-A/B.
use std::sync::Arc;
use zerogravity_artifacts::codecs::{f16_to_f32, f32_to_f16_rne};
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::MatVecRegion;
use zerogravity_core::encoding::StoredTensorEncoding;
use zerogravity_core::placement::PreparedTensorLayout;

pub struct StaticSource {
    pub kind: StoredTensorEncoding,
    pub rows: usize,
    pub cols: usize,
    pub blocks: usize,
    pub gpu_bytes: usize,
    pub gpu_elements: usize,
    weight_scales: Arc<[f32]>,
    biases: Arc<[f32]>,
}
pub enum DynamicStep {
    Q4(crate::quantization::q4_0::PreparedQ4Q8),
    Mixed(crate::quantization::projection::Prepared),
}

pub struct DeviceStep {
    pub kind: StoredTensorEncoding,
    pub layout: crate::quantization::activation::ActivationLayout,
    pub input: Vec<f32>,
    pub scales: Arc<[f32]>,
    pub biases: Arc<[f32]>,
}

impl StaticSource {
    pub fn gpu_bytes_for(region: MatVecRegion<'_>) -> Result<usize> {
        if region.rows == 0 || region.columns == 0 {
            return Err(Error::Malformed("source tensor dimensions zero".into()));
        }
        match region.kind {
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0) => {
                if !region.columns.is_multiple_of(32) || !region.rows.is_multiple_of(8) {
                    return Err(Error::Unsupported(
                        "Q4_0 source requires 8-row alignment".into(),
                    ));
                }
                region
                    .rows
                    .checked_mul(region.columns / 2)
                    .ok_or_else(|| Error::Limit("Q4_0 GPU buffer size overflow".into()))
            }
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_1)
            | StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q8_0)
            | StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K)
            | StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32) => {
                let block = if region.kind
                    == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K)
                {
                    256
                } else {
                    32
                };
                if !region.columns.is_multiple_of(block) {
                    return Err(Error::Unsupported(
                        "GPU source vector alignment invalid".into(),
                    ));
                }
                Ok(region.packed_weights.len())
            }
            _ => Err(Error::Unsupported(
                "GPU source encoding not implemented".into(),
            )),
        }
    }

    pub fn from_region(region: MatVecRegion<'_>) -> Result<(Self, Vec<u8>)> {
        let gpu_bytes = Self::gpu_bytes_for(region)?;
        let layout = PreparedTensorLayout::packed_rows(region.kind, region.columns, region.rows)?;
        if layout.byte_len()? != region.packed_weights.len() {
            return Err(Error::Malformed("GPU source weight length mismatch".into()));
        }
        let block_width = if region.kind
            == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K)
        {
            256
        } else {
            32
        };
        let blocks = region.columns / block_width;
        let count = blocks
            .checked_mul(region.rows)
            .ok_or_else(|| Error::Limit("GPU source block count overflow".into()))?;
        let mut scales = Vec::with_capacity(
            if region.kind
                == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32)
            {
                0
            } else {
                count
            },
        );
        let mut biases = Vec::with_capacity(
            if region.kind
                == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_1)
            {
                count
            } else {
                0
            },
        );
        let mut gpu = Vec::with_capacity(gpu_bytes);
        if region.kind == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32)
        {
            for chunk in region.packed_weights.chunks_exact(4) {
                if !f32::from_le_bytes(chunk.try_into().expect("validated F32 chunk")).is_finite() {
                    return Err(Error::Malformed("GPU source F32 weight nonfinite".into()));
                }
            }
            gpu.extend_from_slice(region.packed_weights);
        } else {
            let physical = match region.kind {
                StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0) => {
                    18
                }
                StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_1) => {
                    20
                }
                StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q8_0) => {
                    34
                }
                StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K) => {
                    210
                }
                _ => unreachable!("admitted source format"),
            };
            for chunk in region.packed_weights.chunks_exact(physical) {
                let scale = if region.kind
                    == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K)
                {
                    f16_to_f32(u16::from_le_bytes([chunk[208], chunk[209]]))
                } else {
                    f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
                };
                if !scale.is_finite() {
                    return Err(Error::Malformed("GPU source scale nonfinite".into()));
                }
                scales.push(scale);
                if region.kind
                    == StoredTensorEncoding::Quantized(
                        zerogravity_core::encoding::QuantScheme::Q4_1,
                    )
                {
                    let bias = f16_to_f32(u16::from_le_bytes([chunk[2], chunk[3]]));
                    if !bias.is_finite() {
                        return Err(Error::Malformed("GPU source bias nonfinite".into()));
                    }
                    biases.push(bias);
                }
                if region.kind
                    == StoredTensorEncoding::Quantized(
                        zerogravity_core::encoding::QuantScheme::Q4_0,
                    )
                {
                    gpu.extend_from_slice(&chunk[2..18]);
                } else {
                    gpu.extend_from_slice(chunk);
                }
            }
        }
        if gpu.len() != gpu_bytes
            || scales.len()
                != if region.kind
                    == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32)
                {
                    0
                } else {
                    count
                }
        {
            return Err(Error::State("static GPU source incomplete".into()));
        }
        let elements = if region.kind
            == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32)
        {
            gpu_bytes / 4
        } else {
            gpu_bytes
        };
        Ok((
            Self {
                kind: region.kind,
                rows: region.rows,
                cols: region.columns,
                blocks,
                gpu_bytes,
                gpu_elements: elements,
                weight_scales: scales.into(),
                biases: biases.into(),
            },
            gpu,
        ))
    }

    /// Immutable miss-only metadata for device-side activation preparation.
    pub(crate) fn weight_scales(&self) -> &[f32] {
        &self.weight_scales
    }
    pub(crate) fn biases(&self) -> &[f32] {
        &self.biases
    }

    pub fn dynamic_device(&self, activation: &[f32]) -> Result<DeviceStep> {
        let layout = crate::quantization::activation::ActivationLayout::new(
            self.kind, self.cols, self.rows,
        )?;
        crate::device_activation::DeviceActivation::validate_inputs(
            &layout,
            activation,
            &self.weight_scales,
            &self.biases,
        )?;
        Ok(DeviceStep {
            kind: self.kind,
            layout,
            input: activation.to_vec(),
            scales: self.weight_scales.clone(),
            biases: self.biases.clone(),
        })
    }

    pub fn dynamic(&self, activation: &[f32]) -> Result<DynamicStep> {
        if activation.len() != self.cols || !activation.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed(
                "GPU live activation geometry/finite mismatch".into(),
            ));
        }
        if self.kind == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32) {
            return Ok(DynamicStep::Mixed(
                crate::quantization::projection::Prepared {
                    kind: self.kind,
                    rows: self.rows,
                    cols: self.cols,
                    blocks: self.blocks,
                    #[cfg(any(test, feature = "qualification"))]
                    source: Vec::new(),
                    scales: Vec::new(),
                    offsets: Vec::new(),
                    activation: Vec::new(),
                    #[cfg(any(test, feature = "qualification"))]
                    weights_f32: Vec::new(),
                    input_f32: activation.to_vec(),
                },
            ));
        }
        let width = if self.kind
            == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K)
        {
            256
        } else {
            32
        };
        let mut input_scales = Vec::with_capacity(self.blocks);
        let mut input_sums = Vec::with_capacity(self.blocks);
        let mut q8 = Vec::with_capacity(self.cols);
        for input in activation.chunks_exact(width) {
            if self.kind
                == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K)
            {
                let mut largest = 0f32;
                let mut signed = 0f32;
                for &v in input {
                    if v.abs() > largest {
                        largest = v.abs();
                        signed = v;
                    }
                }
                let inv = if largest == 0.0 { 0.0 } else { -127.0 / signed };
                input_scales.push(if largest == 0.0 { 0.0 } else { 1.0 / inv });
                input_sums.push(0.0);
                for &v in input {
                    q8.push((v * inv).round_ties_even().clamp(-128.0, 127.0) as i32);
                }
            } else {
                let largest = input.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let d = largest / 127.0;
                let inv = if largest == 0.0 { 0.0 } else { 127.0 / largest };
                let mut sum = 0i32;
                for &v in input {
                    let code = (v * inv).round_ties_even().clamp(-128.0, 127.0) as i32;
                    q8.push(code);
                    sum += code;
                }
                input_scales.push(f16_to_f32(f32_to_f16_rne(d)));
                input_sums.push(f16_to_f32(f32_to_f16_rne(sum as f32 * d)));
            }
        }
        let scales = self
            .weight_scales
            .iter()
            .enumerate()
            .map(|(idx, &s)| s * input_scales[idx % self.blocks])
            .collect::<Vec<_>>();
        let offsets = self
            .biases
            .iter()
            .enumerate()
            .map(|(idx, &s)| s * input_sums[idx % self.blocks])
            .collect::<Vec<_>>();
        if scales.iter().chain(offsets.iter()).any(|s| !s.is_finite()) {
            return Err(Error::Malformed(
                "GPU dynamic source scales nonfinite".into(),
            ));
        }
        if self.kind
            == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0)
        {
            Ok(DynamicStep::Q4(crate::quantization::q4_0::PreparedQ4Q8 {
                rows: self.rows,
                blocks_per_row: self.blocks,
                #[cfg(any(test, feature = "qualification"))]
                weights: Vec::new(),
                scales,
                activations: q8,
            }))
        } else {
            Ok(DynamicStep::Mixed(
                crate::quantization::projection::Prepared {
                    kind: self.kind,
                    rows: self.rows,
                    cols: self.cols,
                    blocks: self.blocks,
                    #[cfg(any(test, feature = "qualification"))]
                    source: Vec::new(),
                    scales,
                    offsets,
                    activation: q8,
                    #[cfg(any(test, feature = "qualification"))]
                    weights_f32: Vec::new(),
                    input_f32: Vec::new(),
                },
            ))
        }
    }
}

#[cfg(all(test, feature = "qualification"))]
mod tests {
    use super::*;
    use cubecl::prelude::CubeElement;
    use zerogravity_core::accelerator::{MatVecKernel, RegionExecution};
    use zerogravity_cpu::source::CpuWeights;

    #[test]
    #[ignore = "requires exact pinned Qwen3.5 0.8B and 2B GGUFs"]
    fn all_five_real_tensor_codecs_have_exact_live_preparation() {
        struct Exact;
        impl MatVecKernel for Exact {
            fn try_execute(&self, r: MatVecRegion<'_>) -> Result<RegionExecution> {
                let (static_source, bytes) = StaticSource::from_region(r)?;
                let prepared = static_source.dynamic(r.activation)?;
                match prepared {
                    DynamicStep::Q4(new) => {
                        let old = crate::quantization::q4_0::prepare_source(r)?;
                        assert_eq!(bytes, old.weights);
                        assert!(new.weights.is_empty());
                        assert_eq!(new.scales, old.scales);
                        assert_eq!(new.activations, old.activations);
                    }
                    DynamicStep::Mixed(new) => {
                        let old = crate::quantization::projection::prepare(r)?;
                        let source = if r.kind
                            == StoredTensorEncoding::Scalar(
                                zerogravity_core::encoding::ScalarDType::F32,
                            ) {
                            f32::as_bytes(&old.weights_f32)
                        } else {
                            &old.source
                        };
                        assert_eq!(&bytes, source);
                        assert!(new.source.is_empty() && new.weights_f32.is_empty());
                        assert_eq!(new.scales, old.scales);
                        assert_eq!(new.offsets, old.offsets);
                        assert_eq!(new.activation, old.activation);
                        assert_eq!(new.input_f32, old.input_f32);
                    }
                }
                Ok(RegionExecution::Unsupported)
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("target/models");
        for (model, names) in [
            ("qwen35-0.8b", vec!["blk.0.attn_gate.weight"]),
            (
                "qwen35-2b",
                vec![
                    "blk.0.attn_gate.weight",
                    "blk.0.ffn_down.weight",
                    "blk.0.ssm_out.weight",
                    "blk.11.attn_q.weight",
                    "blk.0.ssm_alpha.weight",
                ],
            ),
        ] {
            let weights = CpuWeights::open(&root.join(model).join("model.gguf")).unwrap();
            for name in names {
                let cols = weights.tensor(name).unwrap().shape[0] as usize;
                for seed in [0usize, 1, 2, 3] {
                    let input = (0..cols)
                        .map(|i| {
                            if seed == 0 {
                                0.0
                            } else {
                                ((i * 31 + seed * 7) % 1031) as f32 / 317.0 - 1.5
                            }
                        })
                        .collect::<Vec<_>>();
                    weights.matvec_with(name, &input, Some(&Exact)).unwrap();
                }
                println!("P07_D_STATIC_DECODE_ONCE_EXACT model={model} tensor={name}");
            }
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use zerogravity_core::{
        accelerator::SourceIdentity,
        encoding::{QuantScheme, ScalarDType},
    };

    #[test]
    fn prepared_source_extent_matches_each_original_packing_without_a_device() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        for (kind, columns, rows, block_bytes, scale_offset) in [
            (
                StoredTensorEncoding::Quantized(QuantScheme::Q4_0),
                32,
                8,
                18,
                0,
            ),
            (
                StoredTensorEncoding::Quantized(QuantScheme::Q4_1),
                32,
                2,
                20,
                0,
            ),
            (
                StoredTensorEncoding::Quantized(QuantScheme::Q8_0),
                32,
                2,
                34,
                0,
            ),
            (
                StoredTensorEncoding::Quantized(QuantScheme::Q6K),
                256,
                2,
                210,
                208,
            ),
        ] {
            let mut packed = vec![0u8; block_bytes * rows];
            for block in packed.chunks_exact_mut(block_bytes) {
                block[scale_offset..scale_offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
            }
            let input = vec![0.0; columns];
            let region = MatVecRegion {
                source_identity: source,
                tensor_name: "layout.weight",
                kind,
                columns,
                rows,
                packed_weights: &packed,
                activation: &input,
            };
            let (descriptor, bytes) = StaticSource::from_region(region).unwrap();
            assert_eq!(descriptor.rows, rows);
            assert_eq!(descriptor.cols, columns);
            assert_eq!(&*descriptor.weight_scales, vec![0.5; rows].as_slice());
            if kind == StoredTensorEncoding::Quantized(QuantScheme::Q4_0) {
                assert_eq!(bytes, vec![0u8; 16 * rows]);
            } else {
                assert_eq!(bytes, packed);
            }
            assert_eq!(descriptor.gpu_bytes, bytes.len());
            assert!(matches!(
                StaticSource::from_region(MatVecRegion {
                    packed_weights: &packed[..packed.len() - 1],
                    ..region
                }),
                Err(Error::Malformed(_))
            ));
        }
        let packed = 1.0f32.to_le_bytes().repeat(64);
        let input = vec![0.0; 32];
        let region = MatVecRegion {
            source_identity: source,
            tensor_name: "scalar.weight",
            kind: StoredTensorEncoding::Scalar(ScalarDType::F32),
            columns: 32,
            rows: 2,
            packed_weights: &packed,
            activation: &input,
        };
        let (descriptor, bytes) = StaticSource::from_region(region).unwrap();
        assert_eq!(bytes, packed);
        assert_eq!(descriptor.gpu_elements, 64);
        assert!(descriptor.weight_scales.is_empty());
    }
    #[test]
    fn known_q4k_pairing_and_layout_do_not_grant_a_gpu_implementation() {
        let kind = StoredTensorEncoding::Quantized(QuantScheme::Q4K);
        assert_eq!(
            zerogravity_core::numerics::ProjectionContract::Source
                .activation(kind)
                .unwrap(),
            zerogravity_core::numerics::ActivationEncoding::GgmlQ8K
        );
        assert_eq!(
            PreparedTensorLayout::packed_rows(kind, 256, 1)
                .unwrap()
                .byte_len()
                .unwrap(),
            144
        );
        let packed = [0u8; 144];
        let input = [0.0; 256];
        let region = MatVecRegion {
            source_identity: SourceIdentity::for_new_allocation().unwrap(),
            tensor_name: "unsupported.weight",
            kind,
            columns: 256,
            rows: 1,
            packed_weights: &packed,
            activation: &input,
        };
        assert!(matches!(
            StaticSource::from_region(region),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            crate::quantization::activation::ActivationLayout::new(kind, 256, 1),
            Err(Error::Unsupported(_))
        ));
    }
}
