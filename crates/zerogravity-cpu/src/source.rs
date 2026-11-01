//! CPU access and operations over an immutable source owner.

pub mod projection;
mod quantization;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_embedding;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_projection;

use crate::accelerator::{
    EmbeddingRegion, MatVecKernel, MatVecRegion, RegionExecution, SourceIdentity,
    validate_completed,
};
use crate::artifact::TensorInfo;
use crate::codecs::{dequantize, f16_to_f32};
use crate::numerics::torch_cpu::projection::dot_bf16_cpu_avx2;
use crate::{Error, Result};
use quantization::{
    QuantizedActivation, dot_q4_0_repacked_8x8_source_row, dot_row_with_activation,
};
use rayon::prelude::*;
use std::path::Path;
use zerogravity_artifacts::SourceWeights;
use zerogravity_core::encoding::StoredTensorEncoding;
pub(crate) const MAX_MATRIX_ROWS: u64 = 1_000_000;

#[derive(Debug)]
pub struct CpuWeights {
    source: SourceWeights,
    pub architecture: String,
}
impl std::ops::Deref for CpuWeights {
    type Target = SourceWeights;
    fn deref(&self) -> &Self::Target {
        &self.source
    }
}
impl CpuWeights {
    pub fn open(path: &Path) -> Result<Self> {
        let source = SourceWeights::open(path)?;
        let architecture = source.architecture().to_owned();
        Ok(Self {
            source,
            architecture,
        })
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source.source_identity()
    }
    pub fn source_len(&self) -> usize {
        self.source.source_len()
    }
    pub fn contains(&self, key: &str) -> bool {
        self.source.contains(key)
    }
    pub fn tensor(&self, key: &str) -> Result<&TensorInfo> {
        self.source.tensor(key)
    }
    pub fn tensor_bytes(&self, key: &str) -> Result<&[u8]> {
        self.source.tensor_bytes(key)
    }
    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        self.source.vector(name)
    }
    pub fn matrix_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.source.matrix_f32(name)
    }
    pub fn embedding_region<'a>(
        &'a self,
        token: u32,
        name: &'a str,
    ) -> Result<EmbeddingRegion<'a>> {
        self.source.embedding_region(token, name)
    }
    pub fn embed(&self, token: u32, name: &str) -> Result<Vec<f32>> {
        source_reference_embedding(self.embedding_region(token, name)?)
    }
    pub fn matvec_fp32_reference(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        SourceCpu::new(&self.source).matvec_fp32_reference(name, input)
    }
    pub fn matvec(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        SourceCpu::new(&self.source).matvec(name, input)
    }
    pub fn matvec_with(
        &self,
        name: &str,
        input: &[f32],
        provider: Option<&dyn MatVecKernel>,
    ) -> Result<Vec<f32>> {
        matvec_with_source(&self.source, name, input, provider)
    }
}

struct SourceCpu<'a>(&'a SourceWeights);
impl<'a> SourceCpu<'a> {
    fn new(source: &'a SourceWeights) -> Self {
        Self(source)
    }
}
impl std::ops::Deref for SourceCpu<'_> {
    type Target = SourceWeights;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}
impl SourceCpu<'_> {
    /// Pure source-tensor dequantized f32 mathematical diagnostic.
    /// This is NOT the GGUF CPU execution path for quantized tensors:
    /// GGML pairs Q4_0 with Q8_0 activation blocks, Q4_1 with Q8_1, etc.
    /// Used for independently checked FP32 ideal-reference projection probes.
    pub fn matvec_fp32_reference(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.shape.len() != 2 || t.shape[0] != input.len() as u64 || t.shape[1] > MAX_MATRIX_ROWS {
            return Err(Error::Malformed(format!(
                "FP32 reference matvec shape {name}"
            )));
        }
        let bytes = self.bytes(t)?;
        let row_len = t.kind.byte_len(t.shape[0])? as usize;
        let rows = t.shape[1] as usize;
        if rows.checked_mul(row_len) != Some(bytes.len()) {
            return Err(Error::Malformed(
                "FP32 matrix source geometry mismatch".into(),
            ));
        }
        let mut result = vec![0.0f32; rows];
        // Bounded Rayon split: <= MAX_MATRIX_ROWS rows, depth <= 20, no
        // nested Rayon and no per-worker deque growth beyond the splitter.
        crate::runtime::record_generic_rayon();
        result
            .par_iter_mut()
            .zip(bytes.par_chunks_exact(row_len))
            .for_each(|(out, row)| *out = dot_row(t.kind, row, input));
        Ok(result)
    }

    /// Authoritative original Rust source-math implementation, no device.
    pub fn matvec(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        self.matvec_with(name, input, None)
    }

    /// Optionally specialize a fully source-validated closed matvec region.
    /// A Completed result is validated before it can affect model state.
    /// Unsupported alone permits local CPU fallback; a device error does not.
    /// Future providers may cache source weights/device buffers independently.
    pub fn matvec_with(
        &self,
        name: &str,
        input: &[f32],
        provider: Option<&dyn MatVecKernel>,
    ) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.shape.len() != 2 || t.shape[0] != input.len() as u64 || t.shape[1] > MAX_MATRIX_ROWS {
            return Err(Error::Malformed(format!(
                "matvec '{}' mismatch: {:?} vs input {}",
                name,
                t.shape,
                input.len()
            )));
        }
        let rows = t.shape[1] as usize;
        let row_len = t.kind.byte_len(t.shape[0])? as usize;
        let source = self.bytes(t)?;
        if rows.checked_mul(row_len) != Some(source.len()) {
            return Err(Error::Malformed(
                "source matrix byte geometry mismatch".into(),
            ));
        }
        if !input.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed("nonfinite GEMV input".into()));
        }
        if let Some(provider) = provider {
            let request = MatVecRegion {
                source_identity: self.source_identity(),
                tensor_name: name,
                kind: t.kind,
                columns: input.len(),
                rows,
                packed_weights: source,
                activation: input,
            };
            match provider.try_execute(request)? {
                RegionExecution::Unsupported => {}
                RegionExecution::Completed(values) => {
                    validate_completed(request, &values)?;
                    return Ok(values);
                }
            }
        }
        source_reference_projection(MatVecRegion {
            source_identity: self.source_identity(),
            tensor_name: name,
            kind: t.kind,
            columns: input.len(),
            rows,
            packed_weights: source,
            activation: input,
        })
    }
}

pub fn matvec_with_source(
    source: &SourceWeights,
    name: &str,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
) -> Result<Vec<f32>> {
    SourceCpu::new(source).matvec_with(name, input, provider)
}

/// Source row dequantization shared by CPU embedding and physical qualification.
pub fn source_reference_embedding(r: EmbeddingRegion<'_>) -> Result<Vec<f32>> {
    r.validate()?;
    let row = usize::try_from(r.kind.byte_len(r.width as u64)?)
        .map_err(|_| Error::Limit("embedding row size".into()))?;
    let offset = r.token as usize * row;
    let values = dequantize(r.kind, &r.packed_weights[offset..offset + row], r.width)?;
    r.validate_completed(&values)?;
    Ok(values)
}

/// Exact Rust CPU interpretation of a standalone immutable GGUF GEMV region.
///
/// Used both by normal CPU execution and opt-in *physical* accelerator
/// qualification so the latter compares identical runtime activations and
/// source tensor bytes without loading a second model or duplicating the
/// mathematical implementation. Not an idealized dequantized FP32 dot.
pub fn source_reference_projection(region: MatVecRegion<'_>) -> Result<Vec<f32>> {
    source_reference_projection_with_profile(region, SourceProjectionProfile::Source)
}

/// Numerical order is independent of source storage/codec eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceProjectionProfile {
    /// Existing GGUF/source row arithmetic, unchanged.
    Source,
    /// Pinned Torch CPU AVX2 BF16 B1 GEMV, with BF16-exact activations.
    /// FP32 accumulation only; the caller owns the final BF16 cast.
    TorchCpuBf16Avx2,
}

pub fn source_reference_projection_with_profile(
    region: MatVecRegion<'_>,
    profile: SourceProjectionProfile,
) -> Result<Vec<f32>> {
    if region.rows == 0
        || region.rows as u64 > MAX_MATRIX_ROWS
        || region.columns == 0
        || region.activation.len() != region.columns
        || region.tensor_name.is_empty()
        || !region.activation.iter().all(|v| v.is_finite())
    {
        return Err(Error::Malformed(
            "invalid source reference projection region".into(),
        ));
    }
    if profile == SourceProjectionProfile::TorchCpuBf16Avx2 {
        if region.kind
            != StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16)
        {
            return Err(Error::Unsupported(
                "CPU BF16 AVX2 profile requires BF16 source rows".into(),
            ));
        }
        if !region.source_identity.is_bound()
            || region.activation.iter().any(|x| x.to_bits() & 0xffff != 0)
        {
            return Err(Error::Malformed(
                "CPU BF16 AVX2 profile needs bound source and BF16 activations".into(),
            ));
        }
    }
    if region.kind == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4K)
    {
        return Err(Error::Unsupported(
            "GGUF Q4_K x Q8_K physical dot not independently qualified for this CPU runner".into(),
        ));
    }
    let row_bytes = usize::try_from(region.kind.byte_len(region.columns as u64)?)
        .map_err(|_| Error::Limit("source reference row bytes overflow".into()))?;
    if row_bytes.checked_mul(region.rows) != Some(region.packed_weights.len()) {
        return Err(Error::Malformed(
            "source reference projection weight size mismatch".into(),
        ));
    }
    // Prepare source GGML Q8 activation once, not a changed dequantized-F32
    // diagnostic. All dot_row primitives below are shared with CPU Qwen.
    let activation = QuantizedActivation::prepare(region.kind, region.activation)?;
    let mut output = vec![0f32; region.rows];
    // Bounded Rayon split: <= MAX_MATRIX_ROWS rows, depth <= 20, no nested
    // Rayon and no per-worker deque growth beyond the splitter.
    crate::runtime::record_generic_rayon();
    output
        .par_iter_mut()
        .zip(region.packed_weights.par_chunks_exact(row_bytes))
        .for_each(|(dst, row)| {
            *dst = if profile == SourceProjectionProfile::TorchCpuBf16Avx2 {
                dot_bf16_cpu_avx2(row, region.activation)
            } else if region.kind
                == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0)
                && region.rows.is_multiple_of(8)
            {
                match &activation {
                    QuantizedActivation::Small(blocks) => {
                        dot_q4_0_repacked_8x8_source_row(row, blocks)
                    }
                    _ => unreachable!("Q4_0 always prepares Q8_0 blocks"),
                }
            } else {
                dot_row_with_activation(region.kind, row, region.activation, &activation)
            };
        });
    if !output.iter().all(|v| v.is_finite()) {
        return Err(Error::Malformed(format!(
            "nonfinite result from matrix '{}'",
            region.tensor_name
        )));
    }
    Ok(output)
}

/// Inner GEMV reference with direct byte decoding, avoiding a full f32 expansion
/// GGML x86 AVX2+FMA f32 dot numerical reduction, written as portable scalar
/// lanes. Four 8-wide accumulators receive fused multiply-adds in blocks of 32.
/// The two reduction trees and final horizontal adds deliberately preserve
/// the source's FP32 rounding points. A single scalar f32 sum accumulates
/// in a different order; it was the first observable mismatch in Qwen3.5
/// Gated DeltaNet's F32 ssm_alpha projection and caused downstream amplification.
///
/// This is a source/ISA-specific numerical reference, not an optimized SIMD
/// implementation, and is independent of a foreign inference runtime.
fn dot_f32_source_reduction(row: &[u8], x: &[f32]) -> f32 {
    let np = x.len() & !31usize;
    let mut accum = [[0.0f32; 8]; 4];
    for base in (0..np).step_by(32) {
        for (j, lanes) in accum.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let index = base + j * 8 + lane;
                let offset = index * 4;
                let value = f32::from_le_bytes([
                    row[offset],
                    row[offset + 1],
                    row[offset + 2],
                    row[offset + 3],
                ]);
                *sum = value.mul_add(x[index], *sum);
            }
        }
    }
    let mut reduced = [0.0f32; 8];
    for (lane, sum) in reduced.iter_mut().enumerate() {
        *sum = (accum[0][lane] + accum[2][lane]) + (accum[1][lane] + accum[3][lane]);
    }
    // GGML Haswell: low 128 + high 128, two SSE horizontal additions.
    let horizontal = [
        reduced[0] + reduced[4],
        reduced[1] + reduced[5],
        reduced[2] + reduced[6],
        reduced[3] + reduced[7],
    ];
    let mut result = (horizontal[0] + horizontal[1]) + (horizontal[2] + horizontal[3]);
    for (index, &v) in x.iter().enumerate().skip(np) {
        let offset = index * 4;
        let value = f32::from_le_bytes([
            row[offset],
            row[offset + 1],
            row[offset + 2],
            row[offset + 3],
        ]);
        result += value * v;
    }
    result
}

fn dot_row(kind: StoredTensorEncoding, row: &[u8], x: &[f32]) -> f32 {
    match kind {
        StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32) => {
            dot_f32_source_reduction(row, x)
        }
        StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F16)
        | StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16) => row
            .chunks_exact(2)
            .zip(x)
            .map(|(b, &v)| {
                let raw = u16::from_le_bytes([b[0], b[1]]);
                (if kind
                    == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F16)
                {
                    f16_to_f32(raw)
                } else {
                    f32::from_bits(u32::from(raw) << 16)
                }) * v
            })
            .sum(),
        StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q8_0) => {
            let mut sum = 0f32;
            for (block, inp) in row.chunks_exact(34).zip(x.chunks_exact(32)) {
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dot: f32 = block[2..]
                    .iter()
                    .zip(inp)
                    .map(|(&q, &v)| (q as i8 as f32) * v)
                    .sum();
                sum = dot.mul_add(scale, sum);
            }
            sum
        }
        StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0)
        | StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_1) => {
            let len = if kind
                == StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0)
            {
                18
            } else {
                20
            };
            let mut sum = 0f32;
            for (block, inp) in row.chunks_exact(len).zip(x.chunks_exact(32)) {
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let offset = if kind
                    == StoredTensorEncoding::Quantized(
                        zerogravity_core::encoding::QuantScheme::Q4_0,
                    ) {
                    -8.0 * scale
                } else {
                    f16_to_f32(u16::from_le_bytes([block[2], block[3]]))
                };
                let bytes = &block[len - 16..];
                let mut dot = 0f32;
                for i in 0..16 {
                    dot += (f32::from(bytes[i] & 15) * scale + offset) * inp[i];
                    dot += (f32::from(bytes[i] >> 4) * scale + offset) * inp[i + 16];
                }
                sum += dot;
            }
            sum
        }
        StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4K) => {
            let mut sum = 0.0f32;
            for (b, inp) in row.chunks_exact(144).zip(x.chunks_exact(256)) {
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                let dm = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
                let scales = &b[4..16];
                let qs = &b[16..144];
                for group in 0..8 {
                    let (scale, min) = if group < 4 {
                        (scales[group] & 63, scales[group + 4] & 63)
                    } else {
                        (
                            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                        )
                    };
                    let scale = d * f32::from(scale);
                    let min = dm * f32::from(min);
                    let group_bytes = &qs[(group / 2) * 32..(group / 2 + 1) * 32];
                    let group_x = &inp[group * 32..(group + 1) * 32];
                    let high = group % 2 == 1;
                    for (&raw, &x) in group_bytes.iter().zip(group_x) {
                        let q = if high { raw >> 4 } else { raw & 15 };
                        sum += (f32::from(q) * scale - min) * x;
                    }
                }
            }
            sum
        }
        StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K) => {
            let mut sum = 0.0f32;
            for (b, inp) in row.chunks_exact(210).zip(x.chunks_exact(256)) {
                let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
                for half in 0..2 {
                    let low = &b[half * 64..half * 64 + 64];
                    let high = &b[128 + half * 32..128 + half * 32 + 32];
                    let scales = &b[192 + half * 8..192 + half * 8 + 8];
                    for lane in 0..32 {
                        let hi = high[lane];
                        for group in 0..4 {
                            let lo = if group < 2 {
                                low[lane + group * 32] & 15
                            } else {
                                low[lane + (group - 2) * 32] >> 4
                            };
                            let q = ((lo | (((hi >> (group * 2)) & 3) << 4)) as i32 - 32) as f32;
                            let scale = scales[group * 2 + lane / 16] as i8 as f32;
                            sum += d * scale * q * inp[half * 128 + group * 32 + lane];
                        }
                    }
                }
            }
            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_projection(
        weights: &[f32],
        input: &[f32],
        profile: SourceProjectionProfile,
    ) -> Result<Vec<f32>> {
        let packed = weights
            .iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        source_reference_projection_with_profile(
            MatVecRegion {
                source_identity: SourceIdentity::for_new_allocation().unwrap(),
                tensor_name: "synthetic.bf16.weight",
                kind: StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
                columns: input.len(),
                rows: weights.len() / input.len(),
                packed_weights: &packed,
                activation: input,
            },
            profile,
        )
    }

    #[test]
    fn explicit_bf16_avx2_keeps_lane_cancellation_separate_from_legacy_serial_sum() {
        let mut weight = vec![0.0; 128];
        weight[0] = 16384.0;
        weight[1] = 2f32.powi(-10);
        weight[64] = -16384.0;
        let input = [1.0; 128];
        assert_eq!(
            bf16_projection(&weight, &input, SourceProjectionProfile::Source).unwrap(),
            [0.0]
        );
        assert_eq!(
            bf16_projection(&weight, &input, SourceProjectionProfile::TorchCpuBf16Avx2).unwrap(),
            [2f32.powi(-10)]
        );
    }

    #[test]
    fn explicit_bf16_profile_keeps_signed_zero_and_rejects_unbound_or_non_bf16_inputs() {
        for width in [1, 16, 64, 65, 128] {
            for zero in [0.0, -0.0] {
                let result = bf16_projection(
                    &vec![zero; width],
                    &vec![1.0; width],
                    SourceProjectionProfile::TorchCpuBf16Avx2,
                )
                .unwrap()[0];
                assert_eq!(result.to_bits(), 0);
            }
        }
        let packed = 0x3f80u16.to_le_bytes();
        let mut region = MatVecRegion {
            source_identity: SourceIdentity::for_new_allocation().unwrap(),
            tensor_name: "synthetic.one.weight",
            kind: StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            columns: 1,
            rows: 1,
            packed_weights: &packed,
            activation: &[1.0],
        };
        let profile = SourceProjectionProfile::TorchCpuBf16Avx2;
        assert_eq!(
            source_reference_projection_with_profile(region, profile).unwrap(),
            [1.0]
        );
        region.activation = &[1.1];
        assert!(source_reference_projection_with_profile(region, profile).is_err());
        region.activation = &[f32::NAN];
        assert!(source_reference_projection_with_profile(region, profile).is_err());
        region.activation = &[1.0];
        region.source_identity = SourceIdentity::unbound();
        assert!(source_reference_projection_with_profile(region, profile).is_err());
        region.source_identity = SourceIdentity::for_new_allocation().unwrap();
        region.kind = StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F16);
        assert!(source_reference_projection_with_profile(region, profile).is_err());
        region.kind = StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16);
        region.packed_weights = &packed[..1];
        assert!(source_reference_projection_with_profile(region, profile).is_err());
    }

    #[test]
    fn standalone_source_reference_projection_is_bounded_and_exactly_typed() {
        let mut packed = vec![0u8; 8 * 18];
        for chunk in packed.chunks_exact_mut(18) {
            chunk[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        let x = [1.0f32; 32];
        let mut region = MatVecRegion {
            source_identity: SourceIdentity::unbound(),
            tensor_name: "synthetic.quant.weight",
            kind: StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0),
            columns: 32,
            rows: 8,
            packed_weights: &packed,
            activation: &x,
        };
        let result = source_reference_projection(region).unwrap();
        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|v| *v == result[0] && v.is_finite()));
        region.rows = usize::MAX;
        assert!(source_reference_projection(region).is_err());
        region.rows = 8;
        region.columns = 33;
        assert!(source_reference_projection(region).is_err());
        region.columns = 32;
        region.packed_weights = &packed[..18];
        assert!(source_reference_projection(region).is_err());
    }

    #[test]
    fn q4_k_decodes_nonzero_source_bytes_but_source_exact_projection_is_unsupported() {
        let kind = StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4K);
        let mut packed = [0u8; 144];
        packed[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        packed[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 1/2
        packed[4..8].fill(2);
        packed[8..12].fill(3);
        packed[12..16].fill(0x32); // all eight groups: scale = 2, min = 3
        packed[16..].fill(0x51); // alternating 32-value groups: q = 1, q = 5
        let decoded = dequantize(kind, &packed, 256).unwrap();
        assert_eq!(decoded.len(), 256);
        for (group, values) in decoded.chunks_exact(32).enumerate() {
            let expected: f32 = if group % 2 == 0 { 0.5 } else { 8.5 };
            assert!(values.iter().all(|v| v.to_bits() == expected.to_bits()));
        }
        // Valid standalone source geometry; decoding does not qualify Q4_K x Q8_K.
        let input = [1.0; 256];
        let region = MatVecRegion {
            source_identity: SourceIdentity::for_new_allocation().unwrap(),
            tensor_name: "synthetic.q4_k.weight",
            kind,
            columns: 256,
            rows: 1,
            packed_weights: &packed,
            activation: &input,
        };
        match source_reference_projection(region) {
            Err(Error::Unsupported(message)) => assert_eq!(
                message,
                "GGUF Q4_K x Q8_K physical dot not independently qualified for this CPU runner"
            ),
            other => panic!("expected Q4_K source-exact refusal, got {other:?}"),
        }
    }

    #[test]
    fn q4_block_matrix_row_dot_matches_reference() {
        for kind in [
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0),
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_1),
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q8_0),
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4K),
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K),
        ] {
            let (block, bytes) = kind.block();
            let mut v = vec![0u8; bytes as usize];
            // finite scale values for q4_0/q8_0/q4_1; q4k and q6k zero is valid
            if matches!(
                kind,
                StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0)
                    | StoredTensorEncoding::Quantized(
                        zerogravity_core::encoding::QuantScheme::Q4_1
                    )
                    | StoredTensorEncoding::Quantized(
                        zerogravity_core::encoding::QuantScheme::Q8_0
                    )
            ) {
                v[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
            }
            let input = (0..block as usize)
                .map(|i| (i % 17) as f32 - 8.)
                .collect::<Vec<_>>();
            let decoded = dequantize(kind, &v, block as usize).unwrap();
            let oracle = decoded
                .iter()
                .zip(&input)
                .map(|(&a, &b)| a * b)
                .sum::<f32>();
            let actual = dot_row(kind, &v, &input);
            assert!(
                (actual - oracle).abs() < 0.05,
                "{kind:?} {actual} vs {oracle}"
            );
        }
    }
}
