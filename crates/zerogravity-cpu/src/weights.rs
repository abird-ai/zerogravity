//! Original CPU source arithmetic over a borrowed immutable source owner.
use crate::accelerator::{
    EmbeddingRegion, MatVecKernel, MatVecRegion, RegionExecution, SourceIdentity,
    validate_completed,
};
use crate::artifact::TensorInfo;
use crate::quant::{QuantKind, dequantize, f16_to_f32, f32_to_f16_rne};
use crate::{Error, Result};
use rayon::prelude::*;
use std::path::Path;
use zerogravity_artifacts::SourceWeights;
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
        if region.kind != QuantKind::Bf16 {
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
    if region.kind == QuantKind::Q4K {
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
    output
        .par_iter_mut()
        .zip(region.packed_weights.par_chunks_exact(row_bytes))
        .for_each(|(dst, row)| {
            *dst = if profile == SourceProjectionProfile::TorchCpuBf16Avx2 {
                dot_bf16_cpu_avx2(row, region.activation)
            } else if region.kind == QuantKind::Q4_0 && region.rows.is_multiple_of(8) {
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

/// Ephemeral source GGUF activation encoding, independent of model family.
/// Dynamic scales are recomputed for every GEMV input; not a weight codec.
#[derive(Clone)]
struct Q8SmallBlock {
    d: f32, // f16-roundtripped GGUF Q8_0/Q8_1 scale
    s: f32, // f16-roundtripped sum(q_i)*unrounded_scale, Q8_1 only
    qs: [i8; 32],
}

#[derive(Clone)]
struct Q8KBlock {
    d: f32, // GGUF Q8_K signed dynamic f32 scale
    qs: [i8; 256],
}

enum QuantizedActivation {
    Direct,
    Small(Vec<Q8SmallBlock>),
    Big(Vec<Q8KBlock>),
}

impl QuantizedActivation {
    fn prepare(kind: QuantKind, input: &[f32]) -> Result<Self> {
        if !input.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed("nonfinite GEMV input".into()));
        }
        match kind {
            QuantKind::Q4_0 | QuantKind::Q4_1 | QuantKind::Q8_0 => {
                if !input.len().is_multiple_of(32) {
                    return Err(Error::Malformed("Q8 activation width misaligned".into()));
                }
                let mut out = Vec::with_capacity(input.len() / 32);
                for block in input.chunks_exact(32) {
                    let amax = block.iter().map(|x| x.abs()).fold(0f32, f32::max);
                    let d = amax / 127.0;
                    let inv = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                    let mut qs = [0i8; 32];
                    let mut sum = 0i32;
                    for (i, &x) in block.iter().enumerate() {
                        let value = (x * inv).round_ties_even().clamp(-128.0, 127.0) as i8;
                        qs[i] = value;
                        sum += i32::from(value);
                    }
                    let d_half = f16_to_f32(f32_to_f16_rne(d));
                    let s_half = f16_to_f32(f32_to_f16_rne(sum as f32 * d));
                    out.push(Q8SmallBlock {
                        d: d_half,
                        s: s_half,
                        qs,
                    });
                }
                Ok(Self::Small(out))
            }
            QuantKind::Q6K | QuantKind::Q4K => {
                if !input.len().is_multiple_of(256) {
                    return Err(Error::Malformed("Q8_K activation width misaligned".into()));
                }
                let mut out = Vec::with_capacity(input.len() / 256);
                for block in input.chunks_exact(256) {
                    // GGUF Q8_K chooses the *signed* input value with largest
                    // absolute magnitude and uses iscale = -127 / signed_max.
                    let mut max_signed = 0.0f32;
                    let mut amax = 0.0f32;
                    for &x in block {
                        if x.abs() > amax {
                            amax = x.abs();
                            max_signed = x;
                        }
                    }
                    let inv = if amax == 0.0 {
                        0.0
                    } else {
                        -127.0 / max_signed
                    };
                    let d = if amax == 0.0 { 0.0 } else { 1.0 / inv };
                    let mut qs = [0i8; 256];
                    for (i, &x) in block.iter().enumerate() {
                        qs[i] = (x * inv).round_ties_even().clamp(-128.0, 127.0) as i8;
                    }
                    out.push(Q8KBlock { d, qs });
                }
                Ok(Self::Big(out))
            }
            QuantKind::F32 | QuantKind::F16 | QuantKind::Bf16 => Ok(Self::Direct),
        }
    }
}

/// Matches the x86 GGML Q4_0_8x8 versus Q8_0 GEMV numerical contract,
/// while reading original (unrepacked) GGUF row bytes directly.
///
/// The source optimized kernel pre-repacks eight weight rows for efficient
/// memory access, sums all 32 signed integer products into ONE i32 per
/// quantization block, then performs ONE fused f32 multiply-add per block.
/// The single-row GGML vec_dot kernel instead keeps eight partial f32
/// accumulators and horizontally reduces at the end. Both are valid
/// quantized dot implementations but they round DIFFERENTLY in f32.
/// Replication of the 8x8 reduction order does not require weight repacking
/// because each row has independent integer sums and f32 accumulators.
fn dot_q4_0_repacked_8x8_source_row(row: &[u8], blocks: &[Q8SmallBlock]) -> f32 {
    let mut accumulator = 0.0f32;
    for (weights, activation) in row.chunks_exact(18).zip(blocks) {
        let packed = &weights[2..18];
        let mut integer_sum = 0i32;
        for (i, &nibbles) in packed.iter().enumerate() {
            let lo = i32::from(nibbles & 15) - 8;
            let hi = i32::from(nibbles >> 4) - 8;
            integer_sum += lo * i32::from(activation.qs[i]);
            integer_sum += hi * i32::from(activation.qs[i + 16]);
        }
        let scale = f16_to_f32(u16::from_le_bytes([weights[0], weights[1]])) * activation.d;
        accumulator = (integer_sum as f32).mul_add(scale, accumulator);
    }
    accumulator
}

/// Physical source-paired GGUF block integer-dot reference. This is Rust math
/// and never calls a foreign inference or quantization library. Keeping it
/// separate from pure dequantized FP32 matvec allows explicit source semantics
/// versus idealized FP32 diagnostic reference comparisons.
fn dot_row_with_activation(
    kind: QuantKind,
    row: &[u8],
    original: &[f32],
    activation: &QuantizedActivation,
) -> f32 {
    match (kind, activation) {
        (QuantKind::Q4_0, QuantizedActivation::Small(blocks)) => {
            // Source x86 GGML AVX2/FMA reduction topology, expressed as
            // portable Rust scalar lanes: 8 dot accumulators (four i8 pairs
            // each), fused scale*integer accumulation per 32-element block,
            // then a source-ordered horizontal sum. No ISA-specific SIMD is
            // required; this is a numerical specialization, not a throughput
            // optimization. Other architectures can supply their own
            // legally qualified kernel.
            let mut lanes = [0.0f32; 8];
            for (weight, quant) in row.chunks_exact(18).zip(blocks) {
                let d = f16_to_f32(u16::from_le_bytes([weight[0], weight[1]])) * quant.d;
                let packed = &weight[2..18];
                for (lane, accumulator) in lanes.iter_mut().enumerate() {
                    let mut dot = 0i32;
                    for j in lane * 4..lane * 4 + 4 {
                        let quantized = if j < 16 {
                            i32::from(packed[j] & 15) - 8
                        } else {
                            i32::from(packed[j - 16] >> 4) - 8
                        };
                        dot += quantized * i32::from(quant.qs[j]);
                    }
                    *accumulator = d.mul_add(dot as f32, *accumulator);
                }
            }
            let r0 = lanes[0] + lanes[4];
            let r1 = lanes[1] + lanes[5];
            let r2 = lanes[2] + lanes[6];
            let r3 = lanes[3] + lanes[7];
            (r0 + r2) + (r1 + r3)
        }
        (QuantKind::Q8_0, QuantizedActivation::Small(blocks)) => {
            // Source GGML Haswell nonrepacked Q8_0 x Q8_0 uses one f32
            // FMA accumulator per 4-element integer lane, then an ordered
            // 8-lane horizontal sum. A single int32 block sum accumulated in
            // f32 across blocks is algebraically equivalent but rounds
            // differently and can flip later Q8 quantization boundaries.
            let mut lanes = [0.0f32; 8];
            for (weight, quant) in row.chunks_exact(34).zip(blocks) {
                let scale = f16_to_f32(u16::from_le_bytes([weight[0], weight[1]])) * quant.d;
                for (lane, acc) in lanes.iter_mut().enumerate() {
                    let mut pair_dot = 0i32;
                    for i in lane * 4..lane * 4 + 4 {
                        pair_dot += i32::from(weight[2 + i] as i8) * i32::from(quant.qs[i]);
                    }
                    *acc = (pair_dot as f32).mul_add(scale, *acc);
                }
            }
            let r0 = lanes[0] + lanes[4];
            let r1 = lanes[1] + lanes[5];
            let r2 = lanes[2] + lanes[6];
            let r3 = lanes[3] + lanes[7];
            (r0 + r2) + (r1 + r3)
        }
        (QuantKind::Q4_1, QuantizedActivation::Small(blocks)) => {
            // Source Haswell Q4_1 x Q8_1 uses a floating zero-point
            // correction accumulated separately across blocks. The 8 SIMD
            // integer partial dot sums (four values per lane) are combined
            // with FMA into 8 f32 lanes; one final ordered horizontal sum
            // is added to the cumulative zero-point correction.
            let mut lanes = [0.0f32; 8];
            let mut summs = 0.0f32;
            for (weight, quant) in row.chunks_exact(20).zip(blocks) {
                let d0 = f16_to_f32(u16::from_le_bytes([weight[0], weight[1]]));
                let m = f16_to_f32(u16::from_le_bytes([weight[2], weight[3]]));
                summs += m * quant.s;
                let scale = d0 * quant.d;
                let packed = &weight[4..20];
                for (lane, acc) in lanes.iter_mut().enumerate() {
                    let mut partial = 0i32;
                    for i in lane * 4..lane * 4 + 4 {
                        let w = if i < 16 {
                            i32::from(packed[i] & 15)
                        } else {
                            i32::from(packed[i - 16] >> 4)
                        };
                        partial += w * i32::from(quant.qs[i]);
                    }
                    *acc = (partial as f32).mul_add(scale, *acc);
                }
            }
            let r0 = lanes[0] + lanes[4];
            let r1 = lanes[1] + lanes[5];
            let r2 = lanes[2] + lanes[6];
            let r3 = lanes[3] + lanes[7];
            ((r0 + r2) + (r1 + r3)) + summs
        }
        (QuantKind::Q6K, QuantizedActivation::Big(blocks)) => {
            // Source Haswell GGML Q6_K × Q8_K: one signed integer
            // accumulator per 32 consecutive activation positions (8 lanes
            // per 256-value superblock), then FMA each lane with its f32
            // dynamic Q8_K × f16 Q6_K scale and horizontally reduce once.
            // Summing 256 weighted integer products into one f32 partial
            // before accumulation produces different numerical semantics.
            let mut lanes = [0.0f32; 8];
            for (weight, quant) in row.chunks_exact(210).zip(blocks) {
                let scale = quant.d * f16_to_f32(u16::from_le_bytes([weight[208], weight[209]]));
                let mut int_acc = [0i32; 8];
                for half in 0..2 {
                    let low = &weight[half * 64..half * 64 + 64];
                    let high = &weight[128 + half * 32..128 + half * 32 + 32];
                    let scales = &weight[192 + half * 8..192 + half * 8 + 8];
                    for lane in 0..32 {
                        let hi = high[lane];
                        for group in 0..4 {
                            let nibble = if group < 2 {
                                low[lane + group * 32] & 15
                            } else {
                                low[lane + (group - 2) * 32] >> 4
                            };
                            let code = i32::from(nibble | (((hi >> (2 * group)) & 3) << 4));
                            let subscale = i32::from(scales[group * 2 + lane / 16] as i8);
                            let at = half * 128 + group * 32 + lane;
                            // SSE2/AVX2 pairwise madd maps four consecutive
                            // elements into each 32-bit SIMD lane, then adds
                            // the corresponding positions from all eight
                            // 32-value groups. Offset correction below uses
                            // the original eight contiguous 32-value groups,
                            // a DISTINCT lane layout. The final integer sum
                            // is equal but FMA per-lane reduction is not.
                            int_acc[(at % 32) / 4] += code * subscale * i32::from(quant.qs[at]);
                        }
                    }
                }
                let scales = &weight[192..208];
                for (lane, acc) in int_acc.iter_mut().enumerate() {
                    let mut offset = 0i32;
                    for sub in 0..2 {
                        let group16 = lane * 2 + sub;
                        let q8sum = quant.qs[group16 * 16..(group16 + 1) * 16]
                            .iter()
                            .map(|&v| i32::from(v))
                            .sum::<i32>();
                        offset += q8sum * i32::from(scales[group16] as i8);
                    }
                    *acc -= 32 * offset;
                }
                for (acc, &code) in lanes.iter_mut().zip(&int_acc) {
                    *acc = (code as f32).mul_add(scale, *acc);
                }
            }
            let r0 = lanes[0] + lanes[4];
            let r1 = lanes[1] + lanes[5];
            let r2 = lanes[2] + lanes[6];
            let r3 = lanes[3] + lanes[7];
            (r0 + r2) + (r1 + r3)
        }
        // Q4_K is not present in the two initial Qwen checkpoints. Leave its
        // independent FP32 dequant numerical oracle operational pending a
        // separately source-qualified Q4_K x Q8_K block dot.
        (QuantKind::Q4K, QuantizedActivation::Big(_)) => dot_row(kind, row, original),
        (QuantKind::F32 | QuantKind::F16 | QuantKind::Bf16, QuantizedActivation::Direct) => {
            dot_row(kind, row, original)
        }
        _ => unreachable!("shape/codec validated before GEMV dispatch"),
    }
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

// Numerical topology follows PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb,
// ReducedPrecisionFloatGemvFastPathKernel.cpp and functional_base.h (AVX2).
// That reduction carries this ggml attribution; no foreign runtime is called.
//
// MIT License
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
pub(crate) fn dot_bf16_cpu_avx2(row: &[u8], x: &[f32]) -> f32 {
    let weight = |i: usize| {
        let at = 2 * i;
        f32::from_bits(u32::from(u16::from_le_bytes([row[at], row[at + 1]])) << 16)
    };
    let reduce = |v: [f32; 8]| ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
    let aligned = x.len() / 64 * 64;
    let vector_end = x.len() / 16 * 16;
    let mut registers = [[0.0f32; 8]; 8];
    for chunk in (0..aligned).step_by(64) {
        for (register, lanes) in registers.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let i = chunk + register * 8 + lane;
                *sum = weight(i).mul_add(x[i], *sum);
            }
        }
    }
    for offset in [4, 2, 1] {
        let (low, high) = registers.split_at_mut(offset);
        for (lanes, other) in low.iter_mut().zip(high.iter()) {
            for (sum, &value) in lanes.iter_mut().zip(other) {
                *sum += value;
            }
        }
    }
    let mut result = reduce(registers[0]);
    let mut tail = [0.0f32; 8];
    for chunk in (aligned..vector_end).step_by(16) {
        for (lane, sum) in tail.iter_mut().enumerate() {
            let i = chunk + lane;
            *sum = weight(i).mul_add(x[i], *sum);
            *sum = weight(i + 8).mul_add(x[i + 8], *sum);
        }
    }
    result += reduce(tail);
    for (i, &activation) in x.iter().enumerate().skip(vector_end) {
        result += weight(i) * activation;
    }
    result
}

fn dot_row(kind: QuantKind, row: &[u8], x: &[f32]) -> f32 {
    match kind {
        QuantKind::F32 => dot_f32_source_reduction(row, x),
        QuantKind::F16 | QuantKind::Bf16 => row
            .chunks_exact(2)
            .zip(x)
            .map(|(b, &v)| {
                let raw = u16::from_le_bytes([b[0], b[1]]);
                (if kind == QuantKind::F16 {
                    f16_to_f32(raw)
                } else {
                    f32::from_bits(u32::from(raw) << 16)
                }) * v
            })
            .sum(),
        QuantKind::Q8_0 => {
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
        QuantKind::Q4_0 | QuantKind::Q4_1 => {
            let len = if kind == QuantKind::Q4_0 { 18 } else { 20 };
            let mut sum = 0f32;
            for (block, inp) in row.chunks_exact(len).zip(x.chunks_exact(32)) {
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let offset = if kind == QuantKind::Q4_0 {
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
        QuantKind::Q4K => {
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
        QuantKind::Q6K => {
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
                kind: QuantKind::Bf16,
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
    fn explicit_bf16_vector_and_scalar_tails_match_pinned_actual_torch_golden_bits() {
        // Synthetic B1 F.linear, Torch cf30153..., CPU AVX2, no GPU.
        // x=1; w_i=(i%7-3)*2^-10, first=16384, last=-16384 if width>1.
        for (width, expected) in [
            (1, 0x4680_0000u32),
            (2, 0),
            (7, 0),
            (8, 0x3b80_0000),
            (9, 0),
            (15, 0x3b80_0000),
            (16, 0xbb40_0000),
            (17, 0xbb40_0000),
            (31, 0xbb40_0000),
            (32, 0xbb40_0000),
            (33, 0xbb40_0000),
            (47, 0xbb40_0000),
            (48, 0xbb00_0000),
            (49, 0),
            (63, 0),
            (64, 0x3b00_0000),
            (65, 0xba80_0000),
            (79, 0xba80_0000),
            (80, 0xbb40_0000),
            (95, 0xbb80_0000),
            (96, 0xbb40_0000),
            (127, 0x3b00_0000),
            (128, 0xbb40_0000),
            (129, 0xbb40_0000),
            (255, 0xbb00_0000),
            (256, 0xbb40_0000),
            (257, 0xbb40_0000),
            (1024, 0xbb40_0000),
        ] {
            let mut weight = (0..width)
                .map(|i| ((i % 7) as f32 - 3.0) * 2f32.powi(-10))
                .collect::<Vec<_>>();
            weight[0] = 16384.0;
            if width > 1 {
                weight[width - 1] = -16384.0;
            }
            let value = bf16_projection(
                &weight,
                &vec![1.0; width],
                SourceProjectionProfile::TorchCpuBf16Avx2,
            )
            .unwrap()[0];
            let word = value.to_bits();
            let rounded = word.wrapping_add(0x7fff + ((word >> 16) & 1)) & 0xffff_0000;
            assert_eq!(rounded, expected, "synthetic width={width}");
        }
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
            kind: QuantKind::Bf16,
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
        region.kind = QuantKind::F16;
        assert!(source_reference_projection_with_profile(region, profile).is_err());
        region.kind = QuantKind::Bf16;
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
            kind: QuantKind::Q4_0,
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
    fn q4_block_matrix_row_dot_matches_reference() {
        for kind in [
            QuantKind::Q4_0,
            QuantKind::Q4_1,
            QuantKind::Q8_0,
            QuantKind::Q4K,
            QuantKind::Q6K,
        ] {
            let (block, bytes) = kind.block();
            let mut v = vec![0u8; bytes as usize];
            // finite scale values for q4_0/q8_0/q4_1; q4k and q6k zero is valid
            if matches!(kind, QuantKind::Q4_0 | QuantKind::Q4_1 | QuantKind::Q8_0) {
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
