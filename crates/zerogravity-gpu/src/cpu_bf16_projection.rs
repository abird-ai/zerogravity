//! Closed CUDA implementation of the existing CPU BF16 projection contract.
//! This is not cuBLAS/vLLM arithmetic or a complete HF execution profile.
// Numerical topology: existing weights::dot_bf16_cpu_avx2, pinned PyTorch
// cf30153c4c131c8164ee7798e5022d810682e2cb AVX2. Attribution follows that reducer.
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
use cubecl::prelude::*;
use std::ops::Range;
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{MatrixAxes, ProgramNumericalProfile, SourceTensor};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::{Error, Result};

pub(crate) mod adversary;
pub(crate) mod qualification;
mod resident;
pub(crate) use resident::{PreparedProjection, ProjectionReadback};

const MAX_WEIGHT_BYTES: usize = 32 * 1024 * 1024;
const MAX_ROWS: usize = 16384;
const MAX_COLUMNS: usize = 8192;
const LANES: usize = 64;
const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;

fn malformed(message: &str) -> Error {
    Error::Malformed(format!("CPU-profile BF16 CUDA: {message}"))
}
fn finite_bf16(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(2)
        && bytes
            .chunks_exact(2)
            .all(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 != 0x7f80)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Identity {
    descriptor: SourceTensor,
    rows: Range<usize>,
    source_byte_offset: u64,
    original_byte_offset: u64,
    original_byte_len: u64,
    window_bytes: usize,
    bundle_sha256: String,
}
impl Identity {
    pub(crate) fn descriptor(&self) -> &SourceTensor {
        &self.descriptor
    }
    pub(crate) fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }
    fn require_window(&self, window: &TensorWindow) -> Result<()> {
        let raw = window.matrix().tensor();
        if self.descriptor.source != window.matrix().source_identity()
            || self.descriptor.name != raw.name
            || self.rows != window.row_range()
            || self.source_byte_offset != window.source_offset()
            || self.original_byte_offset != raw.byte_offset
            || self.original_byte_len != raw.byte_len
            || self.window_bytes != window.as_slice().len()
            || self.bundle_sha256 != window.verified_bundle_sha256()
        {
            return Err(malformed(
                "private result lost full original window identity",
            ));
        }
        Ok(())
    }
}

/// A real sealed original matrix range, never a reduced SourceTensorRef.
pub(crate) struct Bf16Window {
    source: VerifiedTensorSource,
    owner: Arc<TensorWindow>,
    identity: Identity,
}
impl Bf16Window {
    pub(crate) fn bind(
        source: VerifiedTensorSource,
        owner: Arc<TensorWindow>,
        descriptor: SourceTensor,
        rows: Range<usize>,
        profile: ProgramNumericalProfile,
    ) -> Result<Self> {
        let matrix = owner.matrix();
        let raw = matrix.tensor();
        let full_bytes = descriptor
            .columns
            .checked_mul(descriptor.rows)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| malformed("original full size overflow"))?;
        let offset = u64::try_from(rows.start)
            .ok()
            .and_then(|n| n.checked_mul(descriptor.columns as u64))
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| raw.byte_offset.checked_add(n))
            .ok_or_else(|| malformed("original range offset overflow"))?;
        raw.byte_offset
            .checked_add(raw.byte_len)
            .ok_or_else(|| malformed("original full byte end overflow"))?;
        if profile != PROFILE
            || !descriptor.source.is_bound()
            || source.source_identity() != descriptor.source
            || source.verified_bundle_sha256() != owner.verified_bundle_sha256()
            || descriptor.source != matrix.source_identity()
            || descriptor.name != raw.name
            || descriptor.name.is_empty()
            || descriptor.kind != QuantKind::Bf16
            || raw.kind != QuantKind::Bf16
            || descriptor.axes != MatrixAxes::OutputInput
            || matrix.axes() != MatrixAxes::OutputInput
            || raw.shape.as_slice() != [descriptor.rows as u64, descriptor.columns as u64]
            || matrix.rows() != descriptor.rows
            || matrix.columns() != descriptor.columns
            || raw.byte_len != full_bytes as u64
            || descriptor.rows == 0
            || descriptor.rows > 1_048_576
            || rows.start >= rows.end
            || rows.end > descriptor.rows
            || owner.row_range() != rows
            || owner.source_offset() != offset
        {
            return Err(malformed(
                "full original descriptor/profile/issuer/range differs",
            ));
        }
        let shape = Shape::new(rows.len(), descriptor.columns, 1)?;
        if owner.as_slice().len() != shape.weight_bytes || !finite_bf16(owner.as_slice()) {
            return Err(malformed("authentic range bytes or finite BF16 weights"));
        }
        let identity = Identity {
            descriptor,
            rows,
            source_byte_offset: offset,
            original_byte_offset: raw.byte_offset,
            original_byte_len: raw.byte_len,
            window_bytes: owner.as_slice().len(),
            bundle_sha256: owner.verified_bundle_sha256().into(),
        };
        Ok(Self {
            source,
            owner,
            identity,
        })
    }
}

/// Explicit source-tagged operator input transport; no trained producer claim.
pub(crate) struct Bf16Inputs {
    source: VerifiedTensorSource,
    shape: Shape,
    bytes: Arc<[u8]>,
    _admission: Arc<WindowWorkspace>,
}
impl Bf16Inputs {
    pub(crate) fn bind(
        source: VerifiedTensorSource,
        profile: ProgramNumericalProfile,
        rows: usize,
        columns: usize,
        batch: usize,
        raw: &[u8],
    ) -> Result<Self> {
        let shape = validate_input(source.source_identity(), profile, rows, columns, batch, raw)?;
        let admission = Arc::new(source.reserve_workspace(raw.len())?);
        let bytes = Arc::from(raw); // Same-source admission precedes private copy.
        Ok(Self {
            source,
            shape,
            bytes,
            _admission: admission,
        })
    }
}
fn validate_input(
    source: zerogravity_core::accelerator::SourceIdentity,
    profile: ProgramNumericalProfile,
    rows: usize,
    columns: usize,
    batch: usize,
    bytes: &[u8],
) -> Result<Shape> {
    let shape = Shape::new(rows, columns, batch)?;
    if !source.is_bound()
        || profile != PROFILE
        || bytes.len() != shape.input_bytes
        || !finite_bf16(bytes)
    {
        return Err(malformed(
            "explicit finite BF16 input source/profile/full [B,C]",
        ));
    }
    Ok(shape)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Shape {
    rows: usize,
    columns: usize,
    batch: usize,
    weight_bytes: usize,
    input_bytes: usize,
    outputs: usize,
}
impl Shape {
    fn new(rows: usize, columns: usize, batch: usize) -> Result<Self> {
        if rows == 0
            || rows > MAX_ROWS
            || columns == 0
            || columns > MAX_COLUMNS
            || !columns.is_multiple_of(LANES)
            || ![1, 2, 4, 8].contains(&batch)
        {
            return Err(malformed("closed rows/width-div64/native batch geometry"));
        }
        let weight_bytes = rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(2))
            .filter(|n| *n <= MAX_WEIGHT_BYTES)
            .ok_or_else(|| malformed("bounded original weight range"))?;
        let input_bytes = batch
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| malformed("input bytes overflow"))?;
        let outputs = batch
            .checked_mul(rows)
            .filter(|n| u32::try_from(*n).is_ok())
            .ok_or_else(|| malformed("native output domain"))?;
        Ok(Self {
            rows,
            columns,
            batch,
            weight_bytes,
            input_bytes,
            outputs,
        })
    }
    fn owned_bytes(self) -> Result<usize> {
        self.outputs
            .checked_mul(12)
            .and_then(|n| n.checked_add(self.weight_bytes))
            .and_then(|n| n.checked_add(self.input_bytes))
            .and_then(|n| n.checked_add(4))
            .ok_or_else(|| malformed("all leased payloads overflow"))
    }
}

/// One [request,original output row] per64-thread workgroup. The lane index
/// is exactly CPU register*8+lane; no subgroup reduction or TensorCore math.
#[cube(launch_unchecked)]
fn project_cpu_bf16(
    weights: &[u32],
    input: &[u32],
    zero: &[f32],
    accumulation: &mut [f32],
    rounded: &mut [f32],
    status: &mut [u32],
    #[comptime] rows: usize,
    #[comptime] columns: usize,
) {
    let output = usize::cast_from(CUBE_POS_X);
    let request = output / rows;
    let row = output % rows;
    let lane = usize::cast_from(UNIT_POS);
    let mut sums = Shared::<[f32]>::new_slice(64usize);
    let mut flags = Shared::<[u32]>::new_slice(64usize);
    // Dynamic immutable +0 defeats CubeCL's literal FAdd(x,0) simplifier.
    // CPU initializes all64 partials with +0 and adds its empty-tail +0 last.
    let z = zero[0];
    let mut accumulator = z;
    let mut invalid = 0u32;
    if u32::reinterpret(z) != 0u32 {
        invalid |= 4u32;
    }
    for chunk in 0..columns / 64usize {
        let column = chunk * 64usize + lane;
        let wi = row * columns + column;
        let xi = request * columns + column;
        let wb = (weights[wi / 2usize] >> (u32::cast_from(wi % 2usize) * 16u32)) & 0xffffu32;
        let xb = (input[xi / 2usize] >> (u32::cast_from(xi % 2usize) * 16u32)) & 0xffffu32;
        if (wb & 0x7f80u32) == 0x7f80u32 {
            invalid |= 1u32;
        }
        if (xb & 0x7f80u32) == 0x7f80u32 {
            invalid |= 2u32;
        }
        accumulator = fma(
            f32::reinterpret(wb << 16u32),
            f32::reinterpret(xb << 16u32),
            accumulator,
        );
    }
    sums[lane] = accumulator;
    flags[lane] = invalid;
    sync_cube();
    if lane < 32usize {
        let collapsed = sums[lane] + sums[lane + 32usize];
        sums[lane] = collapsed;
    }
    sync_cube();
    if lane < 16usize {
        let collapsed = sums[lane] + sums[lane + 16usize];
        sums[lane] = collapsed;
    }
    sync_cube();
    if lane < 8usize {
        let collapsed = sums[lane] + sums[lane + 8usize];
        sums[lane] = collapsed;
    }
    sync_cube();
    if lane == 0usize {
        let r0 = sums[0] + sums[4];
        let r1 = sums[1] + sums[5];
        let r2 = sums[2] + sums[6];
        let r3 = sums[3] + sums[7];
        let result = ((r0 + r2) + (r1 + r3)) + z;
        let bits = u32::reinterpret(result);
        let rounded_bits = (bits + (0x7fffu32 + ((bits >> 16u32) & 1u32))) & 0xffff0000u32;
        let mut combined = 0u32;
        for at in 0..64usize {
            combined |= flags[at];
        }
        if (bits & 0x7f800000u32) == 0x7f800000u32 {
            combined |= 8u32;
        }
        if (rounded_bits & 0x7f800000u32) == 0x7f800000u32 {
            combined |= 16u32;
        }
        accumulation[output] = result;
        rounded[output] = f32::reinterpret(rounded_bits);
        status[output] = combined;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_shape_and_exact_allocation_accounting() {
        for batch in [1, 2, 4, 8] {
            let shape = Shape::new(512, 2048, batch).unwrap();
            assert_eq!(
                shape.owned_bytes().unwrap(),
                2 * 512 * 2048 + 2 * batch * 2048 + 12 * batch * 512 + 4
            );
        }
        for (r, c, b) in [
            (0, 1024, 1),
            (1, 0, 1),
            (1, 63, 1),
            (1, 65, 1),
            (1, 8193, 1),
            (1, 64, 3),
            (16385, 64, 1),
            (16384, 8192, 8),
        ] {
            assert!(Shape::new(r, c, b).is_err());
        }
    }
    #[test]
    fn explicit_input_bits_never_cast_or_admit_wrong_source_profile() {
        let source = zerogravity_core::accelerator::SourceIdentity::for_new_allocation().unwrap();
        let mut raw = vec![0; 128];
        for (at, bits) in [0u16, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f].iter().enumerate() {
            raw[2 * at..2 * at + 2].copy_from_slice(&bits.to_le_bytes());
        }
        validate_input(source, PROFILE, 1, 64, 1, &raw).unwrap();
        raw[0..2].copy_from_slice(&0x7f80u16.to_le_bytes());
        assert!(validate_input(source, PROFILE, 1, 64, 1, &raw).is_err());
        assert!(
            validate_input(
                source,
                ProgramNumericalProfile::GgufSourceV1,
                1,
                64,
                1,
                &[0; 128]
            )
            .is_err()
        );
    }
}
