//! Allocation-free causal BF16 attention, CUDA tile-64 online-softmax arithmetic.
use super::kernel::{
    PinnedKernel, PreparedPinnedLaunch, StaticScalarBlock, StaticScalarSchema, StaticScalarType,
    StaticScalarValue,
};
use crate::resident_program::BufferRange;
use cubecl::{client::Client, server::Handle};
use cubecl_runtime::kernel::BufferIOAttr;
use zerogravity_core::{Error, Result};

const SOURCE: &str = include_str!("kernels.cu");

pub(crate) struct Bf16CausalAttention {
    kernel: PinnedKernel,
    grid: [u32; 3],
}
impl Bf16CausalAttention {
    /// Q/O[rows, query_heads, width], K/V[capacity, kv_heads, width].
    /// Query row i attends to K/V[0..=base+i]; no padded tail is consumed.
    pub(crate) fn new(
        rows: usize,
        base: usize,
        capacity: usize,
        query_heads: usize,
        kv_heads: usize,
        width: usize,
        scale: f32,
    ) -> Result<Self> {
        if rows == 0
            || kv_heads == 0
            || query_heads == 0
            || !query_heads.is_multiple_of(kv_heads)
            || width != 256
            || !scale.is_finite()
            || scale <= 0.0
            || base.checked_add(rows).is_none_or(|end| end > capacity)
        {
            return Err(Error::Unsupported(
                "BF16 CUDA attention geometry/scale".into(),
            ));
        }
        let scale_log2 = scale * std::f32::consts::LOG2_E;
        if !scale_log2.is_finite() || scale_log2 == 0.0 {
            return Err(Error::Unsupported(
                "BF16 CUDA attention scale overflow/underflow".into(),
            ));
        }
        let extent = |rows: usize, heads: usize| {
            rows.checked_mul(heads)
                .and_then(|n| n.checked_mul(width * 2))
                .filter(|n| *n <= isize::MAX as usize)
                .ok_or_else(|| Error::Limit("BF16 attention extent overflow".into()))
        };
        let q_bytes = extent(rows, query_heads)?;
        let kv_bytes = extent(capacity, kv_heads)?;
        let grid = match (u32::try_from(rows.div_ceil(16)), u32::try_from(query_heads)) {
            (Ok(x), Ok(y)) if x <= i32::MAX as u32 && y <= 65_535 => [x, y, 1],
            _ => return Err(Error::Limit("BF16 attention grid overflow".into())),
        };
        let schema = StaticScalarSchema::new(
            &[
                (StaticScalarType::U64, 0),
                (StaticScalarType::U64, 8),
                (StaticScalarType::U64, 16),
                (StaticScalarType::U64, 24),
                (StaticScalarType::U64, 32),
                (StaticScalarType::F32Bits, 40),
            ],
            48,
            8,
        )?;
        let scalars = StaticScalarBlock::new(
            schema,
            &[
                StaticScalarValue::U64(rows as u64),
                StaticScalarValue::U64(base as u64),
                StaticScalarValue::U64(capacity as u64),
                StaticScalarValue::U64(query_heads as u64),
                StaticScalarValue::U64(kv_heads as u64),
                StaticScalarValue::F32Bits(scale_log2.to_bits()),
            ],
        )?;
        let kernel = PinnedKernel::new_with_static_scalars(
            "bf16_causal_attention",
            SOURCE.into(),
            128,
            [
                (q_bytes, BufferIOAttr::ReadOnly),
                (kv_bytes, BufferIOAttr::ReadOnly),
                (kv_bytes, BufferIOAttr::ReadOnly),
                (q_bytes, BufferIOAttr::WriteOnly),
                (4, BufferIOAttr::ReadWrite),
            ]
            .into(),
            scalars,
        )?;
        Ok(Self { kernel, grid })
    }

    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    pub(crate) fn prepare(
        &self,
        client: &Client,
        handles: [Handle; 5],
    ) -> Result<PreparedPinnedLaunch> {
        require_disjoint_writes(&handles)?;
        self.kernel.prepare(client, self.grid, handles.into())
    }
}

fn require_disjoint_writes(handles: &[Handle; 5]) -> Result<()> {
    let ranges = handles
        .iter()
        .map(BufferRange::new)
        .collect::<Result<Vec<_>>>()?;
    for write in [3, 4] {
        for other in 0..handles.len() {
            if write != other && ranges[write].overlaps(&ranges[other]) {
                return Err(Error::State(
                    "BF16 attention writable binding overlaps".into(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
