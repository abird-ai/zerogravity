//! Parameterized BF16 GEMM on the caller's CUDA queue.
//! No allocator, BLAS runtime, stream, completion authority, or model semantics.
use super::kernel::{
    PinnedKernel, PreparedPinnedLaunch, StaticScalarBlock, StaticScalarSchema, StaticScalarType,
    StaticScalarValue,
};
use cubecl::{client::Client, server::Handle};
use cubecl_runtime::kernel::BufferIOAttr;
use zerogravity_core::{Error, Result};

const SOURCE: &str = include_str!("kernels.cu");

#[derive(Clone, Copy)]
pub(crate) struct Bf16GemmShape {
    pub(crate) columns: usize,
    pub(crate) rows: usize,
    pub(crate) inner: usize,
    bytes: [usize; 3],
}
impl Bf16GemmShape {
    pub(crate) fn new(columns: usize, rows: usize, inner: usize) -> Result<Self> {
        if columns == 0 || rows == 0 || inner == 0 {
            return Err(Error::Malformed(
                "BF16 GEMM requires nonzero dimensions".into(),
            ));
        }
        let extent = |a: usize, b: usize| {
            a.checked_mul(b)
                .and_then(|n| n.checked_mul(2))
                .filter(|n| *n <= isize::MAX as usize)
                .ok_or_else(|| Error::Limit("BF16 GEMM byte extent overflow".into()))
        };
        Ok(Self {
            columns,
            rows,
            inner,
            bytes: [
                extent(columns, inner)?,
                extent(rows, inner)?,
                extent(rows, columns)?,
            ],
        })
    }
    pub(crate) fn weight_bytes(self) -> usize {
        self.bytes[0]
    }
    pub(crate) fn input_bytes(self) -> usize {
        self.bytes[1]
    }
    pub(crate) fn output_bytes(self) -> usize {
        self.bytes[2]
    }
}

/// Reduction arithmetic is explicit, independent of source/model selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bf16GemmReduction {
    /// One FP32 sum; final BF16 rounding.
    Ordered,
    /// Alternate K segments between two FP32 sums; add before final rounding.
    InterleavedPair { terms: usize },
    /// Round each segment to BF16, then sum segments in ascending order in FP32.
    RoundedSegments { terms: usize },
    /// Strided per-lane segment FMAs, ordered segment sums, ordered lane sum.
    LaneStrided { lanes: usize, terms: usize },
    /// Contiguous per-lane FMA blocks, followed by a halving-tree lane reduction.
    LaneBlocked { lanes: usize, terms: usize },
}

/// W[columns, inner], X[rows, inner], Y[rows, columns], all contiguous BF16.
/// FP32 accumulation with an explicit reduction/rounding schedule.
pub(crate) struct Bf16Gemm {
    kernel: PinnedKernel,
    grid: [u32; 3],
}
impl Bf16Gemm {
    pub(crate) fn new(columns: usize, rows: usize, inner: usize) -> Result<Self> {
        Self::with_reduction(columns, rows, inner, Bf16GemmReduction::Ordered)
    }

    pub(crate) fn with_reduction(
        columns: usize,
        rows: usize,
        inner: usize,
        reduction: Bf16GemmReduction,
    ) -> Result<Self> {
        Self::with_output_stride(columns, rows, inner, reduction, columns)
    }

    pub(crate) fn with_output_stride(
        columns: usize,
        rows: usize,
        inner: usize,
        reduction: Bf16GemmReduction,
        output_stride: usize,
    ) -> Result<Self> {
        let shape = Bf16GemmShape::new(columns, rows, inner)?;
        let output_bytes = output_span(columns, rows, output_stride)?;
        let (terms, lanes) = match reduction {
            Bf16GemmReduction::Ordered => (32, 1),
            Bf16GemmReduction::InterleavedPair { terms }
            | Bf16GemmReduction::RoundedSegments { terms } => (terms, 1),
            Bf16GemmReduction::LaneStrided { lanes, terms }
            | Bf16GemmReduction::LaneBlocked { lanes, terms } => (terms, lanes),
        };
        if !lanes.is_power_of_two() || lanes > 32 {
            return Err(Error::Unsupported(
                "BF16 GEMM requires power-of-two warp lane groups".into(),
            ));
        }
        let granularity = match reduction {
            Bf16GemmReduction::LaneStrided { .. } => lanes,
            Bf16GemmReduction::LaneBlocked { .. } => 1,
            _ => 32,
        };
        if terms == 0 || !terms.is_multiple_of(granularity) {
            return Err(Error::Unsupported(
                "BF16 GEMM reduction requires whole positive groups".into(),
            ));
        }
        terms
            .checked_mul(lanes)
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| Error::Limit("BF16 GEMM reduction span overflow".into()))?;
        let tile_rows = if rows <= 8 { 8 } else { 32 };
        let (tile_columns, tile_rows) = match reduction {
            Bf16GemmReduction::LaneStrided { .. } | Bf16GemmReduction::LaneBlocked { .. } => {
                (128 / lanes, 1)
            }
            _ => (64, tile_rows),
        };
        let grid = [columns.div_ceil(tile_columns), rows.div_ceil(tile_rows), 1].map(u32::try_from);
        let grid = match grid {
            [Ok(x), Ok(y), Ok(z)] if x <= i32::MAX as u32 && y <= 65_535 => [x, y, z],
            _ => return Err(Error::Limit("BF16 GEMM grid overflow".into())),
        };
        let schema = StaticScalarSchema::new(
            &[
                (StaticScalarType::U64, 0),
                (StaticScalarType::U64, 8),
                (StaticScalarType::U64, 16),
                (StaticScalarType::U64, 24),
                (StaticScalarType::U64, 32),
                (StaticScalarType::U64, 40),
            ],
            48,
            8,
        )?;
        let scalars = StaticScalarBlock::new(
            schema,
            &[
                columns,
                rows,
                inner,
                terms,
                lanes.trailing_zeros() as usize,
                output_stride,
            ]
            .map(|n| StaticScalarValue::U64(n as u64)),
        )?;
        let kernel = PinnedKernel::new_with_static_scalars(
            match (reduction, tile_rows == 8) {
                (Bf16GemmReduction::Ordered, true) => "bf16_gemm_narrow",
                (Bf16GemmReduction::Ordered, false) => "bf16_gemm_wide",
                (Bf16GemmReduction::InterleavedPair { .. }, true) => "bf16_gemm_interleaved_narrow",
                (Bf16GemmReduction::InterleavedPair { .. }, false) => "bf16_gemm_interleaved_wide",
                (Bf16GemmReduction::RoundedSegments { .. }, true) => "bf16_gemm_segmented_narrow",
                (Bf16GemmReduction::RoundedSegments { .. }, false) => "bf16_gemm_segmented_wide",
                (Bf16GemmReduction::LaneStrided { .. }, _) => "bf16_gemm_lane_strided",
                (Bf16GemmReduction::LaneBlocked { .. }, _) => "bf16_gemm_lane_blocked",
            },
            SOURCE.into(),
            128,
            [
                (shape.weight_bytes(), BufferIOAttr::ReadOnly),
                (shape.input_bytes(), BufferIOAttr::ReadOnly),
                (output_bytes, BufferIOAttr::WriteOnly),
            ]
            .into(),
            scalars,
        )?
        .with_binding_alignment(2)?;
        Ok(Self { kernel, grid })
    }

    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    pub(crate) fn prepare(
        &self,
        client: &Client,
        weights: Handle,
        input: Handle,
        output: Handle,
    ) -> Result<PreparedPinnedLaunch> {
        require_disjoint_output(&weights, &input, &output)?;
        self.kernel
            .prepare(client, self.grid, vec![weights, input, output])
    }
}

pub(crate) fn output_span(columns: usize, rows: usize, stride: usize) -> Result<usize> {
    if columns == 0 || rows == 0 || stride < columns {
        return Err(Error::Malformed("BF16 GEMM output stride".into()));
    }
    (rows - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(columns))
        .and_then(|n| n.checked_mul(2))
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| Error::Limit("BF16 GEMM strided output extent".into()))
}

fn require_disjoint_output(weights: &Handle, input: &Handle, output: &Handle) -> Result<()> {
    let range = |handle: &Handle| {
        let start = handle.offset_start.unwrap_or(0);
        let end = handle
            .size()
            .checked_sub(handle.offset_end.unwrap_or(0))
            .filter(|end| *end >= start)
            .ok_or_else(|| Error::State("BF16 GEMM invalid binding range".into()))?;
        Ok::<_, Error>((start, end))
    };
    let (ws, we) = range(output)?;
    for read in [weights, input] {
        let (rs, re) = range(read)?;
        if read.service == output.service
            && read.memory.descriptor().id == output.memory.descriptor().id
            && rs < we
            && ws < re
        {
            return Err(Error::State("BF16 GEMM output overlaps an input".into()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
