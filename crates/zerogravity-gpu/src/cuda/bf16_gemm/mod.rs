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

/// Physical input storage only; mathematical rows/reduction are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bf16GemmInputLayout {
    /// X[row, inner], the canonical dense matrix layout.
    RowMajor,
    /// X[inner, row], useful when a narrow batch consumes all rows per k.
    InnerMajor,
}

/// W[columns, inner], X[rows, inner], Y[rows, columns], all contiguous BF16.
/// FP32 accumulation with an explicit reduction/rounding schedule.
pub(crate) struct Bf16Gemm {
    kernel: PinnedKernel,
    grid: [u32; 3],
    certified: bool,
}

/// Exact B8 inner-major LaneStrided<16,512> projection with immediate
/// numeric-zero greedy reduction over the eight columns computed by one CTA.
///
/// This is only a physical projection/reduction primitive. It owns no source,
/// model, sampler, session, root, completion or publication policy.
pub(crate) struct Bf16HeadGreedyPartials {
    kernel: PinnedKernel,
    grid: [u32; 3],
    records: usize,
    partial_bytes: usize,
}

impl Bf16HeadGreedyPartials {
    pub(crate) const ROWS: usize = 8;
    pub(crate) const COLUMNS_PER_RECORD: usize = 8;

    pub(crate) fn records(columns: usize) -> Result<usize> {
        if columns == 0 || u32::try_from(columns).is_err() {
            return Err(Error::Unsupported(
                "BF16 head greedy requires nonzero u32 vocabulary".into(),
            ));
        }
        Ok(columns.div_ceil(Self::COLUMNS_PER_RECORD))
    }

    pub(crate) fn partial_bytes(columns: usize) -> Result<usize> {
        Self::records(columns)?
            .checked_mul(Self::ROWS)
            .and_then(|records| records.checked_mul(3 * 4))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| Error::Limit("BF16 head greedy partial extent".into()))
    }

    pub(crate) fn new(columns: usize, inner: usize) -> Result<Self> {
        let shape = Bf16GemmShape::new(columns, Self::ROWS, inner)?;
        let records = Self::records(columns)?;
        let output_stride = records
            .checked_mul(3)
            .ok_or_else(|| Error::Limit("BF16 head greedy row stride".into()))?;
        let partial_bytes = Self::partial_bytes(columns)?;
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
            &[columns, Self::ROWS, inner, 512, 4, output_stride]
                .map(|value| StaticScalarValue::U64(value as u64)),
        )?;
        let x = u32::try_from(records)
            .ok()
            .filter(|value| *value <= i32::MAX as u32)
            .ok_or_else(|| Error::Limit("BF16 head greedy grid overflow".into()))?;
        let kernel = PinnedKernel::new_with_static_scalars(
            "bf16_gemm_lane_strided_batch8_inner_major_greedy_partials",
            SOURCE.into(),
            128,
            vec![
                (shape.weight_bytes(), BufferIOAttr::ReadOnly),
                (shape.input_bytes(), BufferIOAttr::ReadOnly),
                (partial_bytes, BufferIOAttr::WriteOnly),
                (4, BufferIOAttr::ReadWrite),
            ]
            .into(),
            scalars,
        )?
        .with_binding_alignment(16)?;
        Ok(Self {
            kernel,
            grid: [x, 1, 1],
            records,
            partial_bytes,
        })
    }

    pub(crate) fn record_count(&self) -> usize {
        self.records
    }

    pub(crate) fn output_bytes(&self) -> usize {
        self.partial_bytes
    }

    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    pub(crate) fn prepare(
        &self,
        client: &Client,
        weights: Handle,
        input: Handle,
        partials: Handle,
        status: Handle,
    ) -> Result<PreparedPinnedLaunch> {
        require_disjoint_output(&weights, &input, &partials)?;
        if status.service != client.service_id()
            || status.size_in_used() != 4
            || partials.service != client.service_id()
        {
            return Err(Error::State(
                "BF16 head greedy status service/extent".into(),
            ));
        }
        let partial_range = range(&partials)?;
        let status_range = range(&status)?;
        if partials.memory.descriptor().id == status.memory.descriptor().id
            && partial_range.0 < status_range.1
            && status_range.0 < partial_range.1
        {
            return Err(Error::State(
                "BF16 head greedy partials overlap status".into(),
            ));
        }
        self.kernel
            .prepare(client, self.grid, vec![weights, input, partials, status])
    }
}

/// Two exact B8 projections sharing one row-major activation binding.
///
/// The primary projection retains the accepted Ordered K64 arithmetic and the
/// secondary projection retains its independent LaneBlocked reduction. This is
/// a physical composition primitive only: no source/model/graph/session/root or
/// publication policy is encoded here.
pub(crate) struct Bf16SharedInputProjectionPair {
    kernel: PinnedKernel,
    grid: [u32; 3],
}

impl Bf16SharedInputProjectionPair {
    pub(crate) fn new(
        primary: Bf16GemmShape,
        primary_reduction: Bf16GemmReduction,
        secondary: Bf16GemmShape,
        secondary_reduction: Bf16GemmReduction,
    ) -> Result<Self> {
        if primary.rows != 8
            || secondary.rows != 8
            || primary.inner != secondary.inner
            || primary.inner < 2048
            || primary_reduction != Bf16GemmReduction::Ordered
        {
            return Err(Error::Unsupported(
                "BF16 shared-input pair primary geometry/reduction".into(),
            ));
        }
        let secondary_terms = match secondary_reduction {
            Bf16GemmReduction::LaneBlocked { lanes: 32, terms } if terms != 0 => terms,
            _ => {
                return Err(Error::Unsupported(
                    "BF16 shared-input pair secondary reduction".into(),
                ));
            }
        };
        secondary_terms
            .checked_mul(32)
            .filter(|span| *span <= isize::MAX as usize)
            .ok_or_else(|| Error::Limit("BF16 shared-input pair reduction span".into()))?;
        let blocks = primary
            .columns
            .div_ceil(64)
            .checked_add(
                secondary
                    .columns
                    .div_ceil(4)
                    .checked_mul(8)
                    .ok_or_else(|| Error::Limit("BF16 shared-input pair grid overflow".into()))?,
            )
            .ok_or_else(|| Error::Limit("BF16 shared-input pair grid overflow".into()))?;
        let x = u32::try_from(blocks)
            .ok()
            .filter(|blocks| *blocks <= i32::MAX as u32)
            .ok_or_else(|| Error::Limit("BF16 shared-input pair grid overflow".into()))?;
        let schema = StaticScalarSchema::new(
            &[
                (StaticScalarType::U64, 0),
                (StaticScalarType::U64, 8),
                (StaticScalarType::U64, 16),
                (StaticScalarType::U64, 24),
                (StaticScalarType::U64, 32),
                (StaticScalarType::U64, 40),
                (StaticScalarType::U64, 48),
                (StaticScalarType::U64, 56),
                (StaticScalarType::U64, 64),
                (StaticScalarType::U64, 72),
                (StaticScalarType::U64, 80),
                (StaticScalarType::U64, 88),
            ],
            96,
            8,
        )?;
        let scalars = StaticScalarBlock::new(
            schema,
            &[
                primary.columns,
                primary.rows,
                primary.inner,
                32,
                0,
                primary.columns,
                secondary.columns,
                secondary.rows,
                secondary.inner,
                secondary_terms,
                5,
                secondary.columns,
            ]
            .map(|value| StaticScalarValue::U64(value as u64)),
        )?;
        let kernel = PinnedKernel::new_with_static_scalars(
            "bf16_gemm_shared_input_pair_batch8_ordered_k64_certified",
            SOURCE.into(),
            128,
            vec![
                (primary.weight_bytes(), BufferIOAttr::ReadOnly),
                (secondary.weight_bytes(), BufferIOAttr::ReadOnly),
                (primary.input_bytes(), BufferIOAttr::ReadOnly),
                (primary.output_bytes(), BufferIOAttr::WriteOnly),
                (secondary.output_bytes(), BufferIOAttr::WriteOnly),
                (4, BufferIOAttr::ReadWrite),
            ]
            .into(),
            scalars,
        )?
        .with_binding_alignment(2)?;
        Ok(Self {
            kernel,
            grid: [x, 1, 1],
        })
    }

    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    #[allow(clippy::too_many_arguments)] // Keep both projection/output/status bindings explicit.
    pub(crate) fn prepare(
        &self,
        client: &Client,
        primary_weights: Handle,
        secondary_weights: Handle,
        input: Handle,
        primary_output: Handle,
        secondary_output: Handle,
        status: Handle,
    ) -> Result<PreparedPinnedLaunch> {
        if status.service != client.service_id() || status.size_in_used() != 4 {
            return Err(Error::State(
                "BF16 shared-input pair status service/extent".into(),
            ));
        }
        require_pair_disjoint(
            [&primary_weights, &secondary_weights, &input],
            [&primary_output, &secondary_output],
            &status,
        )?;
        self.kernel.prepare(
            client,
            self.grid,
            vec![
                primary_weights,
                secondary_weights,
                input,
                primary_output,
                secondary_output,
                status,
            ],
        )
    }
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
        Self::with_output_stride_mode(columns, rows, inner, reduction, output_stride, false)
    }

    pub(crate) fn with_reduction_certified(
        columns: usize,
        rows: usize,
        inner: usize,
        reduction: Bf16GemmReduction,
    ) -> Result<Self> {
        Self::with_output_stride_mode(columns, rows, inner, reduction, columns, true)
    }

    pub(crate) fn with_reduction_input_layout_certified(
        columns: usize,
        rows: usize,
        inner: usize,
        reduction: Bf16GemmReduction,
        input_layout: Bf16GemmInputLayout,
    ) -> Result<Self> {
        Self::with_output_stride_layout_mode(
            columns,
            rows,
            inner,
            reduction,
            columns,
            input_layout,
            true,
        )
    }

    pub(crate) fn with_reduction_input_layout(
        columns: usize,
        rows: usize,
        inner: usize,
        reduction: Bf16GemmReduction,
        input_layout: Bf16GemmInputLayout,
    ) -> Result<Self> {
        Self::with_output_stride_layout_mode(
            columns,
            rows,
            inner,
            reduction,
            columns,
            input_layout,
            false,
        )
    }

    pub(crate) fn with_output_stride_certified(
        columns: usize,
        rows: usize,
        inner: usize,
        reduction: Bf16GemmReduction,
        output_stride: usize,
    ) -> Result<Self> {
        Self::with_output_stride_mode(columns, rows, inner, reduction, output_stride, true)
    }

    pub(crate) fn with_output_stride_input_layout_certified(
        columns: usize,
        rows: usize,
        inner: usize,
        reduction: Bf16GemmReduction,
        output_stride: usize,
        input_layout: Bf16GemmInputLayout,
    ) -> Result<Self> {
        Self::with_output_stride_layout_mode(
            columns,
            rows,
            inner,
            reduction,
            output_stride,
            input_layout,
            true,
        )
    }

    fn with_output_stride_mode(
        columns: usize,
        rows: usize,
        inner: usize,
        reduction: Bf16GemmReduction,
        output_stride: usize,
        certified: bool,
    ) -> Result<Self> {
        Self::with_output_stride_layout_mode(
            columns,
            rows,
            inner,
            reduction,
            output_stride,
            Bf16GemmInputLayout::RowMajor,
            certified,
        )
    }

    fn with_output_stride_layout_mode(
        columns: usize,
        rows: usize,
        inner: usize,
        reduction: Bf16GemmReduction,
        output_stride: usize,
        input_layout: Bf16GemmInputLayout,
        certified: bool,
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
        let batch4_lane_strided = matches!(
            reduction,
            Bf16GemmReduction::LaneStrided {
                lanes: 16,
                terms: 512
            }
        ) && rows == 4;
        let batch8_lane_strided = matches!(
            reduction,
            Bf16GemmReduction::LaneStrided {
                lanes: 16,
                terms: 512
            }
        ) && rows == 8;
        let inner_major_batch2_lane_strided = input_layout == Bf16GemmInputLayout::InnerMajor
            && rows == 2
            && matches!(
                reduction,
                Bf16GemmReduction::LaneStrided {
                    lanes: 16,
                    terms: 256 | 512
                }
            );
        let inner_major_batch4_lane_strided = input_layout == Bf16GemmInputLayout::InnerMajor
            && rows == 4
            && matches!(
                reduction,
                Bf16GemmReduction::LaneStrided {
                    lanes: 16,
                    terms: 256 | 512
                }
            );
        let inner_major_batch8_lane_strided = input_layout == Bf16GemmInputLayout::InnerMajor
            && rows == 8
            && matches!(
                reduction,
                Bf16GemmReduction::LaneStrided {
                    lanes: 16,
                    terms: 256 | 512
                }
            );
        let inner_major_lane_strided_terms256 = input_layout == Bf16GemmInputLayout::InnerMajor
            && matches!(
                reduction,
                Bf16GemmReduction::LaneStrided {
                    lanes: 16,
                    terms: 256
                }
            );
        let batch_lane_blocked = matches!(
            reduction,
            Bf16GemmReduction::LaneBlocked {
                lanes: 32,
                terms: 32
            }
        ) && matches!(rows, 2 | 4 | 8);
        if input_layout == Bf16GemmInputLayout::InnerMajor
            && !inner_major_batch2_lane_strided
            && !inner_major_batch4_lane_strided
            && !inner_major_batch8_lane_strided
        {
            return Err(Error::Unsupported(
                "BF16 inner-major input currently requires B2/B4/B8 lane-strided 16x256/512 reduction"
                    .into(),
            ));
        }
        let ordered_narrow_k64 =
            reduction == Bf16GemmReduction::Ordered && rows == 8 && inner >= 2048;
        let segmented_narrow_k64 = matches!(
            reduction,
            Bf16GemmReduction::RoundedSegments { terms }
                if rows == 8 && inner >= 128 && terms.is_multiple_of(64)
        );
        let interleaved_narrow_k64 = matches!(
            reduction,
            Bf16GemmReduction::InterleavedPair { terms }
                if rows == 8 && inner >= 64 && inner.is_multiple_of(64) && terms.is_multiple_of(32)
        );
        let ordered_rows64 = reduction == Bf16GemmReduction::Ordered && rows > 64 && inner >= 2048;
        let mma_tile_rows = if rows <= 8 {
            8
        } else if ordered_rows64 {
            64
        } else {
            32
        };
        let (tile_columns, tile_rows) = match reduction {
            Bf16GemmReduction::LaneStrided { lanes, .. } if inner_major_batch2_lane_strided => {
                (128 / lanes, 2)
            }
            Bf16GemmReduction::LaneStrided { lanes, .. } if inner_major_batch4_lane_strided => {
                (128 / lanes, 4)
            }
            Bf16GemmReduction::LaneStrided { lanes, .. } if inner_major_batch8_lane_strided => {
                (128 / lanes, 8)
            }
            Bf16GemmReduction::LaneStrided { lanes, .. } if batch4_lane_strided => (128 / lanes, 4),
            Bf16GemmReduction::LaneStrided { lanes, .. } if batch8_lane_strided => (128 / lanes, 8),
            Bf16GemmReduction::LaneBlocked { lanes, .. } if batch_lane_blocked => {
                (128 / lanes, rows)
            }
            Bf16GemmReduction::LaneStrided { .. } | Bf16GemmReduction::LaneBlocked { .. } => {
                (128 / lanes, 1)
            }
            _ => (64, mma_tile_rows),
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
        let entry = if ordered_narrow_k64 {
            if certified {
                "bf16_gemm_ordered_narrow_k64_certified"
            } else {
                "bf16_gemm_ordered_narrow_k64"
            }
        } else if segmented_narrow_k64 {
            if certified {
                "bf16_gemm_segmented_narrow_k64_certified"
            } else {
                "bf16_gemm_segmented_narrow_k64"
            }
        } else if interleaved_narrow_k64 {
            if certified {
                "bf16_gemm_interleaved_narrow_k64_certified"
            } else {
                "bf16_gemm_interleaved_narrow_k64"
            }
        } else if ordered_rows64 {
            if certified {
                "bf16_gemm_ordered_rows64_certified"
            } else {
                "bf16_gemm_ordered_rows64"
            }
        } else if inner_major_batch2_lane_strided {
            if certified && inner_major_lane_strided_terms256 {
                "bf16_gemm_lane_strided_batch2_inner_major_terms256_certified"
            } else if inner_major_lane_strided_terms256 {
                "bf16_gemm_lane_strided_batch2_inner_major_terms256"
            } else if certified {
                "bf16_gemm_lane_strided_batch2_inner_major_certified"
            } else {
                "bf16_gemm_lane_strided_batch2_inner_major"
            }
        } else if inner_major_batch4_lane_strided {
            if certified && inner_major_lane_strided_terms256 {
                "bf16_gemm_lane_strided_batch4_inner_major_terms256_certified"
            } else if inner_major_lane_strided_terms256 {
                "bf16_gemm_lane_strided_batch4_inner_major_terms256"
            } else if certified {
                "bf16_gemm_lane_strided_batch4_inner_major_certified"
            } else {
                "bf16_gemm_lane_strided_batch4_inner_major"
            }
        } else if batch4_lane_strided {
            if certified {
                "bf16_gemm_lane_strided_batch4_certified"
            } else {
                "bf16_gemm_lane_strided_batch4"
            }
        } else if inner_major_batch8_lane_strided {
            if certified && inner_major_lane_strided_terms256 {
                "bf16_gemm_lane_strided_batch8_inner_major_terms256_certified"
            } else if inner_major_lane_strided_terms256 {
                "bf16_gemm_lane_strided_batch8_inner_major_terms256"
            } else if certified {
                "bf16_gemm_lane_strided_batch8_inner_major_certified"
            } else {
                "bf16_gemm_lane_strided_batch8_inner_major"
            }
        } else if batch8_lane_strided {
            if certified {
                "bf16_gemm_lane_strided_batch8_certified"
            } else {
                "bf16_gemm_lane_strided_batch8"
            }
        } else if batch_lane_blocked {
            match (certified, rows) {
                (false, 2) => "bf16_gemm_lane_blocked_batch2",
                (true, 2) => "bf16_gemm_lane_blocked_batch2_certified",
                (false, 4) => "bf16_gemm_lane_blocked_batch4",
                (true, 4) => "bf16_gemm_lane_blocked_batch4_certified",
                (false, 8) => "bf16_gemm_lane_blocked_batch8",
                (true, 8) => "bf16_gemm_lane_blocked_batch8_certified",
                _ => unreachable!("validated lane-blocked batch width"),
            }
        } else {
            match (certified, reduction, tile_rows == 8) {
                (false, Bf16GemmReduction::Ordered, true) => "bf16_gemm_narrow",
                (false, Bf16GemmReduction::Ordered, false) => "bf16_gemm_wide",
                (false, Bf16GemmReduction::InterleavedPair { .. }, true) => {
                    "bf16_gemm_interleaved_narrow"
                }
                (false, Bf16GemmReduction::InterleavedPair { .. }, false) => {
                    "bf16_gemm_interleaved_wide"
                }
                (false, Bf16GemmReduction::RoundedSegments { .. }, true) => {
                    "bf16_gemm_segmented_narrow"
                }
                (false, Bf16GemmReduction::RoundedSegments { .. }, false) => {
                    "bf16_gemm_segmented_wide"
                }
                (false, Bf16GemmReduction::LaneStrided { .. }, _) => "bf16_gemm_lane_strided",
                (false, Bf16GemmReduction::LaneBlocked { .. }, _) => "bf16_gemm_lane_blocked",
                (true, Bf16GemmReduction::Ordered, true) => "bf16_gemm_narrow_certified",
                (true, Bf16GemmReduction::Ordered, false) => "bf16_gemm_wide_certified",
                (true, Bf16GemmReduction::InterleavedPair { .. }, true) => {
                    "bf16_gemm_interleaved_narrow_certified"
                }
                (true, Bf16GemmReduction::InterleavedPair { .. }, false) => {
                    "bf16_gemm_interleaved_wide_certified"
                }
                (true, Bf16GemmReduction::RoundedSegments { .. }, true) => {
                    "bf16_gemm_segmented_narrow_certified"
                }
                (true, Bf16GemmReduction::RoundedSegments { .. }, false) => {
                    "bf16_gemm_segmented_wide_certified"
                }
                (true, Bf16GemmReduction::LaneStrided { .. }, _) => {
                    "bf16_gemm_lane_strided_certified"
                }
                (true, Bf16GemmReduction::LaneBlocked { .. }, _) => {
                    "bf16_gemm_lane_blocked_certified"
                }
            }
        };
        let mut bindings = vec![
            (shape.weight_bytes(), BufferIOAttr::ReadOnly),
            (shape.input_bytes(), BufferIOAttr::ReadOnly),
            (output_bytes, BufferIOAttr::WriteOnly),
        ];
        if certified {
            bindings.push((4, BufferIOAttr::ReadWrite));
        }
        let kernel = PinnedKernel::new_with_static_scalars(
            entry,
            SOURCE.into(),
            128,
            bindings.into(),
            scalars,
        )?
        .with_binding_alignment(match (input_layout, rows) {
            (Bf16GemmInputLayout::InnerMajor, 2) => 4,
            (Bf16GemmInputLayout::InnerMajor, 4) => 8,
            (Bf16GemmInputLayout::InnerMajor, 8) => 16,
            _ => 2,
        })?;
        Ok(Self {
            kernel,
            grid,
            certified,
        })
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
        if self.certified {
            return Err(Error::State(
                "certified BF16 GEMM requires explicit status binding".into(),
            ));
        }
        require_disjoint_output(&weights, &input, &output)?;
        self.kernel
            .prepare(client, self.grid, vec![weights, input, output])
    }

    pub(crate) fn prepare_certified(
        &self,
        client: &Client,
        weights: Handle,
        input: Handle,
        output: Handle,
        status: Handle,
    ) -> Result<PreparedPinnedLaunch> {
        if !self.certified {
            return Err(Error::State(
                "ordinary BF16 GEMM has no fused certificate".into(),
            ));
        }
        require_disjoint_output(&weights, &input, &output)?;
        if status.service != client.service_id()
            || status.size_in_used() != 4
            || output.service != client.service_id()
        {
            return Err(Error::State(
                "certified BF16 GEMM status service/extent".into(),
            ));
        }
        let output_range = range(&output)?;
        let status_range = range(&status)?;
        if output.service == status.service
            && output.memory.descriptor().id == status.memory.descriptor().id
            && output_range.0 < status_range.1
            && status_range.0 < output_range.1
        {
            return Err(Error::State(
                "certified BF16 GEMM output overlaps status".into(),
            ));
        }
        self.kernel
            .prepare(client, self.grid, vec![weights, input, output, status])
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

fn require_pair_disjoint(
    reads: [&Handle; 3],
    outputs: [&Handle; 2],
    status: &Handle,
) -> Result<()> {
    for output in outputs {
        let (os, oe) = range(output)?;
        for read in reads {
            let (rs, re) = range(read)?;
            if read.service == output.service
                && read.memory.descriptor().id == output.memory.descriptor().id
                && rs < oe
                && os < re
            {
                return Err(Error::State(
                    "BF16 shared-input pair output overlaps an input".into(),
                ));
            }
        }
    }
    let (a0, a1) = range(outputs[0])?;
    let (b0, b1) = range(outputs[1])?;
    if outputs[0].service == outputs[1].service
        && outputs[0].memory.descriptor().id == outputs[1].memory.descriptor().id
        && a0 < b1
        && b0 < a1
    {
        return Err(Error::State(
            "BF16 shared-input pair outputs overlap".into(),
        ));
    }
    let (ss, se) = range(status)?;
    for output in outputs {
        let (os, oe) = range(output)?;
        if output.service == status.service
            && output.memory.descriptor().id == status.memory.descriptor().id
            && os < se
            && ss < oe
        {
            return Err(Error::State(
                "BF16 shared-input pair output overlaps status".into(),
            ));
        }
    }
    Ok(())
}

fn range(handle: &Handle) -> Result<(u64, u64)> {
    let start = handle.offset_start.unwrap_or(0);
    let end = handle
        .size()
        .checked_sub(handle.offset_end.unwrap_or(0))
        .filter(|end| *end >= start)
        .ok_or_else(|| Error::State("BF16 GEMM invalid binding range".into()))?;
    Ok((start, end))
}

#[cfg(test)]
mod tests;
