//! Retained CUDA performance candidates for explicit cross-model qualification.
//!
//! Nothing in this module is selected by production serving. These launches are kept
//! so a negative A/B result on one model or shape can be revisited on another model
//! without reconstructing implementation code from historical notes.

use crate::{
    cuda::{
        bf16_gemm::{Bf16GemmShape, output_span},
        gdn::{CONV_STATE_BYTES, GdnBuffer, GdnPhase},
        kernel::{
            PinnedKernel, PreparedPinnedLaunch, StaticScalarBlock, StaticScalarSchema,
            StaticScalarType, StaticScalarValue,
        },
    },
    resident_program::BufferRange,
};
use cubecl::{client::Client, server::Handle};
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::{Arc, OnceLock};
use zerogravity_core::{Error, Result};

const THREADS: u32 = 128;
const MAX_WORDS: usize = 2 * 1024 * 1024;
const COPY_ZERO_TAIL_ENTRY: &str = "zg_candidate_copy_zero_tail_words_v1";
const COPY_ZERO_TAIL_SOURCE: &str = r#"// cubecl.cuda.dynamic_shared_bytes=0
#if __CUDA_ARCH__ != 860
#error Closed SM86 candidate copy-zero-tail transport
#endif
struct alignas(8) ZgCandidateCopyZeroTailScalars {
    unsigned long long words[4];
};
static_assert(sizeof(ZgCandidateCopyZeroTailScalars) == 32, "candidate scalar ABI");
static_assert(alignof(ZgCandidateCopyZeroTailScalars) == 8, "candidate scalar alignment");
extern "C" __global__ __launch_bounds__(128) void zg_candidate_copy_zero_tail_words_v1(
    const unsigned int* source,
    unsigned int* destination,
    ZgCandidateCopyZeroTailScalars scalars) {
    const unsigned int copy_words = (unsigned int)scalars.words[0];
    const unsigned int span_words = (unsigned int)scalars.words[1];
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (copy_words != 0u && copy_words <= span_words && index < span_words) {
        destination[index] = index < copy_words ? source[index] : 0u;
    }
}
"#;

/// Actual retained implementation of the rejected paged-append candidate:
/// copy an exact active prefix and zero the remaining destination tail in one
/// device launch. It owns no model, page, StateRoot, allocation or publication
/// semantics.
///
/// The Qwen .8B refill A/B produced only ~0.3% paged gain, so this is qualification
/// only. Other models/shapes may have a different launch/bandwidth balance.
pub fn launch_copy_zero_tail(
    client: &Client,
    source: &Handle,
    destination: &Handle,
    copy_bytes: usize,
    span_bytes: usize,
) -> Result<usize> {
    if copy_bytes == 0
        || span_bytes == 0
        || !copy_bytes.is_multiple_of(4)
        || !span_bytes.is_multiple_of(4)
        || copy_bytes > span_bytes
    {
        return Err(Error::Malformed(
            "candidate copy-zero-tail geometry differs".into(),
        ));
    }
    let copy_words = copy_bytes / 4;
    let span_words = span_bytes / 4;
    if span_words > MAX_WORDS {
        return Err(Error::Limit(
            "candidate copy-zero-tail span exceeds transport bound".into(),
        ));
    }

    let source_range = BufferRange::new(source)?;
    let destination_range = BufferRange::new(destination)?;
    source_range.require_exact_len(copy_bytes)?;
    destination_range.require_exact_len(span_bytes)?;
    if source_range.overlaps(&destination_range) {
        return Err(Error::State(
            "candidate copy-zero-tail source/destination alias".into(),
        ));
    }

    let schema = StaticScalarSchema::new(
        &[
            (StaticScalarType::U32, 0),
            (StaticScalarType::U32, 8),
            (StaticScalarType::U32, 16),
            (StaticScalarType::U32, 24),
        ],
        32,
        8,
    )?;
    let as_u32 = |value: usize, role: &str| {
        u32::try_from(value).map_err(|_| Error::Limit(format!("candidate {role} u32")))
    };
    let scalars = StaticScalarBlock::new(
        schema,
        &[
            StaticScalarValue::U32(u64::from(as_u32(copy_words, "copy words")?)),
            StaticScalarValue::U32(u64::from(as_u32(span_words, "span words")?)),
            StaticScalarValue::U32(0),
            StaticScalarValue::U32(0),
        ],
    )?;
    let kernel = PinnedKernel::new_with_static_scalars(
        COPY_ZERO_TAIL_ENTRY,
        Arc::from(COPY_ZERO_TAIL_SOURCE),
        THREADS,
        vec![
            (copy_bytes, BufferIOAttr::ReadOnly),
            (span_bytes, BufferIOAttr::WriteOnly),
        ]
        .into(),
        scalars,
    )?;
    let grid = [as_u32(span_words, "launch words")?.div_ceil(THREADS), 1, 1];
    kernel
        .prepare(client, grid, vec![source.clone(), destination.clone()])?
        .launch()?;
    Ok(1)
}

const BF16_GEMM_BASE_SOURCE: &str = include_str!("bf16_gemm/kernels.cu");
const BF16_GEMM_CANDIDATE_EXTENSION: &str =
    include_str!("performance_candidates/bf16_gemm_candidates.cu");

fn bf16_gemm_candidate_source() -> Arc<str> {
    static SOURCE: OnceLock<Arc<str>> = OnceLock::new();
    Arc::clone(SOURCE.get_or_init(|| {
        let mut source = String::with_capacity(
            BF16_GEMM_BASE_SOURCE.len() + BF16_GEMM_CANDIDATE_EXTENSION.len() + 1,
        );
        source.push_str(BF16_GEMM_BASE_SOURCE);
        source.push('\n');
        source.push_str(BF16_GEMM_CANDIDATE_EXTENSION);
        Arc::<str>::from(source)
    }))
}

fn transform_bf16_gemm_section(
    start: &str,
    end: &str,
    from: &str,
    to: &str,
    label: &str,
) -> Result<Arc<str>> {
    let begin = BF16_GEMM_BASE_SOURCE
        .find(start)
        .ok_or_else(|| Error::State(format!("{label} start marker differs")))?;
    let section_end = BF16_GEMM_BASE_SOURCE[begin..]
        .find(end)
        .map(|offset| begin + offset)
        .ok_or_else(|| Error::State(format!("{label} end marker differs")))?;
    let section = &BF16_GEMM_BASE_SOURCE[begin..section_end];
    if section.matches(from).count() != 1 {
        return Err(Error::State(format!("{label} transform marker differs")));
    }
    let replaced = section.replacen(from, to, 1);
    let mut source = String::with_capacity(
        BF16_GEMM_BASE_SOURCE.len() + replaced.len().saturating_sub(section.len()),
    );
    source.push_str(&BF16_GEMM_BASE_SOURCE[..begin]);
    source.push_str(&replaced);
    source.push_str(&BF16_GEMM_BASE_SOURCE[section_end..]);
    Ok(Arc::from(source))
}

fn bf16_gemm_paired_boundary_source() -> Result<Arc<str>> {
    static SOURCE: OnceLock<Arc<str>> = OnceLock::new();
    if let Some(source) = SOURCE.get() {
        return Ok(Arc::clone(source));
    }
    const FROM: &str = r#"            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                rounded[i] += widen_bf16(round_bf16(d[i]));
                d[i] = 0;
            }"#;
    const TO: &str = r#"            unsigned pair01, pair23;
            asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
                : "=r"(pair01) : "f"(d[1]), "f"(d[0]));
            asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
                : "=r"(pair23) : "f"(d[3]), "f"(d[2]));
            rounded[0] += widen_bf16(static_cast<unsigned short>(pair01));
            rounded[1] += widen_bf16(static_cast<unsigned short>(pair01 >> 16));
            rounded[2] += widen_bf16(static_cast<unsigned short>(pair23));
            rounded[3] += widen_bf16(static_cast<unsigned short>(pair23 >> 16));
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) d[i] = 0;"#;
    let source = transform_bf16_gemm_section(
        "__device__ __forceinline__ void gemm_segmented_narrow_k64_body(",
        "extern \"C\" __global__ __launch_bounds__(128) void bf16_gemm_segmented_narrow_k64(",
        FROM,
        TO,
        "paired BF16 segment boundary",
    )?;
    Ok(Arc::clone(SOURCE.get_or_init(|| source)))
}

fn bf16_gemm_weight_cg_source() -> Result<Arc<str>> {
    static SOURCE: OnceLock<Arc<str>> = OnceLock::new();
    if let Some(source) = SOURCE.get() {
        return Ok(Arc::clone(source));
    }
    const FROM: &str = r#"        if (aligned) {
            copy16(target, source, valid * 2);
        } else {"#;
    const TO: &str = r#"        if (aligned) {
            if (weight) {
                unsigned address =
                    static_cast<unsigned>(__cvta_generic_to_shared(target));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                             :: "r"(address), "l"(source), "r"(valid * 2));
            } else {
                copy16(target, source, valid * 2);
            }
        } else {"#;
    let source = transform_bf16_gemm_section(
        "template<int BN>\n__device__ __forceinline__ void stage64(",
        "__device__ __forceinline__ float widen_bf16(",
        FROM,
        TO,
        "K64 weight-cache policy",
    )?;
    Ok(Arc::clone(SOURCE.get_or_init(|| source)))
}

/// Dormant exact-arithmetic physical specializations recovered from earlier A/B work.
///
/// None is a production selector. Each keeps the accepted reduction order and varies
/// only storage/load/CTA/tile mechanics so other models can remeasure the economics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16GemmCandidateKind {
    LaneBlockedInnerMajor,
    LaneBlockedSharedRowsTerms64,
    LaneStridedBatch8Threads64,
    LaneStridedBatch8Threads256,
    RoundedSegmentsBatch8M32K64 {
        terms: usize,
    },
    RoundedSegmentsBatch8PairedBoundary {
        terms: usize,
    },
    OrderedBatch8K64WeightCg,
    RoundedSegmentsBatch8K64WeightCg {
        terms: usize,
    },
    LaneStridedBatch8PairedUnpack,
    OrderedBatch8M128K64,
    OrderedN16K64,
    /// Reconstructed from the recorded shared-input/widening strategy; the
    /// original uncommitted working-tree source did not survive Git recovery.
    LaneStridedBatch8RowMajorSharedInputF32,
    LaneStridedBatch8RowMajorPairedColumns,
    LaneStridedBatch8RowMajorSplitRows4,
    LaneStridedBatch8InnerMajorWarpShared,
    LaneStridedBatch8InnerMajorPairedColumns,
    LaneStridedBatch8InnerMajorUnroll8,
}

pub struct Bf16GemmCandidate {
    kernel: PinnedKernel,
    grid: [u32; 3],
}

impl Bf16GemmCandidate {
    pub fn new(
        kind: Bf16GemmCandidateKind,
        columns: usize,
        rows: usize,
        inner: usize,
        output_stride: usize,
    ) -> Result<Self> {
        let shape = Bf16GemmShape::new(columns, rows, inner)?;
        let output_bytes = output_span(columns, rows, output_stride)?;
        let source = match kind {
            Bf16GemmCandidateKind::RoundedSegmentsBatch8PairedBoundary { .. } => {
                bf16_gemm_paired_boundary_source()?
            }
            Bf16GemmCandidateKind::OrderedBatch8K64WeightCg
            | Bf16GemmCandidateKind::RoundedSegmentsBatch8K64WeightCg { .. } => {
                bf16_gemm_weight_cg_source()?
            }
            _ => bf16_gemm_candidate_source(),
        };
        let (entry, threads, terms, lane_shift, tile_columns, tile_rows, alignment) = match kind {
            Bf16GemmCandidateKind::LaneBlockedInnerMajor if matches!(rows, 2 | 4 | 8) => (
                match rows {
                    2 => "zg_candidate_lane_blocked_inner_major_batch2_certified",
                    4 => "zg_candidate_lane_blocked_inner_major_batch4_certified",
                    8 => "zg_candidate_lane_blocked_inner_major_batch8_certified",
                    _ => unreachable!(),
                },
                128u32,
                32usize,
                5usize,
                4usize,
                rows,
                rows * 2,
            ),
            Bf16GemmCandidateKind::LaneBlockedSharedRowsTerms64 if matches!(rows, 2 | 4 | 8) => (
                match rows {
                    2 => "zg_candidate_lane_blocked_batch2_terms64_certified",
                    4 => "zg_candidate_lane_blocked_batch4_terms64_certified",
                    8 => "zg_candidate_lane_blocked_batch8_terms64_certified",
                    _ => unreachable!(),
                },
                128,
                64,
                5,
                4,
                rows,
                2,
            ),
            Bf16GemmCandidateKind::LaneStridedBatch8Threads64 if rows == 8 => (
                "zg_candidate_lane_strided_b8_inner_major_t64_certified",
                64,
                512,
                4,
                4,
                8,
                16,
            ),
            Bf16GemmCandidateKind::LaneStridedBatch8Threads256 if rows == 8 => (
                "zg_candidate_lane_strided_b8_inner_major_t256_certified",
                256,
                512,
                4,
                16,
                8,
                16,
            ),
            Bf16GemmCandidateKind::RoundedSegmentsBatch8M32K64 { terms }
                if rows == 8 && terms != 0 && terms.is_multiple_of(64) && inner >= 64 =>
            {
                (
                    "zg_candidate_segmented_b8_m32_k64_certified",
                    64,
                    terms,
                    0,
                    32,
                    8,
                    2,
                )
            }
            Bf16GemmCandidateKind::RoundedSegmentsBatch8PairedBoundary { terms }
                if rows == 8 && terms != 0 && terms.is_multiple_of(64) && inner >= 64 =>
            {
                (
                    "bf16_gemm_segmented_narrow_k64_certified",
                    128,
                    terms,
                    0,
                    64,
                    8,
                    16,
                )
            }
            Bf16GemmCandidateKind::OrderedBatch8K64WeightCg if rows == 8 && inner >= 64 => (
                "bf16_gemm_ordered_narrow_k64_certified",
                128,
                32,
                0,
                64,
                8,
                16,
            ),
            Bf16GemmCandidateKind::RoundedSegmentsBatch8K64WeightCg { terms }
                if rows == 8 && terms != 0 && terms.is_multiple_of(64) && inner >= 64 =>
            {
                (
                    "bf16_gemm_segmented_narrow_k64_certified",
                    128,
                    terms,
                    0,
                    64,
                    8,
                    16,
                )
            }
            Bf16GemmCandidateKind::LaneStridedBatch8PairedUnpack if rows == 8 => (
                "zg_candidate_lane_strided_b8_paired_unpack_certified",
                128,
                512,
                4,
                8,
                8,
                16,
            ),
            Bf16GemmCandidateKind::OrderedBatch8M128K64 if rows == 8 && inner >= 2048 => (
                "zg_candidate_ordered_b8_m128_k64_certified",
                256,
                32,
                0,
                128,
                8,
                16,
            ),
            Bf16GemmCandidateKind::OrderedN16K64 if rows == 16 && inner >= 2048 => (
                "zg_candidate_ordered_n16_k64_certified",
                128,
                32,
                0,
                64,
                16,
                16,
            ),
            Bf16GemmCandidateKind::LaneStridedBatch8RowMajorSharedInputF32 if rows == 8 => (
                "zg_candidate_lane_strided_b8_row_major_shared_f32_certified",
                128,
                512,
                4,
                8,
                8,
                2,
            ),
            Bf16GemmCandidateKind::LaneStridedBatch8RowMajorPairedColumns if rows == 8 => (
                "zg_candidate_lane_strided_b8_row_major_paired_columns_certified",
                128,
                512,
                4,
                16,
                8,
                2,
            ),
            Bf16GemmCandidateKind::LaneStridedBatch8RowMajorSplitRows4 if rows == 8 => (
                "zg_candidate_lane_strided_b8_row_major_split4_certified",
                128,
                512,
                4,
                4,
                8,
                2,
            ),
            Bf16GemmCandidateKind::LaneStridedBatch8InnerMajorWarpShared if rows == 8 => (
                "zg_candidate_lane_strided_b8_inner_major_warp_shared_certified",
                128,
                512,
                4,
                8,
                8,
                16,
            ),
            Bf16GemmCandidateKind::LaneStridedBatch8InnerMajorPairedColumns if rows == 8 => (
                "zg_candidate_lane_strided_b8_inner_major_paired_columns_certified",
                128,
                512,
                4,
                16,
                8,
                16,
            ),
            Bf16GemmCandidateKind::LaneStridedBatch8InnerMajorUnroll8 if rows == 8 => (
                "zg_candidate_lane_strided_b8_inner_major_unroll8_certified",
                128,
                512,
                4,
                8,
                8,
                16,
            ),
            _ => {
                return Err(Error::Unsupported(
                    "BF16 GEMM candidate geometry is outside its retained A/B contract".into(),
                ));
            }
        };

        let grid = [
            u32::try_from(columns.div_ceil(tile_columns))
                .ok()
                .filter(|value| *value <= i32::MAX as u32)
                .ok_or_else(|| Error::Limit("candidate BF16 GEMM x grid".into()))?,
            u32::try_from(rows.div_ceil(tile_rows))
                .ok()
                .filter(|value| *value <= 65_535)
                .ok_or_else(|| Error::Limit("candidate BF16 GEMM y grid".into()))?,
            1,
        ];
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
            &[columns, rows, inner, terms, lane_shift, output_stride]
                .map(|value| StaticScalarValue::U64(value as u64)),
        )?;
        let kernel = PinnedKernel::new_with_static_scalars(
            entry,
            source,
            threads,
            vec![
                (shape.weight_bytes(), BufferIOAttr::ReadOnly),
                (shape.input_bytes(), BufferIOAttr::ReadOnly),
                (output_bytes, BufferIOAttr::WriteOnly),
                (4, BufferIOAttr::ReadWrite),
            ]
            .into(),
            scalars,
        )?
        .with_binding_alignment(alignment)?;

        Ok(Self { kernel, grid })
    }

    pub fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    pub fn prepare_certified(
        &self,
        client: &Client,
        weights: Handle,
        input: Handle,
        output: Handle,
        status: Handle,
    ) -> Result<PreparedPinnedLaunch> {
        let inputs = [BufferRange::new(&weights)?, BufferRange::new(&input)?];
        let output_range = BufferRange::new(&output)?;
        let status_range = BufferRange::new(&status)?;
        if inputs.iter().any(|range| range.overlaps(&output_range))
            || inputs.iter().any(|range| range.overlaps(&status_range))
            || output_range.overlaps(&status_range)
        {
            return Err(Error::State(
                "candidate BF16 GEMM writable binding aliases another binding".into(),
            ));
        }
        self.kernel
            .prepare(client, self.grid, vec![weights, input, output, status])
    }
}

const RECURRENT_U32_ELEMENTS: usize = 16 * 128 * 128;
const RECURRENT_U32_BYTES: usize = RECURRENT_U32_ELEMENTS * 4;
const RECURRENT_TRANSPOSE_SOURCE: &str =
    include_str!("performance_candidates/recurrent_transpose.cu");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecurrentTransposeDirection {
    CanonicalToNative,
    NativeToCanonical,
}

/// Retained 32x32 padded shared-memory U32 transpose used by the rejected
/// recurrent-state transport A/B. The transpose is its own inverse; direction
/// only selects the original nonfinite status bit.
pub struct RecurrentTransposeCandidate {
    kernel: PinnedKernel,
    grid: [u32; 3],
}

impl RecurrentTransposeCandidate {
    pub fn new(direction: RecurrentTransposeDirection) -> Result<Self> {
        let status_mask = match direction {
            RecurrentTransposeDirection::CanonicalToNative => 1u64 << 2,
            RecurrentTransposeDirection::NativeToCanonical => 1u64 << 3,
        };
        let scalars = StaticScalarBlock::new(
            StaticScalarSchema::new(&[(StaticScalarType::U32, 0)], 8, 8)?,
            &[StaticScalarValue::U32(status_mask)],
        )?;
        Ok(Self {
            kernel: PinnedKernel::new_with_static_scalars(
                "zg_candidate_recurrent_transpose_u32",
                Arc::from(RECURRENT_TRANSPOSE_SOURCE),
                256,
                vec![
                    (RECURRENT_U32_BYTES, BufferIOAttr::ReadOnly),
                    (RECURRENT_U32_BYTES, BufferIOAttr::WriteOnly),
                    (4, BufferIOAttr::ReadWrite),
                ]
                .into(),
                scalars,
            )?
            .with_binding_alignment(4)?,
            grid: [4, 64, 1],
        })
    }

    pub fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    pub fn prepare(
        &self,
        client: &Client,
        source: Handle,
        destination: Handle,
        status: Handle,
    ) -> Result<PreparedPinnedLaunch> {
        let source_range = BufferRange::new(&source)?;
        let destination_range = BufferRange::new(&destination)?;
        let status_range = BufferRange::new(&status)?;
        source_range.require_exact_len(RECURRENT_U32_BYTES)?;
        destination_range.require_exact_len(RECURRENT_U32_BYTES)?;
        status_range.require_exact_len(4)?;
        if source_range.overlaps(&destination_range)
            || source_range.overlaps(&status_range)
            || destination_range.overlaps(&status_range)
        {
            return Err(Error::State(
                "candidate recurrent transpose writable binding aliases another binding".into(),
            ));
        }
        self.kernel
            .prepare(client, self.grid, vec![source, destination, status])
    }
}

const CACHED_GDN_DECODE_SOURCE: &str = include_str!("gdn/text/cached_decode_parameterized.cu");
const CACHED_GDN_DECODE_ENTRY: &str = "zg_gdn_decode_cached_parameterized_v1";
const CACHED_GDN_DECODE_SCALE_BITS: u32 = 1_035_273_459;

/// Retained compiler-occupancy experiment over the frozen cached-GDN decode
/// source. Only the launch-bounds min-block hint changes; body, ABI, grid and
/// scalar packet remain byte-for-byte derived from the accepted source.
pub struct CachedGdnOccupancyCandidate {
    kernel: PinnedKernel,
    grid: [u32; 3],
}

impl CachedGdnOccupancyCandidate {
    pub fn new(min_blocks_per_sm: usize) -> Result<Self> {
        if !matches!(min_blocks_per_sm, 8 | 16 | 24) {
            return Err(Error::Unsupported(
                "cached GDN occupancy candidate requires min blocks 8, 16 or 24".into(),
            ));
        }
        const FROM: &str = "__launch_bounds__(32)";
        if CACHED_GDN_DECODE_SOURCE.matches(FROM).count() != 1 {
            return Err(Error::State(
                "cached GDN launch-bounds source marker differs".into(),
            ));
        }
        let source = CACHED_GDN_DECODE_SOURCE.replacen(
            FROM,
            &format!("__launch_bounds__(32, {min_blocks_per_sm})"),
            1,
        );
        let phase = GdnPhase::CachedDecode;
        let cache_tail = GdnBuffer::CacheArena
            .bytes(phase)
            .checked_sub(CONV_STATE_BYTES)
            .ok_or_else(|| Error::State("cached GDN candidate cache tail extent".into()))?;
        let scalars = StaticScalarBlock::new(
            StaticScalarSchema::new(&[(StaticScalarType::F32Bits, 0)], 8, 8)?,
            &[StaticScalarValue::F32Bits(CACHED_GDN_DECODE_SCALE_BITS)],
        )?;
        use BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
        Ok(Self {
            kernel: PinnedKernel::new_with_static_scalars(
                CACHED_GDN_DECODE_ENTRY,
                source.into(),
                32,
                vec![
                    (GdnBuffer::ConvOutput.bytes(phase), R),
                    (32, R),
                    (32, R),
                    (GdnBuffer::ALog.bytes(phase), R),
                    (GdnBuffer::DtBias.bytes(phase), R),
                    (GdnBuffer::Core.bytes(phase), W),
                    (cache_tail, RW),
                    (cache_tail, RW),
                    (GdnBuffer::CacheIndices.bytes(phase), R),
                ]
                .into(),
                scalars,
            )?,
            grid: [4, 16, 1],
        })
    }

    pub fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }

    pub fn prepare(&self, client: &Client, handles: [Handle; 9]) -> Result<PreparedPinnedLaunch> {
        self.kernel.prepare(client, self.grid, handles.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_geometry_rejects_invalid_or_unbounded_spans() {
        assert_eq!(COPY_ZERO_TAIL_ENTRY, "zg_candidate_copy_zero_tail_words_v1");
        assert!(!COPY_ZERO_TAIL_SOURCE.is_empty());
        assert_eq!(MAX_WORDS, 2 * 1024 * 1024);
    }

    #[test]
    fn recovered_bf16_candidates_keep_bounded_constructor_contracts() {
        for (kind, columns, rows, inner) in [
            (
                Bf16GemmCandidateKind::LaneBlockedSharedRowsTerms64,
                256,
                8,
                2048,
            ),
            (
                Bf16GemmCandidateKind::RoundedSegmentsBatch8PairedBoundary { terms: 2048 },
                256,
                8,
                6144,
            ),
            (
                Bf16GemmCandidateKind::OrderedBatch8K64WeightCg,
                8192,
                8,
                2048,
            ),
            (
                Bf16GemmCandidateKind::RoundedSegmentsBatch8K64WeightCg { terms: 2048 },
                2048,
                8,
                6144,
            ),
        ] {
            Bf16GemmCandidate::new(kind, columns, rows, inner, columns).unwrap();
        }
    }
}
