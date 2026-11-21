//! Retained CUDA performance candidates for explicit cross-model qualification.
//!
//! Nothing in this module is selected by production serving. These launches are kept
//! so a negative A/B result on one model or shape can be revisited on another model
//! without reconstructing implementation code from historical notes.

use crate::{
    cuda::kernel::{
        PinnedKernel, StaticScalarBlock, StaticScalarSchema, StaticScalarType, StaticScalarValue,
    },
    resident_program::BufferRange,
};
use cubecl::{client::Client, server::Handle};
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::Arc;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_geometry_rejects_invalid_or_unbounded_spans() {
        assert_eq!(COPY_ZERO_TAIL_ENTRY, "zg_candidate_copy_zero_tail_words_v1");
        assert!(!COPY_ZERO_TAIL_SOURCE.is_empty());
        assert_eq!(MAX_WORDS, 2 * 1024 * 1024);
    }
}
