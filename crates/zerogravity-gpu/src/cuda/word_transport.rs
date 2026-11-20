//! Source-neutral runtime-geometry u32 transport for pinned CUDA programs.
//!
//! Logical/model layers own roles, offsets, state and publication authority.
//! This primitive only copies or zeros checked word rows inside caller-supplied
//! bounded views. Rows/width/strides are runtime scalar values so arbitrary
//! ragged geometry reuses one compiled kernel/ABI instead of minting code.

use crate::{
    cuda::kernel::{
        PinnedKernel, StaticScalarBlock, StaticScalarSchema, StaticScalarType, StaticScalarValue,
    },
    program_kernels,
};
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::Arc;
use zerogravity_core::{Error, Result};

const THREADS: u32 = 128;
pub(crate) const MAX_RUNTIME_WORD_TRANSPORT_WORDS: usize = 2 * 1024 * 1024;
pub(crate) const MAX_PAGED_PAIR_GATHER_PAGES: usize = 15;
const COPY_ENTRY: &str = "zg_copy_strided_words_runtime_v1";
const ZERO_ENTRY: &str = "zg_zero_strided_words_runtime_v1";
const PAGED_PAIR_GATHER_ENTRY: &str = "zg_gather_paged_pair_words_runtime_v1";
const SOURCE: &str = r#"// cubecl.cuda.dynamic_shared_bytes=0
#if __CUDA_ARCH__ != 860
#error Closed SM86 runtime word transport
#endif
struct alignas(8) ZgWordTransportScalars {
    unsigned long long words[4];
};
static_assert(sizeof(ZgWordTransportScalars) == 32, "ZgWordTransportScalars ABI");
static_assert(alignof(ZgWordTransportScalars) == 8, "ZgWordTransportScalars alignment");
extern "C" __global__ __launch_bounds__(128) void zg_copy_strided_words_runtime_v1(
    const unsigned int* source,
    unsigned int* destination,
    ZgWordTransportScalars scalars) {
    const unsigned int rows = (unsigned int)scalars.words[0];
    const unsigned int width = (unsigned int)scalars.words[1];
    const unsigned int source_stride = (unsigned int)scalars.words[2];
    const unsigned int destination_stride = (unsigned int)scalars.words[3];
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)rows * width;
    if (width != 0u && index < total) {
        const unsigned long long row = index / width;
        const unsigned long long column = index - row * width;
        destination[row * destination_stride + column] =
            source[row * source_stride + column];
    }
}
extern "C" __global__ __launch_bounds__(128) void zg_zero_strided_words_runtime_v1(
    unsigned int* destination,
    ZgWordTransportScalars scalars) {
    const unsigned int rows = (unsigned int)scalars.words[0];
    const unsigned int width = (unsigned int)scalars.words[1];
    const unsigned int destination_stride = (unsigned int)scalars.words[3];
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)rows * width;
    if (width != 0u && index < total) {
        const unsigned long long row = index / width;
        const unsigned long long column = index - row * width;
        destination[row * destination_stride + column] = 0u;
    }
}
"#;

const PAGED_PAIR_GATHER_SOURCE: &str = r#"// cubecl.cuda.dynamic_shared_bytes=0
#if __CUDA_ARCH__ != 860
#error Closed SM86 runtime paged pair gather
#endif
struct alignas(8) ZgPagedPairGatherScalars {
    unsigned long long words[4];
};
static_assert(sizeof(ZgPagedPairGatherScalars) == 32, "ZgPagedPairGatherScalars ABI");
static_assert(alignof(ZgPagedPairGatherScalars) == 8, "ZgPagedPairGatherScalars alignment");
extern "C" __global__ __launch_bounds__(128) void zg_gather_paged_pair_words_runtime_v1(
    const unsigned int* key0, const unsigned int* key1, const unsigned int* key2,
    const unsigned int* key3, const unsigned int* key4, const unsigned int* key5,
    const unsigned int* key6, const unsigned int* key7, const unsigned int* key8,
    const unsigned int* key9, const unsigned int* key10, const unsigned int* key11,
    const unsigned int* key12, const unsigned int* key13, const unsigned int* key14,
    const unsigned int* value0, const unsigned int* value1, const unsigned int* value2,
    const unsigned int* value3, const unsigned int* value4, const unsigned int* value5,
    const unsigned int* value6, const unsigned int* value7, const unsigned int* value8,
    const unsigned int* value9, const unsigned int* value10, const unsigned int* value11,
    const unsigned int* value12, const unsigned int* value13, const unsigned int* value14,
    unsigned int* key_destination, unsigned int* value_destination,
    ZgPagedPairGatherScalars scalars) {
    const unsigned int page_words = (unsigned int)scalars.words[0];
    const unsigned int copy_words = (unsigned int)scalars.words[1];
    const unsigned int destination_offset = (unsigned int)scalars.words[2];
    const unsigned int span_words = (unsigned int)scalars.words[3];
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)span_words * 2ull;
    if (page_words == 0u || copy_words == 0u || span_words < copy_words || index >= total) {
        return;
    }
    const unsigned int plane = (unsigned int)(index / span_words);
    const unsigned int local =
        (unsigned int)(index - (unsigned long long)plane * span_words);
    unsigned int* destination = plane == 0u ? key_destination : value_destination;
    if (local >= copy_words) {
        destination[destination_offset + local] = 0u;
        return;
    }
    const unsigned int page = local / page_words;
    const unsigned int column = local - page * page_words;
    if (page >= 15u) {
        return;
    }
    const unsigned int* source = nullptr;
#define ZG_PAGE_CASE(i) case i: source = plane == 0u ? key##i : value##i; break
    switch (page) {
        ZG_PAGE_CASE(0); ZG_PAGE_CASE(1); ZG_PAGE_CASE(2); ZG_PAGE_CASE(3);
        ZG_PAGE_CASE(4); ZG_PAGE_CASE(5); ZG_PAGE_CASE(6); ZG_PAGE_CASE(7);
        ZG_PAGE_CASE(8); ZG_PAGE_CASE(9); ZG_PAGE_CASE(10); ZG_PAGE_CASE(11);
        ZG_PAGE_CASE(12); ZG_PAGE_CASE(13); ZG_PAGE_CASE(14);
        default: return;
    }
#undef ZG_PAGE_CASE
    destination[destination_offset + local] = source[column];
}
"#;

pub(crate) struct RuntimeWordTransport {
    pub(crate) entry: &'static str,
    pub(crate) source: &'static str,
    pub(crate) grid: [u32; 3],
    pub(crate) kernel: PinnedKernel,
}

pub(crate) struct RuntimePagedPairGather {
    pub(crate) grid: [u32; 3],
    pub(crate) kernel: PinnedKernel,
    pages: usize,
}

impl RuntimePagedPairGather {
    pub(crate) fn can_fill_span(span_bytes: usize) -> bool {
        span_bytes != 0
            && span_bytes.is_multiple_of(4)
            && span_bytes / 4 <= MAX_RUNTIME_WORD_TRANSPORT_WORDS / 2
    }

    pub(crate) fn max_pages(page_bytes: usize) -> Result<usize> {
        if page_bytes == 0 || !page_bytes.is_multiple_of(4) {
            return Err(Error::State(
                "runtime paged pair gather page extent differs".into(),
            ));
        }
        let max_pair_bytes = MAX_RUNTIME_WORD_TRANSPORT_WORDS
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("runtime paged pair gather byte bound".into()))?;
        let max_plane_bytes = max_pair_bytes / 2;
        let pages = (max_plane_bytes / page_bytes).min(MAX_PAGED_PAIR_GATHER_PAGES);
        if pages == 0 {
            return Err(Error::Limit(
                "runtime paged pair gather page exceeds transport bound".into(),
            ));
        }
        Ok(pages)
    }

    pub(crate) fn new(
        page_bytes: usize,
        destination_bytes: usize,
        pages: usize,
        copy_bytes: usize,
        destination_offset_bytes: usize,
        span_bytes: usize,
    ) -> Result<Self> {
        if pages == 0 || pages > Self::max_pages(page_bytes)? {
            return Err(Error::State(
                "runtime paged pair gather page count differs".into(),
            ));
        }
        let page_span = pages
            .checked_mul(page_bytes)
            .ok_or_else(|| Error::Limit("runtime paged pair gather page span".into()))?;
        let prior_span = (pages - 1)
            .checked_mul(page_bytes)
            .ok_or_else(|| Error::Limit("runtime paged pair gather prior span".into()))?;
        if page_bytes == 0
            || destination_bytes == 0
            || copy_bytes == 0
            || !page_bytes.is_multiple_of(4)
            || !destination_bytes.is_multiple_of(4)
            || !copy_bytes.is_multiple_of(4)
            || !destination_offset_bytes.is_multiple_of(4)
            || !Self::can_fill_span(span_bytes)
            || copy_bytes > span_bytes
            || copy_bytes > page_span
            || copy_bytes <= prior_span
            || destination_offset_bytes
                .checked_add(span_bytes)
                .is_none_or(|end| end > destination_bytes)
        {
            return Err(Error::State(
                "runtime paged pair gather geometry differs".into(),
            ));
        }
        let page_words = page_bytes / 4;
        let copy_words = copy_bytes / 4;
        let destination_offset_words = destination_offset_bytes / 4;
        let span_words = span_bytes / 4;
        let total_words = span_words
            .checked_mul(2)
            .filter(|words| *words <= MAX_RUNTIME_WORD_TRANSPORT_WORDS)
            .ok_or_else(|| Error::Limit("runtime paged pair gather words".into()))?;
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
        let scalars = StaticScalarBlock::new(
            schema,
            &[
                StaticScalarValue::U32(as_u32(page_words, "paged page words")?),
                StaticScalarValue::U32(as_u32(copy_words, "paged copy words")?),
                StaticScalarValue::U32(as_u32(
                    destination_offset_words,
                    "paged destination offset",
                )?),
                StaticScalarValue::U32(as_u32(span_words, "paged destination span")?),
            ],
        )?;
        let mut bindings = Vec::with_capacity(32);
        bindings.extend(std::iter::repeat_n(
            (page_bytes, BufferIOAttr::ReadOnly),
            2 * MAX_PAGED_PAIR_GATHER_PAGES,
        ));
        bindings.extend([
            (destination_bytes, BufferIOAttr::WriteOnly),
            (destination_bytes, BufferIOAttr::WriteOnly),
        ]);
        Ok(Self {
            grid: [
                u32::try_from(total_words)
                    .map_err(|_| Error::Limit("runtime paged pair gather launch words".into()))?
                    .div_ceil(THREADS),
                1,
                1,
            ],
            kernel: PinnedKernel::new_with_static_scalars(
                PAGED_PAIR_GATHER_ENTRY,
                Arc::from(PAGED_PAIR_GATHER_SOURCE),
                THREADS,
                bindings.into(),
                scalars,
            )?,
            pages,
        })
    }

    pub(crate) fn prepare(
        &self,
        client: &cubecl::client::Client,
        keys: &[cubecl::server::Handle],
        values: &[cubecl::server::Handle],
        key_destination: cubecl::server::Handle,
        value_destination: cubecl::server::Handle,
    ) -> Result<crate::cuda::kernel::PreparedPinnedLaunch> {
        if keys.len() != self.pages || values.len() != self.pages {
            return Err(Error::State(
                "runtime paged pair gather source coverage differs".into(),
            ));
        }
        let key_fill = keys
            .last()
            .ok_or_else(|| Error::State("runtime paged pair gather keys absent".into()))?;
        let value_fill = values
            .last()
            .ok_or_else(|| Error::State("runtime paged pair gather values absent".into()))?;
        let mut handles = Vec::with_capacity(32);
        handles.extend(keys.iter().cloned());
        handles.extend(std::iter::repeat_n(
            key_fill.clone(),
            MAX_PAGED_PAIR_GATHER_PAGES - keys.len(),
        ));
        handles.extend(values.iter().cloned());
        handles.extend(std::iter::repeat_n(
            value_fill.clone(),
            MAX_PAGED_PAIR_GATHER_PAGES - values.len(),
        ));
        handles.push(key_destination);
        handles.push(value_destination);
        self.kernel.prepare(client, self.grid, handles)
    }
}

impl RuntimeWordTransport {
    pub(crate) fn copy(
        source_bytes: usize,
        destination_bytes: usize,
        rows: usize,
        width: usize,
        source_stride: usize,
        destination_stride: usize,
    ) -> Result<Self> {
        let words = program_kernels::checked_copy_words_with_limit(
            rows,
            width,
            source_stride,
            destination_stride,
            MAX_RUNTIME_WORD_TRANSPORT_WORDS,
        )?;
        require_word_view(source_bytes, rows, source_stride, width, "source")?;
        require_word_view(
            destination_bytes,
            rows,
            destination_stride,
            width,
            "destination",
        )?;
        Self::new(
            COPY_ENTRY,
            vec![
                (source_bytes, BufferIOAttr::ReadOnly),
                (destination_bytes, BufferIOAttr::WriteOnly),
            ],
            words,
            rows,
            width,
            source_stride,
            destination_stride,
        )
    }

    pub(crate) fn zero(
        destination_bytes: usize,
        rows: usize,
        width: usize,
        destination_stride: usize,
    ) -> Result<Self> {
        let words = program_kernels::checked_copy_words_with_limit(
            rows,
            width,
            destination_stride,
            destination_stride,
            MAX_RUNTIME_WORD_TRANSPORT_WORDS,
        )?;
        require_word_view(
            destination_bytes,
            rows,
            destination_stride,
            width,
            "destination",
        )?;
        Self::new(
            ZERO_ENTRY,
            vec![(destination_bytes, BufferIOAttr::WriteOnly)],
            words,
            rows,
            width,
            0,
            destination_stride,
        )
    }

    fn new(
        entry: &'static str,
        bindings: Vec<(usize, BufferIOAttr)>,
        words: usize,
        rows: usize,
        width: usize,
        source_stride: usize,
        destination_stride: usize,
    ) -> Result<Self> {
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
        let scalars = StaticScalarBlock::new(
            schema,
            &[
                StaticScalarValue::U32(as_u32(rows, "rows")?),
                StaticScalarValue::U32(as_u32(width, "width")?),
                StaticScalarValue::U32(as_u32(source_stride, "source stride")?),
                StaticScalarValue::U32(as_u32(destination_stride, "destination stride")?),
            ],
        )?;
        let words = u32::try_from(words)
            .map_err(|_| Error::Limit("runtime word transport launch words".into()))?;
        Ok(Self {
            entry,
            source: SOURCE,
            grid: [words.div_ceil(THREADS), 1, 1],
            kernel: PinnedKernel::new_with_static_scalars(
                entry,
                Arc::from(SOURCE),
                THREADS,
                bindings.into(),
                scalars,
            )?,
        })
    }
}

fn as_u32(value: usize, role: &str) -> Result<u64> {
    u32::try_from(value)
        .map(u64::from)
        .map_err(|_| Error::Limit(format!("runtime word transport {role}")))
}

fn require_word_view(
    bytes: usize,
    rows: usize,
    stride: usize,
    width: usize,
    role: &str,
) -> Result<()> {
    if bytes == 0 || !bytes.is_multiple_of(4) {
        return Err(Error::State(format!(
            "runtime word transport {role} physical extent"
        )));
    }
    let words = bytes / 4;
    let end = (rows - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(width))
        .ok_or_else(|| Error::Limit(format!("runtime word transport {role} span")))?;
    if end > words {
        return Err(Error::State(format!(
            "runtime word transport {role} view too short"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_geometries_share_one_source_and_bounded_abi() {
        let a = RuntimeWordTransport::copy(4096, 4096, 47, 8, 16, 8).unwrap();
        let b = RuntimeWordTransport::copy(4096, 4096, 64, 8, 16, 8).unwrap();
        assert_eq!(a.entry, b.entry);
        assert_eq!(a.source, b.source);
        assert_eq!(a.kernel.binding_sizes(), b.kernel.binding_sizes());
        assert_ne!(a.grid, b.grid);
    }

    #[test]
    fn copy_and_zero_refuse_short_or_overlapping_destinations() {
        assert!(RuntimeWordTransport::copy(32, 32, 2, 8, 8, 4).is_err());
        assert!(RuntimeWordTransport::copy(16, 64, 2, 8, 8, 8).is_err());
        assert!(RuntimeWordTransport::zero(16, 2, 8, 8).is_err());
    }

    #[cfg(all(
        feature = "qualification",
        feature = "cuda-cublas",
        target_os = "linux"
    ))]
    #[test]
    #[ignore = "physical NVIDIA runtime-strided word transport qualification"]
    fn physical_copy_and_zero_preserve_ragged_views_and_guards() -> Result<()> {
        futures_lite::future::block_on(physical())
    }

    #[cfg(all(
        feature = "qualification",
        feature = "cuda-cublas",
        target_os = "linux"
    ))]
    async fn physical() -> Result<()> {
        const GUARD_WORDS: usize = 4;
        const ROWS: usize = 5;
        const WIDTH: usize = 3;
        const SOURCE_STRIDE: usize = 5;
        const DESTINATION_STRIDE: usize = 7;
        const SENTINEL: u32 = 0xdead_beef;
        const GUARD: u32 = 0x55aa_a55a;

        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("runtime-word-transport")?;
        let service = diagnostic.service.clone();
        let client = service.client();
        let source_words = (ROWS - 1) * SOURCE_STRIDE + WIDTH;
        let destination_words = (ROWS - 1) * DESTINATION_STRIDE + WIDTH;
        let encode = |words: &[u32]| {
            words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>()
        };

        let source_host = (0..source_words)
            .map(|index| 0x3f00_0000u32 ^ index as u32)
            .collect::<Vec<_>>();
        let source = client.create_from_slice(&encode(&source_host));
        let mut guarded_host = vec![GUARD; GUARD_WORDS];
        guarded_host.extend(std::iter::repeat_n(SENTINEL, destination_words));
        guarded_host.extend(std::iter::repeat_n(GUARD, GUARD_WORDS));
        let raw_destination = client.create_from_slice(&encode(&guarded_host));
        let destination = raw_destination
            .clone()
            .offset_start((GUARD_WORDS * 4) as u64)
            .offset_end((GUARD_WORDS * 4) as u64);

        let copy = RuntimeWordTransport::copy(
            source_words * 4,
            destination_words * 4,
            ROWS,
            WIDTH,
            SOURCE_STRIDE,
            DESTINATION_STRIDE,
        )?;
        copy.kernel
            .prepare(
                &client,
                copy.grid,
                vec![source.clone(), destination.clone()],
            )?
            .launch()?;
        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("runtime word copy sync: {error}")))?;

        let copied = client
            .read_one(raw_destination.clone())
            .map_err(|error| Error::State(format!("runtime word copy readback: {error}")))?;
        let mut expected = guarded_host.clone();
        for row in 0..ROWS {
            for column in 0..WIDTH {
                expected[GUARD_WORDS + row * DESTINATION_STRIDE + column] =
                    source_host[row * SOURCE_STRIDE + column];
            }
        }
        if &copied[..] != encode(&expected).as_slice() {
            return Err(Error::State(
                "runtime word copy changed gaps/guards or copied wrong values".into(),
            ));
        }

        let zero =
            RuntimeWordTransport::zero(destination_words * 4, ROWS, WIDTH, DESTINATION_STRIDE)?;
        zero.kernel
            .prepare(&client, zero.grid, vec![destination.clone()])?
            .launch()?;
        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("runtime word zero sync: {error}")))?;
        let zeroed = client
            .read_one(raw_destination)
            .map_err(|error| Error::State(format!("runtime word zero readback: {error}")))?;
        for row in 0..ROWS {
            for column in 0..WIDTH {
                expected[GUARD_WORDS + row * DESTINATION_STRIDE + column] = 0;
            }
        }
        if &zeroed[..] != encode(&expected).as_slice() {
            return Err(Error::State(
                "runtime word zero changed gaps/guards or missed active values".into(),
            ));
        }
        println!(
            "RUNTIME_WORD_TRANSPORT_PASS rows={ROWS} width={WIDTH} source_stride={SOURCE_STRIDE} destination_stride={DESTINATION_STRIDE} guards=true"
        );

        client
            .memory_cleanup()
            .map_err(|error| Error::State(format!("runtime word transport cleanup: {error}")))?;
        service.close_model_namespace()?;
        Ok(())
    }
}
