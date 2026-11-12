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
const COPY_ENTRY: &str = "zg_copy_strided_words_runtime_v1";
const ZERO_ENTRY: &str = "zg_zero_strided_words_runtime_v1";
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

pub(crate) struct RuntimeWordTransport {
    pub(crate) entry: &'static str,
    pub(crate) source: &'static str,
    pub(crate) grid: [u32; 3],
    pub(crate) kernel: PinnedKernel,
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
        let words =
            program_kernels::checked_copy_words(rows, width, source_stride, destination_stride)?;
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
        let words = program_kernels::checked_copy_words(
            rows,
            width,
            destination_stride,
            destination_stride,
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
