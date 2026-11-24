//! RDNA3 wave32 BF16 matrix multiplication with native FP32 accumulation.
//! Different reduction identity from the CPU64 reference; never substitute it
//! under that contract. No allocator, stream, model or publication authority.
use cubecl::{
    client::Client,
    prelude::{CubeCount, ElemType, KernelId, Scope},
    server::{Handle, KernelArguments, MetadataBindingInfo},
};
use cubecl_core::ir::{
    AddressType, UIntKind,
    metadata::Info,
    settings::{Dim3, ExecutionMode, KernelSettings},
};
use cubecl_server::kernel::{
    BufferIOAttr, CubeKernel, KernelDefinition, KernelMetadata, PrecompiledSource,
};
use zerogravity_core::{Error, Result};

const SOURCE: &str = include_str!("wmma.hip");
const fn source_hash() -> u64 {
    let bytes = SOURCE.as_bytes();
    let mut hash = 0xcbf29ce484222325u64;
    let mut i = 0;
    while i < bytes.len() {
        hash = (hash ^ bytes[i] as u64).wrapping_mul(0x100000001b3);
        i += 1;
    }
    hash
}

#[allow(dead_code)]
pub(crate) struct Bf16Wmma;
impl KernelMetadata for Bf16Wmma {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .cube_dim(Dim3::new_1d(32))
            .info(source_hash())
    }
    fn address_type(&self) -> ElemType {
        ElemType::UInt(UIntKind::U32)
    }
}
impl CubeKernel for Bf16Wmma {
    fn define(&self) -> KernelDefinition {
        let settings =
            KernelSettings::new(Dim3::new_1d(32), ExecutionMode::Checked, AddressType::U32);
        KernelDefinition {
            body: Scope::root(settings.clone()),
            settings,
            info: Info::default(),
        }
    }
    fn source(&self) -> Option<PrecompiledSource> {
        Some(PrecompiledSource {
            source: SOURCE.into(),
            entrypoint_name: "zg_bf16_wmma".into(),
            lang: "hip",
        })
    }
}

/// Independent input rows times transposed BF16 weight rows. Both inputs
/// are contiguous packed BF16; output is BF16-rounded, widened F32 with
/// an optional larger destination row stride.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub requests: usize,
    pub rows: usize,
    pub columns: usize,
    pub output_stride: usize,
}
impl Geometry {
    fn extents(self) -> Result<[usize; 3]> {
        let Self {
            requests,
            rows,
            columns,
            output_stride,
        } = self;
        let span = || Error::Limit("WMMA projection geometry".into());
        if requests == 0
            || rows == 0
            || columns == 0
            || !columns.is_multiple_of(16)
            || output_stride < rows
        {
            return Err(span());
        }
        let words = [
            rows.checked_mul(columns),
            requests.checked_mul(columns),
            (requests - 1)
                .checked_mul(output_stride)
                .and_then(|n| n.checked_add(rows)),
        ];
        let mut bytes = [0; 3];
        for (i, n) in words.into_iter().enumerate() {
            bytes[i] = n
                .filter(|n| *n <= u32::MAX as usize)
                .and_then(|n| n.checked_mul(if i == 2 { 4 } else { 2 }))
                .ok_or_else(span)?;
        }
        Ok(bytes)
    }

    /// Caller owns the original HIP flight, native capability and non-aliasing
    /// physical buffers. Status is initialized once by the enclosing attempt.
    /// The kernel uses native matrix arithmetic, not CPU64 reference arithmetic.
    ///
    /// # Safety
    /// All four handles must be live, appropriately aligned non-overlapping
    /// views on the same qualified gfx11 client, retained through completion.
    #[allow(dead_code)]
    pub(crate) unsafe fn enqueue(
        self,
        client: &Client,
        weights: &Handle,
        input: &Handle,
        output: &Handle,
        status: &Handle,
    ) -> Result<()> {
        let [w, x, y] = self.extents()?;
        let grid = (self.rows.div_ceil(16), self.requests.div_ceil(16), 1usize);
        let max = client.properties().hardware.max_cube_count;
        if grid.0 > max.0 as usize || grid.1 > max.1 as usize {
            return Err(Error::Limit("WMMA projection grid".into()));
        }
        for (handle, bytes, alignment) in [
            (weights, w, 32),
            (input, x, 32),
            (output, y, 4),
            (status, 4, 4),
        ] {
            let start = handle.offset_start.unwrap_or(0);
            let size = handle
                .size()
                .checked_sub(start)
                .and_then(|n| n.checked_sub(handle.offset_end.unwrap_or(0)));
            if handle.service != client.service_id()
                || size != Some(bytes as u64)
                || start % alignment != 0
            {
                return Err(Error::State(
                    "WMMA projection binding extent/alignment".into(),
                ));
            }
        }
        client.launch(
            Box::new(Bf16Wmma),
            CubeCount::Static(grid.0 as u32, grid.1 as u32, 1),
            KernelArguments::new()
                .with_buffer_io(weights.clone().binding(), BufferIOAttr::ReadOnly)
                .with_buffer_io(input.clone().binding(), BufferIOAttr::ReadOnly)
                .with_buffer_io(output.clone().binding(), BufferIOAttr::WriteOnly)
                .with_buffer_io(status.clone().binding(), BufferIOAttr::ReadWrite)
                .with_info(MetadataBindingInfo::custom(vec![
                    self.requests as u64,
                    self.rows as u64,
                    self.columns as u64,
                    self.output_stride as u64,
                ])),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;
