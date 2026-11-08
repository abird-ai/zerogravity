//! BF16 transport and original-flight CUDA online-softmax attention.

use super::*;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn widen_bf16(
    input: &[u32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let word = input[i / 2usize];
        let shift = u32::cast_from(i % 2usize) * 16u32;
        let bits = (word >> shift) & 0xffffu32;
        if (bits & 0x7f80u32) == 0x7f80u32 {
            status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
        }
        output[i] = f32::reinterpret(bits << 16u32);
    }
}

#[cube(launch_unchecked)]
fn pack_source_bf16(
    input: &[f32],
    output: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let word = ABSOLUTE_POS;
    if word < elements / 2usize {
        let a = u32::reinterpret(input[word * 2usize]);
        let b = u32::reinterpret(input[word * 2usize + 1usize]);
        if (a & 0x7f800000u32) == 0x7f800000u32
            || (b & 0x7f800000u32) == 0x7f800000u32
            || (a & 0xffffu32) != 0u32
            || (b & 0xffffu32) != 0u32
        {
            status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
        }
        output[word] = (a >> 16u32) | (b & 0xffff0000u32);
    }
}

#[cube(launch_unchecked)]
fn zero_words_runtime(destination: &mut [u32], words: u32) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(words) {
        destination[i] = 0u32;
    }
}

#[cube(launch_unchecked)]
fn copy_words_runtime(source: &[u32], destination: &mut [u32], status: &[Atomic<u32>], words: u32) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(words) {
        if i < source.len() && i < destination.len() {
            destination[i] = source[i];
        } else {
            status[0].fetch_or(crate::program_kernels::VIEW_OUT_OF_BOUNDS);
        }
    }
}

#[cube(launch_unchecked)]
fn certify_bf16(input: &[u32], status: &[Atomic<u32>], elements: u32) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(elements) {
        let word = input[i / 2usize];
        let shift = u32::cast_from(i % 2usize) * 16u32;
        let bits = (word >> shift) & 0xffffu32;
        if (bits & 0x7f80u32) == 0x7f80u32 {
            status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
        }
    }
}

fn grid(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(32), 1, 1)
}
fn threads() -> CubeDim {
    CubeDim::new_1d(32)
}
fn arg(handle: &cubecl::server::Handle, elements: usize) -> BufferArg {
    // SAFETY: caller validates exact physical BF16/F32 extents and retains
    // original leases in the enclosing armed flight.
    unsafe { BufferArg::from_raw_parts(handle.clone(), elements) }
}

pub(super) fn widen(
    client: &cubecl::client::Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    status: &cubecl::server::Handle,
    elements: usize,
) -> Result<usize> {
    if elements == 0 || !elements.is_multiple_of(2) {
        return Err(Error::Unsupported(
            "ordinary attention BF16 widen even positive extent".into(),
        ));
    }
    if input.size_in_used() != (elements * 2) as u64
        || output.size_in_used() != (elements * 4) as u64
        || status.size_in_used() != 4
        || input.service != client.service_id()
        || output.service != client.service_id()
        || status.service != client.service_id()
    {
        return Err(Error::State(
            "ordinary attention BF16 widen exact service/extent".into(),
        ));
    }
    unsafe {
        widen_bf16::launch_unchecked(
            client,
            grid(elements),
            threads(),
            arg(input, elements / 2),
            arg(output, elements),
            arg(status, 1),
            elements,
        );
    }
    Ok(1)
}

pub(super) fn pack(
    client: &cubecl::client::Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    status: &cubecl::server::Handle,
    elements: usize,
) -> Result<usize> {
    if elements == 0 || !elements.is_multiple_of(2) {
        return Err(Error::Unsupported(
            "ordinary attention BF16 pack even positive extent".into(),
        ));
    }
    if input.size_in_used() != (elements * 4) as u64
        || output.size_in_used() != (elements * 2) as u64
        || status.size_in_used() != 4
        || input.service != client.service_id()
        || output.service != client.service_id()
        || status.service != client.service_id()
    {
        return Err(Error::State(
            "ordinary attention BF16 pack exact service/extent".into(),
        ));
    }
    unsafe {
        pack_source_bf16::launch_unchecked(
            client,
            grid(elements / 2),
            threads(),
            arg(input, elements),
            arg(output, elements / 2),
            arg(status, 1),
            elements,
        );
    }
    Ok(1)
}

#[cfg(test)]
fn host_widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
#[cfg(test)]
fn host_pack(a: f32, b: f32) -> Result<u32> {
    let a = a.to_bits();
    let b = b.to_bits();
    if [a, b]
        .into_iter()
        .any(|bits| bits & 0x7f80_0000 == 0x7f80_0000 || bits & 0xffff != 0)
    {
        return Err(Error::State(
            "ordinary attention host BF16 pack contract".into(),
        ));
    }
    Ok((a >> 16) | (b & 0xffff_0000))
}

pub(super) fn zero_bf16(
    client: &cubecl::client::Client,
    destination: &cubecl::server::Handle,
    elements: usize,
) -> Result<usize> {
    if elements == 0 || !elements.is_multiple_of(2) {
        return Err(Error::Unsupported(
            "ordinary CUDA BF16 zero even positive extent".into(),
        ));
    }
    let bytes = elements
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("ordinary CUDA BF16 zero bytes".into()))?;
    if destination.size_in_used() != bytes as u64
        || destination.service != client.service_id()
        || u32::try_from(elements / 2).is_err()
    {
        return Err(Error::State(
            "ordinary CUDA BF16 zero exact service/extent".into(),
        ));
    }
    unsafe {
        zero_words_runtime::launch_unchecked(
            client,
            grid(elements / 2),
            threads(),
            arg(destination, elements / 2),
            (elements / 2) as u32,
        );
    }
    Ok(1)
}

pub(super) fn copy_bf16_prefix(
    client: &cubecl::client::Client,
    source: &cubecl::server::Handle,
    destination: &cubecl::server::Handle,
    status: &cubecl::server::Handle,
    elements: usize,
) -> Result<usize> {
    if elements == 0 || !elements.is_multiple_of(2) {
        return Err(Error::Unsupported(
            "ordinary CUDA BF16 prefix copy even positive extent".into(),
        ));
    }
    let bytes = elements
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("ordinary CUDA BF16 prefix copy bytes".into()))?;
    if source.size_in_used() != bytes as u64
        || destination.size_in_used() != bytes as u64
        || status.size_in_used() != 4
        || source.service != client.service_id()
        || destination.service != client.service_id()
        || status.service != client.service_id()
        || u32::try_from(elements / 2).is_err()
    {
        return Err(Error::State(
            "ordinary CUDA BF16 prefix copy exact service/extent".into(),
        ));
    }
    unsafe {
        copy_words_runtime::launch_unchecked(
            client,
            grid(elements / 2),
            threads(),
            arg(source, elements / 2),
            arg(destination, elements / 2),
            arg(status, 1),
            (elements / 2) as u32,
        );
    }
    Ok(1)
}

pub(super) fn certify(
    client: &cubecl::client::Client,
    input: &cubecl::server::Handle,
    status: &cubecl::server::Handle,
    elements: usize,
) -> Result<usize> {
    if elements == 0
        || input.size_in_used() != (elements * 2) as u64
        || status.size_in_used() != 4
        || input.service != client.service_id()
        || status.service != client.service_id()
        || u32::try_from(elements).is_err()
    {
        return Err(Error::State(
            "ordinary CUDA BF16 certificate exact service/extent".into(),
        ));
    }
    unsafe {
        certify_bf16::launch_unchecked(
            client,
            grid(elements),
            threads(),
            arg(input, elements.div_ceil(2)),
            arg(status, 1),
            elements as u32,
        );
    }
    Ok(1)
}

const WIDTH: usize = 256;
const SCALE: f32 = 0.0625;

pub(super) fn managed_bytes(
    capacity: usize,
    geometry: crate::cuda::ordinary_text::Geometry,
) -> Result<usize> {
    let geometry = geometry.validate()?;
    let query_width = geometry.query_width()?;
    let kv_width = geometry.kv_width()?;
    if !(1..=32_768).contains(&capacity) {
        return Err(Error::Unsupported(
            "ordinary CUDA attention context capacity".into(),
        ));
    }
    let kv = capacity
        .checked_mul(kv_width)
        .ok_or_else(|| Error::Limit("ordinary attention KV extent".into()))?;
    let scores = capacity
        .checked_mul(geometry.query_heads)
        .ok_or_else(|| Error::Limit("ordinary attention score extent".into()))?;
    [
        query_width,
        kv,
        kv,
        scores,
        scores,
        geometry.query_heads,
        scores,
        geometry.query_heads,
        geometry.query_heads,
        scores,
        query_width,
        1,
    ]
    .into_iter()
    .try_fold(0usize, |sum, elements| {
        elements
            .checked_mul(4)
            .and_then(|bytes| sum.checked_add(bytes))
            .ok_or_else(|| Error::Limit("ordinary attention managed byte sum".into()))
    })
}

pub(super) struct OrdinaryCudaAttention {
    service: Arc<ResidentService>,
    capacity: usize,
    geometry: crate::cuda::ordinary_text::Geometry,
    // Preserve the admitted workspace envelope until long-context qualification.
    // The tile-local kernel itself consumes no global scratch.
    workspace_reserve: ResidentLease,
}

impl OrdinaryCudaAttention {
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        init: &mut crate::resident_leases::ResidentInitialization<'_>,
        capacity: usize,
        geometry: crate::cuda::ordinary_text::Geometry,
    ) -> Result<Arc<Self>> {
        let bytes = managed_bytes(capacity, geometry)?;
        init.require_service(&service)?;
        let result = Arc::new(Self {
            service,
            capacity,
            geometry,
            workspace_reserve: init.allocate_empty(bytes, BufferRole::Scratch)?,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }

    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        vec![(self.workspace_reserve.clone(), DeviceAccess::Write)]
    }

    fn require_operator(
        &self,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
    ) -> Result<()> {
        use zerogravity_core::accelerator::program::{
            SourceAttentionMask, SourceNumericalContract, SourceOperatorKind,
        };
        operator.validate()?;
        if operator.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
            || operator.call().rows().capacity() != self.capacity
            || !matches!(
                operator.kind(),
                SourceOperatorKind::FlashAttention {
                    geometry,
                    scale,
                    mask: SourceAttentionMask::Causal,
                    ..
                } if geometry.width == WIDTH
                    && geometry.query_heads == self.geometry.query_heads
                    && geometry.kv_heads == self.geometry.kv_heads
                    && scale.to_bits() == SCALE.to_bits()
            )
            || !operator.matrices().is_empty()
            || !operator.parameters().is_empty()
        {
            return Err(Error::State(
                "ordinary CUDA exact source attention contract differs".into(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue(
        &self,
        client: &cubecl::client::Client,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
        query: &cubecl::server::Handle,
        candidate_keys: &crate::resident_program::DeviceBuffer,
        candidate_values: &crate::resident_program::DeviceBuffer,
        output: &cubecl::server::Handle,
        status: &ResidentLease,
    ) -> Result<usize> {
        use crate::resident_program::{ResidentBufferLayout, ResidentPhysicalType};
        self.require_operator(operator)?;
        self.service.ensure_usable()?;
        if client.service_id() != self.service.client().service_id()
            || query.service != client.service_id()
            || output.service != client.service_id()
            || !Arc::ptr_eq(status.service(), &self.service)
            || status.bytes() != 4
        {
            return Err(Error::State(
                "ordinary CUDA attention original service/status differs".into(),
            ));
        }
        let rows = operator.call().rows().len();
        let base = operator.call().rows().base();
        let end = base
            .checked_add(rows)
            .filter(|end| *end <= self.capacity)
            .ok_or_else(|| Error::Limit("ordinary attention active prefix".into()))?;
        let query_elements = rows
            .checked_mul(self.geometry.query_width()?)
            .ok_or_else(|| Error::Limit("ordinary attention query batch".into()))?;
        if query.size_in_used() != (query_elements * 2) as u64
            || output.size_in_used() != (query_elements * 2) as u64
        {
            return Err(Error::State(
                "ordinary attention BF16 query/output extent differs".into(),
            ));
        }

        let state_elements = self
            .capacity
            .checked_mul(self.geometry.kv_width()?)
            .ok_or_else(|| Error::Limit("ordinary attention state extent".into()))?;
        let layout = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, state_elements)?;
        candidate_keys.require_layout(layout)?;
        candidate_values.require_layout(layout)?;
        let keys = candidate_keys.original_lease()?;
        let values = candidate_values.original_lease()?;
        if !Arc::ptr_eq(keys.service(), &self.service)
            || !Arc::ptr_eq(values.service(), &self.service)
            || keys.bytes() != state_elements * 2
            || values.bytes() != state_elements * 2
        {
            return Err(Error::State(
                "ordinary attention candidate KV physical owner/extent differs".into(),
            ));
        }

        let launch = crate::cuda::bf16_attention::Bf16CausalAttention::new(
            rows,
            base,
            self.capacity,
            self.geometry.query_heads,
            self.geometry.kv_heads,
            WIDTH,
            SCALE,
        )?
        .prepare(
            client,
            [
                query.clone(),
                keys.handle(),
                values.handle(),
                output.clone(),
                status.handle(),
            ],
        )?;
        launch.launch_on(client)?;
        debug_assert_eq!(end, base + rows);
        Ok(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_transport_is_bit_exact_and_pack_refuses_unrounded_f32() {
        for bits in [0u16, 0x8000, 1, 0x8001, 0x3f80, 0xbf80, 0x7f7f, 0xff7f] {
            let value = host_widen(bits);
            let packed = host_pack(value, 0.0).unwrap();
            assert_eq!(packed as u16, bits);
        }
        assert!(host_pack(f32::from_bits(0x3f80_0001), 0.0).is_err());
        assert!(host_pack(f32::INFINITY, 0.0).is_err());
        assert!(host_pack(f32::NAN, 0.0).is_err());
    }

    #[test]
    fn attention_inventory_is_capacity_derived_and_bounded() {
        for capacity in [1usize, 512, 8_192, 32_768] {
            let expected_elements = 4_121usize + 1_056usize * capacity;
            assert_eq!(
                managed_bytes(capacity, crate::cuda::ordinary_text::Geometry::qwen35_08()).unwrap(),
                expected_elements * 4
            );
        }
        assert!(managed_bytes(0, crate::cuda::ordinary_text::Geometry::qwen35_08()).is_err());
        assert!(managed_bytes(32_769, crate::cuda::ordinary_text::Geometry::qwen35_08()).is_err());
    }
}
