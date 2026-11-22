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
            "forward attention BF16 widen even positive extent".into(),
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
            "forward attention BF16 widen exact service/extent".into(),
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
            "forward attention BF16 pack even positive extent".into(),
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
            "forward attention BF16 pack exact service/extent".into(),
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
            "forward attention host BF16 pack contract".into(),
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
            "CUDA forward BF16 zero even positive extent".into(),
        ));
    }
    let bytes = elements
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("CUDA forward BF16 zero bytes".into()))?;
    if destination.size_in_used() != bytes as u64
        || destination.service != client.service_id()
        || u32::try_from(elements / 2).is_err()
    {
        return Err(Error::State(
            "CUDA forward BF16 zero exact service/extent".into(),
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
            "CUDA forward BF16 prefix copy even positive extent".into(),
        ));
    }
    let bytes = elements
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("CUDA forward BF16 prefix copy bytes".into()))?;
    if source.size_in_used() != bytes as u64
        || destination.size_in_used() != bytes as u64
        || status.size_in_used() != 4
        || source.service != client.service_id()
        || destination.service != client.service_id()
        || status.service != client.service_id()
        || u32::try_from(elements / 2).is_err()
    {
        return Err(Error::State(
            "CUDA forward BF16 prefix copy exact service/extent".into(),
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
            "CUDA forward BF16 certificate exact service/extent".into(),
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

fn validate_geometry(
    capacity: usize,
    geometry: crate::cuda::text_forward::Geometry,
) -> Result<crate::cuda::text_forward::Geometry> {
    let geometry = geometry.validate()?;
    if !(1..=32_768).contains(&capacity) {
        return Err(Error::Unsupported(
            "CUDA forward attention context capacity".into(),
        ));
    }
    Ok(geometry)
}

/// ZeroGravity-authored online-softmax implementation. This is one physical
/// provider for the semantic Attention operator, not the semantic operation
/// itself.
struct BuiltinCudaAttention {
    service: Arc<ResidentService>,
    capacity: usize,
    geometry: crate::cuda::text_forward::Geometry,
}

impl BuiltinCudaAttention {
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        capacity: usize,
        geometry: crate::cuda::text_forward::Geometry,
    ) -> Result<Arc<Self>> {
        service.ensure_usable()?;
        let geometry = validate_geometry(capacity, geometry)?;
        Ok(Arc::new(Self {
            service,
            capacity,
            geometry,
        }))
    }

    fn require_operator(
        &self,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
    ) -> Result<()> {
        use zerogravity_core::accelerator::program::{
            NumericalContract, SourceAttentionMask, SourceOperatorKind,
        };
        operator.validate()?;
        if operator.numerical_contract() != NumericalContract::CudaSm86Bf16ReferenceV1
            || operator.invocation().tokens().capacity() != self.capacity
            || !matches!(
                operator.kind(),
                SourceOperatorKind::Attention {
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
                "CUDA forward exact source attention contract differs".into(),
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
                "CUDA forward attention original service/status differs".into(),
            ));
        }
        let rows = operator.invocation().tokens().len();
        let base = operator.invocation().tokens().base();
        let end = base
            .checked_add(rows)
            .filter(|end| *end <= self.capacity)
            .ok_or_else(|| Error::Limit("forward attention active prefix".into()))?;
        let query_elements = rows
            .checked_mul(self.geometry.query_width()?)
            .ok_or_else(|| Error::Limit("forward attention query batch".into()))?;
        if query.size_in_used() != (query_elements * 2) as u64
            || output.size_in_used() != (query_elements * 2) as u64
        {
            return Err(Error::State(
                "forward attention BF16 query/output extent differs".into(),
            ));
        }

        let state_elements = self
            .capacity
            .checked_mul(self.geometry.kv_width()?)
            .ok_or_else(|| Error::Limit("forward attention state extent".into()))?;
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
                "forward attention candidate KV physical owner/extent differs".into(),
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_cohort(
        &self,
        client: &cubecl::client::Client,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
        bases: [usize; 4],
        query: &cubecl::server::Handle,
        states: &[(
            &crate::resident_program::DeviceBuffer,
            &crate::resident_program::DeviceBuffer,
        )],
        output: &cubecl::server::Handle,
        status: &ResidentLease,
    ) -> Result<usize> {
        use crate::resident_program::{ResidentBufferLayout, ResidentPhysicalType};
        self.require_operator(operator)?;
        self.service.ensure_usable()?;
        if states.len() != 4
            || operator.invocation().tokens().len() != 1
            || operator.invocation().tokens().base() != bases[0]
            || bases.iter().any(|&base| base >= self.capacity)
            || client.service_id() != self.service.client().service_id()
            || query.service != client.service_id()
            || output.service != client.service_id()
            || !Arc::ptr_eq(status.service(), &self.service)
            || status.bytes() != 4
        {
            return Err(Error::State(
                "CUDA forward attention batch4 service/geometry differs".into(),
            ));
        }
        let query_row = self.geometry.query_width()?;
        let query_elements = query_row
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("forward attention batch4 query extent".into()))?;
        if query.size_in_used() != (query_elements * 2) as u64
            || output.size_in_used() != (query_elements * 2) as u64
        {
            return Err(Error::State(
                "forward attention batch4 BF16 query/output extent differs".into(),
            ));
        }
        let state_elements = self
            .capacity
            .checked_mul(self.geometry.kv_width()?)
            .ok_or_else(|| Error::Limit("forward attention batch4 state extent".into()))?;
        let layout = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, state_elements)?;
        let mut keys = Vec::with_capacity(4);
        let mut values = Vec::with_capacity(4);
        for (key, value) in states {
            key.require_layout(layout)?;
            value.require_layout(layout)?;
            let key = key.original_lease()?;
            let value = value.original_lease()?;
            if !Arc::ptr_eq(key.service(), &self.service)
                || !Arc::ptr_eq(value.service(), &self.service)
                || key.bytes() != state_elements * 2
                || value.bytes() != state_elements * 2
            {
                return Err(Error::State(
                    "forward attention batch4 candidate KV owner/extent differs".into(),
                ));
            }
            keys.push(key.handle());
            values.push(value.handle());
        }
        let launch = crate::cuda::bf16_attention::Bf16CausalAttentionBatch4::new_ragged(
            bases,
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
                keys[0].clone(),
                keys[1].clone(),
                keys[2].clone(),
                keys[3].clone(),
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                values[3].clone(),
                output.clone(),
                status.handle(),
            ],
        )?;
        launch.launch_on(client)?;
        Ok(1)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_prefill_cohort4(
        &self,
        client: &cubecl::client::Client,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
        rows: usize,
        bases: [usize; 4],
        query: &cubecl::server::Handle,
        states: &[(
            &crate::resident_program::DeviceBuffer,
            &crate::resident_program::DeviceBuffer,
        )],
        output: &cubecl::server::Handle,
        status: &ResidentLease,
    ) -> Result<usize> {
        use crate::resident_program::{ResidentBufferLayout, ResidentPhysicalType};
        self.require_operator(operator)?;
        self.service.ensure_usable()?;
        if rows == 0
            || states.len() != 4
            || operator.invocation().tokens().len() != rows
            || operator.invocation().tokens().base() != bases[0]
            || bases
                .iter()
                .any(|&base| base.checked_add(rows).is_none_or(|end| end > self.capacity))
            || client.service_id() != self.service.client().service_id()
            || query.service != client.service_id()
            || output.service != client.service_id()
            || !Arc::ptr_eq(status.service(), &self.service)
            || status.bytes() != 4
        {
            return Err(Error::State(
                "CUDA forward attention prefill cohort4 service/geometry differs".into(),
            ));
        }
        let query_row = self.geometry.query_width()?;
        let query_elements = query_row
            .checked_mul(rows)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| Error::Limit("forward attention prefill cohort4 query extent".into()))?;
        if query.size_in_used() != (query_elements * 2) as u64
            || output.size_in_used() != (query_elements * 2) as u64
        {
            return Err(Error::State(
                "forward attention prefill cohort4 BF16 query/output extent differs".into(),
            ));
        }
        let state_elements = self
            .capacity
            .checked_mul(self.geometry.kv_width()?)
            .ok_or_else(|| Error::Limit("forward attention prefill cohort4 state extent".into()))?;
        let layout = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, state_elements)?;
        let mut keys = Vec::with_capacity(4);
        let mut values = Vec::with_capacity(4);
        for (key, value) in states {
            key.require_layout(layout)?;
            value.require_layout(layout)?;
            let key = key.original_lease()?;
            let value = value.original_lease()?;
            if !Arc::ptr_eq(key.service(), &self.service)
                || !Arc::ptr_eq(value.service(), &self.service)
                || key.bytes() != state_elements * 2
                || value.bytes() != state_elements * 2
            {
                return Err(Error::State(
                    "forward attention prefill cohort4 candidate KV owner/extent differs".into(),
                ));
            }
            keys.push(key.handle());
            values.push(value.handle());
        }
        let launch = crate::cuda::bf16_attention::Bf16CausalAttentionBatch4::new_ragged_rows(
            rows,
            bases,
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
                keys[0].clone(),
                keys[1].clone(),
                keys[2].clone(),
                keys[3].clone(),
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                values[3].clone(),
                output.clone(),
                status.handle(),
            ],
        )?;
        launch.launch_on(client)?;
        Ok(1)
    }
}

/// Phase-specific provider plan selected once during cold model preparation.
/// The wrapper owns no KV/state authority; providers consume the exact retained
/// state and service leases supplied by the enclosing resident program.
pub(super) struct CudaForwardAttention {
    providers: crate::AttentionProviderPlan,
    builtin: Arc<BuiltinCudaAttention>,
}

impl CudaForwardAttention {
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        capacity: usize,
        geometry: crate::cuda::text_forward::Geometry,
        providers: crate::AttentionProviderPlan,
    ) -> Result<Arc<Self>> {
        let builtin = BuiltinCudaAttention::prepare(service, capacity, geometry)?;
        Ok(Arc::new(Self { providers, builtin }))
    }

    fn provider(
        &self,
        phase: zerogravity_core::accelerator::program::InvocationPhase,
    ) -> crate::AttentionProvider {
        self.providers.provider(phase)
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
        match self.provider(operator.invocation().phase()) {
            crate::AttentionProvider::Builtin => self.builtin.enqueue(
                client,
                operator,
                query,
                candidate_keys,
                candidate_values,
                output,
                status,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_cohort(
        &self,
        client: &cubecl::client::Client,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
        bases: [usize; 4],
        query: &cubecl::server::Handle,
        states: &[(
            &crate::resident_program::DeviceBuffer,
            &crate::resident_program::DeviceBuffer,
        )],
        output: &cubecl::server::Handle,
        status: &ResidentLease,
    ) -> Result<usize> {
        match self.provider(operator.invocation().phase()) {
            crate::AttentionProvider::Builtin => self
                .builtin
                .enqueue_cohort(client, operator, bases, query, states, output, status),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_prefill_cohort4(
        &self,
        client: &cubecl::client::Client,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
        rows: usize,
        bases: [usize; 4],
        query: &cubecl::server::Handle,
        states: &[(
            &crate::resident_program::DeviceBuffer,
            &crate::resident_program::DeviceBuffer,
        )],
        output: &cubecl::server::Handle,
        status: &ResidentLease,
    ) -> Result<usize> {
        use zerogravity_core::accelerator::program::InvocationPhase;
        if operator.invocation().phase() != InvocationPhase::Prefill {
            return Err(Error::State(
                "prefill attention cohort routed with non-prefill invocation".into(),
            ));
        }
        match self.providers.prefill() {
            crate::AttentionProvider::Builtin => self.builtin.enqueue_prefill_cohort4(
                client, operator, rows, bases, query, states, output, status,
            ),
        }
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
    fn attention_capacity_is_bounded_without_global_scratch() {
        for capacity in [1usize, 512, 8_192, 32_768] {
            assert!(
                validate_geometry(capacity, crate::cuda::text_forward::Geometry::qwen35_08(),)
                    .is_ok()
            );
        }
        assert!(validate_geometry(0, crate::cuda::text_forward::Geometry::qwen35_08()).is_err());
        assert!(
            validate_geometry(32_769, crate::cuda::text_forward::Geometry::qwen35_08()).is_err()
        );
    }
}
