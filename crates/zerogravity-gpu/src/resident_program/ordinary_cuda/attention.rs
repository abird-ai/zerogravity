//! Source-exact BF16<->F32 transport and reusable ordinary attention scratch.
//! Numerical attention itself remains in the existing source_attention kernels.

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
            assert_eq!(managed_bytes(capacity).unwrap(), expected_elements * 4);
        }
        assert!(managed_bytes(0).is_err());
        assert!(managed_bytes(32_769).is_err());
    }
}

const WIDTH: usize = 256;
const QUERY_HEADS: usize = 8;
const KV_HEADS: usize = 2;
const QUERY_WIDTH: usize = WIDTH * QUERY_HEADS;
const KV_WIDTH: usize = WIDTH * KV_HEADS;
const SCALE: f32 = 0.0625;

pub(super) fn managed_bytes(capacity: usize) -> Result<usize> {
    if !(1..=32_768).contains(&capacity) {
        return Err(Error::Unsupported(
            "ordinary CUDA attention context capacity".into(),
        ));
    }
    let kv = capacity
        .checked_mul(KV_WIDTH)
        .ok_or_else(|| Error::Limit("ordinary attention KV extent".into()))?;
    let scores = capacity
        .checked_mul(QUERY_HEADS)
        .ok_or_else(|| Error::Limit("ordinary attention score extent".into()))?;
    [
        QUERY_WIDTH,
        kv,
        kv,
        scores,
        scores,
        QUERY_HEADS,
        scores,
        QUERY_HEADS,
        QUERY_HEADS,
        scores,
        QUERY_WIDTH,
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
    query: ResidentLease,
    keys: ResidentLease,
    values: ResidentLease,
    dots: ResidentLease,
    scores: ResidentLease,
    maxima: ResidentLease,
    exponentials: ResidentLease,
    sums: ResidentLease,
    inverse: ResidentLease,
    probabilities: ResidentLease,
    output: ResidentLease,
    zero: ResidentLease,
}

impl OrdinaryCudaAttention {
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        init: &mut crate::resident_leases::ResidentInitialization<'_>,
        capacity: usize,
    ) -> Result<Arc<Self>> {
        managed_bytes(capacity)?;
        init.require_service(&service)?;
        let f32_bytes = |elements: usize, role: &str| -> Result<usize> {
            elements
                .checked_mul(4)
                .ok_or_else(|| Error::Limit(format!("ordinary attention {role} bytes")))
        };
        let kv = capacity
            .checked_mul(KV_WIDTH)
            .ok_or_else(|| Error::Limit("ordinary attention KV extent".into()))?;
        let scores = capacity
            .checked_mul(QUERY_HEADS)
            .ok_or_else(|| Error::Limit("ordinary attention score extent".into()))?;
        let mut scratch = |elements, role| {
            f32_bytes(elements, role)
                .and_then(|bytes| init.allocate_empty(bytes, BufferRole::Scratch))
        };
        let result = Arc::new(Self {
            service,
            capacity,
            query: scratch(QUERY_WIDTH, "query")?,
            keys: scratch(kv, "keys")?,
            values: scratch(kv, "values")?,
            dots: scratch(scores, "dots")?,
            scores: scratch(scores, "scores")?,
            maxima: scratch(QUERY_HEADS, "maxima")?,
            exponentials: scratch(scores, "exponentials")?,
            sums: scratch(QUERY_HEADS, "sums")?,
            inverse: scratch(QUERY_HEADS, "inverse")?,
            probabilities: scratch(scores, "probabilities")?,
            output: scratch(QUERY_WIDTH, "output")?,
            zero: init.upload_bytes(&0f32.to_bits().to_le_bytes(), BufferRole::ImmutableWeights)?,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }

    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        [
            &self.query,
            &self.keys,
            &self.values,
            &self.dots,
            &self.scores,
            &self.maxima,
            &self.exponentials,
            &self.sums,
            &self.inverse,
            &self.probabilities,
            &self.output,
        ]
        .into_iter()
        .map(|lease| (lease.clone(), DeviceAccess::Write))
        .chain([(self.zero.clone(), DeviceAccess::Read)])
        .collect()
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
                    && geometry.query_heads == QUERY_HEADS
                    && geometry.kv_heads == KV_HEADS
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
        query: &ResidentLease,
        candidate_keys: &crate::resident_program::DeviceBuffer,
        candidate_values: &crate::resident_program::DeviceBuffer,
        output: &ResidentLease,
        status: &ResidentLease,
    ) -> Result<usize> {
        use crate::resident_program::{ResidentBufferLayout, ResidentPhysicalType};
        self.require_operator(operator)?;
        self.service.ensure_usable()?;
        if client.service_id() != self.service.client().service_id()
            || !Arc::ptr_eq(query.service(), &self.service)
            || !Arc::ptr_eq(output.service(), &self.service)
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
            .checked_mul(QUERY_WIDTH)
            .ok_or_else(|| Error::Limit("ordinary attention query batch".into()))?;
        if query.bytes() != query_elements * 2 || output.bytes() != query_elements * 2 {
            return Err(Error::State(
                "ordinary attention BF16 query/output extent differs".into(),
            ));
        }

        let state_elements = self
            .capacity
            .checked_mul(KV_WIDTH)
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

        let mut launches = 0;
        // Widen once after KV append. Future rows are present in the candidate
        // but masked by each row's exact active causal prefix.
        launches += widen(
            client,
            &keys.handle(),
            &self.keys.handle(),
            &status.handle(),
            state_elements,
        )?;
        launches += widen(
            client,
            &values.handle(),
            &self.values.handle(),
            &status.handle(),
            state_elements,
        )?;

        let score_elements = self.capacity * QUERY_HEADS;
        for row in 0..rows {
            let active = base + row + 1;
            let query_view = super::exact_view(
                query,
                row * QUERY_WIDTH * 2,
                QUERY_WIDTH * 2,
                "attention query row",
            )?;
            let output_view = super::exact_view(
                output,
                row * QUERY_WIDTH * 2,
                QUERY_WIDTH * 2,
                "attention output row",
            )?;
            launches += widen(
                client,
                &query_view,
                &self.query.handle(),
                &status.handle(),
                QUERY_WIDTH,
            )?;
            unsafe {
                crate::source_attention::kernels::qk::launch_unchecked(
                    client,
                    CubeCount::Static(score_elements as u32, 1, 1),
                    CubeDim::new_1d(64),
                    arg(&self.query.handle(), QUERY_WIDTH),
                    arg(&self.keys.handle(), state_elements),
                    arg(&self.dots.handle(), score_elements),
                    arg(&self.scores.handle(), score_elements),
                    arg(&status.handle(), 1),
                    arg(&self.zero.handle(), 1),
                    WIDTH,
                    QUERY_HEADS,
                    KV_HEADS,
                    self.capacity,
                    SCALE,
                    active as u32,
                    0u32,
                );
                crate::source_attention::kernels::reduce::launch_unchecked(
                    client,
                    CubeCount::Static(QUERY_HEADS as u32, 1, 1),
                    threads(),
                    arg(&self.scores.handle(), score_elements),
                    arg(&self.maxima.handle(), QUERY_HEADS),
                    arg(&status.handle(), 1),
                    active as u32,
                    true,
                    self.capacity,
                );
                crate::source_attention::kernels::exponentials::launch_unchecked(
                    client,
                    grid(score_elements),
                    threads(),
                    arg(&self.scores.handle(), score_elements),
                    arg(&self.maxima.handle(), QUERY_HEADS),
                    arg(&self.exponentials.handle(), score_elements),
                    arg(&status.handle(), 1),
                    active as u32,
                    QUERY_HEADS,
                    self.capacity,
                );
                crate::source_attention::kernels::reduce::launch_unchecked(
                    client,
                    CubeCount::Static(QUERY_HEADS as u32, 1, 1),
                    threads(),
                    arg(&self.exponentials.handle(), score_elements),
                    arg(&self.sums.handle(), QUERY_HEADS),
                    arg(&status.handle(), 1),
                    active as u32,
                    false,
                    self.capacity,
                );
                crate::source_attention::kernels::inverse::launch_unchecked(
                    client,
                    grid(QUERY_HEADS),
                    threads(),
                    arg(&self.sums.handle(), QUERY_HEADS),
                    arg(&self.inverse.handle(), QUERY_HEADS),
                    arg(&status.handle(), 1),
                    QUERY_HEADS,
                );
                crate::source_attention::kernels::probabilities::launch_unchecked(
                    client,
                    grid(score_elements),
                    threads(),
                    arg(&self.exponentials.handle(), score_elements),
                    arg(&self.inverse.handle(), QUERY_HEADS),
                    arg(&self.probabilities.handle(), score_elements),
                    arg(&status.handle(), 1),
                    active as u32,
                    QUERY_HEADS,
                    self.capacity,
                );
                crate::source_attention::kernels::pv::launch_unchecked(
                    client,
                    grid(QUERY_WIDTH),
                    threads(),
                    arg(&self.probabilities.handle(), score_elements),
                    arg(&self.values.handle(), state_elements),
                    arg(&self.output.handle(), QUERY_WIDTH),
                    arg(&status.handle(), 1),
                    arg(&self.zero.handle(), 1),
                    WIDTH,
                    QUERY_HEADS,
                    KV_HEADS,
                    self.capacity,
                    active as u32,
                    1u32,
                    active as u32,
                );
            }
            launches += 7;
            launches += pack(
                client,
                &self.output.handle(),
                &output_view,
                &status.handle(),
                QUERY_WIDTH,
            )?;
        }
        debug_assert_eq!(end, base + rows);
        Ok(launches)
    }
}
