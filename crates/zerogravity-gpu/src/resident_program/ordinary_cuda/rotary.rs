//! Source-exact ordinary Qwen rotary execution.
//!
//! BF16 trig cache + FP32 FMA; not the CPU profile's BF16 product rounding.
//! Request coordinates choose angles only; they grant
//! no execution, completion, or StateRoot authority.

use super::attention;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use cubecl::prelude::*;
use std::sync::Arc;
use zerogravity_core::{
    Error, Result,
    device_events::{BufferRole, DeviceAccess},
};

const WIDTH: usize = 256;
const ROTARY: usize = 64;
const HALF: usize = ROTARY / 2;
const ANGLE_BYTES: usize = HALF * 4;
const TRIG_BYTES: usize = HALF * 2 * 4;
const TABLE_BYTES: usize = crate::source_trig::TRIG_TABLES.len() * 8;

pub(super) const fn scratch_bytes(query_heads: usize, kv_heads: usize) -> usize {
    (query_heads + kv_heads) * WIDTH * 4 + ANGLE_BYTES + TRIG_BYTES
}
pub(super) const fn immutable_bytes() -> usize {
    TABLE_BYTES
}

fn grid(elements: usize) -> CubeCount {
    CubeCount::Static((elements as u32).div_ceil(32), 1, 1)
}
fn threads() -> CubeDim {
    CubeDim::new_1d(32)
}
fn arg(handle: &cubecl::server::Handle, elements: usize) -> BufferArg {
    // SAFETY: all callers validate exact byte extents and retain the original
    // leases in the enclosing resident flight.
    unsafe { BufferArg::from_raw_parts(handle.clone(), elements) }
}

pub(super) struct OrdinaryCudaRotary {
    service: Arc<ResidentService>,
    query_heads: usize,
    kv_heads: usize,
    query: ResidentLease,
    key: ResidentLease,
    angles: ResidentLease,
    trig: ResidentLease,
    tables: ResidentLease,
}
impl OrdinaryCudaRotary {
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        query_heads: usize,
        kv_heads: usize,
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
        require_heads(query_heads, kv_heads)?;
        let tables = crate::source_trig::TRIG_TABLES
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        if tables.len() != TABLE_BYTES {
            return Err(Error::State(
                "ordinary CUDA source trig table byte extent differs".into(),
            ));
        }
        let owner = Arc::new(Self {
            service,
            query_heads,
            kv_heads,
            query: init.allocate_empty(query_heads * WIDTH * 4, BufferRole::Scratch)?,
            key: init.allocate_empty(kv_heads * WIDTH * 4, BufferRole::Scratch)?,
            angles: init.allocate_empty(ANGLE_BYTES, BufferRole::Scratch)?,
            trig: init.allocate_empty(TRIG_BYTES, BufferRole::Scratch)?,
            tables: init.upload_bytes(&tables, BufferRole::ImmutableWeights)?,
        });
        init.retain_completion_owner(owner.clone())?;
        Ok(owner)
    }

    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        vec![
            (self.query.clone(), DeviceAccess::Write),
            (self.key.clone(), DeviceAccess::Write),
            (self.angles.clone(), DeviceAccess::Write),
            (self.trig.clone(), DeviceAccess::Write),
            (self.tables.clone(), DeviceAccess::Read),
        ]
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue(
        &self,
        query: &cubecl::server::Handle,
        key: &cubecl::server::Handle,
        inverse_frequency: &ResidentLease,
        status: &ResidentLease,
        base: usize,
        rows: usize,
    ) -> Result<usize> {
        self.service.ensure_usable()?;
        enqueue_source_exact_raw(
            self.service.client(),
            query,
            key,
            &inverse_frequency.handle(),
            &self.query.handle(),
            &self.key.handle(),
            &self.angles.handle(),
            &self.trig.handle(),
            &self.tables.handle(),
            &status.handle(),
            base,
            rows,
            self.query_heads,
            self.kv_heads,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn enqueue_source_exact_raw(
    client: &cubecl::client::Client,
    query: &cubecl::server::Handle,
    key: &cubecl::server::Handle,
    inverse_frequency: &cubecl::server::Handle,
    query_f32: &cubecl::server::Handle,
    key_f32: &cubecl::server::Handle,
    angles: &cubecl::server::Handle,
    trig: &cubecl::server::Handle,
    tables: &cubecl::server::Handle,
    status: &cubecl::server::Handle,
    base: usize,
    rows: usize,
    query_heads: usize,
    kv_heads: usize,
) -> Result<usize> {
    require_heads(query_heads, kv_heads)?;
    let query_elements = query_heads * WIDTH;
    let key_elements = kv_heads * WIDTH;
    let service = client.service_id();
    let query_bytes = rows
        .checked_mul(query_elements * 2)
        .ok_or_else(|| Error::Limit("ordinary rotary query batch bytes".into()))?;
    let key_bytes = rows
        .checked_mul(key_elements * 2)
        .ok_or_else(|| Error::Limit("ordinary rotary key batch bytes".into()))?;
    let exact = [
        (query, query_bytes),
        (key, key_bytes),
        (inverse_frequency, HALF * 4),
        (query_f32, query_heads * WIDTH * 4),
        (key_f32, kv_heads * WIDTH * 4),
        (angles, ANGLE_BYTES),
        (trig, TRIG_BYTES),
        (tables, TABLE_BYTES),
        (status, 4),
    ];
    if !(1..=64).contains(&rows)
        || base.checked_add(rows).is_none()
        || exact.iter().any(|(handle, bytes)| {
            handle.service != service || handle.size_in_used() != *bytes as u64
        })
    {
        return Err(Error::State(
            "ordinary CUDA source rotary exact service/extent differs".into(),
        ));
    }

    let mut launches = 0usize;
    for row in 0..rows {
        let position = base
            .checked_add(row)
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| Error::Limit("ordinary rotary position".into()))?;
        let query_row = exact_handle_view(
            query,
            row * query_elements * 2,
            query_elements * 2,
            "ordinary rotary query row",
        )?;
        let key_row = exact_handle_view(
            key,
            row * key_elements * 2,
            key_elements * 2,
            "ordinary rotary key row",
        )?;

        launches += attention::widen(client, &query_row, query_f32, status, query_elements)?;
        launches += attention::widen(client, &key_row, key_f32, status, key_elements)?;

        // SAFETY: all raw handles were validated to exact service/extent above.
        unsafe {
            crate::source_rotary::kernels::angles::launch_unchecked(
                client,
                grid(HALF),
                threads(),
                arg(inverse_frequency, HALF),
                arg(angles, HALF),
                position,
                HALF,
            );
            crate::source_trig::sincos_angles::launch_unchecked(
                client,
                grid(HALF),
                threads(),
                arg(angles, HALF),
                arg(tables, crate::source_trig::TRIG_TABLES.len()),
                arg(trig, HALF * 2),
                HALF,
            );
            crate::source_rotary::kernels::apply_fp32_fma::launch_unchecked(
                client,
                grid(query_heads * HALF),
                threads(),
                arg(query_f32, query_elements),
                arg(trig, HALF * 2),
                arg(query_f32, query_elements),
                arg(status, 1),
                WIDTH,
                query_elements,
                ROTARY,
            );
            crate::source_rotary::kernels::apply_fp32_fma::launch_unchecked(
                client,
                grid(kv_heads * HALF),
                threads(),
                arg(key_f32, key_elements),
                arg(trig, HALF * 2),
                arg(key_f32, key_elements),
                arg(status, 1),
                WIDTH,
                key_elements,
                ROTARY,
            );
        }
        launches += 4;
        launches += attention::pack(client, query_f32, &query_row, status, query_elements)?;
        launches += attention::pack(client, key_f32, &key_row, status, key_elements)?;
    }
    Ok(launches)
}

fn require_heads(query: usize, kv: usize) -> Result<()> {
    if !(1..=64).contains(&query) || !(1..=64).contains(&kv) || !query.is_multiple_of(kv) {
        return Err(Error::Unsupported(
            "source CUDA rotary grouped heads".into(),
        ));
    }
    Ok(())
}

fn exact_handle_view(
    handle: &cubecl::server::Handle,
    offset: usize,
    bytes: usize,
    role: &str,
) -> Result<cubecl::server::Handle> {
    let available = usize::try_from(handle.size_in_used())
        .map_err(|_| Error::Limit(format!("ordinary rotary {role} handle extent")))?;
    let end = offset
        .checked_add(bytes)
        .filter(|end| *end <= available)
        .ok_or_else(|| Error::State(format!("ordinary rotary {role} view exceeds handle")))?;
    Ok(handle
        .clone()
        .offset_start(offset as u64)
        .offset_end((available - end) as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_exact_rotary_inventory_is_constant_and_small() {
        assert_eq!(scratch_bytes(8, 2), 10_624);
        assert_eq!(scratch_bytes(24, 4), 29_056);
        assert_eq!(immutable_bytes(), 280);
    }
}
