//! Source-exact ordinary Qwen rotary execution.
//!
//! This deliberately reuses the source arithmetic/trig schedule rather than
//! CUDA libdevice sincosf. Request coordinates choose angles only; they grant
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
const QUERY_HEADS: usize = 8;
const KV_HEADS: usize = 2;
const QUERY_ELEMENTS: usize = QUERY_HEADS * WIDTH;
const KEY_ELEMENTS: usize = KV_HEADS * WIDTH;
const QUERY_F32_BYTES: usize = QUERY_ELEMENTS * 4;
const KEY_F32_BYTES: usize = KEY_ELEMENTS * 4;
const ANGLE_BYTES: usize = HALF * 4;
const TRIG_BYTES: usize = HALF * 2 * 4;
const TABLE_BYTES: usize = crate::source_trig::TRIG_TABLES.len() * 8;

pub(super) const fn scratch_bytes() -> usize {
    QUERY_F32_BYTES + KEY_F32_BYTES + ANGLE_BYTES + TRIG_BYTES
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
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
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
            query: init.allocate_empty(QUERY_F32_BYTES, BufferRole::Scratch)?,
            key: init.allocate_empty(KEY_F32_BYTES, BufferRole::Scratch)?,
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
) -> Result<usize> {
    let service = client.service_id();
    let query_bytes = rows
        .checked_mul(QUERY_ELEMENTS * 2)
        .ok_or_else(|| Error::Limit("ordinary rotary query batch bytes".into()))?;
    let key_bytes = rows
        .checked_mul(KEY_ELEMENTS * 2)
        .ok_or_else(|| Error::Limit("ordinary rotary key batch bytes".into()))?;
    let exact = [
        (query, query_bytes),
        (key, key_bytes),
        (inverse_frequency, HALF * 4),
        (query_f32, QUERY_F32_BYTES),
        (key_f32, KEY_F32_BYTES),
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
            row * QUERY_ELEMENTS * 2,
            QUERY_ELEMENTS * 2,
            "ordinary rotary query row",
        )?;
        let key_row = exact_handle_view(
            key,
            row * KEY_ELEMENTS * 2,
            KEY_ELEMENTS * 2,
            "ordinary rotary key row",
        )?;

        launches += attention::widen(client, &query_row, query_f32, status, QUERY_ELEMENTS)?;
        launches += attention::widen(client, &key_row, key_f32, status, KEY_ELEMENTS)?;

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
            crate::source_rotary::kernels::apply::launch_unchecked(
                client,
                grid(QUERY_HEADS * HALF),
                threads(),
                arg(query_f32, QUERY_ELEMENTS),
                arg(trig, HALF * 2),
                arg(query_f32, QUERY_ELEMENTS),
                arg(status, 1),
                WIDTH,
                QUERY_ELEMENTS,
                ROTARY,
            );
            crate::source_rotary::kernels::apply::launch_unchecked(
                client,
                grid(KV_HEADS * HALF),
                threads(),
                arg(key_f32, KEY_ELEMENTS),
                arg(trig, HALF * 2),
                arg(key_f32, KEY_ELEMENTS),
                arg(status, 1),
                WIDTH,
                KEY_ELEMENTS,
                ROTARY,
            );
        }
        launches += 4;
        launches += attention::pack(client, query_f32, &query_row, status, QUERY_ELEMENTS)?;
        launches += attention::pack(client, key_f32, &key_row, status, KEY_ELEMENTS)?;
    }
    Ok(launches)
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
        assert_eq!(scratch_bytes(), 10_624);
        assert_eq!(immutable_bytes(), 280);
    }
}
