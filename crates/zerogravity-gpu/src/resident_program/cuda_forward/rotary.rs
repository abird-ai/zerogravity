//! Source-exact Qwen forward rotary execution.
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
const BATCH_ROWS: usize = 8;

pub(super) const fn scratch_bytes(query_heads: usize, kv_heads: usize) -> usize {
    BATCH_ROWS * ((query_heads + kv_heads) * WIDTH * 4 + ANGLE_BYTES + TRIG_BYTES)
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

pub(super) struct CudaForwardRotary {
    service: Arc<ResidentService>,
    query_heads: usize,
    kv_heads: usize,
    query: ResidentLease,
    key: ResidentLease,
    angles: ResidentLease,
    trig: ResidentLease,
    tables: ResidentLease,
}
impl CudaForwardRotary {
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
                "CUDA forward source trig table byte extent differs".into(),
            ));
        }
        let owner = Arc::new(Self {
            service,
            query_heads,
            kv_heads,
            query: init
                .allocate_empty(BATCH_ROWS * query_heads * WIDTH * 4, BufferRole::Scratch)?,
            key: init.allocate_empty(BATCH_ROWS * kv_heads * WIDTH * 4, BufferRole::Scratch)?,
            angles: init.allocate_empty(BATCH_ROWS * ANGLE_BYTES, BufferRole::Scratch)?,
            trig: init.allocate_empty(BATCH_ROWS * TRIG_BYTES, BufferRole::Scratch)?,
            tables: init.upload_bytes(&tables, BufferRole::ImmutableWeights)?,
        });
        init.retain_completion_owner(owner.clone())?;
        Ok(owner)
    }

    pub(super) fn tables(&self) -> &ResidentLease {
        &self.tables
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
        enqueue_source_exact_grouped_raw(
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

/// Group one causal contiguous span into the already-qualified 8/4/2 batched
/// rotary kernels, retaining the scalar source-exact path only for a final odd
/// row. This is physical launch grouping only: it owns no session/root policy.
#[allow(clippy::too_many_arguments)]
fn enqueue_source_exact_grouped_raw(
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
    if !(1..=64).contains(&rows) || base.checked_add(rows).is_none() {
        return Err(Error::State(
            "CUDA forward grouped rotary causal extent differs".into(),
        ));
    }
    let query_elements = query_heads * WIDTH;
    let key_elements = kv_heads * WIDTH;
    let service = client.service_id();
    let exact = [
        (query, rows * query_elements * 2),
        (key, rows * key_elements * 2),
        (inverse_frequency, HALF * 4),
        (query_f32, BATCH_ROWS * query_elements * 4),
        (key_f32, BATCH_ROWS * key_elements * 4),
        (angles, BATCH_ROWS * ANGLE_BYTES),
        (trig, BATCH_ROWS * TRIG_BYTES),
        (tables, TABLE_BYTES),
        (status, 4),
    ];
    if exact
        .iter()
        .any(|(handle, bytes)| handle.service != service || handle.size_in_used() != *bytes as u64)
    {
        return Err(Error::State(
            "CUDA forward grouped rotary exact service/extent differs".into(),
        ));
    }

    let mut offset = 0usize;
    let mut launches = 0usize;
    while offset < rows {
        let batch = rotary_batch_width(rows - offset);
        let query = exact_handle_view(
            query,
            offset * query_elements * 2,
            batch * query_elements * 2,
            "grouped rotary query",
        )?;
        let key = exact_handle_view(
            key,
            offset * key_elements * 2,
            batch * key_elements * 2,
            "grouped rotary key",
        )?;
        let query_f32 = exact_handle_view(
            query_f32,
            0,
            batch * query_elements * 4,
            "grouped rotary query scratch",
        )?;
        let key_f32 = exact_handle_view(
            key_f32,
            0,
            batch * key_elements * 4,
            "grouped rotary key scratch",
        )?;
        let angles = exact_handle_view(
            angles,
            0,
            batch * ANGLE_BYTES,
            "grouped rotary angle scratch",
        )?;
        let trig = exact_handle_view(trig, 0, batch * TRIG_BYTES, "grouped rotary trig scratch")?;
        let batch_base = base
            .checked_add(offset)
            .ok_or_else(|| Error::Limit("forward grouped rotary position".into()))?;
        launches += if batch == 1 {
            enqueue_source_exact_raw(
                client,
                &query,
                &key,
                inverse_frequency,
                &query_f32,
                &key_f32,
                &angles,
                &trig,
                tables,
                status,
                batch_base,
                1,
                query_heads,
                kv_heads,
            )?
        } else {
            enqueue_source_exact_batch_raw(
                client,
                &query,
                &key,
                inverse_frequency,
                &query_f32,
                &key_f32,
                &angles,
                &trig,
                tables,
                status,
                batch_base,
                batch,
                query_heads,
                kv_heads,
            )?
        };
        offset += batch;
    }
    Ok(launches)
}

/// Source-exact rotary over 2/4/8 independent rows with explicit positions.
/// The mathematical primitive is model-neutral; callers provide all geometry,
/// buffers, positions and publication authority.
#[allow(clippy::too_many_arguments)]
fn enqueue_source_exact_batch_raw(
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
    if !matches!(rows, 2 | 4 | 8) {
        return Err(Error::Unsupported(
            "CUDA forward rotary batch width must be two, four or eight".into(),
        ));
    }
    let mut positions = [0u32; BATCH_ROWS];
    for (row, position) in positions.iter_mut().enumerate().take(rows) {
        *position = base
            .checked_add(row)
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| Error::Limit("forward rotary position".into()))?;
    }
    let query_elements = query_heads * WIDTH;
    let key_elements = kv_heads * WIDTH;
    let service = client.service_id();
    let exact = [
        (query, rows * query_elements * 2),
        (key, rows * key_elements * 2),
        (inverse_frequency, HALF * 4),
        (query_f32, rows * query_elements * 4),
        (key_f32, rows * key_elements * 4),
        (angles, rows * ANGLE_BYTES),
        (trig, rows * TRIG_BYTES),
        (tables, TABLE_BYTES),
        (status, 4),
    ];
    if exact
        .iter()
        .any(|(handle, bytes)| handle.service != service || handle.size_in_used() != *bytes as u64)
    {
        return Err(Error::State(
            "CUDA forward source rotary batch exact service/extent differs".into(),
        ));
    }

    let mut launches = 0usize;
    launches += attention::widen(client, query, query_f32, status, rows * query_elements)?;
    launches += attention::widen(client, key, key_f32, status, rows * key_elements)?;
    // SAFETY: all handles above are exact service-local views. The retained
    // batched kernels are independently physically qualified against per-row
    // source-exact rotary for widths 2/4/8 and heterogeneous positions.
    unsafe {
        crate::source_rotary::kernels::angles_batched_explicit::launch_unchecked(
            client,
            grid(rows * HALF),
            threads(),
            arg(inverse_frequency, HALF),
            arg(angles, rows * HALF),
            positions[0],
            positions[1],
            positions[2],
            positions[3],
            positions[4],
            positions[5],
            positions[6],
            positions[7],
            HALF,
            rows,
        );
        crate::source_trig::sincos_angles::launch_unchecked(
            client,
            grid(rows * HALF),
            threads(),
            arg(angles, rows * HALF),
            arg(tables, crate::source_trig::TRIG_TABLES.len()),
            arg(trig, rows * HALF * 2),
            rows * HALF,
        );
        crate::source_rotary::kernels::apply_fp32_fma_batched::launch_unchecked(
            client,
            grid(rows * query_heads * HALF),
            threads(),
            arg(query_f32, rows * query_elements),
            arg(trig, rows * HALF * 2),
            arg(query_f32, rows * query_elements),
            arg(status, 1),
            WIDTH,
            query_heads,
            rows,
            ROTARY,
        );
        crate::source_rotary::kernels::apply_fp32_fma_batched::launch_unchecked(
            client,
            grid(rows * kv_heads * HALF),
            threads(),
            arg(key_f32, rows * key_elements),
            arg(trig, rows * HALF * 2),
            arg(key_f32, rows * key_elements),
            arg(status, 1),
            WIDTH,
            kv_heads,
            rows,
            ROTARY,
        );
    }
    launches += 4;
    launches += attention::pack(client, query_f32, query, status, rows * query_elements)?;
    launches += attention::pack(client, key_f32, key, status, rows * key_elements)?;
    Ok(launches)
}

fn rotary_batch_width(remaining: usize) -> usize {
    if remaining >= 8 {
        8
    } else if remaining >= 4 {
        4
    } else if remaining >= 2 {
        2
    } else {
        1
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
        .ok_or_else(|| Error::Limit("forward rotary query batch bytes".into()))?;
    let key_bytes = rows
        .checked_mul(key_elements * 2)
        .ok_or_else(|| Error::Limit("forward rotary key batch bytes".into()))?;
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
            "CUDA forward source rotary exact service/extent differs".into(),
        ));
    }

    let mut launches = 0usize;
    for row in 0..rows {
        let position = base
            .checked_add(row)
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| Error::Limit("forward rotary position".into()))?;
        let query_row = exact_handle_view(
            query,
            row * query_elements * 2,
            query_elements * 2,
            "forward rotary query row",
        )?;
        let key_row = exact_handle_view(
            key,
            row * key_elements * 2,
            key_elements * 2,
            "forward rotary key row",
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
        .map_err(|_| Error::Limit(format!("forward rotary {role} handle extent")))?;
    let end = offset
        .checked_add(bytes)
        .filter(|end| *end <= available)
        .ok_or_else(|| Error::State(format!("forward rotary {role} view exceeds handle")))?;
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
        assert_eq!(scratch_bytes(8, 2), 84_992);
        assert_eq!(scratch_bytes(24, 4), 232_448);
        assert_eq!(immutable_bytes(), 280);
    }

    #[test]
    fn grouped_rotary_uses_only_retained_power_of_two_widths() {
        for (rows, expected) in [
            (1, vec![1]),
            (3, vec![2, 1]),
            (7, vec![4, 2, 1]),
            (47, vec![8, 8, 8, 8, 8, 4, 2, 1]),
            (64, vec![8, 8, 8, 8, 8, 8, 8, 8]),
        ] {
            let mut remaining = rows;
            let mut actual = Vec::new();
            while remaining != 0 {
                let width = rotary_batch_width(remaining);
                actual.push(width);
                remaining -= width;
            }
            assert_eq!(actual, expected);
        }
    }

    #[cfg(all(
        feature = "qualification",
        feature = "cuda-cublas",
        target_os = "linux"
    ))]
    #[test]
    #[ignore = "requires physical NVIDIA CUDA"]
    fn grouped_wrapper_matches_scalar_source_exact_across_all_tail_widths() -> Result<()> {
        const ROWS: usize = 47;
        const QUERY_HEADS: usize = 8;
        const KV_HEADS: usize = 2;
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new(
            "forward-source-rotary-grouped-wrapper",
        )?;
        let client = diagnostic.service.client();
        let query_elements = QUERY_HEADS * WIDTH;
        let key_elements = KV_HEADS * WIDTH;
        let bf16_bytes = |elements: usize, seed: u16| {
            (0..elements)
                .flat_map(|index| {
                    let bits = seed.wrapping_add((index % 61) as u16);
                    bits.to_ne_bytes()
                })
                .collect::<Vec<_>>()
        };
        let f32_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let f64_bytes = |values: &[f64]| {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let query_bytes = bf16_bytes(ROWS * query_elements, 0x3e80);
        let key_bytes = bf16_bytes(ROWS * key_elements, 0x3f00);
        let grouped_query = client.create_from_slice(&query_bytes);
        let scalar_query = client.create_from_slice(&query_bytes);
        let grouped_key = client.create_from_slice(&key_bytes);
        let scalar_key = client.create_from_slice(&key_bytes);
        let inverse_frequency = client.create_from_slice(&f32_bytes(
            &(0..HALF)
                .map(|index| 1.0f32 / (1.0 + index as f32 * 0.125))
                .collect::<Vec<_>>(),
        ));
        let tables = client.create_from_slice(&f64_bytes(&crate::source_trig::TRIG_TABLES));
        let grouped_query_f32 =
            client.create_from_slice(&vec![0u8; BATCH_ROWS * query_elements * 4]);
        let grouped_key_f32 = client.create_from_slice(&vec![0u8; BATCH_ROWS * key_elements * 4]);
        let grouped_angles = client.create_from_slice(&vec![0u8; BATCH_ROWS * ANGLE_BYTES]);
        let grouped_trig = client.create_from_slice(&vec![0u8; BATCH_ROWS * TRIG_BYTES]);
        let scalar_query_f32 = client.create_from_slice(&vec![0u8; query_elements * 4]);
        let scalar_key_f32 = client.create_from_slice(&vec![0u8; key_elements * 4]);
        let scalar_angles = client.create_from_slice(&vec![0u8; ANGLE_BYTES]);
        let scalar_trig = client.create_from_slice(&vec![0u8; TRIG_BYTES]);
        let grouped_status = client.create_from_slice(&0u32.to_ne_bytes());
        let scalar_status = client.create_from_slice(&0u32.to_ne_bytes());

        let grouped_launches = enqueue_source_exact_grouped_raw(
            client,
            &grouped_query,
            &grouped_key,
            &inverse_frequency,
            &grouped_query_f32,
            &grouped_key_f32,
            &grouped_angles,
            &grouped_trig,
            &tables,
            &grouped_status,
            37,
            ROWS,
            QUERY_HEADS,
            KV_HEADS,
        )?;
        let scalar_launches = enqueue_source_exact_raw(
            client,
            &scalar_query,
            &scalar_key,
            &inverse_frequency,
            &scalar_query_f32,
            &scalar_key_f32,
            &scalar_angles,
            &scalar_trig,
            &tables,
            &scalar_status,
            37,
            ROWS,
            QUERY_HEADS,
            KV_HEADS,
        )?;
        let read = |handle: cubecl::server::Handle, role: &str| {
            client
                .read_one(handle)
                .map(|bytes| bytes.to_vec())
                .map_err(|error| Error::State(format!("forward rotary {role} readback: {error}")))
        };
        if read(grouped_query, "grouped query")? != read(scalar_query, "scalar query")?
            || read(grouped_key, "grouped key")? != read(scalar_key, "scalar key")?
            || read(grouped_status, "grouped status")? != read(scalar_status, "scalar status")?
        {
            return Err(Error::State(
                "forward grouped rotary wrapper differs from scalar source exact".into(),
            ));
        }
        assert_eq!(grouped_launches, 8 * 8);
        assert_eq!(scalar_launches, ROWS * 8);
        eprintln!(
            "FORWARD_ROTARY_GROUPED_PASS rows={ROWS} grouped_launches={grouped_launches} scalar_launches={scalar_launches} exact=true"
        );
        Ok(())
    }
}
