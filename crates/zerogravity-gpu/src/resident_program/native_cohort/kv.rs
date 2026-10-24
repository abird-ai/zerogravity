//! Actual independent root K/V ↔ one original B4 paged native cache.
//! Copies share the existing integer/BF16 equations; no activation mathematics.
use super::super::*;
use super::attention::CohortAttentionControls;
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue, model_state::BoundModelState, short_hf::ShortHfCallInput,
};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use crate::vllm_ptx_inline::PinnedKernel;
use cubecl_runtime::kernel::BufferIOAttr;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::StateEffect;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

const SOURCE: &str = include_str!("../../vllm_cuda_qualification/model_kv_copy.cu");
const CANONICAL_BYTES: usize = 512 * 512 * 4;
// Actual source pages4/8/12/16, FullAttentionSpec block544, paired K/V lanes.
pub(super) const CACHE_BYTES: usize = 17 * 2 * 544 * 512 * 2;
const METADATA_BYTES: usize = 512 * 1024;

pub(super) fn definition(incoming: bool) -> Result<PinnedKernel> {
    let entry = if incoming {
        "zg_cohort_kv_in"
    } else {
        "zg_cohort_kv_out"
    };
    let text: Arc<str> = format!(
        "// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_ENTRY {entry}\n#define ZG_IN {}\n#define ZG_DYNAMIC_CONTROLS 1\n{SOURCE}",
        usize::from(incoming),
    ).into();
    use BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
    let mut bindings = vec![
        (CANONICAL_BYTES, R),
        (CANONICAL_BYTES, R),
        (CACHE_BYTES, if incoming { RW } else { R }),
    ];
    if !incoming {
        bindings.extend([(CANONICAL_BYTES, W), (CANONICAL_BYTES, W)]);
    }
    bindings.extend([(8, R), (4, R), (4, R), (4, RW)]);
    PinnedKernel::new(entry, text, 128, bindings.into())
}

fn require_metadata_bound(definitions: &[PinnedKernel; 2]) -> Result<()> {
    // Source/ABI/key heaps are shared by all eight copies. Price Vec backing,
    // row/copy wrappers and owner bookkeeping under the existing512KiB lease.
    let retained = definitions.iter().try_fold(0usize, |sum, definition| {
        sum.checked_add(definition.retained_metadata_bytes()?)
            .ok_or_else(|| Error::Limit("B4 KV immutable metadata overflow".into()))
    })?;
    let wrappers = std::mem::size_of::<CohortKvTransport>()
        + 4 * std::mem::size_of::<Row>()
        + 8 * std::mem::size_of::<Copy>()
        + 4 * 16 * std::mem::size_of::<Handle>()
        + 16 * 1024; // Arc/Vec/initialization/access bookkeeping, conservatively.
    if retained
        .checked_add(wrappers)
        .is_none_or(|bytes| bytes > METADATA_BYTES)
    {
        return Err(Error::Limit("B4 KV metadata exceeds original512KiB".into()));
    }
    Ok(())
}

fn checked_footprint(base: usize, end: usize, block: usize) -> Result<()> {
    if base >= end || end > 512 || ![4, 8, 12, 16].contains(&block) {
        return Err(Error::Limit("B4 KV actual position/page bounds".into()));
    }
    let bytes = (block as u64 * 557_056 + 512 + (end as u64 - 1) * 1024 + 256 + 256) * 2;
    if bytes > CACHE_BYTES as u64 {
        return Err(Error::Limit("B4 KV actual native page footprint".into()));
    }
    Ok(())
}

fn control_spans(row: usize, logical_rows: usize) -> Result<[(usize, usize); 3]> {
    if row >= 4 || !matches!(logical_rows, 1 | 47) {
        return Err(Error::State("B4 KV actual control row geometry".into()));
    }
    Ok([(row * logical_rows * 8, 8), (row * 4, 4), (row * 16, 4)])
}

fn control_view(lease: &ResidentLease, offset: usize, bytes: usize) -> Result<Handle> {
    let end = offset
        .checked_add(bytes)
        .filter(|end| {
            *end <= lease.bytes() && matches!(bytes, 4 | 8) && offset.is_multiple_of(bytes)
        })
        .ok_or_else(|| Error::Limit("B4 KV exact readonly control view".into()))?;
    Ok(lease
        .handle()
        .offset_start(offset as u64)
        .offset_end((lease.bytes() - end) as u64))
}

struct Copy {
    kernel: PinnedKernel,
    grid: [u32; 3],
    handles: Vec<Handle>,
}
struct Row {
    input: Arc<ShortHfCallInput>,
    state: Arc<BoundModelState>,
    incoming: Option<Copy>,
    outgoing: Copy,
}
pub(super) struct CohortKvTransport {
    service: Arc<ResidentService>,
    cache: ResidentLease,
    status: ResidentLease,
    rows: Box<[Row]>,
    control_sources: [ResidentLease; 3],
    _controls: Arc<CohortAttentionControls>,
    _metadata: Arc<WindowWorkspace>,
}
impl CohortKvTransport {
    /// Each row owns actual current/candidate canonical planes. Cache is a
    /// private already initialized B4 arena, never captured original state.
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        inputs: &[Arc<ShortHfCallInput>],
        states: &[Arc<BoundModelState>],
        cache: ResidentLease,
        status: ResidentLease,
        controls: Arc<CohortAttentionControls>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
        crate::vllm_cuda_qualification::cohort_native::require_input_shape(inputs)?;
        if inputs.len() != 4
            || states.len() != 4
            || !Arc::ptr_eq(cache.service(), &service)
            || cache.bytes() != CACHE_BYTES
            || cache.role() != BufferRole::RecurrentState
            || !cache.is_initialized()
            || !Arc::ptr_eq(status.service(), &service)
            || status.bytes() != 4
            || status.role() != BufferRole::Scratch
        {
            return Err(Error::State(
                "cohort actual B4 paired cache/status admission".into(),
            ));
        }
        let source = inputs[0].source().source();
        inputs[0]
            .scope()
            .require_record(inputs[0].source(), inputs[0].record())?;
        let metadata = Arc::new(source.source().reserve_workspace(METADATA_BYTES)?);
        init.retain_completion_owner(metadata.clone())?;
        let control_sources = controls.kv_sources(inputs, &service)?;
        init.retain_completion_owner(controls.clone())?;
        let definitions = [definition(true)?, definition(false)?];
        require_metadata_bound(&definitions)?;
        let mut rows = Vec::with_capacity(4);
        let layer = states[0].layer().index();
        for (row, (input, state)) in inputs.iter().zip(states).enumerate() {
            state.require_short_hf(input.scope())?;
            if !state.layer().is_attention()
                || state.layer().index() != layer
                || state.call() != input.call()
                || input.workload().prompt_rows() != 47
                || !matches!((input.rows(), input.base()), (47, 0) | (1, 47..=53))
                || input.rows() != inputs[0].rows()
                || !Arc::ptr_eq(input.source().source(), source)
                || input.call().source() != service.source_identity()
            {
                return Err(Error::State(
                    "cohort KV exact original layer/current span/source".into(),
                ));
            }
            let base = input.base();
            let end = base
                .checked_add(input.rows())
                .ok_or_else(|| Error::Limit("cohort KV current end overflow".into()))?;
            checked_footprint(base, end, 4 + 4 * row)?;
            let spans = control_spans(row, input.rows())?;
            let control_handles = control_sources
                .iter()
                .zip(spans)
                .map(|(lease, (offset, bytes))| control_view(lease, offset, bytes))
                .collect::<Result<Vec<_>>>()?;
            let leases = state.leases();
            if leases.iter().any(|(lease, _)| {
                lease.bytes() != CANONICAL_BYTES
                    || !Arc::ptr_eq(lease.service(), &service)
                    || lease.id() == cache.id()
                    || lease.id() == status.id()
            }) {
                return Err(Error::State(
                    "cohort original canonical K/V plane ownership".into(),
                ));
            }
            let make = |incoming: bool| -> Result<Copy> {
                let mut handles = Vec::with_capacity(if incoming { 7 } else { 9 });
                handles.extend([leases[0].0.handle(), leases[2].0.handle(), cache.handle()]);
                if !incoming {
                    handles.extend([leases[1].0.handle(), leases[3].0.handle()]);
                }
                handles.extend(control_handles.iter().cloned());
                handles.push(status.handle());
                let kernel = definitions[usize::from(!incoming)].clone();
                let count = if incoming { base * 512 } else { 512 * 512 };
                let grid = [count.div_ceil(128) as u32, 1, 1];
                kernel.validate(service.client(), grid, &handles)?;
                Ok(Copy {
                    kernel,
                    grid,
                    handles,
                })
            };
            let incoming = (base != 0).then(|| make(true)).transpose()?;
            let outgoing = make(false)?;
            rows.push(Row {
                input: input.clone(),
                state: state.clone(),
                incoming,
                outgoing,
            });
        }
        // Controls cannot alias any actual state/cache/status owner. Original
        // controls are shared across layers, never mutable transport outputs.
        let state_owners: Vec<_> = rows
            .iter()
            .flat_map(|row| row.state.leases())
            .map(|(lease, _)| lease)
            .collect();
        for (index, control) in control_sources.iter().enumerate() {
            let range = BufferRange::new(&control.handle())?;
            for other in control_sources[..index]
                .iter()
                .chain([&cache, &status])
                .chain(state_owners.iter())
            {
                if range.overlaps(&BufferRange::new(&other.handle())?) {
                    return Err(Error::State(
                        "B4 KV control aliases transport storage".into(),
                    ));
                }
            }
        }
        let result = Arc::new(Self {
            service,
            cache,
            status,
            rows: rows.into_boxed_slice(),
            control_sources,
            _controls: controls,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    fn require_current(
        &self,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
    ) -> Result<()> {
        self.service.ensure_usable()?;
        if permits.len() != 4 || effects.len() != 4 {
            return Err(Error::State(
                "cohort KV all-row current effects/permits".into(),
            ));
        }
        if self
            .control_sources
            .iter()
            .any(|lease| !lease.is_initialized())
        {
            return Err(Error::State(
                "B4 KV original controls not initialized".into(),
            ));
        }
        // Validate every row before the first copy in this partition.
        for ((row, permit), effect) in self.rows.iter().zip(permits).zip(effects) {
            row.input.require_permit(permit, &self.service)?;
            row.state.require_enqueue(&self.service, permit)?;
            row.state.require_effect(*effect, false)?;
        }
        Ok(())
    }
    pub(super) fn before_append(
        &self,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
    ) -> Result<usize> {
        self.require_current(permits, effects)?;
        let mut launches = 0;
        for row in &self.rows {
            if let Some(copy) = &row.incoming {
                copy.kernel
                    .launch(self.service.client(), copy.grid, &copy.handles)?;
                launches += 1;
            }
        }
        Ok(launches)
    }
    pub(super) fn after_append(
        &self,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
    ) -> Result<usize> {
        self.require_current(permits, effects)?;
        for row in &self.rows {
            row.outgoing.kernel.launch(
                self.service.client(),
                row.outgoing.grid,
                &row.outgoing.handles,
            )?;
        }
        Ok(self.rows.len())
    }
    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.rows
            .iter()
            .flat_map(|row| row.state.leases())
            .chain(
                self.control_sources
                    .iter()
                    .cloned()
                    .map(|lease| (lease, DeviceAccess::Read)),
            )
            .chain([
                (self.cache.clone(), DeviceAccess::Write),
                (self.status.clone(), DeviceAccess::Write),
            ])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_runtime::kernel::KernelMetadata;

    #[test]
    fn two_direction_keys_cover_all_actual_b4_positions_and_rows() {
        let incoming = definition(true).unwrap();
        let outgoing = definition(false).unwrap();
        assert_ne!(incoming.id(), outgoing.id());
        require_metadata_bound(&[incoming.clone(), outgoing.clone()]).unwrap();
        for row in 0..4 {
            for (base, rows) in [(0, 47)].into_iter().chain((47..=53).map(|base| (base, 1))) {
                checked_footprint(base, base + rows, 4 + 4 * row).unwrap();
                let spans = control_spans(row, rows).unwrap();
                assert_eq!(spans.map(|(_, bytes)| bytes), [8, 4, 4]);
                assert_eq!(spans[0].0, row * rows * 8);
                for expected in [&incoming, &outgoing] {
                    let current = definition(expected.id() == incoming.id()).unwrap();
                    assert_eq!(expected.id(), current.id());
                    assert_eq!(expected.id().stable_hash(), current.id().stable_hash());
                    assert_eq!(
                        expected.id().mode,
                        cubecl::prelude::ExecutionMode::Unchecked
                    );
                    assert_eq!(expected.id().address_type, cubecl::ir::AddressType::U64);
                }
            }
        }
    }

    #[test]
    fn control_spans_reject_foreign_rows_and_keep_original_position_pitch() {
        assert_eq!(control_spans(3, 47).unwrap(), [(1128, 8), (12, 4), (48, 4)]);
        assert_eq!(control_spans(3, 1).unwrap(), [(24, 8), (12, 4), (48, 4)]);
        for (row, rows) in [(4, 1), (usize::MAX, 47), (0, 0), (0, 4), (0, usize::MAX)] {
            assert!(control_spans(row, rows).is_err());
        }
    }

    #[test]
    fn native_footprint_rejects_empty_overflow_and_unowned_pages() {
        for (base, end, page) in [
            (0, 0, 4),
            (48, 47, 4),
            (0, 513, 16),
            (0, usize::MAX, 16),
            (usize::MAX, 1, 4),
            (0, 47, 0),
            (0, 47, 5),
            (0, 47, 17),
            (0, 47, usize::MAX),
        ] {
            assert!(checked_footprint(base, end, page).is_err());
        }
        // Addressing can cover512 canonical positions without specializing
        // time; the constructor still enforces the original finite admission.
        checked_footprint(511, 512, 16).unwrap();
    }
}
