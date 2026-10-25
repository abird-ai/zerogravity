//! Actual independent canonical GDN planes ↔ one B4 private native cache.
//! Reuses original exact integer permutations, no model arithmetic/Root mint.
//! The per-row kernels stay compiled cold as the independent bit oracle and
//! admission fallback keys; the grouped row-dispatch variant performs the
//! actual per-call transport as ONE launch per partition.
use super::super::*;
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue, model_state::BoundModelState, short_hf::ShortHfCallInput,
};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use crate::resident_state_transport::typed::{
    CONV_ELEMENTS, PreparedTypedTransport, RECURRENT_ELEMENTS, TypedDirection, TypedLayout,
    TypedRow,
};
use crate::vllm_ptx_inline::PinnedKernel;
use cubecl_runtime::kernel::BufferIOAttr;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::StateEffect;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

const SOURCE: &str = include_str!("../../vllm_cuda_gdn/resident/state_copy.cu");
const CONV: usize = 6144 * 3;
const RECURRENT: usize = 16 * 128 * 128;
const SLOT_BYTES: usize = 278_528 * 4;
const CACHE_BYTES: usize = 17 * SLOT_BYTES;

/// One actual row's per-kind transport planes: (source, destination) in the
/// original per-row kernel order. Identical addresses to the per-row kernels.
struct Row {
    input: Arc<ShortHfCallInput>,
    state: Arc<BoundModelState>,
    planes: [(Handle, Handle); 4],
}
pub(super) struct CohortGdnState {
    service: Arc<ResidentService>,
    cache: ResidentLease,
    status: ResidentLease,
    rows: Box<[Row]>,
    prefill_h: Option<(ResidentLease, ResidentLease)>,
    // Cold per-row (bit oracle/fallback) and grouped definitions, cloned from
    // the shared CohortTransportLibrary; grids keep the original equations.
    grouped: Arc<[PinnedKernel; 4]>,
    grids: [[u32; 3]; 4],
    _metadata: Arc<WindowWorkspace>,
}
fn plane(cache: &ResidentLease, offset: usize, bytes: usize) -> Result<Handle> {
    let end = offset
        .checked_add(bytes)
        .filter(|end| *end <= cache.bytes())
        .ok_or_else(|| Error::State("cohort GDN actual native state plane range".into()))?;
    Ok(cache
        .handle()
        .offset_start(offset as u64)
        .offset_end((cache.bytes() - end) as u64))
}
/// Original per-row kernel: exact state transport between one row's canonical
/// plane and its native view. Kept as the independent qualification oracle and
/// admission fallback key; unchanged source/ABI/key.
pub(super) fn definition(kind: usize) -> Result<PinnedKernel> {
    let (elements, input_width, output_width, input_type, output_type) = match kind {
        0 => (CONV, 4, 2, "unsigned int", "unsigned short"),
        1 => (CONV, 2, 4, "unsigned short", "unsigned int"),
        2 | 3 => (RECURRENT, 4, 4, "unsigned int", "unsigned int"),
        _ => return Err(Error::Unsupported("cohort GDN transport kind".into())),
    };
    let entry = [
        "zg_cohort_conv_in",
        "zg_cohort_conv_out",
        "zg_cohort_h_in",
        "zg_cohort_h_out",
    ][kind];
    let text: std::sync::Arc<str> = format!(
        "// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_ENTRY {entry}\n#define ZG_COPY {kind}\n#define ZG_ELEMENTS {elements}u\n#define ZG_INPUT {input_type}\n#define ZG_OUTPUT {output_type}\n{SOURCE}",
    )
    .into();
    PinnedKernel::new(
        entry,
        text,
        128,
        std::sync::Arc::from([
            (elements * input_width, BufferIOAttr::ReadOnly),
            (elements * output_width, BufferIOAttr::WriteOnly),
            (4, BufferIOAttr::ReadWrite),
        ]),
    )
}
impl CohortGdnState {
    #[allow(clippy::too_many_arguments)] // Cold transport definitions are one
    // additional immutable argument over the original admission sequence.
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        inputs: &[Arc<ShortHfCallInput>],
        states: &[Arc<BoundModelState>],
        cache: ResidentLease,
        status: ResidentLease,
        prefill_h: Option<(ResidentLease, ResidentLease)>,
        transport: Arc<super::CohortTransportLibrary>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
        crate::vllm_cuda_qualification::cohort_native::require_input_shape(inputs)?;
        if inputs.len() != 4
            || states.len() != 4
            || cache.bytes() != CACHE_BYTES
            || cache.role() != BufferRole::RecurrentState
            || !cache.is_initialized()
            || !Arc::ptr_eq(cache.service(), &service)
            || status.bytes() != 4
            || status.role() != BufferRole::Scratch
            || !Arc::ptr_eq(status.service(), &service)
        {
            return Err(Error::State(
                "cohort GDN actual B4 native cache/status".into(),
            ));
        }
        match (&prefill_h, inputs[0].rows()) {
            (Some((initial, final_state)), 47)
                if [initial, final_state].into_iter().all(|lease| {
                    Arc::ptr_eq(lease.service(), &service)
                        && lease.bytes() == 4 * RECURRENT * 4
                        && lease.role() == BufferRole::RecurrentState
                }) && !BufferRange::new(&initial.handle())?
                    .overlaps(&BufferRange::new(&final_state.handle())?) => {}
            (None, 1) => {}
            _ => {
                return Err(Error::State(
                    "cohort original prefill contiguous H0/Ht vs cached indexed H ABI".into(),
                ));
            }
        }
        let source = inputs[0].source().source();
        inputs[0]
            .scope()
            .require_record(inputs[0].source(), inputs[0].record())?;
        let metadata = Arc::new(source.source().reserve_workspace(512 * 1024)?);
        init.retain_completion_owner(metadata.clone())?;
        let layer = states[0].layer().index();
        let mut rows = Vec::with_capacity(4);
        for (row, (input, state)) in inputs.iter().zip(states).enumerate() {
            state.require_short_hf(input.scope())?;
            if state.layer().is_attention()
                || state.layer().index() != layer
                || state.call() != input.call()
                || input.workload().prompt_rows() != 47
                || !matches!((input.rows(), input.base()), (47, 0) | (1, 47..=53))
                || input.rows() != inputs[0].rows()
                || !Arc::ptr_eq(input.source().source(), source)
                || input.call().source() != service.source_identity()
            {
                return Err(Error::State(
                    "cohort GDN actual row layer/span/source".into(),
                ));
            }
            let leases = state.leases();
            if leases.iter().enumerate().any(|(index, (lease, _))| {
                lease.bytes() != if index < 2 { CONV * 4 } else { RECURRENT * 4 }
                    || !Arc::ptr_eq(lease.service(), &service)
                    || lease.id() == cache.id()
                    || lease.id() == status.id()
            }) {
                return Err(Error::State(
                    "cohort GDN real prior/candidate typed planes".into(),
                ));
            }
            // Actual original B4 RequestState mapping, no GdnCacheSlot N1 grant.
            let offset = (1 + layer % 4 + 4 * row) * SLOT_BYTES;
            let conv = plane(&cache, offset, CONV * 2)?;
            // Original prefill H kernel consumes/produces contiguous four-row
            // arrays; cached decode indexes the shared native cache directly.
            let (h_read, h_write) = if let Some((initial, final_state)) = &prefill_h {
                (
                    plane(initial, row * RECURRENT * 4, RECURRENT * 4)?,
                    plane(final_state, row * RECURRENT * 4, RECURRENT * 4)?,
                )
            } else {
                let native = plane(&cache, offset + CONV * 2, RECURRENT * 4)?;
                (native.clone(), native)
            };
            // Per-kind (source, destination) planes: identical addresses and
            // extents the four original per-row kernels consumed/produced.
            let planes = [
                (leases[0].0.handle(), conv.clone()),
                (conv, leases[1].0.handle()),
                (leases[2].0.handle(), h_read),
                (h_write, leases[3].0.handle()),
            ];
            rows.push(Row {
                input: input.clone(),
                state: state.clone(),
                planes,
            });
        }
        // Grouped grids keep the original per-row block equations; the row
        // axis moves to blockIdx.y (four actual rows).
        let mut grids = [[1, 1, 1]; 4];
        for (kind, grid) in grids.iter_mut().enumerate() {
            let elements = if kind < 2 {
                CONV_ELEMENTS
            } else {
                RECURRENT_ELEMENTS
            };
            *grid = [(elements as u32).div_ceil(128), 4, 1];
        }
        let result = Arc::new(Self {
            service,
            cache,
            status,
            rows: rows.into_boxed_slice(),
            prefill_h,
            grouped: transport.gdn_state_grouped.clone(),
            grids,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    /// Even tags import actual priors; odd tags produce actual candidates.
    /// All rows validate before the first command; one surrounding B node owns
    /// the real native math between these integer transport partitions.
    pub(super) fn require_storage(
        &self,
        inputs: &[Arc<ShortHfCallInput>],
        cache: &ResidentLease,
        status: &ResidentLease,
        prefill_h: Option<&(ResidentLease, ResidentLease)>,
    ) -> Result<()> {
        let same_h = match (&self.prefill_h, prefill_h) {
            (None, None) => true,
            (Some((a, b)), Some((c, d))) => a.id() == c.id() && b.id() == d.id(),
            _ => false,
        };
        if inputs.len() != self.rows.len()
            || self.cache.id() != cache.id()
            || self.status.id() != status.id()
            || !same_h
            || !Arc::ptr_eq(cache.service(), &self.service)
            || !Arc::ptr_eq(status.service(), &self.service)
            || self
                .rows
                .iter()
                .zip(inputs)
                .any(|(row, input)| !Arc::ptr_eq(&row.input, input))
        {
            return Err(Error::State(
                "B4 GDN transport exact current row/storage owners".into(),
            ));
        }
        self.service.ensure_usable()
    }
    pub(super) fn require_partition(
        &self,
        kind: usize,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
    ) -> Result<()> {
        self.service.ensure_usable()?;
        if kind >= 4 || permits.len() != 4 || effects.len() != 4 {
            return Err(Error::State(
                "cohort GDN all-row original transport partition".into(),
            ));
        }
        for ((row, permit), effect) in self.rows.iter().zip(permits).zip(effects) {
            row.input.require_permit(permit, &self.service)?;
            row.state.require_enqueue(&self.service, permit)?;
            row.state.require_effect(*effect, kind >= 2)?;
        }
        Ok(())
    }
    /// ONE grouped launch across all four actual rows, replacing the four
    /// per-row copies with identical planes, equations, guards and status
    /// epoch. Returns (launches, grouped launches).
    pub(super) fn enqueue(
        &self,
        kind: usize,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
    ) -> Result<(usize, usize)> {
        self.require_partition(kind, permits, effects)?;
        let layout = if kind < 2 {
            TypedLayout::Conv
        } else {
            TypedLayout::RecurrentH
        };
        let direction = if kind.is_multiple_of(2) {
            TypedDirection::Import
        } else {
            TypedDirection::Export
        };
        let rows = self
            .rows
            .iter()
            .map(|row| {
                let (source, destination) = &row.planes[kind];
                if kind.is_multiple_of(2) {
                    TypedRow {
                        prior: Some(source),
                        candidate: None,
                        native: destination,
                        second: None,
                        second_prior: None,
                        controls: None,
                    }
                } else {
                    TypedRow {
                        prior: None,
                        candidate: Some(destination),
                        native: source,
                        second: None,
                        second_prior: None,
                        controls: None,
                    }
                }
            })
            .collect::<Vec<_>>();
        PreparedTypedTransport::prepare(
            self.service.client(),
            &self.grouped[kind],
            &rows,
            &self.status.handle(),
            layout,
            direction,
            self.grids[kind],
        )?
        .launch(self.service.client())?;
        Ok((1, 1))
    }
    /// Per-kind census attribution: canonical prior/candidate plane leases,
    /// the native-side owner and the shared status lease.
    pub(super) fn partition_accesses(&self, kind: usize) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut accesses: Vec<(ResidentLease, DeviceAccess)> = Vec::with_capacity(6);
        for row in &self.rows {
            let leases = row.state.leases();
            let (lease, access) = match kind {
                0 => (leases[0].0.clone(), DeviceAccess::Read),
                1 => (leases[1].0.clone(), DeviceAccess::Write),
                2 => (leases[2].0.clone(), DeviceAccess::Read),
                _ => (leases[3].0.clone(), DeviceAccess::Write),
            };
            accesses.push((lease, access));
        }
        match (kind, &self.prefill_h) {
            (0, _) => accesses.push((self.cache.clone(), DeviceAccess::Write)),
            (1, _) => accesses.push((self.cache.clone(), DeviceAccess::Read)),
            (2, Some((initial, _))) => {
                accesses.push((initial.clone(), DeviceAccess::Write));
            }
            (2, None) => accesses.push((self.cache.clone(), DeviceAccess::Write)),
            (_, Some((_, final_state))) => {
                accesses.push((final_state.clone(), DeviceAccess::Read));
            }
            (_, None) => accesses.push((self.cache.clone(), DeviceAccess::Read)),
        }
        accesses.push((self.status.clone(), DeviceAccess::Write));
        accesses
    }
    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.rows
            .iter()
            .flat_map(|row| row.state.leases())
            .chain([
                (self.cache.clone(), DeviceAccess::Write),
                (self.status.clone(), DeviceAccess::Write),
            ])
            .chain(self.prefill_h.iter().flat_map(|(a, b)| {
                [
                    (a.clone(), DeviceAccess::Write),
                    (b.clone(), DeviceAccess::Write),
                ]
            }))
            .collect()
    }
}
