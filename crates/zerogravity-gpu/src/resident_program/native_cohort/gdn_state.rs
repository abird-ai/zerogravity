//! Actual independent canonical GDN planes ↔ one B4 private native cache.
//! Reuses original exact integer permutations, no model arithmetic/Root mint.
use super::super::*;
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue, model_state::BoundModelState, short_hf::ShortHfCallInput,
};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
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
struct Copy {
    kernel: PinnedKernel,
    grid: [u32; 3],
    handles: Vec<Handle>,
}
struct Row {
    input: Arc<ShortHfCallInput>,
    state: Arc<BoundModelState>,
    copies: [Copy; 4],
}
pub(super) struct CohortGdnState {
    service: Arc<ResidentService>,
    cache: ResidentLease,
    status: ResidentLease,
    rows: Box<[Row]>,
    prefill_h: Option<(ResidentLease, ResidentLease)>,
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
pub(super) fn definitions() -> Result<Vec<(PinnedKernel, [u32; 3])>> {
    (0..4usize).map(|kind| -> Result<(PinnedKernel, [u32; 3])> {
            let (elements, input_width, output_width, input_type, output_type) = match kind {
                0 => (CONV, 4, 2, "unsigned int", "unsigned short"),
                1 => (CONV, 2, 4, "unsigned short", "unsigned int"),
                _ => (RECURRENT, 4, 4, "unsigned int", "unsigned int"),
            };
            let entry = ["zg_cohort_conv_in", "zg_cohort_conv_out", "zg_cohort_h_in", "zg_cohort_h_out"][kind];
            let text: Arc<str> = format!(
                "// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_ENTRY {entry}\n#define ZG_COPY {kind}\n#define ZG_ELEMENTS {elements}u\n#define ZG_INPUT {input_type}\n#define ZG_OUTPUT {output_type}\n{SOURCE}",
            ).into();
            let kernel = PinnedKernel::new(entry, text, 128, Arc::from([
                (elements * input_width, BufferIOAttr::ReadOnly),
                (elements * output_width, BufferIOAttr::WriteOnly),
                (4, BufferIOAttr::ReadWrite),
            ]))?;
            Ok((kernel, [(elements as u32).div_ceil(128), 1, 1]))
        }).collect::<Result<Vec<_>>>()
}
impl CohortGdnState {
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        inputs: &[Arc<ShortHfCallInput>],
        states: &[Arc<BoundModelState>],
        cache: ResidentLease,
        status: ResidentLease,
        prefill_h: Option<(ResidentLease, ResidentLease)>,
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
        // Only four immutable source/ABI definitions, shared across real rows.
        let definitions = definitions()?;
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
            let mut copies = Vec::with_capacity(4);
            for (kind, (from, to)) in [
                (leases[0].0.handle(), conv.clone()),
                (conv, leases[1].0.handle()),
                (leases[2].0.handle(), h_read),
                (h_write, leases[3].0.handle()),
            ]
            .into_iter()
            .enumerate()
            {
                let (kernel, grid) = &definitions[kind];
                let handles = vec![from, to, status.handle()];
                kernel.validate(service.client(), *grid, &handles)?;
                copies.push(Copy {
                    kernel: kernel.clone(),
                    grid: *grid,
                    handles,
                });
            }
            let copies: [Copy; 4] = copies
                .try_into()
                .map_err(|_| Error::State("cohort GDN exact4 transport partitions".into()))?;
            rows.push(Row {
                input: input.clone(),
                state: state.clone(),
                copies,
            });
        }
        let result = Arc::new(Self {
            service,
            cache,
            status,
            rows: rows.into_boxed_slice(),
            prefill_h,
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
    pub(super) fn enqueue(
        &self,
        kind: usize,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
    ) -> Result<usize> {
        self.require_partition(kind, permits, effects)?;
        for row in &self.rows {
            let copy = &row.copies[kind];
            copy.kernel
                .launch(self.service.client(), copy.grid, &copy.handles)?;
        }
        Ok(self.rows.len())
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
