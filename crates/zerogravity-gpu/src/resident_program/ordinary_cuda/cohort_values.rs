//! Private native-cohort activation arena using the ordinary plan's liveness slots.
use super::layout::{self, OrdinaryCudaLayout};
use crate::{
    cuda::gdn::{GdnRuntimeBuffer, GdnRuntimeWorkspaceLayout, RECURRENT_STATE_BYTES},
    resident_leases::{ResidentInitialization, ResidentLease},
    resident_program::cohort_rows::CohortRowSpans,
};
use std::{collections::BTreeMap, sync::Arc};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{StateSlot, ValueId, ValueStorageBinding},
    device_events::{BufferRole, DeviceAccess},
};

pub(super) type CohortValueAccesses = Vec<(ResidentLease, DeviceAccess)>;

pub(crate) struct OrdinaryCudaCohortValues {
    width: usize,
    slots: Arc<[ResidentLease]>,
    bindings: Arc<[ValueStorageBinding]>,
    token_ids: ResidentLease,
    // Generic hidden-width BF16 staging for cohort row transforms. The head
    // projection and any compatible independent projection share it serially.
    row_scratch: ResidentLease,
    // Optional width-8 native recurrent scratch. Canonical session state remains
    // authoritative; this only supplies four additional private physical rows
    // so a qualified B8 GDN core can execute between exact import/export edges.
    gdn_private_recurrent: Option<ResidentLease>,
    gdn_private_indices: Option<ResidentLease>,
    gdn_prefill_layout: Option<Arc<GdnRuntimeWorkspaceLayout>>,
    gdn_prefill_internal: BTreeMap<GdnRuntimeBuffer, ResidentLease>,
}

impl OrdinaryCudaCohortValues {
    fn has_gdn(layout: &OrdinaryCudaLayout) -> bool {
        layout.groups().direct()
            && layout
                .states()
                .iter()
                .any(|state| matches!(state.slot, StateSlot::Recurrent(_)))
    }

    fn build_gdn_prefill_layout(
        layout: &OrdinaryCudaLayout,
        width: usize,
    ) -> Result<Option<Arc<GdnRuntimeWorkspaceLayout>>> {
        if !Self::has_gdn(layout) {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            GdnRuntimeWorkspaceLayout::new_prefill_cohort(layout::gdn_runtime_config()?, width)?,
        )))
    }

    fn gdn_prefill_bytes(layout: &OrdinaryCudaLayout, width: usize) -> Result<usize> {
        let Some(gdn) = Self::build_gdn_prefill_layout(layout, width)? else {
            return Ok(0);
        };
        gdn.buffers()
            .iter()
            .filter(|buffer| {
                layout::owns_gdn_buffer(
                    buffer.role(),
                    buffer.ownership(),
                    !layout.groups().direct(),
                )
            })
            .try_fold(0usize, |sum, buffer| {
                sum.checked_add(buffer.bytes())
                    .ok_or_else(|| Error::Limit("ordinary CUDA cohort GDN workspace bytes".into()))
            })
    }

    fn gdn_private_bytes(layout: &OrdinaryCudaLayout, width: usize) -> Result<usize> {
        let recurrent = width == 8
            && layout.groups().direct()
            && layout.states().iter().any(|state| {
                matches!(state.slot, StateSlot::Recurrent(_))
                    && state.bytes == RECURRENT_STATE_BYTES
            });
        if !recurrent {
            return Ok(0);
        }
        RECURRENT_STATE_BYTES
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(width * 4))
            .ok_or_else(|| Error::Limit("ordinary CUDA private GDN cohort scratch".into()))
    }

    pub(super) fn required_bytes(layout: &OrdinaryCudaLayout, width: usize) -> Result<usize> {
        if width == 0 {
            return Ok(0);
        }
        if !matches!(width, 4 | 8) {
            return Err(Error::Unsupported(
                "ordinary CUDA cohort activation width must be four or eight".into(),
            ));
        }
        let value_bytes = layout.slot_bytes().iter().try_fold(0usize, |sum, &bytes| {
            let scaled = bytes
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("ordinary CUDA cohort activation slot bytes".into()))?;
            if scaled > 16 * 1024 * 1024 {
                return Err(Error::Limit(
                    "ordinary CUDA cohort activation slot exceeds physical bound".into(),
                ));
            }
            sum.checked_add(scaled)
                .filter(|&total| total <= 64 * 1024 * 1024)
                .ok_or_else(|| Error::Limit("ordinary CUDA cohort activation total bytes".into()))
        })?;
        let control_bytes = layout::MAX_ROWS
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .and_then(|tokens| {
                layout
                    .geometry()
                    .hidden_width
                    .checked_mul(width)
                    .and_then(|n| n.checked_mul(2))
                    .and_then(|row_scratch| tokens.checked_add(row_scratch))
            })
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort control bytes".into()))?;
        let private_bytes = Self::gdn_private_bytes(layout, width)?;
        let gdn_prefill_bytes = Self::gdn_prefill_bytes(layout, width)?;
        value_bytes
            .checked_add(control_bytes)
            .and_then(|bytes| bytes.checked_add(private_bytes))
            .and_then(|bytes| bytes.checked_add(gdn_prefill_bytes))
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort activation/control total".into()))
    }

    pub(super) fn allocate(
        layout: &OrdinaryCudaLayout,
        width: usize,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<(Arc<Self>, CohortValueAccesses)> {
        let expected = Self::required_bytes(layout, width)?;
        if expected == 0 {
            return Err(Error::State(
                "ordinary CUDA zero-width cohort activation allocation".into(),
            ));
        }
        let slots = layout
            .slot_bytes()
            .iter()
            .map(|&bytes| {
                bytes
                    .checked_mul(width)
                    .ok_or_else(|| {
                        Error::Limit("ordinary CUDA cohort activation slot extent".into())
                    })
                    .and_then(|bytes| init.allocate_empty(bytes, BufferRole::Activation))
            })
            .collect::<Result<Vec<_>>>()?;
        let token_ids =
            init.allocate_empty(layout::MAX_ROWS * width * 4, BufferRole::Activation)?;
        let row_scratch = init.allocate_empty(
            layout.geometry().hidden_width * width * 2,
            BufferRole::Activation,
        )?;
        let private_bytes = Self::gdn_private_bytes(layout, width)?;
        let (gdn_private_recurrent, gdn_private_indices) = if private_bytes == 0 {
            (None, None)
        } else {
            let recurrent = init.allocate_empty(RECURRENT_STATE_BYTES * 4, BufferRole::Scratch)?;
            let indices = init.upload_bytes(
                &(1i32..=8).flat_map(i32::to_ne_bytes).collect::<Vec<_>>(),
                BufferRole::Scratch,
            )?;
            (Some(recurrent), Some(indices))
        };
        let gdn_prefill_layout = Self::build_gdn_prefill_layout(layout, width)?;
        let mut gdn_prefill_internal = BTreeMap::new();
        if let Some(gdn) = &gdn_prefill_layout {
            for buffer in gdn.buffers().iter().filter(|buffer| {
                layout::owns_gdn_buffer(
                    buffer.role(),
                    buffer.ownership(),
                    !layout.groups().direct(),
                )
            }) {
                let lease = init.allocate_empty(buffer.bytes(), BufferRole::Scratch)?;
                if gdn_prefill_internal.insert(buffer.role(), lease).is_some() {
                    return Err(Error::State(
                        "ordinary CUDA duplicate cohort GDN internal role".into(),
                    ));
                }
            }
        }
        let actual = slots
            .iter()
            .try_fold(0usize, |sum, lease| {
                sum.checked_add(lease.bytes())
                    .ok_or_else(|| Error::Limit("ordinary CUDA cohort activation byte sum".into()))
            })?
            .checked_add(token_ids.bytes())
            .and_then(|sum| sum.checked_add(row_scratch.bytes()))
            .and_then(|sum| {
                sum.checked_add(
                    gdn_private_recurrent
                        .as_ref()
                        .map_or(0, |lease| lease.bytes()),
                )
            })
            .and_then(|sum| {
                sum.checked_add(
                    gdn_private_indices
                        .as_ref()
                        .map_or(0, |lease| lease.bytes()),
                )
            })
            .and_then(|sum| {
                gdn_prefill_internal
                    .values()
                    .try_fold(sum, |sum, lease| sum.checked_add(lease.bytes()))
            })
            .ok_or_else(|| {
                Error::Limit("ordinary CUDA cohort activation/control byte sum".into())
            })?;
        if actual != expected || slots.len() != layout.slot_bytes().len() {
            return Err(Error::State(
                "ordinary CUDA cohort activation allocation coverage".into(),
            ));
        }
        let mut accesses = slots
            .iter()
            .cloned()
            .map(|lease| (lease, DeviceAccess::Write))
            .collect::<Vec<_>>();
        accesses.push((token_ids.clone(), DeviceAccess::Write));
        accesses.push((row_scratch.clone(), DeviceAccess::Write));
        accesses.extend(
            gdn_private_recurrent
                .iter()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Write)),
        );
        accesses.extend(
            gdn_private_indices
                .iter()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Read)),
        );
        accesses.extend(
            gdn_prefill_internal
                .values()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Write)),
        );
        Ok((
            Arc::new(Self {
                width,
                slots: slots.into(),
                bindings: layout.value_bindings().to_vec().into(),
                token_ids,
                row_scratch,
                gdn_private_recurrent,
                gdn_private_indices,
                gdn_prefill_layout,
                gdn_prefill_internal,
            }),
            accesses,
        ))
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn leases(&self) -> impl Iterator<Item = &ResidentLease> {
        self.slots
            .iter()
            .chain(std::iter::once(&self.token_ids))
            .chain(std::iter::once(&self.row_scratch))
            .chain(self.gdn_private_recurrent.iter())
            .chain(self.gdn_private_indices.iter())
            .chain(self.gdn_prefill_internal.values())
    }

    pub(super) fn token_handle(&self, lane: usize, rows: usize) -> Result<cubecl::server::Handle> {
        if lane >= self.width || rows == 0 || rows > layout::MAX_ROWS {
            return Err(Error::State(
                "ordinary CUDA cohort token lane geometry".into(),
            ));
        }
        super::exact_view(
            &self.token_ids,
            lane * layout::MAX_ROWS * 4,
            rows * 4,
            "cohort token IDs",
        )
    }

    /// Fixed-capacity token view for a bounded-row kernel ABI. Only the active
    /// prefix is semantically readable; the kernel receives that active count
    /// separately and must not touch the remaining capacity.
    pub(super) fn token_handle_bounded(
        &self,
        lane: usize,
        active_rows: usize,
        bound_rows: usize,
    ) -> Result<cubecl::server::Handle> {
        if lane >= self.width
            || active_rows == 0
            || active_rows > bound_rows
            || bound_rows > layout::MAX_ROWS
        {
            return Err(Error::State(
                "ordinary CUDA bounded token lane geometry".into(),
            ));
        }
        super::exact_view(
            &self.token_ids,
            lane * layout::MAX_ROWS * 4,
            bound_rows * 4,
            "cohort bounded token IDs",
        )
    }

    pub(crate) fn row_scratch(&self) -> &ResidentLease {
        &self.row_scratch
    }

    pub(crate) fn gdn_private_recurrent_handle(
        &self,
        lane: usize,
    ) -> Result<cubecl::server::Handle> {
        if self.width != 8 || !(4..8).contains(&lane) {
            return Err(Error::State(
                "ordinary CUDA private GDN recurrent lane".into(),
            ));
        }
        let scratch = self.gdn_private_recurrent.as_ref().ok_or_else(|| {
            Error::State("ordinary CUDA private GDN recurrent scratch absent".into())
        })?;
        let offset = (lane - 4)
            .checked_mul(RECURRENT_STATE_BYTES)
            .ok_or_else(|| Error::Limit("ordinary CUDA private GDN recurrent offset".into()))?;
        super::exact_view(
            scratch,
            offset,
            RECURRENT_STATE_BYTES,
            "ordinary CUDA private GDN recurrent row",
        )
    }

    pub(crate) fn gdn_private_indices(&self) -> Result<&ResidentLease> {
        self.gdn_private_indices
            .as_ref()
            .ok_or_else(|| Error::State("ordinary CUDA private GDN B8 indices absent".into()))
    }

    pub(super) fn gdn_prefill_layout(&self) -> Result<Arc<GdnRuntimeWorkspaceLayout>> {
        self.gdn_prefill_layout
            .clone()
            .ok_or_else(|| Error::State("ordinary CUDA cohort GDN layout absent".into()))
    }

    pub(super) fn gdn_prefill_internal(&self, role: GdnRuntimeBuffer) -> Result<&ResidentLease> {
        self.gdn_prefill_internal
            .get(&role)
            .ok_or_else(|| Error::State("ordinary CUDA cohort GDN internal role absent".into()))
    }

    pub(super) fn gdn_prefill_internal_handles(
        &self,
    ) -> BTreeMap<GdnRuntimeBuffer, cubecl::server::Handle> {
        self.gdn_prefill_internal
            .iter()
            .map(|(&role, lease)| (role, lease.handle()))
            .collect()
    }

    pub(crate) fn lease(&self, value: ValueId) -> Result<&ResidentLease> {
        let binding = self
            .bindings
            .get(value.0 as usize)
            .ok_or_else(|| Error::State("ordinary CUDA cohort value binding absent".into()))?;
        self.slots
            .get(binding.slot)
            .ok_or_else(|| Error::State("ordinary CUDA cohort value slot absent".into()))
    }

    pub(crate) fn handle(
        &self,
        value: ValueId,
        elements_per_lane: usize,
        lanes: usize,
    ) -> Result<cubecl::server::Handle> {
        if elements_per_lane == 0 || lanes == 0 || lanes > self.width {
            return Err(Error::State(
                "ordinary CUDA cohort active value geometry".into(),
            ));
        }
        let bytes = elements_per_lane
            .checked_mul(lanes)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort active value bytes".into()))?;
        super::exact_view(self.lease(value)?, 0, bytes, "cohort active BF16 value")
    }

    pub(crate) fn lane_handle(
        &self,
        value: ValueId,
        elements_per_lane: usize,
        lane: usize,
    ) -> Result<cubecl::server::Handle> {
        if elements_per_lane == 0 || lane >= self.width {
            return Err(Error::State(
                "ordinary CUDA cohort lane value geometry".into(),
            ));
        }
        let offset = lane
            .checked_mul(elements_per_lane)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort lane value offset".into()))?;
        let bytes = elements_per_lane
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort lane value bytes".into()))?;
        super::exact_view(self.lease(value)?, offset, bytes, "cohort lane BF16 value")
    }

    /// Exact packed active region for a row-proportional value across ragged
    /// independent lanes. The span object owns only checked row geometry; this
    /// arena still owns the physical allocation and value-slot lifetime.
    pub(crate) fn packed_row_handle(
        &self,
        value: ValueId,
        elements_per_row: usize,
        spans: &CohortRowSpans,
    ) -> Result<cubecl::server::Handle> {
        if spans.len() > self.width {
            return Err(Error::State(
                "ordinary CUDA cohort row spans exceed activation width".into(),
            ));
        }
        let elements = spans.packed_elements(elements_per_row)?;
        let bytes = elements
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary CUDA packed row BF16 bytes".into()))?;
        super::exact_view(self.lease(value)?, 0, bytes, "cohort packed BF16 rows")
    }

    /// Stable-capacity view over one packed row-proportional cohort value.
    /// Active rows remain exactly `spans.total_rows()`; the extra pointer extent
    /// is only for a bounded physical ABI and grants no padded mathematical rows.
    pub(crate) fn packed_row_bounded_handle(
        &self,
        value: ValueId,
        elements_per_row: usize,
        spans: &CohortRowSpans,
        bound_rows: usize,
    ) -> Result<cubecl::server::Handle> {
        let maximum = layout::MAX_ROWS
            .checked_mul(self.width)
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort row capacity".into()))?;
        if spans.len() > self.width
            || elements_per_row == 0
            || spans.total_rows() > bound_rows
            || bound_rows > maximum
        {
            return Err(Error::State(
                "ordinary CUDA bounded packed cohort geometry".into(),
            ));
        }
        let bytes = bound_rows
            .checked_mul(elements_per_row)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA bounded packed BF16 bytes".into()))?;
        super::exact_view(
            self.lease(value)?,
            0,
            bytes,
            "cohort bounded packed BF16 rows",
        )
    }

    /// One lane view within a packed ragged row-proportional value.
    pub(crate) fn lane_row_handle(
        &self,
        value: ValueId,
        elements_per_row: usize,
        spans: &CohortRowSpans,
        lane: usize,
    ) -> Result<cubecl::server::Handle> {
        if spans.len() > self.width {
            return Err(Error::State(
                "ordinary CUDA cohort row spans exceed activation width".into(),
            ));
        }
        let elements = spans.element_span(lane, elements_per_row)?;
        let offset = elements
            .start
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary CUDA packed row BF16 offset".into()))?;
        let bytes = (elements.end - elements.start)
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary CUDA packed row lane BF16 bytes".into()))?;
        super::exact_view(
            self.lease(value)?,
            offset,
            bytes,
            "cohort packed lane BF16 rows",
        )
    }

    /// One contiguous group of packed lanes. This is pure physical row
    /// geometry; callers retain all session/root/model semantics.
    pub(crate) fn lane_group_row_handle(
        &self,
        value: ValueId,
        elements_per_row: usize,
        spans: &CohortRowSpans,
        first_lane: usize,
        lanes: usize,
    ) -> Result<cubecl::server::Handle> {
        if elements_per_row == 0
            || lanes == 0
            || first_lane
                .checked_add(lanes)
                .is_none_or(|end| end > spans.len() || end > self.width)
        {
            return Err(Error::State(
                "ordinary CUDA cohort lane-group geometry".into(),
            ));
        }
        let first = spans.row_span(first_lane)?.start;
        let last = spans.row_span(first_lane + lanes - 1)?.end;
        let offset = first
            .checked_mul(elements_per_row)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA lane-group offset".into()))?;
        let bytes = (last - first)
            .checked_mul(elements_per_row)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA lane-group bytes".into()))?;
        super::exact_view(
            self.lease(value)?,
            offset,
            bytes,
            "cohort packed lane-group BF16 rows",
        )
    }

    /// Capacity-sized view beginning at one packed lane. Bounded-row kernels
    /// may see overlapping *capacity* views, but execute sequentially and are
    /// required to touch only their active prefix. This stabilizes physical
    /// pointer extents without padding mathematical rows.
    pub(crate) fn lane_row_bounded_handle(
        &self,
        value: ValueId,
        elements_per_row: usize,
        spans: &CohortRowSpans,
        lane: usize,
        bound_rows: usize,
    ) -> Result<cubecl::server::Handle> {
        let active_rows = spans.lane_rows(lane)?;
        if spans.len() > self.width
            || elements_per_row == 0
            || active_rows > bound_rows
            || bound_rows > layout::MAX_ROWS
        {
            return Err(Error::State(
                "ordinary CUDA bounded cohort lane geometry".into(),
            ));
        }
        let start = spans
            .row_span(lane)?
            .start
            .checked_mul(elements_per_row)
            .ok_or_else(|| Error::Limit("ordinary CUDA bounded lane offset".into()))?;
        let offset = start
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary CUDA bounded lane BF16 offset".into()))?;
        let bytes = bound_rows
            .checked_mul(elements_per_row)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA bounded lane BF16 bytes".into()))?;
        super::exact_view(
            self.lease(value)?,
            offset,
            bytes,
            "cohort bounded lane BF16 rows",
        )
    }
}
