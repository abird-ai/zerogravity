//! Private native-cohort activation arena using the ordinary plan's liveness slots.
use super::layout::{self, OrdinaryCudaLayout};
use crate::{
    resident_leases::{ResidentInitialization, ResidentLease},
    resident_program::cohort_rows::CohortRowSpans,
};
use std::sync::Arc;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{ValueId, ValueStorageBinding},
    device_events::{BufferRole, DeviceAccess},
};

pub(super) type CohortValueAccesses = Vec<(ResidentLease, DeviceAccess)>;

pub(crate) struct OrdinaryCudaCohortValues {
    width: usize,
    slots: Arc<[ResidentLease]>,
    bindings: Arc<[ValueStorageBinding]>,
    token_ids: ResidentLease,
    head_rows: ResidentLease,
}

impl OrdinaryCudaCohortValues {
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
                    .and_then(|head| tokens.checked_add(head))
            })
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort control bytes".into()))?;
        value_bytes
            .checked_add(control_bytes)
            .filter(|&total| total <= 64 * 1024 * 1024)
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
        let head_rows = init.allocate_empty(
            layout.geometry().hidden_width * width * 2,
            BufferRole::Activation,
        )?;
        let actual = slots
            .iter()
            .try_fold(0usize, |sum, lease| {
                sum.checked_add(lease.bytes())
                    .ok_or_else(|| Error::Limit("ordinary CUDA cohort activation byte sum".into()))
            })?
            .checked_add(token_ids.bytes())
            .and_then(|sum| sum.checked_add(head_rows.bytes()))
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
        accesses.push((head_rows.clone(), DeviceAccess::Write));
        Ok((
            Arc::new(Self {
                width,
                slots: slots.into(),
                bindings: layout.value_bindings().to_vec().into(),
                token_ids,
                head_rows,
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
            .chain(std::iter::once(&self.head_rows))
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

    pub(crate) fn head_rows(&self) -> &ResidentLease {
        &self.head_rows
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
