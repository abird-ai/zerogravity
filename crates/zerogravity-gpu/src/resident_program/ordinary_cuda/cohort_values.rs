//! Private native-cohort activation arena using the ordinary plan's liveness slots.
use super::layout::{self, OrdinaryCudaLayout};
use crate::resident_leases::{ResidentInitialization, ResidentLease};
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
}
