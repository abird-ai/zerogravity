//! Physical state layout/bridge descriptions over original semantic versions.
//! No state allocation, copy, event, Ready or publication authority.
use super::{PreparedRegion, StateBridge};
use crate::{
    Error, Result,
    accelerator::program::ValueStorageFormat,
    semantic::{BoundTextCall, RegionStatePort, StateLayout, StateSlot, StateVersion},
    session_root::{RootAttempt, RootBinding},
    source::ProgramSource,
};
use std::sync::Arc;

/// Prepared resident text state allocation policy. No page allocation or
/// provider eligibility follows from this value alone.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentStatePolicy {
    Contiguous,
    Paged {
        page_tokens: usize,
        max_append_tokens: usize,
    },
}

/// Explicit dense state storage, independently of logical arithmetic. The
/// original semantic program/profile supplies vector ordering and precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateStorage {
    slot: StateSlot,
    layout: StateLayout,
    format: ValueStorageFormat,
}
impl StateStorage {
    pub fn new(slot: StateSlot, layout: StateLayout, format: ValueStorageFormat) -> Result<Self> {
        if matches!(slot, StateSlot::Kv(_)) != matches!(layout, StateLayout::Kv { .. })
            || matches!(slot, StateSlot::Recurrent(_)) && format != ValueStorageFormat::F32
        {
            return Err(Error::Unsupported(
                "state storage domain/precision differs".into(),
            ));
        }
        let storage = Self {
            slot,
            layout,
            format,
        };
        storage.bytes()?;
        Ok(storage)
    }
    pub fn slot(self) -> StateSlot {
        self.slot
    }
    pub fn layout(self) -> StateLayout {
        self.layout
    }
    pub fn format(self) -> ValueStorageFormat {
        self.format
    }
    pub fn bytes(self) -> Result<usize> {
        let (planes, elements) = self.layout.planes()?;
        elements
            .checked_mul(planes)
            .and_then(|n| n.checked_mul(self.format.byte_width()))
            .ok_or_else(|| Error::Limit("state storage byte extent".into()))
    }
    pub fn require_port(self, port: RegionStatePort) -> Result<()> {
        if self.slot != port.version.slot || self.layout != port.layout {
            return Err(Error::State(
                "state storage differs from original region port".into(),
            ));
        }
        Ok(())
    }
}

/// One original invocation/attempt shared by adjacent prepared regions.
/// Describes state versions, not actual storage identity or enqueue authority.
/// Actual providers must still bind the same admitted private state leases.
pub struct BoundRegionState<'a, S: ProgramSource> {
    prepared: &'a PreparedRegion<S>,
    call: BoundTextCall<'a>,
    attempt: &'a RootAttempt,
}
impl<'a, S: ProgramSource> BoundRegionState<'a, S> {
    pub(super) fn new(
        prepared: &'a PreparedRegion<S>,
        call: BoundTextCall<'a>,
        current: RootBinding,
        attempt: &'a RootAttempt,
    ) -> Result<Self> {
        prepared.bind(call)?;
        let input = call.input();
        attempt.require_advance(
            current,
            input.source(),
            input.base() as u64,
            input.frontier() as u64,
        )?;
        Ok(Self {
            prepared,
            call,
            attempt,
        })
    }
    pub fn prior(&self) -> RootBinding {
        self.attempt.base()
    }
    pub fn frontier(&self) -> u64 {
        self.attempt.frontier()
    }
    pub fn inputs(&self) -> &[RegionStatePort] {
        self.prepared.model().region().state_inputs()
    }
    pub fn outputs(&self) -> &[RegionStatePort] {
        self.prepared.model().region().state_outputs()
    }

    /// A provisional producer can feed a later consumer on the exact original
    /// call/attempt. No shape-only retag, committed-root conversion or hidden
    /// retry. Passing this check does not establish physical completion.
    pub fn require_input_from(
        &self,
        producer: &Self,
        version: StateVersion,
        bridge: StateBridge,
    ) -> Result<()> {
        if bridge != StateBridge::OriginalLayout {
            return Err(Error::Unsupported(
                "state conversion requires an explicit qualified bridge".into(),
            ));
        }
        let consumer_region = self.prepared.model();
        let producer_region = producer.prepared.model();
        if version.ordinal != 1
            || !self.call.same_call(producer.call)
            || !std::ptr::eq(self.attempt, producer.attempt)
            || !Arc::ptr_eq(consumer_region.program(), producer_region.program())
            || !consumer_region
                .source()
                .same_issuer(producer_region.source())
            || producer_region.region().range().end > consumer_region.region().range().start
        {
            return Err(Error::State(
                "state bridge original invocation/attempt/order differs".into(),
            ));
        }
        let input = self.inputs().iter().find(|port| port.version == version);
        let output = producer
            .outputs()
            .iter()
            .find(|port| port.version == version);
        if input.is_none()
            || input != output
            || self.prepared.state_storage(version.slot)?
                != producer.prepared.state_storage(version.slot)?
        {
            return Err(Error::State(
                "state bridge original version/layout differs".into(),
            ));
        }
        Ok(())
    }
}
