//! Source-owned physical region contract. No allocation or completion permit.
use super::{BoundRegionState, ProviderRequirements, SelectedMode, StateStorage};
use crate::{
    Error, Result,
    accelerator::program::{SourceRowSelection, ValueId, ValueStorageFormat},
    model::ModelRegion,
    numerics::ActivationPrecision,
    placement::PreparedTensorLayout,
    semantic::{BoundInvocation, RegionPort, StateSlot},
    session_root::{RootAttempt, RootBinding},
    source::{ProgramSource, TensorSource},
};
use std::{ops::RangeInclusive, sync::Arc};

/// Logical precision and physical encoding are independent. These dense views
/// describe caller-owned storage; an actual backend must bind admitted leases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalRegionPort {
    logical: RegionPort,
    format: ValueStorageFormat,
    capacity: PreparedTensorLayout,
}
impl PhysicalRegionPort {
    fn new(logical: RegionPort, format: ValueStorageFormat, max_rows: usize) -> Result<Self> {
        if format == ValueStorageFormat::Bf16
            && logical.precision != ActivationPrecision::Bf16WidenedF32
        {
            return Err(Error::Unsupported(
                "region storage changes logical precision".into(),
            ));
        }
        let bytes = format.byte_width();
        let row_bytes = logical
            .width
            .checked_mul(bytes)
            .ok_or_else(|| Error::Limit("region physical row extent overflow".into()))?;
        let rows = match logical.rows {
            SourceRowSelection::AllActive => max_rows,
            SourceRowSelection::LastActive => 1,
        };
        let capacity = PreparedTensorLayout::dense(bytes, row_bytes, rows)?;
        Ok(Self {
            logical,
            format,
            capacity,
        })
    }
    pub fn value(self) -> ValueId {
        self.logical.value
    }
    pub fn format(self) -> ValueStorageFormat {
        self.format
    }
    pub fn logical(self) -> RegionPort {
        self.logical
    }
    fn layout(self, active_rows: usize) -> Result<PreparedTensorLayout> {
        match self.logical.rows {
            SourceRowSelection::AllActive => self.capacity.prefix_rows(active_rows),
            SourceRowSelection::LastActive => Ok(self.capacity),
        }
    }
}

/// Selected lowering contract over the exact source and semantic region.
/// Preparation/retirement of actual code, workspace, streams and leases remain
/// the responsibility of the original physical owner, not this descriptor.
pub struct PreparedRegion<S: ProgramSource = Arc<dyn TensorSource>> {
    model: ModelRegion<S>,
    selected: SelectedMode,
    requirements: ProviderRequirements,
    rows: RangeInclusive<usize>,
    inputs: Box<[PhysicalRegionPort]>,
    outputs: Box<[PhysicalRegionPort]>,
    states: Box<[StateStorage]>,
}
impl<S: ProgramSource> PreparedRegion<S> {
    pub fn new(
        model: ModelRegion<S>,
        selected: SelectedMode,
        requirements: ProviderRequirements,
        rows: RangeInclusive<usize>,
        input_formats: &[ValueStorageFormat],
        output_formats: &[ValueStorageFormat],
    ) -> Result<Self> {
        Self::with_state(
            model,
            selected,
            requirements,
            rows,
            input_formats,
            output_formats,
            &[],
        )
    }

    /// Exact original state layouts are required for stateful preparation.
    /// Nothing may silently infer state storage from activation format.
    pub fn with_state(
        model: ModelRegion<S>,
        selected: SelectedMode,
        requirements: ProviderRequirements,
        rows: RangeInclusive<usize>,
        input_formats: &[ValueStorageFormat],
        output_formats: &[ValueStorageFormat],
        states: &[StateStorage],
    ) -> Result<Self> {
        requirements.device_bytes()?;
        let region = model.region();
        let state_ports = || region.state_inputs().iter().chain(region.state_outputs());
        for (i, storage) in states.iter().enumerate() {
            if states[..i]
                .iter()
                .any(|prior| prior.slot() == storage.slot())
                || !state_ports().any(|port| port.version.slot == storage.slot())
            {
                return Err(Error::State("region extra/duplicate state storage".into()));
            }
        }
        for &port in state_ports() {
            states
                .iter()
                .find(|storage| storage.slot() == port.version.slot)
                .ok_or_else(|| Error::State("region state storage absent".into()))?
                .require_port(port)?;
        }
        if rows.is_empty() || *rows.start() == 0 || *rows.end() > region.max_rows() {
            return Err(Error::Unsupported(
                "region prepared row band outside semantic bounds".into(),
            ));
        }
        let ports = |logical: &[RegionPort], formats: &[ValueStorageFormat]| {
            if logical.len() != formats.len() {
                return Err(Error::Malformed(
                    "region physical port arity differs".into(),
                ));
            }
            logical
                .iter()
                .zip(formats)
                .map(|(&port, &format)| PhysicalRegionPort::new(port, format, *rows.end()))
                .collect::<Result<Box<[_]>>>()
        };
        let inputs = ports(region.inputs(), input_formats)?;
        let outputs = ports(region.outputs(), output_formats)?;
        Ok(Self {
            model,
            selected,
            requirements,
            rows,
            inputs,
            outputs,
            states: states.into(),
        })
    }
    pub fn model(&self) -> &ModelRegion<S> {
        &self.model
    }
    pub fn selected(&self) -> SelectedMode {
        self.selected
    }
    pub fn requirements(&self) -> ProviderRequirements {
        self.requirements
    }
    pub fn state_storage(&self, slot: StateSlot) -> Result<StateStorage> {
        self.states
            .iter()
            .find(|state| state.slot() == slot)
            .copied()
            .ok_or_else(|| Error::State("prepared region state domain absent".into()))
    }
    pub fn bind_state<'a>(
        &'a self,
        invocation: &'a BoundInvocation,
        current: RootBinding,
        attempt: &'a RootAttempt,
    ) -> Result<BoundRegionState<'a, S>> {
        BoundRegionState::new(self, invocation, current, attempt)
    }
    pub fn supported_rows(&self) -> RangeInclusive<usize> {
        self.rows.clone()
    }
    /// Geometry-only view for consumers that already checked the original
    /// invocation (native whole-program adapters). Other callers use bind.
    /// A shape miss is a pre-enqueue refusal, never an implicit fallback.
    pub fn shape(&self, rows: usize) -> Result<RegionShape<'_>> {
        if !self.rows.contains(&rows) {
            return Err(Error::Unsupported("region prepared row shape miss".into()));
        }
        Ok(RegionShape {
            inputs: &self.inputs,
            outputs: &self.outputs,
            rows,
        })
    }
    pub fn capacity(&self) -> RegionShape<'_> {
        RegionShape {
            inputs: &self.inputs,
            outputs: &self.outputs,
            rows: *self.rows.end(),
        }
    }
    /// Validate the original logical invocation before interpreting its storage.
    pub fn bind(&self, invocation: &BoundInvocation) -> Result<RegionShape<'_>> {
        self.model.region().bind(invocation)?;
        self.shape(invocation.input().token_count())
    }
}

/// Checked physical geometry only, without a handle, lease or Ready witness.
/// The original flight validates actual buffer identity and access.
pub struct RegionShape<'a> {
    inputs: &'a [PhysicalRegionPort],
    outputs: &'a [PhysicalRegionPort],
    rows: usize,
}
impl RegionShape<'_> {
    pub fn inputs(
        &self,
    ) -> impl Iterator<Item = Result<(PhysicalRegionPort, PreparedTensorLayout)>> + '_ {
        self.inputs
            .iter()
            .map(|&port| Ok((port, port.layout(self.rows)?)))
    }
    pub fn outputs(
        &self,
    ) -> impl Iterator<Item = Result<(PhysicalRegionPort, PreparedTensorLayout)>> + '_ {
        self.outputs
            .iter()
            .map(|&port| Ok((port, port.layout(self.rows)?)))
    }
    pub fn input(&self, index: usize) -> Result<PreparedTensorLayout> {
        self.inputs
            .get(index)
            .ok_or_else(|| Error::State("region input port absent".into()))?
            .layout(self.rows)
    }
    pub fn output(&self, index: usize) -> Result<PreparedTensorLayout> {
        self.outputs
            .get(index)
            .ok_or_else(|| Error::State("region output port absent".into()))?
            .layout(self.rows)
    }
}
