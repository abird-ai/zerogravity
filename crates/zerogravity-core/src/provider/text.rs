//! Whole invocation preparation over the existing source/plan owners.
use super::SelectedMode;
use crate::{
    Error, Result,
    accelerator::program::InvocationPhase,
    model::ModelProgram,
    numerics::NumericalProfile,
    semantic::{BoundInvocation, BoundTextCall, InvocationPorts},
    source::{ProgramSource, TensorSource},
};
use std::{ops::RangeInclusive, sync::Arc};

/// A declared physical capability, not observation completion. AtProducer
/// includes requested value sites and original prior/provisional state planes.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ObservationSupport {
    #[default]
    None,
    AtProducer,
}
impl ObservationSupport {
    pub fn require(self, observed: bool) -> Result<()> {
        if observed && self == Self::None {
            return Err(Error::Unsupported(
                "provider has no original producer observation path".into(),
            ));
        }
        Ok(())
    }
}

/// Descriptive selected executable, not a capability or completion permit.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionDescription {
    pub selected: SelectedMode,
    /// None when the adapter has no explicit recorded arithmetic profile.
    pub numerical_profile: Option<NumericalProfile>,
    /// Only a real prepared text template supplies this binding contract.
    pub text_calls: Option<TextCallSupport>,
}
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextCallSupport {
    pub supported_rows: RangeInclusive<usize>,
    pub invocation_ports: InvocationPorts,
    pub observations: ObservationSupport,
}

/// Whole text-template executable contract. Physical state representation,
/// resource envelopes and completion remain with the real resident owner.
/// Unlike a partial region, it can select the original optional output tail.
pub struct PreparedTextProgram<S: ProgramSource = Arc<dyn TensorSource>> {
    model: ModelProgram<S>,
    selected: SelectedMode,
    rows: RangeInclusive<usize>,
    ports: InvocationPorts,
    observations: ObservationSupport,
}
impl<S: ProgramSource> PreparedTextProgram<S> {
    pub fn new(
        model: ModelProgram<S>,
        selected: SelectedMode,
        rows: RangeInclusive<usize>,
        observations: ObservationSupport,
    ) -> Result<Self> {
        let plan = model.program();
        let geometry = plan.source_plan_geometry().ok_or_else(|| {
            Error::Unsupported("prepared text program needs original source template".into())
        })?;
        if plan.text_invocation().is_some()
            || plan.frontier_delta() != 0
            || plan.logits().is_none()
            || plan.state_nodes().is_empty()
            || rows.is_empty()
            || *rows.start() == 0
            || *rows.end() > geometry.max_rows()
        {
            return Err(Error::Unsupported(
                "prepared whole text schedule/row band differs".into(),
            ));
        }
        let ports = InvocationPorts::for_nodes(plan.nodes())?;
        Ok(Self {
            model,
            selected,
            rows,
            ports,
            observations,
        })
    }
    pub fn model(&self) -> &ModelProgram<S> {
        &self.model
    }
    pub fn selected(&self) -> SelectedMode {
        self.selected
    }
    pub fn supported_rows(&self) -> RangeInclusive<usize> {
        self.rows.clone()
    }
    pub fn invocation_ports(&self) -> InvocationPorts {
        self.ports
    }
    pub fn observations(&self) -> ObservationSupport {
        self.observations
    }
    pub fn description(&self) -> ExecutionDescription {
        ExecutionDescription {
            selected: self.selected,
            numerical_profile: Some(self.model.program().numerical_profile()),
            text_calls: Some(TextCallSupport {
                supported_rows: self.rows.clone(),
                invocation_ports: self.ports,
                observations: self.observations,
            }),
        }
    }

    /// Refuse a foreign plan, shape/phase miss or unsupported observation
    /// before materialization, admission or enqueue. No additional allocation.
    pub fn bind<'a>(
        &self,
        invocation: &'a BoundInvocation,
        phase: InvocationPhase,
        observed: bool,
    ) -> Result<BoundTextCall<'a>> {
        if !Arc::ptr_eq(self.model.program(), invocation.plan()) {
            return Err(Error::State(
                "prepared text invocation belongs to another original program".into(),
            ));
        }
        if !self.rows.contains(&invocation.input().token_count()) {
            return Err(Error::Unsupported("prepared text row shape miss".into()));
        }
        self.observations.require(observed)?;
        invocation.call(phase)
    }
}
