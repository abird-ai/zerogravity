//! Coarse source-owned preparation; no execution authority from a descriptor.
mod description;
mod region;
mod resources;
mod selection;
mod state;
mod text;
use crate::{
    Error, Result,
    model::{ModelProgram, ModelRegion},
    semantic::ForwardPlan,
    source::{ProgramSource, RegionSource},
};
pub use description::{ProgramDescription, RegionDescription};
pub use region::{PhysicalRegionPort, PreparedRegion, RegionShape};
pub use resources::{
    ProviderBufferMode, ProviderCompletionMode, ProviderMemoryBudget, ProviderRequirements,
    ResidentModelMemory, ResidentTextAdmission,
};
pub use selection::{
    CompiledProvider, ExecutionMode, SelectedMode, SelectionPolicy, ShapeMissPolicy, StateBridge,
    SwitchingBoundary,
};
pub use state::{BoundRegionState, ResidentStatePolicy, StateStorage};
use std::{future::Future, sync::Arc};
pub use text::{ExecutionDescription, ObservationSupport, PreparedTextProgram, TextCallSupport};

#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderKind {
    Cpu,
    Cuda,
    Hip,
    Vulkan,
    /// Graph-only preparation without a selected native device. Not physical support.
    GraphOnly,
    Remote,
}

/// Original semantic program and provider-owned resources. This interface does
/// not publish roots or expose Ready; existing typed execution adapters do that.
pub trait PreparedExecutable: Send + Sync {
    type Source: ProgramSource;
    fn source(&self) -> impl Future<Output = Result<Self::Source>>;
    fn program(&self) -> impl Future<Output = Result<Arc<ForwardPlan>>>;
    /// The actual selected implementation. Legacy whole-model owners report
    /// their actual selection directly; typed preparations use their contract.
    fn selected(&self) -> Result<SelectedMode> {
        self.prepared_region()
            .map(PreparedRegion::selected)
            .or_else(|| self.prepared_text().map(PreparedTextProgram::selected))
            .ok_or_else(|| Error::State("prepared executable selection absent".into()))
    }
    /// Cold observation through the actual owner, never a fresh recording.
    fn description(&self) -> impl Future<Output = Result<ExecutionDescription>> {
        async {
            if let Some(region) = self.prepared_region() {
                Ok(region.description())
            } else if let Some(text) = self.prepared_text() {
                Ok(text.description())
            } else {
                let plan = self.program().await?;
                Ok(ExecutionDescription::recorded(&plan, self.selected()?))
            }
        }
    }
    /// Partial preparation carries the original region and physical ports.
    fn prepared_region(&self) -> Option<&PreparedRegion<Self::Source>> {
        None
    }
    /// Whole source-template text execution declares its actual call support.
    /// Older non-template owners retain their original typed execution layout.
    fn prepared_text(&self) -> Option<&PreparedTextProgram<Self::Source>> {
        None
    }
}
impl<T: PreparedExecutable + ?Sized> PreparedExecutable for Arc<T> {
    type Source = T::Source;
    fn source(&self) -> impl Future<Output = Result<Self::Source>> {
        self.as_ref().source()
    }
    fn program(&self) -> impl Future<Output = Result<Arc<ForwardPlan>>> {
        self.as_ref().program()
    }
    fn selected(&self) -> Result<SelectedMode> {
        self.as_ref().selected()
    }
    fn description(&self) -> impl Future<Output = Result<ExecutionDescription>> {
        self.as_ref().description()
    }
    fn prepared_region(&self) -> Option<&PreparedRegion<Self::Source>> {
        self.as_ref().prepared_region()
    }
    fn prepared_text(&self) -> Option<&PreparedTextProgram<Self::Source>> {
        self.as_ref().prepared_text()
    }
}

/// Whole-program and explicitly supported closed-region preparation.
/// The provider retains actual device/budget selection and physical ownership.
/// Preparation must complete original initialization or retirement before return.
/// Dropping the returned owner (including on engine verification refusal) must
/// safely retire its resources or retain them in the original quarantine.
/// As with original loaders, callers must drive submitted preparation to completion.
pub trait RegionExecutionProvider: Send + Sync {
    type Executable: PreparedExecutable;
    /// Exact cold selection, captured before preparation begins.
    fn selected(&self) -> SelectedMode;
    fn kind(&self) -> ProviderKind {
        self.selected().provider
    }
    fn prepare(
        &self,
        program: ModelProgram<<Self::Executable as PreparedExecutable>::Source>,
    ) -> impl Future<Output = Result<Self::Executable>>;
    /// Lower the authoritative closed region and bind its original learned
    /// operands via its typed source; no required export/import round trip.
    fn prepare_region(
        &self,
        _region: ModelRegion<<Self::Executable as PreparedExecutable>::Source>,
    ) -> impl Future<Output = Result<Self::Executable>>
    where
        <Self::Executable as PreparedExecutable>::Source: RegionSource,
    {
        std::future::ready(Err(Error::Unsupported(
            "provider does not prepare closed regions".into(),
        )))
    }
}
