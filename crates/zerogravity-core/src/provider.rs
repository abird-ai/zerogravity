//! Coarse source-owned preparation; no execution authority from a descriptor.
mod region;
mod resources;
mod selection;
mod state;
mod text;
use crate::{
    Error, Result,
    model::{ModelProgram, ModelRegion},
    semantic::ForwardPlan,
    source::ProgramSource,
};
pub use region::{PhysicalRegionPort, PreparedRegion, RegionShape};
pub use resources::{ProviderBufferMode, ProviderCompletionMode, ProviderRequirements};
pub use selection::{
    ExecutionMode, SelectedMode, SelectionPolicy, ShapeMissPolicy, StateBridge, SwitchingBoundary,
};
pub use state::{BoundRegionState, StateStorage};
use std::{future::Future, sync::Arc};
pub use text::{ObservationSupport, PreparedTextProgram};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderKind {
    Cpu,
    Cuda,
    Hip,
    Vulkan,
    GraphOnly,
    Remote,
}

/// Original semantic program and provider-owned resources. This interface does
/// not publish roots or expose Ready; existing typed execution adapters do that.
pub trait PreparedExecutable: Send + Sync {
    type Source: ProgramSource;
    fn source(&self) -> impl Future<Output = Result<Self::Source>>;
    fn program(&self) -> impl Future<Output = Result<Arc<ForwardPlan>>>;
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
/// As with original loaders, callers must drive submitted preparation to completion.
pub trait RegionExecutionProvider: Send + Sync {
    type Executable: PreparedExecutable;
    fn kind(&self) -> ProviderKind;
    fn prepare(
        &self,
        program: ModelProgram<<Self::Executable as PreparedExecutable>::Source>,
    ) -> impl Future<Output = Result<Self::Executable>>;
    fn prepare_region(
        &self,
        _region: ModelRegion<<Self::Executable as PreparedExecutable>::Source>,
    ) -> impl Future<Output = Result<Self::Executable>> {
        std::future::ready(Err(Error::Unsupported(
            "provider does not prepare closed regions".into(),
        )))
    }
}
