//! Coarse source-owned preparation; no execution authority from a descriptor.
mod region;
mod resources;
mod selection;
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
use std::{future::Future, sync::Arc};

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
    /// Partial preparation carries the original region and explicit physical
    /// port/shape contract. Whole-program owners retain their existing layout.
    fn prepared_region(&self) -> Option<&PreparedRegion<Self::Source>> {
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
