//! Coarse source-owned preparation; no execution authority from a descriptor.
mod selection;
use crate::{Result, model::ModelProgram, semantic::ForwardPlan, source::ProgramSource};
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
}
impl<T: PreparedExecutable + ?Sized> PreparedExecutable for Arc<T> {
    type Source = T::Source;
    fn source(&self) -> impl Future<Output = Result<Self::Source>> {
        self.as_ref().source()
    }
    fn program(&self) -> impl Future<Output = Result<Arc<ForwardPlan>>> {
        self.as_ref().program()
    }
}

/// Whole-program closed regions only. The provider retains actual device/budget
/// selection; arbitrary placement and partial-region replacement are not exposed.
/// Preparation must complete original initialization or retirement before return.
/// As with original loaders, callers must drive submitted preparation to completion.
pub trait RegionExecutionProvider: Send + Sync {
    type Executable: PreparedExecutable;
    fn kind(&self) -> ProviderKind;
    fn prepare(
        &self,
        program: ModelProgram<<Self::Executable as PreparedExecutable>::Source>,
    ) -> impl Future<Output = Result<Self::Executable>>;
}
