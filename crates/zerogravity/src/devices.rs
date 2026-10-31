//! Device selection descriptions remain separate from discovered/qualified hardware.
use crate::{Error, Result};
pub use zerogravity_core::placement::{PlacementRequest, PlacementTarget, TransferRequest};
pub use zerogravity_core::run_policy::{Backend, DeviceRef};
pub struct Devices<'a> {
    runtime: &'a zerogravity_engine::Runtime,
}
impl<'a> Devices<'a> {
    pub(crate) fn new(runtime: &'a zerogravity_engine::Runtime) -> Self {
        Self { runtime }
    }
    pub fn inventory(&self) -> Result<Vec<DeviceRef>> {
        self.runtime.require_open()?;
        Err(Error::Unsupported(
            "Fabric inventory awaits F1 extraction; CLI inventory unchanged".into(),
        ))
    }
    pub fn plan(&self, request: &PlacementRequest) -> Result<Vec<TransferRequest>> {
        self.runtime.require_open()?;
        zerogravity_engine::placement::plan(request)
    }
}
