//! Read-only device observations and selection policy, not execution eligibility.
//!
//! Discovery may inspect OS metadata and, with Linux CUDA enabled, initialize
//! read-only CUDA/NVML metadata access. It creates no model, primary context,
//! allocation, admission, listener or execution permit. Runtime builders stay cold.
mod backend;
pub use backend::GpuBackend;
#[cfg(target_os = "linux")]
mod dev;
mod inventory;
mod policy;

use crate::{Error, Result};
pub use inventory::{Inventory, Observation, discover, normalize_pci};
#[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
pub use policy::host_available;
pub use policy::{allowed, fresh, observe_selected};
pub use zerogravity_core::placement::{PlacementRequest, PlacementTarget, TransferRequest};
pub use zerogravity_core::run_policy::{Backend, DeviceRef};

pub struct Devices<'a> {
    runtime: &'a zerogravity_engine::Runtime,
}
impl<'a> Devices<'a> {
    pub(crate) fn new(runtime: &'a zerogravity_engine::Runtime) -> Self {
        Self { runtime }
    }

    /// Observe hardware only after checking this same runtime is open.
    /// Observed capacity, backend bindings and exclusions do not qualify a source.
    pub fn inventory(&self) -> Result<Inventory> {
        self.runtime.require_open()?;
        discover()
    }

    /// Resolve an allowlist against a snapshot; does not admit resources or
    /// qualify a model/backend. `/dev` selectors inspect kernel metadata.
    pub fn allowed(
        &self,
        request: &crate::run_policy::Request,
        inventory: &Inventory,
    ) -> Result<Vec<Observation>> {
        self.runtime.require_open()?;
        allowed(request, inventory)
    }

    /// Same-domain placement requires no inter-domain transfer plan.
    /// Cross-domain transports remain unqualified. Not an admission or an
    /// allocation/stream compatibility proof.
    pub fn plan(&self, request: &PlacementRequest) -> Result<Vec<TransferRequest>> {
        self.runtime.require_open()?;
        zerogravity_engine::placement::plan(request)
    }

    /// Original finite CUDA HOST precheck; no GPU discovery or admission.
    #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
    pub fn host_available(
        &self,
        host_bytes: u64,
        headroom: &crate::run_policy::Headroom,
    ) -> Result<()> {
        self.runtime.require_open()?;
        host_available(host_bytes, headroom)
    }
}

/// Resolve a Linux character-device selector to its existing full PCI identity,
/// without opening the GPU node. Input follows `DeviceRef` selector syntax.
pub fn device_node_identity(path: &str) -> Result<String> {
    #[cfg(target_os = "linux")]
    {
        dev::resolve(path)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Err(Error::Unsupported("/dev device nodes require Linux".into()))
    }
}

fn bad(s: impl Into<String>) -> Error {
    Error::Malformed(s.into())
}

fn now_ms() -> Result<u128> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| bad("system clock before epoch"))?
        .as_millis())
}
