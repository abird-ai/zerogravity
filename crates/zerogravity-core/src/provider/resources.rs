//! Additional provider-local resources and execution compatibility.
//! The original physical owner admits and retains them.
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderBufferMode {
    /// Import/borrow the original admitted port allocations.
    Borrowed,
    /// Provider-private storage needs an explicit admitted transfer bridge.
    Private,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderCompletionMode {
    /// All commands participate in the original physical queue's completion.
    OriginalQueue,
    /// Requires an actual external-event/stream completion bridge.
    ExternalFence,
}

/// Incremental device requirements of one prepared implementation, not an
/// allocation, accounting receipt, or completion witness. Ordinary operands,
/// state, existing mathematical scratch, and shared backend context/module
/// envelopes remain priced by their original owners, not once per region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderRequirements {
    /// Exclusive writable workspace supplied by the fabric, including any
    /// explicitly admitted managed private-buffer staging.
    pub workspace_bytes: usize,
    /// Additional model-scoped opaque provider reservation (not measured VRAM).
    pub opaque_device_bytes: usize,
    pub buffers: ProviderBufferMode,
    pub completion: ProviderCompletionMode,
}

impl ProviderRequirements {
    pub const fn original_queue(workspace_bytes: usize, opaque_device_bytes: usize) -> Self {
        Self {
            workspace_bytes,
            opaque_device_bytes,
            buffers: ProviderBufferMode::Borrowed,
            completion: ProviderCompletionMode::OriginalQueue,
        }
    }

    pub fn device_bytes(self) -> Result<usize> {
        self.workspace_bytes
            .checked_add(self.opaque_device_bytes)
            .ok_or_else(|| Error::Limit("provider additional device extent overflow".into()))
    }

    /// Current native/captured adapter has no implicit copy or foreign fence.
    pub fn require_original_queue(self) -> Result<()> {
        if self.buffers != ProviderBufferMode::Borrowed
            || self.completion != ProviderCompletionMode::OriginalQueue
        {
            return Err(Error::Unsupported(
                "provider requires an explicit buffer/completion bridge".into(),
            ));
        }
        Ok(())
    }

    /// Compare with actual original-owner extents after admission. The caller
    /// still proves exact lease, charge, access and queue identity.
    pub fn require_admitted(self, workspace: usize, opaque: usize) -> Result<()> {
        if self.workspace_bytes != workspace || self.opaque_device_bytes != opaque {
            return Err(Error::State(
                "provider admitted resource extents differ".into(),
            ));
        }
        Ok(())
    }
}

/// Cold device/provider envelope, not a reservation. The difference is
/// headroom, including any separately owned backend context. Use the same
/// clamp for discovery and the original actor's live observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderMemoryBudget {
    provider_bytes: usize,
    device_bytes: u64,
}
impl ProviderMemoryBudget {
    pub fn new(provider_bytes: usize, device_bytes: u64) -> Result<Self> {
        if provider_bytes == 0
            || u64::try_from(provider_bytes).is_err()
            || provider_bytes as u64 > device_bytes
        {
            return Err(Error::Limit(
                "provider capacity outside configured physical device envelope".into(),
            ));
        }
        Ok(Self {
            provider_bytes,
            device_bytes,
        })
    }
    pub fn provider_bytes(self) -> usize {
        self.provider_bytes
    }
    pub fn device_bytes(self) -> u64 {
        self.device_bytes
    }
    pub fn headroom_bytes(self) -> u64 {
        self.device_bytes - self.provider_bytes as u64
    }

    /// A current observation may only narrow the requested provider capacity.
    /// A zero usable result is a typed capacity miss, never a lifecycle error.
    pub fn fit_available(self, free_bytes: u64) -> Result<usize> {
        let headroom = self.headroom_bytes();
        let effective = (self.provider_bytes as u64).min(free_bytes.saturating_sub(headroom));
        if effective == 0 {
            return Err(Error::resource(
                "device free memory / VRAM",
                "provider headroom admission",
                headroom.saturating_add(1),
                free_bytes,
            ));
        }
        Ok(effective as usize)
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;
    #[test]
    fn observed_capacity_preserves_headroom_and_typed_refusal() {
        let budget = ProviderMemoryBudget::new(90, 100).unwrap();
        assert_eq!(budget.headroom_bytes(), 10);
        assert_eq!(budget.fit_available(120).unwrap(), 90);
        assert_eq!(budget.fit_available(80).unwrap(), 70);
        assert!(
            matches!(budget.fit_available(10), Err(Error::Resource(r)) if r.requested == 11 && r.available == 10)
        );
        assert!(ProviderMemoryBudget::new(0, 100).is_err());
        assert!(ProviderMemoryBudget::new(101, 100).is_err());
    }
}
