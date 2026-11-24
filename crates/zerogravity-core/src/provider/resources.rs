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
    /// Exclusive writable workspace supplied by the fabric.
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
