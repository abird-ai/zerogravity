//! Actual opaque compiler/retained native host admission; no GPU fit permit.
use super::*;
impl VerifiedTensorSource {
    /// Refuse isolated test arbiters and alternate host keys even if their
    /// declared byte limit happens to match a backend's real host policy.
    pub fn require_process_host_workspace(&self, limit: u64) -> Result<()> {
        if !self.0.admission.arbiter.is_process_shared()
            || self.0.admission.resource
                != zerogravity_core::resource_arbiter::PhysicalDeviceKey::new("host", "system-ram")?
            || self.workspace_limit_bytes()? != limit
        {
            return Err(Error::State(
                "backend workspace requires original process host domain".into(),
            ));
        }
        Ok(())
    }

    /// Backend policy fixes the allowance before compilation. Same source,
    /// canonical host domain and global arbiter as original source windows.
    /// The caller retains this through actual destruction or uncertainty.
    pub fn reserve_backend_workspace(&self, bytes: usize) -> Result<WindowWorkspace> {
        if bytes == 0 || bytes > 8 * 1024 * 1024 * 1024 {
            return Err(Error::Limit(
                "verified source backend workspace bound".into(),
            ));
        }
        self.admit_workspace(
            bytes,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: bytes as u64,
            }],
        )
    }
}

impl VerifiedTensorSource {
    /// Validate a retained detached lease against this original HOST domain.
    pub fn detached_host_workspace_bytes(
        &self,
        lease: &GlobalResourceLease,
        declared_host_limit: u64,
    ) -> Result<u64> {
        if !self.0.admission.arbiter.is_process_shared()
            || self.0.admission.resource.backend != "host"
            || self.workspace_limit_bytes()? != declared_host_limit
        {
            return Err(Error::State(
                "detached workspace HOST policy/domain differs".into(),
            ));
        }
        lease.bytes_in_domain(&self.0.admission.arbiter, &self.0.admission.resource)
    }
    /// Detached reservation on this source's original process-shared HOST key.
    /// No source/window/Client backedge or raw arbiter/admission is returned.
    pub fn reserve_detached_host_workspace(
        &self,
        declared_host_limit: u64,
        owner: zerogravity_core::resource_arbiter::ResourceOwner,
        bytes: u64,
    ) -> Result<GlobalResourceLease> {
        if bytes == 0
            || bytes > declared_host_limit
            || !self.0.admission.arbiter.is_process_shared()
            || self.0.admission.resource.backend != "host"
            || self.workspace_limit_bytes()? != declared_host_limit
        {
            return Err(Error::State(
                "detached workspace requires original process HOST admission and declared limit"
                    .into(),
            ));
        }
        self.0.admission.arbiter.admit(
            &self.0.admission.resource,
            owner,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes,
            }],
        )
    }
}
