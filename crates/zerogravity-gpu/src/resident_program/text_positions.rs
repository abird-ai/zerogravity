//! Source-produced I64 control storage; no float PlanValue or new math engine.
use crate::resident_inference::text_positions::{BoundTextPositions, POSITION_BYTES};
use crate::resident_inference::{ResidentSession, native_fragment::PrefixPhaseEnqueue};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentReadback};
use cubecl::bytes::Bytes;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{Error, Result, device_events::BufferRole};

struct PositionPayload {
    bytes: Vec<u8>,
    binding: Arc<BoundTextPositions>,
    // Fixed payload + pending upload copy + one completion readback bound.
    _host: WindowWorkspace,
}
pub(crate) struct TextPositionsPrepared {
    lease: ResidentLease,
    payload: Arc<PositionPayload>,
}
impl TextPositionsPrepared {
    /// Caller owns SAME preparation fair slot/init and preallocates before modules.
    pub(crate) fn prepare(
        binding: Arc<BoundTextPositions>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        if cancelled.is_some_and(|v| v.load(Ordering::Acquire)) {
            return Err(Error::State("source positions cancelled".into()));
        }
        init.require_service(binding.service())?;
        let host = binding
            .source()
            .source()
            .reserve_workspace(3 * POSITION_BYTES)?;
        let bytes = binding.bytes()?.to_vec();
        if bytes.capacity() > POSITION_BYTES {
            return Err(Error::Limit("source position payload capacity".into()));
        }
        let payload = Arc::new(PositionPayload {
            bytes,
            binding,
            _host: host,
        });
        // Retain source/admissions BEFORE actual original H2D.
        init.retain_completion_owner(payload.clone())?;
        let lease = init.upload_bytes(&payload.bytes, BufferRole::Activation)?;
        let result = Arc::new(Self { lease, payload });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    /// Preparation only; actual lowerer must call require_enqueue before QK.
    pub(crate) fn prepared_lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn require_enqueue(
        &self,
        session: &ResidentSession,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        self.payload
            .binding
            .require_enqueue(session, permit, cancelled)?;
        if !self.lease.is_initialized()
            || !Arc::ptr_eq(self.lease.service(), self.payload.binding.service())
            || self.lease.bytes() != POSITION_BYTES
            || self.lease.source_identity() != self.payload.binding.invocation().teachers().source()
        {
            return Err(Error::State(
                "original position upload not completed or lease changed".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn readback(&self) -> Result<ResidentReadback> {
        self.lease.readback(0, POSITION_BYTES)
    }
    /// Internal complete control proof only; source reference compares ACTIVE
    /// values, not the defined private padding or original unused GPU lanes.
    pub(crate) fn validate_readback(&self, raw: &[Bytes]) -> Result<()> {
        if raw.len() != 1 || &raw[0][..] != self.payload.bytes.as_slice() {
            return Err(Error::State(
                "source-produced original I64 position transport changed".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require_invocation(
        &self,
        invocation: &zerogravity_core::accelerator::program::VllmBf16Invocation,
    ) -> Result<()> {
        invocation.validate()?;
        if self.payload.binding.invocation() != invocation {
            return Err(Error::State(
                "source positions differ from actual same-plan teacher issuer".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
mod natural;

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod canonical_natural;

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod short_hf;
