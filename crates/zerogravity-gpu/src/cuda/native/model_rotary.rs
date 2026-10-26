//! Only the genuine completed source constructor may supply rotary bytes.
use super::source_rotary::SourceRotaryPrefix;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use std::sync::Arc;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::SourceCall;
use zerogravity_core::device_events::BufferRole;
use zerogravity_core::{Error, Result};

pub(crate) struct ModelRotary {
    prefix: Arc<SourceRotaryPrefix>,
    lease: ResidentLease,
    source: VerifiedTensorSource,
    service: Arc<ResidentService>,
    generation: u64,
    text: bool,
}
impl ModelRotary {
    /// Model factory authenticates the selected source/recipe first.
    /// Constructor runs/retirement finished BEFORE full-model W allocation.
    /// The source prefix prepays its actual bytes/H2D copy/readback admission.
    pub(crate) fn prepare(
        source: VerifiedTensorSource,
        service: Arc<ResidentService>,
        prefix: Arc<SourceRotaryPrefix>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        prefix.require_source(&source)?;
        init.require_service(&service)?;
        if source.source_identity() != service.source_identity() || prefix.bytes().len() != 8576 {
            return Err(Error::State(
                "full24 original computed rotary source/service/defined extent".into(),
            ));
        }
        // Retain genuine producer/admission BEFORE the original H2D copy.
        init.retain_completion_owner(prefix.clone())?;
        let lease = init.upload_bytes(prefix.bytes(), BufferRole::ImmutableWeights)?;
        let generation = service.generation();
        let prepared = Arc::new(Self {
            prefix,
            lease,
            source,
            service,
            generation,
            text: false,
        });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }
    pub(crate) fn prepare_text(
        source: VerifiedTensorSource,
        service: Arc<ResidentService>,
        prefix: Arc<SourceRotaryPrefix>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        prefix.require_source(&source)?;
        prefix.require_span(0, 79)?;
        init.require_service(&service)?;
        if !prefix.is_text()
            || prefix.bytes().len() != 79 * 128
            || source.source_identity() != service.source_identity()
        {
            return Err(Error::State("text genuine source prefix79/service".into()));
        }
        init.retain_completion_owner(prefix.clone())?;
        let lease = init.upload_bytes(prefix.bytes(), BufferRole::ImmutableWeights)?;
        let result = Arc::new(Self {
            prefix,
            lease,
            source,
            generation: service.generation(),
            service,
            text: true,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn source_bytes(&self) -> &[u8] {
        self.prefix.bytes()
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn require_call(
        &self,
        source: &VerifiedTensorSource,
        call: &SourceCall,
        service: &Arc<ResidentService>,
    ) -> Result<()> {
        self.prefix.require_source(source)?;
        if call.source() != source.source_identity() {
            return Err(Error::State(
                "computed rotary actual source call differs".into(),
            ));
        }
        self.prefix
            .require_span(call.rows().base(), call.rows().len())?;
        if !source.same_issuer(&self.source)
            || !Arc::ptr_eq(service, &self.service)
            || service.generation() != self.generation
            || !self.lease.is_initialized()
            || self.lease.source_identity() != call.source()
            || self.lease.bytes() != (if self.text { 79 * 128 } else { 8576 })
        {
            return Err(Error::State(
                "full24 computed rotary original source/service/generation differs".into(),
            ));
        }
        service.ensure_usable()
    }
}
