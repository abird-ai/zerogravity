//! Only the genuine completed source constructor may supply rotary bytes.
use super::source_rotary::SourceRotaryPrefix;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use std::sync::Arc;
use zerogravity_core::accelerator::program::Vllm08Call;
use zerogravity_core::device_events::BufferRole;
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

pub(crate) struct ModelRotary {
    prefix: Arc<SourceRotaryPrefix>,
    lease: ResidentLease,
    source: Arc<OriginalWindowModelSource>,
    service: Arc<ResidentService>,
    generation: u64,
    short_hf: bool,
}
impl ModelRotary {
    /// Constructor runs/retirement finished BEFORE full-model W allocation.
    /// The source prefix prepays its actual bytes/H2D copy/readback admission.
    pub(crate) fn prepare(
        source: Arc<OriginalWindowModelSource>,
        service: Arc<ResidentService>,
        prefix: Arc<SourceRotaryPrefix>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        source.require_vllm08_full_source()?;
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
            short_hf: false,
        });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }
    pub(crate) fn prepare_short_hf(
        source: Arc<OriginalWindowModelSource>,
        service: Arc<ResidentService>,
        prefix: Arc<SourceRotaryPrefix>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        source.bind_full_model_source()?;
        prefix.require_source(&source)?;
        prefix.require_span(0, 79)?;
        init.require_service(&service)?;
        if !prefix.is_short_hf()
            || prefix.bytes().len() != 79 * 128
            || source.source_identity() != service.source_identity()
        {
            return Err(Error::State(
                "short-HF genuine source prefix79/service".into(),
            ));
        }
        init.retain_completion_owner(prefix.clone())?;
        let lease = init.upload_bytes(prefix.bytes(), BufferRole::ImmutableWeights)?;
        let result = Arc::new(Self {
            prefix,
            lease,
            source,
            generation: service.generation(),
            service,
            short_hf: true,
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
        source: &Arc<OriginalWindowModelSource>,
        call: &Vllm08Call,
        service: &Arc<ResidentService>,
    ) -> Result<()> {
        self.prefix.require_source(source)?;
        self.prefix
            .require_span(call.request().teachers().base(), call.rows())?;
        if !Arc::ptr_eq(source, &self.source)
            || !Arc::ptr_eq(service, &self.service)
            || service.generation() != self.generation
            || !self.lease.is_initialized()
            || self.lease.source_identity() != call.source()
            || self.lease.bytes() != (if self.short_hf { 79 * 128 } else { 8576 })
            || self.short_hf != call.request().teachers().short_hf_span().is_some()
        {
            return Err(Error::State(
                "full24 computed rotary original source/service/generation differs".into(),
            ));
        }
        service.ensure_usable()
    }
}
