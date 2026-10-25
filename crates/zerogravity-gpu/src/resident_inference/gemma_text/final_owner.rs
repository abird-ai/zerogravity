//! Final committed Output packet; observers retain only Weak/value IDs.
use super::*;
use crate::backend::final_native::{FinalNativeObservation, FinalNativeProbe};
use crate::resident_leases::{ResidentService, final_owner_probe};
use std::sync::Weak;

#[derive(Debug)]
pub struct GemmaFinalOwnerReceipt {
    pub wrapper: &'static str,
    pub fault: bool,
    pub source: String,
    pub lease: String,
    pub service_generation: u64,
    pub lease_bytes: usize,
    pub aliases_inside_wrapper: usize,
    pub candidate_owners_before_drop: usize,
    pub model_owners_before_drop: usize,
    pub candidate_workspace_target_aliases: usize,
    pub candidate_retired: bool,
    pub release_hits: usize,
    pub native_retirement_attempts: usize,
    pub reader_hits_before_native_retirement: usize,
    pub native_modules_before: usize,
    pub native_code_bytes_before: usize,
    pub native_host_bytes_before: usize,
    pub native_device_bytes_before: usize,
    pub native_admission_retained_after: bool,
    pub native_modules_after: usize,
    pub original_service_retained_after: bool,
    pub quarantined_handles_after: usize,
    /// Root frontier of the actual committed feedback packet.
    pub source_frontier: usize,
}
struct Boundary {
    wrapper: &'static str,
    fault: bool,
    source: String,
    reader: final_owner_probe::ReaderProbe,
    candidate: Weak<Gemma4Bf16GpuCandidate>,
    service: Weak<ResidentService>,
    native: FinalNativeProbe,
    inventory: FinalNativeObservation,
    candidate_owners: usize,
    model_owners: usize,
    frontier: usize,
}
impl Boundary {
    fn before(
        wrapper: &'static str,
        loaded: &Arc<Gemma4Bf16GpuCandidate>,
        feedback: Option<&ResidentToken>,
        binding: RootBinding,
        expected_model_owners: usize,
        fault: bool,
    ) -> Result<Self> {
        let candidate_owners = Arc::strong_count(loaded);
        let model_owners = Arc::strong_count(&loaded.model);
        if candidate_owners != 1 || model_owners != expected_model_owners {
            return Err(Error::State(
                "Gemma final wrapper is not the sole candidate/model population".into(),
            ));
        }
        loaded.model.service.ensure_usable()?;
        let token =
            feedback.ok_or_else(|| Error::State("Gemma committed final feedback absent".into()))?;
        let selected = &token.lease;
        if token.producer != Some(binding)
            || binding.frontier() == 0
            || selected.role() != BufferRole::Output
            || selected.bytes() != 12
            || !selected.is_initialized()
            || token.id >= 262144
            || !f32::from_bits(token.max_bits).is_finite()
            || !Arc::ptr_eq(selected.service(), &loaded.model.service)
            || selected.source_identity() != loaded.source.source_identity()
        {
            return Err(Error::State(
                "Gemma final committed Output/source/root packet differs".into(),
            ));
        }
        let native = {
            let workspace = loaded
                .model
                .workspace
                .try_lock()
                .ok_or_else(|| Error::State("Gemma final workspace still busy".into()))?;
            if workspace
                .leases
                .iter()
                .chain(&workspace.state_accesses)
                .chain(&workspace.logit_accesses)
                .any(|(lease, _)| lease.id() == selected.id())
            {
                return Err(Error::State(
                    "Gemma feedback still aliases candidate workspace".into(),
                ));
            }
            workspace.program.gemma_final_owner_native_probe()?
        };
        let inventory = native.observe()?;
        if !inventory.retained || inventory.modules == 0 {
            return Err(Error::State(
                "Gemma actual prepared native admission absent".into(),
            ));
        }
        // The final wrapper has exactly one feedback field. The original probe
        // requires exact total Arc count1, rejecting ANY surviving Ready/token.
        let reader = final_owner_probe::arm(selected, 1, fault)?;
        let frontier = usize::try_from(binding.frontier())
            .map_err(|_| Error::Limit("Gemma final root frontier overflow".into()))?;
        Ok(Self {
            wrapper,
            fault,
            source: format!("{:?}", selected.source_identity()),
            reader,
            candidate: Arc::downgrade(loaded),
            service: Arc::downgrade(&loaded.model.service),
            native,
            inventory,
            candidate_owners,
            model_owners,
            frontier,
        })
    }
    fn after(self) -> Result<GemmaFinalOwnerReceipt> {
        if self.candidate.strong_count() != 0 {
            return Err(Error::State(
                "caller candidate survived final wrapper drop".into(),
            ));
        }
        let order = final_owner_probe::result()?;
        let native = self.native.observe()?;
        // Upgrade only AFTER actual wrapper destruction; no observer keepalive
        // may delay its real final service/root/native retirement.
        let (service_retained, quarantined) = match self.service.upgrade() {
            Some(service) => (
                true,
                crate::physical_quarantine::quarantined_handles_for(service.client())?,
            ),
            None => (false, 0),
        };
        if self.fault {
            if !service_retained
                || quarantined == 0
                || !native.retained
                || native.modules != self.inventory.modules
                || native.code_bytes != self.inventory.code_bytes
                || native.host_bytes != self.inventory.host_bytes
                || native.device_bytes != self.inventory.device_bytes
            {
                return Err(Error::State(
                    "late original reader fault lost actual source/native/handles".into(),
                ));
            }
        } else if service_retained || quarantined != 0 || native.retained {
            return Err(Error::State(
                "known final root/native service did not retire".into(),
            ));
        }
        Ok(GemmaFinalOwnerReceipt {
            wrapper: self.wrapper,
            fault: self.fault,
            source: self.source,
            lease: self.reader.lease,
            service_generation: self.reader.generation,
            lease_bytes: self.reader.bytes,
            aliases_inside_wrapper: self.reader.aliases_inside_wrapper,
            candidate_owners_before_drop: self.candidate_owners,
            model_owners_before_drop: self.model_owners,
            candidate_workspace_target_aliases: 0,
            candidate_retired: true,
            release_hits: order.release_hits,
            native_retirement_attempts: order.native_retirements,
            reader_hits_before_native_retirement: order.hits_before_native_retirement,
            native_modules_before: self.inventory.modules,
            native_code_bytes_before: self.inventory.code_bytes,
            native_host_bytes_before: self.inventory.host_bytes,
            native_device_bytes_before: self.inventory.device_bytes,
            native_admission_retained_after: native.retained,
            native_modules_after: native.modules,
            original_service_retained_after: service_retained,
            quarantined_handles_after: quarantined,
            source_frontier: self.frontier,
        })
    }
}
impl Gemma4Bf16GpuSession {
    pub fn qualify_final_feedback_drop(self, fault: bool) -> Result<GemmaFinalOwnerReceipt> {
        let boundary = Boundary::before(
            "Session",
            &self.loaded,
            self.feedback.as_ref(),
            self.binding(),
            2,
            fault,
        )?;
        drop(self);
        boundary.after()
    }
}
impl Gemma4Bf16GpuSnapshot {
    pub fn qualify_final_feedback_drop(self, fault: bool) -> Result<GemmaFinalOwnerReceipt> {
        let boundary = Boundary::before(
            "Snapshot",
            &self.loaded,
            self.feedback.as_ref(),
            self.binding(),
            1,
            fault,
        )?;
        drop(self);
        boundary.after()
    }
}
