//! Qualification-only views over the original root leases and flight pause.
use super::super::source_native_probe::{SourceTextNativeProbe, SourceTextNativeState};
use super::*;
use crate::resident_leases::final_owner_probe;
use std::sync::Weak;

pub struct ResidentTextFinalReader {
    reader: final_owner_probe::ReaderProbe,
    model: Weak<ResidentTextModel>,
    native: SourceTextNativeProbe,
    before: SourceTextNativeState,
}
#[derive(Debug)]
pub struct ResidentTextReaderObservation {
    pub lease: String,
    pub bytes: usize,
    pub aliases: usize,
    pub model_retained: bool,
    pub release_hits: usize,
    pub native_retirements: usize,
    pub reader_hits_before_native: usize,
    pub before: SourceTextNativeState,
    pub after: SourceTextNativeState,
}
impl ResidentTextFinalReader {
    fn arm(
        model: &Arc<ResidentTextModel>,
        domains: &[Domain],
        ready: &[ResidentLease],
        inner_owners: usize,
        fault: bool,
    ) -> Result<Self> {
        if Arc::strong_count(model) != 1 || Arc::strong_count(&model.inner) != inner_owners {
            return Err(Error::State(
                "text-invocation reader requires sole final wrapper".into(),
            ));
        }
        model.inner.service.ensure_usable()?;
        let lease = domains
            .iter()
            .flat_map(|d| &d.leases)
            .find(|l| l.role() == BufferRole::RecurrentState)
            .ok_or_else(|| Error::State("text-invocation final recurrent reader absent".into()))?;
        if !Arc::ptr_eq(lease.service(), &model.inner.service)
            || lease.source_identity() != model.source.source_identity()
            || !lease.is_initialized()
        {
            return Err(Error::State(
                "text-invocation final reader original binding".into(),
            ));
        }
        let workspace = model
            .inner
            .workspace
            .try_lock()
            .ok_or_else(|| Error::State("text-invocation final workspace busy".into()))?;
        if workspace
            .leases
            .iter()
            .chain(&workspace.state_accesses)
            .chain(&workspace.logit_accesses)
            .any(|(other, _)| other.id() == lease.id())
        {
            return Err(Error::State(
                "text-invocation final reader aliases workspace".into(),
            ));
        }
        let aliases = domains
            .iter()
            .flat_map(|d| &d.leases)
            .chain(
                domains
                    .iter()
                    .flat_map(|d| d.buffers.buffers())
                    .filter_map(DeviceBuffer::final_owner_original_lease),
            )
            .chain(ready)
            .filter(|other| {
                other.id() == lease.id() && Arc::ptr_eq(other.service(), lease.service())
            })
            .count();
        let native = SourceTextNativeProbe::from_original(
            workspace.program.text_execution_native_probe()?,
            &model.inner.service,
        );
        let before = native.observe()?;
        let reader = final_owner_probe::arm(lease, aliases, fault)?;
        Ok(Self {
            reader,
            model: Arc::downgrade(model),
            native,
            before,
        })
    }
    pub fn observe(self) -> Result<ResidentTextReaderObservation> {
        let order = final_owner_probe::result()?;
        Ok(ResidentTextReaderObservation {
            lease: self.reader.lease,
            bytes: self.reader.bytes,
            aliases: self.reader.aliases_inside_wrapper,
            model_retained: self.model.strong_count() != 0,
            release_hits: order.release_hits,
            native_retirements: order.native_retirements,
            reader_hits_before_native: order.hits_before_native_retirement,
            before: self.before,
            after: self.native.observe()?,
        })
    }
}
impl ResidentTextSession {
    pub fn diagnostic_final_reader(&self, fault: bool) -> Result<ResidentTextFinalReader> {
        ResidentTextFinalReader::arm(&self.model, &self.inner.domains, &[], 2, fault)
    }
}
impl ResidentTextSnapshot {
    pub fn diagnostic_final_reader(&self, fault: bool) -> Result<ResidentTextFinalReader> {
        if Arc::strong_count(&self.0) != 1 {
            return Err(Error::State(
                "text-invocation final snapshot has other readers".into(),
            ));
        }
        ResidentTextFinalReader::arm(&self.0.scope.model, &self.0.inner.domains, &[], 1, fault)
    }
}
impl ResidentTextPending {
    pub fn diagnostic_final_reader(&self, fault: bool) -> Result<ResidentTextFinalReader> {
        ResidentTextFinalReader::arm(
            &self.scope.model,
            &self.inner.domains,
            self.inner.ready.outputs(),
            1,
            fault,
        )
    }
}
impl ResidentTextModel {
    /// Execute the original prepared primitive; returned storage retains this model.
    pub async fn diagnostic_trig(
        self: &Arc<Self>,
        angles: &[f32],
    ) -> Result<ResidentTextReadback<Vec<f32>>> {
        self.inner.service.ensure_usable()?;
        let workspace =
            self.inner.workspace.try_lock().ok_or_else(|| {
                Error::State("source trig diagnostic requires idle workspace".into())
            })?;
        let bytes = angles
            .len()
            .checked_mul(20)
            .and_then(|n| n.checked_add(CONTROL_BYTES))
            .ok_or_else(|| Error::Limit("source trig host extent".into()))?;
        let scope = self.reader_scope(bytes)?;
        let data = workspace
            .program
            .text_execution_diagnostic_trig(angles, scope.clone())
            .await?;
        Ok(ResidentTextReadback {
            data,
            _scope: scope,
        })
    }
}
