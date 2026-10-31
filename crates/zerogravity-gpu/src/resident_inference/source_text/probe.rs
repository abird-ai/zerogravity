//! Hardware-only hooks over original owners; case orchestration stays in test support.
use super::*;
use crate::resident_leases::final_owner_probe;
use std::sync::{Weak, atomic::Ordering};

pub struct SourceTextFinalReader {
    reader: final_owner_probe::ReaderProbe,
    candidate: Weak<SourceTextCandidate>,
    native: SourceTextNativeProbe,
    before: SourceTextNativeState,
}
#[derive(Debug)]
pub struct SourceTextReaderObservation {
    pub lease: String,
    pub bytes: usize,
    pub aliases: usize,
    pub candidate_retained: bool,
    pub release_hits: usize,
    pub native_retirements: usize,
    pub reader_hits_before_native: usize,
    pub before: SourceTextNativeState,
    pub after: SourceTextNativeState,
}
impl SourceTextFinalReader {
    fn arm(
        loaded: &Arc<SourceTextCandidate>,
        domains: &[Domain],
        ready: &[ResidentLease],
        model_owners: usize,
        fault: bool,
    ) -> Result<Self> {
        if Arc::strong_count(loaded) != 1 || Arc::strong_count(&loaded.model) != model_owners {
            return Err(Error::State(
                "source reader requires sole final wrapper".into(),
            ));
        }
        loaded.model.service.ensure_usable()?;
        let selected = domains
            .iter()
            .flat_map(|d| &d.leases)
            .find(|l| l.role() == BufferRole::RecurrentState)
            .ok_or_else(|| Error::State("source final recurrent reader absent".into()))?;
        if !Arc::ptr_eq(selected.service(), &loaded.model.service)
            || selected.source_identity() != loaded.source.source_identity()
            || !selected.is_initialized()
        {
            return Err(Error::State(
                "source final reader original binding/initialization".into(),
            ));
        }
        {
            let workspace = loaded
                .model
                .workspace
                .try_lock()
                .ok_or_else(|| Error::State("source final workspace busy".into()))?;
            if workspace
                .leases
                .iter()
                .chain(&workspace.state_accesses)
                .chain(&workspace.logit_accesses)
                .any(|(l, _)| l.id() == selected.id())
            {
                return Err(Error::State(
                    "source final reader aliases shared workspace".into(),
                ));
            }
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
            .filter(|l| l.id() == selected.id() && Arc::ptr_eq(l.service(), selected.service()))
            .count();
        // Pool backings are distinct owners, not aliases of this logical lease.
        let native = loaded.native_probe();
        let before = native.observe()?;
        let reader = final_owner_probe::arm(selected, aliases, fault)?;
        Ok(Self {
            reader,
            candidate: Arc::downgrade(loaded),
            native,
            before,
        })
    }
    pub fn observe(self) -> Result<SourceTextReaderObservation> {
        let order = final_owner_probe::result()?;
        Ok(SourceTextReaderObservation {
            lease: self.reader.lease,
            bytes: self.reader.bytes,
            aliases: self.reader.aliases_inside_wrapper,
            candidate_retained: self.candidate.strong_count() != 0,
            release_hits: order.release_hits,
            native_retirements: order.native_retirements,
            reader_hits_before_native: order.hits_before_native_retirement,
            before: self.before,
            after: self.native.observe()?,
        })
    }
}
impl SourceTextSession {
    pub fn diagnostic_final_reader(&self, fault: bool) -> Result<SourceTextFinalReader> {
        SourceTextFinalReader::arm(&self.loaded, &self.inner.domains, &[], 2, fault)
    }
}
impl SourceTextSnapshot {
    pub fn diagnostic_final_reader(&self, fault: bool) -> Result<SourceTextFinalReader> {
        SourceTextFinalReader::arm(&self.loaded, &self.inner.domains, &[], 1, fault)
    }
}
impl SourceTextPending {
    pub fn diagnostic_final_reader(&self, fault: bool) -> Result<SourceTextFinalReader> {
        SourceTextFinalReader::arm(
            &self.loaded,
            &self.inner.domains,
            self.inner.ready.outputs(),
            1,
            fault,
        )
    }
}
impl SourceTextCandidate {
    pub fn diagnostic_pause_next_launch(&self) -> Result<()> {
        let _idle = self.proposals.try_acquire().ok_or_else(|| {
            Error::State("source launch probe requires idle proposal slot".into())
        })?;
        self.model.service.ensure_usable()?;
        if self.model.qualification_launch_seen() != 0 {
            return Err(Error::State("source launch probe already reached".into()));
        }
        self.model
            .abort_after_launch
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("source launch probe already armed".into()))?;
        Ok(())
    }
    pub fn diagnostic_launch_handles(&self) -> usize {
        self.model.qualification_launch_seen()
    }
}
