//! Existing five actual source graphs, not a second plan or execution permit.
use super::*;
pub struct SourceModelScopes {
    source: VerifiedTensorSource,
    records: [Arc<SourceBindings>; 5],
}
impl SourceModelScopes {
    pub fn new(source: VerifiedTensorSource, records: [Arc<SourceBindings>; 5]) -> Result<Self> {
        Self::require(&source, &records)?;
        Ok(Self { source, records })
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub fn records(&self) -> &[Arc<SourceBindings>; 5] {
        &self.records
    }
    pub(crate) fn into_parts(self) -> (VerifiedTensorSource, [Arc<SourceBindings>; 5]) {
        (self.source, self.records)
    }
    pub(super) fn require(
        source: &VerifiedTensorSource,
        records: &[Arc<SourceBindings>; 5],
    ) -> Result<()> {
        // The original bound computes the same physical source/state/ABI extent.
        provider_bound(source, records)?;
        for (index, record) in records.iter().enumerate() {
            if crate::resident_program::natural_binding::slot(record.call())? != index
                || !source.same_issuer(record.source())
                || record.call().rows().capacity() != 512
                || record.call().rows().vocabulary() != 248320
            {
                return Err(Error::State(
                    "source five-scope actual order/issuer/extent differs".into(),
                ));
            }
        }
        Ok(())
    }
}
