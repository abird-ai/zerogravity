//! Authenticated source data plus original cold-owner retention; no execution permit.
use crate::physical_quarantine::CompletionOwner;
use std::any::Any;
use std::sync::{Arc, Weak};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::SourceTensor;
use zerogravity_core::{Error, Result};

pub struct RetainedSource {
    source: VerifiedTensorSource,
    owner: CompletionOwner,
    _metadata_owner: CompletionOwner,
}
impl RetainedSource {
    /// The family/application authenticates its eligibility before this data join.
    /// Erasing these Arc types preserves the actual original owner allocations.
    pub fn new<T: Any + Send + Sync, M: Any + Send + Sync>(
        source: VerifiedTensorSource,
        owner: Arc<T>,
        metadata_owner: Arc<M>,
    ) -> Arc<Self> {
        Arc::new(Self {
            source,
            owner,
            _metadata_owner: metadata_owner,
        })
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub fn source_identity(&self) -> zerogravity_core::accelerator::SourceIdentity {
        self.source.source_identity()
    }
    pub(crate) fn owner(&self) -> &CompletionOwner {
        &self.owner
    }
    pub(crate) fn weak_owner(&self) -> Weak<dyn Any + Send + Sync> {
        Arc::downgrade(&self.owner)
    }
}

/// Actual authenticated tied matrix and original cold-source retention.
pub struct TiedWeightSource {
    output: SourceTensor,
    // Descriptor payloads drop before their original cold metadata owner.
    retained: Arc<RetainedSource>,
}
impl TiedWeightSource {
    pub fn bind(retained: Arc<RetainedSource>, output: SourceTensor) -> Result<Arc<Self>> {
        let actual = retained.source().matrix(&output.name, output.axes)?;
        if output.source != retained.source_identity()
            || actual.source_identity() != output.source
            || actual.tensor().kind != output.kind
            || actual.rows() != output.rows
            || actual.columns() != output.columns
        {
            return Err(Error::State(
                "tied qualification actual source descriptor differs".into(),
            ));
        }
        Ok(Arc::new(Self { retained, output }))
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        self.retained.source()
    }
    pub fn source_identity(&self) -> zerogravity_core::accelerator::SourceIdentity {
        self.retained.source_identity()
    }
    pub(crate) fn output(&self) -> &SourceTensor {
        &self.output
    }
    pub(crate) fn owner(&self) -> &CompletionOwner {
        self.retained.owner()
    }
}
