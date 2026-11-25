//! Authenticated complete source parameter -> immutable resident bytes.
//!
//! The caller supplies the exact SourceParameter descriptor and explicit
//! storage policy chosen by numerical lowering.
pub(crate) use super::source_parameters::ParameterStorage as ForwardParameterStorage;
use super::{ResidentInitialization, ResidentLease, ResidentService};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{VerifiedParameter, VerifiedTensorSource};
use zerogravity_core::{
    Result,
    accelerator::{SourceIdentity, program::SourceParameter},
    encoding::StoredTensorEncoding,
};

pub(crate) struct ForwardParameter {
    binding: VerifiedParameter,
    descriptor: SourceParameter,
    lease: ResidentLease,
    storage: ForwardParameterStorage,
    kind: StoredTensorEncoding,
}

impl ForwardParameter {
    pub(crate) fn source_identity(&self) -> SourceIdentity {
        self.binding.source_identity()
    }
    pub(crate) fn descriptor(&self) -> &SourceParameter {
        &self.descriptor
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn storage(&self) -> ForwardParameterStorage {
        self.storage
    }
    pub(crate) fn kind(&self) -> StoredTensorEncoding {
        self.kind
    }
    pub(crate) fn elements(&self) -> usize {
        self.binding.elements()
    }
    pub(crate) fn require_descriptor(&self, descriptor: &SourceParameter) -> Result<()> {
        self.binding.validate_source_parameter(descriptor)
    }
}

impl ResidentInitialization<'_> {
    pub(crate) fn upload_forward_parameter(
        &mut self,
        service: &Arc<ResidentService>,
        source: VerifiedTensorSource,
        descriptor: &SourceParameter,
        storage: ForwardParameterStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ForwardParameter>> {
        self.require_service(service)?;
        let bytes = storage.resident_bytes(descriptor)?;
        let kind = storage.kind(descriptor.kind)?;
        let (lease, binding) =
            self.upload_parameter(&source, descriptor, storage, bytes, cancelled)?;
        Ok(Arc::new(ForwardParameter {
            binding,
            descriptor: descriptor.clone(),
            lease,
            storage,
            kind,
        }))
    }
}
