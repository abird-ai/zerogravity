//! Fixed head programs on the same original image/set/source/leases.
use super::super::projection_cubin::resident::{NativeRetentionWitness, PreparedImages};
use super::*;
use crate::resident_leases::ResidentInitialization;
use cubecl::cuda::{Bf16VocabularyHead, OwnedCudaKernel, PinnedBf16Cubin, VocabularyHeadSelection};

#[derive(Clone)]
enum Producer {
    Captured,
    FinalNorm(Arc<FusedRmsPrepared>),
}
#[derive(Clone)]
pub(super) struct HeadInputView {
    lease: ResidentLease,
    offset: usize,
    case: VocabularyHeadCase,
    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
    producer: Producer,
}
impl HeadInputView {
    pub(super) fn captured(
        source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
        case: VocabularyHeadCase,
        lease: ResidentLease,
    ) -> Result<Self> {
        let view = Self {
            lease,
            offset: 0,
            case,
            source,
            producer: Producer::Captured,
        };
        view.validate()?;
        Ok(view)
    }
    pub(super) fn computed(
        source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
        case: VocabularyHeadCase,
        fused: Arc<FusedRmsPrepared>,
    ) -> Result<Self> {
        let view = Self {
            lease: fused.normalized().clone(),
            offset: case.selected_row() * 2048,
            case,
            source,
            producer: Producer::FinalNorm(fused),
        };
        view.validate()?;
        Ok(view)
    }
    fn validate(&self) -> Result<()> {
        let expected = match &self.producer {
            Producer::Captured => 2048,
            Producer::FinalNorm(fused) => {
                if fused.normalized().id() != self.lease.id()
                    || fused.normalized().allocation_generation()
                        != self.lease.allocation_generation()
                    || self.offset != self.case.selected_row() * 2048
                {
                    return Err(Error::State(
                        "head selection must reference actual same-phase normalized producer"
                            .into(),
                    ));
                }
                self.case.phase_rows() * 2048
            }
        };
        if self.lease.bytes() != expected
            || self.lease.source_identity() != self.source.source_identity()
            || self
                .offset
                .checked_add(2048)
                .is_none_or(|end| end > self.lease.bytes())
        {
            return Err(Error::State(
                "head selected original source/phase/view extent".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(super) fn readback(&self) -> Result<ResidentReadback> {
        self.validate()?;
        self.lease.readback(self.offset, 2048)
    }
    fn selection(&self) -> VocabularyHeadSelection {
        match &self.producer {
            Producer::Captured => VocabularyHeadSelection::WholeSelected,
            Producer::FinalNorm(_) => match self.case {
                VocabularyHeadCase::InitialFirst64 => VocabularyHeadSelection::LastOfSixtyFour,
                VocabularyHeadCase::Continued65And66 => VocabularyHeadSelection::LastOfTwo,
                _ => VocabularyHeadSelection::WholeSelected,
            },
        }
    }
}
pub(super) struct HeadPrograms {
    images: Arc<PreparedImages>,
    kernel: Arc<OwnedCudaKernel>,
}
impl HeadPrograms {
    pub(super) fn prepare(
        images: Arc<PreparedImages>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        init.retain_completion_owner(images.clone())?;
        let image = images.image(ProjectionImage::Bf16Gemv2TSm86)?;
        let expected = PinnedBf16Cubin::Bf16Gemv2TSm86.derivation();
        if image.derived_from() != Some(PinnedBf16Cubin::Bf16Gemv2TSm86)
            || image.captured_from().is_some()
            || image.byte_len() != expected.image_bytes
            || image.sha256() != expected.image_sha256
            || image.binary_sm() != 86
        {
            return Err(Error::State(
                "head exact original derived native image".into(),
            ));
        }
        // SAFETY: original real W initialization already retains source/image/
        // native admission; the new set accepts only this entire fixed ABI.
        let kernel = unsafe { images.set().prepare(image, Bf16VocabularyHead::manifest()) }
            .map_err(native_error)?;
        init.retain_completion_owner(kernel.clone())?;
        let programs = Arc::new(Self { images, kernel });
        init.retain_completion_owner(programs.clone())?;
        Ok(programs)
    }
    pub(super) fn enqueue(
        &self,
        weight: &Arc<TiedWeight>,
        input: &HeadInputView,
        output: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        input.validate()?;
        let service = weight.lease().service();
        service.ensure_usable()?;
        self.images.require_service(service)?;
        if !weight.source().same_issuer(&input.source)
            || !Arc::ptr_eq(service, input.lease.service())
            || !Arc::ptr_eq(service, output.service())
            || weight.lease().bytes() != TIED_BYTES
            || !weight.lease().is_initialized()
            || weight.lease().role() != BufferRole::ImmutableWeights
            || output.bytes() != 496_640
            || output.source_identity() != weight.source().source_identity()
            || input.lease.id() == output.id()
            || weight.lease().id() == input.lease.id()
            || weight.lease().id() == output.id()
            || matches!(&input.producer, Producer::Captured) && !input.lease.is_initialized()
        {
            return Err(Error::State(
                "head original complete W/X/Y source/service/geometry".into(),
            ));
        }
        // SAFETY: caller's original armed flight retains W/source/program/X/Y;
        // computed norm, if present, was enqueued earlier in that SAME flight.
        unsafe {
            self.kernel.enqueue_vocabulary_head_selected(
                weight.lease().handle(),
                input.lease.handle(),
                output.handle(),
                input.selection(),
            )
        }
        .map_err(native_error)?;
        Ok(1)
    }
    pub(super) fn images(&self) -> &Arc<PreparedImages> {
        &self.images
    }
    pub(super) fn retention_witness(&self) -> Result<NativeRetentionWitness> {
        self.images
            .retention_witness(std::slice::from_ref(&self.kernel))
    }
    pub(super) unsafe fn retire_kernels_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        // SAFETY: enclosing real original initialized-W read-only guard.
        unsafe { self.kernel.retire_after_physical_completion() }
    }
    pub(super) unsafe fn retire(&self) -> std::result::Result<(), cubecl::server::ServerError> {
        // SAFETY: caller's real initialized-W read-only flight owns the full
        // graph and proved completion of every original head launch.
        unsafe { self.retire_kernels_after_physical_completion() }?;
        unsafe { self.images.set().retire_after_physical_completion() }
    }
}
