//! Finite source-owned learned uploads on the original initialization/actor.
//! Original tied-table transport and the old32MiB packer remain unchanged.
use super::*;
use cubecl::cuda::Vllm08WeightLayout;
use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedMatrix, WindowWorkspace};
use zerogravity_core::accelerator::program::{Vllm08Kind, Vllm08Operator, Vllm08Projection};
use zerogravity_core::quant::QuantKind;
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

mod coverage;
mod qwen2b_mlp;
pub(crate) use qwen2b_mlp::{
    Qwen2bMlpComputeReadCut, Qwen2bMlpSourceVerification, Qwen2bMlpWeightCohort, Qwen2bMlpWeights,
};
pub use qwen2b_mlp::{Qwen2bMlpWeightMode, Qwen2bMlpWeightOutcome, qualify_qwen2b_mlp_weights};
mod verify;
use coverage::Coverage;
const METADATA_BYTES: usize = 64 * 1024;

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "original full-model learned upload cancelled".into(),
        ));
    }
    Ok(())
}
fn require_original08_upload_source(
    source: &Arc<OriginalWindowModelSource>,
    descriptor: &Vllm08Operator,
) -> Result<()> {
    source.require_vllm08_full_source()?;
    if descriptor.kind() == Vllm08Kind::Projection(Vllm08Projection::Head) {
        return Err(Error::Unsupported(
            "old .8B learned wrapper excludes tied head".into(),
        ));
    }
    if descriptor.call().model_profile()
        != zerogravity_core::accelerator::program::FullModelProfile::Qwen08
    {
        return Err(Error::Unsupported(
            "original .8B upload rejects2B layouts".into(),
        ));
    }
    Ok(())
}
fn layout(operator: &Vllm08Operator) -> Result<Vllm08WeightLayout> {
    operator.validate()?;
    if operator.call().model_profile()
        == zerogravity_core::accelerator::program::FullModelProfile::Qwen2B
    {
        return match operator.kind() {
            Vllm08Kind::Projection(Vllm08Projection::Head) => Ok(Vllm08WeightLayout::Vocabulary2b),
            Vllm08Kind::Projection(Vllm08Projection::Qkvz) => Ok(Vllm08WeightLayout::Qwen2bGdnQkvz),
            Vllm08Kind::Projection(Vllm08Projection::Ba) => Ok(Vllm08WeightLayout::Qwen2bGdnBa),
            Vllm08Kind::Projection(Vllm08Projection::GdnOut) => {
                Ok(Vllm08WeightLayout::Qwen2bGdnOutput)
            }
            Vllm08Kind::Projection(Vllm08Projection::AttentionQkv) => {
                Ok(Vllm08WeightLayout::Qwen2bAttentionQkv)
            }
            Vllm08Kind::Projection(Vllm08Projection::AttentionOut) => {
                Ok(Vllm08WeightLayout::Qwen2bAttentionOutput)
            }
            Vllm08Kind::Projection(Vllm08Projection::GateUp) => {
                Ok(Vllm08WeightLayout::Qwen2bMlpGateUp)
            }
            Vllm08Kind::Projection(Vllm08Projection::Down) => Ok(Vllm08WeightLayout::Qwen2bMlpDown),
            _ => Err(Error::Unsupported(
                "finite2B learned source layouts only".into(),
            )),
        };
    }
    match operator.kind() {
        Vllm08Kind::Projection(kind) => Ok(match kind {
            Vllm08Projection::Qkvz => Vllm08WeightLayout::GdnQkvz,
            Vllm08Projection::Ba => Vllm08WeightLayout::GdnBa,
            Vllm08Projection::GdnOut => Vllm08WeightLayout::GdnOutput,
            Vllm08Projection::AttentionQkv => Vllm08WeightLayout::AttentionQkv,
            Vllm08Projection::AttentionOut => Vllm08WeightLayout::AttentionOutput,
            Vllm08Projection::GateUp => Vllm08WeightLayout::MlpGateUp,
            Vllm08Projection::Down => Vllm08WeightLayout::MlpDown,
            Vllm08Projection::Head => Vllm08WeightLayout::Vocabulary08,
        }),
        _ => Err(Error::Unsupported(
            "closed full-model learned projection upload only".into(),
        )),
    }
}
struct SourceWindow(TensorWindow);
impl AsRef<[u8]> for SourceWindow {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}
struct Progress {
    pending: Option<Arc<SourceWindow>>,
    coverage: Coverage,
    enqueued: usize,
    checkpoints: usize,
}
pub(super) struct ModelUploadOwner {
    // Actual pending storage retires before original source/metadata admissions.
    progress: Mutex<Progress>,
    lease: ResidentLease,
    matrices: Vec<VerifiedMatrix>,
    descriptor: Vllm08Operator,
    source: Arc<OriginalWindowModelSource>,
    _metadata: WindowWorkspace,
}
impl ModelUploadOwner {
    fn progress(&self) -> Result<std::sync::MutexGuard<'_, Progress>> {
        self.progress
            .lock()
            .map_err(|_| Error::State("original learned upload progress poisoned".into()))
    }
    pub(super) fn require_complete(&self) -> Result<()> {
        let layout = layout(&self.descriptor)?;
        let progress = self.progress()?;
        progress.coverage.require_complete()?;
        if progress.pending.is_some()
            || progress.enqueued != progress.checkpoints
            || progress.enqueued == 0
            || self.lease.bytes() != layout.bytes()
            || self.lease.role() != BufferRole::ImmutableWeights
            || self.lease.source_identity() != self.source.source_identity()
            || self.descriptor.call().source() != self.source.source_identity()
            || self.matrices.len() != self.descriptor.matrices().len()
        {
            return Err(Error::State(
                "original ordered learned upload incomplete/source changed".into(),
            ));
        }
        for (matrix, descriptor) in self.matrices.iter().zip(self.descriptor.matrices()) {
            if matrix.source_identity() != self.source.source_identity()
                || matrix.tensor().name != descriptor.name
                || matrix.axes() != descriptor.axes
                || matrix.tensor().kind != descriptor.kind
                || matrix.rows() != descriptor.rows
                || matrix.columns() != descriptor.columns
            {
                return Err(Error::State(
                    "original full-model member owner differs".into(),
                ));
            }
        }
        Ok(())
    }
}
pub(crate) struct PendingModelWeight {
    owner: Arc<ModelUploadOwner>,
}
pub(crate) struct ModelWeight {
    owner: Arc<ModelUploadOwner>,
}
impl PendingModelWeight {
    pub(crate) fn bind_initialized(self) -> Result<Arc<ModelWeight>> {
        self.owner.require_complete()?;
        self.owner.lease.service().ensure_usable()?;
        if !self.owner.lease.is_initialized() {
            return Err(Error::State(
                "learned weight original initialization incomplete".into(),
            ));
        }
        Ok(Arc::new(ModelWeight { owner: self.owner }))
    }
}
impl ModelWeight {
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.owner.lease
    }
    /// Immutable W is shared across real requests/phases; this does not transfer
    /// any teacher/Root authority. Node operands retain their own issuer seals.
    pub(crate) fn require_operator(
        &self,
        source: &Arc<OriginalWindowModelSource>,
        descriptor: &Vllm08Operator,
    ) -> Result<()> {
        self.require_initialized()?;
        descriptor.validate()?;
        if !Arc::ptr_eq(source, &self.owner.source)
            || descriptor.call().source() != source.source_identity()
            || descriptor.layer() != self.owner.descriptor.layer()
            || descriptor.kind() != self.owner.descriptor.kind()
            || descriptor.matrices() != self.owner.descriptor.matrices()
            || !descriptor.parameters().is_empty()
        {
            return Err(Error::State(
                "full-model exact shared immutable W source/roles".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require_initialized(&self) -> Result<()> {
        self.owner.require_complete()?;
        self.owner.lease.service().ensure_usable()?;
        if !self.owner.lease.is_initialized() {
            return Err(Error::State(
                "full original model weight not initialized".into(),
            ));
        }
        Ok(())
    }
}

impl ResidentInitialization<'_> {
    /// One source-bound packed W, without retaining a complete host pack. Every
    /// selected native program and managed producer/cache must be preallocated
    /// before activating the native pool, as in the original preparation path.
    pub(crate) async fn upload_original_model_weight(
        &mut self,
        source: Arc<OriginalWindowModelSource>,
        descriptor: Vllm08Operator,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PendingModelWeight> {
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        require_original08_upload_source(&source, &descriptor)?;
        self.upload_model_weight_inner(source, descriptor, cancelled)
            .await
    }
    /// A genuine bound short-HF source authenticates the family/ordered roles.
    /// Old .8B and2B MLP admission wrappers remain unchanged.
    pub(crate) async fn upload_short_hf_model_weight(
        &mut self,
        scope: &Arc<crate::vllm_cuda_qualification::short_hf_model::ShortHfModelBinding>,
        node: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PendingModelWeight> {
        not_cancelled(cancelled)?;
        if !scope.projection_nodes().contains(&node)
            || scope.source().source().source_identity() != self.service.source_identity()
        {
            return Err(Error::State(
                "short-HF real source learned node before H2D".into(),
            ));
        }
        let zerogravity_core::accelerator::program::PlanOperation::TypedVllm08 {
            descriptor, ..
        } = &scope.record().plan().nodes()[node].operation
        else {
            return Err(Error::State(
                "short-HF original typed learned descriptor absent".into(),
            ));
        };
        if descriptor.call() != scope.record().call() {
            return Err(Error::State("short-HF learned call issuer differs".into()));
        }
        self.upload_model_weight_inner(
            scope.source().source().clone(),
            descriptor.as_ref().clone(),
            cancelled,
        )
        .await
    }
    async fn upload_model_weight_inner(
        &mut self,
        source: Arc<OriginalWindowModelSource>,
        descriptor: Vllm08Operator,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PendingModelWeight> {
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        // Only exact .8B, sealed2B MLP or source-bound short-HF wrappers reach here.
        let shape = layout(&descriptor)?;
        if source.source_identity() != self.service.source_identity()
            || descriptor.call().source() != source.source_identity()
            || self.model_source_uploads.iter().any(|entry| {
                entry.descriptor.layer() == descriptor.layer()
                    && entry.descriptor.kind() == descriptor.kind()
            })
        {
            return Err(Error::State(
                "original full-model W source/service/duplicate role".into(),
            ));
        }
        let metadata = source.source().reserve_workspace(METADATA_BYTES)?;
        let mut matrices = Vec::with_capacity(descriptor.matrices().len());
        let coverage = Coverage::new(shape);
        for (member, &rows) in descriptor.matrices().iter().zip(coverage.members()) {
            let matrix = source.source().matrix(&member.name, member.axes)?;
            if matrix.source_identity() != source.source_identity()
                || matrix.tensor().kind != QuantKind::Bf16
                || matrix.rows() != rows
                || matrix.rows() != member.rows
                || matrix.columns() != shape.columns()
                || matrix.columns() != member.columns
                || matrix.axes() != member.axes
            {
                return Err(Error::State(
                    "original learned W exact source member/axes/shape".into(),
                ));
            }
            matrices.push(matrix);
        }
        if matrices.len() != coverage.members().len() {
            return Err(Error::State(
                "complete ordered source member cohort required".into(),
            ));
        }
        let queue = self.service.pinned_cuda_queue()?.clone();
        let lease = self.allocate_empty(shape.bytes(), BufferRole::ImmutableWeights)?;
        let owner = Arc::new(ModelUploadOwner {
            progress: Mutex::new(Progress {
                pending: None,
                coverage,
                enqueued: 0,
                checkpoints: 0,
            }),
            lease,
            matrices,
            descriptor,
            source,
            _metadata: metadata,
        });
        self.retain_completion_owner(owner.clone())?;
        self.model_source_uploads.push(owner.clone());
        loop {
            not_cancelled(cancelled)?;
            let Some(region) = owner.progress()?.coverage.next()? else {
                break;
            };
            let matrix = &owner.matrices[region.member];
            let window = match cancelled {
                Some(flag) => owner.source.source().read_rows_cancellable(
                    matrix,
                    region.first..region.end,
                    flag,
                )?,
                None => owner
                    .source
                    .source()
                    .read_rows(matrix, region.first..region.end)?,
            };
            let source_offset = matrix
                .tensor()
                .byte_offset
                .checked_add((region.first * shape.columns() * 2) as u64)
                .ok_or_else(|| Error::Limit("original source member byte offset".into()))?;
            if window.row_range() != (region.first..region.end)
                || window.source_offset() != source_offset
                || window.as_slice().len() != region.bytes
                || window.matrix().source_identity() != owner.source.source_identity()
                || window.matrix().tensor().name != matrix.tensor().name
                || window.matrix().axes() != matrix.axes()
                || window
                    .as_slice()
                    .chunks_exact(2)
                    .any(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 == 0x7f80)
            {
                return Err(Error::State(
                    "original finite learned source window/offset differs".into(),
                ));
            }
            not_cancelled(cancelled)?;
            let window = Arc::new(SourceWindow(window));
            {
                let mut progress = owner.progress()?;
                progress.coverage.begin(region)?;
                if progress.pending.replace(window.clone()).is_some() {
                    return Err(Error::State(
                        "overlapping original learned host window".into(),
                    ));
                }
            }
            not_cancelled(cancelled)?;
            // SAFETY: exact authenticated ordinary host allocation/admission is
            // retained by THIS original armed init before actual H2D enqueue.
            unsafe {
                queue.enqueue_vllm08_weight_window(
                    owner.lease.handle(),
                    shape,
                    region.destination_offset,
                    window.clone(),
                )
            }
            .map_err(|error| Error::State(format!("original learned H2D enqueue: {error}")))?;
            owner.progress()?.enqueued += 1;
            self.service
                .uploaded_value_bytes
                .fetch_add(region.bytes as u64, Ordering::AcqRel);
            not_cancelled(cancelled)?;
            let view = owner
                .lease
                .handle()
                .offset_start(region.destination_offset as u64)
                .offset_end((shape.bytes() - region.destination_offset - region.bytes) as u64);
            self.service
                .client
                .sync_buffers([&view])
                .await
                .map_err(|error| {
                    Error::State(format!("original learned window completion: {error}"))
                })?;
            self.service.ensure_usable()?;
            not_cancelled(cancelled)?;
            // Sole credit site, after actual original-range physical completion.
            let retired = {
                let mut progress = owner.progress()?;
                progress.coverage.checkpoint(region)?;
                progress.checkpoints += 1;
                progress.pending.take()
            };
            drop(retired);
            drop(window);
        }
        owner.require_complete()?;
        self.uploaded.insert(owner.lease.id());
        Ok(PendingModelWeight { owner })
    }
}

impl ModelWeight {
    /// SAME finite layout selected by the actual original upload allocator.
    pub(crate) fn allocation_bytes(operator: &Vllm08Operator) -> Result<usize> {
        Ok(layout(operator)?.bytes())
    }
}
