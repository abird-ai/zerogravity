//! Finite source-owned learned uploads on the original initialization/actor.
//! Original tied-table transport and the old32MiB packer remain unchanged.
use super::*;
use cubecl::cuda::SourceWeightLayout;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedMatrix, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    SourceNumericalContract, SourceOperator, SourceOperatorKind,
};
use zerogravity_core::quant::QuantKind;

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
fn require_layout(operator: &SourceOperator, shape: SourceWeightLayout) -> Result<()> {
    operator.validate()?;
    let members = Coverage::new(shape);
    if !matches!(operator.kind(), SourceOperatorKind::Projection { .. })
        || operator.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
        || !operator.parameters().is_empty()
        || operator.matrices().len() != members.members().len()
        || operator
            .matrices()
            .iter()
            .zip(members.members())
            .any(|(matrix, &rows)| {
                matrix.source != operator.call().source()
                    || matrix.kind != QuantKind::Bf16
                    || matrix.rows != rows
                    || matrix.columns != shape.columns()
                    || matrix.axes
                        != zerogravity_core::accelerator::program::MatrixAxes::OutputInput
            })
    {
        return Err(Error::Unsupported(
            "actual ordered BF16 members differ from packed layout".into(),
        ));
    }
    Ok(())
}

/// Selects byte packing only; the source factory owns model/recipe eligibility.
pub(crate) fn layout_for_operator(operator: &SourceOperator) -> Result<SourceWeightLayout> {
    for shape in [
        SourceWeightLayout::GdnQkvz,
        SourceWeightLayout::GdnBa,
        SourceWeightLayout::GdnOutput,
        SourceWeightLayout::AttentionQkv,
        SourceWeightLayout::AttentionOutput,
        SourceWeightLayout::MlpGateUp,
        SourceWeightLayout::MlpDown,
        SourceWeightLayout::Vocabulary08,
        SourceWeightLayout::Qwen2bGdnQkvz,
        SourceWeightLayout::Qwen2bGdnBa,
        SourceWeightLayout::Qwen2bGdnOutput,
        SourceWeightLayout::Qwen2bAttentionQkv,
        SourceWeightLayout::Qwen2bAttentionOutput,
        SourceWeightLayout::Qwen2bMlpGateUp,
        SourceWeightLayout::Qwen2bMlpDown,
        SourceWeightLayout::Vocabulary2b,
    ] {
        if require_layout(operator, shape).is_ok() {
            return Ok(shape);
        }
    }
    Err(Error::Unsupported(
        "no existing packed BF16 member layout".into(),
    ))
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
    descriptor: SourceOperator,
    source: VerifiedTensorSource,
    layout: SourceWeightLayout,
    _metadata: WindowWorkspace,
}
impl ModelUploadOwner {
    fn progress(&self) -> Result<std::sync::MutexGuard<'_, Progress>> {
        self.progress
            .lock()
            .map_err(|_| Error::State("original learned upload progress poisoned".into()))
    }
    pub(super) fn require_complete(&self) -> Result<()> {
        let layout = self.layout;
        require_layout(&self.descriptor, layout)?;
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
    pub(crate) fn require_source_operator(
        &self,
        source: &VerifiedTensorSource,
        actual: &SourceOperator,
    ) -> Result<()> {
        self.require_initialized()?;
        actual.validate()?;
        if !source.same_issuer(&self.owner.source)
            || actual.call().source() != source.source_identity()
            || actual.kind() != self.owner.descriptor.kind()
            || actual.numerical_contract() != self.owner.descriptor.numerical_contract()
            || actual.matrices() != self.owner.descriptor.matrices()
            || actual.parameters() != self.owner.descriptor.parameters()
        {
            return Err(Error::State(
                "exact shared immutable W source/members/contract differ".into(),
            ));
        }
        require_layout(actual, self.owner.layout)
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
    pub(crate) async fn upload_model_weight(
        &mut self,
        source: VerifiedTensorSource,
        descriptor: SourceOperator,
        shape: SourceWeightLayout,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PendingModelWeight> {
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        require_layout(&descriptor, shape)?;
        if source.source_identity() != self.service.source_identity()
            || descriptor.call().source() != source.source_identity()
            || self.model_source_uploads.iter().any(|entry| {
                entry.descriptor.matrices().first().map(|m| m.name.as_str())
                    == descriptor.matrices().first().map(|m| m.name.as_str())
            })
        {
            return Err(Error::State(
                "original full-model W source/service/duplicate role".into(),
            ));
        }
        let metadata = source.reserve_workspace(METADATA_BYTES)?;
        let mut matrices = Vec::with_capacity(descriptor.matrices().len());
        let coverage = Coverage::new(shape);
        for (member, &rows) in descriptor.matrices().iter().zip(coverage.members()) {
            let matrix = source.matrix(&member.name, member.axes)?;
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
            layout: shape,
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
                Some(flag) => {
                    owner
                        .source
                        .read_rows_cancellable(matrix, region.first..region.end, flag)?
                }
                None => owner.source.read_rows(matrix, region.first..region.end)?,
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
    pub(crate) fn expected_source_windows(&self) -> Result<usize> {
        let shape = self.owner.layout;
        let rows_per_window = 1024 * 1024 / (shape.columns() * 2);
        Coverage::new(shape)
            .members()
            .iter()
            .try_fold(0usize, |n, rows| {
                n.checked_add(rows.div_ceil(rows_per_window))
                    .ok_or_else(|| Error::Limit("source window count".into()))
            })
    }
    pub(crate) fn packing_layout(operator: &SourceOperator) -> Result<SourceWeightLayout> {
        layout_for_operator(operator)
    }
    pub(crate) fn allocation_bytes(operator: &SourceOperator) -> Result<usize> {
        Ok(layout_for_operator(operator)?.bytes())
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::{
        MatrixAxes, SourceCall, SourceCallMode, SourceRows, SourceTensor, TextCallGeometry,
    };
    fn projection(members: &[(usize, usize)]) -> SourceOperator {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let call = SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, 248320, 512).unwrap(),
                0,
                vec![353].into(),
            )
            .unwrap(),
            SourceCallMode::Prefill,
        )
        .unwrap();
        let matrices = members
            .iter()
            .enumerate()
            .map(|(index, &(rows, columns))| SourceTensor {
                source,
                name: format!("member{index}"),
                kind: QuantKind::Bf16,
                rows,
                columns,
                axes: MatrixAxes::OutputInput,
            })
            .collect();
        SourceOperator::new(
            call,
            SourceOperatorKind::Projection { selected_row: None },
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            matrices,
            vec![],
        )
        .unwrap()
    }
    #[test]
    fn byte_layout_requires_every_ordered_member_and_complete_extent() {
        let correct = projection(&[(6144, 1024), (2048, 1024)]);
        require_layout(&correct, SourceWeightLayout::GdnQkvz).unwrap();
        for members in [
            vec![(2048, 1024), (6144, 1024)],
            vec![(6144, 1024)],
            vec![(6144, 2048), (2048, 2048)],
            vec![(6144, 1024), (2049, 1024)],
        ] {
            assert!(require_layout(&projection(&members), SourceWeightLayout::GdnQkvz).is_err());
        }
        assert_eq!(
            ModelWeight::allocation_bytes(&correct).unwrap(),
            SourceWeightLayout::GdnQkvz.bytes()
        );
    }
}
