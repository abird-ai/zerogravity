//! Actual source/producer joins for the shared prepared BF16 selected-row head.
use super::{
    model_values::ModelValues,
    prepared_bf16_head::{Bf16HeadCall, HeadPreparation, HeadRun, HeadWitness, PreparedBf16Head},
    projection_cubin::resident::PreparedImages,
    text_model::TextModelBinding,
};
use crate::resident_program::source_bindings::SourceBindings;
use crate::{
    cuda::rms::{FullModelRmsScratch, FusedRmsPrepared},
    resident_leases::{
        ResidentInitialization, ResidentLease, TiedWeight, resident_matrix::Bf16MatrixSource,
    },
};
use cubecl::cuda::{Bf16VocabularyHead, OwnedCudaKernel, SourceWeightLayout};
use std::sync::{Arc, Weak, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{SourceOwnerWitness, VerifiedTensorSource};
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::program::PlanOperation;
use zerogravity_core::accelerator::program::SourceTensor;
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::device_events::DeviceAccess;
use zerogravity_core::quant::QuantKind;
mod mode;
pub(crate) use super::prepared_bf16_head::{
    HeadFault as StreamedHeadFault, STREAMED_WINDOW_STAGING_UPPER,
};
pub(crate) use mode::HeadMode;
const VOCAB: usize = 248320;
struct FamilyHead {
    source: VerifiedTensorSource,
    tied: Option<Arc<TiedWeight>>,
    prepared: Arc<PreparedBf16Head>,
    images: Arc<PreparedImages>,
}
/// The original flight retains family seals together with the shared run.
pub(crate) struct Run {
    _head: Arc<StreamedHead>,
    _operation: Arc<HeadRun>,
}
pub(crate) struct StreamedHead {
    cold: Arc<FamilyHead>,
    operation: Arc<Bf16HeadCall>,
    scope: Arc<TextModelBinding>,
    values: Arc<ModelValues>,
    final_normalizer: Arc<FusedRmsPrepared>,
    rms_scope: Arc<FullModelRmsScratch>,
}
fn native(e: cubecl::server::ServerError) -> Error {
    Error::State(format!("streamed head original actor: {e}"))
}
fn head_operand(scope: &TextModelBinding) -> Result<&SourceTensor> {
    let PlanOperation::SourceOperator { descriptor, .. } =
        &scope.record().plan().nodes()[254].operation
    else {
        return Err(Error::State(
            "streamed head actual Head254 descriptor".into(),
        ));
    };
    let [member] = descriptor.matrices() else {
        return Err(Error::State(
            "streamed head one authentic tied member".into(),
        ));
    };
    Ok(member)
}
impl StreamedHead {
    #[expect(
        clippy::too_many_arguments,
        reason = "source/producer/one-set/actual-init and diagnostic transport are distinct owners"
    )]
    pub(crate) fn prepare(
        scope: Arc<TextModelBinding>,
        values: Arc<ModelValues>,
        rms_scope: Arc<FullModelRmsScratch>,
        final_normalizer: Arc<FusedRmsPrepared>,
        images: Arc<PreparedImages>,
        init: &mut ResidentInitialization<'_>,
        cancel: Option<&AtomicBool>,
        audit: bool,
        mode: HeadMode,
        tied: Option<Arc<TiedWeight>>,
    ) -> Result<Arc<Self>> {
        if cancel.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State("streamed head cancelled".into()));
        }
        scope.require_values(&values)?;
        let source = scope.source().clone();
        if mode.resident() != tied.is_some() || (mode.resident() && audit) {
            return Err(Error::State(
                "text resident source mode/weight/audit contract".into(),
            ));
        }
        let member = head_operand(&scope)?;
        let matrix = source.matrix(&member.name, member.axes)?;
        if member.source != source.source_identity()
            || matrix.tensor().kind != member.kind
            || matrix.rows() != member.rows
            || matrix.columns() != member.columns
            || matrix.axes() != member.axes
        {
            return Err(Error::State(
                "streamed head actual source operand header differs".into(),
            ));
        }
        let matrix = Bf16MatrixSource::new(source.clone(), matrix)?;
        let layout = match (matrix.rows(), matrix.columns()) {
            (VOCAB, 1024) => SourceWeightLayout::Vocabulary08,
            (VOCAB, 2048) => SourceWeightLayout::Vocabulary2b,
            _ => {
                return Err(Error::Unsupported(
                    "streamed head native source geometry".into(),
                ));
            }
        };
        if matrix.matrix().tensor().kind != QuantKind::Bf16
            || matrix.source().source_identity() != source.source_identity()
            || !Arc::ptr_eq(rms_scope.record(), scope.record())
        {
            return Err(Error::State(
                "streamed head exact source/profile/record/matrix".into(),
            ));
        }
        let manifest = if mode.resident() {
            Bf16VocabularyHead::resident_manifest(TiedWeight::source_layout(&source, member)?)
                .map_err(native)?
        } else {
            Bf16VocabularyHead::streamed_manifest(layout).map_err(native)?
        };
        images.require_service(values.service())?;
        init.require_service(values.service())?;
        init.retain_completion_owner(scope.clone())?;
        init.retain_completion_owner(values.clone())?;
        init.retain_completion_owner(final_normalizer.completion_owner())?;
        if let Some(weight) = &tied {
            if !weight.source().same_issuer(&source)
                || !weight.lease().is_initialized()
                || weight.lease().bytes() != weight.bytes()
                || weight.lease().role()
                    != zerogravity_core::device_events::BufferRole::ImmutableWeights
                || weight.copy_counts()? != (weight.windows(), weight.windows())
                || !Arc::ptr_eq(weight.lease().service(), values.service())
            {
                return Err(Error::State(
                    "resident Head requires SAME complete original tied source upload".into(),
                ));
            }
            weight
                .matrix()
                .require_matrix(&source, values.service(), matrix.matrix())?;
            init.retain_completion_owner(weight.clone())?;
        }
        let prepared = PreparedBf16Head::prepare(
            HeadPreparation {
                matrix,
                operand: member.clone(),
                numerical: scope.record().plan().numerical_profile(),
                manifest,
                images: images.clone(),
                resident: tied.as_ref().map(|w| w.matrix().clone()),
                audit,
            },
            values.service(),
            init,
            cancel,
        )?;
        let cold = Arc::new(FamilyHead {
            source,
            tied,
            prepared,
            images,
        });
        init.retain_completion_owner(cold.clone())?;
        let result = Self::bind(cold, scope, values, rms_scope, final_normalizer)?;
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    fn bind(
        cold: Arc<FamilyHead>,
        scope: Arc<TextModelBinding>,
        values: Arc<ModelValues>,
        rms_scope: Arc<FullModelRmsScratch>,
        final_normalizer: Arc<FusedRmsPrepared>,
    ) -> Result<Arc<Self>> {
        scope.require_values(&values)?;
        let operation =
            cold.prepared.bind(
                scope.call().rows().len(),
                scope.record().selection().selected_row().ok_or_else(|| {
                    Error::State("streamed head actual selected row absent".into())
                })?,
                head_operand(&scope)?,
                scope.record().plan().numerical_profile(),
                values.lease(ValueId(324))?.clone(),
                values.lease(ValueId(326))?.clone(),
            )?;
        let result = Arc::new(Self {
            cold,
            operation,
            scope,
            values,
            rms_scope,
            final_normalizer,
        });
        result.require_binding()?;
        Ok(result)
    }
    pub(crate) fn bind_call(
        self: &Arc<Self>,
        scope: Arc<TextModelBinding>,
        values: Arc<ModelValues>,
        rms_scope: Arc<FullModelRmsScratch>,
        final_normalizer: Arc<FusedRmsPrepared>,
    ) -> Result<Arc<Self>> {
        self.cold.prepared.service().ensure_usable()?;
        if Arc::ptr_eq(scope.record(), self.scope.record()) || self.cold.prepared.is_active()? {
            return Err(Error::State(
                "streamed head fresh record and completed original run required".into(),
            ));
        }
        Self::bind(
            self.cold.clone(),
            scope,
            values,
            rms_scope,
            final_normalizer,
        )
    }
    fn require_binding(&self) -> Result<()> {
        self.scope.require_values(&self.values)?;
        let hidden = self.scope.hidden_size();
        let view = self.scope.head_view();
        let PlanOperation::SourceOperator { descriptor, .. } =
            &self.scope.record().plan().nodes()[253].operation
        else {
            return Err(Error::State(
                "streamed head final normalizer source producer".into(),
            ));
        };
        self.final_normalizer.require_source_operator(descriptor)?;
        if !self.scope.source().same_issuer(&self.cold.source)
            || !self.cold.prepared.source().same_issuer(&self.cold.source)
            || !Arc::ptr_eq(self.rms_scope.record(), self.scope.record())
            || !Arc::ptr_eq(self.values.service(), self.cold.prepared.service())
            || self.cold.prepared.columns() != hidden
            || self.operation.input().bytes() != view.full_bytes
            || view.offset + hidden * 2 != view.full_bytes
            || view.output_bytes != self.cold.prepared.rows() * 2
            || self.operation.output().bytes() != self.cold.prepared.rows() * 2
            || self.operation.input().id() != self.final_normalizer.normalized().id()
            || self.operation.input().allocation_generation()
                != self.final_normalizer.normalized().allocation_generation()
            || self.operation.input().id() == self.operation.output().id()
            || self.operation.input().id() == self.cold.prepared.lease().id()
            || self.operation.output().id() == self.cold.prepared.lease().id()
        {
            return Err(Error::State(
                "streamed head SAME real source/record/final producer/WXY owners".into(),
            ));
        }
        let node = &self.scope.record().plan().nodes()[254];
        let PlanOperation::SourceOperator { descriptor, state } = &node.operation else {
            return Err(Error::State("streamed head typed node254".into()));
        };
        if descriptor.kind()
            != (zerogravity_core::accelerator::program::SourceOperatorKind::Projection {
                selected_row: Some(self.scope.record().selection().selected_row().ok_or_else(
                    || Error::State("streamed head actual selected row absent".into()),
                )?),
            })
            || descriptor.call() != self.scope.record().call()
            || state.is_some()
            || node.state.is_some()
            || node.inputs != [ValueId(324)]
            || node.outputs != [ValueId(326)]
        {
            return Err(Error::State(
                "streamed head exact original node/SSA/effect/selected row".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.operation.accesses()
    }
    pub(crate) fn register(
        self: &Arc<Self>,
        registration: &mut crate::resident_inference::RmsScratchRegistration<'_, '_>,
    ) -> Result<()> {
        self.require_binding()?;
        registration.require_scope(&self.rms_scope)?;
        registration.require_value_storage_accesses(&self.accesses())?;
        // SAFETY: actual original registration checked SAME RMS producer scope
        // and WXY accesses; closure retains both family seals and shared run.
        unsafe {
            self.operation.register(|operation| {
                registration.retain_streamed_head(Arc::new(Run {
                    _head: self.clone(),
                    _operation: operation,
                }))
            })
        }
    }
    pub(crate) fn enqueue(
        self: &Arc<Self>,
        node: usize,
        cancel: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.require_binding()?;
        self.values.require_storage_node(node)?;
        if node != 254 {
            return Err(Error::State("streamed head actual cursor254".into()));
        }
        // SAFETY: original call/producer/storage cursor254 and exact leases
        // were just rechecked; registered original flight retains this adapter.
        unsafe { self.operation.enqueue(cancel) }
    }
    pub(crate) fn resident_weight(&self) -> Option<&Arc<TiedWeight>> {
        self.cold.tied.as_ref()
    }
    pub(crate) fn enqueue_placement(&self) -> crate::resident_leases::EnqueuePlacement {
        self.operation.enqueue_placement()
    }
    pub(crate) fn arm_fault(&self, cut: StreamedHeadFault) -> Result<()> {
        self.operation.arm_fault(cut)
    }
    pub(crate) fn retention_witness(self: &Arc<Self>) -> Result<StreamedHeadWitness> {
        Ok(StreamedHeadWitness {
            head: Arc::downgrade(self),
            source: self.cold.source.weak_owner(),
            operation: self.operation.retention_witness()?,
        })
    }
    pub(crate) fn record(&self) -> &Arc<SourceBindings> {
        self.scope.record()
    }
    pub(crate) fn complete(&self) -> Result<bool> {
        self.operation.complete()
    }
    pub(crate) fn retire_transfers_after_completion(&self) -> Result<()> {
        self.operation.retire_transfers_after_completion()
    }
    pub(crate) fn retirement_counts(&self) -> Result<[usize; 5]> {
        self.operation.retirement_counts()
    }
    pub(crate) fn counts(&self) -> Result<[usize; 4]> {
        self.operation.counts()
    }
    pub(crate) fn retained_graph(&self) -> Result<bool> {
        self.require_binding()?;
        let weight = match &self.cold.tied {
            Some(tied) => {
                tied.source().same_issuer(&self.cold.source)
                    && tied.copy_counts()? == (tied.windows(), tied.windows())
            }
            None => true,
        };
        Ok(weight && self.operation.retained_graph()?)
    }
    pub(crate) fn kernel(&self) -> &Arc<OwnedCudaKernel> {
        self.operation.kernel()
    }
    pub(crate) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        // SAFETY: caller's original flight proves physical completion; shared kernel/set are SAME.
        unsafe { self.operation.retire_after_physical_completion() }
    }
    pub(crate) fn images(&self) -> &Arc<PreparedImages> {
        &self.cold.images
    }
}
pub(crate) struct StreamedHeadWitness {
    head: Weak<StreamedHead>,
    source: SourceOwnerWitness,
    operation: HeadWitness,
}
pub(crate) struct StreamedHeadRetained {
    pub(crate) graph: bool,
    pub(crate) source: bool,
    pub(crate) run: bool,
    pub(crate) pending_window: bool,
    pub(crate) actual_counts: [usize; 4],
    pub(crate) retirement_counts: [usize; 5],
    pub(crate) source_staging_bytes: usize,
    pub(crate) resident_immutable_weight: bool,
}
impl StreamedHeadWitness {
    pub(crate) fn inspect(&self) -> Result<StreamedHeadRetained> {
        let source_retained = self.source.retained();
        let head = self
            .head
            .upgrade()
            .ok_or_else(|| Error::State("streamed head original graph not retained".into()))?;
        let shared = self.operation.inspect()?;
        Ok(StreamedHeadRetained {
            graph: shared.graph && head.retained_graph()?,
            source: source_retained && self.source.same_issuer(&head.cold.source),
            run: shared.run,
            pending_window: shared.pending_window,
            actual_counts: shared.actual_counts,
            retirement_counts: shared.retirement_counts,
            source_staging_bytes: shared.source_staging_bytes,
            resident_immutable_weight: shared.resident_immutable_weight && head.cold.tied.is_some(),
        })
    }
}
