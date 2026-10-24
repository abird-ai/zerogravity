//! Original computed final-normalizer→native fullV, same sealed value map.
use super::super::model_values::ModelValues;
use super::super::projection_cubin::resident::PreparedImages;
use super::resident::{HeadInputView, HeadPrograms};
use super::*;
use crate::resident_leases::{ResidentInitialization, TiedWeight};
use zerogravity_core::accelerator::program::{
    PlanOperation, Vllm08Kind, Vllm08Projection, VllmBf16Phase,
};

#[derive(Clone)]
struct BoundHead {
    input: HeadInputView,
    values: Arc<ModelValues>,
    weight: Arc<TiedWeight>,
    output: ResidentLease,
    node: usize,
    case: VocabularyHeadCase,
}
pub(crate) struct ModelHeadPrograms {
    programs: Arc<HeadPrograms>,
    bindings: Vec<BoundHead>,
    _metadata: WindowWorkspace,
}
impl ModelHeadPrograms {
    /// Final normalizers are actual previously prepared producers. The SAME
    /// immutable map also feeds the BF16 numeric-tie output consumer.
    pub(crate) fn prepare(
        images: Arc<PreparedImages>,
        source: Arc<OriginalWindowModelSource>,
        weight: Arc<TiedWeight>,
        phases: Vec<(Arc<ModelValues>, Arc<FusedRmsPrepared>)>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        source.require_vllm08_full_source()?;
        if phases.len() != 5
            || phases.capacity() > 8
            || !Arc::ptr_eq(&source, weight.source())
            || !weight.lease().is_initialized()
        {
            return Err(Error::State(
                "full24 original five-phase shared tied head".into(),
            ));
        }
        let metadata = source.source().reserve_workspace(64 * 1024)?;
        let mut bindings = Vec::with_capacity(5);
        for (values, fused) in phases {
            let record = values.record();
            let call = record.call();
            let case = match (
                call.request().legacy_phase()?,
                call.request().teachers().base(),
                call.rows(),
            ) {
                (VllmBf16Phase::InitialOne, 0, 1) => VocabularyHeadCase::Initial11,
                (VllmBf16Phase::CachedOne, 1, 1) => VocabularyHeadCase::Decode353After11,
                (VllmBf16Phase::Initial64, 0, 64) => VocabularyHeadCase::InitialFirst64,
                (VllmBf16Phase::ContinuedTwo, 64, 2) => VocabularyHeadCase::Continued65And66,
                (VllmBf16Phase::CachedOne, 66, 1) => VocabularyHeadCase::Decode353After66,
                _ => {
                    return Err(Error::Unsupported(
                        "full24 head keeps actual full-phase request geometry".into(),
                    ));
                }
            };
            let node = record
                .plan()
                .nodes()
                .len()
                .checked_sub(1)
                .ok_or_else(|| Error::State("full24 genuine head node absent".into()))?;
            let head = &record.plan().nodes()[node];
            let PlanOperation::TypedVllm08 { descriptor, state } = &head.operation else {
                return Err(Error::State(
                    "full24 actual typed head descriptor absent".into(),
                ));
            };
            descriptor.validate()?;
            if descriptor.kind() != Vllm08Kind::Projection(Vllm08Projection::Head)
                || descriptor.call() != call
                || descriptor.layer().is_some()
                || state.is_some()
                || head.state.is_some()
                || head.inputs != [record.normalized()]
                || head.outputs != [record.logits()]
                || !Arc::ptr_eq(values.service(), weight.lease().service())
                || fused.normalized().id() != values.lease(record.normalized())?.id()
                || fused.normalized().allocation_generation()
                    != values.lease(record.normalized())?.allocation_generation()
            {
                return Err(Error::State(
                    "full24 head actual final-normalizer/value/phase seal".into(),
                ));
            }
            let normalized = record
                .plan()
                .vllm_value_provenance(record.normalized())?
                .ok_or_else(|| Error::State("full24 final normalizer provenance absent".into()))?;
            let zerogravity_core::accelerator::program::VllmValueOrigin::FullModel(origin) =
                normalized.origin()
            else {
                return Err(Error::State("full24 actual final RMS origin absent".into()));
            };
            let normalizer = record
                .plan()
                .nodes()
                .get(origin.node_index())
                .ok_or_else(|| Error::State("full24 actual final RMS node absent".into()))?;
            let PlanOperation::TypedVllm08 {
                descriptor: final_descriptor,
                ..
            } = &normalizer.operation
            else {
                return Err(Error::State(
                    "full24 typed final RMS descriptor absent".into(),
                ));
            };
            fused.require_full_model_operator(final_descriptor)?;
            let input = HeadInputView::computed(source.clone(), case, fused)?;
            let output = values.lease(record.logits())?.clone();
            if input.lease().id() != values.lease(record.normalized())?.id()
                || output.bytes() != 496_640
                || output.id() == weight.lease().id()
            {
                return Err(Error::State(
                    "full24 head binds original fullV output and full-phase input once".into(),
                ));
            }
            bindings.push(BoundHead {
                input,
                values,
                weight: weight.clone(),
                output,
                node,
                case,
            });
        }
        if bindings
            .iter()
            .map(|binding| binding.case)
            .collect::<Vec<_>>()
            != VocabularyHeadCase::ALL
        {
            return Err(Error::State(
                "full24 exact original five head phase ordering".into(),
            ));
        }
        init.retain_completion_owner(source)?;
        init.retain_completion_owner(weight)?;
        for binding in &bindings {
            init.retain_completion_owner(binding.values.clone())?;
        }
        let programs = HeadPrograms::prepare(images, init, cancelled)?;
        let result = Arc::new(Self {
            programs,
            bindings,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(in crate::vllm_cuda_qualification) fn retention_witness(
        &self,
    ) -> Result<super::super::projection_cubin::resident::NativeRetentionWitness> {
        self.programs.retention_witness()
    }
    pub(crate) fn require_phase(&self, phase: usize, values: &Arc<ModelValues>) -> Result<()> {
        let prepared = self
            .bindings
            .get(phase)
            .ok_or_else(|| Error::State("full24 head sealed phase absent".into()))?;
        if !Arc::ptr_eq(&prepared.values, values)
            || prepared.output.id() != values.lease(values.record().logits())?.id()
        {
            return Err(Error::State(
                "full24 Head library differs from SAME original phase value map".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn accesses(&self, phase: usize) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let binding = self
            .bindings
            .get(phase)
            .ok_or_else(|| Error::State("full24 head phase absent".into()))?;
        Ok(vec![
            (binding.weight.lease().clone(), DeviceAccess::Read),
            (binding.input.lease().clone(), DeviceAccess::Read),
            (binding.output.clone(), DeviceAccess::Write),
        ])
    }
    pub(crate) fn enqueue(
        &self,
        phase: usize,
        node: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        let binding = self
            .bindings
            .get(phase)
            .ok_or_else(|| Error::State("full24 head phase absent".into()))?;
        if binding.node != node {
            return Err(Error::State("full24 unbound head node".into()));
        }
        self.programs
            .enqueue(&binding.weight, &binding.input, &binding.output, cancelled)
    }
    pub(crate) unsafe fn retire_kernels_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        unsafe { self.programs.retire_kernels_after_physical_completion() }
    }
}

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
mod natural;
