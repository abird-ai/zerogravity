//! Closed source-family forwarding to existing factories; no tensor evaluator.
use super::*;
use crate::resident_leases::Qwen2bMlpWeights;
use crate::vllm_cuda_qualification::canonical47_mlp::{
    Mlp47Programs, Mlp47QualificationInputs, Mlp47Weights,
};
use crate::vllm_cuda_qualification::canonical47_projection::component::ConnectedProjectionPrograms;
use crate::vllm_cuda_qualification::post_attention_projection::OutPrograms;
use crate::vllm_cuda_qualification::qwen2b_mlp::{Qwen2bMlpPrograms, Qwen2bMlpQualificationInputs};
use zerogravity_core::accelerator::program::{Vllm08Kind, Vllm08Operator, Vllm08Projection};
pub(super) enum ComponentProjections {
    Qwen2b(Arc<ProjectionPrograms>, Option<Arc<OutPrograms>>),
    Canonical47(Arc<ConnectedProjectionPrograms>, Option<Arc<OutPrograms>>),
    Canonical47Mlp(Arc<Mlp47Programs>),
    Qwen2bMlp(Arc<Qwen2bMlpPrograms>),
}
impl ComponentProjections {
    pub(super) fn prepare(
        values: Arc<ComponentValues>,
        service: &Arc<ResidentService>,
        inputs: CubinQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let recipe = values.recipe();
        if recipe.cut().includes_mlp() {
            return Err(Error::Unsupported(
                "MLP requires its separate exact activation/source factory".into(),
            ));
        }
        let value=match (recipe.kind(),recipe.cut()) {
            (_,ComponentCut::Mlp | ComponentCut::Qwen2bMlp)=>return Err(Error::Unsupported("MLP inputs absent".into())),
            (ComponentKind::Qwen2b,ComponentCut::Gated)=>Self::Qwen2b(crate::vllm_cuda_qualification::qwen2b_component::prepare_projections(recipe.clone(),service,inputs,init,cancelled)?,None),
            (ComponentKind::Canonical47(_),ComponentCut::Gated)=>Self::Canonical47(crate::vllm_cuda_qualification::canonical47_projection::component::prepare_projections(recipe.clone(),service,inputs,init,cancelled)?,None),
            (ComponentKind::Qwen2b,ComponentCut::PostAttention)=>{let (p,o)=crate::vllm_cuda_qualification::qwen2b_component::prepare_post_attention_projections(values,service,inputs,init,cancelled)?;Self::Qwen2b(p,Some(o))},
            (ComponentKind::Canonical47(_),ComponentCut::PostAttention)=>{let (p,o)=crate::vllm_cuda_qualification::canonical47_projection::component::prepare_post_attention_projections(values,service,inputs,init,cancelled)?;Self::Canonical47(p,Some(o))},
        };
        Ok(Arc::new(value))
    }
    pub(super) fn prepare_mlp(
        values: Arc<ComponentValues>,
        weights: Arc<Mlp47Weights>,
        inputs: Mlp47QualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        if values.recipe().cut() != ComponentCut::Mlp {
            return Err(Error::Unsupported(
                "new MLP factory requires exact eleven-node cut".into(),
            ));
        }
        Ok(Arc::new(Self::Canonical47Mlp(Mlp47Programs::prepare(
            values, weights, inputs, init, cancelled,
        )?)))
    }
    pub(super) fn prepare_qwen2b_mlp(
        values: Arc<ComponentValues>,
        weights: Arc<Qwen2bMlpWeights>,
        inputs: Qwen2bMlpQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        if values.recipe().cut() != ComponentCut::Qwen2bMlp
            || values.recipe().kind() != ComponentKind::Qwen2b
            || values.recipe().qwen2b_arithmetic()
                != Some(crate::vllm_cuda_gdn::Qwen2bGdnArithmetic::OwnedSolve4Recompute3)
        {
            return Err(Error::Unsupported(
                "explicit2B eleven-node source/native cut required".into(),
            ));
        }
        weights.require_record(values.recipe().record(), values.service())?;
        Ok(Arc::new(Self::Qwen2bMlp(Qwen2bMlpPrograms::prepare(
            values, weights, inputs, init, cancelled,
        )?)))
    }
    pub(super) fn arm_qwen2b_down1_fault(&self, cancel: bool) -> Result<()> {
        match self {
            Self::Qwen2bMlp(p) => p.arm_down1_fault(cancel),
            _ => Err(Error::Unsupported(
                "only exact2B Down main group fault".into(),
            )),
        }
    }
    pub(super) fn qwen2b_native_launches(&self) -> Result<usize> {
        match self {
            Self::Qwen2bMlp(p) => Ok(p.native_launches()),
            _ => Err(Error::Unsupported("only actual2B native metadata".into())),
        }
    }
    pub(super) fn down1_enqueues(&self) -> Result<usize> {
        match self {
            Self::Qwen2bMlp(p) => Ok(p.down1_enqueues()),
            _ => Err(Error::Unsupported("only actual2B Down metadata".into())),
        }
    }
    pub(super) fn down1_output_unpromoted(&self) -> Result<bool> {
        match self {
            Self::Qwen2bMlp(p) => p.down1_output_unpromoted(),
            _ => Err(Error::Unsupported(
                "only actual2B Down output metadata".into(),
            )),
        }
    }
    pub(super) fn accesses(
        &self,
        op: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        if let Self::Canonical47Mlp(p) = self {
            return p.accesses(op, input, output, workspace);
        }
        if let Self::Qwen2bMlp(p) = self {
            return p.accesses(op, input, output, workspace);
        }
        if op.kind() == Vllm08Kind::Projection(Vllm08Projection::GdnOut) {
            self.out()?.require_operator(op, input, output)?;
            return self.out()?.accesses();
        }
        match self {
            Self::Qwen2b(p, _) => p.accesses(op, input, output, workspace),
            Self::Canonical47(p, _) => p.accesses(op, input, output, workspace),
            Self::Canonical47Mlp(p) => p.accesses(op, input, output, workspace),
            Self::Qwen2bMlp(p) => p.accesses(op, input, output, workspace),
        }
    }
    pub(super) fn enqueue(
        &self,
        op: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        if let Self::Canonical47Mlp(p) = self {
            return p.enqueue(op, input, output, workspace, cancelled);
        }
        if let Self::Qwen2bMlp(p) = self {
            return p.enqueue(op, input, output, workspace, cancelled);
        }
        if op.kind() == Vllm08Kind::Projection(Vllm08Projection::GdnOut) {
            self.out()?.require_operator(op, input, output)?;
            return self.out()?.enqueue(cancelled);
        }
        match self {
            Self::Qwen2b(p, _) => p.enqueue(op, input, output, workspace, cancelled),
            Self::Canonical47(p, _) => p.enqueue(op, input, output, workspace, cancelled),
            Self::Canonical47Mlp(p) => p.enqueue(op, input, output, workspace, cancelled),
            Self::Qwen2bMlp(p) => p.enqueue(op, input, output, workspace, cancelled),
        }
    }
    pub(super) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        let mut weights = match self {
            Self::Qwen2b(p, _) => p.immutable_weights(),
            Self::Canonical47(p, _) => p.immutable_weights(),
            Self::Canonical47Mlp(p) => p.immutable_weights(),
            Self::Qwen2bMlp(p) => p.immutable_weights(),
        };
        if let Ok(out) = self.out() {
            weights.extend(out.immutable_weights());
        }
        weights
    }
    fn out(&self) -> Result<&Arc<OutPrograms>> {
        match self {
            Self::Qwen2b(_, Some(out)) | Self::Canonical47(_, Some(out)) => Ok(out),
            _ => Err(Error::Unsupported(
                "six-node cut has no Out producer".into(),
            )),
        }
    }
    pub(super) fn retained_post_attention_values(&self, values: &Arc<ComponentValues>) -> bool {
        match self {
            Self::Canonical47Mlp(p) => p.retained_values(values),
            Self::Qwen2bMlp(p) => p.retained_values(values),
            _ => self.out().is_ok_and(|out| out.retained_values(values)),
        }
    }
    pub(super) unsafe fn retire_after_physical_completion(&self) -> Result<()> {
        unsafe {
            if let Ok(out) = self.out() {
                out.retire_kernels_after_physical_completion()?;
            }
            match self {
                Self::Qwen2b(p, _) => p.retire_after_physical_completion(),
                Self::Canonical47(p, _) => p.retire_after_physical_completion(),
                Self::Canonical47Mlp(p) => p.retire_after_physical_completion(),
                Self::Qwen2bMlp(p) => p.retire_after_physical_completion(),
            }
        }
    }
}
