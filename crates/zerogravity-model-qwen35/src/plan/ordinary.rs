//! Call-free ordinary original-HF Qwen plan metadata.
//! The shared core plan owns math/SSA/state effects only; source authentication
//! and family roles stay with the model, while roots/completion stay external.
use super::*;
mod geometry;
use crate::source::binding::{OrdinarySourceSeal, QwenMatrixRole, QwenParameterRole};
pub use geometry::OrdinaryQwenGeometry;

#[derive(Debug)]
struct OrdinaryQwenPlanOwner {
    source: OrdinarySourceSeal,
    plan: Arc<ForwardPlan>,
}

pub const ORDINARY_SOURCE_MAX_CHUNK_ROWS: usize = 64;
pub const ORDINARY_SOURCE_MIN_CONTEXT: usize = 512;
pub const ORDINARY_SOURCE_MAX_CONTEXT: usize = 32_768;
// Existing CUDA callers retain the same bounds and source plan.
pub use ORDINARY_SOURCE_MAX_CHUNK_ROWS as ORDINARY_CUDA_MAX_CHUNK_ROWS;
pub use ORDINARY_SOURCE_MAX_CONTEXT as ORDINARY_CUDA_MAX_CONTEXT;
pub use ORDINARY_SOURCE_MIN_CONTEXT as ORDINARY_CUDA_MIN_CONTEXT;

#[derive(Clone, Debug)]
pub struct OrdinaryQwenPlan {
    owner: Arc<OrdinaryQwenPlanOwner>,
    profile: OrdinaryQwenGeometry,
    boundaries: Arc<[[ValueId; 2]]>,
    normalized: ValueId,
    logits: ValueId,
}
impl OrdinaryQwenPlan {
    pub fn plan(&self) -> &Arc<ForwardPlan> {
        &self.owner.plan
    }
    /// Authenticated source capability retained with the static plan.
    pub fn source_seal(&self) -> &OrdinarySourceSeal {
        &self.owner.source
    }
    /// Exact immutable source backing for GPU preparation and validation.
    pub fn source(&self) -> &zerogravity_artifacts::source_weights::VerifiedTensorSource {
        self.owner.source.source()
    }
    pub fn model_profile(&self) -> OrdinaryQwenGeometry {
        self.profile
    }
    pub fn context_capacity_tokens(&self) -> usize {
        self.owner.plan.prefix_capacity()
    }
    pub fn max_chunk_rows(&self) -> usize {
        ORDINARY_SOURCE_MAX_CHUNK_ROWS
    }
    pub fn boundaries(&self) -> &[[ValueId; 2]] {
        &self.boundaries
    }
    pub fn normalized(&self) -> ValueId {
        self.normalized
    }
    pub fn logits(&self) -> ValueId {
        self.logits
    }
    /// Bind request data without cloning/rebinding the plan or minting a root.
    pub fn bind_invocation(
        &self,
        mode: SourceCallMode,
        base: usize,
        ids: Arc<[u32]>,
        emit_logits: bool,
    ) -> Result<OrdinaryQwenInvocation> {
        require_source_phase(
            self.source_seal().numerical_contract(),
            mode,
            base,
            ids.len(),
        )?;
        let invocation = SourceInvocation::new(
            TextCallGeometry::new(
                self.owner.plan.source(),
                self.profile.vocabulary_size(),
                self.context_capacity_tokens(),
            )?,
            base,
            ids,
            emit_logits,
        )?;
        let bound = self.owner.plan.bind_source_invocation(invocation)?;
        // Phase is explicit; never infer cached decode from row count or base.
        bound.invocation().source_call(mode)?;
        Ok(OrdinaryQwenInvocation {
            bound,
            owner: self.owner.clone(),
            profile: self.profile,
            mode,
        })
    }
}

#[derive(Clone, Debug)]
pub struct OrdinaryQwenInvocation {
    bound: BoundSourceInvocation,
    owner: Arc<OrdinaryQwenPlanOwner>,
    profile: OrdinaryQwenGeometry,
    mode: SourceCallMode,
}
impl OrdinaryQwenInvocation {
    /// Borrow the original core binding; providers materialize it under admission.
    pub fn bound_invocation(&self) -> &BoundSourceInvocation {
        &self.bound
    }
    pub fn plan(&self) -> &Arc<ForwardPlan> {
        self.bound.plan()
    }
    pub fn source_seal(&self) -> &OrdinarySourceSeal {
        &self.owner.source
    }
    pub fn source(&self) -> &zerogravity_artifacts::source_weights::VerifiedTensorSource {
        self.owner.source.source()
    }
    pub fn model_profile(&self) -> OrdinaryQwenGeometry {
        self.profile
    }
    pub fn mode(&self) -> SourceCallMode {
        self.mode
    }
    pub fn source_call(&self) -> Result<SourceCall> {
        self.bound.invocation().source_call(self.mode)
    }
    pub fn materialize_source_plan(&self) -> Result<ForwardPlan> {
        self.bound.materialize_source_plan(self.mode)
    }
    pub fn base(&self) -> usize {
        self.bound.invocation().base()
    }
    pub fn ids(&self) -> &[u32] {
        self.bound.invocation().ids()
    }
    pub fn rows(&self) -> usize {
        self.bound.invocation().rows()
    }
    pub fn frontier(&self) -> usize {
        self.bound.invocation().frontier()
    }
    pub fn emit_logits(&self) -> bool {
        self.bound.invocation().emit_logits()
    }
    pub fn active_nodes(&self) -> &[PlanNode] {
        self.bound.active_nodes()
    }
}

fn require_source_phase(
    numerical: NumericalContract,
    mode: SourceCallMode,
    base: usize,
    rows: usize,
) -> Result<()> {
    // Original HF takes cached recurrence for every initialized single-row call.
    // Refuse a contradictory declaration; never silently rewrite the phase.
    if numerical == NumericalContract::TorchCpuAvx2Bf16MixedV1
        && base > 0
        && rows == 1
        && mode != SourceCallMode::Decode
    {
        return Err(Error::Unsupported(
            "original CPU-source cached row requires explicit decode".into(),
        ));
    }
    Ok(())
}

fn domains(profile: OrdinaryQwenGeometry) -> Result<Vec<StateSlot>> {
    hybrid_state_domains((0..profile.layer_count()).map(|index| index % 4 == 3))
}

struct RecordedOrdinaryPlan {
    plan: Arc<ForwardPlan>,
    boundaries: Arc<[[ValueId; 2]]>,
    normalized: ValueId,
    logits: ValueId,
}

pub(crate) fn build_ordinary_plan(
    source: &OrdinarySourceSeal,
    context_capacity_tokens: usize,
    matrix: impl FnMut(QwenMatrixRole) -> Result<SourceTensor>,
    parameter: impl FnMut(QwenParameterRole) -> Result<SourceParameter>,
) -> Result<OrdinaryQwenPlan> {
    let profile = source.model_profile();
    let recorded = record_ordinary_plan(
        source.source_identity(),
        profile,
        source.numerical_contract(),
        context_capacity_tokens,
        matrix,
        parameter,
    )?;
    Ok(OrdinaryQwenPlan {
        owner: Arc::new(OrdinaryQwenPlanOwner {
            source: source.clone(),
            plan: recorded.plan,
        }),
        profile,
        boundaries: recorded.boundaries,
        normalized: recorded.normalized,
        logits: recorded.logits,
    })
}

fn record_ordinary_plan(
    identity: SourceIdentity,
    profile: impl Into<OrdinaryQwenGeometry>,
    numerical: NumericalContract,
    context_capacity_tokens: usize,
    mut matrix: impl FnMut(QwenMatrixRole) -> Result<SourceTensor>,
    mut parameter: impl FnMut(QwenParameterRole) -> Result<SourceParameter>,
) -> Result<RecordedOrdinaryPlan> {
    let profile = profile.into();
    if !(ORDINARY_SOURCE_MIN_CONTEXT..=ORDINARY_SOURCE_MAX_CONTEXT)
        .contains(&context_capacity_tokens)
    {
        return Err(Error::Unsupported(
            "ordinary source context capacity must be 512..32768".into(),
        ));
    }
    let geometry = SourcePlanGeometry::new(
        identity,
        profile.vocabulary_size(),
        context_capacity_tokens,
        ORDINARY_SOURCE_MAX_CHUNK_ROWS,
    )?;
    let mut recorder = OrdinaryRecorder {
        profile,
        numerical,
        geometry,
        builder: CorePlanBuilder::new_source_template_with_contract(
            geometry,
            domains(profile)?,
            numerical,
        )?,
        matrix: &mut matrix,
        parameter: &mut parameter,
        boundaries: Vec::with_capacity(profile.layer_count()),
        normalized: None,
    };
    // The source seal authenticated this model's complete hybrid schedule.
    let layers = (0..profile.layer_count())
        .map(|index| {
            if index % 4 == 3 {
                crate::config::LayerKind::FullAttention
            } else {
                crate::config::LayerKind::LinearAttention
            }
        })
        .collect::<Vec<_>>();
    let logits = crate::hybrid_decoder::compose(&layers, &mut recorder)?;
    let normalized = recorder
        .normalized
        .ok_or_else(|| Error::State("ordinary final normalization absent".into()))?;
    if recorder.boundaries.len() != profile.layer_count() {
        return Err(Error::State(
            "ordinary plan complete source layer boundaries".into(),
        ));
    }
    let plan = recorder.builder.finish_source_template(logits)?;
    if plan.required_domains().len() != profile.state_domains()
        || plan.proposed_domains().len() != profile.state_domains()
        || plan.logits() != Some(logits.id)
        || plan.frontier_delta() != 0
        || plan.source_call().is_some()
        || plan
            .nodes()
            .iter()
            .any(|node| !matches!(node.operation, PlanOperation::SourceOperatorTemplate { .. }))
    {
        return Err(Error::State(
            "ordinary plan static source/state/logits structure".into(),
        ));
    }
    Ok(RecordedOrdinaryPlan {
        plan: Arc::new(plan),
        boundaries: recorder
            .boundaries
            .into_iter()
            .map(|[a, b]| [a.id, b.id])
            .collect(),
        normalized: normalized.id,
        logits: logits.id,
    })
}

type MatrixResolver<'a> = &'a mut dyn FnMut(QwenMatrixRole) -> Result<SourceTensor>;
type ParameterResolver<'a> = &'a mut dyn FnMut(QwenParameterRole) -> Result<SourceParameter>;
struct OrdinaryRecorder<'a> {
    profile: OrdinaryQwenGeometry,
    numerical: NumericalContract,
    geometry: SourcePlanGeometry,
    builder: CorePlanBuilder,
    matrix: MatrixResolver<'a>,
    parameter: ParameterResolver<'a>,
    boundaries: Vec<[PlanValue; 2]>,
    normalized: Option<PlanValue>,
}
impl OrdinaryRecorder<'_> {
    fn layer_kind(index: usize) -> bool {
        index % 4 == 3
    }
    fn template(
        &mut self,
        layer: Option<usize>,
        operation: QwenOperation,
    ) -> Result<SourceOperatorTemplate> {
        let (matrix_roles, parameter_roles) = roles(layer, operation);
        let matrices = matrix_roles
            .into_iter()
            .map(&mut self.matrix)
            .collect::<Result<Vec<_>>>()?;
        let parameters = parameter_roles
            .into_iter()
            .map(&mut self.parameter)
            .collect::<Result<Vec<_>>>()?;
        SourceOperatorTemplate::new(
            self.geometry,
            if operation == QwenOperation::Projection(QwenProjection::Head) {
                SourceRowSelection::LastActive
            } else {
                SourceRowSelection::AllActive
            },
            mathematical_kind(self.profile, self.numerical, layer, operation)?,
            self.numerical,
            matrices,
            parameters,
        )
    }
    fn apply(
        &mut self,
        layer: Option<usize>,
        operation: QwenOperation,
        inputs: &[PlanValue],
    ) -> Result<Vec<PlanValue>> {
        let descriptor = self.template(layer, operation)?;
        self.builder
            .record_source_template("qwen.ordinary_cuda", &descriptor, inputs)
    }
    fn one(
        &mut self,
        layer: Option<usize>,
        operation: QwenOperation,
        inputs: &[PlanValue],
    ) -> Result<PlanValue> {
        let [value] = self
            .apply(layer, operation, inputs)?
            .try_into()
            .map_err(|_| Error::State("ordinary Qwen single output".into()))?;
        Ok(value)
    }
    fn two(
        &mut self,
        layer: Option<usize>,
        operation: QwenOperation,
        inputs: &[PlanValue],
    ) -> Result<[PlanValue; 2]> {
        self.apply(layer, operation, inputs)?
            .try_into()
            .map_err(|_| Error::State("ordinary Qwen dual output".into()))
    }
    fn project(
        &mut self,
        layer: Option<usize>,
        projection: QwenProjection,
        input: PlanValue,
    ) -> Result<PlanValue> {
        self.one(layer, QwenOperation::Projection(projection), &[input])
    }
}

impl crate::hybrid_decoder::HybridDecoder for OrdinaryRecorder<'_> {
    type Value = PlanValue;
    type GdnInputs = [PlanValue; 2];
    type AttentionInputs = PlanValue;
    type FfnInputs = PlanValue;

    fn embedding(&mut self) -> Result<PlanValue> {
        let descriptor = SourceOperatorTemplate::new(
            self.geometry,
            SourceRowSelection::AllActive,
            SourceOperatorKind::EmbeddingRows,
            self.numerical,
            vec![(self.matrix)(QwenMatrixRole::Embedding)?],
            vec![],
        )?;
        let [value] = self
            .builder
            .record_source_template("qwen.ordinary_cuda.embedding", &descriptor, &[])?
            .try_into()
            .map_err(|_| Error::State("ordinary embedding output".into()))?;
        Ok(value)
    }
    fn input_norm(
        &mut self,
        index: usize,
        branch: PlanValue,
        residual: Option<PlanValue>,
    ) -> Result<(PlanValue, PlanValue)> {
        if index == 0 {
            if residual.is_some() {
                return Err(Error::State("ordinary initial residual".into()));
            }
            Ok((
                self.one(Some(index), QwenOperation::InputRms, &[branch])?,
                branch,
            ))
        } else {
            let residual =
                residual.ok_or_else(|| Error::State("ordinary residual absent".into()))?;
            let [normal, carried] = self.two(
                Some(index),
                QwenOperation::ResidualRms(QwenRmsSite::Input),
                &[branch, residual],
            )?;
            Ok((normal, carried))
        }
    }
    fn gdn_inputs(&mut self, index: usize, normal: &PlanValue) -> Result<Self::GdnInputs> {
        Ok([
            self.project(Some(index), QwenProjection::Qkvz, *normal)?,
            self.project(Some(index), QwenProjection::Ba, *normal)?,
        ])
    }
    fn gdn_evolve(&mut self, index: usize, [packed, ba]: Self::GdnInputs) -> Result<PlanValue> {
        let [activated, z] = self.two(Some(index), QwenOperation::GdnConv, &[packed, ba])?;
        self.one(
            Some(index),
            QwenOperation::GdnRecurrent,
            &[activated, z, ba],
        )
    }
    fn attention_inputs(&mut self, index: usize, normal: &PlanValue) -> Result<PlanValue> {
        self.project(Some(index), QwenProjection::AttentionQkv, *normal)
    }
    fn attention_evolve(&mut self, index: usize, packed: PlanValue) -> Result<PlanValue> {
        let outputs = self.apply(Some(index), QwenOperation::AttentionQk, &[packed])?;
        let [query, key, gate]: [PlanValue; 3] = outputs
            .try_into()
            .map_err(|_| Error::State("ordinary attention output arity".into()))?;
        if !self
            .apply(Some(index), QwenOperation::KvAppend, &[key, packed])?
            .is_empty()
        {
            return Err(Error::State("ordinary KV append output".into()));
        }
        let value = self.one(Some(index), QwenOperation::FlashAttention, &[query])?;
        let gate = self.one(Some(index), QwenOperation::Sigmoid, &[gate])?;
        self.one(Some(index), QwenOperation::Multiply, &[value, gate])
    }
    fn attention_output(
        &mut self,
        index: usize,
        kind: crate::config::LayerKind,
        value: PlanValue,
    ) -> Result<PlanValue> {
        let expected_attention = Self::layer_kind(index);
        if expected_attention != (kind == crate::config::LayerKind::FullAttention) {
            return Err(Error::State("ordinary layer schedule".into()));
        }
        self.project(
            Some(index),
            if expected_attention {
                QwenProjection::AttentionOut
            } else {
                QwenProjection::GdnOut
            },
            value,
        )
    }
    fn post_attention_norm(
        &mut self,
        index: usize,
        branch: PlanValue,
        carried: PlanValue,
    ) -> Result<(PlanValue, PlanValue)> {
        let [normal, carried] = self.two(
            Some(index),
            QwenOperation::ResidualRms(QwenRmsSite::PostAttention),
            &[branch, carried],
        )?;
        Ok((normal, carried))
    }
    fn ffn_inputs(&mut self, index: usize, normal: &PlanValue) -> Result<Self::FfnInputs> {
        self.project(Some(index), QwenProjection::GateUp, *normal)
    }
    fn ffn_activate(&mut self, index: usize, input: Self::FfnInputs) -> Result<PlanValue> {
        self.one(Some(index), QwenOperation::SiluAndMul, &[input])
    }
    fn ffn_output(&mut self, index: usize, input: PlanValue) -> Result<PlanValue> {
        self.project(Some(index), QwenProjection::Down, input)
    }
    fn finish_layer(
        &mut self,
        _: usize,
        branch: PlanValue,
        carried: PlanValue,
    ) -> Result<(PlanValue, Option<PlanValue>)> {
        self.boundaries.push([branch, carried]);
        Ok((branch, Some(carried)))
    }
    fn finish_model(
        &mut self,
        branch: PlanValue,
        residual: Option<PlanValue>,
    ) -> Result<PlanValue> {
        let residual = residual.ok_or_else(|| Error::State("ordinary final residual".into()))?;
        let [normalized, _] = self.two(
            None,
            QwenOperation::ResidualRms(QwenRmsSite::Final),
            &[branch, residual],
        )?;
        self.normalized = Some(normalized);
        self.project(None, QwenProjection::Head, normalized)
    }
}

fn mathematical_kind(
    profile: OrdinaryQwenGeometry,
    numerical: NumericalContract,
    layer: Option<usize>,
    operation: QwenOperation,
) -> Result<SourceOperatorKind> {
    let index = layer.unwrap_or(0);
    let rms = || SourceRmsGeometry {
        width: profile.hidden_size(),
        epsilon: 1e-6,
        gamma: SourceGamma::OnePlus,
    };
    Ok(match operation {
        QwenOperation::InputRms => SourceOperatorKind::Rms(rms()),
        QwenOperation::ResidualRms(_) => SourceOperatorKind::ResidualRms(rms()),
        QwenOperation::Projection(_) => SourceOperatorKind::Projection { selected_row: None },
        QwenOperation::GdnConv => SourceOperatorKind::CausalConvSplit {
            domain: ConvDomain(index),
            geometry: SourceConvGeometry {
                qkv_channels: profile.conv_channels(),
                z_channels: profile.recurrent_width(),
                coefficient_channels: 2 * profile.value_heads(),
                kernel: 4,
            },
        },
        QwenOperation::GdnRecurrent => SourceOperatorKind::GatedDelta {
            domain: RecurrentDomain(index),
            geometry: SourceGdnGeometry {
                key_heads: profile.key_heads(),
                value_heads: profile.value_heads(),
                key_width: 128,
                value_width: 128,
                epsilon: 1e-6,
                contract: if numerical == NumericalContract::TorchCpuAvx2Bf16MixedV1 {
                    GdnContract::TorchCpuAvx2L2SiluBf16V1
                } else {
                    GdnContract::ReferenceCudaSm86L2SiluBf16V1
                },
            },
        },
        QwenOperation::AttentionQk => {
            SourceOperatorKind::QkRmsRotaryGated(SourceQkRotaryGeometry {
                attention: AttentionGeometry {
                    width: 256,
                    query_heads: profile.query_heads(),
                    kv_heads: profile.kv_heads(),
                },
                rotary_dimensions: 64,
                theta: 10_000_000.0,
                epsilon: 1e-6,
            })
        }
        QwenOperation::KvAppend => SourceOperatorKind::KvAppend {
            domain: KvDomain(index),
            key_width: profile.kv_width(),
            packed_value: TensorView {
                offset: 2 * profile.query_width() + profile.kv_width(),
                rows: ORDINARY_SOURCE_MAX_CHUNK_ROWS,
                width: profile.kv_width(),
                row_stride: 2 * (profile.query_width() + profile.kv_width()),
            },
        },
        QwenOperation::FlashAttention => SourceOperatorKind::FlashAttention {
            domain: KvDomain(index),
            geometry: AttentionGeometry {
                width: 256,
                query_heads: profile.query_heads(),
                kv_heads: profile.kv_heads(),
            },
            scale: 0.0625,
            mask: SourceAttentionMask::Causal,
        },
        QwenOperation::Sigmoid => SourceOperatorKind::Sigmoid {
            width: profile.query_width(),
        },
        QwenOperation::Multiply => SourceOperatorKind::Multiply {
            width: profile.query_width(),
        },
        QwenOperation::SiluAndMul => SourceOperatorKind::SiluAndMul {
            width: profile.intermediate_size(),
        },
    })
}

fn roles(
    layer: Option<usize>,
    operation: QwenOperation,
) -> (Vec<QwenMatrixRole>, Vec<QwenParameterRole>) {
    let index = layer.unwrap_or(0);
    use crate::hybrid_decoder::{ParameterGroup as P, ProjectionGroup as M};
    let matrices = match operation {
        QwenOperation::Projection(projection) => match projection {
            QwenProjection::Qkvz => M::GdnInput.roles(index),
            QwenProjection::Ba => M::GdnCoefficients.roles(index),
            QwenProjection::GdnOut => M::GdnOutput.roles(index),
            QwenProjection::AttentionQkv => M::AttentionInput.roles(index),
            QwenProjection::AttentionOut => M::AttentionOutput.roles(index),
            QwenProjection::GateUp => M::FfnInput.roles(index),
            QwenProjection::Down => M::FfnOutput.roles(index),
            QwenProjection::Head => M::Head.roles(index),
        },
        _ => Vec::new(),
    };
    let parameters = match operation {
        QwenOperation::InputRms | QwenOperation::ResidualRms(QwenRmsSite::Input) => {
            P::InputNorm.roles(index)
        }
        QwenOperation::ResidualRms(QwenRmsSite::PostAttention) => P::FfnNorm.roles(index),
        QwenOperation::ResidualRms(QwenRmsSite::Final) => P::FinalNorm.roles(index),
        QwenOperation::GdnConv => P::GdnConvolution.roles(index),
        QwenOperation::GdnRecurrent => P::GdnRecurrent.roles(index),
        QwenOperation::AttentionQk => P::AttentionNorm.roles(index),
        _ => Vec::new(),
    };
    (matrices, parameters)
}

#[cfg(feature = "diagnostic-cpu-checkpoint")]
pub fn diagnostic_ordinary_plan(
    profile: QwenModelVariant,
    context_capacity_tokens: usize,
) -> Result<Arc<ForwardPlan>> {
    let source = SourceIdentity::for_new_allocation()?;
    let profile = OrdinaryQwenGeometry::from(profile);
    let recorded = record_ordinary_plan(
        source,
        profile,
        NumericalContract::CudaSm86Bf16ReferenceV1,
        context_capacity_tokens,
        |role| Ok(diagnostic_matrix(source, profile, role)),
        |role| Ok(diagnostic_parameter(source, profile, role)),
    )?;
    Ok(recorded.plan)
}

#[cfg(feature = "diagnostic-cpu-checkpoint")]
fn diagnostic_matrix(
    source: SourceIdentity,
    profile: OrdinaryQwenGeometry,
    role: QwenMatrixRole,
) -> SourceTensor {
    let hidden = profile.hidden_size();
    let intermediate = profile.intermediate_size();
    let (name, rows, columns) = match role {
        QwenMatrixRole::Embedding | QwenMatrixRole::Output => (
            "model.language_model.embed_tokens.weight".into(),
            profile.vocabulary_size(),
            hidden,
        ),
        QwenMatrixRole::FfnGate(index) => (format!("layer.{index}.gate"), intermediate, hidden),
        QwenMatrixRole::FfnUp(index) => (format!("layer.{index}.up"), intermediate, hidden),
        QwenMatrixRole::FfnDown(index) => (format!("layer.{index}.down"), hidden, intermediate),
        QwenMatrixRole::GdnQkv(index) => (
            format!("layer.{index}.qkv"),
            profile.conv_channels(),
            hidden,
        ),
        QwenMatrixRole::GdnGate(index) => (
            format!("layer.{index}.z"),
            profile.recurrent_width(),
            hidden,
        ),
        QwenMatrixRole::GdnAlpha(index) => {
            (format!("layer.{index}.a"), profile.value_heads(), hidden)
        }
        QwenMatrixRole::GdnBeta(index) => {
            (format!("layer.{index}.b"), profile.value_heads(), hidden)
        }
        QwenMatrixRole::GdnOutput(index) => (
            format!("layer.{index}.gdn_out"),
            hidden,
            profile.recurrent_width(),
        ),
        QwenMatrixRole::Query(index) => (
            format!("layer.{index}.q"),
            2 * profile.query_width(),
            hidden,
        ),
        QwenMatrixRole::Key(index) => (format!("layer.{index}.k"), profile.kv_width(), hidden),
        QwenMatrixRole::Value(index) => (format!("layer.{index}.v"), profile.kv_width(), hidden),
        QwenMatrixRole::AttentionOutput(index) => (
            format!("layer.{index}.attn_out"),
            hidden,
            profile.query_width(),
        ),
    };
    SourceTensor {
        source,
        name,
        kind: StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
        axes: MatrixAxes::OutputInput,
        columns,
        rows,
    }
}

#[cfg(feature = "diagnostic-cpu-checkpoint")]
fn diagnostic_parameter(
    source: SourceIdentity,
    profile: OrdinaryQwenGeometry,
    role: QwenParameterRole,
) -> SourceParameter {
    let hidden = profile.hidden_size();
    let (name, kind, shape) = match role {
        QwenParameterRole::FinalNorm => (
            "final_norm".into(),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![hidden as u64],
        ),
        QwenParameterRole::InputNorm(index) => (
            format!("layer.{index}.input_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![hidden as u64],
        ),
        QwenParameterRole::FfnNorm(index) => (
            format!("layer.{index}.ffn_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![hidden as u64],
        ),
        QwenParameterRole::GdnConvolution(index) => (
            format!("layer.{index}.conv"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![profile.conv_channels() as u64, 1, 4],
        ),
        QwenParameterRole::GdnNorm(index) => (
            format!("layer.{index}.gdn_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32),
            vec![128],
        ),
        QwenParameterRole::GdnA(index) => (
            format!("layer.{index}.A_log"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32),
            vec![profile.value_heads() as u64],
        ),
        QwenParameterRole::GdnDt(index) => (
            format!("layer.{index}.dt"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![profile.value_heads() as u64],
        ),
        QwenParameterRole::QueryNorm(index) => (
            format!("layer.{index}.q_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![256],
        ),
        QwenParameterRole::KeyNorm(index) => (
            format!("layer.{index}.k_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![256],
        ),
    };
    let element_bytes =
        if kind == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16) {
            2
        } else {
            4
        };
    let bytes = shape.iter().map(|n| *n as usize).product::<usize>() * element_bytes;
    SourceParameter {
        source,
        name,
        kind,
        shape,
        bytes,
    }
}

#[cfg(all(test, feature = "cpu"))]
pub(crate) fn build_test_ordinary_plan(
    source: &OrdinarySourceSeal,
    context_capacity_tokens: usize,
) -> Result<OrdinaryQwenPlan> {
    let identity = source.source_identity();
    let profile = source.model_profile();
    build_ordinary_plan(
        source,
        context_capacity_tokens,
        |role| Ok(tests::matrix(identity, profile, role)),
        |role| Ok(tests::parameter(identity, profile, role)),
    )
}

#[cfg(test)]
#[path = "ordinary/tests.rs"]
mod tests;
