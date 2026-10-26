//! Original source recipe, through the existing family layer traversal.
use super::*;
use crate::plan::{
    QwenEmbedding, QwenLayer, QwenOperation, QwenOperator, QwenPlan, QwenPlanBuilder,
    QwenProjection, QwenRmsSite, QwenSourceCall, QwenSourceProfile,
};
use zerogravity_core::accelerator::program::{PlanValue, SourceCall};

impl OriginalWindowModelSource {
    /// Authenticate an original finite reference call without recording a graph.
    pub fn bind_reference_call(&self, call: &QwenSourceCall) -> Result<SourceCall> {
        self.require_small_model_source()?;
        call.validate()?;
        call.request().reference_phase()?;
        let actual = call.request().source_call()?;
        if call.model_profile() != QwenSourceProfile::Qwen08
            || call.source() != self.source_identity()
            || call.request().layer().index() != 0
            || actual.rows().capacity() != 512
            || actual.rows().vocabulary() != 248320
        {
            return Err(Error::State("reference call source/profile/layer/extent differs".into()));
        }
        Ok(actual)
    }
    /// Closed source/config metadata before any device/service/learned upload.
    /// Config pin also binds original mrope fields not exposed by old geometry.
    pub fn require_small_model_source(&self) -> Result<()> {
        use crate::config::{Family, LayerKind};
        use crate::source::safetensors::GdnOutputGate;
        let c = self.owner().config();
        let g = c.geometry();
        if self.owner().config_sha256()
            != "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
            || self.owner().verified_index_sha256()
                != "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e"
            || g.family != Family::QwenHybrid
            || g.hidden_size != 1024
            || g.vocab_size != 248320
            || g.layers.len() != 24
            || g.head_dim != 256
            || g.num_attention_heads != 8
            || g.num_key_value_heads != 2
            || g.recurrent_key_heads != Some(16)
            || g.recurrent_value_heads != Some(16)
            || c.intermediate_size() != 3584
            || c.linear_key_head_dim() != 128
            || c.linear_value_head_dim() != 128
            || c.linear_conv_kernel_dim() != 4
            || c.gdn_output_gate() != GdnOutputGate::Silu
            || !c.attention_output_gate()
            || !c.tie_word_embeddings()
            || c.rms_norm_eps() != 1e-6
            || c.rope_theta() != 10_000_000.0
            || c.rotary_dimensions() != 64
            || g.layers.iter().enumerate().any(|(i, kind)| {
                *kind
                    != if i % 4 == 3 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
            })
        {
            return Err(Error::Unsupported(
                "closed original .8B24 source/config/rotary geometry".into(),
            ));
        }
        let embedding = self.matrix_descriptor(QwenMatrixRole::Embedding)?;
        if self.matrix_descriptor(QwenMatrixRole::Output)? != embedding {
            return Err(Error::State(
                "original full model must share tied embedding/head descriptor".into(),
            ));
        }
        self.validate_matrix(QwenMatrixRole::Embedding, &embedding)?;
        self.validate_matrix(QwenMatrixRole::Output, &embedding)
    }

    /// Records arithmetic, all42 private effects and fullV. No Session/RootReady,
    /// execution eligibility or model publication follows from this record.
    pub fn record_small_model(&self, call: QwenSourceCall) -> Result<QwenPlan> {
        self.require_small_model_source()?;
        if call.model_profile() != QwenSourceProfile::Qwen08 {
            return Err(Error::Unsupported(
                "original .8B recording constructor profile".into(),
            ));
        }
        self.record_full_model_inner(call)
    }
    // Both sealed source profiles use this SAME family traversal/SSA recipe.
    pub(super) fn record_full_model_inner(&self, call: QwenSourceCall) -> Result<QwenPlan> {
        if call.source() != self.source_identity() {
            return Err(Error::State("full-model request/source issuer".into()));
        }
        let mut recorder = FullSourceRecorder {
            source: self,
            call: &call,
            builder: QwenPlanBuilder::new_model(call.clone())?,
            boundaries: Vec::with_capacity(self.geometry().layers.len()),
            normalized: None,
        };
        let logits = crate::hybrid_decoder::compose(&self.geometry().layers, &mut recorder)?;
        let normalized = recorder
            .normalized
            .ok_or_else(|| Error::State("full-model final norm absent".into()))?;
        recorder
            .builder
            .finish_model(recorder.boundaries, normalized, logits)
    }

    /// Exact family roles joined to the original authenticated source owner.
    pub fn validate_operator(&self, operator: &QwenOperator) -> Result<()> {
        validate_source_operator(
            self.source_identity(),
            operator,
            |role, descriptor| self.validate_matrix(role, descriptor),
            |role, descriptor| self.validate_parameter(role, descriptor),
        )
    }

    fn full_projection(
        &self,
        builder: &mut QwenPlanBuilder,
        call: &QwenSourceCall,
        layer: Option<QwenLayer>,
        kind: QwenProjection,
        input: PlanValue,
    ) -> Result<PlanValue> {
        self.full_one(
            builder,
            call,
            layer,
            QwenOperation::Projection(kind),
            &[input],
        )
    }
    fn full_one(
        &self,
        builder: &mut QwenPlanBuilder,
        call: &QwenSourceCall,
        layer: Option<QwenLayer>,
        kind: QwenOperation,
        inputs: &[PlanValue],
    ) -> Result<PlanValue> {
        let [output]: [PlanValue; 1] = self
            .full_apply(builder, call, layer, kind, inputs)?
            .try_into()
            .map_err(|_| Error::State("full-model exact single output".into()))?;
        Ok(output)
    }
    fn full_two(
        &self,
        builder: &mut QwenPlanBuilder,
        call: &QwenSourceCall,
        layer: Option<QwenLayer>,
        kind: QwenOperation,
        inputs: &[PlanValue],
    ) -> Result<[PlanValue; 2]> {
        self.full_apply(builder, call, layer, kind, inputs)?
            .try_into()
            .map_err(|_| Error::State("full-model exact dual output".into()))
    }
    fn full_apply(
        &self,
        builder: &mut QwenPlanBuilder,
        call: &QwenSourceCall,
        layer: Option<QwenLayer>,
        kind: QwenOperation,
        inputs: &[PlanValue],
    ) -> Result<Vec<PlanValue>> {
        let (roles, parameter_roles) = source_roles(layer, kind);
        let mut matrices = Vec::with_capacity(roles.len());
        for role in roles {
            let descriptor = self.matrix_descriptor(role)?;
            self.validate_matrix(role, &descriptor)?;
            matrices.push(descriptor);
        }
        let mut parameters = Vec::with_capacity(parameter_roles.len());
        for role in parameter_roles {
            let descriptor = self.parameter_descriptor(role)?;
            self.validate_parameter(role, &descriptor)?;
            parameters.push(descriptor);
        }
        builder.source_operator(
            &QwenOperator::new(call.clone(), layer, kind, matrices, parameters)?,
            inputs,
        )
    }
}

struct FullSourceRecorder<'a> {
    source: &'a OriginalWindowModelSource,
    call: &'a QwenSourceCall,
    builder: QwenPlanBuilder,
    boundaries: Vec<[PlanValue; 2]>,
    normalized: Option<PlanValue>,
}
impl FullSourceRecorder<'_> {
    fn one(
        &mut self,
        layer: Option<QwenLayer>,
        kind: QwenOperation,
        inputs: &[PlanValue],
    ) -> Result<PlanValue> {
        self.source
            .full_one(&mut self.builder, self.call, layer, kind, inputs)
    }
    fn two(
        &mut self,
        layer: Option<QwenLayer>,
        kind: QwenOperation,
        inputs: &[PlanValue],
    ) -> Result<[PlanValue; 2]> {
        self.source
            .full_two(&mut self.builder, self.call, layer, kind, inputs)
    }
    fn project(
        &mut self,
        layer: Option<QwenLayer>,
        kind: QwenProjection,
        input: PlanValue,
    ) -> Result<PlanValue> {
        self.source
            .full_projection(&mut self.builder, self.call, layer, kind, input)
    }
}
impl crate::hybrid_decoder::HybridDecoder for FullSourceRecorder<'_> {
    type Value = PlanValue;
    type GdnInputs = [PlanValue; 2];
    type AttentionInputs = PlanValue;
    type FfnInputs = PlanValue;
    fn embedding(&mut self) -> Result<PlanValue> {
        self.builder.source_embedding(&QwenEmbedding::for_call(
            self.source.matrix_descriptor(QwenMatrixRole::Embedding)?,
            self.call,
        )?)
    }
    fn input_norm(
        &mut self,
        index: usize,
        branch: PlanValue,
        residual: Option<PlanValue>,
    ) -> Result<(PlanValue, PlanValue)> {
        let layer = self.call.layer(index)?;
        if layer.is_attention()
            != (self.source.geometry().layers[index] == crate::config::LayerKind::FullAttention)
        {
            return Err(Error::State("original source layer schedule/type".into()));
        }
        if index == 0 {
            if residual.is_some() {
                return Err(Error::State(
                    "initial embedding has no pending residual".into(),
                ));
            }
            Ok((
                self.one(Some(layer), QwenOperation::InputRms, &[branch])?,
                branch,
            ))
        } else {
            let residual =
                residual.ok_or_else(|| Error::State("original carried residual absent".into()))?;
            let [normal, carried] = self.two(
                Some(layer),
                QwenOperation::ResidualRms(QwenRmsSite::Input),
                &[branch, residual],
            )?;
            Ok((normal, carried))
        }
    }
    fn gdn_inputs(&mut self, index: usize, normal: &PlanValue) -> Result<Self::GdnInputs> {
        let layer = Some(self.call.layer(index)?);
        let qkvz = self.project(layer, QwenProjection::Qkvz, *normal)?;
        let ba = self.project(layer, QwenProjection::Ba, *normal)?;
        Ok([qkvz, ba])
    }
    fn gdn_evolve(&mut self, index: usize, [packed, ba]: Self::GdnInputs) -> Result<PlanValue> {
        let layer = Some(self.call.layer(index)?);
        let [activated, z] = self.two(layer, QwenOperation::GdnConv, &[packed, ba])?;
        self.one(layer, QwenOperation::GdnRecurrent, &[activated, z, ba])
    }
    fn attention_inputs(&mut self, index: usize, normal: &PlanValue) -> Result<PlanValue> {
        self.project(
            Some(self.call.layer(index)?),
            QwenProjection::AttentionQkv,
            *normal,
        )
    }
    fn attention_evolve(&mut self, index: usize, packed: PlanValue) -> Result<PlanValue> {
        let layer = Some(self.call.layer(index)?);
        let outputs = self.source.full_apply(
            &mut self.builder,
            self.call,
            layer,
            QwenOperation::AttentionQk,
            &[packed],
        )?;
        let [q, k, gate]: [PlanValue; 3] = outputs
            .try_into()
            .map_err(|_| Error::State("attention Q/K/gate output arity".into()))?;
        if !self
            .source
            .full_apply(
                &mut self.builder,
                self.call,
                layer,
                QwenOperation::KvAppend,
                &[k, packed],
            )?
            .is_empty()
        {
            return Err(Error::State(
                "KV append is one private effect without public value".into(),
            ));
        }
        let fa = self.one(layer, QwenOperation::FlashAttention, &[q])?;
        let sigmoid = self.one(layer, QwenOperation::Sigmoid, &[gate])?;
        self.one(layer, QwenOperation::Multiply, &[fa, sigmoid])
    }
    fn attention_output(
        &mut self,
        index: usize,
        kind: crate::config::LayerKind,
        value: PlanValue,
    ) -> Result<PlanValue> {
        self.project(
            Some(self.call.layer(index)?),
            if kind == crate::config::LayerKind::FullAttention {
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
            Some(self.call.layer(index)?),
            QwenOperation::ResidualRms(QwenRmsSite::PostAttention),
            &[branch, carried],
        )?;
        Ok((normal, carried))
    }
    fn ffn_inputs(&mut self, index: usize, normal: &PlanValue) -> Result<PlanValue> {
        self.project(
            Some(self.call.layer(index)?),
            QwenProjection::GateUp,
            *normal,
        )
    }
    fn ffn_activate(&mut self, index: usize, input: PlanValue) -> Result<PlanValue> {
        self.one(
            Some(self.call.layer(index)?),
            QwenOperation::SiluAndMul,
            &[input],
        )
    }
    fn ffn_output(&mut self, index: usize, input: PlanValue) -> Result<PlanValue> {
        self.project(Some(self.call.layer(index)?), QwenProjection::Down, input)
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
        let residual = residual
            .ok_or_else(|| Error::State("final original carried residual absent".into()))?;
        let [normalized, _] = self.two(
            None,
            QwenOperation::ResidualRms(QwenRmsSite::Final),
            &[branch, residual],
        )?;
        self.normalized = Some(normalized);
        self.project(None, QwenProjection::Head, normalized)
    }
}

fn source_roles(
    layer: Option<QwenLayer>,
    kind: QwenOperation,
) -> (Vec<QwenMatrixRole>, Vec<QwenParameterRole>) {
    let i = layer.map(QwenLayer::index).unwrap_or(0);
    use crate::hybrid_decoder::{ParameterGroup as P, ProjectionGroup as M};
    let group = match kind {
        QwenOperation::Projection(p) => Some(match p {
            QwenProjection::Qkvz => M::GdnInput,
            QwenProjection::Ba => M::GdnCoefficients,
            QwenProjection::GdnOut => M::GdnOutput,
            QwenProjection::AttentionQkv => M::AttentionInput,
            QwenProjection::AttentionOut => M::AttentionOutput,
            QwenProjection::GateUp => M::FfnInput,
            QwenProjection::Down => M::FfnOutput,
            QwenProjection::Head => M::Head,
        }),
        _ => None,
    };
    let roles = group.map(|group| group.roles(i)).unwrap_or_default();
    let group = match kind {
        QwenOperation::InputRms | QwenOperation::ResidualRms(QwenRmsSite::Input) => {
            Some(P::InputNorm)
        }
        QwenOperation::ResidualRms(QwenRmsSite::PostAttention) => Some(P::FfnNorm),
        QwenOperation::ResidualRms(QwenRmsSite::Final) => Some(P::FinalNorm),
        QwenOperation::GdnConv => Some(P::GdnConvolution),
        QwenOperation::GdnRecurrent => Some(P::GdnRecurrent),
        QwenOperation::AttentionQk => Some(P::AttentionNorm),
        _ => None,
    };
    let parameter_roles = group.map(|group| group.roles(i)).unwrap_or_default();
    (roles, parameter_roles)
}

pub(super) fn validate_source_operator(
    source: zerogravity_core::accelerator::SourceIdentity,
    operator: &QwenOperator,
    mut matrix: impl FnMut(QwenMatrixRole, &SourceTensor) -> Result<()>,
    mut parameter: impl FnMut(QwenParameterRole, &QwenParameterDescriptor) -> Result<()>,
) -> Result<()> {
    operator.validate()?;
    if operator.call().source() != source {
        return Err(Error::State("Qwen operator source issuer differs".into()));
    }
    let (matrices, parameters) = source_roles(operator.layer(), operator.kind());
    if matrices.len() != operator.matrices().len()
        || parameters.len() != operator.parameters().len()
    {
        return Err(Error::State(
            "Qwen exact ordered source-member roles".into(),
        ));
    }
    for (role, descriptor) in matrices.into_iter().zip(operator.matrices()) {
        matrix(role, descriptor)?;
    }
    for (role, descriptor) in parameters.into_iter().zip(operator.parameters()) {
        parameter(role, descriptor)?;
    }
    Ok(())
}
