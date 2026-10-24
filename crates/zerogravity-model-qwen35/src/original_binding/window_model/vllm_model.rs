//! Original source recipe, through the existing family layer traversal.
use super::*;
use zerogravity_core::accelerator::program::{
    FullModelProfile, MathematicalBackend, PlanBuilder, PlanValue, RecordedVllm08Model, Vllm08Call,
    Vllm08Kind, Vllm08Layer, Vllm08Operator, Vllm08Projection, Vllm08RmsSite, VllmBf16Embedding,
};

impl OriginalWindowModelSource {
    /// Closed source/config metadata before any device/service/learned upload.
    /// Config pin also binds original mrope fields not exposed by old geometry.
    pub fn require_vllm08_full_source(&self) -> Result<()> {
        use zerogravity_artifacts::model::qwen::GdnOutputGate;
        use zerogravity_artifacts::model::{Family, LayerKind};
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
    pub fn record_vllm08_model(&self, call: Vllm08Call) -> Result<RecordedVllm08Model> {
        self.require_vllm08_full_source()?;
        if call.model_profile() != FullModelProfile::Qwen08 {
            return Err(Error::Unsupported(
                "original .8B recording constructor profile".into(),
            ));
        }
        self.record_full_model_inner(call)
    }
    // Both sealed source profiles use this SAME family traversal/SSA recipe.
    pub(super) fn record_full_model_inner(&self, call: Vllm08Call) -> Result<RecordedVllm08Model> {
        if call.source() != self.source_identity() {
            return Err(Error::State("full-model request/source issuer".into()));
        }
        let mut recorder = FullSourceRecorder {
            source: self,
            call: &call,
            builder: PlanBuilder::new_full_model(call.clone())?,
            boundaries: Vec::with_capacity(self.geometry().layers.len()),
            normalized: None,
        };
        let logits = crate::hybrid_decoder::compose(&self.geometry().layers, &mut recorder)?;
        let normalized = recorder
            .normalized
            .ok_or_else(|| Error::State("full-model final norm absent".into()))?;
        recorder
            .builder
            .finish_full_model(recorder.boundaries, normalized, logits)
    }

    fn full_projection(
        &self,
        builder: &mut PlanBuilder,
        call: &Vllm08Call,
        layer: Option<Vllm08Layer>,
        kind: Vllm08Projection,
        input: PlanValue,
    ) -> Result<PlanValue> {
        self.full_one(builder, call, layer, Vllm08Kind::Projection(kind), &[input])
    }
    fn full_one(
        &self,
        builder: &mut PlanBuilder,
        call: &Vllm08Call,
        layer: Option<Vllm08Layer>,
        kind: Vllm08Kind,
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
        builder: &mut PlanBuilder,
        call: &Vllm08Call,
        layer: Option<Vllm08Layer>,
        kind: Vllm08Kind,
        inputs: &[PlanValue],
    ) -> Result<[PlanValue; 2]> {
        self.full_apply(builder, call, layer, kind, inputs)?
            .try_into()
            .map_err(|_| Error::State("full-model exact dual output".into()))
    }
    fn full_apply(
        &self,
        builder: &mut PlanBuilder,
        call: &Vllm08Call,
        layer: Option<Vllm08Layer>,
        kind: Vllm08Kind,
        inputs: &[PlanValue],
    ) -> Result<Vec<PlanValue>> {
        let i = layer.map(Vllm08Layer::index).unwrap_or(0);
        use crate::hybrid_decoder::{ParameterGroup as P, ProjectionGroup as M};
        let group = match kind {
            Vllm08Kind::Projection(p) => Some(match p {
                Vllm08Projection::Qkvz => M::GdnInput,
                Vllm08Projection::Ba => M::GdnCoefficients,
                Vllm08Projection::GdnOut => M::GdnOutput,
                Vllm08Projection::AttentionQkv => M::AttentionInput,
                Vllm08Projection::AttentionOut => M::AttentionOutput,
                Vllm08Projection::GateUp => M::FfnInput,
                Vllm08Projection::Down => M::FfnOutput,
                Vllm08Projection::Head => M::Head,
            }),
            _ => None,
        };
        let roles = group.map(|group| group.roles(i)).unwrap_or_default();
        let group = match kind {
            Vllm08Kind::InputRms | Vllm08Kind::ResidualRms(Vllm08RmsSite::Input) => {
                Some(P::InputNorm)
            }
            Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention) => Some(P::FfnNorm),
            Vllm08Kind::ResidualRms(Vllm08RmsSite::Final) => Some(P::FinalNorm),
            Vllm08Kind::GdnConv => Some(P::GdnConvolution),
            Vllm08Kind::GdnRecurrent => Some(P::GdnRecurrent),
            Vllm08Kind::AttentionQk => Some(P::AttentionNorm),
            _ => None,
        };
        let parameter_roles = group.map(|group| group.roles(i)).unwrap_or_default();
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
        builder.vllm08_operator(
            &Vllm08Operator::new(call.clone(), layer, kind, matrices, parameters)?,
            inputs,
        )
    }
}

struct FullSourceRecorder<'a> {
    source: &'a OriginalWindowModelSource,
    call: &'a Vllm08Call,
    builder: PlanBuilder,
    boundaries: Vec<[PlanValue; 2]>,
    normalized: Option<PlanValue>,
}
impl FullSourceRecorder<'_> {
    fn one(
        &mut self,
        layer: Option<Vllm08Layer>,
        kind: Vllm08Kind,
        inputs: &[PlanValue],
    ) -> Result<PlanValue> {
        self.source
            .full_one(&mut self.builder, self.call, layer, kind, inputs)
    }
    fn two(
        &mut self,
        layer: Option<Vllm08Layer>,
        kind: Vllm08Kind,
        inputs: &[PlanValue],
    ) -> Result<[PlanValue; 2]> {
        self.source
            .full_two(&mut self.builder, self.call, layer, kind, inputs)
    }
    fn project(
        &mut self,
        layer: Option<Vllm08Layer>,
        kind: Vllm08Projection,
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
        self.builder
            .vllm_bf16_source_embedding(&VllmBf16Embedding::for_call(
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
            != (self.source.geometry().layers[index] == crate::model::LayerKind::FullAttention)
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
                self.one(Some(layer), Vllm08Kind::InputRms, &[branch])?,
                branch,
            ))
        } else {
            let residual =
                residual.ok_or_else(|| Error::State("original carried residual absent".into()))?;
            let [normal, carried] = self.two(
                Some(layer),
                Vllm08Kind::ResidualRms(Vllm08RmsSite::Input),
                &[branch, residual],
            )?;
            Ok((normal, carried))
        }
    }
    fn gdn_inputs(&mut self, index: usize, normal: &PlanValue) -> Result<Self::GdnInputs> {
        let layer = Some(self.call.layer(index)?);
        let qkvz = self.project(layer, Vllm08Projection::Qkvz, *normal)?;
        let ba = self.project(layer, Vllm08Projection::Ba, *normal)?;
        Ok([qkvz, ba])
    }
    fn gdn_evolve(&mut self, index: usize, [packed, ba]: Self::GdnInputs) -> Result<PlanValue> {
        let layer = Some(self.call.layer(index)?);
        let [activated, z] = self.two(layer, Vllm08Kind::GdnConv, &[packed, ba])?;
        self.one(layer, Vllm08Kind::GdnRecurrent, &[activated, z, ba])
    }
    fn attention_inputs(&mut self, index: usize, normal: &PlanValue) -> Result<PlanValue> {
        self.project(
            Some(self.call.layer(index)?),
            Vllm08Projection::AttentionQkv,
            *normal,
        )
    }
    fn attention_evolve(&mut self, index: usize, packed: PlanValue) -> Result<PlanValue> {
        let layer = Some(self.call.layer(index)?);
        let outputs = self.source.full_apply(
            &mut self.builder,
            self.call,
            layer,
            Vllm08Kind::AttentionQk,
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
                Vllm08Kind::KvAppend,
                &[k, packed],
            )?
            .is_empty()
        {
            return Err(Error::State(
                "KV append is one private effect without public value".into(),
            ));
        }
        let fa = self.one(layer, Vllm08Kind::FlashAttention, &[q])?;
        let sigmoid = self.one(layer, Vllm08Kind::Sigmoid, &[gate])?;
        self.one(layer, Vllm08Kind::Multiply, &[fa, sigmoid])
    }
    fn attention_output(
        &mut self,
        index: usize,
        kind: crate::model::LayerKind,
        value: PlanValue,
    ) -> Result<PlanValue> {
        self.project(
            Some(self.call.layer(index)?),
            if kind == crate::model::LayerKind::FullAttention {
                Vllm08Projection::AttentionOut
            } else {
                Vllm08Projection::GdnOut
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
            Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention),
            &[branch, carried],
        )?;
        Ok((normal, carried))
    }
    fn ffn_inputs(&mut self, index: usize, normal: &PlanValue) -> Result<PlanValue> {
        self.project(
            Some(self.call.layer(index)?),
            Vllm08Projection::GateUp,
            *normal,
        )
    }
    fn ffn_activate(&mut self, index: usize, input: PlanValue) -> Result<PlanValue> {
        self.one(
            Some(self.call.layer(index)?),
            Vllm08Kind::SiluAndMul,
            &[input],
        )
    }
    fn ffn_output(&mut self, index: usize, input: PlanValue) -> Result<PlanValue> {
        self.project(Some(self.call.layer(index)?), Vllm08Projection::Down, input)
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
            Vllm08Kind::ResidualRms(Vllm08RmsSite::Final),
            &[branch, residual],
        )?;
        self.normalized = Some(normalized);
        self.project(None, Vllm08Projection::Head, normalized)
    }
}
