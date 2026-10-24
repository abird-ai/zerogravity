//! Source-owned metadata recipe for the first SAME-engine stateful fragment.
use super::*;
use zerogravity_core::accelerator::program::{
    GdnPrefixLayer, GdnPrefixRmsSite, MathematicalBackend, PlanBuilder, PlanValue,
    RecordedVllmGdnFragment, ResidualRmsContract, VllmBf16Embedding, VllmBf16Invocation,
    VllmGdnConv, VllmGdnProjection, VllmGdnProjectionKind, VllmGdnRecurrent, VllmInputRms,
    VllmSiluAndMul,
};
impl OriginalWindowModelSource {
    /// Only records the source schedule; no driver, state root or model permit.
    pub fn record_initial_vllm_gdn_fragment(
        &self,
        invocation: VllmBf16Invocation,
    ) -> Result<RecordedVllmGdnFragment> {
        invocation.validate()?;
        if invocation.layer().source() != self.source_identity() || invocation.layer().index() != 0
        {
            return Err(Error::State(
                "original initial GDN recipe source/layer".into(),
            ));
        }
        let mut builder = PlanBuilder::new_vllm_gdn_fragment(invocation.clone())?;
        let embedding = VllmBf16Embedding::new(
            self.matrix_descriptor(QwenMatrixRole::Embedding)?,
            invocation.teachers().clone(),
        )?;
        let rows = builder.vllm_bf16_source_embedding(&embedding)?;
        let normal = builder.vllm_input_rms(
            &VllmInputRms::new(
                invocation.clone(),
                self.parameter_descriptor(QwenParameterRole::InputNorm(0))?,
            )?,
            &rows,
        )?;
        let gated = self.record_native_gdn_layer(&mut builder, &invocation, &normal)?;
        builder.finish_vllm_gdn_fragment(gated)
    }
}

impl OriginalWindowModelSource {
    /// Complete initial layer0 or prefix0..2 through the SAME layer traversal.
    /// The final MLP branch and carried residual are deliberately not added.
    pub fn record_initial_vllm_gdn_mlp_fragment(
        &self,
        invocation: VllmBf16Invocation,
        completed_layers: usize,
    ) -> Result<RecordedVllmGdnFragment> {
        self.record_vllm_gdn_mlp_fragment_inner(invocation, completed_layers, false)
    }
    /// Same source schedule, no cached/root execution permission.
    pub fn record_vllm_gdn_attempt_phase(
        &self,
        invocation: VllmBf16Invocation,
        completed_layers: usize,
    ) -> Result<RecordedVllmGdnFragment> {
        self.record_vllm_gdn_mlp_fragment_inner(invocation, completed_layers, true)
    }
    fn record_vllm_gdn_mlp_fragment_inner(
        &self,
        invocation: VllmBf16Invocation,
        completed_layers: usize,
        attempt_phase: bool,
    ) -> Result<RecordedVllmGdnFragment> {
        invocation.validate()?;
        if invocation.layer().source() != self.source_identity()
            || invocation.layer().index() != 0
            || !matches!(completed_layers, 1 | 3)
            || self.geometry().layers.len() != 24
        {
            return Err(Error::Unsupported(
                "source-owned initial complete GDN prefix0/0..2".into(),
            ));
        }
        let mut builder = if attempt_phase {
            PlanBuilder::new_vllm_gdn_attempt_phase(invocation.clone())?
        } else {
            PlanBuilder::new_vllm_gdn_fragment(invocation.clone())?
        };
        let rows = builder.vllm_bf16_source_embedding(&VllmBf16Embedding::new(
            self.matrix_descriptor(QwenMatrixRole::Embedding)?,
            invocation.teachers().clone(),
        )?)?;
        let (branch, residual) = crate::run_layer_sequence(
            &self.geometry().layers[..completed_layers],
            (rows, None),
            |index, kind, (branch, residual)| {
                if kind != crate::model::LayerKind::LinearAttention {
                    return Err(Error::Unsupported(
                        "closed initial GDN prefix excludes attention".into(),
                    ));
                }
                let layer = GdnPrefixLayer::new(self.source_identity(), index)?;
                // Clone the SAME teacher issuer; no new span/token authority.
                let call = VllmBf16Invocation::new(
                    layer,
                    invocation.legacy_phase()?,
                    invocation.teachers().clone(),
                )?;
                let (normal, carried) = if index == 0 {
                    if residual.is_some() {
                        return Err(Error::State(
                            "initial source rows have no pending residual".into(),
                        ));
                    }
                    (
                        builder.vllm_input_rms(
                            &VllmInputRms::new(
                                call.clone(),
                                self.parameter_descriptor(QwenParameterRole::InputNorm(0))?,
                            )?,
                            &branch,
                        )?,
                        branch,
                    )
                } else {
                    let prior = residual.ok_or_else(|| {
                        Error::State("pending previous layer residual absent".into())
                    })?;
                    let values = builder.residual_rms_profiled(
                        self.parameter_descriptor(QwenParameterRole::InputNorm(index))?,
                        &branch,
                        &prior,
                        ResidualRmsContract::Vllm031CudaGdnPrefix(
                            layer.rms(GdnPrefixRmsSite::InputResidual)?,
                        ),
                    )?;
                    (values.normalized, values.residual)
                };
                let gated = self.record_native_gdn_layer(&mut builder, &call, &normal)?;
                let projected = self.record_native_projection(
                    &mut builder,
                    &call,
                    VllmGdnProjectionKind::GdnOut,
                    &gated,
                )?;
                let post = builder.residual_rms_profiled(
                    self.parameter_descriptor(QwenParameterRole::FfnNorm(index))?,
                    &projected,
                    &carried,
                    ResidualRmsContract::Vllm031CudaGdnPrefix(
                        layer.rms(GdnPrefixRmsSite::PostAttention)?,
                    ),
                )?;
                let gate_up = self.record_native_projection(
                    &mut builder,
                    &call,
                    VllmGdnProjectionKind::GateUp,
                    &post.normalized,
                )?;
                let activated =
                    builder.vllm_silu_and_mul(&VllmSiluAndMul::new(call.clone())?, &gate_up)?;
                let down = self.record_native_projection(
                    &mut builder,
                    &call,
                    VllmGdnProjectionKind::Down,
                    &activated,
                )?;
                Ok((down, Some(post.residual)))
            },
        )?;
        builder.finish_vllm_gdn_mlp_fragment(
            branch,
            residual.ok_or_else(|| Error::State("completed layer residual absent".into()))?,
            completed_layers,
        )
    }
    fn record_native_projection(
        &self,
        builder: &mut PlanBuilder,
        call: &VllmBf16Invocation,
        kind: VllmGdnProjectionKind,
        input: &PlanValue,
    ) -> Result<PlanValue> {
        let i = call.layer().index();
        let roles: Vec<_> = match kind {
            VllmGdnProjectionKind::Qkvz => {
                vec![QwenMatrixRole::GdnQkv(i), QwenMatrixRole::GdnGate(i)]
            }
            VllmGdnProjectionKind::Ba => {
                vec![QwenMatrixRole::GdnBeta(i), QwenMatrixRole::GdnAlpha(i)]
            }
            VllmGdnProjectionKind::GdnOut => vec![QwenMatrixRole::GdnOutput(i)],
            VllmGdnProjectionKind::GateUp => {
                vec![QwenMatrixRole::FfnGate(i), QwenMatrixRole::FfnUp(i)]
            }
            VllmGdnProjectionKind::Down => vec![QwenMatrixRole::FfnDown(i)],
        };
        let mut members = Vec::with_capacity(roles.len());
        for role in roles {
            members.push(self.matrix_descriptor(role)?);
        }
        builder.vllm_gdn_project(&VllmGdnProjection::new(call.clone(), kind, members)?, input)
    }
    fn record_native_gdn_layer(
        &self,
        builder: &mut PlanBuilder,
        call: &VllmBf16Invocation,
        normal: &PlanValue,
    ) -> Result<PlanValue> {
        let i = call.layer().index();
        let qkvz =
            self.record_native_projection(builder, call, VllmGdnProjectionKind::Qkvz, normal)?;
        let ba = self.record_native_projection(builder, call, VllmGdnProjectionKind::Ba, normal)?;
        let convolved = builder.vllm_gdn_conv(
            &VllmGdnConv::new(
                call.clone(),
                self.parameter_descriptor(QwenParameterRole::GdnConvolution(i))?,
            )?,
            &qkvz,
            &ba,
        )?;
        builder.vllm_gdn_recurrent(
            &VllmGdnRecurrent::new(
                call.clone(),
                [
                    self.parameter_descriptor(QwenParameterRole::GdnA(i))?,
                    self.parameter_descriptor(QwenParameterRole::GdnDt(i))?,
                    self.parameter_descriptor(QwenParameterRole::GdnNorm(i))?,
                ],
            )?,
            &convolved.activated,
            &convolved.z,
            &ba,
        )
    }
}
