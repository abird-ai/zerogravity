//! Source-owned metadata recipe for the first SAME-engine stateful fragment.
use super::*;
use crate::plan::{
    GdnProjection, QwenEmbedding, QwenFragmentPlan, QwenGdnCall, QwenGdnConv, QwenGdnProjection,
    QwenGdnRecurrent, QwenInputRms, QwenPlanBuilder, QwenPrefixLayer, QwenPrefixRmsSite,
    QwenResidualRmsContract, QwenSiluAndMul,
};
use zerogravity_core::accelerator::program::PlanValue;
impl OriginalWindowModelSource {
    /// Only records the source schedule; no driver, state root or model permit.
    pub fn record_initial_gdn_fragment(&self, invocation: QwenGdnCall) -> Result<QwenFragmentPlan> {
        invocation.validate()?;
        if invocation.layer().source() != self.source_identity() || invocation.layer().index() != 0
        {
            return Err(Error::State(
                "original initial GDN recipe source/layer".into(),
            ));
        }
        let mut builder = QwenPlanBuilder::new_gdn_fragment(invocation.clone())?;
        let embedding = QwenEmbedding::new(
            self.matrix_descriptor(QwenMatrixRole::Embedding)?,
            invocation.teachers().clone(),
        )?;
        let rows = builder.source_embedding(&embedding)?;
        let normal = builder.input_rms(
            &QwenInputRms::new(
                invocation.clone(),
                self.parameter_descriptor(QwenParameterRole::InputNorm(0))?,
            )?,
            &rows,
        )?;
        let gated = self.record_native_gdn_layer(&mut builder, &invocation, &normal)?;
        builder.finish_gdn_fragment(gated)
    }
}

impl OriginalWindowModelSource {
    /// Complete initial layer0 or prefix0..2 through the SAME layer traversal.
    /// The final MLP branch and carried residual are deliberately not added.
    pub fn record_initial_gdn_mlp_fragment(
        &self,
        invocation: QwenGdnCall,
        completed_layers: usize,
    ) -> Result<QwenFragmentPlan> {
        self.record_gdn_mlp_fragment_inner(invocation, completed_layers, false)
    }
    /// Same source schedule, no cached/root execution permission.
    pub fn record_gdn_attempt_phase(
        &self,
        invocation: QwenGdnCall,
        completed_layers: usize,
    ) -> Result<QwenFragmentPlan> {
        self.record_gdn_mlp_fragment_inner(invocation, completed_layers, true)
    }
    // Previous name: record_vllm_gdn_mlp_fragment_inner.
    fn record_gdn_mlp_fragment_inner(
        &self,
        invocation: QwenGdnCall,
        completed_layers: usize,
        attempt_phase: bool,
    ) -> Result<QwenFragmentPlan> {
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
            QwenPlanBuilder::new_gdn_attempt_phase(invocation.clone())?
        } else {
            QwenPlanBuilder::new_gdn_fragment(invocation.clone())?
        };
        let rows = builder.source_embedding(&QwenEmbedding::new(
            self.matrix_descriptor(QwenMatrixRole::Embedding)?,
            invocation.teachers().clone(),
        )?)?;
        let (branch, residual) = crate::run_layer_sequence(
            &self.geometry().layers[..completed_layers],
            (rows, None),
            |index, kind, (branch, residual)| {
                if kind != crate::config::LayerKind::LinearAttention {
                    return Err(Error::Unsupported(
                        "closed initial GDN prefix excludes attention".into(),
                    ));
                }
                let layer = QwenPrefixLayer::new(self.source_identity(), index)?;
                // Clone the SAME teacher issuer; no new span/token authority.
                let call = QwenGdnCall::new(
                    layer,
                    invocation.reference_phase()?,
                    invocation.teachers().clone(),
                )?;
                let (normal, carried) = if index == 0 {
                    if residual.is_some() {
                        return Err(Error::State(
                            "initial source rows have no pending residual".into(),
                        ));
                    }
                    (
                        builder.input_rms(
                            &QwenInputRms::new(
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
                        QwenResidualRmsContract::GdnPrefix(
                            layer.rms(QwenPrefixRmsSite::InputResidual)?,
                        ),
                    )?;
                    (values.normalized, values.residual)
                };
                let gated = self.record_native_gdn_layer(&mut builder, &call, &normal)?;
                let projected = self.record_native_projection(
                    &mut builder,
                    &call,
                    GdnProjection::GdnOut,
                    &gated,
                )?;
                let post = builder.residual_rms_profiled(
                    self.parameter_descriptor(QwenParameterRole::FfnNorm(index))?,
                    &projected,
                    &carried,
                    QwenResidualRmsContract::GdnPrefix(
                        layer.rms(QwenPrefixRmsSite::PostAttention)?,
                    ),
                )?;
                let gate_up = self.record_native_projection(
                    &mut builder,
                    &call,
                    GdnProjection::GateUp,
                    &post.normalized,
                )?;
                let activated =
                    builder.silu_and_mul(&QwenSiluAndMul::new(call.clone())?, &gate_up)?;
                let down = self.record_native_projection(
                    &mut builder,
                    &call,
                    GdnProjection::Down,
                    &activated,
                )?;
                Ok((down, Some(post.residual)))
            },
        )?;
        builder.finish_gdn_mlp_fragment(
            branch,
            residual.ok_or_else(|| Error::State("completed layer residual absent".into()))?,
            completed_layers,
        )
    }
    fn record_native_projection(
        &self,
        builder: &mut QwenPlanBuilder,
        call: &QwenGdnCall,
        kind: GdnProjection,
        input: &PlanValue,
    ) -> Result<PlanValue> {
        let i = call.layer().index();
        let roles: Vec<_> = match kind {
            GdnProjection::Qkvz => {
                vec![QwenMatrixRole::GdnQkv(i), QwenMatrixRole::GdnGate(i)]
            }
            GdnProjection::Ba => {
                vec![QwenMatrixRole::GdnBeta(i), QwenMatrixRole::GdnAlpha(i)]
            }
            GdnProjection::GdnOut => vec![QwenMatrixRole::GdnOutput(i)],
            GdnProjection::GateUp => {
                vec![QwenMatrixRole::FfnGate(i), QwenMatrixRole::FfnUp(i)]
            }
            GdnProjection::Down => vec![QwenMatrixRole::FfnDown(i)],
        };
        let mut members = Vec::with_capacity(roles.len());
        for role in roles {
            members.push(self.matrix_descriptor(role)?);
        }
        builder.gdn_project(&QwenGdnProjection::new(call.clone(), kind, members)?, input)
    }
    fn record_native_gdn_layer(
        &self,
        builder: &mut QwenPlanBuilder,
        call: &QwenGdnCall,
        normal: &PlanValue,
    ) -> Result<PlanValue> {
        let i = call.layer().index();
        let qkvz = self.record_native_projection(builder, call, GdnProjection::Qkvz, normal)?;
        let ba = self.record_native_projection(builder, call, GdnProjection::Ba, normal)?;
        let convolved = builder.gdn_conv(
            &QwenGdnConv::new(
                call.clone(),
                self.parameter_descriptor(QwenParameterRole::GdnConvolution(i))?,
            )?,
            &qkvz,
            &ba,
        )?;
        builder.gdn_recurrent(
            &QwenGdnRecurrent::new(
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
