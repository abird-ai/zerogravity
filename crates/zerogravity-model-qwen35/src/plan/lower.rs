//! Family role validation precedes this translation to common mathematical data.
use super::*;
const NUMERICAL: SourceNumericalContract = SourceNumericalContract::ReferenceCudaSm86Bf16V1;
fn rms(width: usize) -> SourceRmsGeometry {
    SourceRmsGeometry {
        width,
        epsilon: 1e-6,
        gamma: SourceGamma::OnePlus,
    }
}
fn conv() -> SourceConvGeometry {
    SourceConvGeometry {
        qkv_channels: 6144,
        z_channels: 2048,
        coefficient_channels: 32,
        kernel: 4,
    }
}
fn gdn() -> SourceGdnGeometry {
    SourceGdnGeometry {
        key_heads: 16,
        value_heads: 16,
        key_width: 128,
        value_width: 128,
        epsilon: 1e-6,
        contract: SourceGdnContract::ReferenceCudaSm86L2SiluBf16V1,
    }
}
impl QwenOperator {
    pub fn mathematical_operator(&self) -> Result<SourceOperator> {
        self.validate()?;
        SourceOperator::new(
            self.call().request().source_call()?,
            self.mathematical_kind(),
            NUMERICAL,
            self.matrices().to_vec(),
            self.parameters().to_vec(),
        )
    }
    /// Compare immutable validated data without allocating on native submission.
    pub fn require_mathematical_operator(&self, actual: &SourceOperator) -> Result<()> {
        if actual.call() != &self.call().request().source_call()?
            || actual.kind() != self.mathematical_kind()
            || actual.numerical_contract() != NUMERICAL
            || actual.matrices() != self.matrices()
            || actual.parameters() != self.parameters()
        {
            return Err(Error::State(
                "Qwen source operator differs from prepared mathematical binding".into(),
            ));
        }
        Ok(())
    }
    fn mathematical_kind(&self) -> SourceOperatorKind {
        let layer = self.layer().map_or(0, QwenLayer::index);
        match self.kind() {
            QwenOperation::InputRms => {
                SourceOperatorKind::Rms(rms(self.call().model_profile().hidden_size()))
            }
            QwenOperation::ResidualRms(_) => {
                SourceOperatorKind::ResidualRms(rms(self.call().model_profile().hidden_size()))
            }
            QwenOperation::Projection(_) => SourceOperatorKind::Projection {
                selected_row: self.selected_row(),
            },
            QwenOperation::GdnConv => SourceOperatorKind::CausalConvSplit {
                domain: ConvDomain(layer),
                geometry: conv(),
            },
            QwenOperation::GdnRecurrent => SourceOperatorKind::GatedDelta {
                domain: RecurrentDomain(layer),
                geometry: gdn(),
            },
            QwenOperation::AttentionQk => {
                SourceOperatorKind::QkRmsRotaryGated(SourceQkRotaryGeometry {
                    attention: AttentionGeometry {
                        width: 256,
                        query_heads: 8,
                        kv_heads: 2,
                    },
                    rotary_dimensions: 64,
                    theta: 10000000.0,
                    epsilon: 1e-6,
                })
            }
            QwenOperation::KvAppend => SourceOperatorKind::KvAppend {
                domain: KvDomain(layer),
                key_width: 512,
                packed_value: TensorView {
                    offset: 4608,
                    rows: self.call().rows(),
                    width: 512,
                    row_stride: 5120,
                },
            },
            QwenOperation::FlashAttention => SourceOperatorKind::FlashAttention {
                domain: KvDomain(layer),
                geometry: AttentionGeometry {
                    width: 256,
                    query_heads: 8,
                    kv_heads: 2,
                },
                scale: 0.0625,
                mask: SourceAttentionMask::Causal,
            },
            QwenOperation::Sigmoid => SourceOperatorKind::Sigmoid { width: 2048 },
            QwenOperation::Multiply => SourceOperatorKind::Multiply { width: 2048 },
            QwenOperation::SiluAndMul => SourceOperatorKind::SiluAndMul {
                width: self.call().model_profile().intermediate_size(),
            },
        }
    }
}
impl QwenPlanBuilder {
    pub(super) fn embedding_operator(&self, descriptor: &QwenEmbedding) -> Result<SourceOperator> {
        let call = if let Some(call) = &self.model_call {
            call.request().source_call()?
        } else if let Some(call) = &self.fragment_call {
            call.source_call()?
        } else {
            let rows = descriptor.teachers();
            let mode = if rows.len() == 1 && matches!(rows.base(), 1 | 66) {
                SourceCallMode::Decode
            } else {
                SourceCallMode::Prefill
            };
            rows.source_call(mode)?
        };
        SourceOperator::new(
            call,
            SourceOperatorKind::EmbeddingRows,
            NUMERICAL,
            vec![descriptor.tensor().clone()],
            Vec::new(),
        )
    }
    pub(super) fn projection_operator(
        &self,
        descriptor: &QwenGdnProjection,
    ) -> Result<SourceOperator> {
        SourceOperator::new(
            descriptor.invocation().source_call()?,
            SourceOperatorKind::Projection { selected_row: None },
            NUMERICAL,
            descriptor.members().to_vec(),
            Vec::new(),
        )
    }
    pub(super) fn activation_operator(
        &self,
        descriptor: &QwenSiluAndMul,
    ) -> Result<SourceOperator> {
        SourceOperator::new(
            descriptor.invocation().source_call()?,
            SourceOperatorKind::SiluAndMul { width: 3584 },
            NUMERICAL,
            Vec::new(),
            Vec::new(),
        )
    }
    pub(super) fn input_rms_operator(&self, descriptor: &QwenInputRms) -> Result<SourceOperator> {
        SourceOperator::new(
            descriptor.invocation().source_call()?,
            SourceOperatorKind::Rms(rms(1024)),
            NUMERICAL,
            Vec::new(),
            vec![descriptor.parameter().clone()],
        )
    }
    pub(super) fn conv_operator(&self, descriptor: &QwenGdnConv) -> Result<SourceOperator> {
        SourceOperator::new(
            descriptor.invocation().source_call()?,
            SourceOperatorKind::CausalConvSplit {
                domain: ConvDomain(descriptor.invocation().layer().index()),
                geometry: conv(),
            },
            NUMERICAL,
            Vec::new(),
            vec![descriptor.parameter().clone()],
        )
    }
    pub(super) fn recurrent_operator(
        &self,
        descriptor: &QwenGdnRecurrent,
    ) -> Result<SourceOperator> {
        SourceOperator::new(
            descriptor.invocation().source_call()?,
            SourceOperatorKind::GatedDelta {
                domain: RecurrentDomain(descriptor.invocation().layer().index()),
                geometry: gdn(),
            },
            NUMERICAL,
            Vec::new(),
            descriptor.parameters().to_vec(),
        )
    }
    pub(super) fn residual_operator(
        &self,
        parameter: &SourceParameter,
        inputs: &[PlanValue],
    ) -> Result<SourceOperator> {
        // Operator-only spans need no finite ReferencePhase; use actual SSA call.
        let call = self
            .core
            .source_value_provenance(&inputs[0])?
            .ok_or_else(|| Error::State("Qwen residual actual source call absent".into()))?
            .call()
            .clone();
        SourceOperator::new(
            call,
            SourceOperatorKind::ResidualRms(rms(1024)),
            NUMERICAL,
            Vec::new(),
            vec![parameter.clone()],
        )
    }
}

impl QwenGdnConv {
    pub fn require_mathematical_operator(&self, actual: &SourceOperator) -> Result<()> {
        if actual.call() != &self.invocation().source_call()?
            || actual.kind()
                != (SourceOperatorKind::CausalConvSplit {
                    domain: ConvDomain(self.invocation().layer().index()),
                    geometry: conv(),
                })
            || actual.numerical_contract() != NUMERICAL
            || !actual.matrices().is_empty()
            || actual.parameters() != std::slice::from_ref(self.parameter())
        {
            return Err(Error::State(
                "prepared convolution mathematical binding differs".into(),
            ));
        }
        Ok(())
    }
}
impl QwenGdnRecurrent {
    pub fn require_mathematical_operator(&self, actual: &SourceOperator) -> Result<()> {
        if actual.call() != &self.invocation().source_call()?
            || actual.kind()
                != (SourceOperatorKind::GatedDelta {
                    domain: RecurrentDomain(self.invocation().layer().index()),
                    geometry: gdn(),
                })
            || actual.numerical_contract() != NUMERICAL
            || !actual.matrices().is_empty()
            || actual.parameters() != self.parameters()
        {
            return Err(Error::State(
                "prepared gated-delta mathematical binding differs".into(),
            ));
        }
        Ok(())
    }
}
