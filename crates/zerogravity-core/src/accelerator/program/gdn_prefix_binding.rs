//! Metadata only: exact .8B GDN prefix keys, never a source/runtime permit.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnPrefixLayer {
    source: SourceIdentity,
    index: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnPrefixRmsSite {
    InputResidual,
    PostAttention,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnPrefixRmsBinding {
    layer: GdnPrefixLayer,
    site: GdnPrefixRmsSite,
}
impl GdnPrefixLayer {
    /// Descriptor bounds only. Consumer must validate actual config and issuer.
    pub fn new(source: SourceIdentity, index: usize) -> Result<Self> {
        if !source.is_bound() || index > 2 {
            return Err(Error::Unsupported("closed GDN prefix layers0..2".into()));
        }
        Ok(Self {
            source,
            index: index as u8,
        })
    }
    pub fn source(self) -> SourceIdentity {
        self.source
    }
    pub fn index(self) -> usize {
        self.index as usize
    }
    pub fn rms(self, site: GdnPrefixRmsSite) -> Result<GdnPrefixRmsBinding> {
        if self.index == 0 && site == GdnPrefixRmsSite::InputResidual {
            return Err(Error::Unsupported(
                "layer0 input has no pending residual".into(),
            ));
        }
        Ok(GdnPrefixRmsBinding { layer: self, site })
    }
}
impl GdnPrefixRmsBinding {
    pub fn layer(self) -> GdnPrefixLayer {
        self.layer
    }
    pub fn site(self) -> GdnPrefixRmsSite {
        self.site
    }
    pub fn gamma_name(self) -> String {
        let suffix = match self.site {
            GdnPrefixRmsSite::InputResidual => "input_layernorm.weight",
            GdnPrefixRmsSite::PostAttention => "post_attention_layernorm.weight",
        };
        format!(
            "model.language_model.layers.{}.{suffix}",
            self.layer.index()
        )
    }
    pub fn validate(self, parameter: &SourceParameter, elements: usize) -> Result<usize> {
        if parameter.source != self.layer.source || parameter.name != self.gamma_name() {
            return Err(Error::State("GDN RMS exact layer/site/source key".into()));
        }
        let bytes = match parameter.kind {
            QuantKind::Bf16 => 2048,
            QuantKind::F32 => 4096,
            _ => return Err(Error::Unsupported("GDN RMS original BF16/F32 gamma".into())),
        };
        if parameter.shape != [1024] || parameter.bytes != bytes {
            return Err(Error::Malformed("GDN RMS raw gamma1024".into()));
        }
        if !elements.is_multiple_of(1024) || !matches!(elements / 1024, 1 | 2 | 64) {
            return Err(Error::Unsupported("GDN RMS dense rows1/2/64".into()));
        }
        Ok(elements / 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_keys_refuse_attention_layer_and_layer0_pending_input() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        assert!(GdnPrefixLayer::new(source, 3).is_err());
        assert!(GdnPrefixLayer::new(source, usize::MAX).is_err());
        assert!(
            GdnPrefixLayer::new(source, 0)
                .unwrap()
                .rms(GdnPrefixRmsSite::InputResidual)
                .is_err()
        );
        assert!(
            GdnPrefixLayer::new(source, 0)
                .unwrap()
                .rms(GdnPrefixRmsSite::PostAttention)
                .is_ok()
        );
    }

    #[test]
    fn raw_gamma_cannot_cross_layer_site_source_or_geometry() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let key = GdnPrefixLayer::new(source, 1)
            .unwrap()
            .rms(GdnPrefixRmsSite::InputResidual)
            .unwrap();
        let valid = SourceParameter {
            source,
            name: key.gamma_name(),
            kind: QuantKind::Bf16,
            shape: vec![1024],
            bytes: 2048,
        };
        for rows in [1, 2, 64] {
            assert_eq!(key.validate(&valid, rows * 1024).unwrap(), rows);
        }
        for mode in 0..5 {
            let mut bad = valid.clone();
            match mode {
                0 => {
                    bad.name = GdnPrefixLayer::new(source, 2)
                        .unwrap()
                        .rms(GdnPrefixRmsSite::InputResidual)
                        .unwrap()
                        .gamma_name()
                }
                1 => {
                    bad.name = GdnPrefixLayer::new(source, 1)
                        .unwrap()
                        .rms(GdnPrefixRmsSite::PostAttention)
                        .unwrap()
                        .gamma_name()
                }
                2 => bad.source = SourceIdentity::for_new_allocation().unwrap(),
                3 => bad.bytes = 4096,
                _ => bad.shape = vec![1, 1024],
            }
            assert!(key.validate(&bad, 1024).is_err(), "fault{mode}");
        }
        assert!(key.validate(&valid, 3 * 1024).is_err());
    }

    #[test]
    fn raw_gamma_keeps_explicit_dtype_bytes_and_exact_row_bands() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let binding = GdnPrefixLayer::new(source, 2)
            .unwrap()
            .rms(GdnPrefixRmsSite::PostAttention)
            .unwrap();
        for (kind, bytes) in [(QuantKind::Bf16, 2048), (QuantKind::F32, 4096)] {
            let parameter = SourceParameter {
                source,
                name: binding.gamma_name(),
                kind,
                shape: vec![1024],
                bytes,
            };
            for rows in [1, 2, 64] {
                assert!(binding.validate(&parameter, rows * 1024).is_ok());
            }
            for elements in [0, 1023, 1025, 3 * 1024, 65 * 1024, usize::MAX] {
                assert!(binding.validate(&parameter, elements).is_err());
            }
        }
        let f16 = SourceParameter {
            source,
            name: binding.gamma_name(),
            kind: QuantKind::F16,
            shape: vec![1024],
            bytes: 2048,
        };
        assert!(matches!(
            binding.validate(&f16, 1024),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn bound_rms_records_only_two_bf16_outputs_and_keeps_builder_issuer() {
        for layer in [1, 2] {
            for site in [
                GdnPrefixRmsSite::InputResidual,
                GdnPrefixRmsSite::PostAttention,
            ] {
                for rows in [1, 2, 64] {
                    let source = SourceIdentity::for_new_allocation().unwrap();
                    let key = GdnPrefixLayer::new(source, layer)
                        .unwrap()
                        .rms(site)
                        .unwrap();
                    let descriptor = VllmBf16Embedding::new(
                        SourceTensor {
                            source,
                            name: "model.language_model.embed_tokens.weight".into(),
                            kind: QuantKind::Bf16,
                            axes: MatrixAxes::OutputInput,
                            columns: 1024,
                            rows: 248320,
                        },
                        VllmBf16Teachers::new(source, 0, 64, &vec![11; rows]).unwrap(),
                    )
                    .unwrap();
                    let profile = ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1;
                    let mut builder = PlanBuilder::new_operator_slice(source, profile).unwrap();
                    let branch = builder.vllm_bf16_source_embedding(&descriptor).unwrap();
                    let prior = builder.vllm_bf16_source_embedding(&descriptor).unwrap();
                    let mut other = PlanBuilder::new_operator_slice(source, profile).unwrap();
                    let foreign = other.vllm_bf16_source_embedding(&descriptor).unwrap();
                    let gamma = SourceParameter {
                        source,
                        name: key.gamma_name(),
                        kind: QuantKind::Bf16,
                        shape: vec![1024],
                        bytes: 2048,
                    };
                    let contract = ResidualRmsContract::Vllm031CudaGdnPrefix(key);
                    assert!(
                        builder
                            .residual_rms_profiled(gamma.clone(), &foreign, &prior, contract)
                            .is_err()
                    );
                    let result = builder
                        .residual_rms_profiled(gamma, &branch, &prior, contract)
                        .unwrap();
                    assert_ne!(result.normalized.id, result.residual.id);
                    for output in [result.normalized, result.residual] {
                        assert_eq!(output.precision(), ActivationPrecision::Bf16WidenedF32);
                    }
                    let recorded = builder.finish_operator_slice(result.normalized).unwrap();
                    let plan = recorded.plan();
                    assert_eq!(plan.nodes().len(), 3, "refusal records no node");
                    assert_eq!(plan.values().len(), 4, "F32 sum remains private");
                    assert!(plan.nodes().iter().all(|node| node.state.is_none()));
                    assert_eq!(plan.frontier_delta(), 0);
                }
            }
        }
    }
}
