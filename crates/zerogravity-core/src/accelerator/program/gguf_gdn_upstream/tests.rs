use super::*;
struct Original {
    source: SourceIdentity,
    embedding: Vec<u8>,
    norm: Vec<u8>,
    qkv: Vec<u8>,
    z: Vec<u8>,
    conv: Vec<u8>,
}
impl Original {
    fn new() -> Self {
        Self {
            source: SourceIdentity::for_new_allocation().unwrap(),
            embedding: vec![0; 1680 * 16],
            norm: vec![0; 8192],
            qkv: vec![0; 1152 * 6144],
            z: vec![0; 1152 * 2048],
            conv: vec![0; 4 * 6144 * 4],
        }
    }
    fn descriptor(&self) -> GgufGdnUpstreamDescriptor {
        let span = TeacherSpan::new(self.source, 0, 8448, 16, &[11; 64]).unwrap();
        let prefix =
            GgufPrefixDescriptor::new(span, self, GgufPrefixContract::GgufSource2BFirst64V1, 1e-6)
                .unwrap();
        GgufGdnUpstreamDescriptor::new(
            prefix,
            self,
            GgufGdnUpstreamContract::GgufQ4Repacked2BFirst64V1,
        )
        .unwrap()
    }
}
impl TensorSource for Original {
    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        let (kind, shape, packed_bytes): (_, &[u64], &[u8]) = match name {
            "token_embd.weight" => (QuantKind::Q6K, &[2048, 16], &self.embedding),
            "blk.0.attn_norm.weight" => (QuantKind::F32, &[2048], &self.norm),
            "blk.0.attn_qkv.weight" => (QuantKind::Q4_0, &[2048, 6144], &self.qkv),
            "blk.0.attn_gate.weight" => (QuantKind::Q4_0, &[2048, 2048], &self.z),
            "blk.0.ssm_conv1d.weight" => (QuantKind::F32, &[4, 6144], &self.conv),
            _ => return Err(Error::Malformed("unknown original test tensor".into())),
        };
        Ok(SourceTensorRef {
            kind,
            shape,
            packed_bytes,
        })
    }
}
#[test]
fn original_source_descriptor_refuses_changed_source_or_missing_raw_bytes() {
    let mut source = Original::new();
    let descriptor = source.descriptor();
    assert_eq!(descriptor.tensor(GgufGdnProjection::Qkv).rows, 6144);
    assert_eq!(descriptor.tensor(GgufGdnProjection::Z).rows, 2048);
    source.qkv.pop();
    assert!(
        GgufGdnUpstreamDescriptor::new(descriptor.prefix().clone(), &source, descriptor.contract())
            .is_err()
    );
    source.qkv.push(0);
    source.source = SourceIdentity::for_new_allocation().unwrap();
    assert!(
        GgufGdnUpstreamDescriptor::new(descriptor.prefix().clone(), &source, descriptor.contract())
            .is_err()
    );
}
#[test]
fn five_node_slice_retains_exact_producers_and_refuses_legacy_views_or_role_swaps() {
    let source = Original::new();
    let descriptor = source.descriptor();
    let mut b =
        PlanBuilder::new_operator_slice(source.source, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let input = b.gguf_gdn_input(&descriptor).unwrap();
    assert!(b.silu("legacy", &input, None).is_err());
    assert!(b.view(&input, TensorView::vector(0, 131072)).is_err());
    let qkv = b
        .gguf_gdn_project(&descriptor, GgufGdnProjection::Qkv, &input)
        .unwrap();
    let z = b
        .gguf_gdn_project(&descriptor, GgufGdnProjection::Z, &input)
        .unwrap();
    assert!(b.gguf_gdn_conv_silu(&descriptor, &z).is_err());
    assert!(b.gguf_gdn_qk(&descriptor, &qkv).is_err());
    let conv = b.gguf_gdn_conv_silu(&descriptor, &qkv).unwrap();
    assert!(
        b.gguf_gdn_project(&descriptor, GgufGdnProjection::Qkv, &conv)
            .is_err()
    );
    let operands = b.gguf_gdn_qk(&descriptor, &conv).unwrap();
    assert!(b.silu("legacy", &operands, None).is_err());
    let plan = b.finish_operator_slice(operands).unwrap();
    assert_eq!(plan.plan().nodes().len(), 5);
    assert!(plan.plan().required_domains().is_empty());
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(plan.plan().logits().is_none());
    assert_eq!(plan.plan().frontier_delta(), 0);
}
#[test]
fn equal_ids_foreign_issuers_or_builders_and_other_profiles_cannot_substitute() {
    let source = Original::new();
    let descriptor = source.descriptor();
    let foreign = source.descriptor();
    let mut b =
        PlanBuilder::new_operator_slice(source.source, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let input = b.gguf_gdn_input(&descriptor).unwrap();
    assert!(
        b.gguf_gdn_project(&foreign, GgufGdnProjection::Qkv, &input)
            .is_err()
    );
    let mut other =
        PlanBuilder::new_operator_slice(source.source, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let value = other.gguf_gdn_input(&descriptor).unwrap();
    assert!(
        b.gguf_gdn_project(&descriptor, GgufGdnProjection::Qkv, &value)
            .is_err()
    );
    let mut bf16 = PlanBuilder::new_operator_slice(
        source.source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    assert!(bf16.gguf_gdn_input(&descriptor).is_err());
    let mut full = PlanBuilder::new(
        source.source,
        8448,
        vec![StateSlot::Conv(ConvDomain(0))],
        false,
    )
    .unwrap();
    assert!(full.gguf_gdn_input(&descriptor).is_err());
}
