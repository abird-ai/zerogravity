use super::*;
#[test]
fn default_backend_gdn_and_input_rms_refuse_before_callbacks() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 0).unwrap(),
        VllmBf16Phase::InitialOne,
        VllmBf16Teachers::new(source, 0, 512, &[11]).unwrap(),
    )
    .unwrap();
    let parameter = |suffix: &str, kind: QuantKind, shape: &[u64], bytes| SourceParameter {
        source,
        name: format!("model.language_model.layers.0.{suffix}"),
        kind,
        shape: shape.to_vec(),
        bytes,
    };
    let rms = VllmInputRms::new(
        call.clone(),
        parameter("input_layernorm.weight", QuantKind::Bf16, &[1024], 2048),
    )
    .unwrap();
    let conv = VllmGdnConv::new(
        call.clone(),
        parameter(
            "linear_attn.conv1d.weight",
            QuantKind::Bf16,
            &[6144, 1, 4],
            49152,
        ),
    )
    .unwrap();
    let recurrent = VllmGdnRecurrent::new(
        call,
        [
            parameter("linear_attn.A_log", QuantKind::F32, &[16], 64),
            parameter("linear_attn.dt_bias", QuantKind::Bf16, &[16], 32),
            parameter("linear_attn.norm.weight", QuantKind::F32, &[128], 512),
        ],
    )
    .unwrap();
    let mut backend = UnawareBackend;
    assert!(matches!(
        backend.vllm_input_rms(&rms, &vec![]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        backend.vllm_gdn_conv(&conv, &vec![], &vec![]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        backend.vllm_gdn_recurrent(&recurrent, &vec![], &vec![], &vec![]),
        Err(Error::Unsupported(_))
    ));
}
