use super::*;

#[test]
fn unaware_native_projection_and_activation_refuse_before_any_legacy_or_metadata_callback() {
    let mut backend = UnawareBackend;
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 0).unwrap(),
        VllmBf16Phase::InitialOne,
        VllmBf16Teachers::new(source, 0, 512, &[11]).unwrap(),
    )
    .unwrap();
    let descriptor = VllmGdnProjection::new(
        call.clone(),
        VllmGdnProjectionKind::Ba,
        ["b", "a"]
            .into_iter()
            .map(|role| SourceTensor {
                source,
                name: format!("model.language_model.layers.0.linear_attn.in_proj_{role}.weight"),
                kind: QuantKind::Bf16,
                axes: MatrixAxes::OutputInput,
                columns: 1024,
                rows: 16,
            })
            .collect(),
    )
    .unwrap();
    assert!(matches!(
        backend.vllm_gdn_project(&descriptor, &vec![]),
        Err(Error::Unsupported(_))
    ));
    let activation = VllmSiluAndMul::new(call).unwrap();
    assert!(matches!(
        backend.vllm_silu_and_mul(&activation, &vec![]),
        Err(Error::Unsupported(_))
    ));
}
