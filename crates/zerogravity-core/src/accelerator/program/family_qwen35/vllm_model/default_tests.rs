use super::*;

#[test]
fn unaware_backend_full_model_operator_refuses_before_any_legacy_callback() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let request = VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 0).unwrap(),
        VllmBf16Phase::InitialOne,
        VllmBf16Teachers::new(source, 0, 512, &[11]).unwrap(),
    )
    .unwrap();
    let call = Vllm08Call::new(request).unwrap();
    let d = Vllm08Operator::new(
        call.clone(),
        Some(call.layer(3).unwrap()),
        Vllm08Kind::Sigmoid,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    assert!(matches!(
        UnawareBackend.vllm08_operator(&d, &[vec![0.0; 2048]]),
        Err(Error::Unsupported(_))
    ));
}
