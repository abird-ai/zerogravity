use super::*;

// Included below typed_tests.rs; reuse its panic-on-any-legacy-callback backend.
#[test]
fn unaware_backend_refuses_residual_rms_before_metadata_or_legacy_callbacks() {
    let mut backend = UnawareBackend;
    let parameter = SourceParameter {
        source: SourceIdentity::for_new_allocation().unwrap(),
        name: "model.language_model.layers.0.post_attention_layernorm.weight".into(),
        kind: QuantKind::Bf16,
        shape: vec![1024],
        bytes: 2048,
    };
    assert!(matches!(
        backend.residual_rms_profiled(
            parameter,
            &vec![0.0; 1024],
            &vec![0.0; 1024],
            ResidualRmsContract::Vllm031CudaAtenBf16H1024Rows1_2_64
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn unaware_backend_refuses_source_rows_before_ordinary_embedding_callback() {
    let mut backend = UnawareBackend;
    let source = SourceIdentity::for_new_allocation().unwrap();
    let tensor = SourceTensor {
        source,
        name: "model.language_model.embed_tokens.weight".into(),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        columns: 1024,
        rows: 248_320,
    };
    let teachers = VllmBf16Teachers::new(source, 0, 64, &[11]).unwrap();
    let descriptor = VllmBf16Embedding::new(tensor, teachers).unwrap();
    assert!(matches!(
        backend.vllm_bf16_source_embedding(&descriptor),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn unaware_backend_refuses_bound_layer_rms_before_any_legacy_callback() {
    let mut backend = UnawareBackend;
    let source = SourceIdentity::for_new_allocation().unwrap();
    let binding = GdnPrefixLayer::new(source, 1)
        .unwrap()
        .rms(GdnPrefixRmsSite::InputResidual)
        .unwrap();
    let parameter = SourceParameter {
        source,
        name: binding.gamma_name(),
        kind: QuantKind::Bf16,
        shape: vec![1024],
        bytes: 2048,
    };
    assert!(matches!(
        backend.residual_rms_profiled(
            parameter,
            &vec![0.0; 1024],
            &vec![0.0; 1024],
            ResidualRmsContract::Vllm031CudaGdnPrefix(binding)
        ),
        Err(Error::Unsupported(_))
    ));
}
