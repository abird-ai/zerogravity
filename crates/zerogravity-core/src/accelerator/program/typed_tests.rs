use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn explicit_cpu_pointwise_preserves_legacy_refusal_and_builder_ownership() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let span = |builder: &mut PlanBuilder, elements| {
        builder
            .embed_profiled(
                SourceTensor {
                    source,
                    name: format!("pointwise.span{elements}"),
                    kind: QuantKind::Bf16,
                    rows: 1,
                    columns: elements,
                    axes: MatrixAxes::OutputInput,
                },
                PlanBuilder::inputs().token,
                EmbeddingContract::Bf16OutputInput,
            )
            .unwrap()
    };
    let mut first = builder(source);
    let input = span(&mut first, 16);
    assert!(first.silu("legacy", &input, Some(&input)).is_err());
    assert!(first.add("legacy", &input, &input).is_err());
    let product = first
        .torch_cpu_bf16_silu_mul("staged", &input, &input)
        .unwrap();
    let sum = first
        .torch_cpu_bf16_add("residual", &product, &input)
        .unwrap();
    assert_eq!(sum.precision(), ActivationPrecision::Bf16WidenedF32);
    let mut second = builder(source);
    let foreign = span(&mut second, 16);
    assert!(first.torch_cpu_bf16_add("foreign", &sum, &foreign).is_err());
    let short = span(&mut first, 8);
    assert!(
        first
            .torch_cpu_bf16_silu_mul("scalar", &short, &short)
            .is_err()
    );
    assert!(first.torch_cpu_bf16_add("shape", &sum, &short).is_err());
    let wide = span(&mut first, 17424);
    assert!(first.torch_cpu_bf16_silu_mul("wide", &wide, &wide).is_err());
    let maximum = span(&mut first, 17408);
    assert!(
        first
            .torch_cpu_bf16_silu_mul("maximum", &maximum, &maximum)
            .is_ok()
    );
    let larger = span(&mut first, 32);
    let strided = first
        .view(
            &larger,
            TensorView {
                offset: 0,
                rows: 2,
                width: 8,
                row_stride: 16,
            },
        )
        .unwrap();
    assert!(
        first
            .torch_cpu_bf16_silu_mul("strided", &strided, &strided)
            .is_err()
    );
    let mut other_profile = PlanBuilder::new_operator_slice(
        source,
        ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
    )
    .unwrap();
    assert!(
        other_profile
            .torch_cpu_bf16_add("no_vllm", &sum, &sum)
            .is_err()
    );
    assert!(
        other_profile
            .torch_cpu_bf16_silu_mul("no_vllm", &sum, &sum)
            .is_err()
    );
}

struct Original {
    source: SourceIdentity,
    rows: [u64; 2],
    packed: Vec<u8>,
    reads: AtomicUsize,
}
impl Original {
    fn new() -> Self {
        Self {
            source: SourceIdentity::for_new_allocation().unwrap(),
            rows: [8, 4],
            packed: vec![0; 64],
            reads: AtomicUsize::new(0),
        }
    }
    fn matrix(&self, name: &str) -> SourceTensor {
        SourceTensor::from_source(self, name, MatrixAxes::OutputInput).unwrap()
    }
}
impl TensorSource for Original {
    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if name == "narrow" {
            return Ok(SourceTensorRef {
                kind: QuantKind::Bf16,
                shape: &[8, 2],
                packed_bytes: &self.packed[..32],
            });
        }
        Ok(SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &self.rows,
            packed_bytes: &self.packed,
        })
    }
}
fn builder(source: SourceIdentity) -> PlanBuilder {
    PlanBuilder::new_operator_slice(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap()
}
fn input(b: &mut PlanBuilder, matrix: SourceTensor) -> PlanValue {
    b.embed_profiled(
        matrix,
        PlanBuilder::inputs().token,
        EmbeddingContract::Bf16OutputInput,
    )
    .unwrap()
}

#[test]
fn source_parameter_revalidates_real_kind_shape_bytes_and_foreign_owner_before_read() {
    let original = Original::new();
    let descriptor = SourceParameter::from_source(&original, "raw.parameter").unwrap();
    assert_eq!(descriptor.kind, QuantKind::Bf16);
    assert_eq!(descriptor.shape, [8, 4]);
    assert_eq!(descriptor.bytes, 64);
    descriptor.validate_source(&original).unwrap();
    let foreign = Original::new();
    assert!(descriptor.validate_source(&foreign).is_err());
    assert_eq!(foreign.reads.load(Ordering::Relaxed), 0);
    for changed in [
        SourceParameter {
            kind: QuantKind::F32,
            ..descriptor.clone()
        },
        SourceParameter {
            shape: vec![4, 8],
            ..descriptor.clone()
        },
        SourceParameter {
            bytes: 63,
            ..descriptor.clone()
        },
        SourceParameter {
            source: SourceIdentity::unbound(),
            ..descriptor.clone()
        },
    ] {
        assert!(changed.validate_source(&original).is_err());
    }
    assert!(SourceParameter::from_source(&original, "").is_err());
}

#[test]
fn explicit_precision_refuses_nonfinite_and_non_bf16_words_without_inference() {
    ActivationPrecision::F32.validate_values(&[1.0001]).unwrap();
    assert!(
        ActivationPrecision::Bf16WidenedF32
            .validate_values(&[1.0001])
            .is_err()
    );
    ActivationPrecision::Bf16WidenedF32
        .validate_values(&[1.0, -0.0, f32::from_bits(0x0001_0000)])
        .unwrap();
    for values in [&[][..], &[f32::NAN][..], &[f32::INFINITY][..]] {
        assert!(ActivationPrecision::F32.validate_values(values).is_err());
        assert!(
            ActivationPrecision::Bf16WidenedF32
                .validate_values(values)
                .is_err()
        );
    }
}

#[test]
fn pure_typed_slice_records_contracts_views_and_no_fake_state_or_frontier() {
    let source = Original::new();
    let mut b = builder(source.source);
    let embedded = input(&mut b, source.matrix("embedding"));
    let viewed = b
        .view(
            &embedded,
            TensorView {
                offset: 0,
                rows: 2,
                width: 1,
                row_stride: 2,
            },
        )
        .unwrap();
    assert_eq!(viewed.precision(), ActivationPrecision::Bf16WidenedF32);
    let packed = b.contiguous(&viewed).unwrap();
    assert_eq!(packed.precision(), embedded.precision());
    let matrix = source.matrix("narrow");
    let projected = b
        .project_profiled(matrix, &packed, ProjectionContract::TorchCpuBf16Avx2)
        .unwrap();
    b.assert_finite(0, &projected).unwrap();
    assert!(b.advance_frontier().is_err());
    let slice = b.finish_operator_slice(projected).unwrap();
    let plan = slice.plan();
    assert!(plan.is_operator_slice());
    assert_eq!(
        plan.numerical_profile(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    );
    assert_eq!(plan.values().len(), plan.precisions().len());
    assert!(
        plan.precisions()
            .iter()
            .all(|precision| *precision == ActivationPrecision::Bf16WidenedF32)
    );
    assert!(plan.required_domains().is_empty() && plan.proposed_domains().is_empty());
    assert_eq!(plan.frontier_delta(), 0);
    assert!(plan.logits().is_none());
    assert!(matches!(
        plan.nodes()[0].operation,
        PlanOperation::TypedEmbed { .. }
    ));
    assert!(matches!(
        plan.nodes()[3].operation,
        PlanOperation::TypedProject {
            contract: ProjectionContract::TorchCpuBf16Avx2,
            ..
        }
    ));
}

#[test]
fn typed_recorder_refuses_foreign_values_wrong_precision_axes_codec_and_legacy_math() {
    let source = Original::new();
    let mut b = builder(source.source);
    let embedded = input(&mut b, source.matrix("embedding"));
    let mut foreign = builder(source.source);
    let other = input(&mut foreign, source.matrix("embedding"));
    let projection = source.matrix("projection");
    assert!(
        b.project_profiled(
            projection.clone(),
            &other,
            ProjectionContract::TorchCpuBf16Avx2
        )
        .is_err()
    );
    assert!(b.project(projection.clone(), &embedded).is_err());
    assert!(b.silu("legacy", &embedded, None).is_err());
    assert!(b.add("legacy", &embedded, &embedded).is_err());
    for matrix in [
        SourceTensor {
            kind: QuantKind::F32,
            ..projection.clone()
        },
        SourceTensor {
            axes: MatrixAxes::InputOutput,
            ..projection.clone()
        },
        SourceTensor {
            source: SourceIdentity::for_new_allocation().unwrap(),
            ..projection.clone()
        },
    ] {
        assert!(
            b.project_profiled(matrix, &embedded, ProjectionContract::TorchCpuBf16Avx2)
                .is_err()
        );
    }
    let mut forged = embedded;
    forged.precision = ActivationPrecision::F32;
    assert!(b.contiguous(&forged).is_err());
    assert!(b.finish_operator_slice(other).is_err());
    let mut full = builder(source.source);
    let value = input(&mut full, source.matrix("embedding"));
    assert!(full.finish(Some(value)).is_err());
}

#[test]
fn packed_bf16_does_not_change_legacy_f32_operations_or_select_cpu64() {
    let source = Original::new();
    let mut old =
        PlanBuilder::new_operator_slice(source.source, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let embedded = old
        .embed(source.matrix("embedding"), PlanBuilder::inputs().token)
        .unwrap();
    assert_eq!(embedded.precision(), ActivationPrecision::F32);
    let projected = old.project(source.matrix("projection"), &embedded).unwrap();
    assert!(
        old.project_profiled(
            source.matrix("projection"),
            &embedded,
            ProjectionContract::TorchCpuBf16Avx2
        )
        .is_err()
    );
    let slice = old.finish_operator_slice(projected).unwrap();
    assert!(matches!(
        slice.plan().nodes()[0].operation,
        PlanOperation::Embed { .. }
    ));
    assert!(matches!(
        slice.plan().nodes()[1].operation,
        PlanOperation::Project { .. }
    ));
    assert!(
        slice
            .plan()
            .precisions()
            .iter()
            .all(|precision| *precision == ActivationPrecision::F32)
    );
}

struct NormSource {
    id: SourceIdentity,
    matrix: Vec<u8>,
    matrix_shape: [u64; 2],
    weight: Vec<u8>,
}
impl NormSource {
    fn new() -> Self {
        Self::with_columns(256)
    }
    fn with_columns(columns: usize) -> Self {
        Self {
            id: SourceIdentity::for_new_allocation().unwrap(),
            matrix: vec![0; 8 * columns * 2],
            matrix_shape: [8, columns as u64],
            weight: vec![0; 256 * 2],
        }
    }
    fn matrix(&self) -> SourceTensor {
        SourceTensor::from_source(self, "embedding", MatrixAxes::OutputInput).unwrap()
    }
}
impl TensorSource for NormSource {
    fn source_identity(&self) -> SourceIdentity {
        self.id
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        match name {
            "embedding" => Ok(SourceTensorRef {
                kind: QuantKind::Bf16,
                shape: &self.matrix_shape,
                packed_bytes: &self.matrix,
            }),
            "input.norm" => Ok(SourceTensorRef {
                kind: QuantKind::Bf16,
                shape: &[256],
                packed_bytes: &self.weight,
            }),
            _ => Err(Error::State("unknown synthetic source role".into())),
        }
    }
}

#[test]
fn ordinary_rms_records_raw_parameter_and_bf16_precision_without_static_gamma_or_state() {
    let source = NormSource::new();
    let raw = SourceParameter::from_source(&source, "input.norm").unwrap();
    let mut b = builder(source.id);
    let embedding = input(&mut b, source.matrix());
    let normalized = b
        .ordinary_rms_profiled(
            raw.clone(),
            &embedding,
            OrdinaryRmsContract::TorchCpuBf16Avx2,
        )
        .unwrap();
    assert_eq!(normalized.precision(), ActivationPrecision::Bf16WidenedF32);
    let slice = b.finish_operator_slice(normalized).unwrap();
    let plan = slice.plan();
    assert_eq!(plan.nodes().len(), 2);
    let PlanOperation::TypedOrdinaryRms {
        parameter,
        contract,
    } = &plan.nodes()[1].operation
    else {
        panic!("missing typed raw-parameter ordinary RMS");
    };
    assert_eq!(parameter, &raw);
    assert_eq!(*contract, OrdinaryRmsContract::TorchCpuBf16Avx2);
    assert!(plan.constants().is_empty());
    assert!(plan.required_domains().is_empty() && plan.proposed_domains().is_empty());
    assert_eq!(plan.frontier_delta(), 0);
    assert!(plan.logits().is_none());
    assert!(
        plan.precisions()
            .iter()
            .all(|p| *p == ActivationPrecision::Bf16WidenedF32)
    );
}

#[test]
fn ordinary_rms_refuses_foreign_parameters_values_precision_and_unqualified_metadata() {
    let source = NormSource::new();
    let parameter = SourceParameter::from_source(&source, "input.norm").unwrap();
    let mut b = builder(source.id);
    let embedding = input(&mut b, source.matrix());
    let original_nodes = b.nodes.len();
    for changed in [
        SourceParameter {
            source: SourceIdentity::for_new_allocation().unwrap(),
            ..parameter.clone()
        },
        SourceParameter {
            name: String::new(),
            ..parameter.clone()
        },
        SourceParameter {
            shape: vec![1, 256],
            ..parameter.clone()
        },
        SourceParameter {
            shape: vec![128],
            bytes: 256,
            ..parameter.clone()
        },
        SourceParameter {
            kind: QuantKind::F16,
            ..parameter.clone()
        },
        SourceParameter {
            bytes: 511,
            ..parameter.clone()
        },
    ] {
        assert!(
            b.ordinary_rms_profiled(changed, &embedding, OrdinaryRmsContract::TorchCpuBf16Avx2)
                .is_err()
        );
        assert_eq!(b.nodes.len(), original_nodes);
    }
    let mut foreign = builder(source.id);
    let other = input(&mut foreign, source.matrix());
    assert!(
        b.ordinary_rms_profiled(
            parameter.clone(),
            &other,
            OrdinaryRmsContract::TorchCpuBf16Avx2
        )
        .is_err()
    );
    let mut forged = embedding;
    forged.precision = ActivationPrecision::F32;
    assert!(
        b.ordinary_rms_profiled(
            parameter.clone(),
            &forged,
            OrdinaryRmsContract::TorchCpuBf16Avx2
        )
        .is_err()
    );
    let mut old =
        PlanBuilder::new_operator_slice(source.id, ProgramNumericalProfile::GgufSourceV1).unwrap();
    let old_value = old
        .embed(source.matrix(), PlanBuilder::inputs().token)
        .unwrap();
    assert!(
        old.ordinary_rms_profiled(
            parameter.clone(),
            &old_value,
            OrdinaryRmsContract::TorchCpuBf16Avx2
        )
        .is_err()
    );
    for width in [256usize, 1024, 2048, 5120] {
        for (kind, bytes) in [(QuantKind::Bf16, width * 2), (QuantKind::F32, width * 4)] {
            let descriptor = SourceParameter {
                kind,
                bytes,
                shape: vec![width as u64],
                ..parameter.clone()
            };
            assert_eq!(
                OrdinaryRmsContract::TorchCpuBf16Avx2
                    .validate(&descriptor, width * 2)
                    .unwrap(),
                width
            );
            assert!(
                OrdinaryRmsContract::TorchCpuBf16Avx2
                    .validate(&descriptor, width - 1)
                    .is_err()
            );
            assert!(
                OrdinaryRmsContract::TorchCpuBf16Avx2
                    .validate(&descriptor, width * 8192)
                    .is_err()
            );
        }
    }
}

macro_rules! unaware_callbacks {
    ($($name:ident($($arg:ident:$kind:ty),*) -> $result:ty;)+) => {$(
        fn $name(&mut self,$($arg:$kind),*) -> $result {
            panic!("unaware backend callback must not run");
        }
    )+};
}
pub(super) struct UnawareBackend;
#[allow(clippy::too_many_arguments)]
impl MathematicalBackend for UnawareBackend {
    type Value = Vec<f32>;
    type Index = usize;
    fn elements(&self, _value: &Self::Value) -> usize {
        panic!("unaware metadata callback");
    }
    unaware_callbacks! {
        embed(_tensor:SourceTensor,_token:usize)->Result<Self::Value>;
        project(_tensor:SourceTensor,_input:&Self::Value)->Result<Self::Value>;
        view(_input:&Self::Value,_view:TensorView)->Result<Self::Value>;
        contiguous(_input:&Self::Value)->Result<Self::Value>;
        rms(_name:&str,_input:&Self::Value,_gamma:SourceVector<'_>,_eps:f32)->Result<Self::Value>;
        grouped_rms(_name:&str,_input:&Self::Value,_gamma:SourceVector<'_>,_groups:usize,_eps:f32)->Result<Self::Value>;
        silu(_name:&str,_input:&Self::Value,_multiplier:Option<&Self::Value>)->Result<Self::Value>;
        add(_name:&str,_left:&Self::Value,_right:&Self::Value)->Result<Self::Value>;
        sigmoid_mul(_name:&str,_value:&Self::Value,_gate:&Self::Value)->Result<Self::Value>;
        conv_step(_name:&str,_domain:ConvDomain,_input:&Self::Value,_weights:SourceVector<'_>)->Result<Self::Value>;
        coefficients(_name:&str,_alpha:&Self::Value,_beta:&Self::Value,_dt:SourceVector<'_>,_a:SourceVector<'_>,_scale:f32)->Result<CoefficientValues<Self::Value>>;
        recurrent_step(_name:&str,_domain:RecurrentDomain,_q:&Self::Value,_k:&Self::Value,_v:&Self::Value,_coefficients:&Self::Value,_width:usize)->Result<Self::Value>;
        validate_kv_prefix(_domain:KvDomain,_position:usize)->Result<()>;
        rope(_name:&str,_q:&Self::Value,_k:&Self::Value,_v:&Self::Value,_position:usize,_geometry:RopeGeometry)->Result<RopeValues<Self::Value>>;
        append_kv(_domain:KvDomain,_k:&Self::Value,_v:&Self::Value)->Result<()>;
        attention_qk(_name:&str,_domain:KvDomain,_query:&Self::Value,_valid_tokens:usize,_geometry:AttentionGeometry,_scale:f32)->Result<QkValues<Self::Value>>;
        attention_softmax(_name:&str,_scores:&Self::Value,_heads:usize,_valid_tokens:usize)->Result<Self::Value>;
        attention_value(_name:&str,_domain:KvDomain,_probabilities:&Self::Value,_valid_tokens:usize,_geometry:AttentionGeometry)->Result<Self::Value>;
        assert_finite(_layer:usize,_value:&Self::Value)->Result<()>;
        observe(_layer:usize,_stage:&'static str,_value:&Self::Value)->Result<()>;
        observe_state(_layer:usize,_stage:&'static str,_domain:StateSlot)->Result<()>;
        advance_frontier()->Result<()>;
    }
}
#[test]
fn unaware_backend_refuses_new_contract_without_legacy_or_metadata_callbacks() {
    let source = NormSource::new();
    let parameter = SourceParameter::from_source(&source, "input.norm").unwrap();
    let mut unaware = UnawareBackend;
    let span = vec![1.0; 16];
    assert!(matches!(
        unaware.torch_cpu_bf16_silu_mul("staged", &span, &span),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        unaware.torch_cpu_bf16_add("residual", &span, &span),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        unaware.ordinary_rms_profiled(
            parameter,
            &vec![1.0; 256],
            OrdinaryRmsContract::TorchCpuBf16Avx2
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn ordinary_rms_requires_explicit_packing_for_valid_geometry_strided_input() {
    let source = NormSource::with_columns(1024);
    let parameter = SourceParameter::from_source(&source, "input.norm").unwrap();
    let contract = OrdinaryRmsContract::TorchCpuBf16Avx2;
    let mut b = builder(source.id);
    let embedding = input(&mut b, source.matrix());
    let strided = b
        .view(
            &embedding,
            TensorView {
                offset: 0,
                rows: 2,
                width: 128,
                row_stride: 256,
            },
        )
        .unwrap();
    assert_eq!(strided.elements, 256);
    assert_eq!(
        contract.validate(&parameter, strided.elements).unwrap(),
        256
    );
    let nodes = b.nodes.len();
    assert!(matches!(
        b.ordinary_rms_profiled(parameter.clone(), &strided, contract),
        Err(Error::State(message)) if message.contains("contiguous")
    ));
    assert_eq!(b.nodes.len(), nodes);
    let packed = b.contiguous(&strided).unwrap();
    let normalized = b
        .ordinary_rms_profiled(parameter, &packed, contract)
        .unwrap();
    let slice = b.finish_operator_slice(normalized).unwrap();
    assert_eq!(slice.plan().nodes().len(), 4);
    assert!(
        slice.plan().required_domains().is_empty() && slice.plan().proposed_domains().is_empty()
    );
    assert_eq!(slice.plan().frontier_delta(), 0);
}

#[test]
fn unaware_backend_causal_conv_refuses_before_any_callback() {
    let mut backend = UnawareBackend;
    let parameter = SourceParameter {
        source: SourceIdentity::for_new_allocation().unwrap(),
        name: "actual.conv.weight".into(),
        kind: QuantKind::Bf16,
        shape: vec![32, 1, 4],
        bytes: 256,
    };
    assert!(matches!(
        backend.causal_conv_silu_profiled(
            parameter,
            &vec![0.0; 32],
            &vec![0.0; 96],
            CausalConvSiluContract::TorchCpuBf16Avx2
        ),
        Err(Error::Unsupported(_))
    ));
}

#[path = "residual_rms/default_backend_tests.rs"]
mod residual_rms_defaults;
#[path = "vllm_native/default_tests.rs"]
mod vllm_native_defaults;

#[path = "vllm_gdn/default_tests.rs"]
mod vllm_gdn_defaults;

#[path = "vllm_model/default_tests.rs"]
mod vllm_model_defaults;
