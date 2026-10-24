//! Synthetic original full-header sources; no trained27B parity claim.
use super::*;
use crate::accelerator::program::{
    EmbeddingContract, MathematicalBackend, OrdinaryRmsContract, PlanOperation, TensorView,
};
use crate::operator_slice::OperatorSliceSource;
use crate::original_binding::{OriginalWindowOperators, QwenParameterRole};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowOpenLimits};

fn windows(fixture: &Fixture, loaded: &Loaded) -> (VerifiedTensorSource, TransferAdmission) {
    let path = fixture.0.join("model.safetensors");
    let bytes = fs::read(&path).unwrap();
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
    let map = artifact
        .info
        .tensors
        .iter()
        .map(|info| (info.name.clone(), "model.safetensors"))
        .collect::<BTreeMap<_, _>>();
    let size = artifact
        .info
        .tensors
        .iter()
        .map(|info| info.byte_len)
        .sum::<u64>();
    let index =
        serde_json::to_vec(&json!({"metadata":{"total_size":size},"weight_map":map})).unwrap();
    assert_eq!(digest(&index), loaded.index_sha256);
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "window-operator-tests").unwrap();
    arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
    let admission = TransferAdmission {
        arbiter,
        resource: key,
        owner: ResourceOwner::new("original", "operator-window").unwrap(),
    };
    let source = SourceWeights::prepare_verified_windows(
        &index,
        &loaded.index_sha256,
        vec![
            ExpectedSourceShard::new(
                "model.safetensors".into(),
                artifact,
                bytes.len() as u64,
                digest(&bytes),
            )
            .unwrap(),
        ],
        WindowOpenLimits {
            max_source_bytes: 16 * 1024 * 1024,
            max_metadata_bytes: 32 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 200,
            scan_chunk_bytes: 64 * 1024,
        },
        &admission,
    )
    .unwrap()
    .open()
    .unwrap();
    (source, admission)
}
fn reserved(admission: &TransferAdmission) -> u64 {
    admission
        .arbiter
        .snapshot(&admission.resource)
        .unwrap()
        .reserved
}
fn bind_windows(source: VerifiedTensorSource, loaded: &Loaded) -> OriginalWindowOperators {
    OriginalWindowOperators::bind(
        source,
        &loaded.config,
        &digest(&loaded.config),
        &loaded.index_sha256,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap()
}
fn same(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    assert!(
        expected
            .iter()
            .zip(actual)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

#[test]
fn window_three_nodes_match_owned_source_and_retain_buffers_after_callers_drop() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, true, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let before_bind = reserved(&admission);
    assert!(
        OriginalWindowOperators::bind(
            source.clone(),
            &loaded.config,
            &digest(&loaded.config),
            &loaded.index_sha256,
            QwenNumericalProfile::GgufSourceV1
        )
        .is_err()
    );
    assert_eq!(reserved(&admission), before_bind);
    let operators = bind_windows(source.clone(), &loaded);
    assert_eq!(
        operators.binding().text_tensor_count(),
        QwenTextConfig::from_json(&loaded.config)
            .unwrap()
            .tensor_specs()
            .unwrap()
            .len()
    );
    let plan = operators.record().unwrap();
    assert_eq!(plan.plan().nodes().len(), 3);
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(matches!(
        plan.plan().nodes()[0].operation,
        PlanOperation::TypedEmbed { .. }
    ));
    assert!(matches!(
        plan.plan().nodes()[1].operation,
        PlanOperation::TypedOrdinaryRms { .. }
    ));
    assert!(matches!(
        plan.plan().nodes()[2].operation,
        PlanOperation::TypedProject { .. }
    ));
    let (model, _, _) = bind(loaded);
    assert!(model.forward_plan(8, true, true).is_err());
    assert_eq!(model.initialized_cpu_state_bytes(), 0);
    let before = reserved(&admission);
    let output = operators.execute(11, None).unwrap();
    let expected = model
        .execute_embedding_input_norm_projection_slice(11)
        .unwrap();
    same(expected.embedding(), output.embedding().unwrap());
    same(expected.normalized(), output.normalized().unwrap());
    same(expected.projected(), output.projected().unwrap());
    assert_eq!(output.projected().unwrap(), &[512.0; 8]);
    assert_eq!(reserved(&admission), before + 3 * 1024 * 1024);
    assert_eq!(
        admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .per_class[&zerogravity_core::resource_arbiter::AllocationClass::Output],
        3 * 64 * 1024
    );
    let identity = source.source_identity();
    drop(operators);
    drop(source);
    drop(plan);
    assert_eq!(output.source_identity(), identity);
    assert_eq!(reserved(&admission), 35 * 1024 * 1024);
    same(expected.projected(), output.projected().unwrap());
    drop(output);
    assert_eq!(reserved(&admission), 0);
    assert!(
        !admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .fenced
    );
}

#[test]
fn complete_namespace_and_raw_norm_codec_are_checked_without_model_permits() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |header| {
        header["model.language_model.layers.2.mlp.down_proj.weight"]["shape"] = json!([8, 256]);
    });
    let (source, admission) = windows(&fixture, &loaded);
    let before = reserved(&admission);
    assert!(
        OriginalWindowOperators::bind(
            source.clone(),
            &loaded.config,
            &digest(&loaded.config),
            &loaded.index_sha256,
            QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
        )
        .is_err()
    );
    assert_eq!(reserved(&admission), before);
    drop(source);
    assert_eq!(reserved(&admission), 0);
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, true, 256, true, |header| {
        header["model.language_model.layers.0.input_layernorm.weight"]["dtype"] = json!("F16");
    });
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let before = reserved(&admission);
    assert!(operators.record().is_err());
    assert!(operators.execute(11, None).is_err());
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn cancellation_pool_exhaustion_and_bad_token_release_every_private_buffer() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, true, 256, false, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let before = reserved(&admission);
    assert!(operators.execute(11, Some(&AtomicBool::new(true))).is_err());
    assert!(operators.execute(u32::MAX, None).is_err());
    assert_eq!(reserved(&admission), before);
    let mut pressure = Vec::new();
    loop {
        let remaining = admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .remaining;
        if remaining == 0 {
            break;
        }
        pressure.push(
            source
                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    assert!(operators.execute(11, None).is_err());
    assert_eq!(reserved(&admission), 128 * 1024 * 1024);
    drop(pressure);
    assert_eq!(reserved(&admission), before);
    drop(operators.execute(11, None).unwrap());
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn views_keep_original_leases_and_forbid_bare_vec_escape_or_foreign_activations() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, true, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let mut math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
    let tensor = operators
        .matrix_descriptor(QwenMatrixRole::Embedding)
        .unwrap();
    let embedding = math
        .embed_profiled(tensor.clone(), 11, EmbeddingContract::Bf16OutputInput)
        .unwrap();
    let before = reserved(&admission);
    let view = math
        .view(
            &embedding,
            TensorView {
                offset: 0,
                width: 32,
                rows: 2,
                row_stride: 64,
            },
        )
        .unwrap();
    assert_eq!(reserved(&admission), before);
    assert!(view.clone().into_vec().is_err());
    let packed = math.contiguous(&view).unwrap();
    assert_eq!(reserved(&admission), before + 1024 * 1024);
    assert_eq!(packed.as_slice().unwrap(), &[1.0; 64]);
    let mut forged = tensor.clone();
    forged.columns = 128;
    assert!(
        math.embed_profiled(forged, 11, EmbeddingContract::Bf16OutputInput)
            .is_err()
    );
    let (foreign_source, foreign_admission) = windows(&fixture, &loaded);
    let foreign = bind_windows(foreign_source.clone(), &loaded);
    let mut other = crate::host_math::HostMath::for_window_operator_slice(&foreign, None, None);
    assert!(
        other
            .project_profiled(
                foreign
                    .matrix_descriptor(QwenMatrixRole::GdnQkv(0))
                    .unwrap(),
                &embedding,
                ProjectionContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    drop(foreign);
    drop(foreign_source);
    assert_eq!(reserved(&foreign_admission), 0);
    drop(operators);
    drop(source);
    drop(embedding);
    drop(view);
    assert_eq!(reserved(&admission), 33 * 1024 * 1024);
    assert_eq!(packed.as_slice().unwrap(), &[1.0; 64]);
    drop(packed);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn unaware_provider_never_gets_partial_source_or_legacy_norm_fallback() {
    struct Unaware(AtomicBool);
    impl crate::accelerator::MatVecKernel for Unaware {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            panic!("window provider must refuse before callbacks");
        }
    }
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, true, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let provider = Unaware(AtomicBool::new(false));
    let mut ordinary =
        crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
    let input = ordinary
        .embed_profiled(
            operators
                .matrix_descriptor(QwenMatrixRole::Embedding)
                .unwrap(),
            11,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let before = reserved(&admission);
    let mut unaware =
        crate::host_math::HostMath::for_window_operator_slice(&operators, None, Some(&provider));
    assert!(
        unaware
            .embed_profiled(
                operators
                    .matrix_descriptor(QwenMatrixRole::Embedding)
                    .unwrap(),
                11,
                EmbeddingContract::Bf16OutputInput
            )
            .is_err()
    );
    assert!(
        unaware
            .ordinary_rms_profiled(
                operators
                    .parameter_descriptor(QwenParameterRole::InputNorm(0))
                    .unwrap(),
                &input,
                OrdinaryRmsContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    assert!(
        unaware
            .project_profiled(
                operators
                    .matrix_descriptor(QwenMatrixRole::GdnQkv(0))
                    .unwrap(),
                &input,
                ProjectionContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    assert!(!provider.0.load(Ordering::Relaxed));
    assert_eq!(reserved(&admission), before);
    drop(input);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[path = "attention_slice_tests.rs"]
mod attention;

#[path = "gdn_conv_window_tests.rs"]
mod gdn_conv;

#[path = "operator_slice_window_roles_tests.rs"]
mod pure_roles;

#[path = "attention_tail_window_tests.rs"]
mod attention_tail;
#[path = "full_window_cpu_tests.rs"]
mod full_window_cpu;
#[path = "output_tail_window_tests.rs"]
mod output_tail;

#[test]
fn window_model_metadata_retains_one_original_issuer_and_stays_execution_closed() {
    use crate::original_binding::OriginalWindowModelSource;
    use zerogravity_artifacts::model::qwen_owned::QwenOriginalWindowText;
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, true, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let owner = Arc::new(
        QwenOriginalWindowText::bind(
            source.clone(),
            &loaded.config,
            &digest(&loaded.config),
            &loaded.index_sha256,
        )
        .unwrap(),
    );
    let weak = Arc::downgrade(&owner);
    let model = OriginalWindowModelSource::from_owner(owner.clone()).unwrap();
    assert!(Arc::ptr_eq(model.owner(), &owner));
    assert!(model.source().same_issuer(&source));
    assert_ne!(
        loaded.source.source_identity(),
        model.source_identity(),
        "equal source bytes do not mint the same issuer"
    );
    let view = crate::QwenModelMetadata::VerifiedWindows(&model);
    assert!(Arc::ptr_eq(view.original_window_owner().unwrap(), &owner));
    assert_eq!(view.source_identity(), source.source_identity());
    assert_eq!(view.geometry().layers.len(), view.state_specs().len());
    assert_eq!(
        view.program_numerical_profile(),
        crate::accelerator::program::ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1
    );
    assert!(matches!(
        view.require_execution_profile(),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(view.legacy_loaded(), Err(Error::Unsupported(_))));
    for layer in 0..3 {
        let matrix = view
            .matrix_descriptor(QwenMatrixRole::GdnQkv(layer))
            .unwrap();
        let gamma = view
            .parameter_descriptor(QwenParameterRole::InputNorm(layer))
            .unwrap();
        assert_eq!(matrix.source, source.source_identity());
        assert_eq!(gamma.source, source.source_identity());
        model
            .validate_matrix(QwenMatrixRole::GdnQkv(layer), &matrix)
            .unwrap();
        model
            .validate_parameter(QwenParameterRole::InputNorm(layer), &gamma)
            .unwrap();
    }
    drop(owner);
    drop(source);
    assert!(weak.upgrade().is_some());
    assert!(reserved(&admission) > 0);
    drop(model);
    assert!(weak.upgrade().is_none());
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn equal_original_window_files_cannot_cross_model_role_issuers() {
    use crate::original_binding::OriginalWindowModelSource;
    use zerogravity_artifacts::model::qwen_owned::QwenOriginalWindowText;
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, true, 256, true, |_| {});
    let (first, a) = windows(&fixture, &loaded);
    let (second, b) = windows(&fixture, &loaded);
    assert_eq!(
        first.verified_bundle_sha256(),
        second.verified_bundle_sha256()
    );
    assert!(!first.same_issuer(&second));
    let bind = |source| {
        OriginalWindowModelSource::from_owner(Arc::new(
            QwenOriginalWindowText::bind(
                source,
                &loaded.config,
                &digest(&loaded.config),
                &loaded.index_sha256,
            )
            .unwrap(),
        ))
        .unwrap()
    };
    let first = bind(first);
    let second = bind(second);
    let matrix = first.matrix_descriptor(QwenMatrixRole::FfnGate(1)).unwrap();
    let gamma = first
        .parameter_descriptor(QwenParameterRole::FfnNorm(2))
        .unwrap();
    assert!(
        second
            .validate_matrix(QwenMatrixRole::FfnGate(1), &matrix)
            .is_err()
    );
    assert!(
        second
            .validate_parameter(QwenParameterRole::FfnNorm(2), &gamma)
            .is_err()
    );
    let mut wrong = matrix.clone();
    wrong.name = second
        .matrix_descriptor(QwenMatrixRole::FfnUp(1))
        .unwrap()
        .name;
    assert!(
        first
            .validate_matrix(QwenMatrixRole::FfnGate(1), &wrong)
            .is_err()
    );
    let mut wrong = gamma.clone();
    wrong.bytes -= 1;
    assert!(
        first
            .validate_parameter(QwenParameterRole::FfnNorm(2), &wrong)
            .is_err()
    );
    drop(first);
    drop(second);
    assert_eq!(reserved(&a), 0);
    assert_eq!(reserved(&b), 0);
}

#[test]
fn window_model_role_metadata_pressure_refuses_before_new_payload_allocation() {
    use crate::original_binding::OriginalWindowModelSource;
    use zerogravity_artifacts::model::qwen_owned::QwenOriginalWindowText;
    use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, true, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let owner = Arc::new(
        QwenOriginalWindowText::bind(
            source.clone(),
            &loaded.config,
            &digest(&loaded.config),
            &loaded.index_sha256,
        )
        .unwrap(),
    );
    let before = reserved(&admission);
    let remaining = 128 * 1024 * 1024 - before;
    let hold = admission
        .arbiter
        .admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: remaining,
            }],
        )
        .unwrap();
    let full = reserved(&admission);
    assert!(matches!(
        OriginalWindowModelSource::from_owner(owner.clone()),
        Err(Error::Limit(_))
    ));
    assert_eq!(reserved(&admission), full);
    drop(hold);
    assert_eq!(reserved(&admission), before);
    drop(owner);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
