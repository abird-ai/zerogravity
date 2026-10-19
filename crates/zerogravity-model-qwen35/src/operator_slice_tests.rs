//! Synthetic source ownership/type checks; no trained model or GPU proof.

use super::*;
use crate::LoadedQwenModel;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use zerogravity_artifacts::SourceWeights;
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::model::qwen::QwenTextConfig;
use zerogravity_artifacts::source_weights::{ExpectedSourceShard, SourceOpenLimits};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-hf-typed-slice-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn config(tied: bool) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings":tied,
        "text_config":{
            "model_type":"qwen3_5_text","num_hidden_layers":4,
            "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
            "hidden_size":4,"head_dim":4,"num_attention_heads":2,
            "num_key_value_heads":1,"vocab_size":16,"max_position_embeddings":1024,
            "full_attention_interval":4,"linear_conv_kernel_dim":4,
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":2,"linear_value_head_dim":2,
            "intermediate_size":8,"hidden_act":"silu","attention_bias":false,
            "attention_dropout":0.0,"attn_output_gate":true,"output_gate_type":"swish",
            "rms_norm_eps":0.000001,"tie_word_embeddings":tied,
            "partial_rotary_factor":0.5,
            "rope_parameters":{"rope_type":"default","partial_rotary_factor":0.5,"rope_theta":10000.0}
        }
    })).unwrap()
}
struct Loaded {
    source: SourceWeights,
    config: Vec<u8>,
    index_sha256: String,
    admission: TransferAdmission,
    key: PhysicalDeviceKey,
}
fn source(fixture: &Fixture, tied: bool, mutate: impl FnOnce(&mut Value)) -> Loaded {
    let cfg = config(tied);
    let specs = QwenTextConfig::from_json(&cfg)
        .unwrap()
        .tensor_specs()
        .unwrap();
    let mut header = BTreeMap::new();
    let mut map = BTreeMap::new();
    let mut payload = Vec::new();
    for spec in specs {
        let kind = if spec.shape.len() == 1 && !spec.name.ends_with("dt_bias") {
            "F32"
        } else {
            "BF16"
        };
        let start = payload.len();
        for _ in 0..spec.shape.iter().product::<u64>() {
            match kind {
                "F32" => payload.extend_from_slice(&0.25f32.to_le_bytes()),
                "F16" => payload.extend_from_slice(&0x3400u16.to_le_bytes()),
                "BF16" => {
                    let bits = if spec.name == "lm_head.weight" {
                        0xbf80u16
                    } else {
                        0x3f80
                    };
                    payload.extend_from_slice(&bits.to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        header.insert(
            spec.name.clone(),
            json!({
                "dtype":kind,"shape":spec.shape,"data_offsets":[start,payload.len()]
            }),
        );
        map.insert(spec.name, "model.safetensors");
    }
    let mut header = serde_json::to_value(header).unwrap();
    mutate(&mut header);
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(&payload);
    let path = fixture.0.join("model.safetensors");
    fs::write(&path, &bytes).unwrap();
    let index = serde_json::to_vec(&json!({
        "metadata":{"total_size":payload.len()},"weight_map":map
    }))
    .unwrap();
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
    let shard = ExpectedSourceShard::new(
        "model.safetensors".into(),
        artifact,
        bytes.len() as u64,
        digest(&bytes),
    )
    .unwrap();
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("synthetic-host", "hf-binding").unwrap();
    arbiter.register(key.clone(), 64 * 1024 * 1024).unwrap();
    let admission = TransferAdmission {
        arbiter,
        resource: key.clone(),
        owner: ResourceOwner::new("qwen", "source-binding").unwrap(),
    };
    let index_sha256 = digest(&index);
    let source = SourceWeights::prepare_verified_shards(
        &index,
        &index_sha256,
        vec![shard],
        SourceOpenLimits {
            max_owned_bytes: 8 * 1024 * 1024,
            max_metadata_bytes: 32 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 200,
        },
        &admission,
    )
    .unwrap()
    .materialize()
    .unwrap();
    Loaded {
        source,
        config: cfg,
        index_sha256,
        admission,
        key,
    }
}
fn bind(
    loaded: Loaded,
) -> (
    std::sync::Arc<LoadedQwenModel>,
    TransferAdmission,
    PhysicalDeviceKey,
) {
    let model = LoadedQwenModel::from_original_source(
        loaded.source,
        &loaded.config,
        &digest(&loaded.config),
        &loaded.index_sha256,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    (model, loaded.admission, loaded.key)
}

#[test]
fn original_typed_embedding_projection_uses_real_roles_without_opening_hf_sessions() {
    use crate::accelerator::program::{ActivationPrecision, PlanOperation};
    for tied in [false, true] {
        let fixture = Fixture::new();
        let (model, admission, key) = bind(source(&fixture, tied, |_| {}));
        let source = model.source_identity();
        let role = QwenMatrixRole::GdnQkv(0);
        let tensor = model.matrix_descriptor(role).unwrap();
        let pointer = model.tensor_bytes(&tensor.name).unwrap().as_ptr();
        let reserved = admission.arbiter.snapshot(&key).unwrap().reserved;
        assert_eq!(model.initialized_cpu_state_bytes(), 0);
        let output = model.execute_embedding_projection_slice(11, role).unwrap();
        assert_eq!(output.source_identity(), source);
        assert_eq!(output.precision(), ActivationPrecision::Bf16WidenedF32);
        assert_eq!(output.embedding(), &[1.0; 4]);
        assert_eq!(output.projected(), &[4.0; 8]);
        let vocabulary = model
            .execute_embedding_projection_slice(0, QwenMatrixRole::Output)
            .unwrap();
        assert_eq!(
            vocabulary.projected(),
            &vec![if tied { 4.0 } else { -4.0 }; 16]
        );
        let recorded = model.embedding_projection_slice(role).unwrap();
        assert_eq!(recorded.plan().source(), source);
        assert_eq!(recorded.plan().nodes().len(), 2);
        assert!(matches!(
            recorded.plan().nodes()[0].operation,
            PlanOperation::TypedEmbed { .. }
        ));
        assert!(matches!(
            recorded.plan().nodes()[1].operation,
            PlanOperation::TypedProject { .. }
        ));
        assert!(
            recorded
                .plan()
                .precisions()
                .iter()
                .all(|precision| *precision == ActivationPrecision::Bf16WidenedF32)
        );
        assert!(recorded.plan().required_domains().is_empty());
        assert_eq!(recorded.plan().frontier_delta(), 0);
        assert_eq!(model.tensor_bytes(&tensor.name).unwrap().as_ptr(), pointer);
        assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, reserved);
        assert_eq!(model.initialized_cpu_state_bytes(), 0);
        assert!(model.new_session().is_err());
        assert!(model.forward_plan(256, true, false).is_err());
        assert!(model.execute_embedding_projection_slice(16, role).is_err());
        assert!(
            model
                .embedding_projection_slice(QwenMatrixRole::GdnQkv(3))
                .is_err()
        );
        drop((recorded, vocabulary, output, model));
        assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, 0);
    }
}

#[test]
fn typed_host_values_preserve_source_precision_and_refuse_foreign_or_legacy_paths() {
    use crate::accelerator::program::{ActivationPrecision, MathematicalBackend, TensorView};
    let fixture = Fixture::new();
    let (first, _, _) = bind(source(&fixture, true, |_| {}));
    let second_fixture = Fixture::new();
    let (second, _, _) = bind(source(&second_fixture, true, |_| {}));
    let mut math = crate::host_math::HostMath::for_operator_slice(&first, None);
    let embedded = math
        .embed_profiled(
            first.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            0,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    assert!(math.add("legacy", &embedded, &embedded).is_err());
    assert!(math.silu("legacy", &embedded, None).is_err());
    assert!(math.advance_frontier().is_err());
    let view = math.view(&embedded, TensorView::vector(0, 4)).unwrap();
    let packed = math.contiguous(&view).unwrap();
    assert_eq!(math.precision(&packed), ActivationPrecision::Bf16WidenedF32);
    let projected = math
        .project_profiled(
            first.matrix_descriptor(QwenMatrixRole::GdnQkv(0)).unwrap(),
            &packed,
            ProjectionContract::TorchCpuBf16Avx2,
        )
        .unwrap();
    assert_eq!(projected.into_vec().unwrap(), [4.0; 8]);
    let mut foreign = crate::host_math::HostMath::for_operator_slice(&second, None);
    assert!(
        foreign
            .project_profiled(
                second.matrix_descriptor(QwenMatrixRole::GdnQkv(0)).unwrap(),
                &embedded,
                ProjectionContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    assert_eq!(first.initialized_cpu_state_bytes(), 0);
    assert_eq!(second.initialized_cpu_state_bytes(), 0);
}

#[test]
fn unaware_physical_provider_refuses_before_embedding_or_projection_callback() {
    use crate::accelerator::program::MathematicalBackend;
    struct Unaware(std::sync::atomic::AtomicBool);
    impl crate::accelerator::MatVecKernel for Unaware {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(crate::accelerator::RegionExecution::Unsupported)
        }
    }
    let fixture = Fixture::new();
    let (model, _, _) = bind(source(&fixture, true, |_| {}));
    let provider = Unaware(std::sync::atomic::AtomicBool::new(false));
    let mut math = crate::host_math::HostMath::for_operator_slice(&model, Some(&provider));
    assert!(
        math.embed_profiled(
            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            0,
            EmbeddingContract::Bf16OutputInput
        )
        .is_err()
    );
    assert!(!provider.0.load(Ordering::Relaxed));
    assert_eq!(model.initialized_cpu_state_bytes(), 0);
}
