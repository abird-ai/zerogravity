//! Official metadata and arithmetic fixtures; no trained weights are present.
use serde_json::Value;
use zerogravity_core::{Error, Result};
use zerogravity_model_gemma4::moe::MoeGeometry;
use zerogravity_model_gemma4::variant_26b;

// Unqualified structural arithmetic remains a test-only analytic fixture.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertRoute {
    /// Descending probability order; execution accumulation order is separate.
    pub indices: Vec<usize>,
    /// Selected probabilities normalized first, then learned expert scale.
    pub weights: Vec<f32>,
}

/// Bounded semantic diagnostic for the pinned Transformers routing order.
/// Input is an entire expert-probability row, not raw logits or just top-k.
/// This explicit FP32 left fold is not a qualified Torch sum implementation.
/// A boundary tie has no portable topk identity and is refused explicitly.
fn route_probabilities(
    geometry: &MoeGeometry,
    probabilities: &[f32],
    scales: &[f32],
) -> Result<ExpertRoute> {
    if probabilities.len() != geometry.experts() || scales.len() != geometry.experts() {
        return Err(Error::Malformed("Gemma4 routing row shape mismatch".into()));
    }
    if probabilities
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0)
        || scales.iter().any(|x| !x.is_finite())
    {
        return Err(Error::Malformed(
            "Gemma4 routing requires finite probabilities and scales".into(),
        ));
    }
    let mut indices: Vec<_> = (0..geometry.experts()).collect();
    indices.sort_unstable_by(|a, b| {
        probabilities[*b]
            .total_cmp(&probabilities[*a])
            .then_with(|| a.cmp(b))
    });
    if geometry.selected_experts() < geometry.experts()
        && probabilities[indices[geometry.selected_experts() - 1]]
            == probabilities[indices[geometry.selected_experts()]]
    {
        return Err(Error::Unsupported(
            "Gemma4 top-k boundary tie requires a qualified source tie policy".into(),
        ));
    }
    indices.truncate(geometry.selected_experts());
    let mut denominator = 0.0_f32;
    for &index in &indices {
        denominator += probabilities[index];
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(Error::Malformed(
            "Gemma4 selected probability mass is not positive finite".into(),
        ));
    }
    let weights: Vec<_> = indices
        .iter()
        .map(|&i| (probabilities[i] / denominator) * scales[i])
        .collect();
    if weights.iter().any(|x| !x.is_finite()) {
        return Err(Error::Malformed(
            "Gemma4 scaled routing weights are not finite".into(),
        ));
    }
    Ok(ExpertRoute { indices, weights })
}

// Verbatim config.json from google/gemma-4-26B-A4B at revision
// 24548b62aa021d562695c04aaf7758a1ea47990b, SHA256
// 0364256c5de7863a22b5dd666043d542d42b681c02fd83eb67dc7b483d859e96.
const OFFICIAL_CONFIG: &str = r#"{
  "architectures": [
    "Gemma4ForConditionalGeneration"
  ],
  "audio_config": null,
  "audio_token_id": 258881,
  "boa_token_id": 256000,
  "boi_token_id": 255999,
  "dtype": "bfloat16",
  "eoa_token_id": 258883,
  "eoa_token_index": 258883,
  "eoi_token_id": 258882,
  "image_token_id": 258880,
  "initializer_range": 0.02,
  "model_type": "gemma4",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attention_k_eq_v": true,
    "bos_token_id": 2,
    "dtype": "bfloat16",
    "enable_moe_block": true,
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 2816,
    "hidden_size_per_layer_input": 0,
    "initializer_range": 0.02,
    "intermediate_size": 2112,
    "layer_types": [
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "full_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "full_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "full_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "full_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "full_attention"
    ],
    "max_position_embeddings": 262144,
    "model_type": "gemma4_text",
    "moe_intermediate_size": 704,
    "num_attention_heads": 16,
    "num_experts": 128,
    "num_global_key_value_heads": 2,
    "num_hidden_layers": 30,
    "num_key_value_heads": 8,
    "num_kv_shared_layers": 0,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional"
      },
      "sliding_attention": {
        "rope_theta": 10000.0,
        "rope_type": "default"
      }
    },
    "sliding_window": 1024,
    "tie_word_embeddings": true,
    "top_k_experts": 8,
    "use_bidirectional_attention": "vision",
    "use_cache": true,
    "use_double_wide_mlp": false,
    "vocab_size": 262144,
    "vocab_size_per_layer_input": 262144
  },
  "tie_word_embeddings": true,
  "transformers_version": "5.5.0.dev0",
  "video_token_id": 258884,
  "vision_config": {
    "_name_or_path": "",
    "architectures": null,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "chunk_size_feed_forward": 0,
    "default_output_length": 280,
    "dtype": "bfloat16",
    "global_head_dim": 72,
    "head_dim": 72,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "id2label": {
      "0": "LABEL_0",
      "1": "LABEL_1"
    },
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "is_encoder_decoder": false,
    "label2id": {
      "LABEL_0": 0,
      "LABEL_1": 1
    },
    "max_position_embeddings": 131072,
    "model_type": "gemma4_vision",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "num_key_value_heads": 16,
    "output_attentions": false,
    "output_hidden_states": false,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 10240,
    "problem_type": null,
    "return_dict": true,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "rope_theta": 100.0,
      "rope_type": "default"
    },
    "standardize": true,
    "use_clipped_linears": false
  },
  "vision_soft_tokens_per_image": 280
}
"#;
fn config() -> Value {
    serde_json::from_str(OFFICIAL_CONFIG).unwrap()
}

#[test]
fn official_config_admits_exact_moe_and_two_attention_widths() {
    let value = config();
    let m = variant_26b::validate_config(&value).unwrap();
    assert_eq!(
        (
            m.hidden_size(),
            m.experts(),
            m.selected_experts(),
            m.intermediate_size()
        ),
        (2816, 128, 8, 704)
    );
    let t = &value["text_config"];
    assert_eq!(t["head_dim"], 256);
    assert_eq!(t["global_head_dim"], 512);
    assert_eq!(t["num_key_value_heads"], 8);
    assert_eq!(t["num_global_key_value_heads"], 2);
    assert_eq!(t["attention_k_eq_v"], true);
}

#[test]
fn sparse_and_dense_branches_cannot_be_confused() {
    for (name, replacement) in [
        ("enable_moe_block", serde_json::json!(false)),
        ("num_experts", serde_json::json!(8)),
        ("top_k_experts", serde_json::json!(128)),
        ("moe_intermediate_size", serde_json::json!(2112)),
        ("hidden_size_per_layer_input", serde_json::json!(256)),
        ("num_kv_shared_layers", serde_json::json!(20)),
        ("global_head_dim", serde_json::json!(256)),
        ("attention_k_eq_v", serde_json::json!(false)),
    ] {
        let mut value = config();
        value["text_config"][name] = replacement;
        assert!(variant_26b::validate_config(&value).is_err(), "{name}");
    }
    let mut value = config();
    value["text_config"]["layer_types"][5] = serde_json::json!("sliding_attention");
    assert!(variant_26b::validate_config(&value).is_err());
}

#[test]
fn packed_expert_axes_and_official_tensor_names_are_preserved() {
    let m = variant_26b::validate_config(&config()).unwrap();
    let tensors = m.tensors(29);
    assert_eq!(tensors.len(), 5);
    let expected = [
        ("router.proj.weight", vec![128, 2816]),
        ("router.scale", vec![2816]),
        ("router.per_expert_scale", vec![128]),
        ("experts.gate_up_proj", vec![128, 1408, 2816]),
        ("experts.down_proj", vec![128, 2816, 704]),
    ];
    for (tensor, (suffix, shape)) in tensors.iter().zip(expected) {
        assert_eq!(
            tensor.name,
            format!("model.language_model.layers.29.{suffix}")
        );
        assert_eq!(tensor.shape, shape);
    }
    // Half of the packed gate_up axis is GELU input; the other half is up.
    assert_eq!(tensors[3].shape[1] / 2, m.intermediate_size());
}

#[test]
fn routing_normalizes_before_learned_scale_and_preserves_expert_identity() {
    let m = MoeGeometry::new(2, 5, 2, 3).unwrap();
    let route = route_probabilities(
        &m,
        &[0.125, 0.5, 0.0625, 0.25, 0.0625],
        &[1.0, 0.25, 1.0, 4.0, 1.0],
    )
    .unwrap();
    assert_eq!(route.indices, vec![1, 3]);
    // Analytic FP32 fixtures: (.5/.75)*.25 and (.25/.75)*4.
    assert_eq!(
        route
            .weights
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![0x3e2a_aaab, 0x3faa_aaab]
    );
    assert!(route.weights.iter().sum::<f32>() > 1.0); // No second normalization.
    // Learned scales change output coefficients, never the selected IDs.
    let negative = route_probabilities(
        &m,
        &[0.125, 0.5, 0.0625, 0.25, 0.0625],
        &[99.0, -0.25, 99.0, 0.0, 99.0],
    )
    .unwrap();
    assert_eq!(negative.indices, route.indices);
    assert_eq!(negative.weights[0].to_bits(), 0xbe2a_aaab);
    assert_eq!(negative.weights[1].to_bits(), 0);
}

#[test]
fn ambiguous_boundary_and_invalid_rows_refuse_without_partial_output() {
    let m = MoeGeometry::new(2, 4, 2, 3).unwrap();
    assert!(matches!(
        route_probabilities(&m, &[0.5, 0.25, 0.25, 0.0], &[1.0; 4]),
        Err(Error::Unsupported(_))
    ));
    for p in [
        [0.0; 4],
        [f32::NAN, 0.5, 0.25, 0.25],
        [-0.1, 0.5, 0.25, 0.35],
        [f32::INFINITY, 0.0, 0.0, 0.0],
    ] {
        assert!(route_probabilities(&m, &p, &[1.0; 4]).is_err());
    }
    assert!(route_probabilities(&m, &[0.5, 0.25, 0.125, 0.125], &[1.0; 3]).is_err());
    assert!(
        route_probabilities(
            &m,
            &[0.5, 0.25, 0.125, 0.125],
            &[1.0, f32::INFINITY, 1.0, 1.0]
        )
        .is_err()
    );
    assert!(MoeGeometry::new(0, 128, 8, 704).is_err());
    assert!(MoeGeometry::new(2816, 128, 129, 704).is_err());
    assert!(MoeGeometry::new(2816, usize::MAX, 8, 704).is_err());
}
