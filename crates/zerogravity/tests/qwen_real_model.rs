#![cfg(all(feature = "cpu", feature = "model-qwen35"))]

//! Optional real checkpoint cross-check for the source-backed Qwen math layer.
//!
//! Run with:
//! cargo test --locked --offline --test qwen_real_model -- --ignored --nocapture
//!
//! Does not download checkpoints or execute the independent model-server
//! reference. Fixture-generation script has already verified exact model SHA.

mod common;

#[test]
#[ignore = "requires both locally pinned Qwen models; records the one native schedule"]
fn actual_qwen_symbolic_forward_covers_source_ops_and_private_domains() {
    use zerogravity::accelerator::program::{ParameterSlot, PlanOperation, ValueShape};
    use zerogravity::qwen35::LoadedQwenModel;
    for model in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(model);
        let loaded =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let embedding_bytes = loaded.tensor_bytes("token_embd.weight").unwrap();
        assert_eq!(
            embedding_bytes.len(),
            loaded.tensor_info("token_embd.weight").unwrap().byte_len as usize
        );
        assert!(std::ptr::eq(
            embedding_bytes.as_ptr(),
            loaded.tensor_bytes("token_embd.weight").unwrap().as_ptr()
        ));
        assert!(loaded.tensor_bytes("missing.tensor").is_err());
        let sibling = loaded.new_session().unwrap();
        assert_eq!(
            sibling.loaded_model().source_identity(),
            loaded.source_identity()
        );
        assert!(std::ptr::eq(
            embedding_bytes.as_ptr(),
            sibling
                .loaded_model()
                .tensor_bytes("token_embd.weight")
                .unwrap()
                .as_ptr()
        ));
        assert!(loaded.forward_plan(0, true, false).is_err());
        assert!(
            loaded
                .forward_plan(
                    loaded.geometry().max_position_embeddings as usize + 1,
                    true,
                    false
                )
                .is_err()
        );
        for need_logits in [false, true] {
            let plan = loaded.forward_plan(257, need_logits, false).unwrap();
            assert_eq!(plan.source(), loaded.source_identity());
            assert_eq!(plan.required_domains().len(), 42);
            assert_eq!(plan.proposed_domains().len(), 42);
            assert!(
                plan.proposed_domains()
                    .iter()
                    .all(|state| state.ordinal == 1)
            );
            assert_eq!(plan.frontier_delta(), 1);
            let mut projects = 0;
            let mut embeddings = 0;
            let mut effects = 0;
            let mut qk = 0;
            let mut softmax = 0;
            let mut values = 0;
            let mut silu = 0;
            for node in plan.nodes() {
                effects += usize::from(node.state.is_some());
                match &node.operation {
                    PlanOperation::Embed { tensor, token } => {
                        embeddings += 1;
                        assert_eq!(token.0, ParameterSlot::Token);
                        assert_eq!(tensor.source, loaded.source_identity());
                    }
                    PlanOperation::Project { tensor } => {
                        projects += 1;
                        assert_eq!(tensor.source, loaded.source_identity());
                    }
                    PlanOperation::AttentionQk {
                        state,
                        valid_tokens,
                        ..
                    } => {
                        qk += 1;
                        assert_eq!(state.ordinal, 1);
                        assert_eq!(valid_tokens.0, ParameterSlot::ValidTokens);
                        assert_eq!(
                            plan.values()[node.outputs[0].0 as usize],
                            ValueShape::CausalRows {
                                heads: 8,
                                capacity: 257
                            }
                        );
                    }
                    PlanOperation::AttentionSoftmax { valid_tokens, .. } => {
                        softmax += 1;
                        assert_eq!(valid_tokens.0, ParameterSlot::ValidTokens);
                    }
                    PlanOperation::AttentionValue {
                        state,
                        valid_tokens,
                        padding,
                        ..
                    } => {
                        values += 1;
                        assert_eq!(state.ordinal, 1);
                        assert_eq!(valid_tokens.0, ParameterSlot::ValidTokens);
                        assert_eq!(*padding, 256);
                    }
                    PlanOperation::Silu { .. } => silu += 1,
                    PlanOperation::Observe { .. } | PlanOperation::ObserveState { .. } => {
                        panic!("capture=false retained diagnostic observations")
                    }
                    _ => {}
                }
            }
            assert_eq!(embeddings, 1);
            assert_eq!(projects, 186 + usize::from(need_logits));
            assert_eq!(effects, 42);
            assert_eq!(qk, 6);
            assert_eq!(softmax, 6);
            assert_eq!(values, 6);
            assert_eq!(silu, 60);
            assert_eq!(plan.logits().is_some(), need_logits);
            let mut session = loaded.new_session().unwrap();
            session.prefill_chunk_without_logits(&[11, 353]).unwrap();
            let again = loaded.forward_plan(257, need_logits, false).unwrap();
            assert_eq!(
                format!("{:?}", plan.nodes()),
                format!("{:?}", again.nodes()),
                "live session/root/position entered compiled source schedule"
            );
        }
        let capture = loaded.forward_plan(1, true, true).unwrap();
        let observations = capture
            .nodes()
            .iter()
            .filter(|node| {
                matches!(
                    node.operation,
                    PlanOperation::Observe { .. } | PlanOperation::ObserveState { .. }
                )
            })
            .count();
        let expected = if cfg!(feature = "diagnostic-gdn-trace") {
            477
        } else {
            146
        };
        assert_eq!(observations, expected);
        for capture in [false, true] {
            let full = loaded.forward_plan(256, true, capture).unwrap();
            let state = loaded.forward_plan(256, false, capture).unwrap();
            assert_eq!(
                format!("{:?}", full.state_nodes()),
                format!("{:?}", state.nodes()),
                "state prefix differs from the same schedule without logits"
            );
            assert_eq!(state.state_nodes().len(), state.nodes().len());
            assert_eq!(full.proposed_domains(), state.proposed_domains());
            assert_eq!(full.required_domains(), state.required_domains());
            assert!(
                full.nodes()[full.state_nodes().len()..]
                    .iter()
                    .all(|n| n.state.is_none())
            );
        }
    }
}

use zerogravity::cpu::source::CpuWeights;

fn float_at(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(buf[offset..offset + 4].try_into().expect("f32 fixture"))
}

fn verify_projection(root: &str, expected_cols: usize) {
    let dir = common::workspace_path("target/models").join(root);
    let weights = CpuWeights::open(&dir.join("model.gguf")).expect("known Qwen model");
    let embedding = weights
        .embed(0, "token_embd.weight")
        .expect("embedding token0");
    assert_eq!(embedding.len(), expected_cols);
    let projection = weights
        .matvec_fp32_reference("blk.0.attn_gate.weight", &embedding)
        .expect("run exact dequantized F32 mathematical projection");
    let bytes = std::fs::read(dir.join("gpu-q4-fixture.bin")).expect("model Q4 fixture");
    assert_eq!(&bytes[..4], b"ZQ4P");
    let cols = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    assert_eq!(cols, expected_cols);
    assert_eq!(rows, 16);
    let oracle_offset = 12 + rows * (cols / 32) * 4 + cols * 4;
    for (i, &actual) in projection.iter().enumerate().take(rows) {
        let expected = float_at(&bytes, oracle_offset + i * 4);
        let error = (actual - expected).abs();
        assert!(
            error < 2.0e-4 + expected.abs() * 1.0e-3,
            "model={root} row={i}: Rust CPU GEMV {actual} != independent fixture {expected} (error {error})"
        );
    }
}

#[test]
#[ignore = "requires local SHA-verified Qwen3.5 0.8B/2B GGUF and source projection fixtures"]
fn both_real_qwen_models_cpu_gemv_match_gpu_source_oracles() {
    verify_projection("qwen35-0.8b", 1024);
    verify_projection("qwen35-2b", 2048);
}

#[test]
#[ignore = "requires local Qwen3.5 0.8B GGUF/config and official tokenizer; exercises true GPU-independent CPU state changes"]
fn real_qwen08_cpu_snapshot_replay_transaction_and_foreign_rejection() {
    use zerogravity::qwen35::QwenDenseCpu;
    let dir = common::workspace_path("target/models/qwen35-0.8b");
    let mut engine = QwenDenseCpu::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
    let baseline = engine.snapshot();
    let baseline_hash = engine.state_fingerprint();
    let base_root = engine.root_id();
    assert_eq!(engine.position(), 0);
    let (out_a, prepared_a) = engine.next_logits(11).unwrap();
    assert_eq!(engine.position(), 0);
    assert_eq!(engine.root_id(), base_root);
    assert_eq!(engine.state_fingerprint(), baseline_hash);
    assert!(out_a.iter().all(|v| v.is_finite()));
    engine.commit(prepared_a.clone()).unwrap();
    assert_eq!(engine.position(), 1);
    assert_eq!(engine.root_id(), base_root + 1);
    assert!(
        engine.commit(prepared_a).is_err(),
        "stale draft re-committed"
    );
    let published = engine.snapshot();
    let committed_hash = engine.state_fingerprint();
    let (out_b, prepared_b) = engine.next_logits(13).unwrap();
    assert_eq!(prepared_b.position(), 2);
    assert_eq!(engine.state_fingerprint(), committed_hash);
    engine.restore(&published).unwrap();
    assert!(
        engine.commit(prepared_b).is_err(),
        "old timeline draft committed after restore"
    );
    let repeated = engine.forward(13).unwrap();
    assert_eq!(
        out_b, repeated,
        "restore+decode must reproduce identical logits"
    );
    engine.reset().unwrap();
    assert_eq!(engine.state_fingerprint(), baseline_hash);
    assert_eq!(engine.position(), 0);
    assert!(engine.next_logits(u32::MAX).is_err());
    assert_eq!(engine.state_fingerprint(), baseline_hash);
    let foreign = QwenDenseCpu::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
    assert!(engine.restore(&foreign.snapshot()).is_err());
    assert_eq!(engine.state_fingerprint(), baseline_hash);
    // Restore only snapshots produced by the same session/model.
    engine.restore(&baseline).unwrap();
    assert_eq!(engine.position(), 0);
}

#[test]
#[ignore = "requires original SHA-pinned Qwen3.5 0.8B and 2B source checkpoints"]
fn actual_qwen_two_chunk_cpu_prefill_skips_intermediate_logits_without_state_drift() {
    use zerogravity::qwen35::QwenDenseCpu;
    for root in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(root);
        let make = || {
            QwenDenseCpu::open(&dir.join("model.gguf"), &dir.join("config.json"))
                .expect("original GGUF and geometry")
        };
        let mut monolithic = make();
        let mut no_intermediate_logits = make();
        let ids = [11u32, 353, 1044];
        let expected = monolithic.prefill_transaction(&ids).unwrap();
        no_intermediate_logits
            .prefill_chunk_without_logits(&ids[..2])
            .unwrap();
        assert_eq!(no_intermediate_logits.position(), 2);
        assert_eq!(no_intermediate_logits.root_id(), 1);
        let actual = no_intermediate_logits
            .prefill_transaction(&ids[2..])
            .unwrap();
        assert_eq!(
            actual, expected,
            "{root}: intermediate logits elision changed final logits"
        );
        assert_eq!(no_intermediate_logits.position(), monolithic.position());
        assert_eq!(
            no_intermediate_logits.state_fingerprint(),
            monolithic.state_fingerprint()
        );
        assert_eq!(
            no_intermediate_logits.state_bytes(),
            monolithic.state_bytes()
        );
        assert_eq!(no_intermediate_logits.root_id(), 2);
        assert_eq!(monolithic.root_id(), 1);
        println!(
            "P07_REVIEW_REAL_QWEN_CPU_CHUNK_NO_INTERMEDIATE_LOGITS_PASS model={root} tokens=3 chunk_roots=2 monolithic_roots=1"
        );
    }
}

#[test]
#[ignore = "requires SHA-verified Qwen3.5 0.8B and 2B models, costs real CPU execution"]
fn actual_qwen_prefill_chunking_transaction_parity_both_models() {
    use zerogravity::qwen35::QwenDenseCpu;
    let tokens = [11u32, 353, 1044, 264, 220, 17, 15, 1007];
    for root in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(root);
        let make =
            || QwenDenseCpu::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut sequential = make();
        let mut block = make();
        let mut segmented = make();
        let mut compact = make();
        let original_hash = block.state_fingerprint();
        let mut seq_logits = Vec::new();
        for &token in &tokens {
            seq_logits = sequential.forward(token).unwrap();
        }
        let chunk_logits = block.prefill_transaction(&tokens).unwrap();
        let _ = segmented.prefill_transaction(&tokens[..3]).unwrap();
        let segmented_logits = segmented.prefill_transaction(&tokens[3..]).unwrap();
        // A normal user-facing prefill needs exact state on earlier chunks,
        // not a redundant 248K-vocabulary projection for every boundary.
        compact.prefill_chunk_without_logits(&tokens[..3]).unwrap();
        let compact_logits = compact.prefill_transaction(&tokens[3..]).unwrap();
        assert_eq!(
            seq_logits, chunk_logits,
            "model {root} prefill logits differ"
        );
        assert_eq!(
            chunk_logits, segmented_logits,
            "model {root} chunk layout changes logits"
        );
        assert_eq!(
            chunk_logits, compact_logits,
            "{root} skipped intermediate logits changed output"
        );
        assert_eq!(compact.root_id(), segmented.root_id());
        assert_eq!(compact.position(), tokens.len());
        assert_eq!(compact.state_fingerprint(), segmented.state_fingerprint());
        assert_eq!(sequential.state_fingerprint(), block.state_fingerprint());
        assert_eq!(block.state_fingerprint(), segmented.state_fingerprint());
        assert_eq!(block.position(), tokens.len());
        assert_eq!(block.root_id(), 1);
        assert_eq!(segmented.root_id(), 2);
        assert_eq!(sequential.root_id(), tokens.len() as u64);
        let before_fail = block.snapshot();
        let (out, stale_draft) = block.next_logits(271).unwrap();
        assert!(block.prefill_transaction(&[11, u32::MAX]).is_err());
        assert_eq!(block.state_fingerprint(), before_fail.fingerprint());
        assert_eq!(block.root_id(), 1);
        let checkpoint = block.snapshot();
        block.commit(stale_draft.clone()).unwrap();
        assert_eq!(block.position(), tokens.len() + 1);
        assert_eq!(block.root_id(), 2);
        assert!(block.commit(stale_draft).is_err());
        block.restore(&checkpoint).unwrap();
        assert_eq!(block.forward(271).unwrap(), out);
        block.reset().unwrap();
        let foreign = make();
        assert!(
            block.restore(&foreign.snapshot()).is_err(),
            "{root}: cross-session state accepted"
        );
        assert_eq!(block.position(), 0);
        assert_eq!(block.state_fingerprint(), original_hash);
        assert!(block.prefill_transaction(&[]).is_err());
        assert!(block.prefill_transaction(&vec![11; 262145]).is_err());
        assert_eq!(block.position(), 0);
    }
}

#[test]
#[cfg(feature = "text")]
#[ignore = "requires official Qwen tokenizer source JSON files"]
fn qwen_official_tokenizer_multilingual_and_chat_roundtrip() {
    use zerogravity::tokenize::QwenTokenizer;
    for root in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(root);
        let tokenizer = QwenTokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
        for text in [
            "Hello, world!",
            "Bonjour! 你好 👋",
            "A <think> guarded token",
            "こんにちは世界\nRust! 👨‍💻",
            "math: 2+2=4\n",
        ] {
            let tokens = tokenizer.encode(text).unwrap();
            let decoded = tokenizer.decode(&tokens, false).unwrap();
            assert_eq!(decoded, text, "{root} tokenizer changed Unicode/content");
        }
        let (start, end) = tokenizer.special_ids();
        let chat = tokenizer.prompt_tokens("Hi 👋", true).unwrap();
        assert_eq!(chat[0], start);
        assert!(chat.contains(&end));
        assert_eq!(
            *chat.last().unwrap(),
            198,
            "Qwen assistant header ends in a newline token"
        );
        let full = tokenizer.decode(&chat, false).unwrap();
        assert!(
            full.ends_with("<|im_start|>assistant\n"),
            "{root} chat template suffix changed"
        );
    }
}

#[test]
#[ignore = "requires local Qwen 0.8B/2B weights; numerical layer observations are not yet an independent llama.cpp per-layer oracle"]
fn trained_source_per_layer_trace_preserves_outputs_and_private_state() {
    use zerogravity::qwen35::QwenDenseCpu;
    for root in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(root);
        let model = QwenDenseCpu::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let original_state = model.state_fingerprint();
        let (regular, regular_state) = model.next_logits(11).unwrap();
        let (traced, traced_state, records) = model.next_logits_with_layer_trace(11).unwrap();
        assert!(
            records
                .iter()
                .all(|r| r.position == 0 && r.values.iter().all(|v| v.is_finite()))
        );
        // Diagnostic GDN features add raw tensors of different widths. Keep
        // the original six-stage/146-vector public trace gate exact while
        // checking the additional diagnostics separately.
        let observations = records
            .iter()
            .filter(|r| {
                matches!(
                    r.stage,
                    "embedding"
                        | "final_norm"
                        | "attention_input_norm"
                        | "attention_output"
                        | "post_attention_residual"
                        | "ffn_input_norm"
                        | "ffn_output"
                        | "post_ffn_residual"
                )
            })
            .collect::<Vec<_>>();
        #[cfg(feature = "diagnostic-gdn-trace")]
        {
            assert_eq!(
                records
                    .iter()
                    .filter(|r| r.stage == "gdn_core_output")
                    .count(),
                18
            );
            let state = records
                .iter()
                .filter(|r| r.stage == "gdn_state")
                .collect::<Vec<_>>();
            assert_eq!(state.len(), 1);
            assert_eq!(state[0].layer, 0);
            assert_eq!(state[0].values.len(), 16 * 128 * 128);
        }
        assert_eq!(regular, traced, "{root} trace altered final logits");
        assert_eq!(regular_state.fingerprint(), traced_state.fingerprint());
        assert_eq!(
            model.state_fingerprint(),
            original_state,
            "{root} tracing mutated the visible root"
        );
        assert_eq!(model.root_id(), 0);
        assert_eq!(
            observations.len(),
            2 + 24 * 6,
            "each layer needs six stage snapshots"
        );
        assert_eq!(observations[0].stage, "embedding");
        assert_eq!(observations.last().unwrap().stage, "final_norm");
        let hidden = if root == "qwen35-0.8b" { 1024 } else { 2048 };
        assert!(observations.iter().all(|obs| obs.position == 0
            && obs.values.len() == hidden
            && obs.values.iter().all(|x| x.is_finite())));
        for layer in 0..24 {
            let b = &observations[1 + layer * 6..1 + (layer + 1) * 6];
            assert_eq!(
                b.iter().map(|r| r.stage).collect::<Vec<_>>(),
                [
                    "attention_input_norm",
                    "attention_output",
                    "post_attention_residual",
                    "ffn_input_norm",
                    "ffn_output",
                    "post_ffn_residual",
                ]
            );
            assert!(b.iter().all(|r| r.layer == layer));
            let x = if layer == 0 {
                &observations[0].values
            } else {
                &observations[1 + (layer - 1) * 6 + 5].values
            };
            for (j, &xj) in x.iter().enumerate() {
                assert_eq!(
                    xj + b[1].values[j],
                    b[2].values[j],
                    "{root} layer {layer}: attention residual not exact"
                );
                assert_eq!(
                    b[2].values[j] + b[4].values[j],
                    b[5].values[j],
                    "{root} layer {layer}: FFN residual not exact"
                );
            }
        }
        let encoded = serde_json::to_vec(&observations).unwrap();
        assert!(encoded.len() > 100_000);
        assert!(encoded.len() < 6_000_000);
    }
}

#[cfg(feature = "diagnostic-cpu-checkpoint")]
#[test]
#[ignore = "requires local SHA-pinned Qwen3.5 0.8B and 2B GGUF; exercises real CPU checkpoint replay"]
fn cpu_checkpoint_resumed_vs_monolithic_both_real_models_and_rejection() {
    use zerogravity::qwen35::QwenDenseCpu;
    let ids = [11u32, 353, 1044, 264, 220, 17, 15, 1007];
    for model in ["qwen35-0.8b", "qwen35-2b"] {
        let folder = common::workspace_path("target/models").join(model);
        let mk = || {
            QwenDenseCpu::open(&folder.join("model.gguf"), &folder.join("config.json"))
                .expect("pinned Qwen3.5 source model")
        };
        let mut mono = mk();
        let expected = mono.prefill_transaction(&ids).unwrap();
        let expected_digest = mono.state_fingerprint();
        let expected_bytes = mono.state_bytes();

        let checkpoint_dir = common::workspace_path("target/qualification/cpu-checkpoint-tests");
        std::fs::create_dir_all(&checkpoint_dir).unwrap();
        let checkpoint = checkpoint_dir.join(format!("{model}-cpu-state.bin"));
        let corrupt = checkpoint_dir.join(format!("{model}-corrupt.bin"));
        let good_model = [0x32u8; 32];
        let good_input = [0x58u8; 32];
        let mut partial = mk();
        partial.prefill_chunk_without_logits(&ids[..4]).unwrap();
        let partial_fingerprint = partial.state_fingerprint();
        assert_eq!(partial.position(), 4);
        partial
            .save_cpu_checkpoint(&checkpoint, good_model, good_input)
            .unwrap();
        let bytes = std::fs::read(&checkpoint).unwrap();
        assert!(bytes.len() > 500_000);

        let mut resumed = mk();
        let fresh_fingerprint = resumed.state_fingerprint();
        assert!(
            resumed
                .restore_cpu_checkpoint(&checkpoint, [0; 32], good_input)
                .is_err()
        );
        assert!(
            resumed
                .restore_cpu_checkpoint(&checkpoint, good_model, [0; 32])
                .is_err()
        );
        assert_eq!(resumed.position(), 0);
        assert_eq!(resumed.state_fingerprint(), fresh_fingerprint);
        resumed
            .restore_cpu_checkpoint(&checkpoint, good_model, good_input)
            .unwrap();
        assert_eq!(resumed.position(), 4);
        assert_eq!(resumed.state_fingerprint(), partial_fingerprint);
        let actual = resumed.prefill_transaction(&ids[4..]).unwrap();
        assert_eq!(
            actual, expected,
            "{model} checkpoint changed trained logits"
        );
        assert_eq!(resumed.state_fingerprint(), expected_digest);
        assert_eq!(resumed.state_bytes(), expected_bytes);

        let mut altered = bytes.clone();
        altered[128] ^= 1;
        std::fs::write(&corrupt, altered).unwrap();
        let mut invalid = mk();
        assert!(
            invalid
                .restore_cpu_checkpoint(&corrupt, good_model, good_input)
                .is_err()
        );
        assert_eq!(invalid.position(), 0);
        assert_eq!(invalid.state_fingerprint(), fresh_fingerprint);

        std::fs::write(&corrupt, &bytes[..bytes.len() - 12]).unwrap();
        assert!(
            invalid
                .restore_cpu_checkpoint(&corrupt, good_model, good_input)
                .is_err()
        );
        assert_eq!(invalid.state_fingerprint(), fresh_fingerprint);

        let _ = std::fs::remove_file(checkpoint);
        let _ = std::fs::remove_file(corrupt);
    }
}

#[test]
#[ignore = "requires both locally SHA-pinned Qwen3.5 0.8B and 2B models"]
fn qwen_source_matvec_specialization_fallback_and_device_failure_are_atomic() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use zerogravity::accelerator::{MatVecKernel, MatVecRegion, RegionExecution};
    use zerogravity::qwen35::QwenDenseCpu;
    use zerogravity::{Error, Result};

    struct Fallback(AtomicUsize);
    impl MatVecKernel for Fallback {
        fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
            assert!(!region.tensor_name.is_empty());
            assert_eq!(region.activation.len(), region.columns);
            assert_eq!(
                region.kind.byte_len(region.columns as u64).unwrap() as usize * region.rows,
                region.packed_weights.len(),
            );
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(RegionExecution::Unsupported)
        }
    }
    struct IncompleteOutput;
    impl MatVecKernel for IncompleteOutput {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Completed(Vec::new()))
        }
    }
    struct DeviceFailure;
    impl MatVecKernel for DeviceFailure {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Err(Error::State("test device readback failure".into()))
        }
    }

    for model in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(model);
        let gguf = dir.join("model.gguf");
        let config = dir.join("config.json");
        let weights = CpuWeights::open(&gguf).unwrap();
        let token = weights.embed(11, "token_embd.weight").unwrap();
        let reference = weights.matvec("blk.0.attn_gate.weight", &token).unwrap();
        let fallback = Arc::new(Fallback(AtomicUsize::new(0)));
        let requested: Arc<dyn MatVecKernel> = fallback.clone();
        let projected = weights
            .matvec_with("blk.0.attn_gate.weight", &token, Some(requested.as_ref()))
            .unwrap();
        assert_eq!(
            reference, projected,
            "{model} fallback altered CPU source math"
        );
        assert_eq!(fallback.0.load(Ordering::Relaxed), 1);

        let mut cpu = QwenDenseCpu::open(&gguf, &config).unwrap();
        let mut substituted = QwenDenseCpu::open_with_kernel(&gguf, &config, requested).unwrap();
        let prompts = [11, 353, 1007];
        let cpu_logits = cpu.prefill_transaction(&prompts).unwrap();
        let substitute_logits = substituted.prefill_transaction(&prompts).unwrap();
        assert_eq!(
            cpu_logits, substitute_logits,
            "{model} accelerator seam altered logits"
        );
        assert_eq!(cpu.state_fingerprint(), substituted.state_fingerprint());
        assert_eq!(cpu.root_id(), substituted.root_id());
        assert!(fallback.0.load(Ordering::Relaxed) > 10);

        for kernel in [
            Arc::new(IncompleteOutput) as Arc<dyn MatVecKernel>,
            Arc::new(DeviceFailure) as Arc<dyn MatVecKernel>,
        ] {
            let mut device_model = QwenDenseCpu::open_with_kernel(&gguf, &config, kernel).unwrap();
            let original_state = device_model.state_fingerprint();
            let reason = device_model
                .prefill_transaction(&prompts)
                .unwrap_err()
                .to_string();
            assert!(
                reason.contains("invalid or incomplete output")
                    || reason.contains("test device readback failure"),
                "{model}: unexpected device failure {reason}"
            );
            assert_eq!(device_model.position(), 0);
            assert_eq!(device_model.root_id(), 0);
            assert_eq!(device_model.state_fingerprint(), original_state);
        }
    }
}

#[test]
#[ignore = "requires both local source-pinned Qwen checkpoints; recurrent proposal/rollback gate"]
fn recurrent_proposals_validate_both_outputs_and_preserve_nonzero_root() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zerogravity::{
        Error, Result,
        accelerator::{
            GdnRecurrentCompleted, GdnRecurrentExecution, GdnRecurrentRegion, MatVecKernel,
            MatVecRegion, RegionExecution,
        },
        qwen35::{LoadedQwenModel, source_gdn_recurrent_step},
    };
    struct Proposal {
        failure: usize,
        calls: AtomicUsize,
    }
    impl MatVecKernel for Proposal {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
        fn try_gdn_recurrent(
            &self,
            region: GdnRecurrentRegion<'_>,
        ) -> Result<GdnRecurrentExecution> {
            region.validate()?;
            assert!(region.source_identity.is_bound());
            assert!(
                region.state.iter().any(|&x| x != 0.0),
                "test needs trained nonzero state"
            );
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            let mut state = region.state.to_vec();
            let mut output = vec![0.0; region.q.len()];
            let d = region.width;
            for (h, &coeff) in region.coefficients.iter().enumerate() {
                source_gdn_recurrent_step(
                    &mut state[h * d * d..(h + 1) * d * d],
                    &region.q[h * d..(h + 1) * d],
                    &region.k[h * d..(h + 1) * d],
                    &region.v[h * d..(h + 1) * d],
                    coeff,
                    &mut output[h * d..(h + 1) * d],
                )?;
            }
            // Let an entire token's 18 proposals reach its private draft,
            // then fail the second token without publishing either token.
            if call == 18 {
                match self.failure {
                    1 => {
                        state.pop();
                    }
                    2 => {
                        output.pop();
                    }
                    3 => {
                        state[0] = f32::NAN;
                    }
                    4 => {
                        output[0] = f32::INFINITY;
                    }
                    5 => {
                        return Err(Error::State(
                            "submitted recurrent completion unknown".into(),
                        ));
                    }
                    _ => {}
                }
            }
            Ok(GdnRecurrentExecution::Completed(GdnRecurrentCompleted {
                state,
                output,
            }))
        }
    }
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(name);
        let model =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut reference = model.new_session().unwrap();
        reference.prefill_transaction(&[11]).unwrap();
        let expected = reference.prefill_transaction(&[353, 17]).unwrap();
        for failure in 0..=5 {
            let mut session = model.new_session().unwrap();
            session.prefill_transaction(&[11]).unwrap();
            let before = (
                session.root_id(),
                session.position(),
                session.state_fingerprint(),
            );
            let sibling = model.new_session().unwrap();
            let sibling_before = sibling.state_fingerprint();
            let kernel = Arc::new(Proposal {
                failure,
                calls: AtomicUsize::new(0),
            });
            session = session.with_kernel(kernel.clone());
            let result = session.prefill_transaction(&[353, 17]);
            if failure == 0 {
                let actual = result.unwrap();
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert_eq!(session.state_fingerprint(), reference.state_fingerprint());
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 36);
                // A prepared proposal cannot publish itself or survive rewind.
                let snapshot = session.snapshot();
                let (_, stale) = session.next_logits(42).unwrap();
                session.restore(&snapshot).unwrap();
                assert!(session.commit(stale).is_err());
            } else {
                assert!(
                    result.is_err(),
                    "{name} incomplete/error proposal accepted: {failure}"
                );
                assert_eq!(
                    (
                        session.root_id(),
                        session.position(),
                        session.state_fingerprint()
                    ),
                    before
                );
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 19);
            }
            assert_eq!(sibling.state_fingerprint(), sibling_before);
        }
    }
}

#[test]
#[ignore = "requires both local source-pinned Qwen checkpoints; GDN L2 rollback gate"]
fn grouped_gdn_l2_completion_failure_preserves_nonzero_root() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zerogravity::{
        Error, Result,
        accelerator::{GroupedRmsNormRegion, MatVecKernel, MatVecRegion, RegionExecution},
        qwen35::{LoadedQwenModel, source_grouped_rms_normalize},
    };
    struct L2Completion {
        fault: usize,
        calls: AtomicUsize,
    }
    impl MatVecKernel for L2Completion {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
        fn try_grouped_rms_norm(
            &self,
            region: GroupedRmsNormRegion<'_>,
        ) -> Result<RegionExecution> {
            if !region.tensor_name.ends_with(".gdn_qk_l2") {
                return Ok(RegionExecution::Unsupported);
            }
            assert!(region.source_identity.is_bound());
            assert_eq!((region.group_width, region.groups), (128, 32));
            let mut output = source_grouped_rms_normalize(
                region.input,
                region.weights,
                region.groups,
                region.eps,
            )?;
            if self.calls.fetch_add(1, Ordering::Relaxed) == 18 {
                match self.fault {
                    1 => {
                        output.pop();
                    }
                    2 => {
                        output[0] = f32::NAN;
                    }
                    3 => return Err(Error::State("L2 readback completion unknown".into())),
                    _ => {}
                }
            }
            Ok(RegionExecution::Completed(output))
        }
    }
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(name);
        let model =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut reference = model.new_session().unwrap();
        reference.prefill_transaction(&[11]).unwrap();
        let expected = reference.prefill_transaction(&[353, 17]).unwrap();
        for fault in 0..=3 {
            let mut session = model.new_session().unwrap();
            session.prefill_transaction(&[11]).unwrap();
            let before = (
                session.root_id(),
                session.position(),
                session.state_fingerprint(),
            );
            let sibling = model.new_session().unwrap();
            let sibling_before = sibling.state_fingerprint();
            let kernel = Arc::new(L2Completion {
                fault,
                calls: AtomicUsize::new(0),
            });
            session = session.with_kernel(kernel.clone());
            let result = session.prefill_transaction(&[353, 17]);
            if fault == 0 {
                let actual = result.unwrap();
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert_eq!(session.state_fingerprint(), reference.state_fingerprint());
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 36);
            } else {
                assert!(result.is_err(), "{name}: L2 fault {fault} accepted");
                assert_eq!(
                    (
                        session.root_id(),
                        session.position(),
                        session.state_fingerprint()
                    ),
                    before
                );
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 19);
            }
            assert_eq!(sibling.state_fingerprint(), sibling_before);
        }
    }
}

#[test]
#[ignore = "requires both local source-pinned Qwen checkpoints; GDN SiLU rollback gate"]
fn gdn_silu_completion_failure_preserves_nonzero_root() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zerogravity::{
        Error, Result,
        accelerator::{MatVecKernel, MatVecRegion, RegionExecution, SiluRegion},
        qwen35::LoadedQwenModel,
    };
    struct SiluCompletion {
        fault: usize,
        role: &'static str,
        calls: AtomicUsize,
    }
    impl MatVecKernel for SiluCompletion {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
        fn try_silu(&self, region: SiluRegion<'_>) -> Result<RegionExecution> {
            if !region.operation.ends_with(self.role) {
                return Ok(RegionExecution::Unsupported);
            }
            assert!(region.source_identity.is_bound());
            let mut output: Vec<_> = region
                .input
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let y = zerogravity::cpu::ops::ggml_source_silu(x);
                    region.multiplier.map_or(y, |m| m[i] * y)
                })
                .collect();
            if self.calls.fetch_add(1, Ordering::Relaxed) == 18 {
                match self.fault {
                    1 => {
                        output.pop();
                    }
                    2 => {
                        output[0] = f32::NAN;
                    }
                    3 => return Err(Error::State("SiLU readback completion unknown".into())),
                    _ => {}
                }
            }
            Ok(RegionExecution::Completed(output))
        }
    }
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(name);
        let model =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut reference = model.new_session().unwrap();
        reference.prefill_transaction(&[11]).unwrap();
        let expected = reference.prefill_transaction(&[353, 17]).unwrap();
        for (role, fault) in [".gdn_conv_silu", ".gdn_output_gate"]
            .into_iter()
            .flat_map(|role| (0..=3).map(move |fault| (role, fault)))
        {
            let mut session = model.new_session().unwrap();
            session.prefill_transaction(&[11]).unwrap();
            let before = (
                session.root_id(),
                session.position(),
                session.state_fingerprint(),
            );
            let sibling = model.new_session().unwrap();
            let sibling_before = sibling.state_fingerprint();
            let kernel = Arc::new(SiluCompletion {
                fault,
                role,
                calls: AtomicUsize::new(0),
            });
            session = session.with_kernel(kernel.clone());
            let result = session.prefill_transaction(&[353, 17]);
            if fault == 0 {
                let actual = result.unwrap();
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert_eq!(session.state_fingerprint(), reference.state_fingerprint());
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 36);
            } else {
                assert!(result.is_err(), "{name}: SiLU fault {fault} accepted");
                assert_eq!(
                    (
                        session.root_id(),
                        session.position(),
                        session.state_fingerprint()
                    ),
                    before
                );
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 19);
            }
            assert_eq!(sibling.state_fingerprint(), sibling_before);
        }
    }
}

#[test]
#[ignore = "requires both local source-pinned Qwen checkpoints; GDN coefficient rollback gate"]
fn gdn_coefficients_completion_failure_preserves_nonzero_root() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zerogravity::{
        Error, Result,
        accelerator::{
            GdnCoefficientExecution, GdnCoefficientRegion, MatVecKernel, MatVecRegion,
            RegionExecution,
        },
        qwen35::{LoadedQwenModel, source_gdn_coefficients},
    };
    struct CoefficientCompletion {
        fault: usize,
        calls: AtomicUsize,
    }
    impl MatVecKernel for CoefficientCompletion {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
        fn try_gdn_coefficients(
            &self,
            region: GdnCoefficientRegion<'_>,
        ) -> Result<GdnCoefficientExecution> {
            if !region.operation.ends_with(".gdn_coefficients") {
                return Ok(GdnCoefficientExecution::Unsupported);
            }
            assert!(region.source_identity.is_bound());
            let mut output = source_gdn_coefficients(region)?;
            if self.calls.fetch_add(1, Ordering::Relaxed) == 18 {
                match self.fault {
                    1 => {
                        output.coefficients.pop();
                    }
                    2 => {
                        output.coefficients[0].gain = f32::NAN;
                    }
                    3 => {
                        return Err(Error::State(
                            "coefficient readback completion unknown".into(),
                        ));
                    }
                    4 => {
                        output.softplus.pop();
                    }
                    5 => {
                        output.gate_raw[0] = f32::NAN;
                    }
                    6 => {
                        output.coefficients[0].beta = f32::NAN;
                    }
                    7 => {
                        output.coefficients[0].scale = f32::NAN;
                    }
                    8 => {
                        output.coefficients[0].scale = 0.25;
                    }
                    _ => {}
                }
            }
            Ok(GdnCoefficientExecution::Completed(output))
        }
    }
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(name);
        let model =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut reference = model.new_session().unwrap();
        reference.prefill_transaction(&[11]).unwrap();
        let expected = reference.prefill_transaction(&[353, 17]).unwrap();
        for fault in 0..=8 {
            let mut session = model.new_session().unwrap();
            session.prefill_transaction(&[11]).unwrap();
            let before = (
                session.root_id(),
                session.position(),
                session.state_fingerprint(),
            );
            let sibling = model.new_session().unwrap();
            let sibling_before = sibling.state_fingerprint();
            let kernel = Arc::new(CoefficientCompletion {
                fault,
                calls: AtomicUsize::new(0),
            });
            session = session.with_kernel(kernel.clone());
            let result = session.prefill_transaction(&[353, 17]);
            if fault == 0 {
                let actual = result.unwrap();
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert_eq!(session.state_fingerprint(), reference.state_fingerprint());
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 36);
            } else {
                assert!(
                    result.is_err(),
                    "{name}: coefficient fault {fault} accepted"
                );
                assert_eq!(
                    (
                        session.root_id(),
                        session.position(),
                        session.state_fingerprint()
                    ),
                    before
                );
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 19);
            }
            assert_eq!(sibling.state_fingerprint(), sibling_before);
        }
    }
}

#[test]
#[ignore = "requires both local source-pinned Qwen checkpoints; RoPE/private KV rollback gate"]
fn rope_kv_proposal_failures_preserve_committed_nonzero_prefix() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zerogravity::{
        Error, Result,
        accelerator::{MatVecKernel, MatVecRegion, RegionExecution, RopeKvExecution, RopeKvRegion},
        qwen35::{LoadedQwenModel, source_rope_kv},
    };
    struct Proposal {
        fault: usize,
        calls: AtomicUsize,
    }
    impl MatVecKernel for Proposal {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
        fn try_rope_kv(&self, r: RopeKvRegion<'_>) -> Result<RopeKvExecution> {
            assert!(r.source_identity.is_bound());
            assert!(r.position > 0);
            let mut ready = source_rope_kv(r)?;
            if self.calls.fetch_add(1, Ordering::Relaxed) == 6 {
                match self.fault {
                    1 => {
                        ready.query.pop();
                    }
                    2 => {
                        ready.key.pop();
                    }
                    3 => {
                        ready.value.pop();
                    }
                    4 => {
                        ready.query[0] = f32::NAN;
                    }
                    5 => return Err(Error::State("RoPE/KV physical completion unknown".into())),
                    6 => {
                        ready.value[0] = f32::from_bits(ready.value[0].to_bits() ^ 0x80000000);
                    }
                    7 => {
                        ready.query[r.rotate] =
                            f32::from_bits(ready.query[r.rotate].to_bits() ^ 0x80000000);
                    }
                    8 => {
                        ready.key[r.rotate] =
                            f32::from_bits(ready.key[r.rotate].to_bits() ^ 0x80000000);
                    }
                    _ => {}
                }
            }
            Ok(RopeKvExecution::Completed(ready))
        }
    }
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(name);
        let model =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut reference = model.new_session().unwrap();
        reference.prefill_transaction(&[11]).unwrap();
        let expected = reference.prefill_transaction(&[353, 17]).unwrap();
        for fault in 0..=8 {
            let mut session = model.new_session().unwrap();
            session.prefill_transaction(&[11]).unwrap();
            let before = (
                session.root_id(),
                session.position(),
                session.state_fingerprint(),
            );
            let sibling = model.new_session().unwrap();
            let sibling_before = sibling.state_fingerprint();
            let kernel = Arc::new(Proposal {
                fault,
                calls: AtomicUsize::new(0),
            });
            session = session.with_kernel(kernel.clone());
            let result = session.prefill_transaction(&[353, 17]);
            if fault == 0 {
                let actual = result.unwrap();
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert_eq!(session.state_fingerprint(), reference.state_fingerprint());
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 12);
            } else {
                assert!(
                    result.is_err(),
                    "{name} accepted malformed RoPE/KV fault {fault}"
                );
                assert_eq!(
                    (
                        session.root_id(),
                        session.position(),
                        session.state_fingerprint()
                    ),
                    before
                );
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 7);
            }
            assert_eq!(sibling.state_fingerprint(), sibling_before);
        }
    }
}

#[test]
#[ignore = "requires both original locally pinned Qwen3.5 GGUF checkpoints"]
fn attention_qk_failures_preserve_committed_prefix_after_tentative_token() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zerogravity::{
        Error, Result,
        accelerator::{
            AttentionQkExecution, AttentionQkRegion, MatVecKernel, MatVecRegion, RegionExecution,
        },
        qwen35::{LoadedQwenModel, source_attention_qk},
    };
    struct Proposal {
        fault: usize,
        calls: AtomicUsize,
    }
    impl MatVecKernel for Proposal {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
        fn try_attention_qk(&self, r: AttentionQkRegion<'_>) -> Result<AttentionQkExecution> {
            assert!(r.source_identity.is_bound());
            assert!(r.keys.len() > 1);
            let mut ready = source_attention_qk(r)?;
            if self.calls.fetch_add(1, Ordering::Relaxed) == 6 {
                match self.fault {
                    1 => {
                        ready.dots.pop();
                    }
                    2 => {
                        ready.scores.pop();
                    }
                    3 => {
                        ready.dots[0] = f32::NAN;
                    }
                    4 => {
                        ready.scores[0] = f32::INFINITY;
                    }
                    5 => return Err(Error::State("QK physical completion unknown".into())),
                    6 => {
                        ready.scores[0] = f32::from_bits(ready.scores[0].to_bits() ^ 0x80000000);
                    }
                    _ => {}
                }
            }
            Ok(AttentionQkExecution::Completed(ready))
        }
    }
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(name);
        let model =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut reference = model.new_session().unwrap();
        reference.prefill_transaction(&[11]).unwrap();
        let expected = reference.prefill_transaction(&[353, 17]).unwrap();
        for fault in 0..=6 {
            let mut session = model.new_session().unwrap();
            session.prefill_transaction(&[11]).unwrap();
            let before = (
                session.root_id(),
                session.position(),
                session.state_fingerprint(),
            );
            let sibling = model.new_session().unwrap();
            let sibling_before = sibling.state_fingerprint();
            let kernel = Arc::new(Proposal {
                fault,
                calls: AtomicUsize::new(0),
            });
            session = session.with_kernel(kernel.clone());
            let result = session.prefill_transaction(&[353, 17]);
            if fault == 0 {
                let actual = result.unwrap();
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert_eq!(session.state_fingerprint(), reference.state_fingerprint());
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 12);
            } else {
                assert!(result.is_err(), "{name} accepted QK fault {fault}");
                assert_eq!(
                    (
                        session.root_id(),
                        session.position(),
                        session.state_fingerprint()
                    ),
                    before
                );
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 7);
            }
            assert_eq!(sibling.state_fingerprint(), sibling_before);
        }
    }
}

#[test]
#[ignore = "requires both original locally pinned Qwen3.5 GGUF checkpoints"]
fn attention_softmax_failures_preserve_nonzero_root_after_tentative_token() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zerogravity::{
        Error, Result,
        accelerator::{
            AttentionSoftmaxExecution, AttentionSoftmaxRegion, MatVecKernel, MatVecRegion,
            RegionExecution,
        },
        qwen35::{LoadedQwenModel, source_attention_softmax},
    };
    struct Proposal {
        fault: usize,
        calls: AtomicUsize,
    }
    impl MatVecKernel for Proposal {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
        fn try_attention_softmax(
            &self,
            r: AttentionSoftmaxRegion<'_>,
        ) -> Result<AttentionSoftmaxExecution> {
            assert!(r.source_identity.is_bound());
            assert!(r.width > 1);
            let mut ready = source_attention_softmax(r)?;
            if self.calls.fetch_add(1, Ordering::Relaxed) == 6 {
                match self.fault {
                    1 => {
                        ready.probabilities.pop();
                    }
                    2 => {
                        ready.exponentials.pop();
                    }
                    3 => {
                        ready.sums.pop();
                    }
                    4 => {
                        ready.reciprocals.pop();
                    }
                    5 => {
                        ready.inverse.pop();
                    }
                    6 => {
                        ready.exponentials[0] = f32::NAN;
                    }
                    7 => return Err(Error::State("softmax physical completion unknown".into())),
                    8 => {
                        ready.inverse[0] *= 0.5;
                    }
                    9 => {
                        ready.probabilities[0] = f32::NAN;
                    }
                    10 => {
                        ready.reciprocals[0] *= 0.5;
                    }
                    11 => {
                        ready.exponentials[0] = -0.0;
                        ready.probabilities[0] = -0.0;
                    }
                    _ => {}
                }
            }
            Ok(AttentionSoftmaxExecution::Completed(ready))
        }
    }
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(name);
        let model =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut reference = model.new_session().unwrap();
        reference.prefill_transaction(&[11]).unwrap();
        let expected = reference.prefill_transaction(&[353, 17]).unwrap();
        for fault in 0..=11 {
            let mut session = model.new_session().unwrap();
            session.prefill_transaction(&[11]).unwrap();
            let before = (
                session.root_id(),
                session.position(),
                session.state_fingerprint(),
            );
            let sibling = model.new_session().unwrap();
            let sibling_before = sibling.state_fingerprint();
            let kernel = Arc::new(Proposal {
                fault,
                calls: AtomicUsize::new(0),
            });
            session = session.with_kernel(kernel.clone());
            let result = session.prefill_transaction(&[353, 17]);
            if fault == 0 {
                let actual = result.unwrap();
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert_eq!(session.state_fingerprint(), reference.state_fingerprint());
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 12);
            } else {
                assert!(result.is_err(), "{name} accepted softmax fault {fault}");
                assert_eq!(
                    (
                        session.root_id(),
                        session.position(),
                        session.state_fingerprint()
                    ),
                    before
                );
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 7);
            }
            assert_eq!(sibling.state_fingerprint(), sibling_before);
        }
    }
}

#[test]
#[ignore = "requires both original locally pinned Qwen3.5 GGUF checkpoints"]
fn attention_value_failures_preserve_nonzero_root_after_tentative_token() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zerogravity::{
        Error, Result,
        accelerator::{AttentionValueRegion, MatVecKernel, MatVecRegion, RegionExecution},
        qwen35::{LoadedQwenModel, source_attention_value},
    };
    struct Proposal {
        fault: usize,
        calls: AtomicUsize,
    }
    impl MatVecKernel for Proposal {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
        fn try_attention_value(&self, r: AttentionValueRegion<'_>) -> Result<RegionExecution> {
            assert!(r.source_identity.is_bound());
            assert!(r.values.len() > 1);
            let mut values = source_attention_value(r)?;
            if self.calls.fetch_add(1, Ordering::Relaxed) == 6 {
                match self.fault {
                    1 => {
                        values.pop();
                    }
                    2 => {
                        values.push(0.0);
                    }
                    3 => {
                        values[0] = f32::NAN;
                    }
                    4 => {
                        values[0] = f32::INFINITY;
                    }
                    5 => {
                        return Err(Error::State("value physical completion unknown".into()));
                    }
                    _ => {}
                }
            }
            Ok(RegionExecution::Completed(values))
        }
    }
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(name);
        let model =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut reference = model.new_session().unwrap();
        reference.prefill_transaction(&[11]).unwrap();
        let expected = reference.prefill_transaction(&[353, 17]).unwrap();
        for fault in 0..=5 {
            let mut session = model.new_session().unwrap();
            session.prefill_transaction(&[11]).unwrap();
            let before = (
                session.root_id(),
                session.position(),
                session.state_fingerprint(),
            );
            let sibling = model.new_session().unwrap();
            let sibling_before = sibling.state_fingerprint();
            let kernel = Arc::new(Proposal {
                fault,
                calls: AtomicUsize::new(0),
            });
            session = session.with_kernel(kernel.clone());
            let result = session.prefill_transaction(&[353, 17]);
            if fault == 0 {
                let actual = result.unwrap();
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert_eq!(session.state_fingerprint(), reference.state_fingerprint());
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 12);
            } else {
                assert!(result.is_err(), "{name} accepted value fault {fault}");
                assert_eq!(
                    (
                        session.root_id(),
                        session.position(),
                        session.state_fingerprint()
                    ),
                    before
                );
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 7);
            }
            assert_eq!(sibling.state_fingerprint(), sibling_before);
        }
    }
}

#[test]
#[ignore = "requires both original locally pinned Qwen3.5 GGUF checkpoints"]
fn remaining_operator_proposals_preserve_nonzero_root_on_failure() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zerogravity::{
        Error, Result,
        accelerator::{
            BinaryRegion, EmbeddingRegion, MatVecKernel, MatVecRegion, RegionExecution, SiluRegion,
        },
        cpu::ops::ggml_source_silu,
        cpu::source::source_reference_embedding,
        qwen35::{LoadedQwenModel, source_binary},
    };
    struct Proposal {
        kind: usize,
        fault: usize,
        calls: AtomicUsize,
    }
    impl Proposal {
        fn output(&self, mut values: Vec<f32>) -> Result<RegionExecution> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 1 {
                match self.fault {
                    1 => {
                        values.pop();
                    }
                    2 => {
                        values[0] = f32::NAN;
                    }
                    3 => return Err(Error::State("remaining physical completion unknown".into())),
                    _ => {}
                }
            }
            Ok(RegionExecution::Completed(values))
        }
    }
    impl MatVecKernel for Proposal {
        fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
        fn try_embedding(&self, r: EmbeddingRegion<'_>) -> Result<RegionExecution> {
            if self.kind != 0 {
                return Ok(RegionExecution::Unsupported);
            }
            self.output(source_reference_embedding(r)?)
        }
        fn try_binary(&self, r: BinaryRegion<'_>) -> Result<RegionExecution> {
            if (self.kind == 1 && r.operation == "blk.0.attn_residual")
                || (self.kind == 2 && r.operation == "blk.3.attn_gate")
            {
                self.output(source_binary(r)?)
            } else {
                Ok(RegionExecution::Unsupported)
            }
        }
        fn try_silu(&self, r: SiluRegion<'_>) -> Result<RegionExecution> {
            if self.kind != 3 || r.operation != "blk.0.ffn_silu" {
                return Ok(RegionExecution::Unsupported);
            }
            r.validate()?;
            let m = r.multiplier.unwrap();
            self.output(
                r.input
                    .iter()
                    .zip(m)
                    .map(|(&a, &b)| ggml_source_silu(a) * b)
                    .collect(),
            )
        }
    }
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let dir = common::workspace_path("target/models").join(name);
        let model =
            LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
        let mut reference = model.new_session().unwrap();
        reference.prefill_transaction(&[11]).unwrap();
        let expected = reference.prefill_transaction(&[353, 17]).unwrap();
        for kind in 0..4 {
            for fault in 0..4 {
                let mut session = model.new_session().unwrap();
                session.prefill_transaction(&[11]).unwrap();
                let before = (
                    session.root_id(),
                    session.position(),
                    session.state_fingerprint(),
                );
                let sibling = model.new_session().unwrap();
                let sibling_before = sibling.state_fingerprint();
                let kernel = Arc::new(Proposal {
                    kind,
                    fault,
                    calls: AtomicUsize::new(0),
                });
                session = session.with_kernel(kernel.clone());
                let result = session.prefill_transaction(&[353, 17]);
                if fault == 0 {
                    let actual = result.unwrap();
                    assert_eq!(actual.len(), expected.len());
                    assert!(
                        actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                    assert_eq!(session.state_fingerprint(), reference.state_fingerprint());
                } else {
                    assert!(result.is_err(), "{name} kind{kind} fault{fault}");
                    assert_eq!(
                        (
                            session.root_id(),
                            session.position(),
                            session.state_fingerprint()
                        ),
                        before
                    );
                }
                assert_eq!(kernel.calls.load(Ordering::Relaxed), 2);
                assert_eq!(sibling.state_fingerprint(), sibling_before);
            }
        }
    }
}
