use std::process::Command;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}

#[test]
fn version_reports_package_version() {
    let output = binary().arg("--version").output().expect("run CLI");
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).expect("utf8").trim(),
        format!("zerogravity v{}", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn help_is_available_offline() {
    let output = binary().arg("--help").output().expect("run CLI");
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).expect("utf8");
    assert!(text.contains("USAGE:"));
    assert!(text.contains("NOT IMPLEMENTED"));
    assert_eq!(
        text.contains("GPU OPTIONS:"),
        cfg!(all(
            feature = "model-qwen35",
            feature = "text",
            any(feature = "vulkan", feature = "cuda", feature = "hip")
        ))
    );
}

#[test]
fn serve_requires_an_explicit_feature_and_registered_model() {
    let output = binary().arg("serve").output().expect("run CLI");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    if cfg!(feature = "serve") {
        assert!(stderr.contains("invalid_config"), "{stderr}");
    } else {
        assert!(stderr.contains("requires --features serve"), "{stderr}");
    }
}

#[test]
fn inspect_real_safetensors_directory_without_claiming_generation() {
    use std::fs;
    let path =
        std::env::temp_dir().join(format!("zg-inspect-cli-{}.safetensors", std::process::id()));
    let metadata = r#"{"example.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut bytes = (metadata.len() as u64).to_le_bytes().to_vec();
    bytes.extend(metadata.as_bytes());
    bytes.extend(1f32.to_le_bytes());
    bytes.extend(2f32.to_le_bytes());
    fs::write(&path, bytes).expect("fixture");
    let output = binary()
        .arg("inspect")
        .arg(&path)
        .output()
        .expect("inspect");
    let _ = fs::remove_file(&path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Tensor directory: 1 tensors"));
    assert!(text.contains("TENSOR DIRECTORY ONLY"));
    assert!(!text.contains("model executable"));
}

#[test]
fn inspection_refuses_missing_file() {
    let output = binary()
        .args(["inspect", "/nonexistent-zerogravity-unknown.gguf"])
        .output()
        .expect("inspect missing");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("I/O error"));
}

#[test]
fn inspect_real_local_model_config_geometry() {
    use std::fs;
    let path = std::env::temp_dir().join(format!(
        "zg-model-inspect-{}-config.json",
        std::process::id()
    ));
    let config = serde_json::json!({
        "architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4",
        "text_config":{
            "model_type":"gemma4_text",
            "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                           "sliding_attention","sliding_attention","full_attention"],
            "num_hidden_layers":6,"hidden_size":2816,"head_dim":256,
            "num_attention_heads":16,"num_key_value_heads":8,
            "num_global_key_value_heads":2,"vocab_size":262144,
            "max_position_embeddings":262144,"sliding_window":1024,
            "num_experts":128,"top_k_experts":8,"enable_moe_block":true
        }
    });
    fs::write(&path, config.to_string()).unwrap();
    let output = binary().arg("inspect").arg(&path).output().unwrap();
    fs::remove_file(&path).unwrap();
    // A workspace consumer may also enable the application adapter.
    if cfg!(feature = "model-gemma4") || output.status.success() {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("Gemma4Moe"));
        assert!(text.contains("CONFIG GEOMETRY ONLY"));
    } else {
        assert_eq!(output.status.code(), Some(2));
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("Gemma config inspection adapter is not compiled")
        );
    }
}

#[test]
fn unavailable_inference_capabilities_refuse_before_source_io() {
    for (name, compiled) in [
        ("run-vulkan-amd", cfg!(feature = "vulkan")),
        ("run-vulkan-nvidia", cfg!(feature = "vulkan")),
        ("run-cuda", cfg!(feature = "cuda")),
        ("run-rocm", cfg!(feature = "hip")),
    ] {
        if compiled && cfg!(all(feature = "model-qwen35", feature = "text")) {
            continue;
        }
        let output = binary()
            .args([name, "/nonexistent-zerogravity-source.gguf"])
            .output()
            .expect("disabled GPU command");
        assert_eq!(output.status.code(), Some(2));
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("unsupported:"), "{name}: {error}");
        assert!(
            error.contains("requires compiled features"),
            "{name}: {error}"
        );
        assert!(!error.contains("I/O error"), "{name}: {error}");
    }
}

#[cfg(not(all(feature = "cpu", feature = "model-qwen35", feature = "text")))]
#[test]
fn cpu_run_without_its_capabilities_refuses_before_source_io() {
    let output = binary()
        .args(["run", "/nonexistent-zerogravity-source.gguf"])
        .output()
        .expect("disabled CPU command");
    assert_eq!(output.status.code(), Some(2));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("unsupported:"));
    assert!(error.contains("cpu, model-qwen35 and text"));
    assert!(!error.contains("I/O error"));
}

#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
#[test]
fn feature_aware_help_lists_only_executable_gpu_commands() {
    let output = binary().arg("--help").output().expect("help");
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    for (name, compiled) in [
        ("run-vulkan-amd", cfg!(feature = "vulkan")),
        ("run-vulkan-nvidia", cfg!(feature = "vulkan")),
        ("run-cuda", cfg!(feature = "cuda")),
        ("run-rocm", cfg!(feature = "hip")),
    ] {
        assert_eq!(
            text.contains(name),
            compiled && cfg!(all(feature = "model-qwen35", feature = "text")),
            "{name}: {text}"
        );
    }
    assert!(!text.contains("GPU, and HTTP are NOT IMPLEMENTED"));
}

#[cfg(all(
    feature = "model-qwen35",
    feature = "text",
    any(feature = "vulkan", feature = "cuda", feature = "hip")
))]
#[test]
fn gpu_request_argument_errors_do_not_open_the_source_or_initialize_a_device() {
    for (name, compiled) in [
        ("run-vulkan-amd", cfg!(feature = "vulkan")),
        ("run-cuda", cfg!(feature = "cuda")),
        ("run-rocm", cfg!(feature = "hip")),
    ] {
        if !compiled {
            continue;
        }
        for max_tokens in ["0", "513"] {
            let output = binary()
                .args([
                    name,
                    "/nonexistent-zerogravity-source.gguf",
                    "--max-tokens",
                    max_tokens,
                ])
                .output()
                .expect("invalid GPU request");
            assert_eq!(output.status.code(), Some(2));
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(
                error.contains("positive integer")
                    || error.contains("1..=512")
                    || error.contains("bounded context/output/prompt/prefill request"),
                "{name}: {error}"
            );
            assert!(!error.contains("I/O error"), "{name}: {error}");
        }
    }
}
