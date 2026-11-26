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
    for command in [
        "run",
        "serve",
        "model",
        "device",
        "pipeline",
        "profile",
        "config",
        "doctor",
        "completion",
    ] {
        assert!(text.contains(command), "{command}: {text}");
    }
    assert!(text.contains("Diagnose local configuration and hardware"));
    assert!(!text.contains("pending"));
    assert!(text.lines().count() < 35, "{text}");
    assert!(!text.contains("Phase 1:"));
    assert!(!text.contains("run-cuda"));
    assert!(!text.contains("COMPILED CAPABILITIES:"));
}

#[test]
fn serve_requires_an_explicit_feature_and_registered_model() {
    let root = std::env::temp_dir().join(format!("zg-serve-no-document-{}", std::process::id()));
    assert!(!root.exists());
    let output = binary()
        .arg("serve")
        .env("XDG_CONFIG_HOME", &root)
        .output()
        .expect("run CLI");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    if cfg!(feature = "serve") {
        // Missing implicit default uses built-ins; no model route selected.
        assert!(
            stderr.contains("malformed") && stderr.contains("one or more direct model sources"),
            "{stderr}"
        );
    } else {
        assert!(stderr.contains("requires --features serve"), "{stderr}");
    }
    assert!(!root.exists());
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

#[test]
fn detailed_run_help_distinguishes_compiled_backends() {
    let output = binary().args(["help", "run"]).output().expect("help");
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    let compiled = text
        .lines()
        .find(|line| line.starts_with("Compiled run choices:"))
        .unwrap();
    for (name, enabled) in [
        ("cuda", cfg!(feature = "cuda")),
        ("rocm", cfg!(feature = "hip")),
        ("vulkan", cfg!(feature = "vulkan")),
    ] {
        assert_eq!(
            compiled.contains(name),
            enabled && cfg!(all(feature = "model-qwen35", feature = "text")),
            "{compiled}"
        );
    }
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

#[test]
fn presentation_flags_preserve_primary_output_and_typed_failures() {
    let version = binary()
        .args(["-q", "--color", "always", "--version"])
        .output()
        .unwrap();
    assert_eq!(
        version.stdout,
        format!("zerogravity v{}\n", env!("CARGO_PKG_VERSION")).as_bytes()
    );
    assert!(version.stderr.is_empty());

    let plain = binary()
        .args(["-q", "--format", "text", "inspect"])
        .output()
        .unwrap();
    assert_eq!(plain.status.code(), Some(2));
    assert!(plain.stdout.is_empty());
    let stderr = String::from_utf8(plain.stderr).unwrap();
    assert!(stderr.starts_with("error[invalid_usage]:"));
    assert!(stderr.contains("hint:"));
    assert!(!stderr.contains('\x1b'));

    let colored = binary()
        .env("NO_COLOR", "1")
        .args(["--color=always", "inspect"])
        .output()
        .unwrap();
    assert_eq!(colored.status.code(), Some(2));
    assert!(colored.stdout.is_empty());
    assert!(
        String::from_utf8(colored.stderr)
            .unwrap()
            .starts_with("\x1b[31merror[invalid_usage]")
    );
}

#[test]
fn presentation_rejects_unknown_values_without_echoing_secrets() {
    for args in [
        vec!["--color", "NEVER_ECHO_SECRET", "pull"],
        vec!["NEVER_ECHO_SECRET"],
    ] {
        let output = binary().args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("error["));
        assert!(!stderr.contains("NEVER_ECHO_SECRET"));
    }
}

#[test]
fn canonical_help_completions_and_device_output_are_offline() {
    let help = binary()
        .args(["help", "model", "inspect"])
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("sha256:DIGEST"));
    assert!(help.stderr.is_empty());
    for shell in ["bash", "zsh", "fish", "powershell"] {
        let result = binary().args(["completion", shell]).output().unwrap();
        assert!(result.status.success(), "{shell}: {:?}", result.stderr);
        let script = String::from_utf8_lossy(&result.stdout);
        for name in [
            "zerogravity",
            "zg",
            "model",
            "device",
            "profile",
            "completion",
        ] {
            assert!(script.contains(name), "{shell}: {name}");
        }
        assert!(result.stderr.is_empty());
    }
    for args in [
        vec!["device", "list", "--format", "json"],
        vec!["--format=json", "devices", "list"],
        vec![
            "--color",
            "always",
            "device",
            "--quiet",
            "list",
            "--format=json",
        ],
    ] {
        let result = binary().args(args).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert!(value["devices"].is_array());
        assert!(result.stderr.is_empty());
    }
    let result = binary()
        .args(["device", "inspect", "cpu", "--format", "json"])
        .output()
        .unwrap();
    assert!(result.status.success(), "{:?}", result.stderr);
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["id"], "cpu:0");
    assert!(result.stderr.is_empty());
}

#[test]
fn unsupported_globals_fail_before_effects() {
    for args in [
        vec!["--format=jsonl", "run", "/missing.gguf"],
        vec!["--interactive=always", "device", "list"],
        vec!["--yes", "run", "/missing.gguf"],
        vec![
            "--format=json",
            "model",
            "pull",
            "hf:org/repo@main",
            "--file",
            "model.gguf",
        ],
    ] {
        let result = binary().args(args).output().unwrap();
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
        assert!(String::from_utf8_lossy(&result.stderr).contains("error[invalid_usage]"));
    }
}

#[cfg(all(feature = "acquisition", target_os = "linux"))]
#[test]
fn download_modes_refuse_before_provider_or_cache_io() {
    let root = std::env::temp_dir().join(format!("zg-download-modes-{}", std::process::id()));
    assert!(!root.exists());
    for args in [
        vec![
            "--format=jsonl",
            "model",
            "pull",
            "hf:fixture/repo@main",
            "--file",
            "LICENSE",
            "--dry-run",
        ],
        vec![
            "--format=jsonl",
            "model",
            "pull",
            "hf:fixture/repo@main",
            "--file",
            "LICENSE",
            "--offline",
        ],
        vec![
            "--interactive=always",
            "model",
            "pull",
            "hf:fixture/repo@main",
            "--file",
            "LICENSE",
        ],
    ] {
        let output = binary()
            .args(args)
            .arg("--cache")
            .arg(&root)
            .env("HF_TOKEN", "not-a-valid\nsecret")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("not-a-valid"));
        assert!(!root.exists());
    }
    let output = binary()
        .args(["model", "list", "--partial", "--format=json", "--cache"])
        .arg(&root)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        serde_json::json!([])
    );
    assert!(!root.exists());
}

#[test]
fn doctor_observes_local_policy_without_creating_or_loading_anything() {
    let root = std::env::temp_dir().join(format!("zg-doctor-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    let run = |args: &[&str]| {
        binary()
            .args(args)
            .current_dir(&root)
            .env("HOME", &root)
            .env("XDG_CONFIG_HOME", root.join("config"))
            .env("XDG_CACHE_HOME", root.join("cache"))
            .env("HF_TOKEN", "PRIVATE_DOCTOR_CREDENTIAL")
            .output()
            .unwrap()
    };
    let parse = |out: &std::process::Output| {
        assert!(
            !out.stdout
                .windows(b"PRIVATE_DOCTOR".len())
                .any(|w| w == b"PRIVATE_DOCTOR")
        );
        assert!(
            !out.stderr
                .windows(b"PRIVATE_DOCTOR".len())
                .any(|w| w == b"PRIVATE_DOCTOR")
        );
        serde_json::from_slice::<serde_json::Value>(&out.stdout).unwrap()
    };
    let cold = run(&["doctor", "--format=json"]);
    assert!(cold.status.success(), "{:?}", cold.stderr);
    let report = parse(&cold);
    assert_eq!(report["schema"], "zerogravity.doctor.v1");
    assert_eq!(report["execution_qualified"], false);
    assert_eq!(report["status"], "ok");
    let checks = report["checks"].as_array().unwrap();
    assert!(checks.iter().any(|c| {
        c["component"] == "devices"
            && c["details"]["devices"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == "cpu:0")
    }));
    assert!(!root.join("config").exists());
    assert!(!root.join("cache").exists());
    let document = "schema: zerogravity.pipeline.v2\nmodel:\n  - id: example\n    target: qwen3.5-0.8b\n    source: ./PRIVATE_DOCTOR_MISSING.gguf\nrun:\n  entrypoint: {model: example}\n  device: ['gpu:99999']\n  task: {prompt: PRIVATE_DOCTOR_PROMPT}\n";
    std::fs::write(root.join("local.yaml"), document).unwrap();
    let configured = run(&["doctor", "--config", "local.yaml", "--format=json"]);
    assert!(configured.status.success(), "{:?}", configured.stderr);
    let report = parse(&configured);
    assert_eq!(report["status"], "warning");
    let checks = report["checks"].as_array().unwrap();
    assert!(
        checks
            .iter()
            .any(|c| c["component"] == "model:example" && c["status"] == "warning")
    );
    assert!(
        checks
            .iter()
            .any(|c| c["component"] == "device policy" && c["status"] == "warning")
    );
    std::fs::write(root.join("broken.yaml"), "PRIVATE_DOCTOR_INVALID: true").unwrap();
    let invalid = run(&["--config", "broken.yaml", "doctor", "--format", "json"]);
    assert_eq!(invalid.status.code(), Some(2));
    let report = parse(&invalid);
    assert_eq!(report["status"], "error");
    assert!(
        report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["component"] == "devices")
    );
    let missing = run(&["--profile", "missing", "doctor", "--format=json"]);
    assert_eq!(missing.status.code(), Some(2));
    assert_eq!(parse(&missing)["status"], "error");
    let help = run(&["--config", "broken.yaml", "help", "doctor"]);
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("Cold local"));
    assert_eq!(
        std::fs::read_to_string(root.join("local.yaml")).unwrap(),
        document
    );
    assert!(!root.join("config").exists());
    assert!(!root.join("cache").exists());
}
