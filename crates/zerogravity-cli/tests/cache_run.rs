use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}

#[cfg(not(all(
    feature = "cache",
    feature = "cpu",
    feature = "model-qwen35",
    feature = "text",
    target_os = "linux"
)))]
#[test]
fn cache_run_requires_explicit_capabilities_before_io() {
    let result = binary()
        .args(["run-cache", "missing", "--model-file", "absent.gguf"])
        .env("HF_TOKEN", "invalid\ncredential")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("run-cache requires"));
}

#[cfg(all(
    feature = "cache",
    feature = "cpu",
    feature = "model-qwen35",
    feature = "text",
    target_os = "linux"
))]
#[test]
fn cache_run_is_explicit_offline_and_rejects_unverified_overrides() {
    let root = std::env::temp_dir().join(format!("zg-cache-cli-{}", std::process::id()));
    assert!(!root.exists());
    let help = binary()
        .args(["run-cache", "--help"])
        .env("HF_TOKEN", "invalid\ncredential")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("No network"));
    for args in [
        vec!["run-cache", "missing", "--model-file", "../escape.gguf"],
        vec![
            "run-cache",
            "missing",
            "--model-file",
            "x.gguf",
            "--config",
            "/unverified/config.json",
        ],
        vec![
            "run-cache",
            "missing",
            "--model-file",
            "x.gguf",
            "--token=do-not-log",
        ],
        vec![
            "run-cache",
            "missing",
            "--model-file",
            "x.gguf",
            "--max-tokens",
            "0",
        ],
    ] {
        let result = binary()
            .args(args)
            .arg("--cache")
            .arg(&root)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(2));
        assert!(!String::from_utf8_lossy(&result.stderr).contains("do-not-log"));
        assert!(!root.exists());
    }
    let result = binary()
        .args(["run-cache", "missing", "--model-file", "x.gguf", "--cache"])
        .arg(&root)
        .env("HF_TOKEN", "invalid\ncredential")
        .env("GITHUB_TOKEN", "invalid\ncredential")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(!error.contains("credential"));
    assert!(!error.contains("HTTP"));
    std::fs::remove_dir_all(root).unwrap();
}
