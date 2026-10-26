// Previous test: short_hf_dry_run_physical.
//! Explicit physical Qwen CUDA dry-run trace evidence; never part of ordinary
//! CPU test execution. Requires approved sm86 hardware with the real
//! source/native bundle from ZG_QWEN35_MODEL_DIR/ZG_QWEN35_CUDA_KERNEL_DIR in a
//! fresh owned process.
#![cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
use std::path::{Path, PathBuf};
use std::process::Command;

struct Home(PathBuf);
impl Drop for Home {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn absolute_env(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| panic!("required absolute {name}"))
}

fn run(home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
        .args(args)
        .current_dir(home)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_CACHE_HOME", home.join("cache"))
        .output()
        .unwrap()
}

fn trace(output: &std::process::Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).expect("Qwen CUDA dry-run decision trace JSON")
}

#[test]
#[ignore = "explicit physical Qwen CUDA dry-run trace; approved sm86 hardware with ZG_QWEN35_MODEL_DIR and ZG_QWEN35_CUDA_KERNEL_DIR in a fresh owned process"]
fn physical_qwen35_cuda_dry_run_trace_is_read_only() {
    let home =
        Home(std::env::temp_dir().join(format!("zg-qwen35-cuda-dry-{}", std::process::id())));
    assert!(!home.0.exists());
    std::fs::create_dir(&home.0).unwrap();
    let source = absolute_env("ZG_QWEN35_MODEL_DIR");
    let native = absolute_env("ZG_QWEN35_CUDA_KERNEL_DIR");
    let invocation = [
        "run".to_string(),
        source.to_string_lossy().into_owned(),
        "--native-artifacts".to_string(),
        native.to_string_lossy().into_owned(),
        "--dry-run".to_string(),
        "--prompt".to_string(),
        "a".to_string(),
    ];
    let invocation = invocation.iter().map(String::as_str).collect::<Vec<_>>();
    let out = run(&home.0, &invocation);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let first = trace(&out);
    assert_eq!(first["schema"], "zerogravity.run.qwen35-cuda.decision.v1");
    assert_eq!(first["dry_run"], true);
    assert_eq!(first["reservation"], "none");
    assert_eq!(first["pre_client_admission_gap"], true);
    assert_eq!(first["hard_os_admission"], false);
    assert!(first["device"].is_string(), "approved sm86 card proposed");
    assert!(first["excluded"].as_array().is_some());
    assert!(!home.0.join("cache").exists());

    // A second dry-run in the fresh process-shared ledger observes the same
    // proposal and creates no cache/ledger effects.
    let out = run(&home.0, &invocation);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let second = trace(&out);
    assert_eq!(second["device"], first["device"]);
    assert_eq!(second["dry_run"], true);
    assert_eq!(second["reservation"], "none");
    assert!(!home.0.join("cache").exists());

    // A forced headroom either still fits the card or prints the refusal
    // trace before failing; both contracts stay read-only either way.
    let mut refused_invocation = invocation.clone();
    refused_invocation.extend(["--headroom", "6GiB"]);
    let refused = run(&home.0, &refused_invocation);
    let refused_trace = trace(&refused);
    assert_eq!(
        refused_trace["schema"],
        "zerogravity.run.qwen35-cuda.decision.v1"
    );
    assert_eq!(refused_trace["dry_run"], true);
    assert_eq!(refused_trace["reservation"], "none");
    if refused.status.success() {
        assert!(refused_trace["device"].is_string());
    } else {
        assert_eq!(refused.status.code(), Some(2));
        assert!(refused_trace["device"].is_null());
        assert!(
            refused_trace["excluded"]
                .as_array()
                .is_some_and(|a| !a.is_empty())
        );
        assert!(
            String::from_utf8_lossy(&refused.stderr).contains("resource limit:"),
            "{}",
            String::from_utf8_lossy(&refused.stderr)
        );
    }
    assert!(!home.0.join("cache").exists());
}
