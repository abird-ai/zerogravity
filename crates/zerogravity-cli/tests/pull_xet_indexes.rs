//! Native required-index CLI semantics: explicit apply, metadata-only dry-run,
//! no new grants, flags, or downloads during offline/static refusal.
use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}

#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn native_index_frontend_remains_optional_in_the_default_binary() {
    let output = binary()
        .args([
            "pull",
            "xet",
            "owner/repo",
            "--revision",
            "main",
            "--file",
            "model.safetensors.index.json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("optional acquisition feature"));
}
#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod enabled {
    use super::*;
    #[test]
    fn help_preserves_metadata_only_dryrun_and_explicit_apply_index_policy() {
        let out = binary().args(["pull", "xet", "--help"]).output().unwrap();
        assert!(out.status.success());
        let help = String::from_utf8_lossy(&out.stdout);
        assert!(help.contains("Index") || help.contains("index"));
        assert!(help.contains("required native Xet index reconstruction"));
        assert!(help.contains("no CAS grants or signed ranges"));
        assert!(help.contains("(default apply)"));
        assert!(!help.contains("--enable-native-index"));
    }
    #[test]
    fn unsupported_new_index_flag_and_invalid_budgets_fail_before_credential_io() {
        for suffix in [
            vec!["--enable-native-index"],
            vec!["--xet-max-terms", "0"],
            vec!["--xet-max-requests", "0"],
            vec!["--offline", "--ref", "not-a-digest"],
            vec!["--dry-run", "--offline"],
        ] {
            let out = binary()
                .args([
                    "pull",
                    "xet",
                    "owner/repo",
                    "--revision",
                    "main",
                    "--file",
                    "model.safetensors.index.json",
                ])
                .args(suffix)
                .env("HF_TOKEN", "SECRET_NEVER_PRINT")
                .output()
                .unwrap();
            assert!(!out.status.success());
            let text = String::from_utf8_lossy(&out.stderr);
            assert!(!text.contains("SECRET_NEVER_PRINT"));
        }
    }
}
