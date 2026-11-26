use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Root(PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn binary(root: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_zerogravity"));
    cmd.env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env_remove("VISUAL")
        .env_remove("EDITOR");
    cmd
}
fn run(root: &Path, args: &[&str]) -> Output {
    binary(root).args(args).output().unwrap()
}
fn pass(root: &Path, args: &[&str]) -> Output {
    let out = run(root, args);
    assert!(
        out.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}
fn value(root: &Path, args: &[&str]) -> serde_json::Value {
    serde_json::from_slice(&pass(root, args).stdout).unwrap()
}
fn refuse(root: &Path, args: &[&str]) {
    let out = run(root, args);
    assert_eq!(
        out.status.code(),
        Some(2),
        "{args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.stdout.is_empty());
}

#[test]
fn config_profile_lifecycle_is_cold_and_refuses_invalid_edits() {
    let root = Root(std::env::temp_dir().join(format!("zg-config-cli-{}", std::process::id())));
    std::fs::create_dir(&root.0).unwrap();
    let root = &root.0;
    let base = root.join("config/abird/zerogravity");
    let file = base.join("config.yaml");
    assert_eq!(
        value(root, &["config", "get", "ui.color", "--format=json"])["origin"],
        "built-in"
    );
    assert_eq!(
        value(root, &["profile", "list", "--format=json"])[0]["name"],
        "default"
    );
    assert_eq!(
        value(root, &["config", "path", "--format=json"])["path"],
        file.to_str().unwrap()
    );
    assert_eq!(
        value(root, &["config", "get", "ui.theme", "--format=json"])["value"],
        "auto"
    );
    assert_eq!(
        value(
            root,
            &[
                "config",
                "get",
                "serve.enable_error_details",
                "--format=json"
            ]
        )["value"],
        false
    );
    assert!(!base.exists());
    refuse(
        root,
        &["config", "set", "serve.enable_error_details", "yes-please"],
    );
    refuse(root, &["config", "set", "ui.theme", "carbon"]);
    refuse(root, &["config", "set", "ui.mouse", "invalid"]);
    refuse(root, &["config", "set", "ui.color", "pink"]);
    refuse(root, &["profile", "create", "../escape"]);
    assert!(!base.exists());
    pass(root, &["config", "init"]);
    pass(root, &["config", "set", "run.backend", "cpu"]);
    pass(root, &["config", "set", "run.task.context", "64"]);
    pass(root, &["config", "set", "ui.progress", "never"]);
    pass(root, &["config", "set", "ui.color", "always"]);
    pass(root, &["config", "set", "ui.theme", "light"]);
    pass(root, &["config", "set", "ui.mouse", "never"]);
    assert_eq!(
        value(root, &["config", "get", "ui.mouse", "--format=json"])["value"],
        "never"
    );
    pass(root, &["config", "unset", "ui.mouse"]);
    assert_eq!(
        value(root, &["config", "get", "ui.theme", "--format=json"])["value"],
        "light"
    );
    pass(root, &["config", "unset", "ui.theme"]);
    assert_eq!(
        value(root, &["config", "get", "ui.theme", "--format=json"])["value"],
        "auto"
    );
    let missing = root.join("missing.gguf");
    let args = ["model", "inspect", missing.to_str().unwrap()];
    let colored = run(root, &args);
    assert_eq!(colored.status.code(), Some(2));
    assert!(colored.stderr.contains(&27));
    let plain = run(
        root,
        &[
            "--color=never",
            "model",
            "inspect",
            missing.to_str().unwrap(),
        ],
    );
    assert!(!plain.stderr.contains(&27));
    pass(root, &["config", "set", "run.task.prompt", "--quiet"]);
    let report = value(root, &["config", "explain", "--format=json"]);
    assert_eq!(report["run.task.context"]["value"], 64);
    assert_eq!(report["run.task.context"]["origin"], file.to_str().unwrap());
    assert_eq!(report["serve.max_context"]["value"], 64);
    let before = std::fs::read(&file).unwrap();
    for args in [
        vec!["config", "set", "run.device", "[gpu:0]"],
        vec!["config", "set", "serve.max_context", "0"],
        vec!["config", "set", "downloads.ask_before_pull", "yes-please"],
        vec!["config", "set", "api_key", "secret"],
        vec!["config", "unset", "unknown.key"],
        vec!["config", "init"],
    ] {
        refuse(root, &args);
        assert_eq!(std::fs::read(&file).unwrap(), before);
    }
    assert_eq!(
        value(root, &["config", "get", "run.task.prompt", "--format=json"])["value"],
        "--quiet"
    );
    pass(root, &["profile", "create", "empty"]);
    pass(
        root,
        &[
            "--profile",
            "empty",
            "config",
            "set",
            "serve.enable_error_details",
            "true",
        ],
    );
    assert_eq!(
        value(
            root,
            &[
                "--profile",
                "empty",
                "config",
                "get",
                "serve.enable_error_details",
                "--format=json"
            ]
        )["value"],
        true
    );
    pass(
        root,
        &[
            "--profile",
            "empty",
            "config",
            "unset",
            "serve.enable_error_details",
        ],
    );

    pass(root, &["--profile", "empty", "model", "search"]);
    refuse(root, &["--profile", "missing", "model", "search"]);
    let plain = run(
        root,
        &[
            "--profile",
            "empty",
            "model",
            "inspect",
            missing.to_str().unwrap(),
        ],
    );
    assert_eq!(plain.status.code(), Some(2));
    assert!(!plain.stderr.contains(&27));
    assert_eq!(
        value(
            root,
            &[
                "--profile",
                "empty",
                "config",
                "get",
                "run.backend",
                "--format=json"
            ]
        )["value"],
        "auto"
    );
    pass(root, &["profile", "copy", "default", "travel"]);
    refuse(root, &["profile", "copy", "default", "travel"]);
    pass(
        root,
        &["config", "--profile", "travel", "unset", "ui.progress"],
    );
    assert_eq!(
        value(
            root,
            &[
                "config",
                "--profile",
                "travel",
                "get",
                "ui.progress",
                "--format=json"
            ]
        )["origin"],
        "built-in"
    );
    assert_eq!(
        value(root, &["config", "get", "ui.progress", "--format=json"])["value"],
        "never"
    );
    refuse(root, &["profile", "remove", "travel"]);
    refuse(root, &["profile", "remove", "default", "--yes"]);
    pass(root, &["profile", "remove", "travel", "--yes"]);
    assert!(!base.join("profiles/travel.yaml").exists());
    assert_eq!(std::fs::read(&file).unwrap(), before);
    assert!(!root.join("cache").exists());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let editor = root.join("editor");
        let execute = |body: &str| {
            std::fs::write(&editor, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&editor, std::fs::Permissions::from_mode(0o700)).unwrap();
            binary(root)
                .env("EDITOR", &editor)
                .args(["config", "edit"])
                .output()
                .unwrap()
        };
        assert!(
            !execute("printf 'schema: wrong\\n' > \"$1\"")
                .status
                .success()
        );
        assert!(
            !execute("printf 'schema: zerogravity.pipeline.v2\\n' > \"$1\"; exit 1")
                .status
                .success()
        );
        assert_eq!(std::fs::read(&file).unwrap(), before);
        assert!(execute("printf '# edited\\nschema: zerogravity.pipeline.v2\\nui:\\n  color: never\\n' > \"$1\"").status.success());
        assert!(
            std::fs::read_to_string(&file)
                .unwrap()
                .starts_with("# edited")
        );
        pass(root, &["config", "validate"]);
    }
}
