use std::path::PathBuf;
use zerogravity::configuration::{self as cfg, Change, ConfigFile, INITIAL};
use zerogravity::run::config::DocumentSelection;

struct Root(PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn typed_documents_and_atomic_profile_edits_share_the_same_owner() {
    let root = Root(std::env::temp_dir().join(format!("zg-config-test-{}", std::process::id())));
    let base = root.0.join("abird/zerogravity");
    let path = base.join("config.yaml");
    let missing = ConfigFile::open(path.clone()).unwrap();
    assert!(!missing.exists());
    assert_eq!(
        cfg::get(&path, missing.bytes(), "ui.color").unwrap().origin,
        "built-in"
    );
    assert_eq!(cfg::profiles(&base).unwrap(), ["default"]);
    assert!(!root.0.exists());
    let set = |path, yaml| Change::Set { path, yaml };
    for invalid in [
        set("ui.color", "green"),
        set("run.headroom", "100%"),
        set("run.task.max_tokens", "129"),
        set("serve.listen", "0.0.0.0:8000"),
        set("serve.workers", "0"),
        set("serve.cohort_width", "3"),
        set("serve.cohort_width", "8"),
        set("serve.device_mib", "0"),
        set("run.api_key", "never-persist"),
        set(
            "model",
            "[{id: main, target: qwen3.5-0.8b, source: 'https://fixture.invalid/model.gguf?token=never-persist'}]",
        ),
        set("run.backend", "null"),
    ] {
        assert!(missing.change(invalid).is_err(), "{invalid:?}");
        assert!(!root.0.exists());
    }
    let serving = b"schema: zerogravity.pipeline.v2\nserve:\n  active_per_worker: 8\n  cohort_width: 4\n  device_mib: 24576\n";
    let doc = cfg::validate(&path, serving).unwrap().serve.unwrap();
    assert_eq!(doc.cohort_width, Some(4));
    assert_eq!(doc.device_mib, Some(24576));
    let automatic = cfg::validate(
        &path,
        b"schema: zerogravity.pipeline.v2\nserve: {cohort_width: auto, device_mib: auto}\n",
    )
    .unwrap()
    .serve
    .unwrap();
    assert_eq!((automatic.cohort_width, automatic.device_mib), (None, None));
    missing.create(INITIAL.as_bytes()).unwrap();
    assert!(missing.create(INITIAL.as_bytes()).is_err());
    let first = ConfigFile::open(path.clone()).unwrap();
    let peer = ConfigFile::open(path.clone()).unwrap();
    first.change(set("run.backend", "cpu")).unwrap();
    assert!(peer.change(set("ui.color", "never")).is_err());
    let after = ConfigFile::open(path.clone()).unwrap();
    let original = after.bytes().to_vec();
    assert!(after.change(set("run.device", "[gpu:0]")).is_err());
    assert!(
        after
            .change(set("serve.max_pending_requests", "0"))
            .is_err()
    );
    assert!(
        after
            .change(Change::Unset {
                path: "unknown.field"
            })
            .is_err()
    );
    assert!(
        after
            .edit(|draft| {
                std::fs::write(draft, "schema: broken")?;
                Ok(())
            })
            .is_err()
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    after
        .edit(|draft| {
            std::fs::write(
                draft,
                r#"schema: zerogravity.pipeline.v2
model:
  - id: main
    target: qwen3.5-0.8b
    source: ./weights.gguf
run:
  entrypoint: {model: main}
  backend: cpu
  task: {context: 64, max_tokens: 3}
ui:
  color: never
downloads:
  ask_before_pull: false
"#,
            )?;
            Ok(())
        })
        .unwrap();
    let current = ConfigFile::open(path.clone()).unwrap();
    let report = cfg::explain(&path, current.bytes()).unwrap();
    assert_eq!(report["run.task.context"].value.as_u64(), Some(64));
    assert_eq!(report["serve.max_context"].value.as_u64(), Some(64));
    assert_eq!(
        report["serve.max_context"].origin,
        path.display().to_string()
    );
    assert_eq!(
        report["downloads.ask_before_pull"].value.as_bool(),
        Some(false)
    );
    let named_path = base.join("profiles/laptop.yaml");
    let copied = cfg::copy_document(&path, &named_path, current.bytes()).unwrap();
    let profile = ConfigFile::open(named_path.clone()).unwrap();
    profile.create(&copied).unwrap();
    assert_eq!(cfg::profiles(&base).unwrap(), ["default", "laptop"]);
    let selected = cfg::load_selected(
        DocumentSelection::Profile("laptop"),
        &root.0,
        Some(root.0.as_os_str()),
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(selected.0, named_path);
    let anchored = selected.1.model[0].source.as_deref().unwrap();
    assert_eq!(std::path::Path::new(anchored), base.join("./weights.gguf"));
    let profile = ConfigFile::open(named_path.clone()).unwrap();
    profile.change(Change::Unset { path: "ui.color" }).unwrap();
    let profile = ConfigFile::open(named_path.clone()).unwrap();
    assert_eq!(
        cfg::get(&named_path, profile.bytes(), "ui.color")
            .unwrap()
            .origin,
        "built-in"
    );
    assert_eq!(
        cfg::get(&path, current.bytes(), "ui.color")
            .unwrap()
            .value
            .as_str(),
        Some("never")
    );
    profile.remove().unwrap();
    assert!(!named_path.exists());
    assert_eq!(cfg::profiles(&base).unwrap(), ["default"]);
    #[cfg(unix)]
    {
        let link = base.join("link.yaml");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(ConfigFile::open(link).is_err());
    }
}
