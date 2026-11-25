//! Public SDK coverage adapted from CLI pipeline/config fixtures. No CLI,
//! model features, network, devices or process-global environment changes.
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use zerogravity::{
    Error,
    pipelines::{self, CheckedDocument, MAX_CONFIG_BYTES, load_document, parse_document},
    run_policy::Entrypoint,
};

fn example(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/pipelines")
        .join(name)
}
fn text_example(name: &str) -> String {
    std::fs::read_to_string(example(name)).unwrap()
}
fn checked(text: &str) -> zerogravity::Result<CheckedDocument> {
    parse_document(text.as_bytes())?.check()
}
fn execution_refused(config: &pipelines::Config) {
    assert!(matches!(
        pipelines::run(config),
        Err(Error::Unsupported(message)) if message ==
            "compiled multi-model pipeline execution NOT IMPLEMENTED; no source or device was opened"
    ));
}

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zg-sdk-pipeline-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn examples_check_through_the_facade_without_becoming_runnable() {
    for (file, stages, routes) in [
        ("voice-assistant.yaml", None, 2),
        ("transcribe-diarize.yaml", None, 1),
        ("text-model.yaml", Some("generate"), 1),
        ("image-generation.yaml", None, 1),
        ("image-latent-assembly.yaml", Some("solve -> decode"), 1),
        ("audio-codec.yaml", Some("encode -> decode"), 0),
        ("video-stream.yaml", Some("generate"), 1),
        ("decision.yaml", Some("decide"), 0),
    ] {
        let report = load_document(example(file)).unwrap().check().unwrap();
        assert_eq!(report.shape.route_count, routes, "{file}");
        if let Some(stages) = stages {
            assert!(
                report
                    .shape
                    .ordered
                    .iter()
                    .any(|(_, order)| order.join(" -> ") == stages)
            );
        }
        if file == "voice-assistant.yaml" {
            assert_eq!(report.shape.shared_model_refs, 1);
        }
        assert_eq!(
            pipelines::check(&report.config).unwrap().ordered,
            report.shape.ordered
        );
        execution_refused(&report.config);
    }
}

#[test]
fn language_selection_refuses_unresolved_pins_media_and_cold_execution() {
    let mut graph = checked(&text_example("text-model.yaml")).unwrap().config;
    assert_eq!(graph.language_chain("chat").unwrap()[0].id, "generate");
    assert!(matches!(
        zerogravity::Runtime::builder()
            .build()
            .unwrap()
            .prepare_pipeline(&graph, "chat", &[]),
        Err(Error::Unsupported(_))
    ));
    graph.models[0].source = Some("/not-opened/model.gguf".into());
    assert!(matches!(
        graph.language_chain("chat"),
        Err(Error::Unsupported(_))
    ));
    graph.models[0].source = None;
    graph.pipelines[0].realtime = true;
    assert!(matches!(
        graph.language_chain("chat"),
        Err(Error::Unsupported(_))
    ));
    let media = checked(&text_example("voice-assistant.yaml"))
        .unwrap()
        .config;
    assert!(matches!(
        media.language_chain(&media.pipelines[0].name),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn schemas_ports_cycles_models_and_route_validation_are_not_cli_policy() {
    let baseline = text_example("text-model.yaml");
    for text in [
        baseline.replace("schema: zerogravity.pipeline.v1", "schema: v2"),
        baseline.replace("output: text", "output: audio-pcm"),
        baseline.replace("to: generate.text", "to: generate.missing"),
        baseline.replace("to: output.value", "to: generate.text"),
        baseline.replace("id: generate", "id: input"),
        baseline.replace("protocol: openai-text", "protocol: webrtc"),
        baseline.replace("model: model", "model: missing"),
        baseline.replace("target: qwen3.5-2b", "target: unregistered-future-model"),
        baseline.replace("target: qwen3.5-2b", "target: ''"),
        baseline.replace("kind: llm", "kind: tts"),
        baseline.replace("/v1/chat/completions", "relative-route"),
        baseline.replace("/v1/chat/completions", "/v1/chat?private=true"),
        baseline.replace("/v1/chat/completions", "/v1/chat#fragment"),
        format!("{baseline}\nunknown_command: run-anything\n"),
    ] {
        assert_ne!(text, baseline);
        assert!(checked(&text).is_err(), "{text}");
    }
}

#[test]
fn context_is_opt_in_and_disabling_it_preserves_the_graph() {
    let baseline = text_example("contextual-voice-assistant.yaml");
    let enabled = checked(&baseline).unwrap();
    assert_eq!(enabled.shape.shared_model_refs, 2);
    assert_eq!(enabled.shape.context_configured_pipelines, 1);
    assert_eq!(enabled.shape.context_enabled_pipelines, 1);
    assert!(enabled.shape.ordered.iter().any(|(_, order)| {
        order.join(" -> ") == "listen -> speaker -> translate -> prompt -> reply -> speak"
    }));
    let context = enabled.config.pipelines[0].chat_context.as_ref().unwrap();
    assert_eq!(
        context.selection.sources,
        zerogravity::context_filter::default_sources()
    );
    assert_eq!(context.max_bytes, 4096);
    let disabled = checked(&baseline.replace("enabled: true", "enabled: false")).unwrap();
    assert_eq!(enabled.shape.ordered, disabled.shape.ordered);
    assert_eq!(disabled.shape.context_configured_pipelines, 1);
    assert_eq!(disabled.shape.context_enabled_pipelines, 0);
    execution_refused(&enabled.config);
    execution_refused(&disabled.config);
}

#[test]
fn bad_context_bounds_sources_spoofed_cycles_and_unknown_fields_refuse() {
    let original = text_example("contextual-voice-assistant.yaml");
    for text in [
        original.replace("max_bytes: 4096", "max_bytes: 0"),
        original.replace("max_bytes: 4096", "max_bytes: 1048577"),
        original.replace("from_model: assistant", "from_model: voice"),
        original.replace("from_model: assistant", "from_model: missing"),
        original.replace(
            "    chat_context:\n      enabled: true\n      from_model: assistant\n      max_bytes: 4096\n", ""),
        original.replace("from: context.value", "from: reply.text"),
        original.replace("to: listen.context", "to: listen.audio"),
        original.replace("id: listen", "id: context"),
        original.replace("max_bytes: 4096", "max_bytes: -1"),
        original.replace("max_bytes: 4096", "max_bytes: 4096\n      allow_cross_session: true"),
        original.replace("enabled: true", "enabled: \"yes\""),
    ] {
        assert_ne!(text, original);
        assert!(checked(&text).is_err(), "{text}");
    }
}

#[test]
fn ordered_context_filters_and_decisions_are_structural_only() {
    let baseline = text_example("contextual-voice-filtered.yaml");
    let report = checked(&baseline).unwrap();
    assert_eq!(report.config.models.len(), 5);
    assert_eq!(report.shape.shared_model_refs, 2);
    assert_eq!(report.shape.context_filter_stages, 3);
    assert_eq!(report.shape.context_decision_stages, 1);
    assert_eq!(report.shape.context_enabled_pipelines, 1);
    execution_refused(&report.config);
    for text in [
        baseline.replace("target: nanojev-decision", "target: qwen3.5-2b"),
        baseline.replace("model: gate", "model: missing"),
        baseline.replace("- session/tool_result", "- session/user_message"),
        baseline.replace(
            "shared:team-support/assistant_message",
            "shared:../assistant_message",
        ),
        baseline.replace("min_confidence_bps: 7500", "min_confidence_bps: 10001"),
        baseline.replace("on_unavailable: omit", "on_unavailable: elevate"),
        baseline.replace("field: kind", "field: regex_script"),
        baseline.replace("- tool_call", "- bad_role"),
        baseline.replace("kind: decision", "kind: arbitrary_code"),
        baseline.replace("      sources:\n", "      sources: []\n"),
    ] {
        assert_ne!(text, baseline);
        assert!(checked(&text).is_err(), "{text}");
    }
}

#[test]
fn yaml_anchors_aliases_and_shallow_merges_are_structural_conveniences_only() {
    let baseline = text_example("text-model.yaml");
    for text in [
        baseline
            .replace("id: model", "id: &shared_model_name model")
            .replace("model: model", "model: *shared_model_name"),
        baseline.replace(
            "  - id: model\n    target: qwen3.5-2b",
            "  - <<: &shared_model_fields {target: qwen3.5-2b}\n    id: model",
        ),
    ] {
        assert_ne!(text, baseline);
        let report = checked(&text).unwrap();
        assert_eq!(
            report.shape.ordered,
            vec![("chat".into(), vec!["generate".into()])]
        );
        execution_refused(&report.config);
    }
}

#[test]
fn yaml_tags_duplicate_keys_multiple_documents_cycles_and_legacy_toml_refuse() {
    let baseline = text_example("text-model.yaml");
    for text in [
        baseline.replace(
            "target: qwen3.5-2b",
            "target: !untrusted_constructor qwen3.5-2b",
        ),
        baseline.replace(
            "schema: zerogravity.pipeline.v1",
            "schema: &recursive [*recursive]",
        ),
        baseline.replace(
            "schema: zerogravity.pipeline.v1",
            "schema: zerogravity.pipeline.v1\nschema: v2",
        ),
        baseline.replace(
            "target: qwen3.5-2b",
            "target: qwen3.5-2b\n    target: qwen3.5-2b",
        ),
        format!("{baseline}\n---\nschema: zerogravity.pipeline.v1\n"),
        baseline.replace(
            "  - id: model",
            "  - <<: invalid-scalar-merge\n    id: model",
        ),
        "schema = \"zerogravity.pipeline.v1\"\n[[model]]\nid = \"qwen\"\n".into(),
        format!("{baseline}\nunknown_schema_key: !exec [arbitrary, code]\n"),
    ] {
        assert!(parse_document(text.as_bytes()).is_err(), "{text}");
    }
}

#[test]
fn yaml_size_utf8_mapping_key_depth_and_alias_expansion_bounds() {
    for bytes in [vec![], vec![b' '; MAX_CONFIG_BYTES + 1]] {
        assert!(
            matches!(parse_document(&bytes), Err(Error::Limit(m)) if m == "pipeline config size limit")
        );
    }
    assert!(
        matches!(parse_document(&[0xff]), Err(Error::Malformed(m)) if m == "pipeline config not UTF-8")
    );
    for text in [
        "schema: zerogravity.pipeline.v1\n1: value\n".to_owned(),
        format!(
            "schema: zerogravity.pipeline.v1\n{}: value\n",
            "k".repeat(129)
        ),
    ] {
        assert!(
            matches!(parse_document(text.as_bytes()), Err(Error::Malformed(m)) if m == "YAML mapping keys must be bounded strings")
        );
    }
    let deep = format!(
        "schema: zerogravity.pipeline.v1\nnested: {}0{}\n",
        "[".repeat(33),
        "]".repeat(33)
    );
    assert!(
        matches!(parse_document(deep.as_bytes()), Err(Error::Limit(m)) if m == "expanded YAML pipeline size/depth")
    );
    let chunks = (0..512)
        .map(|n| format!("item{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    let aliases = std::iter::repeat_n("*words", 40)
        .collect::<Vec<_>>()
        .join(", ");
    let bomb = format!(
        "schema: zerogravity.pipeline.v1\nwords: &words [{chunks}]\nexpanded: [{aliases}]\n"
    );
    assert!(bomb.len() < MAX_CONFIG_BYTES);
    assert!(matches!(
        parse_document(bomb.as_bytes()),
        Err(Error::Limit(_)) | Err(Error::Malformed(_))
    ));
    let aliases = std::iter::repeat_n("*word", 80)
        .collect::<Vec<_>>()
        .join(", ");
    let scalars = format!(
        "schema: zerogravity.pipeline.v1\nword: &word {}\nexpanded: [{aliases}]\n",
        "x".repeat(4096)
    );
    assert!(scalars.len() < MAX_CONFIG_BYTES);
    assert!(
        matches!(parse_document(scalars.as_bytes()), Err(Error::Limit(m)) if m == "expanded YAML scalar bytes")
    );
}

const DIRECT: &str = "schema: zerogravity.pipeline.v2\nmodel:\n  - id: main\n    target: qwen3.5-0.8b\n    source: ./tiny.gguf\n    model_config: ./config.json\n    tokenizer: ./tokenizer.json\n    snapshot: declared-only\n    native_artifacts: ./native\nrun:\n  entrypoint: {model: main}\n  backend: cpu\n  device: [cpu]\n  headroom: '0'\n  numerical: source\n  speculation: off\n  draft_device: [cpu]\n  task: {raw: true, max_tokens: 2, prompt: configured, context: 64, teacher_ids: [1, 2], prefill_chunk: 1}\n  diagnostics: {explain: false, dry_run: true, audit_logits_jsonl: ./logits.jsonl, audit_layers_jsonl: ./layers.jsonl}\n";

#[test]
fn v2_run_and_sidecar_schema_remain_typed_and_entrypoint_is_checked() {
    let report = checked(DIRECT).unwrap();
    assert!(report.config.pipelines.is_empty());
    assert_eq!(
        report.shape.ordered,
        vec![("run-direct-entry".into(), vec!["generate".into()])]
    );
    let run = report.run.as_ref().unwrap();
    assert_eq!(
        run.entrypoint.as_ref().unwrap().typed(),
        Entrypoint::Model("main".into())
    );
    let task = run.task.as_ref().unwrap();
    assert_eq!(task.prompt.as_deref(), Some("configured"));
    assert_eq!(task.raw, Some(true));
    assert_eq!(task.teacher_ids.as_deref(), Some([1, 2].as_slice()));
    assert_eq!(run.diagnostics.as_ref().unwrap().explain, Some(false));
    assert_eq!(
        report.config.models[0].tokenizer.as_deref(),
        Some("./tokenizer.json")
    );
    execution_refused(&report.config);
    assert!(checked(&DIRECT.replace("{model: main}", "{model: missing}")).is_err());
    assert!(checked(&DIRECT.replace("{model: main}", "{pipeline: missing}")).is_err());
    let pipeline = format!(
        "{}\nrun:\n  entrypoint: {{pipeline: chat}}\n",
        text_example("text-model.yaml").replace("pipeline.v1", "pipeline.v2")
    );
    assert_eq!(checked(&pipeline).unwrap().shape.ordered[0].0, "chat");
    // Selecting a valid direct model must still validate other declarations.
    let bad_graph = text_example("text-model.yaml").replace("model: model", "model: missing");
    let bad_graph = format!(
        "{}\nrun:\n  entrypoint: {{model: model}}\n",
        bad_graph.replace("pipeline.v1", "pipeline.v2")
    );
    assert!(checked(&bad_graph).is_err());
}

#[test]
fn v1_refuses_run_and_each_v2_sidecar_without_changing_null_compatibility() {
    let baseline = text_example("text-model.yaml");
    for field in ["model_config", "tokenizer", "snapshot", "native_artifacts"] {
        let text = baseline.replace(
            "target: qwen3.5-2b",
            &format!("target: qwen3.5-2b\n    {field}: ./declared"),
        );
        assert!(
            matches!(checked(&text), Err(Error::Malformed(m)) if m == "run/source-sidecar fields require zerogravity.pipeline.v2")
        );
    }
    assert!(
        matches!(checked(&format!("{baseline}\nrun: {{}}\n")), Err(Error::Malformed(m)) if m == "run/source-sidecar fields require zerogravity.pipeline.v2")
    );
    let optional_null =
        baseline.replace("target: qwen3.5-2b", "target: qwen3.5-2b\n    source: null");
    checked(&optional_null).unwrap();
    assert!(
        matches!(checked(&optional_null.replace("pipeline.v1", "pipeline.v2")), Err(Error::Malformed(m)) if m == "pipeline.v2 does not accept null fields; omit optional values")
    );
}

#[test]
fn v2_null_unknown_duplicate_and_ambiguous_run_fields_refuse() {
    for text in [
        DIRECT.replace("raw: true", "raw: null"),
        DIRECT.replace("teacher_ids: [1, 2]", "teacher_ids: [1, null]"),
        DIRECT.replace("backend: cpu", "backend: cpu\n  unknown: true"),
        DIRECT.replace("max_tokens: 2", "max_tokens: 2, max_tokens: 3"),
        DIRECT.replace("raw: true", "unknown: true"),
        DIRECT.replace("explain: false", "unknown: false"),
        DIRECT.replace("{model: main}", "{model: main, pipeline: chat}"),
        DIRECT.replace("{model: main}", "{unknown: main}"),
        DIRECT.replace(
            "target: qwen3.5-0.8b",
            "target: qwen3.5-0.8b\n    unknown: false",
        ),
    ] {
        assert!(parse_document(text.as_bytes()).is_err(), "{text}");
    }
}

#[test]
fn file_loader_is_bounded_and_does_not_open_declared_sources_or_write_diagnostics() {
    let dir = TempDir::new();
    let file = dir.0.join("config.yaml");
    assert!(matches!(load_document(&file), Err(Error::Io(_))));
    std::fs::write(&file, []).unwrap();
    assert!(matches!(load_document(&file), Err(Error::Limit(_))));
    std::fs::write(&file, vec![b' '; MAX_CONFIG_BYTES + 1]).unwrap();
    assert!(matches!(load_document(&file), Err(Error::Limit(_))));
    std::fs::write(&file, [0xff]).unwrap();
    assert!(matches!(load_document(&file), Err(Error::Malformed(_))));
    let log = dir.0.join("not-created.jsonl");
    let text = DIRECT.replace("./logits.jsonl", log.to_str().unwrap());
    std::fs::write(&file, text).unwrap();
    let report = load_document(&file).unwrap().check().unwrap();
    assert_eq!(
        report.config.models[0].source.as_deref(),
        Some("./tiny.gguf")
    );
    assert!(!log.exists());
    execution_refused(&report.config);
    assert!(!log.exists());
    assert!(matches!(
        checked(&DIRECT.replace("./tiny.gguf", "unknown://source")),
        Err(Error::Malformed(_))
    ));
}

#[cfg(unix)]
#[test]
fn explicit_config_symlinks_are_followed_but_source_symlinks_are_not_opened() {
    use std::os::unix::fs::symlink;
    let dir = TempDir::new();
    let file = dir.0.join("config.yaml");
    let alias = dir.0.join("alias.yaml");
    let source = dir.0.join("source.gguf");
    symlink(dir.0.join("absent.gguf"), &source).unwrap();
    let text = DIRECT.replace("./tiny.gguf", source.to_str().unwrap());
    std::fs::write(&file, text).unwrap();
    symlink(&file, &alias).unwrap();
    let report = load_document(&alias).unwrap().check().unwrap();
    assert_eq!(report.config.models[0].source.as_deref(), source.to_str());
    execution_refused(&report.config);
    std::fs::remove_file(&file).unwrap();
    assert!(matches!(load_document(&alias), Err(Error::Io(_))));
}
