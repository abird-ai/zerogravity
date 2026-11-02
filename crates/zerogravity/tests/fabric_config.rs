//! Public-facade-only configuration policy; no source, device or network IO.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use zerogravity::{
    Error,
    pipelines::*,
    run::config::{RunOverrides, resolve},
    run_policy::*,
};

struct Fixture(PathBuf);
impl Fixture {
    fn new(body: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "zg-sdk-config-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("run.yaml"), body).unwrap();
        Self(path)
    }
    fn resolve(&self, overrides: RunOverrides) -> zerogravity::Result<Request> {
        resolve(
            Some(Path::new("run.yaml")),
            overrides,
            &self.0,
            Origin::Caller,
        )
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
const MODEL: &str = "schema: zerogravity.pipeline.v2\nmodel:\n  - id: main\n    target: qwen3.5-0.8b\n    source: ./absent.gguf\n    model_config: ./geometry.json\n    tokenizer: ./tokenizer.json\n    native_artifacts: ./native\nrun:\n  entrypoint: {model: main}\n  backend: cpu\n  task: {raw: true, prompt: configured, max_tokens: 2}\n  diagnostics: {explain: true, audit_logits_jsonl: ./audit.jsonl}\n";
const GRAPH: &str = "schema: zerogravity.pipeline.v2\nmodel:\n  - id: recognizer\n    target: nvidia-canary-qwen-2.5b\n    source: ./recognizer.gguf\n  - id: assistant\n    target: qwen3.5-2b\n    source: ./assistant.gguf\npipeline:\n  - name: talk\n    input: audio-pcm\n    output: text\n    realtime: true\n    chat_context:\n      enabled: true\n      from_model: assistant\n      max_bytes: 64\n      sources: [session/user_message, session/tool_call]\n      filter:\n        - {kind: match, action: exclude, field: kind, values: [tool_call]}\n    step:\n      - {id: listen, kind: asr, model: recognizer}\n      - {id: prompt, kind: transcript-text}\n      - {id: reply, kind: llm, model: assistant}\n    link:\n      - {from: input.value, to: listen.audio}\n      - {from: context.value, to: listen.context}\n      - {from: listen.transcript, to: prompt.transcript}\n      - {from: prompt.text, to: reply.text}\n      - {from: reply.text, to: output.value}\nroute:\n  - {path: /v1/chat/completions, protocol: openai-text, pipeline: talk}\nrun:\n  entrypoint: {pipeline: talk}\n";

#[test]
fn explicit_document_precedence_paths_and_no_source_or_audit_io() {
    let fixture = Fixture::new(MODEL);
    let base = fixture.resolve(RunOverrides::default()).unwrap();
    assert_eq!(
        base.provenance["task.raw"],
        Origin::Config(fixture.0.join("run.yaml"))
    );
    assert_eq!(
        base.graph.models[0].source.as_deref(),
        Some(fixture.0.join("./absent.gguf").to_str().unwrap())
    );
    assert_eq!(
        base.graph.models[0].model_config.as_deref(),
        Some(fixture.0.join("./geometry.json").to_str().unwrap())
    );
    assert_eq!(
        base.graph.models[0].tokenizer.as_deref(),
        Some(fixture.0.join("./tokenizer.json").to_str().unwrap())
    );
    assert_eq!(
        base.graph.models[0].native_artifacts.as_deref(),
        Some(fixture.0.join("./native").to_str().unwrap())
    );
    assert_eq!(
        base.diagnostics.audit_logits_jsonl,
        Some(fixture.0.join("./audit.jsonl"))
    );
    let request = fixture
        .resolve(RunOverrides {
            model: Some("./replacement.gguf".into()),
            tokenizer: Some("./replacement.json".into()),
            task: TaskDoc {
                raw: Some(false),
                prompt: Some(String::new()),
                max_tokens: Some(1),
                ..Default::default()
            },
            diagnostics: DiagnosticDoc {
                explain: Some(false),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
    assert!(!request.task.raw && !request.diagnostics.explain);
    assert_eq!(request.task.prompt.as_deref(), Some(""));
    assert_eq!(request.task.max_tokens, 1);
    assert_eq!(request.provenance["task.raw"], Origin::Caller);
    assert_eq!(request.provenance["model.main.source"], Origin::Caller);
    assert_eq!(request.provenance["model.main.tokenizer"], Origin::Caller);
    assert_eq!(
        request.graph.models[0].tokenizer.as_deref(),
        Some(fixture.0.join("./replacement.json").to_str().unwrap())
    );
    assert_eq!(std::fs::read_dir(&fixture.0).unwrap().count(), 1);
}

#[test]
fn caller_base_and_document_base_are_independent() {
    let fixture = Fixture::new(MODEL);
    let request = resolve(
        Some(&fixture.0.join("run.yaml")),
        RunOverrides {
            model: Some("./caller.gguf".into()),
            diagnostics: DiagnosticDoc {
                audit_layers_jsonl: Some("./layers.jsonl".into()),
                ..Default::default()
            },
            ..Default::default()
        },
        Path::new("/caller"),
        Origin::Caller,
    )
    .unwrap();
    assert_eq!(
        request.graph.models[0].source.as_deref(),
        Some("/caller/./caller.gguf")
    );
    assert_eq!(
        request.diagnostics.audit_layers_jsonl,
        Some(PathBuf::from("/caller/./layers.jsonl"))
    );
    assert_eq!(
        request.diagnostics.audit_logits_jsonl,
        Some(fixture.0.join("./audit.jsonl"))
    );
    assert!(
        resolve(
            None,
            RunOverrides::default(),
            Path::new("relative"),
            Origin::Caller
        )
        .is_err()
    );
}

#[test]
fn absent_false_and_empty_graph_overrides_have_distinct_meanings() {
    let fixture = Fixture::new(GRAPH);
    let base = fixture.resolve(RunOverrides::default()).unwrap();
    assert!(base.graph.pipelines[0].realtime);
    let request = fixture
        .resolve(RunOverrides {
            graph: Some(PipelinePatch {
                realtime: Some(false),
                routes: Some(vec![]),
                chat_context: Some(ChatContextPatch {
                    enabled: Some(false),
                    stages: Some(vec![]),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        })
        .unwrap();
    assert!(!request.graph.pipelines[0].realtime);
    assert!(request.graph.routes.is_empty());
    let context = request.graph.pipelines[0].chat_context.as_ref().unwrap();
    assert!(!context.enabled);
    assert_eq!(context.max_bytes, 64);
    assert_eq!(context.from_model, "assistant");
    assert_eq!(context.selection.sources.len(), 2);
    assert!(context.selection.stages.is_empty());
    assert_eq!(request.graph.pipelines[0].steps.len(), 3);
    assert!(matches!(
        zerogravity::pipelines::run(&request.graph),
        Err(Error::Unsupported(_))
    ));
    for patch in [
        PipelinePatch {
            models: Some(vec![]),
            ..Default::default()
        },
        PipelinePatch {
            steps: Some(vec![]),
            ..Default::default()
        },
        PipelinePatch {
            links: Some(vec![]),
            ..Default::default()
        },
        PipelinePatch {
            chat_context: Some(ChatContextPatch {
                sources: Some(vec![]),
                ..Default::default()
            }),
            ..Default::default()
        },
    ] {
        assert!(
            fixture
                .resolve(RunOverrides {
                    graph: Some(patch),
                    ..Default::default()
                })
                .is_err()
        );
    }
}

#[test]
fn patch_failure_is_atomic_and_structural_success_is_not_execution() {
    let mut graph = parse_document(GRAPH.as_bytes()).unwrap().compile().unwrap();
    let mut entry = Some(Entrypoint::Model("main".into()));
    let before = format!("{graph:?}");
    assert!(
        PipelinePatch {
            name: Some("text".into()),
            models: Some(vec![]),
            ..Default::default()
        }
        .apply(&mut graph, &mut entry)
        .is_err()
    );
    assert_eq!(format!("{graph:?}"), before);
    assert_eq!(entry, Some(Entrypoint::Model("main".into())));
    entry = Some(Entrypoint::Pipeline("other".into()));
    assert!(
        PipelinePatch {
            name: Some("text".into()),
            ..Default::default()
        }
        .apply(&mut graph, &mut entry)
        .is_err()
    );
    assert_eq!(format!("{graph:?}"), before);
    assert!(
        PipelinePatch {
            name: Some("new".into()),
            input: Some(ValueKind::Text),
            output: Some(ValueKind::Text),
            chat_context: Some(ChatContextPatch {
                enabled: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        }
        .compile()
        .is_err()
    );
}

#[test]
fn source_selector_resolution_never_acquires_or_fabricates_a_target() {
    for source in [
        "./missing.gguf",
        "hf:missing/repository",
        "catalog:unpublished",
        "https://invalid.example/model.gguf",
    ] {
        let request = resolve(
            None,
            RunOverrides {
                model: Some(source.into()),
                ..Default::default()
            },
            Path::new("/not-created"),
            Origin::Caller,
        )
        .unwrap();
        assert!(request.graph.models[0].target.is_empty());
        assert_eq!(request.provenance["model.main.source"], Origin::Caller);
        assert_eq!(request.provenance["task.max_tokens"], Origin::Builtin);
    }
    assert!(
        resolve(
            None,
            RunOverrides::default(),
            Path::new("/"),
            Origin::Caller
        )
        .is_err()
    );
}

#[test]
fn typed_model_replacement_anchors_all_source_paths() {
    let fixture = Fixture::new(GRAPH);
    let mut models = fixture
        .resolve(RunOverrides::default())
        .unwrap()
        .graph
        .models;
    for model in &mut models {
        model.source = Some("./replacement.gguf".into());
        model.model_config = Some("./shape.json".into());
        model.tokenizer = Some("./tokens.json".into());
        model.native_artifacts = Some("./native".into());
    }
    let request = fixture
        .resolve(RunOverrides {
            graph: Some(PipelinePatch {
                models: Some(models),
                ..Default::default()
            }),
            ..Default::default()
        })
        .unwrap();
    for model in request.graph.models {
        assert_eq!(
            model.source.as_deref(),
            Some(fixture.0.join("./replacement.gguf").to_str().unwrap())
        );
        assert_eq!(
            model.model_config.as_deref(),
            Some(fixture.0.join("./shape.json").to_str().unwrap())
        );
        assert_eq!(
            model.tokenizer.as_deref(),
            Some(fixture.0.join("./tokens.json").to_str().unwrap())
        );
        assert_eq!(
            model.native_artifacts.as_deref(),
            Some(fixture.0.join("./native").to_str().unwrap())
        );
        assert_eq!(
            request.provenance[&format!("model.{}.source", model.id)],
            Origin::Caller
        );
    }
    assert_eq!(std::fs::read_dir(&fixture.0).unwrap().count(), 1);
}

#[test]
fn malformed_config_is_not_hidden_by_a_valid_override() {
    for document in [
        MODEL.replace("backend: cpu", "backend: invalid"),
        MODEL.replace("raw: true", "raw: null"),
        MODEL.replace("max_tokens: 2", "max_tokens: 2, max_tokens: 3"),
        MODEL.replace("target: qwen3.5-0.8b", "target: ''"),
    ] {
        let fixture = Fixture::new(&document);
        assert!(
            fixture
                .resolve(RunOverrides {
                    backend: Some(Backend::Cpu),
                    ..Default::default()
                })
                .is_err()
        );
    }
}
