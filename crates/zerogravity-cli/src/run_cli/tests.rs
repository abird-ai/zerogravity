use super::*;
fn words(s: &[&str]) -> Vec<String> {
    s.iter().map(|s| s.to_string()).collect()
}
#[test]
fn source_and_alias_policy_before_effects() {
    for s in [
        "hf:org/file.yaml",
        "gh:org/repo.yml",
        "https://host/file.yaml?locator=one",
    ] {
        let a = parse(&words(&[s]), None).unwrap();
        assert!(a.model.is_some());
        assert!(a.config.is_none());
    }
    assert!(matches!(
        SourceSelector::parse("qwen3.5-2b").unwrap(),
        SourceSelector::Catalog(_)
    ));
    assert!(parse(&words(&["x.gguf", "--model", "y.gguf"]), None).is_err());
    assert!(parse(&words(&["x.gguf", "--backend", "auto"]), Some("run-cpu")).is_err());
    assert!(parse(&words(&["x.gguf", "-d", "cpu:0,gpu:0"]), None).is_err());
    assert!(parse(&words(&["x.gguf", "-d", "cpu:0"]), Some("run-cuda")).is_err());
    let a = parse(
        &words(&["x.gguf", "--raw=false", "--prompt=", "--headroom=0"]),
        None,
    )
    .unwrap();
    assert_eq!(a.task.raw, Some(false));
    assert_eq!(a.task.prompt.as_deref(), Some(""));
    assert_eq!(a.headroom, Some(Headroom::Bytes(0)));
}
#[test]
fn full_physical_identity_and_shared_domain() {
    assert_eq!(
        devices::normalize_pci("00000000:01:02.3").unwrap(),
        "0000:01:02.3"
    );
    for s in ["01:02.3", "0000:01:20.0", "0000:01:02.8"] {
        assert!(devices::normalize_pci(s).is_err());
    }
    assert_eq!(
        Headroom::Percent(99).bytes(u64::MAX, false).unwrap(),
        u64::MAX / 100 * 99 + (u64::MAX % 100) * 99 / 100
    );
    assert!(Headroom::parse("100%").is_err());
    assert!(Headroom::parse("18446744073709551615GiB").is_err());
}
#[test]
fn config_xdg_origins_strictness_and_false_override() {
    let root = std::env::temp_dir().join(format!("zg-run-config-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let config = root.join("run.yaml");
    let document = "schema: zerogravity.pipeline.v2\nmodel:\n  - id: main\n    target: qwen3.5-0.8b\n    source: ./tiny.gguf\nrun:\n  entrypoint: {model: main}\n  backend: cpu\n  task: {raw: true, max_tokens: 2, prompt: configured}\n";
    std::fs::write(&config, document).unwrap();
    let a = parse(
        &words(&[
            "--config",
            config.to_str().unwrap(),
            "--raw=false",
            "--max-tokens=1",
            "--prompt=",
        ]),
        None,
    )
    .unwrap();
    let r = config::resolve(a, Path::new("/"), None, Some("/home/test".into())).unwrap();
    assert!(!r.task.raw);
    assert_eq!(r.task.max_tokens, 1);
    assert_eq!(r.task.prompt.as_deref(), Some(""));
    assert_eq!(
        r.graph.models[0].source.as_deref(),
        Some(root.join("./tiny.gguf").to_str().unwrap())
    );
    assert_eq!(r.provenance["task.raw"], Origin::Cli);
    for body in [
        document.replace("raw: true", "raw: null"),
        document.replace("backend: cpu", "backend: cpu\n  unknown: true"),
        document.replace("max_tokens: 2", "max_tokens: 2, max_tokens: 3"),
    ] {
        std::fs::write(&config, body).unwrap();
        assert!(crate::pipeline::load_document(config.to_str().unwrap()).is_err());
    }
    assert!(
        config::resolve(
            Arguments::default(),
            Path::new("/"),
            Some("relative".into()),
            Some("relative-home".into())
        )
        .is_err()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn catalogue_reference_spelling_and_local_precedence() {
    for reference in [
        "huggingface/fixture/tiny/v1",
        "huggingface-bundle/fixture/tiny/v1",
        "github/fixture/tiny/v1",
        "github-bundle/fixture/tiny/v1",
        "catalog:huggingface/fixture/tiny/v1",
        "catalog:model.yaml",
    ] {
        assert!(
            matches!(
                SourceSelector::parse(reference),
                Ok(SourceSelector::Catalog(_))
            ),
            "{reference}"
        );
    }
    assert_eq!(
        SourceSelector::parse("catalog:model.yaml").unwrap(),
        SourceSelector::Catalog("model.yaml".into())
    );
    assert!(SourceSelector::parse("catalog:").is_err());
    assert!(SourceSelector::has_provider_scheme("catalog:model.yaml"));
    for local in [
        "/models/huggingface/org/repo/rev",
        "./huggingface/org/repo/rev",
        "../huggingface/org/repo/rev",
        "./catalog:model.yaml",
    ] {
        assert!(
            matches!(SourceSelector::parse(local), Ok(SourceSelector::Local(_))),
            "{local}"
        );
    }
    // Explicit catalogue schemes beat a local YAML suffix in positionals.
    let a = parse(&words(&["catalog:run.yaml"]), None).unwrap();
    assert_eq!(a.model.as_deref(), Some("catalog:run.yaml"));
    assert!(a.config.is_none());
    let a = parse(&words(&["huggingface/fixture/tiny/v1"]), None).unwrap();
    assert!(a.model.is_some());
    assert!(a.config.is_none());
    let a = parse(&words(&["./huggingface/run.yaml"]), None).unwrap();
    assert!(a.config.is_some());
    assert!(a.model.is_none());
}

const GRAPH: &str = "schema: zerogravity.pipeline.v2\nmodel:\n  - id: recognizer\n    target: nvidia-canary-qwen-2.5b\n    source: ./recognizer.gguf\n  - id: assistant\n    target: qwen3.5-2b\n    source: ./assistant.gguf\npipeline:\n  - name: talk\n    input: audio-pcm\n    output: text\n    chat_context:\n      enabled: true\n      from_model: assistant\n      max_bytes: 64\n      sources: [session/user_message, session/tool_call]\n      filter:\n        - {kind: match, action: exclude, field: kind, values: [tool_call]}\n    step:\n      - {id: listen, kind: asr, model: recognizer}\n      - {id: prompt, kind: transcript-text}\n      - {id: reply, kind: llm, model: assistant}\n    link:\n      - {from: input.value, to: listen.audio}\n      - {from: context.value, to: listen.context}\n      - {from: listen.transcript, to: prompt.transcript}\n      - {from: prompt.text, to: reply.text}\n      - {from: reply.text, to: output.value}\nroute:\n  - {path: /v1/chat/completions, protocol: openai-text, pipeline: talk}\nrun:\n  entrypoint: {pipeline: talk}\n";
fn graph_resolve(tag: &str, extra: &[&str]) -> Result<Request> {
    let root = std::env::temp_dir().join(format!("zg-graph-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let config = root.join("run.yaml");
    std::fs::write(&config, GRAPH).unwrap();
    let mut args = vec!["--config", config.to_str().unwrap()];
    args.extend_from_slice(extra);
    let a = parse(&words(&args), None).unwrap();
    config::resolve(a, Path::new("/"), None, Some("/home/test".into()))
}

#[test]
fn graph_empty_list_spelling_is_field_typed() {
    let base = graph_resolve("base", &[]).unwrap();
    assert_eq!(base.graph.routes.len(), 1);
    let context = base.graph.pipelines[0].chat_context.as_ref().unwrap();
    assert_eq!(context.selection.sources.len(), 2);
    assert_eq!(context.selection.stages.len(), 1);
    // Every scalar field refuses the list-clear spelling.
    for flag in [
        "--name=[]",
        "--input=[]",
        "--output=[]",
        "--chat-context-from=[]",
        "--chat-context-enabled=[]",
        "--chat-context-max-bytes=[]",
    ] {
        assert!(graph_resolve("scalar", &[flag]).is_err(), "{flag}");
    }
    assert!(parse(&words(&["x.gguf", "--realtime=[]"]), None).is_err());
    // A list clear replaces only that whole list.
    let cleared = graph_resolve("stages", &["--chat-context-match=[]"]).unwrap();
    let context = cleared.graph.pipelines[0].chat_context.as_ref().unwrap();
    assert!(context.selection.stages.is_empty());
    assert_eq!(context.selection.sources.len(), 2);
    assert!(context.enabled);
    assert_eq!(context.max_bytes, 64);
    let cleared = graph_resolve("sources", &["--chat-context-source=[]"]);
    // Empty sources are cleared, not restored; existing validation then refuses
    // a policy that selects no record source at all.
    assert!(cleared.is_err());
    let cleared = graph_resolve(
        "both",
        &["--chat-context-source=[]", "--chat-context-match=[]"],
    );
    assert!(cleared.is_err());
    let cleared = graph_resolve(
        "source-replaced",
        &[
            "--chat-context-source=session/user_message",
            "--chat-context-match=[]",
        ],
    )
    .unwrap();
    let context = cleared.graph.pipelines[0].chat_context.as_ref().unwrap();
    assert_eq!(context.selection.sources.len(), 1);
    assert!(context.selection.stages.is_empty());
    let cleared = graph_resolve("routes", &["--route=[]"]).unwrap();
    assert!(cleared.graph.routes.is_empty());
    // Ordinary declarations replace the same whole lists.
    let replaced = graph_resolve(
        "steps",
        &[
            "--step=listen=asr:recognizer",
            "--step=prompt=transcript-text",
            "--step=reply=llm:assistant",
        ],
    )
    .unwrap();
    assert_eq!(replaced.graph.pipelines[0].steps.len(), 3);
    // Clear cannot mix with members; other lists then still validate.
    for mixed in [
        ["--route=[]", "--route=/x=openai-text"],
        [
            "--chat-context-match=[]",
            "--chat-context-match=exclude:kind:tool_call",
        ],
        ["--step=[]", "--step=reply=llm:assistant"],
    ] {
        assert!(graph_resolve("mixed", &mixed).is_err(), "{mixed:?}");
    }
    // Clearing models/steps/links leaves a genuinely invalid graph, not a
    // silently restored prior declaration.
    for invalid in [["--define-model=[]"], ["--step=[]"], ["--link=[]"]] {
        assert!(graph_resolve("invalid", &invalid).is_err(), "{invalid:?}");
    }
}

#[test]
fn declared_empty_model_target_requires_direct_source_inference() {
    let root = std::env::temp_dir().join(format!("zg-target-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let config = root.join("run.yaml");
    let body = "schema: zerogravity.pipeline.v2\nmodel:\n  - id: main\n    target: ''\n    source: ./model.gguf\nrun:\n  entrypoint: {model: main}\n";
    std::fs::write(&config, body).unwrap();
    assert!(
        crate::pipeline::load_document(config.to_str().unwrap())
            .unwrap()
            .compile()
            .is_err()
    );
    let a = parse(&words(&["--config", config.to_str().unwrap()]), None).unwrap();
    assert!(config::resolve(a, Path::new("/"), None, Some("/home/test".into())).is_err());
    // A declared CLI graph model also states its target.
    let a = parse(
        &words(&[
            "--define-model",
            "main=@./model.gguf",
            "--name",
            "p",
            "--input",
            "text",
            "--output",
            "text",
        ]),
        None,
    )
    .unwrap();
    assert!(config::resolve(a, Path::new("/tmp"), None, None).is_err());
    // The direct source keeps its later, source-geometry target inference.
    let a = parse(&words(&["--model", "missing.gguf"]), None).unwrap();
    let request = config::resolve(a, Path::new("/tmp"), None, None).unwrap();
    assert_eq!(request.graph.models.len(), 1);
    assert!(request.graph.models[0].target.is_empty());
    assert!(request.graph.models[0].source.is_some());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn every_explicit_device_compatible_before_source_effects() {
    let a = parse(
        &words(&["missing.gguf", "--backend=cuda", "-d", "gpu:0,gpu:1"]),
        None,
    )
    .unwrap();
    let request = config::resolve(a, Path::new("/"), None, Some("/tmp".into())).unwrap();
    let card = |id: &str, vendor: &str| devices::Observation {
        id: id.into(),
        class: "gpu".into(),
        vendor: vendor.into(),
        ceiling: Some(6 * 1024u64.pow(3)),
        available: Some(6 * 1024u64.pow(3)),
        backends: vec![],
        exclusion: None,
        provenance: "pure metadata fixture".into(),
        cuda_ordinal: None,
        cuda_cc: None,
    };
    let inventory = devices::Inventory {
        devices: vec![
            card("pci:0000:01:00.0", "amd"),
            card("pci:0000:02:00.0", "nvidia"),
        ],
        observed_unix_ms: 0,
    };
    assert!(devices::allowed(&request, &inventory).is_err());
}
