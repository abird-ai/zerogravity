use super::*;
fn words(s: &[&str]) -> Vec<String> {
    s.iter().map(|s| s.to_string()).collect()
}
#[test]
fn device_only_route_shares_exact_qualified_policy_with_advanced_flags() {
    let cmd = |flags: &[&str]| parse(&words(flags), None).expect("typed run arguments");
    let device_only = cmd(&["qwen.gguf", "-d", "cpu"]);
    let advanced = cmd(&["qwen.gguf", "--backend", "auto", "--device", "cpu:0"]);
    assert_eq!(device_only.backend, None);
    assert_eq!(advanced.backend, Some(Backend::Auto));
    assert_eq!(
        Devices::parse(device_only.devices.as_ref().unwrap()).unwrap(),
        Devices::parse(advanced.devices.as_ref().unwrap()).unwrap(),
    );
    assert_eq!(
        DeviceRef::parse("local/cpu").unwrap(),
        DeviceRef::parse("cpu:0").unwrap(),
    );
    assert_eq!(cmd(&["qwen.gguf"]).backend, None);
    let specific_gpu = cmd(&["qwen.gguf", "-d", "gpu:0"]);
    assert_eq!(specific_gpu.backend, None);
    assert!(matches!(
        Devices::parse(specific_gpu.devices.as_ref().unwrap()),
        Ok(Devices::Ordered(_))
    ));
    assert_eq!(
        cmd(&["qwen.gguf", "--backend", "cuda", "-d", "gpu:0"]).backend,
        Some(Backend::Cuda)
    );
    assert!(
        parse(
            &words(&["qwen.gguf", "--backend", "cuda", "-d", "cpu"]),
            None
        )
        .is_err()
    );
    assert!(parse(&words(&["qwen.gguf", "--on", "cpu"]), None).is_err());
}

#[test]
fn absolute_dev_node_selectors_use_the_same_device_policy_as_pci() {
    for node in [
        "/dev/dri/renderD128",
        "/dev/dri/card0",
        "/dev/dri/by-path/pci-0000:01:00.0-render",
        "/dev/nvidia0",
    ] {
        let reference = DeviceRef::parse(node).unwrap();
        assert_eq!(reference.host, "local");
        assert_eq!(reference.selector, node);
        assert_eq!(
            Devices::parse(&[node.into()]).unwrap(),
            Devices::Ordered(vec![reference.clone()]),
        );
        assert!(reference.label().ends_with(node));
        assert!(parse(&words(&["missing.gguf", "-d", node]), None).is_ok());
        assert!(
            parse(
                &words(&["missing.gguf", "--backend", "cpu", "-d", node]),
                None
            )
            .is_err()
        );
    }
    for bad in [
        "/dev/",
        "/dev/dri/../kfd",
        "/dev/dri/./renderD128",
        "/dev/dri//renderD128",
        "/dev/dri/renderD128/",
        "/tmp/renderD128",
    ] {
        assert!(DeviceRef::parse(bad).is_err(), "{bad}");
    }
    assert!(Devices::parse(&words(&["cpu", "/dev/dri/renderD128"])).is_err());
    assert!(Devices::parse(&words(&["/dev/dri/renderD128", "cpu"])).is_ok());
}

#[cfg(target_os = "linux")]
#[test]
fn dev_path_and_pci_alias_use_one_observed_gpu_and_refuse_foreign_backend() {
    let inventory = devices::discover().unwrap();
    for node in ["/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/nvidia0"] {
        if !Path::new(node).exists() {
            continue;
        }
        let physical_id = devices::device_node_identity(node).unwrap();
        let policy = |selector: &str, backend: Option<&str>| {
            let mut args = words(&["missing.gguf", "-d", selector]);
            if let Some(backend) = backend {
                args.extend(words(&["--backend", backend]));
            }
            let parsed = parse(&args, None).unwrap();
            config::resolve(parsed, Path::new("/"), None, None).unwrap()
        };
        let by_path = devices::allowed(&policy(node, None), &inventory).unwrap();
        let by_pci = devices::allowed(&policy(&physical_id, None), &inventory).unwrap();
        assert_eq!(by_path.len(), 1);
        assert_eq!(by_path[0].id, physical_id);
        assert_eq!(by_path[0].id, by_pci[0].id);
        // Alias selection retains original vendor, capacities, CUDA ordinal,
        // exclusions and backend permits. It never invents an eligible card.
        assert_eq!(
            serde_json::to_value(&by_path[0]).unwrap(),
            serde_json::to_value(&by_pci[0]).unwrap()
        );
        let foreign = if by_path[0].vendor == "nvidia" {
            "rocm"
        } else {
            "cuda"
        };
        assert!(devices::allowed(&policy(node, Some(foreign)), &inventory).is_err());
        let pair = words(&["missing.gguf", "-d", node, "-d", &physical_id]);
        let duplicate =
            config::resolve(parse(&pair, None).unwrap(), Path::new("/"), None, None).unwrap();
        assert_eq!(devices::allowed(&duplicate, &inventory).unwrap().len(), 1);
    }
}

#[test]
fn device_inventory_display_keeps_observation_separate_from_eligibility() {
    let inventory = devices::Observation {
        id: "pci:0000:01:00.0".into(),
        class: "gpu".into(),
        vendor: "nvidia".into(),
        ceiling: Some(6 * 1024 * 1024 * 1024),
        available: None,
        backends: vec!["cuda".into()],
        exclusion: None,
        provenance: "fixture".into(),
        cuda_ordinal: Some(0),
        cuda_cc: Some((8, 6)),
    };
    let shown = devices::display(&inventory);
    assert!(shown.contains("total=6144 MiB"), "{shown}");
    assert!(shown.contains("available=unknown"), "{shown}");
    assert!(shown.contains("(inventory only)"), "{shown}");
    assert!(!shown.contains("qualified") && !shown.contains("ready"));
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

#[test]
fn positional_scheme_precedence_over_yaml_suffix() {
    // Provider/catalog/URL schemes are model sources regardless of a yaml suffix.
    for s in [
        "hf:org/file.yaml",
        "gh:o/r.yml",
        "catalog:run.yaml",
        "http://h/x.yaml",
        "https://h/x.yml",
    ] {
        let a = parse(&words(&[s]), None).unwrap();
        assert!(a.model.is_some() && a.config.is_none(), "{s}");
    }
    // Pull-shaped catalog namespaces with a yaml final component are catalog
    // model sources, never run documents.
    for s in [
        "huggingface/o/r/v1.yaml",
        "huggingface-bundle/o/r/v1.yml",
        "github/o/r/v1.yaml",
        "github-bundle/o/r/v1.yml",
    ] {
        let a = parse(&words(&[s]), None).unwrap();
        assert_eq!(a.model.as_deref(), Some(s), "{s}");
        assert!(a.config.is_none(), "{s}");
        assert!(
            matches!(SourceSelector::parse(s), Ok(SourceSelector::Catalog(_))),
            "{s}"
        );
        assert!(!SourceSelector::implies_run_document(s).unwrap(), "{s}");
    }
    // Only a local .yaml/.yml path implies a run document. Case-sensitive
    // schemes leave HF:... local-by-slash, so its yaml suffix still selects
    // a run document exactly as before.
    for s in [
        "x.yaml",
        "x.yml",
        "./x.yaml",
        "/abs/x.yml",
        "foo/x.yaml",
        "./huggingface/run.yaml",
        "model.yaml",
        "HF:org/file.yaml",
    ] {
        let a = parse(&words(&[s]), None).unwrap();
        assert_eq!(a.config.as_deref(), Some(s), "{s}");
        assert!(a.model.is_none(), "{s}");
        assert!(SourceSelector::implies_run_document(s).unwrap(), "{s}");
    }
    // Local non-yaml sources and bare catalog names stay model sources.
    for s in ["model.gguf", "./m.gguf", "/abs/m.gguf", "qwen3.5-2b"] {
        let a = parse(&words(&[s]), None).unwrap();
        assert!(a.model.is_some() && a.config.is_none(), "{s}");
    }
    // Invalid positionals fail at parse with their exact selector errors.
    for (s, message) in [
        ("", "bounded nonempty model source required"),
        ("a\u{7}b.yaml", "bounded nonempty model source required"),
        (
            "foo://x.yaml",
            "source scheme unavailable; use an exact local acquired source",
        ),
    ] {
        assert!(
            matches!(
                parse(&words(&[s]), None),
                Err(Error::Malformed(m)) | Err(Error::Unsupported(m)) if m == message
            ),
            "{s}"
        );
    }
    let oversize = vec!["x".repeat(4097)];
    assert!(matches!(
        parse(&oversize, None),
        Err(Error::Malformed(m)) if m == "bounded nonempty model source required"
    ));
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

#[cfg(all(
    feature = "qwen35-cuda",
    not(feature = "qwen35-cuda-qualification"),
    target_os = "linux"
))]
#[test]
fn ordinary_qwen_cuda_explicit_targets_refuse_before_injected_device_discovery() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let root = std::env::temp_dir().join(format!("zg-explicit-qwen-route-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let config_path = root.join("run.yaml");
    for target in ["qwen3.5-0.8b", "qwen3.5-2b"] {
        std::fs::write(
            &config_path,
            format!(
                "schema: zerogravity.pipeline.v2\nmodel:\n  - id: main\n    target: {target}\n    source: ./source-not-opened\nrun:\n  entrypoint: {{model: main}}\n  backend: cuda\n"
            ),
        )
        .unwrap();
        let parsed = parse(&words(&["--config", config_path.to_str().unwrap()]), None).unwrap();
        let request =
            config::resolve(parsed, Path::new("/"), None, Some("/home/test".into())).unwrap();
        let discoveries = AtomicUsize::new(0);
        let error = execute::run_with_discovery(request, || {
            discoveries.fetch_add(1, Ordering::SeqCst);
            Err(Error::State("injected discovery must not run".into()))
        })
        .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "{target}: {error}");
        assert_eq!(discoveries.load(Ordering::SeqCst), 0, "{target}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(all(
    feature = "qwen35-cuda-serving",
    not(feature = "qwen35-cuda-qualification"),
    target_os = "linux"
))]
#[test]
fn ordinary_cpu_and_gguf_routes_reach_injected_discovery() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    for (source, backend, target) in [
        ("/does-not-exist/extensionless-qwen-gguf", "cuda", None),
        ("/does-not-exist/arbitrary/local-source", "cuda", None),
        ("/does-not-exist/model.GGUF", "cuda", None),
        (
            "/does-not-exist/explicit-qwen-source",
            "cpu",
            Some("qwen3.5-0.8b"),
        ),
        (
            "/does-not-exist/future-qwen-source",
            "cuda",
            Some("qwen3.5-plus"),
        ),
        (
            "/does-not-exist/unknown-qwen-source",
            "cuda",
            Some("qwen3.5-27b"),
        ),
        (
            "/does-not-exist/legacy-prefix-source",
            "cuda",
            Some("qwen35-0.8b"),
        ),
    ] {
        let parsed = parse(&words(&[source, "--backend", backend]), None).unwrap();
        let mut request =
            config::resolve(parsed, Path::new("/"), None, Some("/home/test".into())).unwrap();
        if let Some(target) = target {
            request.graph.models[0].target = target.into();
        }
        let discoveries = AtomicUsize::new(0);
        let error = execute::run_with_discovery(request, || {
            discoveries.fetch_add(1, Ordering::SeqCst);
            Err(Error::State(
                "generic direct-source route reached discovery".into(),
            ))
        })
        .unwrap_err();
        assert!(
            matches!(error, Error::State(message) if message == "generic direct-source route reached discovery"),
            "{source}"
        );
        assert_eq!(discoveries.load(Ordering::SeqCst), 1, "{source}");
    }
}

#[test]
fn qwen_cuda_route_classification_is_purely_declarative() {
    let request = |source: &str, backend: &str| {
        let parsed = parse(&words(&[source, "--backend", backend]), None).unwrap();
        config::resolve(parsed, Path::new("/"), None, Some("/home/test".into())).unwrap()
    };

    let root = std::env::temp_dir().join(format!("zg-route-kind-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let extensionless_gguf = root.join("weights");
    std::fs::write(&extensionless_gguf, b"GGUF").unwrap();
    let arbitrary_directory = root.join("arbitrary-local-source");
    std::fs::create_dir(&arbitrary_directory).unwrap();
    for source in [
        extensionless_gguf.to_str().unwrap(),
        arbitrary_directory.to_str().unwrap(),
        "/does-not-exist/model.GGUF",
    ] {
        let generic = request(source, "cuda");
        assert!(generic.graph.models[0].target.is_empty());
        assert!(
            !qwen35_cuda::selected(&generic.backend, &generic.graph.models[0]),
            "{source}"
        );
    }
    std::fs::remove_dir_all(root).unwrap();

    for target in ["qwen3.5-0.8b", "qwen3.5-2b"] {
        let mut explicit = request("/does-not-exist/source", "cuda");
        explicit.graph.models[0].target = target.into();
        assert!(
            qwen35_cuda::selected(&explicit.backend, &explicit.graph.models[0]),
            "{target}"
        );
        explicit.backend = Backend::Auto;
        assert!(
            !qwen35_cuda::selected(&explicit.backend, &explicit.graph.models[0]),
            "{target}"
        );
        explicit.backend = Backend::Cpu;
        assert!(
            !qwen35_cuda::selected(&explicit.backend, &explicit.graph.models[0]),
            "{target}"
        );
    }

    for target in [
        "qwen3.5-plus",
        "qwen3.5-27b",
        "qwen3.5-unknown",
        "qwen35-0.8b",
    ] {
        let mut unsupported = request("/does-not-exist/source", "cuda");
        unsupported.graph.models[0].target = target.into();
        assert!(
            !qwen35_cuda::selected(&unsupported.backend, &unsupported.graph.models[0]),
            "{target}"
        );
    }

    let mut diagnostic = request("/does-not-exist/source", "auto");
    diagnostic.graph.models[0].native_artifacts = Some("/not-opened/native".into());
    assert!(qwen35_cuda::selected(
        &diagnostic.backend,
        &diagnostic.graph.models[0]
    ));
    diagnostic.backend = Backend::Cpu;
    assert!(!qwen35_cuda::selected(
        &diagnostic.backend,
        &diagnostic.graph.models[0]
    ));
}
