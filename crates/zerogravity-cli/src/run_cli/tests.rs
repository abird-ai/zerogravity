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
fn ordinary_qwen_cuda_explicit_target_refuses_before_injected_device_discovery() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let root = std::env::temp_dir().join(format!("zg-explicit-qwen-route-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let config_path = root.join("run.yaml");
    std::fs::write(
        &config_path,
        "schema: zerogravity.pipeline.v2\nmodel:\n  - id: main\n    target: qwen3.5-0.8b\n    source: ./source-not-opened\nrun:\n  entrypoint: {model: main}\n  backend: cuda\n",
    )
    .unwrap();
    let parsed = parse(&words(&["--config", config_path.to_str().unwrap()]), None).unwrap();
    let request = config::resolve(parsed, Path::new("/"), None, Some("/home/test".into())).unwrap();
    std::fs::remove_dir_all(root).unwrap();
    let discoveries = AtomicUsize::new(0);
    let error = execute::run_with_discovery(request, || {
        discoveries.fetch_add(1, Ordering::SeqCst);
        Err(Error::State("injected discovery must not run".into()))
    })
    .unwrap_err();
    assert!(
        matches!(error, Error::Unsupported(message) if message.contains("no ordinary dynamic GDN implementation"))
    );
    assert_eq!(discoveries.load(Ordering::SeqCst), 0);
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

    let mut explicit = request("/does-not-exist/source", "cuda");
    explicit.graph.models[0].target = "qwen3.5-0.8b".into();
    assert!(qwen35_cuda::selected(
        &explicit.backend,
        &explicit.graph.models[0]
    ));
    explicit.backend = Backend::Auto;
    assert!(!qwen35_cuda::selected(
        &explicit.backend,
        &explicit.graph.models[0]
    ));
    explicit.backend = Backend::Cpu;
    assert!(!qwen35_cuda::selected(
        &explicit.backend,
        &explicit.graph.models[0]
    ));

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

/// Pure finite-route proposal and pre-Client seam unit coverage; no CUDA
/// client or real source is touched here.
#[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
mod qwen35_cuda_route_tests {
    use super::*;
    use zerogravity::resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    };

    const MIB: u64 = 1024 * 1024;
    const HOST_REFUSAL: &str =
        "finite HF original HOST8192MiB plus current routing headroom does not fit";
    const NO_CARD_REFUSAL: &str = "no allowed source-qualified sm86 CUDA card fits original provider+512MiB backend and current free-memory headroom; no split/backend/source fallback";

    fn route_request(args: &[&str]) -> Request {
        let a = parse(&words(args), None).unwrap();
        config::resolve(a, Path::new("/"), None, Some("/home/test".into())).unwrap()
    }

    fn gpu(
        id: &str,
        vendor: &str,
        cc: Option<(i32, i32)>,
        ordinal: Option<usize>,
        ceiling: Option<u64>,
        available: Option<u64>,
        exclusion: Option<&str>,
    ) -> devices::Observation {
        devices::Observation {
            id: id.into(),
            class: "gpu".into(),
            vendor: vendor.into(),
            ceiling,
            available,
            backends: vec![],
            exclusion: exclusion.map(str::to_string),
            provenance: "pure metadata fixture".into(),
            cuda_ordinal: ordinal,
            cuda_cc: cc,
        }
    }

    fn host(ceiling: Option<u64>, available: Option<u64>) -> devices::Observation {
        devices::Observation {
            id: "cpu:0".into(),
            class: "host-memory".into(),
            vendor: "cpu".into(),
            ceiling,
            available,
            backends: vec![],
            exclusion: None,
            provenance: "pure metadata fixture".into(),
            cuda_ordinal: None,
            cuda_cc: None,
        }
    }

    fn inventory(devices: Vec<devices::Observation>) -> devices::Inventory {
        devices::Inventory {
            devices,
            observed_unix_ms: 0,
        }
    }

    fn profile(host_bytes: usize, provider_bytes: usize) -> qwen35_cuda::FiniteRouteProfile {
        qwen35_cuda::FiniteRouteProfile {
            host_bytes,
            provider_bytes,
        }
    }

    fn refusal_message(error: Error) -> String {
        match error {
            Error::Limit(m) => m,
            other => panic!("expected limit refusal: {other:?}"),
        }
    }

    #[test]
    fn native_artifacts_marker_keeps_qualified_auto_route() {
        let request = route_request(&[
            "/source-not-opened/model",
            "--native-artifacts",
            "/native-not-opened",
        ]);
        assert_eq!(request.backend, Backend::Auto);
        assert!(request.graph.models[0].target.is_empty());
        assert!(qwen35_cuda::selected(
            &request.backend,
            &request.graph.models[0]
        ));
    }

    #[test]
    fn sm86_proposal_keeps_ordered_exclusions_for_other_devices() {
        let request = route_request(&["m.gguf"]);
        let profile = profile(8192 * 1024 * 1024, 512 * 1024 * 1024);
        let amd = gpu(
            "pci:0000:01:00.0",
            "amd",
            None,
            None,
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            Some(
                "source-qualified cold/compiler cost and runtime physical binding unavailable in generic run",
            ),
        );
        let fit = gpu(
            "pci:0000:02:00.0",
            "nvidia",
            Some((8, 6)),
            Some(0),
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            None,
        );
        let inv = inventory(vec![
            amd,
            fit.clone(),
            host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB)),
        ]);
        let (proposed, exclusions) =
            qwen35_cuda::select_sm86_card(&request, &profile, &inv, &ResourceArbiter::new())
                .unwrap();
        assert_eq!(proposed.expect("fit card").id, fit.id);
        assert_eq!(
            exclusions,
            vec![
                "pci:0000:01:00.0: source-qualified cold/compiler cost and runtime physical binding unavailable in generic run"
            ]
        );
    }

    #[test]
    fn sm86_card_exclusion_reasons_mirror_every_skip_condition() {
        // Explicit headroom makes the free-memory exclusion deterministic.
        let request = route_request(&["m.gguf", "--headroom", "3GiB"]);
        let profile = profile(8192 * 1024 * 1024, 4 * 1024 * 1024 * 1024usize);
        let inv = inventory(vec![
            gpu(
                "pci:0000:03:00.0",
                "nvidia",
                Some((8, 6)),
                Some(0),
                Some(6 * 1024 * MIB),
                None,
                Some("NVML current free/capacity binding unavailable"),
            ),
            gpu(
                "pci:0000:04:00.0",
                "nvidia",
                Some((8, 6)),
                Some(1),
                None,
                None,
                Some("multiple CUDA views share one PCI identity (MIG/partition ambiguous)"),
            ),
            gpu(
                "pci:0000:05:00.0",
                "nvidia",
                Some((8, 6)),
                Some(2),
                Some(4 * 1024 * MIB),
                Some(4 * 1024 * MIB),
                None,
            ),
            gpu(
                "pci:0000:06:00.0",
                "nvidia",
                Some((8, 6)),
                Some(3),
                Some(6 * 1024 * MIB),
                Some(6 * 1024 * MIB),
                None,
            ),
            gpu(
                "pci:0000:07:00.0",
                "nvidia",
                Some((8, 9)),
                Some(4),
                Some(6 * 1024 * MIB),
                Some(6 * 1024 * MIB),
                None,
            ),
            gpu("pci:0000:08:00.0", "amd", None, None, None, None, None),
            host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB)),
        ]);
        let (proposed, exclusions) =
            qwen35_cuda::select_sm86_card(&request, &profile, &inv, &ResourceArbiter::new())
                .unwrap();
        assert!(proposed.is_none(), "no card passes every condition");
        assert_eq!(
            exclusions,
            vec![
                "pci:0000:03:00.0: NVML current free/capacity binding unavailable",
                "pci:0000:04:00.0: multiple CUDA views share one PCI identity (MIG/partition ambiguous)",
                "pci:0000:05:00.0: card capacity below the 5120MiB operator cap",
                "pci:0000:06:00.0: insufficient current free memory after routing headroom",
                "pci:0000:07:00.0: finite Qwen CUDA route requires qualified sm86 NVIDIA CUDA card",
                "pci:0000:08:00.0: finite Qwen CUDA route requires qualified sm86 NVIDIA CUDA card",
            ]
        );
        assert_eq!(
            refusal_message(qwen35_cuda::no_route_refusal(&profile, &exclusions)),
            NO_CARD_REFUSAL
        );
    }

    #[test]
    fn sm86_ledger_hold_fence_and_cap_conflict_exclude_the_card_exactly() {
        let request = route_request(&["m.gguf"]);
        let profile = profile(8192 * 1024 * 1024, 512 * 1024 * 1024);
        let fit = |id: &str| {
            gpu(
                id,
                "nvidia",
                Some((8, 6)),
                Some(0),
                Some(6 * 1024 * MIB),
                Some(6 * 1024 * MIB),
                None,
            )
        };
        let reserved = |arbiter: &ResourceArbiter,
                        key: &PhysicalDeviceKey|
         -> zerogravity::resource_arbiter::GlobalResourceLease {
            arbiter.register_matching(key.clone(), 5120 * MIB).unwrap();
            arbiter
                .admit(
                    key,
                    ResourceOwner::new("other-route", "session").unwrap(),
                    &[AllocationRequest {
                        kind: AllocationClass::Scratch,
                        bytes: 128 * MIB,
                    }],
                )
                .unwrap()
        };
        for (arbiter, name) in [
            (ResourceArbiter::new(), "reserved"),
            (ResourceArbiter::new(), "fenced"),
            (ResourceArbiter::new(), "cap"),
        ] {
            let key =
                PhysicalDeviceKey::new("pci", "0000:09:00.0").expect("synthetic PCI identity");
            // The reserved case keeps its lease alive across the scan.
            let held = match name {
                "reserved" => Some(reserved(&arbiter, &key)),
                "fenced" => {
                    arbiter.register_matching(key.clone(), 5120 * MIB).unwrap();
                    arbiter.fence_unknown_device(&key).unwrap();
                    None
                }
                _ => {
                    arbiter.register(key.clone(), 4096 * MIB).unwrap();
                    None
                }
            };
            let inv = inventory(vec![
                fit("pci:0000:09:00.0"),
                host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB)),
            ]);
            let (proposed, exclusions) =
                qwen35_cuda::select_sm86_card(&request, &profile, &inv, &arbiter).unwrap();
            assert!(proposed.is_none(), "{name}");
            assert_eq!(
                exclusions,
                vec!["pci:0000:09:00.0: card already serves an in-process route reservation"],
                "{name}"
            );
            drop(held);
        }
    }

    #[test]
    fn sm86_host_unfit_suppresses_proposals_and_refuses_with_host_limit() {
        let request = route_request(&["m.gguf"]);
        let profile = profile(8192 * 1024 * 1024, 512 * 1024 * 1024);
        let fit = gpu(
            "pci:0000:0a:00.0",
            "nvidia",
            Some((8, 6)),
            Some(0),
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            None,
        );
        let amd = gpu(
            "pci:0000:0b:00.0",
            "amd",
            None,
            None,
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            None,
        );
        let unfit_host = host(Some(64 * 1024 * MIB), Some(1));
        let inv = inventory(vec![amd, fit, unfit_host]);
        let (proposed, exclusions) =
            qwen35_cuda::select_sm86_card(&request, &profile, &inv, &ResourceArbiter::new())
                .unwrap();
        assert!(proposed.is_none(), "host unfitness suppresses proposals");
        assert_eq!(
            exclusions.first().map(String::as_str),
            Some("host: finite HF original HOST8192MiB plus current routing headroom does not fit")
        );
        assert!(exclusions.len() >= 2, "card exclusions share the trace");
        assert_eq!(
            refusal_message(qwen35_cuda::no_route_refusal(&profile, &exclusions)),
            HOST_REFUSAL
        );
    }

    #[test]
    fn sm86_observation_gaps_stay_hard_errors_and_empty_traces_refuse_exactly() {
        let request = route_request(&["m.gguf"]);
        let profile = profile(8192 * 1024 * 1024, 512 * 1024 * 1024);
        let arbiter = ResourceArbiter::new();
        let fit = gpu(
            "pci:0000:0c:00.0",
            "nvidia",
            Some((8, 6)),
            Some(0),
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            None,
        );
        // No HOST observation is a hard state error, never a silent exclusion.
        let inv = inventory(vec![fit]);
        assert!(qwen35_cuda::select_sm86_card(&request, &profile, &inv, &arbiter).is_err());
        // An unknown HOST ceiling is a hard state error.
        let inv = inventory(vec![
            gpu(
                "pci:0000:0c:00.0",
                "nvidia",
                Some((8, 6)),
                Some(0),
                Some(6 * 1024 * MIB),
                Some(6 * 1024 * MIB),
                None,
            ),
            host(None, None),
        ]);
        assert!(qwen35_cuda::select_sm86_card(&request, &profile, &inv, &arbiter).is_err());
        // An unavailable explicit device is a policy error, not an exclusion.
        let request = route_request(&["m.gguf", "--device", "gpu:5"]);
        let inv = inventory(vec![
            gpu(
                "pci:0000:0c:00.0",
                "nvidia",
                Some((8, 6)),
                Some(0),
                Some(6 * 1024 * MIB),
                Some(6 * 1024 * MIB),
                None,
            ),
            host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB)),
        ]);
        assert!(qwen35_cuda::select_sm86_card(&request, &profile, &inv, &arbiter).is_err());
        // A fitting host with no GPUs refuses with the exact no-card limit.
        let request = route_request(&["m.gguf"]);
        let inv = inventory(vec![host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB))]);
        let (proposed, exclusions) =
            qwen35_cuda::select_sm86_card(&request, &profile, &inv, &arbiter).unwrap();
        assert!(proposed.is_none());
        assert!(exclusions.is_empty());
        assert_eq!(
            refusal_message(qwen35_cuda::no_route_refusal(&profile, &exclusions)),
            NO_CARD_REFUSAL
        );
    }

    #[test]
    fn decision_trace_fields_split_dry_run_from_executing_admission() {
        let mut dry = serde_json::json!({});
        qwen35_cuda::dry_run_trace_status(&mut dry);
        qwen35_cuda::dry_run_admission_trace(&mut dry);
        assert_eq!(dry["dry_run"], serde_json::json!(true));
        assert_eq!(dry["reservation"], serde_json::json!("none"));
        assert_eq!(dry["pre_client_admission_gap"], serde_json::json!(true));
        assert!(dry.get("pre_client_admission").is_none());
        assert!(dry.get("advisory_boundary").is_none());

        let mut executing = serde_json::json!({});
        qwen35_cuda::executing_admission_trace(&mut executing);
        assert_eq!(
            executing["pre_client_admission"],
            serde_json::json!(
                "transactional card hold before Client; released at verified identity"
            )
        );
        assert_eq!(
            executing["advisory_boundary"],
            serde_json::json!(
                "other processes and driver allocations; post-release pre-backend window"
            )
        );
        assert!(executing.get("pre_client_admission_gap").is_none());
        assert!(executing.get("dry_run").is_none());
        assert!(executing.get("reservation").is_none());
    }

    #[test]
    fn pre_client_card_admit_races_to_exactly_one_holder_then_retries() {
        let arbiter = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("pci", "0000:0d:00.0").unwrap();
        arbiter.register_matching(key.clone(), 5120 * MIB).unwrap();
        let fresh = arbiter.snapshot(&key).unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let (first, second) = std::thread::scope(|scope| {
            let a = scope.spawn({
                let arbiter = &arbiter;
                let key = &key;
                let barrier = barrier.clone();
                move || {
                    barrier.wait();
                    qwen35_cuda::admit_pre_client_card(arbiter, key)
                }
            });
            let b = scope.spawn({
                let arbiter = &arbiter;
                let key = &key;
                move || {
                    barrier.wait();
                    qwen35_cuda::admit_pre_client_card(arbiter, key)
                }
            });
            (a.join().unwrap(), b.join().unwrap())
        });
        let (winner, loser) = match (first, second) {
            (Ok(lease), Err(error)) => (lease, error),
            (Err(error), Ok(lease)) => (lease, error),
            (Ok(_), Ok(_)) => panic!("two pre-Client holds admitted one card"),
            (Err(_), Err(_)) => panic!("no pre-Client hold admitted the free card"),
        };
        assert!(
            matches!(&loser, Error::Limit(m) if m == "physical device memory admission refused"),
            "{loser}"
        );
        let held = arbiter.snapshot(&key).unwrap();
        assert_eq!(held.reserved, 5120 * MIB);
        assert_eq!(held.reservations, 1);
        assert_eq!(
            held.per_class.get(&AllocationClass::ProviderEnvelope),
            Some(&(5120 * MIB))
        );
        assert_eq!(
            held.active_models
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["qwen35-cuda"]
        );
        // The immutable cap stays one shared budget while the hold is alive.
        assert!(arbiter.register(key.clone(), 1).is_err());
        drop(winner);
        // After the hold is released the loser retries successfully and the
        // budget returns to the exact pre-admission snapshot.
        let retried = qwen35_cuda::admit_pre_client_card(&arbiter, &key).unwrap();
        drop(retried);
        let released = arbiter.snapshot(&key).unwrap();
        assert_eq!(released.reserved, fresh.reserved);
        assert_eq!(released.remaining, fresh.remaining);
        assert_eq!(released.reservations, fresh.reservations);
        assert_eq!(released.per_class, fresh.per_class);
        assert_eq!(released.active_models, fresh.active_models);
        assert_eq!(released.fenced, fresh.fenced);
    }

    #[test]
    fn pre_client_card_admit_refuses_fenced_keys_and_conflicting_caps_exactly() {
        let key = PhysicalDeviceKey::new("pci", "0000:0e:00.0").unwrap();
        let fenced = ResourceArbiter::new();
        fenced.register_matching(key.clone(), 5120 * MIB).unwrap();
        fenced.fence_unknown_device(&key).unwrap();
        assert!(matches!(
            &qwen35_cuda::admit_pre_client_card(&fenced, &key),
            Err(Error::State(m)) if m == "physical device completion uncertain: fenced"
        ));
        let conflicting = ResourceArbiter::new();
        conflicting.register(key.clone(), 4096 * MIB).unwrap();
        assert!(matches!(
            &qwen35_cuda::admit_pre_client_card(&conflicting, &key),
            Err(Error::State(m)) if m == "conflicting physical device budget attestation"
        ));
        // Idempotent same-cap registration leaves exactly one budget.
        let idempotent = ResourceArbiter::new();
        idempotent
            .register_matching(key.clone(), 5120 * MIB)
            .unwrap();
        idempotent
            .register_matching(key.clone(), 5120 * MIB)
            .unwrap();
        assert_eq!(idempotent.snapshot(&key).unwrap().limit, 5120 * MIB);
        assert!(idempotent.register(key.clone(), 5120 * MIB).is_err());
        let hold = qwen35_cuda::admit_pre_client_card(&idempotent, &key).unwrap();
        drop(hold);
        assert_eq!(idempotent.snapshot(&key).unwrap().reserved, 0);
    }

    fn absolute_env(name: &str) -> Result<PathBuf> {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .ok_or_else(|| bad(format!("required absolute {name}")))
    }

    #[test]
    #[ignore = "explicit physical Qwen CUDA dry-run; approved sm86 hardware and real source/native bundle, separate fresh owned process"]
    fn physical_qwen35_cuda_dry_run_is_read_only_and_registers_nothing() -> Result<()> {
        let source = absolute_env("ZG_QWEN35_MODEL_DIR")?;
        let native = absolute_env("ZG_QWEN35_CUDA_KERNEL_DIR")?;
        let source = source.to_str().ok_or_else(|| bad("UTF-8 source path"))?;
        let native = native.to_str().ok_or_else(|| bad("UTF-8 native path"))?;
        let request = config::resolve(
            parse(
                &words(&[
                    source,
                    "--native-artifacts",
                    native,
                    "--dry-run",
                    "--prompt",
                    "a",
                ]),
                None,
            )?,
            Path::new("/"),
            None,
            Some("/home/test".into()),
        )?;
        let arbiter = ResourceArbiter::process_shared();
        let keys = devices::discover()?
            .devices
            .iter()
            .filter(|d| d.id.starts_with("pci:"))
            .filter_map(|d| {
                d.id.strip_prefix("pci:")
                    .and_then(|pci| PhysicalDeviceKey::new("pci", pci).ok())
            })
            .collect::<Vec<_>>();
        assert!(!keys.is_empty(), "physical CUDA card observation");
        // A second dry-run in the same process observes the same ledger: a
        // dry-run may not register a card budget or admit anything.
        let before = keys.iter().map(|k| arbiter.snapshot(k)).collect::<Vec<_>>();
        execute::run(request.clone())?;
        execute::run(request)?;
        let after = keys.iter().map(|k| arbiter.snapshot(k)).collect::<Vec<_>>();
        for (before, after) in before.iter().zip(&after) {
            match (before, after) {
                (Err(_), Err(_)) => {}
                (Ok(before), Ok(after)) => {
                    assert_eq!(after.reserved, before.reserved);
                    assert_eq!(after.reservations, before.reservations);
                    assert_eq!(after.per_class, before.per_class);
                    assert_eq!(after.active_models, before.active_models);
                    assert_eq!(after.fenced, before.fenced);
                }
                _ => panic!("dry-run changed the physical-card ledger registration"),
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "explicit physical pre-Client seam; approved sm86 hardware, separate fresh owned process"]
    fn physical_pre_client_card_seam_holds_the_whole_card_budget() -> Result<()> {
        let inventory = devices::discover()?;
        devices::fresh(&inventory)?;
        let card = inventory
            .devices
            .iter()
            .find(|d| {
                d.class == "gpu"
                    && d.vendor == "nvidia"
                    && d.cuda_cc == Some((8, 6))
                    && d.cuda_ordinal.is_some()
                    && d.exclusion.is_none()
            })
            .ok_or_else(|| bad("no allowed source-qualified sm86 CUDA card"))?;
        let key = PhysicalDeviceKey::new(
            "pci",
            card.id
                .strip_prefix("pci:")
                .ok_or_else(|| bad("PCI identity"))?,
        )?;
        let arbiter = ResourceArbiter::process_shared();
        assert!(arbiter.snapshot(&key).is_err());
        let hold = qwen35_cuda::admit_pre_client_card(&arbiter, &key)?;
        // A second route refuses with the arbiter's exact limit while the
        // first route holds the whole finite card budget.
        assert!(matches!(
            &qwen35_cuda::admit_pre_client_card(&arbiter, &key),
            Err(Error::Limit(m)) if m == "physical device memory admission refused"
        ));
        let held = arbiter.snapshot(&key)?;
        assert_eq!(held.reserved, 5120 * MIB);
        assert_eq!(held.reservations, 1);
        let ordinal = card.cuda_ordinal.ok_or_else(|| bad("CUDA ordinal"))?;
        let device = zerogravity::gpu::Device::cuda(ordinal)
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        let actual = zerogravity::gpu::initialize_cuda_physical_identity(&device)?;
        assert_eq!(actual, key);
        drop(hold);
        let released = arbiter.snapshot(&key)?;
        assert_eq!(released.reserved, 0);
        assert_eq!(released.reservations, 0);
        assert!(released.per_class.is_empty());
        assert!(released.active_models.is_empty());
        Ok(())
    }
}
