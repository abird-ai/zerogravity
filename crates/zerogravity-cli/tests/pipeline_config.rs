//! Both single-model and multi-model config forms remain testable on
//! the default CPU-only binary without a media runtime or network listener.
use std::path::PathBuf;
use std::process::Command;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/pipelines")
        .join(name)
}
fn output(args: &[&str]) -> std::process::Output {
    binary()
        .args(args)
        .env("HF_TOKEN", "NEVER_ECHO_SECRET")
        .env("GITHUB_TOKEN", "NEVER_ECHO_SECRET")
        .output()
        .unwrap()
}
#[test]
fn four_human_writable_toml_pipelines_are_validated_without_ever_becoming_servers() {
    for (file, expected, route_count) in [
        (
            "voice-assistant.toml",
            "reused model bindings: 1",
            "declared routes: 2",
        ),
        (
            "transcribe-diarize.toml",
            "transcribe-speakers",
            "declared routes: 1",
        ),
        ("text-model.toml", "chat", "declared routes: 1"),
        ("image-generation.toml", "render", "declared routes: 1"),
    ] {
        let file = example(file);
        let path = file.to_str().unwrap();
        let check = output(&["pipeline", "check", "--config", path]);
        assert!(
            check.status.success(),
            "{file:?} {:?}",
            String::from_utf8_lossy(&check.stderr)
        );
        let text = String::from_utf8(check.stdout).unwrap();
        assert!(text.contains("PIPELINE STRUCTURE VALIDATED"));
        assert!(text.contains(expected));
        assert!(text.contains(route_count));
        assert!(text.contains("NOT IMPLEMENTED"));
        assert!(!text.contains("NEVER_ECHO_SECRET"));
        let run = output(&["pipeline", "run", "--config", path]);
        assert_eq!(run.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&run.stderr).contains("NOT IMPLEMENTED"));
    }
}
#[test]
fn equivalent_repeatable_flags_compile_one_typed_voice_chain_without_execution() {
    let flags = [
        "pipeline",
        "check",
        "--name",
        "voice",
        "--input",
        "audio-pcm",
        "--output",
        "audio-pcm",
        "--model",
        "recognizer=nvidia-canary-qwen-2.5b",
        "--model",
        "speakers=speaker-diarization",
        "--model",
        "assistant=qwen3.5-2b",
        "--model",
        "voice=qwen3-tts-1.7b",
        "--step",
        "listen=asr:recognizer",
        "--step",
        "speaker=diarization:speakers",
        "--step",
        "prompt=speakers-text",
        "--step",
        "reply=llm:assistant",
        "--step",
        "speak=tts:voice",
        "--link",
        "input.value=listen.audio",
        "--link",
        "input.value=speaker.audio",
        "--link",
        "listen.transcript=speaker.transcript",
        "--link",
        "speaker.speakers=prompt.speakers",
        "--link",
        "prompt.text=reply.text",
        "--link",
        "reply.text=speak.text",
        "--link",
        "speak.audio=output.value",
        "--route",
        "/v1/realtime=websocket",
        "--realtime",
    ];
    let check = output(&flags);
    assert!(
        check.status.success(),
        "{:?}",
        String::from_utf8_lossy(&check.stderr)
    );
    let body = String::from_utf8_lossy(&check.stdout);
    assert!(body.contains("listen -> speaker -> prompt -> reply -> speak"));
    assert!(body.contains("Model instances: 4"));
    assert!(body.contains("declared routes: 1"));
    let mut run = flags.to_vec();
    run[1] = "run";
    let refused = output(&run);
    assert_eq!(refused.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("NOT IMPLEMENTED"));
}
#[test]
fn bad_input_schemas_port_types_cycles_and_api_exposure_fail_before_any_runtime() {
    let baseline = std::fs::read_to_string(example("text-model.toml")).unwrap();
    let cases = [
        baseline.replace("schema = \"zerogravity.pipeline.v1\"", "schema = \"v2\""),
        baseline.replace("output = \"text\"", "output = \"audio-pcm\""),
        baseline.replace("to = \"generate.text\"", "to = \"generate.missing\""),
        baseline.replace("to = \"output.value\"", "to = \"generate.text\""),
        baseline.replace("id = \"generate\"", "id = \"input\""),
        baseline.replace("protocol = \"openai-text\"", "protocol = \"webrtc\""),
        baseline.replace("model = \"model\"", "model = \"missing\""),
        baseline.replace(
            "target = \"qwen3.5-2b\"",
            "target = \"unregistered-future-model\"",
        ),
        baseline.replace("kind = \"llm\"", "kind = \"tts\""),
        format!("{baseline}\nunknown_command = \"run-anything\"\n"),
    ];
    let base = std::env::temp_dir().join(format!("zg-pipeline-negative-{}", std::process::id()));
    std::fs::create_dir(&base).unwrap();
    for (i, contents) in cases.into_iter().enumerate() {
        let path = base.join(format!("{i}.toml"));
        std::fs::write(&path, contents).unwrap();
        let check = output(&["pipeline", "check", "--config", path.to_str().unwrap()]);
        assert_eq!(
            check.status.code(),
            Some(2),
            "bad case {i}: {:?}",
            String::from_utf8_lossy(&check.stderr)
        );
        assert!(!String::from_utf8_lossy(&check.stderr).contains("NEVER_ECHO_SECRET"));
    }
    std::fs::remove_dir_all(base).unwrap();
    for extra in [
        vec![
            "pipeline",
            "check",
            "--config",
            "foo",
            "--model",
            "id=qwen3.5-2b",
        ],
        vec!["pipeline", "check", "--name", "foo", "--input", "audio-pcm"],
        vec![
            "pipeline",
            "check",
            "--name",
            "foo",
            "--input",
            "text",
            "--output",
            "text",
            "--step",
            "x=arbitrary-lua",
            "--link",
            "input.value=output.value",
        ],
        vec!["pipeline", "serve", "--config", "foo"],
    ] {
        assert_eq!(output(&extra).status.code(), Some(2), "{extra:?}");
    }
}

#[test]
fn committed_same_session_chat_hints_are_opt_in_and_toggle_without_graph_mutation() {
    let file = example("contextual-voice-assistant.toml");
    let contents = std::fs::read_to_string(&file).unwrap();
    let result = output(&["pipeline", "check", "--config", file.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "{:?}",
        String::from_utf8_lossy(&result.stderr)
    );
    let checked = String::from_utf8(result.stdout).unwrap();
    assert!(checked.contains("listen -> speaker -> translate -> prompt -> reply -> speak"));
    assert!(checked.contains("reused model bindings: 2"));
    assert!(checked.contains("Committed chat context: 1 configured, 1 ENABLED"));
    assert!(checked.contains("context-aware ASR/translation"));
    assert!(checked.contains("NOT IMPLEMENTED"));
    let tmp = std::env::temp_dir().join(format!("zg-context-toggle-{}.toml", std::process::id()));
    assert!(!tmp.exists());
    // Same TOML and typed context links remain; disabled policy emits no
    // snapshot, so no previous-turn context reaches ASR/translation.
    let disabled = contents.replace("enabled = true", "enabled = false");
    assert_ne!(disabled, contents);
    std::fs::write(&tmp, disabled).unwrap();
    let disabled_result = output(&["pipeline", "check", "--config", tmp.to_str().unwrap()]);
    std::fs::remove_file(tmp).unwrap();
    assert!(
        disabled_result.status.success(),
        "{:?}",
        String::from_utf8_lossy(&disabled_result.stderr)
    );
    let checked = String::from_utf8(disabled_result.stdout).unwrap();
    assert!(checked.contains("Committed chat context: 1 configured, 0 ENABLED"));
    assert!(checked.contains("listen -> speaker -> translate -> prompt -> reply -> speak"));
    let run = output(&["pipeline", "run", "--config", file.to_str().unwrap()]);
    assert_eq!(run.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&run.stderr).contains("NOT IMPLEMENTED"));
}

#[test]
fn repeatable_context_flags_compile_identically_and_can_disable_hints() {
    let args = [
        "pipeline",
        "check",
        "--name",
        "contextual-stt",
        "--input",
        "audio-pcm",
        "--output",
        "text",
        "--model",
        "recognizer=nvidia-canary-qwen-2.5b",
        "--model",
        "assistant=qwen3.5-2b",
        "--step",
        "listen=asr:recognizer",
        "--step",
        "format=transcript-text",
        "--step",
        "reply=llm:assistant",
        "--link",
        "input.value=listen.audio",
        "--link",
        "listen.transcript=format.transcript",
        "--link",
        "format.text=reply.text",
        "--link",
        "reply.text=output.value",
        "--chat-context-from",
        "assistant",
        "--chat-context-max-bytes",
        "4096",
        "--link",
        "context.value=listen.context",
        "--link",
        "context.value=format.context",
    ];
    let check = output(&args);
    assert!(
        check.status.success(),
        "{:?}",
        String::from_utf8_lossy(&check.stderr)
    );
    let text = String::from_utf8(check.stdout).unwrap();
    assert!(text.contains("listen -> format -> reply"));
    assert!(text.contains("Committed chat context: 1 configured, 1 ENABLED"));
    let mut disabled = args.to_vec();
    disabled.extend(["--chat-context-enabled", "false"]);
    let disabled_result = output(&disabled);
    assert!(
        disabled_result.status.success(),
        "{:?}",
        String::from_utf8_lossy(&disabled_result.stderr)
    );
    assert!(String::from_utf8_lossy(&disabled_result.stdout).contains("1 configured, 0 ENABLED"));
    let mut run = args.to_vec();
    run[1] = "run";
    let output = output(&run);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("NOT IMPLEMENTED"));
}

#[test]
fn bad_context_sources_bounds_spoofed_cycles_and_unknown_config_refuse() {
    let original = std::fs::read_to_string(example("contextual-voice-assistant.toml")).unwrap();
    let bad = [
        original.replace("max_bytes = 4096", "max_bytes = 0"),
        original.replace("max_bytes = 4096", "max_bytes = 1048577"),
        original.replace("from_model = \"assistant\"", "from_model = \"voice\""),
        original.replace("from_model = \"assistant\"", "from_model = \"missing\""),
        original.replace(
            "chat_context = { enabled = true, from_model = \"assistant\", max_bytes = 4096 }",
            "",
        ),
        original.replace("from = \"context.value\"", "from = \"reply.text\""),
        original.replace("to = \"listen.context\"", "to = \"listen.audio\""),
        original.replace("id = \"listen\"", "id = \"context\""),
        original.replace("max_bytes = 4096", "max_bytes = -1"),
        original.replace(
            "max_bytes = 4096",
            "max_bytes = 4096, allow_cross_session = true",
        ),
        original.replace("enabled = true", "enabled = \"yes\""),
    ];
    let temp =
        std::env::temp_dir().join(format!("zg-chat-context-negative-{}", std::process::id()));
    std::fs::create_dir(&temp).unwrap();
    for (i, bad) in bad.into_iter().enumerate() {
        assert_ne!(bad, original, "negative fixture did not change");
        let file = temp.join(format!("{i}.toml"));
        std::fs::write(&file, bad).unwrap();
        let result = output(&["pipeline", "check", "--config", file.to_str().unwrap()]);
        assert_eq!(
            result.status.code(),
            Some(2),
            "negative {i}: {:?}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!String::from_utf8_lossy(&result.stderr).contains("NEVER_ECHO_SECRET"));
    }
    std::fs::remove_dir_all(temp).unwrap();
    for flags in [
        vec!["--chat-context-enabled", "true"],
        vec!["--chat-context-max-bytes", "1024"],
        vec!["--chat-context-enabled", "maybe"],
        vec![
            "--chat-context-from",
            "assistant",
            "--chat-context-max-bytes",
            "1048577",
        ],
    ] {
        let mut args = vec![
            "pipeline",
            "check",
            "--name",
            "chat",
            "--input",
            "text",
            "--output",
            "text",
            "--model",
            "assistant=qwen3.5-2b",
            "--step",
            "reply=llm:assistant",
            "--link",
            "input.value=reply.text",
            "--link",
            "reply.text=output.value",
        ];
        args.extend(flags);
        let result = output(&args);
        assert_eq!(result.status.code(), Some(2), "{args:?}");
    }
}

#[test]
fn ordered_context_filters_and_nanojev_decision_compile_from_human_toml_without_loading_models() {
    let file = example("contextual-voice-filtered.toml");
    let result = output(&["pipeline", "check", "--config", file.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "{:?}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = String::from_utf8(result.stdout).unwrap();
    assert!(text.contains("Model instances: 5"));
    assert!(text.contains("reused model bindings: 2"));
    assert!(text.contains("1 configured, 1 ENABLED"));
    assert!(text.contains("Context filters: 3 ordered stages, 1 typed decision stages"));
    assert!(text.contains("history/backend access NOT IMPLEMENTED"));
    let execution = output(&["pipeline", "run", "--config", file.to_str().unwrap()]);
    assert_eq!(execution.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&execution.stderr).contains("NOT IMPLEMENTED"));
}

#[test]
fn context_sources_match_filters_and_decision_accepted_as_repeatable_flags() {
    let args = [
        "pipeline",
        "check",
        "--name",
        "chat-context",
        "--input",
        "text",
        "--output",
        "text",
        "--model",
        "assistant=qwen3.5-2b",
        "--model",
        "gate=nanojev-decision",
        "--step",
        "reply=llm:assistant",
        "--link",
        "input.value=reply.text",
        "--link",
        "context.value=reply.context",
        "--link",
        "reply.text=output.value",
        "--chat-context-from",
        "assistant",
        "--chat-context-source",
        "session/user_message",
        "--chat-context-source",
        "session/tool_call",
        "--chat-context-source",
        "session/tool_result",
        "--chat-context-source",
        "application/system_prompt",
        "--chat-context-source",
        "shared:team-support/assistant_message",
        "--chat-context-match",
        "exclude:kind:tool_call",
        "--chat-context-match",
        "exclude:kind:tool_result",
        "--chat-context-decision",
        "gate:8500:omit:Is this useful for the next utterance?",
    ];
    // The LLM's existing text port has no context input: use an ASR stage
    // for the correctly typed policy instead; the malformed destination
    // must refuse rather than smuggle session context into arbitrary ports.
    let bad = output(&args);
    assert_eq!(bad.status.code(), Some(2));
    let mut correct = args.to_vec();
    let index = correct
        .iter()
        .position(|value| *value == "context.value=reply.context")
        .unwrap();
    correct.remove(index);
    correct.remove(index - 1);
    correct.extend(["--model", "recognizer=nvidia-canary-qwen-2.5b"]);
    // Replace all pipeline stages with ASR -> text -> LLM and add one typed
    // context source edge, preserving the same selector/filter/decision flags.
    let mut fixed = Vec::new();
    let mut cursor = 0;
    while cursor < correct.len() {
        if correct[cursor] == "--input" {
            fixed.extend(["--input", "audio-pcm"]);
            cursor += 2;
        } else if correct[cursor] == "--step"
            && correct.get(cursor + 1) == Some(&"reply=llm:assistant")
        {
            fixed.extend([
                "--step",
                "hear=asr:recognizer",
                "--step",
                "to-text=transcript-text",
                "--step",
                "reply=llm:assistant",
            ]);
            cursor += 2;
        } else if correct[cursor] == "--link"
            && correct.get(cursor + 1) == Some(&"input.value=reply.text")
        {
            fixed.extend([
                "--link",
                "input.value=hear.audio",
                "--link",
                "hear.transcript=to-text.transcript",
                "--link",
                "to-text.text=reply.text",
                "--link",
                "context.value=hear.context",
            ]);
            cursor += 2;
        } else {
            fixed.push(correct[cursor]);
            cursor += 1;
        }
    }
    let result = output(&fixed);
    assert!(
        result.status.success(),
        "{:?}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = String::from_utf8(result.stdout).unwrap();
    assert!(text.contains("Context filters: 3 ordered stages, 1 typed decision stages"));
    let mut disabled = fixed.clone();
    disabled.extend(["--chat-context-enabled", "false"]);
    let result = output(&disabled);
    assert!(
        result.status.success(),
        "{:?}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("1 configured, 0 ENABLED"));
}

#[test]
fn decision_source_types_duplicates_malformed_filters_and_unqualified_targets_refuse() {
    let base = std::fs::read_to_string(example("contextual-voice-filtered.toml")).unwrap();
    let cases = [
        base.replace(r#"target = "nanojev-decision""#, r#"target = "qwen3.5-2b""#),
        base.replace(r#"model = "gate""#, r#"model = "missing""#),
        base.replace("sources = [", r#"sources = ["session/tool_result","#),
        base.replace(
            "shared:team-support/assistant_message",
            "shared:../assistant_message",
        ),
        base.replace("min_confidence_bps = 7500", "min_confidence_bps = 10001"),
        base.replace(
            r#"on_unavailable = "omit""#,
            r#"on_unavailable = "elevate""#,
        ),
        base.replace(r#"field = "kind""#, r#"field = "regex_script""#),
        base.replace(r#"values = ["tool_call"]"#, r#"values = ["bad_role"]"#),
        base.replace(r#"kind = "decision""#, r#"kind = "arbitrary_code""#),
        base.replace("sources = [", "sources = [] # invalid sources,"),
    ];
    let temp = std::env::temp_dir().join(format!("zg-context-filters-{}", std::process::id()));
    assert!(!temp.exists());
    std::fs::create_dir(&temp).unwrap();
    for (i, config) in cases.iter().enumerate() {
        assert_ne!(*config, base);
        let path = temp.join(format!("{i}.toml"));
        std::fs::write(&path, config).unwrap();
        let output = output(&["pipeline", "check", "--config", path.to_str().unwrap()]);
        assert_eq!(
            output.status.code(),
            Some(2),
            "case {i} {:?}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!String::from_utf8_lossy(&output.stderr).contains("NEVER_ECHO_SECRET"));
    }
    std::fs::remove_dir_all(temp).unwrap();
    for invalid in [
        ["--chat-context-source", "session/tool_result"].as_slice(),
        ["--chat-context-match", "exclude:kind:tool_call"].as_slice(),
        ["--chat-context-decision", "gate:7500:omit:Question"].as_slice(),
    ] {
        let mut args = vec![
            "pipeline",
            "check",
            "--name",
            "chat",
            "--input",
            "text",
            "--output",
            "text",
            "--model",
            "assistant=qwen3.5-2b",
            "--step",
            "reply=llm:assistant",
            "--link",
            "input.value=reply.text",
            "--link",
            "reply.text=output.value",
        ];
        args.extend_from_slice(invalid);
        assert_eq!(output(&args).status.code(), Some(2));
    }
}
