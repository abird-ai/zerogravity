//! Pipeline argv and presentation; YAML/schema/checking live in the public SDK.
use zerogravity::context_filter::{
    ContextFilter, ContextSource, DecisionUnavailable, FilterAction, MatchField,
};
pub(super) use zerogravity::pipelines::load_document;
use zerogravity::pipelines::{
    self, ApiRoute, ChatContextPatch, ChatContextPolicy, Config, ModelInstance, PipelineEdge,
    PipelineNode, PipelinePatch, ValueKind,
};
use zerogravity::{Error, Result};

fn invalid() -> Error {
    Error::Malformed("invalid pipeline arguments; use pipeline --help".into())
}
fn assignment(input: &str) -> Result<(&str, &str)> {
    let (a, b) = input.split_once('=').ok_or_else(invalid)?;
    if a.is_empty() || b.is_empty() || input.len() > 4096 {
        return Err(invalid());
    }
    Ok((a, b))
}
pub(super) fn flags(args: &[String]) -> Result<Config> {
    patch(args, false)?.compile()
}

/// Argv syntax only; inheritance and whole-list replacement belong to the SDK.
pub(super) fn patch(args: &[String], overrides: bool) -> Result<PipelinePatch> {
    if args.len() > 1024 {
        return Err(Error::Limit("pipeline flag count".into()));
    }
    let mut name = None;
    let mut input = None;
    let mut output = None;
    let mut realtime = None;
    let mut chat_context_from = None;
    let mut chat_context_enabled = None;
    let mut chat_context_max_bytes = None;
    let mut chat_context_sources = None;
    let mut chat_context_stages = None;
    let mut models = None;
    let mut steps = None;
    let mut links = None;
    let mut routes = None;
    let mut seen = std::collections::BTreeSet::new();
    let mut cleared = std::collections::BTreeSet::new();
    let mut lists = std::collections::BTreeSet::new();
    let mut iter = args.iter();
    while let Some(raw) = iter.next() {
        let flag = if overrides && raw == "--define-model" {
            "--model"
        } else {
            raw.as_str()
        };
        if flag == "--realtime" || (overrides && flag == "--realtime-value") {
            if realtime.is_some() {
                return Err(invalid());
            }
            realtime = Some(if flag == "--realtime" {
                true
            } else {
                match iter.next().map(String::as_str) {
                    Some("true") => true,
                    Some("false") => false,
                    _ => return Err(invalid()),
                }
            });
            continue;
        }
        if !matches!(
            flag,
            "--model"
                | "--step"
                | "--link"
                | "--route"
                | "--chat-context-source"
                | "--chat-context-match"
                | "--chat-context-decision"
        ) && !seen.insert(flag)
        {
            return Err(invalid());
        }
        let value = iter.next().ok_or_else(invalid)?;
        if overrides {
            let is_list = matches!(
                flag,
                "--model"
                    | "--step"
                    | "--link"
                    | "--route"
                    | "--chat-context-source"
                    | "--chat-context-match"
                    | "--chat-context-decision"
            );
            if value == "[]" {
                if !is_list || lists.contains(flag) || !cleared.insert(flag) {
                    return Err(invalid());
                }
                match flag {
                    "--model" => {
                        models.get_or_insert_with(Vec::new);
                    }
                    "--step" => {
                        steps.get_or_insert_with(Vec::new);
                    }
                    "--link" => {
                        links.get_or_insert_with(Vec::new);
                    }
                    "--route" => {
                        routes.get_or_insert_with(Vec::new);
                    }
                    "--chat-context-source" => {
                        chat_context_sources.get_or_insert_with(Vec::new);
                    }
                    _ => {
                        chat_context_stages.get_or_insert_with(Vec::new);
                    }
                }
                continue;
            }
            if is_list {
                if cleared.contains(flag) {
                    return Err(invalid());
                }
                lists.insert(flag);
            }
        }
        match flag {
            "--name" => name = Some(value.to_string()),
            "--input" => input = Some(ValueKind::parse(value)?),
            "--output" => output = Some(ValueKind::parse(value)?),
            "--chat-context-from" => chat_context_from = Some(value.to_string()),
            "--chat-context-enabled" => {
                chat_context_enabled = Some(match value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err(invalid()),
                });
            }
            "--chat-context-max-bytes" => {
                chat_context_max_bytes = Some(value.parse::<usize>().map_err(|_| invalid())?);
            }
            "--chat-context-source" => {
                chat_context_sources
                    .get_or_insert_with(Vec::new)
                    .push(ContextSource::parse(value)?);
            }
            "--chat-context-match" => {
                let mut parts = value.splitn(3, ':');
                let action = FilterAction::parse(parts.next().ok_or_else(invalid)?)?;
                let field = MatchField::parse(parts.next().ok_or_else(invalid)?)?;
                let terms = parts.next().ok_or_else(invalid)?;
                chat_context_stages
                    .get_or_insert_with(Vec::new)
                    .push(ContextFilter::Match {
                        action,
                        field,
                        values: terms.split(',').map(str::to_owned).collect(),
                    });
            }
            "--chat-context-decision" => {
                let mut parts = value.splitn(4, ':');
                let model = parts.next().ok_or_else(invalid)?.to_owned();
                let min_confidence_bps = parts
                    .next()
                    .ok_or_else(invalid)?
                    .parse::<u16>()
                    .map_err(|_| invalid())?;
                let unavailable = DecisionUnavailable::parse(parts.next().ok_or_else(invalid)?)?;
                let question = parts.next().ok_or_else(invalid)?.to_owned();
                chat_context_stages
                    .get_or_insert_with(Vec::new)
                    .push(ContextFilter::Decision {
                        model,
                        question,
                        min_confidence_bps,
                        unavailable,
                    });
            }
            "--model" => {
                let (id, right) = assignment(value)?;
                let (target, source) = right
                    .split_once('@')
                    .map_or((right, None), |(a, b)| (a, Some(b)));
                if target.is_empty() {
                    return Err(Error::Malformed(
                        "declared model target must be nonempty".into(),
                    ));
                }
                models.get_or_insert_with(Vec::new).push(ModelInstance {
                    id: id.into(),
                    target: target.into(),
                    source: source.map(str::to_owned),
                    model_config: None,
                    tokenizer: None,
                    snapshot: None,
                    native_artifacts: None,
                });
            }
            "--step" => {
                let (id, right) = assignment(value)?;
                let (kind, model) = right
                    .split_once(':')
                    .map_or((right, None), |(a, b)| (a, Some(b)));
                steps.get_or_insert_with(Vec::new).push(PipelineNode {
                    id: id.into(),
                    kind: kind.into(),
                    model: model.map(str::to_owned),
                });
            }
            "--link" => {
                let (from, to) = assignment(value)?;
                links.get_or_insert_with(Vec::new).push(PipelineEdge {
                    from: from.into(),
                    to: to.into(),
                });
            }
            "--route" => {
                let (path, protocol) = assignment(value)?;
                routes.get_or_insert_with(Vec::new).push(ApiRoute {
                    path: path.into(),
                    protocol: protocol.into(),
                    pipeline: String::new(),
                });
            }
            _ => return Err(invalid()),
        }
    }
    let has_context = chat_context_from.is_some()
        || chat_context_enabled.is_some()
        || chat_context_max_bytes.is_some()
        || chat_context_sources.is_some()
        || chat_context_stages.is_some();
    if !overrides && has_context && chat_context_from.is_none() {
        return Err(invalid());
    }
    Ok(PipelinePatch {
        name,
        input,
        output,
        realtime,
        models,
        steps,
        links,
        routes,
        chat_context: has_context.then_some(ChatContextPatch {
            from_model: chat_context_from,
            enabled: chat_context_enabled,
            max_bytes: chat_context_max_bytes,
            sources: chat_context_sources,
            stages: chat_context_stages,
        }),
    })
}

pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "help" | "-h") {
        println!(
            "USAGE: zerogravity pipeline check --config CONFIG.yaml
        zerogravity pipeline check --name NAME --input audio-pcm --output audio-pcm \\
         --model asr=nvidia-canary-qwen-2.5b --model llm=qwen3.5-2b \\
         --model tts=qwen3-tts-1.7b --model diar=speaker-diarization \\
         --step hear=asr:asr --step speaker=diarization:diar \\
         --step text=speakers-text --step answer=llm:llm --step talk=tts:tts \\
         --link input.value=hear.audio --link hear.transcript=speaker.transcript \\
         --link input.value=speaker.audio --link speaker.speakers=text.speakers \\
         --link text.text=answer.text --link answer.text=talk.text \\
         --link talk.audio=output.value --route /v1/realtime=websocket --realtime
       zerogravity pipeline check --config examples/pipelines/contextual-voice-assistant.yaml
       zerogravity pipeline run --config CONFIG.yaml   (NOT IMPLEMENTED)

OPTIONAL CHAT CONTEXT (off by default):
    chat_context: {{enabled: true, from_model: assistant, max_bytes: 4096}}
    --chat-context-from assistant [--chat-context-max-bytes 4096]
    --chat-context-enabled false    (keeps optional typed ports; supplies no context)
    context.value -> stage.context only when chat_context policy exists
    Default context sources: session/user_message + session/assistant_message.
    Optional sources include session/tool_call and session/tool_result,
    application/system_prompt, application/developer_prompt,
    shared:grant-id/assistant_message when granted by the real source owner.
    --chat-context-source ORIGIN/KIND  (repeat)
    --chat-context-match exclude:kind:tool_call,tool_result  (repeat)
    --chat-context-match exclude:tool_name:read,exec  (repeat)
    --chat-context-decision gate:7500:omit:Relevant to audio?  (repeat)
    NanoJev-style decision model requires explicit verified model binding.
    Source grants must be checked; config never bypasses authorization.

All stages/ports, model instances and routes compile into ONE process graph.
Only structural checking is implemented; no weights, API port, stream or model starts.
The same verified model instance may be referenced by multiple pipelines.
Single-model resident inference continues through zerogravity run MODEL.gguf."
        );
        return Ok(());
    }
    let operation = args.first().ok_or_else(invalid)?;
    if operation != "check" && operation != "run" {
        return Err(invalid());
    }
    let (config, valid) = match &args[1..] {
        [flag, file] if flag == "--config" => {
            let checked = load_document(file)?.check()?;
            (checked.config, checked.shape)
        }
        rest if rest.iter().any(|x| x == "--config") => return Err(invalid()),
        rest => {
            let config = flags(rest)?;
            let valid = pipelines::check(&config)?;
            (config, valid)
        }
    };
    if operation == "run" {
        return pipelines::run(&config);
    }
    println!("PIPELINE STRUCTURE VALIDATED (not runnable; no models or APIs started)");
    for (name, ordered) in &valid.ordered {
        println!(
            "Pipeline: {name}; topological stages: {}",
            ordered.join(" -> ")
        );
    }
    println!(
        "Model instances: {}; reused model bindings: {}; declared routes: {}",
        config.models.len(),
        valid.shared_model_refs,
        valid.route_count
    );
    println!(
        "Committed chat context: {} configured, {} ENABLED (source filters only; history/backend access NOT IMPLEMENTED)",
        valid.context_configured_pipelines, valid.context_enabled_pipelines,
    );
    println!(
        "Context filters: {} ordered stages, {} typed decision stages; max {} configured bytes",
        valid.context_filter_stages,
        valid.context_decision_stages,
        ChatContextPolicy::MAX_BYTES,
    );
    println!(
        "Streaming audio, context-aware ASR/translation, diarization, image frames, API listeners and joint StateRoot publication: NOT IMPLEMENTED."
    );
    Ok(())
}
