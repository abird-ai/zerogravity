//! Pipeline argv and presentation; YAML/schema/checking live in the public SDK.
use zerogravity::context_filter::{
    ContextFilter, ContextSource, DecisionUnavailable, FilterAction, MatchField,
};
#[cfg(test)]
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

pub(super) fn help(child: Option<&str>) {
    match child {
        Some("list") => println!(
            "USAGE: zerogravity [--profile NAME | --config FILE] pipeline list [--format json]"
        ),
        Some("inspect") => println!(
            "USAGE: zerogravity [--profile NAME | --config FILE] pipeline inspect NAME [--format json]"
        ),
        Some("run") => println!(
            "USAGE: zerogravity pipeline run NAME|FILE [--config FILE | --profile NAME]\nValidates the graph, then refuses: general pipeline execution is not qualified. Direct text models use run MODEL."
        ),
        Some("check") => {
            println!(
                "USAGE: zerogravity pipeline check FILE [--format json]
       zerogravity pipeline check --config CONFIG.yaml

ADVANCED STRUCTURAL CHECK:
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
       zerogravity pipeline run --config CONFIG.yaml   (execution refused)

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
        }
        _ => println!(
            "USAGE: zerogravity pipeline list | inspect NAME | check FILE | run NAME|FILE\nSelected config/profile supplies the document. --format json for list/inspect/check.\nChecks are structural only; no source, device, model, history or API is started.\nRun 'zerogravity help pipeline check' for repeatable graph flags."
        ),
    }
}

use crate::args::Format;

/// One document or one flag-defined graph; the original typed compiler is
/// the sole structural authority. Report only declared structure.
pub(super) fn run(
    args: &[String],
    document: Option<(&'static str, String)>,
    format: Format,
) -> Result<()> {
    let (operation, rest) = args.split_first().ok_or_else(invalid)?;
    let mut selection = document
        .as_ref()
        .map(|(flag, value)| (*flag, value.as_str()));
    let mut name = None;
    let mut flags_args = None;
    match (operation.as_str(), rest) {
        ("list", []) => {}
        ("inspect", [id]) => name = Some(id.as_str()),
        ("check", [file]) if selection.is_none() && !file.starts_with('-') => {
            selection = Some(("--config", file))
        }
        ("run", [value]) if !value.starts_with('-') => {
            let path = std::path::Path::new(value);
            if selection.is_none()
                && (path.is_absolute()
                    || value.starts_with('.')
                    || path
                        .extension()
                        .is_some_and(|ext| ext == "yaml" || ext == "yml"))
            {
                selection = Some(("--config", value));
            } else {
                name = Some(value.as_str());
            }
        }
        ("check" | "run", []) => {}
        ("check" | "run", rest)
            if selection.is_none() && rest.first().is_some_and(|arg| arg.starts_with("--")) =>
        {
            flags_args = Some(rest)
        }
        _ => return Err(invalid()),
    }
    let (config, valid) = if let Some(flags_args) = flags_args {
        let config = flags(flags_args)?;
        let valid = pipelines::check(&config)?;
        (config, valid)
    } else {
        let selected = crate::config_cli::read_selected_document(selection)?;
        let Some((_, doc)) = selected else {
            if operation == "list" {
                return list_empty(format, 0);
            }
            return Err(Error::Malformed(
                "select a pipeline document or explicit graph flags".into(),
            ));
        };
        crate::output::install_document(doc.ui.unwrap_or_default());
        if operation == "list" && doc.pipeline.is_empty() && doc.route.is_empty() {
            let config = doc.compile()?;
            config.validate_model_table()?;
            return list_empty(format, config.models.len());
        }
        let checked = doc.check()?;
        (checked.config, checked.shape)
    };
    if let Some(name) = name
        && !config.pipelines.iter().any(|p| p.name == name)
    {
        return Err(Error::Malformed(
            "pipeline not found in selected document".into(),
        ));
    }
    if operation == "run" {
        return pipelines::run(&config);
    }
    let rows = config
        .pipelines
        .iter()
        .filter(|p| name.is_none_or(|name| p.name == name));
    if format == Format::Json {
        let pipelines = rows
            .map(|p| pipeline_json(&config, &valid, p))
            .collect::<Vec<_>>();
        return crate::output::json(&serde_json::json!({
            "schema": "zerogravity.pipeline.structure.v1",
            "structure_valid": true, "execution_qualified": false,
            "pipelines": pipelines, "model_instances": config.models.len(),
            "shared_model_refs": valid.shared_model_refs, "declared_routes": valid.route_count,
        }));
    }
    if operation == "list" {
        println!("Pipelines — declared structure only");
        for p in rows {
            println!(
                "{}  {} -> {}  {} stages  {}",
                p.name,
                p.input.label(),
                p.output.label(),
                p.steps.len(),
                if p.realtime {
                    "realtime declared"
                } else {
                    "request"
                }
            );
        }
        println!("Execution not qualified.");
        return Ok(());
    }
    if operation == "inspect" {
        for p in rows {
            println!(
                "Pipeline: {} · {} -> {}",
                p.name,
                p.input.label(),
                p.output.label()
            );
            for step in ordered_stages(&valid, p) {
                let model = step.model.as_deref().unwrap_or("");
                let target = config
                    .models
                    .iter()
                    .find(|row| row.id == model)
                    .map_or("", |row| row.target.as_str());
                println!("  {}  {}  {} {}", step.id, step.kind, model, target);
            }
            for link in &p.links {
                println!("  {} -> {}", link.from, link.to);
            }
            for route in config
                .routes
                .iter()
                .filter(|route| route.pipeline == p.name)
            {
                println!(
                    "  Route: {} ({}) — declared, not listening",
                    route.path, route.protocol
                );
            }
            if let Some(context) = &p.chat_context {
                println!(
                    "  Chat context: {} · {} bytes · {} filter stages",
                    if context.enabled {
                        "enabled"
                    } else {
                        "disabled"
                    },
                    context.max_bytes,
                    context.selection.stages.len()
                );
            }
        }
        println!("Structure valid; execution not qualified.");
        return Ok(());
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

fn list_empty(format: Format, model_instances: usize) -> Result<()> {
    if format == Format::Json {
        crate::output::json(&serde_json::json!({
            "schema": "zerogravity.pipeline.structure.v1", "structure_valid": true,
            "execution_qualified": false, "pipelines": [], "model_instances": model_instances,
            "shared_model_refs": 0, "declared_routes": 0,
        }))
    } else {
        println!("No pipelines declared in the selected document.");
        Ok(())
    }
}

fn ordered_stages<'a>(
    shape: &'a pipelines::VerifiedShape,
    pipeline: &'a pipelines::Pipeline,
) -> impl Iterator<Item = &'a PipelineNode> {
    shape
        .ordered
        .iter()
        .find(|(name, _)| name == &pipeline.name)
        .into_iter()
        .flat_map(|(_, ids)| ids)
        .filter_map(|id| pipeline.steps.iter().find(|step| &step.id == id))
}

fn pipeline_json(
    config: &Config,
    shape: &pipelines::VerifiedShape,
    pipeline: &pipelines::Pipeline,
) -> serde_json::Value {
    use serde_json::json;
    let stages = ordered_stages(shape, pipeline).map(|step| json!({
        "id": step.id, "kind": step.kind, "model": step.model,
        "target": step.model.as_ref().and_then(|id| config.models.iter().find(|model| &model.id == id)).map(|model| &model.target),
    })).collect::<Vec<_>>();
    json!({
        "name": pipeline.name, "input": pipeline.input.label(), "output": pipeline.output.label(),
        "realtime": pipeline.realtime, "stages": stages,
        "links": pipeline.links.iter().map(|edge| json!({"from":edge.from,"to":edge.to})).collect::<Vec<_>>(),
        "routes": config.routes.iter().filter(|route| route.pipeline == pipeline.name).map(|route| json!({"path":route.path,"protocol":route.protocol})).collect::<Vec<_>>(),
        "chat_context": pipeline.chat_context.as_ref().map(|context| json!({
            "enabled": context.enabled, "from_model": context.from_model, "max_bytes": context.max_bytes,
            "source_count": context.selection.sources.len(), "filter_stages": context.selection.stages.len(),
        })),
        "execution_qualified": false,
    })
}
