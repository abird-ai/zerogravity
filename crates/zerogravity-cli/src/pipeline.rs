//! Human-editable single-process multi-model pipeline compiler FRONTEND.
//! Config checking has no side effects. Execution, serving and every media
//! adapter are NOT IMPLEMENTED until source, physical and protocol gates pass.
use serde::Deserialize;
use std::fs;
use zerogravity::pipeline::{
    ApiRoute, ChatContextPolicy, Config, ModelInstance, Pipeline, PipelineEdge, PipelineNode,
    ValueKind,
};
use zerogravity::{Error, Result};

const MAX_CONFIG_BYTES: usize = 256 * 1024;
const SCHEMA: &str = "zerogravity.pipeline.v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: String,
    #[serde(default)]
    model: Vec<ModelDoc>,
    pipeline: Vec<PipelineDoc>,
    #[serde(default)]
    route: Vec<RouteDoc>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelDoc {
    id: String,
    target: String,
    source: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineDoc {
    name: String,
    input: String,
    output: String,
    #[serde(default)]
    realtime: bool,
    /// Absent = off. Present = explicit same-session prior committed context.
    chat_context: Option<ChatContextDoc>,
    #[serde(default)]
    step: Vec<StepDoc>,
    #[serde(default)]
    link: Vec<LinkDoc>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatContextDoc {
    enabled: bool,
    from_model: String,
    max_bytes: Option<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepDoc {
    id: String,
    kind: String,
    model: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LinkDoc {
    from: String,
    to: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteDoc {
    path: String,
    protocol: String,
    pipeline: String,
}
impl Document {
    fn compile(self) -> Result<Config> {
        if self.schema != SCHEMA {
            return Err(Error::Unsupported(
                "pipeline schema version unavailable".into(),
            ));
        }
        Ok(Config {
            models: self
                .model
                .into_iter()
                .map(|m| ModelInstance {
                    id: m.id,
                    target: m.target,
                    source: m.source,
                })
                .collect(),
            pipelines: self
                .pipeline
                .into_iter()
                .map(|p| {
                    Ok(Pipeline {
                        name: p.name,
                        input: ValueKind::parse(&p.input)?,
                        output: ValueKind::parse(&p.output)?,
                        realtime: p.realtime,
                        chat_context: p.chat_context.map(|c| ChatContextPolicy {
                            enabled: c.enabled,
                            from_model: c.from_model,
                            max_bytes: c.max_bytes.unwrap_or(ChatContextPolicy::DEFAULT_BYTES),
                        }),
                        steps: p
                            .step
                            .into_iter()
                            .map(|s| PipelineNode {
                                id: s.id,
                                kind: s.kind,
                                model: s.model,
                            })
                            .collect(),
                        links: p
                            .link
                            .into_iter()
                            .map(|l| PipelineEdge {
                                from: l.from,
                                to: l.to,
                            })
                            .collect(),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            routes: self
                .route
                .into_iter()
                .map(|r| ApiRoute {
                    path: r.path,
                    protocol: r.protocol,
                    pipeline: r.pipeline,
                })
                .collect(),
        })
    }
}
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
fn flags(args: &[String]) -> Result<Config> {
    if args.len() > 1024 {
        return Err(Error::Limit("pipeline flag count".into()));
    }
    let mut name = None;
    let mut input = None;
    let mut output = None;
    let mut realtime = false;
    let mut chat_context_from = None;
    let mut chat_context_enabled = None;
    let mut chat_context_max_bytes = None;
    let mut models = Vec::new();
    let mut steps = Vec::new();
    let mut links = Vec::new();
    let mut routes = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        if flag == "--realtime" {
            if realtime {
                return Err(invalid());
            }
            realtime = true;
            continue;
        }
        if !matches!(flag.as_str(), "--model" | "--step" | "--link" | "--route")
            && !seen.insert(flag.as_str())
        {
            return Err(invalid());
        }
        let value = iter.next().ok_or_else(invalid)?;
        match flag.as_str() {
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
            "--model" => {
                let (id, right) = assignment(value)?;
                let (target, source) = right
                    .split_once('@')
                    .map_or((right, None), |(a, b)| (a, Some(b)));
                models.push(ModelInstance {
                    id: id.into(),
                    target: target.into(),
                    source: source.map(str::to_owned),
                });
            }
            "--step" => {
                let (id, right) = assignment(value)?;
                let (kind, model) = right
                    .split_once(':')
                    .map_or((right, None), |(a, b)| (a, Some(b)));
                steps.push(PipelineNode {
                    id: id.into(),
                    kind: kind.into(),
                    model: model.map(str::to_owned),
                });
            }
            "--link" => {
                let (from, to) = assignment(value)?;
                links.push(PipelineEdge {
                    from: from.into(),
                    to: to.into(),
                });
            }
            "--route" => {
                let (path, protocol) = assignment(value)?;
                routes.push(ApiRoute {
                    path: path.into(),
                    protocol: protocol.into(),
                    pipeline: String::new(),
                });
            }
            _ => return Err(invalid()),
        }
    }
    if chat_context_from.is_none()
        && (chat_context_enabled.is_some() || chat_context_max_bytes.is_some())
    {
        return Err(invalid());
    }
    let chat_context = chat_context_from.map(|from_model| ChatContextPolicy {
        // The explicit --chat-context-from is opt-in; --chat-context-enabled
        // false is a toggled-off policy, retaining the wired optional ports.
        enabled: chat_context_enabled.unwrap_or(true),
        from_model,
        max_bytes: chat_context_max_bytes.unwrap_or(ChatContextPolicy::DEFAULT_BYTES),
    });
    let name = name.ok_or_else(invalid)?;
    for route in &mut routes {
        route.pipeline.clone_from(&name);
    }
    Ok(Config {
        models,
        pipelines: vec![Pipeline {
            name,
            input: input.ok_or_else(invalid)?,
            output: output.ok_or_else(invalid)?,
            realtime,
            chat_context,
            steps,
            links,
        }],
        routes,
    })
}
fn load(path: &str) -> Result<Config> {
    let bytes = fs::metadata(path)?.len();
    if bytes == 0 || bytes > MAX_CONFIG_BYTES as u64 {
        return Err(Error::Limit("pipeline config size limit".into()));
    }
    // Recheck after read: the source can change between metadata and read.
    let bytes = fs::read(path)?;
    if bytes.is_empty() || bytes.len() > MAX_CONFIG_BYTES {
        return Err(Error::Limit(
            "pipeline config changed or exceeds size".into(),
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| Error::Malformed("pipeline config not UTF-8".into()))?;
    let doc: Document = toml::from_str(text)
        .map_err(|_| Error::Malformed("invalid bounded pipeline TOML schema".into()))?;
    doc.compile()
}
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "help" | "-h") {
        println!(
            "USAGE: zerogravity pipeline check --config CONFIG.toml
        zerogravity pipeline check --name NAME --input audio-pcm --output audio-pcm \\
         --model asr=nvidia-canary-qwen-2.5b --model llm=qwen3.5-2b \\
         --model tts=qwen3-tts-1.7b --model diar=speaker-diarization \\
         --step hear=asr:asr --step speaker=diarization:diar \\
         --step text=speakers-text --step answer=llm:llm --step talk=tts:tts \\
         --link input.value=hear.audio --link hear.transcript=speaker.transcript \\
         --link input.value=speaker.audio --link speaker.speakers=text.speakers \\
         --link text.text=answer.text --link answer.text=talk.text \\
         --link talk.audio=output.value --route /v1/realtime=websocket --realtime
       zerogravity pipeline check --config examples/pipelines/contextual-voice-assistant.toml
       zerogravity pipeline run --config CONFIG.toml   (NOT IMPLEMENTED)

OPTIONAL CHAT CONTEXT (off by default):
    [pipeline.chat_context] enabled=true, from_model='assistant', max_bytes=4096
    --chat-context-from assistant [--chat-context-max-bytes 4096]
    --chat-context-enabled false    (keeps optional typed ports; supplies no context)
    context.value -> stage.context only when chat_context policy exists
    Bound to prior committed, same-session visible chat turns only;
    no current-turn feedback, raw KV, hidden prompts or cross-user context.

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
    let config = match &args[1..] {
        [flag, file] if flag == "--config" => load(file)?,
        rest if rest.iter().any(|x| x == "--config") => return Err(invalid()),
        rest => flags(rest)?,
    };
    let valid = config.validate()?;
    if operation == "run" {
        return Err(Error::Unsupported(
            "compiled multi-model pipeline execution NOT IMPLEMENTED; no source or device was opened".into()
        ));
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
        "Committed chat context: {} configured, {} ENABLED (same-session previous-turn hints; max 16KiB; no backend attached)",
        valid.context_configured_pipelines, valid.context_enabled_pipelines,
    );
    println!(
        "Streaming audio, context-aware ASR/translation, diarization, image frames, API listeners and joint StateRoot publication: NOT IMPLEMENTED."
    );
    Ok(())
}
