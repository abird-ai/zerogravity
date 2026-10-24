//! Human-editable single-process multi-model pipeline compiler FRONTEND.
//! Config checking has no side effects. Execution, serving and every media
//! adapter are NOT IMPLEMENTED until source, physical and protocol gates pass.
use serde::Deserialize;
use std::fs;
use zerogravity::context_filter::{
    ContextFilter, ContextFilterPlan, ContextSource, DecisionUnavailable, FilterAction, MatchField,
    default_sources,
};
use zerogravity::pipeline::{
    ApiRoute, ChatContextPolicy, Config, ModelInstance, Pipeline, PipelineEdge, PipelineNode,
    ValueKind,
};
use zerogravity::{Error, Result};

const MAX_CONFIG_BYTES: usize = 256 * 1024;
const SCHEMA: &str = "zerogravity.pipeline.v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Document {
    pub(super) schema: String,
    #[serde(default)]
    pub(super) model: Vec<ModelDoc>,
    #[serde(default)]
    pub(super) pipeline: Vec<PipelineDoc>,
    #[serde(default)]
    pub(super) route: Vec<RouteDoc>,
    #[serde(default)]
    pub(super) run: Option<crate::run_cli::RunDoc>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelDoc {
    pub(super) id: String,
    pub(super) target: String,
    pub(super) source: Option<String>,
    pub(super) model_config: Option<String>,
    pub(super) tokenizer: Option<String>,
    pub(super) snapshot: Option<String>,
    pub(super) native_artifacts: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PipelineDoc {
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
    sources: Option<Vec<String>>,
    #[serde(default)]
    filter: Vec<ContextFilterDoc>,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ContextFilterDoc {
    Match {
        action: String,
        field: String,
        values: Vec<String>,
    },
    Decision {
        model: String,
        question: String,
        min_confidence_bps: Option<u16>,
        on_unavailable: Option<String>,
    },
}
impl ContextFilterDoc {
    fn compile(self) -> Result<ContextFilter> {
        match self {
            Self::Match {
                action,
                field,
                values,
            } => Ok(ContextFilter::Match {
                action: FilterAction::parse(&action)?,
                field: MatchField::parse(&field)?,
                values,
            }),
            Self::Decision {
                model,
                question,
                min_confidence_bps,
                on_unavailable,
            } => Ok(ContextFilter::Decision {
                model,
                question,
                min_confidence_bps: min_confidence_bps.unwrap_or(7500),
                unavailable: DecisionUnavailable::parse(
                    on_unavailable.as_deref().unwrap_or("omit"),
                )?,
            }),
        }
    }
}
impl ChatContextDoc {
    fn compile(self) -> Result<ChatContextPolicy> {
        let sources = match self.sources {
            Some(names) => names
                .into_iter()
                .map(|s| ContextSource::parse(&s))
                .collect::<Result<Vec<_>>>()?,
            None => default_sources(),
        };
        let stages = self
            .filter
            .into_iter()
            .map(ContextFilterDoc::compile)
            .collect::<Result<Vec<_>>>()?;
        Ok(ChatContextPolicy {
            enabled: self.enabled,
            from_model: self.from_model,
            max_bytes: self.max_bytes.unwrap_or(ChatContextPolicy::DEFAULT_BYTES),
            selection: ContextFilterPlan { sources, stages },
        })
    }
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
pub(super) struct RouteDoc {
    path: String,
    protocol: String,
    pipeline: String,
}
impl Document {
    pub(super) fn compile(self) -> Result<Config> {
        if !matches!(self.schema.as_str(), SCHEMA | "zerogravity.pipeline.v2") {
            return Err(Error::Unsupported(
                "pipeline schema version unavailable".into(),
            ));
        }
        if self.schema == SCHEMA
            && (self.run.is_some()
                || self.model.iter().any(|m| {
                    m.model_config.is_some()
                        || m.tokenizer.is_some()
                        || m.snapshot.is_some()
                        || m.native_artifacts.is_some()
                }))
        {
            return Err(Error::Malformed(
                "run/source-sidecar fields require zerogravity.pipeline.v2".into(),
            ));
        }
        Ok(Config {
            models: self
                .model
                .into_iter()
                .map(|m| {
                    // Only a direct CLI source may infer its target from parsed
                    // source geometry; a declared table states it now.
                    if m.target.is_empty() {
                        return Err(Error::Malformed(
                            "declared model target must be nonempty".into(),
                        ));
                    }
                    Ok(ModelInstance {
                        id: m.id,
                        target: m.target,
                        source: m.source,
                        model_config: m.model_config,
                        tokenizer: m.tokenizer,
                        snapshot: m.snapshot,
                        native_artifacts: m.native_artifacts,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            pipelines: self
                .pipeline
                .into_iter()
                .map(|p| {
                    Ok(Pipeline {
                        name: p.name,
                        input: ValueKind::parse(&p.input)?,
                        output: ValueKind::parse(&p.output)?,
                        realtime: p.realtime,
                        chat_context: p.chat_context.map(ChatContextDoc::compile).transpose()?,
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
pub(super) fn flags(args: &[String]) -> Result<Config> {
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
    let mut chat_context_sources = Vec::new();
    let mut chat_context_stages = Vec::new();
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
        if !matches!(
            flag.as_str(),
            "--model"
                | "--step"
                | "--link"
                | "--route"
                | "--chat-context-source"
                | "--chat-context-match"
                | "--chat-context-decision"
        ) && !seen.insert(flag.as_str())
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
            "--chat-context-source" => {
                chat_context_sources.push(ContextSource::parse(value)?);
            }
            "--chat-context-match" => {
                let mut parts = value.splitn(3, ':');
                let action = FilterAction::parse(parts.next().ok_or_else(invalid)?)?;
                let field = MatchField::parse(parts.next().ok_or_else(invalid)?)?;
                let terms = parts.next().ok_or_else(invalid)?;
                chat_context_stages.push(ContextFilter::Match {
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
                chat_context_stages.push(ContextFilter::Decision {
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
                models.push(ModelInstance {
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
        && (chat_context_enabled.is_some()
            || chat_context_max_bytes.is_some()
            || !chat_context_sources.is_empty()
            || !chat_context_stages.is_empty())
    {
        return Err(invalid());
    }
    let chat_context = chat_context_from.map(|from_model| ChatContextPolicy {
        enabled: chat_context_enabled.unwrap_or(true),
        from_model,
        max_bytes: chat_context_max_bytes.unwrap_or(ChatContextPolicy::DEFAULT_BYTES),
        selection: ContextFilterPlan {
            sources: if chat_context_sources.is_empty() {
                default_sources()
            } else {
                chat_context_sources
            },
            stages: chat_context_stages,
        },
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
/// Bound the *resolved* YAML AST before typed deserialization. YAML
/// anchors/aliases and shallow merge keys are useful for shared settings,
/// but custom tags are not executable or trusted pipeline components.
/// Every expanded scalar and map key is counted within this finite policy.
pub(super) fn check_yaml_shape(value: &serde_yaml::Value) -> Result<()> {
    use serde_yaml::Value;
    const MAX_NODES: usize = 16_384;
    const MAX_DEPTH: usize = 32;
    let mut stack = vec![(value, 0usize)];
    let mut nodes = 0usize;
    let mut scalar_bytes = 0usize;
    while let Some((node, depth)) = stack.pop() {
        nodes += 1;
        if nodes > MAX_NODES || depth > MAX_DEPTH {
            return Err(Error::Limit("expanded YAML pipeline size/depth".into()));
        }
        match node {
            Value::Tagged(_) => {
                return Err(Error::Unsupported(
                    "YAML tags cannot select a pipeline component or executable constructor".into(),
                ));
            }
            Value::Mapping(map) => {
                for (key, value) in map {
                    if !matches!(key, Value::String(s) if s.len() <= 128) {
                        return Err(Error::Malformed(
                            "YAML mapping keys must be bounded strings".into(),
                        ));
                    }
                    stack.push((value, depth + 1));
                    stack.push((key, depth + 1));
                }
            }
            Value::Sequence(items) => {
                stack.extend(items.iter().map(|item| (item, depth + 1)));
            }
            Value::String(text) => {
                scalar_bytes = scalar_bytes
                    .checked_add(text.len())
                    .ok_or_else(|| Error::Limit("expanded YAML scalar bytes".into()))?;
                if scalar_bytes > MAX_CONFIG_BYTES {
                    return Err(Error::Limit("expanded YAML scalar bytes".into()));
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }
    Ok(())
}

pub(super) fn load_document(path: &str) -> Result<Document> {
    let bytes = fs::metadata(path)?.len();
    if bytes == 0 || bytes > MAX_CONFIG_BYTES as u64 {
        return Err(Error::Limit("pipeline config size limit".into()));
    }
    // Recheck after read: the source can change between metadata and read.
    let mut bytes = Vec::new();
    use std::io::Read;
    fs::File::open(path)?
        .take(MAX_CONFIG_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.is_empty() || bytes.len() > MAX_CONFIG_BYTES {
        return Err(Error::Limit(
            "pipeline config changed or exceeds size".into(),
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| Error::Malformed("pipeline config not UTF-8".into()))?;
    // No arbitrary code or custom YAML tags; even anchor/merge-expanded
    // values undergo the same bounded type and source-graph verification.
    let mut node: serde_yaml::Value = serde_yaml::from_str(text)
        .map_err(|_| Error::Malformed("invalid bounded pipeline YAML schema".into()))?;
    check_yaml_shape(&node)?;
    // YAML merge keys are a source-level convenience, not dynamic inheritance
    // across model registry, device capabilities or source authorization.
    node.apply_merge()
        .map_err(|_| Error::Malformed("invalid YAML mapping merge".into()))?;
    check_yaml_shape(&node)?;
    if node.get("schema").and_then(serde_yaml::Value::as_str) == Some("zerogravity.pipeline.v2") {
        fn no_null(v: &serde_yaml::Value) -> bool {
            match v {
                serde_yaml::Value::Null => false,
                serde_yaml::Value::Sequence(s) => s.iter().all(no_null),
                serde_yaml::Value::Mapping(m) => m.iter().all(|(k, v)| no_null(k) && no_null(v)),
                _ => true,
            }
        }
        if !no_null(&node) {
            return Err(Error::Malformed(
                "pipeline.v2 does not accept null fields; omit optional values".into(),
            ));
        }
    }
    let doc: Document = serde_yaml::from_value(node)
        .map_err(|_| Error::Malformed("invalid typed pipeline YAML schema".into()))?;
    Ok(doc)
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
    let (config, entrypoint) = match &args[1..] {
        [flag, file] if flag == "--config" => {
            let doc = load_document(file)?;
            let entrypoint = doc
                .run
                .as_ref()
                .and_then(|r| r.entrypoint.as_ref())
                .map(crate::run_cli::EntryDoc::typed);
            (doc.compile()?, entrypoint)
        }
        rest if rest.iter().any(|x| x == "--config") => return Err(invalid()),
        rest => (flags(rest)?, None),
    };
    let valid = match entrypoint {
        Some(entry) => config.validate_entrypoint(&entry)?,
        None => config.validate()?,
    };
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
