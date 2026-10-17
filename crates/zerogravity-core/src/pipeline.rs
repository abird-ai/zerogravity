//! Source-neutral single-process, multi-model pipeline STRUCTURE.
//!
//! A pure, bounded configuration checker, not a workflow interpreter. Only the
//! later TaskCoordinator may commit neural state and media outputs. It is unsafe
//! to mistake a compiled graph for loaded models, admitted tensors, network
//! listeners, trained inference or realtime media support.
//!
//! Model instances are named once and may be shared across multiple pipeline
//! graphs. Component kinds are registered through typed port contracts, not
//! per-model independent servers or an untyped "pass arbitrary JSON" edge.
use crate::targets::{TaskKind, find};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueKind {
    Text,
    AudioPcm,
    TimedTranscript,
    SpeakerTranscript,
    ImageRgba,
    /// Bounded immutable hints derived from previously committed, same-session
    /// user-visible turns; never raw KV state, private system prompts or tools.
    CommittedChatContext,
}
impl ValueKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::AudioPcm => "audio-pcm",
            Self::TimedTranscript => "timed-transcript",
            Self::SpeakerTranscript => "speaker-transcript",
            Self::ImageRgba => "image-rgba",
            Self::CommittedChatContext => "committed-chat-context",
        }
    }
    pub fn parse(label: &str) -> Result<Self> {
        match label {
            "text" => Ok(Self::Text),
            "audio-pcm" => Ok(Self::AudioPcm),
            "timed-transcript" => Ok(Self::TimedTranscript),
            "speaker-transcript" => Ok(Self::SpeakerTranscript),
            "image-rgba" => Ok(Self::ImageRgba),
            "committed-chat-context" => Err(Error::Unsupported(
                "chat context is a trusted session snapshot, not a caller input".into(),
            )),
            _ => Err(Error::Unsupported("unknown pipeline value type".into())),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Port {
    pub name: &'static str,
    pub value: ValueKind,
    /// Unconnected means None, not fabricated transcript or injected chat state.
    pub optional: bool,
}
const fn port(name: &'static str, value: ValueKind) -> Port {
    Port {
        name,
        value,
        optional: false,
    }
}
const fn optional_port(name: &'static str, value: ValueKind) -> Port {
    Port {
        name,
        value,
        optional: true,
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskBinding {
    Asr,
    Diarization,
    Language,
    Speech,
    Image,
    None,
}
#[derive(Clone, Copy, Debug)]
pub struct ComponentContract {
    pub kind: &'static str,
    pub binding: TaskBinding,
    pub inputs: &'static [Port],
    pub outputs: &'static [Port],
}
use ValueKind::{
    AudioPcm as A, CommittedChatContext as C, ImageRgba as I, SpeakerTranscript as ST, Text as T,
    TimedTranscript as TR,
};
pub const COMPONENTS: &[ComponentContract] = &[
    ComponentContract {
        kind: "asr",
        binding: TaskBinding::Asr,
        inputs: &[port("audio", A), optional_port("context", C)],
        outputs: &[port("transcript", TR)],
    },
    ComponentContract {
        kind: "diarization",
        binding: TaskBinding::Diarization,
        inputs: &[
            port("audio", A),
            port("transcript", TR),
            optional_port("context", C),
        ],
        outputs: &[port("speakers", ST)],
    },
    ComponentContract {
        kind: "transcript-text",
        binding: TaskBinding::None,
        inputs: &[port("transcript", TR), optional_port("context", C)],
        outputs: &[port("text", T)],
    },
    ComponentContract {
        kind: "speakers-text",
        binding: TaskBinding::None,
        inputs: &[port("speakers", ST), optional_port("context", C)],
        outputs: &[port("text", T)],
    },
    // Translation is a separate registered stage. Language hints must not
    // rewrite raw ASR timings or acoustic speaker evidence in place.
    ComponentContract {
        kind: "translate",
        binding: TaskBinding::Language,
        inputs: &[port("speakers", ST), optional_port("context", C)],
        outputs: &[port("speakers", ST)],
    },
    ComponentContract {
        kind: "llm",
        binding: TaskBinding::Language,
        inputs: &[port("text", T)],
        outputs: &[port("text", T)],
    },
    ComponentContract {
        kind: "tts",
        binding: TaskBinding::Speech,
        inputs: &[port("text", T)],
        outputs: &[port("audio", A)],
    },
    ComponentContract {
        kind: "image-generate",
        binding: TaskBinding::Image,
        inputs: &[port("prompt", T)],
        outputs: &[port("image", I)],
    },
];
pub fn component(kind: &str) -> Option<&'static ComponentContract> {
    COMPONENTS.iter().find(|c| c.kind == kind)
}

#[derive(Clone, Debug)]
pub struct ModelInstance {
    pub id: String,
    pub target: String,
    /// Not fetched, opened or granted authority by graph validation.
    pub source: Option<String>,
}
#[derive(Clone, Debug)]
pub struct PipelineNode {
    pub id: String,
    pub kind: String,
    pub model: Option<String>,
}
#[derive(Clone, Debug)]
pub struct PipelineEdge {
    pub from: String,
    pub to: String,
}
#[derive(Clone, Debug)]
pub struct Pipeline {
    pub name: String,
    pub input: ValueKind,
    pub output: ValueKind,
    pub realtime: bool,
    /// Opt-in prior-turn immutable same-session chat hints, never a current
    /// generation feedback loop. If disabled, context channels carry None.
    pub chat_context: Option<ChatContextPolicy>,
    pub steps: Vec<PipelineNode>,
    pub links: Vec<PipelineEdge>,
}
/// Declarative context policy, not an implemented session-history loader.
/// A later model-specific ASR/translation backend must explicitly prove it
/// accepts these hints; incompatible models fail rather than pretending.
#[derive(Clone, Debug)]
pub struct ChatContextPolicy {
    pub enabled: bool,
    pub from_model: String,
    pub max_bytes: usize,
}
impl ChatContextPolicy {
    pub const MAX_BYTES: usize = 16 * 1024;
    pub const DEFAULT_BYTES: usize = 4 * 1024;
}

#[derive(Clone, Debug)]
pub struct ApiRoute {
    pub path: String,
    pub protocol: String,
    pub pipeline: String,
}
#[derive(Clone, Debug)]
pub struct Config {
    pub models: Vec<ModelInstance>,
    pub pipelines: Vec<Pipeline>,
    pub routes: Vec<ApiRoute>,
}
#[derive(Clone, Debug)]
pub struct VerifiedShape {
    /// Deterministic topological order; per-stage execution is NOT implemented.
    pub ordered: Vec<(String, Vec<String>)>,
    pub shared_model_refs: usize,
    pub route_count: usize,
    pub context_configured_pipelines: usize,
    pub context_enabled_pipelines: usize,
}
fn invalid() -> Error {
    Error::Malformed("invalid bounded pipeline graph/config".into())
}
fn unsupported() -> Error {
    Error::Unsupported("pipeline component/model port contract is not registered".into())
}
fn id_ok(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
fn field_ok(value: &str) -> bool {
    !value.is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control)
}
fn port_ref(s: &str) -> Result<(&str, &str)> {
    let (node, p) = s.rsplit_once('.').ok_or_else(invalid)?;
    if !id_ok(node) || !id_ok(p) {
        return Err(invalid());
    }
    Ok((node, p))
}
fn checked_value(
    endpoint: &str,
    output: bool,
    steps: &BTreeMap<&str, &ComponentContract>,
    input: ValueKind,
    finish: ValueKind,
    has_context_policy: bool,
) -> Result<ValueKind> {
    let (node, p) = port_ref(endpoint)?;
    if node == "context" {
        if output && p == "value" && has_context_policy {
            return Ok(ValueKind::CommittedChatContext);
        }
        return Err(invalid());
    }
    if node == "input" {
        if output && p == "value" {
            return Ok(input);
        }
        return Err(invalid());
    }
    if node == "output" {
        if !output && p == "value" {
            return Ok(finish);
        }
        return Err(invalid());
    }
    let contract = steps.get(node).ok_or_else(unsupported)?;
    let ports = if output {
        contract.outputs
    } else {
        contract.inputs
    };
    ports
        .iter()
        .find(|port| port.name == p)
        .map(|port| port.value)
        .ok_or_else(unsupported)
}
impl Config {
    /// No network, disk or resource allocator access, no model admission and no
    /// runtime/API binding. Successful validation NEVER means executable.
    pub fn validate(&self) -> Result<VerifiedShape> {
        if self.models.len() > 128
            || self.pipelines.is_empty()
            || self.pipelines.len() > 64
            || self.routes.len() > 128
        {
            return Err(Error::Limit("pipeline model/graph/route count".into()));
        }
        let mut models = BTreeMap::new();
        for model in &self.models {
            if !id_ok(&model.id)
                || !field_ok(&model.target)
                || model
                    .source
                    .as_ref()
                    .is_some_and(|p| !field_ok(p) || p.contains("://"))
                || models.insert(model.id.as_str(), model).is_some()
            {
                return Err(invalid());
            }
            if find(&model.target).is_none() {
                return Err(Error::Unsupported("unregistered model target".into()));
            }
        }
        let mut compiled = Vec::new();
        let mut names = BTreeSet::new();
        let mut used_models = BTreeSet::new();
        let mut refs = 0usize;
        let mut context_configured_pipelines = 0usize;
        let mut context_enabled_pipelines = 0usize;
        for pipeline in &self.pipelines {
            if !id_ok(&pipeline.name)
                || !names.insert(pipeline.name.as_str())
                || pipeline.steps.is_empty()
                || pipeline.steps.len() > 128
                || pipeline.links.len() > 512
            {
                return Err(invalid());
            }
            if let Some(policy) = &pipeline.chat_context {
                if !id_ok(&policy.from_model)
                    || policy.max_bytes == 0
                    || policy.max_bytes > ChatContextPolicy::MAX_BYTES
                {
                    return Err(Error::Limit("chat context snapshot exceeds bounds".into()));
                }
                let source = models
                    .get(policy.from_model.as_str())
                    .ok_or_else(unsupported)?;
                let target = find(&source.target).ok_or_else(unsupported)?;
                // A context provider is a referenced LLM chat instance, not
                // just an unused entry whose future session might appear.
                // It may live in a second pipeline within this one process.
                if !self.pipelines.iter().any(|graph| {
                    graph.steps.iter().any(|step| {
                        step.kind == "llm"
                            && step.model.as_deref() == Some(policy.from_model.as_str())
                    })
                }) {
                    return Err(Error::Unsupported(
                        "chat context source has no configured LLM conversation stage".into(),
                    ));
                }
                if !pipeline
                    .links
                    .iter()
                    .any(|link| link.from == "context.value")
                {
                    return Err(Error::Malformed(
                        "chat context policy has no typed input consumers".into(),
                    ));
                }
                if !matches!(
                    target.task,
                    TaskKind::Language | TaskKind::TextImage | TaskKind::MultiModal
                ) {
                    return Err(Error::Unsupported(
                        "chat context must bind one declared text-capable model".into(),
                    ));
                }
                context_configured_pipelines += 1;
                context_enabled_pipelines += usize::from(policy.enabled);
            }
            let mut steps = BTreeMap::<&str, &ComponentContract>::new();
            let mut dependencies = BTreeMap::<&str, Vec<&str>>::new();
            let mut indegree = BTreeMap::<&str, usize>::new();
            for stage in &pipeline.steps {
                if !id_ok(&stage.id) || matches!(stage.id.as_str(), "input" | "output" | "context")
                {
                    return Err(invalid());
                }
                let contract = component(&stage.kind).ok_or_else(unsupported)?;
                if steps.insert(stage.id.as_str(), contract).is_some() {
                    return Err(invalid());
                }
                indegree.insert(stage.id.as_str(), 0);
                dependencies.insert(stage.id.as_str(), Vec::new());
                match (contract.binding, stage.model.as_deref()) {
                    (TaskBinding::None, None) => {}
                    (TaskBinding::None, Some(_)) | (_, None) => return Err(invalid()),
                    (task, Some(name)) => {
                        let model = models.get(name).ok_or_else(unsupported)?;
                        let target = find(&model.target).ok_or_else(unsupported)?;
                        let compatible = match task {
                            TaskBinding::Asr | TaskBinding::Diarization => {
                                target.task == TaskKind::SpeechRecognition
                            }
                            TaskBinding::Language => matches!(
                                target.task,
                                TaskKind::Language | TaskKind::TextImage | TaskKind::MultiModal
                            ),
                            TaskBinding::Speech => target.task == TaskKind::SpeechSynthesis,
                            TaskBinding::Image => target.task == TaskKind::ImageGeneration,
                            TaskBinding::None => false,
                        };
                        if !compatible {
                            return Err(unsupported());
                        }
                        used_models.insert(name);
                        refs += 1;
                    }
                }
            }
            let mut wired = BTreeSet::new();
            let mut reach_from = BTreeMap::<&str, Vec<&str>>::new();
            let mut reach_to = BTreeMap::<&str, Vec<&str>>::new();
            for link in &pipeline.links {
                let a = checked_value(
                    &link.from,
                    true,
                    &steps,
                    pipeline.input,
                    pipeline.output,
                    pipeline.chat_context.is_some(),
                )?;
                let b = checked_value(
                    &link.to,
                    false,
                    &steps,
                    pipeline.input,
                    pipeline.output,
                    pipeline.chat_context.is_some(),
                )?;
                if a != b {
                    return Err(Error::Unsupported(
                        "pipeline edge value types differ; explicit typed conversion required"
                            .into(),
                    ));
                }
                if !wired.insert(link.to.as_str()) {
                    return Err(Error::Malformed(
                        "pipeline input has multiple producers".into(),
                    ));
                }
                let (a_node, _) = port_ref(&link.from)?;
                let (b_node, _) = port_ref(&link.to)?;
                if a_node == "output"
                    || b_node == "input"
                    || a_node == b_node
                    || b_node == "context"
                {
                    return Err(invalid());
                }
                // The context source is a previously committed session
                // snapshot. Never add same-turn dependencies or a new model
                // execution step to obtain it.
                if a_node == "context" && b != ValueKind::CommittedChatContext {
                    return Err(invalid());
                }
                reach_from.entry(a_node).or_default().push(b_node);
                reach_to.entry(b_node).or_default().push(a_node);
                if a_node != "input" && a_node != "context" && b_node != "output" {
                    *indegree.get_mut(b_node).ok_or_else(unsupported)? += 1;
                    dependencies
                        .get_mut(a_node)
                        .ok_or_else(unsupported)?
                        .push(b_node);
                }
            }
            if !wired.contains("output.value") {
                return Err(invalid());
            }
            for stage in &pipeline.steps {
                for port in steps[stage.id.as_str()].inputs {
                    if !port.optional
                        && !wired.contains(format!("{}.{}", stage.id, port.name).as_str())
                    {
                        return Err(Error::Malformed(
                            "pipeline component required input is unconnected".into(),
                        ));
                    }
                }
            }
            let mut queue = VecDeque::from_iter(
                indegree
                    .iter()
                    .filter(|(_, degree)| **degree == 0)
                    .map(|(n, _)| *n),
            );
            let mut sorted = Vec::new();
            while let Some(next) = queue.pop_front() {
                sorted.push(next.to_owned());
                for child in dependencies[next].iter().copied() {
                    let pending = indegree.get_mut(child).ok_or_else(unsupported)?;
                    *pending -= 1;
                    if *pending == 0 {
                        queue.push_back(child);
                    }
                }
            }
            if sorted.len() != pipeline.steps.len() {
                return Err(Error::Unsupported("pipeline stage cycle".into()));
            }
            // No abandoned model stages: every registered stage must be on
            // some input-to-output path. Otherwise a config could reserve an
            // invisible side job and claim one coherent pipeline.
            fn reach<'a>(
                root: &'a str,
                graph: &BTreeMap<&'a str, Vec<&'a str>>,
            ) -> BTreeSet<&'a str> {
                let mut seen = BTreeSet::new();
                let mut queue = VecDeque::from([root]);
                while let Some(current) = queue.pop_front() {
                    if seen.insert(current) {
                        queue.extend(
                            graph
                                .get(current)
                                .into_iter()
                                .flat_map(|nodes| nodes.iter().copied()),
                        );
                    }
                }
                seen
            }
            let forwards = reach("input", &reach_from);
            let backwards = reach("output", &reach_to);
            if !forwards.contains("output")
                || pipeline.steps.iter().any(|stage| {
                    !forwards.contains(stage.id.as_str()) || !backwards.contains(stage.id.as_str())
                })
            {
                return Err(Error::Unsupported(
                    "pipeline has a disconnected or unproductive component".into(),
                ));
            }
            compiled.push((pipeline.name.clone(), sorted));
        }
        let mut routes = BTreeSet::new();
        for route in &self.routes {
            if !field_ok(&route.path)
                || !route.path.starts_with('/')
                || route.path.len() > 256
                || route.path.contains(['?', '#', '\\'])
                || route.path.chars().any(char::is_whitespace)
                || !names.contains(route.pipeline.as_str())
                || !matches!(
                    route.protocol.as_str(),
                    "openai-text" | "openai-asr" | "sse" | "websocket" | "webrtc"
                )
                || !routes.insert(route.path.as_str())
            {
                return Err(invalid());
            }
            let graph = self
                .pipelines
                .iter()
                .find(|p| p.name == route.pipeline)
                .ok_or_else(invalid)?;
            match route.protocol.as_str() {
                "openai-text" if graph.output != T => return Err(invalid()),
                "openai-asr" if !matches!(graph.output, TR | ST) => return Err(invalid()),
                "webrtc" if !graph.realtime => return Err(invalid()),
                _ => {}
            }
        }
        // Absence of routes is valid for a local CLI-only pipeline. The
        // transports above are declarations, not live listeners.
        Ok(VerifiedShape {
            ordered: compiled,
            shared_model_refs: refs.saturating_sub(used_models.len()),
            route_count: self.routes.len(),
            context_configured_pipelines,
            context_enabled_pipelines,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn text() -> Config {
        Config {
            models: vec![ModelInstance {
                id: "llm".into(),
                target: "qwen3.5-2b".into(),
                source: None,
            }],
            pipelines: vec![Pipeline {
                name: "chat".into(),
                input: T,
                output: T,
                realtime: false,
                chat_context: None,
                steps: vec![PipelineNode {
                    id: "answer".into(),
                    kind: "llm".into(),
                    model: Some("llm".into()),
                }],
                links: vec![
                    PipelineEdge {
                        from: "input.value".into(),
                        to: "answer.text".into(),
                    },
                    PipelineEdge {
                        from: "answer.text".into(),
                        to: "output.value".into(),
                    },
                ],
            }],
            routes: vec![ApiRoute {
                path: "/v1/chat".into(),
                protocol: "openai-text".into(),
                pipeline: "chat".into(),
            }],
        }
    }
    #[test]
    fn pure_single_model_graph_is_a_bounded_one_stage_pipeline_not_a_loaded_model() {
        let graph = text();
        let view = graph.validate().unwrap();
        assert_eq!(view.ordered[0].1, ["answer"]);
        assert_eq!(view.shared_model_refs, 0);
        assert_eq!(view.route_count, 1);
        assert!(
            find("qwen3.5-2b")
                .unwrap()
                .require_source_bound_execution()
                .is_err()
        );
    }
    #[test]
    fn multi_model_reuse_does_not_create_secondary_model_process_or_router() {
        let mut graph = text();
        let mut second = graph.pipelines[0].clone();
        second.name = "chat2".into();
        graph.pipelines.push(second);
        graph.routes.push(ApiRoute {
            path: "/v1/chat2".into(),
            protocol: "openai-text".into(),
            pipeline: "chat2".into(),
        });
        let result = graph.validate().unwrap();
        assert_eq!(result.shared_model_refs, 1);
        assert_eq!(result.ordered.len(), 2);
    }
    #[test]
    fn disconnected_ambiguous_or_ill_typed_graphs_refuse() {
        let mut graph = text();
        graph.pipelines[0].links.pop();
        assert!(graph.validate().is_err());
        let mut graph = text();
        graph.pipelines[0].links.push(PipelineEdge {
            from: "input.value".into(),
            to: "answer.text".into(),
        });
        assert!(graph.validate().is_err());
        let mut graph = text();
        graph.pipelines[0].output = A;
        assert!(graph.validate().is_err());
        let mut graph = text();
        graph.pipelines[0].steps.push(PipelineNode {
            id: "unused".into(),
            kind: "transcript-text".into(),
            model: None,
        });
        assert!(graph.validate().is_err());
        let mut graph = text();
        graph.routes.push(ApiRoute {
            path: "/v1/chat".into(),
            protocol: "websocket".into(),
            pipeline: "chat".into(),
        });
        assert!(graph.validate().is_err());
        let mut graph = text();
        graph.pipelines[0].steps[0].kind = "image-generate".into();
        assert!(graph.validate().is_err());
    }
}

#[cfg(test)]
mod context_tests {
    use super::*;

    fn contextual() -> Config {
        let graph = Config {
            models: vec![ModelInstance {
                id: "assistant".into(),
                target: "qwen3.5-2b".into(),
                source: None,
            }],
            pipelines: vec![
                Pipeline {
                    name: "chat".into(),
                    input: T,
                    output: T,
                    realtime: false,
                    chat_context: None,
                    steps: vec![PipelineNode {
                        id: "reply".into(),
                        kind: "llm".into(),
                        model: Some("assistant".into()),
                    }],
                    links: vec![
                        PipelineEdge {
                            from: "input.value".into(),
                            to: "reply.text".into(),
                        },
                        PipelineEdge {
                            from: "reply.text".into(),
                            to: "output.value".into(),
                        },
                    ],
                },
                Pipeline {
                    name: "contextual-conversion".into(),
                    input: ST,
                    output: T,
                    realtime: false,
                    chat_context: Some(ChatContextPolicy {
                        enabled: true,
                        from_model: "assistant".into(),
                        max_bytes: 4096,
                    }),
                    steps: vec![PipelineNode {
                        id: "format".into(),
                        kind: "speakers-text".into(),
                        model: None,
                    }],
                    links: vec![
                        PipelineEdge {
                            from: "input.value".into(),
                            to: "format.speakers".into(),
                        },
                        PipelineEdge {
                            from: "context.value".into(),
                            to: "format.context".into(),
                        },
                        PipelineEdge {
                            from: "format.text".into(),
                            to: "output.value".into(),
                        },
                    ],
                },
            ],
            routes: vec![],
        };
        assert_eq!(graph.validate().unwrap().context_enabled_pipelines, 1);
        graph
    }

    #[test]
    fn previous_turn_snapshot_may_enter_optional_context_ports_from_a_reused_chat_model() {
        let mut graph = contextual();
        let shape = graph.validate().unwrap();
        assert_eq!(shape.shared_model_refs, 0);
        assert_eq!(shape.context_configured_pipelines, 1);
        assert_eq!(shape.context_enabled_pipelines, 1);
        assert_eq!(shape.ordered[1].1, ["format"]);
        // Changing only the toggle preserves the declared port topology.
        graph.pipelines[1].chat_context.as_mut().unwrap().enabled = false;
        let disabled = graph.validate().unwrap();
        assert_eq!(disabled.context_configured_pipelines, 1);
        assert_eq!(disabled.context_enabled_pipelines, 0);
    }

    #[test]
    fn context_policy_must_bind_real_language_chat_stage_with_tight_byte_limits() {
        let mut graph = contextual();
        graph.pipelines[1].chat_context.as_mut().unwrap().max_bytes = 0;
        assert!(matches!(graph.validate(), Err(Error::Limit(_))));
        graph.pipelines[1].chat_context.as_mut().unwrap().max_bytes =
            ChatContextPolicy::MAX_BYTES + 1;
        assert!(matches!(graph.validate(), Err(Error::Limit(_))));
        let mut graph = contextual();
        graph.pipelines[1].chat_context.as_mut().unwrap().from_model = "unknown".into();
        assert!(graph.validate().is_err());
        let mut graph = contextual();
        graph.pipelines[0].steps[0].kind = "translate".into();
        assert!(graph.validate().is_err()); // source must have a real configured chat stage
        let mut graph = contextual();
        graph.pipelines[1]
            .links
            .retain(|l| l.from != "context.value");
        assert!(graph.validate().is_err()); // no silent enabled-but-unused toggle
    }

    #[test]
    fn context_cannot_be_spoofed_as_external_audio_or_same_turn_state() {
        let mut graph = contextual();
        graph.pipelines[1].chat_context = None;
        assert!(graph.validate().is_err()); // special context source requires policy
        let mut graph = contextual();
        graph.pipelines[1].links[1].from = "input.value".into();
        assert!(graph.validate().is_err()); // speaker transcript is not context
        let mut graph = contextual();
        graph.pipelines[1].links[1].from = "format.text".into();
        assert!(graph.validate().is_err()); // self-cycle/feedback cannot export chat state
        let mut graph = contextual();
        graph.pipelines[1].steps[0].id = "context".into();
        assert!(graph.validate().is_err()); // special source node is reserved
        assert!(ValueKind::parse("committed-chat-context").is_err());
    }
}
