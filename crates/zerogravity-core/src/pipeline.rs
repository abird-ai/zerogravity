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
use crate::context_filter::{ContextFilter, ContextFilterPlan};
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
    AudioFeatures,
    AudioTokens,
    ImageLatents,
    VideoFrames,
    DecisionVerdict,
    /// Bounded immutable, provenance-labelled context records from explicitly
    /// requested and authorized sources. The default is only prior
    /// same-session user/assistant turns; app prompts, tools and authorized
    /// shared data may be opted in. Never a raw mutable KV pointer.
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
            Self::AudioFeatures => "audio-features",
            Self::AudioTokens => "audio-tokens",
            Self::ImageLatents => "image-latents",
            Self::VideoFrames => "video-frames",
            Self::DecisionVerdict => "decision-verdict",
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
            "audio-features" => Ok(Self::AudioFeatures),
            "audio-tokens" => Ok(Self::AudioTokens),
            "image-latents" => Ok(Self::ImageLatents),
            "video-frames" => Ok(Self::VideoFrames),
            "decision-verdict" => Ok(Self::DecisionVerdict),
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
    Multimodal,
    Decision,
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
    AudioFeatures as AF, AudioPcm as A, AudioTokens as AT, CommittedChatContext as C,
    DecisionVerdict as DV, ImageLatents as IL, ImageRgba as I, SpeakerTranscript as ST, Text as T,
    TimedTranscript as TR, VideoFrames as VF,
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
    // Future multi-component media scaffolds. They are typed structural
    // operators only. There are no codecs, solvers or neural drivers here.
    ComponentContract {
        kind: "audio-feature-extract",
        binding: TaskBinding::None,
        inputs: &[port("audio", A)],
        outputs: &[port("features", AF)],
    },
    ComponentContract {
        kind: "audio-tokenize",
        binding: TaskBinding::None,
        inputs: &[port("audio", A)],
        outputs: &[port("tokens", AT)],
    },
    ComponentContract {
        kind: "audio-detokenize",
        binding: TaskBinding::None,
        inputs: &[port("tokens", AT)],
        outputs: &[port("audio", A)],
    },
    ComponentContract {
        kind: "image-solve",
        binding: TaskBinding::Image,
        inputs: &[port("prompt", T)],
        outputs: &[port("latents", IL)],
    },
    ComponentContract {
        kind: "image-decode",
        binding: TaskBinding::None,
        inputs: &[port("latents", IL)],
        outputs: &[port("image", I)],
    },
    ComponentContract {
        kind: "video-generate",
        binding: TaskBinding::Multimodal,
        inputs: &[port("prompt", T)],
        outputs: &[port("frames", VF)],
    },
    ComponentContract {
        kind: "decision",
        binding: TaskBinding::Decision,
        inputs: &[port("text", T)],
        outputs: &[port("verdict", DV)],
    },
];
pub fn component(kind: &str) -> Option<&'static ComponentContract> {
    COMPONENTS.iter().find(|c| c.kind == kind)
}

/// Source-neutral extension registry for *typed contract descriptions*.
/// It does not register executable model code, start a server, grant context
/// access, load weights or allow unqualified new tensor/codec semantics.
/// An application can add another statically defined port contract and then
/// use Config::validate_with_registry. Separate model-specific TaskDrivers,
/// source-bound implementations and scheduler admission remain required.
#[derive(Default)]
pub struct ComponentRegistry {
    extensions: BTreeMap<&'static str, ComponentContract>,
}
impl ComponentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, descriptor: ComponentContract) -> Result<()> {
        if !id_ok(descriptor.kind)
            || component(descriptor.kind).is_some()
            || self.extensions.contains_key(descriptor.kind)
        {
            return Err(Error::Malformed(
                "duplicate or malformed component registration".into(),
            ));
        }
        if descriptor.inputs.is_empty()
            || descriptor.outputs.is_empty()
            || descriptor.inputs.len() > 16
            || descriptor.outputs.len() > 16
        {
            return Err(Error::Limit("component port count".into()));
        }
        let mut inputs = BTreeSet::new();
        let mut outputs = BTreeSet::new();
        for input in descriptor.inputs {
            if !id_ok(input.name)
                || !inputs.insert(input.name)
                || input.value == ValueKind::CommittedChatContext && !input.optional
            {
                return Err(Error::Malformed(
                    "invalid typed component input port".into(),
                ));
            }
        }
        for output in descriptor.outputs {
            if !id_ok(output.name)
                || !outputs.insert(output.name)
                || output.optional
                || output.value == ValueKind::CommittedChatContext
            {
                return Err(Error::Malformed(
                    "invalid typed component output port".into(),
                ));
            }
        }
        if self.extensions.len() >= 256 {
            return Err(Error::Limit("registered extension component count".into()));
        }
        self.extensions.insert(descriptor.kind, descriptor);
        Ok(())
    }

    pub fn resolve(&self, kind: &str) -> Option<&ComponentContract> {
        component(kind).or_else(|| self.extensions.get(kind))
    }
}

#[derive(Clone, Debug)]
pub struct ModelInstance {
    pub id: String,
    pub target: String,
    /// Not fetched, opened or granted authority by graph validation.
    pub source: Option<String>,
    pub model_config: Option<String>,
    pub tokenizer: Option<String>,
    pub snapshot: Option<String>,
    pub native_artifacts: Option<String>,
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
    /// Opt-in typed context selection/filter pipeline, never a current-turn
    /// feedback loop. If disabled, context channels carry None.
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
    /// Versioned source selectors and ordered generic include/exclude /
    /// decision steps. Configuration requests authority; it grants none.
    pub selection: ContextFilterPlan,
}
impl ChatContextPolicy {
    /// Configurable input capacity is not a hardcoded 16KiB context restriction.
    /// Actual runtime admission may impose lower per-session quotas.
    pub const MAX_BYTES: usize = 1024 * 1024;
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
    pub context_filter_stages: usize,
    pub context_decision_stages: usize,
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
    /// Validate a direct request as the same canonical single-model region,
    /// while still checking every declared graph/route and instance reference.
    pub fn validate_entrypoint(
        &self,
        entry: &crate::run_policy::Entrypoint,
    ) -> Result<VerifiedShape> {
        match entry {
            crate::run_policy::Entrypoint::Pipeline(name) => {
                if !self.pipelines.iter().any(|p| p.name == *name) {
                    return Err(Error::Malformed(
                        "selected pipeline entrypoint is absent".into(),
                    ));
                }
                self.validate()
            }
            crate::run_policy::Entrypoint::Model(id) => {
                if !self.models.iter().any(|m| m.id == *id) {
                    return Err(Error::Malformed(
                        "selected model entrypoint is absent".into(),
                    ));
                }
                let mut graph = self.clone();
                let mut name = "run-direct-entry".to_string();
                while graph.pipelines.iter().any(|p| p.name == name) {
                    name.push('x');
                }
                graph.pipelines.push(Pipeline {
                    name,
                    input: ValueKind::Text,
                    output: ValueKind::Text,
                    realtime: false,
                    chat_context: None,
                    steps: vec![PipelineNode {
                        id: "generate".into(),
                        kind: "llm".into(),
                        model: Some(id.clone()),
                    }],
                    links: vec![
                        PipelineEdge {
                            from: "input.value".into(),
                            to: "generate.text".into(),
                        },
                        PipelineEdge {
                            from: "generate.text".into(),
                            to: "output.value".into(),
                        },
                    ],
                });
                graph.validate()
            }
        }
    }
    /// No network, disk or resource allocator access, no model admission and no
    /// runtime/API binding. Successful validation NEVER means executable.
    pub fn validate(&self) -> Result<VerifiedShape> {
        self.validate_with_registry(&ComponentRegistry::default())
    }

    /// Validate under a caller-supplied finite component schema registry.
    /// Validation is still pure; registration is not execution eligibility.
    pub fn validate_with_registry(&self, registry: &ComponentRegistry) -> Result<VerifiedShape> {
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
                || model.source.as_ref().is_some_and(|p| {
                    !field_ok(p) || crate::run_policy::SourceSelector::parse(p).is_err()
                })
                || model.model_config.as_ref().is_some_and(|p| !field_ok(p))
                || model.tokenizer.as_ref().is_some_and(|p| !field_ok(p))
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
        let mut context_filter_stages = 0usize;
        let mut context_decision_stages = 0usize;
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
                policy.selection.validate()?;
                context_filter_stages += policy.selection.stages.len();
                for stage in &policy.selection.stages {
                    if let ContextFilter::Decision { model, .. } = stage {
                        let instance = models.get(model.as_str()).ok_or_else(unsupported)?;
                        let decision = find(&instance.target).ok_or_else(unsupported)?;
                        if decision.task != TaskKind::Decision {
                            return Err(Error::Unsupported("context decisions require a registered typed decision model instance".into()));
                        }
                        context_decision_stages += 1;
                    }
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
                let contract = registry.resolve(&stage.kind).ok_or_else(unsupported)?;
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
                            TaskBinding::Multimodal => target.task == TaskKind::MultiModal,
                            TaskBinding::Decision => target.task == TaskKind::Decision,
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
            context_filter_stages,
            context_decision_stages,
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
                model_config: None,
                tokenizer: None,
                snapshot: None,
                native_artifacts: None,
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
                model_config: None,
                tokenizer: None,
                snapshot: None,
                native_artifacts: None,
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
                        selection: ContextFilterPlan::default(),
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

#[cfg(test)]
mod extension_tests {
    use super::*;
    fn graph(model: &str, target: &str, kind: &str, to: &str, output: ValueKind) -> Config {
        Config {
            models: vec![ModelInstance {
                id: model.into(),
                target: target.into(),
                source: None,
                model_config: None,
                tokenizer: None,
                snapshot: None,
                native_artifacts: None,
            }],
            pipelines: vec![Pipeline {
                name: "specification".into(),
                input: T,
                output,
                realtime: false,
                chat_context: None,
                steps: vec![PipelineNode {
                    id: "component".into(),
                    kind: kind.into(),
                    model: Some(model.into()),
                }],
                links: vec![
                    PipelineEdge {
                        from: "input.value".into(),
                        to: "component.prompt".into(),
                    },
                    PipelineEdge {
                        from: format!("component.{to}"),
                        to: "output.value".into(),
                    },
                ],
            }],
            routes: vec![],
        }
    }
    #[test]
    fn future_image_video_and_typed_decision_components_validate_without_execution() {
        for (model, target, kind, output_port, output_type) in [
            ("image", "qwen-image-2.1", "image-solve", "latents", IL),
            ("video", "wan-video", "video-generate", "frames", VF),
        ] {
            let config = graph(model, target, kind, output_port, output_type);
            assert!(config.validate().is_ok());
            assert!(
                find(target)
                    .unwrap()
                    .require_source_bound_execution()
                    .is_err()
            );
        }
        let mut decision = graph("gate", "nanojev-decision", "decision", "verdict", DV);
        decision.pipelines[0].links[0].to = "component.text".into();
        assert!(decision.validate().is_ok());
        assert!(
            find("nanojev-decision")
                .unwrap()
                .require_source_bound_execution()
                .is_err()
        );
        decision.models[0].target = "qwen3.5-2b".into();
        assert!(decision.validate().is_err()); // no model-family coercion
    }
    #[test]
    fn audio_codec_structure_requires_typed_token_intermediates() {
        let mut config = graph("image", "qwen-image-2.1", "image-solve", "latents", IL);
        config.models.clear();
        config.pipelines[0].input = A;
        config.pipelines[0].output = A;
        config.pipelines[0].steps = vec![
            PipelineNode {
                id: "encode".into(),
                kind: "audio-tokenize".into(),
                model: None,
            },
            PipelineNode {
                id: "decode".into(),
                kind: "audio-detokenize".into(),
                model: None,
            },
        ];
        config.pipelines[0].links = vec![
            PipelineEdge {
                from: "input.value".into(),
                to: "encode.audio".into(),
            },
            PipelineEdge {
                from: "encode.tokens".into(),
                to: "decode.tokens".into(),
            },
            PipelineEdge {
                from: "decode.audio".into(),
                to: "output.value".into(),
            },
        ];
        assert_eq!(
            config.validate().unwrap().ordered[0].1,
            ["encode", "decode"]
        );
        config.pipelines[0].links[1].from = "input.value".into();
        assert!(config.validate().is_err());
    }
    #[test]
    fn safe_extension_registry_accepts_only_distinct_bounded_static_port_contracts() {
        static INPUTS: &[Port] = &[port("text", T)];
        static OUTPUTS: &[Port] = &[port("text", T)];
        static BAD_CTX_OUTPUT: &[Port] = &[port("context", C)];
        static BAD_CTX_INPUT: &[Port] = &[port("context", C)];
        static BAD_OPTIONAL_OUTPUT: &[Port] = &[optional_port("value", T)];
        static DUPLICATE_INPUT: &[Port] = &[port("text", T), port("text", T)];
        let registered = ComponentContract {
            kind: "my-normalizer",
            binding: TaskBinding::None,
            inputs: INPUTS,
            outputs: OUTPUTS,
        };
        let mut registry = ComponentRegistry::new();
        registry.register(registered).unwrap();
        assert!(registry.register(registered).is_err());
        assert!(
            registry
                .register(ComponentContract {
                    kind: "llm",
                    ..registered
                })
                .is_err()
        );
        assert!(
            registry
                .register(ComponentContract {
                    kind: "unsafe..kind",
                    ..registered
                })
                .is_err()
        );
        assert!(
            registry
                .register(ComponentContract {
                    kind: "source-context",
                    outputs: BAD_CTX_OUTPUT,
                    ..registered
                })
                .is_err()
        );
        assert!(
            registry
                .register(ComponentContract {
                    kind: "source-context-input",
                    inputs: BAD_CTX_INPUT,
                    ..registered
                })
                .is_err()
        );
        assert!(
            registry
                .register(ComponentContract {
                    kind: "optional-output",
                    outputs: BAD_OPTIONAL_OUTPUT,
                    ..registered
                })
                .is_err()
        );
        assert!(
            registry
                .register(ComponentContract {
                    kind: "duplicate-input",
                    inputs: DUPLICATE_INPUT,
                    ..registered
                })
                .is_err()
        );
        assert!(
            registry
                .register(ComponentContract {
                    kind: "no-input",
                    inputs: &[],
                    ..registered
                })
                .is_err()
        );
        assert!(
            registry
                .register(ComponentContract {
                    kind: "no-output",
                    outputs: &[],
                    ..registered
                })
                .is_err()
        );
        let mut config = graph("answer", "qwen3.5-2b", "my-normalizer", "text", T);
        config.pipelines[0].links[0].to = "component.text".into();
        config.pipelines[0].steps[0].model = None;
        assert!(config.validate().is_err()); // not silently added to built-ins
        assert_eq!(
            config.validate_with_registry(&registry).unwrap().ordered[0].1,
            ["component"]
        );
        // Registered type is a schema extension ONLY; no executable backend.
        assert!(
            find("qwen3.5-2b")
                .unwrap()
                .require_source_bound_execution()
                .is_err()
        );
    }
}
