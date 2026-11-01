//! Typed v1/v2 configuration declarations. Parsing never grants execution.
use serde::Deserialize;

use super::{
    ApiRoute, ChatContextPolicy, Config, ModelInstance, Pipeline, PipelineEdge, PipelineNode,
    ValueKind,
};
use crate::context_filter::{
    ContextFilter, ContextFilterPlan, ContextSource, DecisionUnavailable, FilterAction, MatchField,
    default_sources,
};
use crate::run_policy::Entrypoint;
use crate::{Error, Result};

const SCHEMA: &str = "zerogravity.pipeline.v1";

/// YAML declarations before graph compilation or entrypoint validation.
///
/// Paths and source selectors remain verbatim. Callers choosing a config base
/// directory may resolve them before compilation; no source is opened here.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub schema: String,
    #[serde(default)]
    pub model: Vec<ModelDoc>,
    #[serde(default)]
    pub pipeline: Vec<PipelineDoc>,
    #[serde(default)]
    pub route: Vec<RouteDoc>,
    #[serde(default)]
    pub run: Option<RunDoc>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDoc {
    pub id: String,
    pub target: String,
    pub source: Option<String>,
    pub model_config: Option<String>,
    pub tokenizer: Option<String>,
    pub snapshot: Option<String>,
    pub native_artifacts: Option<String>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineDoc {
    pub name: String,
    pub input: String,
    pub output: String,
    #[serde(default)]
    pub realtime: bool,
    /// Absent = off. Present = explicit same-session prior committed context.
    pub chat_context: Option<ChatContextDoc>,
    #[serde(default)]
    pub step: Vec<StepDoc>,
    #[serde(default)]
    pub link: Vec<LinkDoc>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatContextDoc {
    pub enabled: bool,
    pub from_model: String,
    pub max_bytes: Option<usize>,
    pub sources: Option<Vec<String>>,
    #[serde(default)]
    pub filter: Vec<ContextFilterDoc>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContextFilterDoc {
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
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepDoc {
    pub id: String,
    pub kind: String,
    pub model: Option<String>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkDoc {
    pub from: String,
    pub to: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteDoc {
    pub path: String,
    pub protocol: String,
    pub pipeline: String,
}

/// Optional v2 run settings. Semantic run-policy validation and CLI precedence
/// are separate from pipeline structural validation.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunDoc {
    pub entrypoint: Option<EntryDoc>,
    pub backend: Option<String>,
    pub device: Option<Vec<String>>,
    pub headroom: Option<String>,
    pub task: Option<TaskDoc>,
    pub numerical: Option<String>,
    pub speculation: Option<String>,
    pub draft_device: Option<Vec<String>>,
    pub diagnostics: Option<DiagnosticDoc>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum EntryDoc {
    Model(ModelEntry),
    Pipeline(PipelineEntry),
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    pub model: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineEntry {
    pub pipeline: String,
}
impl EntryDoc {
    pub fn typed(&self) -> Entrypoint {
        match self {
            Self::Model(v) => Entrypoint::Model(v.model.clone()),
            Self::Pipeline(v) => Entrypoint::Pipeline(v.pipeline.clone()),
        }
    }
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskDoc {
    pub prompt: Option<String>,
    pub context: Option<usize>,
    pub max_tokens: Option<usize>,
    pub raw: Option<bool>,
    pub teacher_ids: Option<Vec<u32>>,
    pub prefill_chunk: Option<usize>,
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticDoc {
    pub explain: Option<bool>,
    pub dry_run: Option<bool>,
    pub audit_logits_jsonl: Option<String>,
    pub audit_layers_jsonl: Option<String>,
}

impl Document {
    /// Compile declarations into the core graph. Use `check` for graph and
    /// selected-entrypoint validation; compilation alone is not validation.
    pub fn compile(self) -> Result<Config> {
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
                    // Only a direct source request may infer its target from
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

    /// Check every declared graph and route, including the optional v2
    /// entrypoint. No model, source, device, context store or listener is opened.
    pub fn check(self) -> Result<super::CheckedDocument> {
        // Preserve the v1 run-field refusal in compile before extracting run.
        let run = self.run.clone();
        let config = self.compile()?;
        let shape = match run.as_ref().and_then(|r| r.entrypoint.as_ref()) {
            Some(entry) => super::check_entrypoint(&config, &entry.typed())?,
            None => super::check(&config)?,
        };
        Ok(super::CheckedDocument { config, run, shape })
    }
}
