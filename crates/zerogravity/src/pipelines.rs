//! Bounded v1/v2 pipeline YAML and typed structural checking.
//!
//! Successful checking is not execution eligibility: no models, context store,
//! devices or APIs are started. Explicit cold load/bind and loaded language
//! binding both use the original runtime lifecycle and task scheduler.
pub use zerogravity_engine::pipeline::{LanguageBinding, LanguageStage, PreparedLanguagePipeline};
mod document;
pub use crate::models::TextLoadPolicy as LanguageLoadPolicy;
mod patch;
pub use patch::{ChatContextPatch, PipelinePatch};
mod yaml;

pub use document::{
    ChatContextDoc, ContextFilterDoc, DiagnosticDoc, Document, EntryDoc, LinkDoc, ModelDoc,
    ModelEntry, PipelineDoc, PipelineEntry, RouteDoc, RunDoc, ServeDoc, StepDoc, TaskDoc,
};
pub use yaml::{MAX_CONFIG_BYTES, load_document, parse_document, read_document_bytes};
pub use zerogravity_core::pipeline::{
    ApiRoute, ChatContextPolicy, ComponentContract, ComponentRegistry, Config, ModelInstance,
    Pipeline, PipelineEdge, PipelineNode, Port, TaskBinding, ValueKind, VerifiedShape,
};

/// Structural report, not a loaded model, execution permit or source grant.
#[derive(Clone, Debug)]
pub struct CheckedDocument {
    pub config: Config,
    /// Parsed v2 options; pipeline checking does not resolve run precedence or
    /// validate execution settings such as backend/device/task eligibility.
    pub run: Option<RunDoc>,
    pub serve: Option<ServeDoc>,
    pub shape: VerifiedShape,
}

/// Resolve selected declarations only inside the original runtime load slot.
/// The engine commits the router and admitted plan together.
pub(crate) fn load_language(
    runtime: &zerogravity_engine::Runtime,
    config: &Config,
    name: &str,
    policy: LanguageLoadPolicy,
    cache_root: Option<&std::path::Path>,
) -> std::result::Result<PreparedLanguagePipeline, crate::runtime::text::ServiceError> {
    runtime.configure_language_pipeline(config, name, |models| {
        crate::builtins::load_declared_text(models, policy, cache_root, runtime.limits())
    })
}

pub fn check(config: &Config) -> crate::Result<VerifiedShape> {
    config.validate()
}

pub fn check_entrypoint(
    config: &Config,
    entrypoint: &crate::run_policy::Entrypoint,
) -> crate::Result<VerifiedShape> {
    config.validate_entrypoint(entrypoint)
}

/// Context-free compatibility entrypoint. Execution requires an explicit
/// runtime and policy via Runtime::load_pipeline or explicit loaded bindings
/// via prepare_pipeline, then run_pipeline. No implicit device/source selection.
pub fn run(_config: &Config) -> crate::Result<()> {
    Err(crate::Error::Unsupported(
        "compiled multi-model pipeline execution NOT IMPLEMENTED; no source or device was opened"
            .into(),
    ))
}
