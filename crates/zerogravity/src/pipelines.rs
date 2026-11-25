//! Bounded v1/v2 pipeline YAML and typed structural checking.
//!
//! Successful checking is not execution eligibility: no models, context store,
//! devices or APIs are started. Explicit loaded language binding is separate.
pub use zerogravity_engine::pipeline::{LanguageBinding, LanguageStage, PreparedLanguagePipeline};
mod document;
mod patch;
pub use patch::{ChatContextPatch, PipelinePatch};
mod yaml;

pub use document::{
    ChatContextDoc, ContextFilterDoc, DiagnosticDoc, Document, EntryDoc, LinkDoc, ModelDoc,
    ModelEntry, PipelineDoc, PipelineEntry, RouteDoc, RunDoc, StepDoc, TaskDoc,
};
pub use yaml::{MAX_CONFIG_BYTES, load_document, parse_document};
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
    pub shape: VerifiedShape,
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
/// loaded runtime and bindings via Runtime::prepare_pipeline/run_pipeline.
/// No automatic source loading, device selection, listener or terminal IO.
pub fn run(_config: &Config) -> crate::Result<()> {
    Err(crate::Error::Unsupported(
        "compiled multi-model pipeline execution NOT IMPLEMENTED; no source or device was opened"
            .into(),
    ))
}
