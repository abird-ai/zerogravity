//! Typed whole-list overrides; absent fields inherit, present empty lists clear.
use super::{
    ApiRoute, ChatContextPolicy, Config, ModelInstance, Pipeline, PipelineEdge, PipelineNode,
    ValueKind,
};
use crate::context_filter::{ContextFilter, ContextFilterPlan, ContextSource, default_sources};
use crate::{Error, Result, run_policy::Entrypoint};

#[derive(Clone, Debug, Default)]
pub struct ChatContextPatch {
    pub from_model: Option<String>,
    pub enabled: Option<bool>,
    pub max_bytes: Option<usize>,
    pub sources: Option<Vec<ContextSource>>,
    pub stages: Option<Vec<ContextFilter>>,
}

/// Structural configuration only, never context access or model execution.
#[derive(Clone, Debug, Default)]
pub struct PipelinePatch {
    pub name: Option<String>,
    pub input: Option<ValueKind>,
    pub output: Option<ValueKind>,
    pub realtime: Option<bool>,
    pub models: Option<Vec<ModelInstance>>,
    pub steps: Option<Vec<PipelineNode>>,
    pub links: Option<Vec<PipelineEdge>>,
    pub routes: Option<Vec<ApiRoute>>,
    pub chat_context: Option<ChatContextPatch>,
}
impl PipelinePatch {
    /// Merge into a private configuration. The caller must validate the final
    /// graph/entrypoint; this does not load sources or grant execution.
    pub fn apply(self, graph: &mut Config, entry: &mut Option<Entrypoint>) -> Result<()> {
        let name = self
            .name
            .or_else(|| match entry {
                Some(Entrypoint::Pipeline(name)) => Some(name.clone()),
                _ => None,
            })
            .ok_or_else(|| bad("graph overrides require a name or pipeline entrypoint"))?;
        match entry {
            Some(Entrypoint::Model(_)) => {
                return Err(bad("graph flags contradict a direct model entrypoint"));
            }
            Some(Entrypoint::Pipeline(selected)) if selected != &name => {
                return Err(bad("--name contradicts selected pipeline entrypoint"));
            }
            _ => {}
        }
        let old = graph.pipelines.iter().find(|p| p.name == name);
        let input = self
            .input
            .or_else(|| old.map(|p| p.input))
            .ok_or_else(|| bad("graph input required"))?;
        let output = self
            .output
            .or_else(|| old.map(|p| p.output))
            .ok_or_else(|| bad("graph output required"))?;
        let previous_context = old.and_then(|p| p.chat_context.clone());
        let chat_context = match self.chat_context {
            None => previous_context,
            Some(patch) => {
                let previous = previous_context.as_ref();
                Some(ChatContextPolicy {
                    from_model: patch
                        .from_model
                        .or_else(|| previous.map(|p| p.from_model.clone()))
                        .ok_or_else(|| bad("chat context overrides require a source model"))?,
                    enabled: patch
                        .enabled
                        .or_else(|| previous.map(|p| p.enabled))
                        .unwrap_or(true),
                    max_bytes: patch
                        .max_bytes
                        .or_else(|| previous.map(|p| p.max_bytes))
                        .unwrap_or(ChatContextPolicy::DEFAULT_BYTES),
                    selection: ContextFilterPlan {
                        sources: patch
                            .sources
                            .or_else(|| previous.map(|p| p.selection.sources.clone()))
                            .unwrap_or_else(default_sources),
                        stages: patch
                            .stages
                            .or_else(|| previous.map(|p| p.selection.stages.clone()))
                            .unwrap_or_default(),
                    },
                })
            }
        };
        let new = Pipeline {
            name: name.clone(),
            input,
            output,
            realtime: self
                .realtime
                .or_else(|| old.map(|p| p.realtime))
                .unwrap_or(false),
            steps: self
                .steps
                .or_else(|| old.map(|p| p.steps.clone()))
                .unwrap_or_default(),
            links: self
                .links
                .or_else(|| old.map(|p| p.links.clone()))
                .unwrap_or_default(),
            chat_context,
        };
        // No fallible work after replacing any part of the caller's graph.
        if let Some(models) = self.models {
            graph.models = models;
        }
        if let Some(mut routes) = self.routes {
            for route in &mut routes {
                route.pipeline.clone_from(&name);
            }
            graph.routes = routes;
        }
        if let Some(i) = graph.pipelines.iter().position(|p| p.name == name) {
            graph.pipelines[i] = new;
        } else {
            graph.pipelines.push(new);
        }
        *entry = Some(Entrypoint::Pipeline(name));
        Ok(())
    }

    /// Construct declarations only; use `pipelines::check` for validation.
    pub fn compile(self) -> Result<Config> {
        let mut config = Config {
            models: Vec::new(),
            pipelines: Vec::new(),
            routes: Vec::new(),
        };
        self.apply(&mut config, &mut None)?;
        Ok(config)
    }
}
fn bad(message: &str) -> Error {
    Error::Malformed(message.into())
}
