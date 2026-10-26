//! Model-owned finite calls; six physical layouts share forty semantic templates.
use super::*;
use crate::gpu::SourceBindings;
use crate::gpu::cuda::gdn::GdnKernelRecipe;
use crate::qwen35::plan::{PromptSpan, PromptWorkload};
use crate::qwen35::source::binding::{BoundFullModelSource, GdnKernelCapture};
use zerogravity_artifacts::source_weights::WindowWorkspace;

pub(super) struct Templates {
    source: BoundFullModelSource,
    calls: Vec<Template>,
    _metadata: WindowWorkspace,
}
struct Template {
    workload: PromptWorkload,
    base: Option<usize>,
    binding: Arc<SourceBindings>,
    recipe: GdnKernelRecipe,
}
impl Templates {
    pub(super) fn prepare(source: BoundFullModelSource) -> Result<Self> {
        // Admit wrappers before their vectors; each graph retains its own charge.
        let metadata = source.source().source().reserve_workspace(64 * 1024)?;
        let mut calls = Vec::with_capacity(40);
        for count in [47, 57, 62, 67, 72] {
            let workload = PromptWorkload::from_prompt_tokens(count)?;
            for base in std::iter::once(None).chain(workload.cached_bases().map(Some)) {
                let span = match base {
                    None => PromptSpan::prefill(
                        source.source().source_identity(),
                        workload,
                        &vec![0; count],
                    )?,
                    Some(base) => {
                        PromptSpan::cached(source.source().source_identity(), workload, base, 0)?
                    }
                };
                let call = source.bind_prompt_span(span)?;
                let recipe = bindings::gdn_recipe(GdnKernelCapture::for_call(call.call())?)?;
                let record = call.record_prepared()?;
                let prepared = call.prepare_program(record)?;
                let binding = bindings::prepare(source.source(), prepared, None)?;
                calls.push(Template {
                    workload,
                    base,
                    binding,
                    recipe,
                });
            }
        }
        if calls.len() != 40 {
            return Err(Error::State(
                "text semantic template coverage differs".into(),
            ));
        }
        Ok(Self {
            source,
            calls,
            _metadata: metadata,
        })
    }
    pub(super) fn source(&self) -> &BoundFullModelSource {
        &self.source
    }
    pub(super) fn physical_scopes(&self) -> Result<Vec<(Arc<SourceBindings>, GdnKernelRecipe)>> {
        let mut scopes = Vec::with_capacity(6);
        for count in [47, 57, 62, 67, 72] {
            let template = self.find(PromptWorkload::from_prompt_tokens(count)?, None)?;
            scopes.push((template.binding.clone(), template.recipe));
        }
        let cached = self.find(PromptWorkload::from_prompt_tokens(47)?, Some(47))?;
        scopes.push((cached.binding.clone(), cached.recipe));
        Ok(scopes)
    }
    fn find(&self, workload: PromptWorkload, base: Option<usize>) -> Result<&Template> {
        self.calls
            .iter()
            .find(|call| call.workload == workload && call.base == base)
            .ok_or_else(|| Error::Unsupported("text exact semantic template absent".into()))
    }
    fn admit_call(&self, workload: PromptWorkload, base: Option<usize>) -> Result<WindowWorkspace> {
        let bytes = self
            .find(workload, base)?
            .binding
            .rebind_metadata_bytes()?
            .checked_add(4096)
            .ok_or_else(|| Error::Limit("text call metadata extent".into()))?;
        self.source.source().source().reserve_workspace(bytes)
    }
    pub(super) fn bind_prompt(
        &self,
        ids: &[u32],
    ) -> Result<(Arc<SourceBindings>, GdnKernelRecipe)> {
        PromptWorkload::validate_request(ids, 1)?;
        let workload = PromptWorkload::from_prompt_tokens(ids.len())?;
        let metadata = self.admit_call(workload, None)?;
        let span = PromptSpan::prefill(self.source.source().source_identity(), workload, ids)?;
        self.bind(span, metadata)
    }
    pub(super) fn bind_cached_metadata(
        &self,
        prompt_tokens: usize,
        base: usize,
        id: u32,
    ) -> Result<(Arc<SourceBindings>, GdnKernelRecipe)> {
        let workload = PromptWorkload::from_prompt_tokens(prompt_tokens)?;
        let metadata = self.admit_call(workload, Some(base))?;
        let span = PromptSpan::cached(self.source.source().source_identity(), workload, base, id)?;
        self.bind(span, metadata)
    }
    fn bind(
        &self,
        span: PromptSpan,
        metadata: WindowWorkspace,
    ) -> Result<(Arc<SourceBindings>, GdnKernelRecipe)> {
        span.validate()?;
        let base = match span.mode() {
            crate::accelerator::program::SourceCallMode::Prefill => None,
            crate::accelerator::program::SourceCallMode::Decode => Some(span.base()),
        };
        let template = self.find(span.workload(), base)?;
        let call = self.source.bind_prompt_span(span)?;
        let recipe = bindings::gdn_recipe(GdnKernelCapture::for_call(call.call())?)?;
        if recipe != template.recipe {
            return Err(Error::State(
                "text template reference capture differs".into(),
            ));
        }
        // The model validates workload/profile; core retains exact layout and a
        // fresh SourceCall issuer. GPU separately checks genuine feedback/root.
        let binding = template
            .binding
            .rebind(call.call().request().source_call()?, metadata)?;
        Ok((Arc::new(binding), recipe))
    }
}
