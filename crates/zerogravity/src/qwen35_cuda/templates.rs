//! Model-owned finite calls and their distinct physical layouts.
use super::*;
use crate::gpu::SourceBindings;
use crate::gpu::cuda::gdn::GdnKernelRecipe;
use crate::qwen35::plan::{PromptSpan, PromptWorkload, QwenSourceProfile};
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
    fn metadata_bytes(profile: QwenSourceProfile) -> Result<usize> {
        // Include Vec/Arc/call wrappers and transient scope/ID metadata. Graphs
        // retain separate charges; keep the accepted64KiB minimum allowance.
        let calls = profile.prepared_prompt_call_count()?.checked_mul(
            std::mem::size_of::<Template>()
                .checked_add(1024)
                .ok_or_else(|| Error::Limit("prompt template wrapper extent".into()))?,
        );
        let scopes = profile.physical_prompt_scope_count()?.checked_mul(
            std::mem::size_of::<(Arc<SourceBindings>, GdnKernelRecipe)>()
                .checked_add(1024)
                .ok_or_else(|| Error::Limit("prompt scope wrapper extent".into()))?,
        );
        calls
            .and_then(|calls| calls.checked_add(scopes?))
            .and_then(|bytes| bytes.checked_add(4096))
            .and_then(|bytes| bytes.checked_add(64 * 1024 - 1))
            .and_then(|bytes| (bytes / (64 * 1024)).max(1).checked_mul(64 * 1024))
            .ok_or_else(|| Error::Limit("profile prompt template metadata extent".into()))
    }
    pub(super) fn prepare(source: BoundFullModelSource) -> Result<Self> {
        // Admit wrappers before their vectors; each graph retains its own charge.
        let profile = source.model_profile();
        let metadata = source
            .source()
            .source()
            .reserve_workspace(Self::metadata_bytes(profile)?)?;
        let expected = profile.prepared_prompt_call_count()?;
        let mut calls = Vec::with_capacity(expected);
        for &count in profile.prompt_token_counts() {
            let workload = profile.prompt_workload(count)?;
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
        if calls.len() != expected {
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
        let profile = self.source.model_profile();
        let expected = profile.physical_prompt_scope_count()?;
        let mut scopes = Vec::with_capacity(expected);
        for &count in profile.prompt_token_counts() {
            let template = self.find(profile.prompt_workload(count)?, None)?;
            scopes.push((template.binding.clone(), template.recipe));
        }
        let first = *profile
            .prompt_token_counts()
            .first()
            .ok_or_else(|| Error::State("profile has no prompt physical scope".into()))?;
        let cached = self.find(profile.prompt_workload(first)?, Some(first))?;
        scopes.push((cached.binding.clone(), cached.recipe));
        if scopes.len() != expected {
            return Err(Error::State(
                "profile physical prompt scope coverage differs".into(),
            ));
        }
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
        let profile = self.source.model_profile();
        profile.validate_prompt_request(ids, 1)?;
        let workload = profile.prompt_workload(ids.len())?;
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
        let workload = self.source.model_profile().prompt_workload(prompt_tokens)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_profiles_keep_the_original_template_allowance() {
        for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
            assert_eq!(Templates::metadata_bytes(profile).unwrap(), 64 * 1024);
            assert_eq!(profile.prepared_prompt_call_count().unwrap(), 40);
            assert_eq!(profile.physical_prompt_scope_count().unwrap(), 6);
        }
    }
}
