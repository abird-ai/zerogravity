//! Lazy model calls and exact backend implementation binding.
use super::*;
use crate::gpu::SourceBindings;
use crate::gpu::cuda::gdn::GdnKernelRecipe;
use crate::qwen35::plan::{PromptSpan, PromptWorkload};
use crate::qwen35::source::binding::BoundFullModelSource;
use std::collections::VecDeque;
use std::sync::Mutex;
use zerogravity_artifacts::source_weights::WindowWorkspace;

const PREPARED_CAPACITY: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImplementationPolicy {
    Ordinary,
    Qualification,
}

pub(super) struct Templates {
    source: BoundFullModelSource,
    context_capacity_tokens: usize,
    calls: Mutex<VecDeque<Template>>,
    policy: ImplementationPolicy,
    _metadata: WindowWorkspace,
}
struct Template {
    workload: PromptWorkload,
    base: Option<usize>,
    binding: Arc<SourceBindings>,
    recipe: GdnKernelRecipe,
}
impl Templates {
    fn metadata_bytes() -> Result<usize> {
        // Bounded retained wrappers only. Graph/source charges are owned by each
        // retained prepared entry and survive independently while referenced.
        let entries = PREPARED_CAPACITY
            .checked_mul(
                std::mem::size_of::<Template>()
                    .checked_add(1024)
                    .ok_or_else(|| Error::Limit("prompt template wrapper extent".into()))?,
            )
            .ok_or_else(|| Error::Limit("prompt template cache extent".into()))?;
        entries
            .checked_add(std::mem::size_of::<Self>() + 4096)
            .and_then(|bytes| bytes.checked_add(64 * 1024 - 1))
            .and_then(|bytes| (bytes / (64 * 1024)).max(1).checked_mul(64 * 1024))
            .ok_or_else(|| Error::Limit("prompt template metadata extent".into()))
    }
    pub(super) fn prepare(
        source: BoundFullModelSource,
        context_capacity_tokens: usize,
    ) -> Result<Self> {
        Self::prepare_with_policy(
            source,
            context_capacity_tokens,
            ImplementationPolicy::Ordinary,
        )
    }
    pub(super) fn prepare_qualification(
        source: BoundFullModelSource,
        context_capacity_tokens: usize,
    ) -> Result<Self> {
        Self::prepare_with_policy(
            source,
            context_capacity_tokens,
            ImplementationPolicy::Qualification,
        )
    }
    fn prepare_with_policy(
        source: BoundFullModelSource,
        context_capacity_tokens: usize,
        policy: ImplementationPolicy,
    ) -> Result<Self> {
        source
            .model_profile()
            .prompt_workload(1, context_capacity_tokens)?;
        let metadata = source
            .source()
            .source()
            .reserve_workspace(Self::metadata_bytes()?)?;
        Ok(Self {
            source,
            context_capacity_tokens,
            calls: Mutex::new(VecDeque::with_capacity(PREPARED_CAPACITY)),
            policy,
            _metadata: metadata,
        })
    }
    pub(super) fn source(&self) -> &BoundFullModelSource {
        &self.source
    }
    fn implementation(
        &self,
        input_tokens: usize,
        is_decode: bool,
    ) -> Option<bindings::GdnImplementation> {
        match self.policy {
            ImplementationPolicy::Ordinary => None,
            ImplementationPolicy::Qualification => qualification::gdn_implementation(
                self.source.model_profile(),
                input_tokens,
                is_decode,
            ),
        }
    }
    fn find(
        calls: &VecDeque<Template>,
        workload: PromptWorkload,
        base: Option<usize>,
    ) -> Option<&Template> {
        calls
            .iter()
            .find(|call| call.workload == workload && call.base == base)
    }
    fn template(
        &self,
        workload: PromptWorkload,
        base: Option<usize>,
    ) -> Result<(Arc<SourceBindings>, GdnKernelRecipe)> {
        {
            let calls = self
                .calls
                .lock()
                .map_err(|_| Error::State("prompt template cache poisoned".into()))?;
            if let Some(template) = Self::find(&calls, workload, base) {
                return Ok((template.binding.clone(), template.recipe));
            }
        }
        let is_decode = base.is_some();
        let input_tokens = if is_decode {
            1
        } else {
            workload.prompt_tokens()
        };
        let recipe = bindings::gdn_recipe(
            self.implementation(input_tokens, is_decode),
            input_tokens,
            is_decode,
        )?;
        let span = match base {
            None => PromptSpan::prefill(
                self.source.source().source_identity(),
                workload,
                &vec![0; workload.prompt_tokens()],
            )?,
            Some(base) => {
                PromptSpan::cached(self.source.source().source_identity(), workload, base, 0)?
            }
        };
        let call = self.source.bind_prompt_span(span)?;
        let record = call.record_prepared()?;
        let prepared = call.prepare_program(record)?;
        let binding = bindings::prepare(self.source.source(), prepared, None)?;
        let mut calls = self
            .calls
            .lock()
            .map_err(|_| Error::State("prompt template cache poisoned".into()))?;
        if let Some(template) = Self::find(&calls, workload, base) {
            return Ok((template.binding.clone(), template.recipe));
        }
        if calls.len() == PREPARED_CAPACITY {
            calls.pop_front();
        }
        calls.push_back(Template {
            workload,
            base,
            binding: binding.clone(),
            recipe,
        });
        Ok((binding, recipe))
    }
    pub(super) fn physical_scopes(&self) -> Result<Vec<(Arc<SourceBindings>, GdnKernelRecipe)>> {
        if self.policy != ImplementationPolicy::Qualification {
            return Err(Error::Unsupported(
                "CUDA backend has no ordinary dynamic GDN preparation API".into(),
            ));
        }
        let mut scopes = Vec::with_capacity(qualification::PROMPT_TOKEN_COUNTS.len() + 1);
        for &count in qualification::PROMPT_TOKEN_COUNTS {
            let workload = self
                .source
                .model_profile()
                .prompt_workload(count, self.context_capacity_tokens)?;
            scopes.push(self.template(workload, None)?);
        }
        let first = qualification::PROMPT_TOKEN_COUNTS[0];
        let workload = self
            .source
            .model_profile()
            .prompt_workload(first, self.context_capacity_tokens)?;
        scopes.push(self.template(workload, Some(first))?);
        Ok(scopes)
    }
    fn admit_call(&self, binding: &SourceBindings) -> Result<WindowWorkspace> {
        let bytes = binding
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
        profile.validate_prompt_request(ids, 1, self.context_capacity_tokens)?;
        let workload = profile.prompt_workload(ids.len(), self.context_capacity_tokens)?;
        let (template, recipe) = self.template(workload, None)?;
        let metadata = self.admit_call(&template)?;
        let span = PromptSpan::prefill(self.source.source().source_identity(), workload, ids)?;
        self.bind(span, template, recipe, metadata)
    }
    pub(super) fn bind_cached_metadata(
        &self,
        prompt_tokens: usize,
        base: usize,
        id: u32,
    ) -> Result<(Arc<SourceBindings>, GdnKernelRecipe)> {
        let workload = self
            .source
            .model_profile()
            .prompt_workload(prompt_tokens, self.context_capacity_tokens)?;
        let span = PromptSpan::cached(self.source.source().source_identity(), workload, base, id)?;
        let (template, recipe) = self.template(workload, Some(base))?;
        let metadata = self.admit_call(&template)?;
        self.bind(span, template, recipe, metadata)
    }
    fn bind(
        &self,
        span: PromptSpan,
        template: Arc<SourceBindings>,
        recipe: GdnKernelRecipe,
        metadata: WindowWorkspace,
    ) -> Result<(Arc<SourceBindings>, GdnKernelRecipe)> {
        span.validate()?;
        let call = self.source.bind_prompt_span(span)?;
        let binding = template.rebind(call.call().request().source_call()?, metadata)?;
        Ok((Arc::new(binding), recipe))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::plan::QwenSourceProfile;

    #[test]
    fn bounded_lazy_wrapper_charge_does_not_encode_historical_call_count() {
        assert_eq!(Templates::metadata_bytes().unwrap(), 64 * 1024);
        assert_eq!(PREPARED_CAPACITY, 8);
    }

    #[test]
    fn missing_or_mismatched_backend_implementation_refuses() {
        assert!(matches!(
            bindings::gdn_recipe(None, 48, false),
            Err(Error::Unsupported(_))
        ));
        let implementation =
            qualification::gdn_implementation(QwenSourceProfile::Qwen08, 47, false).unwrap();
        assert!(bindings::gdn_recipe(Some(implementation), 48, false).is_err());
        assert!(qualification::gdn_implementation(QwenSourceProfile::Qwen08, 48, false).is_none());
    }
}
