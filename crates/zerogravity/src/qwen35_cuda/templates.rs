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

struct PreparedCache<T> {
    entries: VecDeque<T>,
    capacity: usize,
}
impl<T> PreparedCache<T> {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "prepared cache capacity");
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    fn find(&self, mut predicate: impl FnMut(&T) -> bool) -> Option<&T> {
        self.entries.iter().find(|entry| predicate(entry))
    }
    /// Returns the evicted owner so destruction can happen after the cache lock.
    fn insert(&mut self, entry: T) -> Option<T> {
        let evicted = (self.entries.len() == self.capacity)
            .then(|| self.entries.pop_front().expect("full prepared cache"));
        self.entries.push_back(entry);
        evicted
    }
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

pub(super) struct Templates {
    source: BoundFullModelSource,
    context_capacity_tokens: usize,
    calls: Mutex<PreparedCache<Template>>,
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
    pub(super) fn prepare_qualification(
        source: BoundFullModelSource,
        context_capacity_tokens: usize,
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
            calls: Mutex::new(PreparedCache::new(PREPARED_CAPACITY)),
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
        qualification::gdn_implementation(self.source.model_profile(), input_tokens, is_decode)
    }
    fn find(
        calls: &PreparedCache<Template>,
        workload: PromptWorkload,
        base: Option<usize>,
    ) -> Option<&Template> {
        calls.find(|call| call.workload == workload && call.base == base)
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
        let evicted = calls.insert(Template {
            workload,
            base,
            binding: binding.clone(),
            recipe,
        });
        drop(calls);
        drop(evicted);
        Ok((binding, recipe))
    }
    pub(super) fn physical_scopes(&self) -> Result<Vec<(Arc<SourceBindings>, GdnKernelRecipe)>> {
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
    fn capacity_eight_cache_retains_evicted_in_flight_owner_then_releases_all() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Resource {
            id: usize,
            released: Arc<AtomicUsize>,
        }
        impl Drop for Resource {
            fn drop(&mut self) {
                self.released.fetch_add(1, Ordering::SeqCst);
            }
        }

        let released = Arc::new(AtomicUsize::new(0));
        let mut cache = PreparedCache::new(PREPARED_CAPACITY);
        for id in 0..PREPARED_CAPACITY {
            assert!(
                cache
                    .insert(Arc::new(Resource {
                        id,
                        released: Arc::clone(&released),
                    }))
                    .is_none()
            );
        }
        assert_eq!(cache.len(), 8);
        let in_flight = Arc::clone(cache.find(|resource| resource.id == 0).unwrap());
        let evicted = cache
            .insert(Arc::new(Resource {
                id: PREPARED_CAPACITY,
                released: Arc::clone(&released),
            }))
            .expect("ninth prepared owner evicts the oldest cache entry");
        assert_eq!(cache.len(), 8);
        assert_eq!(evicted.id, 0);
        drop(evicted);
        assert_eq!(released.load(Ordering::SeqCst), 0);
        drop(in_flight);
        assert_eq!(released.load(Ordering::SeqCst), 1);
        drop(cache);
        assert_eq!(released.load(Ordering::SeqCst), 9);
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
