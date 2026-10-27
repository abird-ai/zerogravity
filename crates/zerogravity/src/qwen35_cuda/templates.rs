//! Lazy model calls and exact backend implementation binding.
use super::*;
use crate::gpu::SourceBindings;
use crate::gpu::cuda::gdn::TextGdnImplementation;
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
    implementation: TextGdnImplementation,
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
    ) -> Result<Option<TextGdnImplementation>> {
        let profile = self.source.model_profile();
        let candidate = qualification::gdn_implementation(profile, input_tokens, is_decode)?;
        qualification::seal_gdn_implementation(profile, input_tokens, is_decode, candidate)
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
    ) -> Result<(Arc<SourceBindings>, TextGdnImplementation)> {
        {
            let calls = self
                .calls
                .lock()
                .map_err(|_| Error::State("prompt template cache poisoned".into()))?;
            if let Some(template) = Self::find(&calls, workload, base) {
                return Ok((template.binding.clone(), template.implementation));
            }
        }
        let is_decode = base.is_some();
        let input_tokens = if is_decode {
            1
        } else {
            workload.prompt_tokens()
        };
        let implementation = bindings::gdn_implementation(
            self.implementation(input_tokens, is_decode)?,
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
            return Ok((template.binding.clone(), template.implementation));
        }
        let evicted = calls.insert(Template {
            workload,
            base,
            binding: binding.clone(),
            implementation,
        });
        drop(calls);
        drop(evicted);
        Ok((binding, implementation))
    }
    pub(super) fn physical_scopes(
        &self,
    ) -> Result<Vec<(Arc<SourceBindings>, TextGdnImplementation)>> {
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
    ) -> Result<(Arc<SourceBindings>, TextGdnImplementation)> {
        let profile = self.source.model_profile();
        profile.validate_prompt_request(ids, 1, self.context_capacity_tokens)?;
        let workload = profile.prompt_workload(ids.len(), self.context_capacity_tokens)?;
        let (template, implementation) = self.template(workload, None)?;
        let metadata = self.admit_call(&template)?;
        let span = PromptSpan::prefill(self.source.source().source_identity(), workload, ids)?;
        self.bind(span, template, implementation, metadata)
    }
    pub(super) fn bind_cached_metadata(
        &self,
        prompt_tokens: usize,
        base: usize,
        id: u32,
    ) -> Result<(Arc<SourceBindings>, TextGdnImplementation)> {
        let workload = self
            .source
            .model_profile()
            .prompt_workload(prompt_tokens, self.context_capacity_tokens)?;
        let span = PromptSpan::cached(self.source.source().source_identity(), workload, base, id)?;
        let (template, implementation) = self.template(workload, Some(base))?;
        let metadata = self.admit_call(&template)?;
        self.bind(span, template, implementation, metadata)
    }
    fn bind(
        &self,
        span: PromptSpan,
        template: Arc<SourceBindings>,
        implementation: TextGdnImplementation,
        metadata: WindowWorkspace,
    ) -> Result<(Arc<SourceBindings>, TextGdnImplementation)> {
        span.validate()?;
        let call = self.source.bind_prompt_span(span)?;
        let binding = template.rebind(call.call().request().source_call()?, metadata)?;
        Ok((Arc::new(binding), implementation))
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
    fn missing_or_mismatched_backend_implementation_refuses_before_preparation() {
        use crate::gpu::cuda::gdn::{
            GdnKernelConfig, GdnKktKernel, GdnOutputKernel, GdnRecomputeKernel, GdnStateKernel,
            TextGdnImplementation,
        };

        assert!(matches!(
            bindings::gdn_implementation(None, 48, false),
            Err(Error::Unsupported(_))
        ));
        assert!(
            qualification::seal_gdn_implementation(QwenSourceProfile::Qwen08, 47, false, None,)
                .is_err()
        );

        let qwen08_47 = qualification::gdn_implementation(QwenSourceProfile::Qwen08, 47, false)
            .unwrap()
            .unwrap();
        let qwen2b_47 = qualification::gdn_implementation(QwenSourceProfile::Qwen2B, 47, false)
            .unwrap()
            .unwrap();
        let qwen2b_57 = qualification::gdn_implementation(QwenSourceProfile::Qwen2B, 57, false)
            .unwrap()
            .unwrap();
        assert_eq!(qwen08_47.config(), qwen2b_57.config());
        assert_ne!(qwen08_47.config(), qwen2b_47.config());
        assert!(bindings::gdn_implementation(Some(qwen08_47), 48, false).is_err());

        let explicit = GdnKernelConfig::from_stages(
            GdnKktKernel::Bk32W4S3,
            GdnRecomputeKernel::Bk64Bv64W8S4,
            GdnStateKernel::Bv64W4S2,
            GdnOutputKernel::Bk32Bv32W4S3,
        )
        .unwrap();
        let mismatched = TextGdnImplementation::prefill(
            "b24e5fb2ac4b0e591cb907e8226cd1d80bc24507c0586ecfb5238ea4f3c91f8b",
            47,
            explicit,
        )
        .unwrap();
        assert_eq!(mismatched.config(), Some(explicit));
        assert!(
            qualification::seal_gdn_implementation(
                QwenSourceProfile::Qwen08,
                47,
                false,
                Some(mismatched),
            )
            .is_err()
        );
        assert!(
            qualification::gdn_implementation(QwenSourceProfile::Qwen08, 48, false)
                .unwrap()
                .is_none()
        );
    }
}
