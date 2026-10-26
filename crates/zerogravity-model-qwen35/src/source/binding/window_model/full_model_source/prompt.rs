//! Genuine source/profile join over the original whole-model recorder.
use super::*;
use crate::plan::{PromptCall, PromptSpan};

const PREPARED_PROMPT_CAPACITY: usize = 8;

pub(super) struct PreparedRecord {
    call: PromptCall,
    record: Arc<QwenPlan>,
}
fn same_shape(a: &PromptCall, b: &PromptCall) -> bool {
    a.model_profile() == b.model_profile()
        && a.span().source() == b.span().source()
        && a.span().workload() == b.span().workload()
        && a.span().mode() == b.span().mode()
        && a.span().base() == b.span().base()
        && a.span().len() == b.span().len()
}
/// Source owner stays alive with the actual semantic call; no execution permit.
#[derive(Clone)]
pub struct BoundPromptCall {
    source: BoundFullModelSource,
    call: PromptCall,
}
fn require_call(
    source: crate::accelerator::SourceIdentity,
    profile: QwenSourceProfile,
    call: &PromptCall,
) -> Result<()> {
    call.validate()?;
    if call.span().source() != source || call.model_profile() != profile {
        return Err(Error::State(
            "full-model exact authenticated source/profile issuer".into(),
        ));
    }
    Ok(())
}
impl BoundFullModelSource {
    /// Charge only entries retained by the bounded lazy cache.
    pub fn prompt_prepared_metadata_bytes(&self) -> Result<usize> {
        let retained = self
            .0
            .prompt
            .lock()
            .map_err(|_| Error::State("prepared prompt source cache poisoned".into()))?
            .len();
        retained
            .checked_mul(QwenPlan::PREPARED_METADATA_BYTES)
            .ok_or_else(|| Error::Limit("retained prompt metadata extent".into()))
    }
    pub fn bind_prompt_span(&self, span: PromptSpan) -> Result<BoundPromptCall> {
        let call = PromptCall::new(self.model_profile(), span)?;
        require_call(self.source().source_identity(), self.model_profile(), &call)?;
        Ok(BoundPromptCall {
            source: self.clone(),
            call,
        })
    }
}
impl BoundPromptCall {
    pub fn source(&self) -> &BoundFullModelSource {
        &self.source
    }
    pub fn call(&self) -> &PromptCall {
        &self.call
    }
    /// Prepare source/SSA once per exact semantic shape; return a fresh real issuer.
    /// The caller prices invocation-local wrappers with BINDING_METADATA_BYTES.
    pub fn record_prepared(&self) -> Result<QwenPlan> {
        require_call(
            self.source.source().source_identity(),
            self.source.model_profile(),
            &self.call,
        )?;
        let prepared =
            {
                let mut cache =
                    self.source.0.prompt.lock().map_err(|_| {
                        Error::State("prepared prompt source cache poisoned".into())
                    })?;
                if let Some(prepared) = cache.iter().find(|p| same_shape(&p.call, &self.call)) {
                    prepared.record.clone()
                } else {
                    let metadata = self
                        .source
                        .source()
                        .source()
                        .reserve_workspace(QwenPlan::PREPARED_METADATA_BYTES)?;
                    let record = Arc::new(
                        self.record()?
                            .retain_prepared_metadata(metadata.into_host_workspace()?)?,
                    );
                    if cache.len() == PREPARED_PROMPT_CAPACITY {
                        cache.pop_front();
                    }
                    cache.push_back(PreparedRecord {
                        call: self.call.clone(),
                        record: record.clone(),
                    });
                    record
                }
            };
        // Independent request-local wrappers never hold the source cache lock.
        prepared.bind_prompt_call(&self.call)
    }

    pub fn record(&self) -> Result<QwenPlan> {
        require_call(
            self.source.source().source_identity(),
            self.source.model_profile(),
            &self.call,
        )?;
        self.source
            .source()
            .record_full_model_inner(self.call.as_source_call().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::PromptWorkload;
    use zerogravity_core::accelerator::SourceIdentity;

    fn call(
        source: SourceIdentity,
        profile: QwenSourceProfile,
        prompt_tokens: usize,
        capacity: usize,
        base: Option<usize>,
        id: u32,
    ) -> PromptCall {
        let workload = PromptWorkload::new(prompt_tokens, capacity).unwrap();
        let span = match base {
            None => PromptSpan::prefill(source, workload, &vec![id; prompt_tokens]),
            Some(base) => PromptSpan::cached(source, workload, base, id),
        }
        .unwrap();
        PromptCall::new(profile, span).unwrap()
    }

    #[test]
    fn source_profile_join_rejects_foreign_owner_without_shape_equivalence() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let foreign = SourceIdentity::for_new_allocation().unwrap();
        for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
            let actual = call(source, profile, 73, 512, None, 11);
            require_call(source, profile, &actual).unwrap();
            assert!(require_call(foreign, profile, &actual).is_err());
            let other = if profile == QwenSourceProfile::Qwen08 {
                QwenSourceProfile::Qwen2B
            } else {
                QwenSourceProfile::Qwen08
            };
            assert!(require_call(source, other, &actual).is_err());
        }
    }

    #[test]
    fn shape_keys_preserve_context_frontier_profile_and_issuer() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let foreign = SourceIdentity::for_new_allocation().unwrap();
        let a = call(source, QwenSourceProfile::Qwen08, 48, 512, None, 11);
        let equal_shape = call(source, QwenSourceProfile::Qwen08, 48, 512, None, 353);
        assert_ne!(a, equal_shape);
        assert!(same_shape(&a, &equal_shape));
        for different in [
            call(foreign, QwenSourceProfile::Qwen08, 48, 512, None, 11),
            call(source, QwenSourceProfile::Qwen2B, 48, 512, None, 11),
            call(source, QwenSourceProfile::Qwen08, 49, 512, None, 11),
            call(source, QwenSourceProfile::Qwen08, 48, 1024, None, 11),
            call(source, QwenSourceProfile::Qwen08, 48, 512, Some(48), 11),
            call(source, QwenSourceProfile::Qwen08, 48, 512, Some(49), 11),
        ] {
            assert!(!same_shape(&a, &different));
        }
    }
}
