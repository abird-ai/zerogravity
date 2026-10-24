//! Genuine source/profile join over the original whole-model recorder.
use super::*;
use zerogravity_core::accelerator::program::{ShortHfCall, ShortHfSpan};

// Five workloads × (one prefill + seven exact decode frontiers).
pub(super) const MAX_PREPARED_SHAPES: usize = 40;
pub(super) struct PreparedRecord {
    call: ShortHfCall,
    record: Arc<RecordedVllm08Model>,
}
fn same_shape(a: &ShortHfCall, b: &ShortHfCall) -> bool {
    a.model_profile() == b.model_profile()
        && a.span().source() == b.span().source()
        && a.span().workload() == b.span().workload()
        && a.span().mode() == b.span().mode()
        && a.span().base() == b.span().base()
        && a.span().len() == b.span().len()
}
/// Source owner stays alive with the actual semantic call; no execution permit.
#[derive(Clone)]
pub struct BoundShortHfCall {
    source: BoundFullModelSource,
    call: ShortHfCall,
}
fn require_call(
    source: crate::accelerator::SourceIdentity,
    profile: FullModelProfile,
    call: &ShortHfCall,
) -> Result<()> {
    call.validate()?;
    if call.span().source() != source || call.model_profile() != profile {
        return Err(Error::State(
            "short-HF exact original source/profile issuer".into(),
        ));
    }
    Ok(())
}
impl BoundFullModelSource {
    /// Complete conservative retained cache allowance, including unvisited shapes.
    pub const SHORT_HF_PREPARED_METADATA_BYTES: usize =
        MAX_PREPARED_SHAPES * RecordedVllm08Model::PREPARED_METADATA_BYTES;
    pub fn bind_short_hf_span(&self, span: ShortHfSpan) -> Result<BoundShortHfCall> {
        let call = ShortHfCall::new(self.model_profile(), span)?;
        require_call(self.source().source_identity(), self.model_profile(), &call)?;
        Ok(BoundShortHfCall {
            source: self.clone(),
            call,
        })
    }
}
impl BoundShortHfCall {
    pub fn source(&self) -> &BoundFullModelSource {
        &self.source
    }
    pub fn call(&self) -> &ShortHfCall {
        &self.call
    }
    /// Prepare source/SSA once per exact semantic shape; return a fresh real issuer.
    /// The caller prices invocation-local wrappers with BINDING_METADATA_BYTES.
    pub fn record_prepared(&self) -> Result<RecordedVllm08Model> {
        require_call(
            self.source.source().source_identity(),
            self.source.model_profile(),
            &self.call,
        )?;
        let prepared =
            {
                let mut cache =
                    self.source.0.short_hf.lock().map_err(|_| {
                        Error::State("prepared short-HF source cache poisoned".into())
                    })?;
                if let Some(prepared) = cache.iter().find(|p| same_shape(&p.call, &self.call)) {
                    prepared.record.clone()
                } else {
                    if cache.len() >= MAX_PREPARED_SHAPES {
                        return Err(Error::Limit("prepared short-HF finite shape bound".into()));
                    }
                    let metadata = self
                        .source
                        .source()
                        .source()
                        .reserve_workspace(RecordedVllm08Model::PREPARED_METADATA_BYTES)?;
                    let record = Arc::new(
                        self.record()?
                            .retain_prepared_metadata(metadata.into_host_workspace()?)?,
                    );
                    cache.push(PreparedRecord {
                        call: self.call.clone(),
                        record: record.clone(),
                    });
                    record
                }
            };
        // Independent request-local wrappers never hold the source cache lock.
        prepared.bind_short_hf_call(&self.call)
    }

    pub fn record(&self) -> Result<RecordedVllm08Model> {
        require_call(
            self.source.source().source_identity(),
            self.source.model_profile(),
            &self.call,
        )?;
        self.source
            .source()
            .record_full_model_inner(self.call.as_vllm_call().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::{SourceIdentity, program::ShortHfWorkload};
    #[test]
    fn source_profile_join_rejects_foreign_owner_without_shape_equivalence() {
        // Pure join refusal only; synthetic IDs are not authenticated sources.
        let source = SourceIdentity::for_new_allocation().unwrap();
        let foreign = SourceIdentity::for_new_allocation().unwrap();
        for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
            let span = ShortHfSpan::prefill(source, ShortHfWorkload::Prompt72, &[11; 72]).unwrap();
            let call = ShortHfCall::new(profile, span).unwrap();
            require_call(source, profile, &call).unwrap();
            assert!(require_call(foreign, profile, &call).is_err());
            let other = if profile == FullModelProfile::Qwen08 {
                FullModelProfile::Qwen2B
            } else {
                FullModelProfile::Qwen08
            };
            assert!(require_call(source, other, &call).is_err());
        }
    }
    #[test]
    fn prepared_shape_keys_keep_all_exact_frontiers_profiles_and_issuers() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let foreign = SourceIdentity::for_new_allocation().unwrap();
        for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
            let mut shapes = Vec::new();
            for count in [47, 57, 62, 67, 72] {
                let workload = ShortHfWorkload::from_prompt_rows(count).unwrap();
                for base in std::iter::once(None).chain(workload.cached_bases().map(Some)) {
                    let make = |source, profile, id| {
                        let span = match base {
                            None => ShortHfSpan::prefill(source, workload, &vec![id; count]),
                            Some(base) => ShortHfSpan::cached(source, workload, base, id),
                        }
                        .unwrap();
                        ShortHfCall::new(profile, span).unwrap()
                    };
                    let a = make(source, profile, 11);
                    let b = make(source, profile, 353);
                    assert_ne!(a, b);
                    assert!(same_shape(&a, &b));
                    assert!(!same_shape(&a, &make(foreign, profile, 11)));
                    let other = if profile == FullModelProfile::Qwen08 {
                        FullModelProfile::Qwen2B
                    } else {
                        FullModelProfile::Qwen08
                    };
                    assert!(!same_shape(&a, &make(source, other, 11)));
                    assert!(shapes.iter().all(|prior| !same_shape(prior, &a)));
                    shapes.push(a);
                }
            }
            assert_eq!(shapes.len(), MAX_PREPARED_SHAPES);
        }
    }
}
