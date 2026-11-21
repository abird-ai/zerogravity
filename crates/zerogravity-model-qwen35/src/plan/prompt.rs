//! Source-bound prompt semantics; no native, current-root or completion authority.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptWorkload {
    prompt_tokens: usize,
    context_capacity_tokens: usize,
}
impl PromptWorkload {
    pub fn new(prompt_tokens: usize, context_capacity_tokens: usize) -> Result<Self> {
        if prompt_tokens == 0
            || context_capacity_tokens == 0
            || prompt_tokens > context_capacity_tokens
            || u64::try_from(context_capacity_tokens).is_err()
        {
            return Err(Error::Malformed(
                "positive prompt within configured context required".into(),
            ));
        }
        Ok(Self {
            prompt_tokens,
            context_capacity_tokens,
        })
    }
    /// Semantic validation only. A backend must independently supply an exact
    /// implementation for the resulting span.
    pub fn validate_request(
        self,
        ids: &[u32],
        max_tokens: usize,
        vocabulary_size: usize,
    ) -> Result<()> {
        if ids.len() != self.prompt_tokens
            || max_tokens == 0
            || vocabulary_size == 0
            || ids.iter().any(|&id| id as usize >= vocabulary_size)
        {
            return Err(Error::Malformed(
                "prompt IDs/output span/model vocabulary".into(),
            ));
        }
        self.checked_frontier(max_tokens)?;
        Ok(())
    }
    pub const fn prompt_tokens(self) -> usize {
        self.prompt_tokens
    }
    pub const fn context_capacity_tokens(self) -> usize {
        self.context_capacity_tokens
    }
    pub fn checked_frontier(self, generated_tokens: usize) -> Result<usize> {
        if generated_tokens == 0 {
            return Err(Error::Malformed("positive generated span required".into()));
        }
        self.prompt_tokens
            .checked_add(generated_tokens)
            .filter(|&frontier| frontier <= self.context_capacity_tokens)
            .ok_or_else(|| Error::Limit("prompt frontier exceeds configured context".into()))
    }
    fn validate_cached_base(self, base: usize) -> Result<()> {
        if base < self.prompt_tokens || base >= self.context_capacity_tokens {
            return Err(Error::Malformed(
                "cached prompt frontier outside configured context".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn from_prompt_tokens(prompt_tokens: usize) -> Result<Self> {
        Self::new(prompt_tokens, 512)
    }
    #[cfg(test)]
    pub(crate) fn cached_bases(self) -> std::ops::RangeInclusive<usize> {
        self.prompt_tokens()..=self.prompt_tokens() + 6
    }
    #[cfg(test)]
    pub(crate) fn final_frontier(self) -> usize {
        self.prompt_tokens() + 7
    }
}

/// One immutable actual call issuer. Equal IDs from another mint stay distinct.
/// Cached IDs here are metadata; execution must bind genuine computed feedback.
#[derive(Clone, Debug)]
pub struct PromptSpan(Arc<PromptSpanData>);
#[derive(Debug)]
struct PromptSpanData {
    rows: TokenSpan,
    source: SourceIdentity,
    workload: PromptWorkload,
    phase: InvocationPhase,
    base: usize,
    ids: Arc<[u32]>,
}
impl PartialEq for PromptSpan {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for PromptSpan {}
impl PromptSpan {
    pub fn prefill(source: SourceIdentity, workload: PromptWorkload, ids: &[u32]) -> Result<Self> {
        Self::new(source, workload, InvocationPhase::Prefill, 0, ids)
    }
    pub fn cached(
        source: SourceIdentity,
        workload: PromptWorkload,
        base: usize,
        id: u32,
    ) -> Result<Self> {
        Self::new(source, workload, InvocationPhase::Decode, base, &[id])
    }
    fn new(
        source: SourceIdentity,
        workload: PromptWorkload,
        phase: InvocationPhase,
        base: usize,
        ids: &[u32],
    ) -> Result<Self> {
        Self::require_parts(source, workload, phase, base, ids)?;
        Ok(Self(Arc::new(PromptSpanData {
            rows: TokenSpan::new(
                TextCallGeometry::new(
                    source,
                    QwenModelVariant::Qwen800M.vocabulary_size(),
                    workload.context_capacity_tokens(),
                )?,
                base,
                Arc::from(ids),
            )?,
            source,
            workload,
            phase,
            base,
            ids: Arc::from(ids),
        })))
    }
    fn require_parts(
        source: SourceIdentity,
        workload: PromptWorkload,
        phase: InvocationPhase,
        base: usize,
        ids: &[u32],
    ) -> Result<()> {
        let shape = match phase {
            InvocationPhase::Prefill => base == 0 && ids.len() == workload.prompt_tokens(),
            InvocationPhase::Decode => {
                workload.validate_cached_base(base).is_ok() && ids.len() == 1
            }
        };
        if !source.is_bound()
            || !shape
            || base
                .checked_add(ids.len())
                .is_none_or(|end| end > workload.context_capacity_tokens())
            || ids
                .iter()
                .any(|&id| id as usize >= QwenModelVariant::Qwen800M.vocabulary_size())
        {
            return Err(Error::Malformed(
                "bounded prompt exact source/workload/span/IDs".into(),
            ));
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        Self::require_parts(
            self.source(),
            self.workload(),
            self.phase(),
            self.base(),
            self.ids(),
        )
    }
    pub(super) fn source_rows(&self) -> TokenSpan {
        self.0.rows.clone()
    }
    pub fn source(&self) -> SourceIdentity {
        self.0.source
    }
    pub fn workload(&self) -> PromptWorkload {
        self.0.workload
    }
    pub fn phase(&self) -> InvocationPhase {
        self.0.phase
    }
    pub fn base(&self) -> usize {
        self.0.base
    }
    pub fn capacity(&self) -> usize {
        self.workload().context_capacity_tokens()
    }
    pub fn ids(&self) -> &[u32] {
        &self.0.ids
    }
    pub fn len(&self) -> usize {
        self.ids().len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn frontier(&self) -> usize {
        self.base() + self.len()
    }
}

/// Geometry metadata only; an authenticated model owner binds the profile.
/// Uses the existing whole-model operators/traversal, never a native permit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptInvocation {
    span: PromptSpan,
    model_invocation: QwenModelInvocation,
}
impl PromptInvocation {
    pub fn new(profile: QwenModelVariant, span: PromptSpan) -> Result<Self> {
        profile.validate_prompt_workload(span.workload())?;
        let request = QwenGdnInvocation::from_prompt_span(span.clone())?;
        let model_invocation = QwenModelInvocation::from_request(profile, request)?;
        let value = Self {
            span,
            model_invocation,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn span(&self) -> &PromptSpan {
        &self.span
    }
    pub fn model_profile(&self) -> QwenModelVariant {
        self.model_invocation.model_profile()
    }
    pub fn request(&self) -> &QwenGdnInvocation {
        self.model_invocation.request()
    }
    pub fn model_invocation(&self) -> &QwenModelInvocation {
        &self.model_invocation
    }
    pub fn validate(&self) -> Result<()> {
        self.span.validate()?;
        self.model_invocation.validate()?;
        self.model_profile()
            .validate_prompt_workload(self.span.workload())?;
        if self.request().teachers().prompt_span() != Some(&self.span)
            || self.request().phase() != self.span.phase()
            || self.model_invocation.source() != self.span.source()
        {
            return Err(Error::State(
                "bounded prompt invocation/teacher issuer changed".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "prompt/tests.rs"]
mod tests;
