//! Closed semantic workloads; no native, current-root or completion authority.
use super::*;

// Former Prompt47/57/62/67/72 variants and from_prompt_rows/prompt_rows methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptWorkload {
    prompt_tokens: usize,
}
impl PromptWorkload {
    /// Finite source route policy; neither metadata nor geometry admits inference.
    pub fn validate_request(ids: &[u32], max_tokens: usize) -> Result<()> {
        if !matches!(ids.len(), 47 | 57 | 62 | 67 | 72)
            || ids.iter().any(|&id| id >= 248320)
            || !(1..=8).contains(&max_tokens)
        {
            return Err(Error::Unsupported(
                "text finite prompt tokens47/57/62/67/72 and1..8 outputs only".into(),
            ));
        }
        Ok(())
    }
    pub fn from_prompt_tokens(prompt_tokens: usize) -> Result<Self> {
        if !matches!(prompt_tokens, 47 | 57 | 62 | 67 | 72) {
            return Err(Error::Unsupported(
                "closed bounded prompt tokens47/57/62/67/72".into(),
            ));
        }
        Ok(Self { prompt_tokens })
    }
    pub const fn prompt_tokens(self) -> usize {
        self.prompt_tokens
    }
    pub fn cached_bases(self) -> std::ops::RangeInclusive<usize> {
        self.prompt_tokens()..=self.prompt_tokens() + 6
    }
    pub fn final_frontier(self) -> usize {
        self.prompt_tokens() + 7
    }
}

/// One immutable actual call issuer. Equal IDs from another mint stay distinct.
/// Cached IDs here are metadata; execution must bind genuine computed feedback.
#[derive(Clone, Debug)]
pub struct PromptSpan(Arc<PromptSpanData>);
#[derive(Debug)]
struct PromptSpanData {
    rows: SourceRows,
    source: SourceIdentity,
    workload: PromptWorkload,
    mode: SourceCallMode,
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
        Self::new(source, workload, SourceCallMode::Prefill, 0, ids)
    }
    pub fn cached(
        source: SourceIdentity,
        workload: PromptWorkload,
        base: usize,
        id: u32,
    ) -> Result<Self> {
        Self::new(source, workload, SourceCallMode::Decode, base, &[id])
    }
    fn new(
        source: SourceIdentity,
        workload: PromptWorkload,
        mode: SourceCallMode,
        base: usize,
        ids: &[u32],
    ) -> Result<Self> {
        Self::require_parts(source, workload, mode, base, ids)?;
        Ok(Self(Arc::new(PromptSpanData {
            rows: SourceRows::new(
                TextCallGeometry::new(source, 248320, 512)?,
                base,
                Arc::from(ids),
            )?,
            source,
            workload,
            mode,
            base,
            ids: Arc::from(ids),
        })))
    }
    fn require_parts(
        source: SourceIdentity,
        workload: PromptWorkload,
        mode: SourceCallMode,
        base: usize,
        ids: &[u32],
    ) -> Result<()> {
        let shape = match mode {
            SourceCallMode::Prefill => base == 0 && ids.len() == workload.prompt_tokens(),
            SourceCallMode::Decode => workload.cached_bases().contains(&base) && ids.len() == 1,
        };
        if !source.is_bound()
            || !shape
            || base.checked_add(ids.len()).is_none_or(|end| end > 512)
            || ids.iter().any(|&id| id >= 248_320)
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
            self.mode(),
            self.base(),
            self.ids(),
        )
    }
    pub(super) fn source_rows(&self) -> SourceRows {
        self.0.rows.clone()
    }
    pub fn source(&self) -> SourceIdentity {
        self.0.source
    }
    pub fn workload(&self) -> PromptWorkload {
        self.0.workload
    }
    pub fn mode(&self) -> SourceCallMode {
        self.0.mode
    }
    pub fn base(&self) -> usize {
        self.0.base
    }
    pub fn capacity(&self) -> usize {
        512
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
}

/// Geometry metadata only; an authenticated model owner binds the profile.
/// Uses the existing whole-model operators/traversal, never a native permit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptCall {
    span: PromptSpan,
    call: QwenSourceCall,
}
impl PromptCall {
    pub fn new(profile: QwenSourceProfile, span: PromptSpan) -> Result<Self> {
        let request = QwenGdnCall::from_prompt_span(span.clone())?;
        let call = QwenSourceCall::from_request(profile, request)?;
        let value = Self { span, call };
        value.validate()?;
        Ok(value)
    }
    pub fn span(&self) -> &PromptSpan {
        &self.span
    }
    pub fn model_profile(&self) -> QwenSourceProfile {
        self.call.model_profile()
    }
    pub fn request(&self) -> &QwenGdnCall {
        self.call.request()
    }
    pub fn as_source_call(&self) -> &QwenSourceCall {
        &self.call
    }
    pub fn validate(&self) -> Result<()> {
        self.span.validate()?;
        self.call.validate()?;
        if self.request().teachers().prompt_span() != Some(&self.span)
            || self.request().mode() != self.span.mode()
            || self.call.source() != self.span.source()
        {
            return Err(Error::State(
                "bounded prompt exact call/teacher issuer changed".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "prompt/tests.rs"]
mod tests;
