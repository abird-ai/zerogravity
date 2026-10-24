//! Closed semantic workloads; no native, current-root or completion authority.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortHfWorkload {
    Prompt47,
    Prompt57,
    Prompt62,
    Prompt67,
    Prompt72,
}
impl ShortHfWorkload {
    pub fn from_prompt_rows(rows: usize) -> Result<Self> {
        match rows {
            47 => Ok(Self::Prompt47),
            57 => Ok(Self::Prompt57),
            62 => Ok(Self::Prompt62),
            67 => Ok(Self::Prompt67),
            72 => Ok(Self::Prompt72),
            _ => Err(Error::Unsupported(
                "closed short-HF prompt rows47/57/62/67/72".into(),
            )),
        }
    }
    pub fn prompt_rows(self) -> usize {
        match self {
            Self::Prompt47 => 47,
            Self::Prompt57 => 57,
            Self::Prompt62 => 62,
            Self::Prompt67 => 67,
            Self::Prompt72 => 72,
        }
    }
    pub fn cached_bases(self) -> std::ops::RangeInclusive<usize> {
        self.prompt_rows()..=self.prompt_rows() + 6
    }
    pub fn final_frontier(self) -> usize {
        self.prompt_rows() + 7
    }
}

/// One immutable actual call issuer. Equal IDs from another mint stay distinct.
/// Cached IDs here are metadata; execution must bind genuine computed feedback.
#[derive(Clone, Debug)]
pub struct ShortHfSpan(Arc<ShortHfSpanData>);
#[derive(Debug)]
struct ShortHfSpanData {
    source: SourceIdentity,
    workload: ShortHfWorkload,
    mode: VllmBf16Mode,
    base: usize,
    ids: Arc<[u32]>,
}
impl PartialEq for ShortHfSpan {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for ShortHfSpan {}
impl ShortHfSpan {
    pub fn prefill(source: SourceIdentity, workload: ShortHfWorkload, ids: &[u32]) -> Result<Self> {
        Self::new(source, workload, VllmBf16Mode::Prefill, 0, ids)
    }
    pub fn cached(
        source: SourceIdentity,
        workload: ShortHfWorkload,
        base: usize,
        id: u32,
    ) -> Result<Self> {
        Self::new(source, workload, VllmBf16Mode::Decode, base, &[id])
    }
    fn new(
        source: SourceIdentity,
        workload: ShortHfWorkload,
        mode: VllmBf16Mode,
        base: usize,
        ids: &[u32],
    ) -> Result<Self> {
        Self::require_parts(source, workload, mode, base, ids)?;
        Ok(Self(Arc::new(ShortHfSpanData {
            source,
            workload,
            mode,
            base,
            ids: Arc::from(ids),
        })))
    }
    fn require_parts(
        source: SourceIdentity,
        workload: ShortHfWorkload,
        mode: VllmBf16Mode,
        base: usize,
        ids: &[u32],
    ) -> Result<()> {
        let shape = match mode {
            VllmBf16Mode::Prefill => base == 0 && ids.len() == workload.prompt_rows(),
            VllmBf16Mode::Decode => workload.cached_bases().contains(&base) && ids.len() == 1,
        };
        if !source.is_bound()
            || !shape
            || base.checked_add(ids.len()).is_none_or(|end| end > 512)
            || ids.iter().any(|&id| id >= 248_320)
        {
            return Err(Error::Malformed(
                "short-HF exact source/workload/span/IDs".into(),
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
    pub fn source(&self) -> SourceIdentity {
        self.0.source
    }
    pub fn workload(&self) -> ShortHfWorkload {
        self.0.workload
    }
    pub fn mode(&self) -> VllmBf16Mode {
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
pub struct ShortHfCall {
    span: ShortHfSpan,
    call: Vllm08Call,
}
impl ShortHfCall {
    pub fn new(profile: FullModelProfile, span: ShortHfSpan) -> Result<Self> {
        let request = VllmBf16Invocation::from_short_hf_span(span.clone())?;
        let call = Vllm08Call::from_request(profile, request)?;
        let value = Self { span, call };
        value.validate()?;
        Ok(value)
    }
    pub fn span(&self) -> &ShortHfSpan {
        &self.span
    }
    pub fn model_profile(&self) -> FullModelProfile {
        self.call.model_profile()
    }
    pub fn request(&self) -> &VllmBf16Invocation {
        self.call.request()
    }
    pub fn as_vllm_call(&self) -> &Vllm08Call {
        &self.call
    }
    pub fn validate(&self) -> Result<()> {
        self.span.validate()?;
        self.call.validate()?;
        if self.request().teachers().short_hf_span() != Some(&self.span)
            || self.request().mode() != self.span.mode()
            || self.call.source() != self.span.source()
        {
            return Err(Error::State(
                "short-HF exact call/teacher issuer changed".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
