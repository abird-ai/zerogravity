//! Explicit captured native ABI identity; no source/model/native execution permit.
use crate::resident_program::source_bindings::SourceBindings;
use zerogravity_core::accelerator::program::{SourceCall, SourceCallMode, SourceOperatorKind};
use zerogravity_core::{Error, Result};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cuda::native) enum NativeCatalogue {
    Source1024,
    Source2048,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureMode {
    Historical,
    FreshPrefill,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextKernelCapture {
    catalogue: NativeCatalogue,
    mode: CaptureMode,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextColdScopeRole {
    Prefill(usize),
    Cached,
}
impl TextColdScopeRole {
    pub(crate) fn require_call(self, call: &SourceCall) -> Result<()> {
        let (mode, rows, base) = match self {
            Self::Prefill(rows) => (SourceCallMode::Prefill, rows, 0),
            Self::Cached => (SourceCallMode::Decode, 1, 47),
        };
        if call.mode() != mode || call.rows().len() != rows || call.rows().base() != base {
            return Err(Error::State("text captured cold scope role differs".into()));
        }
        Ok(())
    }
}
impl TextKernelCapture {
    pub fn bind(recipe_sha256: &str) -> Result<Self> {
        let catalogue = match recipe_sha256 {
            CAPTURE_1024 => NativeCatalogue::Source1024,
            CAPTURE_2048 => NativeCatalogue::Source2048,
            _ => {
                return Err(Error::Unsupported(
                    "unknown exact text native capture".into(),
                ));
            }
        };
        Ok(Self {
            catalogue,
            mode: CaptureMode::Historical,
        })
    }
    /// Qualification-only four-scope capture. It admits one genuine prefill
    /// call for each of rows 47 through 50 and never admits cached decode.
    pub fn bind_fresh_prefill(recipe_sha256: &str) -> Result<Self> {
        if recipe_sha256 != CAPTURE_1024 {
            return Err(Error::Unsupported(
                "fresh-prefill capture requires the exact .8B source recipe".into(),
            ));
        }
        Ok(Self {
            catalogue: NativeCatalogue::Source1024,
            mode: CaptureMode::FreshPrefill,
        })
    }
    pub const fn recipe_sha256(self) -> &'static str {
        match self.catalogue {
            NativeCatalogue::Source1024 => CAPTURE_1024,
            NativeCatalogue::Source2048 => CAPTURE_2048,
        }
    }
    pub const fn hidden_width(self) -> usize {
        match self.catalogue {
            NativeCatalogue::Source1024 => 1024,
            NativeCatalogue::Source2048 => 2048,
        }
    }
    pub const fn intermediate_width(self) -> usize {
        match self.catalogue {
            NativeCatalogue::Source1024 => 3584,
            NativeCatalogue::Source2048 => 6144,
        }
    }
    pub const fn supports_initial_batch(self) -> bool {
        matches!(self.catalogue, NativeCatalogue::Source1024)
            && matches!(self.mode, CaptureMode::Historical)
    }
    pub(crate) const fn is_fresh_prefill(self) -> bool {
        matches!(self.mode, CaptureMode::FreshPrefill)
    }
    // Actual native artifact coverage, independent of model prompt issuance.
    // Extending this requires a separately bound capture, not a larger model list.
    pub(crate) const fn captured_prefill_rows(self) -> &'static [usize] {
        match self.mode {
            CaptureMode::Historical => &[47, 57, 62, 67, 72],
            CaptureMode::FreshPrefill => &[47, 48, 49, 50],
        }
    }
    pub(crate) const fn captured_call_count(self) -> usize {
        match self.mode {
            CaptureMode::Historical => 8,
            CaptureMode::FreshPrefill => 1,
        }
    }
    pub(crate) fn prepared_call_count(self) -> Result<usize> {
        self.captured_prefill_rows()
            .len()
            .checked_mul(self.captured_call_count())
            .ok_or_else(|| Error::Limit("text captured prepared call count".into()))
    }
    pub(crate) fn cold_scope_count(self) -> Result<usize> {
        self.captured_prefill_rows()
            .len()
            .checked_add(usize::from(!self.is_fresh_prefill()))
            .ok_or_else(|| Error::Limit("text captured cold scope count".into()))
    }
    pub(crate) fn cold_scope_role(self, index: usize) -> Result<TextColdScopeRole> {
        if let Some(&rows) = self.captured_prefill_rows().get(index) {
            Ok(TextColdScopeRole::Prefill(rows))
        } else if !self.is_fresh_prefill() && index == self.captured_prefill_rows().len() {
            Ok(TextColdScopeRole::Cached)
        } else {
            Err(Error::State("text captured cold scope index".into()))
        }
    }
    pub(crate) fn cold_scope_index(self, mode: SourceCallMode, rows: usize) -> Result<usize> {
        match mode {
            SourceCallMode::Prefill => self
                .captured_prefill_rows()
                .iter()
                .position(|&actual| actual == rows)
                .ok_or_else(|| Error::Unsupported("text captured prefill geometry absent".into())),
            SourceCallMode::Decode if rows == 1 && !self.is_fresh_prefill() => {
                Ok(self.captured_prefill_rows().len())
            }
            SourceCallMode::Decode => Err(Error::Unsupported(
                "text captured cached geometry absent".into(),
            )),
        }
    }
    pub(crate) fn require_cold_scopes<'a>(
        self,
        scopes: impl IntoIterator<Item = &'a super::super::text_model::TextModelBinding>,
    ) -> Result<()> {
        let mut first = None;
        let mut count = 0usize;
        for scope in scopes {
            self.cold_scope_role(count)?.require_call(scope.call())?;
            self.require_bindings(scope.record())?;
            let source = first.get_or_insert_with(|| scope.source());
            if scope.native_capture() != self || !scope.source().same_issuer(source) {
                return Err(Error::State(
                    "text captured cold source/capture differs".into(),
                ));
            }
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::Limit("text captured cold scope sum".into()))?;
        }
        if count != self.cold_scope_count()? {
            return Err(Error::State(
                "text captured cold scope coverage incomplete".into(),
            ));
        }
        Ok(())
    }
    pub(in crate::cuda::native) const fn catalogue(self) -> NativeCatalogue {
        self.catalogue
    }
    pub(in crate::cuda::native) const fn wide(self) -> bool {
        matches!(self.catalogue, NativeCatalogue::Source2048)
    }
    pub(crate) fn require_packet_geometry(
        self,
        prompt_tokens: usize,
        ordinal: usize,
    ) -> Result<()> {
        if !self.captured_prefill_rows().contains(&prompt_tokens)
            || ordinal >= self.captured_call_count()
        {
            return Err(Error::Unsupported(
                "text capture packet geometry absent".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require_bindings(self, bindings: &SourceBindings) -> Result<()> {
        let call = bindings.call();
        let plan = bindings.plan();
        let call_ok = match call.mode() {
            SourceCallMode::Prefill => {
                call.rows().base() == 0 && self.captured_prefill_rows().contains(&call.rows().len())
            }
            SourceCallMode::Decode => {
                !self.is_fresh_prefill()
                    && call.rows().len() == 1
                    && matches!(call.rows().base(), 47..=78)
            }
        };
        if !call_ok
            || call.rows().capacity() != 512
            || call.rows().vocabulary() != 248320
            || plan.nodes().len() != 255
            || plan.values().len() != 327
            || plan.required_domains().len() != 42
            || plan.proposed_domains().len() != 42
            || plan.source() != bindings.source().source_identity()
        {
            return Err(Error::Unsupported(
                "text native capture actual graph/source/call extent differs".into(),
            ));
        }
        for node in plan.nodes() {
            if let zerogravity_core::accelerator::program::PlanOperation::SourceOperator {
                descriptor,
                ..
            } = &node.operation
            {
                match descriptor.kind() {
                    SourceOperatorKind::Rms(g) | SourceOperatorKind::ResidualRms(g)
                        if g.width != self.hidden_width() =>
                    {
                        return Err(Error::State(
                            "text native capture actual RMS width differs".into(),
                        ));
                    }
                    SourceOperatorKind::EmbeddingRows if !matches!(descriptor.matrices(),[m] if m.rows==248320&&m.columns==self.hidden_width()) =>
                    {
                        return Err(Error::State(
                            "text native capture actual embedding header differs".into(),
                        ));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}
const CAPTURE_1024: &str = "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d";
const CAPTURE_2048: &str = "4d77cb6637843a57a17fc093ad1e5241ddf083bf8d9ade0c9bbaa118bc9448e2";

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::{SourceRows, TextCallGeometry};

    #[test]
    fn cold_roles_cover_the_actual_literal_capture_without_admitting_new_prompts() -> Result<()> {
        for pin in [CAPTURE_1024, CAPTURE_2048] {
            let capture = TextKernelCapture::bind(pin)?;
            let mut actual = BTreeMap::<usize, BTreeSet<usize>>::new();
            for site in super::super::generated_sites(capture)? {
                actual
                    .entry(site.key.prompt_tokens)
                    .or_default()
                    .insert(site.key.call);
            }
            assert_eq!(
                actual.keys().copied().collect::<Vec<_>>(),
                capture.captured_prefill_rows()
            );
            let calls = (0..capture.captured_call_count()).collect::<BTreeSet<_>>();
            assert!(actual.values().all(|actual| actual == &calls));
            assert_eq!(
                actual.values().map(BTreeSet::len).sum::<usize>(),
                capture.prepared_call_count()?
            );
            for &rows in actual.keys() {
                let index = capture.cold_scope_index(SourceCallMode::Prefill, rows)?;
                assert_eq!(
                    capture.cold_scope_role(index)?,
                    TextColdScopeRole::Prefill(rows)
                );
            }
            let cached = capture.cold_scope_index(SourceCallMode::Decode, 1)?;
            assert_eq!(capture.cold_scope_role(cached)?, TextColdScopeRole::Cached);
            assert_eq!(cached.checked_add(1), Some(capture.cold_scope_count()?));
            assert!(
                capture
                    .cold_scope_role(capture.cold_scope_count()?)
                    .is_err()
            );
            for rows in [1, 48, 49, 50] {
                assert!(
                    capture
                        .cold_scope_index(SourceCallMode::Prefill, rows)
                        .is_err()
                );
            }
            assert!(
                capture
                    .cold_scope_index(SourceCallMode::Decode, 47)
                    .is_err()
            );
        }

        let fresh = TextKernelCapture::bind_fresh_prefill(CAPTURE_1024)?;
        let mut actual = BTreeMap::<usize, BTreeSet<usize>>::new();
        for site in super::super::generated_sites(fresh)? {
            actual
                .entry(site.key.prompt_tokens)
                .or_default()
                .insert(site.key.call);
        }
        assert_eq!(actual.keys().copied().collect::<Vec<_>>(), [47, 48, 49, 50]);
        assert!(
            actual
                .values()
                .all(|calls| calls == &[0].into_iter().collect())
        );
        assert_eq!(fresh.prepared_call_count()?, 4);
        assert_eq!(fresh.cold_scope_count()?, 4);
        assert!(fresh.cold_scope_index(SourceCallMode::Decode, 1).is_err());
        assert!(TextKernelCapture::bind_fresh_prefill(CAPTURE_2048).is_err());
        Ok(())
    }

    #[test]
    fn cached_cold_role_requires_its_original_mode_and_representative_base() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        let geometry = TextCallGeometry::new(source, 248320, 512)?;
        let call = |mode, base, rows| {
            SourceCall::new(
                SourceRows::new(geometry, base, Arc::from(vec![11; rows]))?,
                mode,
            )
        };
        let fresh = call(SourceCallMode::Prefill, 0, 47)?;
        let cached = call(SourceCallMode::Decode, 47, 1)?;
        TextColdScopeRole::Prefill(47).require_call(&fresh)?;
        TextColdScopeRole::Cached.require_call(&cached)?;
        assert!(TextColdScopeRole::Cached.require_call(&fresh).is_err());
        assert!(
            TextColdScopeRole::Prefill(47)
                .require_call(&cached)
                .is_err()
        );
        assert!(
            TextColdScopeRole::Cached
                .require_call(&call(SourceCallMode::Decode, 48, 1)?)
                .is_err()
        );
        assert!(
            TextColdScopeRole::Cached
                .require_call(&call(SourceCallMode::Prefill, 0, 1)?)
                .is_err()
        );
        Ok(())
    }
}
