//! Retained finite INITIAL1/cached1 qualification entry; no broader eligibility.
use super::*;
use crate::accelerator::program::TextStepBindings;
use std::sync::Arc;

impl OriginalWindowOperators {
    pub fn record_full_source_token(self: &Arc<Self>, position: usize) -> Result<TextStepBindings> {
        if position > 1 || self.geometry().layers.len() != 64 {
            return Err(Error::Unsupported(
                "finite source INITIAL1/cached1 full64".into(),
            ));
        }
        let domains: usize = self
            .geometry()
            .layers
            .iter()
            .map(|kind| match kind {
                crate::config::LayerKind::LinearAttention => 2,
                crate::config::LayerKind::FullAttention => 1,
                _ => 0,
            })
            .sum();
        if domains != 112 {
            return Err(Error::State("source full64 logical112 domains".into()));
        }
        self.record_source_token(position, 2)
    }
}
