//! Explicit captured native ABI identity; no source/model/native execution permit.
use crate::resident_program::source_bindings::SourceBindings;
use zerogravity_core::accelerator::program::{SourceCallMode, SourceOperatorKind};
use zerogravity_core::{Error, Result};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cuda::native) enum NativeCatalogue {
    Source1024,
    Source2048,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextKernelCapture {
    catalogue: NativeCatalogue,
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
        Ok(Self { catalogue })
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
    }
    pub(in crate::cuda::native) const fn catalogue(self) -> NativeCatalogue {
        self.catalogue
    }
    pub(in crate::cuda::native) const fn wide(self) -> bool {
        matches!(self.catalogue, NativeCatalogue::Source2048)
    }
    pub(crate) fn require_bindings(self, bindings: &SourceBindings) -> Result<()> {
        let call = bindings.call();
        let plan = bindings.plan();
        let call_ok = match call.mode() {
            SourceCallMode::Prefill => {
                call.rows().base() == 0 && matches!(call.rows().len(), 47 | 57 | 62 | 67 | 72)
            }
            SourceCallMode::Decode => {
                call.rows().len() == 1 && matches!(call.rows().base(), 47..=78)
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
const CAPTURE_2048: &str = "0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0";
