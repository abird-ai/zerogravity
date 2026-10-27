use super::GdnPhase;
use zerogravity_core::{Error, Result};

/// Closed KKT implementation choices retained by the SM86 backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnKktKernel {
    Bk32W4S3,
    Bk32W8S3,
    Bk64W8S4,
}

/// Closed recompute implementation choices retained by the SM86 backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnRecomputeKernel {
    Bk64Bv64W8S3,
    Bk64Bv64W8S4,
}

/// Closed H/state implementation choices retained by the SM86 backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnStateKernel {
    Bv32W4S2,
    Bv64W4S2,
}

/// Closed Output implementation choices retained by the SM86 backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnOutputKernel {
    Bk32Bv32W4S2,
    Bk32Bv32W4S3,
    Bk32Bv64W4S3,
}

/// A coherent backend numerical implementation contract.
///
/// Each closed choice names an actual retained source/ABI implementation. The
/// configuration is independent of qualification cases and prompt lengths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnKernelConfig {
    kkt: GdnKktKernel,
    recompute: GdnRecomputeKernel,
    state: GdnStateKernel,
    output: GdnOutputKernel,
}

impl GdnKernelConfig {
    pub fn from_stages(
        kkt: GdnKktKernel,
        recompute: GdnRecomputeKernel,
        state: GdnStateKernel,
        output: GdnOutputKernel,
    ) -> Result<Self> {
        Ok(Self {
            kkt,
            recompute,
            state,
            output,
        })
    }

    pub(crate) fn require(config: Option<Self>) -> Result<Self> {
        config.ok_or_else(|| Error::Unsupported("missing explicit GDN kernel config".into()))
    }

    pub fn kkt(self) -> GdnKktKernel {
        self.kkt
    }

    pub fn recompute(self) -> GdnRecomputeKernel {
        self.recompute
    }

    pub fn state(self) -> GdnStateKernel {
        self.state
    }

    pub fn output(self) -> GdnOutputKernel {
        self.output
    }
}

/// Actual backend implementation selected for a checked call geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TextGdnKernel {
    ParameterizedPrefillV1(GdnKernelConfig),
    ParameterizedCachedV1,
}

/// Recipe-free text-GDN execution descriptor.
///
/// Historical finite qualification authenticates a case before constructing
/// this value. Execution retains only call geometry and the actual backend
/// implementation identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextGdnImplementation {
    input_tokens: usize,
    kernel: TextGdnKernel,
}

impl TextGdnImplementation {
    pub fn prefill(input_tokens: usize, config: GdnKernelConfig) -> Result<Self> {
        if input_tokens == 0 || input_tokens > i32::MAX as usize / 16 {
            return Err(Error::Unsupported("text GDN prefill input geometry".into()));
        }
        Ok(Self {
            input_tokens,
            kernel: TextGdnKernel::ParameterizedPrefillV1(config),
        })
    }

    pub fn cached() -> Self {
        Self {
            input_tokens: 1,
            kernel: TextGdnKernel::ParameterizedCachedV1,
        }
    }

    pub fn input_tokens(self) -> usize {
        self.input_tokens
    }

    pub fn is_decode(self) -> bool {
        matches!(self.kernel, TextGdnKernel::ParameterizedCachedV1)
    }

    pub fn config(self) -> Option<GdnKernelConfig> {
        match self.kernel {
            TextGdnKernel::ParameterizedPrefillV1(config) => Some(config),
            TextGdnKernel::ParameterizedCachedV1 => None,
        }
    }

    pub(crate) fn require_phase(self, phase: GdnPhase) -> Result<()> {
        let matches = self.input_tokens == phase.tokens()
            && matches!(
                (phase, self.kernel),
                (
                    GdnPhase::TextPrefill(_),
                    TextGdnKernel::ParameterizedPrefillV1(_)
                ) | (GdnPhase::CachedDecode, TextGdnKernel::ParameterizedCachedV1)
            );
        if !matches {
            return Err(Error::Unsupported(
                "explicit text GDN implementation/phase/config".into(),
            ));
        }
        Ok(())
    }
}
