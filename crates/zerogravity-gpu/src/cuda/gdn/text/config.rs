use super::{GdnPhase, observed::GdnKernelRecipe};
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
}

/// A coherent backend numerical implementation contract.
///
/// Fields are deliberately private: callers can name concrete stage choices,
/// but only combinations observed together in the retained qualification set
/// can become an executable configuration.
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
        use GdnKktKernel::{Bk32W4S3, Bk32W8S3, Bk64W8S4};
        use GdnOutputKernel::{Bk32Bv32W4S2, Bk32Bv32W4S3};
        use GdnRecomputeKernel::{Bk64Bv64W8S3, Bk64Bv64W8S4};
        use GdnStateKernel::{Bv32W4S2, Bv64W4S2};

        if !matches!(
            (kkt, recompute, state, output),
            (Bk32W8S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S2)
                | (Bk32W8S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S3)
                | (Bk64W8S4, Bk64Bv64W8S3, Bv32W4S2, Bk32Bv32W4S3)
                | (Bk32W8S3, Bk64Bv64W8S4, Bv32W4S2, Bk32Bv32W4S2)
                | (Bk32W4S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S3)
                | (Bk32W4S3, Bk64Bv64W8S4, Bv32W4S2, Bk32Bv32W4S3)
        ) {
            return Err(Error::Unsupported(
                "unobserved GDN kernel component combination".into(),
            ));
        }
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

/// Sealed text-GDN implementation selected at an explicit qualification boundary.
///
/// The recipe authenticates the retained qualification envelope. The explicit
/// config alone selects prefill arithmetic; neither value can be changed after
/// construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextGdnImplementation {
    recipe: GdnKernelRecipe,
    prefill_config: Option<GdnKernelConfig>,
}

impl TextGdnImplementation {
    pub fn prefill(
        recipe_sha256: &str,
        input_tokens: usize,
        config: GdnKernelConfig,
    ) -> Result<Self> {
        Ok(Self {
            recipe: GdnKernelRecipe::bind(recipe_sha256, input_tokens, false)?,
            prefill_config: Some(config),
        })
    }

    pub fn cached(recipe_sha256: &str) -> Result<Self> {
        Ok(Self {
            recipe: GdnKernelRecipe::bind(recipe_sha256, 1, true)?,
            prefill_config: None,
        })
    }

    pub fn input_tokens(self) -> usize {
        self.recipe.input_tokens()
    }

    pub fn is_decode(self) -> bool {
        self.recipe.is_decode()
    }

    pub fn config(self) -> Option<GdnKernelConfig> {
        self.prefill_config
    }

    pub(crate) fn require_phase(self, phase: GdnPhase) -> Result<()> {
        let matches = self.input_tokens() == phase.tokens()
            && matches!(
                (phase, self.is_decode(), self.prefill_config),
                (GdnPhase::TextPrefill(_), false, Some(_)) | (GdnPhase::CachedDecode, true, None)
            );
        if !matches {
            return Err(Error::Unsupported(
                "explicit text GDN implementation/phase/config".into(),
            ));
        }
        Ok(())
    }
}
