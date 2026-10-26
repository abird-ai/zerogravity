use super::{CAPTURES, GdnKernelRecipe};
use zerogravity_core::{Error, Result};

impl GdnKernelRecipe {
    pub(in crate::cuda::gdn) fn fixture(
        catalogue: usize,
        input_tokens: usize,
        is_decode: bool,
    ) -> Result<Self> {
        let cell = catalogue
            .checked_mul(6)
            .and_then(|base| {
                CAPTURES
                    .get(base..base + 6)?
                    .iter()
                    .position(|c| c.input_tokens == input_tokens && c.is_decode == is_decode)
                    .map(|i| base + i)
            })
            .ok_or_else(|| {
                Error::Unsupported("actual original text fixture tuple absent".into())
            })?;
        Self::bind(CAPTURES[cell].recipe_sha256, input_tokens, is_decode)
    }
}
