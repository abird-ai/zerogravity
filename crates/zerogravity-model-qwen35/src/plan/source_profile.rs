//! Original source-family geometry and finite prompt inventory.
//! SourceIdentity and ProgramNumericalProfile/provider semantics stay independent.
//! No source/runtime/state authority or learned-source substitution.
use super::{PromptWorkload, QwenProjection};
use crate::{Error, Result};

const ORIGINAL_PROMPT_TOKEN_COUNTS: &[usize] = &[47, 57, 62, 67, 72];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenSourceProfile {
    Qwen08,
    Qwen2B,
}
impl QwenSourceProfile {
    /// Model-owned admitted shapes; geometry alone grants no native execution.
    pub const fn prompt_token_counts(self) -> &'static [usize] {
        match self {
            Self::Qwen08 => ORIGINAL_PROMPT_TOKEN_COUNTS,
            Self::Qwen2B => ORIGINAL_PROMPT_TOKEN_COUNTS,
        }
    }
    pub const fn max_output_tokens(self) -> usize {
        8
    }
    pub const fn context_capacity_tokens(self) -> usize {
        512
    }
    pub fn prompt_workload(self, prompt_tokens: usize) -> Result<PromptWorkload> {
        if !self.prompt_token_counts().contains(&prompt_tokens) {
            return Err(Error::Unsupported(
                "prompt tokens outside source profile inventory".into(),
            ));
        }
        PromptWorkload::from_prompt_tokens(prompt_tokens)
    }
    pub fn validate_prompt_request(self, ids: &[u32], max_tokens: usize) -> Result<()> {
        self.prompt_workload(ids.len())?;
        PromptWorkload::validate_request(ids, max_tokens)
    }
    /// One prefill and each genuine decode frontier through the output limit.
    pub fn prepared_prompt_call_count(self) -> Result<usize> {
        self.prompt_token_counts()
            .len()
            .checked_mul(self.max_output_tokens())
            .ok_or_else(|| Error::Limit("profile prepared prompt call count".into()))
    }
    /// Distinct prefill layouts and one shared single-token cached layout.
    pub fn physical_prompt_scope_count(self) -> Result<usize> {
        self.prompt_token_counts()
            .len()
            .checked_add(1)
            .ok_or_else(|| Error::Limit("profile physical prompt scope count".into()))
    }
    pub fn max_cached_tokens(self) -> Result<usize> {
        self.prompt_token_counts()
            .iter()
            .max()
            .and_then(|tokens| tokens.checked_add(self.max_output_tokens().checked_sub(2)?))
            .ok_or_else(|| Error::Limit("profile cached token extent".into()))
    }
    pub fn hidden_size(self) -> usize {
        match self {
            Self::Qwen08 => 1024,
            Self::Qwen2B => 2048,
        }
    }
    pub fn intermediate_size(self) -> usize {
        match self {
            Self::Qwen08 => 3584,
            Self::Qwen2B => 6144,
        }
    }
    pub fn projection_input(self, kind: QwenProjection) -> usize {
        match kind {
            QwenProjection::GdnOut | QwenProjection::AttentionOut => 2048,
            QwenProjection::Down => self.intermediate_size(),
            _ => self.hidden_size(),
        }
    }
    pub fn projection_output(self, kind: QwenProjection) -> usize {
        match kind {
            QwenProjection::Qkvz => 8192,
            QwenProjection::Ba => 32,
            QwenProjection::AttentionQkv => 5120,
            QwenProjection::GateUp => self.intermediate_size() * 2,
            QwenProjection::Head => 248320,
            _ => self.hidden_size(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_prompt_profiles_preserve_admission_and_preparation_extents() {
        for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
            assert_eq!(profile.prompt_token_counts(), &[47, 57, 62, 67, 72]);
            assert_eq!(profile.prepared_prompt_call_count().unwrap(), 40);
            assert_eq!(profile.physical_prompt_scope_count().unwrap(), 6);
            assert_eq!(profile.max_cached_tokens().unwrap(), 78);
            assert_eq!(profile.max_output_tokens(), 8);
            assert_eq!(profile.context_capacity_tokens(), 512);
            for tokens in profile.prompt_token_counts() {
                let workload = profile.prompt_workload(*tokens).unwrap();
                assert_eq!(
                    workload.cached_bases().count() + 1,
                    profile.max_output_tokens()
                );
            }
            for tokens in [0, 46, 48, 49, 50, 51, 512] {
                assert!(profile.prompt_workload(tokens).is_err());
                assert!(
                    profile
                        .validate_prompt_request(&vec![11; tokens], 1)
                        .is_err()
                );
            }
            assert!(profile.validate_prompt_request(&[248320; 47], 1).is_err());
            assert!(profile.validate_prompt_request(&[11; 47], 9).is_err());
        }
    }
}
