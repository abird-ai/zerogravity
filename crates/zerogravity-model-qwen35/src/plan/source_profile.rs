//! Original source-family geometry.
//! SourceIdentity and ProgramNumericalProfile/provider semantics stay independent.
//! No source/runtime/state authority or learned-source substitution.
use super::{PromptWorkload, QwenProjection};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenSourceProfile {
    Qwen08,
    Qwen2B,
}
impl QwenSourceProfile {
    pub const fn vocabulary_size(self) -> usize {
        248_320
    }
    pub const fn layer_count(self) -> usize {
        24
    }
    pub const fn max_position_embeddings(self) -> usize {
        262_144
    }
    pub fn prompt_workload(
        self,
        prompt_tokens: usize,
        context_capacity_tokens: usize,
    ) -> Result<PromptWorkload> {
        let workload = PromptWorkload::new(prompt_tokens, context_capacity_tokens)?;
        self.validate_prompt_workload(workload)?;
        Ok(workload)
    }
    pub fn validate_prompt_workload(self, workload: PromptWorkload) -> Result<()> {
        if workload.context_capacity_tokens() > self.max_position_embeddings() {
            return Err(Error::Unsupported(
                "prompt context exceeds authenticated model geometry".into(),
            ));
        }
        Ok(())
    }
    pub fn validate_prompt_request(
        self,
        ids: &[u32],
        max_tokens: usize,
        context_capacity_tokens: usize,
    ) -> Result<()> {
        let workload = self.prompt_workload(ids.len(), context_capacity_tokens)?;
        workload.validate_request(ids, max_tokens, self.vocabulary_size())
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
            QwenProjection::Head => self.vocabulary_size(),
            _ => self.hidden_size(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_geometry_accepts_dynamic_spans_but_not_invalid_context_or_ids() {
        for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
            assert_eq!(profile.vocabulary_size(), 248_320);
            assert_eq!(profile.layer_count(), 24);
            assert_eq!(profile.max_position_embeddings(), 262_144);
            for tokens in [1, 47, 48, 49, 50, 73, 511] {
                let workload = profile.prompt_workload(tokens, 512).unwrap();
                assert_eq!(workload.prompt_tokens(), tokens);
                assert_eq!(workload.context_capacity_tokens(), 512);
                profile
                    .validate_prompt_request(&vec![11; tokens], 512 - tokens, 512)
                    .unwrap();
            }
            for (tokens, outputs, context) in [(0, 1, 512), (512, 1, 512), (1, 0, 512)] {
                assert!(
                    profile
                        .validate_prompt_request(&vec![11; tokens], outputs, context)
                        .is_err()
                );
            }
            assert!(profile.validate_prompt_request(&[248_320], 1, 512).is_err());
            assert!(
                profile
                    .prompt_workload(1, profile.max_position_embeddings() + 1)
                    .is_err()
            );
            let full = PromptWorkload::new(
                profile.max_position_embeddings(),
                profile.max_position_embeddings(),
            )
            .unwrap();
            assert!(full.checked_frontier(1).is_err());
        }
    }
}
