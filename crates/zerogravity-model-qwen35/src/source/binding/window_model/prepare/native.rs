//! Independent reference capture expectations; native catalogue/ABI remains GPU-owned.
use super::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceKernelCapture {
    recipe_sha256: &'static str,
}
impl ReferenceKernelCapture {
    /// Inventory selection only; the source factory still authenticates execution.
    pub fn for_profile(profile: QwenSourceProfile) -> Self {
        Self {
            recipe_sha256: match profile {
                QwenSourceProfile::Qwen08 => {
                    "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d"
                }
                QwenSourceProfile::Qwen2B => {
                    "4d77cb6637843a57a17fc093ad1e5241ddf083bf8d9ade0c9bbaa118bc9448e2"
                }
            },
        }
    }
    pub fn recipe_sha256(self) -> &'static str {
        self.recipe_sha256
    }
}
impl BoundFullModelSource {
    pub fn native_kernel_capture(&self) -> ReferenceKernelCapture {
        // Exact source/profile was already joined by bind_full_model_source.
        ReferenceKernelCapture::for_profile(self.model_profile())
    }
}
