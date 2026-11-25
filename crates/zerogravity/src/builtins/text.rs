//! Source-specific text composition. No scheduler, root publisher or duplicate math.
use crate::resource_arbiter::{AllocationClass, GlobalResourceLease};
use crate::tokenize::QwenTokenizer;
use std::alloc::Layout;
use std::path::Path;
use std::sync::{Arc, atomic::AtomicUsize};
use zerogravity_engine::model_instance::TextTokenizer;
use zerogravity_engine::router::{ServiceError, pending::ServingHost};

mod source;
pub use source::read_approved_tokenizer;

#[cfg(any(
    all(feature = "cpu", feature = "cache"),
    feature = "vulkan",
    feature = "cuda",
    feature = "hip"
))]
mod local;
#[cfg(any(
    all(feature = "cpu", feature = "cache"),
    feature = "vulkan",
    feature = "cuda",
    feature = "hip"
))]
pub(crate) use local::SourceDeclarations;

/// Retain the actual tokenizer/source owner, including its original admission.
/// Admit the new wrapper in the same host domain before allocating its Arc.
pub fn tokenizer_owner<T: Send + Sync + 'static>(
    owner: T,
    tokenizer: fn(&T) -> &QwenTokenizer,
    host: &ServingHost,
    alias: &str,
) -> Result<Arc<dyn TextTokenizer>, ServiceError> {
    let charge = host.admit(
        alias,
        "tokenizer-adapter",
        AllocationClass::ImmutableWeights,
        tokenizer_owner_bytes::<T>()?,
    )?;
    Ok(Arc::new(TokenizerOwner {
        owner,
        tokenizer,
        _adapter_charge: charge,
    }))
}
struct TokenizerOwner<T> {
    owner: T,
    tokenizer: fn(&T) -> &QwenTokenizer,
    // Retire the actual parent/tokenizer before releasing its wrapper credit.
    _adapter_charge: GlobalResourceLease,
}

const ADAPTER_ALLOC_SLACK: usize = 64;
fn tokenizer_owner_bytes<T>() -> Result<u64, ServiceError> {
    arc_owner_bytes::<TokenizerOwner<T>>()
}
pub(crate) fn arc_owner_bytes<T>() -> Result<u64, ServiceError> {
    let overflow = || ServiceError::new(400, "size_overflow", "source adapter layout overflow");
    // Pinned std Arc: strong/weak counters, then the aligned value. The host's
    // original lease envelope separately prices reservation/owner bookkeeping.
    let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map_err(|_| overflow())?;
    layout
        .pad_to_align()
        .size()
        .checked_add(ADAPTER_ALLOC_SLACK)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(overflow)
}
impl<T: Send + Sync> TextTokenizer for TokenizerOwner<T> {
    fn eos(&self) -> u32 {
        (self.tokenizer)(&self.owner).eos()
    }
    fn decode(&self, tokens: &[u32], skip_special: bool) -> crate::Result<String> {
        (self.tokenizer)(&self.owner).decode(tokens, skip_special)
    }
    fn prompt_tokens(&self, input: &str, chat: bool) -> crate::Result<Vec<u32>> {
        (self.tokenizer)(&self.owner).prompt_tokens(input, chat)
    }
    fn prompt_token_bound(&self, input_bytes: usize, chat: bool) -> usize {
        (self.tokenizer)(&self.owner).prompt_token_bound(input_bytes, chat)
    }
    fn admitted_prompt_scratch_bound(&self, input_bytes: usize, chat: bool) -> crate::Result<u64> {
        (self.tokenizer)(&self.owner).admitted_prompt_scratch_bound(input_bytes, chat)
    }
    fn decode_scratch_bound(&self, tokens: usize) -> usize {
        (self.tokenizer)(&self.owner).decode_scratch_bound(tokens)
    }
}

#[cfg(all(feature = "cpu", feature = "cache"))]
mod cpu;
#[cfg(all(feature = "cpu", feature = "cache"))]
pub use cpu::load_router;
#[cfg(all(feature = "cpu", feature = "cache"))]
pub(crate) use cpu::load_router_declared;

#[cfg(all(test, feature = "cpu", feature = "cache"))]
mod tests;
#[cfg(test)]
pub(crate) mod tokenizer_tests;

#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
pub(crate) mod gpu;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
pub use gpu::{GpuProfile, load_gpu_router};

#[cfg(all(
    any(
        all(feature = "model-qwen35", feature = "hip"),
        all(feature = "model-qwen35", feature = "cuda")
    ),
    target_os = "linux"
))]
pub(crate) mod qwen_source;
#[cfg(all(
    any(
        all(feature = "model-qwen35", feature = "hip"),
        all(feature = "model-qwen35", feature = "cuda")
    ),
    target_os = "linux"
))]
pub use qwen_source::QwenServingProfile;
