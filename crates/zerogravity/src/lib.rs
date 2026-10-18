//! Zerogravity's Rust-owned model source and reference execution foundation.
//!
//! This crate deliberately contains no foreign inference runtime, model server,
//! mandatory GPU driver requirement or fake token generation. Features are qualified
//! independently rather than inferred from a supported file suffix.
//! `model-gemma4` exposes source metadata contracts. Feature selection alone
//! does not qualify a runnable checkpoint or add a model execution command.

#[cfg(all(feature = "acquisition", target_os = "linux"))]
pub use zerogravity_artifacts::acquisition;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub use zerogravity_artifacts::cache;
pub use zerogravity_artifacts::{artifact, model, quant, sharded};
pub use zerogravity_core::context_filter;
pub use zerogravity_core::context_snapshot;
pub use zerogravity_core::pipeline;
pub use zerogravity_core::targets;
pub use zerogravity_core::{Error, Result, accelerator, session_root};
pub use zerogravity_core::{device_events, device_queue, resource_arbiter, state};
#[cfg(feature = "cpu")]
pub use zerogravity_cpu::{math, original_hf, weights};
#[cfg(all(feature = "cpu", feature = "cache", target_os = "linux"))]
pub mod original_qwen_cpu;
#[cfg(all(
    feature = "cpu",
    feature = "cache",
    feature = "model-qwen35",
    target_os = "linux"
))]
pub mod original_qwen_math;
#[cfg(all(
    feature = "cpu",
    feature = "cache",
    feature = "model-qwen35",
    target_os = "linux"
))]
pub mod original_qwen_precision;
#[cfg(all(
    feature = "cpu",
    feature = "cache",
    feature = "model-qwen35",
    target_os = "linux"
))]
pub mod original_qwen_state;
#[cfg(all(
    feature = "cpu",
    feature = "cache",
    feature = "model-qwen35",
    target_os = "linux"
))]
pub mod original_qwen_trace;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
pub use zerogravity_gpu as gpu;
#[cfg(feature = "model-gemma4")]
pub use zerogravity_model_gemma4 as gemma4;
#[cfg(feature = "model-qwen35")]
pub use zerogravity_model_qwen35 as qwen35;
#[cfg(feature = "model-qwen35")]
pub use zerogravity_model_qwen35::model_catalog;
#[cfg(feature = "text")]
pub use zerogravity_model_qwen35::tokenize;

#[cfg(all(feature = "model-qwen35", feature = "text"))]
pub mod prepare;

#[cfg(test)]
mod tests {
    #[test]
    fn facade_and_direct_core_share_one_physical_admission_authority() {
        use crate::resource_arbiter::{
            AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
        };
        let facade = ResourceArbiter::process_shared();
        let direct = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared();
        let key = PhysicalDeviceKey::new("host-test", "facade-core-single-authority").unwrap();
        direct.register(key.clone(), 64).unwrap();
        let lease = facade
            .admit(
                &key,
                ResourceOwner::new("same-model", "same-session").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ImmutableWeights,
                    bytes: 32,
                }],
            )
            .unwrap();
        assert_eq!(direct.snapshot(&key).unwrap().reserved, 32);
        drop(lease);
        assert_eq!(facade.snapshot(&key).unwrap().reserved, 0);
        assert_eq!(direct.snapshot(&key).unwrap().reserved, 0);
    }
}
