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
pub use zerogravity_artifacts::{artifact, codecs, json_allocation, sharded};
pub mod artifacts;
pub mod builtins;
pub mod configuration;
pub mod control;
pub mod devices;
pub mod diagnostics;
pub mod embedding;
pub mod model;
pub mod models;
pub mod pipelines;
pub mod run;
pub mod runtime;
#[cfg(feature = "serve")]
pub mod serving;
pub mod smart_config;
pub use runtime::{Runtime, RuntimeBuilder};
pub use zerogravity_core::context_filter;
pub use zerogravity_core::context_snapshot;
pub use zerogravity_core::pipeline;
pub use zerogravity_core::run_policy;
pub use zerogravity_core::targets;
pub use zerogravity_core::{Error, ResourceLimit, Result, accelerator, session_root};
pub use zerogravity_core::{device_events, device_queue, resource_arbiter, state};
pub use zerogravity_core::{encoding, identity, numerics, placement, provider, semantic, task};
#[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
pub mod qwen35_cuda;
#[cfg(all(
    target_os = "linux",
    any(
        all(feature = "model-qwen35", feature = "hip"),
        all(feature = "model-qwen35", feature = "cuda")
    )
))]
mod qwen35_source;
#[cfg(feature = "cpu")]
pub use zerogravity_cpu as cpu;
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

/// Process-global tokenizer bootstrap and cold-constructor admission shared by
/// the CPU, GGUF and Qwen CUDA source owners.
#[cfg(feature = "text")]
pub mod tokenize_admission;

#[cfg(all(feature = "qwen27b-hip-qualification", target_os = "linux"))]
pub mod qwen27b_gpu;

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
