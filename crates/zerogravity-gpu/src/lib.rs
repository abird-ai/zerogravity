//! Source-exact resident GPU execution of the shared engine's typed schedule.
//! Provider features are explicit and optional. This library owns the one
//! physical queue/service/quarantine implementation; frontends supply immutable
//! source bindings and the already-recorded mathematical plan.

mod async_bridge;
mod attention_qk;
mod attention_softmax;
mod attention_value;
mod backend;
#[cfg(any(feature = "qualification", test))]
mod batch_attention;
#[cfg(any(feature = "qualification", test))]
mod batch_elementwise;
#[cfg(any(feature = "qualification", test))]
mod batch_kv;
mod batch_projection;
#[cfg(any(feature = "qualification", test))]
mod batch_recurrent;
#[cfg(all(feature = "qualification", target_os = "linux"))]
mod bf16_projection_kernel;
#[cfg(all(feature = "qualification", feature = "cuda", target_os = "linux"))]
mod cpu_bf16_projection;
mod device_activation;
mod elementwise;
mod embedding;
mod gdn_coefficients;
mod gdn_conv;
mod gdn_recurrent;
#[cfg(feature = "qualification")]
mod gpu_telemetry;
mod host_descriptor;
#[cfg(feature = "hardware-abort-probe")]
mod physical_abort_probe;
mod physical_quarantine;
mod physical_queue;
mod physical_service;
mod projection_f32;
mod quantization;
#[cfg(feature = "cuda")]
pub use physical_service::initialize_cuda_physical_identity;
#[cfg(any(feature = "cuda-native-qualification", test))]
mod prepared_kernel_key;
mod program_kernels;
mod residency;
pub mod resident_batch;
mod resident_batch_greedy;
mod resident_bindings;
mod resident_greedy;
#[cfg(feature = "qualification")]
mod resident_greedy_qualification;
mod resident_inference;
mod resident_leases;
pub use resident_leases::ResidentPhysical;
mod resident_program;
#[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
pub use resident_inference::bf16_text::{
    Bf16TextCandidate, Bf16TextPending, Bf16TextResult, Bf16TextSession, Bf16TextSnapshot,
};
#[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
pub use resident_program::bf16_text::{
    TextHostOwners, TextNativeOwners, TextObserver, TextResourceInventory, TextStorage, TextTraffic,
};
// Pure physical byte metadata; no allocation/backend/model eligibility permit.
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use resident_program::qwen2b_component::{ComponentOutput, ComponentRecipe};
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use resident_program::source_bindings::{SourceBindings, SourceSelection};
pub use resident_program::{ResidentBufferLayout, ResidentPhysicalType};
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use resident_program::{StateLayout, StateRequirement};
mod resident_provider;
mod resident_state_transport;
mod rms_norm;
mod rope_angles;
mod rope_kv;
mod silu;
mod source_accumulation;
#[cfg(any(
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
mod source_arithmetic;
#[cfg(any(
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
mod source_attention;
mod source_float;
#[cfg(any(
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
mod source_rms;
#[cfg(any(
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
mod source_rotary;
#[cfg(any(
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
mod source_sleef;
#[cfg(any(
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
mod source_standalone_exp;
mod source_transcendentals;
mod source_trig;
// Shared retirement has no source, provider or qualification dependency.
#[cfg_attr(not(feature = "cuda-native-qualification"), allow(dead_code))]
mod transfer_retirement;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "hip")]
pub mod hip;
#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "qualification")]
pub mod qualification;

#[cfg(feature = "cuda")]
pub use cubecl::cuda::{CudaDeviceMetadata, observe_cuda_devices};
pub use cubecl::{Device, device::WgpuDeviceKind};
pub use resident_inference::{
    InferenceReadback, PreparedInference, PreparedResidentCohort, ResidentModel, ResidentSession,
    ResidentSnapshot, ResidentToken,
};
pub use resident_provider::CubeResidentProvider;

#[cfg(feature = "cuda")]
mod ordered_silu;

#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
pub use cubecl::hip::{SourcePageEventProbeMode, SourcePageEventProbeSnapshot};
#[cfg(all(
    feature = "qwen-source-qualification",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use resident_inference::bf16_text_source::TextSourceFinalOwnerReceipt;
#[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
pub use resident_inference::bf16_text_source::{
    Bf16TextSourceCandidate, Bf16TextSourceGreedyPending, Bf16TextSourceGreedyResult,
    Bf16TextSourcePending, Bf16TextSourceResult, Bf16TextSourceSession, Bf16TextSourceSnapshot,
};
#[cfg(all(
    feature = "hip",
    feature = "hardware-abort-probe",
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    target_os = "linux"
))]
pub use resident_leases::source_page::SourcePageProbeParts;
#[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
pub use resident_program::bf16_text_source::{
    TextSourceFrame, TextSourceFrameKind, TextSourceObserver, TextSourceResourceInventory,
};

#[cfg(all(
    feature = "gemma-bf16-qualification",
    any(feature = "hip", feature = "vulkan"),
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use resident_inference::bf16_text::TextFinalOwnerReceipt;
#[cfg(all(
    feature = "gemma-bf16-qualification",
    feature = "vulkan",
    feature = "hip",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use resident_inference::bf16_text::{LearnedSamePciPeerReceipt, TextVulkanNativeRetention};
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    feature = "hip",
    feature = "vulkan",
    feature = "hardware-abort-probe"
))]
pub use resident_inference::{SamePciPeerReceipt, same_pci_peer};
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    feature = "vulkan",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use vulkan::native::retained_partial_native_modules;

#[cfg(all(
    feature = "hip",
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification",
    target_os = "linux"
))]
pub use hip::native::{HipSourceSetIdentity, SharedHipColdPlan, SharedHipNative};
