//! Source-exact resident GPU execution of the shared engine's typed schedule.
//! Provider features are explicit and optional. This library owns the one
//! physical queue/service/quarantine implementation; frontends supply immutable
//! source bindings and the already-recorded mathematical plan.

mod activation_quant;
mod activation_quant_fused;
#[cfg(feature = "qualification")]
mod activation_stress;
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
mod batch_quant_projection;
#[cfg(feature = "qualification")]
mod batch_quant_qualification;
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
mod mixed_ops;
#[cfg(feature = "qualification")]
mod model_bench;
#[cfg(feature = "qualification")]
mod model_layer;
#[cfg(feature = "qualification")]
mod model_mixed;
#[cfg(feature = "qualification")]
mod model_q4;
#[cfg(feature = "qualification")]
mod model_resident;
#[cfg(feature = "qualification")]
mod model_sessions;
#[cfg(feature = "qualification")]
mod model_workers;
#[cfg(feature = "qualification")]
mod operator_stress;
#[cfg(feature = "hardware-abort-probe")]
mod physical_abort_probe;
mod physical_quarantine;
mod physical_queue;
mod physical_service;
#[cfg(feature = "cuda")]
pub use physical_service::{initialize_cuda_physical_identity, validate_cuda_compiler_environment};
#[cfg(any(feature = "vllm-cuda-qualification", test))]
mod prepared_kernel_key;
mod program_kernels;
#[cfg(feature = "qualification")]
mod q4;
mod q4_q8;
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
pub use resident_inference::gemma_text::{
    Gemma4Bf16GpuCall, Gemma4Bf16GpuCandidate, Gemma4Bf16GpuPending, Gemma4Bf16GpuSession,
    Gemma4Bf16GpuSnapshot,
};
#[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
pub use resident_program::gemma_text::{
    GemmaTextHostOwners, GemmaTextNativeOwners, GemmaTextObserver, GemmaTextResourceInventory,
    GemmaTextStorage, GemmaTextTraffic,
};
// Pure physical byte metadata; no allocation/backend/model eligibility permit.
pub use resident_program::{ResidentBufferLayout, ResidentPhysicalType};
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
#[cfg_attr(not(feature = "vllm-cuda-qualification"), allow(dead_code))]
mod transfer_retirement;

#[cfg(feature = "vllm-cuda-qualification")]
mod vllm_cuda_gdn;
#[cfg(feature = "vllm-cuda-qualification")]
mod vllm_cuda_input_rms;
#[cfg(feature = "vllm-cuda-qualification")]
pub mod vllm_cuda_qualification;
#[cfg(feature = "vllm-cuda-qualification")]
mod vllm_ptx_inline;

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

#[cfg(all(test, feature = "qualification", feature = "hip", target_os = "linux"))]
mod hip_native_qualification;

#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
pub use cubecl::hip::{SourcePageEventProbeMode, SourcePageEventProbeSnapshot};
#[cfg(all(
    feature = "qwen-source-qualification",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use resident_inference::qwen_source::QwenFinalOwnerReceipt;
#[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
pub use resident_inference::qwen_source::{
    Qwen27bNaturalCall, Qwen27bNaturalPending, Qwen27bSourceGpuCandidate, Qwen27bSourceGpuOutput,
    Qwen27bSourceGpuPending, Qwen27bSourceGpuSession, Qwen27bSourceGpuSnapshot,
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
pub use resident_program::qwen_source::{
    QwenSourceFrame, QwenSourceFrameKind, QwenSourceObserver, QwenSourceResourceInventory,
};

#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    feature = "vulkan",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use backend::vulkan_native_policy::retained_partial_native_modules;
#[cfg(all(
    feature = "gemma-bf16-qualification",
    any(feature = "hip", feature = "vulkan"),
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use resident_inference::gemma_text::GemmaFinalOwnerReceipt;
#[cfg(all(
    feature = "gemma-bf16-qualification",
    feature = "vulkan",
    feature = "hip",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use resident_inference::gemma_text::{GemmaVulkanNativeRetention, LearnedSamePciPeerReceipt};
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
    feature = "hip",
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification",
    target_os = "linux"
))]
pub use backend::hip_native_policy::{HipSourceSetIdentity, SharedHipColdPlan, SharedHipNative};
