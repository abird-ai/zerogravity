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
mod resident_program;
mod resident_provider;
mod resident_state_transport;
mod rms_norm;
mod rope_angles;
mod rope_kv;
mod silu;
mod source_accumulation;
mod source_float;
mod source_transcendentals;
mod source_trig;

#[cfg(feature = "qualification")]
pub mod qualification;

pub use cubecl::{Device, device::WgpuDeviceKind};
pub use resident_inference::{
    InferenceReadback, PreparedInference, PreparedResidentCohort, ResidentModel, ResidentSession,
    ResidentSnapshot, ResidentToken,
};
pub use resident_provider::CubeResidentProvider;
