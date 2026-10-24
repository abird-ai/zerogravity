//! Explicit genuine-operand leaf proofs; never a model eligibility permit.
use crate::resident_leases::{ResidentLease, ResidentService};
use crate::vllm_cuda_gdn::{GdnBuffer, GdnStage, VllmGdnPhase, kernel};
use cubecl::{Device, bytes::Bytes};
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{
    ParameterWindow, VerifiedTensorSource, WindowWorkspace,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};
use zerogravity_core::{Error, Result};

const PROVIDER_CAP: usize = 128 * 1024 * 1024;

/// Synthetic predicate/range qualification only; no model or publisher exists.
#[cfg(target_os = "linux")]
pub async fn qualify_bf16_finite_cohorts(
    device: Device,
    host: &zerogravity_core::resource_arbiter::PhysicalDeviceKey,
    owner: ResourceOwner,
) -> Result<usize> {
    crate::resident_program::native::qualify_finite_cohorts(device, host, owner).await
}

#[cfg(target_os = "linux")]
pub use crate::resident_leases::{
    Qwen2bMlpWeightMode, Qwen2bMlpWeightOutcome, qualify_qwen2b_mlp_weights,
};

#[cfg(target_os = "linux")]
pub(crate) mod source_rotary;
#[cfg(target_os = "linux")]
pub use source_rotary::{
    RotaryComparison, RotaryMode, RotaryOutcome, RotaryOutput, qualify_short_hf_rotary_cubins,
    qualify_source_rotary_cubins,
};
#[cfg(target_os = "linux")]
mod tied_weight_upload;
#[cfg(target_os = "linux")]
pub use tied_weight_upload::{
    TiedWeightUploadMode, TiedWeightUploadOutcome, qualify_tied_weight_upload,
};

#[cfg(target_os = "linux")]
pub(crate) mod attention_cubin;
#[cfg(target_os = "linux")]
pub use attention_cubin::{
    CubinAttentionCase, CubinAttentionComparison, CubinAttentionExpected, CubinAttentionFault,
    CubinAttentionFaultOutcome, CubinAttentionFixture, CubinAttentionGatedFixture,
    CubinAttentionOutput, qualify_attention_cubin_fault, qualify_attention_cubins,
    qualify_attention_gated_cubin_fault, qualify_attention_gated_cubins,
};
#[cfg(target_os = "linux")]
mod activation_cubin;
#[cfg(target_os = "linux")]
mod activation_manifest;
#[cfg(target_os = "linux")]
pub(crate) mod canonical47_attention_projection;
#[cfg(target_os = "linux")]
pub(crate) mod canonical47_mlp;
#[cfg(target_os = "linux")]
pub(crate) mod qwen2b_mlp;
#[cfg(target_os = "linux")]
pub use activation_cubin::{
    CubinActivationComparison, CubinActivationFault, CubinActivationFaultOutcome,
    CubinActivationFixture, CubinActivationInputs, qualify_activation_cubin_fault,
    qualify_activation_cubins,
};
#[cfg(target_os = "linux")]
pub use activation_manifest::Activation as CubinActivation;
#[cfg(target_os = "linux")]
pub use canonical47_mlp::Mlp47QualificationInputs;
#[cfg(target_os = "linux")]
pub use qwen2b_mlp::Qwen2bMlpQualificationInputs;

#[cfg(feature = "vllm-cublas-draft")]
mod blas;
#[cfg(target_os = "linux")]
mod cubin_manifests;
#[cfg(target_os = "linux")]
pub(crate) mod short_hf_native;
#[cfg(target_os = "linux")]
pub use cohort_native::short_hf_cohort_source_scope_sha256;
pub use short_hf_native::{
    short_hf_native_provider_bytes, short_hf_native_provider_bytes_for_cohort,
    short_hf_required_images, short_hf_required_images_for_cohort,
};
mod gdn;
#[cfg(target_os = "linux")]
mod solve2b_leaf;
#[cfg(target_os = "linux")]
mod solve47_leaf;
#[cfg(target_os = "linux")]
pub use crate::vllm_cuda_gdn::{Qwen2bGdnArithmetic, Qwen2bOriginalGdnReference};
#[cfg(target_os = "linux")]
pub use solve2b_leaf::{
    Solve2bFault, Solve2bFaultOutcome, Solve2bFixture, Solve2bReadback, Solve2bReceipt,
    qualify_solve2b_owned_fault, qualify_solve2b_owned_leaves,
};
#[cfg(target_os = "linux")]
pub use solve47_leaf::{
    Solve47Fault, Solve47FaultOutcome, Solve47Fixture, Solve47Readback, Solve47Receipt,
    Solve47Variant, qualify_solve47_fault, qualify_solve47_leaves,
};
#[cfg(feature = "vllm-cublas-draft")]
mod layer;
mod layer_common;
#[cfg(target_os = "linux")]
mod mlp_cubin;
#[cfg(target_os = "linux")]
mod mlp_manifests;
#[cfg(target_os = "linux")]
pub(crate) mod native_plan;
#[cfg(target_os = "linux")]
pub use mlp_cubin::{
    CubinMlpComparison, CubinMlpFault, CubinMlpFaultOutcome, CubinMlpFixture,
    qualify_mlp_cubin_fault, qualify_mlp_cubins,
};
#[cfg(target_os = "linux")]
pub use mlp_manifests::MlpProjection as CubinMlpProjection;
#[cfg(target_os = "linux")]
mod layer_cubin;
#[cfg(target_os = "linux")]
mod projection_cubin;
#[cfg(target_os = "linux")]
mod qkvz_cubin;
#[cfg(target_os = "linux")]
pub use crate::vllm_cuda_input_rms::{
    Canonical47DualRmsLeaf, Canonical47RmsFault, Canonical47RmsFaultOutcome,
    Canonical47RmsReadback, Canonical47RmsReceipt, qualify_canonical47_rms,
    qualify_canonical47_rms_fault,
};
pub use crate::vllm_cuda_input_rms::{InputRmsFixture, qualify_input_rms};
#[cfg(target_os = "linux")]
pub use crate::vllm_cuda_input_rms::{
    Qwen2bDualRmsLeaf, Qwen2bRmsFault, Qwen2bRmsFaultOutcome, Qwen2bRmsReadback, Qwen2bRmsReceipt,
    qualify_qwen2b_rms, qualify_qwen2b_rms_fault,
};
#[cfg(feature = "vllm-cublas-draft")]
pub use blas::{ProjectionFixture, qualify_projections};
#[cfg(target_os = "linux")]
pub use cubin_manifests::{Projection as CubinProjection, ProjectionImage as CubinImageKind};
pub use gdn::{GdnComparison, GdnFixture, qualify_gdn};
pub use gdn::{GdnFault, GdnFaultOutcome, qualify_gdn_fault};
#[cfg(feature = "vllm-cublas-draft")]
pub use layer::{LayerComparison, qualify_layer};
pub use layer_common::{LayerExpectedRecord, LayerFixture, LayerOutput};
#[cfg(target_os = "linux")]
pub use layer_cubin::{
    LayerCubinComparison, LayerMlpCubinFault, LayerMlpCubinFaultOutcome, LayerMlpCubinInputs,
    PostnormCubinFault, PostnormCubinFaultOutcome, qualify_first_sequence_cubins,
    qualify_layer_cubins, qualify_layer_mlp_cubin_fault, qualify_layer_mlp_cubins,
    qualify_layer_postnorm_cubin_fault, qualify_layer_postnorm_cubins,
};
#[cfg(target_os = "linux")]
pub use projection_cubin::{
    CubinActivationImage, CubinImage, CubinProjectionComparison, CubinProjectionFault,
    CubinProjectionFaultOutcome, CubinProjectionFixture, CubinQualificationInputs,
    qualify_projection_cubin_fault, qualify_projection_cubins,
};
#[cfg(target_os = "linux")]
pub use qkvz_cubin::{QkvzCubinComparison, QkvzCubinFixture, QkvzCubinImage, qualify_qkvz_cubin};

/// ROOT authenticates these complete actual records before calling this seam.
/// Genuine core/Z are allowed ONLY in this separately labelled leaf proof.
pub struct GatedNormFixture {
    pub core: Vec<u8>,
    pub gate: Vec<u8>,
    pub packed_qkvz: Vec<u8>,
    pub expected: Vec<u8>,
}

struct LeafOwner {
    _source: VerifiedTensorSource,
    _parameter: Arc<ParameterWindow>,
    inputs: Vec<(GdnBuffer, Vec<u8>)>,
    _transport_and_readback: WindowWorkspace,
}

fn finite_bf16(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(2)
        && bytes.chunks_exact(2).all(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            bits & 0x7f80 != 0x7f80
        })
}

fn loaded_gamma(window: &ParameterWindow) -> Result<Vec<u8>> {
    if !window.is_complete() || window.parameter().elements() != 128 {
        return Err(Error::Malformed("complete original GDN gamma128".into()));
    }
    let values = match window.parameter().tensor().kind {
        QuantKind::Bf16 => window.as_slice().to_vec(),
        QuantKind::F32 => {
            let mut out = Vec::with_capacity(256);
            for raw in window.as_slice().chunks_exact(4) {
                let bits = u32::from_le_bytes(raw.try_into().unwrap());
                if !f32::from_bits(bits).is_finite() {
                    return Err(Error::Malformed("nonfinite original GDN gamma".into()));
                }
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                out.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
            }
            out
        }
        _ => return Err(Error::Unsupported("pinned vLLM gamma source dtype".into())),
    };
    if values.len() != 256 || !finite_bf16(&values) {
        return Err(Error::Malformed("finite loaded BF16 GDN gamma".into()));
    }
    Ok(values)
}

async fn gated_leaf(
    service: Arc<ResidentService>,
    source: VerifiedTensorSource,
    parameter: Arc<ParameterWindow>,
    phase: VllmGdnPhase,
    fixture: GatedNormFixture,
    correlation: u64,
) -> Result<usize> {
    let spec = kernel(phase, GdnStage::Gated)?;
    let bytes = phase.tokens() * 4096;
    if fixture.core.len() != bytes
        || fixture.gate.len() != bytes
        || fixture.expected.len() != bytes
        || ![&fixture.core, &fixture.gate, &fixture.expected]
            .into_iter()
            .all(|b| finite_bf16(b))
    {
        return Err(Error::Malformed(
            "actual complete gated RMS leaf operands/output".into(),
        ));
    }
    let gate_role = if phase.tokens() == 1 {
        GdnBuffer::PackedQkvz
    } else {
        GdnBuffer::ZCompact
    };
    let gate = if phase.tokens() == 1 {
        if fixture.packed_qkvz.len() != 16384 || fixture.packed_qkvz[12288..] != fixture.gate {
            return Err(Error::Malformed(
                "actual original packed QKVZ/Z alias".into(),
            ));
        }
        fixture.packed_qkvz
    } else {
        if !fixture.packed_qkvz.is_empty() {
            return Err(Error::Malformed(
                "prefill Z requires its real compact view".into(),
            ));
        }
        fixture.gate
    };
    let inputs = vec![
        (GdnBuffer::Core, fixture.core),
        (gate_role, gate),
        (GdnBuffer::GatedGamma, loaded_gamma(&parameter)?),
    ];
    let physical_bytes = spec.bindings.iter().try_fold(0usize, |total, binding| {
        total
            .checked_add(binding.offset + binding.physical_bytes)
            .ok_or_else(|| Error::Limit("gated leaf extent overflow".into()))
    })?;
    let host_bytes = inputs.iter().map(|(_, raw)| raw.len() * 2).sum::<usize>()
        + physical_bytes
        + fixture.expected.len();
    let owner = Arc::new(LeafOwner {
        _transport_and_readback: source.reserve_workspace(host_bytes)?,
        _source: source,
        _parameter: parameter,
        inputs,
    });
    if service.remaining_capacity()? < physical_bytes {
        return Err(Error::Limit("gated leaf capacity before allocation".into()));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: physical_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(correlation)?;
    init.retain_completion_owner(owner.clone())?;
    let mut leases: Vec<(GdnBuffer, ResidentLease, DeviceAccess)> = Vec::new();
    for binding in &spec.bindings {
        if leases.iter().any(|(role, _, _)| *role == binding.role) {
            return Err(Error::State("gated leaf duplicate owning storage".into()));
        }
        let extent = binding.offset + binding.physical_bytes;
        let (lease, access) =
            if let Some((_, raw)) = owner.inputs.iter().find(|(role, _)| *role == binding.role) {
                if raw.len() != extent || !binding.io.is_readable() || binding.io.is_writable() {
                    return Err(Error::State(
                        "gated leaf original read-only extent/access".into(),
                    ));
                }
                (
                    init.upload_bytes(raw, BufferRole::Activation)?,
                    DeviceAccess::Read,
                )
            } else {
                if binding.offset != 0 || binding.io.is_readable() || !binding.io.is_writable() {
                    return Err(Error::State(
                        "gated leaf complete output extent/access".into(),
                    ));
                }
                (
                    init.allocate_empty(extent, BufferRole::Output)?,
                    DeviceAccess::Write,
                )
            };
        leases.push((binding.role, lease, access));
    }
    let initialized = init.complete_sync().await?;
    slot.finish()?;
    drop(initialized);
    let handles = spec
        .bindings
        .iter()
        .zip(&leases)
        .map(|(binding, (_, lease, _))| {
            lease
                .handle()
                .offset_start(binding.offset as u64)
                .offset_end((lease.bytes() - binding.offset - binding.physical_bytes) as u64)
        })
        .collect::<Vec<_>>();
    spec.kernel
        .validate(service.client(), spec.grid, &handles)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: physical_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let accesses = leases
        .iter()
        .map(|(_, l, a)| (l.clone(), *a))
        .collect::<Vec<_>>();
    let outputs = leases
        .iter()
        .filter(|(_, _, a)| *a == DeviceAccess::Write)
        .map(|(_, l, _)| l.clone())
        .collect::<Vec<_>>();
    let reads = leases
        .iter()
        .map(|(_, l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let mut flight = service.submit(correlation, &accesses)?;
    flight.retain_completion_owner(owner.clone())?;
    spec.kernel.launch(service.client(), spec.grid, &handles)?;
    let (ready, readback) = flight
        .complete_with_readback(&outputs, &reads, |actual: &[Bytes]| {
            for ((role, _, access), raw) in leases.iter().zip(actual) {
                if *role == GdnBuffer::Gated {
                    if &raw[..] != fixture.expected.as_slice() {
                        let first = raw
                            .iter()
                            .zip(&fixture.expected)
                            .position(|(a, b)| a != b)
                            .unwrap_or(0);
                        return Err(Error::State(format!(
                            "pinned vLLM gated RMS differs at byte{first}"
                        )));
                    }
                } else if *access == DeviceAccess::Read {
                    let original = &owner.inputs.iter().find(|(r, _)| r == role).unwrap().1;
                    if &raw[..] != original.as_slice() {
                        return Err(Error::State(
                            "pinned vLLM gated RMS modified original input".into(),
                        ));
                    }
                }
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    drop((
        ready, readback, handles, reads, accesses, outputs, leases, owner,
    ));
    Ok(bytes / 2)
}

/// Five actual stock V1 calls in their observed order. This proves gated norm
/// only; a connected trained layer must derive core/Z from its own producers.
pub async fn qualify_gated_norm(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [GatedNormFixture; 5],
    device_cap: u64,
) -> Result<usize> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release CUDA required for vLLM physical leaf".into(),
        ));
    }
    let parameter = source.parameter("model.language_model.layers.0.linear_attn.norm.weight")?;
    let parameter = Arc::new(source.read_parameter(&parameter, 0..128)?);
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("vllm08-cuda-leaf", "gated-rms")?,
        PROVIDER_CAP,
        device_cap,
    )?;
    let key = service.physical_key();
    let phases = [
        VllmGdnPhase::InitialOne,
        VllmGdnPhase::CachedDecode,
        VllmGdnPhase::Initial64,
        VllmGdnPhase::ContinuedTwo,
        VllmGdnPhase::CachedDecode,
    ];
    let mut compared = 0;
    for (index, (phase, fixture)) in phases.into_iter().zip(fixtures).enumerate() {
        compared += gated_leaf(
            service.clone(),
            source.clone(),
            parameter.clone(),
            phase,
            fixture,
            index as u64 + 1,
        )
        .await?;
        if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
            return Err(Error::State(
                "vLLM leaf retained ordinary completed allocations/events".into(),
            ));
        }
    }
    drop(service);
    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != 0 {
        return Err(Error::State(
            "vLLM leaf original physical envelope did not retire".into(),
        ));
    }
    Ok(compared)
}

#[cfg(target_os = "linux")]
pub(crate) mod native_gdn;
#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    VllmGdnFragmentFault, VllmGdnFragmentFaultOutcome, VllmGdnFragmentReadback,
    VllmGdnFragmentReceipt, VllmGdnSequenceFault, VllmGdnSequenceFaultOutcome,
    VllmGdnSequenceReceipt, VllmGdnSequenceRestoreReceipt, qualify_vllm_gdn_fragment,
    qualify_vllm_gdn_fragment_fault, qualify_vllm_gdn_sequence, qualify_vllm_gdn_sequence_fault,
    qualify_vllm_gdn_sequence_restore_refusal,
};

#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    qualify_vllm_gdn_mlp_fragment, qualify_vllm_gdn_mlp_fragment_fault,
};

#[cfg(target_os = "linux")]
pub(crate) mod vocabulary_head;
#[cfg(target_os = "linux")]
pub use vocabulary_head::{
    FinalNormInputs, VocabularyHeadCase, VocabularyHeadComparison, VocabularyHeadExpected,
    VocabularyHeadFault, VocabularyHeadFaultOutcome, VocabularyHeadFixture, VocabularyHeadFixtures,
    VocabularyHeadOutput, VocabularyHeadScope, qualify_final_norm_head_cubins,
    qualify_vocabulary_head_cubins, qualify_vocabulary_head_fault,
};

#[cfg(target_os = "linux")]
pub(crate) mod canonical47_model;
#[cfg(target_os = "linux")]
pub(crate) mod canonical47_model_native;
#[cfg(target_os = "linux")]
pub(crate) mod canonical47_model_prefix;
#[cfg(target_os = "linux")]
pub(crate) mod full_model_images;
#[cfg(target_os = "linux")]
pub(crate) mod model_cache;
#[cfg(target_os = "linux")]
pub(crate) mod model_gdn;
#[cfg(target_os = "linux")]
pub(crate) mod model_kv_transport;
#[cfg(target_os = "linux")]
pub(crate) mod model_native;
#[cfg(target_os = "linux")]
pub(crate) mod model_values;
#[cfg(target_os = "linux")]
pub(crate) mod model_weights;

#[cfg(target_os = "linux")]
pub(crate) mod full_model_budget;
#[cfg(target_os = "linux")]
pub(crate) mod model_library;
#[cfg(target_os = "linux")]
pub(crate) mod model_rotary;
#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    Vllm08DomainReadback, Vllm08ModelFault, Vllm08ModelFaultOutcome, Vllm08ModelInputs,
    Vllm08ModelReadback, Vllm08ModelReceipt, qualify_vllm08_model, qualify_vllm08_model_fault,
};
#[cfg(target_os = "linux")]
pub use full_model_budget::vllm08_model_host_bound;

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub use crate::resident_inference::{
    Vllm08NaturalFault, Vllm08NaturalFaultExecution, Vllm08NaturalFaultOutcome,
    Vllm08NaturalReceipt, Vllm08NaturalReuseFault, Vllm08NaturalReuseFaultExecution,
    Vllm08NaturalReuseFaultOutcome, Vllm08NaturalReuseReceipt, Vllm08NaturalReuseRound,
    qualify_vllm08_natural_fault, qualify_vllm08_natural_model, qualify_vllm08_natural_reuse,
    qualify_vllm08_natural_reuse_fault,
};

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub use crate::resident_inference::qualify_vllm08_natural_reuse_pressure;

#[cfg(target_os = "linux")]
pub(crate) mod qwen2b_projection;
#[cfg(target_os = "linux")]
pub use qwen2b_projection::{
    Qwen2bProjectionFault, Qwen2bProjectionFaultOutcome, Qwen2bProjectionFixture,
    Qwen2bProjectionReadback, Qwen2bProjectionReceipt, qualify_qwen2b_projection_fault,
    qualify_qwen2b_projections,
};

#[cfg(target_os = "linux")]
pub(crate) mod qwen2b_component;
#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    Qwen2bGdnComponentFault, Qwen2bGdnComponentFaultOutcome, Qwen2bGdnComponentReadback,
    Qwen2bGdnComponentReceipt, qualify_qwen2b_gdn_component,
    qualify_qwen2b_gdn_component_arithmetic, qualify_qwen2b_gdn_component_fault,
};

#[cfg(target_os = "linux")]
pub(crate) mod canonical47_projection;
#[cfg(target_os = "linux")]
pub use canonical47_projection::{
    Canonical47ProjectionFault, Canonical47ProjectionFaultOutcome, Canonical47ProjectionFixture,
    Canonical47ProjectionReadback, Canonical47ProjectionReceipt,
    qualify_canonical47_projection_fault, qualify_canonical47_projections,
};

#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    Canonical47GdnComponentFault, Canonical47GdnComponentFaultOutcome,
    Canonical47GdnComponentReadback, Canonical47GdnComponentReceipt,
    qualify_canonical47_gdn_component, qualify_canonical47_gdn_component_fault,
};
#[cfg(target_os = "linux")]
pub use crate::vllm_cuda_gdn::{Canonical47GdnArithmetic, Canonical47GdnIntermediate};

#[cfg(target_os = "linux")]
pub(crate) mod post_attention_projection;

#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    PostAttentionArithmetic, PostAttentionComponentFault, PostAttentionComponentFaultOutcome,
    PostAttentionComponentReadback, PostAttentionComponentReceipt,
    qualify_post_attention_component, qualify_post_attention_component_fault,
    qualify_qwen2b_post_attention_component_arithmetic,
};

#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    MlpComponentFault, MlpComponentFaultOutcome, MlpComponentReadback, MlpComponentReceipt,
    qualify_mlp_component, qualify_mlp_component_fault,
};

#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    Canonical47ModelPrefixFault, Canonical47ModelPrefixFaultOutcome,
    Canonical47ModelPrefixReadback, Canonical47ModelPrefixReceipt,
    Canonical47QkKvQualificationInputs, qualify_canonical47_model_fa_prefix,
    qualify_canonical47_model_fa_prefix_fault, qualify_canonical47_model_kv_prefix,
    qualify_canonical47_model_kv_prefix_fault, qualify_canonical47_model_layer3_prefix,
    qualify_canonical47_model_layer3_prefix_fault, qualify_canonical47_model_prefix,
    qualify_canonical47_model_prefix_fault, qualify_canonical47_model_qkv_prefix,
    qualify_canonical47_model_qkv_prefix_fault,
};

#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    Canonical47FullModelFault, Canonical47FullModelFaultOutcome, Canonical47FullModelReadback,
    Canonical47FullModelReceipt, qualify_canonical47_full_model,
    qualify_canonical47_full_model_fault,
};
#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    Qwen2bMlpComponentFault, Qwen2bMlpComponentFaultOutcome, Qwen2bMlpComponentReceipt,
    Qwen2bMlpComponentSource, qualify_qwen2b_mlp_component, qualify_qwen2b_mlp_component_fault,
};

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub use crate::resident_inference::{
    Canonical47CallReceipt, Canonical47CloseReceipt, Canonical47EmptySnapshot,
    Canonical47LifecycleProfile, Canonical47Loaded, Canonical47LoadedObservation,
    Canonical47LoadedSnapshot, Canonical47NaturalFault, Canonical47NaturalFaultOutcome,
    Canonical47NaturalReadback, Canonical47Pending, Canonical47Session,
    canonical47_natural_host_bound, prepare_canonical47_natural,
    prepare_canonical47_natural_with_profile,
};

#[cfg(target_os = "linux")]
pub(crate) mod short_hf_head;
#[cfg(target_os = "linux")]
pub(crate) mod short_hf_model;

#[cfg(target_os = "linux")]
pub use crate::resident_inference::{
    ShortHfCallReceipt, ShortHfCloseReceipt, ShortHfCohortFault, ShortHfCohortFaultOutcome,
    ShortHfCohortFaultRetention, ShortHfCohortPreparation, ShortHfCohortReadback,
    ShortHfCohortReceipt, ShortHfCohortRowReceipt, ShortHfEmptySnapshot, ShortHfFault,
    ShortHfFaultOutcome, ShortHfLoaded, ShortHfLoadedObservation, ShortHfLoadedSnapshot,
    ShortHfPending, ShortHfPendingCohort, ShortHfQualificationInputs, ShortHfReadback,
    ShortHfResourceInventory, ShortHfSession, prepare_short_hf, prepare_short_hf_b4,
};

#[cfg(target_os = "linux")]
pub(crate) mod prepared_bf16_head;

pub(crate) mod cohort_native;
