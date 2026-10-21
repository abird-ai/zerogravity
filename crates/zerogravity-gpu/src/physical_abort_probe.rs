//! P07-G3 source-real *physical* Future abandonment proof.
//!
//! Opt-in by --features hardware-abort-probe and CLI model-abort-vulkan-*
//! or model-abort-{norm,grouped,conv,recurrent}-vulkan-*.
//! This code NEVER compiles into the normal CubeCL/CPU app. It deliberately
//! submits one authentic shader and drops the Rust Future before readback.
//! The result must be a terminal global fence and process-lifetime original
//! GPU Handle quarantine. A submission is NOT confirmation of execution
//! completion: this intentionally cannot produce a Ready or published Root.

use crate::resident_provider::CubeResidentProvider;
use cubecl::Device;
use std::{
    future::{Future, pending},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
};
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::{
    AttentionQkExecution, AttentionQkRegion, AttentionSoftmaxExecution, AttentionSoftmaxRegion,
    AttentionValueRegion, BinaryRegion, EmbeddingRegion, GdnCoefficientExecution,
    GdnCoefficientRegion, GdnConvExecution, GdnConvRegion, GdnRecurrentExecution,
    GdnRecurrentRegion, GroupedRmsNormRegion, MatVecKernel, MatVecRegion, RegionExecution,
    RmsNormRegion, RopeKvExecution, RopeKvRegion, SiluRegion,
};
use zerogravity_core::resource_arbiter::{ResourceArbiter, ResourceOwner};
use zerogravity_model_qwen35::QwenDenseCpu;

static ARM_ONE_SUBMISSION: AtomicBool = AtomicBool::new(false);
static ACTUAL_LAUNCH_RETURNED: AtomicBool = AtomicBool::new(false);

/// Called only after the actual CubeCL kernel launch, before read_async.
/// The artificial pending state prevents a false completion receipt; it does
/// not fake a device launch or driver-level kernel completion.
pub async fn after_physical_launch() {
    if ARM_ONE_SUBMISSION.swap(false, Ordering::SeqCst) {
        ACTUAL_LAUNCH_RETURNED.store(true, Ordering::SeqCst);
        pending::<()>().await;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AbortOperator {
    SourceMatVec,
    SourceDeviceQuant,
    RmsNorm,
    GroupedRmsNorm,
    GdnL2,
    GdnCoefficients,
    RopeKv,
    AttentionQk,
    AttentionSoftmax,
    AttentionValue,
    Embedding,
    ResidualAdd,
    AttentionGate,
    FfnSilu,
    Silu,
    SiluGate,
    GdnConv,
    GdnRecurrent,
}

impl AbortOperator {
    fn name(self) -> &'static str {
        match self {
            Self::SourceMatVec => "source_gemv",
            Self::SourceDeviceQuant => "source_device_quant",
            Self::RmsNorm => "rms_norm",
            Self::GroupedRmsNorm => "grouped_rms_norm",
            Self::GdnL2 => "gdn_qk_l2",
            Self::GdnCoefficients => "gdn_coefficients",
            Self::RopeKv => "rope_kv",
            Self::AttentionQk => "attention_qk",
            Self::AttentionSoftmax => "attention_softmax",
            Self::AttentionValue => "attention_value",
            Self::Embedding => "embedding",
            Self::ResidualAdd => "residual_add",
            Self::AttentionGate => "attention_sigmoid_gate",
            Self::FfnSilu => "ffn_silu",
            Self::Silu => "gdn_conv_silu",
            Self::SiluGate => "gdn_output_gate",
            Self::GdnConv => "gdn_conv",
            Self::GdnRecurrent => "gdn_recurrent",
        }
    }
    fn tensor(self) -> &'static str {
        match self {
            Self::SourceMatVec | Self::SourceDeviceQuant => "blk.0.attn_gate.weight",
            Self::RmsNorm => "blk.0.attn_norm.weight",
            Self::GroupedRmsNorm => "blk.0.ssm_norm.weight",
            Self::GdnL2 => "blk.0.gdn_qk_l2",
            Self::GdnCoefficients => "blk.0.gdn_coefficients",
            Self::RopeKv => "blk.3.attn_rope_kv",
            Self::AttentionQk => "blk.3.attn_qk",
            Self::AttentionSoftmax => "blk.3.attn_softmax",
            Self::AttentionValue => "blk.3.attn_value",
            Self::Embedding => "embedding",
            Self::ResidualAdd => "blk.0.attn_residual",
            Self::AttentionGate => "blk.3.attn_gate",
            Self::FfnSilu => "blk.0.ffn_silu",
            Self::Silu => "blk.0.gdn_conv_silu",
            Self::SiluGate => "blk.0.gdn_output_gate",
            Self::GdnConv => "blk.0.ssm_conv1d.weight",
            Self::GdnRecurrent => "blk.0.gdn_recurrent",
        }
    }
}

struct AbortRegion {
    provider: Arc<CubeResidentProvider>,
    operator: AbortOperator,
}
fn drop_submitted_future<F, T>(future: F) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    let mut unfinished = Box::pin(future);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let poll = Future::poll(unfinished.as_mut(), &mut cx);
    if !matches!(poll, Poll::Pending) || !ACTUAL_LAUNCH_RETURNED.load(Ordering::SeqCst) {
        return Err(Error::State(
            "physical Future abort proof did not reach authentic GPU dispatch".into(),
        ));
    }
    // Genuine AsyncMutex, InFlightRegion, PhysicalWork and ActiveSlot Drops.
    drop(unfinished);
    Err(Error::State(
        "physical GPU diagnostic intentionally aborted after launch, before readback".into(),
    ))
}

impl MatVecKernel for AbortRegion {
    fn try_embedding(&self, r: EmbeddingRegion<'_>) -> Result<RegionExecution> {
        if self.operator != AbortOperator::Embedding {
            return Ok(RegionExecution::Unsupported);
        }
        drop_submitted_future(self.provider.try_embedding_async(r))
    }
    fn try_binary(&self, r: BinaryRegion<'_>) -> Result<RegionExecution> {
        if !matches!(
            self.operator,
            AbortOperator::ResidualAdd | AbortOperator::AttentionGate
        ) || r.operation != self.operator.tensor()
        {
            return Ok(RegionExecution::Unsupported);
        }
        drop_submitted_future(self.provider.try_binary_async(r))
    }

    fn try_attention_value(&self, r: AttentionValueRegion<'_>) -> Result<RegionExecution> {
        if self.operator != AbortOperator::AttentionValue || r.operation != self.operator.tensor() {
            return Ok(RegionExecution::Unsupported);
        }
        if r.values.len() < 2 || !r.values[0].iter().any(|&x| x != 0.0) {
            return Err(Error::State(
                "value abort requires genuine nonzero prefix".into(),
            ));
        }
        drop_submitted_future(self.provider.try_attention_value_async(r))
    }

    fn try_attention_softmax(
        &self,
        r: AttentionSoftmaxRegion<'_>,
    ) -> Result<AttentionSoftmaxExecution> {
        if self.operator != AbortOperator::AttentionSoftmax || r.operation != self.operator.tensor()
        {
            return Ok(AttentionSoftmaxExecution::Unsupported);
        }
        if r.width < 2 || !r.scores.iter().any(|&x| x != 0.0) {
            return Err(Error::State(
                "softmax abort requires genuine nonzero prefix scores".into(),
            ));
        }
        drop_submitted_future(self.provider.try_attention_softmax_async(r))
    }

    fn try_attention_qk(&self, r: AttentionQkRegion<'_>) -> Result<AttentionQkExecution> {
        if self.operator != AbortOperator::AttentionQk || r.operation != self.operator.tensor() {
            return Ok(AttentionQkExecution::Unsupported);
        }
        if r.keys.len() < 2 || !r.keys[0].iter().any(|&x| x != 0.0) {
            return Err(Error::State(
                "QK abort requires genuine nonzero prefix".into(),
            ));
        }
        drop_submitted_future(self.provider.try_attention_qk_async(r))
    }

    fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
        if !matches!(
            self.operator,
            AbortOperator::SourceMatVec | AbortOperator::SourceDeviceQuant
        ) || region.tensor_name != self.operator.tensor()
        {
            return Ok(RegionExecution::Unsupported);
        }
        drop_submitted_future(self.provider.try_execute_async(region))
    }

    fn try_rms_norm(&self, region: RmsNormRegion<'_>) -> Result<RegionExecution> {
        if self.operator != AbortOperator::RmsNorm || region.tensor_name != self.operator.tensor() {
            return Ok(RegionExecution::Unsupported);
        }
        drop_submitted_future(self.provider.try_rms_norm_async(region))
    }

    fn try_grouped_rms_norm(&self, region: GroupedRmsNormRegion<'_>) -> Result<RegionExecution> {
        if !matches!(
            self.operator,
            AbortOperator::GroupedRmsNorm | AbortOperator::GdnL2
        ) || region.tensor_name != self.operator.tensor()
        {
            return Ok(RegionExecution::Unsupported);
        }
        drop_submitted_future(self.provider.try_grouped_rms_norm_async(region))
    }

    fn try_gdn_recurrent(&self, region: GdnRecurrentRegion<'_>) -> Result<GdnRecurrentExecution> {
        if self.operator != AbortOperator::GdnRecurrent
            || region.operation != self.operator.tensor()
        {
            return Ok(GdnRecurrentExecution::Unsupported);
        }
        if !region.state.iter().any(|&x| x != 0.0) {
            return Err(Error::State(
                "recurrent abort requires real nonzero trained state".into(),
            ));
        }
        drop_submitted_future(self.provider.try_gdn_recurrent_async(region))
    }

    fn try_rope_kv(&self, r: RopeKvRegion<'_>) -> Result<RopeKvExecution> {
        if self.operator != AbortOperator::RopeKv || r.operation != self.operator.tensor() {
            return Ok(RopeKvExecution::Unsupported);
        }
        if r.position == 0 {
            return Err(Error::State(
                "RoPE abort requires nonzero committed KV prefix".into(),
            ));
        }
        drop_submitted_future(self.provider.try_rope_kv_async(r))
    }

    fn try_gdn_coefficients(
        &self,
        region: GdnCoefficientRegion<'_>,
    ) -> Result<GdnCoefficientExecution> {
        if self.operator != AbortOperator::GdnCoefficients
            || region.operation != self.operator.tensor()
        {
            return Ok(GdnCoefficientExecution::Unsupported);
        }
        drop_submitted_future(self.provider.try_gdn_coefficients_async(region))
    }

    fn try_silu(&self, region: SiluRegion<'_>) -> Result<RegionExecution> {
        if !matches!(
            self.operator,
            AbortOperator::Silu | AbortOperator::SiluGate | AbortOperator::FfnSilu
        ) || region.operation != self.operator.tensor()
        {
            return Ok(RegionExecution::Unsupported);
        }
        drop_submitted_future(self.provider.try_silu_async(region))
    }

    fn try_gdn_conv(&self, region: GdnConvRegion<'_>) -> Result<GdnConvExecution> {
        if self.operator != AbortOperator::GdnConv || region.tensor_name != self.operator.tensor() {
            return Ok(GdnConvExecution::Unsupported);
        }
        drop_submitted_future(self.provider.try_gdn_conv_async(region))
    }
}

pub fn run(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::SourceMatVec)
}

pub fn run_norm(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::RmsNorm)
}

pub fn run_grouped_norm(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::GroupedRmsNorm)
}

pub fn run_l2(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::GdnL2)
}

pub fn run_conv(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::GdnConv)
}

pub fn run_recurrent(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::GdnRecurrent)
}

pub fn run_silu(device: Device, model: &Path, gated: bool) -> std::result::Result<(), String> {
    run_kind(
        device,
        model,
        if gated {
            AbortOperator::SiluGate
        } else {
            AbortOperator::Silu
        },
    )
}

pub fn run_coefficients(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::GdnCoefficients)
}

pub fn run_device_quant(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::SourceDeviceQuant)
}

pub fn run_remaining(device: Device, model: &Path, kind: &str) -> std::result::Result<(), String> {
    run_kind(
        device,
        model,
        match kind {
            "embedding" => AbortOperator::Embedding,
            "residual" => AbortOperator::ResidualAdd,
            "attention-gate" => AbortOperator::AttentionGate,
            "ffn" => AbortOperator::FfnSilu,
            _ => return Err("unknown remaining abort kind".into()),
        },
    )
}

pub fn run_value(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::AttentionValue)
}

pub fn run_softmax(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::AttentionSoftmax)
}

pub fn run_qk(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::AttentionQk)
}

pub fn run_rope(device: Device, model: &Path) -> std::result::Result<(), String> {
    run_kind(device, model, AbortOperator::RopeKv)
}

fn run_kind(
    device: Device,
    model: &Path,
    operator: AbortOperator,
) -> std::result::Result<(), String> {
    let provider_cap = if operator == AbortOperator::Embedding {
        512 * 1024 * 1024
    } else {
        256 * 1024 * 1024
    };
    let device_cap = if operator == AbortOperator::Embedding {
        1024 * 1024 * 1024
    } else {
        512 * 1024 * 1024
    };
    let client = device.client();
    let directory = model.parent().ok_or("GGUF checkpoint has no directory")?;
    let provider = Arc::new(
        CubeResidentProvider::new(
            device.clone(),
            [operator.tensor().to_owned()],
            8192,
            provider_cap,
            device_cap,
            ResourceOwner::new("qwen35-physical-abort", "first-session")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
        .with_device_activation_quantization(operator == AbortOperator::SourceDeviceQuant),
    );
    let card = provider.physical_key();
    let queue = crate::physical_queue::queue_for(card.clone()).map_err(|e| e.to_string())?;
    let original = ResourceArbiter::process_shared()
        .snapshot(&card)
        .map_err(|e| e.to_string())?;
    if original.reserved != provider_cap as u64 || original.fenced {
        return Err(
            "physical GPU abort probe did not begin from an unfenced reserved source".into(),
        );
    }
    let kernel = Arc::new(AbortRegion {
        provider: provider.clone(),
        operator,
    });
    let mut qwen =
        QwenDenseCpu::open(model, &directory.join("config.json")).map_err(|e| e.to_string())?;
    if matches!(
        operator,
        AbortOperator::SourceDeviceQuant
            | AbortOperator::GdnRecurrent
            | AbortOperator::GdnL2
            | AbortOperator::GdnCoefficients
            | AbortOperator::RopeKv
            | AbortOperator::AttentionQk
            | AbortOperator::AttentionSoftmax
            | AbortOperator::AttentionValue
            | AbortOperator::Embedding
            | AbortOperator::ResidualAdd
            | AbortOperator::AttentionGate
            | AbortOperator::FfnSilu
            | AbortOperator::Silu
            | AbortOperator::SiluGate
    ) {
        qwen.prefill_transaction(&[11]).map_err(|e| e.to_string())?;
    }
    qwen = qwen.with_kernel(kernel);
    let root = qwen.root_id();
    let position = qwen.position();
    let fingerprint = qwen.state_fingerprint();
    ARM_ONE_SUBMISSION.store(true, Ordering::SeqCst);
    if qwen.prefill_transaction(&[353u32]).is_ok() {
        return Err("abandoned real GPU command falsely completed Qwen prefill".into());
    }
    if !ACTUAL_LAUNCH_RETURNED.load(Ordering::SeqCst)
        || qwen.root_id() != root
        || qwen.position() != position
        || qwen.state_fingerprint() != fingerprint
    {
        return Err(
            "physical abort either missed GPU dispatch or changed unpublished StateRoot".into(),
        );
    }
    let after = ResourceArbiter::process_shared()
        .snapshot(&card)
        .map_err(|e| e.to_string())?;
    let keepalives = provider
        .quarantined_physical_handles()
        .map_err(|e| e.to_string())?;
    let stats = provider.worker_stats().map_err(|e| e.to_string())?;
    if !after.fenced
        || after.reserved != original.reserved
        || after.reservations != original.reservations
        || keepalives
            < if operator == AbortOperator::SourceDeviceQuant {
                12
            } else if operator == AbortOperator::GdnRecurrent {
                9
            } else if operator == AbortOperator::AttentionSoftmax {
                13
            } else if matches!(
                operator,
                AbortOperator::AttentionQk | AbortOperator::AttentionValue
            ) {
                6
            } else if operator == AbortOperator::RopeKv {
                8
            } else if matches!(operator, AbortOperator::Silu | AbortOperator::ResidualAdd) {
                3
            } else if operator == AbortOperator::AttentionGate {
                6
            } else if matches!(
                operator,
                AbortOperator::GroupedRmsNorm
                    | AbortOperator::GdnL2
                    | AbortOperator::GdnCoefficients
                    | AbortOperator::RmsNorm
            ) {
                5
            } else {
                4
            }
        || stats.active
        || stats.queued != 0
        || stats.completed != 0
    {
        return Err(format!(
            "physical submitted-unknown GPU failed fence/handle/account invariants: {after:?}, handles={keepalives}, queue={stats:?}"
        ));
    }
    if CubeResidentProvider::new(
        device,
        ["blk.0.attn_gate.weight".to_owned()],
        8192,
        128 * 1024 * 1024,
        device_cap,
        ResourceOwner::new("different-model", "blocked-after-unknown")
            .map_err(|e| e.to_string())?,
    )
    .is_ok()
    {
        return Err("unknown physical GPU work admitted another model".into());
    }
    let root_snapshot = qwen.snapshot();
    drop(qwen);
    drop(provider);
    let final_state = ResourceArbiter::process_shared()
        .snapshot(&card)
        .map_err(|e| e.to_string())?;
    let handles_after_teardown =
        crate::physical_quarantine::quarantined_handles_for(&client).map_err(|e| e.to_string())?;
    if !final_state.fenced
        || final_state.reserved != original.reserved
        || handles_after_teardown < keepalives
    {
        return Err(format!(
            "physical source handles/charge released before verified GPU quiescence: {final_state:?} handles={handles_after_teardown}"
        ));
    }
    println!(
        "P07_G3_PHYSICAL_GPU_SUBMITTED_UNKNOWN_FENCED operator={} card={card:?} submitted_without_readback=1 quarantine_handles={handles_after_teardown} global_retained_bytes={} state_root_unchanged=1 rejected_second_model=1",
        operator.name(),
        final_state.reserved
    );
    // Diagnostic process shutdown only: all unknown-completion assertions
    // above were made BEFORE this real service/stream driver barrier. No result is read,
    // no event becomes Ready, and no root/lease/fence is recovered. NVIDIA
    // 595.91.07 faulted in EGL C exit handlers if this submitted work was
    // still outstanding when the isolated process exited normally.
    futures_lite::future::block_on(client.sync())
        .map_err(|e| format!("physical abort cleanup quiescence failed: {e}"))?;
    let after_quiescence = ResourceArbiter::process_shared()
        .snapshot(&card)
        .map_err(|e| e.to_string())?;
    let retained =
        crate::physical_quarantine::quarantined_handles_for(&client).map_err(|e| e.to_string())?;
    let queue_after = queue.stats().map_err(|e| e.to_string())?;
    if !after_quiescence.fenced
        || after_quiescence.reserved != original.reserved
        || retained < handles_after_teardown
        || root_snapshot.fingerprint() != fingerprint
        || root_snapshot.position() != position
        || queue_after.completed != 0
        || queue_after.active
        || queue_after.queued != 0
    {
        return Err("diagnostic shutdown barrier released quarantined state/fence".into());
    }
    println!(
        "PHYSICAL_ABORT_PROCESS_EXIT_QUIESCED cleanup_service_quiesced=1 operator={} quarantine_handles={retained} fence_retained=1 charge_retained=1 no_ready_or_root_publication=1",
        operator.name()
    );
    Ok(())
}

/// Diagnostic finite-input overflow must be rejected by REAL device status,
/// rather than accepted because the host input and shape are well formed.
pub fn run_invalid_quant(device: Device, model: &Path) -> std::result::Result<(), String> {
    struct InvalidQuant(Arc<CubeResidentProvider>);
    impl MatVecKernel for InvalidQuant {
        fn try_execute(&self, r: MatVecRegion<'_>) -> Result<RegionExecution> {
            if r.tensor_name != "blk.0.attn_gate.weight" {
                return Ok(RegionExecution::Unsupported);
            }
            let input = vec![f32::MAX; r.columns];
            self.0.try_execute(MatVecRegion {
                activation: &input,
                ..r
            })
        }
    }
    let client = device.client();
    let provider = Arc::new(
        CubeResidentProvider::new(
            device.clone(),
            ["blk.0.attn_gate.weight".into()],
            8192,
            256 * 1024 * 1024,
            512 * 1024 * 1024,
            ResourceOwner::new("invalid-quant", "session").map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
        .with_device_activation_quantization(true),
    );
    let key = provider.physical_key();
    let mut qwen = QwenDenseCpu::open(
        model,
        &model.parent().ok_or("model parent")?.join("config.json"),
    )
    .map_err(|e| e.to_string())?;
    qwen.prefill_transaction(&[11]).map_err(|e| e.to_string())?;
    let before = (qwen.root_id(), qwen.position(), qwen.state_fingerprint());
    qwen = qwen.with_kernel(Arc::new(InvalidQuant(provider.clone())));
    let error = qwen
        .prefill_transaction(&[353])
        .expect_err("invalid device scale must reject")
        .to_string();
    if !error.contains("device quantization block status")
        || before != (qwen.root_id(), qwen.position(), qwen.state_fingerprint())
        || provider
            .worker_stats()
            .map_err(|e| e.to_string())?
            .completed
            != 0
        || provider
            .quarantined_physical_handles()
            .map_err(|e| e.to_string())?
            < 12
        || !provider.is_fenced().map_err(|e| e.to_string())?
    {
        return Err(format!(
            "physical quant status rejection did not preserve ownership/root: {error}"
        ));
    }
    let snapshot = ResourceArbiter::process_shared()
        .snapshot(&key)
        .map_err(|e| e.to_string())?;
    if !snapshot.fenced || snapshot.reserved != 256 * 1024 * 1024 {
        return Err("invalid quant physical charge released".into());
    }
    if CubeResidentProvider::new(
        device,
        ["other".into()],
        8,
        1024,
        512 * 1024 * 1024,
        ResourceOwner::new("other", "denied").map_err(|e| e.to_string())?,
    )
    .is_ok()
    {
        return Err("invalid quant failed physical fence".into());
    }
    drop(qwen);
    drop(provider);
    futures_lite::future::block_on(client.sync())
        .map_err(|e| format!("invalid quant cleanup failed: {e}"))?;
    let after = ResourceArbiter::process_shared()
        .snapshot(&key)
        .map_err(|e| e.to_string())?;
    if !after.fenced
        || after.reserved != snapshot.reserved
        || crate::physical_quarantine::quarantined_handles_for(&client)
            .map_err(|e| e.to_string())?
            < 12
    {
        return Err("invalid quant cleanup released fence/handles".into());
    }
    println!(
        "P07_H14_PHYSICAL_NONZERO_QUANT_STATUS_REJECTED finite_input=1 block_status_nonzero=1 root_preserved=1 no_ready=1 retained_bytes={} original_buffers=12 fence_retained=1",
        after.reserved
    );
    Ok(())
}
