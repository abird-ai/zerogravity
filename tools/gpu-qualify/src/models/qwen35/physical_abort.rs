//! Qwen source/reference orchestration for the original physical abort probes.
use std::path::Path;
use zerogravity_gpu::Device;
use zerogravity_gpu::qualification::physical_abort::{
    AbortOperator, arm_physical_abort, physical_launch_returned, qualify_abort_operator,
    qualify_invalid_quant,
};
use zerogravity_model_qwen35::QwenDenseCpu;
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
    qualify_abort_operator(device, operator, |kernel| {
        let directory = model.parent().ok_or("GGUF checkpoint has no directory")?;
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
        arm_physical_abort();
        if qwen.prefill_transaction(&[353u32]).is_ok() {
            return Err("abandoned real GPU command falsely completed Qwen prefill".into());
        }
        if !physical_launch_returned()
            || qwen.root_id() != root
            || qwen.position() != position
            || qwen.state_fingerprint() != fingerprint
        {
            return Err(
                "physical abort either missed GPU dispatch or changed unpublished StateRoot".into(),
            );
        }
        let root_snapshot = qwen.snapshot();
        Ok((qwen, move || {
            root_snapshot.fingerprint() == fingerprint && root_snapshot.position() == position
        }))
    })
}

pub fn run_invalid_quant(device: Device, model: &Path) -> std::result::Result<(), String> {
    qualify_invalid_quant(device, |kernel| {
        let mut qwen = QwenDenseCpu::open(
            model,
            &model.parent().ok_or("model parent")?.join("config.json"),
        )
        .map_err(|e| e.to_string())?;
        qwen.prefill_transaction(&[11]).map_err(|e| e.to_string())?;
        let before = (qwen.root_id(), qwen.position(), qwen.state_fingerprint());
        qwen = qwen.with_kernel(kernel);
        let error = qwen
            .prefill_transaction(&[353])
            .expect_err("invalid device scale must reject")
            .to_string();
        if before != (qwen.root_id(), qwen.position(), qwen.state_fingerprint()) {
            return Err(format!(
                "physical quant status rejection did not preserve ownership/root: {error}"
            ));
        }
        Ok((qwen, error))
    })
}
