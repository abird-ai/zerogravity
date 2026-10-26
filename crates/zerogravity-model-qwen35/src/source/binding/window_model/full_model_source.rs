//! Source-owned geometry seal. No native/model/session/publication permit.
use super::*;
use crate::plan::{QwenGdnCall, QwenPlan, QwenSourceCall, QwenSourceProfile};

mod prompt;
pub use prompt::BoundPromptCall;

const CONFIG2B: &str = "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4";
const INDEX2B: &str = "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9";
const BYTES2B: u64 = 4_548_221_488;
const MIB: usize = 1024 * 1024;

// Only bind_full_model_source constructs this after the complete role/profile
// join. Owned config/header/role metadata is immutable; actual window reads
// retain their independent source checks and admission.
struct SourceOwner {
    model: Arc<OriginalWindowModelSource>,
    profile: QwenSourceProfile,
    _metadata: WindowWorkspace,
    prompt: std::sync::Mutex<Vec<prompt::PreparedRecord>>,
}
#[derive(Clone)]
pub struct BoundFullModelSource(Arc<SourceOwner>);
impl BoundFullModelSource {
    pub fn model_profile(&self) -> QwenSourceProfile {
        self.0.profile
    }
    pub fn source(&self) -> &Arc<OriginalWindowModelSource> {
        &self.0.model
    }
    /// Metadata recording only. Actual lowerers must bind this original owner.
    pub fn record(&self, request: QwenGdnCall) -> Result<QwenPlan> {
        if request.layer().source() != self.0.model.source_identity() {
            return Err(Error::State(
                "full-model sealed source/request issuer".into(),
            ));
        }
        self.0
            .model
            .record_full_model_inner(QwenSourceCall::for_profile(self.0.profile, request)?)
    }
}

impl OriginalWindowModelSource {
    pub fn bind_full_model_source(self: &Arc<Self>) -> Result<BoundFullModelSource> {
        let profile = if self.owner().config_sha256() == CONFIG2B {
            QwenSourceProfile::Qwen2B
        } else {
            QwenSourceProfile::Qwen08
        };
        require_source(self, profile)?;
        // Source descriptors/roles were validated before this admission.
        let metadata = self.source().reserve_workspace(64 * 1024)?;
        Ok(BoundFullModelSource(Arc::new(SourceOwner {
            model: self.clone(),
            profile,
            _metadata: metadata,
            prompt: std::sync::Mutex::new(Vec::new()),
        })))
    }
}

fn require_2b_pins(
    config: &str,
    index: &str,
    bytes: u64,
    chunk: usize,
    text: usize,
    other: usize,
) -> Result<()> {
    if config != CONFIG2B
        || index != INDEX2B
        || bytes != BYTES2B
        || chunk != MIB
        || text != 320
        || other != 312
    {
        return Err(Error::Unsupported(
            "closed original2B full-model source/config/index/extent/1MiB".into(),
        ));
    }
    Ok(())
}

fn require_source(model: &OriginalWindowModelSource, profile: QwenSourceProfile) -> Result<()> {
    use crate::config::{Family, LayerKind};
    use crate::source::safetensors::GdnOutputGate;
    if profile == QwenSourceProfile::Qwen08 {
        model.require_small_model_source()?;
    } else {
        require_2b_pins(
            model.owner().config_sha256(),
            model.owner().verified_index_sha256(),
            model.source().source_bytes(),
            model.source().staging_chunk_bytes(),
            model.owner().text_tensor_count(),
            model.owner().other_tensor_count(),
        )?;
    }
    let c = model.owner().config();
    let g = model.geometry();
    if g.family != Family::QwenHybrid
        || g.hidden_size != profile.hidden_size() as u64
        || g.vocab_size != 248320
        || g.layers.len() != 24
        || g.head_dim != 256
        || g.num_attention_heads != 8
        || g.num_key_value_heads != 2
        || g.recurrent_key_heads != Some(16)
        || g.recurrent_value_heads != Some(16)
        || c.intermediate_size() != profile.intermediate_size() as u64
        || c.linear_key_head_dim() != 128
        || c.linear_value_head_dim() != 128
        || c.linear_conv_kernel_dim() != 4
        || c.rms_norm_eps() != 1e-6
        || c.gdn_output_gate() != GdnOutputGate::Silu
        || !c.attention_output_gate()
        || !c.tie_word_embeddings()
        || c.rope_theta() != 10_000_000.0
        || c.rotary_dimensions() != 64
        || g.layers.iter().enumerate().any(|(i, kind)| {
            *kind
                != if i % 4 == 3 {
                    LayerKind::FullAttention
                } else {
                    LayerKind::LinearAttention
                }
        })
    {
        return Err(Error::Unsupported(
            "closed full-model source/profile geometry".into(),
        ));
    }
    let h = profile.hidden_size();
    let m = profile.intermediate_size();
    matrix(model, QwenMatrixRole::Embedding, 248320, h)?;
    if model.matrix_descriptor(QwenMatrixRole::Embedding)?
        != model.matrix_descriptor(QwenMatrixRole::Output)?
    {
        return Err(Error::State(
            "original full-model tied output descriptor differs".into(),
        ));
    }
    matrix(model, QwenMatrixRole::Output, 248320, h)?;
    parameter(
        model,
        QwenParameterRole::FinalNorm,
        QuantKind::Bf16,
        &[h as u64],
    )?;
    for i in 0..24 {
        for role in [
            QwenParameterRole::InputNorm(i),
            QwenParameterRole::FfnNorm(i),
        ] {
            parameter(model, role, QuantKind::Bf16, &[h as u64])?;
        }
        for (role, rows, columns) in [
            (QwenMatrixRole::FfnGate(i), m, h),
            (QwenMatrixRole::FfnUp(i), m, h),
            (QwenMatrixRole::FfnDown(i), h, m),
        ] {
            matrix(model, role, rows, columns)?;
        }
        if i % 4 == 3 {
            for (role, rows, columns) in [
                (QwenMatrixRole::Query(i), 4096, h),
                (QwenMatrixRole::Key(i), 512, h),
                (QwenMatrixRole::Value(i), 512, h),
                (QwenMatrixRole::AttentionOutput(i), h, 2048),
            ] {
                matrix(model, role, rows, columns)?;
            }
            for role in [
                QwenParameterRole::QueryNorm(i),
                QwenParameterRole::KeyNorm(i),
            ] {
                parameter(model, role, QuantKind::Bf16, &[256])?;
            }
        } else {
            for (role, rows, columns) in [
                (QwenMatrixRole::GdnQkv(i), 6144, h),
                (QwenMatrixRole::GdnGate(i), 2048, h),
                (QwenMatrixRole::GdnBeta(i), 16, h),
                (QwenMatrixRole::GdnAlpha(i), 16, h),
                (QwenMatrixRole::GdnOutput(i), h, 2048),
            ] {
                matrix(model, role, rows, columns)?;
            }
            parameter(
                model,
                QwenParameterRole::GdnConvolution(i),
                QuantKind::Bf16,
                &[6144, 1, 4],
            )?;
            parameter(model, QwenParameterRole::GdnA(i), QuantKind::F32, &[16])?;
            parameter(model, QwenParameterRole::GdnDt(i), QuantKind::Bf16, &[16])?;
            parameter(model, QwenParameterRole::GdnNorm(i), QuantKind::F32, &[128])?;
        }
    }
    Ok(())
}

fn matrix(
    model: &OriginalWindowModelSource,
    role: QwenMatrixRole,
    rows: usize,
    columns: usize,
) -> Result<()> {
    let descriptor = model.matrix_descriptor(role)?;
    if descriptor.kind != QuantKind::Bf16
        || descriptor.axes != MatrixAxes::OutputInput
        || descriptor.rows != rows
        || descriptor.columns != columns
    {
        return Err(Error::State(
            "full-model exact original matrix shape/axes/type".into(),
        ));
    }
    model.validate_matrix(role, &descriptor)
}
fn parameter(
    model: &OriginalWindowModelSource,
    role: QwenParameterRole,
    kind: QuantKind,
    shape: &[u64],
) -> Result<()> {
    let descriptor = model.parameter_descriptor(role)?;
    let elements = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
        .ok_or_else(|| Error::Limit("full-model source parameter extent".into()))?;
    let bytes = elements
        .checked_mul(if kind == QuantKind::Bf16 { 2 } else { 4 })
        .ok_or_else(|| Error::Limit("full-model source parameter bytes".into()))?;
    if descriptor.kind != kind || descriptor.shape != shape || descriptor.bytes != bytes {
        return Err(Error::State(
            "full-model exact original parameter shape/type/extent".into(),
        ));
    }
    model.validate_parameter(role, &descriptor)
}

#[cfg(test)]
#[path = "full_model_source/tests.rs"]
mod tests;
