//! Authenticated, bounded original Qwen numerical comparison records.
//!
//! A caller-pinned SHA256 authenticates the reference JSON; the SOURCE
//! bundle digest binds the exact learned shard bytes, not just tensor paths.
//! "trained-framework" is reference provenance ASSERTED by the caller,
//! never independently proven by parsing a JSON string. This module never
//! runs a reference SDK, imports checkpoint files or publishes StateRoot.

use crate::accelerator::SourceIdentity;
use crate::accelerator::program::PlanObservation;
use crate::original_qwen_precision::OriginalNumericalDType;
use crate::{Error, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const SCHEMA: &str = "zerogravity.qwen.original-reference.v1";
const MAX_REFERENCE_JSON: usize = 32 * 1024 * 1024;
const MAX_TRACE_BYTES: usize = 32 * 1024 * 1024;
const MAX_RECORDS: usize = 2048;
const MAX_RECORD_VALUES: usize = 1_048_576;

fn valid_digest(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn valid_stage(stage: &str) -> bool {
    matches!(
        stage,
        "embedding"
            | "attention_input_norm"
            | "gdn_qkv"
            | "gdn_conv_silu"
            | "gdn_recurrent_fp32"
            | "attention_q_norm"
            | "attention_k_norm"
            | "attention_scores"
            | "attention_probs"
            | "attention_value"
            | "attention_output"
            | "post_attention_residual"
            | "ffn_gate"
            | "ffn_up"
            | "ffn_fused"
            | "ffn_output"
            | "post_ffn_residual"
            | "final_logits"
    )
}
pub(crate) const MODEL_LAYER: usize = usize::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceOrigin {
    AnalyticalFixture,
    TrainedFramework,
}
impl ReferenceOrigin {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "analytical-fixture" => Ok(Self::AnalyticalFixture),
            "trained-framework" => Ok(Self::TrainedFramework),
            _ => Err(Error::Malformed(
                "unqualified reference provenance kind".into(),
            )),
        }
    }
}
/// One source- and coordinator-bound TRACE of an unpublished numerical
/// candidate. Stage values are sensitive model data: never log, upload,
/// or share without the source owner's explicit permission.
#[derive(Clone)]
pub struct OriginalQwenTrace {
    pub(crate) source: SourceIdentity,
    pub(crate) coordinator_id: u64,
    pub(crate) root_id: u64,
    pub(crate) epoch: u64,
    pub(crate) position: usize,
    pub(crate) gdn_state_layer: Option<usize>,
    pub(crate) token: u32,
    pub(crate) dtype: OriginalNumericalDType,
    pub(crate) config_sha256: String,
    pub(crate) index_sha256: String,
    pub(crate) bundle_sha256: String,
    pub(crate) records: Vec<PlanObservation>,
}
impl OriginalQwenTrace {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn gdn_state_layer(&self) -> Option<usize> {
        self.gdn_state_layer
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn coordinator_id(&self) -> u64 {
        self.coordinator_id
    }
    pub fn root_id(&self) -> u64 {
        self.root_id
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }
    pub fn index_sha256(&self) -> &str {
        &self.index_sha256
    }
    pub fn token(&self) -> u32 {
        self.token
    }
    pub fn records(&self) -> &[PlanObservation] {
        &self.records
    }
    pub fn dtype(&self) -> OriginalNumericalDType {
        self.dtype
    }
    pub fn bundle_sha256(&self) -> &str {
        &self.bundle_sha256
    }
}

/// Trace records use the exact shared PlanObservation representation,
/// not a parallel numerics/session/state schema.
pub(crate) struct OriginalTraceRecorder {
    position: usize,
    gdn_state_layer: Option<usize>,
    total_bytes: usize,
    records: Vec<PlanObservation>,
}
impl OriginalTraceRecorder {
    /// The 27B Qwen recurrent state is ~3MiB *per* GDN layer; capture
    /// at most ONE selected state layer, never 48 full copies per token.
    pub fn new(position: usize, gdn_state_layer: Option<usize>) -> Result<Self> {
        if gdn_state_layer.is_some_and(|i| i >= 128) {
            return Err(Error::Limit(
                "reference trace chosen GDN layer out of bounds".into(),
            ));
        }
        Ok(Self {
            position,
            gdn_state_layer,
            total_bytes: 0,
            records: Vec::new(),
        })
    }
    pub fn observe(&mut self, layer: usize, stage: &'static str, values: &[f32]) -> Result<()> {
        if !valid_stage(stage)
            || values.is_empty()
            || values.len() > MAX_RECORD_VALUES
            || self.records.len() >= MAX_RECORDS
            || (layer != MODEL_LAYER && layer >= 128)
            || values.iter().any(|x| !x.is_finite())
        {
            return Err(Error::Malformed(
                "invalid bounded original Qwen numerical stage".into(),
            ));
        }
        if stage == "gdn_recurrent_fp32" && self.gdn_state_layer != Some(layer) {
            return Ok(());
        }
        let bytes = values
            .len()
            .checked_mul(4)
            .and_then(|n| self.total_bytes.checked_add(n))
            .ok_or_else(|| Error::Limit("original reference trace bytes".into()))?;
        if bytes > MAX_TRACE_BYTES {
            return Err(Error::Limit(
                "original numerical trace bounded bytes".into(),
            ));
        }
        self.records.push(PlanObservation {
            position: self.position,
            layer,
            stage,
            values: values.to_vec(),
        });
        self.total_bytes = bytes;
        Ok(())
    }
    pub fn finish(self, binding: TraceBinding) -> Result<OriginalQwenTrace> {
        if binding.dtype == OriginalNumericalDType::Bf16Diagnostic
            && self.records.iter().any(|r| {
                r.stage != "gdn_recurrent_fp32"
                    && r.values.iter().any(|v| v.to_bits() & 0xffff != 0)
            })
        {
            return Err(Error::State(
                "claimed BF16 operator stage contains unrounded values".into(),
            ));
        }
        if self.records.is_empty()
            || !valid_digest(&binding.config_sha256)
            || !valid_digest(&binding.index_sha256)
            || !valid_digest(&binding.bundle_sha256)
            || !binding.source.is_bound()
        {
            return Err(Error::State(
                "unbound original Qwen trace source/capture".into(),
            ));
        }
        Ok(OriginalQwenTrace {
            source: binding.source,
            coordinator_id: binding.coordinator_id,
            root_id: binding.root_id,
            epoch: binding.epoch,
            position: self.position,
            gdn_state_layer: self.gdn_state_layer,
            token: binding.token,
            dtype: binding.dtype,
            config_sha256: binding.config_sha256,
            index_sha256: binding.index_sha256,
            bundle_sha256: binding.bundle_sha256,
            records: self.records,
        })
    }
}
pub(crate) struct TraceBinding {
    pub source: SourceIdentity,
    pub coordinator_id: u64,
    pub root_id: u64,
    pub epoch: u64,
    pub token: u32,
    pub dtype: OriginalNumericalDType,
    pub config_sha256: String,
    pub index_sha256: String,
    pub bundle_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceEnvelope {
    schema: String,
    provenance: ReferenceProvenance,
    observations: Vec<ReferenceObservation>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceProvenance {
    kind: String,
    framework_revision: String,
    reference_code_sha256: String,
    config_sha256: String,
    index_sha256: String,
    bundle_sha256: String,
    token: u32,
    position: usize,
    gdn_state_layer: Option<usize>,
    dtype: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceObservation {
    /// None for embedding/final logits; some(layer) for per-layer values.
    layer: Option<usize>,
    stage: String,
    /// Bit-exact IEEE 754 f32 values as 8-digit lowercase hex words.
    bits: Vec<String>,
}
pub struct OriginalReferenceTrace {
    provenance: ReferenceOrigin,
    framework_revision: String,
    reference_code_sha256: String,
    config_sha256: String,
    index_sha256: String,
    bundle_sha256: String,
    token: u32,
    position: usize,
    gdn_state_layer: Option<usize>,
    dtype: OriginalNumericalDType,
    observations: Vec<(usize, String, Vec<u32>)>,
}
/// Reports the first numerical failure in source execution order.
/// A mismatch is not a successful trained reference qualification.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceMismatch {
    pub layer: usize,
    pub stage: String,
    pub element: usize,
    pub reference_bits: u32,
    pub actual_bits: u32,
    pub abs_error: f32,
}
#[derive(Debug)]
pub struct ReferenceComparison {
    pub compared_stages: usize,
    pub compared_values: usize,
    /// IEEE words which differ but were explicitly within FP32 tolerance.
    pub tolerated_values: usize,
    pub first_mismatch: Option<ReferenceMismatch>,
}
impl ReferenceComparison {
    /// No missing/incorrect stages or numerical violations.
    pub fn accepted(&self) -> bool {
        self.first_mismatch.is_none()
    }
    /// Strict bit equality, including signed zeros. A tolerated difference
    /// must NEVER be presented as an exact source/reference match.
    pub fn exact(&self) -> bool {
        self.accepted() && self.tolerated_values == 0
    }
}
#[derive(Clone, Copy)]
pub struct ReferenceTolerance {
    /// Only for FP32 stages; BF16 output stages require bit-exact words.
    pub fp32_abs: f32,
    pub fp32_rel: f32,
}
impl ReferenceTolerance {
    pub fn exact() -> Self {
        Self {
            fp32_abs: 0.0,
            fp32_rel: 0.0,
        }
    }
    fn validate(self) -> Result<()> {
        if !self.fp32_abs.is_finite()
            || !self.fp32_rel.is_finite()
            || self.fp32_abs < 0.0
            || self.fp32_rel < 0.0
            || self.fp32_abs > 0.01
            || self.fp32_rel > 0.01
        {
            return Err(Error::Limit(
                "original reference tolerance exceeds qualification bounds".into(),
            ));
        }
        Ok(())
    }
}
impl OriginalReferenceTrace {
    /// Exact user- or test-pinned bytes. JSON reference "kind" is a CLAIM;
    /// only the trusted capture process and pinned file provenance can
    /// establish independence. Duplicate/unknown fields are rejected.
    pub fn from_pinned_json(data: &[u8], expected_sha256: &str) -> Result<Self> {
        if data.is_empty() || data.len() > MAX_REFERENCE_JSON || !valid_digest(expected_sha256) {
            return Err(Error::Limit("reference manifest admission".into()));
        }
        if format!("{:x}", Sha256::digest(data)) != expected_sha256 {
            return Err(Error::Malformed(
                "original reference JSON SHA256 mismatch".into(),
            ));
        }
        let envelope: ReferenceEnvelope = serde_json::from_slice(data)
            .map_err(|_| Error::Malformed("original reference typed JSON invalid".into()))?;
        if envelope.schema != SCHEMA {
            return Err(Error::Unsupported(
                "original Qwen numerical reference version".into(),
            ));
        }
        let p = envelope.provenance;
        let provenance = ReferenceOrigin::parse(&p.kind)?;
        let dtype = match p.dtype.as_str() {
            "f32" => OriginalNumericalDType::F32,
            "bf16" => OriginalNumericalDType::Bf16Diagnostic,
            _ => return Err(Error::Unsupported("original reference dtype".into())),
        };
        if p.gdn_state_layer.is_some_and(|i| i >= 128)
            || p.position >= 8192
            || p.framework_revision.is_empty()
            || p.framework_revision.len() > 128
            || !p.framework_revision.bytes().all(|b| b.is_ascii_graphic())
            || ![
                &p.reference_code_sha256,
                &p.config_sha256,
                &p.index_sha256,
                &p.bundle_sha256,
            ]
            .iter()
            .all(|v| valid_digest(v))
            || envelope.observations.is_empty()
            || envelope.observations.len() > MAX_RECORDS
        {
            return Err(Error::Malformed(
                "unbound or oversized numerical reference provenance".into(),
            ));
        }
        let mut observed = Vec::new();
        let mut total = 0usize;
        for record in envelope.observations {
            if !valid_stage(&record.stage)
                || record.bits.is_empty()
                || record.bits.len() > MAX_RECORD_VALUES
                || record.layer.is_some_and(|i| i >= 128)
                || ((record.stage == "embedding" || record.stage == "final_logits")
                    != record.layer.is_none())
            {
                return Err(Error::Malformed(
                    "invalid typed numerical reference stage".into(),
                ));
            }
            total = total
                .checked_add(
                    record
                        .bits
                        .len()
                        .checked_mul(4)
                        .ok_or_else(|| Error::Limit("reference stage size".into()))?,
                )
                .ok_or_else(|| Error::Limit("reference aggregate size".into()))?;
            if total > MAX_TRACE_BYTES {
                return Err(Error::Limit("reference diagnostic output budget".into()));
            }
            let mut words = Vec::with_capacity(record.bits.len());
            for bits in record.bits {
                if bits.len() != 8
                    || !bits
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err(Error::Malformed(
                        "reference values must be exact lowercase f32 words".into(),
                    ));
                }
                let parsed = u32::from_str_radix(&bits, 16)
                    .map_err(|_| Error::Malformed("invalid IEEE reference word".into()))?;
                if dtype == OriginalNumericalDType::Bf16Diagnostic
                    && record.stage != "gdn_recurrent_fp32"
                    && parsed & 0xffff != 0
                {
                    return Err(Error::Malformed(
                        "reference BF16 output stage contains non-BF16 bits".into(),
                    ));
                }
                if !f32::from_bits(parsed).is_finite() {
                    return Err(Error::Malformed(
                        "nonfinite numerical reference stage".into(),
                    ));
                }
                words.push(parsed);
            }
            observed.push((record.layer.unwrap_or(MODEL_LAYER), record.stage, words));
        }
        Ok(Self {
            provenance,
            framework_revision: p.framework_revision,
            reference_code_sha256: p.reference_code_sha256,
            config_sha256: p.config_sha256,
            index_sha256: p.index_sha256,
            bundle_sha256: p.bundle_sha256,
            token: p.token,
            position: p.position,
            gdn_state_layer: p.gdn_state_layer,
            dtype,
            observations: observed,
        })
    }
    pub fn origin(&self) -> ReferenceOrigin {
        self.provenance
    }
    pub fn framework_revision(&self) -> &str {
        &self.framework_revision
    }
    pub fn reference_code_sha256(&self) -> &str {
        &self.reference_code_sha256
    }
    pub fn compare(
        &self,
        actual: &OriginalQwenTrace,
        tolerance: ReferenceTolerance,
    ) -> Result<ReferenceComparison> {
        tolerance.validate()?;
        if self.config_sha256 != actual.config_sha256
            || self.index_sha256 != actual.index_sha256
            || self.bundle_sha256 != actual.bundle_sha256
            || self.token != actual.token
            || self.position != actual.position
            || self.gdn_state_layer != actual.gdn_state_layer
            || self.dtype != actual.dtype
        {
            return Err(Error::State(
                "foreign original reference source/config/token/dtype".into(),
            ));
        }
        if self.observations.len() != actual.records.len() {
            return Err(Error::State(
                "incomplete or excess original reference stages".into(),
            ));
        }
        let mut compared = 0usize;
        let mut tolerated = 0usize;
        for ((layer, stage, expected), actual) in self.observations.iter().zip(&actual.records) {
            if *layer != actual.layer
                || stage != actual.stage
                || expected.len() != actual.values.len()
                || actual.position != self.position
            {
                return Err(Error::State(
                    "original trace layer/stage/value shape mismatch".into(),
                ));
            }
            for (i, (&expected, &value)) in expected.iter().zip(&actual.values).enumerate() {
                let got = value.to_bits();
                let reference = f32::from_bits(expected);
                // BF16-value stages compare bit-for-bit. The recurrent
                // accumulator intentionally stays FP32 and tolerates bounded
                // reduction-order differences if caller explicitly permits.
                let fp32_stage =
                    self.dtype == OriginalNumericalDType::F32 || stage == "gdn_recurrent_fp32";
                let abs_error = (reference - value).abs();
                let accepted = got == expected
                    || (fp32_stage
                        && expected != 0x8000_0000
                        && got != 0x8000_0000
                        && abs_error
                            <= tolerance.fp32_abs
                                + tolerance.fp32_rel * reference.abs().max(value.abs()));
                if !accepted {
                    return Ok(ReferenceComparison {
                        compared_stages: self.observations.len(),
                        compared_values: compared + 1,
                        tolerated_values: tolerated,
                        first_mismatch: Some(ReferenceMismatch {
                            layer: *layer,
                            stage: stage.clone(),
                            element: i,
                            reference_bits: expected,
                            actual_bits: got,
                            abs_error,
                        }),
                    });
                }
                if got != expected {
                    tolerated += 1;
                }
                compared += 1;
            }
        }
        Ok(ReferenceComparison {
            compared_stages: self.observations.len(),
            compared_values: compared,
            tolerated_values: tolerated,
            first_mismatch: None,
        })
    }
}
