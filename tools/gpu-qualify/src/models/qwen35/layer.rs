//! P07-H: an opt-in, source-strict GPU matrix placement sweep across an
//! ORIGINAL trained Qwen block (or the entire 24-block model). This is the
//! same Rust decoder and shared model, not a second inference engine.
//!
//! Every selected source GEMV is physically dispatched via the per-card fair
//! CubeCL service, read back and bit-compared to the exact Rust GGUF source
//! GEMV BEFORE any result is allowed to touch the transactional StateRoot.
//! Remaining attention/KV and FFN/residual math stay on CPU. Passing
//! this probe never certifies complete model-on-GPU execution.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Error;
use zerogravity_core::Result as ZgResult;
use zerogravity_core::accelerator::{
    AttentionQkCompleted, AttentionQkExecution, AttentionQkRegion, AttentionSoftmaxCompleted,
    AttentionSoftmaxExecution, AttentionSoftmaxRegion, AttentionValueRegion, BinaryRegion,
    EmbeddingRegion, GdnCoefficientExecution, GdnCoefficientRegion, GdnConvExecution,
    GdnConvRegion, GdnRecurrentExecution, GdnRecurrentRegion, GroupedRmsNormRegion, MatVecKernel,
    MatVecRegion, RegionExecution, RmsNormRegion, RopeKvCompleted, RopeKvExecution, RopeKvRegion,
    SiluRegion,
};
use zerogravity_core::resource_arbiter::{ResourceArbiter, ResourceOwner};
use zerogravity_cpu::source::{source_reference_embedding, source_reference_projection};
use zerogravity_gpu::CubeResidentProvider;
use zerogravity_gpu::Device;
use zerogravity_model_qwen35::config::LayerKind;
use zerogravity_model_qwen35::{
    LoadedQwenModel, source_attention_qk, source_attention_softmax, source_attention_value,
    source_binary, source_gdn_coefficients, source_gdn_recurrent_step,
    source_grouped_rms_normalize, source_rms_normalize, source_rope_kv,
};

const MAX_SELECTED_ROWS: usize = 300_000;
const PROVIDER_CAP: usize = 512 * 1024 * 1024;
const OPERATOR_CAP: u64 = 1024 * 1024 * 1024;

/// CPU source parity is an explicit test gate, not an unmeasured tolerance
/// quietly applied to an optimized production tensor. The inner provider
/// retains physical leases until real CubeCL readback; this wrapper only
/// accepts a source-exact completed region and never falls back on errors.
struct CheckedRegions {
    inner: Arc<CubeResidentProvider>,
    completed: Mutex<BTreeMap<String, u64>>,
    nonzero_recurrent: AtomicUsize,
    preflight_checked: AtomicBool,
    l2_preflight_checked: AtomicBool,
    l2_selected: bool,
    silu_selected: bool,
    coefficients_selected: bool,
    rope_selected: bool,
    qk_selected: bool,
    softmax_selected: bool,
    value_selected: bool,
    remaining_selected: bool,
    binary_preflight_checked: AtomicBool,
    embedding_preflight_checked: AtomicBool,
    value_preflight_checked: AtomicBool,
    softmax_preflight_checked: AtomicBool,
    qk_preflight_checked: AtomicBool,
    rope_preflight_checked: AtomicBool,
    coefficients_preflight_checked: AtomicBool,
    silu_preflight_checked: AtomicUsize,
    recurrent_selected: bool,
    underbudget: Option<Arc<CubeResidentProvider>>,
}

impl MatVecKernel for CheckedRegions {
    fn try_embedding(&self, r: EmbeddingRegion<'_>) -> ZgResult<RegionExecution> {
        if self.remaining_selected
            && !self
                .embedding_preflight_checked
                .swap(true, Ordering::AcqRel)
        {
            let before = self.inner.worker_stats()?;
            for bad in [
                EmbeddingRegion {
                    token: r.vocabulary as u32,
                    ..r
                },
                EmbeddingRegion {
                    packed_weights: &[],
                    ..r
                },
            ] {
                if !matches!(self.inner.try_embedding(bad), Err(Error::Malformed(_))) {
                    return Err(Error::State("invalid embedding admitted".into()));
                }
            }
            if !matches!(
                self.inner.try_embedding(EmbeddingRegion {
                    source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
                    ..r
                }),
                Err(Error::Unsupported(_))
            ) {
                return Err(Error::State("unbound embedding admitted".into()));
            }
            if let Some(poor) = &self.underbudget
                && !matches!(poor.try_embedding(r), Err(Error::Limit(_)))
            {
                return Err(Error::State("embedding memory admission failed".into()));
            }
            if self.inner.worker_stats()?.completed != before.completed
                || self.inner.in_flight_device_bytes()? != 0
            {
                return Err(Error::State("invalid embedding launched".into()));
            }
        }
        let result = self.inner.try_embedding(r)?;
        if let RegionExecution::Completed(ref actual) = result {
            let expected = source_reference_embedding(r)?;
            check_remaining("embedding", &expected, actual)?;
            *self
                .completed
                .lock()
                .map_err(|_| Error::State("embedding proof poisoned".into()))?
                .entry("embedding".into())
                .or_default() += 1;
        }
        Ok(result)
    }
    fn try_binary(&self, r: BinaryRegion<'_>) -> ZgResult<RegionExecution> {
        if self.remaining_selected && !self.binary_preflight_checked.swap(true, Ordering::AcqRel) {
            let before = self.inner.worker_stats()?;
            let nan = vec![f32::NAN; r.left.len()];
            for bad in [
                BinaryRegion { left: &nan, ..r },
                BinaryRegion { right: &[], ..r },
            ] {
                if !matches!(self.inner.try_binary(bad), Err(Error::Malformed(_))) {
                    return Err(Error::State("invalid binary admitted".into()));
                }
            }
            if !matches!(
                self.inner.try_binary(BinaryRegion {
                    source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
                    ..r
                }),
                Err(Error::Unsupported(_))
            ) {
                return Err(Error::State("unbound binary admitted".into()));
            }
            if let Some(poor) = &self.underbudget
                && !matches!(poor.try_binary(r), Err(Error::Limit(_)))
            {
                return Err(Error::State("binary memory admission failed".into()));
            }
            if self.inner.worker_stats()?.completed != before.completed
                || self.inner.in_flight_device_bytes()? != 0
            {
                return Err(Error::State("invalid binary launched".into()));
            }
        }
        let result = self.inner.try_binary(r)?;
        if let RegionExecution::Completed(ref actual) = result {
            check_remaining(r.operation, &source_binary(r)?, actual)?;
            *self
                .completed
                .lock()
                .map_err(|_| Error::State("binary proof poisoned".into()))?
                .entry(r.operation.into())
                .or_default() += 1;
        }
        Ok(result)
    }

    fn try_attention_value(&self, r: AttentionValueRegion<'_>) -> ZgResult<RegionExecution> {
        if self.value_selected && !self.value_preflight_checked.swap(true, Ordering::AcqRel) {
            let before = self.inner.worker_stats()?;
            let nan = vec![f32::NAN; r.probabilities.len()];
            for bad in [
                AttentionValueRegion {
                    probabilities: &nan,
                    ..r
                },
                AttentionValueRegion { values: &[], ..r },
                AttentionValueRegion {
                    padded_tokens: 0,
                    ..r
                },
            ] {
                if !matches!(
                    self.inner.try_attention_value(bad),
                    Err(Error::Malformed(_))
                ) {
                    return Err(Error::State("invalid value input admitted".into()));
                }
            }
            for bad in [
                AttentionValueRegion {
                    source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
                    ..r
                },
                AttentionValueRegion {
                    padded_tokens: r.padded_tokens + 32,
                    ..r
                },
            ] {
                if !matches!(
                    self.inner.try_attention_value(bad),
                    Err(Error::Unsupported(_))
                ) {
                    return Err(Error::State("unqualified value profile admitted".into()));
                }
            }
            if let Some(poor) = &self.underbudget
                && !matches!(poor.try_attention_value(r), Err(Error::Limit(_)))
            {
                return Err(Error::State("value memory admission failed".into()));
            }
            if self.inner.worker_stats()?.completed != before.completed
                || self.inner.in_flight_device_bytes()? != 0
                || self.inner.quarantined_physical_handles()? != 0
            {
                return Err(Error::State("invalid value region launched work".into()));
            }
        }
        let result = self.inner.try_attention_value(r)?;
        if let RegionExecution::Completed(ref actual) = result {
            check_value(r, actual)?;
            *self
                .completed
                .lock()
                .map_err(|_| Error::State("value proof poisoned".into()))?
                .entry(r.operation.to_owned())
                .or_default() += 1;
        }
        Ok(result)
    }

    fn try_attention_softmax(
        &self,
        r: AttentionSoftmaxRegion<'_>,
    ) -> ZgResult<AttentionSoftmaxExecution> {
        if self.softmax_selected && !self.softmax_preflight_checked.swap(true, Ordering::AcqRel) {
            let before = self.inner.worker_stats()?;
            let nan = vec![f32::NAN; r.scores.len()];
            for bad in [
                AttentionSoftmaxRegion { scores: &nan, ..r },
                AttentionSoftmaxRegion { heads: 0, ..r },
                AttentionSoftmaxRegion { width: 0, ..r },
            ] {
                if !matches!(
                    self.inner.try_attention_softmax(bad),
                    Err(Error::Malformed(_))
                ) {
                    return Err(Error::State("invalid softmax admitted".into()));
                }
            }
            let oversized = vec![0.0; 4097];
            for bad in [
                AttentionSoftmaxRegion {
                    source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
                    ..r
                },
                AttentionSoftmaxRegion {
                    scores: &oversized,
                    heads: 1,
                    width: 4097,
                    ..r
                },
            ] {
                if !matches!(
                    self.inner.try_attention_softmax(bad),
                    Err(Error::Unsupported(_))
                ) {
                    return Err(Error::State("out-of-domain softmax admitted".into()));
                }
            }
            if let Some(poor) = &self.underbudget
                && !matches!(poor.try_attention_softmax(r), Err(Error::Limit(_)))
            {
                return Err(Error::State("softmax memory admission failed".into()));
            }
            if self.inner.worker_stats()?.completed != before.completed
                || self.inner.in_flight_device_bytes()? != 0
                || self.inner.quarantined_physical_handles()? != 0
            {
                return Err(Error::State("invalid softmax launched work".into()));
            }
        }
        let result = self.inner.try_attention_softmax(r)?;
        if let AttentionSoftmaxExecution::Completed(ref actual) = result {
            check_softmax(r, actual)?;
            *self
                .completed
                .lock()
                .map_err(|_| Error::State("softmax proof poisoned".into()))?
                .entry(r.operation.to_owned())
                .or_default() += 1;
        }
        Ok(result)
    }
    fn try_attention_qk(&self, r: AttentionQkRegion<'_>) -> ZgResult<AttentionQkExecution> {
        if self.qk_selected && !self.qk_preflight_checked.swap(true, Ordering::AcqRel) {
            let before = self.inner.worker_stats()?;
            let nan = vec![f32::NAN; r.query.len()];
            for bad in [
                AttentionQkRegion { query: &nan, ..r },
                AttentionQkRegion { keys: &[], ..r },
                AttentionQkRegion {
                    scale: f32::NAN,
                    ..r
                },
            ] {
                if !matches!(self.inner.try_attention_qk(bad), Err(Error::Malformed(_))) {
                    return Err(Error::State("invalid QK region admitted".into()));
                }
            }
            let oversized = vec![r.keys[0].clone(); 4097];
            if !matches!(
                self.inner.try_attention_qk(AttentionQkRegion {
                    keys: &oversized,
                    ..r
                }),
                Err(Error::Unsupported(_))
            ) {
                return Err(Error::State("oversized QK prefix admitted".into()));
            }
            if !matches!(
                self.inner.try_attention_qk(AttentionQkRegion {
                    source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
                    ..r
                }),
                Err(Error::Unsupported(_))
            ) {
                return Err(Error::State("unbound QK identity admitted".into()));
            }
            if let Some(poor) = &self.underbudget
                && !matches!(poor.try_attention_qk(r), Err(Error::Limit(_)))
            {
                return Err(Error::State("QK memory admission failed".into()));
            }
            if self.inner.worker_stats()?.completed != before.completed
                || self.inner.in_flight_device_bytes()? != 0
                || self.inner.quarantined_physical_handles()? != 0
            {
                return Err(Error::State("invalid QK input launched work".into()));
            }
        }
        let result = self.inner.try_attention_qk(r)?;
        if let AttentionQkExecution::Completed(ref actual) = result {
            check_qk(r, actual)?;
            *self
                .completed
                .lock()
                .map_err(|_| Error::State("QK proof poisoned".into()))?
                .entry(r.operation.to_owned())
                .or_default() += 1;
        }
        Ok(result)
    }
    fn try_execute(&self, region: MatVecRegion<'_>) -> ZgResult<RegionExecution> {
        let result = self.inner.try_execute(region)?;
        if let RegionExecution::Completed(ref actual) = result {
            let source = source_reference_projection(region)?;
            if source.len() != actual.len()
                || source
                    .iter()
                    .zip(actual)
                    .any(|(&a, &b)| a.to_bits() != b.to_bits())
            {
                let different = source
                    .iter()
                    .zip(actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_abs = source
                    .iter()
                    .zip(actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                return Err(Error::State(format!(
                    "physical GPU GEMV source parity FAIL tensor={} kind={:?} shape={}x{} differing_rows={different} max_abs={max_abs}; refusing Qwen StateRoot",
                    region.tensor_name, region.kind, region.rows, region.columns
                )));
            }
            let mut table = self.completed.lock().map_err(|_| {
                Error::State("strict physical GEMV observation map poisoned".into())
            })?;
            *table.entry(region.tensor_name.to_owned()).or_default() += 1;
        }
        Ok(result)
    }

    fn try_rms_norm(&self, region: RmsNormRegion<'_>) -> ZgResult<RegionExecution> {
        let result = self.inner.try_rms_norm(region)?;
        if let RegionExecution::Completed(ref actual) = result {
            let expected = source_rms_normalize(region.input, region.weights, region.eps)?;
            if expected.len() != actual.len()
                || expected
                    .iter()
                    .zip(actual)
                    .any(|(&a, &b)| a.to_bits() != b.to_bits())
            {
                let diff = expected
                    .iter()
                    .zip(actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_abs = expected
                    .iter()
                    .zip(actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let sum = region.input.iter().map(|&v| f64::from(v * v)).sum::<f64>();
                let mean = (sum / region.input.len() as f64) as f32;
                let norm = mean + region.eps;
                let inv = norm.sqrt().recip();
                let first = expected
                    .iter()
                    .zip(actual)
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                    .unwrap_or(0);
                return Err(Error::State(format!(
                    "physical GPU RMS normalization parity FAIL tensor={} width={} differing_elements={diff} max_abs={max_abs}; source_mean_bits={:08x} normalized_mean_bits={:08x} source_inv_bits={:08x} first={first} source={:?} device={:?} input={:?} weight={:?}; refusing Qwen StateRoot",
                    region.tensor_name,
                    region.input.len(),
                    mean.to_bits(),
                    norm.to_bits(),
                    inv.to_bits(),
                    expected[first],
                    actual[first],
                    region.input[first],
                    region.weights[first],
                )));
            }
            let mut table = self
                .completed
                .lock()
                .map_err(|_| Error::State("strict physical RMS observation map poisoned".into()))?;
            *table.entry(region.tensor_name.to_owned()).or_default() += 1;
        }
        Ok(result)
    }

    fn try_grouped_rms_norm(&self, region: GroupedRmsNormRegion<'_>) -> ZgResult<RegionExecution> {
        if self.l2_selected
            && region.tensor_name.ends_with(".gdn_qk_l2")
            && !self.l2_preflight_checked.swap(true, Ordering::SeqCst)
        {
            let before = self.inner.worker_stats()?;
            let mut malformed = region;
            malformed.groups += 1;
            if self.inner.try_grouped_rms_norm(malformed).is_ok() {
                return Err(Error::State("malformed L2 geometry was admitted".into()));
            }
            malformed = region;
            malformed.eps = f32::NAN;
            if self.inner.try_grouped_rms_norm(malformed).is_ok() {
                return Err(Error::State("nonfinite L2 epsilon was admitted".into()));
            }
            let poor = self
                .underbudget
                .as_ref()
                .ok_or_else(|| Error::State("missing L2 underbudget admission probe".into()))?;
            if poor.try_grouped_rms_norm(region).is_ok()
                || poor.stats()?.completed_regions != 0
                || poor.quarantined_physical_handles()? != 0
                || self.inner.worker_stats()?.completed != before.completed
                || self.inner.quarantined_physical_handles()? != 0
            {
                return Err(Error::State(
                    "L2 pre-submit admission changed physical state".into(),
                ));
            }
            println!(
                "P07_H6_L2_PREFLIGHT_PASS malformed_geometry=refused nonfinite_epsilon=refused underbudget=refused no_submission=1"
            );
        }
        let result = self.inner.try_grouped_rms_norm(region)?;
        if let RegionExecution::Completed(ref actual) = result {
            let expected = source_grouped_rms_normalize(
                region.input,
                region.weights,
                region.groups,
                region.eps,
            )?;
            if expected.len() != actual.len()
                || expected
                    .iter()
                    .zip(actual)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                let differing = expected
                    .iter()
                    .zip(actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_abs = expected
                    .iter()
                    .zip(actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                return Err(Error::State(format!(
                    "physical GPU grouped RMS source mismatch tensor={} groups={} width={} differing={differing} max_abs={max_abs}; refusing StateRoot",
                    region.tensor_name, region.groups, region.group_width
                )));
            }
            let mut table = self.completed.lock().map_err(|_| {
                Error::State("strict grouped-RMS physical proof map mutex poisoned".into())
            })?;
            *table.entry(region.tensor_name.to_owned()).or_default() += 1;
        }
        Ok(result)
    }

    fn try_rope_kv(&self, region: RopeKvRegion<'_>) -> ZgResult<RopeKvExecution> {
        if self.rope_selected && !self.rope_preflight_checked.swap(true, Ordering::AcqRel) {
            let before = self.inner.worker_stats()?;
            let nan = vec![f32::NAN; region.key.len()];
            for bad in [
                RopeKvRegion {
                    key: &nan,
                    ..region
                },
                RopeKvRegion {
                    value: &[],
                    ..region
                },
                RopeKvRegion {
                    rotate: region.head_width + 2,
                    ..region
                },
            ] {
                if !matches!(self.inner.try_rope_kv(bad), Err(Error::Malformed(_))) {
                    return Err(Error::State("malformed RoPE/KV inputs admitted".into()));
                }
            }
            for bad in [
                RopeKvRegion {
                    position: 32769,
                    ..region
                },
                RopeKvRegion {
                    source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
                    ..region
                },
            ] {
                if !matches!(self.inner.try_rope_kv(bad), Err(Error::Unsupported(_))) {
                    return Err(Error::State("out-of-domain RoPE/KV inputs admitted".into()));
                }
            }
            if let Some(poor) = &self.underbudget
                && !matches!(poor.try_rope_kv(region), Err(Error::Limit(_)))
            {
                return Err(Error::State("RoPE/KV memory admission failed".into()));
            }
            if self.inner.worker_stats()?.completed != before.completed
                || self.inner.in_flight_device_bytes()? != 0
                || self.inner.quarantined_physical_handles()? != 0
            {
                return Err(Error::State(
                    "invalid RoPE/KV inputs launched physical work".into(),
                ));
            }
        }
        let result = self.inner.try_rope_kv(region)?;
        if let RopeKvExecution::Completed(ref ready) = result {
            check_rope(region, ready)?;
            *self
                .completed
                .lock()
                .map_err(|_| Error::State("RoPE proof poisoned".into()))?
                .entry(region.operation.to_owned())
                .or_default() += 1;
        }
        Ok(result)
    }

    fn try_gdn_coefficients(
        &self,
        region: GdnCoefficientRegion<'_>,
    ) -> ZgResult<GdnCoefficientExecution> {
        if self.coefficients_selected
            && !self
                .coefficients_preflight_checked
                .swap(true, Ordering::AcqRel)
        {
            let before = self.inner.worker_stats()?;
            let nan = vec![f32::NAN; region.alpha.len()];
            for bad in [
                GdnCoefficientRegion { dt: &[], ..region },
                GdnCoefficientRegion {
                    alpha: &nan,
                    ..region
                },
                GdnCoefficientRegion {
                    scale: f32::NAN,
                    ..region
                },
            ] {
                if !matches!(
                    self.inner.try_gdn_coefficients(bad),
                    Err(Error::Malformed(_))
                ) {
                    return Err(Error::State("invalid coefficient inputs accepted".into()));
                }
            }
            if !matches!(
                self.inner.try_gdn_coefficients(GdnCoefficientRegion {
                    source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
                    ..region
                }),
                Err(Error::Unsupported(_))
            ) {
                return Err(Error::State("unbound coefficient source admitted".into()));
            }
            if let Some(poor) = &self.underbudget
                && !matches!(poor.try_gdn_coefficients(region), Err(Error::Limit(_)))
            {
                return Err(Error::State("underbudget coefficient work accepted".into()));
            }
            if self.inner.worker_stats()?.completed != before.completed
                || self.inner.in_flight_device_bytes()? != 0
                || self.inner.quarantined_physical_handles()? != 0
            {
                return Err(Error::State(
                    "invalid coefficient admission launched work".into(),
                ));
            }
        }
        let result = self.inner.try_gdn_coefficients(region)?;
        if let GdnCoefficientExecution::Completed(ref actual) = result {
            check_coefficients(region, actual)?;
            *self
                .completed
                .lock()
                .map_err(|_| Error::State("coefficient proof poisoned".into()))?
                .entry(region.operation.to_owned())
                .or_default() += 1;
        }
        Ok(result)
    }

    fn try_silu(&self, region: SiluRegion<'_>) -> ZgResult<RegionExecution> {
        let bit = 1usize << usize::from(region.multiplier.is_some());
        if self.silu_selected
            && self.silu_preflight_checked.fetch_or(bit, Ordering::AcqRel) & bit == 0
        {
            let before = self.inner.worker_stats()?;
            let nan = vec![f32::NAN; region.input.len()];
            for bad in [
                SiluRegion {
                    input: &nan,
                    ..region
                },
                SiluRegion {
                    multiplier: Some(&[]),
                    ..region
                },
            ] {
                if !matches!(self.inner.try_silu(bad), Err(Error::Malformed(_))) {
                    return Err(Error::State("malformed SiLU input accepted".into()));
                }
            }
            if let Some(poor) = &self.underbudget
                && !matches!(poor.try_silu(region), Err(Error::Limit(_)))
            {
                return Err(Error::State("SiLU memory admission did not refuse".into()));
            }
            if !matches!(
                self.inner.try_silu(SiluRegion {
                    input: &[257.0],
                    multiplier: None,
                    ..region
                }),
                Err(Error::Unsupported(_))
            ) {
                return Err(Error::State(
                    "out-of-domain SiLU did not fail closed".into(),
                ));
            }
            let after = self.inner.worker_stats()?;
            if before.completed != after.completed || after.active || after.queued != 0 {
                return Err(Error::State("invalid SiLU submitted physical work".into()));
            }
        }
        let result = self.inner.try_silu(region)?;
        if let RegionExecution::Completed(ref actual) = result {
            let expected: Vec<_> = region
                .input
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let y = zerogravity_cpu::ops::ggml_source_silu(x);
                    region.multiplier.map_or(y, |m| m[i] * y)
                })
                .collect();
            if actual.len() != expected.len()
                || expected
                    .iter()
                    .zip(actual)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                let first = expected
                    .iter()
                    .zip(actual)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                return Err(Error::State(format!(
                    "physical source SiLU parity FAIL {} first={first:?} values={:?}; refusing StateRoot",
                    region.operation,
                    first.map(|i| (region.input[i], expected[i].to_bits(), actual[i].to_bits()))
                )));
            }
            *self
                .completed
                .lock()
                .map_err(|_| Error::State("SiLU proof lock poisoned".into()))?
                .entry(region.operation.to_owned())
                .or_default() += 1;
        }
        Ok(result)
    }

    fn try_gdn_conv(&self, region: GdnConvRegion<'_>) -> ZgResult<GdnConvExecution> {
        let result = self.inner.try_gdn_conv(region)?;
        if let GdnConvExecution::Completed(ref actual) = result {
            let mut expected_history = region.history.to_vec();
            let mut expected_raw = vec![0.0f32; region.activation.len()];
            zerogravity_cpu::ops::gdn_conv_step(
                region.activation,
                region.weights,
                &mut expected_history,
                &mut expected_raw,
            )?;
            let exact_raw = expected_raw.len() == actual.raw.len()
                && expected_raw
                    .iter()
                    .zip(&actual.raw)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            let exact_state = expected_history.len() == actual.history.len()
                && expected_history
                    .iter()
                    .zip(&actual.history)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            if !exact_raw || !exact_state {
                let raw_different = expected_raw
                    .iter()
                    .zip(&actual.raw)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let state_different = expected_history
                    .iter()
                    .zip(&actual.history)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let maximum = expected_raw
                    .iter()
                    .zip(&actual.raw)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let preceding_tokens = *self
                    .completed
                    .lock()
                    .map_err(|_| Error::State("conv proof map poisoned".into()))?
                    .get(region.tensor_name)
                    .unwrap_or(&0);
                return Err(Error::State(format!(
                    "physical GDN convolution trained source mismatch tensor={} raw={raw_different} history={state_different} max_abs={maximum} preceding_tokens={preceding_tokens}; no Qwen StateRoot published",
                    region.tensor_name
                )));
            }
            let mut counts = self.completed.lock().map_err(|_| {
                Error::State("strict GDN convolution proof map mutex poisoned".into())
            })?;
            *counts.entry(region.tensor_name.to_owned()).or_default() += 1;
        }
        Ok(result)
    }
    fn try_gdn_recurrent(&self, region: GdnRecurrentRegion<'_>) -> ZgResult<GdnRecurrentExecution> {
        if self.recurrent_selected && !self.preflight_checked.swap(true, Ordering::Relaxed) {
            let before = self.inner.worker_stats()?;
            let bad_state = GdnRecurrentRegion {
                state: &region.state[..region.state.len() - 1],
                ..region
            };
            if !matches!(
                self.inner.try_gdn_recurrent(bad_state),
                Err(Error::Malformed(_))
            ) {
                return Err(Error::State(
                    "invalid recurrent shape did not fail pre-submit".into(),
                ));
            }
            let mut q = region.q.to_vec();
            q[0] = f32::NAN;
            if !matches!(
                self.inner
                    .try_gdn_recurrent(GdnRecurrentRegion { q: &q, ..region }),
                Err(Error::Malformed(_))
            ) || !matches!(
                self.inner.try_gdn_recurrent(GdnRecurrentRegion {
                    source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
                    ..region
                }),
                Err(Error::Malformed(_))
            ) {
                return Err(Error::State(
                    "invalid recurrent provenance/FP32 did not fail pre-submit".into(),
                ));
            }
            if let Some(tiny) = &self.underbudget
                && !matches!(tiny.try_gdn_recurrent(region), Err(Error::Limit(_)))
            {
                return Err(Error::State(
                    "underbudget recurrent work was not refused before submission".into(),
                ));
            }
            if self.inner.worker_stats()? != before
                || self.inner.quarantined_physical_handles()? != 0
            {
                return Err(Error::State(
                    "recoverable recurrent preflight poisoned GPU queue/handles".into(),
                ));
            }
        }
        let result = self.inner.try_gdn_recurrent(region)?;
        if let GdnRecurrentExecution::Completed(ref actual) = result {
            let mut expected_state = region.state.to_vec();
            let mut expected_output = vec![0.0; region.q.len()];
            let d = region.width;
            for (h, &coeff) in region.coefficients.iter().enumerate() {
                source_gdn_recurrent_step(
                    &mut expected_state[h * d * d..(h + 1) * d * d],
                    &region.q[h * d..(h + 1) * d],
                    &region.k[h * d..(h + 1) * d],
                    &region.v[h * d..(h + 1) * d],
                    coeff,
                    &mut expected_output[h * d..(h + 1) * d],
                )?;
            }
            for (kind, expected, received) in [
                ("state", &expected_state, &actual.state),
                ("output", &expected_output, &actual.output),
            ] {
                if expected.len() != received.len()
                    || expected
                        .iter()
                        .zip(received)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    let first = expected
                        .iter()
                        .zip(received)
                        .position(|(a, b)| a.to_bits() != b.to_bits());
                    let different = expected
                        .iter()
                        .zip(received)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    return Err(Error::State(format!(
                        "physical recurrent source parity FAIL operation={} kind={kind} differing={different} first={first:?} expected={:?} actual={:?}; refusing StateRoot",
                        region.operation,
                        first.map(|i| expected[i].to_bits()),
                        first.map(|i| received[i].to_bits())
                    )));
                }
            }
            if region.state.iter().any(|&x| x != 0.0) {
                self.nonzero_recurrent.fetch_add(1, Ordering::Relaxed);
            }
            *self
                .completed
                .lock()
                .map_err(|_| Error::State("recurrent proof map poisoned".into()))?
                .entry(region.operation.to_owned())
                .or_default() += 1;
        }
        Ok(result)
    }
}

fn choose_projection_names(
    model: &LoadedQwenModel,
    blocks: usize,
    include_output: bool,
) -> Result<Vec<String>, String> {
    if blocks == 0 || blocks > model.geometry().layers.len() {
        return Err("GPU source sweep requests out-of-range Qwen block count".into());
    }
    let mut names = Vec::new();
    for (index, &layer) in model.geometry().layers.iter().take(blocks).enumerate() {
        let source_ops = match layer {
            LayerKind::LinearAttention => &[
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_alpha.weight",
                "ssm_beta.weight",
                "ssm_out.weight",
            ][..],
            LayerKind::FullAttention => &[
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            ][..],
            LayerKind::SlidingAttention => {
                return Err("unsupported Qwen sliding-attention in physical GPU sweep".into());
            }
        };
        for projection in source_ops
            .iter()
            .chain(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"].iter())
        {
            names.push(format!("blk.{index}.{projection}"));
        }
    }
    if include_output {
        names.push("token_embd.weight".into());
    }
    for name in &names {
        let info = model.tensor_info(name).map_err(|e| e.to_string())?;
        if info.shape.len() != 2 {
            return Err(format!("required projection {name} must be 2D"));
        }
        let cols = usize::try_from(info.shape[0]).map_err(|_| "columns overflow")?;
        let rows = usize::try_from(info.shape[1]).map_err(|_| "rows overflow")?;
        let qualified = matches!(
            info.kind,
            QuantKind::Q4_0 | QuantKind::Q4_1 | QuantKind::Q8_0 | QuantKind::Q6K | QuantKind::F32
        );
        if !qualified
            || rows == 0
            || rows > MAX_SELECTED_ROWS
            || (matches!(info.kind, QuantKind::Q4_0)
                && (!rows.is_multiple_of(8) || !cols.is_multiple_of(32)))
            || (matches!(info.kind, QuantKind::F32) && !cols.is_multiple_of(32))
            || (matches!(info.kind, QuantKind::Q6K) && !cols.is_multiple_of(256))
            || (matches!(info.kind, QuantKind::Q4_1 | QuantKind::Q8_0) && !cols.is_multiple_of(32))
        {
            return Err(format!(
                "required GPU projection is not source-qualified by current kernels: {name} {:?} {rows}x{cols}",
                info.kind
            ));
        }
    }
    Ok(names)
}

/// Explicit source-model placement for one independent physical acceptance.
/// One composable config, not an expanding positional/boolean API.
pub struct SweepConfig {
    pub blocks: usize,
    pub output_head: bool,
    pub residual_norms: bool,
    pub grouped_heads: bool,
    pub tokens: usize,
    pub gdn_recurrent: bool,
    pub gdn_convolution: bool,
    pub gdn_l2: bool,
    pub gdn_silu: bool,
    pub gdn_coefficients: bool,
    pub rope_kv: bool,
    pub attention_qk: bool,
    pub attention_softmax: bool,
    pub attention_value: bool,
    pub remaining_operators: bool,
    pub device_activation: bool,
}

pub fn run(device: Device, path: &Path, config: SweepConfig) -> Result<(), String> {
    let SweepConfig {
        blocks,
        output_head: output,
        residual_norms: norms,
        grouped_heads,
        tokens: token_count,
        gdn_recurrent: recurrent,
        gdn_convolution: convs,
        gdn_l2: l2,
        gdn_silu: silu,
        gdn_coefficients: coefficients,
        rope_kv: rope,
        attention_qk: qk,
        attention_softmax: softmax,
        attention_value: value,
        remaining_operators: remaining,
        device_activation,
    } = config;
    let config = path
        .parent()
        .ok_or("missing original model directory")?
        .join("config.json");
    let model = LoadedQwenModel::open(path, &config).map_err(|e| e.to_string())?;
    let matrix_names = choose_projection_names(&model, blocks, output)?;
    let mut names = matrix_names.clone();
    if norms {
        for i in 0..blocks {
            for role in ["attn_norm.weight", "post_attention_norm.weight"] {
                let name = format!("blk.{i}.{role}");
                let info = model.tensor_info(&name).map_err(|e| e.to_string())?;
                if info.shape != [model.geometry().hidden_size] {
                    return Err(format!(
                        "normalization source shape incompatible: {name} {:?}",
                        info.shape
                    ));
                }
                names.push(name);
            }
        }
        if output {
            let info = model
                .tensor_info("output_norm.weight")
                .map_err(|e| e.to_string())?;
            if info.shape != [model.geometry().hidden_size] {
                return Err("final normalization source shape incompatible".into());
            }
            names.push("output_norm.weight".into());
        }
    }
    let scalar_norm_count = names.len() - matrix_names.len();
    if grouped_heads {
        for (i, layer) in model.geometry().layers.iter().take(blocks).enumerate() {
            let roles: &[&str] = match layer {
                LayerKind::LinearAttention => &["ssm_norm.weight"],
                LayerKind::FullAttention => &["attn_q_norm.weight", "attn_k_norm.weight"],
                LayerKind::SlidingAttention => return Err("unsupported sliding head norm".into()),
            };
            for &role in roles {
                let name = format!("blk.{i}.{role}");
                let info = model.tensor_info(&name).map_err(|e| e.to_string())?;
                if info.shape.len() != 1 || info.shape[0] == 0 || info.shape[0] > 16_384 {
                    return Err(format!(
                        "GPU grouped head normalization has invalid source shape: {name} {:?}",
                        info.shape
                    ));
                }
                names.push(name);
            }
        }
    }
    let grouped_norm_count = names.len() - matrix_names.len() - scalar_norm_count;
    let mut l2_count = 0;
    if l2 {
        for (i, kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            if *kind == LayerKind::LinearAttention {
                names.push(format!("blk.{i}.gdn_qk_l2"));
                l2_count += 1;
            }
        }
    }
    let conv_count = if convs {
        let mut count = 0usize;
        for (index, &kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            if kind != LayerKind::LinearAttention {
                continue;
            }
            let name = format!("blk.{index}.ssm_conv1d.weight");
            let info = model.tensor_info(&name).map_err(|e| e.to_string())?;
            if info.kind != QuantKind::F32 || info.shape.len() != 2 || info.shape.contains(&0) {
                return Err(format!(
                    "original GDN conv source matrix not F32 rank2: {name} {:?}",
                    info.shape
                ));
            }
            names.push(name);
            count += 1;
        }
        count
    } else {
        0
    };
    let recurrent_count = if recurrent {
        let mut count = 0;
        for (i, kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            if *kind == LayerKind::LinearAttention {
                names.push(format!("blk.{i}.gdn_recurrent"));
                count += 1;
            }
        }
        count
    } else {
        0
    };
    let mut silu_count = 0;
    if silu {
        for (i, kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            if *kind == LayerKind::LinearAttention {
                names.push(format!("blk.{i}.gdn_conv_silu"));
                names.push(format!("blk.{i}.gdn_output_gate"));
                silu_count += 2;
            }
        }
    }
    let mut coefficient_count = 0;
    if coefficients {
        for (i, kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            if *kind == LayerKind::LinearAttention {
                names.push(format!("blk.{i}.gdn_coefficients"));
                coefficient_count += 1;
            }
        }
    }
    let mut rope_count = 0;
    if rope {
        for (i, kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            if *kind == LayerKind::FullAttention {
                names.push(format!("blk.{i}.attn_rope_kv"));
                rope_count += 1;
            }
        }
    }
    let mut qk_count = 0;
    if qk {
        for (i, kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            if *kind == LayerKind::FullAttention {
                names.push(format!("blk.{i}.attn_qk"));
                qk_count += 1;
            }
        }
    }
    let mut softmax_count = 0;
    if softmax {
        for (i, kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            if *kind == LayerKind::FullAttention {
                names.push(format!("blk.{i}.attn_softmax"));
                softmax_count += 1;
            }
        }
    }
    let mut value_count = 0;
    if value {
        for (i, kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            if *kind == LayerKind::FullAttention {
                names.push(format!("blk.{i}.attn_value"));
                value_count += 1;
            }
        }
    }
    if remaining {
        names.push("embedding".into());
        for (i, kind) in model.geometry().layers.iter().take(blocks).enumerate() {
            for role in ["attn_residual", "ffn_residual", "ffn_silu"] {
                names.push(format!("blk.{i}.{role}"));
            }
            if *kind == LayerKind::FullAttention {
                names.push(format!("blk.{i}.attn_gate"));
            }
        }
    }
    let selected: BTreeSet<_> = names.iter().cloned().collect();
    if selected.len() != names.len() {
        return Err("GPU source sweep accidentally selected duplicate tensors".into());
    }
    let mut by_kind = BTreeMap::<String, usize>::new();
    for name in &matrix_names {
        let codec = format!(
            "{:?}",
            model.tensor_info(name).map_err(|e| e.to_string())?.kind
        );
        *by_kind.entry(codec).or_default() += 1;
    }
    println!(
        "P07_H_SOURCE_PLACEMENT card=physical_vulkan model={} blocks={blocks} projection_count={} norm_regions={scalar_norm_count} grouped_head_norm_regions={grouped_norm_count} gpu_gdn_conv_regions={conv_count} quant_codecs={by_kind:?} output_head={output}",
        path.display(),
        matrix_names.len()
    );
    if token_count == 0 || token_count > 4 {
        return Err("source sweep requires 1..=4 tokens".into());
    }
    let tokens = &[11, 353, 17, 42][..token_count];
    let mut reference = model.new_session().map_err(|e| e.to_string())?;
    let expected = reference
        .prefill_transaction(tokens)
        .map_err(|e| e.to_string())?;
    let provider = Arc::new(
        CubeResidentProvider::new(
            device.clone(),
            names.clone(),
            MAX_SELECTED_ROWS,
            PROVIDER_CAP,
            OPERATOR_CAP,
            ResourceOwner::new("qwen35-closed-GPU-regions", "first-session")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
        .with_device_activation_quantization(device_activation),
    );
    let key = provider.physical_key();
    let underbudget = if recurrent || l2 || silu || coefficients || rope || qk || softmax || value {
        Some(Arc::new(
            CubeResidentProvider::new(
                device,
                names.clone(),
                MAX_SELECTED_ROWS,
                if softmax { 512 } else { 1024 },
                OPERATOR_CAP,
                ResourceOwner::new("recurrent-underbudget", "preflight")
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?,
        ))
    } else {
        None
    };
    let checked = Arc::new(CheckedRegions {
        inner: provider.clone(),
        completed: Mutex::new(BTreeMap::new()),
        nonzero_recurrent: AtomicUsize::new(0),
        preflight_checked: AtomicBool::new(false),
        l2_preflight_checked: AtomicBool::new(false),
        l2_selected: l2,
        silu_selected: silu,
        coefficients_selected: coefficients,
        rope_selected: rope,
        qk_selected: qk,
        softmax_selected: softmax,
        value_selected: value,
        remaining_selected: remaining,
        binary_preflight_checked: AtomicBool::new(false),
        embedding_preflight_checked: AtomicBool::new(false),
        value_preflight_checked: AtomicBool::new(false),
        softmax_preflight_checked: AtomicBool::new(false),
        qk_preflight_checked: AtomicBool::new(false),
        rope_preflight_checked: AtomicBool::new(false),
        coefficients_preflight_checked: AtomicBool::new(false),
        silu_preflight_checked: AtomicUsize::new(0),
        recurrent_selected: recurrent,
        underbudget,
    });
    let sibling = model.new_session().map_err(|e| e.to_string())?;
    let sibling_before = (
        sibling.root_id(),
        sibling.position(),
        sibling.state_fingerprint(),
    );
    let mut accelerated = model
        .new_session()
        .map_err(|e| e.to_string())?
        .with_kernel(checked.clone());
    let old_root = accelerated.root_id();
    let old_position = accelerated.position();
    let old_fingerprint = accelerated.state_fingerprint();
    let actual = match accelerated.prefill_transaction(tokens) {
        Ok(x) => x,
        Err(error) => {
            if accelerated.root_id() != old_root
                || accelerated.position() != old_position
                || accelerated.state_fingerprint() != old_fingerprint
            {
                return Err(format!(
                    "GPU parity rejection mutated committed Qwen StateRoot: {error}"
                ));
            }
            return Err(format!(
                "GPU source operation REFUSED; no partial StateRoot published: {error}"
            ));
        }
    };
    if sibling_before
        != (
            sibling.root_id(),
            sibling.position(),
            sibling.state_fingerprint(),
        )
    {
        return Err("GPU recurrence mutated another shared-model session".into());
    }
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(&expected)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        || reference.state_fingerprint() != accelerated.state_fingerprint()
        || reference.root_id() != accelerated.root_id()
        || reference.position() != accelerated.position()
    {
        return Err("full-Qwen logits/StateRoot mismatch after physical GEMV sweep".into());
    }
    let table = checked
        .completed
        .lock()
        .map_err(|_| "physical proof map mutex poisoned")?;
    let calls = u64::try_from(tokens.len()).map_err(|_| "token count overflow")?;
    // Chunked prefill commits all causal layer work per token, but computes
    // the final normalization and vocabulary logits only for the FINAL
    // prefill token. Counting them for every token would manufacture GPU
    // completions that Qwen deliberately never scheduled.
    let final_only = |name: &str| name == "output_norm.weight" || name == "token_embd.weight";
    if table.len() != names.len()
        || names.iter().any(|n| {
            let expected = if final_only(n) { 1 } else { calls };
            table.get(n) != Some(&expected)
        })
    {
        return Err(format!(
            "selected GPU projection was accidentally CPU-fallback or duplicated: counts={table:?}"
        ));
    }
    drop(table);
    if checked.nonzero_recurrent.load(Ordering::Relaxed)
        < recurrent_count * tokens.len().saturating_sub(1)
    {
        return Err("source GPU recurrent sweep lacked actual nonzero causal state".into());
    }
    let queue = provider.worker_stats().map_err(|e| e.to_string())?;
    let stats = provider.stats().map_err(|e| e.to_string())?;
    let final_only_count = names.iter().filter(|n| final_only(n)).count();
    let total_expected = names
        .len()
        .checked_sub(final_only_count)
        .and_then(|active| active.checked_mul(tokens.len()))
        .and_then(|active| active.checked_add(final_only_count))
        .ok_or("GPU expected physical count overflow")?;
    let final_only_matrices = matrix_names.iter().filter(|n| final_only(n)).count();
    let maximum_weight_uploads = matrix_names
        .len()
        .checked_sub(final_only_matrices)
        .and_then(|active| active.checked_mul(tokens.len()))
        .and_then(|active| active.checked_add(final_only_matrices))
        .and_then(|n| n.checked_add(if remaining { tokens.len() } else { 0 }))
        .ok_or("GPU expected weight count overflow")?;
    let quantized_matrices = matrix_names
        .iter()
        .filter(|name| {
            model
                .tensor_info(name)
                .is_ok_and(|t| t.kind != QuantKind::F32)
        })
        .count();
    let quantized_final_only = matrix_names
        .iter()
        .filter(|name| {
            final_only(name)
                && model
                    .tensor_info(name)
                    .is_ok_and(|t| t.kind != QuantKind::F32)
        })
        .count();
    let expected_device_quantized = if device_activation {
        (quantized_matrices - quantized_final_only) * tokens.len() + quantized_final_only
    } else {
        0
    };
    if stats.device_quantized_regions != expected_device_quantized
        || queue.completed != total_expected as u64
        || queue.active
        || queue.queued != 0
        || stats.completed_regions != total_expected
        || stats.source_uploads < matrix_names.len()
        || stats.source_uploads > maximum_weight_uploads
        || provider
            .in_flight_device_bytes()
            .map_err(|e| e.to_string())?
            != 0
        || provider
            .quarantined_physical_handles()
            .map_err(|e| e.to_string())?
            != 0
    {
        return Err(format!(
            "GPU projection sweep had incomplete physical completion or unsafe cache state: {queue:?} {stats:?}"
        ));
    }
    let checked_nonzero = checked.nonzero_recurrent.load(Ordering::Relaxed);
    drop(accelerated);
    drop(checked);
    drop(provider);
    let snapshot = ResourceArbiter::process_shared()
        .snapshot(&key)
        .map_err(|e| e.to_string())?;
    if snapshot.fenced || snapshot.reserved != 0 {
        return Err(format!(
            "GPU source sweep retained leaked logical charge: {snapshot:?}"
        ));
    }
    println!(
        "P07_H_QWEN_PHYSICAL_REGION_SWEEP_PASS card={key:?} blocks={blocks} tokens={} selected_regions={} selected_matrices={} selected_residual_norms={scalar_norm_count} selected_grouped_head_norms={grouped_norm_count} selected_gdn_convs={conv_count} physically_completed={} source_uploads={} full_vocab_logits_bit_exact=1 state_root_bit_exact=1 released_logical_bytes=0 selected_gdn_recurrent={recurrent_count} nonzero_recurrent_regions={} selected_gdn_qk_l2={l2_count} selected_gdn_silu={silu_count} selected_gdn_coefficients={coefficient_count} selected_rope_kv={rope_count} selected_attention_qk={qk_count} selected_attention_softmax={softmax_count} selected_attention_value={value_count} remaining_operators={remaining} gpu_dynamic_quantization={device_activation} completed_device_quantized_regions={expected_device_quantized} host_intermediate_materialization=1",
        tokens.len(),
        names.len(),
        matrix_names.len(),
        queue.completed,
        stats.source_uploads,
        checked_nonzero
    );
    Ok(())
}

/// Bounded source arithmetic branch/quotient gate, independent of whole-model logits.
pub fn silu_stress(device: Device, path: &Path) -> Result<(), String> {
    let config = path.parent().ok_or("model directory")?.join("config.json");
    let model = LoadedQwenModel::open(path, &config).map_err(|e| e.to_string())?;
    let provider = CubeResidentProvider::new(
        device,
        ["silu.stress".into()],
        MAX_SELECTED_ROWS,
        8 * 1024 * 1024,
        OPERATOR_CAP,
        ResourceOwner::new("silu-stress", "source").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut input = vec![
        0.0f32, -0.0, 1.0, -1.0, 20.0, -20.0, 88.0, -88.0, 89.0, -89.0, 133.0, -133.0, 134.0,
        -134.0, 256.0, -256.0,
    ];
    input.extend([
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MIN_POSITIVE * 2.0,
        -f32::MIN_POSITIVE * 2.0,
    ]);
    input.extend((-2048..=2048).map(|i| i as f32 / 8.0));
    for bits in [
        2, 3, 4, 5, 0x007ffffd, 0x007ffffe, 0x007fffff, 0x00800001, 0x00ffffff, 0x01000001,
    ] {
        input.extend([f32::from_bits(bits), -f32::from_bits(bits)]);
    }
    let mut seed = 0x42a51u32;
    for _ in 0..32000 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        input.push((seed as i32 as f64 / i32::MAX as f64 * 80.0) as f32);
    }
    for _ in 0..2000 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        input.push(f32::from_bits(seed & 0x80ffffff));
    }
    let multiplier: Vec<_> = (0..input.len())
        .map(|i| {
            if i % 7 == 0 {
                -0.0
            } else {
                (i as f32 % 31.0 - 15.0) / 8.0
            }
        })
        .collect();
    for m in [None, Some(multiplier.as_slice())] {
        let r = SiluRegion {
            source_identity: model.source_identity(),
            operation: "silu.stress",
            input: &input,
            multiplier: m,
        };
        let RegionExecution::Completed(actual) = provider.try_silu(r).map_err(|e| e.to_string())?
        else {
            return Err("selected stress Unsupported".into());
        };
        for (i, (&x, &actual)) in input.iter().zip(&actual).enumerate() {
            let y = zerogravity_cpu::ops::ggml_source_silu(x);
            let expected = m.map_or(y, |m| m[i] * y);
            if expected.to_bits() != actual.to_bits() {
                return Err(format!(
                    "stress gated={} i={i} x={x} expected={:08x} actual={:08x}",
                    m.is_some(),
                    expected.to_bits(),
                    actual.to_bits()
                ));
            }
        }
    }
    println!(
        "P07_H7_SILU_ARITHMETIC_PASS inputs={} gated_and_ungated=1 source_bits=1",
        input.len()
    );
    Ok(())
}

fn check_coefficients(
    region: GdnCoefficientRegion<'_>,
    actual: &zerogravity_core::accelerator::GdnCoefficientCompleted,
) -> ZgResult<()> {
    actual.validate(region.alpha.len())?;
    let expected = source_gdn_coefficients(region)?;
    for h in 0..region.alpha.len() {
        let a = &actual.coefficients[h];
        let e = &expected.coefficients[h];
        for (role, a, e) in [
            ("softplus", actual.softplus[h], expected.softplus[h]),
            ("gate_raw", actual.gate_raw[h], expected.gate_raw[h]),
            ("gain", a.gain, e.gain),
            ("beta", a.beta, e.beta),
            ("scale", a.scale, e.scale),
        ] {
            if a.to_bits() != e.to_bits() {
                return Err(Error::State(format!(
                    "coefficient source bits FAIL {} head={h} {role} alpha={} dt={} beta={} A={} expected={:08x} actual={:08x}; refusing StateRoot",
                    region.operation,
                    region.alpha[h],
                    region.dt[h],
                    region.beta[h],
                    region.a[h],
                    e.to_bits(),
                    a.to_bits()
                )));
            }
        }
    }
    Ok(())
}

pub fn coefficient_stress(device: Device, path: &Path) -> Result<(), String> {
    let model = LoadedQwenModel::open(
        path,
        &path.parent().ok_or("model directory")?.join("config.json"),
    )
    .map_err(|e| e.to_string())?;
    let provider = CubeResidentProvider::new(
        device,
        ["coefficient.stress".into()],
        MAX_SELECTED_ROWS,
        8 * 1024 * 1024,
        OPERATOR_CAP,
        ResourceOwner::new("coefficient-stress", "source").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut alpha = vec![
        0.0f32,
        -0.0,
        20.0,
        -20.0,
        20.000002,
        -20.000002,
        128.0,
        -128.0,
        88.0,
        -88.0,
        89.0,
        -89.0,
        103.0,
        -103.0,
        104.0,
        -104.0,
        32.564632,
        -32.564632,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
    ];
    alpha.extend((-1024..=1024).map(|i| i as f32 / 8.));
    let mut seed = 42u32;
    for _ in 0..32000 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        alpha.push((seed as i32 as f64 / i32::MAX as f64 * 128.) as f32);
    }
    let boundary_start = alpha.len();
    // Adjacent FP32 inputs around exp table's half-cell boundaries, and
    // overflow/underflow cutoffs. Beta also sees these through reversal.
    for k in (-4800..4096).step_by(4) {
        let x = ((k as f64 + 0.5) * std::f64::consts::LN_2 / 32.0) as f32;
        alpha.extend([
            f32::from_bits(x.to_bits() - 1),
            x,
            f32::from_bits(x.to_bits() + 1),
        ]);
    }
    for bits in [0x42b17217u32, 0x42cff1b4u32] {
        for neighbor in [bits - 1, bits, bits + 1] {
            alpha.extend([f32::from_bits(neighbor), -f32::from_bits(neighbor)]);
        }
    }
    let n = alpha.len();
    let beta: Vec<_> = alpha.iter().rev().copied().collect();
    let dt: Vec<_> = (0..n)
        .map(|i| {
            if i < 22 || i >= boundary_start {
                0.
            } else {
                (i % 31) as f32 - 15.
            }
        })
        .collect();
    let a: Vec<_> = (0..n)
        .map(|i| [-0.0, -0.01, -1., -4., -16., -1e6][i % 6])
        .collect();
    let region = GdnCoefficientRegion {
        source_identity: model.source_identity(),
        operation: "coefficient.stress",
        alpha: &alpha,
        beta: &beta,
        dt: &dt,
        a: &a,
        scale: 1. / 128f32.sqrt(),
    };
    let GdnCoefficientExecution::Completed(ready) = provider
        .try_gdn_coefficients(region)
        .map_err(|e| e.to_string())?
    else {
        return Err("selected coefficient stress Unsupported".into());
    };
    check_coefficients(region, &ready).map_err(|e| e.to_string())?;
    println!(
        "P07_H8_COEFFICIENT_ARITHMETIC_PASS heads={n} source_fp32_values={} bits_exact=1",
        n * 5
    );
    Ok(())
}

fn check_rope(region: RopeKvRegion<'_>, actual: &RopeKvCompleted) -> ZgResult<()> {
    actual.validate(region)?;
    let expected = source_rope_kv(region)?;
    for (role, a, e) in [
        ("query", &actual.query, &expected.query),
        ("key", &actual.key, &expected.key),
        ("value", &actual.value, &expected.value),
    ] {
        if let Some(i) = a
            .iter()
            .zip(e)
            .position(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(Error::State(format!(
                "source RoPE/KV bits FAIL {} position={} {role}[{i}] expected={:08x} actual={:08x}; refusing StateRoot",
                region.operation,
                region.position,
                e[i].to_bits(),
                a[i].to_bits()
            )));
        }
    }
    Ok(())
}

struct RopeSample {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    width: usize,
    qheads: usize,
    kvheads: usize,
    rotate: usize,
    scale: f32,
}
struct CaptureRope(Mutex<Option<RopeSample>>);
impl MatVecKernel for CaptureRope {
    fn try_execute(&self, _: MatVecRegion<'_>) -> ZgResult<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }
    fn try_rope_kv(&self, r: RopeKvRegion<'_>) -> ZgResult<RopeKvExecution> {
        if r.position == 1 && r.operation == "blk.3.attn_rope_kv" {
            *self
                .0
                .lock()
                .map_err(|_| Error::State("capture lock".into()))? = Some(RopeSample {
                q: r.query.to_vec(),
                k: r.key.to_vec(),
                v: r.value.to_vec(),
                width: r.head_width,
                qheads: r.query_heads,
                kvheads: r.kv_heads,
                rotate: r.rotate,
                scale: r.theta_scale,
            });
        }
        Ok(RopeKvExecution::Unsupported)
    }
}
pub fn rope_stress(device: Device, path: &Path) -> Result<(), String> {
    rope_stress_with_device_cap(device, path, OPERATOR_CAP)
}

pub(crate) fn rope_stress_with_device_cap(
    device: Device,
    path: &Path,
    device_cap: u64,
) -> Result<(), String> {
    let model = LoadedQwenModel::open(
        path,
        &path.parent().ok_or("model directory")?.join("config.json"),
    )
    .map_err(|e| e.to_string())?;
    let capture = Arc::new(CaptureRope(Mutex::new(None)));
    let mut session = model
        .new_session()
        .map_err(|e| e.to_string())?
        .with_kernel(capture.clone());
    session
        .prefill_transaction(&[11, 353])
        .map_err(|e| e.to_string())?;
    let mut sample = capture
        .0
        .lock()
        .map_err(|_| "capture lock")?
        .take()
        .ok_or("no actual trained nonzero-position RoPE inputs")?;
    let provider = CubeResidentProvider::new(
        device,
        ["rope.stress".into()],
        MAX_SELECTED_ROWS,
        8 * 1024 * 1024,
        device_cap,
        ResourceOwner::new("rope-stress", "source").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let positions = [
        0, 1, 2, 3, 7, 31, 119, 120, 121, 127, 128, 255, 256, 1023, 4095, 4096, 8191, 8192, 16383,
        16384, 32767, 32768,
    ];
    for edge in [false, true] {
        if edge {
            for (i, x) in [
                0.0,
                -0.0,
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                f32::from_bits(1),
                -f32::from_bits(1),
            ]
            .into_iter()
            .enumerate()
            {
                sample.q[i] = x;
                sample.q[i + 32] = x;
                sample.k[i] = x;
                sample.k[i + 32] = x;
                sample.v[i] = x;
                sample.q[64 + i] = x;
                sample.k[64 + i] = x;
            }
        }
        if edge {
            sample.q[0] = 1.0078125;
            sample.q[32] = -f32::MIN_POSITIVE;
            sample.k[0] = 1.0078125;
            sample.k[32] = -f32::MIN_POSITIVE;
        }
        for position in positions {
            let r = RopeKvRegion {
                source_identity: model.source_identity(),
                operation: "rope.stress",
                query: &sample.q,
                key: &sample.k,
                value: &sample.v,
                head_width: sample.width,
                query_heads: sample.qheads,
                kv_heads: sample.kvheads,
                rotate: sample.rotate,
                position,
                theta_scale: sample.scale,
            };
            let RopeKvExecution::Completed(ready) =
                provider.try_rope_kv(r).map_err(|e| e.to_string())?
            else {
                return Err("selected RoPE stress Unsupported".into());
            };
            check_rope(r, &ready).map_err(|e| e.to_string())?;
        }
    }
    println!(
        "P07_H9_TRAINED_ROPE_OPERATOR_STRESS_PASS positions={} max_position=32768 edge_cohort=1 bits_exact=1 full_long_context_not_claimed=1",
        positions.len()
    );
    Ok(())
}

fn check_qk(r: AttentionQkRegion<'_>, actual: &AttentionQkCompleted) -> ZgResult<()> {
    actual.validate(r)?;
    let expected = source_attention_qk(r)?;
    for (name, a, b) in [
        ("dots", &actual.dots, &expected.dots),
        ("scores", &actual.scores, &expected.scores),
    ] {
        if let Some(i) = a
            .iter()
            .zip(b)
            .position(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(Error::State(format!(
                "source QK bits FAIL {} prefix={} {name}[{i}] expected={:08x} actual={:08x}; refusing StateRoot",
                r.operation,
                r.keys.len(),
                b[i].to_bits(),
                a[i].to_bits()
            )));
        }
    }
    Ok(())
}
struct QkSample {
    query: Vec<f32>,
    keys: Vec<Arc<[f32]>>,
    width: usize,
    qheads: usize,
    kvheads: usize,
    scale: f32,
}
struct CaptureQk(Mutex<Option<QkSample>>);
impl MatVecKernel for CaptureQk {
    fn try_execute(&self, _: MatVecRegion<'_>) -> ZgResult<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }
    fn try_attention_qk(&self, r: AttentionQkRegion<'_>) -> ZgResult<AttentionQkExecution> {
        if r.keys.len() == 4 && r.operation == "blk.3.attn_qk" {
            *self
                .0
                .lock()
                .map_err(|_| Error::State("capture QK lock".into()))? = Some(QkSample {
                query: r.query.to_vec(),
                keys: r.keys.to_vec(),
                width: r.head_width,
                qheads: r.query_heads,
                kvheads: r.kv_heads,
                scale: r.scale,
            });
        }
        Ok(AttentionQkExecution::Unsupported)
    }
}
pub fn qk_stress(device: Device, path: &Path) -> Result<(), String> {
    let model = LoadedQwenModel::open(
        path,
        &path.parent().ok_or("model parent")?.join("config.json"),
    )
    .map_err(|e| e.to_string())?;
    let capture = Arc::new(CaptureQk(Mutex::new(None)));
    let mut session = model.new_session().map_err(|e| e.to_string())?;
    session = session.with_kernel(capture.clone());
    session
        .prefill_transaction(&[11, 353, 17, 42])
        .map_err(|e| e.to_string())?;
    let sample = capture
        .0
        .lock()
        .map_err(|_| "capture QK lock")?
        .take()
        .ok_or("missing genuine QK operands")?;
    let provider = CubeResidentProvider::new(
        device,
        ["qk.stress".into()],
        MAX_SELECTED_ROWS,
        PROVIDER_CAP,
        OPERATOR_CAP,
        ResourceOwner::new("qk-stress", "source").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let prefixes = [
        1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 4095, 4096,
    ];
    let mut count = 0;
    let mut comparisons = 0;
    for cohort in 0..3 {
        let mut q = sample.query.clone();
        let mut originals = sample.keys.iter().map(|k| k.to_vec()).collect::<Vec<_>>();
        if cohort == 1 {
            for (i, x) in q.iter_mut().enumerate() {
                *x = match i % 6 {
                    0 => -0.0,
                    1 => f32::from_bits(1),
                    2 => -f32::from_bits(1),
                    3 => f32::MIN_POSITIVE,
                    4 => -f32::MIN_POSITIVE,
                    _ => 0.0,
                };
            }
            for (t, key) in originals.iter_mut().enumerate() {
                for (i, x) in key.iter_mut().enumerate() {
                    *x = match (i + t) % 4 {
                        0 => 1.0,
                        1 => -1.0,
                        2 => 0.5,
                        _ => -0.5,
                    };
                }
            }
            // Every accumulator of head3 ends with a negative half-minimum
            // product. Its raw reduction must retain -0 through every level.
            let start = 3 * sample.width;
            q[start..start + sample.width].fill(0.0);
            q[start + sample.width - 32..start + sample.width].fill(-f32::from_bits(1));
            for key in &mut originals {
                key[sample.width - 32..sample.width].fill(0.5);
            }
        } else if cohort == 2 {
            // Force a midpoint decision after a real nonzero first FMA block.
            q[..sample.width].fill(0.0);
            q[0] = f32::MIN_POSITIVE;
            q[32] = 1.0078125;
            for key in &mut originals {
                key[..sample.width].fill(0.0);
                key[0] = 1.0;
                key[32] = f32::from_bits(0x3f0a5140);
            }
        }
        for prefix in prefixes {
            // Only prefix4 is the genuine model cache. Other lengths repeat
            // genuine operand slabs for operator geometry, not model contexts.
            let keys: Vec<Arc<[f32]>> = (0..prefix)
                .map(|i| Arc::from(originals[i % originals.len()].clone()))
                .collect();
            let r = AttentionQkRegion {
                source_identity: model.source_identity(),
                operation: "qk.stress",
                query: &q,
                keys: &keys,
                head_width: sample.width,
                query_heads: sample.qheads,
                kv_heads: sample.kvheads,
                scale: sample.scale,
            };
            let AttentionQkExecution::Completed(actual) =
                provider.try_attention_qk(r).map_err(|e| e.to_string())?
            else {
                return Err("selected QK unsupported".into());
            };
            check_qk(r, &actual).map_err(|e| e.to_string())?;
            count += 1;
            comparisons += actual.dots.len() + actual.scores.len();
        }
    }
    for width in [32, 64, 128, 256, 512] {
        let q: Vec<_> = (0..sample.qheads * width)
            .map(|i| sample.query[(i / width) * sample.width + i % width % sample.width])
            .collect();
        let keys: Vec<Arc<[f32]>> = (0..33)
            .map(|t| {
                Arc::from(
                    (0..sample.kvheads * width)
                        .map(|i| {
                            sample.keys[t % 4]
                                [(i / width) * sample.width + i % width % sample.width]
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let r = AttentionQkRegion {
            source_identity: model.source_identity(),
            operation: "qk.stress",
            query: &q,
            keys: &keys,
            head_width: width,
            query_heads: sample.qheads,
            kv_heads: sample.kvheads,
            scale: 1.0 / (width as f32).sqrt(),
        };
        let AttentionQkExecution::Completed(actual) =
            provider.try_attention_qk(r).map_err(|e| e.to_string())?
        else {
            return Err("QK width unsupported".into());
        };
        check_qk(r, &actual).map_err(|e| e.to_string())?;
        count += 1;
        comparisons += actual.dots.len() + actual.scores.len();
    }
    println!(
        "P07_H10_SOURCE_QK_OPERATOR_STRESS_PASS regions={count} compared_fp32={comparisons} prefixes=22 max_prefix=4096 widths=5 cohorts=3 exact_bits=1 constructed_operator_prefixes_not_model_contexts=1"
    );
    Ok(())
}

fn check_softmax(
    r: AttentionSoftmaxRegion<'_>,
    actual: &AttentionSoftmaxCompleted,
) -> ZgResult<()> {
    actual.validate(r)?;
    let expected = source_attention_softmax(r)?;
    for (name, a, b) in [
        ("exponentials", &actual.exponentials, &expected.exponentials),
        (
            "probabilities",
            &actual.probabilities,
            &expected.probabilities,
        ),
        ("inverse", &actual.inverse, &expected.inverse),
    ] {
        if let Some(i) = a
            .iter()
            .zip(b)
            .position(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(Error::State(format!(
                "source softmax bits FAIL {} width={} {name}[{i}] expected={:08x} actual={:08x}; refusing StateRoot",
                r.operation,
                r.width,
                b[i].to_bits(),
                a[i].to_bits()
            )));
        }
    }
    for (name, a, b) in [
        ("sums", &actual.sums, &expected.sums),
        ("reciprocals", &actual.reciprocals, &expected.reciprocals),
    ] {
        if let Some(i) = a
            .iter()
            .zip(b)
            .position(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(Error::State(format!(
                "source softmax double bits FAIL {} width={} {name}[{i}] expected={:016x} actual={:016x}",
                r.operation,
                r.width,
                b[i].to_bits(),
                a[i].to_bits()
            )));
        }
    }
    Ok(())
}
struct CaptureSoftmax(Mutex<Option<Vec<f32>>>);
impl MatVecKernel for CaptureSoftmax {
    fn try_execute(&self, _: MatVecRegion<'_>) -> ZgResult<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }
    fn try_attention_softmax(
        &self,
        r: AttentionSoftmaxRegion<'_>,
    ) -> ZgResult<AttentionSoftmaxExecution> {
        if r.operation == "blk.3.attn_softmax" && r.width == 4 {
            *self
                .0
                .lock()
                .map_err(|_| Error::State("softmax capture lock".into()))? =
                Some(r.scores.to_vec());
        }
        Ok(AttentionSoftmaxExecution::Unsupported)
    }
}
pub fn softmax_stress(device: Device, path: &Path) -> Result<(), String> {
    let model = LoadedQwenModel::open(
        path,
        &path.parent().ok_or("model parent")?.join("config.json"),
    )
    .map_err(|e| e.to_string())?;
    let capture = Arc::new(CaptureSoftmax(Mutex::new(None)));
    let mut session = model
        .new_session()
        .map_err(|e| e.to_string())?
        .with_kernel(capture.clone());
    session
        .prefill_transaction(&[11, 353, 17, 42])
        .map_err(|e| e.to_string())?;
    let trained = capture
        .0
        .lock()
        .map_err(|_| "capture softmax lock")?
        .take()
        .ok_or("missing genuine softmax scores")?;
    let heads = trained.len() / 4;
    let provider = CubeResidentProvider::new(
        device,
        ["softmax.stress".into()],
        MAX_SELECTED_ROWS,
        PROVIDER_CAP,
        OPERATOR_CAP,
        ResourceOwner::new("softmax-stress", "source").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    futures_lite::future::block_on(provider.qualify_softmax_reciprocals())
        .map_err(|e| e.to_string())?;
    let widths = [
        1, 2, 3, 4, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 4095,
        4096,
    ];
    let mut regions = 0;
    let mut fp32 = 0;
    let mut fp64 = 0;
    for cohort in 0..5 {
        for width in widths {
            let scores: Vec<_> = (0..heads * width)
                .map(|i| {
                    let h = i / width;
                    let lane = i % width;
                    match cohort {
                        0 => trained[h * 4 + lane % 4],
                        1 => match lane % 5 {
                            0 => -0.0,
                            1 => f32::from_bits(1),
                            2 => -f32::from_bits(1),
                            3 => f32::MIN_POSITIVE,
                            _ => -f32::MIN_POSITIVE,
                        },
                        2 => [
                            0.0,
                            -0.01,
                            -1.0,
                            -80.0,
                            -87.0,
                            -88.0,
                            -89.0,
                            -90.0,
                            -100.0,
                            -103.0,
                            -104.0,
                            -128.0,
                            -129.0,
                            -f32::MAX,
                        ][lane % 14],
                        3 => {
                            if lane == 0 {
                                f32::MAX
                            } else {
                                -f32::MAX
                            }
                        }
                        _ => {
                            if lane == 0 {
                                0.0
                            } else {
                                -(((lane * 717 + h * 31) % 131072) as f32) / 1024.0
                            }
                        }
                    }
                })
                .collect();
            let r = AttentionSoftmaxRegion {
                source_identity: model.source_identity(),
                operation: "softmax.stress",
                scores: &scores,
                heads,
                width,
            };
            let AttentionSoftmaxExecution::Completed(actual) = provider
                .try_attention_softmax(r)
                .map_err(|e| e.to_string())?
            else {
                return Err("selected softmax Unsupported".into());
            };
            check_softmax(r, &actual).map_err(|e| e.to_string())?;
            regions += 1;
            fp32 += actual.exponentials.len() + actual.probabilities.len() + actual.inverse.len();
            fp64 += actual.sums.len() + actual.reciprocals.len();
        }
    }
    println!(
        "P07_H11_SOURCE_SOFTMAX_OPERATOR_STRESS_PASS regions={regions} compared_fp32={fp32} compared_fp64={fp64} widths=23 max_width=4096 cohorts=5 exact_bits=1 constructed_operator_rows_not_model_contexts=1"
    );
    Ok(())
}

fn check_value(r: AttentionValueRegion<'_>, actual: &[f32]) -> ZgResult<()> {
    r.validate_completed(actual)?;
    let expected = source_attention_value(r)?;
    if let Some(i) = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(Error::State(format!(
            "source value bits differ op={} offset={i} expected={:08x} actual={:08x}",
            r.operation,
            expected[i].to_bits(),
            actual[i].to_bits()
        )));
    }
    Ok(())
}

struct ValueSample {
    probabilities: Vec<f32>,
    values: Vec<Arc<[f32]>>,
    width: usize,
    qheads: usize,
    kvheads: usize,
}
struct CaptureValue(Mutex<Option<ValueSample>>);
impl MatVecKernel for CaptureValue {
    fn try_execute(&self, _: MatVecRegion<'_>) -> ZgResult<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }
    fn try_attention_value(&self, r: AttentionValueRegion<'_>) -> ZgResult<RegionExecution> {
        if r.operation == "blk.3.attn_value" && r.values.len() == 4 {
            *self
                .0
                .lock()
                .map_err(|_| Error::State("value capture lock".into()))? = Some(ValueSample {
                probabilities: r.probabilities.to_vec(),
                values: r.values.to_vec(),
                width: r.head_width,
                qheads: r.query_heads,
                kvheads: r.kv_heads,
            });
        }
        Ok(RegionExecution::Unsupported)
    }
}
pub fn value_stress(device: Device, path: &Path) -> Result<(), String> {
    let model = LoadedQwenModel::open(
        path,
        &path.parent().ok_or("model parent")?.join("config.json"),
    )
    .map_err(|e| e.to_string())?;
    let capture = Arc::new(CaptureValue(Mutex::new(None)));
    model
        .new_session()
        .map_err(|e| e.to_string())?
        .with_kernel(capture.clone())
        .prefill_transaction(&[11, 353, 17, 42])
        .map_err(|e| e.to_string())?;
    let trained = capture
        .0
        .lock()
        .map_err(|_| "value capture lock")?
        .take()
        .ok_or("missing trained value operands")?;
    let provider = CubeResidentProvider::new(
        device,
        ["value.stress".into()],
        MAX_SELECTED_ROWS,
        PROVIDER_CAP,
        OPERATOR_CAP,
        ResourceOwner::new("value-stress", "source").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let widths = [
        1usize, 2, 3, 4, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 224, 225, 255,
        256, 257, 4095, 4096,
    ];
    let mut regions = 0;
    let mut fp32 = 0;
    for cohort in 0..5 {
        for tokens in widths {
            let probabilities: Vec<f32> = (0..trained.qheads * tokens)
                .map(|i| match cohort {
                    0 => trained.probabilities[(i / tokens) * 4 + i % tokens % 4],
                    1 => 1.0 / tokens as f32,
                    2 => {
                        if i % tokens == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    3 => f32::from_bits(1),
                    _ => {
                        if i % tokens % 2 == 0 {
                            0.5
                        } else {
                            f32::from_bits(0x00800000)
                        }
                    }
                })
                .collect();
            let values: Vec<Arc<[f32]>> = (0..tokens)
                .map(|t| {
                    if cohort == 0 {
                        return trained.values[t % 4].clone();
                    }
                    (0..trained.kvheads * trained.width)
                        .map(|j| match cohort {
                            1 => -f32::from_bits(1),
                            2 => {
                                if (j + t) % 2 == 0 {
                                    -0.0
                                } else {
                                    0.0
                                }
                            }
                            3 => [f32::MAX, -f32::MAX, f32::MIN_POSITIVE, -f32::MIN_POSITIVE]
                                [(j + t) % 4],
                            _ => {
                                f32::from_bits(
                                    0x3f000000 + ((j * 13 + t * 727) % 0x01000000) as u32,
                                ) * if (j + t) % 2 == 0 { 1.0 } else { -1.0 }
                            }
                        })
                        .collect::<Vec<_>>()
                        .into()
                })
                .collect();
            let r = AttentionValueRegion {
                source_identity: model.source_identity(),
                operation: "value.stress",
                probabilities: &probabilities,
                values: &values,
                head_width: trained.width,
                query_heads: trained.qheads,
                kv_heads: trained.kvheads,
                padded_tokens: tokens.div_ceil(256) * 256,
            };
            let RegionExecution::Completed(actual) =
                provider.try_attention_value(r).map_err(|e| e.to_string())?
            else {
                return Err("selected value Unsupported".into());
            };
            check_value(r, &actual).map_err(|e| e.to_string())?;
            regions += 1;
            fp32 += actual.len();
        }
    }
    // Exact midpoint product plus a tiny prior in the SAME SIMD lane.
    // Rounding the F64 sum directly loses the tiny positive residual.
    let tokens = 33;
    let mut probabilities = vec![0.0; trained.qheads * tokens];
    for h in 0..trained.qheads {
        probabilities[h * tokens] = f32::MIN_POSITIVE;
        probabilities[h * tokens + 32] = f32::from_bits(0x3f0a5140);
    }
    let mut values: Vec<Arc<[f32]>> =
        vec![Arc::from(vec![1.0; trained.kvheads * trained.width]); tokens];
    values[32] = Arc::from(vec![1.0078125; trained.kvheads * trained.width]);
    let r = AttentionValueRegion {
        source_identity: model.source_identity(),
        operation: "value.stress",
        probabilities: &probabilities,
        values: &values,
        head_width: trained.width,
        query_heads: trained.qheads,
        kv_heads: trained.kvheads,
        padded_tokens: 256,
    };
    let RegionExecution::Completed(actual) =
        provider.try_attention_value(r).map_err(|e| e.to_string())?
    else {
        return Err("midpoint value Unsupported".into());
    };
    check_value(r, &actual).map_err(|e| e.to_string())?;
    if actual.iter().any(|x| x.to_bits() != 0x3f0b65e3) {
        return Err("value midpoint residual fixture failed".into());
    }
    regions += 1;
    fp32 += actual.len();
    println!(
        "P07_H12_SOURCE_VALUE_OPERATOR_STRESS_PASS regions={regions} compared_fp32={fp32} widths=25 max_width=4096 cohorts=5 exact_midpoint_fixture=1 exact_bits=1 constructed_operator_rows_not_model_contexts=1"
    );
    Ok(())
}

fn check_remaining(name: &str, expected: &[f32], actual: &[f32]) -> ZgResult<()> {
    if expected.len() != actual.len() {
        return Err(Error::State(format!("remaining operator size {name}")));
    }
    if let Some(i) = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(Error::State(format!(
            "remaining source bits {name} offset={i} expected={:08x} actual={:08x}",
            expected[i].to_bits(),
            actual[i].to_bits()
        )));
    }
    Ok(())
}
