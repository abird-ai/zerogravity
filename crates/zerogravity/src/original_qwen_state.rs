//! One original Qwen analytical layer, admitted by the existing TaskCoordinator.
//! No independent token loop, trained reference or second StateRoot issuer.
//! State slots use the SAME ForwardPlan StateSlot vocabulary; session bytes
//! are provisional FP32 diagnostics, not a persistent checkpoint format.

use crate::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
use crate::accelerator::{BinaryOperation, BinaryRegion};
use crate::model::LayerKind;
use crate::original_qwen_math::{OriginalFullPrefix, OriginalGdnPrefix, OriginalQwenLayerMath};
use crate::state::{CommittedOutput, PreparedDomain, StateRoot, TaskCoordinator};
use crate::{Error, Result};
use std::sync::Arc;

pub const ORIGINAL_KV_V1: u64 = 0x5157_3801;
pub const ORIGINAL_CONV_V1: u64 = 0x5157_3802;
pub const ORIGINAL_RECURRENCE_V1: u64 = 0x5157_3803;
const MAX_DOMAIN_BYTES: usize = 64 * 1024 * 1024;
const MAX_DIAGNOSTIC_TOKENS: u64 = 8192;
const MAX_AGGREGATE_STATE_BYTES: usize = 128 * 1024 * 1024;
const MAX_QUALIFIED_LAYERS: usize = 128;
fn named(slot: StateSlot) -> (String, u64) {
    match slot {
        StateSlot::Kv(KvDomain(i)) => (format!("qwen.original.kv.{i}"), ORIGINAL_KV_V1),
        StateSlot::Conv(ConvDomain(i)) => (format!("qwen.original.conv.{i}"), ORIGINAL_CONV_V1),
        StateSlot::Recurrent(RecurrentDomain(i)) => (
            format!("qwen.original.recurrent.{i}"),
            ORIGINAL_RECURRENCE_V1,
        ),
    }
}
fn encode(values: &[f32]) -> Result<Vec<u8>> {
    let size = values
        .len()
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("Qwen domain byte overflow".into()))?;
    if size > MAX_DOMAIN_BYTES || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Limit(
            "Qwen domain exceeds finite diagnostic budget".into(),
        ));
    }
    let mut out = Vec::with_capacity(size);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}
fn decode(raw: &[u8], count: usize) -> Result<Vec<f32>> {
    let size = count
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("Qwen state count overflow".into()))?;
    if size > MAX_DOMAIN_BYTES || raw.len() != size {
        return Err(Error::Malformed(
            "Qwen original state domain bytes/length".into(),
        ));
    }
    let result = raw
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().expect("four byte chunks")))
        .collect::<Vec<_>>();
    if result.iter().any(|v| !v.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite committed Qwen domain bytes".into(),
        ));
    }
    Ok(result)
}

/// An unpublished, source-pinned proposal for the existing coordinator.
/// Retaining this borrow prevents its source allocation/host RAM lease from
/// being destroyed before the publication decision.
#[must_use = "dropping a prepared Qwen numerical layer does not publish a root"]
pub struct OriginalPreparedLayer<'m, 'b, 's> {
    math: &'m OriginalQwenLayerMath<'b, 's>,
    coordinator_id: u64,
    epoch: u64,
    base_root: u64,
    next_frontier: u64,
    domains: Vec<PreparedDomain>,
    output: Vec<f32>,
}
impl OriginalPreparedLayer<'_, '_, '_> {
    pub fn output(&self) -> &[f32] {
        &self.output
    }
    pub fn base_root(&self) -> u64 {
        self.base_root
    }
    pub fn proposed_frontier(&self) -> u64 {
        self.next_frontier
    }
    pub fn required_domains(&self) -> usize {
        self.domains.len()
    }
    pub fn publish(self, coordinator: &mut TaskCoordinator) -> Result<CommittedOutput> {
        if coordinator.coordinator_id() != self.coordinator_id
            || coordinator.epoch != self.epoch
            || coordinator.source_identity() != Some(self.math.source_identity())
            || coordinator.root().id != self.base_root
            || coordinator.root().frontier.checked_add(1) != Some(self.next_frontier)
        {
            return Err(Error::State(
                "stale/foreign model, session or root for Qwen publish".into(),
            ));
        }
        let payload = encode(&self.output)?; // all errors before publication
        coordinator.publish_source_bound(
            self.math.source_identity(),
            self.base_root,
            self.next_frontier,
            self.domains,
            payload,
        )
    }
}

/// Complete analytical token candidate: all source-bound original
/// layer state updates and final diagnostic logits are one immutable
/// proposal. The same TaskCoordinator publication path is the ONLY
/// way to commit the whole token, never partial individual layers.
#[must_use = "an original Qwen token proposal is not a published state root"]
pub struct OriginalPreparedToken<'m, 'b, 's> {
    step: OriginalPreparedLayer<'m, 'b, 's>,
}
impl OriginalPreparedToken<'_, '_, '_> {
    pub fn logits(&self) -> &[f32] {
        self.step.output()
    }
    pub fn proposed_frontier(&self) -> u64 {
        self.step.proposed_frontier()
    }
    pub fn base_root(&self) -> u64 {
        self.step.base_root()
    }
    pub fn required_domains(&self) -> usize {
        self.step.required_domains()
    }
    pub fn publish(self, coordinator: &mut TaskCoordinator) -> Result<CommittedOutput> {
        self.step.publish(coordinator)
    }
}

/// Pure source-specific domain/codec bridge. It borrows the original
/// already validated Qwen numerics; it does not own TaskCoordinator or
/// implement a separate source/session execution scheduler.
pub struct OriginalQwenStateBridge<'m, 'b, 's> {
    math: &'m OriginalQwenLayerMath<'b, 's>,
}
impl<'m, 'b, 's> OriginalQwenStateBridge<'m, 'b, 's> {
    pub fn new(math: &'m OriginalQwenLayerMath<'b, 's>) -> Self {
        Self { math }
    }
    pub fn typed_domains(&self, layer: usize) -> Result<Vec<StateSlot>> {
        match self.math.geometry().layers.get(layer) {
            Some(LayerKind::FullAttention) => Ok(vec![StateSlot::Kv(KvDomain(layer))]),
            Some(LayerKind::LinearAttention) => Ok(vec![
                StateSlot::Conv(ConvDomain(layer)),
                StateSlot::Recurrent(RecurrentDomain(layer)),
            ]),
            _ => Err(Error::Unsupported(
                "unregistered Qwen layer for source state bridge".into(),
            )),
        }
    }
    /// Register ALL GDN convolution/recurrent and full-attention KV state
    /// domains on the existing source-bound TaskCoordinator. This extends
    /// the previous single-layer diagnostic; it never owns another session.
    pub fn coordinator_for_model(&self, epoch: u64) -> Result<TaskCoordinator> {
        let layers = &self.math.geometry().layers;
        if layers.is_empty() || layers.len() > MAX_QUALIFIED_LAYERS {
            return Err(Error::Limit("original Qwen diagnostic layer budget".into()));
        }
        let mut domains = Vec::new();
        for layer in 0..layers.len() {
            domains.extend(self.typed_domains(layer)?.into_iter().map(named));
        }
        TaskCoordinator::new_source_bound(epoch, self.math.source_identity(), domains)
    }

    /// Validate the COMPLETE state-domain schema, not a subset. A root
    /// missing a late layer must refuse before computing any next token.
    fn validate_model_owner(&self, coordinator: &TaskCoordinator) -> Result<()> {
        if coordinator.source_identity() != Some(self.math.source_identity())
            || coordinator.root().frontier >= MAX_DIAGNOSTIC_TOKENS
        {
            return Err(Error::State(
                "foreign model root or exhausted original Qwen context".into(),
            ));
        }
        let roles = &self.math.geometry().layers;
        if roles.is_empty() || roles.len() > MAX_QUALIFIED_LAYERS {
            return Err(Error::Limit(
                "unsupported Qwen model diagnostic layer count".into(),
            ));
        }
        let mut expected = 0usize;
        for layer in 0..roles.len() {
            for slot in self.typed_domains(layer)? {
                let (name, schema) = named(slot);
                if coordinator
                    .root()
                    .domains
                    .get(&name)
                    .is_none_or(|v| v.schema != schema)
                {
                    return Err(Error::State(
                        "missing/mismatched original Qwen full-model domain".into(),
                    ));
                }
                expected = expected
                    .checked_add(1)
                    .ok_or_else(|| Error::Limit("original model state domain count".into()))?;
            }
        }
        if expected != coordinator.root().domains.len() {
            return Err(Error::State(
                "extra original Qwen model state domain".into(),
            ));
        }
        Ok(())
    }

    /// Analytical FP32-only full-token DIAGNOSTIC. The source math shares
    /// Qwen's single monomorphized layer traversal but is NOT trained BF16
    /// mathematical-backend parity; it cannot enable public run/serve.
    /// Every domain comes from one root and all next domains are staged
    /// before atomic publication by the existing TaskCoordinator.
    pub fn prepare_diagnostic_token_f32(
        &self,
        coordinator: &TaskCoordinator,
        token: u32,
    ) -> Result<OriginalPreparedToken<'m, 'b, 's>> {
        self.validate_model_owner(coordinator)?;
        let root = coordinator.root();
        let next_frontier = root
            .frontier
            .checked_add(1)
            .ok_or_else(|| Error::Limit("original Qwen next token frontier".into()))?;
        // Model-source token index/embedding validation occurs before
        // registering a single tentative state change.
        let embedded = self.math.embed_token(token)?;
        if embedded.len() != self.math.hidden_width()? {
            return Err(Error::State("original Qwen embedding hidden width".into()));
        }
        let mut domains = Vec::<PreparedDomain>::new();
        let mut staged_bytes = 0usize;
        let last = crate::qwen35::run_layer_sequence(
            &self.math.geometry().layers,
            embedded,
            |layer, _kind, residual| {
                let (attention, mut proposed) =
                    self.prepare_layer(layer, root, &residual, next_frontier)?;
                for entry in &proposed {
                    staged_bytes =
                        staged_bytes.checked_add(entry.bytes.len()).ok_or_else(|| {
                            Error::Limit("original Qwen total state staging overflow".into())
                        })?;
                    if staged_bytes > MAX_AGGREGATE_STATE_BYTES {
                        return Err(Error::Limit(
                            "original Qwen total state exceeds diagnostic bound".into(),
                        ));
                    }
                }
                domains.append(&mut proposed);
                let after_attention = crate::qwen35::source_binary(BinaryRegion {
                    source_identity: self.math.source_identity(),
                    operation: "qwen.original.model.attention_residual",
                    kind: BinaryOperation::Add,
                    left: &residual,
                    right: &attention,
                })?;
                self.math.propose_ffn(layer, &after_attention)
            },
        )?;
        let logits = self.math.final_logits(&last)?;
        let vocab = usize::try_from(self.math.geometry().vocab_size)
            .map_err(|_| Error::Limit("original Qwen trained vocabulary width".into()))?;
        if logits.len() != vocab {
            return Err(Error::State(
                "original Qwen model output head mismatch".into(),
            ));
        }
        let expected_domains =
            self.math
                .geometry()
                .layers
                .iter()
                .try_fold(0usize, |count, kind| {
                    count
                        .checked_add(if *kind == LayerKind::LinearAttention {
                            2
                        } else {
                            1
                        })
                        .ok_or_else(|| Error::Limit("original Qwen model state count".into()))
                })?;
        if domains.len() != expected_domains {
            return Err(Error::State(
                "incomplete original Qwen full-model state proposal".into(),
            ));
        }
        Ok(OriginalPreparedToken {
            step: OriginalPreparedLayer {
                math: self.math,
                coordinator_id: coordinator.coordinator_id(),
                epoch: coordinator.epoch,
                base_root: root.id,
                next_frontier,
                domains,
                output: logits,
            },
        })
    }

    pub fn coordinator_for_layer(&self, layer: usize, epoch: u64) -> Result<TaskCoordinator> {
        TaskCoordinator::new_source_bound(
            epoch,
            self.math.source_identity(),
            self.typed_domains(layer)?.into_iter().map(named),
        )
    }
    fn validate_owner(&self, layer: usize, c: &TaskCoordinator) -> Result<()> {
        if c.source_identity() != Some(self.math.source_identity())
            || c.root().frontier >= MAX_DIAGNOSTIC_TOKENS
        {
            return Err(Error::State(
                "foreign Qwen state source or exhausted context".into(),
            ));
        }
        let required = self.typed_domains(layer)?;
        if c.root().domains.len() != required.len() {
            return Err(Error::State("incomplete Qwen state domains".into()));
        }
        for slot in required {
            let (name, schema) = named(slot);
            if c.root()
                .domains
                .get(&name)
                .is_none_or(|entry| entry.schema != schema)
            {
                return Err(Error::State(
                    "Qwen original state slot/schema mismatch".into(),
                ));
            }
        }
        Ok(())
    }
    fn existing(root: &StateRoot, slot: StateSlot) -> Result<&[u8]> {
        let (name, schema) = named(slot);
        let entry = root
            .domains
            .get(&name)
            .ok_or_else(|| Error::State("missing Qwen committed source domain".into()))?;
        if entry.schema != schema {
            return Err(Error::State("different Qwen state schema".into()));
        }
        Ok(entry.bytes.as_ref())
    }
    /// Prepare one state slot using an immutable committed ROOT, not a
    /// second layer-specific coordinator. Shared by the legacy diagnostic
    /// one-layer test and the full model's single traversal.
    fn prepare_layer(
        &self,
        layer: usize,
        root: &StateRoot,
        hidden: &[f32],
        next_frontier: u64,
    ) -> Result<(Vec<f32>, Vec<PreparedDomain>)> {
        let (output, domains) = match self.math.geometry().layers[layer] {
            LayerKind::FullAttention => {
                // Preflight the next complete source KV slab BEFORE
                // running projections or collecting the new history.
                let heads = usize::try_from(self.math.geometry().num_key_value_heads)
                    .map_err(|_| Error::Limit("next Qwen KV heads".into()))?;
                let width = usize::try_from(self.math.geometry().head_dim)
                    .map_err(|_| Error::Limit("next Qwen KV width".into()))?;
                let next_count = usize::try_from(next_frontier)
                    .ok()
                    .and_then(|t| t.checked_mul(heads))
                    .and_then(|t| t.checked_mul(width))
                    .and_then(|t| t.checked_mul(8))
                    .ok_or_else(|| Error::Limit("next Qwen KV bytes overflow".into()))?;
                if next_count > MAX_DOMAIN_BYTES {
                    return Err(Error::Limit(
                        "next Qwen KV prefix exceeds bounded state budget".into(),
                    ));
                }
                let empty = self.math.empty_full_prefix(layer)?;
                let bytes = Self::existing(root, StateSlot::Kv(KvDomain(layer)))?;
                let prefix = if root.frontier == 0 {
                    if !bytes.is_empty() {
                        return Err(Error::State("nonempty initial Qwen KV".into()));
                    }
                    empty
                } else {
                    let count = usize::try_from(root.frontier)
                        .map_err(|_| Error::Limit("Qwen KV frontier overflow".into()))?;
                    let kvheads = usize::try_from(self.math.geometry().num_key_value_heads)
                        .map_err(|_| Error::Limit("Qwen KV head count".into()))?;
                    let width = usize::try_from(self.math.geometry().head_dim)
                        .map_err(|_| Error::Limit("Qwen KV head width".into()))?;
                    let elems = kvheads
                        .checked_mul(width)
                        .ok_or_else(|| Error::Limit("Qwen KV dimensions overflow".into()))?;
                    let total = count
                        .checked_mul(elems)
                        .and_then(|n| n.checked_mul(2))
                        .ok_or_else(|| Error::Limit("Qwen KV element count overflow".into()))?;
                    let raw = decode(bytes, total)?;
                    let mut keys = Vec::with_capacity(count);
                    let mut values = Vec::with_capacity(count);
                    for slab in raw.chunks_exact(elems * 2) {
                        keys.push(Arc::<[f32]>::from(&slab[..elems]));
                        values.push(Arc::<[f32]>::from(&slab[elems..]));
                    }
                    OriginalFullPrefix {
                        source: self.math.source_identity(),
                        layer,
                        keys,
                        values,
                    }
                };
                let proposed = self.math.propose_full_attention(layer, hidden, &prefix)?;
                if proposed.prefix.len() != usize::try_from(next_frontier).unwrap_or(usize::MAX)
                    || proposed.prefix.source_identity() != self.math.source_identity()
                    || proposed.prefix.layer() != layer
                {
                    return Err(Error::State("Qwen private KV frontier changed".into()));
                }
                let mut collected = Vec::new();
                for (key, value) in proposed.prefix.keys.iter().zip(&proposed.prefix.values) {
                    collected.extend_from_slice(key);
                    collected.extend_from_slice(value);
                }
                let (name, schema) = named(StateSlot::Kv(KvDomain(layer)));
                (
                    proposed.output,
                    vec![PreparedDomain {
                        name,
                        schema,
                        frontier: next_frontier,
                        bytes: encode(&collected)?,
                    }],
                )
            }
            LayerKind::LinearAttention => {
                let empty = self.math.empty_gdn_prefix(layer)?;
                let conv = Self::existing(root, StateSlot::Conv(ConvDomain(layer)))?;
                let rec = Self::existing(root, StateSlot::Recurrent(RecurrentDomain(layer)))?;
                let prefix = if root.frontier == 0 {
                    if !conv.is_empty() || !rec.is_empty() {
                        return Err(Error::State("nonempty initial Qwen GDN".into()));
                    }
                    empty
                } else {
                    OriginalGdnPrefix {
                        source: self.math.source_identity(),
                        layer,
                        position: usize::try_from(root.frontier)
                            .map_err(|_| Error::Limit("Qwen GDN frontier".into()))?,
                        convolution: Arc::new(decode(conv, empty.convolution().len())?),
                        recurrent: Arc::new(decode(rec, empty.recurrent().len())?),
                    }
                };
                let proposed = self.math.propose_gdn(layer, hidden, &prefix)?;
                if proposed.next.position() != usize::try_from(next_frontier).unwrap_or(usize::MAX)
                    || proposed.next.source_identity() != self.math.source_identity()
                    || proposed.next.layer() != layer
                {
                    return Err(Error::State("Qwen private GDN frontier changed".into()));
                }
                let (cn, cs) = named(StateSlot::Conv(ConvDomain(layer)));
                let (rn, rs) = named(StateSlot::Recurrent(RecurrentDomain(layer)));
                let domains = vec![
                    PreparedDomain {
                        name: cn,
                        schema: cs,
                        frontier: next_frontier,
                        bytes: encode(proposed.next.convolution())?,
                    },
                    PreparedDomain {
                        name: rn,
                        schema: rs,
                        frontier: next_frontier,
                        bytes: encode(proposed.next.recurrent())?,
                    },
                ];
                (proposed.output, domains)
            }
            LayerKind::SlidingAttention => {
                return Err(Error::Unsupported(
                    "Qwen sliding attention unqualified".into(),
                ));
            }
        };
        if output.len() != self.math.hidden_width()?
            || output.is_empty()
            || output.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Malformed(
                "Qwen layer output width or finite mismatch".into(),
            ));
        }
        Ok((output, domains))
    }

    pub fn prepare(
        &self,
        layer: usize,
        coordinator: &TaskCoordinator,
        hidden: &[f32],
    ) -> Result<OriginalPreparedLayer<'m, 'b, 's>> {
        self.validate_owner(layer, coordinator)?;
        let root = coordinator.root();
        let next_frontier = root
            .frontier
            .checked_add(1)
            .ok_or_else(|| Error::Limit("Qwen original frontier overflow".into()))?;
        let (output, domains) = self.prepare_layer(layer, root, hidden, next_frontier)?;
        Ok(OriginalPreparedLayer {
            math: self.math,
            coordinator_id: coordinator.coordinator_id(),
            epoch: coordinator.epoch,
            base_root: root.id,
            next_frontier,
            domains,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn domain_encoding_round_trip_preserves_bit_patterns_and_rejects_nonfinite() {
        for v in [vec![1.0, -0.0, 2.0], vec![3.0, 4.0]] {
            let decoded = decode(&encode(&v).unwrap(), v.len()).unwrap();
            assert_eq!(
                decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        assert!(decode(&[0, 0, 0], 1).is_err());
        assert!(decode(&f32::NAN.to_le_bytes(), 1).is_err());
        assert!(encode(&[f32::INFINITY]).is_err());
        assert!(decode(&[], usize::MAX).is_err());
    }
}
