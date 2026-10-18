//! One original Qwen analytical layer, admitted by the existing TaskCoordinator.
//! No independent token loop, trained reference or second StateRoot issuer.
//! State slots use the SAME ForwardPlan StateSlot vocabulary; session bytes
//! are provisional FP32 diagnostics, not a persistent checkpoint format.

use crate::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
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
