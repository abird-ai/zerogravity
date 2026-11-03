use super::*;
use std::collections::BTreeSet;
use zerogravity::accelerator::program::{PlanOperation, SourceOperatorKind, StateSlot};
use zerogravity::gpu::{SourceCallFrame, SourceCallFrameKind, SourceCallObserver};
use zerogravity::qwen35::plan::OrdinaryQwenPlan;

pub struct ObservationMap {
    values: BTreeMap<u32, (String, Option<usize>)>,
    capacity: usize,
}
impl ObservationMap {
    pub fn new(plan: &OrdinaryQwenPlan) -> Result<Self> {
        let mut values = BTreeMap::new();
        for n in plan.plan().nodes() {
            if let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &n.operation
                && matches!(descriptor.kind(), SourceOperatorKind::EmbeddingRows)
            {
                need(n.outputs.len() == 1, "original embedding producer")?;
                values.insert(n.outputs[0].0, ("embedding".into(), None));
            }
        }
        for (layer, boundary) in plan.boundaries().iter().enumerate() {
            let nodes = plan
                .plan()
                .nodes()
                .iter()
                .filter(|n| n.inputs.as_slice() == boundary)
                .collect::<Vec<_>>();
            need(
                nodes.len() == 1 && nodes[0].outputs.len() == 2,
                "original residual-normalization layer producer",
            )?;
            need(
                matches!(&nodes[0].operation,PlanOperation::SourceOperatorTemplate{descriptor,..} if matches!(descriptor.kind(),SourceOperatorKind::ResidualRms{..})),
                "actual residual-add producer, not a reconstructed layer",
            )?;
            need(
                values
                    .insert(nodes[0].outputs[1].0, ("layer_output".into(), Some(layer)))
                    .is_none(),
                "unique original layer producer",
            )?;
        }
        need(values.len() == 25, "complete original embedding/layers")?;
        need(
            values
                .insert(plan.normalized().0, ("normalized".into(), None))
                .is_none()
                && values
                    .insert(plan.logits().0, ("logits".into(), None))
                    .is_none(),
            "original normalized/logit producers",
        )?;
        Ok(Self {
            values,
            capacity: plan.context_capacity_tokens(),
        })
    }
}
pub struct Observer<'a> {
    reference: &'a Expected,
    map: &'a ObservationMap,
    call: Call,
    expected: BTreeSet<usize>,
    done: BTreeSet<usize>,
    active: Option<usize>,
    raw: Vec<u8>,
    next: usize,
    zero_required: BTreeMap<(StateSlot, usize), usize>,
    zero_seen: BTreeMap<(StateSlot, usize), usize>,
    pub words: usize,
    pub initial_zero_words: usize,
    pub kv_tail_words: usize,
    pub position_words: usize,
    pub mismatches: usize,
    pub first: Option<String>,
}
impl<'a> Observer<'a> {
    pub fn new(reference: &'a Expected, map: &'a ObservationMap, call: &Call) -> Result<Self> {
        let expected = reference
            .doc
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| (r.repeat, r.case, r.call) == call.key())
            .map(|(i, _)| i)
            .collect::<BTreeSet<_>>();
        need(
            expected.len() == if call.position == 0 { 76 } else { 124 },
            "complete original call record set",
        )?;
        let mut out = Self {
            reference,
            map,
            call: call.clone(),
            expected,
            done: BTreeSet::new(),
            active: None,
            raw: Vec::new(),
            next: 0,
            zero_required: BTreeMap::new(),
            zero_seen: BTreeMap::new(),
            words: 0,
            initial_zero_words: 0,
            kv_tail_words: 0,
            position_words: 0,
            mismatches: 0,
            first: None,
        };
        let i = reference.find(call, "positions", None)?;
        let r = &reference.doc.records[i];
        let positions = reference.load(i)?;
        need(
            r.dtype == "I64"
                && r.shape == [3, 1, call.rows]
                && positions.chunks_exact(8).enumerate().all(|(i, b)| {
                    i64::from_le_bytes(b.try_into().unwrap())
                        == (call.position + i % call.rows) as i64
                }),
            "original three-axis position descriptors (not GPU readback)",
        )?;
        out.position_words = positions.len() / 8;
        out.done.insert(i);
        if call.position == 0 {
            use zerogravity::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain};
            for &i in &out.expected {
                let r = &reference.doc.records[i];
                let Some(layer) = r.layer else { continue };
                let (slot, plane, words) = match r.label.as_str() {
                    "after.conv4" => (StateSlot::Conv(ConvDomain(layer)), 0, r.bytes / 2),
                    "after.recurrent" => {
                        (StateSlot::Recurrent(RecurrentDomain(layer)), 0, r.bytes / 4)
                    }
                    "after.key" | "after.value" => {
                        need(r.shape.len() == 4, "original KV shape")?;
                        (
                            StateSlot::Kv(KvDomain(layer)),
                            usize::from(r.label == "after.value"),
                            map.capacity * r.shape[1] * r.shape[3],
                        )
                    }
                    _ => continue,
                };
                need(
                    out.zero_required.insert((slot, plane), words).is_none(),
                    "unique initial root plane",
                )?;
            }
            need(
                out.zero_required.len() == 48,
                "complete initial root planes",
            )?;
        }
        Ok(out)
    }
    pub fn finish(&self) -> Result<()> {
        need(
            self.active.is_none()
                && self.done == self.expected
                && self.zero_seen == self.zero_required,
            "all original call records and complete root zero controls observed",
        )?;
        self.reference.unchanged()?;
        need(
            self.mismatches == 0,
            &format!(
                "source-call exact mismatch count={} first={:?}",
                self.mismatches, self.first
            ),
        )
    }
    fn compare(
        &mut self,
        index: usize,
        actual: u32,
        expected: u32,
        label: &str,
        layer: Option<usize>,
    ) {
        if actual != expected {
            self.mismatches += 1;
            if self.first.is_none() {
                self.first = Some(format!(
                    "{label}/{layer:?} word={index} actual={actual:08x} expected={expected:08x}"
                ));
            }
        }
    }
}
impl SourceCallObserver for Observer<'_> {
    fn wants(&self, kind: SourceCallFrameKind) -> bool {
        match kind {
            SourceCallFrameKind::Value(id) => self.map.values.contains_key(&id.0),
            SourceCallFrameKind::State { .. } => true,
        }
    }
    fn completed(&mut self, frame: SourceCallFrame, actual: &[u8]) -> Result<()> {
        need(
            frame.base == self.call.position
                && frame.rows == self.call.rows
                && !actual.is_empty()
                && actual.len().is_multiple_of(4),
            "actual completed call frame identity/extent",
        )?;
        let (label, layer, kv) =
            match frame.kind {
                SourceCallFrameKind::Value(id) => {
                    let (label, layer) =
                        self.map.values.get(&id.0).ok_or_else(|| {
                            Error::State("unexpected value observer frame".into())
                        })?;
                    (label.clone(), *layer, false)
                }
                SourceCallFrameKind::State { slot, plane, prior } => {
                    if prior && self.call.position == 0 {
                        let key = (slot, plane);
                        let total =
                            self.zero_required.get(&key).copied().ok_or_else(|| {
                                Error::State("unexpected initial root plane".into())
                            })?;
                        let next = self.zero_seen.entry(key).or_default();
                        need(
                            frame.total_words == total
                                && frame.offset_words == *next
                                && *next + actual.len() / 4 <= total
                                && actual.iter().all(|b| *b == 0),
                            "canonical complete initial root zero control",
                        )?;
                        *next += actual.len() / 4;
                        self.initial_zero_words += actual.len() / 4;
                        return Ok(());
                    }
                    let (name, layer, kv) = match slot {
                        StateSlot::Conv(d) => {
                            need(plane == 0, "Conv4 plane")?;
                            ("conv4", d.0, false)
                        }
                        StateSlot::Recurrent(d) => {
                            need(plane == 0, "recurrent plane")?;
                            ("recurrent", d.0, false)
                        }
                        StateSlot::Kv(d) => {
                            need(plane < 2, "KV plane")?;
                            (if plane == 0 { "key" } else { "value" }, d.0, true)
                        }
                    };
                    (
                        format!("{}.{name}", if prior { "before" } else { "after" }),
                        Some(layer),
                        kv,
                    )
                }
            };
        let i = self.reference.find(&self.call, &label, layer)?;
        if self.active != Some(i) {
            need(
                self.active.is_none() && !self.done.contains(&i) && frame.offset_words == 0,
                "original producer frame sequence",
            )?;
            self.raw = self.reference.load(i)?;
            self.active = Some(i);
            self.next = 0;
        }
        let r = &self.reference.doc.records[i];
        let width = match r.dtype.as_str() {
            "BF16" => 2,
            "F32" => 4,
            _ => return Err(Error::State("floating producer dtype".into())),
        };
        let (heads, tokens, dimension) = if kv {
            need(
                r.dtype == "BF16" && r.shape.len() == 4 && r.shape[0] == 1,
                "original head-major KV record",
            )?;
            (r.shape[1], r.shape[2], r.shape[3])
        } else {
            (0, 0, 0)
        };
        let total = if kv {
            self.map.capacity * heads * dimension
        } else {
            r.bytes / width
        };
        need(
            frame.total_words == total
                && frame.offset_words == self.next
                && self.next + actual.len() / 4 <= total,
            "complete contiguous original producer extent",
        )?;
        for (local, b) in actual.chunks_exact(4).enumerate() {
            let index = self.next + local;
            let actual = u32::from_le_bytes(b.try_into().unwrap());
            let offset = if kv {
                let token = index / (heads * dimension);
                if token >= tokens {
                    self.compare(index, actual, 0, &label, layer);
                    self.kv_tail_words += 1;
                    continue;
                }
                let head = index / dimension % heads;
                let lane = index % dimension;
                (head * tokens * dimension + token * dimension + lane) * width
            } else {
                index * width
            };
            let expected = if width == 2 {
                (u16::from_le_bytes(self.raw[offset..offset + 2].try_into().unwrap()) as u32) << 16
            } else {
                u32::from_le_bytes(self.raw[offset..offset + 4].try_into().unwrap())
            };
            self.compare(index, actual, expected, &label, layer);
            self.words += 1;
        }
        self.next += actual.len() / 4;
        if self.next == total {
            self.done.insert(i);
            self.active = None;
            self.raw.clear();
            self.next = 0;
        }
        Ok(())
    }
}
