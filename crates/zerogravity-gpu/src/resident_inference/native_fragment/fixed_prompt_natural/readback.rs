//! Borrowed only inside actual trusted completion; old Initial47 readback stays fixed.
use super::*;
use crate::resident_program::source_bindings::SourceBindings;
use std::ops::Range;

#[derive(Clone)]
pub(super) struct Layout {
    values: usize,
    duals: usize,
    packet: usize,
    pub(super) positions: usize,
    pub(super) kv: Vec<(usize, usize)>,
}

impl Layout {
    pub(super) fn new(
        body: usize,
        gdn: usize,
        values: usize,
        dual_nodes: &[usize],
    ) -> Result<Self> {
        let start = 1usize
            .checked_add(gdn)
            .and_then(|n| n.checked_add(3))
            .ok_or_else(|| Error::Limit("canonical natural readback header".into()))?;
        let duals = start
            .checked_add(values)
            .ok_or_else(|| Error::Limit("canonical natural readback values".into()))?;
        let packet = duals
            .checked_add(dual_nodes.len() * 4)
            .ok_or_else(|| Error::Limit("canonical natural dual readbacks".into()))?;
        if gdn != 18
            || values != 327
            || body != packet + 2
            || dual_nodes.len() != 48
            || dual_nodes.windows(2).any(|nodes| nodes[0] >= nodes[1])
            || dual_nodes.last() != Some(&253)
        {
            return Err(Error::State(
                "canonical natural exact computed body and packet readbacks".into(),
            ));
        }
        Ok(Self {
            values: start,
            duals,
            packet,
            positions: body,
            kv: [3, 7, 11, 15, 19, 23]
                .into_iter()
                .enumerate()
                .map(|(i, layer)| (layer, body + 1 + i))
                .collect(),
        })
    }
}

pub struct FixedPromptNaturalReadback<'a> {
    pub(super) record: &'a Arc<SourceBindings>,
    pub(super) raw: &'a [Bytes],
    pub(super) prior: &'a [(StateSlot, Range<usize>)],
    pub(super) candidate: &'a [(StateSlot, Range<usize>)],
    pub(super) layout: &'a Layout,
    pub(super) dual_nodes: &'a [usize],
    pub(super) selected: (u32, u32),
}

impl FixedPromptNaturalReadback<'_> {
    pub fn record(&self) -> &Arc<SourceBindings> {
        self.record
    }
    pub fn value(&self, id: ValueId) -> Result<&[u8]> {
        if id.0 >= 327 {
            return Err(Error::Unsupported(
                "canonical natural value outside original327".into(),
            ));
        }
        self.raw
            .get(self.layout.values + id.0 as usize)
            .map(|bytes| bytes.as_ref())
            .ok_or_else(|| Error::State("canonical natural computed value absent".into()))
    }
    pub fn rms_sum(&self, node: usize) -> Result<&[u8]> {
        let ordinal = self
            .dual_nodes
            .iter()
            .position(|index| *index == node)
            .ok_or_else(|| Error::Unsupported("canonical natural RMS source node absent".into()))?;
        let bytes = self
            .raw
            .get(self.layout.duals + ordinal * 4 + 1)
            .ok_or_else(|| Error::State("canonical natural unrounded sum absent".into()))?;
        let rows = self.record.call().rows().len();
        if !matches!(rows, 1 | 47) || bytes.len() != rows * 1024 * 4 {
            return Err(Error::State(
                "canonical natural source rows and F32 sum extent".into(),
            ));
        }
        Ok(bytes.as_ref())
    }
    pub fn state(&self, slot: StateSlot, candidate: bool) -> Result<(&[u8], Option<&[u8]>)> {
        let keys = if candidate {
            self.candidate
        } else {
            self.prior
        };
        let (_, span) = keys
            .iter()
            .find(|(key, _)| *key == slot)
            .ok_or_else(|| Error::State("canonical natural current domain absent".into()))?;
        if !matches!(span.len(), 1 | 2) {
            return Err(Error::State(
                "canonical natural current domain descriptor arity".into(),
            ));
        }
        let first = self
            .raw
            .get(span.start)
            .ok_or_else(|| Error::State("canonical natural state first descriptor".into()))?;
        let second = if span.len() == 2 {
            Some(
                self.raw
                    .get(span.start + 1)
                    .ok_or_else(|| {
                        Error::State("canonical natural state second descriptor".into())
                    })?
                    .as_ref(),
            )
        } else {
            None
        };
        Ok((first.as_ref(), second))
    }
    pub fn positions(&self) -> Result<&[u8]> {
        self.fixed(
            self.layout.positions,
            1560,
            "canonical natural pitch65 source positions",
        )
    }
    pub fn kv_transport_status(&self, layer: usize) -> Result<&[u8]> {
        let (_, index) = self
            .layout
            .kv
            .iter()
            .find(|(index, _)| *index == layer)
            .ok_or_else(|| Error::Unsupported("canonical natural actual attention layer".into()))?;
        self.fixed(*index, 4, "canonical natural KV status")
    }
    pub fn greedy_packet(&self) -> Result<&[u8]> {
        self.fixed(
            self.layout.packet,
            12,
            "canonical natural actual BF16 packet",
        )
    }
    pub fn greedy(&self) -> (u32, u32) {
        self.selected
    }
    fn fixed(&self, index: usize, bytes: usize, message: &str) -> Result<&[u8]> {
        self.raw
            .get(index)
            .filter(|raw| raw.len() == bytes)
            .map(|raw| raw.as_ref())
            .ok_or_else(|| Error::State(message.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn nodes() -> Vec<usize> {
        let mut nodes = (0..47).map(|i| 2 + i * 5).collect::<Vec<_>>();
        nodes.push(253);
        nodes
    }
    #[test]
    fn exact_layout_includes_real_packet_before_positions_and_transports() {
        let layout = Layout::new(543, 18, 327, &nodes()).unwrap();
        assert_eq!(
            (layout.values, layout.duals, layout.packet, layout.positions),
            (22, 349, 541, 543)
        );
        assert_eq!(
            layout.kv,
            vec![
                (3, 544),
                (7, 545),
                (11, 546),
                (15, 547),
                (19, 548),
                (23, 549)
            ]
        );
        // Legacy549 and standalone541 are different actual body cohorts.
        assert!(Layout::new(549, 18, 327, &nodes()).is_err());
        assert!(Layout::new(541, 18, 327, &nodes()).is_err());
    }
    #[test]
    fn malformed_dual_order_and_partial_map_never_alias_later_packet() {
        for (gdn, values) in [(3, 327), (18, 326), (18, 328)] {
            assert!(Layout::new(543, gdn, values, &nodes()).is_err());
        }
        let mut changed = nodes();
        changed.swap(0, 1);
        assert!(Layout::new(543, 18, 327, &changed).is_err());
        changed.swap(0, 1);
        changed[47] = 252;
        assert!(Layout::new(543, 18, 327, &changed).is_err());
    }
}
