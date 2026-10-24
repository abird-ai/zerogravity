use super::*;
#[derive(Clone, Debug)]
pub(super) struct Layout {
    pub(super) values: usize,
    pub(super) duals: usize,
    pub(super) body: usize,
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
            .ok_or_else(|| Error::Limit("whole readback header overflow".into()))?;
        let duals = start
            .checked_add(values)
            .ok_or_else(|| Error::Limit("whole readback values overflow".into()))?;
        if gdn != 18
            || values != 327
            || body != duals + dual_nodes.len() * 4
            || dual_nodes.len() != 48
            || dual_nodes.windows(2).any(|w| w[0] >= w[1])
            || dual_nodes.last() != Some(&253)
        {
            return Err(Error::State(
                "canonical47 whole actual ordered readback layout".into(),
            ));
        }
        Ok(Self {
            values: start,
            duals,
            body,
            positions: body,
            kv: [3, 7, 11, 15, 19, 23]
                .into_iter()
                .enumerate()
                .map(|(i, layer)| (layer, body + 1 + i))
                .collect(),
        })
    }
}
pub struct Canonical47FullModelReadback<'a> {
    pub(super) record: &'a Arc<RecordedVllm08Model>,
    pub(super) raw: &'a [Bytes],
    pub(super) prior: &'a [(StateSlot, Range<usize>)],
    pub(super) candidate: &'a [(StateSlot, Range<usize>)],
    pub(super) layout: &'a Layout,
    pub(super) dual_nodes: &'a [usize],
}
impl Canonical47FullModelReadback<'_> {
    pub fn record(&self) -> &Arc<RecordedVllm08Model> {
        self.record
    }
    pub fn value(&self, id: ValueId) -> Result<&[u8]> {
        if id.0 >= 327 {
            return Err(Error::Unsupported(
                "whole model value outside original327".into(),
            ));
        }
        self.raw
            .get(self.layout.values + id.0 as usize)
            .map(|b| b.as_ref())
            .ok_or_else(|| Error::State("whole actual value descriptor absent".into()))
    }
    pub fn rms_sum(&self, node: usize) -> Result<&[u8]> {
        let ordinal = self
            .dual_nodes
            .iter()
            .position(|id| *id == node)
            .ok_or_else(|| Error::Unsupported("whole actual RMS node absent".into()))?;
        let value = self
            .raw
            .get(self.layout.duals + ordinal * 4 + 1)
            .ok_or_else(|| Error::State("whole actual RMS sum descriptor absent".into()))?;
        if value.len() != 47 * 1024 * 4 {
            return Err(Error::State("whole exact47x1024 F32 sum".into()));
        }
        Ok(value.as_ref())
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
            .ok_or_else(|| Error::State("whole actual canonical domain absent".into()))?;
        if !matches!(span.len(), 1 | 2) {
            return Err(Error::State("whole canonical descriptor arity".into()));
        }
        Ok((
            self.raw[span.start].as_ref(),
            if span.len() == 2 {
                Some(self.raw[span.start + 1].as_ref())
            } else {
                None
            },
        ))
    }
    pub fn positions(&self) -> Result<&[u8]> {
        let bytes = self
            .raw
            .get(self.layout.positions)
            .ok_or_else(|| Error::State("whole source positions absent".into()))?;
        if bytes.len() != 1560 {
            return Err(Error::State("whole exact pitch65 positions".into()));
        }
        Ok(bytes.as_ref())
    }
    pub fn kv_transport_status(&self, layer: usize) -> Result<&[u8]> {
        let (_, index) = self
            .layout
            .kv
            .iter()
            .find(|(l, _)| *l == layer)
            .ok_or_else(|| Error::Unsupported("whole actual attention layer absent".into()))?;
        let bytes = self
            .raw
            .get(*index)
            .ok_or_else(|| Error::State("whole source KV status absent".into()))?;
        if bytes.len() != 4 {
            return Err(Error::State("whole exact KV status".into()));
        }
        Ok(bytes.as_ref())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_layout_cannot_inherit_three_layer_header_or_future_order() {
        let mut nodes = (0..47).map(|i| 2 + i * 5).collect::<Vec<_>>();
        nodes.push(253);
        let layout = Layout::new(541, 18, 327, &nodes).unwrap();
        assert_eq!(
            (layout.values, layout.duals, layout.positions),
            (22, 349, 541)
        );
        assert_eq!(layout.duals + 47 * 4 + 1, 538);
        for (body, gdn, values) in [
            (541, 3, 327),
            (526, 18, 327),
            (541, 18, 326),
            (541, 18, 328),
        ] {
            assert!(Layout::new(body, gdn, values, &nodes).is_err());
        }
        nodes.swap(0, 1);
        assert!(Layout::new(541, 18, 327, &nodes).is_err());
        nodes.swap(0, 1);
        nodes[47] = 252;
        assert!(Layout::new(541, 18, 327, &nodes).is_err());
    }
}
