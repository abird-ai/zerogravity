//! Borrowed actual completed cohort. Expected bytes can only enter its caller.
use super::*;
use std::collections::BTreeMap;
use zerogravity_core::accelerator::program::{StateSlot, ValueId};

pub struct ShortHfReadback<'a> {
    pub(super) record: &'a Arc<RecordedVllm08Model>,
    pub(super) raw: &'a [Bytes],
    pub(super) values: &'a BTreeMap<u32, usize>,
    pub(super) sums: &'a BTreeMap<usize, usize>,
    pub(super) prior: &'a [(StateSlot, std::ops::Range<usize>)],
    pub(super) candidate: &'a [(StateSlot, std::ops::Range<usize>)],
    pub(super) positions: usize,
    pub(super) kv_status: &'a BTreeMap<usize, usize>,
    pub(super) packet: usize,
    pub(super) selected: (u32, u32),
}
impl ShortHfReadback<'_> {
    pub fn record(&self) -> &RecordedVllm08Model {
        self.record
    }
    pub fn value(&self, id: ValueId) -> Result<&[u8]> {
        let i = *self.values.get(&id.0).ok_or_else(|| {
            Error::Unsupported(
                "short-HF overwritten or absent value is not a retained observation".into(),
            )
        })?;
        self.raw
            .get(i)
            .map(|b| b.as_ref())
            .ok_or_else(|| Error::State("short-HF actual completed value absent".into()))
    }
    pub fn rms_sum(&self, node: usize) -> Result<&[u8]> {
        let i = *self
            .sums
            .get(&node)
            .ok_or_else(|| Error::State("short-HF actual dual RMS sum absent".into()))?;
        self.raw
            .get(i)
            .map(|b| b.as_ref())
            .ok_or_else(|| Error::State("short-HF actual sum cohort absent".into()))
    }
    pub fn state(&self, slot: StateSlot, candidate: bool) -> Result<(&[u8], Option<&[u8]>)> {
        let entries = if candidate {
            self.candidate
        } else {
            self.prior
        };
        let (_, r) = entries
            .iter()
            .find(|(s, _)| *s == slot)
            .ok_or_else(|| Error::State("short-HF canonical state absent".into()))?;
        if !(1..=2).contains(&r.len()) || r.end > self.raw.len() {
            return Err(Error::State("short-HF state descriptor range".into()));
        }
        Ok((
            self.raw[r.start].as_ref(),
            if r.len() == 2 {
                Some(self.raw[r.start + 1].as_ref())
            } else {
                None
            },
        ))
    }
    pub fn positions(&self) -> Result<&[u8]> {
        self.raw
            .get(self.positions)
            .map(|b| b.as_ref())
            .ok_or_else(|| Error::State("short-HF actual positions absent".into()))
    }
    pub fn kv_transport_status(&self, layer: usize) -> Result<&[u8]> {
        let i = *self
            .kv_status
            .get(&layer)
            .ok_or_else(|| Error::State("short-HF actual KV layer absent".into()))?;
        self.raw
            .get(i)
            .map(|b| b.as_ref())
            .ok_or_else(|| Error::State("short-HF actual KV status absent".into()))
    }
    pub fn greedy_packet(&self) -> Result<&[u8]> {
        self.raw
            .get(self.packet)
            .map(|b| b.as_ref())
            .ok_or_else(|| Error::State("short-HF actual greedy packet absent".into()))
    }
    pub fn greedy(&self) -> (u32, u32) {
        self.selected
    }
}
