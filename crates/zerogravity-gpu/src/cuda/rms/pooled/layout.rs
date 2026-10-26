//! Shared storage roles only; source gamma, sums and public values are absent.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ScratchRole {
    GammaF32,
    GammaShifted,
    InputF32,
    Squares,
    Mean,
    Epsilon,
    Inverse,
    Normalized,
    Weighted,
}
impl ScratchRole {
    pub(super) const ALL: [Self; 9] = [
        Self::GammaF32,
        Self::GammaShifted,
        Self::InputF32,
        Self::Squares,
        Self::Mean,
        Self::Epsilon,
        Self::Inverse,
        Self::Normalized,
        Self::Weighted,
    ];
    pub(super) fn bytes(self, rows: usize, columns: usize) -> Result<usize> {
        let elements = match self {
            Self::GammaF32 | Self::GammaShifted => columns,
            Self::Mean | Self::Epsilon | Self::Inverse => rows,
            _ => rows
                .checked_mul(columns)
                .ok_or_else(|| Error::Limit("RMS dense scratch".into()))?,
        };
        elements
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("RMS scratch bytes".into()))
    }
}

#[derive(Default)]
pub(super) struct OrderedActors {
    next: usize,
    busy: bool,
    failed: bool,
}
impl OrderedActors {
    pub(super) fn begin(&mut self, nodes: &[usize], node: usize) -> Result<()> {
        if self.failed || self.busy || nodes.get(self.next) != Some(&node) {
            return Err(Error::State(
                "pooled RMS actual native cursor/order differs".into(),
            ));
        }
        self.busy = true;
        Ok(())
    }
    pub(super) fn require_current(&self, nodes: &[usize], node: usize) -> Result<()> {
        if self.failed || !self.busy || nodes.get(self.next) != Some(&node) {
            return Err(Error::State(
                "pooled RMS current actor lost exclusive borrow".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn completed(&mut self) {
        self.busy = false;
        self.next += 1;
    }
    pub(super) fn fail(&mut self) {
        self.failed = true;
        self.busy = false;
    }
    pub(super) fn require_finished(&self, count: usize) -> Result<()> {
        if self.failed || self.busy || self.next != count {
            return Err(Error::State(
                "pooled RMS original traversal incomplete".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn require_distinct(ids: impl IntoIterator<Item = BufferLeaseId>) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(Error::State(
                "pooled RMS actor input/output/private alias".into(),
            ));
        }
    }
    Ok(())
}
