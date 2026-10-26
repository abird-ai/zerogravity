//! Requested source span; independent of qualification case coverage.
use super::{Error, Result, RotaryOutput};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SourceKind {
    FixedPrompt,
    Text,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    pub(super) kind: SourceKind,
    positions: usize,
}
impl Layout {
    pub(super) fn new(kind: SourceKind, positions: usize) -> Result<Self> {
        let limit = RotaryOutput::Cache.bytes() / 128;
        if positions == 0 || positions > limit {
            return Err(Error::Unsupported(
                "rotary span exceeds source table".into(),
            ));
        }
        let value = Self { kind, positions };
        value.readback_bytes()?;
        value.prefix_reservation_bytes()?;
        Ok(value)
    }
    pub(super) fn positions(self) -> usize {
        self.positions
    }
    pub(super) fn bytes(self) -> usize {
        // Constructor bounds positions by the original complete table extent.
        self.positions * 128
    }
    pub(super) fn prefix_bytes(self, role: RotaryOutput) -> usize {
        match role {
            RotaryOutput::Range1048576 => self.positions * 4,
            RotaryOutput::Frequencies | RotaryOutput::Cos | RotaryOutput::Sin => {
                self.positions * 32 * 4
            }
            RotaryOutput::Cat => self.positions * 64 * 4,
            RotaryOutput::Cache => self.bytes(),
            _ => role.bytes(),
        }
    }
    pub(super) fn readback_bytes(self) -> Result<usize> {
        RotaryOutput::ALL
            .into_iter()
            .try_fold(0usize, |sum, role| sum.checked_add(self.prefix_bytes(role)))
            .map(|bytes| bytes.max(1024 * 1024))
            .ok_or_else(|| Error::Limit("rotary readback extent".into()))
    }
    pub(super) fn prefix_reservation_bytes(self) -> Result<usize> {
        self.bytes()
            .checked_mul(3)
            .ok_or_else(|| Error::Limit("rotary prefix retention extent".into()))
    }
    pub(super) fn retained_host_bytes(self) -> Result<usize> {
        (64usize * 1024)
            .checked_add(self.readback_bytes()?)
            .and_then(|bytes| bytes.checked_add(self.prefix_reservation_bytes().ok()?))
            .ok_or_else(|| Error::Limit("rotary source preparation extent".into()))
    }
    pub(super) fn require_kind(self, kind: SourceKind) -> Result<()> {
        if self.kind != kind {
            return Err(Error::State("rotary source route kind differs".into()));
        }
        Ok(())
    }
    pub(super) fn require_span(self, base: usize, rows: usize) -> Result<()> {
        if rows == 0
            || base
                .checked_add(rows)
                .is_none_or(|end| end > self.positions)
        {
            return Err(Error::Unsupported("rotary source prefix interval".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
