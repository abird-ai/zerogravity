//! Physical bytes only; no numerical profile, allocation, state or Ready permit.
use std::ops::Range;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

/// Explicit physical element storage, independent of logical Plan precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentPhysicalType {
    F32,
    Bf16,
}
impl ResidentPhysicalType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// Dense physical extent; constructing it neither converts nor reinterprets data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentBufferLayout {
    kind: ResidentPhysicalType,
    elements: usize,
}
impl ResidentBufferLayout {
    /// Pure source-plan metadata; no lease binding or reuse authority.
    /// Existing native handles require exact aligned allocation classes. Lowerer
    /// access envelopes, source producers, epochs and actual completion remain
    /// separate gates; ordered diagnostic taps must precede slot overwrite.
    pub fn plan_values<'a>(
        plan: &'a zerogravity_core::accelerator::program::ForwardPlan,
        layouts: &[Self],
        retention: zerogravity_core::accelerator::program::ValueStorageRetention<'_>,
        limits: zerogravity_core::accelerator::program::ValueStorageLimits,
    ) -> Result<zerogravity_core::accelerator::program::ValueStoragePlan<'a>> {
        use zerogravity_core::accelerator::program::{
            ValueStorageFormat, ValueStorageLayout, ValueStorageReusePolicy,
        };
        if limits.reuse_policy != ValueStorageReusePolicy::ExactAlignedSize {
            return Err(Error::Unsupported(
                "resident storage requires exact aligned slots".into(),
            ));
        }
        // Refuse bounded metadata geometry BEFORE deriving another full Vec.
        if layouts.is_empty()
            || plan.nodes().is_empty()
            || layouts.len() != plan.values().len()
            || layouts.len() > limits.max_values
            || plan.nodes().len() > limits.max_nodes
        {
            return Err(Error::Limit(
                "resident storage plan/layout metadata limits".into(),
            ));
        }
        let physical = layouts
            .iter()
            .map(|layout| {
                layout.bytes()?;
                Ok(ValueStorageLayout {
                    format: match layout.kind {
                        ResidentPhysicalType::F32 => ValueStorageFormat::F32,
                        ResidentPhysicalType::Bf16 => ValueStorageFormat::Bf16,
                    },
                    elements: layout.elements,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        plan.plan_value_storage(&physical, retention, limits)
    }

    pub fn new(kind: ResidentPhysicalType, elements: usize) -> Result<Self> {
        if elements == 0 {
            return Err(Error::Malformed("empty typed resident buffer".into()));
        }
        let value = Self { kind, elements };
        value.bytes()?;
        Ok(value)
    }
    // Existing infallible four-byte descriptor; old check remains its authority.
    pub(super) const fn legacy_f32(elements: usize) -> Self {
        Self {
            kind: ResidentPhysicalType::F32,
            elements,
        }
    }
    pub fn kind(self) -> ResidentPhysicalType {
        self.kind
    }
    pub fn elements(self) -> usize {
        self.elements
    }
    pub fn bytes(self) -> Result<usize> {
        self.elements
            .checked_mul(self.kind.byte_width())
            .filter(|_| u32::try_from(self.elements).is_ok())
            .ok_or_else(|| Error::Limit("resident physical index/byte overflow".into()))
    }
    /// Requires the consumer's declared physical kind and element count.
    pub fn require(self, expected: Self) -> Result<()> {
        if self != expected {
            return Err(Error::State(
                "resident physical storage/layout differs".into(),
            ));
        }
        self.bytes()?;
        Ok(())
    }
    /// Exact typed view. An aligned larger allocation is not this same extent.
    pub fn validate_region(self, offset: u64, bytes: u64) -> Result<()> {
        let expected = self.bytes()? as u64;
        if !offset.is_multiple_of(self.kind.byte_width() as u64)
            || bytes != expected
            || offset.checked_add(bytes).is_none()
        {
            return Err(Error::State(
                "resident physical byte extent/alignment differs".into(),
            ));
        }
        Ok(())
    }
    pub fn element_range(self, elements: Range<usize>) -> Result<Range<usize>> {
        if elements.start >= elements.end || elements.end > self.elements {
            return Err(Error::State(
                "resident physical element range differs".into(),
            ));
        }
        let width = self.kind.byte_width();
        let start = elements
            .start
            .checked_mul(width)
            .ok_or_else(|| Error::Limit("resident byte offset overflow".into()))?;
        let end = elements
            .end
            .checked_mul(width)
            .ok_or_else(|| Error::Limit("resident byte end overflow".into()))?;
        Ok(start..end)
    }
    /// Dense activations/scratch only; canonical root state remains F32.
    /// This says nothing about initialization or actual read/write completion.
    pub fn validate_role(self, role: BufferRole, access: DeviceAccess) -> Result<()> {
        self.bytes()?;
        match (role, access) {
            (
                BufferRole::Activation | BufferRole::Output | BufferRole::Scratch,
                DeviceAccess::Read | DeviceAccess::Write,
            ) => Ok(()),
            (BufferRole::RecurrentState, DeviceAccess::Read | DeviceAccess::Write)
                if self.kind == ResidentPhysicalType::F32 =>
            {
                Ok(())
            }
            _ => Err(Error::Unsupported(
                "resident physical role/storage contract".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod value_storage_tests;
