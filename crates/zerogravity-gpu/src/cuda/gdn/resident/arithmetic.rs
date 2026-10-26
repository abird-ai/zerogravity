//! Bound instance metadata; no fresh call issuer or qualification authority.
use super::*;
use crate::cuda::gdn::arithmetic::{GdnFamilyIdentity, StageArithmetic};

#[derive(Clone, PartialEq, Eq)]
pub(super) struct CopyArithmetic {
    pub(super) kind: CopyKind,
    pub(super) geometry: CopyGeometry,
    pub(super) source: Arc<str>,
    pub(super) entry: &'static str,
    pub(super) grid: [u32; 3],
    pub(super) bindings: Arc<[(usize, BufferIOAttr)]>,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct GdnArithmetic {
    family: GdnFamilyIdentity,
    phase: GdnPhase,
    slot: GdnCacheSlot,
    typed_outputs: bool,
    stages: Arc<[StageArithmetic]>,
    // Exact bytes passed to the original initialization uploads, including padding.
    controls: Arc<[(GdnBuffer, Arc<[u8]>)]>,
    packing: Arc<[CopyArithmetic]>,
    inverse_zero: Option<CopyArithmetic>,
    initial: Option<CopyArithmetic>,
    scatter: Option<CopyArithmetic>,
}
impl std::fmt::Debug for GdnArithmetic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GdnArithmetic")
            .field("phase", &self.phase)
            .field("slot", &self.slot)
            .field("typed_outputs", &self.typed_outputs)
            .field("stages", &self.stages.len())
            .field("control_roles", &self.controls.len())
            .finish_non_exhaustive()
    }
}
impl GdnArithmetic {
    pub(super) fn new(
        phase: GdnPhase,
        slot: GdnCacheSlot,
        typed_outputs: bool,
        controls: Arc<[(GdnBuffer, Arc<[u8]>)]>,
        stages: &[Stage],
        packing: &[Copy],
        copies: [Option<&Copy>; 3],
    ) -> Result<Self> {
        slot.validate(phase)?;
        if stages.len() != phase.stages().len() {
            return Err(Error::State(
                "complete GDN arithmetic stage membership".into(),
            ));
        }
        let family = *stages
            .first()
            .ok_or_else(|| Error::State("missing actual GDN code family".into()))?
            .spec
            .arithmetic
            .family();
        for (actual, &expected) in stages.iter().zip(phase.stages()) {
            actual.spec.arithmetic.require_stage(phase, expected)?;
            family.require_same(actual.spec.arithmetic.family())?;
        }
        let [inverse_zero, initial, scatter] = copies;
        if phase == GdnPhase::CachedDecode {
            if inverse_zero.is_some() || initial.is_some() || scatter.is_some() {
                return Err(Error::State(
                    "decode has no prefill initialization copies".into(),
                ));
            }
        } else if inverse_zero.is_none_or(|c| c.arithmetic.kind != CopyKind::ZeroInverse)
            || initial.is_none_or(|c| {
                c.arithmetic.kind
                    != if phase.has_initial_state() {
                        CopyKind::Gather
                    } else {
                        CopyKind::Zero
                    }
            })
            || scatter.is_none_or(|c| c.arithmetic.kind != CopyKind::Scatter)
        {
            return Err(Error::State(
                "exact zero-Ai and initial/final state arithmetic copies".into(),
            ));
        }
        let result = Self {
            family,
            phase,
            slot,
            typed_outputs,
            controls,
            stages: stages.iter().map(|s| s.spec.arithmetic.clone()).collect(),
            packing: packing.iter().map(|c| c.arithmetic.clone()).collect(),
            inverse_zero: inverse_zero.map(|c| c.arithmetic.clone()),
            initial: initial.map(|c| c.arithmetic.clone()),
            scatter: scatter.map(|c| c.arithmetic.clone()),
        };
        if result.logical_metadata_bytes() > 64 * 1024 {
            return Err(Error::Limit("bounded GDN arithmetic metadata".into()));
        }
        Ok(result)
    }
    // Owned payload/descriptor upper bound, including construction scratch.
    // Existing caller metadata reservations cover this; not RSS/allocator proof.
    fn logical_metadata_bytes(&self) -> usize {
        let controls = self
            .controls
            .iter()
            .map(|(_, b)| b.len() + 16)
            .sum::<usize>()
            + self.controls.len() * std::mem::size_of::<(GdnBuffer, Arc<[u8]>)>();
        let stages = self
            .stages
            .iter()
            .map(|s| s.metadata_bytes() + std::mem::size_of::<StageArithmetic>())
            .sum::<usize>();
        let copies = self
            .packing
            .iter()
            .chain(self.inverse_zero.iter())
            .chain(self.initial.iter())
            .chain(self.scatter.iter())
            .map(|c| {
                2 * std::mem::size_of::<CopyArithmetic>()
                    + c.source.len()
                    + 16
                    + c.bindings.len() * std::mem::size_of::<(usize, BufferIOAttr)>()
            })
            .sum::<usize>();
        std::mem::size_of::<Self>() + controls + stages + copies + 16 * 1024
    }
    pub(crate) fn require_instance(
        &self,
        phase: GdnPhase,
        slot: GdnCacheSlot,
        typed_outputs: bool,
    ) -> Result<()> {
        if self.phase != phase || self.slot != slot || self.typed_outputs != typed_outputs {
            return Err(Error::State(
                "different bound GDN phase/slot/typed-output instance".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn family(&self) -> &GdnFamilyIdentity {
        &self.family
    }
    /// Exact metadata comparison only. No baseline or physical acceptance implied.
    pub(crate) fn require_same(&self, other: &Self) -> Result<()> {
        if self != other {
            return Err(Error::Unsupported(
                "different exact GDN arithmetic instance".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
