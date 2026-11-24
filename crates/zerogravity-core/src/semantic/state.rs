//! Source-template state versions and geometry. No device storage or publication.
use crate::{
    Error, Result,
    accelerator::program::{
        ForwardPlan, PlanOperation, SourceOperatorKind, SourceStateAccess, StateEffect, StateSlot,
        StateVersion,
    },
    numerics::ActivationPrecision,
};
use std::collections::BTreeMap;

/// Plane geometry retained by the original state owner. Vector ordering and
/// arithmetic belong to the original plan/profile, not to an equal byte extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateLayout {
    Vector { elements: usize },
    Kv { slab: usize, capacity: usize },
}
impl StateLayout {
    pub fn planes(self) -> Result<(usize, usize)> {
        let (planes, elements) = match self {
            Self::Vector { elements } => (1, elements),
            Self::Kv { slab, capacity } => (2, product(slab, capacity)?),
        };
        if elements == 0 {
            return Err(Error::Malformed("empty state plane".into()));
        }
        Ok((planes, elements))
    }
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| n != 0)
        .ok_or_else(|| Error::Limit("state positive element extent".into()))
}

/// One original state writer; all writes remain private until the whole
/// invocation is completed and the existing root owner publishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceStateDomain {
    pub effect: StateEffect,
    pub writer: usize,
    pub layout: StateLayout,
}
impl SourceStateDomain {
    pub fn slot(self) -> StateSlot {
        self.effect.write.slot
    }
}

/// Exact external state version at a region boundary. Version 0 is the
/// committed input, version 1 the invocation's provisional result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionStatePort {
    pub version: StateVersion,
    pub layout: StateLayout,
}
impl RegionStatePort {
    pub fn precision(self) -> ActivationPrecision {
        match self.version.slot {
            StateSlot::Recurrent(_) => ActivationPrecision::F32,
            _ => ActivationPrecision::Bf16WidenedF32,
        }
    }
}

impl ForwardPlan {
    /// One cold derivation of the original call-free state contract. Native
    /// preflight and closed-region lowering use these same writer/read edges.
    /// StateCarry lacks a source layout and cannot establish a physical domain.
    pub fn source_state_domains(&self) -> Result<Vec<SourceStateDomain>> {
        let geometry = self.source_plan_geometry().ok_or_else(|| {
            Error::Unsupported("state domains require an original source template".into())
        })?;
        if self.text_invocation().is_some() || self.is_operator_slice() {
            return Err(Error::Unsupported(
                "state domains require a whole call-free plan".into(),
            ));
        }
        let mut domains = BTreeMap::<StateSlot, SourceStateDomain>::new();
        let frontier = self.state_nodes().len();
        for (index, node) in self.nodes().iter().enumerate() {
            let PlanOperation::SourceOperatorTemplate { descriptor, state } = &node.operation
            else {
                return Err(Error::Unsupported(
                    "state domain non-template operation".into(),
                ));
            };
            if descriptor.source() != self.source()
                || descriptor.max_rows() != geometry.max_rows()
                || descriptor.vocabulary() != geometry.vocabulary()
                || descriptor.numerical_contract().program_profile() != self.numerical_profile()
            {
                return Err(Error::State("state domain source/profile differs".into()));
            }
            match descriptor.state_access() {
                SourceStateAccess::None => {
                    if node.state.is_some() || state.is_some() {
                        return Err(Error::State("stateless node has state effects".into()));
                    }
                }
                SourceStateAccess::Write(slot) => {
                    let expected = StateEffect {
                        read: StateVersion { slot, ordinal: 0 },
                        write: StateVersion { slot, ordinal: 1 },
                    };
                    if index >= frontier
                        || node.state != Some(expected)
                        || state.is_some()
                        || !self.required_domains().contains(&slot)
                    {
                        return Err(Error::State("state writer effect/version differs".into()));
                    }
                    let layout = match descriptor.kind() {
                        SourceOperatorKind::CausalConvSplit { geometry: g, .. } => {
                            // The original mixed contract retains the complete
                            // convolution window; CUDA retains only history.
                            let history =
                                if descriptor.numerical_contract().uses_torch_mixed_operators() {
                                    g.kernel
                                } else {
                                    g.kernel.checked_sub(1).ok_or_else(|| {
                                        Error::Limit("convolution state history".into())
                                    })?
                                };
                            StateLayout::Vector {
                                elements: product(g.qkv_channels, history)?,
                            }
                        }
                        SourceOperatorKind::GatedDelta { geometry: g, .. } => StateLayout::Vector {
                            elements: product(product(g.value_heads, g.key_width)?, g.value_width)?,
                        },
                        SourceOperatorKind::KvAppend { key_width, .. } => StateLayout::Kv {
                            slab: key_width,
                            capacity: geometry.context_capacity(),
                        },
                        _ => {
                            return Err(Error::Unsupported(
                                "state writer has no declared layout".into(),
                            ));
                        }
                    };
                    layout.planes()?;
                    if domains
                        .insert(
                            slot,
                            SourceStateDomain {
                                effect: expected,
                                writer: index,
                                layout,
                            },
                        )
                        .is_some()
                    {
                        return Err(Error::State("state domain has multiple writers".into()));
                    }
                }
                SourceStateAccess::Read(slot) => {
                    let domain = domains
                        .get(&slot)
                        .ok_or_else(|| Error::State("state reader precedes its writer".into()))?;
                    if index >= frontier
                        || node.state.is_some()
                        || *state != Some(domain.effect.write)
                    {
                        return Err(Error::State(
                            "state reader provisional version differs".into(),
                        ));
                    }
                    let SourceOperatorKind::Attention { geometry: g, .. } = descriptor.kind()
                    else {
                        return Err(Error::Unsupported("state reader layout absent".into()));
                    };
                    if domain.layout
                        != (StateLayout::Kv {
                            slab: product(g.kv_heads, g.width)?,
                            capacity: geometry.context_capacity(),
                        })
                    {
                        return Err(Error::State(
                            "attention state layout differs from append".into(),
                        ));
                    }
                }
            }
        }
        if domains
            .keys()
            .copied()
            .ne(self.required_domains().iter().copied())
            || domains
                .values()
                .map(|d| d.effect.write)
                .ne(self.proposed_domains().iter().copied())
        {
            return Err(Error::State("source state domain coverage differs".into()));
        }
        Ok(domains.into_values().collect())
    }
}
