//! One-use cursor consumed by the ORIGINAL ResidentProgram node loop.
use super::recipe::{ComponentCut, ComponentRecipe};
use crate::resident_inference::{gdn_state::BoundGdnDomains, native_fragment::PrefixPhaseEnqueue};
use crate::resident_leases::ResidentService;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_core::accelerator::program::{PlanNode, PlanOperation, StateEffect};
use zerogravity_core::{Error, Result};

struct NodeOrder {
    cut: ComponentCut,
    launched: AtomicBool,
    next: AtomicUsize,
}
impl Default for NodeOrder {
    fn default() -> Self {
        Self::for_cut(ComponentCut::Gated)
    }
}
impl NodeOrder {
    fn for_cut(cut: ComponentCut) -> Self {
        Self {
            cut,
            launched: AtomicBool::new(false),
            next: AtomicUsize::new(0),
        }
    }
    fn begin(&self) -> Result<()> {
        self.launched
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("source2B prefix storage is one-use".into()))?;
        Ok(())
    }
    fn take(&self, index: usize) -> Result<()> {
        if !self.launched.load(Ordering::Acquire) || index >= self.cut.node_end() {
            return Err(Error::State("source2B exact six-node active cursor".into()));
        }
        self.next
            .compare_exchange(index, index + 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("source2B node duplicated, reordered or skipped".into()))?;
        Ok(())
    }
    fn finished(&self) -> Result<()> {
        if !self.launched.load(Ordering::Acquire)
            || self.next.load(Ordering::Acquire) != self.cut.node_end()
        {
            return Err(Error::State(
                "source2B actual six-node prefix incomplete".into(),
            ));
        }
        Ok(())
    }
}
pub(super) struct ComponentCursor {
    recipe: Arc<ComponentRecipe>,
    order: NodeOrder,
}
impl ComponentCursor {
    pub(super) fn new(recipe: Arc<ComponentRecipe>) -> Self {
        let order = NodeOrder::for_cut(recipe.cut());
        Self { recipe, order }
    }
    pub(super) fn begin(
        &self,
        service: &Arc<ResidentService>,
        permit: &PrefixPhaseEnqueue<'_>,
    ) -> Result<()> {
        if service.source_identity() != self.recipe.bindings().call().source() {
            return Err(Error::State(
                "source2B original service/plan issuer changed".into(),
            ));
        }
        // Original borrowed Session rechecks current root/epoch/generation.
        // One updated GDN domain pair is a component scope, no fullmodel permit.
        permit.require_invocation(service, self.recipe.bindings().call(), 0, 1)?;
        self.order.begin()
    }
    pub(super) fn take(&self, index: usize, node: &PlanNode) -> Result<()> {
        let original = self
            .recipe
            .nodes()
            .get(index)
            .ok_or_else(|| Error::Unsupported("source2B stop before GdnOut".into()))?;
        if node.inputs != original.inputs
            || node.outputs != original.outputs
            || node.state != original.state
        {
            return Err(Error::State(
                "source2B original SSA/effect cursor changed".into(),
            ));
        }
        match (&node.operation, &original.operation) {
            (
                PlanOperation::SourceOperator {
                    descriptor: a,
                    state: sa,
                },
                PlanOperation::SourceOperator {
                    descriptor: b,
                    state: sb,
                },
            ) if a == b && sa == sb => {}
            _ => {
                return Err(Error::State(
                    "source2B exact original mathematical operator changed".into(),
                ));
            }
        }
        // Burn before enqueue: failure/unknown never resets this cursor.
        self.order.take(index)
    }
    pub(super) fn require_stateful_node(
        &self,
        index: usize,
        domains: &BoundGdnDomains,
        service: &Arc<ResidentService>,
        permit: &PrefixPhaseEnqueue<'_>,
    ) -> Result<StateEffect> {
        let convolution = match index {
            4 => true,
            5 => false,
            _ => return Err(Error::State("not source2B stateful node".into())),
        };
        permit.require_invocation(service, self.recipe.bindings().call(), 0, 1)?;
        domains.require_enqueue(service, Some(permit))?;
        let effect = self.recipe.effects()?[usize::from(!convolution)];
        if domains.invocation() != self.recipe.bindings().call()
            || domains.effect(convolution) != effect
        {
            return Err(Error::State(
                "source2B original current-root/source/effect join changed".into(),
            ));
        }
        Ok(effect)
    }
    pub(super) fn finished(&self) -> Result<()> {
        self.order.finished()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn order_cannot_replay_skip_or_execute_gdn_out() {
        let order = NodeOrder::default();
        assert!(order.take(0).is_err());
        assert!(order.finished().is_err());
        order.begin().unwrap();
        assert!(order.begin().is_err());
        assert!(order.take(1).is_err());
        for index in 0..ComponentCut::Gated.node_end() {
            order.take(index).unwrap();
            assert!(order.take(index).is_err());
        }
        order.finished().unwrap();
        assert!(order.take(ComponentCut::Gated.node_end()).is_err());
        assert!(order.begin().is_err());
    }
}

#[cfg(test)]
mod post_attention_tests {
    use super::*;
    #[test]
    fn eight_node_cut_cannot_stop_after_gated_replay_or_run_gate_up() {
        let order = NodeOrder::for_cut(ComponentCut::PostAttention);
        order.begin().unwrap();
        for i in 0..6 {
            order.take(i).unwrap();
        }
        assert!(order.finished().is_err());
        assert!(order.take(7).is_err());
        order.take(6).unwrap();
        order.take(7).unwrap();
        order.finished().unwrap();
        assert!(order.take(7).is_err());
        assert!(order.take(8).is_err());
        assert!(order.begin().is_err());
        assert_eq!(ComponentCut::Gated.value_end(), 7);
        assert_eq!(ComponentCut::PostAttention.value_end(), 10);
    }
}

#[cfg(test)]
mod mlp_order_tests {
    use super::*;
    #[test]
    fn actual_eleven_node_cursor_is_once_ordered_and_stops_before_layer1() {
        let order = NodeOrder::for_cut(ComponentCut::Mlp);
        assert!(order.take(0).is_err());
        order.begin().unwrap();
        assert!(order.begin().is_err());
        for i in 0..11 {
            assert!(order.finished().is_err());
            if i + 1 < 11 {
                assert!(order.take(i + 1).is_err());
            }
            order.take(i).unwrap();
            assert!(order.take(i).is_err());
        }
        order.finished().unwrap();
        assert!(order.take(11).is_err());
    }
}
