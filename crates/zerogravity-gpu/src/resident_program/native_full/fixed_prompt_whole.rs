//! WholeInitial47 binding in the same original body/cursor/flight.
use super::fixed_prompt_prefix::PrefixPhase;
use super::*;
use crate::cuda::gdn::FixedPromptGdnArithmetic;
use crate::cuda::native::{
    fixed_prompt_model_native::FixedPromptModelNativePrograms,
    fixed_prompt_model_prefix::FixedPromptPrefixCut,
};
impl FullModelProgram {
    pub(crate) fn bind_fixed_prompt_whole(
        &self,
        cut: Arc<FixedPromptPrefixCut>,
        native: Arc<FixedPromptModelNativePrograms>,
        states: Vec<Arc<BoundModelState>>,
        gdn: BTreeMap<usize, Arc<ModelGdnPrepared>>,
        arithmetic: FixedPromptGdnArithmetic,
    ) -> Result<()> {
        cut.require_whole()?;
        arithmetic.require_owned()?;
        let scope = self
            .fixed_prompt
            .as_ref()
            .ok_or_else(|| Error::State("canonical47 model source scope absent".into()))?;
        if !Arc::ptr_eq(scope, cut.scope())
            || !Arc::ptr_eq(&self.record, scope.record())
            || !self.source.same_issuer(scope.source())
            || self.bound.get().is_some()
            || self.launched.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != 0
            || self.output_submitted.load(Ordering::Acquire)
            || self.greedy.get().is_some()
            || self.natural.is_some()
            || self.fused.len() != cut.duals()
            || states.len() != 24
            || gdn.len() != 18
        {
            return Err(Error::State(
                "wholeInitial47 exactly once-bound real24 model body".into(),
            ));
        }
        scope.require_values(&self.values)?;
        native.require_values(&cut, &self.values)?;
        for id in cut.values() {
            self.values.lease(*id)?;
        }
        if self
            .values
            .lease(ValueId(cut.values().len() as u32))
            .is_ok()
        {
            return Err(Error::State(
                "canonical47 prefix cannot retain future uncomputed value leases".into(),
            ));
        }
        let mut layers = BTreeSet::new();
        for state in &states {
            if state.call() != self.record.call()
                || !layers.insert(state.layer())
                || states
                    .first()
                    .is_some_and(|first| !first.same_request(state))
            {
                return Err(Error::State(
                    "canonical47 same original schedule/source/layer states".into(),
                ));
            }
            state.require_fixed_prompt_whole(&cut)?;
            if state.kv_domain().is_ok() {
                let attention = native
                    .attention_bindings()?
                    .get(&state.layer())
                    .ok_or_else(|| {
                        Error::State("wholeInitial47 actual attention owner absent".into())
                    })?;
                if !Arc::ptr_eq(state, attention.domains()) {
                    return Err(Error::State(
                        "wholeInitial47 same original KV state/native owner".into(),
                    ));
                }
            } else {
                let prepared = gdn.get(&state.layer()).ok_or_else(|| {
                    Error::State("wholeInitial47 actual indexed GDN absent".into())
                })?;
                prepared.require_value_state(&self.values, state)?;
                prepared.require_fixed_prompt_binding(scope, arithmetic)?;
            }
        }
        if layers != (0..24).collect()
            || gdn.keys().copied().collect::<BTreeSet<_>>()
                != (0..24).filter(|l| l % 4 != 3).collect()
        {
            return Err(Error::State(
                "wholeInitial47 exact24 original state /18 GDN owners".into(),
            ));
        }
        let gdn_arithmetic = gdn
            .values()
            .next()
            .ok_or_else(|| Error::State("canonical47 GDN arithmetic absent".into()))?
            .arithmetic()
            .clone();
        self.gdn_family.require_same(gdn_arithmetic.family())?;
        for prepared in gdn.values() {
            gdn_arithmetic.require_same(prepared.arithmetic())?;
        }
        let expected_fused = cut
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(i, node)| {
                matches!(&node.operation,PlanOperation::SourceOperator {descriptor,..}
                if matches!(descriptor.kind(),SourceOperatorKind::ResidualRms(_)))
                .then_some(i)
            })
            .collect::<BTreeSet<_>>();
        if self.fused.keys().copied().collect::<BTreeSet<_>>() != expected_fused {
            return Err(Error::State(
                "canonical47 original five indexed dual RMS actors".into(),
            ));
        }
        let mut accesses = BTreeMap::new();
        for (lease, access) in self
            .values
            .accesses()
            .into_iter()
            .chain(self.rms.accesses().iter().cloned())
        {
            insert_access(&mut accesses, lease, access);
        }
        for fused in self.fused.values() {
            for (lease, access) in fused.accesses() {
                insert_access(&mut accesses, lease.clone(), *access);
            }
        }
        for state in &states {
            for (lease, access) in state.leases() {
                insert_access(&mut accesses, lease, access);
            }
        }
        for prepared in gdn.values() {
            for (lease, access) in prepared.accesses() {
                insert_access(&mut accesses, lease, access);
            }
        }
        for (lease, access) in native.accesses() {
            insert_access(&mut accesses, lease.clone(), *access);
        }
        self.bound
            .set(BoundPhase {
                gdn_arithmetic,
                phase: 0,
                library: None,
                fixed_prompt_prefix: Some(PrefixPhase { cut, native }),
                canonical_call: None,
                text: None,
                states,
                gdn,
                attention: BTreeMap::new(),
                accesses: accesses.into_values().collect(),
            })
            .map_err(|_| Error::State("canonical47 model prefix already bound".into()))
    }
    pub(crate) fn arm_fixed_prompt_whole_fault(&self, mode: usize) -> Result<()> {
        let cut = &self
            .bound
            .get()
            .and_then(|b| b.fixed_prompt_prefix.as_ref())
            .ok_or_else(|| Error::State("wholeInitial47 fault before binding".into()))?
            .cut;
        cut.require_whole()?;
        if !matches!(mode, 0 | 17..=21)
            || self.launched.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != 0
            || self.fault.load(Ordering::Acquire) != 0
        {
            return Err(Error::State(
                "wholeInitial47 closed once-armed fault mode".into(),
            ));
        }
        self.fault.store(mode, Ordering::Release);
        Ok(())
    }
    pub(crate) fn whole_fault_cut_seen(&self) -> bool {
        self.fault_cut.load(Ordering::Acquire)
    }
}
