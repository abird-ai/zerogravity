//! Whole initial47 metadata over the SAME original immutable full record.
use super::*;
use zerogravity_core::accelerator::program::SourceCall;
use zerogravity_core::accelerator::program::StateEffect;

#[derive(Clone)]
pub(crate) struct AttentionSite {
    layer: usize,
    nodes: [usize; 7],
    values: [ValueId; 8],
    effect: StateEffect,
}
impl AttentionSite {
    pub(crate) fn layer(&self) -> usize {
        self.layer
    }
    /// QKV,QK,KV,FA,Sigmoid,Multiply,Out; source-owned indices only.
    pub(crate) fn nodes(&self) -> [usize; 7] {
        self.nodes
    }
    /// packed,Q,K,gate,FA,Sigmoid,Multiply,Out.
    pub(crate) fn values(&self) -> [ValueId; 8] {
        self.values
    }
    pub(crate) fn effect(&self) -> StateEffect {
        self.effect
    }
    fn bind(scope: &FixedPromptModelBinding, index: usize) -> Result<Self> {
        Self::bind_nodes(scope.record().call(), scope.record().plan().nodes(), index)
    }
    pub(crate) fn bind_nodes(call: &SourceCall, nodes: &[PlanNode], index: usize) -> Result<Self> {
        let mut found=nodes.iter().enumerate().filter(|(_,node)| {
            matches!(&node.operation,PlanOperation::SourceOperator {descriptor,..}
                if matches!(descriptor.kind(),SourceOperatorKind::KvAppend {domain,..} if domain==KvDomain(index)))
        });
        let kv = found
            .next()
            .ok_or_else(|| Error::State("actual KV domain absent".into()))?
            .0;
        let qk = kv
            .checked_sub(1)
            .ok_or_else(|| Error::State("actual rotary predecessor absent".into()))?;
        if found.next().is_some() || qk == 0 {
            return Err(Error::State("actual rotary domain repeated/first".into()));
        }
        let fa_count=nodes.iter().filter(|node| matches!(&node.operation,
            PlanOperation::SourceOperator{descriptor,..}
            if matches!(descriptor.kind(),SourceOperatorKind::FlashAttention{domain,..} if domain==KvDomain(index)))).count();
        if fa_count != 1 {
            return Err(Error::State(
                "whole actual attention domain repeated/absent".into(),
            ));
        }
        let layer = index;
        let ids = [qk - 1, qk, qk + 1, qk + 2, qk + 3, qk + 4, qk + 5];
        for (ordinal, &i) in ids.iter().enumerate() {
            let descriptor = checked_operator(nodes, i)?;
            let valid = match ordinal {
                0 | 6 => matches!(
                    descriptor.kind(),
                    SourceOperatorKind::Projection { selected_row: None }
                ),
                1 => matches!(descriptor.kind(), SourceOperatorKind::QkRmsRotaryGated(_)),
                2 => {
                    matches!(descriptor.kind(),SourceOperatorKind::KvAppend{domain,..} if domain==KvDomain(index))
                }
                3 => {
                    matches!(descriptor.kind(),SourceOperatorKind::FlashAttention{domain,..} if domain==KvDomain(index))
                }
                4 => matches!(descriptor.kind(), SourceOperatorKind::Sigmoid { .. }),
                5 => matches!(descriptor.kind(), SourceOperatorKind::Multiply { .. }),
                _ => false,
            };
            if !valid || descriptor.call() != call {
                return Err(Error::State(
                    "whole actual attention operator/call differs".into(),
                ));
            }
        }
        if ids.windows(2).any(|p| p[1] != p[0] + 1) {
            return Err(Error::State("whole attention node order".into()));
        }
        let [packed] = nodes[ids[0]].outputs.as_slice() else {
            return Err(Error::State("whole QKV output".into()));
        };
        let [q, k, gate] = nodes[ids[1]].outputs.as_slice() else {
            return Err(Error::State("whole QK outputs".into()));
        };
        let one = |i: usize| -> Result<ValueId> {
            let [out] = nodes[i].outputs.as_slice() else {
                return Err(Error::State("whole unary output".into()));
            };
            Ok(*out)
        };
        let (fa, sig, mul, out) = (one(ids[3])?, one(ids[4])?, one(ids[5])?, one(ids[6])?);
        let effect = nodes[ids[2]]
            .state
            .ok_or_else(|| Error::State("whole KV effect absent".into()))?;
        if nodes[ids[1]].inputs.as_slice() != [*packed]
            || nodes[ids[2]].inputs.as_slice() != [*k, *packed]
            || !nodes[ids[2]].outputs.is_empty()
            || nodes[ids[3]].inputs.as_slice() != [*q]
            || nodes[ids[4]].inputs.as_slice() != [*gate]
            || nodes[ids[5]].inputs.as_slice() != [fa, sig]
            || nodes[ids[6]].inputs.as_slice() != [mul]
            || effect.read.slot != StateSlot::Kv(KvDomain(index))
            || effect.write.slot != effect.read.slot
            || effect.read.ordinal != 0
            || effect.write.ordinal != 1
            || !matches!(&nodes[ids[3]].operation,PlanOperation::SourceOperator{state:Some(read),..} if *read==effect.write)
        {
            return Err(Error::State(
                "whole actual attention SSA/state version differs".into(),
            ));
        }
        for (ordinal, i) in ids.into_iter().enumerate() {
            if ordinal != 2 && nodes[i].state.is_some() {
                return Err(Error::State("whole attention extra state effect".into()));
            }
            if ordinal != 3
                && !matches!(
                    &nodes[i].operation,
                    PlanOperation::SourceOperator { state: None, .. }
                )
            {
                return Err(Error::State("whole attention imported state read".into()));
            }
        }
        Ok(Self {
            layer,
            nodes: ids,
            values: [*packed, *q, *k, *gate, fa, sig, mul, out],
            effect,
        })
    }
}
impl FixedPromptPrefixCut {
    pub(crate) fn is_whole(&self) -> bool {
        self.kind == FixedPromptPrefixKind::WholePrefill
    }
    pub(crate) fn require_whole(&self) -> Result<()> {
        if !self.is_whole()
            || self.end != 255
            || self.values.len() != 327
            || !self.carries.is_empty()
            || self.whole_sites.keys().copied().collect::<Vec<_>>() != [3, 7, 11, 15, 19, 23]
        {
            return Err(Error::Unsupported(
                "exact original wholeInitial47 source cut required".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn attention_site(&self, layer: usize) -> Result<&AttentionSite> {
        self.require_whole()?;
        self.whole_sites
            .get(&layer)
            .ok_or_else(|| Error::State("whole actual attention layer absent".into()))
    }
    pub(crate) fn bind_whole(scope: Arc<FixedPromptModelBinding>) -> Result<Arc<Self>> {
        if scope.kind() != FixedPromptCallKind::Prefill {
            return Err(Error::Unsupported("whole initial47 only".into()));
        }
        let record = scope.record();
        let plan = record.plan();
        crate::resident_program::native_full::FullModelProgram::require_record(record)?;
        let mut produced = BTreeSet::new();
        let mut changed = BTreeSet::new();
        let mut duals = 0;
        for (i, node) in plan.nodes().iter().enumerate() {
            if node.inputs.iter().any(|id| !produced.contains(&id.0)) {
                return Err(Error::State("whole consumed future SSA".into()));
            }
            if i == 0 {
                if !matches!(&node.operation, PlanOperation::SourceOperator { descriptor, .. } if matches!(descriptor.kind(), zerogravity_core::accelerator::program::SourceOperatorKind::EmbeddingRows))
                {
                    return Err(Error::State("whole original embedding".into()));
                }
            } else {
                let Some(descriptor) = record.operator(i)? else {
                    return Err(Error::State("whole typed original node".into()));
                };
                if descriptor.call() != record.call() {
                    return Err(Error::State("whole original source/call/layer".into()));
                }
                if matches!(descriptor.kind(), SourceOperatorKind::ResidualRms(_)) {
                    duals += 1;
                    scope.rms(i)?;
                }
            }
            for id in &node.outputs {
                if !produced.insert(id.0) {
                    return Err(Error::State("whole repeated SSA output".into()));
                }
            }
            if let Some(effect) = node.state
                && (effect.read.slot != effect.write.slot
                    || effect.read.ordinal != 0
                    || effect.write.ordinal != 1
                    || !changed.insert(effect.write.slot))
            {
                return Err(Error::State("whole repeated/foreign state effect".into()));
            }
        }
        if produced.iter().copied().ne(0..327)
            || changed != plan.required_domains().iter().copied().collect()
            || changed.len() != 42
            || duals != 48
        {
            return Err(Error::State("whole327/42/48 original coverage".into()));
        }
        let final_node = &plan.nodes()[253];
        let head = &plan.nodes()[254];
        if !(matches!(
            &final_node.operation,
            PlanOperation::SourceOperator { state: None, .. }
        ) && record.operator(253)?.is_some_and(|descriptor| {
            matches!(descriptor.kind(), SourceOperatorKind::ResidualRms(_))
        })) || final_node.outputs.as_slice() != [ValueId(324), ValueId(325)]
            || final_node.state.is_some()
            || !(matches!(
                &head.operation,
                PlanOperation::SourceOperator { state: None, .. }
            ) && record.operator(254)?.is_some_and(|descriptor| {
                matches!(descriptor.kind(), SourceOperatorKind::Projection { .. })
            }))
            || head.inputs.as_slice() != [ValueId(324)]
            || head.outputs.as_slice() != [ValueId(326)]
            || head.state.is_some()
        {
            return Err(Error::State("whole genuine final253/head254 SSA".into()));
        }
        for layer in (0..24).filter(|i| i % 4 != 3) {
            scope.gdn_site(layer)?;
        }
        let whole_sites = [3, 7, 11, 15, 19, 23]
            .into_iter()
            .map(|l| Ok((l, AttentionSite::bind(&scope, l)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let result = Arc::new(Self {
            kind: FixedPromptPrefixKind::WholePrefill,
            scope,
            end: 255,
            values: (0..327).map(ValueId).collect(),
            carries: Vec::new(),
            whole_sites,
        });
        result.require_whole()?;
        Ok(result)
    }
}

#[cfg(test)]
#[path = "whole/tests.rs"]
mod tests;
