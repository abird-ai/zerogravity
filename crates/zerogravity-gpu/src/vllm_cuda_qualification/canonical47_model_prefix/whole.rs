//! Whole initial47 metadata over the SAME original immutable full record.
use super::*;
use zerogravity_core::accelerator::program::{StateEffect, Vllm08Call, Vllm08Layer, Vllm08RmsSite};

#[derive(Clone)]
pub(crate) struct AttentionSite {
    layer: Vllm08Layer,
    nodes: [usize; 7],
    values: [ValueId; 8],
    effect: StateEffect,
}
impl AttentionSite {
    pub(crate) fn layer(&self) -> Vllm08Layer {
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
    fn bind(scope: &Canonical47ModelBinding, index: usize) -> Result<Self> {
        Self::bind_nodes(scope.record().call(), scope.record().plan().nodes(), index)
    }
    fn bind_nodes(call: &Vllm08Call, nodes: &[PlanNode], index: usize) -> Result<Self> {
        let layer = call.layer(index)?;
        if !layer.is_attention() {
            return Err(Error::State("whole attention source layer".into()));
        }
        let find = |kind| -> Result<usize> {
            let mut found = nodes.iter().enumerate().filter(|(_, n)| {
                matches!(&n.operation,
                PlanOperation::TypedVllm08{descriptor,..} if descriptor.call()==call
                && descriptor.layer()==Some(layer) && descriptor.kind()==kind)
            });
            let i = found
                .next()
                .ok_or_else(|| Error::State("whole original attention site absent".into()))?
                .0;
            if found.next().is_some() {
                return Err(Error::State("whole attention source site repeated".into()));
            }
            Ok(i)
        };
        let ids = [
            find(Vllm08Kind::Projection(Vllm08Projection::AttentionQkv))?,
            find(Vllm08Kind::AttentionQk)?,
            find(Vllm08Kind::KvAppend)?,
            find(Vllm08Kind::FlashAttention)?,
            find(Vllm08Kind::Sigmoid)?,
            find(Vllm08Kind::Multiply)?,
            find(Vllm08Kind::Projection(Vllm08Projection::AttentionOut))?,
        ];
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
            || !matches!(&nodes[ids[3]].operation,PlanOperation::TypedVllm08{state:Some(read),..} if *read==effect.write)
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
                    PlanOperation::TypedVllm08 { state: None, .. }
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
impl Canonical47PrefixCut {
    pub(crate) fn is_whole(&self) -> bool {
        self.kind == Canonical47PrefixKind::WholeInitial47
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
    pub(crate) fn bind_whole(scope: Arc<Canonical47ModelBinding>) -> Result<Arc<Self>> {
        if scope.kind() != Canonical47CallKind::Prefill {
            return Err(Error::Unsupported("whole initial47 only".into()));
        }
        let record = scope.record();
        let plan = record.plan();
        crate::resident_program::native_full::FullModelProgram::require_record(
            scope.source().source(),
            record,
        )?;
        let mut produced = BTreeSet::new();
        let mut changed = BTreeSet::new();
        let mut duals = 0;
        for (i, node) in plan.nodes().iter().enumerate() {
            if node.inputs.iter().any(|id| !produced.contains(&id.0)) {
                return Err(Error::State("whole consumed future SSA".into()));
            }
            if i == 0 {
                if !matches!(node.operation, PlanOperation::TypedVllmBf16Embedding { .. }) {
                    return Err(Error::State("whole original embedding".into()));
                }
            } else {
                let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                    return Err(Error::State("whole typed original node".into()));
                };
                if descriptor.call() != record.call()
                    || (i < 253 && descriptor.layer().is_none())
                    || (i >= 253 && descriptor.layer().is_some())
                {
                    return Err(Error::State("whole original source/call/layer".into()));
                }
                if matches!(descriptor.kind(), Vllm08Kind::ResidualRms(_)) {
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
        if !matches!(&final_node.operation,PlanOperation::TypedVllm08{descriptor,state:None}
            if descriptor.kind()==Vllm08Kind::ResidualRms(Vllm08RmsSite::Final))
            || final_node.outputs.as_slice() != [ValueId(324), ValueId(325)]
            || final_node.state.is_some()
            || !matches!(&head.operation,PlanOperation::TypedVllm08{descriptor,state:None}
                if descriptor.kind()==Vllm08Kind::Projection(Vllm08Projection::Head))
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
            kind: Canonical47PrefixKind::WholeInitial47,
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
