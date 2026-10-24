//! Finite cut over the original full record; no alternate plan or state issuer.
use super::canonical47_model::{Canonical47CallKind, Canonical47ModelBinding};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
mod whole;
pub(crate) use whole::AttentionSite;
use zerogravity_core::accelerator::program::{
    ConvDomain, KvDomain, PlanNode, PlanOperation, RecurrentDomain, StateSlot, ValueId, Vllm08Kind,
    Vllm08Projection,
};
use zerogravity_core::{Error, Result};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Canonical47PrefixKind {
    FirstThree,
    ThroughAttentionQkv,
    ThroughAttentionKv,
    ThroughAttentionFa,
    ThroughLayer3,
    WholeInitial47,
}
impl Canonical47PrefixKind {
    pub(crate) const fn values(self) -> usize {
        match self {
            Self::WholeInitial47 => 327,
            Self::FirstThree => 39,
            Self::ThroughAttentionQkv => 42,
            Self::ThroughAttentionKv => 45,
            Self::ThroughAttentionFa => 46,
            Self::ThroughLayer3 => 54,
        }
    }
    pub(crate) const fn duals(self) -> usize {
        match self {
            Self::WholeInitial47 => 48,
            Self::FirstThree => 5,
            Self::ThroughAttentionQkv | Self::ThroughAttentionKv | Self::ThroughAttentionFa => 6,
            Self::ThroughLayer3 => 7,
        }
    }
    pub(crate) const fn weight_groups(self) -> usize {
        match self {
            Self::WholeInitial47 => 114,
            Self::FirstThree => 15,
            Self::ThroughAttentionQkv | Self::ThroughAttentionKv | Self::ThroughAttentionFa => 16,
            Self::ThroughLayer3 => 19,
        }
    }
    pub(crate) const fn source_windows(self) -> usize {
        match self {
            Self::WholeInitial47 => 1008,
            Self::FirstThree => 132,
            Self::ThroughAttentionQkv | Self::ThroughAttentionKv | Self::ThroughAttentionFa => 142,
            Self::ThroughLayer3 => 168,
        }
    }
    pub(crate) const fn native_records(self) -> usize {
        match self {
            Self::WholeInitial47 => 16,
            Self::FirstThree => 8,
            Self::ThroughAttentionQkv => 10,
            Self::ThroughAttentionKv => 12,
            Self::ThroughAttentionFa => 13,
            Self::ThroughLayer3 => 15,
        }
    }
}
pub(crate) struct Canonical47PrefixCut {
    kind: Canonical47PrefixKind,
    scope: Arc<Canonical47ModelBinding>,
    end: usize,
    values: Vec<ValueId>,
    carries: Vec<StateSlot>,
    whole_sites: BTreeMap<usize, AttentionSite>,
}
impl Canonical47PrefixCut {
    pub(crate) fn bind(scope: Arc<Canonical47ModelBinding>) -> Result<Arc<Self>> {
        Self::bind_kind(scope, Canonical47PrefixKind::FirstThree)
    }
    pub(crate) fn bind_kind(
        scope: Arc<Canonical47ModelBinding>,
        kind: Canonical47PrefixKind,
    ) -> Result<Arc<Self>> {
        if kind == Canonical47PrefixKind::WholeInitial47 {
            return Self::bind_whole(scope);
        }
        if scope.kind() != Canonical47CallKind::Prefill {
            return Err(Error::Unsupported(
                "canonical47 first3 cut is initial Prefill47 only".into(),
            ));
        }
        let record = scope.record();
        let boundary = record
            .boundaries()
            .get(2)
            .ok_or_else(|| Error::State("canonical47 actual layer2 boundary absent".into()))?;
        let mut last = record
            .plan()
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                matches!(&node.operation,PlanOperation::TypedVllm08 {descriptor,..}
                if descriptor.layer().map(|l|l.index())==Some(2)
                    && descriptor.kind()==Vllm08Kind::Projection(Vllm08Projection::Down))
            });
        let (index, node) = last
            .next()
            .ok_or_else(|| Error::State("canonical47 layer2 Down absent".into()))?;
        if last.next().is_some() || node.outputs.as_slice() != [boundary[0]] {
            return Err(Error::State(
                "canonical47 exact real layer2 Down boundary".into(),
            ));
        }
        let first3_end = index + 1;
        let next = record
            .plan()
            .nodes()
            .get(first3_end)
            .ok_or_else(|| Error::State("canonical47 stop-node absent".into()))?;
        if first3_end != 31
            || !matches!(&next.operation,PlanOperation::TypedVllm08 {descriptor,..}
            if descriptor.layer().map(|l|l.index())==Some(3)
                && descriptor.kind()==Vllm08Kind::ResidualRms(zerogravity_core::accelerator::program::Vllm08RmsSite::Input))
        {
            return Err(Error::Unsupported(
                "canonical47 exact cut before layer3 input dual/attention".into(),
            ));
        }
        let end = match kind {
            Canonical47PrefixKind::WholeInitial47 => {
                return Err(Error::State("whole source cut dispatch differs".into()));
            }
            Canonical47PrefixKind::FirstThree => first3_end,
            Canonical47PrefixKind::ThroughAttentionQkv => {
                let nodes = record.plan().nodes();
                require_qkv_extension(nodes)?;
                33
            }
            Canonical47PrefixKind::ThroughAttentionKv => {
                require_qkv_extension(record.plan().nodes())?;
                require_kv_extension(record.plan().nodes())?;
                35
            }
            Canonical47PrefixKind::ThroughAttentionFa => {
                require_qkv_extension(record.plan().nodes())?;
                require_kv_extension(record.plan().nodes())?;
                require_fa_extension(record.plan().nodes())?;
                36
            }
            Canonical47PrefixKind::ThroughLayer3 => {
                require_qkv_extension(record.plan().nodes())?;
                require_kv_extension(record.plan().nodes())?;
                require_fa_extension(record.plan().nodes())?;
                require_layer3_extension(record.plan().nodes())?;
                43
            }
        };
        let mut produced = BTreeSet::new();
        let mut changed = BTreeSet::new();
        for (i, node) in record.plan().nodes()[..end].iter().enumerate() {
            if node.inputs.iter().any(|id| !produced.contains(&id.0)) {
                return Err(Error::State(
                    "canonical47 prefix consumed a future full-record value".into(),
                ));
            }
            if i == 0 {
                if !matches!(node.operation, PlanOperation::TypedVllmBf16Embedding { .. }) {
                    return Err(Error::State(
                        "canonical47 real original source embedding first".into(),
                    ));
                }
            } else if !matches!(&node.operation,PlanOperation::TypedVllm08 {descriptor,..}
                if descriptor.call()==record.call()
                    && (descriptor.layer().is_some_and(|l|l.index()<3 && !l.is_attention())
                        || (kind != Canonical47PrefixKind::FirstThree
                            && (matches!(i,31|32)
                                || (matches!(kind,Canonical47PrefixKind::ThroughAttentionKv|Canonical47PrefixKind::ThroughAttentionFa|Canonical47PrefixKind::ThroughLayer3) && matches!(i,33|34))
                                || (matches!(kind,Canonical47PrefixKind::ThroughAttentionFa|Canonical47PrefixKind::ThroughLayer3) && i==35)
                                || (kind == Canonical47PrefixKind::ThroughLayer3 && (36..43).contains(&i)))
                            && descriptor.layer().is_some_and(|l|l.index()==3 && l.is_attention()))))
            {
                return Err(Error::State(
                    "canonical47 prefix cannot contain attention/head/later layer".into(),
                ));
            }
            for id in &node.outputs {
                if !produced.insert(id.0) {
                    return Err(Error::State(
                        "canonical47 actual output ValueId repeated".into(),
                    ));
                }
            }
            if let Some(effect) = node.state {
                changed.insert(effect.write.slot);
            }
        }
        let values = produced.into_iter().map(ValueId).collect::<Vec<_>>();
        let carries = record
            .plan()
            .required_domains()
            .iter()
            .filter(|slot| !changed.contains(*slot))
            .copied()
            .collect::<Vec<_>>();
        let mut expected_changed = (0..3)
            .flat_map(|i| {
                [
                    StateSlot::Conv(ConvDomain(i)),
                    StateSlot::Recurrent(RecurrentDomain(i)),
                ]
            })
            .collect::<BTreeSet<_>>();
        if matches!(
            kind,
            Canonical47PrefixKind::ThroughAttentionKv
                | Canonical47PrefixKind::ThroughAttentionFa
                | Canonical47PrefixKind::ThroughLayer3
        ) {
            expected_changed.insert(StateSlot::Kv(KvDomain(3)));
        }
        if values != (0..kind.values() as u32).map(ValueId).collect::<Vec<_>>()
            || changed != expected_changed
            || carries.len() != 42 - expected_changed.len()
            || record.boundaries()[..3]
                .iter()
                .flatten()
                .any(|v| !values.contains(v))
        {
            return Err(Error::State(
                "canonical47 exact39 computed values/6 effects/36 real carries".into(),
            ));
        }
        Ok(Arc::new(Self {
            kind,
            scope,
            end,
            values,
            carries,
            whole_sites: BTreeMap::new(),
        }))
    }
    pub(crate) fn kind(&self) -> Canonical47PrefixKind {
        self.kind
    }
    pub(crate) fn includes_qkv(&self) -> bool {
        self.kind != Canonical47PrefixKind::FirstThree
    }
    pub(crate) fn includes_kv(&self) -> bool {
        matches!(
            self.kind,
            Canonical47PrefixKind::ThroughAttentionKv
                | Canonical47PrefixKind::ThroughAttentionFa
                | Canonical47PrefixKind::ThroughLayer3
                | Canonical47PrefixKind::WholeInitial47
        )
    }
    pub(crate) fn includes_fa(&self) -> bool {
        matches!(
            self.kind,
            Canonical47PrefixKind::ThroughAttentionFa
                | Canonical47PrefixKind::ThroughLayer3
                | Canonical47PrefixKind::WholeInitial47
        )
    }
    pub(crate) fn includes_layer3(&self) -> bool {
        self.kind == Canonical47PrefixKind::ThroughLayer3
    }
    pub(crate) fn require_layer3_tail(&self) -> Result<()> {
        if !self.includes_layer3() {
            return Err(Error::Unsupported(
                "canonical47 complete layer3 cut absent".into(),
            ));
        }
        require_layer3_extension(self.scope.record().plan().nodes())
    }
    pub(crate) fn is_attention_cut(&self) -> bool {
        matches!(
            (self.kind, self.nodes().len(), self.values().len()),
            (Canonical47PrefixKind::ThroughAttentionKv, 35, 45)
                | (Canonical47PrefixKind::ThroughAttentionFa, 36, 46)
                | (Canonical47PrefixKind::ThroughLayer3, 43, 54)
        )
    }
    pub(crate) fn require_fa_read(&self) -> Result<()> {
        if !self.includes_fa() {
            return Err(Error::Unsupported(
                "canonical47 finite FA cut absent".into(),
            ));
        }
        require_fa_extension(self.scope.record().plan().nodes())
    }
    pub(crate) fn duals(&self) -> usize {
        self.kind.duals()
    }
    pub(crate) fn weight_groups(&self) -> usize {
        self.kind.weight_groups()
    }
    pub(crate) fn source_windows(&self) -> usize {
        self.kind.source_windows()
    }
    pub(crate) fn native_records(&self) -> usize {
        self.kind.native_records()
    }
    pub(crate) fn scope(&self) -> &Arc<Canonical47ModelBinding> {
        &self.scope
    }
    pub(crate) fn nodes(&self) -> &[PlanNode] {
        &self.scope.record().plan().nodes()[..self.end]
    }
    pub(crate) fn values(&self) -> &[ValueId] {
        &self.values
    }
    pub(crate) fn carries(&self) -> &[StateSlot] {
        &self.carries
    }
    pub(crate) fn require_finished(&self, next: usize) -> Result<()> {
        if next != self.end {
            return Err(Error::State(
                "canonical47 original prefix traversal incomplete".into(),
            ));
        }
        Ok(())
    }
}

fn require_qkv_extension(nodes: &[PlanNode]) -> Result<()> {
    use zerogravity_core::accelerator::program::Vllm08RmsSite;
    let (Some(dual), Some(qkv), Some(stop)) = (nodes.get(31), nodes.get(32), nodes.get(33)) else {
        return Err(Error::State(
            "canonical47 original QKV cut nodes absent".into(),
        ));
    };
    let typed = |node: &PlanNode, kind| {
        matches!(&node.operation,
        PlanOperation::TypedVllm08 { descriptor,.. }
        if descriptor.layer().is_some_and(|l|l.index()==3 && l.is_attention())
            && descriptor.kind()==kind)
    };
    if !typed(dual, Vllm08Kind::ResidualRms(Vllm08RmsSite::Input))
        || dual.inputs.as_slice() != [ValueId(38), ValueId(35)]
        || dual.outputs.as_slice() != [ValueId(39), ValueId(40)]
        || !typed(qkv, Vllm08Kind::Projection(Vllm08Projection::AttentionQkv))
        || qkv.inputs.as_slice() != [ValueId(39)]
        || qkv.outputs.as_slice() != [ValueId(41)]
        || !typed(stop, Vllm08Kind::AttentionQk)
        || dual.state.is_some()
        || qkv.state.is_some()
    {
        return Err(Error::Unsupported(
            "canonical47 exact nodes31/32 before33QK with genuine producers".into(),
        ));
    }
    Ok(())
}

fn require_kv_extension(nodes: &[PlanNode]) -> Result<()> {
    use zerogravity_core::accelerator::program::{StateEffect, StateVersion};
    let (Some(qk), Some(kv), Some(stop)) = (nodes.get(33), nodes.get(34), nodes.get(35)) else {
        return Err(Error::State(
            "canonical47 exact QK/KV/FA endpoint absent".into(),
        ));
    };
    let effect = StateEffect {
        read: StateVersion {
            slot: StateSlot::Kv(KvDomain(3)),
            ordinal: 0,
        },
        write: StateVersion {
            slot: StateSlot::Kv(KvDomain(3)),
            ordinal: 1,
        },
    };
    let call = match nodes.get(32).map(|n| &n.operation) {
        Some(PlanOperation::TypedVllm08 { descriptor, .. }) => descriptor.call(),
        _ => {
            return Err(Error::State(
                "canonical47 original node32 call issuer absent".into(),
            ));
        }
    };
    let typed = |node: &PlanNode, kind| {
        matches!(&node.operation,
        PlanOperation::TypedVllm08 { descriptor, .. }
        if descriptor.call()==call && descriptor.layer().is_some_and(|l| l.index() == 3 && l.is_attention())
            && descriptor.kind() == kind)
    };
    if !typed(qk, Vllm08Kind::AttentionQk)
        || qk.inputs.as_slice() != [ValueId(41)]
        || qk.outputs.as_slice() != [ValueId(42), ValueId(43), ValueId(44)]
        || qk.state.is_some()
        || !typed(kv, Vllm08Kind::KvAppend)
        || kv.inputs.as_slice() != [ValueId(43), ValueId(41)]
        || !kv.outputs.is_empty()
        || kv.state != Some(effect)
        || !matches!(
            &kv.operation,
            PlanOperation::TypedVllm08 { state: None, .. }
        )
        || !typed(stop, Vllm08Kind::FlashAttention)
        || stop.inputs.as_slice() != [ValueId(42)]
        || !matches!(&stop.operation,PlanOperation::TypedVllm08 {state:Some(v),..} if *v == effect.write)
    {
        return Err(Error::State(
            "canonical47 original nodes33/34 before35FA and exact privateKv3 version".into(),
        ));
    }
    Ok(())
}

fn require_fa_extension(nodes: &[PlanNode]) -> Result<()> {
    let (Some(kv), Some(fa), Some(stop)) = (nodes.get(34), nodes.get(35), nodes.get(36)) else {
        return Err(Error::State("canonical47 FA/sigmoid cursor absent".into()));
    };
    let PlanOperation::TypedVllm08 {
        descriptor: append,
        state: None,
    } = &kv.operation
    else {
        return Err(Error::State(
            "canonical47 original KV descriptor absent".into(),
        ));
    };
    let Some(effect) = kv.state else {
        return Err(Error::State(
            "canonical47 original append effect absent".into(),
        ));
    };
    let valid = matches!(&fa.operation,PlanOperation::TypedVllm08 {descriptor,state:Some(read)}
        if descriptor.call()==append.call() && descriptor.layer()==append.layer()
            && descriptor.kind()==Vllm08Kind::FlashAttention && *read==effect.write
            && read.slot==StateSlot::Kv(KvDomain(3)) && read.ordinal==1
            && descriptor.parameters().is_empty() && descriptor.matrices().is_empty())
        && fa.inputs.as_slice() == [ValueId(42)]
        && fa.outputs.as_slice() == [ValueId(45)]
        && fa.state.is_none()
        && matches!(&stop.operation,PlanOperation::TypedVllm08 {descriptor,state:None}
            if descriptor.call()==append.call() && descriptor.layer()==append.layer()
                && descriptor.kind()==Vllm08Kind::Sigmoid)
        && stop.inputs.as_slice() == [ValueId(44)]
        && stop.outputs.as_slice() == [ValueId(46)]
        && stop.state.is_none();
    if !valid {
        return Err(Error::State(
            "canonical47 exact node35 appendedKv3 read and computed45 before36sigmoid".into(),
        ));
    }
    Ok(())
}

fn require_layer3_extension(nodes: &[PlanNode]) -> Result<()> {
    let Some(base) = nodes.get(35) else {
        return Err(Error::State(
            "canonical47 original FA producer absent".into(),
        ));
    };
    let PlanOperation::TypedVllm08 { descriptor: fa, .. } = &base.operation else {
        return Err(Error::State(
            "canonical47 original FA descriptor absent".into(),
        ));
    };
    for (index, kind, inputs, outputs) in [
        (
            36,
            Vllm08Kind::Sigmoid,
            vec![ValueId(44)],
            vec![ValueId(46)],
        ),
        (
            37,
            Vllm08Kind::Multiply,
            vec![ValueId(45), ValueId(46)],
            vec![ValueId(47)],
        ),
        (
            38,
            Vllm08Kind::Projection(Vllm08Projection::AttentionOut),
            vec![ValueId(47)],
            vec![ValueId(48)],
        ),
        (
            39,
            Vllm08Kind::ResidualRms(
                zerogravity_core::accelerator::program::Vllm08RmsSite::PostAttention,
            ),
            vec![ValueId(48), ValueId(40)],
            vec![ValueId(49), ValueId(50)],
        ),
        (
            40,
            Vllm08Kind::Projection(Vllm08Projection::GateUp),
            vec![ValueId(49)],
            vec![ValueId(51)],
        ),
        (
            41,
            Vllm08Kind::SiluAndMul,
            vec![ValueId(51)],
            vec![ValueId(52)],
        ),
        (
            42,
            Vllm08Kind::Projection(Vllm08Projection::Down),
            vec![ValueId(52)],
            vec![ValueId(53)],
        ),
    ] {
        let Some(node) = nodes.get(index) else {
            return Err(Error::State(
                "canonical47 layer3 original tail node absent".into(),
            ));
        };
        if !matches!(&node.operation,PlanOperation::TypedVllm08{descriptor,state:None}
            if descriptor.kind()==kind && descriptor.call()==fa.call() && descriptor.layer()==fa.layer())
            || node.inputs != inputs
            || node.outputs != outputs
            || node.state.is_some()
        {
            return Err(Error::State(
                "canonical47 exact layer3 tail source/SSA/operation changed".into(),
            ));
        }
    }
    let Some(stop) = nodes.get(43) else {
        return Err(Error::State("canonical47 layer4 stop absent".into()));
    };
    if !matches!(&stop.operation,PlanOperation::TypedVllm08{descriptor,state:None}
        if descriptor.call()==fa.call() && descriptor.layer().is_some_and(|l|l.index()==4)
        && descriptor.kind()==Vllm08Kind::ResidualRms(zerogravity_core::accelerator::program::Vllm08RmsSite::Input))
        || stop.inputs.as_slice() != [ValueId(53), ValueId(50)]
        || stop.outputs.as_slice() != [ValueId(54), ValueId(55)]
        || stop.state.is_some()
    {
        return Err(Error::State(
            "canonical47 must stop before original layer4 input dual".into(),
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests;
