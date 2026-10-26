//! Finite cut over the original full record; no alternate plan or state issuer.
use super::fixed_prompt_model::{FixedPromptCallKind, FixedPromptModelBinding};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use zerogravity_core::accelerator::program::{SourceOperator, SourceOperatorKind};
mod whole;
pub(crate) use whole::AttentionSite;
use zerogravity_core::accelerator::program::ConvDomain;
use zerogravity_core::accelerator::program::KvDomain;
use zerogravity_core::accelerator::program::PlanNode;
use zerogravity_core::accelerator::program::PlanOperation;
use zerogravity_core::accelerator::program::RecurrentDomain;
use zerogravity_core::accelerator::program::StateSlot;
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::{Error, Result};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FixedPromptPrefixKind {
    FirstThree,
    ThroughAttentionQkv,
    ThroughAttentionKv,
    ThroughAttentionFa,
    ThroughLayer3,
    WholePrefill,
}
impl FixedPromptPrefixKind {
    pub(crate) const fn values(self) -> usize {
        match self {
            Self::WholePrefill => 327,
            Self::FirstThree => 39,
            Self::ThroughAttentionQkv => 42,
            Self::ThroughAttentionKv => 45,
            Self::ThroughAttentionFa => 46,
            Self::ThroughLayer3 => 54,
        }
    }
    pub(crate) const fn duals(self) -> usize {
        match self {
            Self::WholePrefill => 48,
            Self::FirstThree => 5,
            Self::ThroughAttentionQkv | Self::ThroughAttentionKv | Self::ThroughAttentionFa => 6,
            Self::ThroughLayer3 => 7,
        }
    }
    pub(crate) const fn weight_groups(self) -> usize {
        match self {
            Self::WholePrefill => 114,
            Self::FirstThree => 15,
            Self::ThroughAttentionQkv | Self::ThroughAttentionKv | Self::ThroughAttentionFa => 16,
            Self::ThroughLayer3 => 19,
        }
    }
    pub(crate) const fn source_windows(self) -> usize {
        match self {
            Self::WholePrefill => 1008,
            Self::FirstThree => 132,
            Self::ThroughAttentionQkv | Self::ThroughAttentionKv | Self::ThroughAttentionFa => 142,
            Self::ThroughLayer3 => 168,
        }
    }
    pub(crate) const fn native_records(self) -> usize {
        match self {
            Self::WholePrefill => 16,
            Self::FirstThree => 8,
            Self::ThroughAttentionQkv => 10,
            Self::ThroughAttentionKv => 12,
            Self::ThroughAttentionFa => 13,
            Self::ThroughLayer3 => 15,
        }
    }
}
pub(crate) struct FixedPromptPrefixCut {
    kind: FixedPromptPrefixKind,
    scope: Arc<FixedPromptModelBinding>,
    end: usize,
    values: Vec<ValueId>,
    carries: Vec<StateSlot>,
    whole_sites: BTreeMap<usize, AttentionSite>,
}
impl FixedPromptPrefixCut {
    pub(crate) fn bind(scope: Arc<FixedPromptModelBinding>) -> Result<Arc<Self>> {
        Self::bind_kind(scope, FixedPromptPrefixKind::FirstThree)
    }
    pub(crate) fn bind_kind(
        scope: Arc<FixedPromptModelBinding>,
        kind: FixedPromptPrefixKind,
    ) -> Result<Arc<Self>> {
        if kind == FixedPromptPrefixKind::WholePrefill {
            return Self::bind_whole(scope);
        }
        if scope.kind() != FixedPromptCallKind::Prefill {
            return Err(Error::Unsupported(
                "fixed_prompt first3 cut is initial Prefill47 only".into(),
            ));
        }
        let record = scope.record();
        let boundary = record
            .selection()
            .boundaries()
            .get(2)
            .ok_or_else(|| Error::State("fixed_prompt actual layer2 boundary absent".into()))?;
        let mut last = record
            .plan()
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, node)| node.outputs.as_slice() == [boundary[0]]);
        let (index, node) = last
            .next()
            .ok_or_else(|| Error::State("fixed_prompt layer2 Down absent".into()))?;
        if last.next().is_some() || node.outputs.as_slice() != [boundary[0]] {
            return Err(Error::State(
                "fixed_prompt exact real layer2 Down boundary".into(),
            ));
        }
        let first3_end = index + 1;
        let next = record
            .plan()
            .nodes()
            .get(first3_end)
            .ok_or_else(|| Error::State("fixed_prompt stop-node absent".into()))?;
        if first3_end != 31
            || !record
                .operator(first3_end)?
                .is_some_and(|d| matches!(d.kind(), SourceOperatorKind::ResidualRms(_)))
            || next.inputs != [boundary[0], boundary[1]]
        {
            return Err(Error::Unsupported(
                "fixed prompt exact boundary before next input dual".into(),
            ));
        }
        let end = match kind {
            FixedPromptPrefixKind::WholePrefill => {
                return Err(Error::State("whole source cut dispatch differs".into()));
            }
            FixedPromptPrefixKind::FirstThree => first3_end,
            FixedPromptPrefixKind::ThroughAttentionQkv => {
                let nodes = record.plan().nodes();
                require_qkv_extension(nodes)?;
                33
            }
            FixedPromptPrefixKind::ThroughAttentionKv => {
                require_qkv_extension(record.plan().nodes())?;
                require_kv_extension(record.plan().nodes())?;
                35
            }
            FixedPromptPrefixKind::ThroughAttentionFa => {
                require_qkv_extension(record.plan().nodes())?;
                require_kv_extension(record.plan().nodes())?;
                require_fa_extension(record.plan().nodes())?;
                36
            }
            FixedPromptPrefixKind::ThroughLayer3 => {
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
                    "fixed_prompt prefix consumed a future full-record value".into(),
                ));
            }
            if i == 0 {
                if !matches!(&node.operation, PlanOperation::SourceOperator { descriptor, .. } if matches!(descriptor.kind(), zerogravity_core::accelerator::program::SourceOperatorKind::EmbeddingRows))
                {
                    return Err(Error::State(
                        "fixed_prompt real original source embedding first".into(),
                    ));
                }
            } else if !record
                .operator(i)?
                .is_some_and(|descriptor| descriptor.call() == record.call())
            {
                return Err(Error::State(
                    "fixed prompt actual prefix call differs".into(),
                ));
            }
            for id in &node.outputs {
                if !produced.insert(id.0) {
                    return Err(Error::State(
                        "fixed_prompt actual output ValueId repeated".into(),
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
            FixedPromptPrefixKind::ThroughAttentionKv
                | FixedPromptPrefixKind::ThroughAttentionFa
                | FixedPromptPrefixKind::ThroughLayer3
        ) {
            expected_changed.insert(StateSlot::Kv(KvDomain(3)));
        }
        if values != (0..kind.values() as u32).map(ValueId).collect::<Vec<_>>()
            || changed != expected_changed
            || carries.len() != 42 - expected_changed.len()
            || record.selection().boundaries()[..3]
                .iter()
                .flatten()
                .any(|v| !values.contains(v))
        {
            return Err(Error::State(
                "fixed_prompt exact39 computed values/6 effects/36 real carries".into(),
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
    pub(crate) fn kind(&self) -> FixedPromptPrefixKind {
        self.kind
    }
    pub(crate) fn includes_qkv(&self) -> bool {
        self.kind != FixedPromptPrefixKind::FirstThree
    }
    pub(crate) fn includes_kv(&self) -> bool {
        matches!(
            self.kind,
            FixedPromptPrefixKind::ThroughAttentionKv
                | FixedPromptPrefixKind::ThroughAttentionFa
                | FixedPromptPrefixKind::ThroughLayer3
                | FixedPromptPrefixKind::WholePrefill
        )
    }
    pub(crate) fn includes_fa(&self) -> bool {
        matches!(
            self.kind,
            FixedPromptPrefixKind::ThroughAttentionFa
                | FixedPromptPrefixKind::ThroughLayer3
                | FixedPromptPrefixKind::WholePrefill
        )
    }
    pub(crate) fn includes_layer3(&self) -> bool {
        self.kind == FixedPromptPrefixKind::ThroughLayer3
    }
    pub(crate) fn require_layer3_tail(&self) -> Result<()> {
        if !self.includes_layer3() {
            return Err(Error::Unsupported(
                "fixed_prompt complete layer3 cut absent".into(),
            ));
        }
        require_layer3_extension(self.scope.record().plan().nodes())
    }
    pub(crate) fn is_attention_cut(&self) -> bool {
        matches!(
            (self.kind, self.nodes().len(), self.values().len()),
            (FixedPromptPrefixKind::ThroughAttentionKv, 35, 45)
                | (FixedPromptPrefixKind::ThroughAttentionFa, 36, 46)
                | (FixedPromptPrefixKind::ThroughLayer3, 43, 54)
        )
    }
    pub(crate) fn require_fa_read(&self) -> Result<()> {
        if !self.includes_fa() {
            return Err(Error::Unsupported(
                "fixed_prompt finite FA cut absent".into(),
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
    pub(crate) fn scope(&self) -> &Arc<FixedPromptModelBinding> {
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
                "fixed_prompt original prefix traversal incomplete".into(),
            ));
        }
        Ok(())
    }
}

fn checked_operator(nodes: &[PlanNode], index: usize) -> Result<&SourceOperator> {
    let node = nodes
        .get(index)
        .ok_or_else(|| Error::State("actual source node absent".into()))?;
    let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
        return Err(Error::State(
            "actual mathematical source operator absent".into(),
        ));
    };
    Ok(descriptor)
}
fn require_qkv_extension(nodes: &[PlanNode]) -> Result<()> {
    let (Some(dual), Some(qkv), Some(_)) = (nodes.get(31), nodes.get(32), nodes.get(33)) else {
        return Err(Error::State(
            "fixed prompt actual QKV endpoint absent".into(),
        ));
    };
    let (a, b, c) = (
        checked_operator(nodes, 31)?,
        checked_operator(nodes, 32)?,
        checked_operator(nodes, 33)?,
    );
    if !matches!(a.kind(), SourceOperatorKind::ResidualRms(_))
        || !matches!(
            b.kind(),
            SourceOperatorKind::Projection { selected_row: None }
        )
        || !matches!(c.kind(), SourceOperatorKind::QkRmsRotaryGated(_))
        || a.call() != b.call()
        || b.call() != c.call()
        || dual.inputs != [ValueId(38), ValueId(35)]
        || dual.outputs != [ValueId(39), ValueId(40)]
        || qkv.inputs != [ValueId(39)]
        || qkv.outputs != [ValueId(41)]
        || dual.state.is_some()
        || qkv.state.is_some()
    {
        return Err(Error::State(
            "fixed prompt QKV actual operations/SSA/call differ".into(),
        ));
    }
    Ok(())
}
fn require_kv_extension(nodes: &[PlanNode]) -> Result<()> {
    use zerogravity_core::accelerator::program::{StateEffect, StateVersion};
    let (Some(qk), Some(kv), Some(fa)) = (nodes.get(33), nodes.get(34), nodes.get(35)) else {
        return Err(Error::State("fixed prompt actual QK/KV/FA absent".into()));
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
    let (a, b, c) = (
        checked_operator(nodes, 33)?,
        checked_operator(nodes, 34)?,
        checked_operator(nodes, 35)?,
    );
    if !matches!(a.kind(), SourceOperatorKind::QkRmsRotaryGated(_))
        || !matches!(b.kind(),SourceOperatorKind::KvAppend {domain,..} if domain==KvDomain(3))
        || !matches!(c.kind(),SourceOperatorKind::FlashAttention {domain,..} if domain==KvDomain(3))
        || a.call() != b.call()
        || b.call() != c.call()
        || qk.inputs != [ValueId(41)]
        || qk.outputs != [ValueId(42), ValueId(43), ValueId(44)]
        || qk.state.is_some()
        || kv.inputs != [ValueId(43), ValueId(41)]
        || !kv.outputs.is_empty()
        || kv.state != Some(effect)
        || !matches!(
            &kv.operation,
            PlanOperation::SourceOperator { state: None, .. }
        )
        || fa.inputs != [ValueId(42)]
        || !matches!(&fa.operation,PlanOperation::SourceOperator{state:Some(v),..} if *v==effect.write)
    {
        return Err(Error::State(
            "fixed prompt actual appended KV3 state/SSA differs".into(),
        ));
    }
    Ok(())
}
fn require_fa_extension(nodes: &[PlanNode]) -> Result<()> {
    require_kv_extension(nodes)?;
    let (Some(fa), Some(sigmoid)) = (nodes.get(35), nodes.get(36)) else {
        return Err(Error::State("fixed prompt actual FA/sigmoid absent".into()));
    };
    let a = checked_operator(nodes, 35)?;
    let b = checked_operator(nodes, 36)?;
    if a.call() != b.call()
        || !matches!(b.kind(), SourceOperatorKind::Sigmoid { .. })
        || fa.outputs != [ValueId(45)]
        || fa.state.is_some()
        || sigmoid.inputs != [ValueId(44)]
        || sigmoid.outputs != [ValueId(46)]
        || sigmoid.state.is_some()
        || !matches!(
            &sigmoid.operation,
            PlanOperation::SourceOperator { state: None, .. }
        )
    {
        return Err(Error::State(
            "fixed prompt actual FA/sigmoid source/SSA differs".into(),
        ));
    }
    Ok(())
}
fn require_layer3_extension(nodes: &[PlanNode]) -> Result<()> {
    require_fa_extension(nodes)?;
    let call = checked_operator(nodes, 35)?.call();
    for (index, inputs, outputs) in [
        (36, vec![44], vec![46]),
        (37, vec![45, 46], vec![47]),
        (38, vec![47], vec![48]),
        (39, vec![48, 40], vec![49, 50]),
        (40, vec![49], vec![51]),
        (41, vec![51], vec![52]),
        (42, vec![52], vec![53]),
        (43, vec![53, 50], vec![54, 55]),
    ] {
        let node = nodes
            .get(index)
            .ok_or_else(|| Error::State("fixed prompt tail node absent".into()))?;
        let descriptor = checked_operator(nodes, index)?;
        let kind = match index {
            36 => matches!(descriptor.kind(), SourceOperatorKind::Sigmoid { .. }),
            37 => matches!(descriptor.kind(), SourceOperatorKind::Multiply { .. }),
            39 | 43 => matches!(descriptor.kind(), SourceOperatorKind::ResidualRms(_)),
            41 => matches!(descriptor.kind(), SourceOperatorKind::SiluAndMul { .. }),
            _ => matches!(
                descriptor.kind(),
                SourceOperatorKind::Projection { selected_row: None }
            ),
        };
        descriptor.validate()?;
        let (input_widths, output_widths): (&[usize], &[usize]) = match index {
            36 => (&[2048], &[2048]),
            37 => (&[2048, 2048], &[2048]),
            38 => (&[2048], &[1024]),
            39 | 43 => (&[1024, 1024], &[1024, 1024]),
            40 => (&[1024], &[7168]),
            41 => (&[7168], &[3584]),
            42 => (&[3584], &[1024]),
            _ => {
                return Err(Error::State(
                    "fixed prompt tail operation index differs".into(),
                ));
            }
        };
        if !kind
            || descriptor
                .input_elements()?
                .into_iter()
                .ne(input_widths.iter().map(|width| 47 * width))
            || descriptor
                .output_elements()?
                .into_iter()
                .ne(output_widths.iter().map(|width| 47 * width))
            || descriptor.call() != call
            || node.state.is_some()
            || !matches!(
                &node.operation,
                PlanOperation::SourceOperator { state: None, .. }
            )
            || node.inputs.iter().map(|v| v.0).ne(inputs)
            || node.outputs.iter().map(|v| v.0).ne(outputs)
        {
            return Err(Error::State(
                "fixed prompt actual tail operation/source/SSA differs".into(),
            ));
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests;
