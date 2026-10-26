//! Source-owned GDN role checks.
//! Native source/ABI/catalogues and all physical authorities remain backend-owned.
use super::*;
use crate::plan::{QwenOperation, QwenOperator};

/// Family-authenticated immutable data; no second plan, permit or device owner.
pub struct PreparedGdnParameters {
    nodes: [usize; 2],
    operators: [SourceOperator; 2],
    parameters: [SourceParameter; 4],
    site: usize,
}
impl PreparedGdnParameters {
    pub fn nodes(&self) -> [usize; 2] {
        self.nodes
    }
    pub fn operators(&self) -> &[SourceOperator; 2] {
        &self.operators
    }
    pub fn parameters(&self) -> &[SourceParameter; 4] {
        &self.parameters
    }
    pub fn site(&self) -> usize {
        self.site
    }
    pub(super) fn prepare(
        source: &OriginalWindowModelSource,
        conv: &QwenOperator,
        recurrent: &QwenOperator,
        nodes: [usize; 2],
        plan: &ForwardPlan,
    ) -> Result<Self> {
        conv.validate()?;
        recurrent.validate()?;
        let layer = conv
            .layer()
            .ok_or_else(|| Error::State("full24 GDN source layer absent".into()))?;
        if layer.is_attention()
            || conv.kind() != QwenOperation::GdnConv
            || recurrent.kind() != QwenOperation::GdnRecurrent
            || conv.call() != recurrent.call()
            || recurrent.layer() != Some(layer)
            || conv.call().source() != source.source_identity()
            || nodes[0] >= nodes[1]
            || conv.parameters().len() != 1
            || recurrent.parameters().len() != 3
        {
            return Err(Error::State(
                "full24 actual paired GDN source descriptors".into(),
            ));
        }
        let parameters = [
            conv.parameters()[0].clone(),
            recurrent.parameters()[0].clone(),
            recurrent.parameters()[1].clone(),
            recurrent.parameters()[2].clone(),
        ];
        for (descriptor, role) in parameters
            .iter()
            .zip(crate::hybrid_decoder::ParameterGroup::GdnAll.roles(layer.index()))
        {
            source.validate_parameter(role, descriptor)?;
        }
        let operators = nodes
            .map(
                |index| match plan.nodes().get(index).map(|node| &node.operation) {
                    Some(PlanOperation::SourceOperator { descriptor, .. }) => {
                        Ok((**descriptor).clone())
                    }
                    _ => Err(Error::State("GDN actual mathematical node absent".into())),
                },
            )
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("GDN exact common operator pair".into()))?;
        let operators: [SourceOperator; 2] = operators;
        conv.require_mathematical_operator(&operators[0])?;
        recurrent.require_mathematical_operator(&operators[1])?;
        Ok(Self {
            nodes,
            operators,
            parameters,
            site: layer.index(),
        })
    }
}

impl PreparedGdnParameters {
    pub fn metadata_bytes(&self) -> Result<usize> {
        self.parameters
            .iter()
            .try_fold(std::mem::size_of::<Self>(), |bytes, parameter| {
                parameter
                    .shape
                    .capacity()
                    .checked_mul(std::mem::size_of::<u64>())
                    .and_then(|shape| shape.checked_add(parameter.name.capacity()))
                    .and_then(|extra| bytes.checked_add(extra))
                    .ok_or_else(|| Error::Limit("prepared GDN source metadata extent".into()))
            })
    }
}
pub(super) fn for_record(
    source: &OriginalWindowModelSource,
    record: &QwenPlan,
) -> Result<Box<[PreparedGdnParameters]>> {
    let mut result = Vec::with_capacity(18);
    for index in 0..record.plan().nodes().len() {
        let Some(conv) = record.operator(index)? else {
            continue;
        };
        if conv.kind() != QwenOperation::GdnConv {
            continue;
        }
        let recurrent = record
            .operator(index + 1)?
            .ok_or_else(|| Error::State("actual paired GDN recurrence absent".into()))?;
        result.push(PreparedGdnParameters::prepare(
            source,
            &conv,
            &recurrent,
            [index, index + 1],
            record.plan(),
        )?);
    }
    Ok(result.into_boxed_slice())
}
pub(super) fn for_fragment(
    source: &OriginalWindowModelSource,
    record: QwenFragmentView<'_>,
    pairs: &[[usize; 2]],
) -> Result<Box<[PreparedGdnParameters]>> {
    let mut result = Vec::with_capacity(pairs.len());
    let call = crate::plan::QwenSourceCall::new(record.invocation().clone())?;
    for &nodes in pairs {
        let conv = record
            .plan()
            .nodes()
            .get(nodes[0])
            .ok_or_else(|| Error::State("actual fragment convolution absent".into()))?;
        let site = match source_descriptor(conv, record.invocation())?.kind() {
            SourceOperatorKind::CausalConvSplit { domain, .. } => domain.0,
            _ => {
                return Err(Error::State(
                    "actual fragment convolution domain absent".into(),
                ));
            }
        };
        let mut family = Vec::with_capacity(2);
        for (index, kind) in [
            (nodes[0], QwenOperation::GdnConv),
            (nodes[1], QwenOperation::GdnRecurrent),
        ] {
            let node = record
                .plan()
                .nodes()
                .get(index)
                .ok_or_else(|| Error::State("actual fragment GDN node absent".into()))?;
            let descriptor = source_descriptor(node, record.invocation())?;
            family.push(QwenOperator::new(
                call.clone(),
                Some(call.layer(site)?),
                kind,
                Vec::new(),
                descriptor.parameters().to_vec(),
            )?);
        }
        result.push(PreparedGdnParameters::prepare(
            source,
            &family[0],
            &family[1],
            nodes,
            record.plan(),
        )?);
    }
    Ok(result.into_boxed_slice())
}
pub(super) fn metadata_bytes(pairs: &[PreparedGdnParameters]) -> Result<usize> {
    pairs.iter().try_fold(0usize, |sum, pair| {
        sum.checked_add(pair.metadata_bytes()?)
            .ok_or_else(|| Error::Limit("prepared GDN pair metadata extent".into()))
    })
}
