//! Source-owned GDN role checks and independent recorded-reference pins.
//! Native source/ABI/catalogues and all physical authorities remain backend-owned.
use super::*;
use crate::plan::{PromptCall, QwenOperation, QwenOperator, QwenSourceProfile};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnKernelCapture {
    recipe_sha256: &'static str,
    input_tokens: usize,
    is_decode: bool,
}
impl GdnKernelCapture {
    pub fn recipe_sha256(self) -> &'static str {
        self.recipe_sha256
    }
    pub fn input_tokens(self) -> usize {
        self.input_tokens
    }
    pub fn is_decode(self) -> bool {
        self.is_decode
    }
    pub fn contract(self) -> SourceGdnContract {
        SourceGdnContract::ReferenceCudaSm86L2SiluBf16V1
    }
    pub fn for_call(call: &PromptCall) -> Result<Self> {
        call.validate()?;
        let is_decode = call.span().mode() == SourceCallMode::Decode;
        let input_tokens = if is_decode {
            1
        } else {
            call.span().workload().prompt_tokens()
        };
        let start = match call.model_profile() {
            QwenSourceProfile::Qwen08 => 0,
            QwenSourceProfile::Qwen2B => 6,
        };
        EXPECTED[start..start + 6]
            .iter()
            .copied()
            .find(|p| p.input_tokens == input_tokens && p.is_decode == is_decode)
            .ok_or_else(|| {
                Error::Unsupported("actual original text source tuple not present".into())
            })
    }
}
// Fingerprints cover ordered stage/entry/source SHA/PTX/launch tuples. Previous
// selector: cuda::gdn::text::observed; native catalogue remains backend-owned.
const EXPECTED: &[GdnKernelCapture] = &[
    GdnKernelCapture {
        recipe_sha256: "b24e5fb2ac4b0e591cb907e8226cd1d80bc24507c0586ecfb5238ea4f3c91f8b",
        input_tokens: 47,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "9ec9c52e82e99b9654d3136f22e10523e1fa6f786a6136f7adefaaedec2aed43",
        input_tokens: 57,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "f487c0f5bcb5580da48fd33f047c4cf67d4098e496e1040e07a4eeafca0c5a65",
        input_tokens: 62,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "213a13cfb815791936bca556a1ab66730b03ac71fe9c72f6586cc5fcab0be3e5",
        input_tokens: 67,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "d2226726eda06c919709e3b7767d019e0dd33f633db925ce266a9f67b48f1d9e",
        input_tokens: 72,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "cf3ef9d3fa84410540c6d969212a3e29c912255d9b404d2430905155c1cf201e",
        input_tokens: 1,
        is_decode: true,
    },
    GdnKernelCapture {
        recipe_sha256: "aaae809312359b22c8404b6ea5ba18073738db731e3df64192372c7f40d82cb6",
        input_tokens: 47,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "ab5aef404095ded631a2267be095f51b8a4d239a03289d354f3c7b26a1991741",
        input_tokens: 57,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "b95983abb04db60da606ffe7ecccc547062cf714dc8b11361cd829be41fd452b",
        input_tokens: 62,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "484de0c7dd651e8646758103db6634e8c34750a44c0d03f4d291deb7c93191df",
        input_tokens: 67,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "f79c878120e21e36ed8413c5e24f73f159eb4e8a45aaba90e2c4a120cdb30df1",
        input_tokens: 72,
        is_decode: false,
    },
    GdnKernelCapture {
        recipe_sha256: "18faad78538d9d190c5a31668710f7b3aced6e2dd5d922618108b4044b3355d8",
        input_tokens: 1,
        is_decode: true,
    },
];

/// Family-authenticated immutable data; no second plan, permit or device owner.
pub struct PreparedGdnParameters {
    nodes: [usize; 2],
    operators: [SourceOperator; 2],
    parameters: [SourceParameter; 4],
    capture: Option<GdnKernelCapture>,
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
    pub fn capture(&self) -> Option<GdnKernelCapture> {
        self.capture
    }
    pub fn site(&self) -> usize {
        self.site
    }
    pub(super) fn prepare(
        source: &OriginalWindowModelSource,
        conv: &QwenOperator,
        recurrent: &QwenOperator,
        nodes: [usize; 2],
        capture: Option<GdnKernelCapture>,
        plan: &ForwardPlan,
    ) -> Result<Self> {
        if let Some(expected) = capture {
            let span = conv
                .call()
                .request()
                .teachers()
                .prompt_span()
                .ok_or_else(|| {
                    Error::State("text GDN genuine source/profile/call issuer".into())
                })?;
            let call = PromptCall::new(conv.call().model_profile(), span.clone())?;
            if expected != GdnKernelCapture::for_call(&call)?
                || recurrent
                    .call()
                    .request()
                    .teachers()
                    .prompt_span()
                    .is_none()
            {
                return Err(Error::State(
                    "text GDN genuine source/profile/call issuer".into(),
                ));
            }
        }
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
            capture,
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
    capture: Option<GdnKernelCapture>,
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
            capture,
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
            None,
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
