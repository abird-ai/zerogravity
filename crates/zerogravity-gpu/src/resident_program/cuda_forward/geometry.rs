//! Derive implementation dimensions from typed operations, never model identities.
use crate::cuda::{gdn::head_tile::HeadGroups, text_forward::Geometry};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        ForwardPlan, GdnContract, PlanOperation, SourceGamma, SourceOperatorKind,
    },
};

fn consistent<T: Copy + PartialEq>(slot: &mut Option<T>, value: T) -> Result<()> {
    if slot.replace(value).is_some_and(|prior| prior != value) {
        return Err(Error::Unsupported(
            "CUDA forward prepared operator geometry is heterogeneous".into(),
        ));
    }
    Ok(())
}

pub(super) fn derive(plan: &ForwardPlan) -> Result<(Geometry, HeadGroups)> {
    let mut hidden = None;
    let mut intermediate = None;
    let mut vocabulary = None;
    let mut attention = None;
    let mut rotary = None;
    let mut conv = None;
    let mut groups = None;
    for node in plan.nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            return Err(Error::Unsupported(
                "CUDA forward requires source templates".into(),
            ));
        };
        consistent(&mut vocabulary, descriptor.vocabulary())?;
        match descriptor.kind() {
            SourceOperatorKind::Rms(g) | SourceOperatorKind::ResidualRms(g) => {
                if g.gamma != SourceGamma::OnePlus {
                    return Err(Error::Unsupported(
                        "CUDA forward RMS gamma convention".into(),
                    ));
                }
                consistent(&mut hidden, g.width)?;
            }
            SourceOperatorKind::SiluAndMul { width } => consistent(&mut intermediate, width)?,
            SourceOperatorKind::QkRmsRotaryGated(g) => {
                if g.rotary_dimensions != 64
                    || g.theta.to_bits() != 10_000_000f32.to_bits()
                    || g.epsilon.to_bits() != 1e-6f32.to_bits()
                {
                    return Err(Error::Unsupported(
                        "CUDA forward rotary numerical contract".into(),
                    ));
                }
                consistent(&mut attention, g.attention)?;
                consistent(&mut rotary, g.rotary_dimensions)?;
            }
            SourceOperatorKind::FlashAttention {
                geometry, scale, ..
            } => {
                if scale.to_bits() != 0.0625f32.to_bits() {
                    return Err(Error::Unsupported("CUDA forward attention scale".into()));
                }
                consistent(&mut attention, geometry)?;
            }
            SourceOperatorKind::CausalConvSplit { geometry, .. } => {
                if geometry.kernel != 4 {
                    return Err(Error::Unsupported(
                        "CUDA forward convolution kernel width".into(),
                    ));
                }
                consistent(&mut conv, geometry)?;
            }
            SourceOperatorKind::GatedDelta { geometry, .. } => {
                if geometry.contract != GdnContract::ReferenceCudaSm86L2SiluBf16V1
                    || geometry.key_width != 128
                    || geometry.value_width != 128
                    || geometry.epsilon.to_bits() != 1e-6f32.to_bits()
                {
                    return Err(Error::Unsupported(
                        "CUDA forward GDN kernel geometry".into(),
                    ));
                }
                consistent(
                    &mut groups,
                    HeadGroups::new(geometry.key_heads, geometry.value_heads)?,
                )?;
            }
            _ => {}
        }
    }
    let missing = || Error::State("CUDA forward operator geometry absent".into());
    let attention = attention.ok_or_else(missing)?;
    let groups = groups.ok_or_else(missing)?;
    let conv = conv.ok_or_else(missing)?;
    if conv.qkv_channels != groups.channels()
        || conv.z_channels != groups.value * 128
        || conv.coefficient_channels != groups.value * 2
    {
        return Err(Error::State(
            "CUDA forward convolution/GDN geometry differs".into(),
        ));
    }
    let text = Geometry {
        hidden_width: hidden.ok_or_else(missing)?,
        intermediate_width: intermediate.ok_or_else(missing)?,
        vocabulary: vocabulary.ok_or_else(missing)?,
        query_heads: attention.query_heads,
        kv_heads: attention.kv_heads,
        head_width: attention.width,
        rotary_width: rotary.ok_or_else(missing)?,
    }
    .validate()?;
    Ok((text, groups))
}
