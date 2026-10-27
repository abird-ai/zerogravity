//! Cold model-to-engine data join; no kernel launch, root or publication authority.
use crate::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
#[cfg(feature = "qwen35-cuda-qualification")]
use crate::gpu::cuda::gdn::TextGdnImplementation;
use crate::gpu::{
    ComponentOutput as EngineComponentOutput, ComponentRecipe, SourceBindings, SourceSelection,
    StateLayout, StateRequirement,
};
use crate::qwen35::StateSpec;
use crate::qwen35::source::binding::{
    ComponentOutput, OriginalWindowModelSource, PreparedComponent, PreparedPlan,
    PreparedQualificationSource,
};
use crate::{Error, Result};
use std::sync::Arc;
use zerogravity_artifacts::source_weights::WindowWorkspace;

pub(super) fn retained(
    prepared: PreparedQualificationSource,
) -> Arc<crate::gpu::cuda::native::RetainedSource> {
    let actual = prepared.source().clone();
    let owner = prepared.owner().clone();
    crate::gpu::cuda::native::RetainedSource::new(actual, owner, Arc::new(prepared))
}

/// Preserve the original source issuer and complete model state geometry, including
/// domains carried without arithmetic in this call. The engine checks the actual graph.
pub(super) fn prepare(
    source: &OriginalWindowModelSource,
    prepared: PreparedPlan,
    record_admission: Option<Arc<WindowWorkspace>>,
) -> Result<Arc<SourceBindings>> {
    if !source.source().same_issuer(prepared.source()) {
        return Err(Error::State("prepared model source owner differs".into()));
    }
    let count = state_count(source.state_specs())?;
    if count != prepared.plan().required_domains().len() {
        return Err(Error::State(
            "prepared complete model state set differs".into(),
        ));
    }
    // Admit the cloned selectors/state vector before allocating either. The
    // consumed graph keeps its original model preparation charge as well.
    let metadata = prepared
        .source()
        .reserve_workspace(SourceBindings::metadata_bytes(
            prepared.plan(),
            prepared.boundaries().len(),
            count,
        )?)?;
    let selection = SourceSelection::new(
        prepared.output(),
        prepared.residual_output(),
        prepared.normalized(),
        prepared.boundaries().to_vec().into_boxed_slice(),
        prepared.selected_row(),
        prepared.completed_layers(),
        prepared.first_mlp_down_node(),
    );
    let states = state_requirements(source.state_specs(), prepared.plan().prefix_capacity())?;
    let (actual_source, plan, _, _) = prepared.into_parts();
    let bindings = SourceBindings::new(actual_source, plan, selection, states, metadata)?;
    let bindings = match record_admission {
        Some(ticket) => bindings.retain_source_record_admission(ticket)?,
        None => bindings,
    };
    Ok(Arc::new(bindings))
}

/// Require the sealed implementation selected by explicit qualification before
/// any source record or workspace is prepared.
#[cfg(feature = "qwen35-cuda-qualification")]
pub(super) fn gdn_implementation(
    implementation: Option<TextGdnImplementation>,
    input_tokens: usize,
    is_decode: bool,
) -> Result<TextGdnImplementation> {
    let implementation = implementation.ok_or_else(|| {
        Error::Unsupported("CUDA GDN backend has no implementation for requested call".into())
    })?;
    if implementation.input_tokens() != input_tokens || implementation.is_decode() != is_decode {
        return Err(Error::Unsupported(
            "CUDA GDN implementation geometry differs from requested call".into(),
        ));
    }
    if is_decode != implementation.config().is_none() {
        return Err(Error::Unsupported(
            "CUDA GDN implementation has missing or mismatched explicit config".into(),
        ));
    }
    Ok(implementation)
}

/// Move the same source-record ticket into the graph owner before RMS or upload.
pub(super) fn component(
    source: &OriginalWindowModelSource,
    prepared: PreparedComponent,
) -> Result<Arc<ComponentRecipe>> {
    let output = match prepared.output() {
        ComponentOutput::Gated => EngineComponentOutput::Gated,
        ComponentOutput::PostAttention => EngineComponentOutput::PostAttention,
        ComponentOutput::Mlp => EngineComponentOutput::Mlp,
    };
    let (prepared, capture, admission) = prepared.into_parts();
    let bindings = prepare(source, prepared, admission.clone())?;
    ComponentRecipe::bind(
        bindings,
        output,
        capture.recipe_sha256(),
        capture.explicit_owned(),
        admission,
    )
}

fn state_count(specs: &[StateSpec]) -> Result<usize> {
    specs.iter().try_fold(0usize, |count, spec| {
        count
            .checked_add(match spec {
                StateSpec::Delta { .. } => 2,
                StateSpec::Full { .. } => 1,
            })
            .ok_or_else(|| Error::Limit("model state-domain count overflow".into()))
    })
}
fn state_bytes(elements: usize, planes: usize) -> Result<usize> {
    elements
        .checked_mul(planes)
        .and_then(|n| n.checked_mul(4))
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| Error::Limit("model F32 state storage extent".into()))
}
fn state_requirements(specs: &[StateSpec], capacity: usize) -> Result<Vec<StateRequirement>> {
    let mut states = Vec::with_capacity(state_count(specs)?);
    for (layer, spec) in specs.iter().enumerate() {
        match *spec {
            StateSpec::Delta { conv, state } => {
                for (slot, elements) in [
                    (StateSlot::Conv(ConvDomain(layer)), conv),
                    (StateSlot::Recurrent(RecurrentDomain(layer)), state),
                ] {
                    states.push(StateRequirement {
                        slot,
                        layout: StateLayout::Vector { elements },
                        bytes: state_bytes(elements, 1)?,
                    });
                }
            }
            StateSpec::Full { kv_width } => {
                let elements = kv_width
                    .checked_mul(capacity)
                    .ok_or_else(|| Error::Limit("model KV plane extent overflow".into()))?;
                states.push(StateRequirement {
                    slot: StateSlot::Kv(KvDomain(layer)),
                    layout: StateLayout::Kv {
                        slab: kv_width,
                        capacity,
                    },
                    bytes: state_bytes(elements, 2)?,
                });
            }
        }
    }
    states.sort_unstable_by_key(|state| state.slot);
    Ok(states)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_state_specs_keep_carried_domains_and_kv_planes() -> Result<()> {
        let states = state_requirements(
            &[
                StateSpec::Delta { conv: 6, state: 8 },
                StateSpec::Full { kv_width: 3 },
                StateSpec::Delta { conv: 9, state: 12 },
            ],
            7,
        )?;
        assert_eq!(states.len(), 5);
        assert_eq!(
            states.iter().map(|s| s.slot).collect::<Vec<_>>(),
            [
                StateSlot::Conv(ConvDomain(0)),
                StateSlot::Conv(ConvDomain(2)),
                StateSlot::Recurrent(RecurrentDomain(0)),
                StateSlot::Recurrent(RecurrentDomain(2)),
                StateSlot::Kv(KvDomain(1)),
            ]
        );
        assert_eq!(
            states.iter().map(|s| s.bytes).collect::<Vec<_>>(),
            [24, 36, 32, 48, 168]
        );
        assert!(matches!(
            states[4].layout,
            StateLayout::Kv {
                slab: 3,
                capacity: 7
            }
        ));
        Ok(())
    }

    #[test]
    fn state_storage_rejects_zero_and_overflow() {
        for spec in [
            StateSpec::Delta { conv: 0, state: 8 },
            StateSpec::Delta {
                conv: 1,
                state: usize::MAX,
            },
            StateSpec::Full {
                kv_width: usize::MAX,
            },
            StateSpec::Full { kv_width: 0 },
        ] {
            assert!(state_requirements(&[spec], 7).is_err());
        }
        assert!(state_requirements(&[StateSpec::Full { kv_width: 3 }], 0).is_err());
    }
}
