//! One existing native library/W owner across closed request phase plans.
//! Arithmetic/launch/retirement remain the parent's unchanged implementations.
use super::*;
use zerogravity_core::accelerator::program::VllmBf16Phase;

pub(crate) struct NativePlanPhase<'a> {
    pub context: NativePlanContext,
    pub plan: &'a ForwardPlan,
    pub values: &'a [DeviceBuffer],
}

fn require_phase_list(invocations: &[&VllmBf16Invocation]) -> Result<()> {
    let expected: &[(VllmBf16Phase, usize)] = match invocations {
        [_, _] => &[
            (VllmBf16Phase::InitialOne, 0),
            (VllmBf16Phase::CachedOne, 1),
        ],
        [_, _, _] => &[
            (VllmBf16Phase::Initial64, 0),
            (VllmBf16Phase::ContinuedTwo, 64),
            (VllmBf16Phase::CachedOne, 66),
        ],
        _ => {
            return Err(Error::Unsupported(
                "two closed native request phase lists".into(),
            ));
        }
    };
    let source = invocations[0].layer().source();
    for (invocation, &(phase, base)) in invocations.iter().zip(expected) {
        invocation.validate()?;
        if invocation.layer().source() != source
            || invocation.layer().index() != 0
            || invocation.legacy_phase()? != phase
            || invocation.teachers().base() != base
            || invocation.teachers().capacity() != 512
        {
            return Err(Error::State(
                "native request source/phase/span differs".into(),
            ));
        }
    }
    Ok(())
}

/// Immutable learned W has source/layer/role identity, not teacher identity.
/// Each actual X/Y node separately retains its exact request provenance.
fn same_learned_weight(a: &VllmGdnProjection, b: &VllmGdnProjection) -> bool {
    a.invocation().layer() == b.invocation().layer()
        && a.kind() == b.kind()
        && a.members() == b.members()
}

impl NativePlanPrograms {
    /// All phase values/RMS/GDN/state/status allocations precede this call.
    /// Actual source W and one workspace are the final managed allocations;
    /// then one union of original closed factory records is prepared once.
    pub(crate) fn prepare_phases(
        phases: Vec<NativePlanPhase<'_>>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<Arc<Self>>> {
        not_cancelled(cancelled)?;
        if phases.capacity() > 3 {
            return Err(Error::Limit("native phase metadata capacity".into()));
        }
        require_phase_list(
            &phases
                .iter()
                .map(|p| &p.context.invocation)
                .collect::<Vec<_>>(),
        )?;
        let first = &phases[0].context;
        first.images.activation()?;
        for phase in &phases {
            if !Arc::ptr_eq(&first.service, &phase.context.service)
                || !Arc::ptr_eq(&first.source_owner, &phase.context.source_owner)
                || !Arc::ptr_eq(&first.images, &phase.context.images)
            {
                return Err(Error::State(
                    "one original native source/service/image set required".into(),
                ));
            }
        }
        // All original per-plan joins, metadata admission and producer leases
        // are validated BEFORE the first shared W source read/H2D.
        let mut bindings = Vec::with_capacity(phases.len());
        let mut earlier_values = BTreeSet::new();
        for phase in phases {
            let binding =
                prepare_bindings(phase.context, phase.plan, phase.values, init, cancelled)?;
            let current = binding
                .nodes
                .values()
                .flat_map(|n| [n.input.id(), n.output.id()])
                .collect::<BTreeSet<_>>();
            if current.iter().any(|id| earlier_values.contains(id)) {
                return Err(Error::State(
                    "distinct phase activation storage aliases".into(),
                ));
            }
            earlier_values.extend(current);
            bindings.push(binding);
        }
        let canonical = &bindings[0];
        if !matches!(canonical.descriptors.len(), 5 | 15) || canonical.activation_records.is_empty()
        {
            return Err(Error::Unsupported(
                "complete layer0/prefix native phase bindings only".into(),
            ));
        }
        for binding in &bindings[1..] {
            if binding.descriptors.len() != canonical.descriptors.len()
                || binding.activation_records.is_empty()
                || canonical.descriptors.iter().any(|(key, descriptor)| {
                    binding
                        .descriptors
                        .get(key)
                        .is_none_or(|actual| !same_learned_weight(descriptor, actual))
                })
            {
                return Err(Error::State(
                    "every phase must use the same actual learned source roles".into(),
                ));
            }
        }
        let mut weights = BTreeMap::new();
        for (key, descriptor) in &canonical.descriptors {
            weights.insert(
                *key,
                prepare_weight(&canonical.context, descriptor, init, cancelled)?,
            );
        }
        let workspace = if bindings.iter().any(|b| b.needs_workspace) {
            Some(init.allocate_empty(WORKSPACE_BYTES, BufferRole::Scratch)?)
        } else {
            None
        };
        let projection_records = bindings
            .iter()
            .flat_map(|b| b.projection_records.iter().copied())
            .collect::<BTreeSet<_>>();
        let mlp_records = bindings
            .iter()
            .flat_map(|b| b.mlp_records.iter().copied())
            .collect::<BTreeSet<_>>();
        let activation_records = bindings
            .iter()
            .flat_map(|b| b.activation_records.iter().copied())
            .collect::<BTreeSet<_>>();
        let images = canonical.context.images.clone();
        // SAME7/12/21 policy and exact factories. No managed allocation/upload
        // follows first module prepare; only admitted metadata/owner wrapping.
        let projections = Some(ProjectionPrograms::prepare(
            images.clone(),
            &projection_records.into_iter().collect::<Vec<_>>(),
            init,
            cancelled,
        )?);
        let mlp = Some(MlpPrograms::prepare(
            images.clone(),
            &mlp_records.into_iter().collect::<Vec<_>>(),
            init,
            cancelled,
        )?);
        let activations = Some(ActivationPrograms::prepare(
            images,
            &activation_records.into_iter().collect::<Vec<_>>(),
            init,
            cancelled,
        )?);
        let mut result = Vec::with_capacity(bindings.len());
        for binding in bindings {
            // Only small original-lease/Arc metadata clones; no payload copy,
            // duplicate W/window allocation or second library/module set.
            let prepared = Arc::new(Self {
                nodes: binding.nodes,
                weights: weights.clone(),
                workspace: workspace.clone(),
                projections: projections.clone(),
                mlp: mlp.clone(),
                activations: activations.clone(),
                context: binding.context,
                _metadata: binding.metadata,
            });
            init.retain_completion_owner(prepared.clone())?;
            result.push(prepared);
        }
        not_cancelled(cancelled)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::{
        SourceIdentity,
        program::{GdnPrefixLayer, VllmBf16Teachers},
    };
    fn call(source: SourceIdentity, phase: VllmBf16Phase, base: usize) -> VllmBf16Invocation {
        VllmBf16Invocation::new(
            GdnPrefixLayer::new(source, 0).unwrap(),
            phase,
            VllmBf16Teachers::new(source, base, 512, &vec![11; phase.rows()]).unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn exact_ordered_phase_list_accepts_real_span_shapes_and_refuses_wrong_join() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let one = call(source, VllmBf16Phase::InitialOne, 0);
        let cached = call(source, VllmBf16Phase::CachedOne, 1);
        let sixty_four = call(source, VllmBf16Phase::Initial64, 0);
        let continued = call(source, VllmBf16Phase::ContinuedTwo, 64);
        let cached66 = call(source, VllmBf16Phase::CachedOne, 66);
        assert!(require_phase_list(&[&one, &cached]).is_ok());
        assert!(require_phase_list(&[&sixty_four, &continued, &cached66]).is_ok());
        assert!(require_phase_list(&[&one, &cached66]).is_err());
        assert!(require_phase_list(&[&cached, &one]).is_err());
        assert!(require_phase_list(&[&sixty_four, &cached66]).is_err());
        let foreign = call(
            SourceIdentity::for_new_allocation().unwrap(),
            VllmBf16Phase::CachedOne,
            1,
        );
        assert!(require_phase_list(&[&one, &foreign]).is_err());
    }
    #[test]
    fn learned_weight_reuse_is_source_layer_role_bound_and_request_independent() {
        use zerogravity_core::accelerator::program::MatrixAxes;
        use zerogravity_core::quant::QuantKind;
        let source = SourceIdentity::for_new_allocation().unwrap();
        let first = call(source, VllmBf16Phase::Initial64, 0);
        let next = call(source, VllmBf16Phase::ContinuedTwo, 64);
        let out = |invocation: VllmBf16Invocation| {
            let layer = invocation.layer();
            VllmGdnProjection::new(
                invocation,
                VllmGdnProjectionKind::GdnOut,
                vec![SourceTensor {
                    source: layer.source(),
                    name: format!(
                        "model.language_model.layers.{}.linear_attn.out_proj.weight",
                        layer.index()
                    ),
                    kind: QuantKind::Bf16,
                    axes: MatrixAxes::OutputInput,
                    rows: 1024,
                    columns: 2048,
                }],
            )
            .unwrap()
        };
        let a = out(first);
        let b = out(next.clone());
        assert_ne!(a.invocation(), b.invocation());
        assert!(same_learned_weight(&a, &b));
        assert!(
            same_request(a.invocation(), b.invocation()).is_err(),
            "producer request identity stays exact"
        );
        let other_layer = VllmBf16Invocation::new(
            GdnPrefixLayer::new(source, 1).unwrap(),
            next.legacy_phase().unwrap(),
            next.teachers().clone(),
        )
        .unwrap();
        assert!(!same_learned_weight(&a, &out(other_layer)));
        let other_source = call(
            SourceIdentity::for_new_allocation().unwrap(),
            VllmBf16Phase::Initial64,
            0,
        );
        assert!(!same_learned_weight(&a, &out(other_source)));
    }
}
