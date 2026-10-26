//! One existing native library/W owner across closed request phase plans.
//! Arithmetic/launch/retirement remain the parent's unchanged implementations.
use super::*;
use zerogravity_core::accelerator::program::SourceCallMode;

pub(crate) struct NativePlanPhase<'a> {
    pub context: NativePlanContext,
    pub plan: &'a ForwardPlan,
    pub values: &'a [DeviceBuffer],
}

fn require_phase_list(invocations: &[&SourceCall]) -> Result<()> {
    let expected: &[(SourceCallMode, usize, usize)] = match invocations {
        [_, _] => &[
            (SourceCallMode::Prefill, 0, 1),
            (SourceCallMode::Decode, 1, 1),
        ],
        [_, _, _] => &[
            (SourceCallMode::Prefill, 0, 64),
            (SourceCallMode::Prefill, 64, 2),
            (SourceCallMode::Decode, 66, 1),
        ],
        _ => {
            return Err(Error::Unsupported(
                "two closed native request phase lists".into(),
            ));
        }
    };
    let source = invocations[0].source();
    for (invocation, &(mode, base, rows)) in invocations.iter().zip(expected) {
        if invocation.source() != source
            || invocation.mode() != mode
            || invocation.rows().base() != base
            || invocation.rows().len() != rows
            || invocation.rows().capacity() != 512
            || invocation.rows().vocabulary() != 248320
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
fn same_learned_weight(a: &SourceOperator, b: &SourceOperator) -> bool {
    a.call().rows().source() == b.call().rows().source()
        && a.numerical_contract() == b.numerical_contract()
        && a.kind() == b.kind()
        && a.matrices() == b.matrices()
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
                .map(|p| p.context.bindings.call())
                .collect::<Vec<_>>(),
        )?;
        let first = &phases[0].context;
        first.images.activation()?;
        for phase in &phases {
            if !Arc::ptr_eq(&first.service, &phase.context.service)
                || !first
                    .bindings
                    .source()
                    .same_issuer(phase.context.bindings.source())
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
                key.clone(),
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
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::{SourceRows, TextCallGeometry};
    fn call(source: SourceIdentity, mode: SourceCallMode, base: usize, rows: usize) -> SourceCall {
        SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, 248320, 512).unwrap(),
                base,
                vec![11; rows].into(),
            )
            .unwrap(),
            mode,
        )
        .unwrap()
    }
    fn require_calls(calls: &[&SourceCall]) -> Result<()> {
        require_phase_list(calls)
    }
    #[test]
    fn exact_ordered_phase_list_accepts_real_span_shapes_and_refuses_wrong_join() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let one = call(source, SourceCallMode::Prefill, 0, 1);
        let cached = call(source, SourceCallMode::Decode, 1, 1);
        let sixty_four = call(source, SourceCallMode::Prefill, 0, 64);
        let continued = call(source, SourceCallMode::Prefill, 64, 2);
        let cached66 = call(source, SourceCallMode::Decode, 66, 1);
        assert!(require_calls(&[&one, &cached]).is_ok());
        assert!(require_calls(&[&sixty_four, &continued, &cached66]).is_ok());
        assert!(require_calls(&[&one, &cached66]).is_err());
        assert!(require_calls(&[&cached, &one]).is_err());
        assert!(require_calls(&[&sixty_four, &cached66]).is_err());
        let foreign = call(
            SourceIdentity::for_new_allocation().unwrap(),
            SourceCallMode::Decode,
            1,
            1,
        );
        assert!(require_calls(&[&one, &foreign]).is_err());
    }
    #[test]
    fn learned_weight_reuse_is_source_layer_role_bound_and_request_independent() {
        use zerogravity_core::accelerator::program::MatrixAxes;
        use zerogravity_core::quant::QuantKind;
        let source = SourceIdentity::for_new_allocation().unwrap();
        let first = call(source, SourceCallMode::Prefill, 0, 64);
        let next = call(source, SourceCallMode::Prefill, 64, 2);
        let out = |invocation: SourceCall, layer: usize| {
            let source = invocation.source();
            SourceOperator::new(invocation,SourceOperatorKind::Projection{selected_row:None},zerogravity_core::accelerator::program::SourceNumericalContract::ReferenceCudaSm86Bf16V1,
                vec![SourceTensor{source,name:format!("layer{layer}.output"),kind:QuantKind::Bf16,axes:MatrixAxes::OutputInput,rows:1024,columns:2048}],vec![]).unwrap()
        };
        let a = out(first, 0);
        let b = out(next.clone(), 0);
        assert_ne!(a.call(), b.call());
        assert!(same_learned_weight(&a, &b));
        assert!(
            same_request(a.call(), b.call()).is_err(),
            "producer request identity stays exact"
        );
        assert!(!same_learned_weight(&a, &out(next, 1)));
        let other_source = call(
            SourceIdentity::for_new_allocation().unwrap(),
            SourceCallMode::Prefill,
            0,
            64,
        );
        assert!(!same_learned_weight(&a, &out(other_source, 0)));
    }
}
