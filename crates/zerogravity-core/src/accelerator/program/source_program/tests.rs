use super::*;
const NUMERICAL: SourceNumericalContract = SourceNumericalContract::ReferenceCudaSm86Bf16V1;
fn call(source: SourceIdentity, ids: &[u32]) -> SourceCall {
    SourceCall::new(
        SourceRows::new(
            TextCallGeometry::new(source, 17, 8).unwrap(),
            0,
            Arc::from(ids),
        )
        .unwrap(),
        SourceCallMode::Prefill,
    )
    .unwrap()
}
fn embedding(call: &SourceCall) -> SourceOperator {
    SourceOperator::new(
        call.clone(),
        SourceOperatorKind::EmbeddingRows,
        NUMERICAL,
        vec![SourceTensor {
            source: call.source(),
            name: "actual.embedding".into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            columns: 16,
            rows: 17,
        }],
        vec![],
    )
    .unwrap()
}
fn projection(call: &SourceCall, selected_row: Option<usize>) -> SourceOperator {
    SourceOperator::new(
        call.clone(),
        SourceOperatorKind::Projection { selected_row },
        NUMERICAL,
        [3, 5]
            .into_iter()
            .enumerate()
            .map(|(index, rows)| SourceTensor {
                source: call.source(),
                name: format!("actual.member{index}"),
                kind: QuantKind::Bf16,
                axes: MatrixAxes::OutputInput,
                columns: 16,
                rows,
            })
            .collect(),
        vec![],
    )
    .unwrap()
}
#[test]
fn row_issuer_requires_original_extent_and_reuses_the_same_adopted_teacher() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let a = call(source, &[1, 2]);
    assert_eq!(a, a.clone());
    assert_ne!(a, call(source, &[1, 2]));
    let span = TeacherSpan::new(source, 0, 8, 17, &[1, 2]).unwrap();
    assert_eq!(
        SourceRows::from_teacher_span(span.clone(), 17).unwrap(),
        SourceRows::from_teacher_span(span.clone(), 17).unwrap()
    );
    assert_ne!(
        SourceRows::from_teacher_span(span.clone(), 17).unwrap(),
        SourceRows::from_teacher_span(span, 18).unwrap()
    );
    let geometry = TextCallGeometry::new(source, 17, 8).unwrap();
    for (base, ids) in [
        (0, vec![]),
        (8, vec![1]),
        (usize::MAX, vec![1]),
        (0, vec![17]),
    ] {
        assert!(SourceRows::new(geometry, base, Arc::from(ids)).is_err());
    }
    assert!(SourceCall::new(a.rows().clone(), SourceCallMode::Decode).is_err());
}
#[test]
fn ordered_projection_segments_and_selected_rows_are_explicit_checked_math() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = call(source, &[1, 2]);
    let all = projection(&call, None);
    assert_eq!(all.input_elements().unwrap(), [32]);
    assert_eq!(all.output_elements().unwrap(), [16]);
    assert_eq!(
        all.matrices().iter().map(|m| m.rows).collect::<Vec<_>>(),
        [3, 5]
    );
    assert_eq!(projection(&call, Some(1)).output_elements().unwrap(), [8]);
    assert!(
        SourceOperator::new(
            call.clone(),
            SourceOperatorKind::Projection {
                selected_row: Some(2)
            },
            NUMERICAL,
            all.matrices().to_vec(),
            vec![]
        )
        .is_err()
    );
    for fault in 0..4 {
        let mut matrices = all.matrices().to_vec();
        match fault {
            0 => matrices[1].source = SourceIdentity::for_new_allocation().unwrap(),
            1 => matrices[1].axes = MatrixAxes::InputOutput,
            2 => matrices[1].kind = QuantKind::F16,
            _ => matrices[1].columns = 32,
        }
        assert!(
            SourceOperator::new(call.clone(), all.kind(), NUMERICAL, matrices, vec![]).is_err()
        );
    }
}
#[test]
fn core_owns_call_ssa_precision_and_atomic_state_effects() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = call(source, &[1, 2]);
    let domain = StateSlot::Kv(KvDomain(7));
    let mut builder = PlanBuilder::new_source_program(call.clone(), vec![domain]).unwrap();
    let raw = builder.source_operator(&embedding(&call), &[]).unwrap()[0];
    let mut other = PlanBuilder::new_source_program(call.clone(), vec![domain]).unwrap();
    let foreign = other.source_operator(&embedding(&call), &[]).unwrap()[0];
    let before = (
        builder.nodes().len(),
        builder.values().len(),
        builder.proposed_domains().len(),
    );
    assert!(
        builder
            .source_operator(&projection(&call, None), &[foreign])
            .is_err()
    );
    let other_call = super::tests::call(source, &[1, 2]);
    assert!(
        builder
            .source_operator(&projection(&other_call, None), &[raw])
            .is_err()
    );
    assert_eq!(
        (
            builder.nodes().len(),
            builder.values().len(),
            builder.proposed_domains().len()
        ),
        before
    );
    let output = builder
        .source_operator(&projection(&call, Some(1)), &[raw])
        .unwrap()[0];
    assert_eq!(output.elements(), 8);
    assert_eq!(output.precision(), ActivationPrecision::Bf16WidenedF32);
    let tag = builder.source_value_provenance(&output).unwrap().unwrap();
    assert_eq!(tag.call(), &call);
    assert_eq!(tag.inputs(), [raw.id]);
    assert_eq!((tag.node_index(), tag.output_index()), (1, 0));
    let carry = SourceOperator::new(
        call.clone(),
        SourceOperatorKind::StateCarry { slot: domain },
        NUMERICAL,
        vec![],
        vec![],
    )
    .unwrap();
    builder.source_operator(&carry, &[]).unwrap();
    let before = builder.nodes().len();
    assert!(builder.source_operator(&carry, &[]).is_err());
    assert_eq!(builder.nodes().len(), before);
    let plan = builder.finish_source_program().unwrap();
    assert_eq!(
        plan.proposed_domains(),
        [StateVersion {
            slot: domain,
            ordinal: 1
        }]
    );
    assert!(plan.logits().is_none());
    assert_eq!(plan.frontier_delta(), 0);
}
#[test]
fn call_free_plan_binds_dynamic_rows_without_node_or_id_capture() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let geometry = SourcePlanGeometry::new(source, 17, 512, 64).unwrap();
    let slot = StateSlot::Kv(KvDomain(3));
    let matrix = SourceTensor {
        source,
        name: "actual.embedding".into(),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        columns: 16,
        rows: 17,
    };
    let mut builder = PlanBuilder::new_source_template(geometry, vec![slot]).unwrap();
    let raw = builder
        .record_source_template(
            "embedding",
            &SourceOperatorTemplate::new(
                geometry,
                SourceRowSelection::AllActive,
                SourceOperatorKind::EmbeddingRows,
                NUMERICAL,
                vec![matrix.clone()],
                vec![],
            )
            .unwrap(),
            &[],
        )
        .unwrap()[0];
    builder
        .record_source_template(
            "state",
            &SourceOperatorTemplate::new(
                geometry,
                SourceRowSelection::AllActive,
                SourceOperatorKind::StateCarry { slot },
                NUMERICAL,
                vec![],
                vec![],
            )
            .unwrap(),
            &[],
        )
        .unwrap();
    let logits = builder
        .record_source_template(
            "head",
            &SourceOperatorTemplate::new(
                geometry,
                SourceRowSelection::LastActive,
                SourceOperatorKind::Projection { selected_row: None },
                NUMERICAL,
                vec![matrix],
                vec![],
            )
            .unwrap(),
            &[raw],
        )
        .unwrap()[0];
    assert_eq!(logits.elements(), 17);
    let plan = Arc::new(builder.finish_source_template(logits).unwrap());
    assert!(plan.source_call().is_none());
    assert_eq!(plan.source_plan_geometry(), Some(geometry));
    assert_eq!(plan.required_domains(), [slot]);
    assert_eq!(plan.frontier_delta(), 0);
    assert!(
        plan.nodes()
            .iter()
            .all(|node| matches!(node.operation, PlanOperation::SourceOperatorTemplate { .. }))
    );

    let bind = |base, ids: &[u32], emit_logits| {
        SourceInvocation::new(
            TextCallGeometry::new(source, 17, 512).unwrap(),
            base,
            Arc::from(ids),
            emit_logits,
        )
        .and_then(|invocation| plan.bind_source_invocation(invocation))
    };
    let a = bind(0, &[1, 2], true).unwrap();
    let b = bind(29, &[3], false).unwrap();
    assert!(Arc::ptr_eq(a.plan(), b.plan()));
    assert_eq!((a.invocation().rows(), a.invocation().frontier()), (2, 2));
    assert_eq!((b.invocation().base(), b.invocation().frontier()), (29, 30));
    assert!(!b.invocation().emit_logits());
    assert!(std::ptr::eq(a.active_nodes(), plan.nodes()));
    assert!(std::ptr::eq(b.active_nodes(), plan.state_nodes()));
    assert_eq!(a.active_nodes().len(), b.active_nodes().len() + 1);

    assert!(bind(500, &[1; 13], true).is_err());
    assert!(bind(0, &[1; 65], true).is_err());
    assert!(bind(0, &[17], true).is_err());
    let foreign = SourceInvocation::new(
        TextCallGeometry::new(SourceIdentity::for_new_allocation().unwrap(), 17, 512).unwrap(),
        0,
        Arc::from([1]),
        true,
    )
    .unwrap();
    assert!(plan.bind_source_invocation(foreign).is_err());
}

#[test]
fn prepared_rebind_shares_topology_semantics_and_accounts_local_allocations() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    use crate::resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    };
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "source-rebinding-metadata").unwrap();
    arbiter.register(key.clone(), 32 * 1024).unwrap();
    let owner = Arc::new(vec![1u8, 2, 3]);
    let witness = Arc::downgrade(&owner);
    let charge = |bytes| {
        arbiter
            .admit(
                &key,
                ResourceOwner::new("source", "binding").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes,
                }],
            )
            .unwrap()
            .into_host_workspace(source, owner.clone())
            .unwrap()
    };
    let first = call(source, &[1, 2]);
    let slot = StateSlot::Kv(KvDomain(3));
    let mut builder = PlanBuilder::new_source_program(first.clone(), vec![slot]).unwrap();
    let raw = builder.source_operator(&embedding(&first), &[]).unwrap()[0];
    builder
        .source_operator(&projection(&first, Some(1)), &[raw])
        .unwrap();
    builder
        .source_operator(
            &SourceOperator::new(
                first.clone(),
                SourceOperatorKind::StateCarry { slot },
                NUMERICAL,
                vec![],
                vec![],
            )
            .unwrap(),
            &[],
        )
        .unwrap();
    let prepared = builder
        .finish_source_program()
        .unwrap()
        .retain_prepared_metadata(charge(8192))
        .unwrap()
        .retain_binding_metadata(charge(4096))
        .unwrap();
    let second = call(source, &[3, 4]);
    let bound = prepared
        .rebind_source_call(second.clone())
        .unwrap()
        .retain_binding_metadata(charge(4096))
        .unwrap();
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 16 * 1024);
    drop(owner);
    assert!(prepared.shares_semantics(&bound));
    // Vec::clone may compact spare capacity; charge each actual allocation.
    let local_vectors = |p: &ForwardPlan| {
        p.nodes.capacity() * std::mem::size_of::<PlanNode>()
            + p.source_provenance.capacity() * std::mem::size_of::<Option<SourceValueProvenance>>()
    };
    assert_eq!(
        prepared.source_binding_metadata_bytes().unwrap() - local_vectors(&prepared),
        bound.source_binding_metadata_bytes().unwrap() - local_vectors(&bound)
    );
    let mut reserved = bound.clone();
    let before = reserved.source_binding_metadata_bytes().unwrap();
    let vectors_before = local_vectors(&reserved);
    reserved.nodes.reserve(17);
    reserved.source_provenance.reserve(19);
    let growth = local_vectors(&reserved) - vectors_before;
    assert!(growth > 0);
    assert_eq!(
        reserved.source_binding_metadata_bytes().unwrap(),
        before + growth
    );
    for (a, b) in prepared.nodes().iter().zip(bound.nodes()) {
        assert!(a.shares_topology(b));
        let (
            PlanOperation::SourceOperator { descriptor: a, .. },
            PlanOperation::SourceOperator { descriptor: b, .. },
        ) = (&a.operation, &b.operation)
        else {
            panic!("foreign source operation")
        };
        assert!(a.shares_semantics(b));
        assert_eq!(a.call(), &first);
        assert_eq!(b.call(), &second);
    }
    assert!(bound.source_binding_metadata_bytes().unwrap() >= std::mem::size_of_val(bound.nodes()));
    for id in [ValueId(0), ValueId(1)] {
        assert_eq!(
            bound.source_value_provenance(id).unwrap().unwrap().call(),
            &second
        );
    }
    assert!(prepared.rebind_source_call(call(source, &[3])).is_err());
    assert!(
        prepared
            .rebind_source_call(call(SourceIdentity::for_new_allocation().unwrap(), &[3, 4]))
            .is_err()
    );
    drop(reserved);
    drop(prepared);
    assert!(witness.upgrade().is_some());
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 12 * 1024);
    assert_eq!(bound.prepared_metadata_bytes(), Some(8192));
    drop(bound);
    assert!(witness.upgrade().is_none());
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}
