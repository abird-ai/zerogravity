use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{
    ConvDomain, MathematicalBackend, MatrixAxes, PlanBuilder, SourceNumericalContract,
    SourceOperatorTemplate, SourcePlanGeometry, SourceRowSelection, StateSlot,
};
use zerogravity_core::quant::QuantKind;
use zerogravity_model_qwen35::plan::QwenPlanBuilder;
use zerogravity_model_qwen35::plan::{
    GdnProjection, QwenEmbedding, QwenGdnCall, QwenGdnProjection, QwenInputRows, QwenPrefixLayer,
    QwenSiluAndMul, ReferencePhase,
};

fn invocation(layer: usize, phase: ReferencePhase) -> QwenGdnCall {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let base = match phase {
        ReferencePhase::InitialOne | ReferencePhase::Initial64 => 0,
        ReferencePhase::CachedOne => 66,
        ReferencePhase::ContinuedTwo => 64,
    };
    QwenGdnCall::new(
        QwenPrefixLayer::new(source, layer).unwrap(),
        phase,
        QwenInputRows::new(source, base, 512, &vec![11; phase.rows()]).unwrap(),
    )
    .unwrap()
}
fn descriptor(call: &QwenGdnCall, kind: GdnProjection) -> QwenGdnProjection {
    use GdnProjection::{Ba, Down, GateUp, GdnOut, Qkvz};
    let parts: &[(&str, usize)] = match kind {
        Qkvz => &[
            ("linear_attn.in_proj_qkv.weight", 6144),
            ("linear_attn.in_proj_z.weight", 2048),
        ],
        Ba => &[
            ("linear_attn.in_proj_b.weight", 16),
            ("linear_attn.in_proj_a.weight", 16),
        ],
        GdnOut => &[("linear_attn.out_proj.weight", 1024)],
        GateUp => &[("mlp.gate_proj.weight", 3584), ("mlp.up_proj.weight", 3584)],
        Down => &[("mlp.down_proj.weight", 1024)],
    };
    let members = parts
        .iter()
        .map(|&(suffix, rows)| SourceTensor {
            source: call.layer().source(),
            name: format!(
                "model.language_model.layers.{}.{suffix}",
                call.layer().index()
            ),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            rows,
            columns: kind.input_width(),
        })
        .collect();
    QwenGdnProjection::new(call.clone(), kind, members).unwrap()
}

fn neutral(d: &QwenGdnProjection) -> SourceOperator {
    SourceOperator::new(
        d.invocation().source_call().unwrap(),
        SourceOperatorKind::Projection { selected_row: None },
        zerogravity_core::accelerator::program::SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        d.members().to_vec(),
        vec![],
    )
    .unwrap()
}

#[test]
fn captured_native_image_selection_refuses_call_free_templates() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let geometry = SourcePlanGeometry::new(source, 16, 512, 1).unwrap();
    let matrix = |name: &str, rows, columns| SourceTensor {
        source,
        name: name.into(),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        columns,
        rows,
    };
    let slot = StateSlot::Conv(ConvDomain(0));
    let mut builder = PlanBuilder::new_source_template(geometry, vec![slot]).unwrap();
    let embedding = SourceOperatorTemplate::new(
        geometry,
        SourceRowSelection::AllActive,
        SourceOperatorKind::EmbeddingRows,
        SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        vec![matrix("embedding", 16, 4)],
        vec![],
    )
    .unwrap();
    let [input] = builder
        .record_source_template("embedding", &embedding, &[])
        .unwrap()
        .try_into()
        .unwrap();
    let carry = SourceOperatorTemplate::new(
        geometry,
        SourceRowSelection::AllActive,
        SourceOperatorKind::StateCarry { slot },
        SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        vec![],
        vec![],
    )
    .unwrap();
    assert!(
        builder
            .record_source_template("state", &carry, &[])
            .unwrap()
            .is_empty()
    );
    let head = SourceOperatorTemplate::new(
        geometry,
        SourceRowSelection::LastActive,
        SourceOperatorKind::Projection { selected_row: None },
        SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        vec![matrix("head", 8, 4)],
        vec![],
    )
    .unwrap();
    let [output] = builder
        .record_source_template("head", &head, &[input])
        .unwrap()
        .try_into()
        .unwrap();
    let plan = builder.finish_source_template(output).unwrap();
    assert!(matches!(
        projection_images(&plan),
        Err(Error::Unsupported(message))
            if message.contains("call-free source templates")
    ));
}

#[test]
fn existing_manifest_physical_extents_match_every_closed_source_join() {
    for phase in [
        ReferencePhase::InitialOne,
        ReferencePhase::CachedOne,
        ReferencePhase::Initial64,
        ReferencePhase::ContinuedTwo,
    ] {
        for layer in 0..3 {
            let call = invocation(layer, phase);
            for kind in [
                GdnProjection::Qkvz,
                GdnProjection::Ba,
                GdnProjection::GdnOut,
                GdnProjection::GateUp,
                GdnProjection::Down,
            ] {
                let d = descriptor(&call, kind);
                let (weights, input, output) = match projection_operation(&neutral(&d)).unwrap() {
                    NativeOperation::Projection(p) => {
                        (p.weight_bytes(), p.input_bytes(), p.output_bytes())
                    }
                    NativeOperation::Mlp(p) => {
                        (p.weight_bytes(), p.input_bytes(), p.output_bytes())
                    }
                    NativeOperation::Activation(_) => panic!("projection became ACT"),
                };
                assert_eq!(
                    weights,
                    d.members()
                        .iter()
                        .map(|m| m.rows * m.columns * 2)
                        .sum::<usize>()
                );
                assert_eq!(input, d.input_elements() * 2);
                assert_eq!(output, d.output_elements() * 2);
            }
            let NativeOperation::Activation(a) = activation_operation(call.rows()).unwrap() else {
                panic!("ACT became projection")
            };
            let d = QwenSiluAndMul::new(call).unwrap();
            assert_eq!(a.input_bytes(), d.input_elements() * 2);
            assert_eq!(a.output_bytes(), d.output_elements() * 2);
        }
    }
    for rows in [0, 3, 63, 65, usize::MAX] {
        assert!(activation_operation(rows).is_err());
    }
}

#[test]
fn source_layer_weights_stay_distinct_while_actual_kernel_records_share_one_set() {
    let mut projections = BTreeSet::new();
    let mut mlp = BTreeSet::new();
    let mut activations = BTreeSet::new();
    let mut weights = BTreeSet::new();
    for phase in [
        ReferencePhase::InitialOne,
        ReferencePhase::CachedOne,
        ReferencePhase::Initial64,
        ReferencePhase::ContinuedTwo,
    ] {
        for layer in 0..3 {
            let call = invocation(layer, phase);
            for kind in [
                GdnProjection::Qkvz,
                GdnProjection::Ba,
                GdnProjection::GdnOut,
                GdnProjection::GateUp,
                GdnProjection::Down,
            ] {
                let d = descriptor(&call, kind);
                weights.insert(WeightKey::of(&neutral(&d)));
                match projection_operation(&neutral(&d)).unwrap() {
                    NativeOperation::Projection(p) => {
                        projections.insert(p);
                    }
                    NativeOperation::Mlp(p) => {
                        mlp.insert(p);
                    }
                    _ => panic!("projection became ACT"),
                }
            }
            let NativeOperation::Activation(a) = activation_operation(call.rows()).unwrap() else {
                unreachable!()
            };
            activations.insert(a);
        }
    }
    assert_eq!(weights.len(), 15);
    assert_eq!((projections.len(), mlp.len(), activations.len()), (9, 6, 3));
    let projection_records = projections
        .iter()
        .map(|p| p.launches().len())
        .sum::<usize>();
    let mlp_records = mlp
        .iter()
        .flat_map(|p| p.operations())
        .filter(|o| matches!(o, MlpOperation::Kernel(_)))
        .count();
    assert_eq!(
        (projection_records, mlp_records, activations.len()),
        (11, 7, 3)
    );
    assert_eq!(projection_records + mlp_records + activations.len(), 21);
}

#[test]
fn actual_workspace_and_ordered_clear_requirements_survive_operation_mapping() {
    for phase in [
        ReferencePhase::InitialOne,
        ReferencePhase::Initial64,
        ReferencePhase::ContinuedTwo,
    ] {
        let call = invocation(0, phase);
        let NativeOperation::Mlp(down) =
            projection_operation(&neutral(&descriptor(&call, GdnProjection::Down))).unwrap()
        else {
            unreachable!()
        };
        let operations = down.operations();
        let clears = operations
            .iter()
            .enumerate()
            .filter_map(|(i, op)| match op {
                MlpOperation::Clear(c) => Some((i, *c)),
                _ => None,
            })
            .collect::<Vec<_>>();
        match phase {
            ReferencePhase::InitialOne => assert!(clears.is_empty()),
            ReferencePhase::Initial64 | ReferencePhase::ContinuedTwo => {
                assert_eq!(clears.len(), 1);
                let (index, clear) = clears[0];
                assert_eq!(clear.bytes(), if phase.rows() == 64 { 32 } else { 64 });
                assert_eq!(clear.binding_bytes(), WORKSPACE_BYTES);
                assert!(matches!(
                    operations.get(index + 1),
                    Some(MlpOperation::Kernel(_))
                ));
            }
            _ => unreachable!(),
        }
        assert!(NativeOperation::Mlp(down).needs_workspace());
        assert!(
            !activation_operation(phase.rows())
                .unwrap()
                .needs_workspace()
        );
    }
    let call = invocation(0, ReferencePhase::InitialOne);
    assert!(
        !projection_operation(&neutral(&descriptor(&call, GdnProjection::Qkvz)))
            .unwrap()
            .needs_workspace()
    );
}

#[test]
fn equal_teacher_bytes_from_another_issuer_cannot_bind_existing_request() {
    let a = invocation(0, ReferencePhase::Initial64);
    let clone_other_layer = QwenGdnCall::new(
        QwenPrefixLayer::new(a.layer().source(), 2).unwrap(),
        a.reference_phase().unwrap(),
        a.teachers().clone(),
    )
    .unwrap();
    assert!(
        same_request(
            &clone_other_layer.source_call().unwrap(),
            &a.source_call().unwrap()
        )
        .is_ok()
    );
    let reminted = QwenGdnCall::new(
        a.layer(),
        a.reference_phase().unwrap(),
        QwenInputRows::new(a.layer().source(), 0, 512, a.teachers().ids()).unwrap(),
    )
    .unwrap();
    assert!(same_request(&reminted.source_call().unwrap(), &a.source_call().unwrap()).is_err());
    assert!(
        same_request(
            &invocation(0, ReferencePhase::Initial64)
                .source_call()
                .unwrap(),
            &a.source_call().unwrap()
        )
        .is_err()
    );
    assert!(
        same_request(
            &invocation(0, ReferencePhase::ContinuedTwo)
                .source_call()
                .unwrap(),
            &a.source_call().unwrap()
        )
        .is_err()
    );
}

#[test]
fn immutable_plan_value_join_refuses_generic_tag_loss_wrong_id_and_precision() {
    let call = invocation(0, ReferencePhase::InitialOne);
    let source = call.layer().source();
    let mut b =
        QwenPlanBuilder::new_operator_slice(source, ProgramNumericalProfile::SourceCudaSm86Bf16V1)
            .unwrap();
    let input = b
        .source_embedding(
            &QwenEmbedding::new(
                SourceTensor {
                    source,
                    name: "model.language_model.embed_tokens.weight".into(),
                    kind: QuantKind::Bf16,
                    axes: MatrixAxes::OutputInput,
                    rows: 248320,
                    columns: 1024,
                },
                call.teachers().clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let d = descriptor(&call, GdnProjection::GateUp);
    let output = b.gdn_project(&d, &input).unwrap();
    let packed = b.contiguous(&output).unwrap();
    let slice = b.finish_operator_slice(packed).unwrap();
    let plan = slice.plan();
    let input_id = plan.nodes()[0].outputs[0];
    let output_id = plan.nodes()[1].outputs[0];
    let packed_id = plan.nodes()[2].outputs[0];
    assert!(require_value(plan, input_id, &call.source_call().unwrap(), 1024).is_ok());
    assert!(require_value(plan, output_id, &call.source_call().unwrap(), 7168).is_ok());
    assert!(require_value(plan, packed_id, &call.source_call().unwrap(), 7168).is_err());
    assert!(require_value(plan, ValueId(u32::MAX), &call.source_call().unwrap(), 1024).is_err());
    assert!(require_value(plan, input_id, &call.source_call().unwrap(), 7168).is_err());
    assert!(
        require_value(
            plan,
            input_id,
            &invocation(0, ReferencePhase::InitialOne)
                .source_call()
                .unwrap(),
            1024
        )
        .is_err()
    );
    let mut legacy =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1).unwrap();
    let value = legacy
        .embed(
            SourceTensor {
                source,
                name: "raw".into(),
                kind: QuantKind::F32,
                axes: MatrixAxes::OutputInput,
                rows: 248320,
                columns: 1024,
            },
            PlanBuilder::inputs().token,
        )
        .unwrap();
    let legacy = legacy.finish_operator_slice(value).unwrap();
    assert!(
        require_value(
            legacy.plan(),
            legacy.output(),
            &call.source_call().unwrap(),
            1024
        )
        .is_err()
    );
}

#[test]
fn native_operation_cannot_silently_consume_multi_input_stateful_or_aliased_node() {
    let call = invocation(0, ReferencePhase::InitialOne);
    let mut node = PlanNode::new(
        "ACT".into(),
        PlanOperation::SourceOperator {
            descriptor: Box::new(SourceOperator::new(call.source_call().unwrap(),
                SourceOperatorKind::SiluAndMul { width: 3584 },
                zerogravity_core::accelerator::program::SourceNumericalContract::ReferenceCudaSm86Bf16V1,
                vec![], vec![]).unwrap()), state: None,
        },
        vec![ValueId(1)],
        vec![ValueId(2)],
        None,
    );
    assert_eq!(node_arity(&node).unwrap(), (ValueId(1), ValueId(2)));
    node.outputs = vec![ValueId(1)];
    assert!(node_arity(&node).is_err());
    node.outputs = vec![ValueId(2), ValueId(3)];
    assert!(node_arity(&node).is_err());
    node.outputs = vec![ValueId(2)];
    node.inputs.clear();
    assert!(node_arity(&node).is_err());
    node.inputs = vec![ValueId(1), ValueId(0)];
    assert!(node_arity(&node).is_err());
    node.inputs = vec![ValueId(1)];
    use zerogravity_core::accelerator::program::{
        ConvDomain, StateEffect, StateSlot, StateVersion,
    };
    let slot = StateSlot::Conv(ConvDomain(0));
    node.state = Some(StateEffect {
        read: StateVersion { slot, ordinal: 0 },
        write: StateVersion { slot, ordinal: 1 },
    });
    assert!(node_arity(&node).is_err());
}

#[test]
fn original_ledger_identity_cannot_be_bound_as_two_different_plan_values() {
    use zerogravity_core::device_events::DeviceEventLedger;
    let mut ledger = DeviceEventLedger::new(1, 128).unwrap();
    let a = ledger.admit(32, BufferRole::Activation).unwrap();
    let b = ledger.admit(32, BufferRole::Activation).unwrap();
    let mut originals = BTreeMap::new();
    bind_value_identity(&mut originals, ValueId(1), a).unwrap();
    bind_value_identity(&mut originals, ValueId(1), a).unwrap();
    let before = originals.clone();
    assert!(bind_value_identity(&mut originals, ValueId(2), a).is_err());
    assert_eq!(originals, before);
    bind_value_identity(&mut originals, ValueId(2), b).unwrap();
    assert_eq!(originals.len(), 2);
}

#[test]
fn default_image_factory_derives_exact_projection_set_and_refuses_activation_upgrade() {
    for activate in [false, true] {
        let call = invocation(0, ReferencePhase::InitialOne);
        let source = call.layer().source();
        let mut b = QwenPlanBuilder::new_operator_slice(
            source,
            ProgramNumericalProfile::SourceCudaSm86Bf16V1,
        )
        .unwrap();
        let input = b
            .source_embedding(
                &QwenEmbedding::new(
                    SourceTensor {
                        source,
                        name: "model.language_model.embed_tokens.weight".into(),
                        kind: QuantKind::Bf16,
                        axes: MatrixAxes::OutputInput,
                        rows: 248320,
                        columns: 1024,
                    },
                    call.teachers().clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let output = if activate {
            let gate = b
                .gdn_project(&descriptor(&call, GdnProjection::GateUp), &input)
                .unwrap();
            b.silu_and_mul(&QwenSiluAndMul::new(call).unwrap(), &gate)
                .unwrap()
        } else {
            b.gdn_project(&descriptor(&call, GdnProjection::Qkvz), &input)
                .unwrap();
            b.gdn_project(&descriptor(&call, GdnProjection::Ba), &input)
                .unwrap()
        };
        let plan = b.finish_operator_slice(output).unwrap();
        if activate {
            assert!(matches!(
                projection_images(plan.plan()),
                Err(Error::Unsupported(_))
            ));
        } else {
            let expected = [Projection::Qkvz1, Projection::Ba1]
                .into_iter()
                .flat_map(|p| p.launches())
                .map(|launch| launch.image())
                .collect::<BTreeSet<_>>();
            assert_eq!(projection_images(plan.plan()).unwrap(), expected);
        }
    }
}

#[test]
fn connected_image_requirement_requires_typed_activation_but_default_still_refuses() {
    let call = invocation(0, ReferencePhase::InitialOne);
    let mut b = QwenPlanBuilder::new_operator_slice(
        call.layer().source(),
        ProgramNumericalProfile::SourceCudaSm86Bf16V1,
    )
    .unwrap();
    let emb = QwenEmbedding::new(
        SourceTensor {
            source: call.layer().source(),
            name: "model.language_model.embed_tokens.weight".into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            rows: 248320,
            columns: 1024,
        },
        call.teachers().clone(),
    )
    .unwrap();
    let raw = b.source_embedding(&emb).unwrap();
    let gate = b
        .gdn_project(&descriptor(&call, GdnProjection::GateUp), &raw)
        .unwrap();
    let act = b
        .silu_and_mul(&QwenSiluAndMul::new(call).unwrap(), &gate)
        .unwrap();
    let record = b.finish_operator_slice(act).unwrap();
    let p = record.plan();
    assert!(projection_images(p).is_err());
    assert!(!projection_images_inner(p, true).unwrap().is_empty());
}

#[test]
fn native_selection_keeps_explicit_segment_order_rows_and_selected_row_limits() {
    let call = invocation(0, ReferencePhase::InitialOne);
    let valid = neutral(&descriptor(&call, GdnProjection::Qkvz));
    let mut reversed = valid.matrices().to_vec();
    reversed.reverse();
    let wrong_order = SourceOperator::new(
        valid.call().clone(),
        valid.kind(),
        valid.numerical_contract(),
        reversed,
        vec![],
    )
    .unwrap();
    assert!(projection_operation(&wrong_order).is_err());
    let selected = SourceOperator::new(
        valid.call().clone(),
        SourceOperatorKind::Projection {
            selected_row: Some(0),
        },
        valid.numerical_contract(),
        valid.matrices().to_vec(),
        vec![],
    )
    .unwrap();
    assert!(projection_operation(&selected).is_err());
    let rows = zerogravity_core::accelerator::program::SourceRows::new(
        zerogravity_core::accelerator::program::TextCallGeometry::new(
            call.layer().source(),
            248320,
            512,
        )
        .unwrap(),
        0,
        Arc::from([11, 12, 13]),
    )
    .unwrap();
    let three = SourceOperator::new(
        SourceCall::new(
            rows,
            zerogravity_core::accelerator::program::SourceCallMode::Prefill,
        )
        .unwrap(),
        valid.kind(),
        valid.numerical_contract(),
        valid.matrices().to_vec(),
        vec![],
    )
    .unwrap();
    assert!(projection_operation(&three).is_err());
    let changed = valid
        .matrices()
        .iter()
        .map(|m| SourceTensor {
            columns: 2048,
            ..m.clone()
        })
        .collect();
    let wrong_width = SourceOperator::new(
        valid.call().clone(),
        valid.kind(),
        valid.numerical_contract(),
        changed,
        vec![],
    )
    .unwrap();
    assert!(projection_operation(&wrong_width).is_err());
}
