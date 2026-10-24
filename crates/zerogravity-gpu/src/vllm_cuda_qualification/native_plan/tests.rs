use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{
    GdnPrefixLayer, MathematicalBackend, MatrixAxes, PlanBuilder, VllmBf16Embedding, VllmBf16Phase,
    VllmBf16Teachers, VllmSiluAndMul,
};
use zerogravity_core::quant::QuantKind;

fn invocation(layer: usize, phase: VllmBf16Phase) -> VllmBf16Invocation {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let base = match phase {
        VllmBf16Phase::InitialOne | VllmBf16Phase::Initial64 => 0,
        VllmBf16Phase::CachedOne => 66,
        VllmBf16Phase::ContinuedTwo => 64,
    };
    VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, layer).unwrap(),
        phase,
        VllmBf16Teachers::new(source, base, 512, &vec![11; phase.rows()]).unwrap(),
    )
    .unwrap()
}
fn descriptor(call: &VllmBf16Invocation, kind: VllmGdnProjectionKind) -> VllmGdnProjection {
    use VllmGdnProjectionKind::{Ba, Down, GateUp, GdnOut, Qkvz};
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
    VllmGdnProjection::new(call.clone(), kind, members).unwrap()
}

#[test]
fn existing_manifest_physical_extents_match_every_closed_source_join() {
    for phase in [
        VllmBf16Phase::InitialOne,
        VllmBf16Phase::CachedOne,
        VllmBf16Phase::Initial64,
        VllmBf16Phase::ContinuedTwo,
    ] {
        for layer in 0..3 {
            let call = invocation(layer, phase);
            for kind in [
                VllmGdnProjectionKind::Qkvz,
                VllmGdnProjectionKind::Ba,
                VllmGdnProjectionKind::GdnOut,
                VllmGdnProjectionKind::GateUp,
                VllmGdnProjectionKind::Down,
            ] {
                let d = descriptor(&call, kind);
                let (weights, input, output) = match projection_operation(&d).unwrap() {
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
            let d = VllmSiluAndMul::new(call).unwrap();
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
        VllmBf16Phase::InitialOne,
        VllmBf16Phase::CachedOne,
        VllmBf16Phase::Initial64,
        VllmBf16Phase::ContinuedTwo,
    ] {
        for layer in 0..3 {
            let call = invocation(layer, phase);
            for kind in [
                VllmGdnProjectionKind::Qkvz,
                VllmGdnProjectionKind::Ba,
                VllmGdnProjectionKind::GdnOut,
                VllmGdnProjectionKind::GateUp,
                VllmGdnProjectionKind::Down,
            ] {
                let d = descriptor(&call, kind);
                weights.insert(WeightKey::of(&d));
                match projection_operation(&d).unwrap() {
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
        VllmBf16Phase::InitialOne,
        VllmBf16Phase::Initial64,
        VllmBf16Phase::ContinuedTwo,
    ] {
        let call = invocation(0, phase);
        let NativeOperation::Mlp(down) =
            projection_operation(&descriptor(&call, VllmGdnProjectionKind::Down)).unwrap()
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
            VllmBf16Phase::InitialOne => assert!(clears.is_empty()),
            VllmBf16Phase::Initial64 | VllmBf16Phase::ContinuedTwo => {
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
    let call = invocation(0, VllmBf16Phase::InitialOne);
    assert!(
        !projection_operation(&descriptor(&call, VllmGdnProjectionKind::Qkvz))
            .unwrap()
            .needs_workspace()
    );
}

#[test]
fn equal_teacher_bytes_from_another_issuer_cannot_bind_existing_request() {
    let a = invocation(0, VllmBf16Phase::Initial64);
    let clone_other_layer = VllmBf16Invocation::new(
        GdnPrefixLayer::new(a.layer().source(), 2).unwrap(),
        a.legacy_phase().unwrap(),
        a.teachers().clone(),
    )
    .unwrap();
    assert!(same_request(&clone_other_layer, &a).is_ok());
    let reminted = VllmBf16Invocation::new(
        a.layer(),
        a.legacy_phase().unwrap(),
        VllmBf16Teachers::new(a.layer().source(), 0, 512, a.teachers().ids()).unwrap(),
    )
    .unwrap();
    assert!(same_request(&reminted, &a).is_err());
    assert!(same_request(&invocation(0, VllmBf16Phase::Initial64), &a).is_err());
    assert!(same_request(&invocation(0, VllmBf16Phase::ContinuedTwo), &a).is_err());
}

#[test]
fn immutable_plan_value_join_refuses_generic_tag_loss_wrong_id_and_precision() {
    let call = invocation(0, VllmBf16Phase::InitialOne);
    let source = call.layer().source();
    let mut b = PlanBuilder::new_operator_slice(
        source,
        ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
    )
    .unwrap();
    let input = b
        .vllm_bf16_source_embedding(
            &VllmBf16Embedding::new(
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
    let d = descriptor(&call, VllmGdnProjectionKind::GateUp);
    let output = b.vllm_gdn_project(&d, &input).unwrap();
    let packed = b.contiguous(&output).unwrap();
    let slice = b.finish_operator_slice(packed).unwrap();
    let plan = slice.plan();
    let input_id = plan.nodes()[0].outputs[0];
    let output_id = plan.nodes()[1].outputs[0];
    let packed_id = plan.nodes()[2].outputs[0];
    assert!(require_value(plan, input_id, &call, 1024).is_ok());
    assert!(require_value(plan, output_id, &call, 7168).is_ok());
    assert!(require_value(plan, packed_id, &call, 7168).is_err());
    assert!(require_value(plan, ValueId(u32::MAX), &call, 1024).is_err());
    assert!(require_value(plan, input_id, &call, 7168).is_err());
    assert!(
        require_value(
            plan,
            input_id,
            &invocation(0, VllmBf16Phase::InitialOne),
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
    assert!(require_value(legacy.plan(), legacy.output(), &call, 1024).is_err());
}

#[test]
fn native_operation_cannot_silently_consume_multi_input_stateful_or_aliased_node() {
    let call = invocation(0, VllmBf16Phase::InitialOne);
    let mut node = PlanNode::new(
        "ACT".into(),
        PlanOperation::TypedVllmSiluAndMul {
            descriptor: Box::new(VllmSiluAndMul::new(call).unwrap()),
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
        let call = invocation(0, VllmBf16Phase::InitialOne);
        let source = call.layer().source();
        let mut b = PlanBuilder::new_operator_slice(
            source,
            ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
        )
        .unwrap();
        let input = b
            .vllm_bf16_source_embedding(
                &VllmBf16Embedding::new(
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
                .vllm_gdn_project(&descriptor(&call, VllmGdnProjectionKind::GateUp), &input)
                .unwrap();
            b.vllm_silu_and_mul(&VllmSiluAndMul::new(call).unwrap(), &gate)
                .unwrap()
        } else {
            b.vllm_gdn_project(&descriptor(&call, VllmGdnProjectionKind::Qkvz), &input)
                .unwrap();
            b.vllm_gdn_project(&descriptor(&call, VllmGdnProjectionKind::Ba), &input)
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
    let call = invocation(0, VllmBf16Phase::InitialOne);
    let mut b = PlanBuilder::new_operator_slice(
        call.layer().source(),
        ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
    )
    .unwrap();
    let emb = VllmBf16Embedding::new(
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
    let raw = b.vllm_bf16_source_embedding(&emb).unwrap();
    let gate = b
        .vllm_gdn_project(&descriptor(&call, VllmGdnProjectionKind::GateUp), &raw)
        .unwrap();
    let act = b
        .vllm_silu_and_mul(&VllmSiluAndMul::new(call).unwrap(), &gate)
        .unwrap();
    let record = b.finish_operator_slice(act).unwrap();
    let p = record.plan();
    assert!(projection_images(p).is_err());
    assert!(!projection_images_inner(p, true).unwrap().is_empty());
}
