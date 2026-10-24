//! Metadata fixtures only; no source contents, device, Ready or inference claim.
use super::*;
use crate::resident_program::qwen2b_component::CANONICAL47_PROMPT;
use zerogravity_core::{
    accelerator::{
        SourceIdentity,
        program::{
            GdnPrefixLayer, MatrixAxes, SourceParameter, SourceTensor, TeacherSpan, Vllm08Call,
            Vllm08Operator, Vllm08RmsSite, VllmBf16Invocation, VllmBf16Mode, VllmBf16Teachers,
        },
    },
    quant::QuantKind,
};

fn extension() -> Vec<PlanNode> {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let teachers = VllmBf16Teachers::from_span(
        TeacherSpan::new(source, 0, 512, 248320, &CANONICAL47_PROMPT).unwrap(),
    )
    .unwrap();
    let call = Vllm08Call::from_span(
        VllmBf16Invocation::from_span(
            GdnPrefixLayer::new(source, 0).unwrap(),
            VllmBf16Mode::Prefill,
            teachers,
        )
        .unwrap(),
    )
    .unwrap();
    let layer = call.layer(3).unwrap();
    let parameter = |suffix: &str, width: usize| SourceParameter {
        source,
        name: format!("model.language_model.layers.3.{suffix}"),
        kind: QuantKind::Bf16,
        shape: vec![width as u64],
        bytes: width * 2,
    };
    let dual = Vllm08Operator::new(
        call.clone(),
        Some(layer),
        Vllm08Kind::ResidualRms(Vllm08RmsSite::Input),
        vec![],
        vec![parameter("input_layernorm.weight", 1024)],
    )
    .unwrap();
    let qkv = Vllm08Operator::new(
        call.clone(),
        Some(layer),
        Vllm08Kind::Projection(Vllm08Projection::AttentionQkv),
        [
            ("q_proj.weight", 4096),
            ("k_proj.weight", 512),
            ("v_proj.weight", 512),
        ]
        .into_iter()
        .map(|(name, rows)| SourceTensor {
            source,
            name: format!("model.language_model.layers.3.self_attn.{name}"),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            rows,
            columns: 1024,
        })
        .collect(),
        vec![],
    )
    .unwrap();
    let qk = Vllm08Operator::new(
        call,
        Some(layer),
        Vllm08Kind::AttentionQk,
        vec![],
        vec![
            parameter("self_attn.q_norm.weight", 256),
            parameter("self_attn.k_norm.weight", 256),
        ],
    )
    .unwrap();
    let node = |descriptor, inputs, outputs| {
        PlanNode::new(
            "test metadata".into(),
            PlanOperation::TypedVllm08 {
                descriptor: Box::new(descriptor),
                state: None,
            },
            inputs,
            outputs,
            None,
        )
    };
    let dual = node(
        dual,
        vec![ValueId(38), ValueId(35)],
        vec![ValueId(39), ValueId(40)],
    );
    let mut nodes = vec![dual.clone(); 34];
    nodes[31] = dual;
    nodes[32] = node(qkv, vec![ValueId(39)], vec![ValueId(41)]);
    nodes[33] = node(
        qk,
        vec![ValueId(41)],
        vec![ValueId(42), ValueId(43), ValueId(44)],
    );
    nodes
}

#[test]
fn genuine_producer_coordinates_stop_before_qk_and_refuse_substitution() {
    let nodes = extension();
    require_qkv_extension(&nodes).unwrap();
    for mutation in 0..6 {
        let mut wrong = nodes.clone();
        match mutation {
            0 => wrong[31].inputs.swap(0, 1),
            1 => wrong[31].outputs.swap(0, 1),
            2 => wrong[32].inputs[0] = ValueId(40),
            3 => wrong[32].outputs[0] = ValueId(42),
            4 => wrong.swap(32, 33),
            _ => {
                wrong.pop();
            }
        }
        assert!(require_qkv_extension(&wrong).is_err(), "mutation{mutation}");
    }
}

#[test]
fn new_cut_cannot_relabel_old_value_weight_window_or_record_cohorts() {
    use Canonical47PrefixKind as K;
    for (kind, values, duals, weights, windows, records) in [
        (K::FirstThree, 39, 5, 15, 132, 8),
        (K::ThroughAttentionQkv, 42, 6, 16, 142, 10),
        (K::ThroughAttentionKv, 45, 6, 16, 142, 12),
        (K::ThroughAttentionFa, 46, 6, 16, 142, 13),
        (K::ThroughLayer3, 54, 7, 19, 168, 15),
    ] {
        assert_eq!(kind.values(), values);
        assert_eq!(kind.duals(), duals);
        assert_eq!(kind.weight_groups(), weights);
        assert_eq!(kind.source_windows(), windows);
        assert_eq!(kind.native_records(), records);
    }
}

fn kv_extension() -> Vec<PlanNode> {
    use zerogravity_core::accelerator::program::{StateEffect, StateVersion};
    let mut nodes = extension();
    let call = match &nodes[33].operation {
        PlanOperation::TypedVllm08 { descriptor, .. } => descriptor.call().clone(),
        _ => unreachable!(),
    };
    let layer = call.layer(3).unwrap();
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
    let kv = Vllm08Operator::new(
        call.clone(),
        Some(layer),
        Vllm08Kind::KvAppend,
        vec![],
        vec![],
    )
    .unwrap();
    nodes.push(PlanNode::new(
        "actual metadata Kv3".into(),
        PlanOperation::TypedVllm08 {
            descriptor: Box::new(kv),
            state: None,
        },
        vec![ValueId(43), ValueId(41)],
        vec![],
        Some(effect),
    ));
    let fa = Vllm08Operator::new(
        call,
        Some(layer),
        Vllm08Kind::FlashAttention,
        vec![],
        vec![],
    )
    .unwrap();
    nodes.push(PlanNode::new(
        "excluded actual FA".into(),
        PlanOperation::TypedVllm08 {
            descriptor: Box::new(fa),
            state: Some(effect.write),
        },
        vec![ValueId(42)],
        vec![ValueId(45)],
        None,
    ));
    nodes
}
#[test]
fn kv_extension_preserves_call_ssa_effect_and_exact_endpoint() {
    let nodes = kv_extension();
    require_kv_extension(&nodes).unwrap();
    for mutation in 0..8 {
        let mut wrong = nodes.clone();
        match mutation {
            0 => wrong[33].outputs.swap(0, 1),
            1 => wrong[34].inputs.swap(0, 1),
            2 => wrong[34].state.as_mut().unwrap().write.ordinal = 2,
            3 => wrong[34].state.as_mut().unwrap().write.slot = StateSlot::Kv(KvDomain(7)),
            4 => wrong[34].outputs.push(ValueId(45)),
            5 => {
                if let PlanOperation::TypedVllm08 { state, .. } = &mut wrong[35].operation {
                    *state = None;
                }
            }
            6 => wrong[35].inputs[0] = ValueId(44),
            _ => {
                wrong.pop();
            }
        }
        assert!(require_kv_extension(&wrong).is_err(), "mutation{mutation}");
    }
}

fn fa_extension() -> Vec<PlanNode> {
    let mut nodes = kv_extension();
    let call = match &nodes[35].operation {
        PlanOperation::TypedVllm08 { descriptor, .. } => descriptor.call().clone(),
        _ => unreachable!(),
    };
    let sigmoid = Vllm08Operator::new(
        call.clone(),
        Some(call.layer(3).unwrap()),
        Vllm08Kind::Sigmoid,
        vec![],
        vec![],
    )
    .unwrap();
    nodes.push(PlanNode::new(
        "excluded actual sigmoid".into(),
        PlanOperation::TypedVllm08 {
            descriptor: Box::new(sigmoid),
            state: None,
        },
        vec![ValueId(44)],
        vec![ValueId(46)],
        None,
    ));
    nodes
}
#[test]
fn fa_reads_the_actual_appended_kv_version_and_stops_before_sigmoid() {
    let nodes = fa_extension();
    require_kv_extension(&nodes).unwrap();
    require_fa_extension(&nodes).unwrap();
    for mutation in 0..9 {
        let mut wrong = nodes.clone();
        match mutation {
            0 => {
                if let PlanOperation::TypedVllm08 {
                    state: Some(read), ..
                } = &mut wrong[35].operation
                {
                    read.ordinal = 0
                }
            }
            1 => {
                if let PlanOperation::TypedVllm08 {
                    state: Some(read), ..
                } = &mut wrong[35].operation
                {
                    read.slot = StateSlot::Kv(KvDomain(7))
                }
            }
            2 => wrong[35].inputs[0] = ValueId(44),
            3 => wrong[35].outputs[0] = ValueId(46),
            4 => wrong[36].inputs[0] = ValueId(45),
            5 => wrong[36].outputs[0] = ValueId(47),
            6 => wrong.swap(35, 36),
            7 => {
                wrong.pop();
            }
            _ => wrong[35].state = wrong[34].state,
        }
        assert!(require_fa_extension(&wrong).is_err(), "mutation {mutation}");
    }
}

fn layer3_extension() -> Vec<PlanNode> {
    let mut nodes = fa_extension();
    let call = match &nodes[35].operation {
        PlanOperation::TypedVllm08 { descriptor, .. } => descriptor.call().clone(),
        _ => unreachable!(),
    };
    let source = call.source();
    let layer = call.layer(3).unwrap();
    let matrix = |suffix: &str, rows, columns| SourceTensor {
        source,
        name: format!("model.language_model.layers.3.{suffix}"),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        rows,
        columns,
    };
    let make = |kind, matrices, parameters| {
        Vllm08Operator::new(call.clone(), Some(layer), kind, matrices, parameters).unwrap()
    };
    let parameter = |layer, suffix: &str| SourceParameter {
        source,
        name: format!("model.language_model.layers.{layer}.{suffix}"),
        kind: QuantKind::Bf16,
        shape: vec![1024],
        bytes: 2048,
    };
    for (descriptor, inputs, outputs) in [
        (
            make(Vllm08Kind::Multiply, vec![], vec![]),
            vec![ValueId(45), ValueId(46)],
            vec![ValueId(47)],
        ),
        (
            make(
                Vllm08Kind::Projection(Vllm08Projection::AttentionOut),
                vec![matrix("self_attn.o_proj.weight", 1024, 2048)],
                vec![],
            ),
            vec![ValueId(47)],
            vec![ValueId(48)],
        ),
        (
            make(
                Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention),
                vec![],
                vec![parameter(3, "post_attention_layernorm.weight")],
            ),
            vec![ValueId(48), ValueId(40)],
            vec![ValueId(49), ValueId(50)],
        ),
        (
            make(
                Vllm08Kind::Projection(Vllm08Projection::GateUp),
                vec![
                    matrix("mlp.gate_proj.weight", 3584, 1024),
                    matrix("mlp.up_proj.weight", 3584, 1024),
                ],
                vec![],
            ),
            vec![ValueId(49)],
            vec![ValueId(51)],
        ),
        (
            make(Vllm08Kind::SiluAndMul, vec![], vec![]),
            vec![ValueId(51)],
            vec![ValueId(52)],
        ),
        (
            make(
                Vllm08Kind::Projection(Vllm08Projection::Down),
                vec![matrix("mlp.down_proj.weight", 1024, 3584)],
                vec![],
            ),
            vec![ValueId(52)],
            vec![ValueId(53)],
        ),
    ] {
        nodes.push(PlanNode::new(
            "actual layer3 tail metadata".into(),
            PlanOperation::TypedVllm08 {
                descriptor: Box::new(descriptor),
                state: None,
            },
            inputs,
            outputs,
            None,
        ));
    }
    let next = Vllm08Operator::new(
        call.clone(),
        Some(call.layer(4).unwrap()),
        Vllm08Kind::ResidualRms(Vllm08RmsSite::Input),
        vec![],
        vec![parameter(4, "input_layernorm.weight")],
    )
    .unwrap();
    nodes.push(PlanNode::new(
        "excluded layer4".into(),
        PlanOperation::TypedVllm08 {
            descriptor: Box::new(next),
            state: None,
        },
        vec![ValueId(53), ValueId(50)],
        vec![ValueId(54), ValueId(55)],
        None,
    ));
    nodes
}
#[test]
fn full_layer3_requires_real_operations_and_ssa_before_original_layer4() {
    let nodes = layer3_extension();
    require_layer3_extension(&nodes).unwrap();
    for mutation in 0..12 {
        let mut wrong = nodes.clone();
        match mutation {
            0 => wrong[37].inputs.swap(0, 1),
            1 => wrong[38].inputs[0] = ValueId(45),
            2 => wrong[38].outputs[0] = ValueId(49),
            3 => wrong[39].inputs.swap(0, 1),
            4 => wrong[40].inputs[0] = ValueId(50),
            5 => wrong[41].inputs[0] = ValueId(49),
            6 => wrong[42].outputs[0] = ValueId(54),
            7 => wrong[43].inputs[1] = ValueId(40),
            8 => wrong[43].outputs.swap(0, 1),
            9 => wrong.swap(38, 42),
            10 => {
                wrong.pop();
            }
            _ => wrong[38].operation = nodes[40].operation.clone(),
        }
        assert!(
            require_layer3_extension(&wrong).is_err(),
            "mutation{mutation}"
        );
    }
}
