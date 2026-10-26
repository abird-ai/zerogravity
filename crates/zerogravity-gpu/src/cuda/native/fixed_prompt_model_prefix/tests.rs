//! Metadata fixtures only; no source contents, device, Ready or inference claim.
use super::*;
use crate::resident_program::qwen2b_component::FIXED_PROMPT_IDS;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::MatrixAxes;
use zerogravity_core::accelerator::program::SourceCallMode;
use zerogravity_core::accelerator::program::SourceParameter;
use zerogravity_core::accelerator::program::SourceTensor;
use zerogravity_core::accelerator::program::TeacherSpan;
use zerogravity_core::quant::QuantKind;
use zerogravity_model_qwen35::plan::QwenGdnCall;
use zerogravity_model_qwen35::plan::QwenInputRows;
use zerogravity_model_qwen35::plan::QwenOperator;
use zerogravity_model_qwen35::plan::QwenPrefixLayer;
use zerogravity_model_qwen35::plan::QwenRmsSite;
use zerogravity_model_qwen35::plan::QwenSourceCall;
use zerogravity_model_qwen35::plan::{QwenOperation, QwenProjection};

fn extension() -> (Vec<PlanNode>, BTreeMap<usize, QwenOperator>) {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let teachers = QwenInputRows::from_span(
        TeacherSpan::new(source, 0, 512, 248320, &FIXED_PROMPT_IDS).unwrap(),
    )
    .unwrap();
    let call = QwenSourceCall::from_span(
        QwenGdnCall::from_span(
            QwenPrefixLayer::new(source, 0).unwrap(),
            SourceCallMode::Prefill,
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
    let dual = QwenOperator::new(
        call.clone(),
        Some(layer),
        QwenOperation::ResidualRms(QwenRmsSite::Input),
        vec![],
        vec![parameter("input_layernorm.weight", 1024)],
    )
    .unwrap();
    let qkv = QwenOperator::new(
        call.clone(),
        Some(layer),
        QwenOperation::Projection(QwenProjection::AttentionQkv),
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
    let qk = QwenOperator::new(
        call,
        Some(layer),
        QwenOperation::AttentionQk,
        vec![],
        vec![
            parameter("self_attn.q_norm.weight", 256),
            parameter("self_attn.k_norm.weight", 256),
        ],
    )
    .unwrap();
    let metadata = BTreeMap::from([(31, dual.clone()), (32, qkv.clone()), (33, qk.clone())]);
    let node = |descriptor: QwenOperator, inputs, outputs| {
        PlanNode::new(
            "test metadata".into(),
            PlanOperation::SourceOperator {
                descriptor: Box::new(descriptor.mathematical_operator().unwrap()),
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
    (nodes, metadata)
}

#[test]
fn genuine_producer_coordinates_stop_before_qk_and_refuse_substitution() {
    let (nodes, _metadata) = extension();
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
    use FixedPromptPrefixKind as K;
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

fn kv_extension() -> (Vec<PlanNode>, BTreeMap<usize, QwenOperator>) {
    use zerogravity_core::accelerator::program::{StateEffect, StateVersion};
    let (mut nodes, mut metadata) = extension();
    let call = metadata[&33].call().clone();
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
    let kv = QwenOperator::new(
        call.clone(),
        Some(layer),
        QwenOperation::KvAppend,
        vec![],
        vec![],
    )
    .unwrap();
    metadata.insert(34, kv.clone());
    nodes.push(PlanNode::new(
        "actual metadata Kv3".into(),
        PlanOperation::SourceOperator {
            descriptor: Box::new(kv.mathematical_operator().unwrap()),
            state: None,
        },
        vec![ValueId(43), ValueId(41)],
        vec![],
        Some(effect),
    ));
    let fa = QwenOperator::new(
        call,
        Some(layer),
        QwenOperation::FlashAttention,
        vec![],
        vec![],
    )
    .unwrap();
    metadata.insert(35, fa.clone());
    nodes.push(PlanNode::new(
        "excluded actual FA".into(),
        PlanOperation::SourceOperator {
            descriptor: Box::new(fa.mathematical_operator().unwrap()),
            state: Some(effect.write),
        },
        vec![ValueId(42)],
        vec![ValueId(45)],
        None,
    ));
    (nodes, metadata)
}
#[test]
fn kv_extension_preserves_call_ssa_effect_and_exact_endpoint() {
    let (nodes, _metadata) = kv_extension();
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
                if let PlanOperation::SourceOperator { state, .. } = &mut wrong[35].operation {
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

fn fa_extension() -> (Vec<PlanNode>, BTreeMap<usize, QwenOperator>) {
    let (mut nodes, mut metadata) = kv_extension();
    let call = metadata[&35].call().clone();
    let sigmoid = QwenOperator::new(
        call.clone(),
        Some(call.layer(3).unwrap()),
        QwenOperation::Sigmoid,
        vec![],
        vec![],
    )
    .unwrap();
    metadata.insert(36, sigmoid.clone());
    nodes.push(PlanNode::new(
        "excluded actual sigmoid".into(),
        PlanOperation::SourceOperator {
            descriptor: Box::new(sigmoid.mathematical_operator().unwrap()),
            state: None,
        },
        vec![ValueId(44)],
        vec![ValueId(46)],
        None,
    ));
    (nodes, metadata)
}
#[test]
fn fa_reads_the_actual_appended_kv_version_and_stops_before_sigmoid() {
    let (nodes, _metadata) = fa_extension();
    require_kv_extension(&nodes).unwrap();
    require_fa_extension(&nodes).unwrap();
    for mutation in 0..9 {
        let mut wrong = nodes.clone();
        match mutation {
            0 => {
                if let PlanOperation::SourceOperator {
                    state: Some(read), ..
                } = &mut wrong[35].operation
                {
                    read.ordinal = 0
                }
            }
            1 => {
                if let PlanOperation::SourceOperator {
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

fn layer3_extension() -> (Vec<PlanNode>, BTreeMap<usize, QwenOperator>) {
    let (mut nodes, mut metadata) = fa_extension();
    let call = metadata[&35].call().clone();
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
        QwenOperator::new(call.clone(), Some(layer), kind, matrices, parameters).unwrap()
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
            make(QwenOperation::Multiply, vec![], vec![]),
            vec![ValueId(45), ValueId(46)],
            vec![ValueId(47)],
        ),
        (
            make(
                QwenOperation::Projection(QwenProjection::AttentionOut),
                vec![matrix("self_attn.o_proj.weight", 1024, 2048)],
                vec![],
            ),
            vec![ValueId(47)],
            vec![ValueId(48)],
        ),
        (
            make(
                QwenOperation::ResidualRms(QwenRmsSite::PostAttention),
                vec![],
                vec![parameter(3, "post_attention_layernorm.weight")],
            ),
            vec![ValueId(48), ValueId(40)],
            vec![ValueId(49), ValueId(50)],
        ),
        (
            make(
                QwenOperation::Projection(QwenProjection::GateUp),
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
            make(QwenOperation::SiluAndMul, vec![], vec![]),
            vec![ValueId(51)],
            vec![ValueId(52)],
        ),
        (
            make(
                QwenOperation::Projection(QwenProjection::Down),
                vec![matrix("mlp.down_proj.weight", 1024, 3584)],
                vec![],
            ),
            vec![ValueId(52)],
            vec![ValueId(53)],
        ),
    ] {
        metadata.insert(nodes.len(), descriptor.clone());
        nodes.push(PlanNode::new(
            "actual layer3 tail metadata".into(),
            PlanOperation::SourceOperator {
                descriptor: Box::new(descriptor.mathematical_operator().unwrap()),
                state: None,
            },
            inputs,
            outputs,
            None,
        ));
    }
    let next = QwenOperator::new(
        call.clone(),
        Some(call.layer(4).unwrap()),
        QwenOperation::ResidualRms(QwenRmsSite::Input),
        vec![],
        vec![parameter(4, "input_layernorm.weight")],
    )
    .unwrap();
    metadata.insert(43, next.clone());
    nodes.push(PlanNode::new(
        "excluded layer4".into(),
        PlanOperation::SourceOperator {
            descriptor: Box::new(next.mathematical_operator().unwrap()),
            state: None,
        },
        vec![ValueId(53), ValueId(50)],
        vec![ValueId(54), ValueId(55)],
        None,
    ));
    (nodes, metadata)
}
#[test]
fn full_layer3_requires_real_operations_and_ssa_before_original_layer4() {
    let (nodes, _metadata) = layer3_extension();
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
