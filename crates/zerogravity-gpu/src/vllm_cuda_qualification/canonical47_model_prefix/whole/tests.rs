//! Source metadata fixtures only. No device, contents, Ready or runtime claim.
use super::*;
use crate::resident_program::qwen2b_component::CANONICAL47_PROMPT;
use zerogravity_core::{
    accelerator::{
        SourceIdentity,
        program::{
            GdnPrefixLayer, MatrixAxes, SourceParameter, SourceTensor, StateVersion, TeacherSpan,
            VllmBf16Invocation, VllmBf16Mode, VllmBf16Teachers,
        },
    },
    quant::QuantKind,
};
fn fixture(index: usize) -> (Vllm08Call, Vec<PlanNode>) {
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
    let layer = call.layer(index).unwrap();
    let matrix = |name: &str, rows, columns| SourceTensor {
        source,
        name: format!("model.language_model.layers.{index}.self_attn.{name}.weight"),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        rows,
        columns,
    };
    let gamma = |name: &str| SourceParameter {
        source,
        name: format!("model.language_model.layers.{index}.self_attn.{name}.weight"),
        kind: QuantKind::Bf16,
        shape: vec![256],
        bytes: 512,
    };
    let effect = StateEffect {
        read: StateVersion {
            slot: StateSlot::Kv(KvDomain(index)),
            ordinal: 0,
        },
        write: StateVersion {
            slot: StateSlot::Kv(KvDomain(index)),
            ordinal: 1,
        },
    };
    let mut nodes = Vec::new();
    for (kind, matrices, parameters, inputs, outputs, state_read, state_effect) in [
        (
            Vllm08Kind::Projection(Vllm08Projection::AttentionQkv),
            vec![
                matrix("q_proj", 4096, 1024),
                matrix("k_proj", 512, 1024),
                matrix("v_proj", 512, 1024),
            ],
            vec![],
            vec![ValueId(0)],
            vec![ValueId(1)],
            None,
            None,
        ),
        (
            Vllm08Kind::AttentionQk,
            vec![],
            vec![gamma("q_norm"), gamma("k_norm")],
            vec![ValueId(1)],
            vec![ValueId(2), ValueId(3), ValueId(4)],
            None,
            None,
        ),
        (
            Vllm08Kind::KvAppend,
            vec![],
            vec![],
            vec![ValueId(3), ValueId(1)],
            vec![],
            None,
            Some(effect),
        ),
        (
            Vllm08Kind::FlashAttention,
            vec![],
            vec![],
            vec![ValueId(2)],
            vec![ValueId(5)],
            Some(effect.write),
            None,
        ),
        (
            Vllm08Kind::Sigmoid,
            vec![],
            vec![],
            vec![ValueId(4)],
            vec![ValueId(6)],
            None,
            None,
        ),
        (
            Vllm08Kind::Multiply,
            vec![],
            vec![],
            vec![ValueId(5), ValueId(6)],
            vec![ValueId(7)],
            None,
            None,
        ),
        (
            Vllm08Kind::Projection(Vllm08Projection::AttentionOut),
            vec![matrix("o_proj", 1024, 2048)],
            vec![],
            vec![ValueId(7)],
            vec![ValueId(8)],
            None,
            None,
        ),
    ] {
        nodes.push(PlanNode::new(
            "whole site fixture".into(),
            PlanOperation::TypedVllm08 {
                descriptor: Box::new(
                    zerogravity_core::accelerator::program::Vllm08Operator::new(
                        call.clone(),
                        Some(layer),
                        kind,
                        matrices,
                        parameters,
                    )
                    .unwrap(),
                ),
                state: state_read,
            },
            inputs,
            outputs,
            state_effect,
        ));
    }
    (call, nodes)
}
#[test]
fn every_whole_attention_site_preserves_exact_local_ssa_and_kv_read_ordinal() {
    for layer in [3, 7, 11, 15, 19, 23] {
        let (call, nodes) = fixture(layer);
        let site = AttentionSite::bind_nodes(&call, &nodes, layer).unwrap();
        assert_eq!(site.layer().index(), layer);
        assert_eq!(site.nodes(), [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(site.values(), [1, 2, 3, 4, 5, 6, 7, 8].map(ValueId));
        assert_eq!(site.effect().read.ordinal, 0);
        assert_eq!(site.effect().write.ordinal, 1);
    }
}
#[test]
fn whole_attention_refuses_input_source_layer_effect_endpoint_and_repetition_edits() {
    let (call, nodes) = fixture(3);
    for mutation in 0..10 {
        let mut wrong = nodes.clone();
        match mutation {
            0 => wrong[1].inputs[0] = ValueId(0),
            1 => wrong[2].inputs.swap(0, 1),
            2 => wrong[4].inputs[0] = ValueId(5),
            3 => wrong[5].inputs.swap(0, 1),
            4 => wrong[6].inputs[0] = ValueId(6),
            5 => wrong[2].state.as_mut().unwrap().write.ordinal = 2,
            6 => wrong[2].state.as_mut().unwrap().write.slot = StateSlot::Kv(KvDomain(7)),
            7 => {
                let other = fixture(7);
                wrong[0].operation = other.1[0].operation.clone();
            }
            8 => {
                wrong.push(wrong[3].clone());
            }
            _ => {
                wrong.pop();
            }
        }
        assert!(
            AttentionSite::bind_nodes(&call, &wrong, 3).is_err(),
            "mutation{mutation}"
        );
    }
}
#[test]
fn whole_metadata_counts_do_not_change_any_old_cut_contract() {
    assert_eq!(
        (
            Canonical47PrefixKind::WholeInitial47.values(),
            Canonical47PrefixKind::WholeInitial47.duals(),
            Canonical47PrefixKind::WholeInitial47.weight_groups(),
            Canonical47PrefixKind::WholeInitial47.source_windows(),
            Canonical47PrefixKind::WholeInitial47.native_records()
        ),
        (327, 48, 114, 1008, 16)
    );
    for (kind, expected) in [
        (Canonical47PrefixKind::FirstThree, (39, 5, 15, 132, 8)),
        (
            Canonical47PrefixKind::ThroughAttentionQkv,
            (42, 6, 16, 142, 10),
        ),
        (
            Canonical47PrefixKind::ThroughAttentionKv,
            (45, 6, 16, 142, 12),
        ),
        (
            Canonical47PrefixKind::ThroughAttentionFa,
            (46, 6, 16, 142, 13),
        ),
        (Canonical47PrefixKind::ThroughLayer3, (54, 7, 19, 168, 15)),
    ] {
        assert_eq!(
            (
                kind.values(),
                kind.duals(),
                kind.weight_groups(),
                kind.source_windows(),
                kind.native_records()
            ),
            expected
        );
    }
}
