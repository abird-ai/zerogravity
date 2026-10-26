//! Exact state-read coordinate accepted by the original full24 dispatcher.
use zerogravity_core::accelerator::program::{
    PlanNode, PlanOperation, SourceOperator, SourceOperatorKind, StateSlot,
};
use zerogravity_core::{Error, Result};

pub(super) fn descriptor(node: &PlanNode) -> Result<&SourceOperator> {
    let PlanOperation::SourceOperator { descriptor, state } = &node.operation else {
        return Err(Error::State(
            "full24 actual typed node descriptor absent".into(),
        ));
    };
    let valid = match (descriptor.kind(), state) {
        (SourceOperatorKind::FlashAttention { domain, .. }, Some(version)) => {
            node.state.is_none() && version.slot == StateSlot::Kv(domain) && version.ordinal == 1
        }
        (SourceOperatorKind::FlashAttention { .. }, None) => false,
        (_, None) => true,
        (_, Some(_)) => false,
    };
    if !valid {
        return Err(Error::State(
            "full24 state read must be exact private appended FA KV; non-FA has no state read"
                .into(),
        ));
    }
    Ok(descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::{
        AttentionGeometry, KvDomain, SourceAttentionMask, SourceCall, SourceCallMode,
        SourceNumericalContract, SourceRows, TensorView, TextCallGeometry,
    };
    use zerogravity_core::accelerator::program::{
        ConvDomain, RecurrentDomain, StateEffect, StateVersion, ValueId,
    };
    fn call(mode: SourceCallMode, base: usize, rows: usize) -> SourceCall {
        let source = SourceIdentity::for_new_allocation().unwrap();
        SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, 248320, 512).unwrap(),
                base,
                vec![353; rows].into(),
            )
            .unwrap(),
            mode,
        )
        .unwrap()
    }
    // Fixture math matches the original cases; no model/profile eligibility.
    fn fa(layer: usize) -> SourceOperatorKind {
        SourceOperatorKind::FlashAttention {
            domain: KvDomain(layer),
            geometry: AttentionGeometry {
                width: 256,
                query_heads: 8,
                kv_heads: 2,
            },
            scale: 0.0625,
            mask: SourceAttentionMask::Causal,
        }
    }
    fn operator(call: &SourceCall, kind: SourceOperatorKind) -> SourceOperator {
        SourceOperator::new(
            call.clone(),
            kind,
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            Vec::new(),
            Vec::new(),
        )
        .unwrap()
    }
    fn version(layer: usize, ordinal: u32) -> StateVersion {
        StateVersion {
            slot: StateSlot::Kv(KvDomain(layer)),
            ordinal,
        }
    }
    fn node(operator: SourceOperator, state: Option<StateVersion>) -> PlanNode {
        PlanNode::new(
            "source.full_state_read".into(),
            PlanOperation::SourceOperator {
                descriptor: Box::new(operator),
                state,
            },
            vec![ValueId(1)],
            vec![ValueId(2)],
            None,
        )
    }
    fn set_read(node: &mut PlanNode, state: Option<StateVersion>) {
        let PlanOperation::SourceOperator { state: read, .. } = &mut node.operation else {
            unreachable!()
        };
        *read = state;
    }

    #[test]
    fn every_original_attention_layer_and_phase_accepts_only_its_appended_read() {
        for (mode, base, rows) in [
            (SourceCallMode::Prefill, 0, 1),
            (SourceCallMode::Decode, 1, 1),
            (SourceCallMode::Prefill, 0, 64),
            (SourceCallMode::Prefill, 64, 2),
            (SourceCallMode::Decode, 66, 1),
        ] {
            let call = call(mode, base, rows);
            for layer in [3, 7, 11, 15, 19, 23] {
                let original = operator(&call, fa(layer));
                let actual = node(original.clone(), Some(version(layer, 1)));
                let returned = descriptor(&actual).unwrap();
                let PlanOperation::SourceOperator {
                    descriptor: retained,
                    ..
                } = &actual.operation
                else {
                    unreachable!()
                };
                assert!(std::ptr::eq(returned, retained.as_ref()));
                assert_eq!(returned, &original);
                assert_eq!(returned.call(), &call);
            }
        }
    }

    #[test]
    fn fa_rejects_missing_foreign_and_wrong_version_reads_on_every_actual_layer() {
        let call = call(SourceCallMode::Prefill, 0, 1);
        for layer in [3, 7, 11, 15, 19, 23] {
            let actual = node(operator(&call, fa(layer)), Some(version(layer, 1)));
            let foreign_layer = if layer == 3 { 7 } else { 3 };
            for read in [
                None,
                Some(version(foreign_layer, 1)),
                Some(version(layer, 0)),
                Some(version(layer, 2)),
                Some(StateVersion {
                    slot: StateSlot::Conv(ConvDomain(layer)),
                    ordinal: 1,
                }),
                Some(StateVersion {
                    slot: StateSlot::Recurrent(RecurrentDomain(layer)),
                    ordinal: 1,
                }),
            ] {
                let mut bad = actual.clone();
                set_read(&mut bad, read);
                assert!(descriptor(&bad).is_err(), "layer={layer} read={read:?}");
            }
        }
    }

    #[test]
    fn fa_cannot_write_even_when_its_read_is_the_exact_appended_kv_version() {
        let call = call(SourceCallMode::Prefill, 0, 64);
        for layer in [3, 7, 11, 15, 19, 23] {
            let mut actual = node(operator(&call, fa(layer)), Some(version(layer, 1)));
            actual.state = Some(StateEffect {
                read: version(layer, 0),
                write: version(layer, 1),
            });
            assert!(descriptor(&actual).is_err());
        }
    }

    #[test]
    fn non_fa_reads_refuse_but_real_append_write_effect_remains_separate() {
        let call = call(SourceCallMode::Prefill, 64, 2);
        for layer in [3, 7, 11, 15, 19, 23] {
            for kind in [
                SourceOperatorKind::KvAppend {
                    domain: KvDomain(layer),
                    key_width: 512,
                    packed_value: TensorView {
                        offset: 4608,
                        rows: call.rows().len(),
                        width: 512,
                        row_stride: 5120,
                    },
                },
                SourceOperatorKind::Sigmoid { width: 2048 },
                SourceOperatorKind::Multiply { width: 2048 },
            ] {
                let mut actual = node(operator(&call, kind), None);
                if matches!(kind, SourceOperatorKind::KvAppend { .. }) {
                    actual.state = Some(StateEffect {
                        read: version(layer, 0),
                        write: version(layer, 1),
                    });
                }
                assert!(descriptor(&actual).is_ok());
                set_read(&mut actual, Some(version(layer, 1)));
                assert!(descriptor(&actual).is_err());
            }
        }
    }

    #[test]
    fn unrelated_plan_operation_never_becomes_a_full24_descriptor() {
        let node = PlanNode::new(
            "unrelated".into(),
            PlanOperation::Observe {
                layer: 0,
                stage: "unrelated",
            },
            Vec::new(),
            Vec::new(),
            None,
        );
        assert!(descriptor(&node).is_err());
    }
}
