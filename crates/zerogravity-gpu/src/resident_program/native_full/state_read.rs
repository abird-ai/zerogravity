//! Exact state-read coordinate accepted by the original full24 dispatcher.
use zerogravity_core::accelerator::program::{
    KvDomain, PlanNode, PlanOperation, StateSlot, Vllm08Kind, Vllm08Operator,
};
use zerogravity_core::{Error, Result};

pub(super) fn descriptor(node: &PlanNode) -> Result<&Vllm08Operator> {
    let PlanOperation::TypedVllm08 { descriptor, state } = &node.operation else {
        return Err(Error::State(
            "full24 actual typed node descriptor absent".into(),
        ));
    };
    let valid = match (descriptor.kind(), state) {
        (Vllm08Kind::FlashAttention, Some(version)) => {
            node.state.is_none()
                && descriptor.layer().is_some_and(|layer| {
                    layer.is_attention()
                        && version.slot == StateSlot::Kv(KvDomain(layer.index()))
                        && version.ordinal == 1
                })
        }
        (Vllm08Kind::FlashAttention, None) => false,
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
        ConvDomain, GdnPrefixLayer, RecurrentDomain, StateEffect, StateVersion, ValueId,
        Vllm08Call, VllmBf16Invocation, VllmBf16Phase, VllmBf16Teachers,
    };

    fn call(phase: VllmBf16Phase, base: usize) -> Vllm08Call {
        let source = SourceIdentity::for_new_allocation().unwrap();
        Vllm08Call::new(
            VllmBf16Invocation::new(
                GdnPrefixLayer::new(source, 0).unwrap(),
                phase,
                VllmBf16Teachers::new(source, base, 512, &vec![353; phase.rows()]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn operator(call: &Vllm08Call, layer: usize, kind: Vllm08Kind) -> Vllm08Operator {
        let operator = Vllm08Operator::new(
            call.clone(),
            Some(call.layer(layer).unwrap()),
            kind,
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        operator.validate().unwrap();
        operator
    }
    fn version(layer: usize, ordinal: u32) -> StateVersion {
        StateVersion {
            slot: StateSlot::Kv(KvDomain(layer)),
            ordinal,
        }
    }
    fn node(operator: Vllm08Operator, state: Option<StateVersion>) -> PlanNode {
        PlanNode::new(
            "vllm.original.full24".into(),
            PlanOperation::TypedVllm08 {
                descriptor: Box::new(operator),
                state,
            },
            vec![ValueId(1)],
            vec![ValueId(2)],
            None,
        )
    }
    fn set_read(node: &mut PlanNode, state: Option<StateVersion>) {
        let PlanOperation::TypedVllm08 { state: read, .. } = &mut node.operation else {
            unreachable!()
        };
        *read = state;
    }

    #[test]
    fn every_original_attention_layer_and_phase_accepts_only_its_appended_read() {
        for (phase, base) in [
            (VllmBf16Phase::InitialOne, 0),
            (VllmBf16Phase::CachedOne, 1),
            (VllmBf16Phase::Initial64, 0),
            (VllmBf16Phase::ContinuedTwo, 64),
            (VllmBf16Phase::CachedOne, 66),
        ] {
            let call = call(phase, base);
            for layer in [3, 7, 11, 15, 19, 23] {
                let original = operator(&call, layer, Vllm08Kind::FlashAttention);
                let actual = node(original.clone(), Some(version(layer, 1)));
                let returned = descriptor(&actual).unwrap();
                let PlanOperation::TypedVllm08 {
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
        let call = call(VllmBf16Phase::InitialOne, 0);
        for layer in [3, 7, 11, 15, 19, 23] {
            let actual = node(
                operator(&call, layer, Vllm08Kind::FlashAttention),
                Some(version(layer, 1)),
            );
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
        let call = call(VllmBf16Phase::Initial64, 0);
        for layer in [3, 7, 11, 15, 19, 23] {
            let mut actual = node(
                operator(&call, layer, Vllm08Kind::FlashAttention),
                Some(version(layer, 1)),
            );
            actual.state = Some(StateEffect {
                read: version(layer, 0),
                write: version(layer, 1),
            });
            assert!(descriptor(&actual).is_err());
        }
    }

    #[test]
    fn non_fa_reads_refuse_but_real_append_write_effect_remains_separate() {
        let call = call(VllmBf16Phase::ContinuedTwo, 64);
        for layer in [3, 7, 11, 15, 19, 23] {
            for kind in [
                Vllm08Kind::KvAppend,
                Vllm08Kind::Sigmoid,
                Vllm08Kind::Multiply,
            ] {
                let mut actual = node(operator(&call, layer, kind), None);
                if kind == Vllm08Kind::KvAppend {
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
            PlanOperation::TypedVllmStateCarry,
            Vec::new(),
            Vec::new(),
            None,
        );
        assert!(descriptor(&node).is_err());
    }
}
