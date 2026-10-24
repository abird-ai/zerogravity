//! Source metadata fixtures; no trained values, GPU owners or permits.
use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{FullModelProfile, ShortHfSpan, ShortHfWorkload};

#[test]
fn actual_last_prompt_row_and_both_source_widths_are_not_legacy_case_retags() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        let row = profile.hidden_size() * 2;
        for count in [47, 57, 62, 67, 72] {
            let workload = ShortHfWorkload::from_prompt_rows(count)?;
            let span = ShortHfSpan::prefill(source, workload, &vec![11; count])?;
            let call = ShortHfCall::new(profile, span)?;
            let head = ShortHfHeadView::for_call(&call)?;
            assert_eq!(
                head,
                ShortHfHeadView {
                    full_bytes: count * row,
                    offset: (count - 1) * row,
                    selected_bytes: row,
                    output_bytes: 496640
                }
            );
            assert!(call.request().legacy_phase().is_err());
            for base in workload.cached_bases() {
                let call =
                    ShortHfCall::new(profile, ShortHfSpan::cached(source, workload, base, 353)?)?;
                let head = ShortHfHeadView::for_call(&call)?;
                assert_eq!(
                    (head.full_bytes, head.offset, head.selected_bytes),
                    (row, 0, row)
                );
                assert_eq!(head.output_bytes, 496640);
            }
        }
    }
    Ok(())
}

#[test]
fn fa_uses_the_original_appended_private_read_and_never_a_fake_write() -> Result<()> {
    use zerogravity_core::accelerator::program::{ConvDomain, StateEffect, StateVersion};
    let source = SourceIdentity::for_new_allocation()?;
    let call = ShortHfCall::new(
        FullModelProfile::Qwen2B,
        ShortHfSpan::prefill(source, ShortHfWorkload::Prompt67, &[11; 67])?,
    )?;
    for layer in [3, 7, 11, 15, 19, 23] {
        let descriptor = Vllm08Operator::new(
            call.as_vllm_call().clone(),
            Some(call.as_vllm_call().layer(layer)?),
            Vllm08Kind::FlashAttention,
            Vec::new(),
            Vec::new(),
        )?;
        let read = StateVersion {
            slot: StateSlot::Kv(KvDomain(layer)),
            ordinal: 1,
        };
        let actual = PlanNode::new(
            "actual FA read".into(),
            PlanOperation::TypedVllm08 {
                descriptor: Box::new(descriptor.clone()),
                state: Some(read),
            },
            vec![ValueId(1)],
            vec![ValueId(2)],
            None,
        );
        require_state_read(&actual, &descriptor)?;
        for bad in [
            None,
            Some(StateVersion { ordinal: 0, ..read }),
            Some(StateVersion { ordinal: 2, ..read }),
            Some(StateVersion {
                slot: StateSlot::Kv(KvDomain((layer + 4) % 24)),
                ..read
            }),
            Some(StateVersion {
                slot: StateSlot::Conv(ConvDomain(layer)),
                ..read
            }),
        ] {
            let mut node = actual.clone();
            if let PlanOperation::TypedVllm08 { state, .. } = &mut node.operation {
                *state = bad;
            }
            assert!(require_state_read(&node, &descriptor).is_err());
        }
        let mut fake_write = actual.clone();
        fake_write.state = Some(StateEffect {
            read: StateVersion { ordinal: 0, ..read },
            write: read,
        });
        assert!(require_state_read(&fake_write, &descriptor).is_err());
    }
    Ok(())
}
