//! Source metadata fixtures; no trained values, GPU owners or permits.
use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::KvDomain;
use zerogravity_model_qwen35::plan::PromptSpan;
use zerogravity_model_qwen35::plan::PromptWorkload;
use zerogravity_model_qwen35::plan::QwenSourceProfile;
use zerogravity_model_qwen35::plan::{PromptCall, QwenOperation, QwenOperator};

#[test]
fn actual_last_prompt_row_and_both_source_widths_are_not_legacy_case_retags() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        let row = profile.hidden_size() * 2;
        for count in [47, 57, 62, 67, 72] {
            let workload = PromptWorkload::from_prompt_tokens(count)?;
            let span = PromptSpan::prefill(source, workload, &vec![11; count])?;
            let call = PromptCall::new(profile, span)?;
            let head = TextHeadView::new(
                call.span().len(),
                profile.hidden_size(),
                248320,
                call.span().len() - 1,
            )?;
            assert_eq!(
                head,
                TextHeadView {
                    full_bytes: count * row,
                    offset: (count - 1) * row,
                    selected_bytes: row,
                    output_bytes: 496640
                }
            );
            assert!(call.request().reference_phase().is_err());
            for base in workload.cached_bases() {
                let call =
                    PromptCall::new(profile, PromptSpan::cached(source, workload, base, 353)?)?;
                let head = TextHeadView::new(
                    call.span().len(),
                    profile.hidden_size(),
                    248320,
                    call.span().len() - 1,
                )?;
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
    let call = PromptCall::new(
        QwenSourceProfile::Qwen2B,
        PromptSpan::prefill(
            source,
            PromptWorkload::from_prompt_tokens(67).unwrap(),
            &[11; 67],
        )?,
    )?;
    for layer in [3, 7, 11, 15, 19, 23] {
        let descriptor = QwenOperator::new(
            call.as_source_call().clone(),
            Some(call.as_source_call().layer(layer)?),
            QwenOperation::FlashAttention,
            Vec::new(),
            Vec::new(),
        )?;
        let read = StateVersion {
            slot: StateSlot::Kv(KvDomain(layer)),
            ordinal: 1,
        };
        let actual = PlanNode::new(
            "actual FA read".into(),
            PlanOperation::SourceOperator {
                descriptor: Box::new(descriptor.mathematical_operator()?),
                state: Some(read),
            },
            vec![ValueId(1)],
            vec![ValueId(2)],
            None,
        );
        require_state_read(&actual, &descriptor.mathematical_operator()?)?;
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
            if let PlanOperation::SourceOperator { state, .. } = &mut node.operation {
                *state = bad;
            }
            assert!(require_state_read(&node, &descriptor.mathematical_operator()?).is_err());
        }
        let mut fake_write = actual.clone();
        fake_write.state = Some(StateEffect {
            read: StateVersion { ordinal: 0, ..read },
            write: read,
        });
        assert!(require_state_read(&fake_write, &descriptor.mathematical_operator()?).is_err());
    }
    Ok(())
}
