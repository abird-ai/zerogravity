//! Actual common graph fixtures; no trained source or physical permit.
use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_model_qwen35::plan::{PromptWorkload, QwenSourceProfile};
#[path = "fixture.rs"]
mod fixture;
fn selection(record: &zerogravity_model_qwen35::plan::QwenPlan) -> SourceSelection {
    SourceSelection::new(
        record.logits(),
        None,
        Some(record.normalized()),
        record.boundaries().to_vec().into_boxed_slice(),
        Some(record.selected_row()),
        24,
        None,
    )
}
#[test]
fn actual_source_sites_and_head_values_survive_only_same_math_rebinding() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        for prompt_tokens in [47, 57, 62, 67, 72] {
            let workload = PromptWorkload::from_prompt_tokens(prompt_tokens)?;
            for base in std::iter::once(None).chain(workload.cached_bases().map(Some)) {
                let first = fixture::call(source, profile, workload, base, 11);
                let template = fixture::record(&first);
                let selected = selection(&template);
                let analysis = analyze_plan(template.plan(), &selected)?;
                assert_eq!(
                    (
                        analysis.rms.len(),
                        analysis.projections.len(),
                        analysis.gdn.len(),
                        analysis.attention.len()
                    ),
                    (49, 114, 18, 6)
                );
                assert!(PreparedTextModel::metadata_bytes(&analysis)? <= 512 * 1024);
                for id in [353, 353, 78] {
                    let current = fixture::call(source, profile, workload, base, id);
                    let bound = template.bind_prompt_call(&current)?;
                    assert!(template.plan().shares_semantics(bound.plan()));
                    assert_ne!(template.plan().source_call(), bound.plan().source_call());
                    let actual = analyze_plan(bound.plan(), &selected)?;
                    assert_eq!(analysis.head, actual.head);
                    for (&domain, site) in &actual.gdn {
                        assert!(matches!(operator(bound.plan(), site.nodes[0])?.kind(),
                            SourceOperatorKind::CausalConvSplit { domain: d, .. } if d.0 == domain));
                        assert_eq!(
                            Some(operator(bound.plan(), site.nodes[0])?.call()),
                            bound.plan().source_call()
                        );
                    }
                    for site in actual.attention.values() {
                        let node = &bound.plan().nodes()[site.nodes[3]];
                        require_state_read(node, operator(bound.plan(), site.nodes[3])?)?;
                    }
                }
                assert!(
                    !template
                        .plan()
                        .shares_semantics(fixture::record(&first).plan())
                );
            }
        }
    }
    Ok(())
}
#[test]
fn selected_head_cannot_replace_actual_normalized_input_row_or_producer() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let call = fixture::call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47)?,
        None,
        11,
    );
    let record = fixture::record(&call);
    for wrong in [
        SourceSelection::new(
            ValueId(327),
            None,
            Some(record.normalized()),
            Box::new([]),
            Some(46),
            24,
            None,
        ),
        SourceSelection::new(
            record.logits(),
            None,
            Some(ValueId(0)),
            Box::new([]),
            Some(46),
            24,
            None,
        ),
        SourceSelection::new(
            record.logits(),
            None,
            Some(record.normalized()),
            Box::new([]),
            Some(45),
            24,
            None,
        ),
        SourceSelection::new(
            record.logits(),
            None,
            None,
            Box::new([]),
            Some(46),
            24,
            None,
        ),
    ] {
        assert!(analyze_plan(record.plan(), &wrong).is_err());
    }
    Ok(())
}
