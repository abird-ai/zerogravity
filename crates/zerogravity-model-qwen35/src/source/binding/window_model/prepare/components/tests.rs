//! Relocated component source/issuer/arithmetic refusals; no device qualification.
use super::*;
mod fixed_prompt_tests {
    // Metadata rejection tests; actual source/lease/physical gates remain separate.
    use super::ComponentImplementation as A;
    use super::*;
    use crate::plan::{QwenGdnCall, QwenInputRows, QwenPrefixLayer};
    use zerogravity_core::accelerator::{SourceIdentity, program::TeacherSpan};
    fn request(ids: &[u32], base: usize, capacity: usize, layer: usize) -> QwenGdnCall {
        let source = SourceIdentity::for_new_allocation().unwrap();
        QwenGdnCall::from_span(
            QwenPrefixLayer::new(source, layer).unwrap(),
            SourceCallMode::Prefill,
            QwenInputRows::from_span(
                TeacherSpan::new(source, base, capacity, 248320, ids).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn finite47_request_is_not_legacy64_or_plain_tuple_or_another_prompt() {
        let kind = ModelComponentKind::FixedPrompt(A::OwnedFixedPromptSolve4);
        let exact = request(&FIXED_PROMPT_IDS, 0, 512, 0);
        kind.require_request(&exact).unwrap();
        assert!(exact.reference_phase().is_err());
        assert!(ModelComponentKind::Qwen2b.require_request(&exact).is_err());
        assert!(
            ModelComponentKind::FixedPrompt(A::OriginalPlainSolve2)
                .require_request(&exact)
                .is_err()
        );
        let mut changed = FIXED_PROMPT_IDS;
        changed[12] = 353;
        for invalid in [
            request(&changed, 0, 512, 0),
            request(&[11; 64], 0, 512, 0),
            request(&FIXED_PROMPT_IDS, 1, 512, 0),
            request(&FIXED_PROMPT_IDS, 0, 1024, 0),
            request(&FIXED_PROMPT_IDS, 0, 512, 1),
        ] {
            assert!(kind.require_request(&invalid).is_err());
        }
    }
    #[test]
    fn old_source_family_and_equal_id_new_teacher_issuer_remain_distinct() {
        let old = request(&[11; 64], 0, 512, 0);
        ModelComponentKind::Qwen2b.require_request(&old).unwrap();
        assert_eq!(
            ModelComponentKind::Qwen2b.profile(),
            QwenSourceProfile::Qwen2B
        );
        assert_eq!(ModelComponentKind::Qwen2b.columns(), 2048);
        let new = ModelComponentKind::FixedPrompt(A::OwnedFixedPromptSolve4);
        assert_eq!(new.profile(), QwenSourceProfile::Qwen08);
        assert_eq!(new.columns(), 1024);
        assert!(new.require_request(&old).is_err());
        let a = request(&FIXED_PROMPT_IDS, 0, 512, 0);
        let b = QwenGdnCall::from_span(
            a.layer(),
            a.mode(),
            QwenInputRows::from_span(
                TeacherSpan::new(a.layer().source(), 0, 512, 248320, &FIXED_PROMPT_IDS).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(a.teachers().ids(), b.teachers().ids());
        assert_ne!(a.teachers(), b.teachers());
        assert_ne!(a, b);
    }

    #[test]
    fn text_issuer_cannot_reuse_initial47_component_kind() {
        use crate::plan::{PromptCall, PromptSpan, PromptWorkload};
        let source = SourceIdentity::for_new_allocation().unwrap();
        let kind = ModelComponentKind::FixedPrompt(A::OwnedFixedPromptSolve4);
        kind.require_request(&request(&FIXED_PROMPT_IDS, 0, 512, 0))
            .unwrap();
        let short = PromptCall::new(
            QwenSourceProfile::Qwen08,
            PromptSpan::prefill(
                source,
                PromptWorkload::from_prompt_tokens(47).unwrap(),
                &FIXED_PROMPT_IDS,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(short.request().teachers().ids(), FIXED_PROMPT_IDS);
        assert!(short.request().reference_phase().is_err());
        assert!(matches!(
            kind.require_request(short.request()),
            Err(Error::Unsupported(_))
        ));
        for base in 47..=53 {
            let short = PromptCall::new(
                QwenSourceProfile::Qwen08,
                PromptSpan::cached(
                    source,
                    PromptWorkload::from_prompt_tokens(47).unwrap(),
                    base,
                    11,
                )
                .unwrap(),
            )
            .unwrap();
            assert!(matches!(
                kind.require_request(short.request()),
                Err(Error::Unsupported(_))
            ));
        }
    }
}
mod explicit2b_post_attention_tests {
    // Metadata rejection tests; original source and physical tests remain separate.
    use super::ComponentImplementation as FixedPromptGdnArithmetic;
    use super::ComponentImplementation as Qwen2bGdnArithmetic;
    use super::*;
    use crate::plan::{QwenGdnCall, QwenInputRows, QwenPrefixLayer};
    use zerogravity_core::accelerator::{SourceIdentity, program::TeacherSpan};

    fn request(
        rows: usize,
        base: usize,
        capacity: usize,
        layer: usize,
        mode: SourceCallMode,
    ) -> QwenGdnCall {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let ids = vec![11; rows];
        let teachers = if rows == 1 {
            QwenInputRows::new(source, base, capacity, &ids).unwrap()
        } else {
            QwenInputRows::from_span(
                TeacherSpan::new(source, base, capacity, 248320, &ids).unwrap(),
            )
            .unwrap()
        };
        QwenGdnCall::from_span(QwenPrefixLayer::new(source, layer).unwrap(), mode, teachers)
            .unwrap()
    }
    #[test]
    fn explicit_tuple_extends_only_initial2b_gated_to_post_attention() {
        let owned = Qwen2bGdnArithmetic::OwnedSolve4Recompute3;
        assert_eq!(
            ModelComponentKind::Qwen2b.profile(),
            QwenSourceProfile::Qwen2B
        );
        assert_eq!(ModelComponentKind::Qwen2b.columns(), 2048);
        assert_eq!(ModelComponentCut::PostAttention.node_end(), 8);
        assert_eq!(ModelComponentCut::PostAttention.value_end(), 10);
        for rows in [1, 64] {
            for cut in [ModelComponentCut::Gated, ModelComponentCut::PostAttention] {
                require_qwen2b_arithmetic(ModelComponentKind::Qwen2b, cut, rows, owned).unwrap();
            }
            assert!(
                require_qwen2b_arithmetic(
                    ModelComponentKind::Qwen2b,
                    ModelComponentCut::Mlp,
                    rows,
                    owned
                )
                .is_err()
            );
        }
        for rows in [0, 2, 47, 63, 65, 512] {
            assert!(
                require_qwen2b_arithmetic(
                    ModelComponentKind::Qwen2b,
                    ModelComponentCut::PostAttention,
                    rows,
                    owned
                )
                .is_err()
            );
        }
        assert!(
            require_qwen2b_arithmetic(
                ModelComponentKind::FixedPrompt(FixedPromptGdnArithmetic::OwnedFixedPromptSolve4),
                ModelComponentCut::PostAttention,
                47,
                owned,
            )
            .is_err()
        );
    }
    #[test]
    fn exact_request_still_requires_initial_layer0_base0_capacity512() {
        for rows in [1, 64] {
            ModelComponentKind::Qwen2b
                .require_request(&request(rows, 0, 512, 0, SourceCallMode::Prefill))
                .unwrap();
        }
        for wrong in [
            request(47, 0, 512, 0, SourceCallMode::Prefill),
            request(1, 1, 512, 0, SourceCallMode::Prefill),
            request(64, 0, 1024, 0, SourceCallMode::Prefill),
            request(64, 0, 512, 1, SourceCallMode::Prefill),
            request(1, 64, 512, 0, SourceCallMode::Decode),
        ] {
            assert!(ModelComponentKind::Qwen2b.require_request(&wrong).is_err());
        }
    }
    #[test]
    fn same_ids_do_not_replace_the_original_teacher_issuer() {
        let original = request(64, 0, 512, 0, SourceCallMode::Prefill);
        let replacement = QwenGdnCall::from_span(
            original.layer(),
            original.mode(),
            QwenInputRows::from_span(
                TeacherSpan::new(
                    original.layer().source(),
                    0,
                    512,
                    248320,
                    original.teachers().ids(),
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(original.teachers().ids(), replacement.teachers().ids());
        assert_ne!(original.teachers(), replacement.teachers());
        assert_ne!(original, replacement);
    }
}

#[cfg(test)]
mod mlp_cut_tests {
    use super::*;
    #[test]
    fn mlp_is_additive47_only_with_distinct_stop_and_value_extent() {
        assert_eq!(
            (
                ModelComponentCut::Mlp.node_end(),
                ModelComponentCut::Mlp.value_end()
            ),
            (11, 13)
        );
        assert_eq!(ModelComponentCut::Mlp.provider_bytes(), 224 * 1024 * 1024);
        assert!(
            ModelComponentCut::Mlp
                .require_kind(ModelComponentKind::Qwen2b)
                .is_err()
        );
        assert!(
            ModelComponentCut::Mlp
                .require_kind(ModelComponentKind::FixedPrompt(
                    ComponentImplementation::OwnedFixedPromptSolve4
                ))
                .is_ok()
        );
        assert_eq!(
            (
                ModelComponentCut::Gated.node_end(),
                ModelComponentCut::Gated.value_end()
            ),
            (6, 7)
        );
        assert_eq!(
            (
                ModelComponentCut::PostAttention.node_end(),
                ModelComponentCut::PostAttention.value_end()
            ),
            (8, 10)
        );
    }
}

#[cfg(test)]
mod qwen2b_mlp_scope_tests {
    use super::*;
    #[test]
    fn new_mlp_cut_keeps_old47_and_six_eight_node_guards() {
        let cut = ModelComponentCut::Qwen2bMlp;
        assert_eq!(
            (cut.node_end(), cut.value_end(), cut.provider_bytes()),
            (11, 13, 320 * 1024 * 1024)
        );
        assert!(cut.includes_mlp() && cut.includes_post_attention());
        assert!(cut.require_kind(ModelComponentKind::Qwen2b).is_ok());
        let old = ModelComponentKind::FixedPrompt(ComponentImplementation::OwnedFixedPromptSolve4);
        assert!(cut.require_kind(old).is_err());
        assert!(
            ModelComponentCut::Mlp
                .require_kind(ModelComponentKind::Qwen2b)
                .is_err()
        );
        assert_eq!(ModelComponentCut::Mlp.provider_bytes(), 224 * 1024 * 1024);
        assert_eq!(
            (
                ModelComponentCut::Gated.node_end(),
                ModelComponentCut::Gated.value_end()
            ),
            (6, 7)
        );
        assert_eq!(
            (
                ModelComponentCut::PostAttention.node_end(),
                ModelComponentCut::PostAttention.value_end()
            ),
            (8, 10)
        );
    }
    #[test]
    fn arithmetic_seal_does_not_retag_old47_or_accept_new_rows() {
        let a = ComponentImplementation::OwnedSolve4Recompute3;
        for rows in [1, 64] {
            assert!(
                require_qwen2b_arithmetic(
                    ModelComponentKind::Qwen2b,
                    ModelComponentCut::Qwen2bMlp,
                    rows,
                    a
                )
                .is_ok()
            );
        }
        for rows in [0, 2, 47, 65] {
            assert!(
                require_qwen2b_arithmetic(
                    ModelComponentKind::Qwen2b,
                    ModelComponentCut::Qwen2bMlp,
                    rows,
                    a
                )
                .is_err()
            );
        }
        assert!(
            require_qwen2b_arithmetic(ModelComponentKind::Qwen2b, ModelComponentCut::Mlp, 1, a)
                .is_err()
        );
    }
}

mod weight_source_policy {
    // Pure geometry/issuer rejection; no source/native qualification.
    use super::*;
    use crate::plan::QwenInputRows;
    use crate::plan::QwenPrefixLayer;
    use crate::plan::{QwenGdnCall, QwenSourceProfile};
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::TeacherSpan;
    fn request(
        rows: usize,
        base: usize,
        capacity: usize,
        layer: usize,
        mode: SourceCallMode,
    ) -> QwenGdnCall {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let ids = vec![11; rows];
        let teachers = if rows == 1 {
            QwenInputRows::new(source, base, capacity, &ids).unwrap()
        } else {
            QwenInputRows::from_span(
                TeacherSpan::new(source, base, capacity, 248320, &ids).unwrap(),
            )
            .unwrap()
        };
        QwenGdnCall::from_span(QwenPrefixLayer::new(source, layer).unwrap(), mode, teachers)
            .unwrap()
    }
    #[test]
    fn finite_source_family_initial_span_rejects_other_phase_geometry() {
        for (rows, phase) in [
            (1, ReferencePhase::InitialOne),
            (64, ReferencePhase::Initial64),
        ] {
            let req = request(rows, 0, 512, 0, SourceCallMode::Prefill);
            assert_eq!(
                require_weight_request(QwenSourceProfile::Qwen2B, &req).unwrap(),
                phase
            );
            assert!(require_weight_request(QwenSourceProfile::Qwen08, &req).is_err());
        }
        for req in [
            request(47, 0, 512, 0, SourceCallMode::Prefill),
            request(64, 0, 512, 1, SourceCallMode::Prefill),
            request(1, 1, 512, 0, SourceCallMode::Prefill),
            request(64, 0, 1024, 0, SourceCallMode::Prefill),
            request(1, 64, 512, 0, SourceCallMode::Decode),
        ] {
            assert!(require_weight_request(QwenSourceProfile::Qwen2B, &req).is_err());
        }
    }
    #[test]
    fn matching_teacher_words_do_not_replace_actual_invocation_issuer() {
        let req = request(64, 0, 512, 0, SourceCallMode::Prefill);
        let replacement = QwenGdnCall::from_span(
            req.layer(),
            req.mode(),
            QwenInputRows::from_span(
                TeacherSpan::new(req.layer().source(), 0, 512, 248320, req.teachers().ids())
                    .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(req.teachers().ids(), replacement.teachers().ids());
        assert_ne!(req, replacement);
    }
}

mod projection_source_policy {
    // Structural metadata only; no trained tensor or native qualification.
    use super::ModelProjectionRole as Role;
    use super::*;
    const PROMPT: [u32; 47] = FIXED_PROMPT_IDS;
    fn require_same_operator(a: &QwenOperator, b: &QwenOperator) -> Result<()> {
        if a != b {
            return Err(Error::State("actual source/teacher issuer differs".into()));
        }
        Ok(())
    }
    use crate::plan::QwenGdnCall;
    use crate::plan::QwenInputRows;
    use crate::plan::QwenPrefixLayer;
    use crate::plan::QwenSourceCall;
    use crate::plan::{QwenOperation, QwenOperator, QwenProjection, QwenSourceProfile};
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::MatrixAxes;
    use zerogravity_core::accelerator::program::SourceTensor;
    use zerogravity_core::accelerator::program::TeacherSpan;
    use zerogravity_core::quant::QuantKind;
    fn call(ids: &[u32], base: usize) -> QwenSourceCall {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let teachers = if ids.len() == 1 {
            QwenInputRows::new(source, base, 512, ids).unwrap()
        } else {
            QwenInputRows::from_span(TeacherSpan::new(source, base, 512, 248_320, ids).unwrap())
                .unwrap()
        };
        QwenSourceCall::from_span(
            QwenGdnCall::from_span(
                QwenPrefixLayer::new(source, 0).unwrap(),
                SourceCallMode::Prefill,
                teachers,
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn operator(call: &QwenSourceCall, projection: QwenProjection, layer: usize) -> QwenOperator {
        let members = match projection {
            QwenProjection::Qkvz => [("in_proj_qkv", 6144), ("in_proj_z", 2048)],
            QwenProjection::Ba => [("in_proj_b", 16), ("in_proj_a", 16)],
            _ => unreachable!(),
        };
        let matrices = members
            .into_iter()
            .map(|(name, rows)| SourceTensor {
                source: call.source(),
                name: format!("model.language_model.layers.{layer}.linear_attn.{name}.weight"),
                kind: QuantKind::Bf16,
                axes: MatrixAxes::OutputInput,
                rows,
                columns: 1024,
            })
            .collect();
        QwenOperator::new(
            call.clone(),
            Some(call.layer(layer).unwrap()),
            QwenOperation::Projection(projection),
            matrices,
            Vec::new(),
        )
        .unwrap()
    }
    #[test]
    fn fixed_prompt_role_requires_actual_span_prompt_and_layer_without_legacy_retag() {
        let good = call(&PROMPT, 0);
        assert!(good.request().reference_phase().is_err());
        assert_eq!(
            require_component_projection(&operator(&good, QwenProjection::Qkvz, 0)).unwrap(),
            Role::Qkvz
        );
        assert_eq!(
            require_component_projection(&operator(&good, QwenProjection::Ba, 0)).unwrap(),
            Role::Ba
        );
        assert!(require_component_projection(&operator(&good, QwenProjection::Ba, 1)).is_err());
        let mut wrong = PROMPT;
        wrong[46] = 11;
        assert!(
            require_component_projection(&operator(&call(&wrong, 0), QwenProjection::Qkvz, 0))
                .is_err()
        );
        assert!(
            require_component_projection(&operator(&call(&PROMPT, 1), QwenProjection::Ba, 0))
                .is_err()
        );
        for ids in [&PROMPT[..1], &PROMPT[..2], &PROMPT[..46]] {
            assert!(
                require_component_projection(&operator(&call(ids, 0), QwenProjection::Ba, 0))
                    .is_err()
            );
        }
    }
    #[test]
    fn fixed_prompt_equal_ids_new_teacher_issuer_cannot_reuse_prepared_weights() {
        let original = call(&PROMPT, 0);
        let source = original.source();
        let foreign = QwenSourceCall::from_span(
            QwenGdnCall::from_span(
                QwenPrefixLayer::new(source, 0).unwrap(),
                SourceCallMode::Prefill,
                QwenInputRows::from_span(
                    TeacherSpan::new(source, 0, 512, 248_320, &PROMPT).unwrap(),
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        for projection in [QwenProjection::Qkvz, QwenProjection::Ba] {
            let a = operator(&original, projection, 0);
            let b = operator(&foreign, projection, 0);
            assert_eq!(a.matrices(), b.matrices());
            assert_ne!(a.call(), b.call());
            assert!(require_same_operator(&a, &a.clone()).is_ok());
            assert!(require_same_operator(&a, &b).is_err());
        }
    }

    #[test]
    fn text_issuer_cannot_reuse_original47_projection_roles() {
        use crate::plan::PromptCall;
        use crate::plan::PromptSpan;
        use crate::plan::PromptWorkload;
        let source = SourceIdentity::for_new_allocation().unwrap();
        let short = PromptCall::new(
            QwenSourceProfile::Qwen08,
            PromptSpan::prefill(
                source,
                PromptWorkload::from_prompt_tokens(47).unwrap(),
                &PROMPT,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(short.request().teachers().ids(), PROMPT);
        assert!(short.request().reference_phase().is_err());
        for projection in [QwenProjection::Qkvz, QwenProjection::Ba] {
            assert!(
                require_component_projection(&operator(&call(&PROMPT, 0), projection, 0)).is_ok()
            );
            assert!(matches!(
                require_component_projection(&operator(short.as_source_call(), projection, 0)),
                Err(Error::Unsupported(_))
            ));
        }
        for base in 47..=53 {
            let short = PromptCall::new(
                QwenSourceProfile::Qwen08,
                PromptSpan::cached(
                    source,
                    PromptWorkload::from_prompt_tokens(47).unwrap(),
                    base,
                    11,
                )
                .unwrap(),
            )
            .unwrap();
            assert!(matches!(
                require_component_projection(&operator(
                    short.as_source_call(),
                    QwenProjection::Qkvz,
                    0
                )),
                Err(Error::Unsupported(_))
            ));
        }
    }
}
