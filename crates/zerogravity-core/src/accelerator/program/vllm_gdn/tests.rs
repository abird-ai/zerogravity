use super::*;
fn call(rows: usize) -> VllmBf16Invocation {
    let source = SourceIdentity::for_new_allocation().unwrap();
    VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 0).unwrap(),
        if rows == 1 {
            VllmBf16Phase::InitialOne
        } else {
            VllmBf16Phase::Initial64
        },
        VllmBf16Teachers::new(source, 0, 512, &vec![11; rows]).unwrap(),
    )
    .unwrap()
}
fn parameter(
    call: &VllmBf16Invocation,
    suffix: &str,
    kind: QuantKind,
    shape: &[u64],
) -> SourceParameter {
    SourceParameter {
        source: call.layer().source(),
        name: format!("model.language_model.layers.0.{suffix}"),
        kind,
        shape: shape.to_vec(),
        bytes: shape.iter().product::<u64>() as usize * if kind == QuantKind::Bf16 { 2 } else { 4 },
    }
}
fn embedding(call: &VllmBf16Invocation) -> VllmBf16Embedding {
    VllmBf16Embedding::new(
        SourceTensor {
            source: call.layer().source(),
            name: "model.language_model.embed_tokens.weight".into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            columns: 1024,
            rows: 248320,
        },
        call.teachers().clone(),
    )
    .unwrap()
}
fn projection(call: &VllmBf16Invocation, kind: VllmGdnProjectionKind) -> VllmGdnProjection {
    let members: &[(&str, usize)] = match kind {
        VllmGdnProjectionKind::Qkvz => &[("qkv", 6144), ("z", 2048)],
        VllmGdnProjectionKind::Ba => &[("b", 16), ("a", 16)],
        _ => unreachable!(),
    };
    VllmGdnProjection::new(
        call.clone(),
        kind,
        members
            .iter()
            .map(|&(role, rows)| SourceTensor {
                source: call.layer().source(),
                name: format!("model.language_model.layers.0.linear_attn.in_proj_{role}.weight"),
                kind: QuantKind::Bf16,
                axes: MatrixAxes::OutputInput,
                columns: 1024,
                rows,
            })
            .collect(),
    )
    .unwrap()
}
fn recurrent(call: &VllmBf16Invocation) -> VllmGdnRecurrent {
    VllmGdnRecurrent::new(
        call.clone(),
        [
            parameter(call, "linear_attn.A_log", QuantKind::F32, &[16]),
            parameter(call, "linear_attn.dt_bias", QuantKind::Bf16, &[16]),
            parameter(call, "linear_attn.norm.weight", QuantKind::F32, &[128]),
        ],
    )
    .unwrap()
}
fn prefix(call: &VllmBf16Invocation) -> (PlanBuilder, PlanValue, PlanValue) {
    let mut b = PlanBuilder::new_vllm_gdn_fragment(call.clone()).unwrap();
    let raw = b.vllm_bf16_source_embedding(&embedding(call)).unwrap();
    let rms = VllmInputRms::new(
        call.clone(),
        parameter(call, "input_layernorm.weight", QuantKind::Bf16, &[1024]),
    )
    .unwrap();
    let normalized = b.vllm_input_rms(&rms, &raw).unwrap();
    assert_ne!(raw.id, normalized.id);
    assert_eq!(
        b.vllm_value_provenance(&raw).unwrap().unwrap().origin(),
        &VllmValueOrigin::SourceRows
    );
    assert_eq!(
        b.vllm_value_provenance(&normalized)
            .unwrap()
            .unwrap()
            .origin(),
        &VllmValueOrigin::InputNormalized {
            layer: call.layer()
        }
    );
    let qkvz = b
        .vllm_gdn_project(&projection(call, VllmGdnProjectionKind::Qkvz), &normalized)
        .unwrap();
    let ba = b
        .vllm_gdn_project(&projection(call, VllmGdnProjectionKind::Ba), &normalized)
        .unwrap();
    (b, qkvz, ba)
}
#[test]
fn original_initial_fragment_has_two_real_effects_forty_carries_and_no_publication() {
    for rows in [1, 64] {
        let c = call(rows);
        let (mut b, qkvz, ba) = prefix(&c);
        let d = VllmGdnConv::new(
            c.clone(),
            parameter(
                &c,
                "linear_attn.conv1d.weight",
                QuantKind::Bf16,
                &[6144, 1, 4],
            ),
        )
        .unwrap();
        let out = b.vllm_gdn_conv(&d, &qkvz, &ba).unwrap();
        assert_eq!(out.activated.elements, rows * 6144);
        assert_eq!(out.z.elements, rows * 2048);
        let gated = b
            .vllm_gdn_recurrent(&recurrent(&c), &out.activated, &out.z, &ba)
            .unwrap();
        assert!(b.advance_frontier().is_err());
        let record = b.finish_vllm_gdn_fragment(gated).unwrap();
        let p = record.plan();
        assert_eq!(p.required_domains().len(), 42);
        assert!(p.required_domains().contains(&StateSlot::Kv(KvDomain(3))));
        assert!(p.required_domains().contains(&StateSlot::Kv(KvDomain(23))));
        assert!(
            p.required_domains()
                .contains(&StateSlot::Recurrent(RecurrentDomain(22)))
        );
        assert!(!p.required_domains().contains(&StateSlot::Kv(KvDomain(0))));
        assert!(
            !p.required_domains()
                .contains(&StateSlot::Conv(ConvDomain(3)))
        );
        assert_eq!(p.proposed_domains().len(), 42);
        assert_eq!(p.frontier_delta(), 0);
        assert!(p.logits().is_none());
        assert_eq!(p.vllm_gdn_fragment(), Some(&c));
        assert_eq!(
            p.nodes()
                .iter()
                .filter(|n| matches!(n.operation, PlanOperation::TypedVllmStateCarry))
                .count(),
            40
        );
        assert_eq!(p.nodes().iter().filter(|n| n.state.is_some()).count(), 42);
        assert!(p.proposed_domains().iter().all(|v| v.ordinal == 1));
    }
}
#[test]
fn role_cohort_alias_faults_leave_no_partial_effects() {
    let c = call(1);
    let (mut b, qkvz, ba) = prefix(&c);
    let d = VllmGdnConv::new(
        c.clone(),
        parameter(
            &c,
            "linear_attn.conv1d.weight",
            QuantKind::Bf16,
            &[6144, 1, 4],
        ),
    )
    .unwrap();
    let before = (b.nodes.len(), b.values.len(), b.domains.len());
    let packed = b.contiguous(&qkvz).unwrap();
    let after_view = (b.nodes.len(), b.values.len(), b.domains.len());
    assert!(b.vllm_gdn_conv(&d, &packed, &ba).is_err());
    assert_eq!((b.nodes.len(), b.values.len(), b.domains.len()), after_view);
    assert!(b.vllm_gdn_conv(&d, &ba, &qkvz).is_err());
    let out = b.vllm_gdn_conv(&d, &qkvz, &ba).unwrap();
    assert!(b.nodes.len() > before.0);
    let new_source = b.vllm_bf16_source_embedding(&embedding(&c)).unwrap();
    let other_ba = b
        .vllm_gdn_project(&projection(&c, VllmGdnProjectionKind::Ba), &new_source)
        .unwrap();
    let before = (b.nodes.len(), b.values.len(), b.domains.len());
    assert!(
        b.vllm_gdn_recurrent(&recurrent(&c), &out.activated, &out.z, &other_ba)
            .is_err()
    );
    assert_eq!((b.nodes.len(), b.values.len(), b.domains.len()), before);
    assert!(
        b.vllm_gdn_recurrent(&recurrent(&c), &out.z, &out.activated, &ba)
            .is_err()
    );
    assert_eq!((b.nodes.len(), b.values.len(), b.domains.len()), before);
    assert!(b.vllm_gdn_conv(&d, &qkvz, &ba).is_err());
    assert_eq!(
        b.current(StateSlot::Conv(ConvDomain(0))).unwrap().ordinal,
        1
    );
}
#[test]
fn exact_original_parameters_and_initial_scope_cannot_retag_dtype_layer_or_phase() {
    let c = call(1);
    let raw = parameter(
        &c,
        "linear_attn.conv1d.weight",
        QuantKind::Bf16,
        &[6144, 1, 4],
    );
    for fault in 0..5 {
        let mut p = raw.clone();
        match fault {
            0 => p.kind = QuantKind::F32,
            1 => p.shape = vec![6144, 4, 1],
            2 => p.bytes -= 1,
            3 => p.source = SourceIdentity::for_new_allocation().unwrap(),
            _ => p.name = p.name.replace("layers.0", "layers.1"),
        };
        assert!(VllmGdnConv::new(c.clone(), p).is_err());
    }
    let mut params = recurrent(&c).parameters().clone();
    params.swap(0, 2);
    assert!(VllmGdnRecurrent::new(c.clone(), params).is_err());
    let cached = VllmBf16Invocation::new(
        c.layer(),
        VllmBf16Phase::CachedOne,
        VllmBf16Teachers::new(c.layer().source(), 1, 512, &[353]).unwrap(),
    )
    .unwrap();
    assert!(PlanBuilder::new_vllm_gdn_fragment(cached).is_err());
    let mut b = PlanBuilder::new_vllm_gdn_fragment(c).unwrap();
    assert!(b.advance_frontier().is_err());
    assert!(b.finish(None).is_err());
}

#[test]
fn dual_rms_joins_same_teachers_and_phase_with_distinct_actual_producer_roles() {
    let c = call(1);
    let mut b = PlanBuilder::new_operator_slice(
        c.layer().source(),
        ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
    )
    .unwrap();
    let raw = b.vllm_bf16_source_embedding(&embedding(&c)).unwrap();
    let normalized = b
        .vllm_input_rms(
            &VllmInputRms::new(
                c.clone(),
                parameter(&c, "input_layernorm.weight", QuantKind::Bf16, &[1024]),
            )
            .unwrap(),
            &raw,
        )
        .unwrap();
    let output = b
        .residual_rms_profiled(
            parameter(
                &c,
                "post_attention_layernorm.weight",
                QuantKind::Bf16,
                &[1024],
            ),
            &normalized,
            &raw,
            ResidualRmsContract::Vllm031CudaAtenBf16H1024Rows1_2_64,
        )
        .unwrap();
    let n = b
        .vllm_value_provenance(&output.normalized)
        .unwrap()
        .unwrap();
    let r = b.vllm_value_provenance(&output.residual).unwrap().unwrap();
    assert_ne!(n.origin(), r.origin());
    n.require_invocation(&c).unwrap();
    r.require_invocation(&c).unwrap();
}
#[test]
fn equal_teacher_ids_cannot_remint_the_actual_conv_producer_issuer() {
    let c = call(1);
    let (mut b, qkvz, ba) = prefix(&c);
    let reminted = VllmBf16Invocation::new(
        c.layer(),
        c.legacy_phase().unwrap(),
        VllmBf16Teachers::new(c.layer().source(), 0, 512, c.teachers().ids()).unwrap(),
    )
    .unwrap();
    let foreign = b.vllm_bf16_source_embedding(&embedding(&reminted)).unwrap();
    let other_ba = b
        .vllm_gdn_project(&projection(&reminted, VllmGdnProjectionKind::Ba), &foreign)
        .unwrap();
    let d = VllmGdnConv::new(
        c.clone(),
        parameter(
            &c,
            "linear_attn.conv1d.weight",
            QuantKind::Bf16,
            &[6144, 1, 4],
        ),
    )
    .unwrap();
    let before = (b.nodes.len(), b.values.len(), b.domains.len());
    assert!(b.vllm_gdn_conv(&d, &qkvz, &other_ba).is_err());
    assert_eq!((b.nodes.len(), b.values.len(), b.domains.len()), before);
    assert_eq!(
        b.current(StateSlot::Conv(ConvDomain(0))).unwrap().ordinal,
        0
    );
    let out = b.vllm_gdn_conv(&d, &qkvz, &ba).unwrap();
    let packed_z = b.contiguous(&out.z).unwrap();
    let before = (b.nodes.len(), b.values.len(), b.domains.len());
    assert!(
        b.vllm_gdn_recurrent(&recurrent(&c), &out.activated, &packed_z, &ba)
            .is_err()
    );
    assert_eq!((b.nodes.len(), b.values.len(), b.domains.len()), before);
    assert_eq!(
        b.current(StateSlot::Recurrent(RecurrentDomain(0)))
            .unwrap()
            .ordinal,
        0
    );
}

fn complete_parameter(
    c: &VllmBf16Invocation,
    suffix: &str,
    kind: QuantKind,
    shape: &[u64],
) -> SourceParameter {
    let mut p = parameter(c, suffix, kind, shape);
    p.name = p
        .name
        .replace("layers.0", &format!("layers.{}", c.layer().index()));
    p
}
fn complete_projection(c: &VllmBf16Invocation, kind: VllmGdnProjectionKind) -> VllmGdnProjection {
    let parts: &[(&str, usize)] = match kind {
        VllmGdnProjectionKind::Qkvz => &[
            ("linear_attn.in_proj_qkv.weight", 6144),
            ("linear_attn.in_proj_z.weight", 2048),
        ],
        VllmGdnProjectionKind::Ba => &[
            ("linear_attn.in_proj_b.weight", 16),
            ("linear_attn.in_proj_a.weight", 16),
        ],
        VllmGdnProjectionKind::GdnOut => &[("linear_attn.out_proj.weight", 1024)],
        VllmGdnProjectionKind::GateUp => {
            &[("mlp.gate_proj.weight", 3584), ("mlp.up_proj.weight", 3584)]
        }
        VllmGdnProjectionKind::Down => &[("mlp.down_proj.weight", 1024)],
    };
    VllmGdnProjection::new(
        c.clone(),
        kind,
        parts
            .iter()
            .map(|(suffix, rows)| SourceTensor {
                source: c.layer().source(),
                name: format!("model.language_model.layers.{}.{suffix}", c.layer().index()),
                kind: QuantKind::Bf16,
                axes: MatrixAxes::OutputInput,
                columns: kind.input_width(),
                rows: *rows,
            })
            .collect(),
    )
    .unwrap()
}
fn complete_metadata(rows: usize, layers: usize) -> (PlanBuilder, PlanValue, PlanValue) {
    let initial = call(rows);
    let mut b = PlanBuilder::new_vllm_gdn_fragment(initial.clone()).unwrap();
    let raw = b.vllm_bf16_source_embedding(&embedding(&initial)).unwrap();
    let mut branch = raw;
    let mut residual = raw;
    for i in 0..layers {
        let layer = GdnPrefixLayer::new(initial.layer().source(), i).unwrap();
        let c = VllmBf16Invocation::new(
            layer,
            initial.legacy_phase().unwrap(),
            initial.teachers().clone(),
        )
        .unwrap();
        let (normal, carried) = if i == 0 {
            (
                b.vllm_input_rms(
                    &VllmInputRms::new(
                        c.clone(),
                        complete_parameter(&c, "input_layernorm.weight", QuantKind::Bf16, &[1024]),
                    )
                    .unwrap(),
                    &branch,
                )
                .unwrap(),
                raw,
            )
        } else {
            let pair = b
                .residual_rms_profiled(
                    complete_parameter(&c, "input_layernorm.weight", QuantKind::Bf16, &[1024]),
                    &branch,
                    &residual,
                    ResidualRmsContract::Vllm031CudaGdnPrefix(
                        layer.rms(GdnPrefixRmsSite::InputResidual).unwrap(),
                    ),
                )
                .unwrap();
            (pair.normalized, pair.residual)
        };
        let q = b
            .vllm_gdn_project(
                &complete_projection(&c, VllmGdnProjectionKind::Qkvz),
                &normal,
            )
            .unwrap();
        let ba = b
            .vllm_gdn_project(&complete_projection(&c, VllmGdnProjectionKind::Ba), &normal)
            .unwrap();
        let conv = b
            .vllm_gdn_conv(
                &VllmGdnConv::new(
                    c.clone(),
                    complete_parameter(
                        &c,
                        "linear_attn.conv1d.weight",
                        QuantKind::Bf16,
                        &[6144, 1, 4],
                    ),
                )
                .unwrap(),
                &q,
                &ba,
            )
            .unwrap();
        let gated = b
            .vllm_gdn_recurrent(
                &VllmGdnRecurrent::new(
                    c.clone(),
                    [
                        complete_parameter(&c, "linear_attn.A_log", QuantKind::F32, &[16]),
                        complete_parameter(&c, "linear_attn.dt_bias", QuantKind::Bf16, &[16]),
                        complete_parameter(&c, "linear_attn.norm.weight", QuantKind::F32, &[128]),
                    ],
                )
                .unwrap(),
                &conv.activated,
                &conv.z,
                &ba,
            )
            .unwrap();
        let out = b
            .vllm_gdn_project(
                &complete_projection(&c, VllmGdnProjectionKind::GdnOut),
                &gated,
            )
            .unwrap();
        let pair = b
            .residual_rms_profiled(
                complete_parameter(
                    &c,
                    "post_attention_layernorm.weight",
                    QuantKind::Bf16,
                    &[1024],
                ),
                &out,
                &carried,
                ResidualRmsContract::Vllm031CudaGdnPrefix(
                    layer.rms(GdnPrefixRmsSite::PostAttention).unwrap(),
                ),
            )
            .unwrap();
        let gate = b
            .vllm_gdn_project(
                &complete_projection(&c, VllmGdnProjectionKind::GateUp),
                &pair.normalized,
            )
            .unwrap();
        let act = b
            .vllm_silu_and_mul(&VllmSiluAndMul::new(c.clone()).unwrap(), &gate)
            .unwrap();
        branch = b
            .vllm_gdn_project(&complete_projection(&c, VllmGdnProjectionKind::Down), &act)
            .unwrap();
        residual = pair.residual;
    }
    (b, branch, residual)
}
#[test]
fn complete_prefix_preserves_pending_branch_and_residual_all42_and_initial_scope() {
    for rows in [1, 64] {
        for layers in [1, 3] {
            let (b, branch, residual) = complete_metadata(rows, layers);
            assert_ne!(branch.id, residual.id);
            let record = b
                .finish_vllm_gdn_mlp_fragment(branch, residual, layers)
                .unwrap();
            assert_eq!(record.completed_layers(), layers);
            assert_eq!(record.residual_output(), Some(residual.id));
            let p = record.plan();
            assert_eq!(p.required_domains().len(), 42);
            assert_eq!(p.proposed_domains().len(), 42);
            assert_eq!(p.frontier_delta(), 0);
            assert!(p.logits().is_none());
            assert_eq!(
                p.nodes()
                    .iter()
                    .filter(|n| matches!(n.operation, PlanOperation::TypedVllmStateCarry))
                    .count(),
                42 - 2 * layers
            );
            assert_eq!(
                p.nodes()
                    .iter()
                    .filter(|n| matches!(n.operation, PlanOperation::TypedResidualRms { .. }))
                    .count(),
                2 * layers - 1
            );
            assert_eq!(p.values().len(), 13 * layers);
            assert_eq!(
                p.nodes()
                    .iter()
                    .filter(|n| matches!(n.operation, PlanOperation::TypedVllmInputRms { .. }))
                    .count(),
                1
            );
            // No eager rounded addition can replace either pending branch/residual join.
            assert!(
                !p.nodes()
                    .iter()
                    .any(|n| matches!(n.operation, PlanOperation::Add))
            );
        }
    }
}
#[test]
fn prefix_finish_cannot_swap_outputs_or_invent_a_missing_layer_effect() {
    let (b, branch, residual) = complete_metadata(1, 1);
    assert!(b.finish_vllm_gdn_mlp_fragment(residual, branch, 1).is_err());
    let (b, branch, residual) = complete_metadata(1, 1);
    assert!(b.finish_vllm_gdn_mlp_fragment(branch, residual, 3).is_err());
    let (b, branch, residual) = complete_metadata(1, 3);
    assert!(b.finish_vllm_gdn_mlp_fragment(branch, residual, 1).is_err());
    let (b, branch, residual) = complete_metadata(64, 1);
    assert!(b.finish_vllm_gdn_mlp_fragment(branch, residual, 2).is_err());
}
#[test]
fn prefix_layer_change_does_not_remint_teacher_or_conv_producer_authority() {
    let (b, branch, residual) = complete_metadata(1, 1);
    let plan = b.finish_vllm_gdn_mlp_fragment(branch, residual, 1).unwrap();
    let root = plan.plan().vllm_gdn_fragment().unwrap();
    let remint = VllmBf16Invocation::new(
        GdnPrefixLayer::new(root.layer().source(), 1).unwrap(),
        root.legacy_phase().unwrap(),
        VllmBf16Teachers::new(root.layer().source(), 0, 512, root.teachers().ids()).unwrap(),
    )
    .unwrap();
    assert!(
        plan.plan()
            .vllm_value_provenance(branch.id)
            .unwrap()
            .unwrap()
            .require_invocation(&remint)
            .is_err()
    );
}

// Append inside the existing vllm_gdn/tests.rs module; metadata tests only.
#[test]
fn transaction_phase_records_preserve_all42_zero_frontier_and_old_initial_refusal() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for (phase, base, rows) in [
        (VllmBf16Phase::CachedOne, 1, 1),
        (VllmBf16Phase::ContinuedTwo, 64, 2),
        (VllmBf16Phase::CachedOne, 66, 1),
    ] {
        let c = VllmBf16Invocation::new(
            GdnPrefixLayer::new(source, 0).unwrap(),
            phase,
            VllmBf16Teachers::new(source, base, 512, &vec![353; rows]).unwrap(),
        )
        .unwrap();
        assert!(PlanBuilder::new_vllm_gdn_fragment(c.clone()).is_err());
        let mut builder = PlanBuilder::new_vllm_gdn_attempt_phase(c.clone()).unwrap();
        let raw = builder.vllm_bf16_source_embedding(&embedding(&c)).unwrap();
        let norm = builder
            .vllm_input_rms(
                &VllmInputRms::new(
                    c.clone(),
                    parameter(&c, "input_layernorm.weight", QuantKind::Bf16, &[1024]),
                )
                .unwrap(),
                &raw,
            )
            .unwrap();
        let qkvz = builder
            .vllm_gdn_project(&projection(&c, VllmGdnProjectionKind::Qkvz), &norm)
            .unwrap();
        let ba = builder
            .vllm_gdn_project(&projection(&c, VllmGdnProjectionKind::Ba), &norm)
            .unwrap();
        let output = builder
            .vllm_gdn_conv(
                &VllmGdnConv::new(
                    c.clone(),
                    parameter(
                        &c,
                        "linear_attn.conv1d.weight",
                        QuantKind::Bf16,
                        &[6144, 1, 4],
                    ),
                )
                .unwrap(),
                &qkvz,
                &ba,
            )
            .unwrap();
        let gated = builder
            .vllm_gdn_recurrent(&recurrent(&c), &output.activated, &output.z, &ba)
            .unwrap();
        let record = builder.finish_vllm_gdn_fragment(gated).unwrap();
        assert_eq!(record.plan().vllm_gdn_fragment(), Some(&c));
        assert_eq!(record.plan().required_domains().len(), 42);
        assert_eq!(record.plan().proposed_domains().len(), 42);
        assert_eq!(record.plan().frontier_delta(), 0);
        assert!(record.plan().logits().is_none());
        assert_eq!(
            record
                .plan()
                .nodes()
                .iter()
                .filter(|node| matches!(node.operation, PlanOperation::TypedVllmStateCarry))
                .count(),
            40
        );
    }
}
#[test]
fn cached_record_rejects_same_shape_other_phase_or_reminted_teacher_issuer() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let c = VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 0).unwrap(),
        VllmBf16Phase::CachedOne,
        VllmBf16Teachers::new(source, 1, 512, &[353]).unwrap(),
    )
    .unwrap();
    let mut builder = PlanBuilder::new_vllm_gdn_attempt_phase(c.clone()).unwrap();
    let raw = builder.vllm_bf16_source_embedding(&embedding(&c)).unwrap();
    let reminted = VllmBf16Invocation::new(
        c.layer(),
        c.legacy_phase().unwrap(),
        VllmBf16Teachers::new(source, 1, 512, &[353]).unwrap(),
    )
    .unwrap();
    let wrong = VllmInputRms::new(
        reminted,
        parameter(&c, "input_layernorm.weight", QuantKind::Bf16, &[1024]),
    )
    .unwrap();
    let before = (builder.nodes.len(), builder.values.len());
    assert!(builder.vllm_input_rms(&wrong, &raw).is_err());
    assert_eq!((builder.nodes.len(), builder.values.len()), before);
    let initial = VllmBf16Invocation::new(
        c.layer(),
        VllmBf16Phase::InitialOne,
        VllmBf16Teachers::new(source, 0, 512, &[353]).unwrap(),
    )
    .unwrap();
    let wrong = VllmInputRms::new(
        initial,
        parameter(&c, "input_layernorm.weight", QuantKind::Bf16, &[1024]),
    )
    .unwrap();
    assert!(builder.vllm_input_rms(&wrong, &raw).is_err());
    assert_eq!((builder.nodes.len(), builder.values.len()), before);
}
