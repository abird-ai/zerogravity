use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{SourceNumericalContract, SourceOperatorKind};

#[test]
fn exact_source_dimensions_and_actual_native_record_union_stay_closed() {
    let mut functions = BTreeSet::new();
    let mut images = BTreeSet::new();
    let mut records = 0;
    for rows in [1, 2, 64] {
        for role in [Role::Qkvz, Role::Ba] {
            for launch in manifests::launches(rows, role).unwrap() {
                assert_eq!(launch.manifest.binary_sm, launch.image.binary_sm());
                records += 1;
                images.insert(launch.image);
                functions.insert(launch.manifest.entry.clone());
                for (binding, physical) in launch.roles.iter().zip(&launch.manifest.bindings) {
                    let expected = match binding {
                        Binding::W => role.bytes(),
                        Binding::X => rows * 4096,
                        Binding::Y => rows * role.rows() * 2,
                        Binding::Workspace => WORKSPACE_BYTES,
                    };
                    assert_eq!(physical.bytes, expected as u64);
                }
                for fix in &launch.manifest.fixups {
                    assert!(fix.extent <= launch.manifest.bindings[fix.binding].bytes);
                    assert_eq!(fix.binding_offset, 0);
                    assert_eq!(
                        &launch.manifest.parameters[fix.parameter].template
                            [fix.byte_offset..fix.byte_offset + 8],
                        &[0; 8]
                    );
                }
            }
        }
    }
    assert_eq!((records, images.len(), functions.len()), (7, 4, 6));
    for rows in [0, 3, 47, 63, 65, usize::MAX] {
        assert!(manifests::launches(rows, Role::Qkvz).is_err());
        assert!(manifests::launches(rows, Role::Ba).is_err());
    }
}
#[test]
fn original_five_phases_have_eleven_native_launches_and_exact_output_words() {
    let mut native = 0;
    let mut words = 0;
    for rows in [1, 1, 64, 2, 1] {
        for role in [Role::Qkvz, Role::Ba] {
            native += manifests::launches(rows, role).unwrap().len();
            words += rows * role.rows();
        }
    }
    assert_eq!(native, 11);
    assert_eq!(words, 567456);
    assert_eq!(Role::Qkvz.bytes(), 33_554_432);
    assert_eq!(Role::Ba.bytes(), 131_072);
}

#[test]
fn actual2b_scalar_k_and_complete_w_differ_from_preserved1024_defaults() {
    for rows in [1, 2] {
        let launch = manifests::launches(rows, Role::Qkvz).unwrap().remove(0);
        let packet = &launch.manifest.parameters[0].template;
        assert_eq!(&packet[0..4], &(rows as u32).to_le_bytes());
        assert_eq!(&packet[4..8], &8192u32.to_le_bytes());
        assert_eq!(&packet[8..12], &2048u32.to_le_bytes());
        assert_eq!(launch.manifest.bindings[0].bytes, 33_554_432);
        assert_eq!(launch.manifest.bindings[1].bytes, rows as u64 * 4096);
    }
    let old = super::super::cubin_manifests::Projection::Qkvz2;
    assert_eq!(old.weight_bytes(), 16_777_216);
    assert_eq!(old.input_bytes(), 4096);
}

#[test]
fn actual_splitk_16_part_dependency_is_bounded_by_original_managed_workspace() {
    let launches = manifests::launches(64, Role::Ba).unwrap();
    assert_eq!(launches.len(), 2);
    for (index, launch) in launches.iter().enumerate() {
        let wi = launch
            .roles
            .iter()
            .position(|role| matches!(role, Binding::Workspace))
            .unwrap();
        assert_eq!(launch.manifest.bindings[wi].bytes, 8_519_680);
        assert_eq!(
            launch.manifest.bindings[wi].io,
            if index == 0 {
                CubinIo::Write
            } else {
                CubinIo::Read
            }
        );
        assert_eq!(
            launch
                .manifest
                .fixups
                .iter()
                .find(|fix| fix.binding == wi)
                .unwrap()
                .extent,
            65_536
        );
    }
    assert_eq!(
        &launches[0].manifest.parameters[0].template[256..260],
        &u32::MAX.to_le_bytes()
    );
    assert_eq!(
        &launches[0].manifest.parameters[0].template[260..264],
        &[0; 4]
    );
}
#[test]
fn matching_width_does_not_replace_actual_native_operator_kind() {
    use zerogravity_core::accelerator::program::*;
    for columns in [1024, 2048] {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let call = SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, 248320, 512).unwrap(),
                0,
                Arc::from([11]),
            )
            .unwrap(),
            SourceCallMode::Prefill,
        )
        .unwrap();
        let op = SourceOperator::new(
            call,
            SourceOperatorKind::Rms(SourceRmsGeometry {
                width: columns,
                epsilon: 1e-6,
                gamma: SourceGamma::OnePlus,
            }),
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            vec![],
            vec![SourceParameter {
                source,
                name: "gamma".into(),
                kind: zerogravity_core::quant::QuantKind::Bf16,
                shape: vec![columns as u64],
                bytes: columns * 2,
            }],
        )
        .unwrap();
        assert!(Role::of(&op).is_err());
    }
}
#[test]
fn exact_reduce_formal_gap_and_unread_host_scalar_fields_remain_separate() {
    let launch = manifests::launches(64, Role::Ba).unwrap().remove(1);
    let parameters = &launch.manifest.parameters;
    assert_eq!(parameters.len(), 21);
    assert_eq!(
        (parameters[0].offset, parameters[0].template.len()),
        (0, 60)
    );
    for (parameter, offset) in parameters[1..].iter().zip((64..224).step_by(8)) {
        assert_eq!((parameter.offset, parameter.template.len()), (offset, 8));
    }
    assert_eq!(&parameters[0].template[44..48], &[0; 4]);
    for offset in [96, 104] {
        let index = parameters
            .iter()
            .position(|parameter| parameter.offset == offset)
            .unwrap();
        assert_eq!(parameters[index].template, [0; 8]);
        assert!(
            launch
                .manifest
                .fixups
                .iter()
                .all(|fix| fix.parameter != index)
        );
    }
}

#[test]
fn shared_weights_preserve_reference_call_eligibility_for_fresh_issuers() -> Result<()> {
    use zerogravity_core::accelerator::program::{
        MatrixAxes, SourceCall, SourceCallMode, SourceOperator, SourceRows, SourceTensor,
        TextCallGeometry,
    };
    use zerogravity_core::quant::QuantKind;
    let source = SourceIdentity::for_new_allocation()?;
    let call = SourceCall::new(
        SourceRows::new(
            TextCallGeometry::new(source, 248320, 512)?,
            0,
            Arc::from([11]),
        )?,
        SourceCallMode::Prefill,
    )?;
    let member = |suffix: &str, rows| SourceTensor {
        source,
        name: format!("model.language_model.layers.0.linear_attn.{suffix}.weight"),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        rows,
        columns: 2048,
    };
    let descriptors = [
        SourceOperator::new(
            call.clone(),
            SourceOperatorKind::Projection { selected_row: None },
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            vec![member("in_proj_qkv", 6144), member("in_proj_z", 2048)],
            vec![],
        )?,
        SourceOperator::new(
            call.clone(),
            SourceOperatorKind::Projection { selected_row: None },
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            vec![member("in_proj_b", 16), member("in_proj_a", 16)],
            vec![],
        )?,
    ];
    let actual = |expected: &SourceOperator, mode, base, rows, capacity, vocabulary, issuer| {
        let ids: Arc<[u32]> = vec![353; rows].into();
        let call = SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(issuer, vocabulary, capacity)?,
                base,
                ids,
            )?,
            mode,
        )?;
        let template = expected.clone();
        SourceOperator::new(
            call,
            template.kind(),
            template.numerical_contract(),
            template.matrices().to_vec(),
            template.parameters().to_vec(),
        )
    };
    for (mode, base, rows) in [
        (SourceCallMode::Prefill, 0, 1),
        (SourceCallMode::Prefill, 0, 64),
        (SourceCallMode::Prefill, 64, 2),
        (SourceCallMode::Decode, 1, 1),
        (SourceCallMode::Decode, 66, 1),
    ] {
        for (expected, role) in descriptors.iter().zip([Role::Qkvz, Role::Ba]) {
            let op = actual(expected, mode, base, rows, 512, 248320, source)?;
            assert_eq!(weights::role_for_operator(&descriptors, &op)?, role);
            assert_ne!(op.call(), expected.call());
        }
    }
    for (mode, base, rows, capacity, vocabulary) in [
        (SourceCallMode::Prefill, 2, 1, 512, 248320),
        (SourceCallMode::Prefill, 1, 1, 512, 248320),
        (SourceCallMode::Prefill, 1, 64, 512, 248320),
        (SourceCallMode::Prefill, 63, 2, 512, 248320),
        (SourceCallMode::Decode, 2, 1, 512, 248320),
        (SourceCallMode::Decode, 64, 1, 512, 248320),
        (SourceCallMode::Prefill, 0, 1, 513, 248320),
        (SourceCallMode::Prefill, 0, 1, 512, 248321),
    ] {
        for expected in &descriptors {
            let op = actual(expected, mode, base, rows, capacity, vocabulary, source)?;
            assert!(weights::role_for_operator(&descriptors, &op).is_err());
        }
    }
    let foreign = SourceIdentity::for_new_allocation()?;
    let foreign_call = SourceCall::new(
        SourceRows::new(
            TextCallGeometry::new(foreign, 248320, 512)?,
            0,
            Arc::from([353]),
        )?,
        SourceCallMode::Prefill,
    )?;
    assert!(weights::require_reference_call(descriptors[0].call(), &foreign_call).is_err());
    let template = descriptors[0].clone();
    let mut foreign_members = template.matrices().to_vec();
    for member in &mut foreign_members {
        member.source = foreign;
    }
    let foreign_operator = SourceOperator::new(
        foreign_call,
        template.kind(),
        template.numerical_contract(),
        foreign_members,
        vec![],
    )?;
    assert!(weights::role_for_operator(&descriptors, &foreign_operator).is_err());
    Ok(())
}
