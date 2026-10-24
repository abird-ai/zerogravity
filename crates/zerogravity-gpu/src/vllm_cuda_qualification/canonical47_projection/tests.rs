//! Structural metadata only; no trained tensor or native qualification.
use super::*;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{
        GdnPrefixLayer, MatrixAxes, SourceTensor, TeacherSpan, Vllm08Call, VllmBf16Invocation,
        VllmBf16Teachers,
    },
};
use zerogravity_core::quant::QuantKind;
fn call(ids: &[u32], base: usize) -> Vllm08Call {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let teachers = if ids.len() == 1 {
        VllmBf16Teachers::new(source, base, 512, ids).unwrap()
    } else {
        VllmBf16Teachers::from_span(TeacherSpan::new(source, base, 512, 248_320, ids).unwrap())
            .unwrap()
    };
    Vllm08Call::from_span(
        VllmBf16Invocation::from_span(
            GdnPrefixLayer::new(source, 0).unwrap(),
            VllmBf16Mode::Prefill,
            teachers,
        )
        .unwrap(),
    )
    .unwrap()
}
fn operator(call: &Vllm08Call, projection: Vllm08Projection, layer: usize) -> Vllm08Operator {
    let members = match projection {
        Vllm08Projection::Qkvz => [("in_proj_qkv", 6144), ("in_proj_z", 2048)],
        Vllm08Projection::Ba => [("in_proj_b", 16), ("in_proj_a", 16)],
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
    Vllm08Operator::new(
        call.clone(),
        Some(call.layer(layer).unwrap()),
        Vllm08Kind::Projection(projection),
        matrices,
        Vec::new(),
    )
    .unwrap()
}
#[test]
fn canonical47_role_requires_actual_span_prompt_and_layer_without_legacy_retag() {
    let good = call(&PROMPT, 0);
    assert!(good.request().legacy_phase().is_err());
    assert_eq!(
        Role::of(&operator(&good, Vllm08Projection::Qkvz, 0)).unwrap(),
        Role::Qkvz
    );
    assert_eq!(
        Role::of(&operator(&good, Vllm08Projection::Ba, 0)).unwrap(),
        Role::Ba
    );
    assert!(Role::of(&operator(&good, Vllm08Projection::Ba, 1)).is_err());
    let mut wrong = PROMPT;
    wrong[46] = 11;
    assert!(Role::of(&operator(&call(&wrong, 0), Vllm08Projection::Qkvz, 0)).is_err());
    assert!(Role::of(&operator(&call(&PROMPT, 1), Vllm08Projection::Ba, 0)).is_err());
    for ids in [&PROMPT[..1], &PROMPT[..2], &PROMPT[..46]] {
        assert!(Role::of(&operator(&call(ids, 0), Vllm08Projection::Ba, 0)).is_err());
    }
}
#[test]
fn canonical47_exact_union_and_lease_extents_fit_preserved_scope() {
    let mut images = BTreeSet::new();
    let mut functions = BTreeSet::new();
    let mut count = 0;
    for role in [Role::Qkvz, Role::Ba] {
        for launch in manifests::launches(47, role).unwrap() {
            images.insert(launch.image);
            functions.insert(launch.manifest.entry.clone());
            count += 1;
            assert_eq!(launch.manifest.binary_sm, launch.image.binary_sm());
            for (role_binding, physical) in launch.roles.iter().zip(&launch.manifest.bindings) {
                let expected = match role_binding {
                    Binding::W => role.bytes(),
                    Binding::X => 96256,
                    Binding::Y => 47 * role.rows() * 2,
                    Binding::Workspace => WORKSPACE_BYTES,
                };
                assert_eq!(physical.bytes, expected as u64);
            }
            for fix in &launch.manifest.fixups {
                assert_eq!(fix.binding_offset, 0);
                assert!(fix.extent <= launch.manifest.bindings[fix.binding].bytes);
                assert_eq!(
                    &launch.manifest.parameters[fix.parameter].template
                        [fix.byte_offset..fix.byte_offset + 8],
                    &[0; 8]
                );
            }
        }
    }
    assert_eq!((images.len(), functions.len(), count), (3, 3, 3));
    assert_eq!(Role::Qkvz.bytes() + Role::Ba.bytes(), 16_842_752);
    for rows in [0, 1, 2, 46, 48, 64, usize::MAX] {
        for role in [Role::Qkvz, Role::Ba] {
            assert!(manifests::launches(rows, role).is_err());
        }
    }
}
#[test]
fn canonical47_ba_splitk_bytes_match_reducer_read_without_workspace_tail_access() {
    let launches = manifests::launches(47, Role::Ba).unwrap();
    assert_eq!(launches.len(), 2);
    assert_eq!(launches[0].manifest.grid, [1, 1, 16]);
    assert_eq!(launches[0].manifest.block, [128, 1, 1]);
    assert_eq!(launches[0].manifest.dynamic_shared_bytes, 49152);
    assert_eq!(launches[1].manifest.grid, [1, 3, 1]);
    assert_eq!(launches[1].manifest.block, [32, 16, 1]);
    assert_eq!(
        launches[0]
            .manifest
            .fixups
            .iter()
            .map(|f| f.byte_offset)
            .collect::<Vec<_>>(),
        [344, 352, 360, 368]
    );
    for (index, launch) in launches.iter().enumerate() {
        let wi = launch
            .roles
            .iter()
            .position(|r| matches!(r, Binding::Workspace))
            .unwrap();
        assert_eq!(launch.manifest.bindings[wi].bytes, 8_519_680);
        let f = launch
            .manifest
            .fixups
            .iter()
            .find(|f| f.binding == wi)
            .unwrap();
        assert_eq!(f.extent, 47 * 32 * 16 * 2);
        assert_eq!(
            f.io,
            if index == 0 {
                CubinIo::Write
            } else {
                CubinIo::Read
            }
        );
    }
    let p = &launches[0].manifest.parameters[0].template;
    assert_eq!(p.len(), 440);
    assert_eq!(&p[312..316], &1f32.to_le_bytes());
    assert_eq!(&p[316..320], &0f32.to_le_bytes());
    assert_eq!(&p[320..336], &[0; 16]);
    assert_eq!(&p[336..340], &0x7fffu32.to_le_bytes());
    assert!(p[376..440].iter().all(|b| *b == 0));
}
#[test]
fn canonical47_actual_k_and_reduction_layout_cannot_alias_old64_or_2b() {
    let q = manifests::launches(47, Role::Qkvz)
        .unwrap()
        .remove(0)
        .manifest;
    let p = &q.parameters[0].template;
    assert_eq!(p.len(), 176);
    assert_eq!(&p[108..112], &47u32.to_le_bytes());
    assert_eq!(&p[112..116], &1024u32.to_le_bytes());
    let r = manifests::launches(47, Role::Ba)
        .unwrap()
        .remove(1)
        .manifest;
    assert_eq!(r.parameters.len(), 21);
    assert_eq!(
        (r.parameters[0].offset, r.parameters[0].template.len()),
        (0, 60)
    );
    assert_eq!(&r.parameters[0].template[12..16], &47u32.to_le_bytes());
    assert_eq!(&r.parameters[0].template[36..40], &1504u32.to_le_bytes());
    assert_eq!(&r.parameters[0].template[40..44], &16u32.to_le_bytes());
    assert_eq!(&r.parameters[0].template[44..48], &[0; 4]);
    for (p, offset) in r.parameters[1..].iter().zip((64..224).step_by(8)) {
        assert_eq!((p.offset, p.template.len()), (offset, 8));
    }
    for offset in [96, 104] {
        let i = r
            .parameters
            .iter()
            .position(|p| p.offset == offset)
            .unwrap();
        assert_eq!(r.parameters[i].template, [0; 8]);
        assert!(r.fixups.iter().all(|f| f.parameter != i));
    }
}

#[test]
fn canonical47_equal_ids_new_teacher_issuer_cannot_reuse_prepared_weights() {
    let original = call(&PROMPT, 0);
    let source = original.source();
    let foreign = Vllm08Call::from_span(
        VllmBf16Invocation::from_span(
            GdnPrefixLayer::new(source, 0).unwrap(),
            VllmBf16Mode::Prefill,
            VllmBf16Teachers::from_span(
                TeacherSpan::new(source, 0, 512, 248_320, &PROMPT).unwrap(),
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    for projection in [Vllm08Projection::Qkvz, Vllm08Projection::Ba] {
        let a = operator(&original, projection, 0);
        let b = operator(&foreign, projection, 0);
        assert_eq!(a.matrices(), b.matrices());
        assert_ne!(a.call(), b.call());
        assert!(require_same_operator(&a, &a.clone()).is_ok());
        assert!(require_same_operator(&a, &b).is_err());
    }
}

#[test]
fn short_hf_issuer_cannot_reuse_original47_projection_roles() {
    use zerogravity_core::accelerator::program::{ShortHfCall, ShortHfSpan, ShortHfWorkload};
    let source = SourceIdentity::for_new_allocation().unwrap();
    let short = ShortHfCall::new(
        FullModelProfile::Qwen08,
        ShortHfSpan::prefill(source, ShortHfWorkload::Prompt47, &PROMPT).unwrap(),
    )
    .unwrap();
    assert_eq!(short.request().teachers().ids(), PROMPT);
    assert!(short.request().legacy_phase().is_err());
    for projection in [Vllm08Projection::Qkvz, Vllm08Projection::Ba] {
        assert!(Role::of(&operator(&call(&PROMPT, 0), projection, 0)).is_ok());
        assert!(matches!(
            Role::of(&operator(short.as_vllm_call(), projection, 0)),
            Err(Error::Unsupported(_))
        ));
    }
    for base in 47..=53 {
        let short = ShortHfCall::new(
            FullModelProfile::Qwen08,
            ShortHfSpan::cached(source, ShortHfWorkload::Prompt47, base, 11).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            Role::of(&operator(short.as_vllm_call(), Vllm08Projection::Qkvz, 0)),
            Err(Error::Unsupported(_))
        ));
    }
}
