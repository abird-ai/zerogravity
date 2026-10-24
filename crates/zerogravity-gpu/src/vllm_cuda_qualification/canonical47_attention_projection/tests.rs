//! Pure fixed packet/site guards. Actual source/owner/GPU acceptance is separate.
use super::*;
use crate::resident_program::qwen2b_component::CANONICAL47_PROMPT;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{
    GdnPrefixLayer, TeacherSpan, VllmBf16Mode, VllmBf16Teachers,
};

fn request(mode: VllmBf16Mode, base: usize, ids: &[u32]) -> VllmBf16Invocation {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let teachers = if ids.len() == 1 {
        VllmBf16Teachers::new(source, base, 512, ids).unwrap()
    } else {
        VllmBf16Teachers::from_span(TeacherSpan::new(source, base, 512, 248320, ids).unwrap())
            .unwrap()
    };
    VllmBf16Invocation::from_span(GdnPrefixLayer::new(source, 0).unwrap(), mode, teachers).unwrap()
}
fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[test]
fn only_genuine_initial47_node32_source_coordinates_select_the_packets() {
    let exact = request(VllmBf16Mode::Prefill, 0, &CANONICAL47_PROMPT);
    require_site(&exact, 3, 32, &[ValueId(39)], &[ValueId(41)]).unwrap();
    for (layer, node, input, output) in [
        (2, 32, 39, 41),
        (3, 31, 39, 41),
        (3, 33, 39, 41),
        (3, 32, 40, 41),
        (3, 32, 39, 42),
    ] {
        assert!(require_site(&exact, layer, node, &[ValueId(input)], &[ValueId(output)]).is_err());
    }
    let mut changed = CANONICAL47_PROMPT;
    changed[12] = 353;
    for wrong in [
        request(VllmBf16Mode::Prefill, 0, &[11; 64]),
        request(VllmBf16Mode::Prefill, 0, &changed),
        request(VllmBf16Mode::Prefill, 1, &CANONICAL47_PROMPT),
        request(VllmBf16Mode::Decode, 47, &[11]),
    ] {
        assert!(require_site(&wrong, 3, 32, &[ValueId(39)], &[ValueId(41)]).is_err());
    }
}

#[test]
fn actual256x128_main47_and_f32_two_plane_reducer_stay_distinct_from_legacy64_ba() {
    let records = raw_packets();
    assert_eq!(records.len(), 2);
    let (main, reducer) = (&records[0].manifest, &records[1].manifest);
    assert_eq!(
        (main.grid, main.block, main.dynamic_shared_bytes),
        ([20, 1, 2], [256, 1, 1], 24576)
    );
    assert_eq!(
        (reducer.grid, reducer.block, reducer.dynamic_shared_bytes),
        ([160, 3, 1], [32, 16, 1], 0)
    );
    assert_eq!(main.binary_sm, 80);
    assert_eq!(reducer.binary_sm, 86);
    assert!(main.entry.contains("256x128_ldg8_stages_32x3_tn"));
    assert!(reducer.entry.contains("Li16Eif13__nv_bfloat16f"));
    let a = &main.parameters[0].template;
    for (offset, value) in [
        (72, 240640),
        (80, 240640),
        (108, 47),
        (112, 1024),
        (128, 512),
        (132, 0x24),
        (152, 0x3f800000),
        (156, 0),
        (160, 0),
    ] {
        assert_eq!(word(a, offset), value);
    }
    let b = &reducer.parameters[0].template;
    for (offset, value) in [(8, 5120), (12, 47), (36, 240640), (40, 2), (44, 0), (56, 1)] {
        assert_eq!(word(b, offset), value);
    }
}

#[test]
fn real47_extents_and_one_original_workspace_join_all_three_occurrences() {
    let records = raw_packets();
    let (main, reducer) = (&records[0].manifest, &records[1].manifest);
    assert_eq!(
        main.bindings.iter().map(|b| b.bytes).collect::<Vec<_>>(),
        [10485760, 96256, 481280, 8519680]
    );
    assert_eq!(
        reducer.bindings.iter().map(|b| b.bytes).collect::<Vec<_>>(),
        [481280, 8519680]
    );
    let partials = main
        .fixups
        .iter()
        .filter(|f| [16, 120].contains(&f.byte_offset))
        .collect::<Vec<_>>();
    assert_eq!(partials.len(), 2);
    for fixup in partials {
        assert_eq!(
            (fixup.binding, fixup.binding_offset, fixup.extent),
            (3, 0, 1925120)
        );
        assert!(matches!(fixup.io, CubinIo::Write));
    }
    let input = &reducer.fixups[0];
    assert_eq!(
        (
            input.parameter,
            input.binding,
            input.binding_offset,
            input.extent
        ),
        (1, 1, 0, 1925120)
    );
    assert!(matches!(input.io, CubinIo::Read));
    assert!(matches!(records[0].roles[3], Binding::Workspace));
    assert!(matches!(records[1].roles[1], Binding::Workspace));
    assert!(
        reducer.fixups[1..]
            .iter()
            .all(|f| f.binding == 0 && f.extent == 481280)
    );
}

#[test]
fn reducer_preserves21_reflected_parameters_and_unobserved_padding_gap() {
    let packet = reducer_packet();
    let params = &packet.manifest.parameters;
    assert_eq!(params.len(), 21);
    assert_eq!((params[0].offset, params[0].template.len()), (0, 60));
    for (i, parameter) in params[1..].iter().enumerate() {
        assert_eq!(
            (parameter.offset, parameter.template.len()),
            (64 + 8 * i, 8)
        );
    }
    assert!(!params.iter().any(|p| p.offset == 60));
    for parameter in &params[5..=6] {
        assert_eq!(parameter.template, [0; 8]);
    }
    assert!(
        packet
            .manifest
            .fixups
            .iter()
            .all(|p| p.parameter != 5 && p.parameter != 6)
    );
    assert_eq!(word(&params[0].template, 44), 0);
}
