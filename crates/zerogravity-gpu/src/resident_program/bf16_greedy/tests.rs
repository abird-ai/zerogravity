//! Pure integer/range contracts only. No client, actor, CUDA or model execution.
use super::*;
use crate::resident_program::{ResidentBufferLayout, ResidentPhysicalType};
use cubecl_common::device::DeviceId;
use cubecl_core::{
    environment::stream::StreamId,
    server::{MemoryLayoutDescriptor, MemoryLayoutPolicy},
};
use cubecl_server::allocator::ContiguousMemoryLayoutPolicy;

fn service() -> ServiceId {
    ServiceId::of::<ContiguousMemoryLayoutPolicy>(DeviceId::new(0, 0))
}
fn original_handles(alignment: usize, sizes: &[usize]) -> Vec<cubecl::server::Handle> {
    let descriptors = sizes
        .iter()
        .map(|&bytes| MemoryLayoutDescriptor::contiguous([bytes].into(), 1))
        .collect::<Vec<_>>();
    let (_, layouts) = ContiguousMemoryLayoutPolicy::new(alignment).apply(
        service(),
        StreamId::current(),
        &descriptors,
    );
    layouts.into_iter().map(|layout| layout.memory).collect()
}
fn independent_handles(alignment: usize) -> Vec<cubecl::server::Handle> {
    [LOGITS_BYTES, PARTIAL_BYTES, PACKET_BYTES, 4]
        .into_iter()
        .map(|bytes| original_handles(alignment, &[bytes]).remove(0))
        .collect()
}
fn packet(id: u32, bits: u32, flags: u32) -> Vec<u8> {
    [id, bits, flags]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}

#[test]
fn every_finite_bf16_key_has_exact_numeric_order_and_only_zero_signs_tie() {
    let mut finite = (0..=u16::MAX)
        .filter(|b| b & 0x7f80 != 0x7f80)
        .map(|bits| (kernels::numeric_key(bits as u32), bits))
        .collect::<Vec<_>>();
    finite.sort_unstable();
    assert_eq!(finite.len(), 65_280);
    for pair in finite.windows(2) {
        let [(a_key, a), (b_key, b)] = [pair[0], pair[1]];
        let expected = f32::from_bits((a as u32) << 16)
            .partial_cmp(&f32::from_bits((b as u32) << 16))
            .unwrap();
        assert_eq!(a_key.cmp(&b_key), expected);
        if a_key == b_key {
            assert_eq!([a, b], [0, 0x8000]);
        }
    }
    assert_eq!(kernels::numeric_key(0), kernels::numeric_key(0x8000));
    assert!(
        f32::from_bits(0)
            .total_cmp(&f32::from_bits(0x8000_0000))
            .is_gt(),
        "existing F32 total_cmp intentionally differs and stays unchanged"
    );
}

#[test]
fn numeric_ties_choose_lowest_global_id_across_blocks_and_zero_sign_order() {
    for (a, b) in [(0, 0x8000), (0x8000, 0), (0x3f80, 0x3f80), (0xbf80, 0xbf80)] {
        for (low, high) in [(0, 1), (1023, 1024), (17, 248_319)] {
            let key = kernels::numeric_key(a);
            let equal = kernels::numeric_key(b);
            assert_eq!(key, equal);
            assert!(kernels::precedes(key, low, equal, high));
            assert!(!kernels::precedes(equal, high, key, low));
        }
    }
    for (low, high) in [(0xff7f, 0x8001), (0x8001, 0x8000), (0, 1), (1, 0x7f7f)] {
        assert!(kernels::precedes(
            kernels::numeric_key(high),
            248_319,
            kernels::numeric_key(low),
            0
        ));
    }
}

#[test]
fn full_vocabulary_is_two_byte_storage_with_exact_512_id_final_tile() {
    assert_eq!(kernels::VOCABULARY, 248_320);
    assert_eq!(kernels::BLOCKS, 243);
    assert_eq!(
        kernels::VOCABULARY - (kernels::BLOCKS - 1) * kernels::TILE,
        512
    );
    assert_eq!(
        (LOGITS_BYTES, PARTIAL_BYTES, PACKET_BYTES),
        (496_640, 2916, 12)
    );
    assert_eq!(PreparedBf16Greedy::managed_bytes(), 2928);
    let bf16 = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, kernels::VOCABULARY).unwrap();
    assert_eq!(bf16.bytes().unwrap(), LOGITS_BYTES);
    assert!(
        bf16.require(
            ResidentBufferLayout::new(ResidentPhysicalType::F32, kernels::VOCABULARY).unwrap()
        )
        .is_err()
    );
}

#[test]
fn packet_keeps_original_bf16_zero_sign_and_subnormal_source_bits() {
    for bits in [0u16, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x7f7f, 0xff7f] {
        let expected = (bits as u32) << 16;
        assert_eq!(
            parse_packet(&packet(1023, expected, 0), &[0; 4]).unwrap(),
            (1023, expected)
        );
    }
}

#[test]
fn packet_cannot_hide_an_earlier_status_or_nonfinite_or_forged_f32_word() {
    for flags in [1u32, 2, 4, 7, u32::MAX] {
        assert!(parse_packet(&packet(0, 0, flags), &[0; 4]).is_err());
        assert!(parse_packet(&packet(0, 0, 0), &flags.to_le_bytes()).is_err());
    }
    for bits in [0x7f80_0000, 0xff80_0000, 0x7fc1_0000, 0x3f80_0001] {
        assert!(parse_packet(&packet(0, bits, 0), &[0; 4]).is_err());
    }
    for id in [248_320, u32::MAX] {
        assert!(parse_packet(&packet(id, 0, 0), &[0; 4]).is_err());
    }
    assert!(parse_packet(&[0; 11], &[0; 4]).is_err());
    assert!(parse_packet(&[0; 12], &[0; 3]).is_err());
}

#[test]
fn original_allocator_whole_handles_admit_exact_extents_including_tail_padding() {
    for alignment in [256, 512] {
        let handles = independent_handles(alignment);
        require_ranges(
            service(),
            &handles,
            &[LOGITS_BYTES, PARTIAL_BYTES, PACKET_BYTES, 4],
        )
        .unwrap();
        assert!(handles.iter().all(|handle| handle.offset_start == Some(0)));
        assert!(
            handles
                .iter()
                .any(|handle| handle.offset_end.unwrap_or(0) > 0)
        );
    }
}

#[test]
fn foreign_cropped_and_malformed_original_ranges_refuse_without_raw_subtraction() {
    for fault in 0..7 {
        let mut handles = independent_handles(512);
        match fault {
            0 => handles[0].service = ServiceId::of::<u32>(DeviceId::new(0, 0)),
            1 => handles[0].offset_start = Some(4),
            2 => handles[1].offset_end = None,
            3 => handles[2].offset_end = Some(u64::MAX),
            4 => handles[2].offset_start = Some(u64::MAX),
            5 => handles[2].offset_end = Some(handles[2].size() + 1),
            _ => handles[3].offset_end = Some(handles[3].size()),
        }
        assert!(
            require_ranges(
                service(),
                &handles,
                &[LOGITS_BYTES, PARTIAL_BYTES, PACKET_BYTES, 4]
            )
            .is_err()
        );
    }
}

#[test]
fn shared_original_allocations_and_incomplete_binding_sets_refuse() {
    let one = original_handles(512, &[4]).remove(0);
    assert!(require_ranges(service(), &[one.clone(), one], &[4, 4]).is_err());
    let coallocated = original_handles(512, &[4, 4]);
    assert!(require_ranges(service(), &coallocated, &[4, 4]).is_err());
    assert!(require_ranges(service(), &[], &[]).is_err());
    assert!(require_ranges(service(), &independent_handles(512), &[4]).is_err());
}
