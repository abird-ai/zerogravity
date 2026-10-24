//! Original allocator metadata only; no CUDA client, actor or device allocation.
use super::*;
use cubecl_common::device::DeviceId;
use cubecl_core::server::{MemoryLayoutDescriptor, MemoryLayoutPolicy};
use cubecl_environment::stream::StreamId;
use cubecl_server::allocator::ContiguousMemoryLayoutPolicy;

fn service() -> ServiceId {
    ServiceId::of::<ContiguousMemoryLayoutPolicy>(DeviceId::new(0, 0))
}

fn handles(alignment: usize, sizes: &[usize]) -> (Handle, Vec<Handle>) {
    let descriptors = sizes
        .iter()
        .map(|&bytes| MemoryLayoutDescriptor::contiguous([bytes].into(), 1))
        .collect::<Vec<_>>();
    let (base, layouts) = ContiguousMemoryLayoutPolicy::new(alignment).apply(
        service(),
        StreamId::current(),
        &descriptors,
    );
    (
        base,
        layouts.into_iter().map(|layout| layout.memory).collect(),
    )
}

#[test]
fn genuine_single_allocations_admit_only_their_original_zero_start_tail_padding() {
    for alignment in [256, 512] {
        for bytes in [64, 2048, 16_384, 16_777_216] {
            let (base, handles) = handles(alignment, &[bytes]);
            let handle = &handles[0];
            assert_eq!(handle.offset_start, Some(0));
            assert_eq!(handle.offset_end, Some(base.size() - bytes as u64));
            canonical_handle(handle, service(), bytes as u64, alignment as u64).unwrap();
            if bytes == 2048 {
                // The genuine original unoffset base is equivalent when no tail exists.
                canonical_handle(&base, service(), bytes as u64, alignment as u64).unwrap();
            }
        }
    }
}

#[test]
fn original_policy_handles_refuse_foreign_cropped_offset_and_malformed_views() {
    let (_, handles) = handles(512, &[64]);
    let original = &handles[0];
    for fault in 0..8 {
        let mut bad = original.clone();
        match fault {
            0 => bad.service = ServiceId::of::<u32>(DeviceId::new(0, 0)),
            1 => {
                bad.offset_start = Some(16);
                bad.offset_end = Some(432);
            }
            2 => bad.offset_end = Some(449),
            3 => bad.offset_end = None,
            4 => bad.offset_end = Some(u64::MAX),
            5 => bad.offset_start = Some(u64::MAX),
            6 => {
                bad.offset_start = Some(u64::MAX);
                bad.offset_end = Some(u64::MAX);
            }
            _ => bad.offset_end = Some(bad.size() + 1),
        }
        assert!(
            canonical_handle(&bad, service(), 64, 512).is_err(),
            "fault{fault}"
        );
    }
    for (bytes, alignment) in [
        (0, 512),
        (64, 0),
        (64, 3),
        (u64::MAX, 512),
        (u64::MAX - 1, 512),
    ] {
        assert!(canonical_handle(original, service(), bytes, alignment).is_err());
    }
    assert!(canonical_handle(original, service(), 64, 256).is_err());
}

#[test]
fn original_coallocated_views_are_not_a_single_original_allocation() {
    let (_, handles) = handles(256, &[64, 64]);
    for handle in handles {
        assert_eq!(handle.size_in_used(), 64);
        assert!(canonical_handle(&handle, service(), 64, 256).is_err());
    }
}
