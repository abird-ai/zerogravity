use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn object() -> Vec<i8> {
    let mut data = vec![0u8; 256];
    data[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
    put(&mut data, 16, 3, 2);
    put(&mut data, 18, 224, 2);
    put(&mut data, 32, 64, 8);
    put(&mut data, 52, 64, 2);
    put(&mut data, 54, 56, 2);
    put(&mut data, 56, 1, 2);
    put(&mut data, 64, 1, 4);
    put(&mut data, 72, 128, 8);
    put(&mut data, 80, 128, 8);
    put(&mut data, 96, 128, 8);
    put(&mut data, 104, 1024, 8);
    put(&mut data, 112, 64, 8);
    data.into_iter().map(|n| n as i8).collect()
}
fn put(data: &mut [u8], offset: usize, n: u64, bytes: usize) {
    data[offset..offset + bytes].copy_from_slice(&n.to_le_bytes()[..bytes]);
}
fn change(data: &mut [i8], offset: usize, n: u64, bytes: usize) {
    for i in 0..bytes {
        data[offset + i] = n.to_le_bytes()[i] as i8;
    }
}
#[test]
fn actual_segments_price_payload_and_alignment_not_heap_or_driver_peak() {
    let inventory = code_inventory(&object()).unwrap();
    assert_eq!(
        inventory,
        HipNativeInventory {
            modules: 1,
            retained_host_bytes: 512 + MODULE_METADATA_BYTES,
            code_object_load_bytes: 1087
        }
    );
}
#[test]
fn malformed_headers_and_truncated_tables_refuse() {
    for (offset, n, bytes) in [
        (0, 0, 1),
        (4, 1, 1),
        (5, 2, 1),
        (16, 2, 2),
        (18, 62, 2),
        (52, 63, 2),
        (54, 55, 2),
        (56, 0, 2),
        (56, 129, 2),
        (32, u64::MAX, 8),
    ] {
        let mut code = object();
        change(&mut code, offset, n, bytes);
        assert!(code_inventory(&code).is_err(), "offset {offset}");
    }
    for end in [0, 4, 63, 100, 119, 255] {
        assert!(code_inventory(&object()[..end]).is_err());
    }
}
#[test]
fn hostile_load_extents_alignment_and_overflow_refuse() {
    for (offset, n) in [
        (64, 0),
        (72, 250),
        (80, u64::MAX - 128),
        (96, 2048),
        (104, 0),
        (112, 0),
        (112, 3),
    ] {
        let mut code = object();
        change(&mut code, offset, n, 8);
        assert!(code_inventory(&code).is_err(), "offset {offset}");
    }
}
#[test]
fn accounting_overflow_never_wraps() {
    let a = HipNativeInventory {
        modules: usize::MAX,
        retained_host_bytes: 0,
        code_object_load_bytes: 0,
    };
    assert!(
        a.add(HipNativeInventory {
            modules: 1,
            ..Default::default()
        })
        .is_err()
    );
    let a = HipNativeInventory {
        retained_host_bytes: usize::MAX,
        ..Default::default()
    };
    assert!(
        a.add(HipNativeInventory {
            retained_host_bytes: 1,
            ..Default::default()
        })
        .is_err()
    );
    let a = HipNativeInventory {
        code_object_load_bytes: usize::MAX,
        ..Default::default()
    };
    assert!(
        a.add(HipNativeInventory {
            code_object_load_bytes: 1,
            ..Default::default()
        })
        .is_err()
    );
}
struct DropSpy(Arc<AtomicUsize>);
impl Drop for DropSpy {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn retained_actual_code_keeps_charge_until_native_owner_retirement() {
    let drops = Arc::new(AtomicUsize::new(0));
    let owner: NativeRetention = Arc::new(DropSpy(drops.clone()));
    let weak = Arc::downgrade(&owner);
    let mut retained = LoadedNativeOwner::new(&object(), owner.clone());
    drop(owner);
    assert!(weak.upgrade().is_some());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(retained._code.as_deref().unwrap(), object());
    retained.mark_unloaded();
    drop(retained);
    assert!(weak.upgrade().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn unknown_context_drop_cannot_release_native_admission() {
    let drops = Arc::new(AtomicUsize::new(0));
    let owner: NativeRetention = Arc::new(DropSpy(drops.clone()));
    let weak = Arc::downgrade(&owner);
    drop(LoadedNativeOwner::new(&object(), owner));
    assert!(weak.upgrade().is_some());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}

#[test]
fn module_retirement_does_not_release_distinct_persistent_context() {
    let context_drops = Arc::new(AtomicUsize::new(0));
    let module_drops = Arc::new(AtomicUsize::new(0));
    let context: NativeRetention = Arc::new(DropSpy(context_drops.clone()));
    let weak = Arc::downgrade(&context);
    let persistent = PersistentNativeOwner::new(context.clone());
    let mut module = LoadedNativeOwner::new(&object(), Arc::new(DropSpy(module_drops.clone())));
    drop(context);
    module.mark_unloaded();
    drop(module);
    assert_eq!(module_drops.load(Ordering::SeqCst), 1);
    assert_eq!(context_drops.load(Ordering::SeqCst), 0);
    assert!(weak.upgrade().is_some());
    drop(persistent);
    assert!(weak.upgrade().is_some());
    assert_eq!(context_drops.load(Ordering::SeqCst), 0);
}

#[test]
fn subsequent_sets_require_same_original_context_proof() {
    let owner: NativeRetention = Arc::new(DropSpy(Arc::new(AtomicUsize::new(0))));
    let foreign: NativeRetention = Arc::new(DropSpy(Arc::new(AtomicUsize::new(0))));
    let retained = PersistentNativeOwner::new(owner.clone());
    assert!(retained.same(&owner));
    assert!(!retained.same(&foreign));
}
