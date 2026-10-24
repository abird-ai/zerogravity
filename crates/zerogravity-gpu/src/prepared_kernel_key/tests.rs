use super::*;
use std::{
    collections::HashSet,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Counted {
    text: Arc<str>,
}
static HASH_CALLS: AtomicUsize = AtomicUsize::new(0);
impl PartialEq for Counted {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text
    }
}
impl Eq for Counted {}
impl Hash for Counted {
    fn hash<H: Hasher>(&self, state: &mut H) {
        HASH_CALLS.fetch_add(1, Ordering::Relaxed);
        self.text.hash(state);
    }
}

#[test]
fn original_content_is_hashed_once_across_clones_and_cache_lookups() {
    HASH_CALLS.store(0, Ordering::Relaxed);
    let key = PreparedKernelKey::new(Counted {
        text: Arc::from("complete immutable compiler input"),
    });
    assert_eq!(HASH_CALLS.load(Ordering::Relaxed), 1);
    let prepared = key.clone();
    assert!(Arc::ptr_eq(&key.0, &prepared.0));
    assert_eq!(
        StableHasher::hash_one(&key),
        StableHasher::hash_one(&prepared)
    );
    let mut cache = HashSet::new();
    cache.insert(key);
    assert!(cache.contains(&prepared));
    assert_eq!(HASH_CALLS.load(Ordering::Relaxed), 1);
}

#[test]
fn separate_equal_content_and_forced_collisions_use_full_equality() {
    let first = PreparedKernelKey::new(String::from("complete source"));
    let equal = PreparedKernelKey::new(String::from("complete source"));
    assert!(!Arc::ptr_eq(&first.0, &equal.0));
    assert_ne!(first.as_ptr(), equal.as_ptr());
    assert_eq!(first, equal);
    assert_eq!(
        StableHasher::hash_one(&first),
        StableHasher::hash_one(&equal)
    );
    assert_eq!(format!("{first:?}"), format!("{:?}", first.0.value));

    let collide = |value: &str| {
        PreparedKernelKey(Arc::new(PreparedValue {
            value: value.to_string(),
            hash: 0,
        }))
    };
    let a = collide("complete source");
    let b = collide("complete sourcf");
    assert_eq!(StableHasher::hash_one(&a), StableHasher::hash_one(&b));
    assert_ne!(a, b);
    let cache = HashSet::from([a, b]);
    assert_eq!(cache.len(), 2);
    assert!(cache.contains(&collide("complete source")));
    assert!(!cache.contains(&collide("complete sourcg")));
}

#[test]
fn capsule_layout_counts_aligned_payload_and_arc_counters_once() {
    for bytes in [
        PreparedKernelKey::<u8>::ALLOCATION_BYTES,
        PreparedKernelKey::<Arc<str>>::ALLOCATION_BYTES,
    ] {
        assert!(bytes >= 2 * std::mem::size_of::<usize>());
        assert!(bytes.is_multiple_of(std::mem::align_of::<StableHash>()));
    }
    assert_eq!(
        std::mem::size_of::<PreparedKernelKey<String>>(),
        std::mem::size_of::<Arc<String>>()
    );
}
