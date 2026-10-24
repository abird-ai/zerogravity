//! Immutable compiler-key metadata; hashing grants no execution authority.
use cubecl_common::hash::{StableHash, StableHasher};
use std::{
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
    sync::Arc,
};

struct PreparedValue<T> {
    value: T,
    hash: StableHash,
}

/// Share one prepared digest with its complete immutable content. A digest is
/// only a hash-table hint: different storage still compares the complete value.
pub(crate) struct PreparedKernelKey<T>(Arc<PreparedValue<T>>);

impl<T: Hash> PreparedKernelKey<T> {
    pub(crate) fn new(value: T) -> Self {
        let hash = StableHasher::hash_one(&value);
        Self(Arc::new(PreparedValue { value, hash }))
    }
}
impl<T> Clone for PreparedKernelKey<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> PreparedKernelKey<T> {
    // Arc's two counters + aligned payload, matching the existing metadata
    // admission convention. Allocator/RSS peaks remain separate physical facts.
    pub(crate) const ALLOCATION_BYTES: usize = {
        let align = std::mem::align_of::<PreparedValue<T>>();
        let header = 2 * std::mem::size_of::<usize>();
        let allocation_align = if align > std::mem::align_of::<usize>() {
            align
        } else {
            std::mem::align_of::<usize>()
        };
        (header.next_multiple_of(align) + std::mem::size_of::<PreparedValue<T>>())
            .next_multiple_of(allocation_align)
    };
}
impl<T> Deref for PreparedKernelKey<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.value
    }
}
impl<T: fmt::Debug> fmt::Debug for PreparedKernelKey<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.value.fmt(f)
    }
}
impl<T: PartialEq> PartialEq for PreparedKernelKey<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0.value == other.0.value
    }
}
impl<T: Eq> Eq for PreparedKernelKey<T> {}
impl<T> Hash for PreparedKernelKey<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u128(self.0.hash);
    }
}

#[cfg(test)]
mod tests;
