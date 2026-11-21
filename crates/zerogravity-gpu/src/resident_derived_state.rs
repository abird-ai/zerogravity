//! Optional device-native derivatives of canonical resident state.
//!
//! Canonical session domains and SessionAuthority remain the sole publication,
//! rollback and checkpoint authority.  A derived entry is merely a cached
//! physical representation of one exact committed RootBinding.  A stale,
//! unavailable or busy entry is always a cache miss; callers retain their
//! canonical path.

use crate::{
    resident_leases::{ResidentLease, ResidentService},
    resident_program::ResidentBufferLayout,
};
use std::sync::{Arc, Mutex};

const MAX_DERIVED_STATE_ENTRIES: usize = 256;
use zerogravity_core::{
    Error, Result, accelerator::program::StateSlot, device_events::BufferRole,
    session_root::RootBinding,
};

/// Opaque backend-defined representation identity.
///
/// The shared cache never interprets the label.  Backends/models may define
/// stable labels for reusable physical layouts without teaching the cache about
/// those layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DerivedStateEncoding(&'static str);

impl DerivedStateEncoding {
    pub(crate) const fn new(label: &'static str) -> Self {
        Self(label)
    }

    pub(crate) const fn label(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DerivedStateKey {
    slot: StateSlot,
    encoding: DerivedStateEncoding,
    layout: ResidentBufferLayout,
}

struct Entry {
    key: DerivedStateKey,
    root: RootBinding,
    lease: ResidentLease,
    busy: bool,
}

#[derive(Default)]
struct CacheInner {
    entries: Vec<Entry>,
}

fn is_forward_retag(base: RootBinding, root: RootBinding) -> bool {
    root.source() == base.source()
        && root.session() == base.session()
        && base.root().checked_add(1) == Some(root.root())
        && root.session_epoch() == base.session_epoch()
        && root.frontier() > base.frontier()
}

/// Per-session cache of optional physical derivatives.
///
/// The cache has no publication authority.  Acquiring an exact-root hit marks
/// that entry busy before mutation; a competing attempt therefore falls back to
/// canonical execution rather than sharing mutable derived state.
#[derive(Clone, Default)]
pub(crate) struct DerivedStateCache {
    inner: Arc<Mutex<CacheInner>>,
}

/// One mutable use of a cached derivative.
///
/// Dropping an unpublished use invalidates the entry.  After physical
/// completion, the caller may retain this guard in PreparedInference and retag
/// it only after the canonical root has successfully published.
pub(crate) struct DerivedStateUse {
    cache: Arc<Mutex<CacheInner>>,
    key: DerivedStateKey,
    base: RootBinding,
    lease: ResidentLease,
    needs_import: bool,
    armed: bool,
}

impl DerivedStateCache {
    pub(crate) fn try_acquire(
        &self,
        service: &Arc<ResidentService>,
        root: RootBinding,
        slot: StateSlot,
        encoding: DerivedStateEncoding,
        layout: ResidentBufferLayout,
    ) -> Result<Option<DerivedStateUse>> {
        if root.source() != service.source_identity() || encoding.label().is_empty() {
            return Err(Error::State(
                "derived resident state source/encoding differs".into(),
            ));
        }
        let bytes = layout.bytes()?;
        let key = DerivedStateKey {
            slot,
            encoding,
            layout,
        };
        // Cache corruption must never make canonical state unavailable.  A
        // poisoned cache therefore degrades to an ordinary cache miss.
        let Ok(mut cache) = self.inner.lock() else {
            return Ok(None);
        };

        if let Some(index) = cache.entries.iter().position(|entry| entry.key == key) {
            if cache.entries[index].busy {
                return Ok(None);
            }
            if cache.entries[index].root == root
                && cache.entries[index].lease.is_initialized()
                && cache.entries[index].lease.bytes() == bytes
                && cache.entries[index].lease.role() == BufferRole::RecurrentState
                && Arc::ptr_eq(cache.entries[index].lease.service(), service)
            {
                let lease = cache.entries[index].lease.clone();
                cache.entries[index].busy = true;
                return Ok(Some(DerivedStateUse {
                    cache: self.inner.clone(),
                    key,
                    base: root,
                    lease,
                    needs_import: false,
                    armed: true,
                }));
            }
            // A non-busy stale/uninitialized representation has no authority.
            cache.entries.swap_remove(index);
        }

        if cache.entries.len() >= MAX_DERIVED_STATE_ENTRIES {
            return Ok(None);
        }

        let lease = match service.allocate_uninitialized(bytes, BufferRole::RecurrentState) {
            Ok(lease) => lease,
            // This is an optional optimization.  Capacity pressure retains the
            // canonical execution path instead of failing the request.
            Err(Error::Limit(_) | Error::Resource(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        cache.entries.push(Entry {
            key,
            root,
            lease: lease.clone(),
            busy: true,
        });
        Ok(Some(DerivedStateUse {
            cache: self.inner.clone(),
            key,
            base: root,
            lease,
            needs_import: true,
            armed: true,
        }))
    }

    /// Explicitly discard every derived representation after restore/reset or
    /// another canonical root replacement that did not originate in a retained
    /// DerivedStateUse.
    pub(crate) fn invalidate(&self) {
        if let Ok(mut cache) = self.inner.lock() {
            cache.entries.clear();
        }
    }
}

impl DerivedStateUse {
    pub(crate) fn base(&self) -> RootBinding {
        self.base
    }

    pub(crate) fn slot(&self) -> StateSlot {
        self.key.slot
    }

    pub(crate) fn encoding(&self) -> DerivedStateEncoding {
        self.key.encoding
    }

    pub(crate) fn layout(&self) -> ResidentBufferLayout {
        self.key.layout
    }

    /// A miss owns fresh uninitialized storage and must materialize canonical
    /// state before the derived consumer reads it. An exact-root hit may skip
    /// that import.
    pub(crate) fn needs_import(&self) -> bool {
        self.needs_import
    }

    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }

    /// Best-effort cache retag after canonical publication.  Failure to update
    /// the optimization cache cannot undo or fail an already-published root.
    pub(crate) fn retag_after_publish(&mut self, root: RootBinding) -> bool {
        if !is_forward_retag(self.base, root) || !self.lease.is_initialized() {
            return false;
        }
        let Ok(mut cache) = self.cache.lock() else {
            return false;
        };
        let Some(entry) = cache.entries.iter_mut().find(|entry| {
            entry.key == self.key
                && entry.busy
                && entry.root == self.base
                && entry.lease.id() == self.lease.id()
        }) else {
            return false;
        };
        entry.root = root;
        entry.busy = false;
        self.armed = false;
        true
    }
}

impl Drop for DerivedStateUse {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Ok(mut cache) = self.cache.lock()
            && let Some(index) = cache.entries.iter().position(|entry| {
                entry.key == self.key
                    && entry.busy
                    && entry.root == self.base
                    && entry.lease.id() == self.lease.id()
            })
        {
            cache.entries.swap_remove(index);
        }
    }
}

#[cfg(test)]
mod root_tests {
    use super::*;
    use zerogravity_core::{accelerator::SourceIdentity, session_root::SessionAuthority};

    #[test]
    fn retag_accepts_only_the_next_forward_root() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        let mut authority = SessionAuthority::new(source)?;
        let root0 = authority.binding();
        let forward = authority.begin_advance(4)?;
        let root1 = authority.prepare_publish(forward)?.commit();
        assert!(is_forward_retag(root0, root1));

        let future = authority.begin_advance(3)?;
        let root2 = authority.prepare_publish(future)?.commit();
        assert!(!is_forward_retag(root0, root2));

        let restore = authority.begin_restore(root1)?;
        let restored = authority.prepare_publish(restore)?.commit();
        assert!(!is_forward_retag(root2, restored));

        let reset = authority.begin_reset()?;
        let reset = authority.prepare_publish(reset)?.commit();
        assert!(!is_forward_retag(restored, reset));
        Ok(())
    }
}

#[cfg(all(test, feature = "qualification", feature = "cuda", target_os = "linux"))]
mod tests {
    use super::*;
    use crate::cuda::test_support::CudaDiagnostic;
    use crate::resident_program::ResidentPhysicalType;
    use zerogravity_core::{
        accelerator::program::{RecurrentDomain, StateSlot},
        session_root::SessionAuthority,
    };

    #[test]
    #[ignore = "physical CUDA derived-state lifecycle"]
    fn exact_root_busy_retag_and_abandon_are_fail_closed() -> Result<()> {
        futures_lite::future::block_on(async {
            let diagnostic = CudaDiagnostic::new("derived-state-cache")?;
            let service = diagnostic.service;
            let mut authority = SessionAuthority::new(service.source_identity())?;
            let cache = DerivedStateCache::default();
            let encoding = DerivedStateEncoding::new("qualification.opaque.f32.v1");
            let layout = ResidentBufferLayout::new(ResidentPhysicalType::F32, 64)?;
            let slot = StateSlot::Recurrent(RecurrentDomain(7));
            let root0 = authority.binding();

            let mut first = cache
                .try_acquire(&service, root0, slot, encoding, layout)?
                .ok_or_else(|| Error::State("first derived cache allocation absent".into()))?;
            assert!(first.needs_import());
            assert!(
                cache
                    .try_acquire(&service, root0, slot, encoding, layout)?
                    .is_none()
            );
            let first_id = first.lease().id();

            let mut init = service.begin_initialization(root0.root())?;
            init.upload_zeroed_into(first.lease())?;
            drop(init.complete_sync().await?);
            let attempt = authority.begin_advance(1)?;
            let root1 = authority.prepare_publish(attempt)?.commit();
            assert!(first.retag_after_publish(root1));

            let hit = cache
                .try_acquire(&service, root1, slot, encoding, layout)?
                .ok_or_else(|| Error::State("exact-root derived cache hit absent".into()))?;
            assert!(!hit.needs_import());
            assert_eq!(hit.lease().id(), first_id);

            // A dropped mutable use may have changed its bytes; invalidate it
            // rather than exposing that representation to a later root user.
            drop(hit);
            let replacement = cache
                .try_acquire(&service, root1, slot, encoding, layout)?
                .ok_or_else(|| Error::State("abandoned derived cache replacement absent".into()))?;
            assert!(replacement.needs_import());
            assert_ne!(replacement.lease().id(), first_id);
            drop(replacement);

            cache.invalidate();
            let after_invalidate = cache
                .try_acquire(&service, root1, slot, encoding, layout)?
                .ok_or_else(|| {
                    Error::State("invalidated derived cache replacement absent".into())
                })?;
            assert!(after_invalidate.needs_import());
            Ok(())
        })
    }
}
