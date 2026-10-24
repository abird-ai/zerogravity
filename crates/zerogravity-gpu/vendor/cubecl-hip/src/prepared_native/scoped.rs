//! Bounded original-actor ownership routing; logical IDs never name physical slots.
use super::{PreparedModules, failure};
use cubecl_core::{prelude::KernelId, server::ServerError};
use cubecl_environment::stream::StreamId;
use std::collections::{BTreeMap, HashMap, HashSet};

pub(crate) const MAX_SETS: usize = 2;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeMode {
    SingleCompat,
    Scoped,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NativeKernelScope {
    logical: StreamId,
    original: KernelId,
}
/// Cold upper bound for retained full keys, table buckets, source-client/slot
/// control, and the two transient catalogue vectors. Not native-code admission.
pub fn scoped_metadata_bytes(kernels: usize) -> Result<usize, ServerError> {
    if kernels == 0 || kernels > super::MAX_KERNELS {
        return Err(failure("HIP bounded scoped metadata extent"));
    }
    // <=1/2 load leaves room for hashbrown's capacity rounding/control bytes.
    let buckets = kernels
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| failure("HIP scoped bucket count overflow"))?
        .max(4);
    let key = std::mem::size_of::<KernelId>();
    let entry = key
        .checked_mul(3)
        .and_then(|n| n.checked_add(32))
        .ok_or_else(|| failure("HIP scoped table size overflow"))?;
    let node = std::mem::size_of::<NativeKernelScope>()
        .checked_add(2 * std::mem::size_of::<usize>() + 64)
        .and_then(|n| n.checked_add(4 * key))
        .ok_or_else(|| failure("HIP scoped payload size overflow"))?;
    buckets
        .checked_mul(entry)
        .and_then(|n| kernels.checked_mul(node).and_then(|m| n.checked_add(m)))
        .and_then(|n| n.checked_add(std::mem::size_of::<SetSlot>() + 4096))
        .ok_or_else(|| failure("HIP scoped metadata sum overflow"))
}
#[derive(Debug)]
struct SetSlot {
    logical: StreamId,
    modules: Option<PreparedModules>,
    keys: HashMap<KernelId, KernelId>,
    routed: HashSet<KernelId>,
}
impl SetSlot {
    fn prepare_keys(&mut self, scoped: bool, originals: &[KernelId], catalogue: &[KernelId]) {
        if scoped {
            self.keys = originals
                .iter()
                .cloned()
                .zip(catalogue.iter().cloned())
                .collect();
            self.routed = catalogue.iter().cloned().collect();
        }
    }
}
#[derive(Debug, Default)]
pub(crate) struct NativeSets {
    mode: Option<NativeMode>,
    sets: BTreeMap<u64, SetSlot>,
}
impl NativeSets {
    pub(crate) fn active(&self) -> bool {
        self.mode.is_some()
    }
    pub(crate) fn select_scoped(&mut self) -> Result<(), ServerError> {
        if self.mode.is_some_and(|mode| mode != NativeMode::Scoped) {
            return Err(failure("HIP native actor mode already fixed"));
        }
        self.mode = Some(NativeMode::Scoped);
        Ok(())
    }
    pub(crate) fn register(
        &mut self,
        key: u64,
        logical: StreamId,
        mode: NativeMode,
    ) -> Result<(), ServerError> {
        if key == 0
            || self.mode.is_some_and(|fixed| fixed != mode)
            || self.sets.contains_key(&key)
            || self.sets.values().any(|slot| slot.logical == logical)
            || match mode {
                NativeMode::SingleCompat => logical.value != 0 || !self.sets.is_empty(),
                NativeMode::Scoped => logical.value == 0 || self.sets.len() >= MAX_SETS,
            }
        {
            return Err(failure(
                "HIP fixed native mode/logical set identity/capacity",
            ));
        }
        self.mode = Some(mode);
        self.sets.insert(
            key,
            SetSlot {
                logical,
                modules: None,
                keys: HashMap::new(),
                routed: HashSet::new(),
            },
        );
        Ok(())
    }
    pub(crate) fn catalogue(
        &self,
        key: u64,
        originals: &[KernelId],
    ) -> Result<Vec<KernelId>, ServerError> {
        let slot = self
            .sets
            .get(&key)
            .ok_or_else(|| failure("HIP unregistered native set"))?;
        if slot.modules.is_some() || originals.is_empty() || originals.len() > super::MAX_KERNELS {
            return Err(failure("HIP finite native catalogue state/extent"));
        }
        let mut unique = HashSet::with_capacity(originals.len());
        originals
            .iter()
            .map(|original| {
                if original.type_name() == std::any::type_name::<NativeKernelScope>() {
                    return Err(failure("HIP native key already scoped"));
                }
                let routed = match self.mode {
                    Some(NativeMode::Scoped) => KernelId::new::<NativeKernelScope>()
                        .cube_dim(original.cube_dim)
                        .address_type(original.address_type)
                        .mode(original.mode)
                        .info(NativeKernelScope {
                            logical: slot.logical,
                            original: original.clone(),
                        }),
                    Some(NativeMode::SingleCompat) => original.clone(),
                    None => return Err(failure("HIP native mode absent")),
                };
                if !unique.insert(routed.clone()) {
                    return Err(failure("HIP duplicate concrete native specialization"));
                }
                Ok(routed)
            })
            .collect()
    }
    pub(crate) fn is_scoped(&self) -> bool {
        self.mode == Some(NativeMode::Scoped)
    }
    pub(crate) fn install(
        &mut self,
        key: u64,
        modules: PreparedModules,
        originals: &[KernelId],
        catalogue: &[KernelId],
    ) {
        let scoped = self.is_scoped();
        let slot = self.sets.get_mut(&key).expect("validated native slot");
        // Cold retained full-Eq keys; enqueue clones the prepared Arc payload.
        // Tables belong only to this live set, never a context-wide tombstone map.
        slot.prepare_keys(scoped, originals, catalogue);
        slot.modules = Some(modules);
    }
    pub(crate) fn require_stream(&self, logical: StreamId) -> Result<(), ServerError> {
        if !self.active() {
            return Ok(());
        }
        let slot = self
            .sets
            .values()
            .find(|slot| slot.logical == logical)
            .ok_or_else(|| failure("HIP default/foreign/retired logical stream"))?;
        // Registration alone grants no allocation, IO or native launch.
        let p = slot
            .modules
            .as_ref()
            .ok_or_else(|| failure("HIP native source stream is not prepared"))?;
        {
            if p.failed || p.retired {
                return Err(failure("HIP failed/retired native source stream"));
            }
            p.admission.require_usable()?;
        }
        Ok(())
    }
    pub(crate) fn launch_key(
        &self,
        logical: StreamId,
        original: &KernelId,
    ) -> Result<KernelId, ServerError> {
        if !self.active() {
            return Ok(original.clone());
        }
        self.require_stream(logical)?;
        let slot = self
            .sets
            .values()
            .find(|slot| slot.logical == logical)
            .ok_or_else(|| failure("HIP native logical set missing"))?;
        let p = slot
            .modules
            .as_ref()
            .filter(|p| p.sealed && !p.failed && !p.retired)
            .ok_or_else(|| failure("HIP native set not sealed/live"))?;
        p.admission.require_usable()?;
        if original.type_name() == std::any::type_name::<NativeKernelScope>() {
            return Err(failure("HIP nested native launch scope"));
        }
        let routed =
            match self.mode {
                Some(NativeMode::SingleCompat) => original.clone(),
                Some(NativeMode::Scoped) => slot.keys.get(original).cloned().ok_or_else(|| {
                    failure("HIP sealed original specialization miss; no fallback")
                })?,
                None => unreachable!(),
            };
        if !p.kernels.contains_key(&routed) {
            return Err(failure("HIP sealed native catalogue miss; no fallback"));
        }
        Ok(routed)
    }
    pub(crate) fn prepared(&self, kernel: &KernelId) -> Option<&PreparedModules> {
        self.sets
            .values()
            .find(|slot| {
                self.mode == Some(NativeMode::SingleCompat) || slot.routed.contains(kernel)
            })?
            .modules
            .as_ref()
    }
    pub(crate) fn prepared_mut(&mut self, kernel: &KernelId) -> Option<&mut PreparedModules> {
        let compat = self.mode == Some(NativeMode::SingleCompat);
        self.sets
            .values_mut()
            .find(|slot| compat || slot.routed.contains(kernel))?
            .modules
            .as_mut()
    }
    pub(crate) fn set(&self, key: u64) -> Option<&PreparedModules> {
        self.sets.get(&key)?.modules.as_ref()
    }
    pub(crate) fn set_mut(&mut self, key: u64) -> Option<&mut PreparedModules> {
        self.sets.get_mut(&key)?.modules.as_mut()
    }
    pub(crate) fn cancel_registration(&mut self, key: u64) -> Result<(), ServerError> {
        if self
            .sets
            .get(&key)
            .is_none_or(|slot| slot.modules.is_some())
        {
            return Err(failure(
                "HIP nonempty/absent native registration cannot cancel",
            ));
        }
        self.sets.remove(&key);
        Ok(())
    }
    pub(crate) fn retire(&mut self, key: u64) {
        self.sets.remove(&key);
    }
    pub(crate) fn observation(&self) -> Result<super::HipScopedObservation, ServerError> {
        if !self.is_scoped() {
            return Err(failure("HIP observation requires fixed Scoped mode"));
        }
        let mut pair = self.sets.values();
        let overlap = match (pair.next(), pair.next()) {
            (Some(a), Some(b)) => a
                .keys
                .keys()
                .filter(|key| b.keys.contains_key(*key))
                .count(),
            _ => 0,
        };
        enum SealedMiss {}
        let missing = KernelId::new::<SealedMiss>();
        let sets = self
            .sets
            .iter()
            .map(|(&key, slot)| {
                let p = slot.modules.as_ref();
                Ok(super::HipScopedSetObservation {
                    set_key: key,
                    logical_stream: slot.logical.value,
                    catalogue_keys: slot.keys.len(),
                    non_null_modules: p.map_or(0, |p| {
                        p.kernels.values().filter(|k| !k.module.is_null()).count()
                    }),
                    admitted_native: p
                        .map_or_else(super::HipNativeInventory::default, |p| p.inventory),
                    scoped_metadata_bytes: if p.is_some() {
                        scoped_metadata_bytes(slot.keys.len())?
                    } else {
                        0
                    },
                    sealed: p.is_some_and(|p| p.sealed),
                    usable: self.require_stream(slot.logical).is_ok(),
                    sealed_miss_refused: self.launch_key(slot.logical, &missing).is_err(),
                })
            })
            .collect::<Result<Vec<_>, ServerError>>()?;
        Ok(super::HipScopedObservation {
            sets,
            original_key_overlap: overlap,
            default_stream_refused: self.require_stream(StreamId { value: 0 }).is_err(),
        })
    }
    pub(crate) fn fence_all(&mut self) {
        for slot in self.sets.values_mut() {
            if let Some(p) = &mut slot.modules {
                p.failed = true;
                p.admission.fence_unknown();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_modes_bounded_registration_and_retired_default_refusal() {
        let mut sets = NativeSets::default();
        assert!(
            sets.register(1, StreamId { value: 0 }, NativeMode::Scoped)
                .is_err()
        );
        sets.register(1, StreamId { value: 1 }, NativeMode::Scoped)
            .unwrap();
        sets.register(2, StreamId { value: 2 }, NativeMode::Scoped)
            .unwrap();
        assert!(
            sets.register(3, StreamId { value: 3 }, NativeMode::Scoped)
                .is_err()
        );
        assert!(sets.require_stream(StreamId { value: 0 }).is_err());
        sets.cancel_registration(1).unwrap();
        assert!(sets.require_stream(StreamId { value: 1 }).is_err());
        assert!(
            sets.register(4, StreamId { value: 0 }, NativeMode::SingleCompat)
                .is_err()
        );
        sets.register(4, StreamId { value: 4 }, NativeMode::Scoped)
            .unwrap();
    }
    #[test]
    fn complete_original_info_cube_mode_address_and_scope_are_not_replaced() {
        let mut sets = NativeSets::default();
        sets.register(1, StreamId { value: 1 }, NativeMode::Scoped)
            .unwrap();
        sets.register(2, StreamId { value: 2 }, NativeMode::Scoped)
            .unwrap();
        let first = KernelId::new::<u32>().info((7u32, "original specializations"));
        let other = KernelId::new::<u32>().info((8u32, "original specializations"));
        let a = sets.catalogue(1, &[first.clone(), other.clone()]).unwrap();
        let b = sets.catalogue(2, &[first.clone()]).unwrap();
        assert_ne!(a[0], a[1]);
        assert_ne!(a[0], b[0]);
        assert!(sets.catalogue(1, &[first.clone(), first]).is_err());
        assert!(sets.catalogue(1, &[a[0].clone()]).is_err());
        sets.cancel_registration(1).unwrap();
        sets.register(3, StreamId { value: 3 }, NativeMode::Scoped)
            .unwrap();
        assert!(sets.catalogue(3, &[a[0].clone()]).is_err());
        assert_eq!(a[0].cube_dim, other.cube_dim);
    }
    #[test]
    fn all_original_specializations_survive_scoping() {
        use cubecl_core::ir::{
            AddressType,
            settings::{Dim3, ExecutionMode},
        };
        let mut sets = NativeSets::default();
        sets.register(1, StreamId { value: 1 }, NativeMode::Scoped)
            .unwrap();
        let base = KernelId::new::<u32>()
            .info(7u32)
            .cube_dim(Dim3::new_1d(32))
            .address_type(AddressType::U32)
            .mode(ExecutionMode::Checked);
        let originals = [
            base.clone(),
            base.clone().cube_dim(Dim3::new_1d(64)),
            base.clone().address_type(AddressType::U64),
            base.clone().mode(ExecutionMode::Unchecked),
            KernelId::new::<u64>()
                .info(7u32)
                .cube_dim(Dim3::new_1d(32))
                .address_type(AddressType::U32)
                .mode(ExecutionMode::Checked),
        ];
        let routed = sets.catalogue(1, &originals).unwrap();
        assert_eq!(routed.iter().collect::<HashSet<_>>().len(), originals.len());
        for (a, b) in originals.iter().zip(routed) {
            assert_eq!(a.cube_dim, b.cube_dim);
            assert_eq!(a.address_type, b.address_type);
            assert_eq!(a.mode, b.mode);
        }
    }
    #[test]
    fn cold_full_keys_retire_without_tombstones_or_new_enqueue_payloads() {
        let mut sets = NativeSets::default();
        sets.register(1, StreamId { value: 1 }, NativeMode::Scoped)
            .unwrap();
        sets.register(2, StreamId { value: 2 }, NativeMode::Scoped)
            .unwrap();
        let original = KernelId::new::<u32>().info((7u32, "full concrete original"));
        let a = sets.catalogue(1, std::slice::from_ref(&original)).unwrap();
        let b = sets.catalogue(2, std::slice::from_ref(&original)).unwrap();
        sets.sets
            .get_mut(&1)
            .unwrap()
            .prepare_keys(true, std::slice::from_ref(&original), &a);
        sets.sets
            .get_mut(&2)
            .unwrap()
            .prepare_keys(true, std::slice::from_ref(&original), &b);
        for _ in 0..16 {
            assert_eq!(sets.sets[&1].keys[&original], a[0]);
            assert_eq!(sets.sets[&2].keys[&original], b[0]);
        }
        // No modules/handles/driver are fabricated: this exercises cold key
        // ownership only; real sibling execution/unload remains a physical gate.
        sets.retire(1);
        assert_eq!(sets.sets.len(), 1);
        assert_eq!(sets.sets[&2].keys[&original], b[0]);
        sets.register(3, StreamId { value: 3 }, NativeMode::Scoped)
            .unwrap();
        let c = sets.catalogue(3, std::slice::from_ref(&original)).unwrap();
        assert_ne!(c[0], a[0]);
        assert!(sets.catalogue(3, &a).is_err());
        assert!(sets.require_stream(StreamId { value: 1 }).is_err());
        assert!(sets.require_stream(StreamId { value: 3 }).is_err());
    }
    #[test]
    fn scoped_metadata_is_finite_checked_and_covers_cold_payloads() {
        assert!(scoped_metadata_bytes(0).is_err());
        assert!(scoped_metadata_bytes(super::super::MAX_KERNELS + 1).is_err());
        assert!(scoped_metadata_bytes(usize::MAX).is_err());
        let small = scoped_metadata_bytes(117).unwrap();
        let resident = scoped_metadata_bytes(184).unwrap();
        let policy = scoped_metadata_bytes(1024).unwrap();
        assert!(small <= resident && resident <= policy);
        assert!(policy < 2 * 1024 * 1024);
        println!("HIP_SCOPED_METADATA qwen117={small} gemma184={resident} policy1024={policy}");
    }
}
