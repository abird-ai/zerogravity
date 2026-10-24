//! Bounded exact-issuer rows in the existing page; only the caller's real
//! armed-flight consumer checkpoint can publish a slot after begin().
use super::*;
pub(crate) const MAX_ROWS: usize = 64;
#[derive(Clone, Copy, Debug)]
pub(super) struct RowPlacement {
    pub slot: usize,
    pub offset: usize,
    pub hit: bool,
    capacity: usize,
    same_matrix: bool,
}
#[derive(Default)]
pub(super) struct RowCache {
    matrix: Option<VerifiedMatrix>,
    rows: Slots,
}
#[derive(Clone)]
struct Slots {
    rows: [Option<usize>; MAX_ROWS],
    next: usize,
}
impl Default for Slots {
    fn default() -> Self {
        Self {
            rows: [None; MAX_ROWS],
            next: 0,
        }
    }
}
impl Slots {
    fn select(&self, row: usize, capacity: usize) -> (usize, bool) {
        if let Some(slot) = self.rows[..capacity].iter().position(|r| *r == Some(row)) {
            return (slot, true);
        }
        (
            self.rows[..capacity]
                .iter()
                .position(Option::is_none)
                .unwrap_or(self.next % capacity),
            false,
        )
    }
}
pub(crate) fn capacity(row_bytes: usize) -> Result<usize> {
    if row_bytes == 0 || row_bytes > PAGE_BYTES || !row_bytes.is_multiple_of(16) {
        return Err(Error::Limit(
            "cached source row aligned complete extent".into(),
        ));
    }
    Ok((PAGE_BYTES / row_bytes).min(MAX_ROWS))
}
impl RowCache {
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
    pub(super) fn select(
        &self,
        matrix: &VerifiedMatrix,
        row: usize,
        row_bytes: usize,
    ) -> Result<RowPlacement> {
        if row >= matrix.rows()
            || row_bytes
                != matrix
                    .columns()
                    .checked_mul(2)
                    .ok_or_else(|| Error::Limit("cached row extent".into()))?
        {
            return Err(Error::State("cached source original row/stride".into()));
        }
        let capacity = capacity(row_bytes)?;
        let same_matrix = self.matrix.as_ref().is_some_and(|m| m.same_binding(matrix));
        let (slot, hit) = if same_matrix {
            self.rows.select(row, capacity)
        } else {
            (0, false)
        };
        Ok(RowPlacement {
            slot,
            offset: slot * row_bytes,
            hit,
            capacity,
            same_matrix,
        })
    }
    pub(super) fn begin(&mut self, matrix: &VerifiedMatrix, place: RowPlacement) {
        if !place.same_matrix {
            self.clear();
            self.matrix = Some(matrix.clone());
        }
        self.rows.rows[place.slot] = None;
    }
    pub(super) fn complete(&mut self, place: RowPlacement, row: usize) -> Result<()> {
        if self.matrix.is_none()
            || place.slot >= place.capacity
            || self.rows.rows[place.slot].is_some()
        {
            return Err(Error::State("cached source row checkpoint slot".into()));
        }
        self.rows.rows[place.slot] = Some(row);
        if !place.hit {
            self.rows.next = (place.slot + 1) % place.capacity;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_ple_rows_fit_original_page_and_control() {
        assert_eq!(capacity(17920).unwrap(), 58); // E2B35*256*BF16
        assert_eq!(capacity(21504).unwrap(), 48); // E4B42*256*BF16; no route permit
        assert_eq!(capacity(16).unwrap(), 64);
        assert!(std::mem::size_of::<Progress>() <= CONTROL_BYTES);
        for n in [0, 1, PAGE_BYTES + 16, usize::MAX] {
            assert!(capacity(n).is_err());
        }
    }

    #[test]
    fn exact_issued_binding_and_completed_checkpoint_govern_reuse() {
        use sha2::{Digest, Sha256};
        use std::fs::{self, File};
        use zerogravity_artifacts::source_weights::{
            ExpectedSourceShard, SourceWeights, WindowOpenLimits,
        };
        use zerogravity_artifacts::{artifact::Artifact, cache::TransferAdmission};
        use zerogravity_core::resource_arbiter::{
            PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
        };
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zg-page-rows-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        struct Remove(std::path::PathBuf);
        impl Drop for Remove {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let _remove = Remove(path.clone());
        let header=br#"{"a":{"dtype":"BF16","shape":[64,64],"data_offsets":[0,8192]},"b":{"dtype":"BF16","shape":[64,64],"data_offsets":[8192,16384]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.resize(bytes.len() + 16384, 0);
        fs::write(path.join("model.safetensors"), &bytes).unwrap();
        let index=br#"{"metadata":{"total_size":16384},"weight_map":{"a":"model.safetensors","b":"model.safetensors"}}"#;
        let sha = |b: &[u8]| format!("{:x}", Sha256::digest(b));
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("host", "page-row-source").unwrap();
        arbiter
            .register(resource.clone(), 64 * 1024 * 1024)
            .unwrap();
        let admission = TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("page-row-source", "cpu-only").unwrap(),
        };
        let open = || {
            SourceWeights::prepare_verified_windows(
                index,
                &sha(index),
                vec![
                    ExpectedSourceShard::new(
                        "model.safetensors".into(),
                        Artifact::from_file(
                            File::open(path.join("model.safetensors")).unwrap(),
                            std::path::Path::new("model.safetensors"),
                        )
                        .unwrap(),
                        bytes.len() as u64,
                        sha(&bytes),
                    )
                    .unwrap(),
                ],
                WindowOpenLimits {
                    max_source_bytes: 1024 * 1024,
                    max_metadata_bytes: 16 * 1024 * 1024,
                    max_shards: 1,
                    max_tensors: 2,
                    scan_chunk_bytes: 64,
                },
                &admission,
            )
            .unwrap()
            .open()
            .unwrap()
        };
        let first = open();
        let second = open();
        let a = first.matrix("a", MatrixAxes::OutputInput).unwrap();
        let b = first.matrix("b", MatrixAxes::OutputInput).unwrap();
        let foreign = second.matrix("a", MatrixAxes::OutputInput).unwrap();
        assert_ne!(a.source_identity(), foreign.source_identity());
        assert!(first.require_matrix_binding(&foreign).is_err());
        let mut cache = RowCache::default();
        let place = cache.select(&a, 7, 128).unwrap();
        assert!(!place.hit);
        assert!(cache.complete(place, 7).is_err());
        cache.begin(&a, place);
        assert!(!cache.select(&a, 7, 128).unwrap().hit);
        cache.complete(place, 7).unwrap();
        assert!(cache.complete(place, 7).is_err());
        let hit = cache.select(&a.clone(), 7, 128).unwrap();
        assert!(hit.hit);
        assert!(!cache.select(&a, 8, 128).unwrap().hit);
        assert!(!cache.select(&b, 7, 128).unwrap().hit);
        assert!(!cache.select(&foreign, 7, 128).unwrap().hit);
        assert!(cache.select(&a, 64, 128).is_err());
        assert!(cache.select(&a, 7, 256).is_err());
        // begin invalidates even a hit; dropping/unwinding before its actual
        // completion cannot leave that destination reusable.
        cache.begin(&a, hit);
        assert!(!cache.select(&a, 7, 128).unwrap().hit);
        cache.complete(hit, 7).unwrap();
        assert!(cache.select(&a, 7, 128).unwrap().hit);
        cache.clear();
        assert!(!cache.select(&a, 7, 128).unwrap().hit);
        first.require_current_matrix(&a).unwrap();
        assert!(first.require_current_matrix(&foreign).is_err());
        // A hit must never mask in-place mutation of the original held source.
        use std::io::{Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(path.join("model.safetensors"))
            .unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[1]).unwrap();
        file.sync_all().unwrap();
        assert!(first.require_current_matrix(&a).is_err());
    }

    #[test]
    fn row_slots_hit_evict_and_invalidate_only_exact_rows() {
        let mut slots = Slots::default();
        for row in 0..58 {
            let (slot, hit) = slots.select(row, 58);
            assert_eq!(slot, row);
            assert!(!hit);
            slots.rows[slot] = Some(row);
            slots.next = (slot + 1) % 58;
        }
        assert_eq!(slots.select(17, 58), (17, true));
        assert_eq!(slots.select(100, 58), (0, false));
        slots.rows[0] = None; // No row remains reusable while overwritten/uncertain.
        assert_eq!(slots.select(0, 58), (0, false));
        slots.rows[0] = Some(100);
        slots.next = 1;
        assert_eq!(slots.select(101, 58), (1, false));
        slots = Slots::default();
        assert_eq!(slots.select(17, 58), (0, false));
    }
}
