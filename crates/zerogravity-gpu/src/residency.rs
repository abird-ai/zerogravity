//! P07-C bounded source-versioned residency and completion ownership.
//! One provider holds this table across requests. Its mutex remains locked
//! through GPU dispatch and synchronous readback; eviction cannot retire
//! buffers referenced by in-flight work.
//! Host-only policy tests do not require GPU drivers.

use std::collections::BTreeMap;
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::{MatVecRegion, SourceIdentity};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ResidencyKey {
    source: SourceIdentity,
    tensor: String,
    kind: QuantKind,
    columns: usize,
    rows: usize,
}
impl ResidencyKey {
    pub fn from_region(region: MatVecRegion<'_>) -> Result<Self> {
        if !region.source_identity.is_bound() {
            return Err(Error::State(
                "unbound source identity cannot be resident".into(),
            ));
        }
        let row_bytes = usize::try_from(region.kind.byte_len(region.columns as u64)?)
            .map_err(|_| Error::Limit("GPU resident row bytes overflow".into()))?;
        if region.tensor_name.is_empty()
            || region.columns == 0
            || region.rows == 0
            || row_bytes.checked_mul(region.rows) != Some(region.packed_weights.len())
        {
            return Err(Error::Malformed(
                "GPU residency source geometry invalid".into(),
            ));
        }
        Ok(Self {
            source: region.source_identity,
            tensor: region.tensor_name.to_owned(),
            kind: region.kind,
            columns: region.columns,
            rows: region.rows,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyStats {
    pub source_uploads: usize,
    pub source_cache_hits: usize,
    pub source_evictions: usize,
    pub source_uploaded_bytes: usize,
    pub resident_device_bytes: usize,
    pub cached_source_tensors: usize,
    pub completed_regions: usize,
    pub device_quantized_regions: usize,
    pub failed_regions: usize,
    pub dynamic_uploaded_bytes: usize,
    pub completed_readback_bytes: usize,
}
struct Entry<T> {
    value: T,
    bytes: usize,
    last_used: u64,
}

/// Bounded cache independent of CubeCL: T is the retained physical buffer
/// lease plus host-side immutable tensor metadata. Eviction is safe ONLY when
/// caller holds exclusive execution ownership until a confirmed Ready fence.
pub struct ResidentCache<T> {
    budget_bytes: usize,
    entries: BTreeMap<ResidencyKey, Entry<T>>,
    consumed: usize,
    clock: u64,
    poisoned: bool,
    stats: ResidencyStats,
}
impl<T> ResidentCache<T> {
    pub fn new(budget_bytes: usize) -> Result<Self> {
        if budget_bytes == 0 {
            return Err(Error::Limit("zero GPU resident memory budget".into()));
        }
        Ok(Self {
            budget_bytes,
            entries: BTreeMap::new(),
            consumed: 0,
            clock: 0,
            poisoned: false,
            stats: ResidencyStats::default(),
        })
    }
    pub fn stats(&self) -> ResidencyStats {
        self.stats
    }

    /// Combined in-provider admission. P07-E tracked transient scratch
    /// separately, but using a full weight budget *and* a full scratch
    /// budget would permit aggregate device overcommit. This check includes
    /// retained static weights plus the region's activation/scale and output.
    ///
    /// The caller exclusively holds its device-service resident-cache lock
    /// until completion, so the source-weight footprint cannot change during
    /// an admitted kernel. Under G2a this local weight/scratch subdivision
    /// is nested inside exactly one shared physical provider envelope, never
    /// an independent global physical budget or additional device charge.
    pub fn ensure_weight_and_scratch_budget(
        &self,
        dynamic_bytes: usize,
        output_bytes: usize,
    ) -> Result<()> {
        let requested = self
            .consumed
            .checked_add(dynamic_bytes)
            .and_then(|x| x.checked_add(output_bytes))
            .ok_or_else(|| {
                Error::Limit("combined GPU weight/scratch accounting overflow".into())
            })?;
        if requested > self.budget_bytes {
            return Err(Error::Limit(format!(
                "GPU region needs {requested} bytes including resident weights, exceeds per-provider budget {}",
                self.budget_bytes
            )));
        }
        Ok(())
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// A failed GPU readback may leave device work in-flight. Fence this
    /// provider; keep retained handles alive until provider teardown.
    pub fn poison(&mut self) {
        self.poisoned = true;
        self.stats.failed_regions += 1;
    }

    /// Legacy source-only policy entrypoint retained for host test fixtures.
    /// Physical CUDA/Vulkan providers always reserve source + scratch together.
    #[cfg(test)]
    pub fn get_or_try_upload(
        &mut self,
        key: ResidencyKey,
        bytes: usize,
        upload: impl FnOnce() -> Result<T>,
    ) -> Result<&T> {
        self.get_or_try_upload_with_scratch(key, bytes, 0, 0, upload)
    }

    /// Borrow immutable source metadata without changing LRU accounting.
    /// Used ONLY while the physical provider holds its exclusive cache lock
    /// to validate live activation math before any device command is issued.
    pub fn cached(&self, key: &ResidencyKey) -> Option<&T> {
        self.entries.get(key).map(|entry| &entry.value)
    }

    /// Reconcile idle LRU residency with the *next* source + scratch working
    /// set while the physical card is still unsubmitted. No device command
    /// is launched here. Reserve the selected source even on cache hits.
    pub fn ensure_source_and_scratch_room(
        &mut self,
        key: &ResidencyKey,
        bytes: usize,
        dynamic_bytes: usize,
        output_bytes: usize,
    ) -> Result<()> {
        if self.poisoned {
            return Err(Error::State(
                "GPU residency fenced after uncertain completion".into(),
            ));
        }
        let available = self
            .budget_bytes
            .checked_sub(dynamic_bytes)
            .and_then(|n| n.checked_sub(output_bytes))
            .ok_or_else(|| Error::Limit("GPU scratch/output exceeds provider budget".into()))?;
        if bytes == 0 || bytes > available {
            return Err(Error::Limit(format!(
                "GPU source {bytes} bytes plus scratch/output exceeds provider budget {}",
                self.budget_bytes
            )));
        }
        let hit = self.entries.get(key).map(|entry| entry.bytes);
        if hit.is_some_and(|size| size != bytes) {
            return Err(Error::State(
                "GPU source lease size changed for one key".into(),
            ));
        }
        // Overflow errors are detected before ANY eviction.
        let mut required = self
            .consumed
            .checked_add(if hit.is_some() { 0 } else { bytes })
            .ok_or_else(|| Error::Limit("resident size overflow".into()))?;
        self.clock
            .checked_add(1)
            .ok_or_else(|| Error::Limit("resident LRU clock overflow".into()))?;
        while required > available {
            let victim = self
                .entries
                .iter()
                .filter(|(other, _)| *other != key)
                .min_by_key(|(other, entry)| (entry.last_used, *other))
                .map(|(other, _)| other.clone())
                .ok_or_else(|| {
                    Error::State("GPU resident working set cannot be admitted".into())
                })?;
            let old = self
                .entries
                .remove(&victim)
                .ok_or_else(|| Error::State("resident victim disappeared".into()))?;
            self.consumed -= old.bytes;
            self.stats.source_evictions += 1;
            self.stats.resident_device_bytes = self.consumed;
            self.stats.cached_source_tensors = self.entries.len();
            required = self
                .consumed
                .checked_add(if hit.is_some() { 0 } else { bytes })
                .ok_or_else(|| Error::Limit("resident size overflow".into()))?;
        }
        Ok(())
    }

    /// Admit the entire *working set*, not only static weights. Evict only
    /// other, idle cached tensors to leave room for this region's activation
    /// and output before creating any new GPU source handle. The provider
    /// exclusively holds this cache through physical readback, so an evicted
    /// handle has no in-flight users. The current source may never be evicted
    /// to satisfy its own scratch reservation.
    pub fn get_or_try_upload_with_scratch(
        &mut self,
        key: ResidencyKey,
        bytes: usize,
        dynamic_bytes: usize,
        output_bytes: usize,
        upload: impl FnOnce() -> Result<T>,
    ) -> Result<&T> {
        self.ensure_source_and_scratch_room(&key, bytes, dynamic_bytes, output_bytes)?;
        let hit = self.entries.contains_key(&key);
        let required = self
            .consumed
            .checked_add(if hit { 0 } else { bytes })
            .ok_or_else(|| Error::Limit("resident size overflow".into()))?;
        self.clock += 1; // checked by ensure_source_and_scratch_room

        if hit {
            let entry = self
                .entries
                .get_mut(&key)
                .expect("validated resident cache hit");
            entry.last_used = self.clock;
            self.stats.source_cache_hits += 1;
            return Ok(&entry.value);
        }
        // A failed upload never installs an incomplete source. A prior idle
        // eviction may still have occurred; the cache remains internally
        // consistent and the physical provider handles uncertain H2D work.
        let value = upload()?;
        self.consumed = required;
        self.entries.insert(
            key.clone(),
            Entry {
                value,
                bytes,
                last_used: self.clock,
            },
        );
        self.stats.source_uploads += 1;
        self.stats.source_uploaded_bytes += bytes;
        self.stats.resident_device_bytes = self.consumed;
        self.stats.cached_source_tensors = self.entries.len();
        Ok(&self
            .entries
            .get(&key)
            .expect("resident just inserted")
            .value)
    }

    /// Reclaim ONLY safely idle immutable source tensors to fit a transient
    /// non-matrix GPU region (e.g. RMS normalization). Caller must hold the
    /// provider's exclusive source cache lock through physical readback. An
    /// impossible transient request fails without mutating the cache or GPU.
    pub fn ensure_transient_room(
        &mut self,
        dynamic_bytes: usize,
        output_bytes: usize,
    ) -> Result<()> {
        if self.poisoned {
            return Err(Error::State(
                "GPU residency fenced after uncertain completion".into(),
            ));
        }
        let available = self
            .budget_bytes
            .checked_sub(dynamic_bytes)
            .and_then(|n| n.checked_sub(output_bytes))
            .ok_or_else(|| {
                Error::Limit("GPU transient region exceeds held provider budget".into())
            })?;
        while self.consumed > available {
            let key = self
                .entries
                .iter()
                .min_by_key(|(key, entry)| (entry.last_used, *key))
                .map(|(key, _)| key.clone())
                .ok_or_else(|| {
                    Error::State("GPU transient cache accounting inconsistent".into())
                })?;
            let victim = self
                .entries
                .remove(&key)
                .ok_or_else(|| Error::State("GPU transient LRU victim vanished".into()))?;
            self.consumed -= victim.bytes;
            self.stats.source_evictions += 1;
            self.stats.resident_device_bytes = self.consumed;
            self.stats.cached_source_tensors = self.entries.len();
        }
        Ok(())
    }

    /// Called only when GPU output and all memory effects are Ready and
    /// host-visible. Typed ready receipt stays owned by sole coordinator.
    pub fn complete(&mut self, dynamic_bytes: usize, readback_bytes: usize) -> Result<()> {
        if self.poisoned {
            return Err(Error::State("cannot complete fenced GPU provider".into()));
        }
        let completed = self
            .stats
            .completed_regions
            .checked_add(1)
            .ok_or_else(|| Error::Limit("completed region counter overflow".into()))?;
        let dynamic = self
            .stats
            .dynamic_uploaded_bytes
            .checked_add(dynamic_bytes)
            .ok_or_else(|| Error::Limit("dynamic upload counter overflow".into()))?;
        let readback = self
            .stats
            .completed_readback_bytes
            .checked_add(readback_bytes)
            .ok_or_else(|| Error::Limit("readback counter overflow".into()))?;
        self.stats.completed_regions = completed;
        self.stats.dynamic_uploaded_bytes = dynamic;
        self.stats.completed_readback_bytes = readback;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "qualification")]
    use zerogravity_cpu::weights::CpuWeights;

    /// A CPU-only mathematical interpretation of the *physically resident*
    /// Q4_0 nibble buffer. No device is invoked by this test-only kernel.
    #[cfg(feature = "qualification")]
    fn host_resident_q4(prepared: &crate::q4_q8::PreparedQ4Q8, stored: &[u8]) -> Vec<f32> {
        assert_eq!(stored.len(), prepared.rows * prepared.blocks_per_row * 16);
        assert!(prepared.weights.is_empty());
        let mut values = vec![0f32; prepared.rows];
        for (row, out) in values.iter_mut().enumerate() {
            let mut acc = 0f32;
            for block in 0..prepared.blocks_per_row {
                let bid = row * prepared.blocks_per_row + block;
                let mut dot = 0i32;
                for i in 0..16 {
                    let v = stored[bid * 16 + i];
                    dot += (i32::from(v & 15) - 8) * prepared.activations[block * 32 + i];
                    dot += (i32::from(v >> 4) - 8) * prepared.activations[block * 32 + i + 16];
                }
                acc = (dot as f32).mul_add(prepared.scales[bid], acc);
            }
            *out = acc;
        }
        values
    }

    #[test]
    #[cfg(feature = "qualification")]
    #[ignore = "requires source-trained Qwen3.5 0.8B and 2B checkpoints"]
    fn real_model_cache_reuses_weights_across_tokens_and_preserves_state() {
        use std::sync::{Arc, Mutex};
        use zerogravity_core::accelerator::{MatVecKernel, RegionExecution};
        use zerogravity_model_qwen35::QwenDenseCpu;

        struct EmulatedResident {
            host: crate::host_descriptor::StaticSource,
            gpu_bytes: Vec<u8>,
        }
        struct ModeledCachedSource {
            cache: Mutex<ResidentCache<EmulatedResident>>,
        }
        impl MatVecKernel for ModeledCachedSource {
            fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
                if region.tensor_name != "blk.0.attn_gate.weight" {
                    return Ok(RegionExecution::Unsupported);
                }
                let key = ResidencyKey::from_region(region)?;
                let resident_bytes = crate::host_descriptor::StaticSource::gpu_bytes_for(region)?;
                let mut cache = self
                    .cache
                    .lock()
                    .map_err(|_| Error::State("test resident mutex poisoned".into()))?;
                let cached = cache.get_or_try_upload(key, resident_bytes, || {
                    let (host, gpu_bytes) =
                        crate::host_descriptor::StaticSource::from_region(region)?;
                    Ok(EmulatedResident { host, gpu_bytes })
                })?;
                let prepared = match cached.host.dynamic(region.activation)? {
                    crate::host_descriptor::DynamicStep::Q4(q) => q,
                    crate::host_descriptor::DynamicStep::Mixed(_) => {
                        return Err(Error::State("expected Q4_0 source".into()));
                    }
                };
                let result = host_resident_q4(&prepared, &cached.gpu_bytes);
                cache.complete(
                    prepared.scales.len() * 4 + prepared.activations.len() * 4,
                    result.len() * 4,
                )?;
                Ok(RegionExecution::Completed(result))
            }
        }
        let mut backend_bytes = 0usize;
        for model in ["qwen35-0.8b", "qwen35-2b"] {
            let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("target/models")
                .join(model);
            let model_path = directory.join("model.gguf");
            let config = directory.join("config.json");
            let tokens = [11u32, 353, 1007];
            let mut cpu = QwenDenseCpu::open(&model_path, &config).unwrap();
            let expected = cpu.prefill_transaction(&tokens).unwrap();
            let shared = Arc::new(ModeledCachedSource {
                cache: Mutex::new(ResidentCache::new(32 * 1024 * 1024).unwrap()),
            });
            let mut source =
                QwenDenseCpu::open_with_kernel(&model_path, &config, shared.clone()).unwrap();
            let actual = source.prefill_transaction(&tokens).unwrap();
            assert_eq!(actual, expected, "{model} resident source changed logits");
            assert_eq!(source.state_fingerprint(), cpu.state_fingerprint());
            assert_eq!(source.position(), tokens.len());
            assert_eq!(source.root_id(), cpu.root_id());
            let stats = shared.cache.lock().unwrap().stats();
            assert_eq!(stats.source_uploads, 1);
            assert_eq!(stats.source_cache_hits, tokens.len() - 1);
            assert_eq!(stats.completed_regions, tokens.len());
            assert_eq!(stats.cached_source_tensors, 1);
            assert_eq!(stats.source_uploaded_bytes, stats.resident_device_bytes);
            assert!(stats.dynamic_uploaded_bytes > 0);
            assert_eq!(stats.completed_readback_bytes, tokens.len() * 2048 * 4);
            backend_bytes += stats.source_uploaded_bytes;
            println!(
                "P07_D_DECODE_ONCE_RESIDENT_MODEL_PASS model={model} upload_once_bytes={} cache_hits={} completed={}",
                stats.source_uploaded_bytes, stats.source_cache_hits, stats.completed_regions
            );
        }
        assert!(backend_bytes > 1_000_000);
    }

    #[test]
    fn scratch_aware_lru_keeps_active_tensor_and_evicts_only_idle_weights() {
        fn key(name: &str) -> ResidencyKey {
            // Synthetic key valid ONLY for host cache policy tests. Real GPU
            // residency requires SourceIdentity from CpuWeights::open.
            ResidencyKey {
                source: SourceIdentity::unbound(),
                tensor: name.into(),
                kind: QuantKind::Q4_0,
                columns: 32,
                rows: 8,
            }
        }
        let mut cache = ResidentCache::new(120).unwrap();
        cache.get_or_try_upload(key("a"), 64, || Ok(10u64)).unwrap();
        cache.get_or_try_upload(key("b"), 40, || Ok(20u64)).unwrap();
        assert_eq!(cache.stats().resident_device_bytes, 104);

        let active = cache
            .get_or_try_upload_with_scratch(key("a"), 64, 32, 8, || -> Result<u64> {
                panic!("live source cache hit must never upload")
            })
            .unwrap();
        assert_eq!(*active, 10);
        assert_eq!(cache.stats().source_cache_hits, 1);
        assert_eq!(cache.stats().source_evictions, 1);
        assert_eq!(cache.stats().resident_device_bytes, 64);
        cache.ensure_weight_and_scratch_budget(32, 8).unwrap();

        // An impossible working set is rejected without evictions or upload.
        let before = cache.stats();
        assert!(
            cache
                .get_or_try_upload_with_scratch(key("a"), 64, 55, 8, || -> Result<u64> {
                    panic!("impossible static+scratch request must not upload")
                })
                .is_err()
        );
        assert_eq!(cache.stats(), before);
        // A new cache miss also reserves future scratch BEFORE uploading.
        let value = cache
            .get_or_try_upload_with_scratch(key("c"), 48, 50, 10, || Ok(30u64))
            .unwrap();
        assert_eq!(*value, 30);
        assert_eq!(cache.stats().source_evictions, 2);
        assert_eq!(cache.stats().resident_device_bytes, 48);
        cache.ensure_weight_and_scratch_budget(50, 10).unwrap();
    }

    #[test]
    fn transient_norm_evicts_only_idle_source_and_refuses_impossible_scratch() {
        fn key(name: &str) -> ResidencyKey {
            ResidencyKey {
                source: SourceIdentity::unbound(),
                tensor: name.to_owned(),
                kind: QuantKind::Q4_0,
                columns: 32,
                rows: 8,
            }
        }
        let mut cache = ResidentCache::new(128).unwrap();
        cache.get_or_try_upload(key("old"), 64, || Ok(1u8)).unwrap();
        cache
            .get_or_try_upload(key("recent"), 32, || Ok(2u8))
            .unwrap();
        let before = cache.stats();
        assert!(matches!(
            cache.ensure_transient_room(129, 1),
            Err(Error::Limit(_))
        ));
        assert_eq!(cache.stats(), before, "rejected request changed source LRU");
        cache.ensure_transient_room(65, 15).unwrap();
        assert_eq!(cache.stats().resident_device_bytes, 32);
        assert_eq!(cache.stats().cached_source_tensors, 1);
        assert_eq!(cache.stats().source_evictions, 1);
        cache.ensure_transient_room(96, 32).unwrap();
        assert_eq!(cache.stats().resident_device_bytes, 0);
        assert_eq!(cache.stats().source_evictions, 2);
        assert!(
            !cache.is_poisoned(),
            "safe preflight eviction fenced device"
        );
    }

    #[test]
    fn resident_plus_inflight_scratch_share_one_budget() {
        let mut cache = ResidentCache::<u8>::new(120).unwrap();
        // A test-only synthetic key is generated from source checkpoint below;
        // this direct conservative budget check requires no GGUF.
        assert!(cache.ensure_weight_and_scratch_budget(100, 20).is_ok());
        assert!(cache.ensure_weight_and_scratch_budget(100, 21).is_err());
        assert!(
            cache
                .ensure_weight_and_scratch_budget(usize::MAX, 2)
                .is_err()
        );
        // Simulate a physically resident 80-byte weight lease. This is
        // ledger accounting, not a claim about device memory hardware.
        cache.consumed = 80;
        assert!(cache.ensure_weight_and_scratch_budget(20, 20).is_ok());
        assert!(cache.ensure_weight_and_scratch_budget(20, 21).is_err());
        assert!(cache.ensure_weight_and_scratch_budget(41, 1).is_err());
        cache.consumed = 0;
    }

    #[test]
    fn unbound_source_and_budget_are_rejected() {
        assert!(!SourceIdentity::unbound().is_bound());
        assert!(ResidentCache::<u64>::new(0).is_err());
        let packed = [0u8; 18];
        let x = [0f32; 32];
        assert!(
            ResidencyKey::from_region(MatVecRegion {
                source_identity: SourceIdentity::unbound(),
                tensor_name: "test",
                kind: QuantKind::Q4_0,
                columns: 32,
                rows: 1,
                packed_weights: &packed,
                activation: &x
            })
            .is_err()
        );
    }

    #[test]
    #[cfg(feature = "qualification")]
    #[ignore = "requires original SHA-pinned local Qwen3.5 checkpoint"]
    fn source_reopen_cache_hit_eviction_failure_and_fence() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("target/models/qwen35-0.8b/model.gguf");
        let first = CpuWeights::open(&path).unwrap();
        let second = CpuWeights::open(&path).unwrap();
        assert_ne!(first.source_identity(), second.source_identity());
        let packed = [0u8; 18];
        let x = [0f32; 32];
        let key = |source, tensor: &str| {
            ResidencyKey::from_region(MatVecRegion {
                source_identity: source,
                tensor_name: tensor,
                kind: QuantKind::Q4_0,
                columns: 32,
                rows: 1,
                packed_weights: &packed,
                activation: &x,
            })
            .unwrap()
        };
        let a = key(first.source_identity(), "x");
        let b = key(second.source_identity(), "x");
        assert_ne!(a, b);
        let mut c = ResidentCache::new(100).unwrap();
        assert_eq!(*c.get_or_try_upload(a.clone(), 64, || Ok(7u64)).unwrap(), 7);
        assert_eq!(
            *c.get_or_try_upload(a.clone(), 64, || -> Result<u64> {
                panic!("should never upload on cache hit")
            })
            .unwrap(),
            7
        );
        assert!(c.get_or_try_upload(b.clone(), 101, || Ok(8u64)).is_err());
        assert_eq!(*c.get_or_try_upload(b.clone(), 64, || Ok(8u64)).unwrap(), 8);
        assert_eq!(c.stats().source_uploads, 2);
        assert_eq!(c.stats().source_cache_hits, 1);
        assert_eq!(c.stats().source_evictions, 1);
        assert_eq!(c.stats().resident_device_bytes, 64);
        c.complete(60, 16).unwrap();
        assert_eq!(c.stats().completed_regions, 1);
        assert_eq!(c.stats().dynamic_uploaded_bytes, 60);
        assert_eq!(c.stats().completed_readback_bytes, 16);
        let bad = c.get_or_try_upload(a.clone(), 64, || {
            Err(Error::State("injected upload failed".into()))
        });
        assert!(bad.is_err());
        assert_eq!(c.stats().source_uploads, 2);
        c.poison();
        assert!(c.is_poisoned());
        assert!(c.get_or_try_upload(b, 64, || Ok(9u64)).is_err());
        assert!(c.complete(4, 4).is_err());
        assert_eq!(c.stats().failed_regions, 1);
    }
}
