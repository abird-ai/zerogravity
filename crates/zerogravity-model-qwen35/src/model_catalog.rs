//! Minimal native model catalog: alias -> ONE immutable source-backed model.
//!
//! This is the model-lifecycle foundation for a future local/distributed router,
//! not a pretend HTTP server or a second inference runtime. A model can have
//! many independent Qwen sessions, all sharing the same source GGUF allocation
//! and immutable predecoded layer constants. Loads for one source are deduped;
//! unrelated source loads don't block each other. No hidden background threads,
//! auto-download, fallback model substitution or automatic device allocation.
//!
//! Catalog removal revokes NEW lookups but does not invalidate already-created
//! sessions, prepared StateRoots or GPU-held immutable source buffers.
#[cfg(feature = "cpu")]
use crate::QwenDenseCpu;
use crate::{Error, LoadedQwenModel, Result};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, Weak},
};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SourceKey {
    gguf: PathBuf,
    config: PathBuf,
}

struct ModelSlot {
    source: SourceKey,
    model: OnceLock<Arc<LoadedQwenModel>>,
    load_guard: Mutex<()>,
}

impl ModelSlot {
    fn load(&self) -> Result<Arc<LoadedQwenModel>> {
        if let Some(model) = self.model.get() {
            return Ok(model.clone());
        }
        // Only the same source serializes through this short-lived load guard.
        // Different models may open concurrently; a failed load is retryable.
        let _loading = self
            .load_guard
            .lock()
            .map_err(|_| Error::State("model load lock poisoned".into()))?;
        if let Some(model) = self.model.get() {
            return Ok(model.clone());
        }
        let ready = LoadedQwenModel::open(&self.source.gguf, &self.source.config)?;
        self.model
            .set(ready.clone())
            .map_err(|_| Error::State("concurrent model publication bypassed load guard".into()))?;
        Ok(ready)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogEntry {
    pub alias: String,
    pub loaded: bool,
    pub source_bytes: Option<usize>,
}

#[derive(Default)]
struct CatalogState {
    aliases: BTreeMap<String, Arc<ModelSlot>>,
    /// Weak-only mapping lets a removed/unreferenced source disappear without
    /// keeping a second hidden large-GGUF cache alive. Explicit registration
    /// of multiple aliases of an active source dedupes to the same slot.
    by_source: BTreeMap<SourceKey, Weak<ModelSlot>>,
}

/// An explicit, thread-safe, source-identity-preserving model registry.
///
/// New sessions always receive fresh mutable recurrence/KV caches; no catalog
/// method can modify or force-release another session's committed root.
#[derive(Clone, Default)]
pub struct ModelCatalog {
    state: Arc<Mutex<CatalogState>>,
}

impl ModelCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register immutable local artifacts by a stable human-friendly alias.
    /// Repeating the exact same alias/source is idempotent; alias replacement
    /// requires explicit unregister/drain rather than silent model switching.
    ///
    /// Canonicalize paths to avoid alternate aliases for one source filename;
    /// actual GGUF bytes are loaded once from an opened descriptor with
    /// bounded reparse, not re-resolved from the pathname per session.
    pub fn register(&self, alias: &str, gguf: &Path, config: &Path) -> Result<()> {
        if alias.is_empty()
            || alias.len() > 128
            || !alias
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
        {
            return Err(Error::Malformed(
                "model alias must be 1..128 ASCII letters/digits/dash/underscore/dot".into(),
            ));
        }
        let key = SourceKey {
            gguf: fs::canonicalize(gguf)?,
            config: fs::canonicalize(config)?,
        };
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::State("model catalog mutex poisoned".into()))?;
        if let Some(existing) = state.aliases.get(alias) {
            return if existing.source == key {
                Ok(())
            } else {
                Err(Error::State(
                    "model alias already bound to another source".into(),
                ))
            };
        }
        let slot = match state.by_source.get(&key).and_then(Weak::upgrade) {
            Some(active) => active,
            None => {
                let fresh = Arc::new(ModelSlot {
                    source: key.clone(),
                    model: OnceLock::new(),
                    load_guard: Mutex::new(()),
                });
                state.by_source.insert(key, Arc::downgrade(&fresh));
                fresh
            }
        };
        state.aliases.insert(alias.to_owned(), slot);
        Ok(())
    }

    /// Only a successful model open is cached. A failed load does not poison
    /// the entire catalog or lock out an unrelated registered model.
    pub fn load(&self, alias: &str) -> Result<Arc<LoadedQwenModel>> {
        let slot = {
            let state = self
                .state
                .lock()
                .map_err(|_| Error::State("model catalog mutex poisoned".into()))?;
            state
                .aliases
                .get(alias)
                .cloned()
                .ok_or_else(|| Error::State(format!("model '{alias}' not registered")))?
        };
        slot.load()
    }

    #[cfg(feature = "cpu")]
    pub fn new_session(&self, alias: &str) -> Result<QwenDenseCpu> {
        self.load(alias)?.new_session()
    }

    /// Revokes future name-based sessions, not current sessions. Already
    /// created requests keep their own immutable model Arc and StateRoot.
    /// The caller is responsible for draining sessions before capacity reuse.
    pub fn unregister(&self, alias: &str) -> Result<bool> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::State("model catalog mutex poisoned".into()))?;
        let removed = state.aliases.remove(alias).is_some();
        state.by_source.retain(|_, slot| slot.strong_count() > 0);
        Ok(removed)
    }

    /// Source provenance and lifecycle reporting without per-model mutexes,
    /// tensor copies or accidentally materializing/starting a model.
    pub fn list(&self) -> Result<Vec<CatalogEntry>> {
        let state = self
            .state
            .lock()
            .map_err(|_| Error::State("model catalog mutex poisoned".into()))?;
        Ok(state
            .aliases
            .iter()
            .map(|(alias, slot)| {
                let model = slot.model.get();
                CatalogEntry {
                    alias: alias.clone(),
                    loaded: model.is_some(),
                    source_bytes: model.map(|m| m.source_bytes()),
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_catalog_aliases_rejected_without_disk_or_gpu_side_effects() {
        let catalog = ModelCatalog::new();
        let absent = Path::new("/nonexistent-zerogravity-source.gguf");
        for alias in [
            "",
            "../other",
            "name/other",
            " ",
            "bad
name",
        ] {
            assert!(catalog.register(alias, absent, absent).is_err());
        }
        assert_eq!(catalog.list().unwrap(), vec![]);
        assert!(catalog.load("unknown").is_err());
        assert!(!catalog.unregister("unknown").unwrap());
    }

    #[test]
    fn catalog_lists_without_implicitly_opening_or_starting_model() {
        let catalog = ModelCatalog::new();
        let dir = std::env::temp_dir().join(format!("zerogravity-catalog-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("model.gguf");
        let config = dir.join("config.json");
        fs::write(&gguf, b"invalid source but registration is lazy").unwrap();
        fs::write(&config, b"{}").unwrap();
        catalog.register("small-model", &gguf, &config).unwrap();
        catalog.register("other-alias", &gguf, &config).unwrap();
        catalog.register("small-model", &gguf, &config).unwrap();
        let list = catalog.list().unwrap();
        assert_eq!(list.len(), 2);
        assert!(list.iter().all(|x| !x.loaded && x.source_bytes.is_none()));
        assert!(catalog.register("small-model", &config, &gguf).is_err());
        assert!(catalog.load("small-model").is_err());
        assert!(catalog.load("other-alias").is_err());
        assert!(catalog.unregister("small-model").unwrap());
        assert_eq!(catalog.list().unwrap().len(), 1);
        assert!(catalog.unregister("other-alias").unwrap());
        assert!(catalog.list().unwrap().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }
}
