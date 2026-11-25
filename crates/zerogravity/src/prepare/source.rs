//! Cold GGUF source selection shared by direct planning and SDK preparation.
//! No source issuer or backend permit. Local files retain their opened
//! descriptor; cached references resolve to an exact immutable manifest.
use crate::{
    Error, Result, artifact::Artifact, qwen35::config::ModelGeometry, run_policy::SourceSelector,
};
use std::{
    fs::File,
    path::{Path, PathBuf},
};

pub(crate) struct LocalGguf {
    pub weights: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}
impl LocalGguf {
    pub(crate) fn new(
        weights: PathBuf,
        config: Option<PathBuf>,
        tokenizer: Option<PathBuf>,
    ) -> Self {
        let parent = weights.parent().unwrap_or(Path::new("."));
        let config = config.unwrap_or_else(|| parent.join("config.json"));
        let tokenizer = tokenizer.unwrap_or_else(|| parent.join("tokenizer.json"));
        Self {
            weights,
            config,
            tokenizer,
        }
    }

    pub(crate) fn open_weights(&self) -> Result<HeldWeights> {
        let file = File::open(&self.weights)?;
        HeldWeights::new(file, self.weights.clone())
    }

    pub(crate) fn read_config(&self) -> Result<(Vec<u8>, ModelGeometry)> {
        use std::io::Read;
        let file = File::open(&self.config)?;
        let stamp = file.metadata()?;
        if !stamp.is_file() || stamp.len() > 1024 * 1024 {
            return Err(Error::Limit("regular config1MiB required".into()));
        }
        let mut bytes = Vec::new();
        file.take(stamp.len() + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 != stamp.len() {
            return Err(Error::State("held config extent changed".into()));
        }
        let geometry = ModelGeometry::from_json(&bytes)?;
        Ok((bytes, geometry))
    }
}

/// Extent and actual opened file description before admission or inspection.
/// The path is only a format hint; it is never reopened for materialization.
pub(crate) struct HeldWeights {
    file: File,
    path: PathBuf,
    pub bytes: u64,
}
impl HeldWeights {
    pub(crate) fn new(file: File, path: PathBuf) -> Result<Self> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(Error::Malformed(
                "model source is not a regular file".into(),
            ));
        }
        Ok(Self {
            file,
            path,
            bytes: metadata.len(),
        })
    }

    pub(crate) fn inspect(self) -> Result<Artifact> {
        let artifact = Artifact::from_file(self.file, &self.path)?;
        if artifact.info.file_len != self.bytes {
            return Err(Error::State(
                "model extent changed after source selection".into(),
            ));
        }
        Ok(artifact)
    }
}

/// Actual geometry/target/context check shared by planning and all loaders.
pub(crate) fn require_geometry(
    geometry: &ModelGeometry,
    target: Option<&str>,
    context: usize,
) -> Result<String> {
    let actual = super::PreparedQwen::routing_target(geometry)?;
    if target.is_some_and(|t| !t.is_empty() && t != actual) {
        return Err(Error::Malformed(
            "declared model target differs from actual source geometry".into(),
        ));
    }
    if context as u64 > geometry.max_position_embeddings {
        return Err(Error::Limit(
            "requested context exceeds actual model geometry".into(),
        ));
    }
    Ok(actual)
}

pub(crate) enum GgufSource {
    Local(LocalGguf),
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Cached(Box<CachedGguf>),
}
impl GgufSource {
    /// Explicit existing source only. Never acquires or follows a reference
    /// again after selecting its immutable snapshot.
    pub(crate) fn resolve(
        model: &crate::pipeline::ModelInstance,
        cache_root: Option<&Path>,
    ) -> Result<Self> {
        if model.native_artifacts.is_some() {
            return Err(Error::Unsupported(
                "native artifact override is not a GGUF source".into(),
            ));
        }
        let selector = SourceSelector::parse(model.source.as_deref().ok_or_else(|| {
            Error::Malformed("selected model requires an explicit acquired source".into())
        })?)?;
        match selector {
            SourceSelector::Local(weights) => {
                if model.snapshot.is_some() {
                    return Err(Error::Malformed(
                        "snapshot is for cached provider/catalog sources".into(),
                    ));
                }
                Ok(Self::Local(LocalGguf::new(
                    weights,
                    model.model_config.as_ref().map(PathBuf::from),
                    model.tokenizer.as_ref().map(PathBuf::from),
                )))
            }
            other => {
                #[cfg(all(feature = "cache", target_os = "linux"))]
                {
                    Ok(Self::Cached(Box::new(CachedGguf::resolve(
                        model, other, cache_root,
                    )?)))
                }
                #[cfg(not(all(feature = "cache", target_os = "linux")))]
                {
                    let _ = (other, cache_root);
                    Err(Error::Unsupported(
                        "cached source selection requires Linux and cache; acquire explicitly"
                            .into(),
                    ))
                }
            }
        }
    }
}

#[cfg(all(feature = "cache", target_os = "linux"))]
pub(crate) struct CachedGguf {
    pub root: PathBuf,
    disk: u64,
    #[cfg(feature = "cpu")]
    metadata: crate::cache::CacheMetadata,
    snapshot: crate::cache::SnapshotMetadata,
    pub weights: crate::cache::ManifestEntry,
    pub config: crate::cache::ManifestEntry,
    pub tokenizer: crate::cache::ManifestEntry,
}
#[cfg(all(feature = "cache", target_os = "linux"))]
impl CachedGguf {
    fn resolve(
        model: &crate::pipeline::ModelInstance,
        selector: SourceSelector,
        cache_root: Option<&Path>,
    ) -> Result<Self> {
        use crate::cache::{CacheMetadata, FileRole};
        if model.model_config.is_some() || model.tokenizer.is_some() {
            return Err(Error::Malformed("cached run requires role-bound config/tokenizer in the selected immutable snapshot; local sidecar overrides would break provenance".into()));
        }
        // Validate provider locator even when the cache root is absent.
        match &selector {
            SourceSelector::HuggingFace(value) | SourceSelector::GitHub(value) => {
                let provider = if matches!(selector, SourceSelector::HuggingFace(_)) {
                    "huggingface"
                } else {
                    "github"
                };
                crate::cache::provenance::ProviderLocator::parse(provider, value)?;
            }
            SourceSelector::Http(url) => {
                #[cfg(feature = "acquisition")]
                crate::acquisition::object::HttpsObjectSpec::validate_cached_locator(url)?;
                #[cfg(not(feature = "acquisition"))]
                {
                    let _ = url;
                    return Err(Error::Unsupported("HTTP source form requires compiled acquisition contract parser; no request is performed".into()));
                }
            }
            _ => {}
        }
        let root = crate::cache::paths::resolve_root(
            cache_root,
            std::env::var_os("XDG_CACHE_HOME").as_deref(),
            std::env::var_os("HOME").as_deref(),
        )?;
        let metadata = CacheMetadata::open(&root)?;
        let snapshot = metadata.select(&selector, model.snapshot.as_deref())?;
        let unique = |role| -> Result<crate::cache::ManifestEntry> {
            let mut entries = snapshot
                .manifest
                .entries()
                .iter()
                .filter(|e| e.role() == role);
            let entry = entries.next().ok_or_else(|| {
                Error::Malformed("selected snapshot lacks weights/config/tokenizer role".into())
            })?;
            if entries.next().is_some() {
                return Err(Error::Malformed(
                    "selected snapshot has ambiguous weights/config/tokenizer role".into(),
                ));
            }
            Ok(entry.clone())
        };
        let weights = unique(FileRole::Weights)?;
        let config = unique(FileRole::Config)?;
        let tokenizer = unique(FileRole::Tokenizer)?;
        if config.bytes() > 1024 * 1024 || tokenizer.bytes() > 64 * 1024 * 1024 {
            return Err(Error::Limit(
                "cached GGUF sidecar extent exceeds parser bound".into(),
            ));
        }
        let disk = metadata.disk_limit_bytes()?;
        Ok(Self {
            root,
            disk,
            #[cfg(feature = "cpu")]
            metadata,
            snapshot,
            weights,
            config,
            tokenizer,
        })
    }

    pub(crate) fn digest(&self) -> &str {
        &self.snapshot.digest
    }

    /// Read-only routing metadata. The actual prepared owner must authenticate
    /// and validate the selected snapshot again through its held descriptors.
    #[cfg(feature = "cpu")]
    pub(crate) fn inspect(&self) -> Result<(ModelGeometry, crate::artifact::ArtifactInfo)> {
        let geometry =
            ModelGeometry::from_json(&self.metadata.config(&self.snapshot, self.config.path())?)?;
        let info = self
            .metadata
            .inspect_weights(&self.snapshot, self.weights.path())?;
        Ok((geometry, info))
    }

    pub(crate) fn open(&self) -> Result<crate::cache::SnapshotLease> {
        let store = crate::cache::CacheStore::open_existing(crate::cache::CacheOptions::new(
            Some(self.root.clone()),
            self.disk,
        ))?;
        store.open_snapshot(self.digest())
    }
}
