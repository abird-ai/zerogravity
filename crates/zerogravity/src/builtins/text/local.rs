//! Shared local GGUF source selection and retained host admission.
//! Family math and source identity remain with the original model.
use super::*;
use crate::{
    prepare::source::{GgufSource, LocalGguf, require_geometry},
    qwen35::LoadedQwenModel,
};
use std::{collections::BTreeMap, path::PathBuf};
use zerogravity_core::{
    model::{ModelDefinition, ProgramRequest},
    semantic::ForwardPlan,
    source::{SourceTensorRef, TensorSource},
    task::TaskKind,
};

/// The legacy path requests and selected graph/SDK declarations use one
/// original cold preparation. Explicit execution policy checks run first.
pub(crate) enum SourceDeclarations<'a> {
    Paths(Vec<(String, PathBuf)>),
    Models(&'a [&'a crate::pipeline::ModelInstance], Option<&'a Path>),
}
impl SourceDeclarations<'_> {
    #[cfg(all(feature = "cpu", feature = "cache"))]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Paths(paths) => paths.len(),
            Self::Models(models, _) => models.len(),
        }
    }
    pub(super) fn resolve(self) -> Result<GgufRoutes, ServiceError> {
        match self {
            Self::Paths(paths) => GgufRoutes::resolve(paths),
            Self::Models(models, cache_root) => GgufRoutes::resolve_models(models, cache_root),
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum SourceKey {
    Local(PathBuf),
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Cached(PathBuf, String),
}
fn key(source: &GgufSource) -> SourceKey {
    match source {
        GgufSource::Local(source) => SourceKey::Local(source.weights.clone()),
        #[cfg(all(feature = "cache", target_os = "linux"))]
        GgufSource::Cached(source) => {
            SourceKey::Cached(source.root.clone(), source.digest().to_owned())
        }
    }
}
struct RouteSource {
    alias: String,
    target: Option<String>,
    source: GgufSource,
}
/// Canonical paths or exact immutable cache snapshots share one prepared
/// owner. A cached ref cannot be re-resolved at materialization.
pub(super) struct GgufRoutes(Vec<RouteSource>);
impl GgufRoutes {
    fn canonical(path: &Path) -> Result<PathBuf, ServiceError> {
        std::fs::canonicalize(path)
            .map_err(|_| ServiceError::new(400, "missing_model", "cannot resolve model source"))
    }
    pub(super) fn resolve(declared: Vec<(String, PathBuf)>) -> Result<Self, ServiceError> {
        zerogravity_engine::router::validate_aliases(
            declared.iter().map(|(alias, _)| alias.as_str()),
        )?;
        let mut routes = Vec::with_capacity(declared.len());
        for (alias, path) in declared {
            routes.push(RouteSource {
                alias,
                target: None,
                source: GgufSource::Local(LocalGguf::new(Self::canonical(&path)?, None, None)),
            });
        }
        Ok(Self(routes))
    }
    fn resolve_models(
        models: &[&crate::pipeline::ModelInstance],
        cache_root: Option<&Path>,
    ) -> Result<Self, ServiceError> {
        zerogravity_engine::router::validate_aliases(models.iter().map(|model| model.id.as_str()))?;
        for model in models {
            if model.model_config.is_some()
                || model.tokenizer.is_some()
                || model.native_artifacts.is_some()
            {
                return Err(ServiceError::new(
                    400,
                    "unsupported",
                    "served GGUF requires original role-bound config/tokenizer; sidecar overrides are not qualified",
                ));
            }
            if !matches!(model.target.as_str(), "qwen3.5-0.8b" | "qwen3.5-2b") {
                return Err(ServiceError::new(
                    400,
                    "unsupported",
                    "declared target is not a qualified GGUF family",
                ));
            }
        }
        let mut routes = Vec::with_capacity(models.len());
        let mut groups = BTreeMap::new();
        for model in models {
            let source = GgufSource::resolve(model, cache_root).map_err(ServiceError::inference)?;
            let source = match source {
                GgufSource::Local(source) => GgufSource::Local(LocalGguf::new(
                    Self::canonical(&source.weights)?,
                    None,
                    None,
                )),
                #[cfg(all(feature = "cache", target_os = "linux"))]
                other => other,
            };
            if groups
                .insert(key(&source), model.target.as_str())
                .is_some_and(|prior| prior != model.target)
            {
                return Err(ServiceError::new(
                    400,
                    "invalid_model",
                    "aliases of one source disagree on target",
                ));
            }
            routes.push(RouteSource {
                alias: model.id.clone(),
                target: Some(model.target.clone()),
                source,
            });
        }
        Ok(Self(routes))
    }

    pub(super) fn prepare<M: ?Sized>(
        self,
        mut prepare: impl FnMut(&str, &GgufSource, Option<&str>, usize) -> Result<Arc<M>, ServiceError>,
    ) -> Result<BTreeMap<String, Arc<M>>, ServiceError> {
        let mut models = BTreeMap::new();
        let mut unique = BTreeMap::<SourceKey, Arc<M>>::new();
        for route in self.0 {
            let key = key(&route.source);
            let model = match unique.get(&key) {
                Some(model) => model.clone(),
                None => {
                    let model = prepare(
                        &route.alias,
                        &route.source,
                        route.target.as_deref(),
                        unique.len(),
                    )?;
                    unique.insert(key, model.clone());
                    model
                }
            };
            models.insert(route.alias, model);
        }
        Ok(models)
    }
}

/// Keep the host-weight reservation on the actual source owner that reaches
/// GPU completion/quarantine, not just its serving alias or registry.
pub(super) struct AdmittedGguf {
    model: Arc<LoadedQwenModel>,
    _weight_charge: GlobalResourceLease,
    // Cached SourceWeights owns its original exact raw-byte lease itself.
    cached_raw_bytes: u64,
    cache_snapshot: Option<String>,
}
impl AdmittedGguf {
    pub(super) fn model(&self) -> &Arc<LoadedQwenModel> {
        &self.model
    }
    pub(super) fn cache_snapshot(&self) -> Option<&str> {
        self.cache_snapshot.as_deref()
    }
    pub(super) fn host_reserved_bytes(&self) -> crate::Result<u64> {
        self._weight_charge
            .reserved_bytes()?
            .checked_add(self.cached_raw_bytes)
            .ok_or_else(|| crate::Error::Limit("source reservation extent overflow".into()))
    }
}
pub(super) fn artifact_description(
    cache_snapshot: Option<String>,
) -> zerogravity_engine::diagnostics::ArtifactDescription {
    zerogravity_engine::diagnostics::ArtifactDescription {
        family: "qwen35".into(),
        format: zerogravity_artifacts::artifact::ArtifactFormat::Gguf,
        cache_snapshot,
        revision: None,
    }
}
impl TensorSource for AdmittedGguf {
    fn source_identity(&self) -> crate::identity::SourceIdentity {
        self.model.source_identity()
    }
    fn tensor(&self, name: &str) -> crate::Result<SourceTensorRef<'_>> {
        self.model.tensor(name)
    }
}
impl ModelDefinition for AdmittedGguf {
    fn source(&self) -> crate::identity::SourceIdentity {
        self.model.source()
    }
    fn tasks(&self) -> &[TaskKind] {
        self.model.tasks()
    }
    fn program(&self, request: ProgramRequest) -> crate::Result<ForwardPlan> {
        self.model.program(request)
    }
}

pub(super) struct GgufModel {
    pub(super) weights: Arc<AdmittedGguf>,
    pub(super) tokenizer: Arc<dyn TextTokenizer>,
    pub(super) max_token_bytes: usize,
}
struct AdmittedTokenizer {
    tokenizer: QwenTokenizer,
    _inventory_charge: crate::tokenize_admission::Retained,
}

pub(super) fn load(
    alias: &str,
    source: &GgufSource,
    host: &ServingHost,
    expected_target: Option<&str>,
    context: usize,
) -> Result<GgufModel, ServiceError> {
    // Preserve the held regular descriptor and the original tokenizer-first
    // retained-inventory/constructor admission ordering.
    let held = match source {
        GgufSource::Local(source) => Some(source.open_weights().map_err(|error| match error {
            crate::Error::Io(_) => {
                ServiceError::new(400, "missing_model", "model file unavailable")
            }
            other => ServiceError::inference(other),
        })?),
        #[cfg(all(feature = "cache", target_os = "linux"))]
        GgufSource::Cached(_) => None,
    };
    let (input_len, cached_raw_bytes) = match source {
        GgufSource::Local(_) => (held.as_ref().expect("held local source").bytes, 0),
        #[cfg(all(feature = "cache", target_os = "linux"))]
        GgufSource::Cached(source) => (source.weights.bytes(), source.weights.bytes()),
    };
    let tokenizer_bytes = crate::tokenize::APPROVED_TOKENIZER_BYTES;
    // The cached raw bytes are charged by the original verified source
    // owner, not twice through the wrapper's conservative source envelope.
    let charged = input_len
        .checked_sub(cached_raw_bytes)
        .and_then(|n| n.checked_add(tokenizer_bytes.saturating_mul(2)))
        .and_then(|n| n.checked_add(128 * 1024 * 1024))
        .ok_or_else(|| ServiceError::new(400, "model_budget", "model source size overflow"))?;

    // Original CPU budget ordering: admit the approved tokenizer inventory
    // and constructor before parsing; weight credit before materialization.
    let tokenizer_admission = host.transfer_admission(alias, "tokenizer-source")?;
    let bootstrap = crate::tokenize_admission::approve_arbiter(&tokenizer_admission, "bootstrap")?;
    let inventory_charge = bootstrap.retain_arbiter(&tokenizer_admission, "retained-inventory")
        .map_err(|error| {
            let message = error.to_string();
            match error {
                crate::Error::Limit(_) => ServiceError::new(503, "host_budget",
                    format!("stage=retained-inventory owner=model={alias} session=retained-inventory: {message}")),
                other => ServiceError::inference(other),
            }
        })?;
    let tokenizer_copy = host.charge(
        alias,
        "tokenizer-source-copy",
        AllocationClass::Scratch,
        tokenizer_bytes,
    )?;
    #[cfg(all(feature = "cache", target_os = "linux"))]
    let cached = match source {
        GgufSource::Cached(source) => Some(source.open().map_err(ServiceError::inference)?),
        GgufSource::Local(_) => None,
    };
    let tokenizer_bytes = match source {
        GgufSource::Local(source) => read_approved_tokenizer(&source.tokenizer)?,
        #[cfg(all(feature = "cache", target_os = "linux"))]
        GgufSource::Cached(source) => {
            if source.tokenizer.bytes() != crate::tokenize::APPROVED_TOKENIZER_BYTES {
                return Err(ServiceError::new(
                    400,
                    "tokenizer_inventory",
                    "approved tokenizer extent mismatch",
                ));
            }
            cached
                .as_ref()
                .expect("opened cached source")
                .read_verified(
                    source.tokenizer.path(),
                    crate::tokenize::APPROVED_TOKENIZER_BYTES,
                )
                .map_err(ServiceError::inference)?
        }
    };
    let tokenizer = {
        let _constructor = bootstrap.constructor_of_arbiter(
            &tokenizer_admission,
            "tokenizer-constructor",
            &tokenizer_bytes,
        )?;
        QwenTokenizer::from_approved_bytes(&tokenizer_bytes).map_err(ServiceError::inference)?
    };
    drop(tokenizer_bytes);
    drop(tokenizer_copy);
    bootstrap.verify(&tokenizer)?;

    let weight_charge = host
        .admit_cold(
            alias,
            "immutable-source",
            AllocationClass::ImmutableWeights,
            charged,
        )
        .map_err(|error| {
            ServiceError::new(
                error.status,
                error.code,
                format!(
                    "stage=immutable-source owner=model={alias} session=immutable-source: {}",
                    error.message
                ),
            )
        })?;
    let (config, geometry) = match source {
        GgufSource::Local(source) => source.read_config().map_err(ServiceError::inference)?,
        #[cfg(all(feature = "cache", target_os = "linux"))]
        GgufSource::Cached(source) => {
            let bytes = cached
                .as_ref()
                .expect("opened cached source")
                .read_verified(source.config.path(), 1024 * 1024)
                .map_err(ServiceError::inference)?;
            let geometry = crate::qwen35::config::ModelGeometry::from_json(&bytes)
                .map_err(ServiceError::inference)?;
            (bytes, geometry)
        }
    };
    require_geometry(&geometry, expected_target, context).map_err(|error| match error {
        crate::Error::Malformed(message) if expected_target.is_some() => {
            ServiceError::new(400, "invalid_model", message)
        }
        other => ServiceError::inference(other),
    })?;
    let model = match source {
        GgufSource::Local(_) => {
            let artifact = held
                .expect("held local source")
                .inspect()
                .map_err(ServiceError::inference)?;
            LoadedQwenModel::from_inspected_gguf(artifact, geometry)
                .map_err(ServiceError::inference)?
        }
        #[cfg(all(feature = "cache", target_os = "linux"))]
        GgufSource::Cached(source) => {
            let lease = cached.as_ref().expect("opened cached source");
            let artifact = zerogravity_artifacts::artifact::Artifact::from_file(
                lease
                    .file(source.weights.path())
                    .map_err(ServiceError::inference)?
                    .try_clone()
                    .map_err(|e| ServiceError::inference(e.into()))?,
                Path::new(source.weights.path()),
            )
            .map_err(ServiceError::inference)?;
            LoadedQwenModel::validate_gguf_info(&artifact.info, &geometry)
                .map_err(ServiceError::inference)?;
            let admission = host.transfer_admission(alias, "immutable-cached-source")?;
            admission
                .arbiter
                .check_host_cold_allocation(cached_raw_bytes)
                .map_err(ServiceError::inference)?;
            let weights = zerogravity_artifacts::SourceWeights::from_verified_artifact(
                artifact,
                source.weights.bytes(),
                source.weights.digest(),
                &admission,
            )
            .map_err(ServiceError::inference)?;
            LoadedQwenModel::from_source(weights, &config).map_err(ServiceError::inference)?
        }
    };
    drop(config);
    let cache_snapshot = match source {
        GgufSource::Local(_) => None,
        #[cfg(all(feature = "cache", target_os = "linux"))]
        GgufSource::Cached(source) => Some(source.digest().to_owned()),
    };
    let weights = Arc::new(AdmittedGguf {
        model,
        _weight_charge: weight_charge,
        cached_raw_bytes,
        cache_snapshot,
    });
    let max_token_bytes = tokenizer
        .max_token_bytes()
        .map_err(ServiceError::inference)?;
    let tokenizer = tokenizer_owner(
        AdmittedTokenizer {
            tokenizer,
            _inventory_charge: inventory_charge,
        },
        |owner| &owner.tokenizer,
        host,
        alias,
    )?;
    Ok(GgufModel {
        weights,
        tokenizer,
        max_token_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new(tag: &str) -> Self {
            let root =
                std::env::temp_dir().join(format!("zg-held-source-{}-{tag}", std::process::id()));
            std::fs::create_dir(&root).unwrap();
            Self(root)
        }
        fn path(&self) -> PathBuf {
            self.0.join("source.gguf")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn empty_gguf(architecture: &str) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        let key = b"general.architecture";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key);
        bytes.extend(8u32.to_le_bytes());
        bytes.extend((architecture.len() as u64).to_le_bytes());
        bytes.extend(architecture.as_bytes());
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes
    }
    #[test]
    fn local_routes_validate_all_aliases_before_resolving_and_share_canonical_owner() {
        let fixture = Fixture::new("routes");
        let path = fixture.path();
        let missing = fixture.0.join("missing.gguf");
        let error = GgufRoutes::resolve(vec![
            ("same".into(), missing),
            ("same".into(), path.clone()),
        ])
        .err()
        .unwrap();
        assert_eq!(error.code, "invalid_model");
        std::fs::write(&path, empty_gguf("qwen35")).unwrap();
        let routes = GgufRoutes::resolve(vec![
            ("model".into(), path),
            ("alias".into(), fixture.0.join("./source.gguf")),
        ])
        .unwrap();
        let mut count = 0;
        let models = routes
            .prepare(|_, _, _, group| {
                count += 1;
                assert_eq!(group, 0);
                Ok(Arc::new(()))
            })
            .unwrap();
        assert_eq!(count, 1);
        assert!(Arc::ptr_eq(&models["model"], &models["alias"]));
    }
    #[test]
    fn admitted_source_uses_held_descriptor_after_path_replacement() {
        let fixture = Fixture::new("replace");
        let path = fixture.path();
        let original = empty_gguf("qwen35");
        std::fs::write(&path, &original).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        std::fs::rename(&path, fixture.0.join("original.gguf")).unwrap();
        let mut replacement = empty_gguf("gemma4");
        replacement.resize(replacement.len() + 32, 0);
        std::fs::write(&path, replacement).unwrap();
        let artifact = crate::prepare::source::HeldWeights::new(file, path.clone())
            .unwrap()
            .inspect()
            .unwrap();
        let source =
            zerogravity_artifacts::SourceWeights::from_inspected_artifact(artifact).unwrap();
        assert_eq!(source.architecture(), "qwen35");
        assert_eq!(source.source_len(), original.len());
    }
    #[test]
    fn admitted_source_rejects_growth_before_materialization() {
        let fixture = Fixture::new("grow");
        let path = fixture.path();
        let original = empty_gguf("qwen35");
        std::fs::write(&path, &original).unwrap();
        let held = crate::prepare::source::HeldWeights::new(
            std::fs::File::open(&path).unwrap(),
            path.clone(),
        )
        .unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0; 32])
            .unwrap();
        let error = held.inspect().err().unwrap();
        assert!(matches!(error, crate::Error::State(_)));
    }
}
