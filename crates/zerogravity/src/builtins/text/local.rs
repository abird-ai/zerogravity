//! Shared local GGUF source selection and retained host admission.
//! Family math and source identity remain with the original model.
use super::*;
use crate::qwen35::LoadedQwenModel;
use std::{collections::BTreeMap, path::PathBuf};
use zerogravity_core::{
    model::{ModelDefinition, ProgramRequest},
    semantic::ForwardPlan,
    source::{SourceTensorRef, TensorSource},
    task::TaskKind,
};

/// Canonical source paths are resolved once for this cold route set. Aliases
/// reuse its exact prepared owner, never another file opened by alias name.
pub(super) struct LocalRoutes(Vec<(String, PathBuf)>);
impl LocalRoutes {
    pub(super) fn resolve(mut declared: Vec<(String, PathBuf)>) -> Result<Self, ServiceError> {
        zerogravity_engine::router::validate_aliases(
            declared.iter().map(|(alias, _)| alias.as_str()),
        )?;
        for (_, path) in &mut declared {
            *path = std::fs::canonicalize(&path).map_err(|_| {
                ServiceError::new(400, "missing_model", "cannot resolve model source")
            })?;
        }
        Ok(Self(declared))
    }

    pub(super) fn prepare<M: ?Sized>(
        self,
        mut prepare: impl FnMut(&str, &Path, usize) -> Result<Arc<M>, ServiceError>,
    ) -> Result<BTreeMap<String, Arc<M>>, ServiceError> {
        let mut models = BTreeMap::new();
        let mut unique = BTreeMap::<PathBuf, Arc<M>>::new();
        for (alias, path) in self.0 {
            let model = match unique.get(&path) {
                Some(model) => model.clone(),
                None => {
                    let model = prepare(&alias, &path, unique.len())?;
                    unique.insert(path, model.clone());
                    model
                }
            };
            models.insert(alias, model);
        }
        Ok(models)
    }
}

/// Keep the host-weight reservation on the actual source owner that reaches
/// GPU completion/quarantine, not just its serving alias or registry.
pub(super) struct AdmittedGguf {
    model: Arc<LoadedQwenModel>,
    _weight_charge: GlobalResourceLease,
}
impl AdmittedGguf {
    pub(super) fn model(&self) -> &Arc<LoadedQwenModel> {
        &self.model
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

pub(super) struct LocalModel {
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
    path: &Path,
    host: &ServingHost,
) -> Result<LocalModel, ServiceError> {
    let parent = path.parent().unwrap_or(Path::new("."));
    // Hold the actual regular-file descriptor across all cold work.
    let file = std::fs::File::open(path)
        .map_err(|_| ServiceError::new(400, "missing_model", "model file unavailable"))?;
    let metadata = file
        .metadata()
        .map_err(|error| ServiceError::inference(error.into()))?;
    if !metadata.is_file() {
        return Err(ServiceError::new(
            400,
            "invalid_artifact",
            "model source is not a regular file",
        ));
    }
    let input_len = metadata.len();
    let tokenizer_bytes = crate::tokenize::APPROVED_TOKENIZER_BYTES;
    let charged = input_len
        .checked_add(tokenizer_bytes.saturating_mul(2))
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
    let tokenizer_bytes = read_approved_tokenizer(&parent.join("tokenizer.json"))?;
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
        .admit(
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
    let config = zerogravity_artifacts::config::read(&parent.join("config.json"), 1024 * 1024)
        .map_err(ServiceError::inference)?;
    let source = materialize_weights(file, path, input_len)?;
    let model = LoadedQwenModel::from_source(source, &config).map_err(ServiceError::inference)?;
    let weights = Arc::new(AdmittedGguf {
        model,
        _weight_charge: weight_charge,
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
    Ok(LocalModel {
        weights,
        tokenizer,
        max_token_bytes,
    })
}

// Metadata inspection and the owned copy use the descriptor whose extent was
// admitted; pathname replacement must not change either source or allocation.
fn materialize_weights(
    file: std::fs::File,
    path: &Path,
    admitted_bytes: u64,
) -> Result<zerogravity_artifacts::source_weights::SourceWeights, ServiceError> {
    let artifact = zerogravity_artifacts::artifact::Artifact::from_file(file, path)
        .map_err(ServiceError::inference)?;
    if artifact.info.file_len != admitted_bytes {
        return Err(ServiceError::new(
            400,
            "model_source_changed",
            "model extent changed after admission",
        ));
    }
    zerogravity_artifacts::source_weights::SourceWeights::from_inspected_artifact(artifact)
        .map_err(ServiceError::inference)
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
        let error = LocalRoutes::resolve(vec![
            ("same".into(), missing),
            ("same".into(), path.clone()),
        ])
        .err()
        .unwrap();
        assert_eq!(error.code, "invalid_model");
        std::fs::write(&path, empty_gguf("qwen35")).unwrap();
        let routes = LocalRoutes::resolve(vec![
            ("model".into(), path),
            ("alias".into(), fixture.0.join("./source.gguf")),
        ])
        .unwrap();
        let mut count = 0;
        let models = routes
            .prepare(|_, _, group| {
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
        let source = materialize_weights(file, &path, original.len() as u64).unwrap();
        assert_eq!(source.architecture(), "qwen35");
        assert_eq!(source.source_len(), original.len());
    }
    #[test]
    fn admitted_source_rejects_growth_before_materialization() {
        let fixture = Fixture::new("grow");
        let path = fixture.path();
        let original = empty_gguf("qwen35");
        std::fs::write(&path, &original).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0; 32])
            .unwrap();
        let error = materialize_weights(file, &path, original.len() as u64).unwrap_err();
        assert_eq!((error.status, error.code), (400, "model_source_changed"));
    }
}
