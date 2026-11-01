// Source loading and its original admissions, before scheduler creation.
use super::*;

pub(super) fn load_models(
    declared: Vec<(String, PathBuf)>,
    host: &ServingHost,
    max_context: usize,
) -> Result<BTreeMap<String, Arc<dyn LogitModel>>, ServiceError> {
    let mut models: BTreeMap<String, Arc<dyn LogitModel>> = BTreeMap::new();
    // Canonical paths reuse the SAME immutable model/host lease across
    // route aliases. A second alias does not republish a source ID.
    let mut unique_sources = BTreeMap::<PathBuf, Arc<Model>>::new();
    for (id, path) in declared {
        if id.is_empty()
            || id.len() > 96
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            || models.contains_key(&id)
        {
            return Err(ServiceError::new(
                400,
                "invalid_model",
                "duplicate or invalid model alias",
            ));
        }
        let canonical = std::fs::canonicalize(&path)
            .map_err(|_| ServiceError::new(400, "missing_model", "cannot resolve model source"))?;
        if let Some(existing) = unique_sources.get(&canonical) {
            models.insert(id, existing.clone());
            continue;
        }
        let path = canonical;
        let parent = path.parent().unwrap_or(Path::new("."));
        let file = std::fs::File::open(&path)
            .map_err(|_| ServiceError::new(400, "missing_model", "model file unavailable"))?;
        let input_len = file
            .metadata()
            .map_err(|error| ServiceError::inference(error.into()))?
            .len();
        let tokenizer_bytes = crate::tokenize::APPROVED_TOKENIZER_BYTES;
        let charged = input_len
            .checked_add(tokenizer_bytes.saturating_mul(2))
            .and_then(|n| n.checked_add(128 * 1024 * 1024))
            .ok_or_else(|| ServiceError::new(400, "model_budget", "model source size overflow"))?;
        // Preflight/admission BEFORE any tokenizer parse, compiled regex,
        // NFC/ByteLevel bootstrap or weight mapping. The approved retained
        // plan needs only the pinned pool capacity and layout constants,
        // so the process-lifetime ByteLevel owner, retained inventory and
        // cold constructor are charged first; the parsed source must then
        // agree with the plan it was admitted under. The immutable weight
        // source is admitted immediately before materializing the held descriptor,
        // after the constructor scratch/copy are released and verified.
        let tokenizer_admission = host.transfer_admission(&id, "tokenizer-source")?;
        let bootstrap =
            crate::tokenize_admission::approve_arbiter(&tokenizer_admission, "bootstrap")?;
        let inventory_charge = bootstrap
                .retain_arbiter(&tokenizer_admission, "retained-inventory")
                .map_err(|error| {
                    let message = error.to_string();
                    match error {
                        crate::Error::Limit(_) => ServiceError::new(
                            503,
                            "host_budget",
                            format!(
                                "stage=retained-inventory owner=model={id} session=retained-inventory: {message}"
                            ),
                        ),
                        other => ServiceError::inference(other),
                    }
                })?;
        // The exact held source copy is priced before allocation; the
        // constructor census and parsed tokenizer are priced from those
        // same bytes. SHA-256 is checked before the parser.
        let tokenizer_copy = host.charge(
            &id,
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
        // The held copy charge covers these exact bytes: release the
        // payload first, then its precharged source-copy owner.
        drop(tokenizer_bytes);
        drop(tokenizer_copy);
        bootstrap.verify(&tokenizer)?;
        // The unchanged immutable weight source is admitted last, directly
        // before the original weight open; no weight bytes are mapped while
        // only metadata/path work or tokenizer construction is in flight.
        let weight_charge = host
            .admit(
                &id,
                "immutable-source",
                AllocationClass::ImmutableWeights,
                charged,
            )
            .map_err(|error| {
                ServiceError::new(
                    error.status,
                    error.code,
                    format!(
                        "stage=immutable-source owner=model={id} session=immutable-source: {}",
                        error.message
                    ),
                )
            })?;
        let config = std::fs::read(parent.join("config.json"))
            .map_err(|error| ServiceError::inference(error.into()))?;
        let source = materialize_weights(file, &path, input_len)?;
        let weights =
            LoadedQwenModel::from_source(source, &config).map_err(ServiceError::inference)?;
        let max_session_bytes = session_budget(&weights, max_context)?;
        let max_token_bytes = tokenizer
            .max_token_bytes()
            .map_err(ServiceError::inference)?;
        let shared = Arc::new(Model {
            weights,
            tokenizer: tokenizer_owner(
                AdmittedTokenizer {
                    tokenizer,
                    _inventory_charge: inventory_charge,
                },
                |owner| &owner.tokenizer,
                host,
                &id,
            )?,
            max_context,
            max_session_bytes,
            max_token_bytes,
            _weight_charge: weight_charge,
        });
        unique_sources.insert(path, Arc::clone(&shared));
        models.insert(id, shared);
    }
    Ok(models)
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
