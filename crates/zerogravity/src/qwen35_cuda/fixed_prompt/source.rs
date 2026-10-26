//! Compiled original source registry; supplied directory labels grant no authority.
use super::*;
use crate::qwen35::source::owned::QwenOriginalWindowText;
use paths::{Directory, digest};
use zerogravity_artifacts::{
    artifact::Artifact,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};

const CONFIG: &str = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204";
const INDEX: &str = "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e";
const SHARD: &str = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";
const TOKENIZER: &str = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42";
pub(super) fn open(
    directory: &Path,
    host: &TransferAdmission,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<OriginalWindowModelSource>> {
    not_cancelled(cancelled)?;
    let dir = Directory::open(directory)?;
    let config = dir.read("config.json", 2907, Some(&digest(CONFIG)), cancelled)?;
    let index = dir.read(
        "model.safetensors.index.json",
        50900,
        Some(&digest(INDEX)),
        cancelled,
    )?;
    let basename = "model.safetensors-00001-of-00001.safetensors";
    let shard = ExpectedSourceShard::new(
        basename.into(),
        Artifact::from_file(dir.file(basename)?, Path::new(basename))?,
        1_746_942_600,
        SHARD.into(),
    )?;
    let plan = SourceWeights::prepare_verified_windows(
        &index,
        INDEX,
        vec![shard],
        WindowOpenLimits {
            max_source_bytes: 2 * 1024 * MIB as u64,
            max_metadata_bytes: (64 * MIB) as u64,
            max_shards: 1,
            max_tensors: 1024,
            scan_chunk_bytes: MIB,
        },
        host,
    )?;
    not_cancelled(cancelled)?;
    let owner = Arc::new(QwenOriginalWindowText::bind(
        plan.open()?,
        &config,
        CONFIG,
        INDEX,
    )?);
    let model = OriginalWindowModelSource::from_owner(owner)?;
    // Same full source/320-role seal used by the engine, never a shape fallback.
    let _bound = model.bind_full_model_source()?;
    Ok(model)
}
pub(super) fn tokenizer(
    directory: &Path,
    host: &TransferAdmission,
    cancelled: Option<&AtomicBool>,
) -> Result<QwenTokenizer> {
    // The process-global ByteLevel statics and the cold constructor are
    // admitted before the 12.8 MiB approved source is parsed.
    let bootstrap = crate::tokenize_admission::approve_arbiter(host, "fixed-prompt-tokenizer")?;
    let dir = Directory::open(directory)?;
    let raw = dir.read(
        "tokenizer.json",
        12_807_982,
        Some(&digest(TOKENIZER)),
        cancelled,
    )?;
    let tokenizer = {
        let _constructor =
            bootstrap.constructor_of_arbiter(host, "fixed-prompt-constructor", &raw)?;
        // The held extent and SHA-256 above are the approved source; this
        // rechecks that exact identity before the parser sees the bytes.
        QwenTokenizer::from_approved_bytes(&raw)?
    };
    bootstrap.verify(&tokenizer)?;
    not_cancelled(cancelled)?;
    Ok(tokenizer)
}
