//! Selection metadata has no SourceIdentity or execution eligibility. Runtime
//! reopens exact immutable snapshots through the existing verified bridge.
use super::*;
use std::io::Read;
use zerogravity::artifact::Artifact;
#[cfg(all(feature = "cache", target_os = "linux"))]
use zerogravity::artifact::ArtifactInfo;
use zerogravity::model::ModelGeometry;
pub(super) enum Resolved {
    Local {
        weights: Artifact,
        config: Vec<u8>,
        tokenizer: std::fs::File,
        tokenizer_bytes: u64,
    },
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Cached {
        root: PathBuf,
        disk: u64,
        snapshot: String,
        weights: String,
        config: String,
        tokenizer: String,
        layout: ArtifactInfo,
    },
}
pub(super) struct PreparedSelection {
    pub resolved: Resolved,
    pub geometry: ModelGeometry,
    pub planning_bytes: u64,
    pub model_target: String,
    pub source_label: String,
    pub raw_ticket_peak: u64,
}
fn bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let f = std::fs::File::open(path)?;
    let m = f.metadata()?;
    if !m.is_file() || m.len() > limit {
        return Err(Error::Limit("source routing metadata size".into()));
    }
    let mut bytes = Vec::with_capacity(m.len() as usize);
    f.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != m.len() {
        return Err(Error::State("source metadata changed while reading".into()));
    }
    Ok(bytes)
}
pub(super) fn resolve(
    model: &zerogravity::pipeline::ModelInstance,
    context: usize,
) -> Result<PreparedSelection> {
    let selector = SourceSelector::parse(
        model
            .source
            .as_deref()
            .ok_or_else(|| bad("selected model requires an explicit acquired source"))?,
    )?;
    match &selector {
        SourceSelector::HuggingFace(value) | SourceSelector::GitHub(value) => {
            #[cfg(all(feature = "cache", target_os = "linux"))]
            {
                let provider = if matches!(selector, SourceSelector::HuggingFace(_)) {
                    "huggingface"
                } else {
                    "github"
                };
                zerogravity::cache::provenance::ProviderLocator::parse(provider, value)?;
            }
            #[cfg(not(all(feature = "cache", target_os = "linux")))]
            {
                let _ = value;
                return Err(Error::Unsupported("provider cache selection requires Linux and cache; no acquisition is performed".into()));
            }
        }
        SourceSelector::Http(url) => {
            #[cfg(feature = "acquisition")]
            zerogravity::acquisition::object::HttpsObjectSpec::validate_cached_locator(url)?;
            #[cfg(not(feature = "acquisition"))]
            {
                let _ = url;
                return Err(Error::Unsupported("HTTP source form requires compiled acquisition contract parser; no request is performed".into()));
            }
        }
        _ => {}
    }
    match selector {
        SourceSelector::Local(weights) => {
            if model.snapshot.is_some() {
                return Err(bad("--snapshot is for cached provider/catalog sources"));
            }
            let parent = weights.parent().unwrap_or(Path::new("."));
            let config = model
                .model_config
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| parent.join("config.json"));
            let tokenizer = model
                .tokenizer
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| parent.join("tokenizer.json"));
            let config = bounded(&config, 1024 * 1024)?;
            let geometry = ModelGeometry::from_json(&config)?;
            let model_target = zerogravity::prepare::PreparedQwen::routing_target(&geometry)?;
            let weights = Artifact::open(&weights)?;
            let tokenizer = std::fs::File::open(tokenizer)?;
            let stamp = tokenizer.metadata()?;
            let tokenizer_bytes = stamp.len();
            if !stamp.is_file() || tokenizer_bytes > 64 * 1024 * 1024 {
                return Err(Error::Limit("regular tokenizer64MiB required".into()));
            }
            let planning_bytes = zerogravity::prepare::PreparedQwen::routing_bytes(
                &weights.info,
                &geometry,
                tokenizer_bytes,
                context,
            )?;
            Ok(PreparedSelection {
                resolved: Resolved::Local {
                    weights,
                    config,
                    tokenizer,
                    tokenizer_bytes,
                },
                geometry,
                planning_bytes,
                model_target,
                source_label:
                    "local original GGUF; same inspected descriptor retained through preparation"
                        .into(),
                raw_ticket_peak: 0,
            })
        }
        other => cached(model, other, context),
    }
}
#[cfg(not(all(feature = "cache", target_os = "linux")))]
fn cached(
    _: &zerogravity::pipeline::ModelInstance,
    _: SourceSelector,
    _: usize,
) -> Result<PreparedSelection> {
    Err(Error::Unsupported("cached provider/catalog run requires Linux and the cache feature; acquire an exact source explicitly using models/pull".into()))
}
#[cfg(all(feature = "cache", target_os = "linux"))]
fn cached(
    model: &zerogravity::pipeline::ModelInstance,
    selector: SourceSelector,
    context: usize,
) -> Result<PreparedSelection> {
    use zerogravity::cache::{CacheMetadata, FileRole};
    if model.model_config.is_some() || model.tokenizer.is_some() {
        return Err(bad(
            "cached run requires role-bound config/tokenizer in the selected immutable snapshot; local sidecar overrides would break provenance",
        ));
    }
    let root = zerogravity::cache::paths::resolve_root(
        None,
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?;
    let metadata = CacheMetadata::open(&root)?;
    let snapshot = metadata.select(&selector, model.snapshot.as_deref())?;
    let unique = |role| -> Result<String> {
        let entries = snapshot
            .manifest
            .entries()
            .iter()
            .filter(|e| e.role() == role)
            .collect::<Vec<_>>();
        if entries.len() != 1 {
            return Err(bad(
                "selected snapshot needs exactly one weights/config/tokenizer role; select a complete unambiguous acquired snapshot",
            ));
        }
        Ok(entries[0].path().into())
    };
    let weights = unique(FileRole::Weights)?;
    let config = unique(FileRole::Config)?;
    let tokenizer = unique(FileRole::Tokenizer)?;
    let geometry = ModelGeometry::from_json(&metadata.config(&snapshot, &config)?)?;
    let model_target = zerogravity::prepare::PreparedQwen::routing_target(&geometry)?;
    let info = metadata.inspect_weights(&snapshot, &weights)?;
    let tokenizer_bytes = snapshot
        .manifest
        .entries()
        .iter()
        .find(|e| e.path() == tokenizer)
        .expect("unique role")
        .bytes();
    if tokenizer_bytes > 64 * 1024 * 1024 {
        return Err(Error::Limit("tokenizer64MiB".into()));
    }
    let planning_bytes = zerogravity::prepare::PreparedQwen::routing_bytes(
        &info,
        &geometry,
        tokenizer_bytes,
        context,
    )?;
    let raw_ticket_peak = info
        .file_len
        .checked_add(
            snapshot
                .manifest
                .entries()
                .iter()
                .find(|e| e.path() == config)
                .expect("unique config")
                .bytes(),
        )
        .and_then(|n| n.checked_add(tokenizer_bytes))
        .and_then(|n| n.checked_add(64 * 1024))
        .ok_or_else(|| Error::Limit("raw source ticket planning overflow".into()))?;
    let disk = metadata.disk_limit_bytes()?;
    Ok(PreparedSelection {
        resolved: Resolved::Cached {
            root,
            disk,
            snapshot: snapshot.digest.clone(),
            weights,
            config,
            tokenizer,
            layout: info,
        },
        geometry,
        planning_bytes,
        model_target,
        source_label: format!("immutable cached snapshot {}", snapshot.digest),
        raw_ticket_peak,
    })
}
