//! Private pure model closure shared by repository adapters. Provider I/O and
//! byte/request budgets stay with the original admitted metadata owner.
use super::{ExternalSidecars, PullLimits};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
/// Finite, optional, same-directory companions for original model pulls.
/// Never infer optional filenames from the model family, parse executable
/// templates, traverse arbitrary subdirectories, or require absent metadata.
pub(super) const OPTIONAL_MODEL_COMPANIONS: &[&str] = &[
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "tokenizer.tiktoken",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
    "video_processor_config.json",
    "feature_extractor_config.json",
    "audio_processor_config.json",
    "LICENSE",
    "LICENSE.md",
    "README.md",
];
fn malformed() -> Error {
    Error::Malformed("invalid repository model selection".into())
}
fn is_weight(path: &str) -> bool {
    path.ends_with(".gguf") || path.ends_with(".safetensors")
}
fn folder(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(parent, _)| parent)
}
fn sibling(path: &str, leaf: &str) -> String {
    let folder = folder(path);
    if folder.is_empty() {
        leaf.to_string()
    } else {
        format!("{folder}/{leaf}")
    }
}
fn split_gguf(path: &str) -> Result<Option<(String, usize)>> {
    Ok(split_gguf_parts(path)?.map(|(prefix, _, total)| (prefix, total)))
}
pub(super) fn split_gguf_parts(path: &str) -> Result<Option<(String, usize, usize)>> {
    let Some(stem) = path.strip_suffix(".gguf") else {
        return Ok(None);
    };
    let Some((part, total)) = stem.rsplit_once("-of-") else {
        return Ok(None);
    };
    let (prefix, ordinal) = part.rsplit_once('-').ok_or_else(malformed)?;
    if total.len() != 5
        || ordinal.len() != 5
        || !total
            .bytes()
            .chain(ordinal.bytes())
            .all(|c| c.is_ascii_digit())
    {
        return Err(malformed());
    }
    let total = total.parse::<usize>().map_err(|_| malformed())?;
    let ordinal = ordinal.parse::<usize>().map_err(|_| malformed())?;
    if total == 0 || ordinal == 0 || ordinal > total {
        return Err(malformed());
    }
    Ok(Some((prefix.to_owned(), ordinal, total)))
}
fn require(
    entries: &BTreeMap<String, bool>,
    selected: &mut BTreeSet<String>,
    path: String,
    exclusions: &super::filters::Exclusions,
) -> Result<()> {
    exclusions.required(&path)?;
    if !entries.get(&path).copied().unwrap_or(false) {
        return Err(Error::Unsupported(
            "required repository shard or sidecar is absent or not regular".into(),
        ));
    }
    selected.insert(path);
    Ok(())
}
pub(super) fn close_model_selection(
    entries: &BTreeMap<String, bool>,
    selected: &mut BTreeSet<String>,
    limits: &PullLimits,
    sidecars: ExternalSidecars,
    read_index: &mut impl FnMut(&str) -> Result<Vec<u8>>,
) -> Result<()> {
    close_model_selection_filtered(
        entries,
        selected,
        limits,
        sidecars,
        &super::filters::Exclusions::default(),
        read_index,
    )
}
pub(super) fn close_model_selection_filtered(
    entries: &BTreeMap<String, bool>,
    selected: &mut BTreeSet<String>,
    limits: &PullLimits,
    sidecars: ExternalSidecars,
    exclusions: &super::filters::Exclusions,
    read_index: &mut impl FnMut(&str) -> Result<Vec<u8>>,
) -> Result<()> {
    let initial = selected.clone();
    let mut indexes = BTreeSet::new();
    for path in &initial {
        if path.ends_with(".safetensors.index.json") {
            indexes.insert(path.clone());
        }
        if path.ends_with(".safetensors") {
            for candidate in entries.keys().filter(|candidate| {
                candidate.ends_with(".safetensors.index.json") && folder(candidate) == folder(path)
            }) {
                indexes.insert(candidate.clone());
            }
        }
        if let Some((prefix, total)) = split_gguf(path)? {
            if total > limits.cache.max_files {
                return Err(Error::Limit("repository GGUF split count".into()));
            }
            for part in 1..=total {
                require(
                    entries,
                    selected,
                    format!("{prefix}-{part:05}-of-{total:05}.gguf"),
                    exclusions,
                )?;
            }
        }
    }
    let mut covered = BTreeMap::<String, usize>::new();
    let mut index_shards = BTreeSet::new();
    for path in indexes {
        // Closure would inspect this adjacent source index: veto before GET,
        // not after reading an excluded body or silently skipping membership.
        exclusions.required(&path)?;
        let bytes = read_index(&path)?;
        let index = crate::sharded::SafetensorsIndex::from_json(&bytes)
            .map_err(|_| Error::Malformed("invalid repository safetensors index".into()))?;
        let source_shards = index.shards();
        if source_shards.len() > limits.cache.max_files {
            return Err(Error::Limit("repository index shard count".into()));
        }
        let shards = source_shards
            .into_iter()
            .map(|shard| sibling(&path, shard))
            .collect::<Vec<_>>();
        let relevant =
            initial.contains(&path) || shards.iter().any(|shard| initial.contains(shard));
        if relevant {
            require(entries, selected, path.clone(), exclusions)?;
            for shard in shards {
                require(entries, selected, shard.clone(), exclusions)?;
                *covered.entry(shard.clone()).or_default() += 1;
                index_shards.insert(shard);
            }
        }
    }
    if covered.values().any(|count| *count > 1) {
        return Err(Error::Unsupported(
            "ambiguous repository shard indexes".into(),
        ));
    }
    let mut families = BTreeSet::new();
    let mut folders = BTreeSet::new();
    let mut split_counts = BTreeMap::new();
    for path in selected.iter() {
        if path.ends_with(".safetensors.index.json") {
            families.insert(path.clone());
        }
        if is_weight(path) || path.ends_with(".safetensors.index.json") {
            folders.insert(folder(path).to_string());
        }
        if is_weight(path) && !index_shards.contains(path) {
            if path.ends_with(".safetensors") && path.contains("-of-") {
                return Err(Error::Unsupported(
                    "repository safetensors shard requires its source index".into(),
                ));
            }
            let family = if let Some((prefix, total)) = split_gguf(path)? {
                if split_counts
                    .insert(prefix.clone(), total)
                    .is_some_and(|previous| previous != total)
                {
                    return Err(malformed());
                }
                prefix
            } else {
                path.clone()
            };
            families.insert(family);
        }
    }
    if families.len() > 1 {
        return Err(Error::Unsupported(
            "ambiguous repository model weight variants".into(),
        ));
    }
    for folder in folders {
        let path = |leaf: &str| {
            if folder.is_empty() {
                leaf.to_owned()
            } else {
                format!("{folder}/{leaf}")
            }
        };
        for leaf in ["config.json", "tokenizer.json"] {
            let candidate = path(leaf);
            if entries.contains_key(&candidate) || !sidecars.allows_missing(leaf) {
                require(entries, selected, candidate, exclusions)?;
            }
        }
        for &leaf in OPTIONAL_MODEL_COMPANIONS {
            let candidate = path(leaf);
            if entries.contains_key(&candidate) && exclusions.allows(&candidate) {
                require(entries, selected, candidate, exclusions)?;
            }
        }
    }
    if selected.len() > limits.cache.max_files {
        return Err(Error::Limit(
            "repository selected closure exceeds file bound".into(),
        ));
    }
    Ok(())
}
