//! Pure human-name normalization and deterministic collision policy.
//!
//! No filesystem access, cache authority, or runtime model selection lives here.

use super::*;

pub(super) const MAX_ALIAS_BYTES: usize = 240;
pub(super) const MAX_ALIAS_INPUT_BYTES: usize = 4096;
const AUTO_REPOSITORY_BYTES: usize = 176;
const AUTO_REVISION_BYTES: usize = 24;

fn slug(value: &str, maximum: usize) -> String {
    let mut out = String::new();
    let mut separator = false;
    for byte in value.bytes() {
        let mapped = match byte {
            b'A'..=b'Z' => Some((byte + (b'a' - b'A')) as char),
            b'a'..=b'z' | b'0'..=b'9' => Some(byte as char),
            b'.' => Some('.'),
            _ => None,
        };
        match mapped {
            Some(ch) => {
                if separator && !out.is_empty() && !out.ends_with('-') {
                    out.push('-');
                }
                separator = false;
                if out.len() < maximum {
                    out.push(ch);
                }
            }
            None => separator = true,
        }
        if out.len() >= maximum {
            break;
        }
    }
    while out.ends_with(['-', '.']) {
        out.pop();
    }
    while out.starts_with(['-', '.']) {
        out.remove(0);
    }
    out
}

pub(super) fn validate_name(name: &str) -> Result<()> {
    if !zerogravity_core::run_policy::SourceSelector::is_human_name(name) {
        return Err(Error::Malformed(
            "invalid lowercase flat by-name alias".into(),
        ));
    }
    Ok(())
}

pub(super) fn normalize_alias(value: &str) -> Result<String> {
    if value.is_empty()
        || value.len() > MAX_ALIAS_INPUT_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(Error::Malformed("invalid by-name alias".into()));
    }
    let normalized = slug(value, MAX_ALIAS_BYTES);
    if normalized.is_empty() {
        return Err(Error::Malformed(
            "by-name alias needs at least one ASCII letter or digit".into(),
        ));
    }
    validate_name(&normalized)?;
    Ok(normalized)
}

fn provider_slug(provider: &str) -> String {
    match provider {
        "huggingface" => "hf".into(),
        "huggingface-bundle" => "hf-bundle".into(),
        "github" => "gh".into(),
        "github-bundle" => "gh-bundle".into(),
        "modelscope" => "ms".into(),
        "local-hf-cache" => "hf-local".into(),
        "https-object" => "https".into(),
        other => {
            let value = slug(other, 32);
            if value.is_empty() {
                "source".into()
            } else {
                value
            }
        }
    }
}

fn revision_slug(revision: &str) -> String {
    if revision.len() >= 8 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        revision[..8].to_ascii_lowercase()
    } else {
        let value = slug(revision, AUTO_REVISION_BYTES);
        if value.is_empty() {
            "revision".into()
        } else {
            value
        }
    }
}

fn repository_slug(repository: &str, maximum: usize) -> String {
    let (owner, model) = repository.rsplit_once('/').unwrap_or(("", repository));
    let model = slug(model, maximum);
    let owner = slug(owner, maximum);
    let value = match (model.is_empty(), owner.is_empty()) {
        (false, false) => format!("{model}-{owner}"),
        (false, true) => model,
        (true, false) => owner,
        (true, true) => "model".into(),
    };
    slug(&value, maximum)
}

fn repository_subject(source: &ManifestSource, maximum: usize) -> String {
    let repository = if source.provider() == "modelscope" {
        source
            .repository()
            .split_once('/')
            .filter(|(first, _)| first.starts_with("origin-"))
            .map_or(source.repository(), |(_, rest)| rest)
    } else {
        source.repository()
    };
    repository_slug(repository, maximum)
}

fn logical_weight_subject(manifest: &Manifest, maximum: usize) -> Option<String> {
    let path = manifest
        .entries()
        .iter()
        .find(|entry| entry.role() == FileRole::Weights)?
        .path();
    let base = path.rsplit('/').next().unwrap_or(path);
    let mut stem = base;
    for marker in [".safetensors", ".gguf", ".onnx", ".pth", ".pt", ".bin"] {
        if let Some((prefix, _)) = stem.split_once(marker)
            && !prefix.is_empty()
        {
            stem = prefix;
            break;
        }
    }
    let value = slug(stem, maximum);
    (!value.is_empty()).then_some(value)
}

pub(super) fn auto_name(manifest: &Manifest) -> String {
    let source = manifest.source();
    let provider = provider_slug(source.provider());
    let revision = revision_slug(source.revision());
    let reserved = provider
        .len()
        .saturating_add(revision.len())
        .saturating_add(2);
    let subject_limit = MAX_ALIAS_BYTES
        .saturating_sub(reserved)
        .clamp(1, AUTO_REPOSITORY_BYTES);
    let repository_backed = matches!(
        source.provider(),
        "huggingface"
            | "huggingface-bundle"
            | "github"
            | "github-bundle"
            | "modelscope"
            | "local-hf-cache"
    );
    let subject = if repository_backed {
        repository_subject(source, subject_limit)
    } else {
        logical_weight_subject(manifest, subject_limit)
            .unwrap_or_else(|| repository_subject(source, subject_limit))
    };
    let name = slug(&format!("{subject}-{provider}-{revision}"), MAX_ALIAS_BYTES);
    debug_assert!(validate_name(&name).is_ok());
    name
}

pub(super) fn target(snapshot: &str) -> Result<String> {
    validate_digest(snapshot)?;
    Ok(format!("../snapshots/{snapshot}"))
}

fn auto_collision_key(manifest: &Manifest, snapshot: &str) -> String {
    let source = manifest.source();
    let material = format!(
        "{}|{}|{}|{}",
        source.provider(),
        source.repository(),
        source.revision(),
        snapshot
    );
    digest(material.as_bytes())
}

fn with_suffix(base: &str, suffix: &str) -> String {
    let keep = MAX_ALIAS_BYTES
        .saturating_sub(suffix.len())
        .saturating_sub(1);
    let base = slug(base, keep.max(1));
    format!("{base}-{suffix}")
}

pub(super) struct AutoCandidates {
    base: String,
    hash: String,
    index: usize,
}

impl AutoCandidates {
    pub(super) fn new(manifest: &Manifest, snapshot: &str) -> Self {
        Self {
            base: auto_name(manifest),
            hash: auto_collision_key(manifest, snapshot),
            index: 0,
        }
    }
}

impl Iterator for AutoCandidates {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        const WIDTHS: [usize; 7] = [8, 12, 16, 24, 32, 48, 64];
        let index = self.index;
        self.index = self.index.checked_add(1)?;
        if index == 0 {
            return Some(self.base.clone());
        }
        if let Some(width) = WIDTHS.get(index - 1) {
            return Some(with_suffix(&self.base, &self.hash[..*width]));
        }
        let suffix = format!("{}-{}", self.hash, index - WIDTHS.len());
        Some(with_suffix(&self.base, &suffix))
    }
}
