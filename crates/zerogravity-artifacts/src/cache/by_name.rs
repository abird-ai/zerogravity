//! Rebuildable human-facing names for immutable cache snapshots.
//!
//! The by-name directory is never a source of cache authority. Canonical
//! snapshots, manifests and refs are the only inputs. Every published entry is
//! a flat lowercase relative symlink directly to ../snapshots/<digest>.

use super::*;
use rustix::fs::{self, AtFlags, FileType, RenameFlags};
use std::collections::{BTreeMap, BTreeSet};

const MAX_BY_NAME_ENTRIES: usize = 4096;
const MAX_ALIAS_BYTES: usize = 240;
const AUTO_REPOSITORY_BYTES: usize = 176;
const AUTO_REVISION_BYTES: usize = 24;

#[derive(Clone, Debug)]
struct Candidate {
    base: String,
    target: String,
    key: String,
    custom: bool,
}

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
    if out.is_empty() { "alias".into() } else { out }
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
        other => slug(other, 32),
    }
}

fn revision_slug(revision: &str) -> String {
    if revision.len() >= 8 && revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        revision[..8].to_ascii_lowercase()
    } else {
        slug(revision, AUTO_REVISION_BYTES)
    }
}

fn auto_name(source: &ManifestSource) -> String {
    let provider = provider_slug(source.provider());
    let repository = slug(source.repository(), AUTO_REPOSITORY_BYTES);
    let revision = revision_slug(source.revision());
    let reserved = provider
        .len()
        .saturating_add(revision.len())
        .saturating_add(2);
    let repository_limit = MAX_ALIAS_BYTES.saturating_sub(reserved).max(1);
    let repository = slug(&repository, repository_limit);
    slug(
        &format!("{repository}-{provider}-{revision}"),
        MAX_ALIAS_BYTES,
    )
}

fn custom_name(reference: &str) -> String {
    slug(reference, MAX_ALIAS_BYTES)
}

fn default_reference(source: &ManifestSource) -> String {
    format!(
        "{}/{}/{}",
        source.provider(),
        source.repository(),
        source.revision()
    )
}

fn target(digest: &str) -> Result<String> {
    validate_digest(digest)?;
    Ok(format!("../snapshots/{digest}"))
}

fn suffix_key(candidate: &Candidate) -> String {
    let material = format!("{}|{}", candidate.key, candidate.target);
    digest(material.as_bytes())
}

fn with_suffix(base: &str, suffix: &str) -> String {
    let keep = MAX_ALIAS_BYTES
        .saturating_sub(suffix.len())
        .saturating_sub(1);
    let base = slug(base, keep.max(1));
    format!("{base}-{suffix}")
}

fn allocate(mut candidates: Vec<Candidate>) -> Result<BTreeMap<String, String>> {
    if candidates.len() > MAX_BY_NAME_ENTRIES {
        return Err(Error::Limit(
            "derived by-name entry count exceeds bound".into(),
        ));
    }
    candidates.sort_unstable_by(|a, b| {
        a.base
            .cmp(&b.base)
            .then_with(|| b.custom.cmp(&a.custom))
            .then_with(|| a.key.cmp(&b.key))
            .then_with(|| a.target.cmp(&b.target))
    });
    let mut result = BTreeMap::new();
    for candidate in candidates {
        if result
            .get(&candidate.base)
            .is_some_and(|existing| existing == &candidate.target)
        {
            continue;
        }
        if !result.contains_key(&candidate.base) {
            result.insert(candidate.base.clone(), candidate.target.clone());
            continue;
        }
        let hash = suffix_key(&candidate);
        let mut chosen = None;
        for width in [8usize, 12, 16, 24, 32, 48, 64] {
            let name = with_suffix(&candidate.base, &hash[..width]);
            match result.get(&name) {
                None => {
                    chosen = Some(name);
                    break;
                }
                Some(existing) if existing == &candidate.target => {
                    chosen = Some(name);
                    break;
                }
                Some(_) => {}
            }
        }
        let name = chosen
            .ok_or_else(|| Error::State("unable to disambiguate derived by-name entry".into()))?;
        result.insert(name, candidate.target);
    }
    Ok(result)
}

fn scan_existing(directory: &File) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for entry in fs::Dir::read_from(directory).map_err(io)? {
        let entry = entry.map_err(io)?;
        let name = entry
            .file_name()
            .to_str()
            .map_err(|_| Error::Malformed("non-UTF8 by-name entry".into()))?;
        if matches!(name, "." | "..") {
            continue;
        }
        if result.len() == MAX_BY_NAME_ENTRIES {
            return Err(Error::Limit(
                "derived by-name directory entry bound exceeded".into(),
            ));
        }
        let stat = fs::statat(directory, name, AtFlags::SYMLINK_NOFOLLOW).map_err(io)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::Symlink {
            return Err(Error::State(
                "by-name contains a non-symlink foreign entry".into(),
            ));
        }
        let link = fs::readlinkat(directory, name, Vec::new()).map_err(io)?;
        let bytes = link.to_bytes();
        let value = std::str::from_utf8(bytes)
            .map_err(|_| Error::Malformed("non-UTF8 by-name target".into()))?;
        result.insert(name.to_owned(), value.to_owned());
    }
    Ok(result)
}

fn temp_name(name: &str, target: &str) -> String {
    let token = digest(format!("{name}|{target}").as_bytes());
    format!(".tmp-{}", &token[..24])
}

fn install(
    directory: &File,
    existing: &mut BTreeMap<String, String>,
    name: &str,
    target: &str,
) -> Result<()> {
    if existing.get(name).is_some_and(|value| value == target) {
        return Ok(());
    }
    let temporary = temp_name(name, target);
    match fs::statat(directory, temporary.as_str(), AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => {
            if FileType::from_raw_mode(stat.st_mode) != FileType::Symlink {
                return Err(Error::State(
                    "by-name temporary path is not a symlink".into(),
                ));
            }
            fs::unlinkat(directory, temporary.as_str(), AtFlags::empty()).map_err(io)?;
        }
        Err(rustix::io::Errno::NOENT) => {}
        Err(error) => return Err(io(error)),
    }
    fs::symlinkat(target, directory, temporary.as_str()).map_err(io)?;
    if existing.contains_key(name) {
        let stat = fs::statat(directory, name, AtFlags::SYMLINK_NOFOLLOW).map_err(io)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::Symlink {
            return Err(Error::State(
                "by-name destination changed to a non-symlink".into(),
            ));
        }
        fs::unlinkat(directory, name, AtFlags::empty()).map_err(io)?;
    }
    fs::renameat_with(
        directory,
        temporary.as_str(),
        directory,
        name,
        RenameFlags::NOREPLACE,
    )
    .map_err(io)?;
    existing.insert(name.to_owned(), target.to_owned());
    Ok(())
}

impl CacheStore {
    fn desired_by_name_locked(&self) -> Result<BTreeMap<String, String>> {
        let nodes = self.inner.root.walk_canonical_tree(MAX_NODES)?;
        let mut manifests = BTreeMap::new();
        let mut candidates = Vec::new();

        for node in &nodes {
            let Some(snapshot) = node.path.strip_prefix("snapshots/") else {
                continue;
            };
            if !node.is_directory || snapshot.contains('/') {
                continue;
            }
            validate_digest(snapshot)?;
            let manifest = Manifest::from_json(
                &self.read(
                    &format!("snapshots/{snapshot}/manifest.json"),
                    self.inner.options.limits.max_manifest_bytes,
                )?,
                &self.inner.options.limits,
            )?;
            if manifest.digest()? != snapshot {
                return Err(Error::Malformed(
                    "by-name snapshot manifest identity mismatch".into(),
                ));
            }
            candidates.push(Candidate {
                base: auto_name(manifest.source()),
                target: target(snapshot)?,
                key: format!(
                    "auto:{}:{}:{}:{}",
                    manifest.source().provider(),
                    manifest.source().repository(),
                    manifest.source().revision(),
                    snapshot
                ),
                custom: false,
            });
            manifests.insert(snapshot.to_owned(), manifest);
        }

        for node in &nodes {
            let Some(leaf) = node.path.strip_prefix("refs/") else {
                continue;
            };
            if node.is_directory || leaf.contains('/') || !leaf.ends_with(".json") {
                return Err(Error::Malformed("foreign cache reference path".into()));
            }
            let bytes = self.read(&node.path, CONTROL_LIMIT * 2)?;
            let reference: Reference = serde_json::from_slice(&bytes)
                .map_err(|_| Error::Malformed("invalid by-name reference".into()))?;
            if reference.schema != "zerogravity.cache.reference.v1"
                || reference_key(&reference.name)? != node.path
            {
                return Err(Error::Malformed(
                    "by-name reference identity mismatch".into(),
                ));
            }
            let manifest = manifests.get(&reference.snapshot).ok_or_else(|| {
                Error::State("by-name reference points to a missing canonical snapshot".into())
            })?;
            if reference.name == default_reference(manifest.source()) {
                continue;
            }
            candidates.push(Candidate {
                base: custom_name(&reference.name),
                target: target(&reference.snapshot)?,
                key: format!("custom:{}", reference.name),
                custom: true,
            });
        }

        allocate(candidates)
    }

    pub(crate) fn refresh_by_name_locked(&self) -> Result<BTreeMap<String, String>> {
        let desired = self.desired_by_name_locked()?;
        let directory = self.inner.root.open_dir("by-name", true)?;
        let mut existing = scan_existing(&directory)?;

        for (name, target) in &desired {
            install(&directory, &mut existing, name, target)?;
        }

        let desired_names = desired.keys().cloned().collect::<BTreeSet<_>>();
        for name in existing.keys().cloned().collect::<Vec<_>>() {
            if !desired_names.contains(&name) {
                fs::unlinkat(&directory, name.as_str(), AtFlags::empty()).map_err(io)?;
            }
        }
        directory.sync_all()?;
        self.sync_dir("")?;
        Ok(desired)
    }

    /// Rebuild the flat lowercase human-only by-name projection from canonical
    /// manifests and machine refs. Runtime/model selection never consults it.
    pub fn refresh_by_name(&self) -> Result<Vec<String>> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        Ok(self.refresh_by_name_locked()?.into_keys().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(provider: &str, repository: &str, revision: &str) -> ManifestSource {
        ManifestSource::new(provider, repository, revision).unwrap()
    }

    #[test]
    fn automatic_names_are_flat_lowercase_and_readable() {
        assert_eq!(
            auto_name(&source(
                "huggingface",
                "Qwen/Qwen3.5-0.8B",
                "2FC06364715B967F1860AEA9CF38778875588B17",
            )),
            "qwen-qwen3.5-0.8b-hf-2fc06364"
        );
        assert_eq!(
            auto_name(&source("github", "Org/My_Model", "Release/V1")),
            "org-my-model-gh-release-v1"
        );
    }

    #[test]
    fn custom_names_are_lowercase_flat_and_bounded() {
        assert_eq!(
            custom_name("My Team/Primary_Model"),
            "my-team-primary-model"
        );
        let value = custom_name(&"A".repeat(1000));
        assert!(value.len() <= MAX_ALIAS_BYTES);
        assert_eq!(value, value.to_ascii_lowercase());
        assert!(!value.contains('/'));
    }

    #[test]
    fn collisions_get_deterministic_lowercase_suffixes() {
        let candidates = vec![
            Candidate {
                base: "model".into(),
                target:
                    "../snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .into(),
                key: "auto-a".into(),
                custom: false,
            },
            Candidate {
                base: "model".into(),
                target:
                    "../snapshots/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                        .into(),
                key: "auto-b".into(),
                custom: false,
            },
        ];
        let aliases = allocate(candidates).unwrap();
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains_key("model"));
        assert!(
            aliases
                .keys()
                .any(|name| name.starts_with("model-") && name != "model")
        );
        assert!(
            aliases
                .keys()
                .all(|name| name == &name.to_ascii_lowercase())
        );
    }

    #[test]
    fn custom_alias_wins_unsuffixed_collision() {
        let target_a =
            "../snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let target_b =
            "../snapshots/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let aliases = allocate(vec![
            Candidate {
                base: "friendly".into(),
                target: target_a.into(),
                key: "auto".into(),
                custom: false,
            },
            Candidate {
                base: "friendly".into(),
                target: target_b.into(),
                key: "custom".into(),
                custom: true,
            },
        ])
        .unwrap();
        assert_eq!(aliases.get("friendly").map(String::as_str), Some(target_b));
    }
}
