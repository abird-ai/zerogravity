//! Bounded GitHub metadata resolution. Git object IDs identify a pinned tree;
//! they are never interpreted as raw file SHA256. This module does no transfer
//! or cache publication and executes no repository code or LFS hooks.
use super::http::{HttpAccept, HttpRequest, HttpTransport};
use super::{Provider, PullLimits, PullRequest, ResolvedFile, ResolvedPullPlan, Selection};
use crate::cache::{FileRole, ManifestSource, paths::validate_relative_path};
use crate::{Error, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;

#[derive(Deserialize)]
struct Commit {
    sha: String,
    commit: CommitDetails,
}
#[derive(Deserialize)]
struct CommitDetails {
    tree: ObjectId,
}
#[derive(Deserialize)]
struct ObjectId {
    sha: String,
}
#[derive(Deserialize)]
struct Tree {
    sha: String,
    truncated: bool,
    tree: Vec<TreeEntry>,
}
#[derive(Deserialize)]
struct TreeEntry {
    path: String,
    mode: String,
    #[serde(rename = "type")]
    kind: String,
    sha: String,
    size: Option<u64>,
    #[serde(skip)]
    validator: Option<String>,
}
#[derive(Deserialize)]
struct Release {
    id: u64,
    tag_name: String,
}
#[derive(Deserialize)]
struct Asset {
    id: u64,
    name: String,
    size: Option<u64>,
    state: String,
    digest: Option<String>,
}

fn malformed() -> Error {
    Error::Malformed("invalid GitHub metadata or selection".into())
}
fn object_id(id: &str) -> Result<()> {
    if id.len() != 40
        || !id
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(malformed());
    }
    Ok(())
}
fn repository(repository: &str) -> Result<()> {
    let parts: Vec<_> = repository.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|p| {
            p.is_empty()
                || p.len() > 100
                || p.starts_with('.')
                || !p
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
        })
    {
        return Err(malformed());
    }
    Ok(())
}
fn component(value: &str) -> String {
    let mut result = String::new();
    for c in value.bytes() {
        if c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b'~') {
            result.push(char::from(c));
        } else {
            use std::fmt::Write;
            write!(&mut result, "%{c:02X}").expect("writing to string");
        }
    }
    result
}
fn logical_path(path: &str, subfolder: Option<&str>) -> Result<String> {
    validate_relative_path(path)?;
    if let Some(subfolder) = subfolder {
        validate_relative_path(subfolder)?;
        let full = format!("{subfolder}/{path}");
        validate_relative_path(&full)?;
        Ok(full)
    } else {
        Ok(path.to_owned())
    }
}
fn role(path: &str) -> FileRole {
    let leaf = path.rsplit('/').next().unwrap_or(path);
    let lower = leaf.to_ascii_lowercase();
    if lower.ends_with(".index.json") {
        FileRole::Index
    } else if lower.ends_with(".gguf") || lower.ends_with(".safetensors") {
        FileRole::Weights
    } else if lower.starts_with("license") || lower.starts_with("copying") {
        FileRole::License
    } else if lower.contains("tokenizer")
        || matches!(
            lower.as_str(),
            "vocab.json" | "merges.txt" | "special_tokens_map.json"
        )
    {
        FileRole::Tokenizer
    } else if lower.ends_with("config.json") {
        FileRole::Config
    } else {
        FileRole::Auxiliary
    }
}
fn raw_url(repository: &str, pin: &str, path: &str) -> String {
    let encoded = path.split('/').map(component).collect::<Vec<_>>().join("/");
    format!("https://raw.githubusercontent.com/{repository}/{pin}/{encoded}")
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
    Ok(Some((prefix.to_owned(), total)))
}
fn require(
    entries: &BTreeMap<String, TreeEntry>,
    selected: &mut BTreeSet<String>,
    path: String,
) -> Result<()> {
    let entry = entries.get(&path).ok_or_else(|| {
        Error::Unsupported("required GitHub shard or sidecar is absent in pinned repository".into())
    })?;
    if entry.kind != "blob" || !matches!(entry.mode.as_str(), "100644" | "100755") {
        return Err(Error::Unsupported(
            "required GitHub file is not a bounded regular blob".into(),
        ));
    }
    selected.insert(path);
    Ok(())
}
fn close_model_selection(
    transport: &(impl HttpTransport + ?Sized),
    source: (&str, &str, super::ExternalSidecars, &mut usize),
    entries: &mut BTreeMap<String, TreeEntry>,
    selected: &mut BTreeSet<String>,
    limits: &PullLimits,
    remaining: &mut usize,
    cancel: &AtomicBool,
) -> Result<()> {
    let (repository, pin, sidecars, probes) = source;
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
                return Err(Error::Limit("GitHub GGUF split count".into()));
            }
            for part in 1..=total {
                require(
                    entries,
                    selected,
                    format!("{prefix}-{part:05}-of-{total:05}.gguf"),
                )?;
            }
        }
    }
    let mut covered = BTreeMap::<String, usize>::new();
    let mut index_shards = BTreeSet::new();
    for path in indexes {
        let entry = entries.get_mut(&path).ok_or_else(malformed)?;
        if entry.kind != "blob" || !matches!(entry.mode.as_str(), "100644" | "100755") {
            return Err(Error::Unsupported(
                "GitHub index is not a regular blob".into(),
            ));
        }
        let size = ensure_length(
            entry,
            transport,
            raw_url(repository, pin, &path),
            limits.cache.max_file_bytes.min(*remaining as u64),
            probes,
            cancel,
            pin,
        )?;
        let request = HttpRequest::new(raw_url(repository, pin, &path), size);
        let response = transport.get(&request, cancel)?;
        if response.status != 200 {
            return Err(Error::Unsupported(
                "GitHub index metadata request failed".into(),
            ));
        }
        super::http::validate_validator(entry.validator.as_deref(), &response)?;
        let bytes = response.read_bounded(size as usize, cancel)?;
        *remaining -= bytes.len();
        if bytes.len() as u64 != size {
            return Err(malformed());
        }
        if bytes.starts_with(b"version https://git-lfs.github.com/spec/v1") {
            return Err(Error::Unsupported(
                "GitHub LFS index requires a separately qualified LFS resolver".into(),
            ));
        }
        let index = crate::sharded::SafetensorsIndex::from_json(&bytes)
            .map_err(|_| Error::Malformed("invalid GitHub safetensors index".into()))?;
        let source_shards = index.shards();
        if source_shards.len() > limits.cache.max_files {
            return Err(Error::Limit("GitHub index shard count".into()));
        }
        let shards = source_shards
            .into_iter()
            .map(|shard| sibling(&path, shard))
            .collect::<Vec<_>>();
        let relevant =
            initial.contains(&path) || shards.iter().any(|shard| initial.contains(shard));
        if relevant {
            require(entries, selected, path.clone())?;
            for shard in shards {
                require(entries, selected, shard.clone())?;
                *covered.entry(shard.clone()).or_default() += 1;
                index_shards.insert(shard);
            }
        }
    }
    if covered.values().any(|count| *count > 1) {
        return Err(Error::Unsupported("ambiguous GitHub shard indexes".into()));
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
                    "GitHub safetensors shard requires its source index".into(),
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
            "ambiguous GitHub model weight variants".into(),
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
                require(entries, selected, candidate)?;
            }
        }
        for leaf in [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "LICENSE",
            "LICENSE.md",
            "README.md",
        ] {
            let candidate = path(leaf);
            if entries.contains_key(&candidate) {
                require(entries, selected, candidate)?;
            }
        }
    }
    if selected.len() > limits.cache.max_files {
        return Err(Error::Limit(
            "GitHub selected closure exceeds file bound".into(),
        ));
    }
    Ok(())
}

fn metadata<T: for<'de> Deserialize<'de>>(
    transport: &(impl HttpTransport + ?Sized),
    url: String,
    remaining: &mut usize,
    cancel: &AtomicBool,
) -> Result<T> {
    if *remaining == 0 {
        return Err(Error::Limit("GitHub metadata budget exhausted".into()));
    }
    let mut request = HttpRequest::new(url, *remaining as u64);
    request.accept = HttpAccept::GitHubJson;
    let response = transport.get(&request, cancel)?;
    if response.status != 200 {
        return Err(Error::Unsupported(
            "GitHub metadata request did not succeed".into(),
        ));
    }
    let bytes = response.read_bounded(*remaining, cancel)?;
    *remaining -= bytes.len();
    serde_json::from_slice(&bytes).map_err(|_| malformed())
}

/// Only the parent admitted front door may invoke provider metadata work.
pub(super) fn resolve(
    transport: &(impl HttpTransport + ?Sized),
    request: &PullRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    resolve_component(
        transport,
        request,
        limits,
        cancel,
        super::ExternalSidecars::default(),
    )
}

pub(super) fn resolve_component(
    transport: &(impl HttpTransport + ?Sized),
    request: &PullRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
    sidecars: super::ExternalSidecars,
) -> Result<ResolvedPullPlan> {
    request.validate(limits)?;
    limits.validate()?;
    if request.provider != Provider::GitHub {
        return Err(malformed());
    }
    repository(&request.repository)?;
    if request.revision.is_empty()
        || request.revision.len() > 256
        || request.revision.chars().any(char::is_control)
    {
        return Err(malformed());
    }
    let api = format!("https://api.github.com/repos/{}", request.repository);
    let mut metadata_left = limits.max_metadata_bytes;
    let mut probes_left = limits.max_pages;
    match &request.selection {
        Selection::Files { paths, subfolder } => {
            if paths.is_empty() || paths.len() > limits.cache.max_files {
                return Err(Error::Limit(
                    "GitHub file selection exceeds configured bounds".into(),
                ));
            }
            let mut selected = BTreeSet::new();
            for path in paths {
                if !selected.insert(logical_path(path, subfolder.as_deref())?) {
                    return Err(malformed());
                }
            }
            let commit: Commit = metadata(
                transport,
                format!("{api}/commits/{}", component(&request.revision)),
                &mut metadata_left,
                cancel,
            )?;
            object_id(&commit.sha)?;
            if request.revision.len() == 40
                && request.revision.bytes().all(|c| c.is_ascii_hexdigit())
                && !request.revision.eq_ignore_ascii_case(&commit.sha)
            {
                return Err(malformed());
            }
            object_id(&commit.commit.tree.sha)?;
            let tree: Tree = metadata(
                transport,
                format!("{api}/git/trees/{}?recursive=1", commit.commit.tree.sha),
                &mut metadata_left,
                cancel,
            )?;
            if tree.truncated {
                return Err(Error::Unsupported(
                    "GitHub tree is truncated; complete selection cannot be proved".into(),
                ));
            }
            if tree.sha != commit.commit.tree.sha {
                return Err(malformed());
            }
            if tree.tree.len() > limits.cache.max_files {
                return Err(Error::Limit(
                    "GitHub tree exceeds configured entry bound".into(),
                ));
            }
            let mut entries = BTreeMap::new();
            for entry in tree.tree {
                validate_relative_path(&entry.path)?;
                object_id(&entry.sha)?;
                if entries.insert(entry.path.clone(), entry).is_some() {
                    return Err(malformed());
                }
            }
            close_model_selection(
                transport,
                (&request.repository, &commit.sha, sidecars, &mut probes_left),
                &mut entries,
                &mut selected,
                limits,
                &mut metadata_left,
                cancel,
            )?;
            let mut files = Vec::with_capacity(selected.len());
            for path in selected {
                let entry = entries
                    .get_mut(&path)
                    .ok_or_else(|| Error::Unsupported("required GitHub path is absent".into()))?;
                if entry.kind != "blob" || !matches!(entry.mode.as_str(), "100644" | "100755") {
                    return Err(Error::Unsupported(
                        "GitHub symlinks, submodules and directories are not downloadable files"
                            .into(),
                    ));
                }
                let bytes = ensure_length(
                    entry,
                    transport,
                    raw_url(&request.repository, &commit.sha, &path),
                    limits.cache.max_file_bytes,
                    &mut probes_left,
                    cancel,
                    &commit.sha,
                )?;
                let encoded_path = path.split('/').map(component).collect::<Vec<_>>().join("/");
                files.push(
                    ResolvedFile::new(
                        &path,
                        format!(
                            "https://raw.githubusercontent.com/{}/{}/{encoded_path}",
                            request.repository, commit.sha
                        ),
                        role(&path),
                        bytes,
                        None,
                        format!("git-commit:{}:blob:{}", commit.sha, entry.sha),
                    )?
                    .with_validator(entry.validator.clone())?,
                );
            }
            ResolvedPullPlan::new(
                ManifestSource::new("github", &request.repository, commit.sha)?,
                files,
                limits,
            )
        }
        Selection::ReleaseAsset { name } => {
            validate_relative_path(name)?;
            if name.contains('/') {
                return Err(malformed());
            }
            let release: Release = metadata(
                transport,
                format!("{api}/releases/tags/{}", component(&request.revision)),
                &mut metadata_left,
                cancel,
            )?;
            if release.id == 0 || release.tag_name != request.revision {
                return Err(malformed());
            }
            let mut ids = BTreeSet::new();
            let mut names = BTreeSet::new();
            let mut selected = None;
            let mut complete = false;
            for page in 1..=limits.max_pages {
                let assets: Vec<Asset> = metadata(
                    transport,
                    format!(
                        "{api}/releases/{}/assets?per_page=100&page={page}",
                        release.id
                    ),
                    &mut metadata_left,
                    cancel,
                )?;
                if assets.len() > 100
                    || ids.len().saturating_add(assets.len()) > limits.cache.max_files
                {
                    return Err(Error::Limit("GitHub release exceeds asset bound".into()));
                }
                let last = assets.len() < 100;
                for asset in assets {
                    if asset.id == 0 || !ids.insert(asset.id) || !names.insert(asset.name.clone()) {
                        return Err(malformed());
                    }
                    validate_relative_path(&asset.name)?;
                    if asset.name == *name {
                        selected = Some(asset);
                    }
                }
                if last {
                    complete = true;
                    break;
                }
            }
            if !complete {
                return Err(Error::Limit(
                    "GitHub release pagination exceeds configured bound".into(),
                ));
            }
            let asset = selected.ok_or_else(|| {
                Error::Unsupported("required GitHub release asset is absent".into())
            })?;
            if asset.state != "uploaded" {
                return Err(Error::Unsupported(
                    "GitHub release asset is not uploaded".into(),
                ));
            }
            let digest = match asset.digest {
                None => None,
                Some(value) => {
                    let digest = value.strip_prefix("sha256:").ok_or_else(|| {
                        Error::Unsupported("GitHub asset checksum is not SHA256".into())
                    })?;
                    crate::cache::paths::validate_digest(digest)?;
                    Some(digest.to_owned())
                }
            };
            let asset_url = format!("{api}/releases/assets/{}", asset.id);
            let (bytes, validator) = match asset.size {
                Some(bytes) => (bytes, None),
                None => super::http::probe_length(
                    transport,
                    asset_url.clone(),
                    limits.cache.max_file_bytes,
                    &mut probes_left,
                    cancel,
                    None,
                )?,
            };
            let file = ResolvedFile::new(
                name,
                asset_url,
                role(name),
                bytes,
                digest,
                format!("release:{}:asset:{}", release.id, asset.id),
            )?
            .with_validator(validator)?;
            ResolvedPullPlan::new(
                ManifestSource::new(
                    "github",
                    &request.repository,
                    format!("release-{}-asset-{}", release.id, asset.id),
                )?,
                vec![file],
                limits,
            )
        }
    }
}

fn ensure_length(
    entry: &mut TreeEntry,
    transport: &(impl HttpTransport + ?Sized),
    url: String,
    maximum: u64,
    probes: &mut usize,
    cancel: &AtomicBool,
    pin: &str,
) -> Result<u64> {
    let size = match entry.size {
        Some(size) => size,
        None => {
            let (size, validator) =
                super::http::probe_length(transport, url, maximum, probes, cancel, Some(pin))?;
            entry.size = Some(size);
            entry.validator = validator;
            size
        }
    };
    if size > maximum {
        return Err(Error::Limit("GitHub selected file length limit".into()));
    }
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::super::http::HttpResponse;
    use super::*;
    use serde_json::{Value, json};
    use std::collections::VecDeque;
    use std::io::Cursor;
    use std::sync::Mutex;

    struct Fixture(Mutex<VecDeque<(String, Value)>>);
    impl Fixture {
        fn new(rows: Vec<(String, Value)>) -> Self {
            Self(Mutex::new(rows.into()))
        }
        fn finished(&self) {
            assert!(self.0.lock().unwrap().is_empty());
        }
    }
    impl HttpTransport for Fixture {
        fn get(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            let (url, value) = self
                .0
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected provider request");
            assert_eq!(request.url, url);
            let raw = request
                .url
                .starts_with("https://raw.githubusercontent.com/");
            assert!(matches!(
                (raw, request.accept),
                (true, HttpAccept::Binary) | (false, HttpAccept::GitHubJson)
            ));
            assert!(request.range.is_none());
            Ok(HttpResponse {
                status: 200,
                headers: BTreeMap::new(),
                body: Box::new(Cursor::new(if raw && value.is_string() {
                    value.as_str().unwrap().as_bytes().to_vec()
                } else {
                    serde_json::to_vec(&value).unwrap()
                })),
            })
        }
    }
    const COMMIT: &str = "1111111111111111111111111111111111111111";
    const TREE: &str = "2222222222222222222222222222222222222222";
    const BLOB: &str = "3333333333333333333333333333333333333333";
    fn request() -> PullRequest {
        PullRequest {
            provider: Provider::GitHub,
            repository: "owner/repo".into(),
            revision: "branch/name".into(),
            selection: Selection::Files {
                paths: vec!["tiny.gguf".into()],
                subfolder: None,
            },
        }
    }
    fn entries(mode: &str) -> Value {
        json!({"sha":TREE,"truncated":false,"tree":[{"path":"tiny.gguf","mode":mode,"type":"blob","sha":BLOB,"size":7},{"path":"config.json","mode":"100644","type":"blob","sha":BLOB,"size":2},{"path":"tokenizer.json","mode":"100644","type":"blob","sha":BLOB,"size":2}]})
    }
    fn repo(tree: Value) -> Fixture {
        Fixture::new(vec![
            (
                "https://api.github.com/repos/owner/repo/commits/branch%2Fname".into(),
                json!({"sha":COMMIT,"commit":{"tree":{"sha":TREE}}}),
            ),
            (
                format!("https://api.github.com/repos/owner/repo/git/trees/{TREE}?recursive=1"),
                tree,
            ),
        ])
    }
    #[test]
    fn github_branch_resolves_once_and_git_oid_is_not_content_sha256() {
        let fixture = repo(entries("100644"));
        let plan = resolve(
            &fixture,
            &request(),
            &PullLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap();
        fixture.finished();
        assert_eq!(plan.source().revision(), COMMIT);
        assert_eq!(plan.total_bytes(), 11);
        let weight = plan
            .files()
            .iter()
            .find(|file| file.path() == "tiny.gguf")
            .unwrap();
        assert_eq!(weight.sha256(), None);
        assert_eq!(
            weight.identity(),
            format!("git-commit:{COMMIT}:blob:{BLOB}")
        );
        assert_eq!(
            weight.url(),
            format!("https://raw.githubusercontent.com/owner/repo/{COMMIT}/tiny.gguf")
        );
    }
    #[test]
    fn github_tree_truncation_wrong_identity_and_unsafe_entries_refuse() {
        for (field, value) in [
            ("truncated", json!(true)),
            ("sha", json!(COMMIT)),
            (
                "tree",
                json!([{"path":"../tiny.gguf","mode":"100644","type":"blob","sha":BLOB,"size":7}]),
            ),
        ] {
            let mut tree = entries("100644");
            tree[field] = value;
            let fixture = repo(tree);
            assert!(
                resolve(
                    &fixture,
                    &request(),
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
            fixture.finished();
        }
        for mode in ["120000", "160000"] {
            let fixture = repo(entries(mode));
            assert!(matches!(
                resolve(
                    &fixture,
                    &request(),
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                ),
                Err(Error::Unsupported(_))
            ));
            fixture.finished();
        }
    }
    #[test]
    fn github_missing_duplicate_and_unknown_length_files_refuse() {
        let mut trees = vec![];
        let mut absent = entries("100644");
        absent["tree"] = json!([]);
        trees.push(absent);
        let mut duplicate = entries("100644");
        let twice = duplicate["tree"][0].clone();
        duplicate["tree"].as_array_mut().unwrap().push(twice);
        trees.push(duplicate);
        let mut no_size = entries("100644");
        no_size["tree"][0].as_object_mut().unwrap().remove("size");
        trees.push(no_size);
        for tree in trees {
            let fixture = repo(tree);
            assert!(
                resolve(
                    &fixture,
                    &request(),
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
            fixture.finished();
        }
    }
    #[test]
    fn github_subfolder_remains_a_repo_relative_logical_path() {
        let mut tree = entries("100644");
        for entry in tree["tree"].as_array_mut().unwrap() {
            entry["path"] = json!(format!("models/{}", entry["path"].as_str().unwrap()));
        }
        let fixture = repo(tree);
        let mut req = request();
        req.selection = Selection::Files {
            paths: vec!["tiny.gguf".into()],
            subfolder: Some("models".into()),
        };
        let plan = resolve(
            &fixture,
            &req,
            &PullLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(
            plan.files()
                .iter()
                .any(|file| file.path() == "models/tiny.gguf")
        );
        fixture.finished();
    }
    fn release(digest: Value) -> Fixture {
        Fixture::new(vec![
            (
                "https://api.github.com/repos/owner/repo/releases/tags/v1".into(),
                json!({"id":91,"tag_name":"v1"}),
            ),
            (
                "https://api.github.com/repos/owner/repo/releases/91/assets?per_page=100&page=1"
                    .into(),
                json!([{"id":113,"name":"tiny.gguf","size":7,"state":"uploaded","digest":digest,"browser_download_url":"https://untrusted.invalid/?token=secret"}]),
            ),
        ])
    }
    fn release_request() -> PullRequest {
        PullRequest {
            provider: Provider::GitHub,
            repository: "owner/repo".into(),
            revision: "v1".into(),
            selection: Selection::ReleaseAsset {
                name: "tiny.gguf".into(),
            },
        }
    }
    #[test]
    fn github_release_asset_id_and_optional_trusted_digest_are_explicit() {
        for digest in [Value::Null, json!(format!("sha256:{}", "a".repeat(64)))] {
            let fixture = release(digest.clone());
            let plan = resolve(
                &fixture,
                &release_request(),
                &PullLimits::default(),
                &AtomicBool::new(false),
            )
            .unwrap();
            fixture.finished();
            let file = &plan.files()[0];
            assert_eq!(
                file.url(),
                "https://api.github.com/repos/owner/repo/releases/assets/113"
            );
            assert_eq!(file.identity(), "release:91:asset:113");
            assert_eq!(plan.source().revision(), "release-91-asset-113");
            assert_eq!(file.sha256().is_some(), !digest.is_null());
            assert!(!format!("{plan:?}").contains("token=secret"));
        }
    }
    #[test]
    fn github_release_invalid_digest_and_incomplete_pages_refuse() {
        let fixture = release(json!("sha1:secret-token"));
        let error = resolve(
            &fixture,
            &release_request(),
            &PullLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(!error.to_string().contains("secret-token"));
        fixture.finished();
        let assets: Vec<_> = (1..=100).map(|i| json!({"id":i,"name":format!("asset{i}"),"size":0,"state":"uploaded","digest":null})).collect();
        let fixture = Fixture::new(vec![
            (
                "https://api.github.com/repos/owner/repo/releases/tags/v1".into(),
                json!({"id":91,"tag_name":"v1"}),
            ),
            (
                "https://api.github.com/repos/owner/repo/releases/91/assets?per_page=100&page=1"
                    .into(),
                json!(assets),
            ),
        ]);
        let limits = PullLimits {
            max_pages: 1,
            ..PullLimits::default()
        };
        assert!(matches!(
            resolve(
                &fixture,
                &release_request(),
                &limits,
                &AtomicBool::new(false)
            ),
            Err(Error::Limit(_))
        ));
        fixture.finished();
    }
    #[test]
    fn github_bad_local_selection_refuses_without_transport() {
        for repository in [
            "owner/repo?token=secret",
            "https://github.com/owner/repo",
            "a/b/c",
        ] {
            let fixture = Fixture::new(vec![]);
            let mut req = request();
            req.repository = repository.into();
            let error = resolve(
                &fixture,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false),
            )
            .unwrap_err();
            assert!(!error.to_string().contains("secret"));
            fixture.finished();
        }
    }
    #[test]
    fn github_metadata_budget_is_shared_across_commit_and_tree() {
        let commit = json!({"sha":COMMIT,"commit":{"tree":{"sha":TREE}}});
        let tree = entries("100644");
        let bytes =
            serde_json::to_vec(&commit).unwrap().len() + serde_json::to_vec(&tree).unwrap().len();
        let fixture = repo(tree);
        let limits = PullLimits {
            max_metadata_bytes: bytes - 1,
            ..PullLimits::default()
        };
        assert!(matches!(
            resolve(&fixture, &request(), &limits, &AtomicBool::new(false)),
            Err(Error::Limit(_))
        ));
        fixture.finished();
    }
    #[test]
    fn github_explicit_commit_cannot_resolve_to_different_commit() {
        let mut req = request();
        req.revision = BLOB.into();
        let fixture = Fixture::new(vec![(
            format!("https://api.github.com/repos/owner/repo/commits/{BLOB}"),
            json!({"sha":COMMIT,"commit":{"tree":{"sha":TREE}}}),
        )]);
        assert!(
            resolve(
                &fixture,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        fixture.finished();
    }
    #[test]
    fn github_uppercase_explicit_pin_cannot_resolve_another_commit() {
        let mut req = request();
        req.revision = "A".repeat(40);
        let fixture = Fixture::new(vec![(
            format!(
                "https://api.github.com/repos/owner/repo/commits/{}",
                req.revision
            ),
            json!({"sha":COMMIT,"commit":{"tree":{"sha":TREE}}}),
        )]);
        assert!(
            resolve(
                &fixture,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        fixture.finished();
    }
    fn model_index() -> Value {
        json!({"metadata":{"total_size":14},"weight_map":{"a":"a.safetensors","b":"b.safetensors"}})
    }
    fn model_tree(index_size: u64) -> Value {
        let entries:Vec<_>=[("model.safetensors.index.json",index_size),("a.safetensors",7),("b.safetensors",7),("config.json",2),("tokenizer.json",2),("LICENSE",3)]
            .iter().map(|(path,size)|json!({"path":path,"size":size,"mode":"100644","type":"blob","sha":BLOB})).collect();
        json!({"sha":TREE,"truncated":false,"tree":entries})
    }
    fn with_index(tree: Value, index: Value) -> Fixture {
        let fixture = repo(tree);
        fixture.0.lock().unwrap().push_back((
            format!(
                "https://raw.githubusercontent.com/owner/repo/{COMMIT}/model.safetensors.index.json"
            ),
            index,
        ));
        fixture
    }
    #[test]
    fn github_selected_or_adjacent_index_closes_shards_and_available_sidecars() {
        let index = model_index();
        let size = serde_json::to_vec(&index).unwrap().len() as u64;
        for path in ["model.safetensors.index.json", "a.safetensors"] {
            let fixture = with_index(model_tree(size), index.clone());
            let mut req = request();
            req.selection = Selection::Files {
                paths: vec![path.into()],
                subfolder: None,
            };
            let plan = resolve(
                &fixture,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(plan.files().len(), 6);
            for required in [
                "a.safetensors",
                "b.safetensors",
                "config.json",
                "tokenizer.json",
                "LICENSE",
            ] {
                assert!(plan.files().iter().any(|file| file.path() == required));
            }
            fixture.finished();
        }
    }
    #[test]
    fn github_missing_index_shard_or_required_sidecar_refuses_before_weight_body() {
        let index = model_index();
        let size = serde_json::to_vec(&index).unwrap().len() as u64;
        for missing in ["b.safetensors", "config.json", "tokenizer.json"] {
            let mut tree = model_tree(size);
            tree["tree"]
                .as_array_mut()
                .unwrap()
                .retain(|entry| entry["path"] != missing);
            let fixture = with_index(tree, index.clone());
            let mut req = request();
            req.selection = Selection::Files {
                paths: vec!["a.safetensors".into()],
                subfolder: None,
            };
            assert!(
                resolve(
                    &fixture,
                    &req,
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
            fixture.finished();
        }
    }
    #[test]
    fn github_lfs_pointer_index_refuses_instead_of_becoming_weights() {
        let pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nsize 99\n";
        let fixture = with_index(model_tree(pointer.len() as u64), json!(pointer));
        let mut req = request();
        req.selection = Selection::Files {
            paths: vec!["model.safetensors.index.json".into()],
            subfolder: None,
        };
        assert!(matches!(
            resolve(
                &fixture,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false)
            ),
            Err(Error::Unsupported(_))
        ));
        fixture.finished();
    }
    #[test]
    fn github_split_gguf_collects_every_part_or_refuses_incomplete_set() {
        for complete in [false, true] {
            let mut tree = entries("100644");
            tree["tree"][0]["path"] = json!("q4-00001-of-00002.gguf");
            if complete {
                let mut second = tree["tree"][0].clone();
                second["path"] = json!("q4-00002-of-00002.gguf");
                tree["tree"].as_array_mut().unwrap().push(second);
            }
            let fixture = repo(tree);
            let mut req = request();
            req.selection = Selection::Files {
                paths: vec!["q4-00001-of-00002.gguf".into()],
                subfolder: None,
            };
            let result = resolve(
                &fixture,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false),
            );
            assert_eq!(result.is_ok(), complete);
            if let Ok(plan) = result {
                assert_eq!(plan.files().len(), 4);
            }
            fixture.finished();
        }
    }
    #[test]
    fn github_index_plus_an_unrelated_weight_is_not_one_variant() {
        let index = model_index();
        let size = serde_json::to_vec(&index).unwrap().len() as u64;
        let mut tree = model_tree(size);
        tree["tree"].as_array_mut().unwrap().push(
            json!({"path":"other.safetensors","size":7,"mode":"100644","type":"blob","sha":BLOB}),
        );
        let fixture = with_index(tree, index);
        let mut req = request();
        req.selection = Selection::Files {
            paths: vec![
                "model.safetensors.index.json".into(),
                "other.safetensors".into(),
            ],
            subfolder: None,
        };
        assert!(matches!(
            resolve(
                &fixture,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false)
            ),
            Err(Error::Unsupported(_))
        ));
        fixture.finished();
    }
}
