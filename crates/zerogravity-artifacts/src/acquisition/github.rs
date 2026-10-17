//! Bounded GitHub metadata resolution. Git object IDs identify a pinned tree;
//! they are never interpreted as raw file SHA256. This module does no transfer
//! or cache publication and executes no repository code or LFS hooks.
use super::http::{HttpAccept, HttpRequest, HttpTransport};
use super::{
    ByteLength, Provider, PullLimits, PullRequest, ResolvedFile, ResolvedPullPlan, Selection,
};
use crate::cache::{FileRole, ManifestSource, paths::validate_relative_path};
use crate::{Error, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;
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

/// A stable release-ID binding for a previously resolved alias. This does not
/// freeze its asset set; each label still lists/selects its own pinned assets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ReleasePin {
    id: NonZeroU64,
}

/// Canonical original per-asset cache locator, never a tag or payload digest.
pub(super) fn parse_release_locator(value: &str) -> Option<(ReleasePin, u64)> {
    fn positive(value: &str) -> Option<NonZeroU64> {
        let number = value.parse::<NonZeroU64>().ok()?;
        (number.to_string() == value).then_some(number)
    }
    let (release, asset) = value.strip_prefix("release-")?.split_once("-asset-")?;
    Some((
        ReleasePin {
            id: positive(release)?,
        },
        positive(asset)?.get(),
    ))
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
pub(super) fn repository(repository: &str) -> Result<()> {
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
pub(super) fn role(path: &str) -> FileRole {
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
    let available = entries
        .iter()
        .map(|(path, entry)| {
            (
                path.clone(),
                entry.kind == "blob" && matches!(entry.mode.as_str(), "100644" | "100755"),
            )
        })
        .collect();
    super::selection::close_model_selection(&available, selected, limits, sidecars, &mut |path| {
        let entry = entries.get_mut(path).ok_or_else(malformed)?;
        if entry.kind != "blob" || !matches!(entry.mode.as_str(), "100644" | "100755") {
            return Err(Error::Unsupported(
                "GitHub index is not a regular blob".into(),
            ));
        }
        let size = ensure_length(
            entry,
            transport,
            raw_url(repository, pin, path),
            limits.cache.max_file_bytes.min(*remaining as u64),
            probes,
            cancel,
            pin,
        )?;
        let request = HttpRequest::new(raw_url(repository, pin, path), size);
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
        Ok(bytes)
    })
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

fn checked_api(request: &PullRequest, limits: &PullLimits) -> Result<String> {
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
    Ok(format!(
        "https://api.github.com/repos/{}",
        request.repository
    ))
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
    let api = checked_api(request, limits)?;
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
            let (commit, mut entries) = repository_tree(
                transport,
                &api,
                &request.revision,
                limits,
                &mut metadata_left,
                cancel,
            )?;
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
            let model_selection = selected
                .iter()
                .any(|path| matches!(role(path), FileRole::Weights | FileRole::Index));
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
                let file_role = role(&path);
                let length = ensure_length_with_cap(
                    entry,
                    transport,
                    raw_url(&request.repository, &commit.sha, &path),
                    (
                        limits.cache.max_file_bytes,
                        if file_role == FileRole::Auxiliary && !model_selection {
                            limits.unknown_length_cap
                        } else {
                            None
                        },
                    ),
                    &mut probes_left,
                    cancel,
                    &commit.sha,
                )?;
                let encoded_path = path.split('/').map(component).collect::<Vec<_>>().join("/");
                files.push(
                    ResolvedFile::new_with_length(
                        &path,
                        format!(
                            "https://raw.githubusercontent.com/{}/{}/{encoded_path}",
                            request.repository, commit.sha
                        ),
                        file_role,
                        length,
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
        Selection::ReleaseAsset { .. } => {
            resolve_release_component(transport, request, limits, cancel, None)
                .map(|(plan, _)| plan)
        }
    }
}

// Preserve the original commit/tree policy for both resolution and discovery.
fn repository_tree(
    transport: &(impl HttpTransport + ?Sized),
    api: &str,
    revision: &str,
    limits: &PullLimits,
    metadata_left: &mut usize,
    cancel: &AtomicBool,
) -> Result<(Commit, BTreeMap<String, TreeEntry>)> {
    let commit: Commit = metadata(
        transport,
        format!("{api}/commits/{}", component(revision)),
        metadata_left,
        cancel,
    )?;
    object_id(&commit.sha)?;
    if revision.len() == 40
        && revision.bytes().all(|c| c.is_ascii_hexdigit())
        && !revision.eq_ignore_ascii_case(&commit.sha)
    {
        return Err(malformed());
    }
    object_id(&commit.commit.tree.sha)?;
    let tree: Tree = metadata(
        transport,
        format!("{api}/git/trees/{}?recursive=1", commit.commit.tree.sha),
        metadata_left,
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
    Ok((commit, entries))
}

pub(super) fn discover_candidates(
    transport: &impl HttpTransport,
    request: &super::candidates::CandidateRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
) -> Result<(ManifestSource, Vec<super::candidates::CandidateFile>)> {
    let api = format!("https://api.github.com/repos/{}", request.repository());
    let mut remaining = limits.max_metadata_bytes;
    let (commit, entries) = repository_tree(
        transport,
        &api,
        request.revision(),
        limits,
        &mut remaining,
        cancel,
    )?;
    let mut files = Vec::new();
    for entry in entries.into_values() {
        super::http::check_cancel(cancel)?;
        // Nonregular tree members are not download candidates, even when their
        // names resemble a GGUF. Their paths/OIDs were still validated above.
        if entry.kind == "blob" && matches!(entry.mode.as_str(), "100644" | "100755") {
            let role = role(&entry.path);
            files.push(super::candidates::CandidateFile::from_metadata(
                entry.path, entry.size, role, None,
            )?);
        }
    }
    Ok((
        ManifestSource::new("github", request.repository(), commit.sha)?,
        files,
    ))
}

/// Reuse only a typed release pin. Every call retains the supplied label's
/// transport and bounded asset listing; a prior pin never becomes a fake tag.
pub(super) fn resolve_release_component(
    transport: &(impl HttpTransport + ?Sized),
    request: &PullRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
    prior: Option<ReleasePin>,
) -> Result<(ResolvedPullPlan, ReleasePin)> {
    let api = checked_api(request, limits)?;
    let Selection::ReleaseAsset { name } = &request.selection else {
        return Err(malformed());
    };
    super::http::check_cancel(cancel)?;
    let mut metadata_left = limits.max_metadata_bytes;
    let mut probes_left = limits.max_pages;
    validate_relative_path(name)?;
    if name.contains('/') {
        return Err(malformed());
    }
    let release = match prior {
        Some(pin) => pin,
        None => {
            let resolved: Release = metadata(
                transport,
                format!("{api}/releases/tags/{}", component(&request.revision)),
                &mut metadata_left,
                cancel,
            )?;
            if resolved.tag_name != request.revision {
                return Err(malformed());
            }
            ReleasePin {
                id: NonZeroU64::new(resolved.id).ok_or_else(malformed)?,
            }
        }
    };
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
        if assets.len() > 100 || ids.len().saturating_add(assets.len()) > limits.cache.max_files {
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
    let asset = selected
        .ok_or_else(|| Error::Unsupported("required GitHub release asset is absent".into()))?;
    if asset.state != "uploaded" {
        return Err(Error::Unsupported(
            "GitHub release asset is not uploaded".into(),
        ));
    }
    let digest = match asset.digest {
        None => None,
        Some(value) => {
            let digest = value
                .strip_prefix("sha256:")
                .ok_or_else(|| Error::Unsupported("GitHub asset checksum is not SHA256".into()))?;
            crate::cache::paths::validate_digest(digest)?;
            Some(digest.to_owned())
        }
    };
    let asset_url = format!("{api}/releases/assets/{}", asset.id);
    let (bytes, validator) = match asset.size {
        Some(bytes) => (ByteLength::Exact(bytes), None),
        None => super::http::probe_length_with_cap(
            transport,
            asset_url.clone(),
            limits.cache.max_file_bytes,
            &mut probes_left,
            cancel,
            None,
            if role(name) == FileRole::Auxiliary {
                limits.unknown_length_cap
            } else {
                None
            },
        )?,
    };
    let file = ResolvedFile::new_with_length(
        name,
        asset_url,
        role(name),
        bytes,
        digest,
        format!("release:{}:asset:{}", release.id, asset.id),
    )?
    .with_validator(validator)?;
    let plan = ResolvedPullPlan::new(
        ManifestSource::new(
            "github",
            &request.repository,
            format!("release-{}-asset-{}", release.id, asset.id),
        )?,
        vec![file],
        limits,
    )?;
    Ok((plan, release))
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
    ensure_length_with_cap(entry, transport, url, (maximum, None), probes, cancel, pin)?
        .exact()
        .ok_or_else(|| Error::Unsupported("exact GitHub object length required".into()))
}

fn ensure_length_with_cap(
    entry: &mut TreeEntry,
    transport: &(impl HttpTransport + ?Sized),
    url: String,
    bounds: (u64, Option<u64>),
    probes: &mut usize,
    cancel: &AtomicBool,
    pin: &str,
) -> Result<ByteLength> {
    let (maximum, cap) = bounds;
    let size = match entry.size {
        Some(size) => ByteLength::Exact(size),
        None => {
            let (size, validator) = super::http::probe_length_with_cap(
                transport,
                url,
                maximum,
                probes,
                cancel,
                Some(pin),
                cap,
            )?;
            entry.size = size.exact();
            entry.validator = validator;
            size
        }
    };
    if size.maximum() > maximum {
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
        assert_eq!(plan.total_bytes(), Some(11));
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
    fn canonical_release_locators_preserve_distinct_asset_ids_and_reject_aliases() {
        let (pin, asset) = parse_release_locator("release-91-asset-113").unwrap();
        assert_eq!(pin.id.get(), 91);
        assert_eq!(asset, 113);
        let (same, other_asset) = parse_release_locator("release-91-asset-114").unwrap();
        assert_eq!(pin, same);
        assert_ne!(asset, other_asset);
        assert!(
            parse_release_locator(&format!("release-{}-asset-{}", u64::MAX, u64::MAX)).is_some()
        );
        for bad in [
            "v1",
            "release-91",
            "release-0-asset-1",
            "release-1-asset-0",
            "release-01-asset-1",
            "release-1-asset-01",
            "release-+1-asset-1",
            "release-1-asset-+1",
            "release-18446744073709551616-asset-1",
            "release-1-asset-18446744073709551616",
            "release-1-asset-2-asset-3",
            "release-1-asset-2\n",
            "release-１-asset-2",
            "release--1-asset-2",
            "release-1-asset-",
        ] {
            assert!(parse_release_locator(bad).is_none(), "{bad:?}");
        }
    }
    #[test]
    fn prior_release_pin_skips_alias_lookup_and_selects_original_asset_sources() {
        let fixture = Fixture::new(vec![
            (
                "https://api.github.com/repos/owner/repo/releases/tags/v1".into(),
                json!({"id":91,"tag_name":"v1"}),
            ),
            (
                "https://api.github.com/repos/owner/repo/releases/91/assets?per_page=100&page=1"
                    .into(),
                json!([{"id":113,"name":"tiny.gguf","size":7,"state":"uploaded","digest":null}]),
            ),
            (
                "https://api.github.com/repos/owner/repo/releases/91/assets?per_page=100&page=1"
                    .into(),
                json!([{"id":114,"name":"config.json","size":2,"state":"uploaded","digest":format!("sha256:{}", "a".repeat(64))}]),
            ),
        ]);
        let (weights, pin) = resolve_release_component(
            &fixture,
            &release_request(),
            &PullLimits::default(),
            &AtomicBool::new(false),
            None,
        )
        .unwrap();
        let mut config_request = release_request();
        config_request.selection = Selection::ReleaseAsset {
            name: "config.json".into(),
        };
        let (config, reused) = resolve_release_component(
            &fixture,
            &config_request,
            &PullLimits::default(),
            &AtomicBool::new(false),
            Some(pin),
        )
        .unwrap();
        assert_eq!(pin, reused);
        assert_eq!(weights.source().revision(), "release-91-asset-113");
        assert_eq!(config.source().revision(), "release-91-asset-114");
        assert_eq!(
            parse_release_locator(config.source().revision()).unwrap(),
            (pin, 114)
        );
        assert_eq!(weights.files()[0].sha256(), None);
        assert_eq!(config.files()[0].sha256(), Some("a".repeat(64).as_str()));
        assert_eq!(
            config.files()[0].url(),
            "https://api.github.com/repos/owner/repo/releases/assets/114"
        );
        fixture.finished();
    }
    #[test]
    fn prior_release_pin_does_not_freeze_asset_membership_or_retry_the_tag() {
        let pin = parse_release_locator("release-91-asset-113").unwrap().0;
        let fixture = Fixture::new(vec![
            (
                "https://api.github.com/repos/owner/repo/releases/91/assets?per_page=100&page=1"
                    .into(),
                json!([{"id":999,"name":"tiny.gguf","size":7,"state":"uploaded","digest":null}]),
            ),
            (
                "https://api.github.com/repos/owner/repo/releases/91/assets?per_page=100&page=1"
                    .into(),
                json!([]),
            ),
        ]);
        let (plan, reused) = resolve_release_component(
            &fixture,
            &release_request(),
            &PullLimits::default(),
            &AtomicBool::new(false),
            Some(pin),
        )
        .unwrap();
        assert_eq!(reused, pin);
        assert_eq!(plan.source().revision(), "release-91-asset-999");
        assert!(matches!(
            resolve_release_component(
                &fixture,
                &release_request(),
                &PullLimits::default(),
                &AtomicBool::new(false),
                Some(pin)
            ),
            Err(Error::Unsupported(_))
        ));
        fixture.finished();
    }
    #[test]
    fn prior_release_pin_still_validates_source_selection_and_cancellation_before_io() {
        let pin = parse_release_locator("release-91-asset-113").unwrap().0;
        let fixture = Fixture::new(Vec::new());
        for invalid in 0..5 {
            let mut req = release_request();
            match invalid {
                0 => req.provider = Provider::HuggingFace,
                1 => {
                    req.selection = Selection::Files {
                        paths: vec!["config.json".into()],
                        subfolder: None,
                    }
                }
                2 => req.repository = "owner/repo?secret".into(),
                3 => req.revision.clear(),
                _ => {
                    req.selection = Selection::ReleaseAsset {
                        name: "nested/config.json".into(),
                    }
                }
            }
            let error = resolve_release_component(
                &fixture,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false),
                Some(pin),
            )
            .unwrap_err();
            assert!(!error.to_string().contains("secret"));
        }
        assert!(matches!(
            resolve_release_component(
                &fixture,
                &release_request(),
                &PullLimits::default(),
                &AtomicBool::new(true),
                Some(pin)
            ),
            Err(Error::State(_))
        ));
        fixture.finished();
    }
    #[test]
    fn release_pin_requires_matching_alias_and_positive_id_before_asset_listing() {
        for (id, tag) in [(0, "v1"), (91, "changed")] {
            let fixture = Fixture::new(vec![(
                "https://api.github.com/repos/owner/repo/releases/tags/v1".into(),
                json!({"id":id,"tag_name":tag}),
            )]);
            assert!(matches!(
                resolve_release_component(
                    &fixture,
                    &release_request(),
                    &PullLimits::default(),
                    &AtomicBool::new(false),
                    None
                ),
                Err(Error::Malformed(_))
            ));
            fixture.finished();
        }
    }
    #[test]
    fn prior_release_pin_keeps_duplicate_id_name_state_and_metadata_limits() {
        let pin = parse_release_locator("release-91-asset-113").unwrap().0;
        for assets in [
            json!([{"id":113,"name":"tiny.gguf","size":7,"state":"uploaded"},{"id":113,"name":"other.txt","size":1,"state":"uploaded"}]),
            json!([{"id":113,"name":"tiny.gguf","size":7,"state":"uploaded"},{"id":114,"name":"tiny.gguf","size":1,"state":"uploaded"}]),
            json!([{"id":0,"name":"tiny.gguf","size":7,"state":"uploaded"}]),
            json!([{"id":113,"name":"tiny.gguf","size":7,"state":"new"}]),
        ] {
            let fixture = Fixture::new(vec![(
                "https://api.github.com/repos/owner/repo/releases/91/assets?per_page=100&page=1"
                    .into(),
                assets,
            )]);
            assert!(
                resolve_release_component(
                    &fixture,
                    &release_request(),
                    &PullLimits::default(),
                    &AtomicBool::new(false),
                    Some(pin)
                )
                .is_err()
            );
            fixture.finished();
        }
        let fixture = Fixture::new(vec![(
            "https://api.github.com/repos/owner/repo/releases/91/assets?per_page=100&page=1".into(),
            json!([]),
        )]);
        let limits = PullLimits {
            max_metadata_bytes: 1,
            ..Default::default()
        };
        assert!(matches!(
            resolve_release_component(
                &fixture,
                &release_request(),
                &limits,
                &AtomicBool::new(false),
                Some(pin)
            ),
            Err(Error::Limit(_))
        ));
        fixture.finished();
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
