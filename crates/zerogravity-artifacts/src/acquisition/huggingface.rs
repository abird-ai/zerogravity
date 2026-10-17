//! Native HF metadata resolution. Every request after revision resolution uses
//! the same immutable commit. Configuration never authorizes another repo.

use super::http::{HttpAccept, HttpRequest, HttpTransport, check_cancel};
use super::{
    ByteLength, ExternalSidecars, Provider, PullLimits, PullRequest, ResolvedFile,
    ResolvedPullPlan, Selection,
};
use crate::cache::paths::validate_relative_path;
use crate::cache::{FileRole, ManifestSource};
use crate::sharded::SafetensorsIndex;
use crate::{Error, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use url::Url;

#[derive(Deserialize)]
struct Revision {
    sha: String,
}
#[derive(Deserialize)]
struct Lfs {
    oid: String,
    size: u64,
}
#[derive(Deserialize)]
struct Entry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: Option<u64>,
    lfs: Option<Lfs>,
    #[serde(skip)]
    validator: Option<String>,
}

/// Internal resolver: the public front door holds metadata admission first.
pub(super) fn resolve(
    transport: &impl HttpTransport,
    request: &PullRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    resolve_from(
        transport,
        request,
        limits,
        cancel,
        "https://huggingface.co/",
        ExternalSidecars::default(),
    )
}

pub(super) fn resolve_component(
    transport: &impl HttpTransport,
    request: &PullRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
    sidecars: ExternalSidecars,
) -> Result<ResolvedPullPlan> {
    resolve_from(
        transport,
        request,
        limits,
        cancel,
        "https://huggingface.co/",
        sidecars,
    )
}

#[cfg(feature = "diagnostic-acquisition")]
pub fn diagnostic_resolve(
    transport: &impl HttpTransport,
    request: &PullRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
    endpoint: &str,
) -> Result<ResolvedPullPlan> {
    let endpoint =
        Url::parse(endpoint).map_err(|_| Error::Malformed("invalid HF fixture endpoint".into()))?;
    if endpoint.scheme() != "http"
        || !matches!(endpoint.host(), Some(url::Host::Ipv4(ip)) if ip.is_loopback())
            && !matches!(endpoint.host(), Some(url::Host::Ipv6(ip)) if ip.is_loopback())
    {
        return Err(Error::Unsupported(
            "HF fixture endpoint requires literal loopback".into(),
        ));
    }
    resolve_from(
        transport,
        request,
        limits,
        cancel,
        endpoint.as_str(),
        ExternalSidecars::default(),
    )
}

fn resolve_from(
    transport: &impl HttpTransport,
    request: &PullRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
    endpoint: &str,
    sidecars: ExternalSidecars,
) -> Result<ResolvedPullPlan> {
    request.validate(limits)?;
    if request.provider != Provider::HuggingFace {
        return Err(Error::Unsupported("HF provider request required".into()));
    }
    validate_repo(&request.repository)?;
    let Selection::Files { paths, subfolder } = &request.selection else {
        return Err(Error::Unsupported(
            "HF release assets are not supported".into(),
        ));
    };
    let base = Url::parse(endpoint).map_err(|_| Error::Malformed("invalid HF endpoint".into()))?;
    if !base.username().is_empty()
        || base.password().is_some()
        || base.query().is_some()
        || base.fragment().is_some()
        || base.path() != "/"
    {
        return Err(Error::Malformed("invalid HF endpoint authority".into()));
    }
    let repo: Vec<_> = request.repository.split('/').collect();
    let mut revision_segments = vec!["api", "models"];
    revision_segments.extend(&repo);
    revision_segments.extend(["revision", &request.revision]);
    let mut metadata_left = limits.max_metadata_bytes;
    let mut probes_left = limits.max_pages;
    let (bytes, _) = metadata(
        transport,
        make_url(&base, &revision_segments)?,
        &mut metadata_left,
        cancel,
    )?;
    let pin: Revision = serde_json::from_slice(&bytes)
        .map_err(|_| Error::Malformed("invalid HF revision metadata".into()))?;
    if pin.sha.len() != 40 || !pin.sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::Malformed(
            "HF revision did not resolve to exact commit".into(),
        ));
    }
    let pin = pin.sha.to_ascii_lowercase();
    if request.revision.len() == 40
        && request.revision.bytes().all(|b| b.is_ascii_hexdigit())
        && !request.revision.eq_ignore_ascii_case(&pin)
    {
        return Err(Error::State("HF resolved commit changed".into()));
    }
    let mut tree_segments = vec!["api", "models"];
    tree_segments.extend(&repo);
    tree_segments.extend(["tree", pin.as_str()]);
    let mut next = make_url(&base, &tree_segments)?;
    next.set_query(Some("recursive=true&expand=false&limit=1000"));
    let expected_tree_path = next.path().to_string();
    let mut visited = BTreeSet::new();
    let mut entries = BTreeMap::new();
    for page in 0..limits.max_pages {
        check_cancel(cancel)?;
        if !visited.insert(next.as_str().to_string()) {
            return Err(Error::Malformed("HF pagination cycle".into()));
        }
        let (body, link) = metadata(transport, next.clone(), &mut metadata_left, cancel)?;
        let listed: Vec<Entry> = serde_json::from_slice(&body)
            .map_err(|_| Error::Malformed("invalid HF tree metadata".into()))?;
        for entry in listed {
            validate_relative_path(&entry.path)?;
            if entry.kind == "directory" {
                continue;
            }
            if entry.kind != "file" {
                return Err(Error::Unsupported("unsupported HF tree entry type".into()));
            }
            if entries.len() >= limits.cache.max_files {
                return Err(Error::Limit("HF tree file limit".into()));
            }
            if entries.insert(entry.path.clone(), entry).is_some() {
                return Err(Error::Malformed("duplicate HF tree path".into()));
            }
        }
        match link.as_deref().map(next_link).transpose()?.flatten() {
            None => break,
            Some(url) => {
                if page + 1 == limits.max_pages {
                    return Err(Error::Limit("HF tree page limit".into()));
                }
                next = base
                    .join(&url)
                    .map_err(|_| Error::Malformed("invalid HF pagination URL".into()))?;
                if next.origin() != base.origin()
                    || next.path() != expected_tree_path
                    || next.fragment().is_some()
                    || !next.username().is_empty()
                    || next.password().is_some()
                {
                    return Err(Error::Malformed("HF pagination escaped pinned tree".into()));
                }
            }
        }
    }
    let available: Vec<_> = entries.keys().map(String::as_str).collect();
    let mut selected = select_paths(paths, subfolder.as_deref(), &available)?;
    let initial = selected.clone();
    let mut indexes = BTreeSet::new();
    let mut index_shards = BTreeSet::new();
    for path in &initial {
        if path.ends_with(".safetensors.index.json") {
            indexes.insert(path.clone());
        }
        if path.ends_with(".safetensors") {
            let folder = path.rsplit_once('/').map(|(folder, _)| folder);
            for candidate in entries.keys().filter(|candidate| {
                candidate.ends_with(".safetensors.index.json")
                    && candidate.rsplit_once('/').map(|(folder, _)| folder) == folder
            }) {
                indexes.insert(candidate.clone());
            }
        }
        if let Some((prefix, _, total)) = split_gguf(path) {
            if total == 0 || total as usize > limits.cache.max_files {
                return Err(Error::Limit("GGUF split count".into()));
            }
            for part in 1..=total {
                require(
                    &entries,
                    &mut selected,
                    format!("{prefix}-{part:05}-of-{total:05}.gguf"),
                )?;
            }
        }
    }
    let mut covered = BTreeMap::<String, usize>::new();
    for path in indexes {
        let url = file_url(&base, &repo, &pin, &path)?;
        let entry = entries
            .get_mut(&path)
            .ok_or_else(|| Error::Malformed("HF index disappeared".into()))?;
        let size = ensure_length(
            entry,
            transport,
            url.to_string(),
            limits.cache.max_file_bytes.min(metadata_left as u64),
            &mut probes_left,
            cancel,
            &pin,
        )?;
        let (body, _) = metadata_checked(
            transport,
            url,
            &mut metadata_left,
            cancel,
            entry.validator.as_deref(),
        )?;
        if body.len() as u64 != size {
            return Err(Error::Malformed("HF index length changed".into()));
        }
        verify_metadata_digest(entry, &body)?;
        let index = SafetensorsIndex::from_json(&body)
            .map_err(|_| Error::Malformed("invalid HF safetensors index".into()))?;
        let folder = path.rsplit_once('/').map(|(prefix, _)| prefix);
        let shards: Vec<_> = index
            .shards()
            .into_iter()
            .map(|shard| match folder {
                Some(folder) => format!("{folder}/{shard}"),
                None => shard.to_string(),
            })
            .collect();
        if initial.contains(&path) || shards.iter().any(|shard| initial.contains(shard)) {
            require(&entries, &mut selected, path)?;
            for shard in shards {
                require(&entries, &mut selected, shard.clone())?;
                *covered.entry(shard.clone()).or_default() += 1;
                index_shards.insert(shard);
            }
        }
    }
    if covered.values().any(|count| *count > 1) {
        return Err(Error::Unsupported("ambiguous HF shard indexes".into()));
    }
    // Selected or adjacent pinned indexes supply complete shard closure.
    // Multiple unrelated variants never resolve by taking the first match.
    let mut families = BTreeSet::new();
    let mut split_counts = BTreeMap::new();
    for path in &selected {
        if path.ends_with(".safetensors") && path.contains("-of-") && !index_shards.contains(path) {
            return Err(Error::Unsupported(
                "HF safetensors shard requires a complete pinned index".into(),
            ));
        }
        if path.ends_with(".gguf") && path.contains("-of-") {
            let Some((prefix, _, total)) = split_gguf(path) else {
                return Err(Error::Malformed("invalid HF GGUF split name".into()));
            };
            if split_counts
                .insert(prefix, total)
                .is_some_and(|prior| prior != total)
            {
                return Err(Error::Malformed("inconsistent HF GGUF split count".into()));
            }
        }
        if path.ends_with(".safetensors.index.json")
            || is_weight(path) && !index_shards.contains(path)
        {
            families.insert(
                split_gguf(path)
                    .map(|(prefix, _, _)| prefix)
                    .unwrap_or_else(|| path.clone()),
            );
        }
    }
    if families.len() > 1 {
        return Err(Error::Unsupported("ambiguous HF weight variants".into()));
    }
    let weight_folders: BTreeSet<_> = selected
        .iter()
        .filter(|p| is_weight(p) || p.ends_with(".safetensors.index.json"))
        .map(|p| p.rsplit_once('/').map(|(prefix, _)| prefix.to_string()))
        .collect();
    for folder in weight_folders {
        for required in ["config.json", "tokenizer.json"] {
            let path = folder
                .as_ref()
                .map(|f| format!("{f}/{required}"))
                .unwrap_or_else(|| required.to_string());
            if entries.contains_key(&path) || !sidecars.allows_missing(required) {
                require(&entries, &mut selected, path)?;
            }
        }
        for optional in [
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
            let path = folder
                .as_ref()
                .map(|f| format!("{f}/{optional}"))
                .unwrap_or_else(|| optional.to_string());
            if entries.contains_key(&path) {
                selected.insert(path);
            }
        }
    }
    let model_selection = selected
        .iter()
        .any(|path| is_weight(path) || path.ends_with(".index.json"));
    let files = selected
        .into_iter()
        .map(|path| {
            let url = file_url(&base, &repo, &pin, &path)?;
            let entry = entries
                .get_mut(&path)
                .ok_or_else(|| Error::Malformed("HF selection disappeared".into()))?;
            let role = file_role(&path);
            let length = ensure_length_with_cap(
                entry,
                transport,
                url.to_string(),
                (
                    limits.cache.max_file_bytes,
                    if role == FileRole::Auxiliary && !model_selection {
                        limits.unknown_length_cap
                    } else {
                        None
                    },
                ),
                &mut probes_left,
                cancel,
                &pin,
            )?;
            let digest = match &entry.lfs {
                Some(lfs) => {
                    if length.exact() != Some(lfs.size)
                        || lfs.oid.len() != 64
                        || !lfs.oid.bytes().all(|b| b.is_ascii_hexdigit())
                    {
                        return Err(Error::Malformed("invalid HF LFS identity or length".into()));
                    }
                    Some(lfs.oid.to_ascii_lowercase())
                }
                None => None, // A git blob OID is not the raw content SHA256.
            };
            ResolvedFile::new_with_length(
                &path,
                url.to_string(),
                role,
                length,
                digest,
                format!("hf:{}:{pin}:{path}", request.repository),
            )?
            .with_validator(entry.validator.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    let source = ManifestSource::new("huggingface", &request.repository, pin)?;
    ResolvedPullPlan::new(source, files, limits)
}

fn file_role(path: &str) -> FileRole {
    if is_weight(path) {
        FileRole::Weights
    } else if path.ends_with(".index.json") {
        FileRole::Index
    } else if path.ends_with("config.json") && !path.contains("tokenizer") {
        FileRole::Config
    } else if path.contains("tokenizer")
        || path.ends_with("vocab.json")
        || path.ends_with("merges.txt")
        || path.ends_with("added_tokens.json")
    {
        FileRole::Tokenizer
    } else if path
        .rsplit('/')
        .next()
        .is_some_and(|name| name.starts_with("LICENSE"))
    {
        FileRole::License
    } else {
        FileRole::Auxiliary
    }
}

fn ensure_length(
    entry: &mut Entry,
    transport: &impl HttpTransport,
    url: String,
    maximum: u64,
    probes: &mut usize,
    cancel: &AtomicBool,
    pin: &str,
) -> Result<u64> {
    ensure_length_with_cap(entry, transport, url, (maximum, None), probes, cancel, pin)?
        .exact()
        .ok_or_else(|| Error::Unsupported("exact HF object length required".into()))
}

fn ensure_length_with_cap(
    entry: &mut Entry,
    transport: &impl HttpTransport,
    url: String,
    bounds: (u64, Option<u64>),
    probes: &mut usize,
    cancel: &AtomicBool,
    pin: &str,
) -> Result<ByteLength> {
    let (maximum, cap) = bounds;
    if let Some(lfs) = &entry.lfs {
        crate::cache::paths::validate_digest(&lfs.oid.to_ascii_lowercase())?;
        if entry.size.is_some_and(|size| size != lfs.size) {
            return Err(Error::Malformed(
                "HF listed and LFS lengths conflict".into(),
            ));
        }
        entry.size = Some(lfs.size);
    }
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
        return Err(Error::Limit("HF selected file length limit".into()));
    }
    Ok(size)
}

fn metadata(
    transport: &impl HttpTransport,
    url: Url,
    remaining: &mut usize,
    cancel: &AtomicBool,
) -> Result<(Vec<u8>, Option<String>)> {
    metadata_checked(transport, url, remaining, cancel, None)
}
fn metadata_checked(
    transport: &impl HttpTransport,
    url: Url,
    remaining: &mut usize,
    cancel: &AtomicBool,
    validator: Option<&str>,
) -> Result<(Vec<u8>, Option<String>)> {
    if *remaining == 0 {
        return Err(Error::Limit("HF metadata aggregate limit".into()));
    }
    let mut request = HttpRequest::new(url.to_string(), *remaining as u64);
    request.accept = HttpAccept::Json;
    let response = transport.get(&request, cancel)?;
    if response.status != 200 {
        return Err(Error::State("HF metadata request failed".into()));
    }
    super::http::validate_validator(validator, &response)?;
    let link = response.header("link").map(str::to_string);
    let body = response.read_bounded(*remaining, cancel)?;
    *remaining -= body.len();
    Ok((body, link))
}

fn validate_repo(repo: &str) -> Result<()> {
    let parts: Vec<_> = repo.split('/').collect();
    if parts.is_empty()
        || parts.len() > 2
        || parts.iter().any(|p| {
            p.is_empty()
                || *p == "."
                || *p == ".."
                || !p
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-.".contains(&b))
        })
    {
        return Err(Error::Malformed("invalid HF repository identifier".into()));
    }
    Ok(())
}

fn make_url(base: &Url, parts: &[&str]) -> Result<Url> {
    let mut url = base.clone();
    url.path_segments_mut()
        .map_err(|_| Error::Malformed("invalid HF endpoint".into()))?
        .clear()
        .extend(parts.iter().copied());
    Ok(url)
}
fn file_url(base: &Url, repo: &[&str], pin: &str, path: &str) -> Result<Url> {
    let mut parts = repo.to_vec();
    parts.extend(["resolve", pin]);
    parts.extend(path.split('/'));
    make_url(base, &parts)
}
fn require(
    entries: &BTreeMap<String, Entry>,
    selected: &mut BTreeSet<String>,
    path: String,
) -> Result<()> {
    if !entries.contains_key(&path) {
        return Err(Error::Unsupported(
            "required HF shard or sidecar missing; no implicit other repository".into(),
        ));
    }
    selected.insert(path);
    Ok(())
}
fn is_weight(path: &str) -> bool {
    path.ends_with(".gguf") || path.ends_with(".safetensors") || path.ends_with(".bin")
}
fn split_gguf(path: &str) -> Option<(String, u32, u32)> {
    let (part, total) = path.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (prefix, index) = part.rsplit_once('-')?;
    if index.len() != 5
        || total.len() != 5
        || !index
            .bytes()
            .chain(total.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let (index, total) = (index.parse().ok()?, total.parse().ok()?);
    if index == 0 || index > total {
        return None;
    }
    Some((prefix.to_string(), index, total))
}
/// Pure byte-exact initial HF selection, shared by pinned online trees and
/// verified offline snapshots. This selects paths; model/shard closure remains
/// the resolver/executor's separate responsibility. Wildcards never cross '/'.
pub fn select_paths(
    patterns: &[String],
    subfolder: Option<&str>,
    available: &[&str],
) -> Result<BTreeSet<String>> {
    if patterns.is_empty() || patterns.len() > 4096 || available.len() > 4096 {
        return Err(Error::Limit("HF filter input count".into()));
    }
    if let Some(folder) = subfolder {
        crate::cache::paths::validate_relative_path(folder)?;
    }
    let mut path_bytes = 0usize;
    for path in patterns
        .iter()
        .map(String::as_str)
        .chain(available.iter().copied())
    {
        crate::cache::paths::validate_relative_path(path)?;
        path_bytes = path_bytes
            .checked_add(path.len())
            .filter(|n| *n <= 8 * 1024 * 1024)
            .ok_or_else(|| Error::Limit("HF filter path bytes".into()))?;
    }
    let available_paths: BTreeSet<_> = available.iter().copied().collect();
    if available_paths.len() != available.len() {
        return Err(Error::Malformed("duplicate available HF path".into()));
    }
    let mut selected = BTreeSet::new();
    let mut filter_work = 16 * 1024 * 1024_usize;
    for requested in patterns {
        let pattern = match subfolder {
            Some(folder) => format!("{folder}/{requested}"),
            None => requested.clone(),
        };
        crate::cache::paths::validate_relative_path(&pattern)?;
        let mut matches = Vec::new();
        if !pattern.contains(['*', '?']) {
            if available_paths.contains(pattern.as_str()) {
                matches.push(pattern.clone());
            }
        } else {
            for path in available {
                if glob_match(&pattern, path, &mut filter_work)? {
                    matches.push((*path).to_string());
                }
            }
        }
        if matches.is_empty() {
            return Err(Error::Unsupported(
                "requested HF file or filter is missing".into(),
            ));
        }
        if matches.len() > 1 && matches.iter().any(|p| p.ends_with(".gguf")) {
            let families: BTreeSet<_> = matches
                .iter()
                .map(|p| {
                    split_gguf(p)
                        .map(|(prefix, _, _)| prefix)
                        .unwrap_or_else(|| p.clone())
                })
                .collect();
            if families.len() != 1 {
                return Err(Error::Unsupported(
                    "ambiguous HF quant selection; select exact file".into(),
                ));
            }
        }
        selected.extend(matches);
    }
    Ok(selected)
}

fn glob_match(pattern: &str, value: &str, work: &mut usize) -> Result<bool> {
    // Bounded byte-exact filters: '*' matches within a path component only.
    let (p, v) = (pattern.as_bytes(), value.as_bytes());
    let (mut i, mut j, mut star, mut retry) = (0, 0, None, 0);
    while j < v.len() {
        *work = work
            .checked_sub(1)
            .ok_or_else(|| Error::Limit("HF filter work limit".into()))?;
        if i < p.len() && (p[i] == v[j] || p[i] == b'?' && v[j] != b'/') {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == b'*' {
            star = Some(i);
            i += 1;
            retry = j;
        } else if let Some(s) = star {
            if retry >= v.len() || v[retry] == b'/' {
                return Ok(false);
            }
            retry += 1;
            j = retry;
            i = s + 1;
        } else {
            return Ok(false);
        }
    }
    while i < p.len() && p[i] == b'*' {
        i += 1;
    }
    Ok(i == p.len())
}
fn next_link(value: &str) -> Result<Option<String>> {
    let mut result = None;
    for part in value.split(',') {
        let Some((target, params)) = part.split_once('>') else {
            return Err(Error::Malformed("invalid HF pagination Link".into()));
        };
        if params
            .split(';')
            .any(|p| p.trim() == "rel=\"next\"" || p.trim() == "rel=next")
        {
            if result.is_some() {
                return Err(Error::Malformed("duplicate HF next page".into()));
            }
            let target = target
                .trim()
                .strip_prefix('<')
                .ok_or_else(|| Error::Malformed("invalid HF pagination Link".into()))?;
            result = Some(target.to_string());
        }
    }
    Ok(result)
}
fn verify_metadata_digest(entry: &Entry, body: &[u8]) -> Result<()> {
    if let Some(lfs) = &entry.lfs {
        use sha2::{Digest, Sha256};
        if lfs.size != body.len() as u64
            || format!("{:x}", Sha256::digest(body)) != lfs.oid.to_ascii_lowercase()
        {
            return Err(Error::Malformed(
                "HF metadata content identity changed".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::http::HttpResponse;
    use super::*;
    use std::io::Cursor;
    use std::sync::Mutex;
    const PIN: &str = "0123456789abcdef0123456789abcdef01234567";
    struct Script {
        responses: Mutex<std::collections::VecDeque<(String, Option<String>)>>,
        requests: Mutex<Vec<String>>,
    }
    impl Script {
        fn new(values: Vec<(String, Option<String>)>) -> Self {
            Self {
                responses: Mutex::new(values.into()),
                requests: Mutex::new(Vec::new()),
            }
        }
    }
    impl HttpTransport for Script {
        fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
            check_cancel(cancel)?;
            self.requests.lock().unwrap().push(request.url.clone());
            let (body, link) = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected metadata call");
            let headers = link
                .map(|v| [("link".into(), v)].into_iter().collect())
                .unwrap_or_default();
            Ok(HttpResponse {
                status: 200,
                headers,
                body: Box::new(Cursor::new(body.into_bytes())),
            })
        }
    }
    fn request(paths: &[&str]) -> PullRequest {
        PullRequest {
            provider: Provider::HuggingFace,
            repository: "owner/model".into(),
            revision: "main".into(),
            selection: Selection::Files {
                paths: paths.iter().map(|v| v.to_string()).collect(),
                subfolder: None,
            },
        }
    }
    fn revision() -> (String, Option<String>) {
        (format!("{{\"sha\":\"{PIN}\"}}"), None)
    }
    fn file(path: &str, size: u64) -> serde_json::Value {
        serde_json::json!({"type":"file","path":path,"size":size})
    }

    #[test]
    fn public_selection_bounds_missing_ambiguity_and_shared_byte_semantics() {
        let available = ["q/model.gguf", "q/config.json", "q/日本.json"];
        assert_eq!(
            select_paths(&["m?del.*".into()], Some("q"), &available).unwrap(),
            BTreeSet::from(["q/model.gguf".into()])
        );
        assert_eq!(
            select_paths(&["日*.json".into()], Some("q"), &available).unwrap(),
            BTreeSet::from(["q/日本.json".into()])
        );
        assert!(select_paths(&["*.gguf".into()], None, &available).is_err());
        assert!(select_paths(&["*.gguf".into()], None, &["q4.gguf", "q8.gguf"]).is_err());
        assert!(select_paths(&["../x".into()], None, &available).is_err());
        assert!(select_paths(&["x".into()], Some("/x"), &available).is_err());
        assert!(select_paths(&["x".into()], None, &["x", "x"]).is_err());
        assert!(select_paths(&vec!["x".into(); 4097], None, &["x"]).is_err());
        let values: Vec<_> = (0..4096)
            .map(|i| format!("{i}/{}", "a".repeat(250)))
            .collect();
        let paths: Vec<_> = values.iter().map(String::as_str).collect();
        assert!(matches!(
            select_paths(&[format!("*/*{}b", "a".repeat(200))], None, &paths),
            Err(Error::Limit(_))
        ));
    }

    #[test]
    fn pinned_safetensors_index_closes_shards_and_required_sidecars() {
        let index = r#"{"metadata":{"total_size":10},"weight_map":{"a":"model-01.safetensors","b":"model-02.safetensors"}}"#;
        let tree = serde_json::json!([
            file("model.safetensors.index.json", index.len() as u64),
            file("model-01.safetensors", 8),
            file("model-02.safetensors", 9),
            file("config.json", 2),
            file("tokenizer.json", 2)
        ])
        .to_string();
        for selected in ["model.safetensors.index.json", "model-01.safetensors"] {
            let script = Script::new(vec![revision(), (tree.clone(), None), (index.into(), None)]);
            let plan = resolve(
                &script,
                &request(&[selected]),
                &PullLimits::default(),
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(plan.source().revision(), PIN);
            assert_eq!(plan.files().len(), 5);
            let urls = script.requests.lock().unwrap();
            for url in &urls[1..] {
                assert!(url.contains(PIN));
                assert!(!url.contains("main"));
            }
            assert!(plan.files().iter().all(|f| f.url().contains(PIN)));
            assert!(plan.files().iter().all(|f| f.sha256().is_none()));
        }
    }

    #[test]
    fn adjacent_index_missing_shard_refuses_before_weight_download() {
        let index = r#"{"metadata":{"total_size":10},"weight_map":{"a":"a.safetensors","b":"missing.safetensors"}}"#;
        let tree = serde_json::json!([
            file("model.safetensors.index.json", index.len() as u64),
            file("a.safetensors", 5),
            file("config.json", 2),
            file("tokenizer.json", 2)
        ])
        .to_string();
        let script = Script::new(vec![revision(), (tree, None), (index.into(), None)]);
        assert!(
            resolve(
                &script,
                &request(&["a.safetensors"]),
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        let requests = script.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(requests[2].ends_with("model.safetensors.index.json"));
    }

    #[test]
    fn split_names_without_complete_source_binding_refuse() {
        for name in [
            "model-00001-of-00002.safetensors",
            "model-00000-of-00002.gguf",
        ] {
            let tree = serde_json::json!([
                file(name, 1),
                file("config.json", 2),
                file("tokenizer.json", 2)
            ])
            .to_string();
            let script = Script::new(vec![revision(), (tree, None)]);
            assert!(
                resolve(
                    &script,
                    &request(&[name]),
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
            assert_eq!(script.requests.lock().unwrap().len(), 2);
        }
    }

    #[test]
    fn subfolder_split_gguf_has_one_quant_and_trusted_lfs_content_hash() {
        let digest = "a".repeat(64);
        let tree=serde_json::json!([
            {"type":"file","path":"q/model-q4-00001-of-00002.gguf","size":9,"lfs":{"oid":digest,"size":9}},
            file("q/model-q4-00002-of-00002.gguf",10),file("q/config.json",2),file("q/tokenizer.json",2),file("other.gguf",9)
        ]).to_string();
        let script = Script::new(vec![revision(), (tree, None)]);
        let mut request = request(&["model-q4-00001-of-00002.gguf"]);
        request.selection = Selection::Files {
            paths: vec!["model-q4-00001-of-00002.gguf".into()],
            subfolder: Some("q".into()),
        };
        let plan = resolve(
            &script,
            &request,
            &PullLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(plan.files().len(), 4);
        assert_eq!(
            plan.files()
                .iter()
                .find(|f| f.path().ends_with("00001-of-00002.gguf"))
                .unwrap()
                .sha256(),
            Some(digest.as_str())
        );
    }

    #[test]
    fn ambiguity_missing_closure_and_changed_pin_refuse() {
        for (tree, paths) in [
            (
                serde_json::json!([file("q4.gguf", 1), file("q8.gguf", 1)]),
                vec!["*.gguf"],
            ),
            (serde_json::json!([file("one.gguf", 1)]), vec!["one.gguf"]),
            (
                serde_json::json!([
                    file("a-00001-of-00002.gguf", 1),
                    file("config.json", 2),
                    file("tokenizer.json", 2)
                ]),
                vec!["a-00001-of-00002.gguf"],
            ),
            (
                serde_json::json!([{"type":"file","path":"config.json"}]),
                vec!["config.json"],
            ),
        ] {
            let script = Script::new(vec![revision(), (tree.to_string(), None)]);
            assert!(
                resolve(
                    &script,
                    &request(&paths),
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
        }
        let script = Script::new(vec![revision()]);
        let mut changed = request(&["config.json"]);
        changed.revision = "f".repeat(40);
        assert!(
            resolve(
                &script,
                &changed,
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(script.requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn pagination_is_pinned_bounded_and_deduplicated() {
        let page = serde_json::json!([file("config.json", 2)]).to_string();
        for next in [
            format!("https://other.invalid/api/models/owner/model/tree/{PIN}?cursor=1"),
            "https://huggingface.co/api/models/owner/model/tree/main?cursor=1".into(),
        ] {
            let script = Script::new(vec![
                revision(),
                (page.clone(), Some(format!("<{next}>; rel=\"next\""))),
            ]);
            assert!(
                resolve(
                    &script,
                    &request(&["config.json"]),
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
        }
        let next = format!(
            "<https://huggingface.co/api/models/owner/model/tree/{PIN}?cursor=1>; rel=\"next\""
        );
        let script = Script::new(vec![
            revision(),
            (page.clone(), Some(next.clone())),
            (page.clone(), None),
        ]);
        assert!(
            resolve(
                &script,
                &request(&["config.json"]),
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        let script = Script::new(vec![revision(), (page, Some(next))]);
        let limits = PullLimits {
            max_pages: 1,
            ..PullLimits::default()
        };
        assert!(
            resolve(
                &script,
                &request(&["config.json"]),
                &limits,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(glob_match("q/*.gguf", "q/a.gguf", &mut 100).unwrap());
        assert!(!glob_match("*.gguf", "q/a.gguf", &mut 100).unwrap());
        assert!(glob_match("*aaaaaaaaab", &"a".repeat(4096), &mut 100).is_err());
    }

    #[test]
    #[cfg(feature = "diagnostic-acquisition")]
    fn actual_ureq_hf_revision_tree_and_index_use_one_pin() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::time::{Duration, Instant};
        let index = r#"{"metadata":{"total_size":1},"weight_map":{"a":"model.safetensors"}}"#;
        let tree = serde_json::json!([
            file("model.safetensors.index.json", index.len() as u64),
            file("model.safetensors", 3),
            file("config.json", 2),
            file("tokenizer.json", 2)
        ])
        .to_string();
        let replies = [revision().0, tree, index.to_string()];
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for body in replies {
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error)
                            if error.kind() == std::io::ErrorKind::WouldBlock
                                && Instant::now() < deadline =>
                        {
                            std::thread::sleep(Duration::from_millis(2))
                        }
                        Err(error) => panic!("HF fixture accept: {error}"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut request = Vec::new();
                let mut byte = [0_u8; 1];
                while !request.ends_with(b"\r\n\r\n") {
                    assert!(request.len() < 16384);
                    assert_eq!(stream.read(&mut byte).unwrap(), 1);
                    request.push(byte[0]);
                }
                requests.push(String::from_utf8(request).unwrap());
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            }
            requests
        });
        let client = super::super::http::HttpsClient::diagnostic_loopback(
            &origin,
            None,
            super::super::http::HttpOptions::default(),
        )
        .unwrap();
        let plan = diagnostic_resolve(
            &client,
            &request(&["model.safetensors.index.json"]),
            &PullLimits::default(),
            &AtomicBool::new(false),
            &origin,
        )
        .unwrap();
        assert_eq!(plan.files().len(), 4);
        let requests = worker.join().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(requests[0].contains("/revision/main"));
        assert!(requests[1].contains(&format!("/tree/{PIN}?")));
        assert!(requests[2].contains(&format!("/resolve/{PIN}/model.safetensors.index.json")));
    }
}
