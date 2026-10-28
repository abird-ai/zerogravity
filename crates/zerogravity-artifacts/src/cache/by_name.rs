//! Durable human-facing names for immutable cache snapshots.
//!
//! Canonical snapshots and refs remain the only runtime/cache authority. This
//! module owns a separate bounded human metadata record plus a disposable flat
//! lowercase symlink projection. Runtime selection never reads either one.

use super::*;
use rustix::fs::{self, AtFlags, FileType, Mode, OFlags, RenameFlags};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::fs::File;
use std::os::unix::fs::MetadataExt;

const STATE_SCHEMA: &str = "zerogravity.cache.by-name.v1";
const STATE_PATH: &str = "by-name.json";
const STATE_NEXT: &str = ".by-name.json.next";
const LIVE_DIR: &str = "by-name";
const NEXT_DIR: &str = ".by-name.next";
const MAX_BY_NAME_ENTRIES: usize = 8192;
const MAX_ALIAS_BYTES: usize = 240;
const MAX_ALIAS_INPUT_BYTES: usize = 4096;
const MAX_STATE_BYTES: usize = 4 * 1024 * 1024;
const MAX_AUX_NODES: usize = 131_072;
const MAX_AUX_DEPTH: usize = 64;
const AUTO_REPOSITORY_BYTES: usize = 176;
const AUTO_REVISION_BYTES: usize = 24;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ByNameState {
    schema: String,
    automatic: BTreeMap<String, String>,
    aliases: BTreeMap<String, String>,
}

impl ByNameState {
    fn empty() -> Self {
        Self {
            schema: STATE_SCHEMA.into(),
            automatic: BTreeMap::new(),
            aliases: BTreeMap::new(),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.schema != STATE_SCHEMA {
            return Err(Error::Unsupported(
                "unsupported by-name metadata schema".into(),
            ));
        }
        if self
            .automatic
            .len()
            .checked_add(self.aliases.len())
            .is_none_or(|count| count > MAX_BY_NAME_ENTRIES)
        {
            return Err(Error::Limit("by-name metadata entry limit".into()));
        }
        let mut automatic_targets = BTreeSet::new();
        for (name, snapshot) in &self.automatic {
            validate_name(name)?;
            validate_digest(snapshot)?;
            if !automatic_targets.insert(snapshot.as_str()) {
                return Err(Error::Malformed(
                    "one snapshot has multiple automatic by-name claims".into(),
                ));
            }
        }
        for (name, snapshot) in &self.aliases {
            validate_name(name)?;
            validate_digest(snapshot)?;
            if self.automatic.contains_key(name) {
                return Err(Error::Malformed(
                    "automatic and explicit by-name entries collide".into(),
                ));
            }
        }
        Ok(())
    }

    fn total_entries(&self) -> Result<usize> {
        self.automatic
            .len()
            .checked_add(self.aliases.len())
            .filter(|count| *count <= MAX_BY_NAME_ENTRIES)
            .ok_or_else(|| Error::Limit("by-name metadata entry limit".into()))
    }

    fn name_taken(&self, name: &str) -> bool {
        self.automatic.contains_key(name) || self.aliases.contains_key(name)
    }

    fn automatic_for_snapshot(&self, snapshot: &str) -> Option<&str> {
        self.automatic
            .iter()
            .find_map(|(name, digest)| (digest == snapshot).then_some(name.as_str()))
    }
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
    out
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > MAX_ALIAS_BYTES
        || name.starts_with(['-', '.'])
        || name.ends_with(['-', '.'])
        || !name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'.')
        })
    {
        return Err(Error::Malformed(
            "invalid lowercase flat by-name alias".into(),
        ));
    }
    Ok(())
}

fn normalize_alias(value: &str) -> Result<String> {
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

fn auto_name(source: &ManifestSource) -> String {
    let provider = provider_slug(source.provider());
    let revision = revision_slug(source.revision());
    let reserved = provider
        .len()
        .saturating_add(revision.len())
        .saturating_add(2);
    let repository_limit = MAX_ALIAS_BYTES
        .saturating_sub(reserved)
        .clamp(1, AUTO_REPOSITORY_BYTES);
    let repository = repository_slug(source.repository(), repository_limit);
    let name = format!("{repository}-{provider}-{revision}");
    let name = slug(&name, MAX_ALIAS_BYTES);
    debug_assert!(validate_name(&name).is_ok());
    name
}

fn target(snapshot: &str) -> Result<String> {
    validate_digest(snapshot)?;
    Ok(format!("../snapshots/{snapshot}"))
}

fn auto_collision_key(source: &ManifestSource, snapshot: &str) -> String {
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

fn claim_automatic(
    state: &mut ByNameState,
    source: &ManifestSource,
    snapshot: &str,
) -> Result<(String, bool)> {
    validate_digest(snapshot)?;
    if let Some(name) = state.automatic_for_snapshot(snapshot) {
        return Ok((name.to_owned(), false));
    }
    state.total_entries()?;
    if state.automatic.len() + state.aliases.len() == MAX_BY_NAME_ENTRIES {
        return Err(Error::Limit("by-name metadata entry limit".into()));
    }

    let base = auto_name(source);
    if !state.name_taken(&base) {
        state.automatic.insert(base.clone(), snapshot.into());
        return Ok((base, true));
    }

    let hash = auto_collision_key(source, snapshot);
    for width in [8usize, 12, 16, 24, 32, 48, 64] {
        let candidate = with_suffix(&base, &hash[..width]);
        if !state.name_taken(&candidate) {
            state.automatic.insert(candidate.clone(), snapshot.into());
            return Ok((candidate, true));
        }
    }
    Err(Error::State(
        "unable to allocate a stable automatic by-name alias".into(),
    ))
}

fn entries(state: &ByNameState) -> Vec<ByNameEntry> {
    let mut result = Vec::with_capacity(state.automatic.len() + state.aliases.len());
    result.extend(state.automatic.iter().map(|(name, snapshot)| ByNameEntry {
        name: name.clone(),
        snapshot: snapshot.clone(),
        kind: ByNameKind::Automatic,
    }));
    result.extend(state.aliases.iter().map(|(name, snapshot)| ByNameEntry {
        name: name.clone(),
        snapshot: snapshot.clone(),
        kind: ByNameKind::Alias,
    }));
    result.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    result
}

fn desired_projection(state: &ByNameState) -> Result<BTreeMap<String, String>> {
    state.validate()?;
    let mut desired = BTreeMap::new();
    for (name, snapshot) in state.automatic.iter().chain(state.aliases.iter()) {
        if desired.insert(name.clone(), target(snapshot)?).is_some() {
            return Err(Error::Malformed(
                "duplicate by-name projection alias".into(),
            ));
        }
    }
    Ok(desired)
}

fn root_copy(store: &CacheStore) -> Result<File> {
    let root = File::from(
        fs::openat(
            store.inner.root.as_fd(),
            ".",
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io)?,
    );
    let metadata = root.metadata()?;
    if (metadata.dev(), metadata.ino()) != store.inner.root.identity() {
        return Err(Error::State(
            "by-name root capability identity changed".into(),
        ));
    }
    Ok(root)
}

fn remove_entry_at(parent: &File, name: &CStr, nodes: &mut usize, depth: usize) -> Result<()> {
    if depth > MAX_AUX_DEPTH {
        return Err(Error::Limit("by-name cleanup depth limit".into()));
    }
    *nodes = nodes
        .checked_add(1)
        .filter(|count| *count <= MAX_AUX_NODES)
        .ok_or_else(|| Error::Limit("by-name cleanup entry limit".into()))?;

    let before = match fs::statat(parent, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(rustix::io::Errno::NOENT) => return Ok(()),
        Err(error) => return Err(io(error)),
    };
    if FileType::from_raw_mode(before.st_mode) != FileType::Directory {
        fs::unlinkat(parent, name, AtFlags::empty()).map_err(io)?;
        return Ok(());
    }

    let directory = File::from(
        fs::openat(
            parent,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io)?,
    );
    let metadata = directory.metadata()?;
    if metadata.dev() != before.st_dev || metadata.ino() != before.st_ino {
        return Err(Error::State(
            "by-name cleanup directory changed during inspection".into(),
        ));
    }
    for entry in fs::Dir::read_from(&directory).map_err(io)? {
        let entry = entry.map_err(io)?;
        let child = entry.file_name().to_owned();
        if child.as_bytes() == b"." || child.as_bytes() == b".." {
            continue;
        }
        remove_entry_at(&directory, &child, nodes, depth + 1)?;
    }
    let current = fs::statat(parent, name, AtFlags::SYMLINK_NOFOLLOW).map_err(io)?;
    if current.st_dev != before.st_dev || current.st_ino != before.st_ino {
        return Err(Error::State(
            "by-name cleanup directory binding changed".into(),
        ));
    }
    fs::unlinkat(parent, name, AtFlags::REMOVEDIR).map_err(io)
}

fn remove_auxiliary(store: &CacheStore, name: &str) -> Result<()> {
    if !crate::cache::paths::is_by_name_auxiliary(name) {
        return Err(Error::State(
            "refusing to remove a non by-name auxiliary entry".into(),
        ));
    }
    let root = root_copy(store)?;
    let name = CString::new(name)
        .map_err(|_| Error::Malformed("invalid by-name auxiliary path".into()))?;
    let mut nodes = 0;
    remove_entry_at(&root, &name, &mut nodes, 0)
}

fn install_symlink(directory: &File, name: &str, link_target: &str) -> Result<()> {
    validate_name(name)?;
    let token = digest(format!("{name}|{link_target}").as_bytes());
    let temporary = format!(".next-{}", &token[..24]);
    match fs::statat(directory, temporary.as_str(), AtFlags::SYMLINK_NOFOLLOW) {
        Ok(_) => {
            fs::unlinkat(directory, temporary.as_str(), AtFlags::empty()).map_err(io)?;
        }
        Err(rustix::io::Errno::NOENT) => {}
        Err(error) => return Err(io(error)),
    }
    fs::symlinkat(link_target, directory, temporary.as_str()).map_err(io)?;

    match fs::statat(directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Err(rustix::io::Errno::NOENT) => {
            fs::renameat_with(
                directory,
                temporary.as_str(),
                directory,
                name,
                RenameFlags::NOREPLACE,
            )
            .map_err(io)?;
        }
        Ok(stat) if FileType::from_raw_mode(stat.st_mode) == FileType::Symlink => {
            fs::renameat_with(
                directory,
                temporary.as_str(),
                directory,
                name,
                RenameFlags::empty(),
            )
            .map_err(io)?;
        }
        Ok(_) => {
            fs::unlinkat(directory, temporary.as_str(), AtFlags::empty()).map_err(io)?;
            return Err(Error::State(
                "by-name alias is occupied by a non-symlink entry".into(),
            ));
        }
        Err(error) => {
            let _ = fs::unlinkat(directory, temporary.as_str(), AtFlags::empty());
            return Err(io(error));
        }
    }
    directory.sync_all()?;
    Ok(())
}

impl CacheStore {
    fn read_by_name_state_locked(&self) -> Result<Option<ByNameState>> {
        let bytes = match self.read(STATE_PATH, MAX_STATE_BYTES) {
            Ok(bytes) => bytes,
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let state: ByNameState = serde_json::from_slice(&bytes)
            .map_err(|_| Error::Malformed("invalid by-name metadata".into()))?;
        state.validate()?;
        Ok(Some(state))
    }

    fn write_by_name_state_locked(&self, state: &ByNameState) -> Result<()> {
        state.validate()?;
        let mut bytes = serde_json::to_vec(state)
            .map_err(|error| Error::Malformed(format!("by-name metadata encoding: {error}")))?;
        bytes.push(b'\n');
        if bytes.len() > MAX_STATE_BYTES {
            return Err(Error::Limit("by-name metadata byte limit".into()));
        }

        remove_auxiliary(self, STATE_NEXT)?;
        let mut file = self.inner.root.open_file(
            STATE_NEXT,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        )?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        self.rename(STATE_NEXT, STATE_PATH, true)?;
        self.sync_dir("")
    }

    fn snapshot_source_locked(&self, snapshot: &str) -> Result<ManifestSource> {
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
        Ok(manifest.source().clone())
    }

    fn snapshot_sources_locked(&self) -> Result<Vec<(String, ManifestSource)>> {
        let snapshots = self.inner.root.open_dir("snapshots", false)?;
        let mut result = Vec::new();
        for entry in fs::Dir::read_from(&snapshots).map_err(io)? {
            let entry = entry.map_err(io)?;
            let name = entry
                .file_name()
                .to_str()
                .map_err(|_| Error::Malformed("non-UTF8 snapshot identity".into()))?;
            if name == "." || name == ".." {
                continue;
            }
            validate_digest(name)?;
            let stat = fs::statat(&snapshots, name, AtFlags::SYMLINK_NOFOLLOW).map_err(io)?;
            if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
                return Err(Error::Malformed(
                    "snapshot namespace contains a non-directory entry".into(),
                ));
            }
            if result.len() == MAX_BY_NAME_ENTRIES {
                return Err(Error::Limit(
                    "automatic by-name snapshot count exceeds bound".into(),
                ));
            }
            result.push((name.to_owned(), self.snapshot_source_locked(name)?));
        }
        result.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(result)
    }

    fn reconcile_by_name_state_locked(&self, state: &mut ByNameState) -> Result<bool> {
        state.validate()?;
        let snapshots = self.snapshot_sources_locked()?;
        let present = snapshots
            .iter()
            .map(|(snapshot, _)| snapshot.as_str())
            .collect::<BTreeSet<_>>();
        let before_automatic = state.automatic.len();
        let before_aliases = state.aliases.len();
        state
            .automatic
            .retain(|_, snapshot| present.contains(snapshot.as_str()));
        state
            .aliases
            .retain(|_, snapshot| present.contains(snapshot.as_str()));
        let mut changed =
            before_automatic != state.automatic.len() || before_aliases != state.aliases.len();

        for (snapshot, source) in snapshots {
            let (_, inserted) = claim_automatic(state, &source, &snapshot)?;
            changed |= inserted;
        }
        state.validate()?;
        Ok(changed)
    }

    fn load_or_initialize_by_name_state_locked(&self) -> Result<(ByNameState, bool)> {
        if let Some(state) = self.read_by_name_state_locked()? {
            return Ok((state, false));
        }
        let mut state = ByNameState::empty();
        self.reconcile_by_name_state_locked(&mut state)?;
        self.write_by_name_state_locked(&state)?;
        Ok((state, true))
    }

    fn rebuild_by_name_projection_locked(&self, state: &ByNameState) -> Result<()> {
        let desired = desired_projection(state)?;
        remove_auxiliary(self, NEXT_DIR)?;
        let next = self.inner.root.open_dir(NEXT_DIR, true)?;
        for (name, link_target) in &desired {
            fs::symlinkat(link_target.as_str(), &next, name.as_str()).map_err(io)?;
        }
        next.sync_all()?;
        drop(next);

        let root = root_copy(self)?;
        match fs::statat(&root, LIVE_DIR, AtFlags::SYMLINK_NOFOLLOW) {
            Err(rustix::io::Errno::NOENT) => {
                fs::renameat_with(&root, NEXT_DIR, &root, LIVE_DIR, RenameFlags::NOREPLACE)
                    .map_err(io)?;
                root.sync_all()?;
                Ok(())
            }
            Ok(_) => {
                fs::renameat_with(&root, NEXT_DIR, &root, LIVE_DIR, RenameFlags::EXCHANGE)
                    .map_err(io)?;
                root.sync_all()?;
                remove_auxiliary(self, NEXT_DIR)?;
                root.sync_all()?;
                Ok(())
            }
            Err(error) => Err(io(error)),
        }
    }

    fn install_by_name_alias_locked(&self, name: &str, snapshot: &str) -> Result<()> {
        // Missing/non-directory projection means it was externally damaged.
        // Do not silently recreate a partial directory containing only this
        // snapshot; caller falls back to the complete bounded state projection.
        let directory = self.inner.root.open_dir(LIVE_DIR, false)?;
        install_symlink(&directory, name, &target(snapshot)?)
    }

    pub(crate) fn publish_by_name_locked(
        &self,
        source: &ManifestSource,
        snapshot: &str,
    ) -> Result<String> {
        let (mut state, initialized) = self.load_or_initialize_by_name_state_locked()?;
        let (name, changed) = claim_automatic(&mut state, source, snapshot)?;
        if changed {
            self.write_by_name_state_locked(&state)?;
        }
        if initialized {
            self.rebuild_by_name_projection_locked(&state)?;
        } else if self.install_by_name_alias_locked(&name, snapshot).is_err() {
            // Projection loss/corruption is human-layer damage only. Recover
            // from bounded persisted state without scanning canonical snapshots.
            self.rebuild_by_name_projection_locked(&state)?;
        }
        Ok(name)
    }

    pub(crate) fn remove_snapshot_by_name_locked(&self, snapshot: &str) -> Result<()> {
        let (mut state, _) = self.load_or_initialize_by_name_state_locked()?;
        let before = state.automatic.len() + state.aliases.len();
        state.automatic.retain(|_, digest| digest != snapshot);
        state.aliases.retain(|_, digest| digest != snapshot);
        if before != state.automatic.len() + state.aliases.len() {
            self.write_by_name_state_locked(&state)?;
        }
        self.rebuild_by_name_projection_locked(&state)
    }

    /// List intended human names. This never resolves a model or opens payloads.
    pub fn list_by_name(&self) -> Result<Vec<ByNameEntry>> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        let state = match self.read_by_name_state_locked()? {
            Some(state) => state,
            None => {
                let mut state = ByNameState::empty();
                self.reconcile_by_name_state_locked(&mut state)?;
                state
            }
        };
        Ok(entries(&state))
    }

    /// Create or update an explicit human alias. Automatic aliases cannot be
    /// replaced; use another human name. A projection warning leaves the human
    /// metadata saved and canonical model data untouched.
    pub fn set_by_name_alias(
        &self,
        alias: &str,
        snapshot: &str,
        replace: bool,
    ) -> Result<ByNameUpdate> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        self.snapshot_source_locked(snapshot)?;
        let name = normalize_alias(alias)?;
        let (mut state, _) = self.load_or_initialize_by_name_state_locked()?;

        if state.automatic.contains_key(&name) {
            return Err(Error::State(
                "human alias collides with a stable automatic by-name entry; choose another alias"
                    .into(),
            ));
        }
        let changed = match state.aliases.get(&name) {
            Some(existing) if existing == snapshot => false,
            Some(_) if !replace => {
                return Err(Error::State(
                    "human alias already points to another snapshot; use --replace explicitly"
                        .into(),
                ));
            }
            _ => true,
        };
        if changed {
            state.aliases.insert(name.clone(), snapshot.into());
            state.validate()?;
            self.write_by_name_state_locked(&state)?;
        }
        let projection_warning = self
            .rebuild_by_name_projection_locked(&state)
            .err()
            .map(|error| error.to_string());
        Ok(ByNameUpdate {
            name,
            snapshot: Some(snapshot.into()),
            changed,
            projection_warning,
        })
    }

    /// Remove only an explicit human alias. Stable automatic aliases are
    /// intentionally not mutable through this API.
    pub fn remove_by_name_alias(&self, alias: &str) -> Result<ByNameUpdate> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        let name = normalize_alias(alias)?;
        let (mut state, _) = self.load_or_initialize_by_name_state_locked()?;
        if state.automatic.contains_key(&name) {
            return Err(Error::State(
                "automatic by-name entries are derived and cannot be removed as human aliases"
                    .into(),
            ));
        }
        let changed = state.aliases.remove(&name).is_some();
        if changed {
            self.write_by_name_state_locked(&state)?;
        }
        let projection_warning = self
            .rebuild_by_name_projection_locked(&state)
            .err()
            .map(|error| error.to_string());
        Ok(ByNameUpdate {
            name,
            snapshot: None,
            changed,
            projection_warning,
        })
    }

    /// Reconcile automatic claims with canonical snapshots, prune aliases whose
    /// canonical snapshot no longer exists, then atomically replace the entire
    /// disposable symlink projection. Existing projection contents are never a
    /// source of truth.
    pub fn refresh_by_name(&self) -> Result<Vec<ByNameEntry>> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        let (mut state, initialized) = self.load_or_initialize_by_name_state_locked()?;
        if !initialized && self.reconcile_by_name_state_locked(&mut state)? {
            self.write_by_name_state_locked(&state)?;
        }
        self.rebuild_by_name_projection_locked(&state)?;
        Ok(entries(&state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(provider: &str, repository: &str, revision: &str) -> ManifestSource {
        ManifestSource::new(provider, repository, revision).unwrap()
    }

    #[test]
    fn automatic_names_are_model_first_flat_lowercase_and_readable() {
        assert_eq!(
            auto_name(&source(
                "huggingface",
                "Qwen/Qwen3.5-0.8B",
                "2FC06364715B967F1860AEA9CF38778875588B17",
            )),
            "qwen3.5-0.8b-qwen-hf-2fc06364"
        );
        assert_eq!(
            auto_name(&source("github", "Org/My_Model", "Release/V1")),
            "my-model-org-gh-release-v1"
        );
    }

    #[test]
    fn explicit_aliases_are_lowercase_flat_and_reject_empty_normalization() {
        assert_eq!(
            normalize_alias("My Team/Primary_Model").unwrap(),
            "my-team-primary-model"
        );
        assert!(normalize_alias("////").is_err());
        let value = normalize_alias(&"A".repeat(MAX_ALIAS_INPUT_BYTES)).unwrap();
        assert!(value.len() <= MAX_ALIAS_BYTES);
        assert_eq!(value, value.to_ascii_lowercase());
        assert!(!value.contains('/'));
    }

    #[test]
    fn later_auto_collision_never_retargets_existing_primary_name() {
        let mut state = ByNameState::empty();
        let first = source("local", "owner/my_model", "v1");
        let second = source("local", "owner/my-model", "v1");
        let (first_name, _) = claim_automatic(
            &mut state,
            &first,
            "1111111111111111111111111111111111111111111111111111111111111111",
        )
        .unwrap();
        let (second_name, _) = claim_automatic(
            &mut state,
            &second,
            "2222222222222222222222222222222222222222222222222222222222222222",
        )
        .unwrap();
        assert_eq!(
            state.automatic.get(&first_name).map(String::as_str),
            Some("1111111111111111111111111111111111111111111111111111111111111111")
        );
        assert_ne!(first_name, second_name);
        assert!(second_name.starts_with(&(auto_name(&second) + "-")));
    }

    #[test]
    fn explicit_alias_cannot_steal_automatic_name_in_state() {
        let mut state = ByNameState::empty();
        let source = source("local", "owner/model", "v1");
        let (automatic, _) = claim_automatic(
            &mut state,
            &source,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        .unwrap();
        state.aliases.insert(
            automatic,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
        );
        assert!(state.validate().is_err());
    }
}
