//! Durable human-facing names for immutable cache snapshots.
//!
//! Canonical snapshots and refs remain the only runtime/cache authority. Human
//! naming state is stored as small per-entry claims under .by-name-state/ and
//! projected into a disposable flat lowercase by-name/ symlink directory.
//!
//! Layout:
//! - automatic/<name> -> snapshot digest
//! - aliases/<name> -> snapshot digest
//! - snapshots/<digest> -> automatic name (derived reverse index)
//! - dirty -> recovery intent while state/projection mutation is in flight
//!
//! Healthy automatic publication is constant-size work: reverse lookup,
//! collision probes, one claim/reverse update and one symlink update. Whole
//! state/canonical scans are reserved for migration, explicit refresh, reset or
//! crash recovery. A durable dirty marker makes every interrupted human-layer
//! mutation detectable without making this layer runtime authority.

use super::*;
use crate::cache::paths::{mount_id, require_mount};
use rustix::fs::{self, AtFlags, FileType, Mode, OFlags, RenameFlags};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

const STATE_SCHEMA: &[u8] = b"zerogravity.cache.by-name.v2";
const LEGACY_SCHEMA: &str = "zerogravity.cache.by-name.v1";
const STATE_DIR: &str = ".by-name-state";
const STATE_NEXT: &str = ".by-name-state.next";
const STATE_SCHEMA_PATH: &str = ".by-name-state/schema";
const DIRTY_PATH: &str = ".by-name-state/dirty";
const AUTO_DIR: &str = ".by-name-state/automatic";
const ALIAS_DIR: &str = ".by-name-state/aliases";
const REVERSE_DIR: &str = ".by-name-state/snapshots";
const REVERSE_NEXT: &str = ".by-name-state/snapshots.next";
const INIT_MARKER: &str = ".by-name-initialized";
const LEGACY_STATE: &str = "by-name.json";
const LEGACY_NEXT: &str = ".by-name.json.next";
const LIVE_DIR: &str = "by-name";
const NEXT_DIR: &str = ".by-name.next";

const MAX_AUTOMATIC_ENTRIES: usize = 131_072;
const MAX_ALIAS_ENTRIES: usize = 65_536;
const MAX_STATE_ENTRIES: usize = MAX_AUTOMATIC_ENTRIES + MAX_ALIAS_ENTRIES;
// Crash recovery may observe automatic claims, explicit aliases, both the live
// and staged reverse indexes, plus bounded temporary/control entries.
const MAX_STATE_CLEANUP_NODES: usize =
    (MAX_AUTOMATIC_ENTRIES * 3) + MAX_ALIAS_ENTRIES + (4 * 1024) + 64;
const MAX_DIRTY_BYTES: usize = 96;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClaimKind {
    Automatic,
    Alias,
}

impl ClaimKind {
    const fn directory(self) -> &'static str {
        match self {
            Self::Automatic => AUTO_DIR,
            Self::Alias => ALIAS_DIR,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DirtyMode {
    Projection,
    Reconcile,
    Pending(String),
}

impl DirtyMode {
    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::Projection => b"projection".to_vec(),
            Self::Reconcile => b"reconcile".to_vec(),
            Self::Pending(snapshot) => format!("pending:{snapshot}").into_bytes(),
        }
    }

    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes == b"projection" {
            return Ok(Self::Projection);
        }
        if bytes == b"reconcile" {
            return Ok(Self::Reconcile);
        }
        let Some(snapshot) = bytes.strip_prefix(b"pending:") else {
            return Err(Error::Malformed("invalid by-name dirty marker".into()));
        };
        let snapshot = std::str::from_utf8(snapshot)
            .map_err(|_| Error::Malformed("invalid by-name pending digest".into()))?;
        validate_digest(snapshot)?;
        Ok(Self::Pending(snapshot.into()))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyState {
    schema: String,
    automatic: BTreeMap<String, String>,
    aliases: BTreeMap<String, String>,
}

#[path = "by_name/fs_support.rs"]
mod fs_support;
#[path = "by_name/naming.rs"]
mod naming;

use fs_support::{MAX_AUX_NODES, parent_path, remove_auxiliary, remove_relative_tree, root_copy};
use naming::{AutoCandidates, MAX_ALIAS_BYTES, normalize_alias, target, validate_name};
#[cfg(test)]
use naming::{MAX_ALIAS_INPUT_BYTES, auto_name};

impl CacheMetadata {
    /// Explicit human name -> exact canonical manifest. Never follow the
    /// disposable symlink projection or repair state during cold selection.
    pub(super) fn human_reference(&self, name: &str) -> Result<SnapshotMetadata> {
        self.human_reference_optional(name)?.ok_or_else(|| {
            Error::Unsupported("human model name is not acquired; use model list".into())
        })
    }

    pub(super) fn human_reference_optional(&self, name: &str) -> Result<Option<SnapshotMetadata>> {
        validate_name(name)?;
        let optional = |path: &str, limit| match self.read(path, limit) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        };
        if optional(STATE_SCHEMA_PATH, STATE_SCHEMA.len())?.is_none()
            && optional(INIT_MARKER, 64)?.is_none()
            && optional(LEGACY_STATE, 4 * 1024 * 1024)?.is_none()
            && self.directory_absent(STATE_DIR)?
            && self.directory_absent(STATE_NEXT)?
        {
            return Ok(None);
        }
        let clean = || -> Result<()> {
            if optional(STATE_SCHEMA_PATH, STATE_SCHEMA.len())?.as_deref() != Some(STATE_SCHEMA) {
                return Err(Error::Unsupported(
                    "human name state unavailable; use a canonical reference or snapshot digest"
                        .into(),
                ));
            }
            if optional(DIRTY_PATH, MAX_DIRTY_BYTES)?.is_some() {
                return Err(Error::State(
                    "human name state needs refresh; use an exact snapshot digest".into(),
                ));
            }
            Ok(())
        };
        clean()?;
        let automatic = optional(&format!("{AUTO_DIR}/{name}"), 64)?;
        let alias = optional(&format!("{ALIAS_DIR}/{name}"), 64)?;
        clean()?;
        let bytes = match (automatic, alias) {
            (Some(bytes), None) | (None, Some(bytes)) => bytes,
            (None, None) => return Ok(None),
            (Some(_), Some(_)) => {
                return Err(Error::Malformed(
                    "ambiguous human model name; use an exact digest".into(),
                ));
            }
        };
        let digest = std::str::from_utf8(&bytes)
            .map_err(|_| Error::Malformed("invalid human name claim".into()))?;
        self.snapshot(digest).map(Some)
    }
}

impl CacheStore {
    #[cfg(feature = "diagnostic-cache")]
    fn by_name_point(&self, operation: CacheOperation, moment: CacheMoment) -> Result<()> {
        self.point(operation, moment, None)
    }

    fn read_optional(&self, path: &str, limit: usize) -> Result<Option<Vec<u8>>> {
        match self.read(path, limit) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn sync_human_dir(&self, path: &str) -> Result<()> {
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(
            CacheOperation::ByNameStateDirectorySync,
            CacheMoment::Before,
        )?;
        self.sync_dir(path)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateDirectorySync, CacheMoment::After)?;
        Ok(())
    }

    fn write_human_staged_new(&self, path: &str, bytes: &[u8]) -> Result<()> {
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateFileSync, CacheMoment::Before)?;
        self.write_new(path, bytes)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateFileSync, CacheMoment::After)?;
        Ok(())
    }

    fn write_human_new(&self, path: &str, bytes: &[u8]) -> Result<()> {
        self.write_human_staged_new(path, bytes)?;
        self.sync_human_dir(parent_path(path))
    }

    fn write_human_replace(&self, path: &str, bytes: &[u8]) -> Result<()> {
        let parent = parent_path(path);
        let token = digest(format!("{path}|{}", digest(bytes)).as_bytes());
        let temporary = if parent.is_empty() {
            format!(".next-{}", &token[..24])
        } else {
            format!("{parent}/.next-{}", &token[..24])
        };
        let _ = remove_relative_tree(self, &temporary);
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateFileSync, CacheMoment::Before)?;
        let mut file = self.inner.root.open_file(
            &temporary,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        )?;
        file.write_all(bytes)?;
        file.sync_all()?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateFileSync, CacheMoment::After)?;
        drop(file);
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateInstall, CacheMoment::Before)?;
        self.rename(&temporary, path, true)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateInstall, CacheMoment::After)?;
        self.sync_human_dir(parent)
    }

    fn marker_exists_locked(&self) -> Result<bool> {
        let Some(bytes) = self.read_optional(INIT_MARKER, STATE_SCHEMA.len())? else {
            return Ok(false);
        };
        if bytes != STATE_SCHEMA {
            return Err(Error::Malformed(
                "invalid by-name initialization marker".into(),
            ));
        }
        Ok(true)
    }

    fn create_marker_locked(&self) -> Result<()> {
        if self.marker_exists_locked()? {
            return Ok(());
        }
        self.write_human_staged_new(INIT_MARKER, STATE_SCHEMA)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameRootSync, CacheMoment::Before)?;
        self.sync_dir("")?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameRootSync, CacheMoment::After)?;
        Ok(())
    }

    fn validate_state_layout_locked(&self) -> Result<()> {
        let root_mount = mount_id(self.inner.root.as_fd())?;
        let state = self.inner.root.open_dir(STATE_DIR, false)?;
        require_mount(&state, root_mount)?;
        for path in [AUTO_DIR, ALIAS_DIR, REVERSE_DIR] {
            let directory = self.inner.root.open_dir(path, false)?;
            require_mount(&directory, root_mount)?;
        }
        let schema = self.read(STATE_SCHEMA_PATH, STATE_SCHEMA.len())?;
        if schema != STATE_SCHEMA {
            return Err(Error::Malformed("invalid by-name state schema".into()));
        }
        Ok(())
    }

    fn state_directory_exists_locked(&self) -> Result<bool> {
        let root = root_copy(self)?;
        match fs::statat(&root, STATE_DIR, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => {
                if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
                    return Err(Error::State("by-name state path is not a directory".into()));
                }
                Ok(true)
            }
            Err(rustix::io::Errno::NOENT) => Ok(false),
            Err(error) => Err(io(error)),
        }
    }

    fn read_claim_locked(&self, kind: ClaimKind, name: &str) -> Result<Option<String>> {
        validate_name(name)?;
        let path = format!("{}/{name}", kind.directory());
        let Some(bytes) = self.read_optional(&path, 64)? else {
            return Ok(None);
        };
        let digest = std::str::from_utf8(&bytes)
            .map_err(|_| Error::Malformed("non-UTF8 by-name claim".into()))?;
        validate_digest(digest)?;
        Ok(Some(digest.into()))
    }

    fn read_reverse_locked(&self, snapshot: &str) -> Result<Option<String>> {
        validate_digest(snapshot)?;
        let path = format!("{REVERSE_DIR}/{snapshot}");
        let Some(bytes) = self.read_optional(&path, MAX_ALIAS_BYTES)? else {
            return Ok(None);
        };
        let name = std::str::from_utf8(&bytes)
            .map_err(|_| Error::Malformed("non-UTF8 by-name reverse claim".into()))?;
        validate_name(name)?;
        Ok(Some(name.into()))
    }

    fn write_claim_new_locked(&self, kind: ClaimKind, name: &str, snapshot: &str) -> Result<()> {
        validate_name(name)?;
        validate_digest(snapshot)?;
        self.write_human_new(&format!("{}/{name}", kind.directory()), snapshot.as_bytes())
    }

    fn write_claim_replace_locked(
        &self,
        kind: ClaimKind,
        name: &str,
        snapshot: &str,
    ) -> Result<()> {
        validate_name(name)?;
        validate_digest(snapshot)?;
        self.write_human_replace(&format!("{}/{name}", kind.directory()), snapshot.as_bytes())
    }

    fn write_reverse_new_locked(&self, snapshot: &str, name: &str) -> Result<()> {
        validate_digest(snapshot)?;
        validate_name(name)?;
        self.write_human_new(&format!("{REVERSE_DIR}/{snapshot}"), name.as_bytes())
    }

    fn unlink_human_file_locked(&self, path: &str) -> Result<bool> {
        match self.unlink(path, false) {
            Ok(()) => {
                self.sync_human_dir(parent_path(path))?;
                Ok(true)
            }
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn scan_claims_locked(&self, kind: ClaimKind) -> Result<BTreeMap<String, String>> {
        let root_mount = mount_id(self.inner.root.as_fd())?;
        let directory = self.inner.root.open_dir(kind.directory(), false)?;
        require_mount(&directory, root_mount)?;
        let maximum = match kind {
            ClaimKind::Automatic => MAX_AUTOMATIC_ENTRIES,
            ClaimKind::Alias => MAX_ALIAS_ENTRIES,
        };
        let mut result = BTreeMap::new();
        let mut observed = 0usize;
        for entry in fs::Dir::read_from(&directory).map_err(io)? {
            let entry = entry.map_err(io)?;
            let raw = entry
                .file_name()
                .to_str()
                .map_err(|_| Error::Malformed("non-UTF8 by-name state entry".into()))?;
            if raw == "." || raw == ".." {
                continue;
            }
            observed = observed
                .checked_add(1)
                .filter(|count| *count <= maximum.saturating_add(1024))
                .ok_or_else(|| Error::Limit("by-name state directory entry limit".into()))?;
            if raw.starts_with(".next-") {
                continue;
            }
            if result.len() == maximum {
                return Err(Error::Limit("by-name claim count exceeds bound".into()));
            }
            validate_name(raw)?;
            let snapshot = self
                .read_claim_locked(kind, raw)?
                .ok_or_else(|| Error::State("by-name claim disappeared during scan".into()))?;
            if result.insert(raw.into(), snapshot).is_some() {
                return Err(Error::Malformed("duplicate by-name claim".into()));
            }
        }
        Ok(result)
    }

    fn state_entries_locked(&self) -> Result<Vec<ByNameEntry>> {
        let automatic = self.scan_claims_locked(ClaimKind::Automatic)?;
        let aliases = self.scan_claims_locked(ClaimKind::Alias)?;
        for name in aliases.keys() {
            if automatic.contains_key(name) {
                return Err(Error::Malformed(
                    "automatic and explicit by-name entries collide".into(),
                ));
            }
        }
        let mut targets = BTreeSet::new();
        for snapshot in automatic.values() {
            if !targets.insert(snapshot.as_str()) {
                return Err(Error::Malformed(
                    "one snapshot has multiple automatic by-name claims".into(),
                ));
            }
        }
        let mut result = Vec::with_capacity(automatic.len() + aliases.len());
        result.extend(automatic.into_iter().map(|(name, snapshot)| ByNameEntry {
            name,
            snapshot,
            kind: ByNameKind::Automatic,
        }));
        result.extend(aliases.into_iter().map(|(name, snapshot)| ByNameEntry {
            name,
            snapshot,
            kind: ByNameKind::Alias,
        }));
        result.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        Ok(result)
    }

    fn dirty_mode_locked(&self) -> Result<Option<DirtyMode>> {
        let Some(bytes) = self.read_optional(DIRTY_PATH, MAX_DIRTY_BYTES)? else {
            return Ok(None);
        };
        Ok(Some(DirtyMode::parse(&bytes)?))
    }

    fn mark_dirty_locked(&self, mode: DirtyMode) -> Result<()> {
        if self.dirty_mode_locked()?.is_some() {
            return Err(Error::State(
                "by-name mutation began before prior recovery completed".into(),
            ));
        }
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameDirtyFileSync, CacheMoment::Before)?;
        self.write_new(DIRTY_PATH, &mode.bytes())?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameDirtyFileSync, CacheMoment::After)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(
            CacheOperation::ByNameDirtyDirectorySync,
            CacheMoment::Before,
        )?;
        self.sync_dir(STATE_DIR)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameDirtyDirectorySync, CacheMoment::After)?;
        Ok(())
    }

    fn clear_dirty_locked(&self) -> Result<()> {
        if self.dirty_mode_locked()?.is_none() {
            return Ok(());
        }
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameDirtyRemove, CacheMoment::Before)?;
        self.unlink(DIRTY_PATH, false)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameDirtyRemove, CacheMoment::After)?;
        self.sync_human_dir(STATE_DIR)
    }

    fn snapshot_manifest_optional_locked(&self, snapshot: &str) -> Result<Option<Manifest>> {
        validate_digest(snapshot)?;
        let path = format!("snapshots/{snapshot}/manifest.json");
        let bytes = match self.read(&path, self.inner.options.limits.max_manifest_bytes) {
            Ok(bytes) => bytes,
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let manifest = Manifest::from_json(&bytes, &self.inner.options.limits)?;
        if manifest.digest()? != snapshot {
            return Err(Error::Malformed(
                "by-name snapshot manifest identity mismatch".into(),
            ));
        }
        Ok(Some(manifest))
    }

    fn snapshot_manifests_locked(&self) -> Result<Vec<(String, Manifest)>> {
        let snapshots = self.inner.root.open_dir("snapshots", false)?;
        let mut result: Vec<(String, Manifest)> = Vec::new();
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
            if result.len() == MAX_AUTOMATIC_ENTRIES {
                return Err(Error::Limit(
                    "automatic by-name snapshot count exceeds bound".into(),
                ));
            }
            let manifest = self
                .snapshot_manifest_optional_locked(name)?
                .ok_or_else(|| Error::State("snapshot disappeared during by-name scan".into()))?;
            result.push((name.into(), manifest));
        }
        result.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(result)
    }

    fn legacy_state_locked(&self) -> Result<Option<LegacyState>> {
        let Some(bytes) = self.read_optional(LEGACY_STATE, 4 * 1024 * 1024)? else {
            return Ok(None);
        };
        let legacy: LegacyState = serde_json::from_slice(&bytes)
            .map_err(|_| Error::Malformed("invalid legacy by-name metadata".into()))?;
        if legacy.schema != LEGACY_SCHEMA
            || legacy.automatic.len() > MAX_AUTOMATIC_ENTRIES
            || legacy.aliases.len() > MAX_ALIAS_ENTRIES
        {
            return Err(Error::Malformed(
                "unsupported or oversized legacy by-name metadata".into(),
            ));
        }
        let mut targets = BTreeSet::new();
        for (name, snapshot) in &legacy.automatic {
            validate_name(name)?;
            validate_digest(snapshot)?;
            if !targets.insert(snapshot.as_str()) {
                return Err(Error::Malformed(
                    "legacy by-name automatic target duplicated".into(),
                ));
            }
        }
        for (name, snapshot) in &legacy.aliases {
            validate_name(name)?;
            validate_digest(snapshot)?;
            if legacy.automatic.contains_key(name) {
                return Err(Error::Malformed(
                    "legacy by-name alias collides with automatic name".into(),
                ));
            }
        }
        Ok(Some(legacy))
    }

    fn initialize_state_locked(&self) -> Result<()> {
        if self.marker_exists_locked()? {
            return Err(Error::State(
                "by-name state is missing after prior initialization; refusing to discard human aliases"
                    .into(),
            ));
        }
        remove_auxiliary(self, STATE_NEXT)?;
        let root_mount = mount_id(self.inner.root.as_fd())?;
        let automatic_next = format!("{STATE_NEXT}/automatic");
        let aliases_next = format!("{STATE_NEXT}/aliases");
        let snapshots_next = format!("{STATE_NEXT}/snapshots");
        for path in [
            STATE_NEXT,
            automatic_next.as_str(),
            aliases_next.as_str(),
            snapshots_next.as_str(),
        ] {
            let directory = self.inner.root.open_dir(path, true)?;
            require_mount(&directory, root_mount)?;
        }
        self.write_human_staged_new(&format!("{STATE_NEXT}/schema"), STATE_SCHEMA)?;
        self.write_human_staged_new(&format!("{STATE_NEXT}/dirty"), b"projection")?;

        let snapshots = self.snapshot_manifests_locked()?;
        let present = snapshots
            .iter()
            .map(|(snapshot, _)| snapshot.as_str())
            .collect::<BTreeSet<_>>();
        let legacy = self.legacy_state_locked()?;
        let mut used = BTreeSet::new();
        let mut automatic_targets = BTreeSet::new();

        if let Some(legacy) = &legacy {
            for (name, snapshot) in &legacy.automatic {
                if !present.contains(snapshot.as_str()) {
                    continue;
                }
                if !used.insert(name.clone()) || !automatic_targets.insert(snapshot.clone()) {
                    return Err(Error::Malformed(
                        "legacy by-name state is not one-to-one".into(),
                    ));
                }
                self.write_human_staged_new(
                    &format!("{STATE_NEXT}/automatic/{name}"),
                    snapshot.as_bytes(),
                )?;
                self.write_human_staged_new(
                    &format!("{STATE_NEXT}/snapshots/{snapshot}"),
                    name.as_bytes(),
                )?;
            }
            for (name, snapshot) in &legacy.aliases {
                if !present.contains(snapshot.as_str()) {
                    continue;
                }
                if !used.insert(name.clone()) {
                    return Err(Error::Malformed(
                        "legacy by-name state contains a name collision".into(),
                    ));
                }
                self.write_human_staged_new(
                    &format!("{STATE_NEXT}/aliases/{name}"),
                    snapshot.as_bytes(),
                )?;
            }
        }

        for (snapshot, manifest) in &snapshots {
            if automatic_targets.contains(snapshot) {
                continue;
            }
            let mut chosen = None;
            for candidate in AutoCandidates::new(manifest, snapshot).take(MAX_STATE_ENTRIES + 16) {
                if used.insert(candidate.clone()) {
                    chosen = Some(candidate);
                    break;
                }
            }
            let name = chosen.ok_or_else(|| {
                Error::State("unable to allocate migrated automatic by-name alias".into())
            })?;
            automatic_targets.insert(snapshot.clone());
            self.write_human_staged_new(
                &format!("{STATE_NEXT}/automatic/{name}"),
                snapshot.as_bytes(),
            )?;
            self.write_human_staged_new(
                &format!("{STATE_NEXT}/snapshots/{snapshot}"),
                name.as_bytes(),
            )?;
        }

        for path in [
            automatic_next.as_str(),
            aliases_next.as_str(),
            snapshots_next.as_str(),
            STATE_NEXT,
        ] {
            self.sync_human_dir(path)?;
        }

        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateInstall, CacheMoment::Before)?;
        self.rename(STATE_NEXT, STATE_DIR, false)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateInstall, CacheMoment::After)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameRootSync, CacheMoment::Before)?;
        self.sync_dir("")?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameRootSync, CacheMoment::After)?;

        self.rebuild_projection_locked()?;
        self.clear_dirty_locked()?;
        self.create_marker_locked()?;
        let _ = remove_auxiliary(self, LEGACY_STATE);
        let _ = remove_auxiliary(self, LEGACY_NEXT);
        Ok(())
    }

    fn cleanup_state_temporaries_locked(&self) -> Result<()> {
        remove_relative_tree(self, REVERSE_NEXT)?;
        for directory_path in [AUTO_DIR, ALIAS_DIR, REVERSE_DIR] {
            let directory = self.inner.root.open_dir(directory_path, false)?;
            let mut stale = Vec::new();
            let mut observed = 0usize;
            for entry in fs::Dir::read_from(&directory).map_err(io)? {
                let entry = entry.map_err(io)?;
                let name = entry
                    .file_name()
                    .to_str()
                    .map_err(|_| Error::Malformed("non-UTF8 by-name temporary".into()))?;
                if name == "." || name == ".." {
                    continue;
                }
                observed = observed
                    .checked_add(1)
                    .filter(|count| *count <= MAX_AUX_NODES)
                    .ok_or_else(|| Error::Limit("by-name temporary scan entry limit".into()))?;
                if name.starts_with(".next-") {
                    stale.push(format!("{directory_path}/{name}"));
                }
            }
            for path in stale {
                remove_relative_tree(self, &path)?;
            }
        }
        Ok(())
    }

    fn rebuild_reverse_locked(&self) -> Result<()> {
        let automatic = self.scan_claims_locked(ClaimKind::Automatic)?;
        let mut by_snapshot = BTreeMap::new();
        for (name, snapshot) in &automatic {
            if by_snapshot.insert(snapshot.clone(), name.clone()).is_some() {
                return Err(Error::Malformed(
                    "one snapshot has multiple automatic by-name claims".into(),
                ));
            }
        }

        remove_relative_tree(self, REVERSE_NEXT)?;
        let root_mount = mount_id(self.inner.root.as_fd())?;
        let next = self.inner.root.open_dir(REVERSE_NEXT, true)?;
        require_mount(&next, root_mount)?;
        for (snapshot, name) in &by_snapshot {
            self.write_human_staged_new(&format!("{REVERSE_NEXT}/{snapshot}"), name.as_bytes())?;
        }
        self.sync_human_dir(REVERSE_NEXT)?;
        drop(next);

        let state = self.inner.root.open_dir(STATE_DIR, false)?;
        require_mount(&state, root_mount)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateInstall, CacheMoment::Before)?;
        fs::renameat_with(
            &state,
            "snapshots.next",
            &state,
            "snapshots",
            RenameFlags::EXCHANGE,
        )
        .map_err(io)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameStateInstall, CacheMoment::After)?;
        self.sync_human_dir(STATE_DIR)?;
        remove_relative_tree(self, REVERSE_NEXT)?;
        self.sync_human_dir(STATE_DIR)
    }

    fn rebuild_projection_locked(&self) -> Result<()> {
        let entries = self.state_entries_locked()?;
        if entries.len() > MAX_STATE_ENTRIES {
            return Err(Error::Limit("by-name projection entry limit".into()));
        }
        remove_auxiliary(self, NEXT_DIR)?;
        let root_mount = mount_id(self.inner.root.as_fd())?;
        let next = self.inner.root.open_dir(NEXT_DIR, true)?;
        require_mount(&next, root_mount)?;
        for entry in &entries {
            fs::symlinkat(target(entry.snapshot())?.as_str(), &next, entry.name()).map_err(io)?;
        }
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(
            CacheOperation::ByNameProjectionDirectorySync,
            CacheMoment::Before,
        )?;
        next.sync_all()?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(
            CacheOperation::ByNameProjectionDirectorySync,
            CacheMoment::After,
        )?;
        drop(next);

        let root = root_copy(self)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(
            CacheOperation::ByNameProjectionExchange,
            CacheMoment::Before,
        )?;
        match fs::statat(&root, LIVE_DIR, AtFlags::SYMLINK_NOFOLLOW) {
            Err(rustix::io::Errno::NOENT) => {
                fs::renameat_with(&root, NEXT_DIR, &root, LIVE_DIR, RenameFlags::NOREPLACE)
                    .map_err(io)?;
            }
            Ok(_) => {
                fs::renameat_with(&root, NEXT_DIR, &root, LIVE_DIR, RenameFlags::EXCHANGE)
                    .map_err(io)?;
            }
            Err(error) => return Err(io(error)),
        }
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameProjectionExchange, CacheMoment::After)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameRootSync, CacheMoment::Before)?;
        root.sync_all()?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(CacheOperation::ByNameRootSync, CacheMoment::After)?;

        let _ = remove_auxiliary(self, NEXT_DIR);
        root.sync_all()?;
        Ok(())
    }

    fn install_projection_alias_locked(&self, name: &str, snapshot: &str) -> Result<()> {
        validate_name(name)?;
        let link_target = target(snapshot)?;
        let root_mount = mount_id(self.inner.root.as_fd())?;
        let directory = self.inner.root.open_dir(LIVE_DIR, false)?;
        require_mount(&directory, root_mount)?;

        match fs::statat(&directory, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) if FileType::from_raw_mode(stat.st_mode) == FileType::Symlink => {
                let current = fs::readlinkat(&directory, name, Vec::new()).map_err(io)?;
                if current.to_bytes() == link_target.as_bytes() {
                    return Ok(());
                }
            }
            Ok(_) => {
                return Err(Error::State(
                    "by-name alias is occupied by a non-symlink entry".into(),
                ));
            }
            Err(rustix::io::Errno::NOENT) => {}
            Err(error) => return Err(io(error)),
        }

        let token = digest(format!("{name}|{link_target}").as_bytes());
        let temporary = format!(".next-{}", &token[..24]);
        match fs::statat(&directory, temporary.as_str(), AtFlags::SYMLINK_NOFOLLOW) {
            Ok(_) => {
                fs::unlinkat(&directory, temporary.as_str(), AtFlags::empty()).map_err(io)?;
            }
            Err(rustix::io::Errno::NOENT) => {}
            Err(error) => return Err(io(error)),
        }
        fs::symlinkat(link_target.as_str(), &directory, temporary.as_str()).map_err(io)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(
            CacheOperation::ByNameProjectionEntryInstall,
            CacheMoment::Before,
        )?;
        fs::renameat_with(
            &directory,
            temporary.as_str(),
            &directory,
            name,
            RenameFlags::empty(),
        )
        .map_err(io)?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(
            CacheOperation::ByNameProjectionEntryInstall,
            CacheMoment::After,
        )?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(
            CacheOperation::ByNameProjectionDirectorySync,
            CacheMoment::Before,
        )?;
        directory.sync_all()?;
        #[cfg(feature = "diagnostic-cache")]
        self.by_name_point(
            CacheOperation::ByNameProjectionDirectorySync,
            CacheMoment::After,
        )?;
        Ok(())
    }

    fn ensure_automatic_under_dirty_locked(
        &self,
        manifest: &Manifest,
        snapshot: &str,
    ) -> Result<String> {
        if let Some(name) = self.read_reverse_locked(snapshot)? {
            match self.read_claim_locked(ClaimKind::Automatic, &name)? {
                Some(existing) if existing == snapshot => return Ok(name),
                _ => {
                    return Err(Error::State(
                        "by-name automatic reverse index is inconsistent".into(),
                    ));
                }
            }
        }

        for name in AutoCandidates::new(manifest, snapshot).take(MAX_STATE_ENTRIES + 16) {
            let automatic = self.read_claim_locked(ClaimKind::Automatic, &name)?;
            let alias = self.read_claim_locked(ClaimKind::Alias, &name)?;
            match (automatic, alias) {
                (Some(existing), None) if existing == snapshot => {
                    self.write_reverse_new_locked(snapshot, &name)?;
                    return Ok(name);
                }
                (None, None) => {
                    self.write_claim_new_locked(ClaimKind::Automatic, &name, snapshot)?;
                    self.write_reverse_new_locked(snapshot, &name)?;
                    return Ok(name);
                }
                _ => {}
            }
        }
        Err(Error::State(
            "unable to allocate a stable automatic by-name alias".into(),
        ))
    }

    fn remove_claims_for_snapshot_under_dirty_locked(&self, snapshot: &str) -> Result<bool> {
        let mut changed = false;
        if let Some(name) = self.read_reverse_locked(snapshot)? {
            changed |= self.unlink_human_file_locked(&format!("{AUTO_DIR}/{name}"))?;
            changed |= self.unlink_human_file_locked(&format!("{REVERSE_DIR}/{snapshot}"))?;
        } else {
            // An interrupted automatic claim can exist before its reverse index.
            // This scan is recovery-only; the healthy path is the lookup above.
            let automatic = self.scan_claims_locked(ClaimKind::Automatic)?;
            for (name, target) in automatic {
                if target == snapshot {
                    changed |= self.unlink_human_file_locked(&format!("{AUTO_DIR}/{name}"))?;
                }
            }
        }

        let aliases = self.scan_claims_locked(ClaimKind::Alias)?;
        for (name, target) in aliases {
            if target == snapshot {
                changed |= self.unlink_human_file_locked(&format!("{ALIAS_DIR}/{name}"))?;
            }
        }
        Ok(changed)
    }

    fn reconcile_claims_with_canonical_under_dirty_locked(&self) -> Result<()> {
        let snapshots = self.snapshot_manifests_locked()?;
        let present = snapshots
            .iter()
            .map(|(snapshot, _)| snapshot.as_str())
            .collect::<BTreeSet<_>>();

        for kind in [ClaimKind::Automatic, ClaimKind::Alias] {
            let claims = self.scan_claims_locked(kind)?;
            for (name, snapshot) in claims {
                if !present.contains(snapshot.as_str()) {
                    self.unlink_human_file_locked(&format!("{}/{name}", kind.directory()))?;
                }
            }
        }

        self.rebuild_reverse_locked()?;
        for (snapshot, manifest) in snapshots {
            self.ensure_automatic_under_dirty_locked(&manifest, &snapshot)?;
        }
        Ok(())
    }

    fn recover_dirty_locked(&self, mode: DirtyMode) -> Result<()> {
        match mode {
            DirtyMode::Projection => {
                self.cleanup_state_temporaries_locked()?;
                self.rebuild_reverse_locked()?;
                self.rebuild_projection_locked()?;
            }
            DirtyMode::Reconcile => {
                self.cleanup_state_temporaries_locked()?;
                self.reconcile_claims_with_canonical_under_dirty_locked()?;
                self.rebuild_projection_locked()?;
            }
            DirtyMode::Pending(snapshot) => {
                if let Some(manifest) = self.snapshot_manifest_optional_locked(&snapshot)? {
                    let name = self.ensure_automatic_under_dirty_locked(&manifest, &snapshot)?;
                    if self
                        .install_projection_alias_locked(&name, &snapshot)
                        .is_err()
                    {
                        self.rebuild_projection_locked()?;
                    }
                } else {
                    self.remove_claims_for_snapshot_under_dirty_locked(&snapshot)?;
                    self.rebuild_reverse_locked()?;
                    self.rebuild_projection_locked()?;
                }
            }
        }
        self.clear_dirty_locked()
    }

    fn ensure_state_locked(&self) -> Result<()> {
        if !self.state_directory_exists_locked()? {
            if self.marker_exists_locked()? {
                return Err(Error::State(
                    "by-name state is missing after prior initialization; refusing to discard human aliases"
                        .into(),
                ));
            }
            return self.initialize_state_locked();
        }

        self.validate_state_layout_locked()?;
        let had_marker = self.marker_exists_locked()?;
        if let Some(mode) = self.dirty_mode_locked()? {
            self.recover_dirty_locked(mode)?;
        }
        if !had_marker {
            // Existing v2 state without the durable initialization marker can
            // only be an interrupted adoption or foreign human-layer state.
            // Reconcile it against canonical snapshots before blessing it.
            self.mark_dirty_locked(DirtyMode::Reconcile)?;
            self.recover_dirty_locked(DirtyMode::Reconcile)?;
        }
        self.create_marker_locked()?;
        let _ = remove_auxiliary(self, STATE_NEXT);
        let _ = remove_auxiliary(self, LEGACY_STATE);
        let _ = remove_auxiliary(self, LEGACY_NEXT);
        Ok(())
    }

    pub(crate) fn prepare_by_name_reconcile_locked(&self) -> Result<()> {
        self.ensure_state_locked()?;
        self.mark_dirty_locked(DirtyMode::Reconcile)
    }

    pub(crate) fn prepare_by_name_pending_locked(&self, snapshot: &str) -> Result<()> {
        validate_digest(snapshot)?;
        self.ensure_state_locked()?;
        if self.read_reverse_locked(snapshot)?.is_some() {
            return Ok(());
        }
        self.mark_dirty_locked(DirtyMode::Pending(snapshot.into()))
    }

    pub(crate) fn publish_by_name_locked(
        &self,
        manifest: &Manifest,
        snapshot: &str,
    ) -> Result<String> {
        validate_digest(snapshot)?;
        self.ensure_state_locked()?;
        if let Some(name) = self.read_reverse_locked(snapshot)? {
            let claim = self
                .read_claim_locked(ClaimKind::Automatic, &name)?
                .ok_or_else(|| {
                    Error::State("by-name automatic reverse index has no claim".into())
                })?;
            if claim != snapshot {
                return Err(Error::State(
                    "by-name automatic reverse index points to another snapshot".into(),
                ));
            }
            if self
                .install_projection_alias_locked(&name, snapshot)
                .is_err()
            {
                self.mark_dirty_locked(DirtyMode::Projection)?;
                let result = self.rebuild_projection_locked();
                if result.is_ok() {
                    self.clear_dirty_locked()?;
                }
                result?;
            }
            return Ok(name);
        }

        self.mark_dirty_locked(DirtyMode::Projection)?;
        let name = self.ensure_automatic_under_dirty_locked(manifest, snapshot)?;
        let projection = self
            .install_projection_alias_locked(&name, snapshot)
            .or_else(|_| self.rebuild_projection_locked());
        if projection.is_ok() {
            self.clear_dirty_locked()?;
        }
        projection?;
        Ok(name)
    }

    pub(crate) fn remove_snapshot_by_name_locked(&self, snapshot: &str) -> Result<()> {
        validate_digest(snapshot)?;
        self.ensure_state_locked()?;
        let has_automatic = self.read_reverse_locked(snapshot)?.is_some();
        let has_alias = self
            .scan_claims_locked(ClaimKind::Alias)?
            .values()
            .any(|target| target == snapshot);
        if !has_automatic && !has_alias {
            return Ok(());
        }
        self.mark_dirty_locked(DirtyMode::Reconcile)?;
        self.remove_claims_for_snapshot_under_dirty_locked(snapshot)?;
        let result = self.rebuild_projection_locked();
        if result.is_ok() {
            self.clear_dirty_locked()?;
        }
        result
    }

    pub fn list_by_name(&self) -> Result<Vec<ByNameEntry>> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        self.ensure_state_locked()?;
        self.state_entries_locked()
    }

    pub fn set_by_name_alias(
        &self,
        alias: &str,
        snapshot: &str,
        replace: bool,
    ) -> Result<ByNameUpdate> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        self.ensure_state_locked()?;
        self.snapshot_manifest_optional_locked(snapshot)?
            .ok_or_else(|| Error::State("human alias target snapshot is not canonical".into()))?;
        let name = normalize_alias(alias)?;

        if self
            .read_claim_locked(ClaimKind::Automatic, &name)?
            .is_some()
        {
            return Err(Error::State(
                "human alias collides with a stable automatic by-name entry; choose another alias"
                    .into(),
            ));
        }

        let existing = self.read_claim_locked(ClaimKind::Alias, &name)?;
        let changed = match existing.as_deref() {
            Some(current) if current == snapshot => false,
            Some(_) if !replace => {
                return Err(Error::State(
                    "human alias already points to another snapshot; use --replace explicitly"
                        .into(),
                ));
            }
            _ => true,
        };

        if !changed {
            let warning = match self.install_projection_alias_locked(&name, snapshot) {
                Ok(()) => None,
                Err(_) => {
                    self.mark_dirty_locked(DirtyMode::Projection)?;
                    match self.rebuild_projection_locked() {
                        Ok(()) => {
                            self.clear_dirty_locked()?;
                            None
                        }
                        Err(error) => Some(error.to_string()),
                    }
                }
            };
            return Ok(ByNameUpdate {
                name,
                snapshot: Some(snapshot.into()),
                changed: false,
                projection_warning: warning,
            });
        }

        if existing.is_none()
            && self.scan_claims_locked(ClaimKind::Alias)?.len() == MAX_ALIAS_ENTRIES
        {
            return Err(Error::Limit(
                "explicit by-name alias count exceeds bound".into(),
            ));
        }

        self.mark_dirty_locked(DirtyMode::Projection)?;
        if existing.is_some() {
            self.write_claim_replace_locked(ClaimKind::Alias, &name, snapshot)?;
        } else {
            self.write_claim_new_locked(ClaimKind::Alias, &name, snapshot)?;
        }

        let projection = self
            .install_projection_alias_locked(&name, snapshot)
            .or_else(|_| self.rebuild_projection_locked());
        let warning = match projection {
            Ok(()) => {
                self.clear_dirty_locked()?;
                None
            }
            Err(error) => Some(error.to_string()),
        };
        Ok(ByNameUpdate {
            name,
            snapshot: Some(snapshot.into()),
            changed: true,
            projection_warning: warning,
        })
    }

    pub fn remove_by_name_alias(&self, alias: &str) -> Result<ByNameUpdate> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        self.ensure_state_locked()?;
        let name = normalize_alias(alias)?;
        if self
            .read_claim_locked(ClaimKind::Automatic, &name)?
            .is_some()
        {
            return Err(Error::State(
                "automatic by-name entries are derived and cannot be removed as human aliases"
                    .into(),
            ));
        }
        if self.read_claim_locked(ClaimKind::Alias, &name)?.is_none() {
            return Ok(ByNameUpdate {
                name,
                snapshot: None,
                changed: false,
                projection_warning: None,
            });
        }

        self.mark_dirty_locked(DirtyMode::Projection)?;
        self.unlink_human_file_locked(&format!("{ALIAS_DIR}/{name}"))?;
        let projection = self.rebuild_projection_locked();
        let warning = match projection {
            Ok(()) => {
                self.clear_dirty_locked()?;
                None
            }
            Err(error) => Some(error.to_string()),
        };
        Ok(ByNameUpdate {
            name,
            snapshot: None,
            changed: true,
            projection_warning: warning,
        })
    }

    /// Explicitly discard all human naming metadata and rebuild automatic
    /// names from canonical snapshots. This never changes runtime refs,
    /// snapshots, provenance, or model payloads. Callers must present their own
    /// explicit destructive-intent UX before invoking it.
    pub fn reset_by_name(&self) -> Result<Vec<ByNameEntry>> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        for path in [
            NEXT_DIR,
            STATE_NEXT,
            STATE_DIR,
            INIT_MARKER,
            LEGACY_STATE,
            LEGACY_NEXT,
            LIVE_DIR,
        ] {
            remove_auxiliary(self, path)?;
        }
        self.initialize_state_locked()?;
        self.state_entries_locked()
    }

    pub fn refresh_by_name(&self) -> Result<Vec<ByNameEntry>> {
        let publication = self.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        self.ensure_state_locked()?;
        self.mark_dirty_locked(DirtyMode::Reconcile)?;
        let result = self
            .reconcile_claims_with_canonical_under_dirty_locked()
            .and_then(|()| self.rebuild_projection_locked());
        if result.is_ok() {
            self.clear_dirty_locked()?;
        }
        result?;
        self.state_entries_locked()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheLimits;

    fn manifest(provider: &str, repository: &str, revision: &str, path: &str) -> Manifest {
        Manifest::new(
            ManifestSource::new(provider, repository, revision).unwrap(),
            vec![
                ManifestEntry::new(
                    path,
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    1,
                    FileRole::Weights,
                )
                .unwrap(),
            ],
            &CacheLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn automatic_names_are_model_first_flat_lowercase_and_readable() {
        let qwen = manifest(
            "huggingface",
            "Qwen/Qwen3.5-0.8B",
            "2FC06364715B967F1860AEA9CF38778875588B17",
            "model.safetensors",
        );
        assert_eq!(auto_name(&qwen), "qwen3.5-0.8b-qwen-hf-2fc06364");

        let github = manifest("github", "Org/My_Model", "Release/V1", "model.bin");
        assert_eq!(auto_name(&github), "my-model-org-gh-release-v1");

        let object = manifest(
            "https-object",
            "origin-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "weights/llama-8b.gguf",
        );
        assert_eq!(auto_name(&object), "llama-8b-https-bbbbbbbb");
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
    fn candidate_suffixes_are_deterministic_and_bounded() {
        let model = manifest("local", "owner/my_model", "v1", "model.bin");
        let names = AutoCandidates::new(
            &model,
            "1111111111111111111111111111111111111111111111111111111111111111",
        )
        .take(10)
        .collect::<Vec<_>>();
        assert_eq!(names.len(), 10);
        assert!(names.iter().all(|name| name.len() <= MAX_ALIAS_BYTES));
        assert!(names.iter().all(|name| validate_name(name).is_ok()));
        assert!(names[1].starts_with(&(names[0].clone() + "-")));
        assert!(names[9].ends_with("-2"));
    }

    #[test]
    fn dirty_modes_roundtrip_without_ambiguity() {
        for mode in [
            DirtyMode::Projection,
            DirtyMode::Reconcile,
            DirtyMode::Pending(
                "1111111111111111111111111111111111111111111111111111111111111111".into(),
            ),
        ] {
            assert_eq!(DirtyMode::parse(&mode.bytes()).unwrap(), mode);
        }
    }
}
