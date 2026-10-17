//! Read-only capabilities for an explicitly selected local HF cache snapshot.
//! Directory labels and blob names describe layout, not authenticated origin.
use crate::cache::CacheLimits;
use crate::cache::paths::validate_relative_path;
use crate::{Error, Result};
use rustix::fs::{self, AtFlags, FileType, Mode, OFlags, StatxFlags};
use std::collections::BTreeSet;
use std::fs::{File, Metadata};
use std::io::{self, Read};
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Component, Path};
use std::sync::atomic::{AtomicBool, Ordering};

const MAX_PATH: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_NODES: usize = 131_072;
const MAX_INVENTORY_PATH_BYTES: usize = 8 * 1024 * 1024;
const ENTRY_OVERHEAD: usize = 512;

fn os<T>(result: rustix::io::Result<T>) -> Result<T> {
    result.map_err(|error| io::Error::from(error).into())
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        return Err(Error::State("local cache import cancelled".into()));
    }
    Ok(())
}

fn dir_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
}

fn file_flags() -> OFlags {
    // NONBLOCK prevents a substituted FIFO from blocking before fstat refuses it.
    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK
}

fn identity(metadata: &Metadata) -> (u64, u64) {
    (metadata.dev(), metadata.ino())
}

fn mount_id(file: BorrowedFd<'_>) -> Result<u64> {
    let stat = fs::statx(file, "", AtFlags::EMPTY_PATH, StatxFlags::MNT_ID).map_err(|_| {
        Error::Unsupported("local cache import requires readable mount identity".into())
    })?;
    if stat.stx_mask & StatxFlags::MNT_ID.bits() == 0 {
        return Err(Error::Unsupported(
            "local cache import requires mount identity support".into(),
        ));
    }
    Ok(stat.stx_mnt_id)
}

fn same_mount(file: &File, expected: u64) -> Result<()> {
    if mount_id(file.as_fd())? != expected {
        return Err(Error::Unsupported(
            "crossed mounts in local cache import are unsupported".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FilesystemKind {
    Btrfs,
    Tmpfs,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RootIdentity {
    mount: u64,
    device: u64,
    inode: u64,
    // Read only for differing-mount policy. Existing same-mount imports do not
    // gain a filesystem-kind or fstatfs requirement.
    kind: Option<FilesystemKind>,
}

impl RootIdentity {
    fn read(file: &File) -> Result<Self> {
        let metadata = file.metadata()?;
        Ok(Self {
            mount: mount_id(file.as_fd())?,
            device: metadata.dev(),
            inode: metadata.ino(),
            kind: None,
        })
    }

    fn with_kind(mut self, file: &File) -> Result<Self> {
        let stat = os(fs::fstatfs(file))?;
        // Linux filesystem magic values occupy the low 32 bits of FsWord,
        // whose signedness/width varies by target architecture.
        self.kind = Some(match stat.f_type as u32 {
            0x9123_683e => FilesystemKind::Btrfs,
            0x0102_1994 => FilesystemKind::Tmpfs,
            _ => FilesystemKind::Other,
        });
        Ok(self)
    }

    fn inode_key(self) -> (u64, u64) {
        (self.device, self.inode)
    }

    fn ensure_bound(self, file: &File) -> Result<()> {
        let current = Self::read(file)?;
        if current.mount != self.mount || current.inode_key() != self.inode_key() {
            return Err(Error::State("local HF root identity changed".into()));
        }
        Ok(())
    }
}

/// A policy decision, independently qualified by actual cross-filesystem
/// copies. Device numbers alone never authorize an arbitrary filesystem pair.
fn cross_filesystem(source: RootIdentity, target: RootIdentity) -> Result<bool> {
    if source.mount == target.mount {
        return Ok(false);
    }
    if source.device == target.device {
        return Err(Error::Unsupported(
            "same-device mount aliases in local HF import are unsupported".into(),
        ));
    }
    use FilesystemKind::{Btrfs, Tmpfs};
    if !matches!(
        (source.kind, target.kind),
        (Some(Btrfs), Some(Tmpfs)) | (Some(Tmpfs), Some(Btrfs)) | (Some(Tmpfs), Some(Tmpfs))
    ) {
        return Err(Error::Unsupported(
            "local HF cross-filesystem pair is not qualified".into(),
        ));
    }
    Ok(true)
}

fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Pure request validation, shared with the public request constructor. Opening
/// the view repeats these checks after the caller has obtained admission.
pub(super) fn validate_request(
    root: &Path,
    repository: &str,
    commit: &str,
    selected: &[String],
    limits: &CacheLimits,
) -> Result<()> {
    limits.validate()?;
    if !root.is_absolute()
        || root.as_os_str().as_bytes().len() > MAX_PATH
        || root
            .components()
            .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
    {
        return Err(Error::Malformed(
            "local HF repository-cache root must be an absolute normal path".into(),
        ));
    }
    let mut pieces = repository.split('/');
    let owner = pieces.next().unwrap_or_default();
    let name = pieces.next().unwrap_or_default();
    let valid_piece = |piece: &str| {
        !piece.is_empty()
            && piece.len() <= 96
            && piece != "."
            && piece != ".."
            && piece
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    };
    if pieces.next().is_some() || !valid_piece(owner) || !valid_piece(name) {
        return Err(Error::Malformed(
            "local HF repository needs two bounded owner/name components".into(),
        ));
    }
    let expected = format!("models--{owner}--{name}");
    if root.file_name().and_then(|leaf| leaf.to_str()) != Some(expected.as_str()) {
        return Err(Error::Malformed(
            "local HF repository-cache directory does not match repository".into(),
        ));
    }
    if !lower_hex(commit, 40) {
        return Err(Error::Malformed(
            "local HF commit must be 40 lowercase hexadecimal digits".into(),
        ));
    }
    if selected.is_empty() || selected.len() > limits.max_files {
        return Err(Error::Limit("local HF selected file count".into()));
    }
    let mut paths = BTreeSet::new();
    let mut metadata_bytes = 0usize;
    for path in selected {
        validate_relative_path(path)?;
        if path.split('/').any(|part| part.ends_with(".incomplete")) {
            return Err(Error::Malformed("incomplete local HF file refused".into()));
        }
        metadata_bytes = metadata_bytes
            .checked_add(path.len())
            .and_then(|total| total.checked_add(ENTRY_OVERHEAD))
            .ok_or_else(|| Error::Limit("local HF metadata arithmetic".into()))?;
        if metadata_bytes > limits.max_manifest_bytes {
            return Err(Error::Limit("local HF selected metadata limit".into()));
        }
        if !paths.insert(path.as_str()) {
            return Err(Error::Malformed("duplicate local HF selected path".into()));
        }
    }
    for path in &paths {
        for (position, _) in path.match_indices('/') {
            if paths.contains(&path[..position]) {
                return Err(Error::Malformed(
                    "local HF selected file/directory conflict".into(),
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Stamp {
    device: u64,
    inode: u64,
    bytes: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}

impl Stamp {
    fn of(file: &File) -> Result<Self> {
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.nlink() == 0 {
            return Err(Error::State(
                "local HF source is not a live regular file".into(),
            ));
        }
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            bytes: metadata.len(),
            mtime: metadata.mtime(),
            mtime_ns: metadata.mtime_nsec(),
            ctime: metadata.ctime(),
            ctime_ns: metadata.ctime_nsec(),
        })
    }

    fn identity(self) -> (u64, u64) {
        (self.device, self.inode)
    }
}

pub(super) struct SourceView {
    root: File,
    identity: RootIdentity,
    entries: Vec<SourceFile>,
}

pub(super) struct SourceFile {
    path: String,
    file: File,
    stamp: Stamp,
}

impl SourceView {
    pub(super) fn open(
        root: &Path,
        repository: &str,
        commit: &str,
        selected: &[String],
        limits: &CacheLimits,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        validate_request(root, repository, commit, selected, limits)?;
        check_cancel(cancel)?;
        let mut directory = File::from(os(fs::open("/", dir_flags(), Mode::empty()))?);
        for component in root.components() {
            if let Component::Normal(name) = component {
                check_cancel(cancel)?;
                directory = File::from(os(fs::openat(
                    &directory,
                    name,
                    dir_flags(),
                    Mode::empty(),
                ))?);
            }
        }
        let root_identity = RootIdentity::read(&directory)?;
        let mount = root_identity.mount;
        let snapshots = open_directory(&directory, "snapshots", mount)?;
        let snapshot = open_directory(&snapshots, commit, mount)?;
        let mut entries = Vec::with_capacity(selected.len());
        let mut total = 0u64;
        for path in selected {
            check_cancel(cancel)?;
            let (parent, leaf) = path.rsplit_once('/').unwrap_or(("", path.as_str()));
            let mut parent_directory = snapshot.try_clone()?;
            for component in parent.split('/').filter(|component| !component.is_empty()) {
                check_cancel(cancel)?;
                parent_directory = open_directory(&parent_directory, component, mount)?;
            }
            let before = os(fs::statat(
                &parent_directory,
                leaf,
                AtFlags::SYMLINK_NOFOLLOW,
            ))?;
            let file = match FileType::from_raw_mode(before.st_mode) {
                FileType::RegularFile => File::from(os(fs::openat(
                    &parent_directory,
                    leaf,
                    file_flags(),
                    Mode::empty(),
                ))?),
                FileType::Symlink => {
                    let mut buffer = [0u8; MAX_PATH + 1];
                    let length = os(fs::readlinkat_raw(&parent_directory, leaf, &mut buffer[..]))?;
                    if length > MAX_PATH {
                        return Err(Error::Limit("local HF leaf link is too long".into()));
                    }
                    let target = std::str::from_utf8(&buffer[..length])
                        .map_err(|_| Error::Malformed("non-UTF8 local HF leaf link".into()))?;
                    let blob = canonical_blob(target, path.split('/').count() + 1)?;
                    let blobs = open_directory(&directory, "blobs", mount)?;
                    let file =
                        File::from(os(fs::openat(&blobs, blob, file_flags(), Mode::empty()))?);
                    let after = os(fs::statat(
                        &parent_directory,
                        leaf,
                        AtFlags::SYMLINK_NOFOLLOW,
                    ))?;
                    if before.st_dev != after.st_dev
                        || before.st_ino != after.st_ino
                        || before.st_mtime != after.st_mtime
                        || before.st_mtime_nsec != after.st_mtime_nsec
                        || before.st_ctime != after.st_ctime
                        || before.st_ctime_nsec != after.st_ctime_nsec
                    {
                        return Err(Error::State(
                            "local HF leaf link changed during opening".into(),
                        ));
                    }
                    file
                }
                _ => return Err(Error::State("special local HF source file refused".into())),
            };
            same_mount(&file, mount)?;
            let stamp = Stamp::of(&file)?;
            if FileType::from_raw_mode(before.st_mode) == FileType::RegularFile
                && (stamp.device != before.st_dev || stamp.inode != before.st_ino)
            {
                return Err(Error::State(
                    "local HF source changed during opening".into(),
                ));
            }
            if stamp.bytes > limits.max_file_bytes {
                return Err(Error::Limit("local HF source file length limit".into()));
            }
            total = total
                .checked_add(stamp.bytes)
                .ok_or_else(|| Error::Limit("local HF total length arithmetic".into()))?;
            if total > limits.max_snapshot_bytes {
                return Err(Error::Limit("local HF source snapshot length limit".into()));
            }
            entries.push(SourceFile {
                path: path.clone(),
                file,
                stamp,
            });
        }
        let view = Self {
            root: directory,
            identity: root_identity,
            entries,
        };
        view.ensure_stable(cancel)?;
        Ok(view)
    }

    pub(super) fn entries(&self) -> &[SourceFile] {
        &self.entries
    }

    pub(super) fn ensure_stable(&self, cancel: &AtomicBool) -> Result<()> {
        check_cancel(cancel)?;
        self.identity.ensure_bound(&self.root)?;
        for entry in &self.entries {
            entry.ensure_stable(cancel)?;
        }
        check_cancel(cancel)
    }

    /// Prove bounded topology disjointness under each root's own identity.
    /// Distinct-mount pairs are deliberately limited to qualified filesystems;
    /// same-device aliases still refuse. No source paths are reopened.
    pub(super) fn ensure_disjoint(
        &self,
        target: BorrowedFd<'_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.ensure_stable(cancel)?;
        let target_file = File::from(os(fs::openat(target, ".", dir_flags(), Mode::empty()))?);
        let mut source_identity = self.identity;
        let mut target_identity = RootIdentity::read(&target_file)?;
        if source_identity.mount != target_identity.mount
            && source_identity.device != target_identity.device
        {
            source_identity = source_identity.with_kind(&self.root)?;
            target_identity = target_identity.with_kind(&target_file)?;
        }
        let crossfs = cross_filesystem(source_identity, target_identity)?;
        if contains_ancestor(&self.root, target_identity.inode_key(), cancel)?
            || contains_ancestor(&target_file, source_identity.inode_key(), cancel)?
        {
            return Err(Error::State(
                "local HF source and destination trees overlap".into(),
            ));
        }
        // Compact inode keys are the only retained inventory. The walk never
        // reads refs or unselected payload contents and keeps at most one
        // directory stream per depth; symlinks are metadata-only capabilities.
        let mut source_walk = TreeWalk::new(source_identity, crossfs, None, cancel);
        source_walk.collected.insert(source_identity.inode_key());
        source_walk
            .collected
            .extend(self.entries.iter().map(|entry| entry.stamp.identity()));
        if crossfs {
            for entry in &self.entries {
                check_cancel(cancel)?;
                if entry.stamp.device != source_identity.device {
                    return Err(Error::Unsupported(
                        "crossed source devices in local HF import are unsupported".into(),
                    ));
                }
                same_mount(&entry.file, source_identity.mount)?;
            }
        }
        source_walk.inspect(&self.root, 0, 0)?;
        if source_walk.collected.contains(&target_identity.inode_key()) {
            return Err(Error::State(
                "local HF source inode aliases destination root".into(),
            ));
        }
        TreeWalk::new(
            target_identity,
            crossfs,
            Some(&source_walk.collected),
            cancel,
        )
        .inspect(&target_file, 0, 0)?;
        source_identity.ensure_bound(&self.root)?;
        target_identity.ensure_bound(&target_file)?;
        self.ensure_stable(cancel)
    }
}

fn open_directory(parent: &File, name: &str, mount: u64) -> Result<File> {
    let file = File::from(os(fs::openat(parent, name, dir_flags(), Mode::empty()))?);
    same_mount(&file, mount)?;
    Ok(file)
}

fn canonical_blob(target: &str, parents: usize) -> Result<&str> {
    let mut remaining = target;
    for _ in 0..parents {
        remaining = remaining
            .strip_prefix("../")
            .ok_or_else(|| Error::Malformed("noncanonical local HF leaf link".into()))?;
    }
    let blob = remaining
        .strip_prefix("blobs/")
        .ok_or_else(|| Error::Malformed("local HF leaf link must target same-root blobs".into()))?;
    if !lower_hex(blob, 40) && !lower_hex(blob, 64) {
        return Err(Error::Malformed("invalid local HF blob name".into()));
    }
    Ok(blob)
}

impl SourceFile {
    pub(super) fn path(&self) -> &str {
        &self.path
    }
    pub(super) fn bytes(&self) -> u64 {
        self.stamp.bytes
    }
    pub(super) fn ensure_stable(&self, cancel: &AtomicBool) -> Result<()> {
        check_cancel(cancel)?;
        if Stamp::of(&self.file)? != self.stamp {
            return Err(Error::State("local HF source changed during import".into()));
        }
        Ok(())
    }
    pub(super) fn reader<'a>(&'a self, cancel: &'a AtomicBool) -> Result<SourceReader<'a>> {
        self.ensure_stable(cancel)?;
        Ok(SourceReader {
            source: self,
            cancel,
            offset: 0,
        })
    }
}

/// Independent positioned cursor over the original read-only descriptor. This
/// exposes no source descriptor, pathname reopen, lock or writable operation.
pub(super) struct SourceReader<'a> {
    source: &'a SourceFile,
    cancel: &'a AtomicBool,
    offset: u64,
}

impl Read for SourceReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let fail = |error: Error| io::Error::other(error.to_string());
        self.source.ensure_stable(self.cancel).map_err(fail)?;
        if buffer.is_empty() {
            return Ok(0);
        }
        let remaining = self.source.bytes().saturating_sub(self.offset);
        let maximum = remaining.saturating_add(1).min(buffer.len() as u64) as usize;
        let count = self
            .source
            .file
            .read_at(&mut buffer[..maximum], self.offset)?;
        self.source.ensure_stable(self.cancel).map_err(fail)?;
        if count as u64 > remaining || (count == 0 && remaining != 0) {
            return Err(io::Error::other(
                "local HF source length changed during import",
            ));
        }
        self.offset += count as u64;
        Ok(count)
    }
}

fn contains_ancestor(start: &File, candidate: (u64, u64), cancel: &AtomicBool) -> Result<bool> {
    let mut current = start.try_clone()?;
    for _ in 0..256 {
        check_cancel(cancel)?;
        let current_identity = identity(&current.metadata()?);
        if current_identity == candidate {
            return Ok(true);
        }
        let parent = File::from(os(fs::openat(&current, "..", dir_flags(), Mode::empty()))?);
        if identity(&parent.metadata()?) == current_identity {
            return Ok(false);
        }
        current = parent;
    }
    Err(Error::Limit("local cache ancestry depth limit".into()))
}

struct TreeWalk<'a> {
    mount: u64,
    device: Option<u64>,
    forbidden: Option<&'a BTreeSet<(u64, u64)>>,
    collected: BTreeSet<(u64, u64)>,
    cancel: &'a AtomicBool,
    nodes: usize,
    path_bytes: usize,
}

impl<'a> TreeWalk<'a> {
    fn new(
        root: RootIdentity,
        enforce_device: bool,
        forbidden: Option<&'a BTreeSet<(u64, u64)>>,
        cancel: &'a AtomicBool,
    ) -> Self {
        Self {
            mount: root.mount,
            device: enforce_device.then_some(root.device),
            forbidden,
            collected: BTreeSet::new(),
            cancel,
            nodes: 0,
            path_bytes: 0,
        }
    }

    fn inspect(&mut self, directory: &File, depth: usize, prefix_bytes: usize) -> Result<()> {
        check_cancel(self.cancel)?;
        if depth > MAX_DEPTH {
            return Err(Error::Limit("local import topology depth limit".into()));
        }
        for entry in os(fs::Dir::read_from(directory))? {
            check_cancel(self.cancel)?;
            let entry = os(entry)?;
            let name = entry.file_name();
            if name.to_bytes() == b"." || name.to_bytes() == b".." {
                continue;
            }
            self.nodes += 1;
            let path_bytes = prefix_bytes
                .checked_add(name.to_bytes().len() + 1)
                .ok_or_else(|| Error::Limit("local import target path arithmetic".into()))?;
            self.path_bytes = self
                .path_bytes
                .checked_add(path_bytes)
                .ok_or_else(|| Error::Limit("local import target inventory arithmetic".into()))?;
            if self.nodes > MAX_NODES
                || self.path_bytes > MAX_INVENTORY_PATH_BYTES
                || path_bytes > MAX_PATH
            {
                return Err(Error::Limit("local import target inventory limit".into()));
            }
            let before = os(fs::statat(directory, name, AtFlags::SYMLINK_NOFOLLOW))?;
            let kind = FileType::from_raw_mode(before.st_mode);
            let flags = match kind {
                FileType::Directory => dir_flags(),
                FileType::RegularFile => OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                FileType::Symlink if self.forbidden.is_none() => {
                    OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC
                }
                _ => {
                    return Err(Error::State(
                        "symlink/special local import target entry refused".into(),
                    ));
                }
            };
            let file = File::from(os(fs::openat(directory, name, flags, Mode::empty()))?);
            let metadata = file.metadata()?;
            if metadata.dev() != before.st_dev || metadata.ino() != before.st_ino {
                return Err(Error::State(
                    "local import target changed during inspection".into(),
                ));
            }
            same_mount(&file, self.mount)?;
            if self.device.is_some_and(|device| metadata.dev() != device) {
                return Err(Error::Unsupported(
                    "crossed topology devices in local HF import are unsupported".into(),
                ));
            }
            let inode = identity(&metadata);
            if self
                .forbidden
                .is_some_and(|forbidden| forbidden.contains(&inode))
            {
                return Err(Error::State(
                    "local HF source inode aliases destination".into(),
                ));
            }
            if self.forbidden.is_none() {
                self.collected.insert(inode);
            }
            if kind == FileType::Directory {
                self.inspect(&file, depth + 1, path_bytes)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs as stdfs;
    use std::io::Write;
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt, symlink};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const BLOB: &str = "abcdef0123456789abcdef0123456789abcdef01";

    fn policy_root(device: u64, mount: u64, kind: Option<FilesystemKind>) -> RootIdentity {
        RootIdentity {
            device,
            mount,
            inode: 1,
            kind,
        }
    }

    // These pure checks exercise policy only. The independent actual-filesystem
    // fixtures establish cross-filesystem behavior; synthetic IDs do not.
    #[test]
    fn cross_filesystem_policy_accepts_only_the_qualified_kind_pairs() {
        use FilesystemKind::{Btrfs, Other, Tmpfs};
        for source in [Btrfs, Tmpfs, Other] {
            for target in [Btrfs, Tmpfs, Other] {
                let accepted = matches!(
                    (source, target),
                    (Btrfs, Tmpfs) | (Tmpfs, Btrfs) | (Tmpfs, Tmpfs)
                );
                assert_eq!(
                    cross_filesystem(
                        policy_root(1, 10, Some(source)),
                        policy_root(2, 20, Some(target))
                    )
                    .is_ok(),
                    accepted,
                    "{source:?} -> {target:?}"
                );
            }
        }
        assert!(
            cross_filesystem(policy_root(1, 10, None), policy_root(2, 20, Some(Tmpfs))).is_err()
        );
    }

    #[test]
    fn cross_filesystem_policy_refuses_same_device_mount_aliases() {
        use FilesystemKind::{Btrfs, Other, Tmpfs};
        for source in [Btrfs, Tmpfs, Other] {
            for target in [Btrfs, Tmpfs, Other] {
                assert!(matches!(
                    cross_filesystem(
                        policy_root(1, 10, Some(source)),
                        policy_root(1, 20, Some(target))
                    ),
                    Err(Error::Unsupported(_))
                ));
            }
        }
    }

    #[test]
    fn same_mount_policy_retains_original_device_and_kind_behavior() {
        for device in [1, 2] {
            for kind in [
                None,
                Some(FilesystemKind::Other),
                Some(FilesystemKind::Btrfs),
            ] {
                assert!(
                    !cross_filesystem(policy_root(1, 10, None), policy_root(device, 10, kind))
                        .unwrap()
                );
            }
        }
    }

    struct Fixture {
        base: PathBuf,
        root: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let base = std::env::temp_dir().join(format!(
                "zerogravity-hf-source-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            stdfs::DirBuilder::new().mode(0o700).create(&base).unwrap();
            let root = base.join("models--owner--repo");
            stdfs::create_dir_all(root.join(format!("snapshots/{COMMIT}"))).unwrap();
            stdfs::create_dir(root.join("blobs")).unwrap();
            stdfs::set_permissions(&root, stdfs::Permissions::from_mode(0o755)).unwrap();
            Self { base, root }
        }
        fn snapshot(&self, path: &str) -> PathBuf {
            self.root.join(format!("snapshots/{COMMIT}/{path}"))
        }
        fn direct(&self, path: &str, data: &[u8]) {
            let target = self.snapshot(path);
            stdfs::create_dir_all(target.parent().unwrap()).unwrap();
            stdfs::write(target, data).unwrap();
        }
        fn view(&self, paths: &[&str]) -> Result<SourceView> {
            SourceView::open(
                &self.root,
                "owner/repo",
                COMMIT,
                &paths
                    .iter()
                    .map(|path| (*path).to_owned())
                    .collect::<Vec<_>>(),
                &CacheLimits::default(),
                &AtomicBool::new(false),
            )
        }
        fn target(&self) -> File {
            let path = self.base.join("target");
            stdfs::DirBuilder::new().mode(0o700).create(&path).unwrap();
            File::open(path).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = stdfs::remove_dir_all(&self.base);
        }
    }

    #[test]
    fn canonical_links_have_exact_depth_and_flat_lowercase_blob_ids() {
        assert_eq!(
            canonical_blob(&format!("../../blobs/{BLOB}"), 2).unwrap(),
            BLOB
        );
        assert!(canonical_blob(&format!("../../../blobs/{}", "a".repeat(64)), 3).is_ok());
        for target in [
            format!("../blobs/{BLOB}"),
            format!("../../../blobs/{BLOB}"),
            format!("../../blobs/{BLOB}/extra"),
            format!("/blobs/{BLOB}"),
            format!("../../blobs/{}.incomplete", BLOB),
            format!("../../blobs/{}", "A".repeat(40)),
            format!("../../blobs/./{BLOB}"),
            format!("../../snapshots/{BLOB}"),
        ] {
            assert!(canonical_blob(&target, 2).is_err(), "{target}");
        }
    }

    #[test]
    fn direct_and_nested_blob_entries_keep_exact_readonly_original_bytes() {
        let fixture = Fixture::new();
        fixture.direct("direct.json", b"direct");
        stdfs::create_dir(fixture.snapshot("nested")).unwrap();
        stdfs::write(fixture.root.join(format!("blobs/{BLOB}")), b"linked").unwrap();
        symlink(
            format!("../../../blobs/{BLOB}"),
            fixture.snapshot("nested/file.json"),
        )
        .unwrap();
        let before = stdfs::metadata(fixture.snapshot("direct.json")).unwrap();
        let view = fixture.view(&["direct.json", "nested/file.json"]).unwrap();
        let cancel = AtomicBool::new(false);
        for (entry, expected) in view.entries().iter().zip([b"direct", b"linked"]) {
            assert_eq!(entry.bytes(), expected.len() as u64);
            assert!(!entry.path().is_empty());
            let mut first = entry.reader(&cancel).unwrap();
            let mut one = [0];
            first.read_exact(&mut one).unwrap();
            assert_eq!(one[0], expected[0]);
            let mut independent = Vec::new();
            entry
                .reader(&cancel)
                .unwrap()
                .read_to_end(&mut independent)
                .unwrap();
            assert_eq!(independent, expected);
        }
        let after = stdfs::metadata(fixture.snapshot("direct.json")).unwrap();
        assert_eq!(before.mode(), after.mode());
        assert_eq!(
            (before.mtime(), before.mtime_nsec()),
            (after.mtime(), after.mtime_nsec())
        );
        view.ensure_disjoint(fixture.target().as_fd(), &cancel)
            .unwrap();
    }

    #[test]
    fn replacement_of_source_root_path_never_redirects_opened_descriptor() {
        let fixture = Fixture::new();
        fixture.direct("data", b"original");
        let view = fixture.view(&["data"]).unwrap();
        stdfs::rename(&fixture.root, fixture.base.join("old-source")).unwrap();
        stdfs::create_dir_all(fixture.root.join(format!("snapshots/{COMMIT}"))).unwrap();
        stdfs::write(fixture.snapshot("data"), b"replacement").unwrap();
        let mut bytes = Vec::new();
        view.entries()[0]
            .reader(&AtomicBool::new(false))
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(bytes, b"original");
        view.ensure_disjoint(fixture.target().as_fd(), &AtomicBool::new(false))
            .unwrap();
    }

    #[test]
    fn read_boundary_refuses_in_place_mutation_growth_and_truncation() {
        for replacement in [&b"changed!"[..], &b"longer-than-original"[..], &b"x"[..]] {
            let fixture = Fixture::new();
            fixture.direct("data", b"original");
            let view = fixture.view(&["data"]).unwrap();
            let cancel = AtomicBool::new(false);
            let mut reader = view.entries()[0].reader(&cancel).unwrap();
            let mut byte = [0];
            reader.read_exact(&mut byte).unwrap();
            let mut writer = stdfs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(fixture.snapshot("data"))
                .unwrap();
            writer.write_all(replacement).unwrap();
            assert!(reader.read(&mut byte).is_err());
            assert!(view.ensure_stable(&cancel).is_err());
        }
    }

    #[test]
    fn selected_intermediate_links_blob_chains_and_special_files_refuse() {
        let fixture = Fixture::new();
        fixture.direct("data", b"source");
        symlink(fixture.snapshot("data"), fixture.snapshot("absolute")).unwrap();
        assert!(fixture.view(&["absolute"]).is_err());
        symlink(".", fixture.snapshot("intermediate")).unwrap();
        assert!(fixture.view(&["intermediate/data"]).is_err());
        symlink(format!("../../blobs/{BLOB}"), fixture.snapshot("chained")).unwrap();
        symlink(
            fixture.snapshot("data"),
            fixture.root.join(format!("blobs/{BLOB}")),
        )
        .unwrap();
        assert!(fixture.view(&["chained"]).is_err());
        fs::mknodat(
            fs::CWD,
            fixture.snapshot("fifo").as_path(),
            FileType::Fifo,
            Mode::RUSR | Mode::WUSR,
            0,
        )
        .unwrap();
        assert!(fixture.view(&["fifo"]).is_err());
    }

    #[test]
    fn overlap_rejects_both_ancestor_directions_and_hardlinked_unselected_files() {
        let fixture = Fixture::new();
        fixture.direct("data", b"source");
        let view = fixture.view(&["data"]).unwrap();
        let cancel = AtomicBool::new(false);
        let ancestor = File::open(&fixture.base).unwrap();
        assert!(view.ensure_disjoint(ancestor.as_fd(), &cancel).is_err());
        let child = fixture.root.join("destination");
        stdfs::create_dir(&child).unwrap();
        assert!(
            view.ensure_disjoint(File::open(child).unwrap().as_fd(), &cancel)
                .is_err()
        );
        stdfs::write(fixture.root.join("unselected"), b"do not mutate").unwrap();
        let target = fixture.target();
        stdfs::hard_link(
            fixture.root.join("unselected"),
            fixture.base.join("target/alias"),
        )
        .unwrap();
        assert!(view.ensure_disjoint(target.as_fd(), &cancel).is_err());
    }

    #[test]
    fn topology_symlinks_are_metadata_only_and_target_symlinks_refuse() {
        let fixture = Fixture::new();
        fixture.direct("data", b"source");
        symlink(
            "/unavailable/not-followed",
            fixture.root.join("unselected-link"),
        )
        .unwrap();
        let view = fixture.view(&["data"]).unwrap();
        let target = fixture.target();
        let cancel = AtomicBool::new(false);
        view.ensure_disjoint(target.as_fd(), &cancel).unwrap();
        symlink(
            "/unavailable/not-followed",
            fixture.base.join("target/alias"),
        )
        .unwrap();
        assert!(view.ensure_disjoint(target.as_fd(), &cancel).is_err());
    }

    #[test]
    fn pure_request_limits_and_cancellation_precede_source_access() {
        let fixture = Fixture::new();
        fixture.direct("data", b"source");
        for paths in [
            vec!["data", "data"],
            vec!["data", "data-x", "data/child"],
            vec!["x.incomplete"],
            vec!["../data"],
        ] {
            assert!(fixture.view(&paths).is_err());
        }
        assert!(
            SourceView::open(
                &fixture.root,
                "owner/repo",
                COMMIT,
                &["data".into()],
                &CacheLimits::default(),
                &AtomicBool::new(true)
            )
            .is_err()
        );
        let limits = CacheLimits {
            max_file_bytes: 1,
            ..Default::default()
        };
        assert!(
            SourceView::open(
                &fixture.root,
                "owner/repo",
                COMMIT,
                &["data".into()],
                &limits,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(
            validate_request(
                &fixture.root,
                "owner/repo",
                &COMMIT.to_uppercase(),
                &["data".into()],
                &CacheLimits::default()
            )
            .is_err()
        );
        assert!(
            validate_request(
                &fixture.root,
                "other/repo",
                COMMIT,
                &["data".into()],
                &CacheLimits::default()
            )
            .is_err()
        );
    }
}
