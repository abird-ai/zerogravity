//! Mount-bounded mutation helpers for disposable human-only namespaces.

use super::*;
use crate::cache::paths::{mount_id, require_mount, require_mount_at};
use std::ffi::{CStr, CString};
use std::fs::File;
use std::os::fd::AsFd;
use std::os::unix::fs::MetadataExt;

pub(super) const MAX_AUX_NODES: usize = MAX_STATE_CLEANUP_NODES;
const MAX_AUX_DEPTH: usize = 64;

pub(super) fn root_copy(store: &CacheStore) -> Result<File> {
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

fn remove_entry_at(
    parent: &File,
    name: &CStr,
    expected_mount: u64,
    nodes: &mut usize,
    depth: usize,
) -> Result<()> {
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
    require_mount_at(parent, name, expected_mount)?;
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
    require_mount(&directory, expected_mount)?;
    let metadata = directory.metadata()?;
    if metadata.dev() != before.st_dev || metadata.ino() != before.st_ino {
        return Err(Error::State(
            "by-name cleanup directory changed during inspection".into(),
        ));
    }
    for entry in fs::Dir::read_from(&directory).map_err(io)? {
        let entry = entry.map_err(io)?;
        let child = entry.file_name().to_owned();
        if child.to_bytes() == b"." || child.to_bytes() == b".." {
            continue;
        }
        remove_entry_at(&directory, &child, expected_mount, nodes, depth + 1)?;
    }
    let current = fs::statat(parent, name, AtFlags::SYMLINK_NOFOLLOW).map_err(io)?;
    if current.st_dev != before.st_dev || current.st_ino != before.st_ino {
        return Err(Error::State(
            "by-name cleanup directory binding changed".into(),
        ));
    }
    fs::unlinkat(parent, name, AtFlags::REMOVEDIR).map_err(io)
}

pub(super) fn remove_relative_tree(store: &CacheStore, path: &str) -> Result<()> {
    validate_relative_path(path)?;
    let root = root_copy(store)?;
    let expected_mount = mount_id(root.as_fd())?;
    let (parent_path, leaf) = path.rsplit_once('/').unwrap_or(("", path));
    let parent = store.inner.root.open_dir(parent_path, false)?;
    require_mount(&parent, expected_mount)?;
    let leaf =
        CString::new(leaf).map_err(|_| Error::Malformed("invalid by-name cleanup path".into()))?;
    let mut nodes = 0;
    remove_entry_at(&parent, &leaf, expected_mount, &mut nodes, 0)
}

pub(super) fn remove_auxiliary(store: &CacheStore, name: &str) -> Result<()> {
    if !crate::cache::paths::is_by_name_auxiliary(name) {
        return Err(Error::State(
            "refusing to remove a non by-name auxiliary entry".into(),
        ));
    }
    remove_relative_tree(store, name)
}

pub(super) fn parent_path(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(parent, _)| parent)
}
