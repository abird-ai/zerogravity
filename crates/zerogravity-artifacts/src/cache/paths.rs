//! Linux directory capabilities and cooperative stable-inode advisory locks.
//! No pathname reopening, source issuer, cache quota or publication authority.
use crate::{Error, Result};
use rustix::fs::{self, AtFlags, FileType, FlockOperation, Mode, OFlags};
use std::ffi::OsStr;
use std::fs::File;
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const MAX_PATH: usize = 4096;
const MAX_COMPONENTS: usize = 64;
const MAX_INVENTORY_PATH_BYTES: usize = 8 * 1024 * 1024;

fn os<T>(result: rustix::io::Result<T>) -> Result<T> {
    result.map_err(|error| std::io::Error::from(error).into())
}

/// Logical paths are case-sensitive UTF8 bytes on Linux. No Unicode/case
/// normalization is performed; other platform filename policies are unqualified.
pub fn validate_relative_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.len() > MAX_PATH
        || path.starts_with('/')
        || path.contains('\\')
        || path.chars().any(char::is_control)
    {
        return Err(Error::Malformed("invalid cache relative path".into()));
    }
    let mut count = 0;
    for component in path.split('/') {
        count += 1;
        if component.is_empty()
            || component == "."
            || component == ".."
            || component.len() > 255
            || count > MAX_COMPONENTS
        {
            return Err(Error::Malformed("invalid cache path component".into()));
        }
    }
    Ok(())
}

pub fn validate_digest(digest: &str) -> Result<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(Error::Malformed("expected lowercase SHA256 digest".into()));
    }
    Ok(())
}

/// Pure XDG resolution; callers supply environment values without changing the
/// process environment. Relative/empty XDG is ignored; explicit roots must be absolute.
pub fn resolve_root(
    explicit: Option<&Path>,
    xdg: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Result<PathBuf> {
    if let Some(root) = explicit {
        if !root.is_absolute() {
            return Err(Error::Malformed(
                "explicit cache root must be absolute".into(),
            ));
        }
        return Ok(root.to_owned());
    }
    if let Some(path) = xdg.map(Path::new).filter(|p| p.is_absolute()) {
        return Ok(path.join("abird/zerogravity/models"));
    }
    let home = home
        .map(Path::new)
        .filter(|p| p.is_absolute())
        .ok_or_else(|| {
            Error::State("absolute HOME needed when XDG_CACHE_HOME is absent/relative".into())
        })?;
    Ok(home.join(".cache/abird/zerogravity/models"))
}

fn dir_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
}
fn private_dir(file: &File) -> Result<()> {
    let metadata = file.metadata()?;
    if !metadata.is_dir()
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || metadata.mode() & 0o7777 != 0o700
        || metadata.nlink() == 0
    {
        return Err(Error::State(
            "cache directory must be live, owner-private 0700".into(),
        ));
    }
    Ok(())
}
fn private_file(file: &File) -> Result<()> {
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || !matches!(metadata.mode() & 0o7777, 0o400 | 0o600)
        || metadata.nlink() == 0
    {
        return Err(Error::State(
            "cache file must be live, owner-private and regular".into(),
        ));
    }
    Ok(())
}
fn child_dir(parent: &File, name: &OsStr, create: bool) -> Result<File> {
    match fs::openat(parent, name, dir_flags(), Mode::empty()) {
        Ok(fd) => Ok(File::from(fd)),
        Err(rustix::io::Errno::NOENT) if create => {
            match fs::mkdirat(parent, name, Mode::RWXU) {
                // Persist every newly created pathname edge, including absolute
                // root ancestors. A later snapshot/ref sync cannot replace it.
                Ok(()) => parent.sync_all()?,
                Err(rustix::io::Errno::EXIST) => {}
                Err(error) => return Err(std::io::Error::from(error).into()),
            }
            Ok(File::from(os(fs::openat(
                parent,
                name,
                dir_flags(),
                Mode::empty(),
            ))?))
        }
        Err(error) => Err(std::io::Error::from(error).into()),
    }
}

/// An opened root inode is the authority. Concurrent renaming/rebinding of its
/// old pathname never redirects this capability through replacement ancestors.
/// Same-UID uncooperative writers can still mutate opened inodes; this is not
/// OS immutability, and eventual executable loading must verify its actual copy.
pub(crate) struct RootDir {
    file: File,
    identity: (u64, u64),
}
pub(crate) struct CacheNode {
    pub(crate) path: String,
    pub(crate) bytes: u64,
    pub(crate) dev: u64,
    pub(crate) ino: u64,
    pub(crate) allocated_bytes: u64,
    pub(crate) is_directory: bool,
}
impl RootDir {
    pub(crate) fn open(path: &Path, create: bool) -> Result<Self> {
        if !path.is_absolute() {
            return Err(Error::Malformed("cache directory must be absolute".into()));
        }
        let mut file = File::from(os(fs::open("/", dir_flags(), Mode::empty()))?);
        for component in path.components() {
            match component {
                Component::RootDir => {}
                Component::Normal(name) => file = child_dir(&file, name, create)?,
                _ => {
                    return Err(Error::Malformed(
                        "cache root contains non-normal components".into(),
                    ));
                }
            }
        }
        private_dir(&file)?;
        let metadata = file.metadata()?;
        Ok(Self {
            identity: (metadata.dev(), metadata.ino()),
            file,
        })
    }
    pub(crate) fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
    pub(crate) fn identity(&self) -> (u64, u64) {
        self.identity
    }
    pub(crate) fn open_dir(&self, path: &str, create: bool) -> Result<File> {
        private_dir(&self.file)?;
        let mut directory = File::from(os(fs::openat(
            self.as_fd(),
            ".",
            dir_flags(),
            Mode::empty(),
        ))?);
        let metadata = directory.metadata()?;
        if (metadata.dev(), metadata.ino()) != self.identity() {
            return Err(Error::State(
                "cache root capability identity changed".into(),
            ));
        }
        if path.is_empty() || path == "." {
            return Ok(directory);
        }
        validate_relative_path(path)?;
        for name in path.split('/') {
            directory = child_dir(&directory, OsStr::new(name), create)?;
            private_dir(&directory)?;
        }
        Ok(directory)
    }
    /// Return only a validated flat leaf and its pinned parent. Store mutations
    /// must use these descriptors, never rootfd + an unchecked nested pathname.
    pub(crate) fn parent_dir(&self, path: &str, create: bool) -> Result<(File, String)> {
        validate_relative_path(path)?;
        let (parent, leaf) = path.rsplit_once('/').unwrap_or(("", path));
        Ok((self.open_dir(parent, create)?, leaf.to_owned()))
    }
    pub(crate) fn open_file(&self, path: &str, flags: OFlags, mode: Mode) -> Result<File> {
        if flags.intersects(
            OFlags::WRONLY | OFlags::RDWR | OFlags::TRUNC | OFlags::APPEND | OFlags::CREATE,
        ) && !flags.contains(OFlags::CREATE | OFlags::EXCL)
        {
            return Err(Error::State(
                "cache writes require exclusive new-file creation".into(),
            ));
        }
        let (parent, name) = self.parent_dir(path, flags.contains(OFlags::CREATE))?;
        // NONBLOCK avoids hanging if an uncooperative writer substitutes a FIFO
        // between lookup and open. The opened descriptor is checked as regular.
        let file = File::from(os(fs::openat(
            &parent,
            name,
            flags | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            mode,
        ))?);
        private_file(&file)?;
        Ok(file)
    }
    /// Bounded no-follow inventory includes the root and all empty directories.
    /// Identity and metadata must be charged by the store, not this path layer.
    pub(crate) fn walk_tree(&self, maximum: usize) -> Result<Vec<CacheNode>> {
        if maximum == 0 {
            return Err(Error::Limit("cache inventory limit is zero".into()));
        }
        let root = self.open_dir("", false)?;
        let mut nodes = vec![node(".".into(), &root)?];
        let mut pending = vec![(String::new(), root)];
        let mut path_bytes = 1usize;
        while let Some((prefix, directory)) = pending.pop() {
            for entry in os(fs::Dir::read_from(&directory))? {
                let entry = os(entry)?;
                let name = entry
                    .file_name()
                    .to_str()
                    .map_err(|_| Error::Malformed("non-UTF8 cache entry".into()))?;
                if name == "." || name == ".." {
                    continue;
                }
                if nodes.len() == maximum {
                    return Err(Error::Limit("cache inventory entry limit".into()));
                }
                let length = prefix
                    .len()
                    .checked_add(usize::from(!prefix.is_empty()))
                    .and_then(|n| n.checked_add(name.len()))
                    .ok_or_else(|| Error::Limit("cache inventory path arithmetic".into()))?;
                reserve_path_bytes(&mut path_bytes, length)?;
                let path = if prefix.is_empty() {
                    name.to_owned()
                } else {
                    format!("{prefix}/{name}")
                };
                validate_relative_path(&path)?;
                let before = os(fs::statat(&directory, name, AtFlags::SYMLINK_NOFOLLOW))?;
                let kind = FileType::from_raw_mode(before.st_mode);
                let file = match kind {
                    FileType::Directory => child_dir(&directory, OsStr::new(name), false)?,
                    FileType::RegularFile => File::from(os(fs::openat(
                        &directory,
                        name,
                        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
                        Mode::empty(),
                    ))?),
                    _ => {
                        return Err(Error::State(
                            "symlink/special cache inventory entry refused".into(),
                        ));
                    }
                };
                let metadata = file.metadata()?;
                if metadata.dev() != before.st_dev || metadata.ino() != before.st_ino {
                    return Err(Error::State("cache entry changed during inventory".into()));
                }
                if kind == FileType::Directory {
                    private_dir(&file)?;
                    nodes.push(node(path.clone(), &file)?);
                    pending.push((path, file));
                } else {
                    private_file(&file)?;
                    nodes.push(node(path, &file)?);
                }
            }
        }
        nodes.sort_unstable_by(|a, b| a.path.cmp(&b.path));
        Ok(nodes)
    }
    /// Stable names are internal ASCII keys, not provider/repository paths.
    /// Cache APIs must never unlink or replace a lock inode.
    pub(crate) fn lock(
        &self,
        name: &str,
        timeout: Duration,
        cancel: &AtomicBool,
    ) -> Result<CacheLock> {
        if name.is_empty()
            || name.len() > 128
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
            || matches!(name, "." | "..")
        {
            return Err(Error::Malformed("invalid cache lock key".into()));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::State("cache lock wait canceled".into()));
        }
        let parent = self.open_dir("locks", true)?;
        let file = File::from(os(fs::openat(
            &parent,
            name,
            OFlags::RDWR | OFlags::CREATE | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            Mode::RUSR | Mode::WUSR,
        ))?);
        let lock = CacheLock {
            parent,
            file,
            name: name.into(),
        };
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| Error::Limit("cache lock deadline overflow".into()))?;
        loop {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::State("cache lock wait canceled".into()));
            }
            lock.ensure_bound()?;
            match fs::flock(&lock.file, FlockOperation::NonBlockingLockExclusive) {
                Ok(()) => {
                    lock.ensure_bound()?;
                    return Ok(lock);
                }
                Err(rustix::io::Errno::WOULDBLOCK | rustix::io::Errno::INTR) => {}
                Err(error) => return Err(std::io::Error::from(error).into()),
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Error::Limit("cache lock wait deadline exceeded".into()));
            }
            std::thread::sleep(remaining.min(Duration::from_millis(5)));
        }
    }
}
fn reserve_path_bytes(total: &mut usize, length: usize) -> Result<()> {
    // Conservatively retain two copies even for files and already popped
    // directories; bounds all node strings plus pending-directory clones.
    *total = length
        .checked_mul(2)
        .and_then(|bytes| total.checked_add(bytes))
        .filter(|bytes| *bytes <= MAX_INVENTORY_PATH_BYTES)
        .ok_or_else(|| Error::Limit("cache inventory aggregate path bytes".into()))?;
    Ok(())
}
fn node(path: String, file: &File) -> Result<CacheNode> {
    let metadata = file.metadata()?;
    Ok(CacheNode {
        path,
        bytes: metadata.len(),
        dev: metadata.dev(),
        ino: metadata.ino(),
        allocated_bytes: metadata
            .blocks()
            .checked_mul(512)
            .ok_or_else(|| Error::Limit("cache allocated byte count".into()))?,
        is_directory: metadata.is_dir(),
    })
}

/// Closing the sole lock descriptor releases the OS advisory lock. No clone,
/// PID inference, explicit inode deletion or unlink-based authority is exposed.
pub(crate) struct CacheLock {
    parent: File,
    file: File,
    name: String,
}
impl CacheLock {
    pub(crate) fn ensure_bound(&self) -> Result<()> {
        private_dir(&self.parent)?;
        private_file(&self.file)?;
        let metadata = self.file.metadata()?;
        let named = os(fs::statat(
            &self.parent,
            self.name.as_str(),
            AtFlags::SYMLINK_NOFOLLOW,
        ))?;
        if metadata.mode() & 0o777 != 0o600
            || metadata.nlink() != 1
            || named.st_dev != metadata.dev()
            || named.st_ino != metadata.ino()
            || FileType::from_raw_mode(named.st_mode) != FileType::RegularFile
        {
            return Err(Error::State("cache lock inode binding changed".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::sync::atomic::AtomicU64;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "zerogravity-cache-path-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    #[test]
    fn xdg_resolution_uses_explicit_values_without_environment_mutation() {
        let home = OsStr::new("/home/cache-fixture");
        for xdg in [None, Some(OsStr::new("")), Some(OsStr::new("relative"))] {
            assert_eq!(
                resolve_root(None, xdg, Some(home)).unwrap(),
                Path::new("/home/cache-fixture/.cache/abird/zerogravity/models")
            );
        }
        assert_eq!(
            resolve_root(None, Some(OsStr::new("/cache")), None).unwrap(),
            Path::new("/cache/abird/zerogravity/models")
        );
        assert_eq!(
            resolve_root(
                Some(Path::new("/explicit")),
                Some(OsStr::new("/cache")),
                Some(home)
            )
            .unwrap(),
            Path::new("/explicit")
        );
        assert!(resolve_root(Some(Path::new("relative")), None, Some(home)).is_err());
        assert!(resolve_root(None, None, Some(OsStr::new("relative"))).is_err());
    }
    #[test]
    fn logical_paths_and_digest_have_no_implicit_normalization() {
        for bad in [
            "", "/abs", "a//b", "a/", ".", "..", "a/../b", "a\\b", "a\0b", "a\nb",
        ] {
            assert!(validate_relative_path(bad).is_err(), "{bad:?}");
        }
        assert!(validate_relative_path(&"x".repeat(256)).is_err());
        assert!(validate_relative_path(&vec!["x"; 65].join("/")).is_err());
        for good in ["model.gguf", "A/config.json", "β/模型", "a..b"] {
            validate_relative_path(good).unwrap();
        }
        assert!(validate_digest(&"a".repeat(64)).is_ok());
        assert!(validate_digest(&"A".repeat(64)).is_err());
        assert!(validate_digest(&"a".repeat(63)).is_err());
    }
    #[test]
    fn pinned_capability_survives_name_replacement_without_following_it() {
        let fixture = Fixture::new();
        let path = fixture.0.join("cache");
        let root = RootDir::open(&path, true).unwrap();
        let identity = root.identity();
        let retired = fixture.0.join("retired");
        std::fs::rename(&path, &retired).unwrap();
        std::fs::create_dir(fixture.0.join("outside")).unwrap();
        symlink(fixture.0.join("outside"), &path).unwrap();
        assert!(RootDir::open(&path, false).is_err());
        root.open_file(
            "object",
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        )
        .unwrap()
        .write_all(b"capability")
        .unwrap();
        assert_eq!(root.identity(), identity);
        assert_eq!(
            std::fs::read(retired.join("object")).unwrap(),
            b"capability"
        );
        assert!(!fixture.0.join("outside/object").exists());
    }
    #[test]
    fn inventory_counts_empty_directories_and_refuses_symlinks() {
        let fixture = Fixture::new();
        let root = RootDir::open(&fixture.0, false).unwrap();
        root.open_dir("empty/nested", true).unwrap();
        let nodes = root.walk_tree(3).unwrap();
        assert_eq!(nodes.len(), 3);
        assert!(nodes.iter().all(|n| n.is_directory));
        assert!(nodes.iter().any(|n| n.path == "."));
        assert!(root.walk_tree(2).is_err());
        symlink("empty", fixture.0.join("redirect")).unwrap();
        assert!(root.open_dir("redirect/nested", false).is_err());
        assert!(root.walk_tree(10).is_err());
    }
    #[test]
    fn lock_wait_deadline_cancel_and_inode_binding_are_real() {
        let fixture = Fixture::new();
        let root = RootDir::open(&fixture.0, false).unwrap();
        let cancel = AtomicBool::new(false);
        let first = root
            .lock("policy", Duration::from_secs(1), &cancel)
            .unwrap();
        let metadata = first.file.metadata().unwrap();
        assert!(matches!(
            root.lock("policy", Duration::from_millis(1), &cancel),
            Err(Error::Limit(_))
        ));
        assert!(
            root.lock("other", Duration::from_secs(1), &AtomicBool::new(true))
                .is_err()
        );
        drop(first);
        let again = root
            .lock("policy", Duration::from_secs(1), &cancel)
            .unwrap();
        assert_eq!(again.file.metadata().unwrap().ino(), metadata.ino());
        std::fs::rename(
            fixture.0.join("locks/policy"),
            fixture.0.join("locks/retired-policy"),
        )
        .unwrap();
        assert!(again.ensure_bound().is_err());
    }
    #[test]
    fn preexisting_hardlink_is_never_opened_for_truncation() {
        let fixture = Fixture::new();
        let root = RootDir::open(&fixture.0, false).unwrap();
        let original = fixture.0.join("original");
        std::fs::write(&original, b"preserved").unwrap();
        std::fs::set_permissions(&original, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::hard_link(&original, fixture.0.join("substituted")).unwrap();
        assert!(
            root.open_file("substituted", OFlags::WRONLY | OFlags::TRUNC, Mode::empty())
                .is_err()
        );
        assert!(
            root.open_file(
                "substituted",
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
                Mode::RUSR | Mode::WUSR
            )
            .is_err()
        );
        assert_eq!(std::fs::read(original).unwrap(), b"preserved");
    }
    #[test]
    fn inventory_path_reservation_is_bounded_before_clone_and_atomic_on_error() {
        let mut total = MAX_INVENTORY_PATH_BYTES - 2;
        reserve_path_bytes(&mut total, 1).unwrap();
        assert_eq!(total, MAX_INVENTORY_PATH_BYTES);
        assert!(reserve_path_bytes(&mut total, 1).is_err());
        assert_eq!(total, MAX_INVENTORY_PATH_BYTES);
        assert!(reserve_path_bytes(&mut total, usize::MAX).is_err());
        assert_eq!(total, MAX_INVENTORY_PATH_BYTES);
    }
}
