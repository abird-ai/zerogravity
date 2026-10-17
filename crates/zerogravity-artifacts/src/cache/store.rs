use super::paths::{
    CacheLock, CacheNode, RootDir, resolve_root, validate_digest, validate_relative_path,
};
#[cfg(feature = "diagnostic-cache")]
use super::{CacheHook, CacheMoment, CacheOperation, CachePoint};
use super::{
    CacheOptions, CommitOutcome, DigestProvenance, DiskUsage, FileRole, FilesystemProfile,
    Manifest, ManifestEntry, ManifestSource, PublishedSnapshot, SnapshotLease, TransferAdmission,
};
use crate::{Error, Result};
use rustix::fs::{AtFlags, Mode, OFlags, RenameFlags};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};

const UNIT: u64 = 4096;
const MAX_NODES: usize = 131_072;
const CONTROL_LIMIT: usize = 4096;

#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Policy {
    schema: String,
    disk_limit_bytes: u64,
    block_allowance: u64,
    inode_allowance: u64,
    max_nodes: usize,
    filesystem_block_bytes: u64,
    filesystem_fragment_bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reservation {
    schema: String,
    remaining: u64,
    snapshot: String,
    objects: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    schema: String,
    name: String,
    snapshot: String,
}

struct Inner {
    root: RootDir,
    options: CacheOptions,
    filesystem: FilesystemProfile,
}

#[derive(Clone)]
pub struct CacheStore {
    inner: Arc<Inner>,
    #[cfg(feature = "diagnostic-cache")]
    hook: Option<Arc<dyn CacheHook>>,
}

pub struct CacheTransaction {
    store: CacheStore,
    manifest: Manifest,
    digest: String,
    attempt: String,
    complete: BTreeSet<String>,
    cancel: Arc<AtomicBool>,
    buffer: Vec<u8>,
    // Keep ownership until cleanup has finished; kernel lock ownership, not a
    // PID test, determines whether an abandoned reservation may be recovered.
    owner_lock: Option<CacheLock>,
    _memory: GlobalResourceLease,
}

/// An exact listed length is required before reservation. A missing raw SHA256
/// is first-seen content, not permission for an unbounded transfer.
#[derive(Clone, Debug)]
pub struct DiscoveredEntry {
    path: String,
    bytes: u64,
    role: FileRole,
    sha256: Option<String>,
}
impl DiscoveredEntry {
    pub fn new(
        path: impl Into<String>,
        bytes: u64,
        role: FileRole,
        sha256: Option<String>,
    ) -> Result<Self> {
        let path = path.into();
        validate_relative_path(&path)?;
        if let Some(hash) = &sha256 {
            validate_digest(hash)?;
        }
        Ok(Self {
            path,
            bytes,
            role,
            sha256,
        })
    }
}

/// Same owned attempt, reservation, publisher and cleanup as A1. Planning
/// digests are private accounting keys: this type has no way to publish them.
/// Only the final actual-byte Manifest can reach CacheTransaction::commit.
pub struct DiscoveredTransaction {
    inner: CacheTransaction,
    entries: Vec<DiscoveredEntry>,
    verified: BTreeMap<String, ManifestEntry>,
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::Limit("cache accounting overflow".into()))
}

fn charge(bytes: u64) -> Result<u64> {
    let rounded = bytes
        .checked_add(UNIT - 1)
        .ok_or_else(|| Error::Limit("cache size overflow".into()))?
        / UNIT
        * UNIT;
    add(rounded, UNIT)
}

// At most MAX_NODES borrowed pointers, independently bounded from inode
// payloads. Sorting avoids an allocator-dependent B-tree per cached inode.
fn inode_charge<'a>(nodes: impl Iterator<Item = &'a CacheNode>) -> Result<u64> {
    let mut ordered: Vec<_> = nodes.collect();
    ordered.sort_unstable_by_key(|node| (node.dev, node.ino));
    let mut total = 0;
    let mut previous = None;
    let mut largest = 0;
    for node in ordered {
        let key = (node.dev, node.ino);
        if previous != Some(key) {
            total = add(total, largest)?;
            largest = 0;
            previous = Some(key);
        }
        largest = largest.max(charge(node.bytes)?.max(add(node.allocated_bytes, UNIT)?));
    }
    add(total, largest)
}

fn io(error: rustix::io::Errno) -> Error {
    Error::Io(error.into())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::State("cache operation canceled".into()))
    } else {
        Ok(())
    }
}

fn reference_key(name: &str) -> Result<String> {
    if name.is_empty() || name.len() > 4096 || name.chars().any(char::is_control) {
        return Err(Error::Malformed("invalid cache reference name".into()));
    }
    Ok(format!("refs/{}.json", digest(name.as_bytes())))
}

fn checked_filesystem(block_bytes: u64, fragment_bytes: u64) -> Result<FilesystemProfile> {
    if block_bytes == 0
        || fragment_bytes == 0
        || block_bytes > UNIT
        || fragment_bytes > UNIT
        || !block_bytes.is_power_of_two()
        || !fragment_bytes.is_power_of_two()
    {
        return Err(Error::Unsupported(
            "A1 cache quota requires a local filesystem allocation profile at most 4096 bytes"
                .into(),
        ));
    }
    Ok(FilesystemProfile {
        block_bytes,
        fragment_bytes,
    })
}

// A cooperative hardlink changes ctime/nlink without changing payload. Verify
// the bounded observed bytes and opened inode/length/mtime, not link metadata.
// This does not freeze a mutable same-UID inode: A3 must hash its owned copy.
// The private generic observer enables deterministic unit interleavings; the
// production instantiation is a zero-sized no-op, with no public hook/runtime.
fn verify_opened_file(
    mut file: File,
    expected: &str,
    bytes: u64,
    mut after_chunk: impl FnMut(&File) -> Result<()>,
) -> Result<File> {
    let before = file.metadata()?;
    if before.len() != bytes {
        return Err(Error::Malformed("cache object length mismatch".into()));
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut remaining = bytes;
    while remaining != 0 {
        let count = buffer.len().min(remaining.min(usize::MAX as u64) as usize);
        let read = file.read(&mut buffer[..count])?;
        if read == 0 {
            return Err(Error::Malformed(
                "cache object truncated while verifying".into(),
            ));
        }
        hasher.update(&buffer[..read]);
        remaining -= read as u64;
        after_chunk(&file)?;
    }
    if file.read(&mut buffer[..1])? != 0 {
        return Err(Error::Malformed("cache object grew while verifying".into()));
    }
    let after = file.metadata()?;
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
        || format!("{:x}", hasher.finalize()) != expected
    {
        return Err(Error::Malformed(
            "cache object digest or identity mismatch".into(),
        ));
    }
    file.seek(SeekFrom::Start(0))?;
    Ok(file)
}

impl CacheStore {
    pub fn open(options: CacheOptions) -> Result<Self> {
        options.limits.validate()?;
        if options.disk_limit_bytes == 0
            || options.buffer_bytes == 0
            || options.buffer_bytes > 1024 * 1024
            || options.lock_timeout.is_zero()
        {
            return Err(Error::Limit(
                "invalid cache quota, buffer or lock deadline".into(),
            ));
        }
        let path = resolve_root(
            options.root.as_deref(),
            std::env::var_os("XDG_CACHE_HOME").as_deref(),
            std::env::var_os("HOME").as_deref(),
        )?;
        let root = RootDir::open(&path, true)?;
        let stat = rustix::fs::fstatvfs(root.as_fd()).map_err(io)?;
        let filesystem = checked_filesystem(stat.f_bsize, stat.f_frsize)?;
        for name in [
            "locks",
            "blobs/sha256",
            "manifests",
            "snapshots",
            "refs",
            "reference-records",
            "staging",
        ] {
            root.open_dir(name, true)?;
        }
        let result = Self {
            inner: Arc::new(Inner {
                root,
                options,
                filesystem,
            }),
            #[cfg(feature = "diagnostic-cache")]
            hook: None,
        };
        let cancel = AtomicBool::new(false);
        let lock = result.lock("publication", &cancel)?;
        let expected = Policy {
            schema: "zerogravity.cache.policy.v1".into(),
            disk_limit_bytes: result.inner.options.disk_limit_bytes,
            block_allowance: UNIT,
            inode_allowance: UNIT,
            max_nodes: MAX_NODES,
            filesystem_block_bytes: filesystem.block_bytes,
            filesystem_fragment_bytes: filesystem.fragment_bytes,
        };
        match result.read("policy.json", CONTROL_LIMIT) {
            Ok(bytes) => {
                let actual: Policy =
                    serde_json::from_slice(&bytes).map_err(|e| Error::Malformed(e.to_string()))?;
                if actual != expected {
                    return Err(Error::State(
                        "cache policy differs from persistent root policy".into(),
                    ));
                }
            }
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                result.write_new(
                    "policy.json",
                    &serde_json::to_vec(&expected).map_err(|e| Error::Malformed(e.to_string()))?,
                )?;
                result.sync_dir("")?;
            }
            Err(e) => return Err(e),
        }
        lock.ensure_bound()?;
        let usage = result.usage_locked()?;
        if add(usage.used_bytes, usage.reserved_bytes)? > usage.limit_bytes {
            return Err(Error::Limit(
                "cache already exceeds configured quota".into(),
            ));
        }
        Ok(result)
    }

    #[cfg(feature = "diagnostic-cache")]
    pub fn with_hook(mut self, hook: Arc<dyn CacheHook>) -> Self {
        self.hook = Some(hook);
        self
    }

    #[cfg(feature = "diagnostic-cache")]
    fn point(
        &self,
        operation: CacheOperation,
        moment: CacheMoment,
        object: Option<usize>,
    ) -> Result<()> {
        if let Some(hook) = &self.hook {
            hook.point(CachePoint {
                operation,
                moment,
                object,
            })?;
        }
        Ok(())
    }

    fn lock(&self, name: &str, cancel: &AtomicBool) -> Result<CacheLock> {
        self.inner
            .root
            .lock(name, self.inner.options.lock_timeout, cancel)
    }

    fn read(&self, path: &str, limit: usize) -> Result<Vec<u8>> {
        let file = self
            .inner
            .root
            .open_file(path, OFlags::RDONLY, Mode::empty())?;
        let mut bytes = Vec::new();
        file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
        if bytes.len() > limit {
            return Err(Error::Limit("cache metadata exceeds bound".into()));
        }
        Ok(bytes)
    }

    fn write_new(&self, path: &str, bytes: &[u8]) -> Result<()> {
        let mut file = self.inner.root.open_file(
            path,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        )?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }

    fn sync_dir(&self, path: &str) -> Result<()> {
        if path.is_empty() {
            rustix::fs::fsync(self.inner.root.as_fd()).map_err(io)?;
            return Ok(());
        }
        self.inner.root.open_dir(path, false)?.sync_all()?;
        Ok(())
    }

    fn unlink(&self, path: &str, directory: bool) -> Result<()> {
        let (parent, leaf) = self.inner.root.parent_dir(path, false)?;
        rustix::fs::unlinkat(
            &parent,
            leaf.as_str(),
            if directory {
                AtFlags::REMOVEDIR
            } else {
                AtFlags::empty()
            },
        )
        .map_err(io)
    }

    fn link(&self, from: &str, to: &str) -> Result<()> {
        let (source, a) = self.inner.root.parent_dir(from, false)?;
        let (target, b) = self.inner.root.parent_dir(to, true)?;
        rustix::fs::linkat(&source, a.as_str(), &target, b.as_str(), AtFlags::empty()).map_err(io)
    }

    fn rename(&self, from: &str, to: &str, replace: bool) -> Result<()> {
        let (source, a) = self.inner.root.parent_dir(from, false)?;
        let (target, b) = self.inner.root.parent_dir(to, true)?;
        rustix::fs::renameat_with(
            &source,
            a.as_str(),
            &target,
            b.as_str(),
            if replace {
                RenameFlags::empty()
            } else {
                RenameFlags::NOREPLACE
            },
        )
        .map_err(io)
    }

    fn verify_file(&self, path: &str, expected: &str, bytes: u64) -> Result<File> {
        let file = self
            .inner
            .root
            .open_file(path, OFlags::RDONLY, Mode::empty())?;
        verify_opened_file(file, expected, bytes, |_| Ok(()))
    }

    /// Conservative cooperative allocation accounting: count each physical
    /// inode once, at least rounded logical bytes, observed allocated blocks,
    /// plus fixed metadata allowance. It does not reserve filesystem free space.
    fn usage_locked(&self) -> Result<DiskUsage> {
        let nodes = self.inner.root.walk_tree(MAX_NODES)?;
        let used = inode_charge(nodes.iter())?;
        let mut reserved = 0;
        for node in &nodes {
            if !node.is_directory
                && node.path.starts_with("staging/")
                && node.path.ends_with("/reservation.json")
            {
                let parts: Vec<_> = node.path.split('/').collect();
                if parts.len() != 3 {
                    return Err(Error::Malformed("foreign cache reservation path".into()));
                }
                let attempt_lock = format!("attempt-{}", parts[1]);
                match self
                    .inner
                    .root
                    .lock(&attempt_lock, Duration::ZERO, &AtomicBool::new(false))
                {
                    Ok(guard) => {
                        guard.ensure_bound()?;
                    } // dead owner's future reservation ends; actual bytes stay charged
                    Err(Error::Limit(_)) => {
                        let record: Reservation = serde_json::from_slice(&self.read(
                            &node.path,
                            self.inner.options.limits.max_manifest_bytes + CONTROL_LIMIT,
                        )?)
                        .map_err(|e| Error::Malformed(e.to_string()))?;
                        if record.schema != "zerogravity.cache.reservation.v1"
                            || record.objects.len() > self.inner.options.limits.max_files
                        {
                            return Err(Error::Malformed("unknown cache reservation".into()));
                        }
                        validate_digest(&record.snapshot)?;
                        let mut paths = BTreeSet::from([format!("locks/{attempt_lock}")]);
                        for object in &record.objects {
                            validate_digest(object)?;
                            paths.insert(format!("blobs/sha256/{object}"));
                            paths.insert(format!("locks/object-{object}"));
                        }
                        let staging = format!("staging/{}", parts[1]);
                        let staging_prefix = staging.clone() + "/";
                        let consumed = inode_charge(nodes.iter().filter(|allocated| {
                            allocated.path == staging
                                || allocated.path.starts_with(&staging_prefix)
                                || paths.contains(&allocated.path)
                        }))?;
                        // Inventory and reservation use the same observed inode
                        // cohort. Existing/shared blobs consume no future space;
                        // partially written files retain their remaining peak.
                        // Final snapshot/manifest metadata NEVER credits a new
                        // attempt: it still builds its own private directory and
                        // manifest copy, even on an identical-snapshot cache hit.
                        // Post-install double charging until Drop is conservative.
                        reserved = add(reserved, record.remaining.saturating_sub(consumed))?;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(DiskUsage {
            used_bytes: used,
            reserved_bytes: reserved,
            limit_bytes: self.inner.options.disk_limit_bytes,
        })
    }

    /// Identity of the pinned root capability, independent of later pathname
    /// replacement. It is not a model/source identity or disk-free attestation.
    pub fn root_identity(&self) -> (u64, u64) {
        self.inner.root.identity()
    }

    pub fn filesystem_profile(&self) -> FilesystemProfile {
        self.inner.filesystem
    }

    pub fn usage(&self) -> Result<DiskUsage> {
        let lock = self.lock("publication", &AtomicBool::new(false))?;
        lock.ensure_bound()?;
        self.usage_locked()
    }

    /// Bounded offline listing of complete immutable snapshots. This does not
    /// confer model readiness or authenticate a cached first-seen upstream.
    pub fn list_snapshots(&self) -> Result<Vec<String>> {
        let lock = self.lock("publication", &AtomicBool::new(false))?;
        lock.ensure_bound()?;
        let mut result = Vec::new();
        for node in self.inner.root.walk_tree(MAX_NODES)? {
            if let Some(name) = node.path.strip_prefix("snapshots/")
                && node.is_directory
                && !name.contains('/')
            {
                validate_digest(name)?;
                // A complete snapshot always has a valid bound manifest;
                // malformed/incomplete entries refuse rather than look ready.
                let manifest = Manifest::from_json(
                    &self.read(
                        &format!("snapshots/{name}/manifest.json"),
                        self.inner.options.limits.max_manifest_bytes,
                    )?,
                    &self.inner.options.limits,
                )?;
                if manifest.digest()? != name {
                    return Err(Error::Malformed(
                        "snapshot manifest identity mismatch".into(),
                    ));
                }
                result.push(name.to_string());
            }
        }
        result.sort_unstable();
        Ok(result)
    }

    pub fn begin_discovered(
        &self,
        source: ManifestSource,
        mut entries: Vec<DiscoveredEntry>,
        admission: &TransferAdmission,
        cancel: Arc<AtomicBool>,
    ) -> Result<DiscoveredTransaction> {
        check_cancel(&cancel)?;
        let memory = self.admit_transfer(admission)?;
        if entries.is_empty() || entries.len() > self.inner.options.limits.max_files {
            return Err(Error::Limit("discovered cache file count".into()));
        }
        entries.sort_unstable_by(|a, b| a.path.cmp(&b.path));
        let mut used: BTreeSet<String> = entries.iter().filter_map(|e| e.sha256.clone()).collect();
        let mut next = 0u64;
        let mut planned = Vec::new();
        for entry in &entries {
            let key = match &entry.sha256 {
                Some(hash) => hash.clone(),
                None => loop {
                    let key = format!("{next:064x}");
                    next = next
                        .checked_add(1)
                        .ok_or_else(|| Error::Limit("planning keys exhausted".into()))?;
                    if used.insert(key.clone()) {
                        break key;
                    }
                },
            };
            // Distinct unknown files reserve worst-case bytes independently.
            // The final fixed-width digest and provenance occupy exactly the
            // same metadata envelope as this non-publishable planned entry.
            planned.push(
                ManifestEntry::new(&entry.path, key, entry.bytes, entry.role)?
                    .with_digest_provenance(if entry.sha256.is_some() {
                        DigestProvenance::TrustedSha256
                    } else {
                        DigestProvenance::FirstSeen
                    }),
            );
        }
        let manifest = Manifest::new(source, planned, &self.inner.options.limits)?;
        let known_objects = entries
            .iter()
            .filter_map(|entry| entry.sha256.clone())
            .collect();
        let inner = self.begin_admitted(manifest, cancel, memory, Some(known_objects))?;
        Ok(DiscoveredTransaction {
            inner,
            entries,
            verified: BTreeMap::new(),
        })
    }

    pub fn begin(
        &self,
        manifest: Manifest,
        admission: &TransferAdmission,
        cancel: Arc<AtomicBool>,
    ) -> Result<CacheTransaction> {
        check_cancel(&cancel)?;
        let memory = self.admit_transfer(admission)?;
        self.begin_admitted(manifest, cancel, memory, None)
    }

    fn admit_transfer(&self, admission: &TransferAdmission) -> Result<GlobalResourceLease> {
        // Reserve a conservative operation-local envelope before creating
        // canonical/parsed control copies or a transfer buffer. Inventory is
        // independently bounded to 8MiB of reserved path characters (including
        // pending clones) and MAX_NODES nodes: <=16MiB node storage at our
        // checked128B ceiling, <=12MiB String capacity allowance, <=8MiB pending
        // directory work, <=2MiB sorted inode references, and <=4MiB planned
        // object/lock keys. 64MiB rounds these logical allocations up with
        // headroom. Another8x configured manifest bound covers serialization,
        // parsing, reservations and retained source/control copies. Allocator
        // overhead/kernel memory are not a global RAM or UMA census.
        if std::mem::size_of::<CacheNode>() > 128 {
            return Err(Error::Unsupported(
                "cache inventory node exceeds qualified control envelope".into(),
            ));
        }
        let envelope = add(
            add(
                64 * 1024 * 1024,
                (self.inner.options.limits.max_manifest_bytes as u64)
                    .checked_mul(8)
                    .ok_or_else(|| Error::Limit("cache control envelope overflow".into()))?,
            )?,
            add(self.inner.options.buffer_bytes as u64, 64 * 1024)?,
        )?;
        admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: envelope,
            }],
        )
    }

    fn begin_admitted(
        &self,
        manifest: Manifest,
        cancel: Arc<AtomicBool>,
        memory: GlobalResourceLease,
        reservation_objects: Option<Vec<String>>,
    ) -> Result<CacheTransaction> {
        // Revalidate against this store's bounds when a caller used wider
        // limits. No file mutation or input read precedes this validation.
        let manifest =
            Manifest::from_json(&manifest.canonical_bytes()?, &self.inner.options.limits)?;
        let digest = manifest.digest()?;
        for entry in manifest.entries() {
            validate_relative_path(&format!(
                "staging/{}-{}-{}/snapshot/files/{}",
                u32::MAX,
                u128::MAX,
                u64::MAX,
                entry.path()
            ))?;
        }
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(self.inner.options.buffer_bytes)
            .map_err(|_| Error::Limit("cache transfer buffer allocation refused".into()))?;
        buffer.resize(self.inner.options.buffer_bytes, 0);
        let mut peak = 64 * UNIT;
        let mut objects = BTreeSet::new();
        for entry in manifest.entries() {
            if objects.insert(entry.digest()) {
                peak = add(peak, charge(entry.bytes())?)?;
            }
            peak = add(
                peak,
                (entry.path().split('/').count() as u64 + 3) * 2 * UNIT,
            )?;
        }
        peak = add(
            peak,
            charge(manifest.canonical_bytes()?.len() as u64)?
                .checked_mul(3)
                .ok_or_else(|| Error::Limit("cache metadata peak overflow".into()))?,
        )?;
        let quota = self.lock("publication", &cancel)?;
        quota.ensure_bound()?;
        let usage = self.usage_locked()?;
        if add(add(usage.used_bytes, usage.reserved_bytes)?, peak)? > usage.limit_bytes {
            return Err(Error::Limit("cache disk peak admission refused".into()));
        }
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let nonce = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Limit("cache attempt identifiers exhausted".into()))?;
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Error::State("cache clock precedes epoch".into()))?
            .as_nanos();
        let attempt = format!("{}-{time}-{nonce}", std::process::id());
        let owner_lock = self.lock(&format!("attempt-{attempt}"), &cancel)?;
        self.inner
            .root
            .open_dir(&format!("staging/{attempt}"), true)?;
        let record = Reservation {
            schema: "zerogravity.cache.reservation.v1".into(),
            remaining: peak,
            snapshot: digest.clone(),
            // Discovery keys must never credit an unrelated named blob.
            // Only known upstream digests enter this cohort; discovered bytes
            // remain observed through their original staging inode links.
            objects: reservation_objects
                .unwrap_or_else(|| objects.into_iter().map(str::to_string).collect()),
        };
        let mut reservation_file = self.inner.root.open_file(
            &format!("staging/{attempt}/reservation.json"),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        )?;
        reservation_file.write_all(
            &serde_json::to_vec(&record).map_err(|e| Error::Malformed(e.to_string()))?,
        )?;
        #[cfg(feature = "diagnostic-cache")]
        self.point(
            CacheOperation::ReservationFileSync,
            CacheMoment::Before,
            None,
        )?;
        reservation_file.sync_all()?;
        #[cfg(feature = "diagnostic-cache")]
        self.point(
            CacheOperation::ReservationFileSync,
            CacheMoment::After,
            None,
        )?;
        #[cfg(feature = "diagnostic-cache")]
        self.point(
            CacheOperation::ReservationDirectorySync,
            CacheMoment::Before,
            None,
        )?;
        self.sync_dir(&format!("staging/{attempt}"))?;
        #[cfg(feature = "diagnostic-cache")]
        self.point(
            CacheOperation::ReservationDirectorySync,
            CacheMoment::After,
            None,
        )?;
        #[cfg(feature = "diagnostic-cache")]
        self.point(
            CacheOperation::StagingDirectorySync,
            CacheMoment::Before,
            None,
        )?;
        self.sync_dir("staging")?;
        #[cfg(feature = "diagnostic-cache")]
        self.point(
            CacheOperation::StagingDirectorySync,
            CacheMoment::After,
            None,
        )?;
        owner_lock.ensure_bound()?;
        Ok(CacheTransaction {
            store: self.clone(),
            manifest,
            digest,
            attempt,
            complete: BTreeSet::new(),
            cancel,
            buffer,
            owner_lock: Some(owner_lock),
            _memory: memory,
        })
    }

    pub fn open_snapshot(&self, digest: &str) -> Result<SnapshotLease> {
        validate_digest(digest)?;
        let manifest = Manifest::from_json(
            &self.read(
                &format!("snapshots/{digest}/manifest.json"),
                self.inner.options.limits.max_manifest_bytes,
            )?,
            &self.inner.options.limits,
        )?;
        if manifest.digest()? != digest {
            return Err(Error::Malformed("snapshot manifest digest mismatch".into()));
        }
        let mut files = BTreeMap::new();
        for entry in manifest.entries() {
            files.insert(
                entry.path().to_string(),
                self.verify_file(
                    &format!("snapshots/{digest}/files/{}", entry.path()),
                    entry.digest(),
                    entry.bytes(),
                )?,
            );
        }
        Ok(SnapshotLease { manifest, files })
    }

    pub fn verify_snapshot(&self, digest: &str) -> Result<SnapshotLease> {
        self.open_snapshot(digest)
    }

    pub fn resolve_ref(&self, name: &str) -> Result<SnapshotLease> {
        let reference: Reference =
            serde_json::from_slice(&self.read(&reference_key(name)?, CONTROL_LIMIT * 2)?)
                .map_err(|e| Error::Malformed(e.to_string()))?;
        if reference.schema != "zerogravity.cache.reference.v1" || reference.name != name {
            return Err(Error::Malformed("cache reference identity mismatch".into()));
        }
        self.open_snapshot(&reference.snapshot)
    }

    pub fn remove_snapshot(&self, _digest: &str) -> Result<()> {
        Err(Error::Unsupported(
            "cache eviction/removal is not qualified in A1".into(),
        ))
    }
}

impl CacheTransaction {
    fn stage(&self, leaf: &str) -> String {
        format!("staging/{}/{}", self.attempt, leaf)
    }

    fn ensure_owner(&self) -> Result<()> {
        check_cancel(&self.cancel)?;
        self.owner_lock
            .as_ref()
            .ok_or_else(|| Error::State("cache transaction closed".into()))?
            .ensure_bound()
    }

    pub fn import_object(&mut self, logical_path: &str, reader: &mut impl Read) -> Result<()> {
        let entry = self
            .manifest
            .entries()
            .iter()
            .find(|e| e.path() == logical_path)
            .ok_or_else(|| Error::Malformed("object not in resolved cache manifest".into()))?;
        let hash = entry.digest().to_string();
        self.import_verified(logical_path, reader, Some(&hash), false)?;
        Ok(())
    }

    fn import_verified(
        &mut self,
        logical_path: &str,
        reader: &mut impl Read,
        expected: Option<&str>,
        retain_stage: bool,
    ) -> Result<ManifestEntry> {
        self.ensure_owner()?;
        let ordinal = self
            .manifest
            .entries()
            .iter()
            .position(|x| x.path() == logical_path)
            .ok_or_else(|| Error::Malformed("object not in resolved cache manifest".into()))?;
        if self.complete.contains(logical_path) {
            return Err(Error::State("cache object already imported".into()));
        }
        let bytes = self.manifest.entries()[ordinal].bytes();
        let role = self.manifest.entries()[ordinal].role();
        let mut object_lock = expected
            .map(|hash| self.store.lock(&format!("object-{hash}"), &self.cancel))
            .transpose()?;
        if let Some(lock) = &object_lock {
            lock.ensure_bound()?;
        }
        if let Some(hash) = expected {
            match self
                .store
                .verify_file(&format!("blobs/sha256/{hash}"), hash, bytes)
            {
                Ok(_) => {
                    self.complete.insert(logical_path.to_string());
                    return ManifestEntry::new(logical_path, hash, bytes, role);
                }
                Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        let temporary = self.stage(&format!("object-{ordinal}"));
        let mut target = self.store.inner.root.open_file(
            &temporary,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        )?;
        let mut remaining = bytes;
        let mut hasher = Sha256::new();
        while remaining != 0 {
            self.ensure_owner()?;
            let count = self
                .buffer
                .len()
                .min(remaining.min(usize::MAX as u64) as usize);
            let read = reader.read(&mut self.buffer[..count])?;
            if read == 0 {
                return Err(Error::Malformed("cache source truncated".into()));
            }
            hasher.update(&self.buffer[..read]);
            target.write_all(&self.buffer[..read])?;
            remaining -= read as u64;
        }
        self.ensure_owner()?;
        if reader.read(&mut self.buffer[..1])? != 0 {
            return Err(Error::Malformed("cache source length mismatch".into()));
        }
        let actual = format!("{:x}", hasher.finalize());
        if expected.is_some_and(|value| value != actual) {
            return Err(Error::Malformed("cache source digest mismatch".into()));
        }
        #[cfg(feature = "diagnostic-cache")]
        self.store.point(
            CacheOperation::ObjectFileSync,
            CacheMoment::Before,
            Some(ordinal),
        )?;
        target.sync_all()?;
        #[cfg(feature = "diagnostic-cache")]
        self.store.point(
            CacheOperation::ObjectFileSync,
            CacheMoment::After,
            Some(ordinal),
        )?;
        if object_lock.is_none() {
            object_lock = Some(self.store.lock(&format!("object-{actual}"), &self.cancel)?);
        }
        let lock = object_lock.as_ref().expect("actual object lock acquired");
        lock.ensure_bound()?;
        self.ensure_owner()?;
        let blob = format!("blobs/sha256/{actual}");
        // Unknown checksums may converge after their single-pass downloads.
        // Verify an existing inode rather than replacing it or trusting its name.
        let exists = if expected.is_none() {
            match self.store.verify_file(&blob, &actual, bytes) {
                Ok(_) => true,
                Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => false,
                Err(e) => return Err(e),
            }
        } else {
            false
        };
        if !exists {
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::ObjectInstall,
                CacheMoment::Before,
                Some(ordinal),
            )?;
            lock.ensure_bound()?;
            self.ensure_owner()?;
            self.store.link(&temporary, &blob)?;
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::ObjectInstall,
                CacheMoment::After,
                Some(ordinal),
            )?;
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::ObjectDirectorySync,
                CacheMoment::Before,
                Some(ordinal),
            )?;
            self.store.sync_dir("blobs/sha256")?;
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::ObjectDirectorySync,
                CacheMoment::After,
                Some(ordinal),
            )?;
        }
        // Discovery reservations do not know the final blob key in advance.
        // Retain the original staging inode link so usage_locked observes and
        // credits its real bytes against that live attempt's reserved peak.
        // If a dedup hit uses another inode, retaining the private copy also
        // keeps its physical temporary bytes counted until exact-owned cleanup.
        if !retain_stage {
            self.store.unlink(&temporary, false)?;
        }
        self.complete.insert(logical_path.to_string());
        ManifestEntry::new(logical_path, actual, bytes, role)
    }

    pub fn commit(self, reference: Option<&str>) -> Result<CommitOutcome> {
        self.ensure_owner()?;
        let reference_path = reference.map(reference_key).transpose()?;
        let reference_bytes = reference
            .map(|name| -> Result<Vec<u8>> {
                let record = Reference {
                    schema: "zerogravity.cache.reference.v1".into(),
                    name: name.to_string(),
                    snapshot: self.digest.clone(),
                };
                let bytes =
                    serde_json::to_vec(&record).map_err(|e| Error::Malformed(e.to_string()))?;
                if bytes.len() > CONTROL_LIMIT * 2 {
                    return Err(Error::Limit("encoded cache reference exceeds bound".into()));
                }
                Ok(bytes)
            })
            .transpose()?;
        if self.complete.len() != self.manifest.entries().len() {
            return Err(Error::State("snapshot is incomplete".into()));
        }
        let manifest_bytes = self.manifest.canonical_bytes()?;
        let stage_manifest = self.stage("manifest.json");
        let mut file = self.store.inner.root.open_file(
            &stage_manifest,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        )?;
        file.write_all(&manifest_bytes)?;
        #[cfg(feature = "diagnostic-cache")]
        self.store
            .point(CacheOperation::ManifestFileSync, CacheMoment::Before, None)?;
        file.sync_all()?;
        #[cfg(feature = "diagnostic-cache")]
        self.store
            .point(CacheOperation::ManifestFileSync, CacheMoment::After, None)?;
        let stage_snapshot = self.stage("snapshot");
        self.store.inner.root.open_dir(&stage_snapshot, true)?;
        self.store
            .link(&stage_manifest, &format!("{stage_snapshot}/manifest.json"))?;
        let mut directories = BTreeSet::from([stage_snapshot.clone()]);
        for (ordinal, entry) in self.manifest.entries().iter().enumerate() {
            self.ensure_owner()?;
            #[cfg(not(feature = "diagnostic-cache"))]
            let _ = ordinal;
            let blob = format!("blobs/sha256/{}", entry.digest());
            let source = self
                .store
                .verify_file(&blob, entry.digest(), entry.bytes())?;
            let destination = format!("{stage_snapshot}/files/{}", entry.path());
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::SnapshotEntryInstall,
                CacheMoment::Before,
                Some(ordinal),
            )?;
            self.store.link(&blob, &destination)?;
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::SnapshotEntryInstall,
                CacheMoment::After,
                Some(ordinal),
            )?;
            let linked =
                self.store
                    .inner
                    .root
                    .open_file(&destination, OFlags::RDONLY, Mode::empty())?;
            if (source.metadata()?.dev(), source.metadata()?.ino())
                != (linked.metadata()?.dev(), linked.metadata()?.ino())
            {
                return Err(Error::State(
                    "blob binding changed while preparing snapshot".into(),
                ));
            }
            let mut parent = destination.rsplit_once('/').expect("internal path").0;
            while parent.starts_with(&stage_snapshot) {
                directories.insert(parent.to_string());
                let Some((next, _)) = parent.rsplit_once('/') else {
                    break;
                };
                parent = next;
            }
        }
        let mut directories: Vec<_> = directories.into_iter().collect();
        directories.sort_by_key(|x| std::cmp::Reverse(x.len()));
        for directory in &directories {
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::SnapshotDirectorySync,
                CacheMoment::Before,
                None,
            )?;
            self.store.sync_dir(directory)?;
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::SnapshotDirectorySync,
                CacheMoment::After,
                None,
            )?;
        }
        let quota = self.store.lock("publication", &self.cancel)?;
        quota.ensure_bound()?;
        self.ensure_owner()?;
        let manifest_target = format!("manifests/{}.json", self.digest);
        #[cfg(feature = "diagnostic-cache")]
        self.store
            .point(CacheOperation::ManifestInstall, CacheMoment::Before, None)?;
        quota.ensure_bound()?;
        match self.store.link(&stage_manifest, &manifest_target) {
            Ok(()) => {}
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if self.store.read(
                    &manifest_target,
                    self.store.inner.options.limits.max_manifest_bytes,
                )? != manifest_bytes
                {
                    return Err(Error::Malformed("immutable manifest collision".into()));
                }
            }
            Err(e) => return Err(e),
        }
        #[cfg(feature = "diagnostic-cache")]
        self.store
            .point(CacheOperation::ManifestInstall, CacheMoment::After, None)?;
        #[cfg(feature = "diagnostic-cache")]
        self.store.point(
            CacheOperation::ManifestDirectorySync,
            CacheMoment::Before,
            None,
        )?;
        self.store.sync_dir("manifests")?;
        #[cfg(feature = "diagnostic-cache")]
        self.store.point(
            CacheOperation::ManifestDirectorySync,
            CacheMoment::After,
            None,
        )?;
        #[cfg(feature = "diagnostic-cache")]
        self.store
            .point(CacheOperation::SnapshotInstall, CacheMoment::Before, None)?;
        quota.ensure_bound()?;
        self.ensure_owner()?;
        match self.store.rename(
            &stage_snapshot,
            &format!("snapshots/{}", self.digest),
            false,
        ) {
            Ok(()) => {}
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                self.store.open_snapshot(&self.digest)?;
            }
            Err(e) => return Err(e),
        }
        #[cfg(feature = "diagnostic-cache")]
        self.store
            .point(CacheOperation::SnapshotInstall, CacheMoment::After, None)?;
        #[cfg(feature = "diagnostic-cache")]
        self.store.point(
            CacheOperation::SnapshotParentSync,
            CacheMoment::Before,
            None,
        )?;
        self.store.sync_dir("snapshots")?;
        #[cfg(feature = "diagnostic-cache")]
        self.store
            .point(CacheOperation::SnapshotParentSync, CacheMoment::After, None)?;
        let published = PublishedSnapshot {
            digest: self.digest.clone(),
            reference: reference.map(str::to_string),
        };
        if let Some(path) = reference_path {
            let temporary = self.stage("reference.json");
            let mut file = self.store.inner.root.open_file(
                &temporary,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
                Mode::RUSR | Mode::WUSR,
            )?;
            file.write_all(reference_bytes.as_deref().expect("paired reference bytes"))?;
            #[cfg(feature = "diagnostic-cache")]
            self.store
                .point(CacheOperation::RefFileSync, CacheMoment::Before, None)?;
            file.sync_all()?;
            #[cfg(feature = "diagnostic-cache")]
            self.store
                .point(CacheOperation::RefFileSync, CacheMoment::After, None)?;
            drop(file);
            // Preserve a named immutable backing for every reference version.
            // Replacing the alias may unlink its old name while a resolver holds
            // that descriptor; retained backing keeps those bytes charged.
            let record_bytes = reference_bytes.as_deref().expect("paired reference bytes");
            let record_digest = digest(record_bytes);
            let record_target = format!("reference-records/{record_digest}.json");
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::ReferenceRecordInstall,
                CacheMoment::Before,
                None,
            )?;
            quota.ensure_bound()?;
            self.ensure_owner()?;
            match self.store.link(&temporary, &record_target) {
                Ok(()) => {}
                Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    self.store.verify_file(
                        &record_target,
                        &record_digest,
                        record_bytes.len() as u64,
                    )?;
                    // Reuse the backing inode itself, not merely equal bytes:
                    // every published alias must retain a named second link.
                    self.store.unlink(&temporary, false)?;
                    self.store.link(&record_target, &temporary)?;
                }
                Err(error) => return Err(error),
            }
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::ReferenceRecordInstall,
                CacheMoment::After,
                None,
            )?;
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::ReferenceRecordDirectorySync,
                CacheMoment::Before,
                None,
            )?;
            self.store.sync_dir("reference-records")?;
            #[cfg(feature = "diagnostic-cache")]
            self.store.point(
                CacheOperation::ReferenceRecordDirectorySync,
                CacheMoment::After,
                None,
            )?;
            self.ensure_owner()?;
            quota.ensure_bound()?;
            #[cfg(feature = "diagnostic-cache")]
            self.store
                .point(CacheOperation::RefRename, CacheMoment::Before, None)?;
            quota.ensure_bound()?;
            self.ensure_owner()?;
            self.store.rename(&temporary, &path, true)?;
            let durability = (|| -> Result<()> {
                #[cfg(feature = "diagnostic-cache")]
                self.store
                    .point(CacheOperation::RefRename, CacheMoment::After, None)?;
                #[cfg(feature = "diagnostic-cache")]
                self.store
                    .point(CacheOperation::RefDirectorySync, CacheMoment::Before, None)?;
                self.store.sync_dir("refs")?;
                #[cfg(feature = "diagnostic-cache")]
                self.store
                    .point(CacheOperation::RefDirectorySync, CacheMoment::After, None)?;
                Ok(())
            })();
            if let Err(cause) = durability {
                return Ok(CommitOutcome::DurabilityUnknown {
                    snapshot: published,
                    reference: reference.map(str::to_string),
                    cause,
                });
            }
        }
        // Reservation is released by exact-owned staging cleanup. Installed
        // unreferenced snapshots/blobs remain in the next inventory charge.
        Ok(CommitOutcome::Durable(published))
    }
}

impl DiscoveredTransaction {
    fn entry(&self, logical_path: &str) -> Result<&DiscoveredEntry> {
        self.entries
            .iter()
            .find(|e| e.path == logical_path)
            .ok_or_else(|| Error::Malformed("object not in acquisition plan".into()))
    }

    pub fn try_cached(&mut self, logical_path: &str) -> Result<Option<ManifestEntry>> {
        self.inner.ensure_owner()?;
        if self.verified.contains_key(logical_path) {
            return Err(Error::State("cache object already imported".into()));
        }
        let entry = self.entry(logical_path)?;
        let Some(hash) = &entry.sha256 else {
            return Ok(None);
        };
        let lock = self
            .inner
            .store
            .lock(&format!("object-{hash}"), &self.inner.cancel)?;
        lock.ensure_bound()?;
        match self
            .inner
            .store
            .verify_file(&format!("blobs/sha256/{hash}"), hash, entry.bytes)
        {
            Ok(_) => {
                let actual = ManifestEntry::new(&entry.path, hash, entry.bytes, entry.role)?
                    .with_digest_provenance(DigestProvenance::TrustedSha256);
                self.inner.complete.insert(logical_path.to_string());
                self.verified
                    .insert(logical_path.to_string(), actual.clone());
                Ok(Some(actual))
            }
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn import_object(
        &mut self,
        logical_path: &str,
        reader: &mut impl Read,
    ) -> Result<ManifestEntry> {
        let expected = self.entry(logical_path)?.sha256.clone();
        let actual = self
            .inner
            .import_verified(logical_path, reader, expected.as_deref(), true)?
            .with_digest_provenance(if expected.is_some() {
                DigestProvenance::TrustedSha256
            } else {
                DigestProvenance::FirstSeen
            });
        self.verified
            .insert(logical_path.to_string(), actual.clone());
        Ok(actual)
    }

    /// Only small already-verified index/control bodies can be materialized.
    /// Model-weight objects never pass through a whole-file RAM buffer.
    pub fn read_object(&self, logical_path: &str, limit: usize) -> Result<Vec<u8>> {
        self.inner.ensure_owner()?;
        if limit > 8 * 1024 * 1024 {
            return Err(Error::Limit("acquisition control object bound".into()));
        }
        let entry = self
            .verified
            .get(logical_path)
            .ok_or_else(|| Error::State("object not verified".into()))?;
        if entry.bytes() > limit as u64 {
            return Err(Error::Limit("acquisition control object too large".into()));
        }
        let mut file = self.inner.store.verify_file(
            &format!("blobs/sha256/{}", entry.digest()),
            entry.digest(),
            entry.bytes(),
        )?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(entry.bytes() as usize)
            .map_err(|_| Error::Limit("control object allocation refused".into()))?;
        Read::by_ref(&mut file)
            .take(entry.bytes() + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 != entry.bytes() || digest(&bytes) != entry.digest() {
            return Err(Error::Malformed(
                "acquisition control content changed".into(),
            ));
        }
        Ok(bytes)
    }

    pub fn read_prefix(&self, logical_path: &str, output: &mut [u8]) -> Result<usize> {
        self.inner.ensure_owner()?;
        if output.len() > 4096 {
            return Err(Error::Limit("acquisition prefix bound".into()));
        }
        let entry = self
            .verified
            .get(logical_path)
            .ok_or_else(|| Error::State("object not verified".into()))?;
        let mut file = self.inner.store.inner.root.open_file(
            &format!("blobs/sha256/{}", entry.digest()),
            OFlags::RDONLY,
            Mode::empty(),
        )?;
        let count = output
            .len()
            .min(entry.bytes().min(usize::MAX as u64) as usize);
        file.read_exact(&mut output[..count])?;
        Ok(count)
    }

    pub fn manifest(&self) -> Result<Manifest> {
        self.inner.ensure_owner()?;
        if self.verified.len() != self.entries.len() {
            return Err(Error::State("acquisition snapshot incomplete".into()));
        }
        Manifest::new(
            self.inner.manifest.source().clone(),
            self.verified.values().cloned().collect(),
            &self.inner.store.inner.options.limits,
        )
    }

    pub fn commit(mut self, reference: Option<&str>) -> Result<CommitOutcome> {
        let manifest = self.manifest()?;
        // Fixed-width real digests and the same provenance fields cannot grow
        // beyond the exact planned metadata reservation. The reservation's
        // snapshot field is only a private accounting identity, never a ref or
        // a promise that a manifest exists. Its objects remain planned keys;
        // actual payload consumption is pinned by retained staging hardlinks.
        if manifest.canonical_bytes()?.len() > self.inner.manifest.canonical_bytes()?.len() {
            return Err(Error::Limit(
                "discovered manifest exceeds planned peak".into(),
            ));
        }
        #[cfg(feature = "diagnostic-cache")]
        self.inner.store.point(
            CacheOperation::DiscoveredManifestReady,
            CacheMoment::Before,
            None,
        )?;
        self.inner.digest = manifest.digest()?;
        self.inner.manifest = manifest;
        #[cfg(feature = "diagnostic-cache")]
        self.inner.store.point(
            CacheOperation::DiscoveredManifestReady,
            CacheMoment::After,
            None,
        )?;
        self.inner.commit(reference)
    }
}

impl Drop for CacheTransaction {
    fn drop(&mut self) {
        // Best-effort cleanup ONLY this unique owned attempt. A failed unlink
        // leaves counted orphans; never infer released bytes from Drop alone.
        let Ok(_quota) = self.store.lock("publication", &AtomicBool::new(false)) else {
            return;
        };
        if self
            .owner_lock
            .as_ref()
            .is_none_or(|x| x.ensure_bound().is_err())
        {
            return;
        }
        let prefix = format!("staging/{}/", self.attempt);
        if let Ok(nodes) = self.store.inner.root.walk_tree(MAX_NODES) {
            let mut owned: Vec<_> = nodes
                .into_iter()
                .filter(|x| x.path.starts_with(&prefix))
                .collect();
            owned.sort_by_key(|x| std::cmp::Reverse(x.path.len()));
            for node in owned {
                let _ = self.store.unlink(&node.path, node.is_directory);
            }
            let _ = self
                .store
                .unlink(&format!("staging/{}", self.attempt), true);
        }
        self.owner_lock.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quota_profile_refuses_unqualified_or_zero_allocation_units() {
        for (block, fragment) in [
            (0, 4096),
            (4096, 0),
            (8192, 4096),
            (4096, 8192),
            (3000, 4096),
        ] {
            assert!(matches!(
                checked_filesystem(block, fragment),
                Err(Error::Unsupported(_))
            ));
        }
        assert_eq!(
            checked_filesystem(4096, 4096).unwrap(),
            FilesystemProfile {
                block_bytes: 4096,
                fragment_bytes: 4096
            }
        );
    }

    struct VerificationFixture {
        directory: std::path::PathBuf,
        path: std::path::PathBuf,
        bytes: Vec<u8>,
    }

    impl VerificationFixture {
        fn new() -> Self {
            use std::os::unix::fs::DirBuilderExt;
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "zerogravity-cache-verify-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&directory)
                .unwrap();
            let path = directory.join("object");
            let bytes = vec![7u8; 128 * 1024];
            std::fs::write(&path, &bytes).unwrap();
            Self {
                directory,
                path,
                bytes,
            }
        }
    }

    impl Drop for VerificationFixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    #[test]
    fn verification_accepts_same_inode_hardlink_during_hash() {
        let fixture = VerificationFixture::new();
        let before = std::fs::metadata(&fixture.path).unwrap();
        let alias = fixture.directory.join("snapshot-link");
        let mut linked = false;
        let file = verify_opened_file(
            File::open(&fixture.path).unwrap(),
            &digest(&fixture.bytes),
            fixture.bytes.len() as u64,
            |_| {
                if !linked {
                    // Cross a kernel coarse-clock tick so the historical ctime
                    // guard demonstrably rejects this unchanged-payload operation.
                    std::thread::sleep(Duration::from_millis(20));
                    std::fs::hard_link(&fixture.path, &alias)?;
                    linked = true;
                }
                Ok(())
            },
        )
        .unwrap();
        let after = file.metadata().unwrap();
        assert!(linked);
        assert_eq!((before.dev(), before.ino()), (after.dev(), after.ino()));
        assert_eq!(after.nlink(), before.nlink() + 1);
        assert_eq!(
            (before.mtime(), before.mtime_nsec()),
            (after.mtime(), after.mtime_nsec())
        );
        assert_ne!(
            (before.ctime(), before.ctime_nsec()),
            (after.ctime(), after.ctime_nsec())
        );
        assert_eq!(std::fs::read(&alias).unwrap(), fixture.bytes);
    }

    #[test]
    fn verification_rejects_changed_payload_with_restored_mtime() {
        let fixture = VerificationFixture::new();
        let before = std::fs::metadata(&fixture.path).unwrap();
        let mut changed = false;
        let result = verify_opened_file(
            File::open(&fixture.path).unwrap(),
            &digest(&fixture.bytes),
            fixture.bytes.len() as u64,
            |_| {
                if !changed {
                    let mut writer = std::fs::OpenOptions::new()
                        .write(true)
                        .open(&fixture.path)?;
                    writer.seek(SeekFrom::Start(64 * 1024))?;
                    writer.write_all(&vec![9u8; 64 * 1024])?;
                    writer.set_times(std::fs::FileTimes::new().set_modified(before.modified()?))?;
                    changed = true;
                }
                Ok(())
            },
        );
        assert!(changed);
        let after = std::fs::metadata(&fixture.path).unwrap();
        assert_eq!(
            (before.mtime(), before.mtime_nsec()),
            (after.mtime(), after.mtime_nsec())
        );
        assert!(matches!(result, Err(Error::Malformed(_))));
    }

    #[test]
    fn verification_bounds_growth_and_truncation_during_hash() {
        for truncate in [false, true] {
            let fixture = VerificationFixture::new();
            let mut changed = false;
            let result = verify_opened_file(
                File::open(&fixture.path).unwrap(),
                &digest(&fixture.bytes),
                fixture.bytes.len() as u64,
                |_| {
                    if !changed {
                        let mut writer = std::fs::OpenOptions::new()
                            .write(true)
                            .open(&fixture.path)?;
                        if truncate {
                            writer.set_len(64 * 1024)?;
                        } else {
                            writer.seek(SeekFrom::End(0))?;
                            writer.write_all(&[0u8; 4096])?;
                        }
                        changed = true;
                    }
                    Ok(())
                },
            );
            assert!(changed);
            assert!(matches!(result, Err(Error::Malformed(_))));
        }
    }
}
