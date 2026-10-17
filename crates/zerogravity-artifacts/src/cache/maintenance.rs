//! Explicit, exact-target reclamation for protected cache roots. The activity
//! barrier excludes every transaction and new managed open before graph capture.
//! Actual payload descriptor locks cover escaped dup/clone/fork readers. Neither
//! this protocol nor its rebuilt cooperative charge is an OS free-space census.

use super::super::{ReclaimMode, ReclaimOutcome, ReclaimReport};
use super::*;

const GRAPH_BYTES: usize = 64 * 1024 * 1024;
// A rename adds one retirement-directory entry before the old directory can
// contract. Match A1's conservative fixed control headroom; this coordinates
// cache charge, rather than reserving global filesystem metadata/free space.
const RETIREMENT_PEAK: u64 = 64 * UNIT;

struct Name {
    path: String,
    dev: u64,
    ino: u64,
    bytes: u64,
    directory: bool,
}

impl Name {
    fn new(node: &CacheNode) -> Self {
        Self {
            path: node.path.clone(),
            dev: node.dev,
            ino: node.ino,
            bytes: node.bytes,
            directory: node.is_directory,
        }
    }
}

struct Prepared {
    root: Option<Name>,
    tree: Vec<Name>,
    blobs: Vec<Name>,
    backing: Option<Name>,
    references: usize,
    retained_blobs: usize,
    retired: bool,
}

#[derive(Clone, Copy)]
enum Step {
    Rename,
    Unlink,
    DirectorySync,
}

struct Work<'a> {
    store: &'a CacheStore,
    cancel: &'a AtomicBool,
    activity: &'a CacheLock,
    publication: &'a CacheLock,
    next: usize,
    mutated: bool,
}

impl Work<'_> {
    fn ensure(&self) -> Result<()> {
        check_cancel(self.cancel)?;
        self.activity.ensure_bound()?;
        self.publication.ensure_bound()
    }

    fn point(&self, step: Step, index: usize, before: bool) -> Result<()> {
        #[cfg(feature = "diagnostic-cache")]
        self.store.point(
            match step {
                Step::Rename => CacheOperation::RetirementRename,
                Step::Unlink => CacheOperation::ReclaimUnlink,
                Step::DirectorySync => CacheOperation::ReclaimDirectorySync,
            },
            if before {
                CacheMoment::Before
            } else {
                CacheMoment::After
            },
            Some(index),
        )?;
        #[cfg(not(feature = "diagnostic-cache"))]
        let _ = (step, index, before);
        Ok(())
    }

    fn index(&mut self) -> Result<usize> {
        let index = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| Error::Limit("cache reclamation step overflow".into()))?;
        Ok(index)
    }

    fn bound(&self, name: &Name) -> Result<()> {
        self.ensure()?;
        let (parent, leaf) = self.store.inner.root.parent_dir(&name.path, false)?;
        let stat =
            rustix::fs::statat(&parent, leaf.as_str(), AtFlags::SYMLINK_NOFOLLOW).map_err(io)?;
        let kind = rustix::fs::FileType::from_raw_mode(stat.st_mode);
        if stat.st_dev != name.dev
            || stat.st_ino != name.ino
            || (name.directory && kind != rustix::fs::FileType::Directory)
            || (!name.directory && kind != rustix::fs::FileType::RegularFile)
        {
            return Err(Error::State("reclamation candidate binding changed".into()));
        }
        Ok(())
    }

    fn sync(&mut self, path: &str) -> Result<()> {
        self.ensure()?;
        let index = self.index()?;
        self.point(Step::DirectorySync, index, true)?;
        self.ensure()?;
        self.store.sync_dir(path)?;
        self.point(Step::DirectorySync, index, false)
    }

    fn unlink(&mut self, name: &Name, report: &mut ReclaimReport) -> Result<()> {
        self.ensure()?;
        let index = self.index()?;
        self.point(Step::Unlink, index, true)?;
        // Check after the diagnostic boundary too. No locally owned collector descriptor of
        // the final inode survives this check or the unlink call.
        self.bound(name)?;
        check_cancel(self.cancel)?;
        self.store.unlink(&name.path, name.directory)?;
        self.mutated = true;
        if name.directory {
            report.removed_directories += 1;
        } else {
            report.removed_files += 1;
        }
        self.point(Step::Unlink, index, false)?;
        self.sync(
            name.path
                .rsplit_once('/')
                .expect("internal target parent")
                .0,
        )
    }
}

impl CacheStore {
    fn reclamation_manifest(
        &self,
        path: &str,
        expected: Option<&str>,
        metadata: &mut usize,
        edges: &mut usize,
        cancel: &AtomicBool,
    ) -> Result<Manifest> {
        check_cancel(cancel)?;
        let bytes = self.read(path, self.inner.options.limits.max_manifest_bytes)?;
        *metadata = metadata
            .checked_add(bytes.len())
            .filter(|n| *n <= GRAPH_BYTES)
            .ok_or_else(|| Error::Limit("cache reclamation aggregate metadata bound".into()))?;
        let manifest = Manifest::from_json(&bytes, &self.inner.options.limits)?;
        if let Some(identity) = expected
            && manifest.digest()? != identity
        {
            return Err(Error::Malformed(
                "reclamation manifest identity mismatch".into(),
            ));
        }
        *edges = edges
            .checked_add(manifest.entries().len())
            .filter(|n| *n <= MAX_NODES)
            .ok_or_else(|| Error::Limit("cache reclamation aggregate graph bound".into()))?;
        check_cancel(cancel)?;
        Ok(manifest)
    }

    fn prepare_reclamation(&self, digest: &str, cancel: &AtomicBool) -> Result<Prepared> {
        let nodes = self.inner.root.walk_tree(MAX_NODES)?;
        // walk_tree returns path-sorted entries. Graph validation stays bounded
        // by O(nodes log nodes), including roots with many empty markers.
        let find = |path: &str| {
            nodes
                .binary_search_by(|node| node.path.as_str().cmp(path))
                .ok()
                .map(|index| &nodes[index])
        };
        let has_children = |path: &str| {
            let prefix = format!("{path}/");
            let index = nodes.partition_point(|node| node.path.as_str() < prefix.as_str());
            nodes
                .get(index)
                .is_some_and(|node| node.path.starts_with(&prefix))
        };
        let active = format!("snapshots/{digest}");
        let retirement = format!("retirement/{digest}");
        let a = find(&active);
        let r = find(&retirement);
        if a.is_some() && r.is_some() {
            return Err(Error::State(
                "snapshot exists in active and retired namespaces".into(),
            ));
        }
        let root = r.or(a);
        if root.is_some_and(|node| !node.is_directory) {
            return Err(Error::Malformed(
                "snapshot target is not a directory".into(),
            ));
        }
        let mut metadata = 0usize;
        let mut edges = 0usize;
        let mut references = 0usize;
        let mut shared = BTreeSet::new();
        let mut locations: BTreeMap<String, Option<String>> = BTreeMap::new();
        for node in &nodes {
            check_cancel(cancel)?;
            if let Some(leaf) = node.path.strip_prefix("refs/") {
                if node.is_directory || leaf.contains('/') || !leaf.ends_with(".json") {
                    return Err(Error::Malformed("foreign cache reference path".into()));
                }
                let bytes = self.read(&node.path, CONTROL_LIMIT * 2)?;
                metadata = metadata
                    .checked_add(bytes.len())
                    .filter(|n| *n <= GRAPH_BYTES)
                    .ok_or_else(|| {
                        Error::Limit("cache reclamation aggregate metadata bound".into())
                    })?;
                let reference: Reference = serde_json::from_slice(&bytes)
                    .map_err(|_| Error::Malformed("invalid reclamation reference".into()))?;
                if reference.schema != "zerogravity.cache.reference.v1"
                    || reference_key(&reference.name)? != node.path
                {
                    return Err(Error::Malformed(
                        "reclamation reference identity mismatch".into(),
                    ));
                }
                validate_digest(&reference.snapshot)?;
                if reference.snapshot == digest {
                    references += 1;
                }
            }
            if let Some(leaf) = node.path.strip_prefix("manifests/") {
                if node.is_directory || leaf.contains('/') {
                    return Err(Error::Malformed("foreign manifest backing path".into()));
                }
                let identity = leaf
                    .strip_suffix(".json")
                    .ok_or_else(|| Error::Malformed("foreign manifest backing name".into()))?;
                validate_digest(identity)?;
                if identity != digest {
                    locations.insert(node.path.clone(), Some(identity.to_string()));
                }
            }
            for namespace in ["snapshots", "retirement"] {
                let prefix = format!("{namespace}/");
                if let Some(identity) = node.path.strip_prefix(&prefix)
                    && !identity.contains('/')
                {
                    validate_digest(identity)?;
                    if !node.is_directory {
                        return Err(Error::Malformed("foreign snapshot namespace entry".into()));
                    }
                    if identity != digest {
                        let manifest = format!("{}/{identity}/manifest.json", namespace);
                        if find(&manifest).is_some() {
                            locations.insert(manifest, Some(identity.to_string()));
                        } else if namespace == "snapshots" || has_children(&node.path) {
                            return Err(Error::Malformed(
                                "snapshot lacks its retained manifest".into(),
                            ));
                        }
                    }
                }
            }
            // Abandoned prepared snapshots/manifests are conservative graph
            // roots too. Their payloads and all staging names remain retained.
            if !node.is_directory
                && node.path.starts_with("staging/")
                && node.path.ends_with("/manifest.json")
            {
                locations.insert(node.path.clone(), None);
            }
        }
        for (path, identity) in locations {
            let manifest = self.reclamation_manifest(
                &path,
                identity.as_deref(),
                &mut metadata,
                &mut edges,
                cancel,
            )?;
            shared.extend(
                manifest
                    .entries()
                    .iter()
                    .map(|entry| entry.digest().to_string()),
            );
        }
        let Some(root) = root else {
            // Absence grants no authority to sweep an orphan manifest/blob.
            return Ok(Prepared {
                root: None,
                tree: Vec::new(),
                blobs: Vec::new(),
                backing: None,
                references,
                retained_blobs: 0,
                retired: false,
            });
        };
        let manifest_path = format!("{}/manifest.json", root.path);
        let Some(manifest_node) = find(&manifest_path) else {
            if r.is_none() || has_children(&root.path) {
                return Err(Error::Malformed(
                    "target lacks its retained manifest".into(),
                ));
            }
            if find(&format!("manifests/{digest}.json")).is_some() {
                return Err(Error::Malformed(
                    "empty retirement marker has unexpected backing".into(),
                ));
            }
            return Ok(Prepared {
                root: Some(Name::new(root)),
                tree: Vec::new(),
                blobs: Vec::new(),
                backing: None,
                references,
                retained_blobs: 0,
                retired: true,
            });
        };
        if manifest_node.is_directory {
            return Err(Error::Malformed("target manifest is not regular".into()));
        }
        let manifest = self.reclamation_manifest(
            &manifest_path,
            Some(digest),
            &mut metadata,
            &mut edges,
            cancel,
        )?;
        let mut files = BTreeSet::from([manifest_path]);
        let mut directories = BTreeSet::new();
        let mut blobs = BTreeMap::new();
        let mut retained = BTreeSet::new();
        for entry in manifest.entries() {
            let path = format!("{}/files/{}", root.path, entry.path());
            files.insert(path.clone());
            let mut parent = path.rsplit_once('/').expect("internal file parent").0;
            while parent != root.path {
                directories.insert(parent.to_string());
                parent = parent
                    .rsplit_once('/')
                    .expect("internal directory parent")
                    .0;
            }
            let alias = find(&path);
            let blob_path = format!("blobs/sha256/{}", entry.digest());
            let blob = find(&blob_path);
            if r.is_none() && (alias.is_none() || blob.is_none()) {
                return Err(Error::Malformed(
                    "active snapshot payload binding missing".into(),
                ));
            }
            for node in [alias, blob].into_iter().flatten() {
                if node.is_directory || node.bytes != entry.bytes() {
                    return Err(Error::Malformed(
                        "snapshot payload type or length mismatch".into(),
                    ));
                }
            }
            if let (Some(alias), Some(blob)) = (alias, blob)
                && (alias.dev, alias.ino) != (blob.dev, blob.ino)
            {
                return Err(Error::State(
                    "snapshot payload does not alias its blob".into(),
                ));
            }
            if shared.contains(entry.digest()) {
                retained.insert(entry.digest().to_string());
            } else if let Some(blob) = blob {
                blobs.insert(blob_path, Name::new(blob));
            }
        }
        let mut tree = Vec::new();
        for node in nodes
            .iter()
            .filter(|node| node.path.starts_with(&(root.path.clone() + "/")))
        {
            if (node.is_directory && !directories.contains(&node.path))
                || (!node.is_directory && !files.contains(&node.path))
            {
                return Err(Error::Malformed(
                    "unexpected entry in reclamation target".into(),
                ));
            }
            tree.push(Name::new(node));
        }
        let backing = find(&format!("manifests/{digest}.json"));
        if let Some(node) = backing {
            self.reclamation_manifest(&node.path, Some(digest), &mut metadata, &mut edges, cancel)?;
        } else if r.is_none() {
            return Err(Error::Malformed(
                "active snapshot manifest backing missing".into(),
            ));
        }
        Ok(Prepared {
            root: Some(Name::new(root)),
            tree,
            blobs: blobs.into_values().collect(),
            backing: backing.map(Name::new),
            references,
            retained_blobs: retained.len(),
            retired: r.is_some(),
        })
    }

    /// Reclaim one exact, unreferenced snapshot in a protected fresh-policy
    /// root. Managed descriptor readers and transactions cooperate with this
    /// protocol; arbitrary raw-path consumers must be quiescent. No automatic
    /// eviction, reference deletion, staging cleanup or live migration occurs.
    pub fn reclaim_snapshot(
        &self,
        digest: &str,
        mode: ReclaimMode,
        cancel: &AtomicBool,
    ) -> Result<ReclaimOutcome> {
        check_cancel(cancel)?;
        validate_digest(digest)?;
        self.require_protected_root()?;
        let Some(activity) = self.try_maintenance_activity(cancel)? else {
            return Ok(ReclaimOutcome::Busy);
        };
        activity.ensure_bound()?;
        let publication = self.lock("publication", cancel)?;
        publication.ensure_bound()?;
        let initial_usage = self.usage_locked()?;
        let before = initial_usage.used_bytes;
        let mut prepared = self.prepare_reclamation(digest, cancel)?;
        if prepared.references != 0 {
            return Ok(ReclaimOutcome::Referenced {
                references: prepared.references,
            });
        }
        let mut pins = Vec::new();
        let mut identities = BTreeSet::new();
        let control_manifest = prepared
            .root
            .as_ref()
            .map(|root| format!("{}/manifest.json", root.path));
        for name in prepared
            .tree
            .iter()
            .filter(|name| !name.directory && Some(&name.path) != control_manifest.as_ref())
            .chain(prepared.blobs.iter())
        {
            if !identities.insert((name.dev, name.ino)) {
                continue;
            }
            let Some(file) = self.inner.root.try_exclusive_file(&name.path, cancel)? else {
                return Ok(ReclaimOutcome::Busy);
            };
            let actual = file.metadata()?;
            if (actual.dev(), actual.ino()) != (name.dev, name.ino) || actual.len() != name.bytes {
                return Err(Error::State(
                    "payload binding changed before reclamation".into(),
                ));
            }
            pins.push(file);
        }
        // All local collector payload FDs close before unlink. Forked CLOEXEC
        // copies and delayed kernel close effects preclude a global final-close
        // attestation; the report measures named charge, not physical release.
        // New protected opens/transactions remain excluded by activity.
        drop(pins);
        let mut report = ReclaimReport {
            snapshot: digest.to_string(),
            retired: prepared.retired,
            used_before_bytes: before,
            used_after_bytes: Some(before),
            candidate_files: prepared.tree.iter().filter(|n| !n.directory).count()
                + prepared.blobs.len()
                + usize::from(prepared.backing.is_some()),
            candidate_directories: prepared.tree.iter().filter(|n| n.directory).count()
                + usize::from(prepared.root.is_some()),
            removed_files: 0,
            removed_directories: 0,
            retained_blobs: prepared.retained_blobs,
        };
        if prepared.root.is_some()
            && !prepared.retired
            && add(add(before, initial_usage.reserved_bytes)?, RETIREMENT_PEAK)?
                > initial_usage.limit_bytes
        {
            return Err(Error::Limit(
                "cache retirement control peak admission refused".into(),
            ));
        }
        if mode == ReclaimMode::DryRun {
            activity.ensure_bound()?;
            publication.ensure_bound()?;
            check_cancel(cancel)?;
            return Ok(ReclaimOutcome::DryRun(report));
        }
        let mut work = Work {
            store: self,
            cancel,
            activity: &activity,
            publication: &publication,
            next: 0,
            mutated: prepared.retired,
        };
        let result = (|| -> Result<()> {
            if let Some(root) = &mut prepared.root {
                if !prepared.retired {
                    check_cancel(cancel)?;
                    let index = work.index()?;
                    work.point(Step::Rename, index, true)?;
                    work.bound(root)?;
                    check_cancel(cancel)?;
                    let retired = format!("retirement/{digest}");
                    self.rename(&root.path, &retired, false)?;
                    work.mutated = true;
                    report.retired = true;
                    for name in &mut prepared.tree {
                        name.path = name.path.replacen(&root.path, &retired, 1);
                    }
                    root.path = retired;
                    work.point(Step::Rename, index, false)?;
                }
                // Retry these on already-retired roots too; a prior process may
                // have exited after rename but before either parent sync.
                work.sync("snapshots")?;
                work.sync("retirement")?;
                prepared.tree.sort_unstable_by(|a, b| {
                    b.path
                        .len()
                        .cmp(&a.path.len())
                        .then_with(|| a.path.cmp(&b.path))
                });
                let manifest_path = format!("{}/manifest.json", root.path);
                for name in prepared
                    .tree
                    .iter()
                    .filter(|name| name.path != manifest_path)
                {
                    work.unlink(name, &mut report)?;
                }
                for name in &prepared.blobs {
                    work.unlink(name, &mut report)?;
                }
                work.sync("blobs/sha256")?;
                if let Some(name) = &prepared.backing {
                    work.unlink(name, &mut report)?;
                }
                work.sync("manifests")?;
                if let Some(name) = prepared.tree.iter().find(|name| name.path == manifest_path) {
                    work.unlink(name, &mut report)?;
                }
                work.unlink(root, &mut report)?;
            } else {
                // Current absence is idempotent. It establishes neither a past
                // successful removal nor authority over orphaned payloads.
                for path in ["snapshots", "retirement", "manifests", "blobs/sha256"] {
                    work.sync(path)?;
                }
            }
            Ok(())
        })();
        report.used_after_bytes = activity
            .ensure_bound()
            .and_then(|()| publication.ensure_bound())
            .and_then(|()| self.usage_locked())
            .ok()
            .map(|usage| usage.used_bytes);
        match result {
            Ok(()) if report.used_after_bytes.is_some() => Ok(ReclaimOutcome::Durable(report)),
            Ok(()) => Ok(ReclaimOutcome::Incomplete {
                report,
                cause: Error::State("post-reclamation inventory unavailable".into()),
            }),
            Err(cause) if work.mutated => Ok(ReclaimOutcome::Incomplete { report, cause }),
            Err(cause) => Err(cause),
        }
    }
}
