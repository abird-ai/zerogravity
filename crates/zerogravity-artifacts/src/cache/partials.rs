//! Durable partials belong to an existing admitted A1 attempt. Recovery copies
//! a rehashed committed prefix into a new exclusive file: it never reopens a
//! possibly hardlinked cache inode for writing. Retained attempts remain charged
//! and are not reclaimed by this protocol.

use super::*;
use std::os::unix::fs::FileExt;

const PARTIAL_SCHEMA: &str = "zerogravity.cache.partial.v1";
const PARTIAL_LIMIT: usize = 4096;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialRecord {
    schema: String,
    key: String,
    ordinal: usize,
    total: u64,
    bytes: u64,
    sha256: String,
    validator: String,
    device: u64,
    inode: u64,
}

fn strong_validator(value: &str) -> Result<()> {
    if value.len() < 2
        || value.len() > 1024
        || !value.starts_with('"')
        || !value.ends_with('"')
        || value.as_bytes()[1..value.len() - 1]
            .iter()
            .any(|byte| !matches!(byte, 0x21 | 0x23..=0x7e))
    {
        return Err(Error::Malformed("invalid strong partial validator".into()));
    }
    Ok(())
}

/// One single-flight object transfer, borrowing its existing transaction's
/// admission and publisher. No writable descriptor or mutable path is exposed.
/// Call checkpoint periodically and after an interrupted read; write_chunk does
/// not synchronize every network chunk. A checkpoint is a committed prefix,
/// rather than a promise that an uncommitted crash tail can be appended.
pub struct PartialObject<'a> {
    transaction: &'a mut DiscoveredTransaction,
    logical_path: String,
    key: String,
    ordinal: usize,
    target: File,
    hasher: Sha256,
    bytes: u64,
    resumed: u64,
    checkpointed: u64,
    validator: Option<String>,
    partial_lock: CacheLock,
    // Trusted digests share one payload reservation. Match the original A1
    // importer: serialize their transfer and refuse an existing blob before
    // allocating another private copy against that deduplicated reservation.
    object_lock: Option<CacheLock>,
    write_failed: bool,
}

/// One transaction-owned object acquisition. A verified concurrent fill never
/// allocates a second private payload or asks the caller for another body.
pub enum DiscoveredObject<'a> {
    Cached(ManifestEntry),
    Partial(Box<PartialObject<'a>>),
}

impl DiscoveredTransaction {
    /// Start one admitted transfer after checking try_cached. A concurrent
    /// trusted-digest fill may refuse here before stage creation; callers must
    /// use the verified cached path rather than allocate a duplicate payload.
    pub fn partial_object(
        &mut self,
        logical_path: &str,
        immutable_identity: &str,
    ) -> Result<PartialObject<'_>> {
        match self.prepare_object(logical_path, immutable_identity, false)? {
            DiscoveredObject::Partial(partial) => Ok(*partial),
            DiscoveredObject::Cached(_) => Err(Error::State(
                "known object is already cached; use try_cached".into(),
            )),
        }
    }

    /// Recheck a trusted digest after acquiring the original transfer locks.
    /// A fill between try_cached and this call returns a verified cached result.
    pub fn cached_or_partial_object(
        &mut self,
        logical_path: &str,
        immutable_identity: &str,
    ) -> Result<DiscoveredObject<'_>> {
        self.prepare_object(logical_path, immutable_identity, true)
    }

    fn prepare_object(
        &mut self,
        logical_path: &str,
        immutable_identity: &str,
        accept_cached: bool,
    ) -> Result<DiscoveredObject<'_>> {
        self.inner.ensure_owner()?;
        if self.verified.contains_key(logical_path) {
            return Err(Error::State("cache object already imported".into()));
        }
        if immutable_identity.is_empty()
            || immutable_identity.len() > 4096
            || immutable_identity.chars().any(char::is_control)
            || immutable_identity.contains("://")
            || immutable_identity.contains(['?', '#', '@', '\\'])
        {
            return Err(Error::Malformed(
                "partial identity must be bounded secret-free metadata".into(),
            ));
        }
        let entry = self.entry(logical_path)?;
        let total = entry.bytes;
        let role = entry.role;
        let expected = entry.sha256.clone();
        let key = digest(
            &serde_json::to_vec(&(
                self.inner.manifest.source(),
                logical_path,
                immutable_identity,
                total,
                entry.role,
                &entry.sha256,
            ))
            .map_err(|_| Error::Malformed("partial binding encoding failed".into()))?,
        );
        let ordinal = self
            .inner
            .manifest
            .entries()
            .iter()
            .position(|entry| entry.path() == logical_path)
            .ok_or_else(|| Error::Malformed("partial path absent from plan".into()))?;
        let partial_lock = self
            .inner
            .store
            .lock(&format!("partial-{key}"), &self.inner.cancel)?;
        partial_lock.ensure_bound()?;
        let object_lock = expected
            .as_ref()
            .map(|hash| {
                self.inner
                    .store
                    .lock(&format!("object-{hash}"), &self.inner.cancel)
            })
            .transpose()?;
        if let Some(lock) = &object_lock {
            lock.ensure_bound()?;
        }
        if let Some(hash) = &expected {
            match self.inner.store.verify_file_cancellable(
                &format!("blobs/sha256/{hash}"),
                hash,
                total,
                &self.inner.cancel,
            ) {
                Ok(_) => {
                    if accept_cached {
                        let actual = ManifestEntry::new(logical_path, hash, total, role)?
                            .with_digest_provenance(DigestProvenance::TrustedSha256);
                        self.inner.complete.insert(logical_path.to_string());
                        self.verified
                            .insert(logical_path.to_string(), actual.clone());
                        return Ok(DiscoveredObject::Cached(actual));
                    }
                    return Err(Error::State(
                        "known object is already cached; use try_cached".into(),
                    ));
                }
                Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        let saved = recover_candidate(&self.inner, &key, total)?;
        let temporary = self.inner.stage(&format!("object-{ordinal}"));
        let mut target = self.inner.store.inner.root.open_file(
            &temporary,
            OFlags::RDWR | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        )?;
        let mut hasher = Sha256::new();
        let mut resumed = 0;
        let mut validator = None;
        if let Some((record, mut file, owner)) = saved {
            let before = file.metadata()?;
            if before.dev() != record.device
                || before.ino() != record.inode
                || before.len() < record.bytes
                || before.len() > total
            {
                return Err(Error::Malformed("partial inode or length changed".into()));
            }
            let mut remaining = record.bytes;
            while remaining != 0 {
                self.inner.ensure_owner()?;
                partial_lock.ensure_bound()?;
                owner.ensure_bound()?;
                let count = self
                    .inner
                    .buffer
                    .len()
                    .min(remaining.min(usize::MAX as u64) as usize);
                let read = file.read(&mut self.inner.buffer[..count])?;
                self.inner.ensure_owner()?;
                if read == 0 {
                    return Err(Error::Malformed("saved partial truncated".into()));
                }
                hasher.update(&self.inner.buffer[..read]);
                target.write_all(&self.inner.buffer[..read])?;
                remaining -= read as u64;
            }
            let after = file.metadata()?;
            if before.dev() != after.dev()
                || before.ino() != after.ino()
                || before.len() != after.len()
                || before.mtime() != after.mtime()
                || before.mtime_nsec() != after.mtime_nsec()
                || format!("{:x}", hasher.clone().finalize()) != record.sha256
            {
                return Err(Error::Malformed("saved partial prefix changed".into()));
            }
            resumed = record.bytes;
            validator = Some(record.validator);
            // The source descriptor/old owner close before any later attempt
            // cleanup. Old named files continue to be charged, including tails.
        }
        Ok(DiscoveredObject::Partial(Box::new(PartialObject {
            transaction: self,
            logical_path: logical_path.to_string(),
            key,
            ordinal,
            target,
            hasher,
            bytes: resumed,
            resumed,
            checkpointed: resumed,
            validator,
            partial_lock,
            object_lock,
            write_failed: false,
        })))
    }
}

fn recover_candidate(
    transaction: &CacheTransaction,
    key: &str,
    total: u64,
) -> Result<Option<(PartialRecord, File, CacheLock)>> {
    let store = &transaction.store;
    let publication = store.lock("publication", &transaction.cancel)?;
    publication.ensure_bound()?;
    let mut selected: Option<(PartialRecord, File, CacheLock)> = None;
    for node in store.inner.root.walk_tree(MAX_NODES)? {
        let parts: Vec<_> = node.path.split('/').collect();
        if node.is_directory
            || parts.len() != 3
            || parts[0] != "staging"
            || parts[1] == transaction.attempt
            || !parts[2].starts_with("partial-")
            || !parts[2].ends_with(".json")
        {
            continue;
        }
        let owner = match store.inner.root.lock(
            &format!("attempt-{}", parts[1]),
            Duration::ZERO,
            &transaction.cancel,
        ) {
            Ok(owner) => owner,
            Err(Error::Limit(_)) => continue,
            Err(error) => return Err(error),
        };
        owner.ensure_bound()?;
        let record: PartialRecord = serde_json::from_slice(&store.read(&node.path, PARTIAL_LIMIT)?)
            .map_err(|_| Error::Malformed("invalid partial checkpoint".into()))?;
        if record.key != key {
            continue;
        }
        if record.schema != PARTIAL_SCHEMA
            || record.total != total
            || record.bytes == 0
            || record.bytes >= total
            || record.ordinal >= store.inner.options.limits.max_files
            || parts[2] != format!("partial-{}.json", record.ordinal)
        {
            return Err(Error::Malformed(
                "partial checkpoint binding mismatch".into(),
            ));
        }
        validate_digest(&record.sha256)?;
        strong_validator(&record.validator)?;
        if selected
            .as_ref()
            .is_some_and(|(previous, _, _)| previous.bytes >= record.bytes)
        {
            continue;
        }
        let file = store.inner.root.open_file(
            &format!("staging/{}/object-{}", parts[1], record.ordinal),
            OFlags::RDONLY,
            Mode::empty(),
        )?;
        selected = Some((record, file, owner));
    }
    Ok(selected)
}

impl PartialObject<'_> {
    pub fn resume_bytes(&self) -> u64 {
        self.resumed
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn validator(&self) -> Option<&str> {
        self.validator.as_deref()
    }

    pub fn set_validator(&mut self, validator: &str) -> Result<()> {
        self.ensure_owner()?;
        strong_validator(validator)?;
        if self
            .validator
            .as_deref()
            .is_some_and(|old| old != validator)
        {
            return Err(Error::Malformed("partial validator changed".into()));
        }
        self.validator = Some(validator.to_string());
        Ok(())
    }

    pub fn read_prefix(&self, output: &mut [u8]) -> Result<usize> {
        self.ensure_owner()?;
        if output.len() > 4096 {
            return Err(Error::Limit("partial inspection prefix bound".into()));
        }
        let count = output.len().min(self.bytes.min(usize::MAX as u64) as usize);
        self.target.read_exact_at(&mut output[..count], 0)?;
        Ok(count)
    }

    fn ensure_owner(&self) -> Result<()> {
        self.transaction.inner.ensure_owner()?;
        self.partial_lock.ensure_bound()?;
        if let Some(lock) = &self.object_lock {
            lock.ensure_bound()?;
        }
        Ok(())
    }

    pub fn write_chunk(&mut self, bytes: &[u8]) -> Result<()> {
        self.ensure_owner()?;
        if self.write_failed {
            return Err(Error::State("partial write previously failed".into()));
        }
        let total = self.transaction.entry(&self.logical_path)?.bytes;
        let count = self
            .bytes
            .checked_add(bytes.len() as u64)
            .filter(|count| *count <= total)
            .ok_or_else(|| Error::Malformed("partial exceeded planned length".into()))?;
        if let Err(error) = self.target.write_all(bytes) {
            self.write_failed = true;
            return Err(error.into());
        }
        self.hasher.update(bytes);
        self.bytes = count;
        Ok(())
    }

    /// Synchronize an already written prefix even after cancellation. This
    /// acquires no new transfer memory and performs no input/network read.
    pub fn checkpoint(&mut self) -> Result<()> {
        self.transaction
            .inner
            .owner_lock
            .as_ref()
            .ok_or_else(|| Error::State("cache transaction closed".into()))?
            .ensure_bound()?;
        self.partial_lock.ensure_bound()?;
        if let Some(lock) = &self.object_lock {
            lock.ensure_bound()?;
        }
        let Some(validator) = &self.validator else {
            return Ok(());
        };
        // Exact-full bytes do not prove that the upstream body reached EOF.
        // Keep any earlier proper-prefix checkpoint; require a fresh response
        // and terminal framing check before a future invocation can finish.
        if self.bytes == self.checkpointed
            || self.bytes == self.transaction.entry(&self.logical_path)?.bytes
        {
            return Ok(());
        }
        let store = &self.transaction.inner.store;
        #[cfg(feature = "diagnostic-cache")]
        store.point(
            CacheOperation::PartialFileSync,
            CacheMoment::Before,
            Some(self.ordinal),
        )?;
        self.target.sync_all()?;
        #[cfg(feature = "diagnostic-cache")]
        store.point(
            CacheOperation::PartialFileSync,
            CacheMoment::After,
            Some(self.ordinal),
        )?;
        let metadata = self.target.metadata()?;
        let record = PartialRecord {
            schema: PARTIAL_SCHEMA.into(),
            key: self.key.clone(),
            ordinal: self.ordinal,
            total: self.transaction.entry(&self.logical_path)?.bytes,
            bytes: self.bytes,
            sha256: format!("{:x}", self.hasher.clone().finalize()),
            validator: validator.clone(),
            device: metadata.dev(),
            inode: metadata.ino(),
        };
        let encoded = serde_json::to_vec(&record)
            .map_err(|_| Error::Malformed("partial checkpoint encoding failed".into()))?;
        if encoded.len() > PARTIAL_LIMIT {
            return Err(Error::Limit("partial checkpoint metadata bound".into()));
        }
        let publication = store.lock("publication", &AtomicBool::new(false))?;
        publication.ensure_bound()?;
        let temporary = self
            .transaction
            .inner
            .stage(&format!("partial-{}.tmp", self.ordinal));
        let checkpoint = self
            .transaction
            .inner
            .stage(&format!("partial-{}.json", self.ordinal));
        self.transaction
            .inner
            .retained_partials
            .insert(self.ordinal);
        let mut file = store.inner.root.open_file(
            &temporary,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        )?;
        file.write_all(&encoded)?;
        #[cfg(feature = "diagnostic-cache")]
        store.point(
            CacheOperation::PartialRecordSync,
            CacheMoment::Before,
            Some(self.ordinal),
        )?;
        file.sync_all()?;
        #[cfg(feature = "diagnostic-cache")]
        store.point(
            CacheOperation::PartialRecordSync,
            CacheMoment::After,
            Some(self.ordinal),
        )?;
        drop(file);
        #[cfg(feature = "diagnostic-cache")]
        store.point(
            CacheOperation::PartialRecordInstall,
            CacheMoment::Before,
            Some(self.ordinal),
        )?;
        self.partial_lock.ensure_bound()?;
        store.rename(&temporary, &checkpoint, true)?;
        #[cfg(feature = "diagnostic-cache")]
        store.point(
            CacheOperation::PartialRecordInstall,
            CacheMoment::After,
            Some(self.ordinal),
        )?;
        #[cfg(feature = "diagnostic-cache")]
        store.point(
            CacheOperation::PartialDirectorySync,
            CacheMoment::Before,
            Some(self.ordinal),
        )?;
        store.sync_dir(&format!("staging/{}", self.transaction.inner.attempt))?;
        #[cfg(feature = "diagnostic-cache")]
        store.point(
            CacheOperation::PartialDirectorySync,
            CacheMoment::After,
            Some(self.ordinal),
        )?;
        self.checkpointed = self.bytes;
        Ok(())
    }

    pub fn finish(mut self) -> Result<ManifestEntry> {
        self.ensure_owner()?;
        if self.write_failed {
            return Err(Error::State("partial write previously failed".into()));
        }
        let entry = self.transaction.entry(&self.logical_path)?;
        if self.bytes != entry.bytes {
            return Err(Error::Malformed("partial source truncated".into()));
        }
        let actual = format!("{:x}", self.hasher.clone().finalize());
        if entry
            .sha256
            .as_ref()
            .is_some_and(|expected| expected != &actual)
        {
            return Err(Error::Malformed("partial source digest mismatch".into()));
        }
        let provenance = if entry.sha256.is_some() {
            DigestProvenance::TrustedSha256
        } else {
            DigestProvenance::FirstSeen
        };
        let temporary = self
            .transaction
            .inner
            .stage(&format!("object-{}", self.ordinal));
        let installed = self
            .transaction
            .inner
            .install_staged(
                &self.logical_path,
                StagedObject {
                    path: &temporary,
                    file: &self.target,
                    digest: &actual,
                },
                self.object_lock.take(),
                true,
                true,
            )?
            .with_digest_provenance(provenance);
        if self
            .transaction
            .inner
            .retained_partials
            .contains(&self.ordinal)
        {
            let store = &self.transaction.inner.store;
            let publication = store.lock("publication", &self.transaction.inner.cancel)?;
            publication.ensure_bound()?;
            store.unlink(
                &self
                    .transaction
                    .inner
                    .stage(&format!("partial-{}.json", self.ordinal)),
                false,
            )?;
            store.sync_dir(&format!("staging/{}", self.transaction.inner.attempt))?;
            self.transaction
                .inner
                .retained_partials
                .remove(&self.ordinal);
        }
        self.transaction
            .verified
            .insert(self.logical_path.clone(), installed.clone());
        Ok(installed)
    }
}
