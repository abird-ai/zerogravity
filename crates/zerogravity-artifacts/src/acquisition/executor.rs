use super::http::{HttpAccept, HttpRequest, HttpTransport};
use super::resume::{ResumeDisposition, ResumeRequest, strong_etag, validate_range_response};
use super::{DigestProvenance, ResolvedFile, ResolvedPullPlan};
use crate::cache::{
    CacheStore, CommitOutcome, DiscoveredEntry, DiscoveredObject, DiscoveredTransaction, Manifest,
    ManifestEntry, ManifestSource, PartialObject, TransferAdmission,
};
use crate::{Error, Result};
use std::collections::BTreeSet;
use std::io::{self, Read};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct AcquiredFile {
    path: String,
    digest: String,
    bytes: u64,
    provenance: DigestProvenance,
    identity: String,
}
impl AcquiredFile {
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn provenance(&self) -> DigestProvenance {
        self.provenance
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
}

#[derive(Debug)]
pub struct PullReceipt {
    manifest: Manifest,
    files: Vec<AcquiredFile>,
    outcome: CommitOutcome,
}
impl PullReceipt {
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn files(&self) -> &[AcquiredFile] {
        &self.files
    }
    pub fn outcome(&self) -> &CommitOutcome {
        &self.outcome
    }
}

/// Secret-free progress for one immutable source and its current object.
/// Logical bytes include verified cached objects and rehashed saved prefixes;
/// they do not establish that the eventual snapshot has been published.
#[derive(Clone, Copy, Debug)]
pub struct PullProgress<'a> {
    source: &'a ManifestSource,
    file: &'a ResolvedFile,
    file_bytes: u64,
    total_bytes: u64,
    total_expected_bytes: Option<u64>,
    total_maximum_bytes: u64,
    file_complete: bool,
    cached_bytes: u64,
    resumed_bytes: u64,
}
impl PullProgress<'_> {
    pub fn source(&self) -> &ManifestSource {
        self.source
    }
    pub fn path(&self) -> &str {
        self.file.path()
    }
    pub fn identity(&self) -> &str {
        self.file.identity()
    }
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }
    pub fn file_total_bytes(&self) -> Option<u64> {
        self.file.bytes()
    }
    pub fn file_maximum_bytes(&self) -> u64 {
        self.file.maximum_bytes()
    }
    pub fn file_complete(&self) -> bool {
        self.file_complete
    }
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
    pub fn total_expected_bytes(&self) -> Option<u64> {
        self.total_expected_bytes
    }
    pub fn total_maximum_bytes(&self) -> u64 {
        self.total_maximum_bytes
    }
    pub fn cached_bytes(&self) -> u64 {
        self.cached_bytes
    }
    pub fn resumed_bytes(&self) -> u64 {
        self.resumed_bytes
    }
    pub fn downloaded_bytes(&self) -> u64 {
        self.total_bytes - self.cached_bytes - self.resumed_bytes
    }
}

/// The cache admits complete listed lengths before any HTTP body is requested.
pub fn execute(
    store: &CacheStore,
    plan: &ResolvedPullPlan,
    transport: &impl HttpTransport,
    admission: &TransferAdmission,
    cancel: &Arc<AtomicBool>,
    reference: Option<&str>,
) -> Result<PullReceipt> {
    execute_with_progress(
        store,
        plan,
        transport,
        admission,
        cancel,
        reference,
        &mut |_| {},
    )
}

/// Observe logical acquisition progress on the caller's thread. The callback
/// may set the same cancellation flag; it is checked immediately afterwards.
pub fn execute_with_progress(
    store: &CacheStore,
    plan: &ResolvedPullPlan,
    transport: &impl HttpTransport,
    admission: &TransferAdmission,
    cancel: &Arc<AtomicBool>,
    reference: Option<&str>,
    progress: &mut impl FnMut(&PullProgress<'_>),
) -> Result<PullReceipt> {
    validate_split_closure(plan)?;
    let entries = plan
        .files()
        .iter()
        .map(|file| {
            DiscoveredEntry::new_with_length(
                file.path(),
                file.length(),
                file.role(),
                file.sha256().map(str::to_string),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut transaction =
        store.begin_discovered(plan.source().clone(), entries, admission, cancel.clone())?;
    let mut files = Vec::new();
    let mut completed_bytes = 0;
    let mut cached_bytes = 0;
    let mut resumed_bytes = 0;
    for file in plan.files() {
        let mut event = PullProgress {
            source: plan.source(),
            file,
            file_bytes: 0,
            total_bytes: completed_bytes,
            total_expected_bytes: plan.total_bytes(),
            total_maximum_bytes: plan.maximum_bytes(),
            file_complete: false,
            cached_bytes,
            resumed_bytes,
        };
        progress(&event);
        super::http::check_cancel(cancel)?;
        let mut prefix = [0u8; 128];
        let actual = if let Some(cached) = transaction.try_cached(file.path())? {
            let count = transaction.read_prefix(file.path(), &mut prefix)?;
            reject_pointer(&prefix[..count])?;
            cached_bytes += cached.bytes();
            event.file_bytes = cached.bytes();
            event.total_bytes = completed_bytes + cached.bytes();
            event.cached_bytes = cached_bytes;
            event.file_complete = true;
            progress(&event);
            super::http::check_cancel(cancel)?;
            cached
        } else {
            let mut report = |count, resumed, cached| {
                event.file_bytes = count;
                event.total_bytes = completed_bytes + count;
                event.resumed_bytes = resumed_bytes + resumed;
                event.cached_bytes = cached_bytes + cached;
                progress(&event);
            };
            let (actual, resumed, cached) =
                transfer_object(&mut transaction, file, transport, cancel, &mut report)?;
            resumed_bytes += resumed;
            cached_bytes += cached;
            actual
        };
        if !event.file_complete {
            event.file_bytes = actual.bytes();
            event.total_bytes = completed_bytes + actual.bytes();
            event.file_complete = true;
            progress(&event);
            super::http::check_cancel(cancel)?;
        }
        files.push(acquired(file, &actual));
        completed_bytes += actual.bytes();
    }
    let paths: BTreeSet<_> = plan.files().iter().map(ResolvedFile::path).collect();
    for file in plan
        .files()
        .iter()
        .filter(|file| file.path().ends_with(".safetensors.index.json"))
    {
        let bytes = transaction.read_object(file.path(), plan.limits().max_metadata_bytes)?;
        let index = crate::sharded::SafetensorsIndex::from_json(&bytes)
            .map_err(|_| Error::Malformed("invalid acquired safetensors index".into()))?;
        let parent = file.path().rsplit_once('/').map(|(parent, _)| parent);
        for shard in index.shards() {
            let path =
                parent.map_or_else(|| shard.to_string(), |parent| format!("{parent}/{shard}"));
            if !paths.contains(path.as_str()) {
                return Err(Error::Malformed(
                    "selected safetensors index has missing shards".into(),
                ));
            }
        }
        // total_size is tensor payload, never a claim about shard file lengths.
    }
    let manifest = transaction.manifest()?;
    let outcome = transaction.commit(reference)?;
    Ok(PullReceipt {
        manifest,
        files,
        outcome,
    })
}

fn transfer_object(
    transaction: &mut DiscoveredTransaction,
    file: &ResolvedFile,
    transport: &impl HttpTransport,
    cancel: &AtomicBool,
    report: &mut impl FnMut(u64, u64, u64),
) -> Result<(ManifestEntry, u64, u64)> {
    let mut partial = match transaction.cached_or_partial_object(file.path(), file.identity())? {
        DiscoveredObject::Cached(actual) => {
            let mut prefix = [0u8; 128];
            let count = transaction.read_prefix(file.path(), &mut prefix)?;
            reject_pointer(&prefix[..count])?;
            report(actual.bytes(), 0, actual.bytes());
            super::http::check_cancel(cancel)?;
            let cached = actual.bytes();
            return Ok((actual, 0, cached));
        }
        DiscoveredObject::Partial(partial) => partial,
    };
    let resumed = partial.resume_bytes();
    let mut prefix = [0u8; 128];
    let saved_prefix = partial.read_prefix(&mut prefix)?;
    reject_pointer(&prefix[..saved_prefix])?;
    if resumed != 0 {
        report(resumed, resumed, 0);
        if let Err(cause) = super::http::check_cancel(cancel) {
            checkpoint_if_exact(&mut partial, file)?;
            return Err(cause);
        }
    }
    let range = if resumed == 0 {
        None
    } else {
        Some(ResumeRequest {
            start: resumed,
            total: file
                .bytes()
                .ok_or_else(|| Error::Unsupported("capped resume is unsupported".into()))?,
            validator: partial
                .validator()
                .ok_or_else(|| Error::State("saved download lacks strong validator".into()))?
                .to_string(),
        })
    };
    let mut reader = RetryReader::open(transport, file, cancel, range)?;
    if let Some(validator) = &reader.validator
        && file.bytes().is_some()
    {
        partial.set_validator(validator)?;
    }
    // Inspect the original prefix before adding bytes to a retained checkpoint.
    let prefix_size = prefix
        .len()
        .min(file.maximum_bytes().min(usize::MAX as u64) as usize);
    let mut prefix_count = saved_prefix;
    let mut eof = false;
    if saved_prefix < prefix_size {
        while prefix_count < prefix_size {
            match reader.read(&mut prefix[prefix_count..prefix_size]) {
                Ok(0) => {
                    eof = true;
                    break;
                }
                Ok(count) => prefix_count += count,
                Err(_) => {
                    checkpoint_if_exact(&mut partial, file)?;
                    return Err(Error::State("HTTP object prefix read failed".into()));
                }
            }
        }
        reject_pointer(&prefix[..prefix_count])?;
        if let Err(cause) = partial.write_chunk(&prefix[saved_prefix..prefix_count]) {
            checkpoint_if_exact(&mut partial, file)?;
            return Err(cause);
        }
        report(partial.bytes(), resumed, 0);
    }
    let mut checkpoint_bytes = partial.bytes();
    let mut checkpoint_time = Instant::now();
    let mut buffer = [0u8; 64 * 1024];
    while !eof {
        if let Err(cause) = super::http::check_cancel(cancel) {
            checkpoint_if_exact(&mut partial, file)?;
            return Err(cause);
        }
        let read = match reader.read(&mut buffer) {
            Ok(read) => read,
            Err(_) => {
                checkpoint_if_exact(&mut partial, file)?;
                return Err(Error::State("HTTP object read failed".into()));
            }
        };
        if read == 0 {
            eof = true;
            continue;
        }
        if let Err(cause) = partial.write_chunk(&buffer[..read]) {
            checkpoint_if_exact(&mut partial, file)?;
            return Err(cause);
        }
        if partial.bytes() - checkpoint_bytes >= 8 * 1024 * 1024
            || checkpoint_time.elapsed() >= Duration::from_secs(1)
        {
            checkpoint_if_exact(&mut partial, file)?;
            checkpoint_bytes = partial.bytes();
            checkpoint_time = Instant::now();
        }
        report(partial.bytes(), resumed, 0);
    }
    Ok((partial.finish_after_eof()?, resumed, 0))
}

fn checkpoint_if_exact(partial: &mut PartialObject<'_>, file: &ResolvedFile) -> Result<()> {
    if file.bytes().is_some() {
        partial.checkpoint()?;
    }
    Ok(())
}

fn acquired(file: &ResolvedFile, entry: &ManifestEntry) -> AcquiredFile {
    AcquiredFile {
        path: entry.path().to_string(),
        digest: entry.digest().to_string(),
        bytes: entry.bytes(),
        provenance: entry.digest_provenance(),
        identity: file.identity().to_string(),
    }
}

fn validate_split_closure(plan: &ResolvedPullPlan) -> Result<()> {
    let paths: BTreeSet<_> = plan.files().iter().map(ResolvedFile::path).collect();
    for file in plan.files() {
        let Some(stem) = file.path().strip_suffix(".gguf") else {
            continue;
        };
        let Some((prefix, count)) = stem.rsplit_once("-of-") else {
            continue;
        };
        let Some((base, ordinal)) = prefix.rsplit_once('-') else {
            continue;
        };
        if ordinal.len() != 5
            || count.len() != 5
            || !ordinal
                .bytes()
                .chain(count.bytes())
                .all(|b| b.is_ascii_digit())
        {
            return Err(Error::Malformed("invalid split GGUF filename".into()));
        }
        let count = count
            .parse::<usize>()
            .map_err(|_| Error::Malformed("split GGUF count".into()))?;
        let ordinal = ordinal
            .parse::<usize>()
            .map_err(|_| Error::Malformed("split GGUF ordinal".into()))?;
        if count == 0 || count > plan.limits().cache.max_files || ordinal == 0 || ordinal > count {
            return Err(Error::Limit("split GGUF bounds".into()));
        }
        for index in 1..=count {
            if !paths.contains(format!("{base}-{index:05}-of-{count:05}.gguf").as_str()) {
                return Err(Error::Malformed("selected split GGUF is incomplete".into()));
            }
        }
    }
    Ok(())
}

struct RetryReader<'a, T> {
    transport: &'a T,
    file: &'a ResolvedFile,
    cancel: &'a AtomicBool,
    body: Box<dyn Read + Send>,
    validator: Option<String>,
    count: u64,
    response_length: Option<u64>,
    retries: usize,
}
impl<'a, T: HttpTransport> RetryReader<'a, T> {
    fn open(
        transport: &'a T,
        file: &'a ResolvedFile,
        cancel: &'a AtomicBool,
        range: Option<ResumeRequest>,
    ) -> Result<Self> {
        super::http::check_cancel(cancel)?;
        let count = range.as_ref().map_or(0, |range| range.start);
        let response = transport
            .get(
                &HttpRequest {
                    url: file.url().to_string(),
                    accept: HttpAccept::Binary,
                    range: range.clone(),
                    max_body_bytes: file.maximum_bytes() - count,
                },
                cancel,
            )
            .map_err(|_| Error::State("HTTP object request failed".into()))?;
        super::http::check_cancel(cancel)?;
        super::http::validate_validator(file.validator(), &response)?;
        if let Some(range) = &range {
            if validate_range_response(range, &response)? != ResumeDisposition::Append {
                return Err(Error::State(
                    "HTTP range ignored; saved prefix retained".into(),
                ));
            }
        } else {
            if response.status != 200 {
                return Err(Error::State(format!(
                    "HTTP object status {}",
                    response.status
                )));
            }
            if response
                .header("content-encoding")
                .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
            {
                return Err(Error::Unsupported(
                    "download requires identity encoding".into(),
                ));
            }
            let declared = response
                .header("content-length")
                .map(super::http::parse_u64)
                .transpose()?;
            if let Some(expected) = file.bytes() {
                if declared.is_some_and(|actual| actual != expected) {
                    return Err(Error::Malformed(
                        "download metadata/body length mismatch".into(),
                    ));
                }
            } else {
                if response.header("content-range").is_some() {
                    return Err(Error::Malformed("capped response range invalid".into()));
                }
                if declared.is_some_and(|bytes| bytes > file.maximum_bytes()) {
                    return Err(Error::Limit(
                        "capped response declared length exceeds cap".into(),
                    ));
                }
                let transfer = response.header("transfer-encoding");
                if transfer.is_some_and(|value| {
                    declared.is_some() || !value.eq_ignore_ascii_case("chunked")
                }) {
                    return Err(Error::Malformed("ambiguous capped response framing".into()));
                }
                if declared.is_none() && transfer.is_none() {
                    return Err(Error::Unsupported(
                        "capped response requires declared length or chunked completion".into(),
                    ));
                }
            }
        }
        let response_length = file.bytes().or(response
            .header("content-length")
            .map(super::http::parse_u64)
            .transpose()?);
        let validator = response
            .header("etag")
            .filter(|value| strong_etag(value).is_ok())
            .map(str::to_string);
        Ok(Self {
            transport,
            file,
            cancel,
            body: response.body,
            validator,
            count,
            response_length,
            retries: 0,
        })
    }
}
impl<T: HttpTransport> Read for RetryReader<'_, T> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            if self.cancel.load(Ordering::Acquire) {
                return Err(io::Error::other("acquisition canceled"));
            }
            let limit = self
                .file
                .maximum_bytes()
                .min(self.response_length.unwrap_or(u64::MAX));
            let allowed = limit.saturating_sub(self.count).saturating_add(1);
            let count = buffer
                .len()
                .min(usize::try_from(allowed).unwrap_or(usize::MAX));
            match self.body.read(&mut buffer[..count]) {
                Ok(read) if read != 0 => {
                    self.count = self
                        .count
                        .checked_add(read as u64)
                        .filter(|v| {
                            *v <= self.file.maximum_bytes()
                                && self.response_length.is_none_or(|length| *v <= length)
                        })
                        .ok_or_else(|| io::Error::other("download exceeded planned length"))?;
                    // The owning cache guard checks cancellation before accepting
                    // these bytes. Its caller checkpoints earlier accepted bytes
                    // if cancellation arrived while this blocking read completed.
                    return Ok(read);
                }
                Ok(0)
                    if self
                        .response_length
                        .is_none_or(|length| self.count == length) =>
                {
                    return Ok(0);
                }
                Ok(_) | Err(_) => {}
            }
            let Some(total) = self.file.bytes() else {
                return Err(io::Error::other(
                    "capped download interrupted without terminal EOF",
                ));
            };
            if self.retries == 2 || self.count == 0 || self.count >= total {
                return Err(io::Error::other(
                    "download interrupted without resumable prefix",
                ));
            }
            let validator = self
                .validator
                .clone()
                .ok_or_else(|| io::Error::other("interrupted download lacks strong validator"))?;
            self.retries += 1;
            let range = ResumeRequest {
                start: self.count,
                total,
                validator,
            };
            let response = self
                .transport
                .get(
                    &HttpRequest {
                        url: self.file.url().to_string(),
                        accept: HttpAccept::Binary,
                        range: Some(range.clone()),
                        max_body_bytes: total - self.count,
                    },
                    self.cancel,
                )
                .map_err(|_| io::Error::other("HTTP range retry failed"))?;
            if validate_range_response(&range, &response)
                .map_err(|_| io::Error::other("HTTP range validation failed"))?
                != ResumeDisposition::Append
            {
                return Err(io::Error::other(
                    "range ignored; owned partial restart unsupported",
                ));
            }
            self.body = response.body;
        }
    }
}

pub(super) fn reject_pointer(prefix: &[u8]) -> Result<()> {
    if prefix.starts_with(b"version https://git-lfs.github.com/spec/v1") {
        return Err(Error::Unsupported(
            "Git LFS pointer requires qualified LFS resolver".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod cached_start_tests {
    use super::*;
    use crate::cache::{CacheOptions, FileRole};
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::DirBuilderExt;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;
    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture {
        root: PathBuf,
        store: CacheStore,
        admission: TransferAdmission,
        file: ResolvedFile,
        source: ManifestSource,
        previous: String,
    }
    impl Fixture {
        fn new(bytes: &[u8]) -> Self {
            let root = std::env::temp_dir().join(format!(
                "zerogravity-executor-cached-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&root)
                .unwrap();
            let store =
                CacheStore::open(CacheOptions::new(Some(root.clone()), 8 * 1024 * 1024)).unwrap();
            let resource = PhysicalDeviceKey::new("test-host", "executor-cached-start").unwrap();
            let arbiter = ResourceArbiter::new();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("tiny-local-fixture", "cached-start").unwrap(),
            };
            let hash = format!("{:x}", Sha256::digest(bytes));
            let source = ManifestSource::new("github", "fixture/repo", "1".repeat(40)).unwrap();
            let file = ResolvedFile::new(
                "payload.bin",
                "https://example.invalid/payload",
                FileRole::Auxiliary,
                bytes.len() as u64,
                Some(hash.clone()),
                "immutable/payload",
            )
            .unwrap();
            let manifest = Manifest::new(
                source.clone(),
                vec![
                    ManifestEntry::new(file.path(), hash, file.bytes().unwrap(), file.role())
                        .unwrap(),
                ],
                &Default::default(),
            )
            .unwrap();
            let mut publisher = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            let mut input = bytes;
            publisher.import_object(file.path(), &mut input).unwrap();
            let CommitOutcome::Durable(previous) = publisher.commit(Some("previous")).unwrap()
            else {
                panic!("fixture publication must be durable");
            };
            Self {
                root,
                store,
                admission,
                file,
                source,
                previous: previous.digest().to_string(),
            }
        }
        fn transaction(&self, cancel: Arc<AtomicBool>) -> DiscoveredTransaction {
            self.store
                .begin_discovered(
                    self.source.clone(),
                    vec![
                        DiscoveredEntry::new(
                            self.file.path(),
                            self.file.bytes().unwrap(),
                            self.file.role(),
                            self.file.sha256().map(str::to_string),
                        )
                        .unwrap(),
                    ],
                    &self.admission,
                    cancel,
                )
                .unwrap()
        }
        fn unchanged(&self) {
            assert_eq!(
                self.store
                    .resolve_ref("previous")
                    .unwrap()
                    .manifest()
                    .digest()
                    .unwrap(),
                self.previous
            );
            assert_eq!(self.store.usage().unwrap().reserved_bytes, 0);
            assert_eq!(
                self.admission
                    .arbiter
                    .snapshot(&self.admission.resource)
                    .unwrap()
                    .reserved,
                0
            );
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
    struct NoRequest;
    impl HttpTransport for NoRequest {
        fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<super::super::http::HttpResponse> {
            panic!("concurrent verified fill must never request another body");
        }
    }

    #[test]
    fn concurrent_cached_start_reports_only_verified_cached_bytes_without_a_request() {
        let fixture = Fixture::new(b"tiny cached data");
        let cancel = Arc::new(AtomicBool::new(false));
        let mut transaction = fixture.transaction(cancel.clone());
        let mut progress = Vec::new();
        let (entry, resumed, cached) = transfer_object(
            &mut transaction,
            &fixture.file,
            &NoRequest,
            &cancel,
            &mut |bytes, resumed, cached| progress.push((bytes, resumed, cached)),
        )
        .unwrap();
        assert_eq!(entry.digest(), fixture.file.sha256().unwrap());
        assert_eq!((resumed, cached), (0, fixture.file.bytes().unwrap()));
        assert_eq!(
            progress,
            [(
                fixture.file.bytes().unwrap(),
                0,
                fixture.file.bytes().unwrap()
            )]
        );
        drop(transaction);
        fixture.unchanged();
    }
    #[test]
    fn concurrent_cached_progress_cancellation_is_immediate_and_preserves_previous_ref() {
        let fixture = Fixture::new(b"tiny cached data");
        let cancel = Arc::new(AtomicBool::new(false));
        let mut transaction = fixture.transaction(cancel.clone());
        let mut calls = 0;
        let error = transfer_object(
            &mut transaction,
            &fixture.file,
            &NoRequest,
            &cancel,
            &mut |bytes, resumed, cached| {
                assert_eq!(
                    (bytes, resumed, cached),
                    (
                        fixture.file.bytes().unwrap(),
                        0,
                        fixture.file.bytes().unwrap()
                    )
                );
                calls += 1;
                cancel.store(true, Ordering::Release);
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::State(_)));
        assert_eq!(calls, 1);
        drop(transaction);
        fixture.unchanged();
    }
    #[test]
    fn concurrent_cached_lfs_pointer_refuses_before_progress_or_requests() {
        let fixture =
            Fixture::new(b"version https://git-lfs.github.com/spec/v1\noid sha256:placeholder\n");
        let cancel = Arc::new(AtomicBool::new(false));
        let mut transaction = fixture.transaction(cancel.clone());
        let error = transfer_object(
            &mut transaction,
            &fixture.file,
            &NoRequest,
            &cancel,
            &mut |_, _, _| panic!("pointer cannot be reported as accepted progress"),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)));
        drop(transaction);
        fixture.unchanged();
    }
}
