use super::http::{HttpAccept, HttpRequest, HttpTransport};
use super::resume::{ResumeDisposition, ResumeRequest, strong_etag, validate_range_response};
use super::{DigestProvenance, ResolvedFile, ResolvedPullPlan};
use crate::cache::{
    CacheStore, CommitOutcome, DiscoveredEntry, DiscoveredTransaction, Manifest, ManifestEntry,
    ManifestSource, TransferAdmission,
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
    total_expected_bytes: u64,
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
    pub fn file_total_bytes(&self) -> u64 {
        self.file.bytes()
    }
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
    pub fn total_expected_bytes(&self) -> u64 {
        self.total_expected_bytes
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
            DiscoveredEntry::new(
                file.path(),
                file.bytes(),
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
            cached_bytes,
            resumed_bytes,
        };
        progress(&event);
        super::http::check_cancel(cancel)?;
        let mut prefix = [0u8; 128];
        let actual = if let Some(cached) = transaction.try_cached(file.path())? {
            let count = transaction.read_prefix(file.path(), &mut prefix)?;
            reject_pointer(&prefix[..count])?;
            cached_bytes += file.bytes();
            event.file_bytes = file.bytes();
            event.total_bytes = completed_bytes + file.bytes();
            event.cached_bytes = cached_bytes;
            progress(&event);
            super::http::check_cancel(cancel)?;
            cached
        } else {
            let mut report = |count, resumed| {
                event.file_bytes = count;
                event.total_bytes = completed_bytes + count;
                event.resumed_bytes = resumed_bytes + resumed;
                progress(&event);
            };
            let (actual, resumed) =
                transfer_object(&mut transaction, file, transport, cancel, &mut report)?;
            resumed_bytes += resumed;
            actual
        };
        files.push(acquired(file, &actual));
        completed_bytes += file.bytes();
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
    report: &mut impl FnMut(u64, u64),
) -> Result<(ManifestEntry, u64)> {
    let mut partial = transaction.partial_object(file.path(), file.identity())?;
    let resumed = partial.resume_bytes();
    let mut prefix = [0u8; 128];
    let saved_prefix = partial.read_prefix(&mut prefix)?;
    reject_pointer(&prefix[..saved_prefix])?;
    if resumed != 0 {
        report(resumed, resumed);
        if let Err(cause) = super::http::check_cancel(cancel) {
            partial.checkpoint()?;
            return Err(cause);
        }
    }
    let range = if resumed == 0 {
        None
    } else {
        Some(ResumeRequest {
            start: resumed,
            total: file.bytes(),
            validator: partial
                .validator()
                .ok_or_else(|| Error::State("saved download lacks strong validator".into()))?
                .to_string(),
        })
    };
    let mut reader = RetryReader::open(transport, file, cancel, range)?;
    if let Some(validator) = &reader.validator {
        partial.set_validator(validator)?;
    }
    // Inspect the original prefix before adding bytes to a retained checkpoint.
    let prefix_size = prefix
        .len()
        .min(file.bytes().min(usize::MAX as u64) as usize);
    if saved_prefix < prefix_size {
        if reader
            .read_exact(&mut prefix[saved_prefix..prefix_size])
            .is_err()
        {
            partial.checkpoint()?;
            return Err(Error::State("HTTP object prefix read failed".into()));
        }
        reject_pointer(&prefix[..prefix_size])?;
        if let Err(cause) = partial.write_chunk(&prefix[saved_prefix..prefix_size]) {
            partial.checkpoint()?;
            return Err(cause);
        }
        report(partial.bytes(), resumed);
    }
    let mut checkpoint_bytes = partial.bytes();
    let mut checkpoint_time = Instant::now();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        if let Err(cause) = super::http::check_cancel(cancel) {
            partial.checkpoint()?;
            return Err(cause);
        }
        let read = match reader.read(&mut buffer) {
            Ok(read) => read,
            Err(_) => {
                partial.checkpoint()?;
                return Err(Error::State("HTTP object read failed".into()));
            }
        };
        if read == 0 {
            break;
        }
        if let Err(cause) = partial.write_chunk(&buffer[..read]) {
            partial.checkpoint()?;
            return Err(cause);
        }
        if partial.bytes() - checkpoint_bytes >= 8 * 1024 * 1024
            || checkpoint_time.elapsed() >= Duration::from_secs(1)
        {
            partial.checkpoint()?;
            checkpoint_bytes = partial.bytes();
            checkpoint_time = Instant::now();
        }
        report(partial.bytes(), resumed);
    }
    Ok((partial.finish()?, resumed))
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
                    max_body_bytes: file.bytes() - count,
                },
                cancel,
            )
            .map_err(|_| Error::State("HTTP object request failed".into()))?;
        super::http::check_cancel(cancel)?;
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
            if let Some(value) = response.header("content-length")
                && super::http::parse_u64(value)? != file.bytes()
            {
                return Err(Error::Malformed(
                    "download metadata/body length mismatch".into(),
                ));
            }
        }
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
            match self.body.read(buffer) {
                Ok(read) if read != 0 => {
                    self.count = self
                        .count
                        .checked_add(read as u64)
                        .filter(|v| *v <= self.file.bytes())
                        .ok_or_else(|| io::Error::other("download exceeded planned length"))?;
                    // The owning cache guard checks cancellation before accepting
                    // these bytes. Its caller checkpoints earlier accepted bytes
                    // if cancellation arrived while this blocking read completed.
                    return Ok(read);
                }
                Ok(0) if self.count == self.file.bytes() => return Ok(0),
                Ok(_) | Err(_) => {}
            }
            if self.retries == 2 || self.count == 0 || self.count >= self.file.bytes() {
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
                total: self.file.bytes(),
                validator,
            };
            let response = self
                .transport
                .get(
                    &HttpRequest {
                        url: self.file.url().to_string(),
                        accept: HttpAccept::Binary,
                        range: Some(range.clone()),
                        max_body_bytes: self.file.bytes() - self.count,
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

fn reject_pointer(prefix: &[u8]) -> Result<()> {
    if prefix.starts_with(b"version https://git-lfs.github.com/spec/v1") {
        return Err(Error::Unsupported(
            "Git LFS pointer requires qualified LFS resolver".into(),
        ));
    }
    Ok(())
}
