use super::http::{HttpAccept, HttpRequest, HttpTransport};
use super::resume::{ResumeDisposition, ResumeRequest, strong_etag, validate_range_response};
use super::{DigestProvenance, ResolvedFile, ResolvedPullPlan};
use crate::cache::{
    CacheStore, CommitOutcome, DiscoveredEntry, Manifest, ManifestEntry, TransferAdmission,
};
use crate::{Error, Result};
use std::collections::BTreeSet;
use std::io::{self, Read};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

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

/// The cache admits complete listed lengths before any HTTP body is requested.
/// Cross-invocation partial-file resume remains Unsupported; a live transfer may
/// resume at most twice, only against the same strong ETag and exact total.
pub fn execute(
    store: &CacheStore,
    plan: &ResolvedPullPlan,
    transport: &impl HttpTransport,
    admission: &TransferAdmission,
    cancel: &Arc<AtomicBool>,
    reference: Option<&str>,
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
    for file in plan.files() {
        let mut prefix = [0u8; 128];
        let actual = if let Some(cached) = transaction.try_cached(file.path())? {
            let count = transaction.read_prefix(file.path(), &mut prefix)?;
            reject_pointer(&prefix[..count])?;
            cached
        } else {
            let mut reader = RetryReader::open(transport, file, cancel)?;
            let count = prefix
                .len()
                .min(file.bytes().min(usize::MAX as u64) as usize);
            reader.read_exact(&mut prefix[..count])?;
            reject_pointer(&prefix[..count])?;
            transaction.import_object(
                file.path(),
                &mut io::Cursor::new(&prefix[..count]).chain(reader),
            )?
        };
        files.push(acquired(file, &actual));
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
    fn open(transport: &'a T, file: &'a ResolvedFile, cancel: &'a AtomicBool) -> Result<Self> {
        let response = transport.get(&HttpRequest::new(file.url(), file.bytes()), cancel)?;
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
            && value.parse::<u64>().ok() != Some(file.bytes())
        {
            return Err(Error::Malformed(
                "download metadata/body length mismatch".into(),
            ));
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
            count: 0,
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
