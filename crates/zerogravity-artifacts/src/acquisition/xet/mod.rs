//! Native Xet1.1 source reconstruction through the original acquisition owner.
//! Explicit native selection is separate from HF compatibility HTTP redirects.
//! Bounded reconstruction feeds the original exact-byte importer and publisher;
//! native compressed transfers use ephemeral staging without durable prefixes.
mod access;
pub(super) mod budget;
mod codec;
mod hash;
mod multipart;
mod reader;
pub(super) use reader::XetReader;
pub(super) mod reconstruction;

use super::http::{HttpRequest, HttpResponse, HttpTarget, HttpTransport, XetHubMode};
use super::{Provider, PullLimits, PullRequest, ResolvedPullPlan};
use crate::cache::{Manifest, SnapshotLease, TransferAdmission};
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};

/// Immutable protocol identity only. Expiring grants/URLs live in Access.
#[derive(Clone, Debug)]
pub(super) struct XetSpec {
    file_id: hash::XetHash,
    repository: String,
    commit: String,
}
impl XetSpec {
    pub(super) fn metadata_bytes(&self) -> usize {
        self.repository.len() + self.commit.len() + 64
    }
}
#[derive(Clone, Debug)]
pub(super) struct PlanState {
    options: XetOptions,
    seed: budget::Seed,
}

struct Hub<'a, T: ?Sized> {
    transport: &'a T,
    budget: Arc<budget::Budget>,
}
impl<T: HttpTransport + ?Sized> HttpTransport for Hub<'_, T> {
    fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        let target = HttpTarget::xet_hub(&request.url, self.budget.clone(), XetHubMode::Metadata)?;
        self.transport.get_target(request, &target, cancel)
    }
    fn head(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        let target = HttpTarget::xet_hub(&request.url, self.budget.clone(), XetHubMode::Metadata)?;
        self.transport.head_target(request, &target, cancel)
    }
}

/// Resolve one explicit native HF request. Metadata and discovery use the same
/// bounded HTTP owner. Default options never acquire CAS grants or signed
/// bodies during resolution. Explicit with_index_reconstruction permits only
/// required bounded index metadata through the same native reader.
pub fn resolve(
    transport: &impl HttpTransport,
    request: &PullRequest,
    limits: &PullLimits,
    options: &XetOptions,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    request.validate(limits)?;
    resolve_choice(
        transport,
        NativeChoice::Files(request),
        limits,
        options,
        admission,
        cancel,
    )
}
/// Explicit GGUF filename selection through the SAME native metadata/grant owner.
/// No candidate projection, repin or compatibility fallback.
pub fn resolve_quant(
    transport: &impl HttpTransport,
    request: &super::quant::GgufQuantRequest,
    limits: &PullLimits,
    options: &XetOptions,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    limits.validate()?;
    if request.source().provider() != Provider::HuggingFace || limits.unknown_length_cap.is_some() {
        return Err(Error::Unsupported(
            "native GGUF quant requires exact HF model selection".into(),
        ));
    }
    resolve_choice(
        transport,
        NativeChoice::Quant(request),
        limits,
        options,
        admission,
        cancel,
    )
}
/// Additive full-path filters through the same Hub inventory and native
/// discovery, never by re-resolving a filtered compatibility plan.
pub fn resolve_filtered(
    transport: &impl HttpTransport,
    request: &super::filters::FilterRequest,
    limits: &PullLimits,
    options: &XetOptions,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    super::quant::limits_for_quant(limits)?;
    if request.is_bundle() {
        return Err(Error::Unsupported(
            "native Xet opaque bundle selection is not qualified; use original HF bundle".into(),
        ));
    }
    if request.source().provider() != Provider::HuggingFace {
        return Err(Error::Unsupported("native filters require HF".into()));
    }
    resolve_choice(
        transport,
        NativeChoice::Filtered(request),
        limits,
        options,
        admission,
        cancel,
    )
}
#[derive(Clone, Copy)]
enum NativeChoice<'a> {
    Files(&'a PullRequest),
    Quant(&'a super::quant::GgufQuantRequest),
    Filtered(&'a super::filters::FilterRequest),
}
fn resolve_choice(
    transport: &impl HttpTransport,
    choice: NativeChoice<'_>,
    limits: &PullLimits,
    options: &XetOptions,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    options.validate()?;
    check_cancel(cancel)?;
    if matches!(choice,NativeChoice::Files(request) if request.provider!=Provider::HuggingFace) {
        return Err(Error::Unsupported(
            "native Xet requires an HF file request".into(),
        ));
    }
    let _control = if matches!(choice, NativeChoice::Filtered(_)) {
        super::quant::metadata_admission_extra(limits, admission, super::filters::CONTROL_BYTES)?
    } else {
        let bytes = (limits.max_metadata_bytes as u64)
            .checked_mul(8)
            .and_then(|v| v.checked_add(16 * 1024 * 1024))
            .ok_or_else(|| Error::Limit("Xet resolution admission overflow".into()))?;
        admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes,
            }],
        )?
    };
    let budget = Arc::new(budget::Budget::new(
        options,
        limits.max_metadata_bytes,
        budget::Seed::default(),
    )?);
    let counted = super::candidates::MetadataCalls::new(transport, limits.max_pages);
    let transport: &dyn HttpTransport =
        if matches!(choice, NativeChoice::Quant(_) | NativeChoice::Filtered(_)) {
            &counted
        } else {
            transport
        };
    let hub = Hub {
        transport,
        budget: budget.clone(),
    };
    let (mut plan, mut indexed) = {
        let source = match choice {
            NativeChoice::Files(request) => {
                let super::Selection::Files { paths, subfolder } = &request.selection else {
                    return Err(Error::Unsupported(
                        "HF release assets are not supported".into(),
                    ));
                };
                (
                    &*request.repository,
                    &*request.revision,
                    super::quant::InitialChoice::Files {
                        paths,
                        subfolder: subfolder.as_deref(),
                    },
                )
            }
            NativeChoice::Quant(request) => (
                request.source().repository(),
                request.source().revision(),
                super::quant::InitialChoice::Quant(request),
            ),
            NativeChoice::Filtered(request) => (
                request.source().repository(),
                request.source().revision(),
                super::quant::InitialChoice::Filtered(request),
            ),
        };
        let mut indexes = IndexReconstruction {
            transport,
            options,
            budget: budget.clone(),
            metadata: limits.max_metadata_bytes,
            admission,
            cancel,
            invocation: None,
            specs: BTreeMap::new(),
        };
        let plan = {
            super::huggingface::resolve_choice(
                &hub,
                source,
                limits,
                cancel,
                &format!("{}/", options.endpoints.hub),
                super::ExternalSidecars::default(),
                Some(&mut indexes),
            )?
        };
        (plan, indexes.specs)
    };
    let mut native = 0usize;
    for file in &mut plan.files {
        check_cancel(cancel)?;
        match file.hf_lfs {
            Some(false) => continue,
            Some(true) => {}
            None => {
                return Err(Error::Unsupported(
                    "native Xet needs pinned HF storage metadata".into(),
                ));
            }
        }
        if let Some(spec) = indexed.remove(file.path()) {
            file.xet = Some(spec);
        } else {
            let expected = file
                .bytes()
                .ok_or_else(|| Error::Unsupported("native Xet requires exact file size".into()))?;
            file.xet = Some(discover(
                transport,
                Discovery {
                    repository: plan.source.repository(),
                    commit: plan.source.revision(),
                    url: file.url(),
                    bytes: expected,
                    sha256: file.sha256(),
                    validator: None,
                },
                budget.clone(),
                limits.max_metadata_bytes,
                cancel,
            )?);
        }
        native += 1;
    }
    if native == 0 {
        return Err(Error::Unsupported(
            "explicit native route needs an announced Xet file".into(),
        ));
    }
    let mut result = ResolvedPullPlan::new(plan.source, plan.files, limits)?;
    result.xet = Some(PlanState {
        options: options.clone(),
        seed: budget.seed()?,
    });
    check_cancel(cancel)?;
    Ok(result)
}

struct Discovery<'a> {
    repository: &'a str,
    commit: &'a str,
    url: &'a str,
    bytes: u64,
    sha256: Option<&'a str>,
    validator: Option<&'a str>,
}
fn discover(
    transport: &dyn HttpTransport,
    contract: Discovery<'_>,
    budget: Arc<budget::Budget>,
    metadata: usize,
    cancel: &AtomicBool,
) -> Result<XetSpec> {
    check_cancel(cancel)?;
    let request = HttpRequest::new(contract.url, metadata as u64);
    let target = HttpTarget::xet_hub(&request.url, budget, XetHubMode::Discovery)?;
    let response = transport.get_target(&request, &target, cancel)?;
    check_cancel(cancel)?;
    if response.status != 302 {
        return Err(Error::Unsupported(
            "pinned HF file did not announce native Xet".into(),
        ));
    }
    super::http::validate_validator(contract.validator, &response)?;
    if response
        .header("x-repo-commit")
        .is_some_and(|value| value != contract.commit)
        || response
            .header("x-linked-size")
            .map(super::http::parse_u64)
            .transpose()?
            .is_some_and(|value| value != contract.bytes)
        || response.header("x-linked-etag").is_some_and(|value| {
            let value = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(value);
            contract.sha256 != Some(value)
        })
    {
        return Err(Error::State(
            "native Xet discovery contradicts the pinned source".into(),
        ));
    }
    let file_id = hash::XetHash::from_text(
        response
            .header("x-xet-hash")
            .ok_or_else(|| Error::Malformed("native Xet file identity is absent".into()))?,
    )?;
    Ok(XetSpec {
        file_id,
        repository: contract.repository.to_owned(),
        commit: contract.commit.to_owned(),
    })
}
/// Only one sequential, already admitted native reader; the bounded Vec is
/// covered by the outer HF resolution control envelope. No staged body cache.
struct IndexReconstruction<'a> {
    transport: &'a dyn HttpTransport,
    options: &'a XetOptions,
    budget: Arc<budget::Budget>,
    metadata: usize,
    admission: &'a TransferAdmission,
    cancel: &'a AtomicBool,
    invocation: Option<Invocation>,
    specs: BTreeMap<String, XetSpec>,
}
impl IndexReconstruction<'_> {
    fn read(&mut self, index: super::huggingface::PinnedIndex<'_>) -> Result<Vec<u8>> {
        check_cancel(self.cancel)?;
        if !self.options.index_reconstruction {
            return Err(Error::Unsupported("required native Xet index reconstruction needs explicit opt-in; metadata-only resolution never acquires CAS grants".into()));
        }
        if index.sha256.len() != 64 || !index.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::Malformed(
                "native Xet index requires pinned raw SHA256".into(),
            ));
        }
        let length = usize::try_from(index.bytes)
            .ok()
            .filter(|n| *n <= self.metadata && *n <= 8 * 1024 * 1024)
            .ok_or_else(|| Error::Limit("native Xet index metadata length".into()))?;
        let spec = discover(
            self.transport,
            Discovery {
                repository: index.repository,
                commit: index.commit,
                url: index.url,
                bytes: index.bytes,
                sha256: Some(index.sha256),
                validator: index.validator,
            },
            self.budget.clone(),
            self.metadata,
            self.cancel,
        )?;
        // Conservative exact logical-metadata charge before CAS. It is separate
        // from compressed physical body bytes charged by the original owner.
        self.budget.charge_metadata(index.bytes)?;
        if self.invocation.is_none() {
            self.invocation = Some(Invocation::from_budget(
                self.options,
                self.budget.clone(),
                self.metadata,
                self.admission,
                self.cancel,
            )?);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| Error::Limit("native Xet index allocation refused".into()))?;
        bytes.resize(length, 0);
        let mut reader = self
            .invocation
            .as_mut()
            .ok_or_else(|| Error::State("Xet index owner absent".into()))?
            .reader_spec(self.transport, &spec, index.bytes, self.cancel)?;
        let mut done = 0;
        let mut digest = Sha256::new();
        while done < length {
            check_cancel(self.cancel)?;
            let end = (done + 8192).min(length);
            let read = reader
                .read(&mut bytes[done..end])
                .map_err(|_| Error::Malformed("native Xet index reconstruction failed".into()))?;
            check_cancel(self.cancel)?;
            if read == 0 {
                return Err(Error::Malformed("native Xet index truncated".into()));
            }
            digest.update(&bytes[done..done + read]);
            done += read;
        }
        let mut extra = [0];
        if reader
            .read(&mut extra)
            .map_err(|_| Error::Malformed("native Xet index terminal verification failed".into()))?
            != 0
        {
            return Err(Error::Malformed(
                "native Xet index exceeded exact length".into(),
            ));
        }
        check_cancel(self.cancel)?;
        if !format!("{:x}", digest.finalize()).eq_ignore_ascii_case(index.sha256) {
            return Err(Error::Malformed(
                "native Xet index raw SHA256 mismatch".into(),
            ));
        }
        super::executor::reject_pointer(&bytes[..bytes.len().min(128)])?;
        self.specs.insert(index.path.to_owned(), spec);
        Ok(bytes)
    }
}

impl super::huggingface::IndexReader for IndexReconstruction<'_> {
    fn read_index(&mut self, index: super::huggingface::PinnedIndex<'_>) -> Result<Vec<u8>> {
        self.read(index)
    }
}

/// Offline eligibility only, after CacheStore has opened and fully verified
/// the original managed snapshot. Cached bytes do not attest native protocol
/// execution, remote freshness or current authorization.
pub fn validate_cached_selection(
    lease: &SnapshotLease,
    request: &PullRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
) -> Result<()> {
    request.validate(limits)?;
    check_cancel(cancel)?;
    if request.provider != Provider::HuggingFace {
        return Err(Error::Unsupported(
            "native Xet cache selection requires HF".into(),
        ));
    }
    let manifest = lease.manifest();
    if manifest.entries().len() > limits.cache.max_files
        || manifest.total_bytes() > limits.cache.max_snapshot_bytes
        || manifest
            .entries()
            .iter()
            .any(|entry| entry.bytes() > limits.cache.max_file_bytes)
    {
        return Err(Error::Limit(
            "native Xet cached selection exceeds current bounds".into(),
        ));
    }
    // Reuse original manifest validation under the supplied current limits.
    Manifest::new(
        manifest.source().clone(),
        manifest.entries().to_vec(),
        &limits.cache,
    )?;
    super::validate_cached_selection(lease, request)?;
    super::executor::reject_cached_pointers(lease, cancel)
}

pub(super) struct Invocation {
    budget: Arc<budget::Budget>,
    scratch: reader::Scratch,
    options: XetOptions,
    metadata_limit: usize,
    // Scratch descriptors/vectors close before the original admission lease.
    _lease: GlobalResourceLease,
}
impl Invocation {
    pub(super) fn new(
        plan: &ResolvedPullPlan,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<Option<Self>> {
        let Some(state) = &plan.xet else {
            return Ok(None);
        };
        let budget = Arc::new(budget::Budget::new(
            &state.options,
            plan.limits.max_metadata_bytes,
            state.seed,
        )?);
        Ok(Some(Self::from_budget(
            &state.options,
            budget,
            plan.limits.max_metadata_bytes,
            admission,
            cancel,
        )?))
    }
    fn from_budget(
        options: &XetOptions,
        budget: Arc<budget::Budget>,
        metadata: usize,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        check_cancel(cancel)?;
        let bytes = reader::Scratch::reservation(options, metadata)?;
        let lease = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes,
            }],
        )?;
        Ok(Self {
            budget,
            scratch: reader::Scratch::new(options)?,
            options: options.clone(),
            metadata_limit: metadata,
            _lease: lease,
        })
    }
    pub(super) fn ordinary(&self, file: &super::ResolvedFile) -> Result<super::ResolvedFile> {
        Ok(file.clone().with_request_target(HttpTarget::xet_hub(
            file.url(),
            self.budget.clone(),
            XetHubMode::OrdinaryPayload,
        )?))
    }
    pub(super) fn reader<'a>(
        &'a mut self,
        transport: &'a impl HttpTransport,
        file: &super::ResolvedFile,
        cancel: &'a AtomicBool,
    ) -> Result<reader::XetReader<'a>> {
        let spec = file
            .xet
            .as_ref()
            .ok_or_else(|| Error::State("Xet file identity is absent".into()))?;
        let bytes = file
            .bytes()
            .ok_or_else(|| Error::Unsupported("native Xet requires exact length".into()))?;
        self.reader_spec(transport, spec, bytes, cancel)
    }
    fn reader_spec<'a>(
        &'a mut self,
        transport: &'a dyn HttpTransport,
        spec: &XetSpec,
        bytes: u64,
        cancel: &'a AtomicBool,
    ) -> Result<reader::XetReader<'a>> {
        let mut access = access::Access::new(
            transport,
            spec,
            bytes,
            &self.options,
            self.budget.clone(),
            self.metadata_limit,
            cancel,
        )?;
        let reconstruction = access.reconstruction()?;
        reader::XetReader::new(
            Box::new(access),
            reconstruction,
            (spec.file_id, bytes),
            self.budget.clone(),
            &mut self.scratch,
            cancel,
        )
    }
}

/// Finite work bounds for one explicit native operation, including discarded
/// and retried range work. Original HTTP per-request deadlines still apply;
/// there is no new single whole-operation elapsed-time promise.
/// Terms/groups/ranges count accepted reconstruction entries cumulatively,
/// including later files and grant reissues, as well as bounding each document.
/// A reissue of a full-size document can therefore exhaust its cardinality cap.
#[derive(Clone, Debug)]
pub struct XetOptions {
    pub max_terms: usize,
    pub max_fetch_groups: usize,
    pub max_ranges: usize,
    pub max_chunks: usize,
    pub max_requests: usize,
    pub max_wire_bytes: u64,
    pub max_refreshes: usize,
    endpoints: Endpoints,
    index_reconstruction: bool,
}
#[derive(Clone, Debug)]
struct Endpoints {
    hub: String,
    cas: String,
    diagnostic: bool,
}
impl Default for XetOptions {
    fn default() -> Self {
        Self {
            max_terms: 4096,
            max_fetch_groups: 4096,
            max_ranges: 8192,
            max_chunks: 262144,
            max_requests: 16384,
            max_wire_bytes: 64 * 1024 * 1024 * 1024,
            max_refreshes: 2,
            index_reconstruction: false,
            endpoints: Endpoints {
                hub: "https://huggingface.co".into(),
                cas: "https://cas-server.xethub.hf.co".into(),
                diagnostic: false,
            },
        }
    }
}
impl XetOptions {
    /// Explicitly permit bounded required native index metadata reconstruction
    /// during resolution. Default remains metadata/discovery-only, with no CAS
    /// grants or signed bodies. Index and later payload consume ONE work budget.
    pub fn with_index_reconstruction(mut self) -> Self {
        self.index_reconstruction = true;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_terms == 0
            || self.max_terms > 65536
            || self.max_fetch_groups == 0
            || self.max_fetch_groups > 65536
            || self.max_ranges == 0
            || self.max_ranges > 65536
            || self.max_chunks == 0
            || self.max_chunks > hash::MAX_HASH_CHUNKS
            || self.max_requests == 0
            || self.max_requests > 65536
            || self.max_wire_bytes == 0
            || self.max_wire_bytes > 1024 * 1024 * 1024 * 1024
            || self.max_refreshes > 8
        {
            return Err(Error::Limit("invalid bounded Xet options".into()));
        }
        Ok(())
    }
    #[cfg(any(test, feature = "diagnostic-acquisition"))]
    pub fn diagnostic_endpoints(mut self, hub: &str, cas: &str) -> Result<Self> {
        fn fixture(value: &str) -> Result<String> {
            let parsed = super::http::parse_url(value, true)?;
            if parsed.path() != "/" || parsed.query().is_some() || parsed.fragment().is_some() {
                return Err(Error::Malformed("invalid Xet diagnostic origin".into()));
            }
            Ok(parsed.origin().ascii_serialization())
        }
        self.endpoints = Endpoints {
            hub: fixture(hub)?,
            cas: fixture(cas)?,
            diagnostic: true,
        };
        self.validate()?;
        Ok(self)
    }
}
fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::State("Xet acquisition canceled".into()))
    } else {
        Ok(())
    }
}
