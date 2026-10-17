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

struct Hub<'a, T> {
    transport: &'a T,
    budget: Arc<budget::Budget>,
}
impl<T: HttpTransport> HttpTransport for Hub<'_, T> {
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
/// bounded HTTP owner; this never acquires CAS grants or signed payload bodies.
pub fn resolve(
    transport: &impl HttpTransport,
    request: &PullRequest,
    limits: &PullLimits,
    options: &XetOptions,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    request.validate(limits)?;
    options.validate()?;
    check_cancel(cancel)?;
    if request.provider != Provider::HuggingFace {
        return Err(Error::Unsupported(
            "native Xet requires an HF file request".into(),
        ));
    }
    let bytes = (limits.max_metadata_bytes as u64)
        .checked_mul(8)
        .and_then(|v| v.checked_add(16 * 1024 * 1024))
        .ok_or_else(|| Error::Limit("Xet resolution admission overflow".into()))?;
    let _control = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes,
        }],
    )?;
    let budget = Arc::new(budget::Budget::new(
        options,
        limits.max_metadata_bytes,
        budget::Seed::default(),
    )?);
    let hub = Hub {
        transport,
        budget: budget.clone(),
    };
    let mut plan = super::huggingface::resolve_from(
        &hub,
        request,
        limits,
        cancel,
        &format!("{}/", options.endpoints.hub),
        super::ExternalSidecars::default(),
    )?;
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
        let expected = file
            .bytes()
            .ok_or_else(|| Error::Unsupported("native Xet requires exact file size".into()))?;
        let request = HttpRequest::new(file.url(), limits.max_metadata_bytes as u64);
        let target = HttpTarget::xet_hub(&request.url, budget.clone(), XetHubMode::Discovery)?;
        let response = transport.get_target(&request, &target, cancel)?;
        if response.status != 302 {
            return Err(Error::Unsupported(
                "pinned HF file did not announce native Xet".into(),
            ));
        }
        if response
            .header("x-repo-commit")
            .is_some_and(|value| value != plan.source.revision())
            || response
                .header("x-linked-size")
                .map(super::http::parse_u64)
                .transpose()?
                .is_some_and(|value| value != expected)
            || response.header("x-linked-etag").is_some_and(|value| {
                let value = value
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .unwrap_or(value);
                file.sha256() != Some(value)
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
        file.xet = Some(XetSpec {
            file_id,
            repository: plan.source.repository().to_string(),
            commit: plan.source.revision().to_string(),
        });
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
        seed: budget.seed(),
    });
    check_cancel(cancel)?;
    Ok(result)
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
        check_cancel(cancel)?;
        let bytes = reader::Scratch::reservation(&state.options, plan.limits.max_metadata_bytes)?;
        let lease = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes,
            }],
        )?;
        Ok(Some(Self {
            budget: Arc::new(budget::Budget::new(
                &state.options,
                plan.limits.max_metadata_bytes,
                state.seed,
            )?),
            scratch: reader::Scratch::new(&state.options)?,
            options: state.options.clone(),
            metadata_limit: plan.limits.max_metadata_bytes,
            _lease: lease,
        }))
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
            endpoints: Endpoints {
                hub: "https://huggingface.co".into(),
                cas: "https://cas-server.xethub.hf.co".into(),
                diagnostic: false,
            },
        }
    }
}
impl XetOptions {
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
