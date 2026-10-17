//! Explicit source components retain their original immutable snapshots.
//! Successful data acquisition is neither model readiness nor an atomic ref
//! spanning sources. Earlier complete snapshots may remain after a later error.

use super::http::{HttpRequest, HttpResponse, HttpTransport, check_cancel};
use super::{
    ExternalSidecars, Provider, PullLimits, PullProgress, PullReceipt, PullRequest,
    ResolvedPullPlan, Selection, execute_with_progress,
};
use crate::cache::{CacheStore, CommitOutcome, FileRole, SnapshotLease, TransferAdmission};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

const MAX_COMPONENTS: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ComponentKind {
    Weights,
    Config,
    Tokenizer,
    Auxiliary,
}

#[derive(Clone, Debug)]
pub struct ComponentRequest {
    label: String,
    kind: ComponentKind,
    request: PullRequest,
}
impl ComponentRequest {
    pub fn new(
        label: impl Into<String>,
        kind: ComponentKind,
        request: PullRequest,
    ) -> Result<Self> {
        let label = label.into();
        if label.is_empty()
            || label.len() > 64
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte))
        {
            return Err(Error::Malformed("invalid component label".into()));
        }
        Ok(Self {
            label,
            kind,
            request,
        })
    }
    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn kind(&self) -> ComponentKind {
        self.kind
    }
    pub fn request(&self) -> &PullRequest {
        &self.request
    }
}

#[derive(Clone, Debug)]
pub struct ComponentPullRequest {
    components: Vec<ComponentRequest>,
}
impl ComponentPullRequest {
    pub fn new(components: Vec<ComponentRequest>, limits: &PullLimits) -> Result<Self> {
        let request = Self { components };
        request.validate(limits)?;
        Ok(request)
    }
    pub fn components(&self) -> &[ComponentRequest] {
        &self.components
    }

    fn validate(&self, limits: &PullLimits) -> Result<()> {
        limits.validate()?;
        if self.components.is_empty() || self.components.len() > MAX_COMPONENTS {
            return Err(Error::Limit("component count exceeds bounds".into()));
        }
        let mut labels = BTreeSet::new();
        let mut suppliers = BTreeSet::new();
        let mut files = 0usize;
        let mut metadata = 0usize;
        for component in &self.components {
            component.request.validate(limits)?;
            if !labels.insert(component.label()) {
                return Err(Error::Malformed("duplicate component label".into()));
            }
            if component.kind != ComponentKind::Auxiliary && !suppliers.insert(component.kind) {
                return Err(Error::Malformed("duplicate component role supplier".into()));
            }
            let request = component.request();
            let path_bytes = match &request.selection {
                Selection::Files { paths, subfolder } => {
                    files = files.checked_add(paths.len()).ok_or_else(limit)?;
                    paths.iter().map(String::len).sum::<usize>()
                        + subfolder.as_ref().map_or(0, String::len)
                }
                Selection::ReleaseAsset { name } => {
                    files = files.checked_add(1).ok_or_else(limit)?;
                    name.len()
                }
            };
            metadata = metadata
                .checked_add(
                    component.label.len()
                        + request.repository.len()
                        + request.revision.len()
                        + path_bytes
                        + 128,
                )
                .ok_or_else(limit)?;
            if files > limits.cache.max_files || metadata > limits.cache.max_manifest_bytes {
                return Err(limit());
            }
        }
        if !suppliers.contains(&ComponentKind::Weights) {
            return Err(Error::Malformed("one weights component is required".into()));
        }
        Ok(())
    }
    fn sidecars(&self) -> ExternalSidecars {
        ExternalSidecars {
            config: self
                .components
                .iter()
                .any(|c| c.kind == ComponentKind::Config),
            tokenizer: self
                .components
                .iter()
                .any(|c| c.kind == ComponentKind::Tokenizer),
        }
    }
}

/// The caller maps explicit labels to its existing verified HTTP clients and
/// credential scopes. This interface does not create clients or infer origins.
pub trait ComponentTransport: Send + Sync {
    fn get(&self, label: &str, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse>;
    fn head(
        &self,
        _label: &str,
        _request: &HttpRequest,
        _cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        Err(Error::Unsupported(
            "component HTTP HEAD metadata is unavailable".into(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedComponent {
    label: String,
    kind: ComponentKind,
    plan: ResolvedPullPlan,
}
impl ResolvedComponent {
    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn kind(&self) -> ComponentKind {
        self.kind
    }
    pub fn plan(&self) -> &ResolvedPullPlan {
        &self.plan
    }
}
#[derive(Clone, Debug)]
pub struct ResolvedComponentPlan {
    components: Vec<ResolvedComponent>,
    total_bytes: Option<u64>,
    maximum_bytes: u64,
}
impl ResolvedComponentPlan {
    pub fn components(&self) -> &[ResolvedComponent] {
        &self.components
    }
    pub fn total_bytes(&self) -> Option<u64> {
        self.total_bytes
    }
    pub fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
    }
}

#[derive(Debug)]
pub struct VerifiedComponent {
    label: String,
    kind: ComponentKind,
    lease: SnapshotLease,
}
impl VerifiedComponent {
    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn kind(&self) -> ComponentKind {
        self.kind
    }
    pub fn lease(&self) -> &SnapshotLease {
        &self.lease
    }
}
/// Every member's original descriptor lease is verified before this data-only
/// set is returned. No mixed-source manifest or mutable composite ref exists.
#[derive(Debug)]
pub struct VerifiedComponentSet {
    components: Vec<VerifiedComponent>,
}
impl VerifiedComponentSet {
    pub fn components(&self) -> &[VerifiedComponent] {
        &self.components
    }
    pub fn component(&self, label: &str) -> Result<&VerifiedComponent> {
        self.components
            .iter()
            .find(|component| component.label == label)
            .ok_or_else(|| Error::Malformed("component is absent from verified set".into()))
    }
}
#[derive(Debug)]
pub struct ComponentReceipt {
    receipts: Vec<PullReceipt>,
    verified: VerifiedComponentSet,
}
impl ComponentReceipt {
    pub fn receipts(&self) -> &[PullReceipt] {
        &self.receipts
    }
    pub fn verified_set(&self) -> &VerifiedComponentSet {
        &self.verified
    }
    pub fn into_verified_set(self) -> VerifiedComponentSet {
        self.verified
    }
}

fn limit() -> Error {
    Error::Limit("aggregate component bounds exceeded".into())
}

struct MetadataBudget {
    bytes: usize,
    requests: usize,
}
struct Adapter<'a, T> {
    transport: &'a T,
    label: &'a str,
    budget: Option<Arc<Mutex<MetadataBudget>>>,
}
impl<T: ComponentTransport> HttpTransport for Adapter<'_, T> {
    fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        check_cancel(cancel)?;
        let mut bounded = HttpRequest {
            url: request.url.clone(),
            accept: request.accept,
            range: request.range.clone(),
            max_body_bytes: request.max_body_bytes,
        };
        if let Some(budget) = &self.budget {
            let mut budget = budget
                .lock()
                .map_err(|_| Error::State("component budget unavailable".into()))?;
            if budget.requests == 0 || budget.bytes == 0 {
                return Err(limit());
            }
            budget.requests -= 1;
            bounded.max_body_bytes = bounded.max_body_bytes.min(budget.bytes as u64);
        }
        let mut response = self
            .transport
            .get(self.label, &bounded, cancel)
            .map_err(|_| Error::State("component HTTP request failed".into()))?;
        if let Some(budget) = &self.budget {
            response.body = Box::new(BudgetBody {
                inner: response.body,
                budget: budget.clone(),
            });
        }
        Ok(response)
    }
    fn head(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        check_cancel(cancel)?;
        if let Some(budget) = &self.budget {
            let mut budget = budget
                .lock()
                .map_err(|_| Error::State("component budget unavailable".into()))?;
            if budget.requests == 0 || budget.bytes == 0 {
                return Err(limit());
            }
            budget.requests -= 1;
        }
        // max_body_bytes bounds the described object, not HEAD metadata bytes.
        // Never clamp a potentially large exact length to the JSON body budget.
        self.transport
            .head(self.label, request, cancel)
            .map_err(super::http::redact_head_error)
    }
}
struct BudgetBody {
    inner: Box<dyn Read + Send>,
    budget: Arc<Mutex<MetadataBudget>>,
}
impl Read for BudgetBody {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let mut budget = self
            .budget
            .lock()
            .map_err(|_| io::Error::other("component budget unavailable"))?;
        let count = buffer.len().min(budget.bytes.saturating_add(1));
        let read = self
            .inner
            .read(&mut buffer[..count])
            .map_err(|_| io::Error::other("component metadata read failed"))?;
        budget.bytes = budget
            .bytes
            .checked_sub(read)
            .ok_or_else(|| io::Error::other("aggregate component metadata limit"))?;
        Ok(read)
    }
}

/// Resolve only declared sources. Repeated file-source aliases are replaced
/// with the first exact commit before resolving another selector from that source.
/// Release aliases use a separate typed release-ID cohort. Every label still
/// lists/selects assets through its own adapter; release IDs do not freeze assets.
pub fn resolve_components(
    request: &ComponentPullRequest,
    transport: &impl ComponentTransport,
    limits: &PullLimits,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<ResolvedComponentPlan> {
    request.validate(limits)?;
    check_cancel(cancel)?;
    let memory = (limits.max_metadata_bytes as u64)
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(16 * 1024 * 1024))
        .ok_or_else(limit)?;
    let _lease = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: memory,
        }],
    )?;
    let budget = Arc::new(Mutex::new(MetadataBudget {
        bytes: limits.max_metadata_bytes,
        requests: limits.max_pages,
    }));
    let sidecars = request.sidecars();
    let mut file_pins = BTreeMap::<(&'static str, String, String), String>::new();
    let mut release_pins =
        BTreeMap::<(&'static str, String, String), super::github::ReleasePin>::new();
    let mut components = Vec::new();
    let mut totals = Totals::default();
    let mut all_exact = true;
    for component in request.components() {
        check_cancel(cancel)?;
        let mut selected = component.request.clone();
        let key = (
            selected.provider.label(),
            selected.repository.clone(),
            selected.revision.clone(),
        );
        if matches!(selected.selection, Selection::Files { .. })
            && let Some(pin) = file_pins.get(&key)
        {
            selected.revision = pin.clone();
        }
        let adapter = Adapter {
            transport,
            label: component.label(),
            budget: Some(budget.clone()),
        };
        let policy = if component.kind == ComponentKind::Weights {
            sidecars
        } else {
            ExternalSidecars::default()
        };
        let plan = match selected.provider {
            Provider::HuggingFace => {
                super::huggingface::resolve_component(&adapter, &selected, limits, cancel, policy)?
            }
            Provider::GitHub => {
                if matches!(selected.selection, Selection::ReleaseAsset { .. }) {
                    let (plan, pin) = super::github::resolve_release_component(
                        &adapter,
                        &selected,
                        limits,
                        cancel,
                        release_pins.get(&key).copied(),
                    )?;
                    if release_pins
                        .insert(key.clone(), pin)
                        .is_some_and(|prior| prior != pin)
                    {
                        return Err(Error::State(
                            "components disagree on a shared release pin".into(),
                        ));
                    }
                    plan
                } else {
                    super::github::resolve_component(&adapter, &selected, limits, cancel, policy)?
                }
            }
        };
        if matches!(selected.selection, Selection::Files { .. }) {
            file_pins.insert(key, plan.source().revision().to_string());
        }
        totals.add(
            plan.files()
                .iter()
                .map(|file| (file.path(), file.maximum_bytes())),
            plan.files()
                .iter()
                .map(|file| {
                    file.url().len()
                        + file.identity().len()
                        + file.validator().map_or(0, str::len)
                        + 128
                })
                .sum::<usize>()
                + component.label.len()
                + plan.source().provider().len()
                + plan.source().repository().len()
                + plan.source().revision().len(),
            limits,
        )?;
        all_exact &= plan.total_bytes().is_some();
        validate_role(
            component.kind,
            plan.files().iter().map(|file| (file.path(), file.role())),
        )?;
        if component.kind == ComponentKind::Weights {
            validate_sidecars(
                plan.files().iter().map(|file| (file.path(), file.role())),
                sidecars,
            )?;
        }
        components.push(ResolvedComponent {
            label: component.label.clone(),
            kind: component.kind,
            plan,
        });
    }
    Ok(ResolvedComponentPlan {
        components,
        total_bytes: all_exact.then_some(totals.bytes),
        maximum_bytes: totals.bytes,
    })
}

#[derive(Default)]
struct Totals {
    files: usize,
    bytes: u64,
    metadata: usize,
}
impl Totals {
    fn add<'a>(
        &mut self,
        entries: impl Iterator<Item = (&'a str, u64)>,
        extra: usize,
        limits: &PullLimits,
    ) -> Result<()> {
        self.metadata = self.metadata.checked_add(extra).ok_or_else(limit)?;
        for (path, bytes) in entries {
            if bytes > limits.cache.max_file_bytes {
                return Err(limit());
            }
            self.files = self.files.checked_add(1).ok_or_else(limit)?;
            self.bytes = self.bytes.checked_add(bytes).ok_or_else(limit)?;
            self.metadata = self.metadata.checked_add(path.len()).ok_or_else(limit)?;
        }
        if self.files > limits.cache.max_files
            || self.bytes > limits.cache.max_snapshot_bytes
            || self.metadata > limits.cache.max_manifest_bytes
        {
            return Err(limit());
        }
        Ok(())
    }
}
fn validate_role<'a>(
    kind: ComponentKind,
    entries: impl Iterator<Item = (&'a str, FileRole)>,
) -> Result<()> {
    let entries: Vec<_> = entries.collect();
    let has = |leaf, role| {
        entries
            .iter()
            .any(|(path, actual)| path.rsplit('/').next() == Some(leaf) && *actual == role)
    };
    let valid = match kind {
        ComponentKind::Weights => entries.iter().any(|(_, role)| *role == FileRole::Weights),
        ComponentKind::Config => has("config.json", FileRole::Config),
        ComponentKind::Tokenizer => has("tokenizer.json", FileRole::Tokenizer),
        ComponentKind::Auxiliary => true,
    };
    if !valid
        || kind != ComponentKind::Weights
            && entries
                .iter()
                .any(|(_, role)| matches!(role, FileRole::Weights | FileRole::Index))
    {
        return Err(Error::Unsupported(
            "selected files do not satisfy declared component role".into(),
        ));
    }
    Ok(())
}
fn validate_sidecars<'a>(
    entries: impl Iterator<Item = (&'a str, FileRole)>,
    sidecars: ExternalSidecars,
) -> Result<()> {
    let entries: Vec<_> = entries.collect();
    for (weight, role) in &entries {
        if !matches!(role, FileRole::Weights | FileRole::Index) {
            continue;
        }
        let folder = weight.rsplit_once('/').map(|(folder, _)| folder);
        for leaf in ["config.json", "tokenizer.json"] {
            let path = folder.map_or_else(|| leaf.to_string(), |folder| format!("{folder}/{leaf}"));
            if !sidecars.allows_missing(leaf)
                && !entries.iter().any(|(candidate, _)| *candidate == path)
            {
                return Err(Error::Unsupported(
                    "required component sidecar has no declared supplier".into(),
                ));
            }
        }
    }
    Ok(())
}

pub fn execute_components(
    store: &CacheStore,
    plan: &ResolvedComponentPlan,
    transport: &impl ComponentTransport,
    admission: &TransferAdmission,
    cancel: &Arc<AtomicBool>,
) -> Result<ComponentReceipt> {
    execute_components_with_progress(store, plan, transport, admission, cancel, &mut |_, _| {})
}
/// Publish each immutable source snapshot through the original executor, without
/// changing any ref. A later failure can leave earlier complete data snapshots.
pub fn execute_components_with_progress(
    store: &CacheStore,
    plan: &ResolvedComponentPlan,
    transport: &impl ComponentTransport,
    admission: &TransferAdmission,
    cancel: &Arc<AtomicBool>,
    progress: &mut impl FnMut(&str, &PullProgress<'_>),
) -> Result<ComponentReceipt> {
    let mut receipts = Vec::new();
    let mut components = Vec::new();
    for component in plan.components() {
        check_cancel(cancel)?;
        let adapter = Adapter {
            transport,
            label: component.label(),
            budget: None,
        };
        let receipt = execute_with_progress(
            store,
            component.plan(),
            &adapter,
            admission,
            cancel,
            None,
            &mut |event| progress(component.label(), event),
        )?;
        let CommitOutcome::Durable(snapshot) = receipt.outcome() else {
            return Err(Error::State(
                "component snapshot durability is unknown; inspect before retrying".into(),
            ));
        };
        let lease = store.open_snapshot_cancellable(snapshot.digest(), cancel)?;
        components.push(VerifiedComponent {
            label: component.label.clone(),
            kind: component.kind,
            lease,
        });
        receipts.push(receipt);
    }
    check_cancel(cancel)?;
    Ok(ComponentReceipt {
        receipts,
        verified: VerifiedComponentSet { components },
    })
}

/// Verify cached source and initial selection without constructing any remote
/// client. Mutable selector freshness and token revocation cannot be checked offline.
pub fn validate_cached_selection(lease: &SnapshotLease, request: &PullRequest) -> Result<()> {
    request.validate(&PullLimits::default())?;
    let source = lease.manifest().source();
    if source.provider() != request.provider.label() || source.repository() != request.repository {
        return Err(Error::Malformed(
            "cached reference has a different provider/repository".into(),
        ));
    }
    let locator_ok = match &request.selection {
        Selection::Files { .. } => {
            source.revision().len() == 40
                && source
                    .revision()
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        Selection::ReleaseAsset { .. } => {
            super::github::parse_release_locator(source.revision()).is_some()
        }
    };
    if !locator_ok {
        return Err(Error::Malformed(
            "cached reference has a different or unpinned locator kind".into(),
        ));
    }
    if matches!(request.selection, Selection::Files { .. })
        && request.revision.len() == 40
        && request.revision.bytes().all(|b| b.is_ascii_hexdigit())
        && !source.revision().eq_ignore_ascii_case(&request.revision)
    {
        return Err(Error::Malformed(
            "cached reference has a different pinned revision".into(),
        ));
    }
    let paths: Vec<String> = match &request.selection {
        Selection::Files { paths, subfolder } if request.provider == Provider::HuggingFace => {
            let available: Vec<_> = lease
                .manifest()
                .entries()
                .iter()
                .map(|entry| entry.path())
                .collect();
            super::huggingface::select_paths(paths, subfolder.as_deref(), &available)?
                .into_iter()
                .collect()
        }
        Selection::Files { paths, subfolder } => paths
            .iter()
            .map(|path| match subfolder {
                Some(folder) => format!("{folder}/{path}"),
                None => path.clone(),
            })
            .collect(),
        Selection::ReleaseAsset { name } => vec![name.clone()],
    };
    if paths.iter().any(|path| {
        !lease
            .manifest()
            .entries()
            .iter()
            .any(|entry| entry.path() == path)
    }) {
        return Err(Error::Unsupported(
            "cached snapshot lacks required selected files".into(),
        ));
    }
    // Legacy offline selection remains initial-only and role agnostic. Reuse
    // the acquisition pointer policy without adding a signal-handler contract.
    super::executor::reject_cached_pointers(lease, &AtomicBool::new(false))
}

/// Open an explicitly bound cached set. This entry point has no transport,
/// credential or transfer-admission input and never falls back to resolution.
pub fn open_components(
    store: &CacheStore,
    request: &ComponentPullRequest,
    bindings: &[(String, String)],
    limits: &PullLimits,
) -> Result<VerifiedComponentSet> {
    request.validate(limits)?;
    if bindings.len() != request.components.len() {
        return Err(Error::Malformed(
            "each component requires one cached binding".into(),
        ));
    }
    let mut selected = BTreeMap::new();
    for (label, snapshot) in bindings {
        if selected.insert(label.as_str(), snapshot.as_str()).is_some() {
            return Err(Error::Malformed(
                "duplicate cached component binding".into(),
            ));
        }
    }
    let mut components = Vec::new();
    let mut file_pins = BTreeMap::new();
    let mut release_pins = BTreeMap::new();
    let mut totals = Totals::default();
    let mut index_bytes = 0usize;
    let mut index_requests = 0usize;
    for component in request.components() {
        let snapshot = selected
            .remove(component.label())
            .ok_or_else(|| Error::Malformed("cached component binding is absent".into()))?;
        let lease = if snapshot.len() == 64
            && snapshot
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            store.open_snapshot(snapshot)?
        } else {
            store.resolve_ref(snapshot)?
        };
        validate_cached_selection(&lease, component.request())?;
        let source = lease.manifest().source();
        let key = (
            component.request.provider.label(),
            &component.request.repository,
            &component.request.revision,
        );
        match &component.request.selection {
            Selection::Files { .. } => {
                if file_pins
                    .insert(key, source.revision().to_string())
                    .is_some_and(|previous| previous != source.revision())
                {
                    return Err(Error::Malformed(
                        "cached components disagree on a shared source pin".into(),
                    ));
                }
            }
            Selection::ReleaseAsset { .. } => {
                let (pin, _) = super::github::parse_release_locator(source.revision())
                    .ok_or_else(|| Error::Malformed("invalid cached release locator".into()))?;
                if release_pins
                    .insert(key, pin)
                    .is_some_and(|previous| previous != pin)
                {
                    return Err(Error::Malformed(
                        "cached components disagree on a shared release pin".into(),
                    ));
                }
            }
        }
        validate_role(
            component.kind,
            lease
                .manifest()
                .entries()
                .iter()
                .map(|entry| (entry.path(), entry.role())),
        )?;
        if component.kind == ComponentKind::Weights {
            validate_sidecars(
                lease
                    .manifest()
                    .entries()
                    .iter()
                    .map(|entry| (entry.path(), entry.role())),
                request.sidecars(),
            )?;
            let entries: BTreeMap<_, _> = lease
                .manifest()
                .entries()
                .iter()
                .map(|entry| (entry.path().to_string(), true))
                .collect();
            let mut selected = entries.keys().cloned().collect();
            super::selection::close_model_selection(
                &entries,
                &mut selected,
                limits,
                request.sidecars(),
                &mut |path| {
                    let ordinal = lease
                        .manifest()
                        .entries()
                        .binary_search_by(|entry| entry.path().cmp(path))
                        .map_err(|_| {
                            Error::Malformed("cached index absent from manifest".into())
                        })?;
                    let bytes = usize::try_from(lease.manifest().entries()[ordinal].bytes())
                        .map_err(|_| limit())?;
                    index_requests = index_requests.checked_add(1).ok_or_else(limit)?;
                    index_bytes = index_bytes.checked_add(bytes).ok_or_else(limit)?;
                    if index_requests > limits.max_pages || index_bytes > limits.max_metadata_bytes
                    {
                        return Err(limit());
                    }
                    lease.read_verified(path, limits.max_metadata_bytes as u64)
                },
            )?;
        }
        totals.add(
            lease
                .manifest()
                .entries()
                .iter()
                .map(|entry| (entry.path(), entry.bytes())),
            lease.manifest().canonical_bytes()?.len(),
            limits,
        )?;
        components.push(VerifiedComponent {
            label: component.label.clone(),
            kind: component.kind,
            lease,
        });
    }
    Ok(VerifiedComponentSet { components })
}
