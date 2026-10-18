//! Source-grant-checked, immutable, bounded context snapshots.
//!
//! A trusted application's ContextOwner commits records and controls grants.
//! Its ContextReader is bound to ONE authenticated tenant/session identity and
//! can only capture records eligible for that identity. A configured source
//! selector or model decision does not issue grants. This is a real in-memory
//! owner/reader and filter path; it is NOT a connected conversation provider,
//! a model-runtime NanoJev adapter, or a TaskCoordinator StateRoot publisher.
//!
//! Never hold the ledger mutex while invoking a decision adapter. An epoch
//! change (including revocation or concurrent source update) between reading
//! and finishing decisions invalidates the complete candidate snapshot.
use crate::context_filter::{AuthorizedRecord, ContextSource, DecisionVerdict};
use crate::pipeline::{ChatContextPolicy, ValueKind};
use crate::{Error, Result};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

const MAX_STORED_RECORDS: usize = 4096;
const MAX_STORED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECORD_BYTES: usize = ChatContextPolicy::MAX_BYTES;
const MAX_CANDIDATE_RECORDS: usize = 1024;
const MAX_CANDIDATE_BYTES: usize = 4 * ChatContextPolicy::MAX_BYTES;
static NEXT_CONTEXT_OWNER: AtomicU64 = AtomicU64::new(1);

fn state_lock(state: &Mutex<Ledger>) -> Result<MutexGuard<'_, Ledger>> {
    state
        .lock()
        .map_err(|_| Error::State("context source owner poisoned".into()))
}
fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && !value.starts_with('.')
        && !value.contains("..")
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SessionScope {
    pub tenant: String,
    pub application: String,
    pub session: String,
}
impl SessionScope {
    /// Legacy/local default application identity.
    pub fn new(tenant: &str, session: &str) -> Result<Self> {
        Self::in_application(tenant, "default", session)
    }
    /// Bind both the caller and application-owned prompt records to the
    /// exact app ID; another app in the same tenant does not gain access.
    pub fn in_application(tenant: &str, application: &str, session: &str) -> Result<Self> {
        if !valid_id(tenant) || !valid_id(application) || !valid_id(session) {
            return Err(Error::Malformed(
                "invalid authenticated tenant/application/session".into(),
            ));
        }
        Ok(Self {
            tenant: tenant.into(),
            application: application.into(),
            session: session.into(),
        })
    }
}

struct SourceRecord {
    id: u64,
    publisher_scope: SessionScope,
    source: ContextSource,
    /// For session-local records, this is the committed turn of the original
    /// session; the current request's turn is never a legal context input.
    turn: u64,
    tool_name: Option<String>,
    text: String,
}
struct Ledger {
    owner_id: u64,
    epoch: u64,
    next_record: u64,
    records: Vec<SourceRecord>,
    stored_bytes: usize,
    // Exact reader/selector grants, not a capability to scan all sources.
    grants: BTreeSet<(SessionScope, ContextSource)>,
}

/// Exclusive trusted source-owner handle. Do not expose it to a remote
/// request/client or checkpoint-supplied code. It is the only authority to
/// commit context records, grant sharing and revoke grants.
pub struct ContextOwner {
    ledger: Arc<Mutex<Ledger>>,
}

/// Read-only capability minted by the trusted owner for one authenticated
/// scope. A caller cannot retarget a reader to another scope or add grants.
#[derive(Clone)]
pub struct ContextReader {
    ledger: Arc<Mutex<Ledger>>,
    scope: SessionScope,
}
/// Immutable source record view with stable original provenance.
#[derive(Clone)]
pub struct FrozenContextRecord {
    id: u64,
    source: ContextSource,
    publisher: SessionScope,
    committed_turn: u64,
    tool_name: Option<String>,
    text: String,
}
impl FrozenContextRecord {
    pub const fn id(&self) -> u64 {
        self.id
    }
    pub fn source(&self) -> &ContextSource {
        &self.source
    }
    pub fn published_by(&self) -> &SessionScope {
        &self.publisher
    }
    pub const fn committed_turn(&self) -> u64 {
        self.committed_turn
    }
    pub fn tool_name(&self) -> Option<&str> {
        self.tool_name.as_deref()
    }
    pub fn text(&self) -> &str {
        &self.text
    }
}
/// A snapshot is evidence for a specific owner/grant/source epoch. Before
/// giving its contents to an external model or media consumer, re-check
/// freshness with the issuing ContextReader at the delivery boundary.
/// Previously disclosed data cannot be retroactively withdrawn.
pub struct ContextSnapshot {
    owner_id: u64,
    epoch: u64,
    scope: SessionScope,
    records: Vec<FrozenContextRecord>,
    /// Secret-free typed verdict metadata. No source text or private
    /// questions are put into the audit trail.
    decision_audit: Vec<DecisionAudit>,
    bytes: usize,
    truncated: bool,
}
#[derive(Clone)]
pub struct DecisionAudit {
    record_id: u64,
    model: String,
    stage: usize,
    verdict: Option<DecisionVerdict>,
}
impl DecisionAudit {
    pub const fn record_id(&self) -> u64 {
        self.record_id
    }
    pub fn model(&self) -> &str {
        &self.model
    }
    pub const fn stage(&self) -> usize {
        self.stage
    }
    pub const fn verdict(&self) -> Option<DecisionVerdict> {
        self.verdict
    }
}
impl ContextSnapshot {
    pub fn records(&self) -> &[FrozenContextRecord] {
        &self.records
    }
    pub fn decision_audit(&self) -> &[DecisionAudit] {
        &self.decision_audit
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn truncated(&self) -> bool {
        self.truncated
    }
    pub fn value_kind(&self) -> ValueKind {
        ValueKind::CommittedChatContext
    }
    pub fn source_epoch(&self) -> u64 {
        self.epoch
    }
}

/// Exact record text, record origin, tool name and current source identity
/// of a *previously granted* candidate are available to the adapter.
/// Model routing and numerical calibration are caller-owned and NOT supplied
/// by this structural interface. A missing adapter may return None, making
/// the existing omit/reject/keep_authorized behavior explicit.
pub struct DecisionQuery<'a> {
    pub model: &'a str,
    pub question: &'a str,
    pub stage: usize,
    pub record: &'a AuthorizedRecord<'a>,
    pub record_id: u64,
}
pub trait ContextDecisionAdapter {
    fn decide(&mut self, request: DecisionQuery<'_>) -> Result<Option<DecisionVerdict>>;
}
impl<F> ContextDecisionAdapter for F
where
    F: for<'a> FnMut(DecisionQuery<'a>) -> Result<Option<DecisionVerdict>>,
{
    fn decide(&mut self, request: DecisionQuery<'_>) -> Result<Option<DecisionVerdict>> {
        self(request)
    }
}
/// No model used: the configured per-stage unavailability policy governs.
pub struct NoDecisionModel;
impl ContextDecisionAdapter for NoDecisionModel {
    fn decide(&mut self, _: DecisionQuery<'_>) -> Result<Option<DecisionVerdict>> {
        Ok(None)
    }
}
impl ContextOwner {
    pub fn new() -> Result<Self> {
        let owner_id = NEXT_CONTEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| Error::Limit("context owner IDs exhausted".into()))?;
        Ok(Self {
            ledger: Arc::new(Mutex::new(Ledger {
                owner_id,
                epoch: 0,
                next_record: 1,
                records: Vec::new(),
                stored_bytes: 0,
                grants: BTreeSet::new(),
            })),
        })
    }
    /// Mint after the enclosing application authenticates tenant/session.
    /// Ownership of this ContextOwner already confers permission to mint
    /// scoped readers and to maintain source grants.
    pub fn reader(&self, authenticated_scope: SessionScope) -> ContextReader {
        ContextReader {
            ledger: Arc::clone(&self.ledger),
            scope: authenticated_scope,
        }
    }
    /// An explicit grant is necessary even for application prompts in the
    /// same tenant; cross-user sharing requires the named shared source's
    /// exact grant. No wildcard and no implicit cross-tenant application use.
    pub fn grant(&self, reader: &SessionScope, source: &ContextSource) -> Result<bool> {
        ContextSource::parse(&source.label())?;
        let mut inner = state_lock(&self.ledger)?;
        if inner.grants.contains(&(reader.clone(), source.clone())) {
            return Ok(false);
        }
        let next = inner
            .epoch
            .checked_add(1)
            .ok_or_else(|| Error::Limit("context source epoch".into()))?;
        if inner.grants.len() >= 4096 {
            return Err(Error::Limit("context grant count".into()));
        }
        inner.grants.insert((reader.clone(), source.clone()));
        inner.epoch = next;
        Ok(true)
    }
    pub fn revoke(&self, reader: &SessionScope, source: &ContextSource) -> Result<bool> {
        let mut inner = state_lock(&self.ledger)?;
        if !inner.grants.contains(&(reader.clone(), source.clone())) {
            return Ok(false);
        }
        let next = inner
            .epoch
            .checked_add(1)
            .ok_or_else(|| Error::Limit("context source epoch".into()))?;
        inner.grants.remove(&(reader.clone(), source.clone()));
        inner.epoch = next;
        Ok(true)
    }
    /// Commit an app-authenticated record; this does not grant reader access.
    /// Session-sourced records must have a nonzero committed turn. A prompt
    /// registered with the application may use turn zero.
    pub fn commit(
        &self,
        publisher: &SessionScope,
        source: &ContextSource,
        turn: u64,
        tool_name: Option<&str>,
        text: &str,
    ) -> Result<u64> {
        ContextSource::parse(&source.label())?;
        if text.is_empty()
            || text.len() > MAX_RECORD_BYTES
            || tool_name.is_some_and(|name| !valid_id(name))
            || source.origin == "session" && turn == 0
        {
            return Err(Error::Malformed(
                "invalid bounded committed context record".into(),
            ));
        }
        // Every record is immutable once accepted. Publication is serialized
        // and all metadata + bodies are owned by the source authority.
        let mut inner = state_lock(&self.ledger)?;
        if source.origin == "session"
            && inner
                .records
                .iter()
                .rev()
                .find(|r| r.source.origin == "session" && r.publisher_scope == *publisher)
                .is_some_and(|prior| prior.turn > turn)
        {
            return Err(Error::State("out-of-order committed session turn".into()));
        }
        let weight = text
            .len()
            .checked_add(tool_name.map_or(0, str::len))
            .and_then(|n| n.checked_add(source.label().len()))
            .ok_or_else(|| Error::Limit("context retained bytes".into()))?;
        let next_bytes = inner
            .stored_bytes
            .checked_add(weight)
            .ok_or_else(|| Error::Limit("context retained bytes".into()))?;
        if next_bytes > MAX_STORED_BYTES || inner.records.len() >= MAX_STORED_RECORDS {
            return Err(Error::Limit(
                "context store capacity; retire old records first".into(),
            ));
        }
        let new_epoch = inner
            .epoch
            .checked_add(1)
            .ok_or_else(|| Error::Limit("context source epoch".into()))?;
        let next_id = inner
            .next_record
            .checked_add(1)
            .ok_or_else(|| Error::Limit("context record IDs".into()))?;
        let id = inner.next_record;
        inner.records.push(SourceRecord {
            id,
            publisher_scope: publisher.clone(),
            source: source.clone(),
            turn,
            tool_name: tool_name.map(str::to_owned),
            text: text.to_owned(),
        });
        inner.stored_bytes = next_bytes;
        inner.epoch = new_epoch;
        inner.next_record = next_id;
        Ok(id)
    }
    /// Trusted explicit retention/erasure API. Session and shared data must
    /// never be silently evicted while source identity is externally asserted.
    pub fn erase(&self, record_id: u64) -> Result<bool> {
        let mut inner = state_lock(&self.ledger)?;
        let Some(index) = inner.records.iter().position(|r| r.id == record_id) else {
            return Ok(false);
        };
        let next = inner
            .epoch
            .checked_add(1)
            .ok_or_else(|| Error::Limit("context source epoch".into()))?;
        let record = inner.records.remove(index);
        let size = record.text.len()
            + record.tool_name.as_ref().map_or(0, String::len)
            + record.source.label().len();
        inner.stored_bytes -= size;
        inner.epoch = next;
        Ok(true)
    }
}
impl ContextReader {
    pub fn scope(&self) -> &SessionScope {
        &self.scope
    }

    fn permitted(
        &self,
        inner: &Ledger,
        record: &SourceRecord,
        current_turn: u64,
        requested: &BTreeSet<ContextSource>,
    ) -> bool {
        if !requested.contains(&record.source) {
            return false;
        }
        if record.source.origin == "session" {
            return record.publisher_scope == self.scope && record.turn < current_turn;
        }
        // Source service/host explicitly controls app data in its own tenant.
        if record.source.origin == "application"
            && (record.publisher_scope.tenant != self.scope.tenant
                || record.publisher_scope.application != self.scope.application)
        {
            return false;
        }
        inner
            .grants
            .contains(&(self.scope.clone(), record.source.clone()))
    }

    /// Capture a bounded, immutable, original-order snapshot under one source
    /// epoch. With disabled policy, skip even reading the owner. The reader
    /// never grants itself new source/tenant access. The candidate window is
    /// at most 1024 latest complete records/4MiB; selected output is at most
    /// the configured snapshot byte bound, retaining latest complete records
    /// on over-budget results and explicitly setting `truncated`.
    pub fn capture(
        &self,
        policy: &ChatContextPolicy,
        current_turn: u64,
        decision: &mut impl ContextDecisionAdapter,
    ) -> Result<Option<ContextSnapshot>> {
        if !policy.enabled {
            return Ok(None);
        }
        if !valid_id(&policy.from_model) {
            return Err(Error::Malformed("invalid context model instance".into()));
        }
        if current_turn == 0
            || policy.max_bytes == 0
            || policy.max_bytes > ChatContextPolicy::MAX_BYTES
        {
            return Err(Error::Limit("context capture turn/bytes".into()));
        }
        policy.selection.validate()?;
        let requested: BTreeSet<_> = policy.selection.sources.iter().cloned().collect();
        let (owner_id, epoch, candidate, had_more) = {
            let inner = state_lock(&self.ledger)?;
            let mut candidate = Vec::new();
            let mut candidate_bytes = 0usize;
            let mut had_more = false;
            for record in inner.records.iter().rev() {
                if !self.permitted(&inner, record, current_turn, &requested) {
                    continue;
                }
                let charge = record.text.len()
                    + record.tool_name.as_ref().map_or(0, String::len)
                    + record.source.label().len();
                if candidate.len() >= MAX_CANDIDATE_RECORDS
                    || candidate_bytes
                        .checked_add(charge)
                        .is_none_or(|n| n > MAX_CANDIDATE_BYTES)
                {
                    had_more = true;
                    break;
                }
                candidate_bytes += charge;
                candidate.push(FrozenContextRecord {
                    id: record.id,
                    source: record.source.clone(),
                    publisher: record.publisher_scope.clone(),
                    committed_turn: record.turn,
                    tool_name: record.tool_name.clone(),
                    text: record.text.clone(),
                });
            }
            candidate.reverse();
            (inner.owner_id, inner.epoch, candidate, had_more)
        };
        let views: Vec<_> = candidate
            .iter()
            .map(|item| AuthorizedRecord {
                source: item.source.clone(),
                tool_name: item.tool_name.as_deref(),
                text: &item.text,
            })
            .collect();
        let mut decision_audit = Vec::new();
        let selected = policy.selection.select_authorized_with_record(
            &views,
            |model, question, stage, index, item| {
                let verdict = decision.decide(DecisionQuery {
                    model,
                    question,
                    stage,
                    record: item,
                    record_id: candidate[index].id,
                })?;
                decision_audit.push(DecisionAudit {
                    record_id: candidate[index].id,
                    model: model.to_owned(),
                    stage,
                    verdict,
                });
                Ok(verdict)
            },
        )?;
        let mut kept = Vec::new();
        let mut bytes = 0usize;
        let mut truncated = had_more;
        // Keep latest complete records only. Apply the final selected output
        // budget after filtering: the decision may remove large candidates.
        for index in selected.into_iter().rev() {
            let record = &candidate[index];
            let weight = record.text.len()
                + record.tool_name.as_ref().map_or(0, String::len)
                + record.source.label().len();
            if bytes
                .checked_add(weight)
                .is_none_or(|n| n > policy.max_bytes)
            {
                // Keep one contiguous newest accepted history window,
                // never silently drop an intervening record and then
                // splice unrelated older records into the transcript.
                truncated = true;
                break;
            }
            bytes += weight;
            kept.push(record.clone());
        }
        kept.reverse();
        // Filters and decision model must never run while holding the ledger
        // lock. Any revoke, grant, erase or new commit during a decision
        // invalidates the candidate and refuses all publication.
        let current = state_lock(&self.ledger)?;
        if current.owner_id != owner_id || current.epoch != epoch {
            return Err(Error::State(
                "context source or grant changed during snapshot".into(),
            ));
        }
        drop(current);
        Ok(Some(ContextSnapshot {
            owner_id,
            epoch,
            scope: self.scope.clone(),
            records: kept,
            decision_audit,
            bytes,
            truncated,
        }))
    }
    /// Revalidate at the consuming coordinator's publication/admission
    /// boundary. A previously delivered copy may still contain old bytes:
    /// authorization cannot retroactively erase data after delivery.
    pub fn require_fresh(&self, snapshot: &ContextSnapshot) -> Result<()> {
        let inner = state_lock(&self.ledger)?;
        if inner.owner_id != snapshot.owner_id
            || inner.epoch != snapshot.epoch
            || snapshot.scope != self.scope
        {
            return Err(Error::State("stale or foreign context snapshot".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_filter::{
        ContextFilter, ContextFilterPlan, DecisionUnavailable, FilterAction, MatchField, RecordKind,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn user() -> ContextSource {
        ContextSource::parse("session/user_message").unwrap()
    }
    fn assistant() -> ContextSource {
        ContextSource::parse("session/assistant_message").unwrap()
    }
    fn tool() -> ContextSource {
        ContextSource::parse("session/tool_result").unwrap()
    }
    fn system() -> ContextSource {
        ContextSource::parse("application/system_prompt").unwrap()
    }
    fn share() -> ContextSource {
        ContextSource::parse("shared:team-docs/assistant_message").unwrap()
    }
    fn scope(tenant: &str, session: &str) -> SessionScope {
        SessionScope::new(tenant, session).unwrap()
    }
    fn policy() -> ChatContextPolicy {
        ChatContextPolicy {
            enabled: true,
            from_model: "assistant".into(),
            max_bytes: 8192,
            selection: ContextFilterPlan::default(),
        }
    }
    #[test]
    fn session_scope_and_committed_turn_gate_prevent_foreign_or_future_context() {
        let owner = ContextOwner::new().unwrap();
        let a = scope("alice", "chat");
        let b = scope("alice", "other");
        let c = scope("bob", "chat");
        let reader = owner.reader(a.clone());
        let a1 = owner
            .commit(&a, &user(), 1, None, "a previous user turn")
            .unwrap();
        let a2 = owner
            .commit(&a, &assistant(), 1, None, "a previous assistant turn")
            .unwrap();
        owner
            .commit(&a, &user(), 2, None, "current in progress")
            .unwrap();
        owner.commit(&b, &user(), 1, None, "other session").unwrap();
        owner
            .commit(&c, &user(), 1, None, "different tenant")
            .unwrap();
        assert!(a2 > a1);
        let first = reader
            .capture(&policy(), 2, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        assert_eq!(first.records().len(), 2);
        assert_eq!(first.records()[0].id(), a1);
        assert_eq!(first.records()[1].id(), a2);
        assert_eq!(first.value_kind(), ValueKind::CommittedChatContext);
        assert!(first.records()[0].published_by() == &a);
        assert!(!first.truncated());
        reader.require_fresh(&first).unwrap();
        assert!(owner.reader(b.clone()).require_fresh(&first).is_err());
        assert!(owner.reader(c.clone()).require_fresh(&first).is_err());
        assert!(
            ContextOwner::new()
                .unwrap()
                .reader(a)
                .require_fresh(&first)
                .is_err()
        );
        assert!(reader.capture(&policy(), 0, &mut NoDecisionModel).is_err());
        assert!(owner.commit(&b, &user(), 0, None, "not committed").is_err());
        assert!(owner.commit(&b, &user(), 0, None, "not committed").is_err());
        assert!(owner.commit(&b, &user(), 2, None, "next turn").is_ok());
        assert!(owner.commit(&b, &user(), 1, None, "older turn").is_err());
    }
    #[test]
    fn explicit_app_and_cross_tenant_shared_grants_are_exact_scoped_and_revocable() {
        let owner = ContextOwner::new().unwrap();
        let alice = scope("team", "talk");
        let colleague = scope("team", "other");
        let foreign = scope("different-team", "shared-chat");
        let alice_reader = owner.reader(alice.clone());
        owner.commit(&alice, &user(), 1, None, "hello").unwrap();
        owner
            .commit(&alice, &tool(), 1, Some("web"), "tool data")
            .unwrap();
        owner
            .commit(&alice, &system(), 0, None, "tenant owned system prompt")
            .unwrap();
        owner
            .commit(&foreign, &system(), 0, None, "foreign tenant prompt")
            .unwrap();
        owner
            .commit(
                &foreign,
                &share(),
                7,
                None,
                "explicitly shared cross-team summary",
            )
            .unwrap();
        let mut plan = policy();
        plan.selection.sources.extend([tool(), system(), share()]);
        // Requested sources do not grant read permission.
        let foreign_app = SessionScope::in_application("team", "other-app", "config").unwrap();
        owner
            .commit(
                &foreign_app,
                &system(),
                0,
                None,
                "foreign app in the same tenant",
            )
            .unwrap();
        let before = alice_reader
            .capture(&plan, 2, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        assert_eq!(before.records().len(), 2);
        assert_eq!(before.records()[0].text(), "hello");
        assert_eq!(before.records()[1].text(), "tool data");
        assert!(owner.grant(&alice, &tool()).unwrap()); // the explicit grant is redundant for same-session records
        assert!(owner.grant(&alice, &system()).unwrap());
        assert!(owner.grant(&alice, &share()).unwrap());
        assert!(!owner.grant(&alice, &share()).unwrap());
        let after = alice_reader
            .capture(&plan, 2, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        assert_eq!(after.records().len(), 4);
        assert!(
            after
                .records()
                .iter()
                .any(|r| r.text() == "tenant owned system prompt")
        );
        assert!(
            after
                .records()
                .iter()
                .any(|r| r.text() == "explicitly shared cross-team summary")
        );
        assert!(
            !after
                .records()
                .iter()
                .any(|r| r.text() == "foreign tenant prompt")
        );
        assert!(
            !after
                .records()
                .iter()
                .any(|r| r.text() == "foreign app in the same tenant")
        );
        assert!(
            owner
                .reader(colleague)
                .capture(&plan, 2, &mut NoDecisionModel)
                .unwrap()
                .unwrap()
                .records()
                .is_empty()
        );
        owner.revoke(&alice, &share()).unwrap();
        assert!(alice_reader.require_fresh(&after).is_err());
        let revised = alice_reader
            .capture(&plan, 2, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        assert_eq!(revised.records().len(), 3);
        assert!(
            !revised
                .records()
                .iter()
                .any(|r| r.source().origin.starts_with("shared:"))
        );
        assert!(!owner.revoke(&alice, &share()).unwrap());
    }
    #[test]
    fn granted_tool_results_can_be_filtered_without_reconstructing_source_or_running_model() {
        let owner = ContextOwner::new().unwrap();
        let scope = scope("team", "talk");
        let reader = owner.reader(scope.clone());
        owner
            .commit(&scope, &user(), 1, None, "Tell me about the plan")
            .unwrap();
        owner
            .commit(
                &scope,
                &tool(),
                1,
                Some("search"),
                "confidential tool output",
            )
            .unwrap();
        owner
            .commit(&scope, &assistant(), 1, None, "I found the plan")
            .unwrap();
        let mut plan = policy();
        plan.selection.sources.push(tool());
        plan.selection.stages.push(ContextFilter::Match {
            action: FilterAction::Exclude,
            field: MatchField::Kind,
            values: vec![RecordKind::ToolResult.label().into()],
        });
        let snap = reader
            .capture(&plan, 2, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        assert_eq!(snap.records().len(), 2);
        assert!(
            !snap
                .records()
                .iter()
                .any(|r| r.source().kind == RecordKind::ToolResult)
        );
        assert_eq!(snap.records()[1].text(), "I found the plan");
    }

    struct CheckingDecision {
        seen: Vec<(u64, String)>,
    }
    impl ContextDecisionAdapter for CheckingDecision {
        fn decide(&mut self, query: DecisionQuery<'_>) -> Result<Option<DecisionVerdict>> {
            assert_eq!(query.model, "nanojev");
            assert_eq!(query.question, "Relevant to speech?");
            assert_eq!(query.stage, 1);
            self.seen
                .push((query.record_id, query.record.text.to_owned()));
            Ok(Some(DecisionVerdict {
                include: query.record.text.contains("speech"),
                confidence_bps: 9_000,
            }))
        }
    }
    #[test]
    fn decision_adapter_only_receives_survivors_and_absent_decision_uses_explicit_policy() {
        let owner = ContextOwner::new().unwrap();
        let scope = scope("team", "talk");
        let reader = owner.reader(scope.clone());
        let a = owner
            .commit(&scope, &user(), 1, None, "speech term pronunciation")
            .unwrap();
        owner
            .commit(&scope, &assistant(), 1, None, "off topic")
            .unwrap();
        owner
            .commit(&scope, &tool(), 1, Some("db"), "secret tools")
            .unwrap();
        let mut plan = policy();
        plan.selection.stages.push(ContextFilter::Match {
            action: FilterAction::Exclude,
            field: MatchField::TextContains,
            values: vec!["off topic".into()],
        });
        plan.selection.stages.push(ContextFilter::Decision {
            model: "nanojev".into(),
            question: "Relevant to speech?".into(),
            min_confidence_bps: 8_000,
            unavailable: DecisionUnavailable::Omit,
        });
        let mut adapter = CheckingDecision { seen: Vec::new() };
        let snap = reader.capture(&plan, 2, &mut adapter).unwrap().unwrap();
        assert_eq!(adapter.seen, [(a, "speech term pronunciation".into())]);
        assert_eq!(snap.records().len(), 1);
        assert_eq!(snap.decision_audit().len(), 1);
        assert_eq!(snap.decision_audit()[0].record_id(), a);
        assert_eq!(snap.decision_audit()[0].model(), "nanojev");
        assert!(snap.decision_audit()[0].verdict().unwrap().include);
        let empty = reader
            .capture(&plan, 2, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        assert!(empty.records().is_empty());
        if let ContextFilter::Decision { unavailable, .. } = &mut plan.selection.stages[1] {
            *unavailable = DecisionUnavailable::KeepAuthorized;
        }
        assert_eq!(
            reader
                .capture(&plan, 2, &mut NoDecisionModel)
                .unwrap()
                .unwrap()
                .records()
                .len(),
            1
        );
        if let ContextFilter::Decision { unavailable, .. } = &mut plan.selection.stages[1] {
            *unavailable = DecisionUnavailable::Reject;
        }
        assert!(reader.capture(&plan, 2, &mut NoDecisionModel).is_err());
    }
    struct RevokingDecision<'a> {
        owner: &'a ContextOwner,
        scope: SessionScope,
        source: ContextSource,
        calls: AtomicUsize,
    }
    impl ContextDecisionAdapter for RevokingDecision<'_> {
        fn decide(&mut self, query: DecisionQuery<'_>) -> Result<Option<DecisionVerdict>> {
            assert_eq!(query.record.source, share());
            self.calls.fetch_add(1, Ordering::Relaxed);
            // This must not deadlock: no owner mutex is held during adapter execution.
            self.owner.revoke(&self.scope, &self.source)?;
            Ok(Some(DecisionVerdict {
                include: true,
                confidence_bps: 10_000,
            }))
        }
    }
    #[test]
    fn revocation_during_decision_refuses_entire_snapshot_without_deadlocking() {
        let owner = ContextOwner::new().unwrap();
        let reader_scope = scope("team", "talk");
        let source_scope = scope("other-team", "research");
        owner
            .commit(&source_scope, &share(), 1, None, "shared item")
            .unwrap();
        owner.grant(&reader_scope, &share()).unwrap();
        let reader = owner.reader(reader_scope.clone());
        let mut plan = policy();
        plan.selection.sources = vec![share()];
        plan.selection.stages.push(ContextFilter::Decision {
            model: "nanojev".into(),
            question: "Relevant?".into(),
            min_confidence_bps: 5_000,
            unavailable: DecisionUnavailable::Reject,
        });
        let mut adapter = RevokingDecision {
            owner: &owner,
            scope: reader_scope,
            source: share(),
            calls: AtomicUsize::new(0),
        };
        assert!(matches!(
            reader.capture(&plan, 2, &mut adapter),
            Err(Error::State(_))
        ));
        assert_eq!(adapter.calls.load(Ordering::Relaxed), 1);
        assert!(
            reader
                .capture(&plan, 2, &mut NoDecisionModel)
                .unwrap()
                .unwrap()
                .records()
                .is_empty()
        );
    }
    #[test]
    fn source_mutation_and_erase_invalidate_old_snapshots_without_mutating_them() {
        let owner = ContextOwner::new().unwrap();
        let scope = scope("team", "talk");
        let reader = owner.reader(scope.clone());
        let old = owner.commit(&scope, &user(), 1, None, "first").unwrap();
        let a = reader
            .capture(&policy(), 2, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        owner.commit(&scope, &user(), 2, None, "second").unwrap();
        assert!(reader.require_fresh(&a).is_err());
        assert_eq!(a.records()[0].text(), "first"); // a frozen copy, not live.
        assert!(owner.erase(old).unwrap());
        assert!(!owner.erase(old).unwrap());
        let b = reader
            .capture(&policy(), 3, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        assert_eq!(b.records().len(), 1);
        assert_eq!(b.records()[0].text(), "second");
        reader.require_fresh(&b).unwrap();
    }
    #[test]
    fn output_window_is_bounded_and_keeps_latest_complete_chronological_records() {
        let owner = ContextOwner::new().unwrap();
        let scope = scope("team", "talk");
        let reader = owner.reader(scope.clone());
        for i in 1..=30 {
            owner
                .commit(&scope, &user(), i, None, &format!("record {i:02}"))
                .unwrap();
        }
        let mut plan = policy();
        plan.max_bytes = 70;
        let snap = reader
            .capture(&plan, 31, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        assert!(snap.truncated());
        assert!(snap.bytes() <= 70);
        assert!(!snap.records().is_empty());
        assert_eq!(snap.records().last().unwrap().text(), "record 30");
        for pair in snap.records().windows(2) {
            assert!(pair[0].id() < pair[1].id());
        }
        plan.enabled = false;
        assert!(
            reader
                .capture(&plan, 0, &mut NoDecisionModel)
                .unwrap()
                .is_none()
        );
        plan.enabled = true;
        plan.max_bytes = ChatContextPolicy::MAX_BYTES + 1;
        assert!(reader.capture(&plan, 31, &mut NoDecisionModel).is_err());
    }
    #[test]
    fn invalid_record_scope_and_exact_grant_are_enforced_before_mutation() {
        assert!(SessionScope::new("..", "talk").is_err());
        assert!(SessionScope::new("team", "talk/path").is_err());
        let owner = ContextOwner::new().unwrap();
        let a = scope("team", "talk");
        assert!(owner.commit(&a, &user(), 1, None, "").is_err());
        assert!(
            owner
                .commit(&a, &user(), 1, Some("../hidden"), "msg")
                .is_err()
        );
        assert!(
            owner
                .commit(&a, &user(), 1, None, &"a".repeat(MAX_RECORD_BYTES + 1))
                .is_err()
        );
        let mut plan = policy();
        plan.selection.sources = vec![system()];
        owner.commit(&a, &system(), 0, None, "own config").unwrap();
        // Merely requesting even a same-tenant system prompt is not a grant.
        let reader = owner.reader(a.clone());
        assert!(
            reader
                .capture(&plan, 1, &mut NoDecisionModel)
                .unwrap()
                .unwrap()
                .records()
                .is_empty()
        );
        owner.grant(&a, &system()).unwrap();
        assert_eq!(
            reader
                .capture(&plan, 1, &mut NoDecisionModel)
                .unwrap()
                .unwrap()
                .records()
                .len(),
            1
        );
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;
    use crate::context_filter::{ContextFilter, ContextFilterPlan, DecisionUnavailable};

    fn scope() -> SessionScope {
        SessionScope::new("app", "conversation").unwrap()
    }
    fn user() -> ContextSource {
        ContextSource::parse("session/user_message").unwrap()
    }
    fn config() -> ChatContextPolicy {
        ChatContextPolicy {
            enabled: true,
            from_model: "qwen".into(),
            max_bytes: 1024 * 1024,
            selection: ContextFilterPlan::default(),
        }
    }
    struct RejectingAdapter;
    impl ContextDecisionAdapter for RejectingAdapter {
        fn decide(&mut self, _: DecisionQuery<'_>) -> Result<Option<DecisionVerdict>> {
            Err(Error::State(
                "decision adapter failed without any verdict".into(),
            ))
        }
    }
    #[test]
    fn decision_backend_error_is_never_reclassified_as_permissive_unavailable() {
        let owner = ContextOwner::new().unwrap();
        let session = scope();
        owner
            .commit(&session, &user(), 1, None, "a source-valid item")
            .unwrap();
        let reader = owner.reader(session);
        let mut policy = config();
        policy.selection.stages.push(ContextFilter::Decision {
            model: "nanojev".into(),
            question: "include?".into(),
            min_confidence_bps: 0,
            unavailable: DecisionUnavailable::KeepAuthorized,
        });
        assert!(matches!(
            reader.capture(&policy, 2, &mut RejectingAdapter),
            Err(Error::State(_))
        ));
    }
    #[test]
    fn candidate_stage_and_snapshot_byte_budgets_are_both_bounded_without_partial_records() {
        let owner = ContextOwner::new().unwrap();
        let scope = scope();
        let reader = owner.reader(scope.clone());
        let message = "u".repeat(128 * 1024);
        for turn in 1..=40 {
            owner.commit(&scope, &user(), turn, None, &message).unwrap();
        }
        let result = reader
            .capture(&config(), 41, &mut NoDecisionModel)
            .unwrap()
            .unwrap();
        assert!(result.truncated()); // candidate window is smaller than all history.
        assert!(result.bytes() <= ChatContextPolicy::MAX_BYTES);
        assert!(!result.records().is_empty());
        assert!(result.records().len() < 10); // entire last records only, never clipped mid-byte.
        assert_eq!(result.records().last().unwrap().committed_turn(), 40);
        assert!(
            result
                .records()
                .iter()
                .all(|entry| entry.text().len() == message.len())
        );
        reader.require_fresh(&result).unwrap();
        // Explicitly authorizing larger logical context does not relax the
        // process's finite input or output memory policy.
        let mut policy = config();
        policy.max_bytes = ChatContextPolicy::MAX_BYTES + 1;
        assert!(matches!(
            reader.capture(&policy, 41, &mut NoDecisionModel),
            Err(Error::Limit(_))
        ));
    }
    #[test]
    fn unauthorized_records_are_never_seen_by_a_model_adapter_even_when_requested_in_config() {
        let owner = ContextOwner::new().unwrap();
        let current = scope();
        let foreign = SessionScope::new("other-app", "private").unwrap();
        let source = ContextSource::parse("shared:notes/assistant_message").unwrap();
        owner
            .commit(
                &foreign,
                &source,
                1,
                None,
                "private shared note without grant",
            )
            .unwrap();
        owner
            .commit(&current, &user(), 1, None, "ordinary visible message")
            .unwrap();
        let mut policy = config();
        policy.selection.sources.push(source);
        policy.selection.stages.push(ContextFilter::Decision {
            model: "decision".into(),
            question: "relevant".into(),
            min_confidence_bps: 1,
            unavailable: DecisionUnavailable::Reject,
        });
        let reader = owner.reader(current);
        let mut seen = 0usize;
        let mut decide = |q: DecisionQuery<'_>| -> Result<Option<DecisionVerdict>> {
            seen += 1;
            assert_ne!(q.record.text, "private shared note without grant");
            Ok(Some(DecisionVerdict {
                include: true,
                confidence_bps: 9_900,
            }))
        };
        let result = reader.capture(&policy, 2, &mut decide).unwrap().unwrap();
        assert_eq!(seen, 1);
        assert_eq!(result.records().len(), 1);
    }
}
