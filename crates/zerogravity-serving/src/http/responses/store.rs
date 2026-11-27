//! Listener-local protocol records, never model state. Reservations cover
//! pending writes and retained readers, including deleted/expired records.
use super::{BoundedWriter, Identity, Options};
use crate::TextService;
use bytes::Bytes;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, Weak},
    time::{Duration, Instant},
};
use zerogravity_engine::router::{
    ChatMessage, ChatRole, RequestEvents, SemanticRequest, ServiceError, TextInput,
    pending::ServingCharge,
};

mod conversation;
mod replay;
use conversation::{Append, Conversation, ItemKey};
pub(in crate::http) use conversation::{Lease as ConversationLease, route as conversation_route};
pub(super) use replay::{Batch as ReplayBatch, Replay};

type Result<T> = std::result::Result<T, ServiceError>;

pub(super) const MAX_RECORDS: usize = 64;
const MAX_BYTES: usize = 32 * 1024 * 1024;
const TTL: Duration = Duration::from_secs(30 * 60);
const TEMPORARY_TTL: Duration = Duration::from_secs(10 * 60);

#[derive(Default)]
struct Budget {
    count: usize,
    bytes: usize,
}

#[derive(Default)]
pub(in crate::http) struct Store {
    state: Mutex<State>,
    budget: Arc<Mutex<Budget>>,
}
#[derive(Default)]
struct State {
    responses: BTreeMap<String, Entry>,
    conversations: BTreeMap<String, Conversation>,
}
impl Store {
    pub(in crate::http) fn expire(&self) {
        self.expire_at(Instant::now());
    }
    fn expire_at(&self, now: Instant) {
        let mut state = self.state.lock().expect("protocol store");
        state.responses.retain(|_, entry| match entry {
            Entry::Active(_) => true,
            Entry::Terminal { expires, .. } => *expires > now,
        });
        state
            .conversations
            .retain(|_, conversation| conversation.retained_at(now));
    }

    fn reserve(&self, bytes: usize, service: &dyn TextService) -> Result<Credit> {
        self.expire();
        {
            let mut budget = self.budget.lock().expect("protocol budget");
            if budget.count >= MAX_RECORDS
                || budget
                    .bytes
                    .checked_add(bytes)
                    .is_none_or(|n| n > MAX_BYTES)
            {
                return Err(ServiceError::new(
                    503,
                    "response_store_full",
                    "protocol store capacity exhausted",
                ));
            }
            budget.count += 1;
            budget.bytes += bytes;
        }
        // The quota is rolled back even when physical host admission refuses.
        let mut credit = Credit {
            bytes,
            budget: self.budget.clone(),
            charge: None,
        };
        credit.charge = Some(service.charge_ingress("http-response-store", bytes as u64)?);
        Ok(credit)
    }

    pub(super) fn get(&self, id: &str) -> Result<Arc<Record>> {
        self.expire();
        self.state
            .lock()
            .expect("protocol store")
            .responses
            .get(id)
            .map(Entry::record)
            .ok_or_else(missing)
    }

    fn remove(&self, id: &str) -> Result<()> {
        self.expire();
        let mut state = self.state.lock().expect("protocol store");
        let records = &mut state.responses;
        if matches!(records.get(id), Some(Entry::Active(_))) {
            return Err(ServiceError::new(
                409,
                "response_in_progress",
                "cancel the active response before deleting it",
            ));
        }
        records.remove(id).ok_or_else(missing)?;
        Ok(())
    }

    fn cancel(&self, id: &str) -> Result<Arc<Record>> {
        self.expire();
        let mut state = self.state.lock().expect("protocol store");
        let records = &mut state.responses;
        match records.get(id).ok_or_else(missing)? {
            Entry::Terminal { record, .. } if record.background => Ok(record.clone()),
            Entry::Terminal { .. } => Err(ServiceError::new(
                400,
                "invalid_request",
                "only background responses can be cancelled",
            )
            .with_parameter("background")),
            Entry::Active(active) => {
                let active = active.clone();
                let record = active.cancelled.clone();
                // Serialize cancellation vs completion by the same store lock.
                // Publish only after signaling the original receipt.
                if let Some(owner) = active.owner.upgrade() {
                    owner.cancel();
                }
                active.wake.notify_one();
                records.insert(id.to_owned(), Entry::terminal(record.clone()));
                // The public SSE grammar has no cancelled terminal event.
                // Preserve already-published deltas, then close at EOF.
                if let Some(replay) = &record.replay {
                    replay.finish(None);
                }
                Ok(record)
            }
        }
    }
}

enum Entry {
    Active(Arc<Active>),
    Terminal {
        record: Arc<Record>,
        expires: Instant,
    },
}
impl Entry {
    fn terminal(record: Arc<Record>) -> Self {
        let ttl = if record.stored { TTL } else { TEMPORARY_TTL };
        Self::Terminal {
            record,
            expires: Instant::now() + ttl,
        }
    }
    fn record(&self) -> Arc<Record> {
        match self {
            Self::Active(active) => active.initial.clone(),
            Self::Terminal { record, .. } => record.clone(),
        }
    }
}

// Immutable initial and cancellation projections share the same input and
// reservation as the eventual terminal record. No per-poll snapshots.
pub(super) struct Active {
    initial: Arc<Record>,
    cancelled: Arc<Record>,
    owner: Weak<RequestEvents>,
    pub(super) wake: tokio::sync::Notify,
}

fn missing() -> ServiceError {
    ServiceError::new(404, "response_not_found", "response not found")
}
struct Credit {
    bytes: usize,
    budget: Arc<Mutex<Budget>>,
    charge: Option<ServingCharge>,
}
impl Drop for Credit {
    fn drop(&mut self) {
        drop(self.charge.take());
        let mut budget = self.budget.lock().expect("protocol budget");
        budget.count -= 1;
        budget.bytes -= self.bytes;
    }
}

pub(super) struct Record {
    json: Vec<u8>,
    // Normalized text used by this request, excluding request-local
    // instructions. A continuation copies these turns and the output,
    // never references a previous physical session or a chain of records.
    pub(super) input: Arc<Vec<ChatMessage>>,
    input_keys: Option<Arc<Vec<ItemKey>>>,
    pub(super) output: ChatMessage,
    pub(super) continuable: bool,
    background: bool,
    stored: bool,
    replay: Option<Arc<Replay>>,
    _credit: Arc<Credit>,
}
struct Retained(Arc<Record>);
impl AsRef<[u8]> for Retained {
    fn as_ref(&self) -> &[u8] {
        &self.0.json
    }
}
impl Record {
    pub(super) fn bytes(self: Arc<Self>) -> Bytes {
        Bytes::from_owner(Retained(self))
    }
}

// Retained response reservation and optional context mutation, both prepared
// before original model admission. Stateless conversation turns need no
// independent response slot.
pub(in crate::http) struct Pending {
    store: Arc<Store>,
    id: String,
    input: Arc<Vec<ChatMessage>>,
    limit: usize,
    background: bool,
    stored: bool,
    credit: Option<Arc<Credit>>,
    input_keys: Option<Arc<Vec<ItemKey>>>,
    append: Option<Append>,
    pub(super) replay: Option<Arc<Replay>>,
}
impl Pending {
    pub(in crate::http) fn prepare(
        store: &Arc<Store>,
        service: &dyn TextService,
        request: &SemanticRequest,
        options: &Options,
        conversation: Option<ConversationLease>,
        request_id: &str,
        output_limit: usize,
    ) -> Result<Option<Self>> {
        let retain = options.store || options.background;
        if !retain && conversation.is_none() {
            return Ok(None);
        }
        let TextInput::Conversation(messages) = &request.input else {
            return Err(ServiceError::new(
                500,
                "response_state",
                "Responses requires conversation intent",
            ));
        };
        let input = &messages[usize::from(options.instructions.is_some())..];
        let limit = output_limit.min(super::super::MAX_BODY);
        // Retained input backing + independent terminal wire snapshot +
        // assistant text + bounded serialization growth/metadata headroom.
        let bytes = input
            .iter()
            .try_fold(
                input
                    .len()
                    .saturating_mul(std::mem::size_of::<ChatMessage>()),
                |n, turn| n.checked_add(turn.text.len()),
            )
            .and_then(|n| {
                n.checked_add(limit.checked_mul(if options.background { 7 } else { 3 })?)
            })
            .and_then(|n| {
                n.checked_add(if options.background {
                    64 * 1024
                } else {
                    16 * 1024
                })
            })
            .ok_or_else(|| {
                ServiceError::new(400, "size_overflow", "response store bound overflow")
            })?;
        let streaming = options.background && request.stream;
        let bytes = if streaming {
            Replay::reservation(limit)
                .and_then(|extra| bytes.checked_add(extra))
                .ok_or_else(|| ServiceError::new(400, "size_overflow", "replay bound overflow"))?
        } else {
            bytes
        };
        let identity = Identity::new(request_id);
        let credit = if retain {
            // Conversation item IDs/statuses are retained along with the
            // normalized response snapshot, independently of later edits.
            let bytes = bytes
                .checked_add(if conversation.is_some() { 32 * 1024 } else { 0 })
                .ok_or_else(|| {
                    ServiceError::new(400, "size_overflow", "response store bound overflow")
                })?;
            Some(Arc::new(store.reserve(bytes, service)?))
        } else {
            None
        };
        let replay = streaming.then(|| {
            Arc::new(Replay::new(
                limit,
                credit.as_ref().expect("background reservation").clone(),
            ))
        });
        let (append, input_keys) = if let Some(conversation) = conversation {
            let (append, keys) = conversation.prepare(input, &identity)?;
            (Some(append), retain.then(|| Arc::new(keys)))
        } else {
            (None, None)
        };
        Ok(Some(Self {
            store: store.clone(),
            id: identity.response,
            input: Arc::new(if retain { input.to_vec() } else { Vec::new() }),
            limit,
            background: options.background,
            stored: options.store,
            credit,
            input_keys,
            append,
            replay,
        }))
    }

    pub(super) fn record<T: Serialize>(
        &self,
        view: &T,
        output: &str,
        continuable: bool,
    ) -> Result<Arc<Record>> {
        let mut buffer = BoundedWriter {
            bytes: Vec::new(),
            limit: self.limit,
        };
        serde_json::to_writer(&mut buffer, view)
            .map_err(|_| ServiceError::new(503, "output_limit", "stored response exceeds bound"))?;
        if output.len() > self.limit {
            return Err(ServiceError::new(
                503,
                "output_limit",
                "stored output exceeds bound",
            ));
        }
        Ok(Arc::new(Record {
            json: buffer.bytes,
            input: self.input.clone(),
            input_keys: self.input_keys.clone(),
            output: ChatMessage {
                role: ChatRole::Assistant,
                text: output.to_owned(),
            },
            continuable: self.stored && continuable,
            background: self.background,
            stored: self.stored,
            replay: self.replay.clone(),
            _credit: self
                .credit
                .as_ref()
                .expect("retained response reservation")
                .clone(),
        }))
    }

    pub(super) fn finish<T: Serialize>(
        mut self,
        view: &T,
        output: &str,
        status: Option<&'static str>,
    ) -> Result<()> {
        let record = self
            .credit
            .is_some()
            .then(|| self.record(view, output, status.is_some()))
            .transpose()?;
        let mut state = self.store.state.lock().expect("protocol store");
        if let (Some(append), Some(status)) = (&mut self.append, status) {
            append.publish(&mut state, output, status);
        }
        if let Some(record) = record {
            let previous = state.responses.insert(self.id, Entry::terminal(record));
            assert!(previous.is_none(), "server response ID reused");
        }
        Ok(())
    }

    pub(super) fn publish_active(
        &self,
        owner: &Arc<RequestEvents>,
        initial: Arc<Record>,
        cancelled: Arc<Record>,
    ) -> Arc<Active> {
        let active = Arc::new(Active {
            initial,
            cancelled,
            owner: Arc::downgrade(owner),
            wake: tokio::sync::Notify::new(),
        });
        let previous = self
            .store
            .state
            .lock()
            .expect("protocol store")
            .responses
            .insert(self.id.clone(), Entry::Active(active.clone()));
        assert!(previous.is_none(), "server response ID reused");
        active
    }

    pub(super) fn finish_active(
        mut self,
        active: &Arc<Active>,
        record: Arc<Record>,
        status: Option<&'static str>,
        terminal: Option<ReplayBatch>,
    ) {
        let mut state = self.store.state.lock().expect("protocol store");
        // Cancellation may have already won. Never recreate deleted records
        // or overwrite the cancelled snapshot with a later worker result.
        if matches!(state.responses.get(&self.id), Some(Entry::Active(current)) if Arc::ptr_eq(current, active))
        {
            if let (Some(append), Some(status)) = (&mut self.append, status) {
                append.publish(&mut state, &record.output.text, status);
            }
            state.responses.insert(self.id, Entry::terminal(record));
            if let Some(replay) = &self.replay {
                replay.finish(terminal);
            }
        }
    }
}

#[derive(Serialize)]
struct Part<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    annotations: Option<[(); 0]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<[(); 0]>,
}
#[derive(Serialize)]
struct Item<'a> {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    status: &'static str,
    content: [Part<'a>; 1],
}
fn item_id(response_id: &str, index: usize) -> String {
    format!(
        "msg_{}_i{index}",
        response_id.strip_prefix("resp_").unwrap_or(response_id)
    )
}
impl<'a> Item<'a> {
    fn with_status(id: String, turn: &'a ChatMessage, status: &'static str) -> Self {
        let assistant = turn.role == ChatRole::Assistant;
        Self {
            id,
            kind: "message",
            role: turn.role.as_str(),
            status,
            content: [Part {
                kind: if assistant {
                    "output_text"
                } else {
                    "input_text"
                },
                text: &turn.text,
                annotations: assistant.then_some([]),
                logprobs: assistant.then_some([]),
            }],
        }
    }
}
#[derive(Serialize)]
struct Page<'a> {
    object: &'static str,
    data: &'a [Item<'a>],
    first_id: Option<&'a str>,
    last_id: Option<&'a str>,
    has_more: bool,
}

struct Pagination<'a> {
    after: Option<&'a str>,
    limit: usize,
    ascending: bool,
}
impl<'a> Pagination<'a> {
    fn parse(query: Option<&'a str>) -> Result<Self> {
        let invalid = || ServiceError::new(400, "invalid_request", "invalid item pagination");
        let mut page = Self {
            after: None,
            limit: 20,
            ascending: false,
        };
        let mut seen = 0u8;
        for pair in query
            .into_iter()
            .flat_map(|q| q.split('&'))
            .filter(|s| !s.is_empty())
        {
            let (key, value) = pair.split_once('=').ok_or_else(invalid)?;
            let bit = match key {
                "after" if valid_id(value, "msg_") => {
                    page.after = Some(value);
                    1
                }
                "limit" => {
                    page.limit = value
                        .parse()
                        .ok()
                        .filter(|n| (1..=100).contains(n))
                        .ok_or_else(invalid)?;
                    2
                }
                "order" if matches!(value, "asc" | "desc") => {
                    page.ascending = value == "asc";
                    4
                }
                _ => return Err(invalid()),
            };
            if seen & bit != 0 {
                return Err(invalid());
            }
            seen |= bit;
        }
        Ok(page)
    }
}
fn valid_id(id: &str, prefix: &str) -> bool {
    id.starts_with(prefix)
        && id.len() <= 128
        && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

struct Retrieval {
    stream: bool,
    after: Option<usize>,
}
impl Retrieval {
    fn parse(query: Option<&str>) -> Result<Self> {
        let mut result = Self {
            stream: false,
            after: None,
        };
        let mut seen = 0u8;
        for pair in query
            .into_iter()
            .flat_map(|q| q.split('&'))
            .filter(|s| !s.is_empty())
        {
            let (key, value) = pair.split_once('=').ok_or_else(|| {
                ServiceError::new(400, "invalid_request", "invalid response retrieval query")
            })?;
            let bit = match key {
                "stream" if matches!(value, "true" | "false") => {
                    result.stream = value == "true";
                    1
                }
                "starting_after" => {
                    result.after = Some(
                        value
                            .parse::<usize>()
                            .ok()
                            .filter(|_| {
                                !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit())
                            })
                            .ok_or_else(|| {
                                ServiceError::new(400, "invalid_request", "invalid replay cursor")
                                    .with_parameter("starting_after")
                            })?,
                    );
                    2
                }
                "include_obfuscation" if value == "false" => 4,
                _ => {
                    return Err(ServiceError::new(
                        400,
                        "unsupported_parameter",
                        "unsupported response retrieval option",
                    ));
                }
            };
            if seen & bit != 0 {
                return Err(ServiceError::new(
                    400,
                    "invalid_request",
                    "duplicate response retrieval option",
                ));
            }
            seen |= bit;
        }
        if !result.stream && seen & 6 != 0 {
            return Err(ServiceError::new(
                400,
                "invalid_request",
                "replay options require stream=true",
            )
            .with_parameter("stream"));
        }
        Ok(result)
    }
}

use super::super::body::{self, Reply};
use hyper::{Method, Response};
use serde_json::json;

pub(in crate::http) fn route(
    store: &Store,
    service: &dyn TextService,
    method: &Method,
    uri: &hyper::Uri,
    raw: &[u8],
    limit: usize,
) -> Option<Result<Response<Reply>>> {
    let path = uri.path().strip_prefix("/v1/responses/")?;
    // Reserved create/count subroutes are not resource IDs.
    if path == "input_tokens" {
        return None;
    }
    Some((|| {
        let (id, action) = path.split_once('/').unwrap_or((path, ""));
        if !valid_id(id, "resp_") {
            return Err(missing());
        }
        if !raw.is_empty() {
            return Err(ServiceError::new(
                400,
                "invalid_request",
                "resource request body must be empty",
            ));
        }
        match (method, action) {
            (&Method::GET, "") => {
                let retrieval = Retrieval::parse(uri.query())?;
                let record = store.get(id)?;
                if retrieval.stream {
                    return record
                        .replay
                        .as_ref()
                        .ok_or_else(|| {
                            ServiceError::new(
                                400,
                                "unsupported_parameter",
                                "response was not created as a background stream",
                            )
                            .with_parameter("stream")
                        })?
                        .reply(retrieval.after);
                }
                Ok(body::retained_json(record.bytes(), None))
            }
            (&Method::DELETE, "") if uri.query().is_none() => {
                store.remove(id)?;
                Ok(body::json_response(
                    200,
                    &json!({"id":id,"object":"response.deleted","deleted":true}),
                    None,
                ))
            }
            (&Method::POST, "cancel") if uri.query().is_none() => {
                Ok(body::retained_json(store.cancel(id)?.bytes(), None))
            }
            (&Method::GET, "input_items") => {
                let page = Pagination::parse(uri.query())?;
                let record = store.get(id)?;
                let n = record.input.len();
                let order = |position: usize| {
                    if page.ascending {
                        position
                    } else {
                        n - 1 - position
                    }
                };
                let key = |index: usize| {
                    record
                        .input_keys
                        .as_ref()
                        .map(|keys| keys[index].clone())
                        .unwrap_or_else(|| ItemKey::new(item_id(id, index), "completed"))
                };
                let start = match page.after {
                    None => 0,
                    Some(after) => (0..n)
                        .find(|&i| {
                            let index = order(i);
                            match &record.input_keys {
                                Some(keys) => keys[index].id == after,
                                None => item_id(id, index) == after,
                            }
                        })
                        .map(|i| i + 1)
                        .ok_or_else(|| {
                            ServiceError::new(400, "invalid_request", "unknown pagination cursor")
                                .with_parameter("after")
                        })?,
                };
                // Bounded page serialization; record stays charged while
                // its text is borrowed, output gets its own short lease.
                let count = (n - start).min(page.limit);
                let allowance = limit
                    .checked_mul(3)
                    .and_then(|n| n.checked_add(32 * 1024))
                    .ok_or_else(|| {
                        ServiceError::new(400, "size_overflow", "page bound overflow")
                    })?;
                let charge =
                    Arc::new(service.charge_ingress("http-response-items", allowance as u64)?);
                let data = (start..start + count)
                    .map(|i| {
                        let index = order(i);
                        let key = key(index);
                        Item::with_status(key.id, &record.input[index], key.status)
                    })
                    .collect::<Vec<_>>();
                let view = Page {
                    object: "list",
                    first_id: data.first().map(|x| x.id.as_str()),
                    last_id: data.last().map(|x| x.id.as_str()),
                    data: &data,
                    has_more: start + count < n,
                };
                let mut buffer = BoundedWriter {
                    bytes: Vec::new(),
                    limit,
                };
                serde_json::to_writer(&mut buffer, &view).map_err(|_| {
                    ServiceError::new(503, "output_limit", "input items exceed output capacity")
                })?;
                Ok(body::retained_json(buffer.bytes.into(), Some(charge)))
            }
            _ => Err(ServiceError::new(404, "not_found", "unsupported endpoint")),
        }
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{self, HostBytes};

    #[test]
    fn quota_covers_pending_and_expired_but_retained_wire_readers() {
        let _lock = test_support::lock();
        let host = test_support::host();
        let baseline = host.serving_bytes();
        let service = test_support::idle(&host);
        let store = Store::default();

        // A pending reservation counts against the bound before a model
        // request can be published, and a failed reservation changes nothing.
        let credits = (0..MAX_RECORDS)
            .map(|_| store.reserve(1024, service.as_ref()).unwrap())
            .collect::<Vec<_>>();
        assert!(store.reserve(1, service.as_ref()).is_err());
        drop(credits);
        assert_eq!(host.serving_bytes(), baseline);
        let credit = store.reserve(MAX_BYTES, service.as_ref()).unwrap();
        assert!(store.reserve(1, service.as_ref()).is_err());
        let now = Instant::now();
        store.state.lock().unwrap().responses.insert(
            "resp_fixture".into(),
            Entry::Terminal {
                expires: now + TTL,
                record: Arc::new(Record {
                    json: b"{}".to_vec(),
                    input: Arc::new(Vec::new()),
                    input_keys: None,
                    output: ChatMessage {
                        role: ChatRole::Assistant,
                        text: String::new(),
                    },
                    continuable: true,
                    background: false,
                    stored: true,
                    replay: None,
                    _credit: Arc::new(credit),
                }),
            },
        );
        let read = store.get("resp_fixture").unwrap().bytes();
        store.expire_at(now + TTL);
        assert!(store.get("resp_fixture").is_err());
        assert_eq!(&read[..], b"{}");
        assert!(
            store.reserve(1, service.as_ref()).is_err(),
            "retained reader still owns the slot/bytes"
        );
        drop(read);
        assert_eq!(host.serving_bytes(), baseline);
        let credit = store.reserve(1, service.as_ref()).unwrap();
        drop(credit);
        assert_eq!(host.serving_bytes(), baseline);
    }
}
