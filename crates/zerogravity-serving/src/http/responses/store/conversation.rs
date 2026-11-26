//! Conversation resources share the response store's lock and quota.
//! A generation holds an exclusive context mutation lease, not a model session.
use super::{
    BoundedWriter, Credit, Identity, Item, Page, Pagination, Result, State, Store, TTL, item_id,
    valid_id,
};
use crate::{
    TextService,
    http::{
        MAX_BODY,
        body::{self, Reply},
        parse_json,
        responses::input,
    },
};
use hyper::{Method, Response, http::request::Parts};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, sync::Arc, time::Instant};
use zerogravity_engine::router::{
    ChatMessage, ChatRole, MAX_CHAT_MESSAGES, ServiceError, pending::ServingCharge,
};

// The admitted prompt is at most MAX_BODY and the terminal text at most
// MAX_BODY. Price their complete retained backing plus bounded item IDs,
// containers, metadata (including replacement overlap) and lease scratch.
const TEXT_LIMIT: usize = 2 * MAX_BODY;
const CREDIT_BYTES: usize = TEXT_LIMIT + 128 * 1024;
const MAX_BATCH: usize = 20;
// At most 16 metadata pairs of 64/512 Unicode characters, including
// JSON escaping. Small resource objects need no full item-page allowance.
const OBJECT_LIMIT: usize = 64 * 1024;

#[derive(Clone)]
pub(super) struct ItemKey {
    pub(super) id: String,
    pub(super) status: &'static str,
}
impl ItemKey {
    pub(super) fn new(id: String, status: &'static str) -> Self {
        Self { id, status }
    }
}
struct StoredItem {
    key: ItemKey,
    turn: ChatMessage,
}
impl StoredItem {
    fn view(&self) -> Item<'_> {
        Item::with_status(self.key.id.clone(), &self.turn, self.key.status)
    }
}

pub(super) struct Conversation {
    created: u64,
    metadata: BTreeMap<String, String>,
    items: Vec<StoredItem>,
    expires: Instant,
    busy: bool,
    _credit: Credit,
}
#[derive(Serialize)]
struct View<'a> {
    id: &'a str,
    object: &'static str,
    created_at: u64,
    metadata: &'a BTreeMap<String, String>,
}
impl Conversation {
    fn view<'a>(&'a self, id: &'a str) -> View<'a> {
        View {
            id,
            object: "conversation",
            created_at: self.created,
            metadata: &self.metadata,
        }
    }
    fn touch(&mut self) {
        self.expires = Instant::now() + TTL;
    }
    pub(super) fn retained_at(&self, now: Instant) -> bool {
        self.busy || self.expires > now
    }
    fn editable(&self) -> Result<()> {
        if self.busy {
            Err(ServiceError::new(
                409,
                "conversation_in_progress",
                "conversation already has an active response",
            ))
        } else {
            Ok(())
        }
    }
    fn text_bytes(&self) -> usize {
        self.items.iter().map(|item| item.turn.text.len()).sum()
    }
    fn admit_items(&self, count: usize, bytes: usize) -> Result<()> {
        if self.items.len() + count > MAX_CHAT_MESSAGES || self.text_bytes() + bytes > TEXT_LIMIT {
            return Err(ServiceError::new(
                413,
                "conversation_limit",
                "conversation context exceeds capacity",
            ));
        }
        Ok(())
    }
}
fn missing() -> ServiceError {
    ServiceError::new(404, "conversation_not_found", "conversation not found")
}
fn missing_item() -> ServiceError {
    ServiceError::new(
        404,
        "conversation_item_not_found",
        "conversation item not found",
    )
}

impl Store {
    pub(in crate::http) fn lease_conversation(self: &Arc<Self>, id: &str) -> Result<Lease> {
        if !valid_id(id, "conv_") {
            return Err(missing());
        }
        self.expire();
        let mut state = self.state.lock().expect("protocol store");
        let conversation = state.conversations.get_mut(id).ok_or_else(missing)?;
        conversation.editable()?;
        conversation.busy = true;
        Ok(Lease {
            store: self.clone(),
            id: id.to_owned(),
            active: true,
        })
    }
}

/// While leased, neither CRUD nor another generation can replace or mutate
/// the context. Drop on admission failure, disconnect, cancellation or
/// shutdown releases the lease without publishing any new items.
pub(in crate::http) struct Lease {
    store: Arc<Store>,
    id: String,
    active: bool,
}
impl Lease {
    fn with<T>(&self, read: impl FnOnce(&Conversation) -> T) -> T {
        let state = self.store.state.lock().expect("protocol store");
        let conversation = state
            .conversations
            .get(&self.id)
            .expect("leased conversation");
        assert!(conversation.busy);
        read(conversation)
    }
    pub(in crate::http) fn measure(&self) -> (usize, usize) {
        self.with(|conversation| (conversation.items.len(), conversation.text_bytes()))
    }
    pub(in crate::http) fn extend(&self, target: &mut Vec<ChatMessage>) {
        self.with(|conversation| {
            target.extend(conversation.items.iter().map(|item| item.turn.clone()))
        });
    }
    pub(super) fn prepare(
        self,
        input: &[ChatMessage],
        identity: &Identity,
    ) -> Result<(Append, Vec<ItemKey>)> {
        let (new, keys) = self.with(|conversation| {
            let old = conversation.items.len();
            let bytes = input.iter().map(|turn| turn.text.len()).sum::<usize>();
            // The semantic input has already been measured, but validate
            // the precise retained context boundary before model admission.
            if input.len() < old
                || input.len() >= MAX_CHAT_MESSAGES
                || bytes > MAX_BODY
                || !conversation
                    .items
                    .iter()
                    .zip(input)
                    .all(|(item, turn)| item.turn == *turn)
            {
                return Err(ServiceError::new(
                    500,
                    "conversation_state",
                    "conversation context changed before admission",
                ));
            }
            let new = input[old..]
                .iter()
                .enumerate()
                .map(|(i, turn)| StoredItem {
                    key: ItemKey::new(item_id(&identity.response, old + i), "completed"),
                    turn: turn.clone(),
                })
                .collect::<Vec<_>>();
            let keys = conversation
                .items
                .iter()
                .chain(&new)
                .map(|item| item.key.clone())
                .collect();
            Ok((new, keys))
        })?;
        Ok((
            Append {
                lease: self,
                new,
                output_id: identity.message.clone(),
            },
            keys,
        ))
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        if self.active {
            let mut state = self.store.state.lock().expect("protocol store");
            let conversation = state
                .conversations
                .get_mut(&self.id)
                .expect("leased conversation");
            assert!(conversation.busy);
            conversation.busy = false;
        }
    }
}

pub(super) struct Append {
    lease: Lease,
    new: Vec<StoredItem>,
    output_id: String,
}
impl Append {
    /// Called under the same lock as terminal response publication and
    /// cancellation. Bounds are established at admission and serialization.
    pub(super) fn publish(&mut self, state: &mut State, output: &str, status: &'static str) {
        let conversation = state
            .conversations
            .get_mut(&self.lease.id)
            .expect("leased conversation");
        assert!(self.lease.active && conversation.busy);
        assert!(output.len() <= MAX_BODY);
        let bytes = self
            .new
            .iter()
            .map(|item| item.turn.text.len())
            .sum::<usize>()
            + output.len();
        assert!(conversation.admit_items(self.new.len() + 1, bytes).is_ok());
        let output = StoredItem {
            key: ItemKey::new(std::mem::take(&mut self.output_id), status),
            turn: ChatMessage {
                role: ChatRole::Assistant,
                text: output.to_owned(),
            },
        };
        conversation.items.append(&mut self.new);
        conversation.items.push(output);
        conversation.touch();
        conversation.busy = false;
        self.lease.active = false;
    }
}

// Read replies are independent, charged byte copies, never borrowed
// conversation storage past the lock or after deletion/expiry.
fn render<T: Serialize>(
    service: &dyn TextService,
    limit: usize,
    view: &T,
) -> Result<Response<Reply>> {
    let allowance = limit
        .checked_mul(3)
        .and_then(|n| n.checked_add(32 * 1024))
        .ok_or_else(|| {
            ServiceError::new(
                400,
                "size_overflow",
                "conversation projection bound overflow",
            )
        })?;
    let charge =
        Arc::new(service.charge_ingress("http-conversation-projection", allowance as u64)?);
    let mut buffer = BoundedWriter {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut buffer, view).map_err(|_| {
        ServiceError::new(
            503,
            "output_limit",
            "conversation projection exceeds output capacity",
        )
    })?;
    Ok(body::retained_json(buffer.bytes.into(), Some(charge)))
}
fn render_page(
    service: &dyn TextService,
    limit: usize,
    data: &[Item<'_>],
    has_more: bool,
) -> Result<Response<Reply>> {
    render(
        service,
        limit,
        &Page {
            object: "list",
            data,
            first_id: data.first().map(|item| item.id.as_str()),
            last_id: data.last().map(|item| item.id.as_str()),
            has_more,
        },
    )
}

fn batch(value: Option<&Value>, required: bool) -> Result<Option<&Value>> {
    let Some(value) = value else {
        return if required {
            Err(input::invalid("items", "items required"))
        } else {
            Ok(None)
        };
    };
    let items = value
        .as_array()
        .ok_or_else(|| input::invalid("items", "items must be a message array"))?;
    if items.len() > MAX_BATCH || (required && items.is_empty()) {
        return Err(input::invalid("items", "invalid item count"));
    }
    if items
        .iter()
        .any(|item| item.get("id").is_some_and(|id| !id.is_null()))
    {
        return Err(input::unsupported("items"));
    }
    Ok((!items.is_empty()).then_some(value))
}
fn measure_batch(value: Option<&Value>) -> Result<(usize, usize)> {
    let mut count = 0;
    let mut bytes = 0;
    if let Some(value) = value {
        input::turns(value, |role, content| {
            count += 1;
            input::fragments(content, role, |text| {
                bytes = input::add(bytes, text.len())?;
                Ok(())
            })
        })
        .map_err(|e| e.with_parameter("items"))?;
    }
    Ok((count, bytes))
}
fn decode_batch(
    value: Option<&Value>,
    count: usize,
    identity: &Identity,
) -> Result<Vec<StoredItem>> {
    let mut items = Vec::with_capacity(count);
    if let Some(value) = value {
        input::turns(value, |role, content| {
            items.push(StoredItem {
                key: ItemKey::new(item_id(&identity.response, items.len()), "completed"),
                turn: input::message(role, content)?,
            });
            Ok(())
        })
        .map_err(|e| e.with_parameter("items"))?;
    }
    Ok(items)
}
fn object(value: &Value) -> Result<&Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| input::invalid("body", "expected JSON object"))
}
fn not_found() -> ServiceError {
    ServiceError::new(404, "not_found", "unsupported endpoint")
}

pub(in crate::http) fn route(
    store: &Store,
    service: &dyn TextService,
    parts: &Parts,
    raw: &[u8],
    ingress: &mut Vec<ServingCharge>,
    request_id: &str,
    limit: usize,
) -> Result<Response<Reply>> {
    let path = parts.uri.path();
    let query = parts.uri.query();
    if parts.method != Method::POST && !raw.is_empty() {
        return Err(input::invalid(
            "body",
            "resource request body must be empty",
        ));
    }
    let identity = Identity::new(request_id);
    store.expire();
    if path == "/v1/conversations" && parts.method == Method::POST && query.is_none() {
        let value = if raw.is_empty() {
            json!({})
        } else {
            parse_json(raw, Some(service), ingress)?
        };
        let map = object(&value)?;
        input::only(map, &["metadata", "items"], "body")?;
        let metadata = input::metadata(map)?;
        let value = batch(input::optional(map, "items"), false)?;
        let (count, bytes) = measure_batch(value)?;
        let credit = store.reserve(CREDIT_BYTES, service)?;
        let mut conversation = Conversation {
            created: identity.created,
            metadata: input::owned_metadata(metadata),
            items: Vec::with_capacity(MAX_CHAT_MESSAGES),
            expires: Instant::now() + TTL,
            busy: false,
            _credit: credit,
        };
        conversation.admit_items(count, bytes)?;
        conversation
            .items
            .extend(decode_batch(value, count, &identity)?);
        let id = format!(
            "conv_{}",
            identity
                .response
                .strip_prefix("resp_")
                .expect("response ID")
        );
        let reply = render(service, limit.min(OBJECT_LIMIT), &conversation.view(&id))?;
        let old = store
            .state
            .lock()
            .expect("protocol store")
            .conversations
            .insert(id, conversation);
        assert!(old.is_none(), "server conversation ID reused");
        return Ok(reply);
    }
    let path = path
        .strip_prefix("/v1/conversations/")
        .ok_or_else(not_found)?;
    let (id, action) = path.split_once('/').unwrap_or((path, ""));
    if !valid_id(id, "conv_") {
        return Err(missing());
    }
    let mut state = store.state.lock().expect("protocol store");
    let conversation = state.conversations.get_mut(id).ok_or_else(missing)?;
    match (&parts.method, action) {
        (&Method::GET, "") if query.is_none() => {
            render(service, limit.min(OBJECT_LIMIT), &conversation.view(id))
        }
        (&Method::POST, "") if query.is_none() => {
            conversation.editable()?;
            let value = parse_json(raw, Some(service), ingress)?;
            let map = object(&value)?;
            input::only(map, &["metadata"], "metadata")?;
            if !map.contains_key("metadata") {
                return Err(input::invalid("metadata", "metadata required"));
            }
            let metadata = input::owned_metadata(input::metadata(map)?);
            let reply = render(
                service,
                limit.min(OBJECT_LIMIT),
                &View {
                    metadata: &metadata,
                    ..conversation.view(id)
                },
            )?;
            conversation.metadata = metadata;
            conversation.touch();
            Ok(reply)
        }
        (&Method::DELETE, "") if query.is_none() => {
            conversation.editable()?;
            // Fixed-size deletion receipt is covered by connection headroom,
            // just like response deletion; do not need a page charge to free.
            state.conversations.remove(id);
            Ok(body::json_response(
                200,
                &json!({"id":id,"object":"conversation.deleted","deleted":true}),
                None,
            ))
        }
        (&Method::GET, "items") => {
            let page = Pagination::parse(query)?;
            let n = conversation.items.len();
            let order = |i| if page.ascending { i } else { n - 1 - i };
            let start = match page.after {
                None => 0,
                Some(after) => (0..n)
                    .find(|&i| conversation.items[order(i)].key.id == after)
                    .map(|i| i + 1)
                    .ok_or_else(|| input::invalid("after", "unknown pagination cursor"))?,
            };
            let count = (n - start).min(page.limit);
            let data = (start..start + count)
                .map(|i| conversation.items[order(i)].view())
                .collect::<Vec<_>>();
            render_page(service, limit, &data, start + count < n)
        }
        (&Method::POST, "items") if query.is_none() => {
            conversation.editable()?;
            let value = parse_json(raw, Some(service), ingress)?;
            let map = object(&value)?;
            input::only(map, &["items"], "items")?;
            let value = batch(input::optional(map, "items"), true)?;
            let (count, bytes) = measure_batch(value)?;
            conversation.admit_items(count, bytes)?;
            let items = decode_batch(value, count, &identity)?;
            let data = items.iter().map(StoredItem::view).collect::<Vec<_>>();
            let reply = render_page(service, limit, &data, false)?;
            conversation.items.extend(items);
            conversation.touch();
            Ok(reply)
        }
        (method, action) if query.is_none() && action.starts_with("items/") => {
            let key = action.strip_prefix("items/").expect("items path");
            if !valid_id(key, "msg_") {
                return Err(missing_item());
            }
            let index = conversation
                .items
                .iter()
                .position(|item| item.key.id == key)
                .ok_or_else(missing_item)?;
            match *method {
                Method::GET => render(service, limit, &conversation.items[index].view()),
                Method::DELETE => {
                    conversation.editable()?;
                    let reply = render(service, limit.min(OBJECT_LIMIT), &conversation.view(id))?;
                    conversation.items.remove(index);
                    conversation.touch();
                    Ok(reply)
                }
                _ => Err(not_found()),
            }
        }
        _ => Err(not_found()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{self, HostBytes};

    #[test]
    fn conversation_and_response_share_quota_and_lease_defers_expiry() {
        let _lock = test_support::lock();
        let host = test_support::host();
        let baseline = host.serving_bytes();
        let service = test_support::idle(&host);
        let store = Arc::new(Store::default());
        let now = Instant::now();
        let credit = store.reserve(CREDIT_BYTES, service.as_ref()).unwrap();
        store.state.lock().unwrap().conversations.insert(
            "conv_fixture".into(),
            Conversation {
                created: 0,
                metadata: BTreeMap::new(),
                items: Vec::with_capacity(MAX_CHAT_MESSAGES),
                expires: now + TTL,
                busy: false,
                _credit: credit,
            },
        );
        let other = store
            .reserve(super::super::MAX_BYTES - CREDIT_BYTES, service.as_ref())
            .unwrap();
        assert!(store.reserve(1, service.as_ref()).is_err());
        drop(other);
        let lease = store.lease_conversation("conv_fixture").unwrap();
        store.expire_at(now + TTL);
        assert!(store.lease_conversation("conv_fixture").is_err());
        assert_eq!(lease.measure(), (0, 0));
        assert!(
            store
                .reserve(super::super::MAX_BYTES, service.as_ref())
                .is_err()
        );
        drop(lease);
        store.expire_at(now + TTL);
        assert!(store.lease_conversation("conv_fixture").is_err());
        let credit = store
            .reserve(super::super::MAX_BYTES, service.as_ref())
            .unwrap();
        drop(credit);
        assert_eq!(host.serving_bytes(), baseline);
    }
}
