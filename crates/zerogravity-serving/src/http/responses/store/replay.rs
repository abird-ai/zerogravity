//! One bounded append-only event log for an originally streamed background
//! response. Readers retain bytes/credit, never the inference receipt.
use super::Credit;
use crate::http::{
    body::{self, Data, Reply},
    responses::{BoundedWriter, Result, StreamEvent, wire_error},
};
use bytes::Bytes;
use hyper::{Response, body::Frame, header};
use std::sync::{Arc, Mutex};
use tokio::{sync::Notify, time};
use zerogravity_engine::router::{MAX_OUTPUT_TOKENS, ServiceError};

// Four opening events, at most one delta per original output token,
// four terminal events. Refuse an embedding adapter that exceeds the bound.
const MAX_EVENTS: usize = MAX_OUTPUT_TOKENS + 8;

pub(in crate::http::responses) struct Replay {
    log: Mutex<Log>,
    changed: Notify,
    credit: Arc<Credit>,
}
struct Log {
    frames: Vec<Bytes>,
    remaining: usize,
    // Always leave room for a small canonical failure event.
    fallback: usize,
    failure_buffer: Option<Vec<u8>>,
    closed: bool,
}
struct RetainedEvent {
    bytes: Vec<u8>,
    _credit: Arc<Credit>,
}
impl AsRef<[u8]> for RetainedEvent {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

/// Prepared before terminal response/conversation publication. Either the
/// complete successful terminal group fits, or none of it becomes visible.
pub(in crate::http::responses) struct Batch {
    start: usize,
    bytes: usize,
    frames: Vec<Vec<u8>>,
}
impl Log {
    fn prepare<'a>(
        &self,
        events: impl IntoIterator<Item = StreamEvent<'a>>,
        terminal: bool,
    ) -> Result<Batch> {
        let start = self.frames.len();
        let mut remaining = self.remaining;
        let mut max = MAX_EVENTS;
        if !terminal {
            remaining = remaining.checked_sub(self.fallback).ok_or_else(full)?;
            max -= 1;
        }
        let available = remaining;
        let mut frames = Vec::new();
        if self.closed {
            return Err(full());
        }
        for event in events {
            let sequence = start + frames.len();
            if sequence >= max {
                return Err(full());
            }
            let bytes = event.encode(sequence, remaining).map_err(wire_error)?;
            remaining -= bytes.len();
            frames.push(bytes);
        }
        Ok(Batch {
            start,
            bytes: available - remaining,
            frames,
        })
    }

    fn publish(&mut self, batch: Batch, credit: &Arc<Credit>) {
        assert!(!self.closed && self.frames.len() == batch.start);
        self.remaining -= batch.bytes;
        self.frames.extend(batch.frames.into_iter().map(|bytes| {
            Bytes::from_owner(RetainedEvent {
                bytes,
                _credit: credit.clone(),
            })
        }));
    }
}
fn full() -> ServiceError {
    ServiceError::new(
        503,
        "output_limit",
        "background event replay capacity exhausted",
    )
}

impl Replay {
    pub(super) fn reservation(limit: usize) -> Option<usize> {
        // Geometric event backing/encoding overlap (3x), retained fallback
        // buffer overlap (1x), partial text accumulator (2x), frame metadata.
        limit.checked_mul(6)?.checked_add(128 * 1024)
    }

    pub(super) fn new(limit: usize, credit: Arc<Credit>) -> Self {
        Self {
            log: Mutex::new(Log {
                frames: Vec::with_capacity(MAX_EVENTS),
                remaining: limit,
                fallback: 0,
                failure_buffer: None,
                closed: false,
            }),
            changed: Notify::new(),
            credit,
        }
    }

    pub(in crate::http::responses) fn initialize(
        &self,
        initial: [StreamEvent<'_>; 4],
        fallback: StreamEvent<'_>,
    ) -> Result<()> {
        let mut log = self.log.lock().expect("background replay");
        // Size at the largest possible sequence. The deterministic failed
        // response can be encoded at any actual sequence within this bound.
        let bytes = fallback
            .encode(MAX_EVENTS - 1, log.remaining)
            .map_err(wire_error)?;
        log.fallback = bytes.len();
        log.failure_buffer = Some(bytes);
        let batch = log.prepare(initial, false)?;
        log.publish(batch, &self.credit);
        Ok(())
    }

    pub(in crate::http::responses) fn delta(&self, event: StreamEvent<'_>) -> Result<()> {
        let mut log = self.log.lock().expect("background replay");
        if log.closed || log.frames.len() >= MAX_EVENTS - 1 {
            return Err(full());
        }
        let available = log.remaining.checked_sub(log.fallback).ok_or_else(full)?;
        let bytes = event
            .encode(log.frames.len(), available)
            .map_err(wire_error)?;
        log.remaining -= bytes.len();
        log.frames.push(Bytes::from_owner(RetainedEvent {
            bytes,
            _credit: self.credit.clone(),
        }));
        drop(log);
        self.changed.notify_waiters();
        Ok(())
    }

    pub(in crate::http::responses) fn prepare_terminal<'a>(
        &self,
        events: impl IntoIterator<Item = StreamEvent<'a>>,
    ) -> Result<Batch> {
        self.log
            .lock()
            .expect("background replay")
            .prepare(events, true)
    }

    /// Reuse the already allocated failure buffer. The identical view at
    /// a shorter/equal sequence number cannot need more bytes or grow it.
    pub(in crate::http::responses) fn fallback(&self, event: StreamEvent<'_>) -> Batch {
        let mut log = self.log.lock().expect("background replay");
        let mut bytes = log
            .failure_buffer
            .take()
            .expect("preflighted failure projection");
        bytes.clear();
        let start = log.frames.len();
        let mut writer = BoundedWriter {
            bytes,
            limit: log.fallback,
        };
        event
            .encode_into(start, &mut writer)
            .expect("preflighted failure event");
        Batch {
            start,
            bytes: writer.bytes.len(),
            frames: vec![writer.bytes],
        }
    }

    // Caller holds the original store's state lock and has won terminal
    // publication. No late event after cancellation or terminal EOF.
    pub(super) fn finish(&self, terminal: Option<Batch>) {
        let mut log = self.log.lock().expect("background replay");
        assert!(!log.closed, "background replay already terminal");
        if let Some(batch) = terminal {
            log.publish(batch, &self.credit);
        }
        log.closed = true;
        log.failure_buffer = None;
        drop(log);
        self.changed.notify_waiters();
    }

    pub(in crate::http::responses) fn reply(
        self: &Arc<Self>,
        after: Option<usize>,
    ) -> Result<Response<Reply>> {
        let mut index = match after {
            None => 0,
            Some(cursor) => {
                if cursor >= self.log.lock().expect("background replay").frames.len() {
                    return Err(
                        ServiceError::new(400, "invalid_request", "unknown replay cursor")
                            .with_parameter("starting_after"),
                    );
                }
                cursor + 1
            }
        };
        let replay = self.clone();
        let stream = async_stream::stream! {
            loop {
                // Register before observing the log. notify_waiters also
                // wakes a Notified created but not yet polled.
                let changed = replay.changed.notified();
                let (next, closed) = {
                    let log = replay.log.lock().expect("background replay");
                    (log.frames.get(index).cloned(), log.closed)
                };
                if let Some(bytes) = next {
                    index += 1;
                    yield Ok(Frame::data(Data::retained(bytes)));
                } else if closed {
                    break;
                } else if time::timeout(body::KEEPALIVE, changed).await.is_err() {
                    // Transport heartbeat, not a retained/cursor event.
                    yield Ok(Frame::data(Data::retained(Bytes::from_static(b": keep-alive\n\n"))));
                }
            }
        };
        Ok(Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache, no-transform")
            .header("x-accel-buffering", "no")
            .body(Reply::Stream(Box::pin(stream)))
            .expect("fixed background SSE headers"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        http::responses::{
            Identity, delta_event,
            store::{Entry, MAX_BYTES, Record, Store, TTL},
        },
        test_support::{self, HostBytes},
    };
    use std::time::Instant;
    use zerogravity_engine::router::{ChatMessage, ChatRole};

    #[test]
    fn replay_readers_and_wire_frames_hold_original_quota_after_expiry() {
        let _lock = test_support::lock();
        let host = test_support::host();
        let baseline = host.serving_bytes();
        let service = test_support::idle(&host);
        let store = Store::default();
        let credit = Arc::new(store.reserve(MAX_BYTES, service.as_ref()).unwrap());
        let replay = Arc::new(Replay::new(4096, credit.clone()));
        replay
            .delta(delta_event(&Identity::new("req_fixture"), "retained"))
            .unwrap();
        replay.finish(None);
        let frame = replay.log.lock().unwrap().frames[0].clone();
        let reader = replay.reply(None).unwrap();
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
                    continuable: false,
                    background: true,
                    stored: true,
                    replay: Some(replay),
                    _credit: credit,
                }),
            },
        );
        store.expire_at(now + TTL);
        assert!(store.get("resp_fixture").is_err());
        assert!(store.reserve(1, service.as_ref()).is_err());
        drop(reader);
        assert!(store.reserve(1, service.as_ref()).is_err());
        assert!(std::str::from_utf8(&frame).unwrap().contains("retained"));
        drop(frame);
        assert_eq!(host.serving_bytes(), baseline);
        drop(store.reserve(MAX_BYTES, service.as_ref()).unwrap());
        assert_eq!(host.serving_bytes(), baseline);
    }
}
