//! Joined background protocol work; the engine still owns every inference.
//! Polling records never retain a strong request receipt.
use super::{
    Identity, Message, Options, Pending, Result, StreamEvent, Terminal, TextPart, Usage,
    delta_event, done_events, initial_events, response_error, store::Replay, terminal,
    terminal_status,
};
use crate::http::{
    ErrorReporting,
    body::{self, Reply},
};
use hyper::Response;
use serde_json::json;
use std::{
    future::{Future, poll_fn},
    sync::{Arc, Mutex},
};
use tokio::{
    sync::Notify,
    task::{JoinError, JoinSet},
    time,
};
use zerogravity_engine::router::{RequestEvents, ResponseEvent, ServiceError};

/// Single listener-owned join set. A new task consumes an already reserved
/// protocol record slot and an original engine receipt, not a second queue.
/// Reap completed tasks even while no new HTTP connections arrive.
#[derive(Default)]
pub(in crate::http) struct Jobs {
    tasks: Mutex<JoinSet<()>>,
    changed: Notify,
}
impl Jobs {
    pub(in crate::http) fn require_capacity(&self) -> Result<()> {
        if self.tasks.lock().expect("background jobs").len() >= super::store::MAX_RECORDS {
            return Err(ServiceError::new(
                503,
                "response_store_full",
                "background task capacity exhausted",
            ));
        }
        Ok(())
    }

    fn spawn(&self, work: impl Future<Output = ()> + Send + 'static) {
        self.tasks.lock().expect("background jobs").spawn(work);
        self.changed.notify_one();
    }

    pub(in crate::http) async fn join_next(&self) -> std::result::Result<(), JoinError> {
        loop {
            let changed = self.changed.notified();
            let joined = poll_fn(|cx| {
                self.tasks
                    .lock()
                    .expect("background jobs")
                    .poll_join_next(cx)
            })
            .await;
            if let Some(result) = joined {
                return result;
            }
            changed.await;
        }
    }

    /// Called after ingress tasks have joined, so nothing can submit again.
    pub(in crate::http) async fn stop(&self) -> std::result::Result<(), JoinError> {
        let mut tasks = std::mem::take(&mut *self.tasks.lock().expect("background jobs"));
        tasks.abort_all();
        let mut result = Ok(());
        while let Some(joined) = tasks.join_next().await {
            if let Err(error) = joined
                && !error.is_cancelled()
            {
                result = Err(error);
            }
        }
        result
    }
}

/// Consume the original streaming receipt independently of any subscriber.
/// Replay publication supplies the same decoded-output check as foreground.
async fn streamed_terminal(
    owner: &RequestEvents,
    identity: &Identity,
    replay: &Replay,
    partial: &mut String,
) -> Result<Terminal> {
    let deadline = time::Instant::now() + body::RESPONSE_TIMEOUT;
    loop {
        if time::Instant::now() >= deadline {
            return Err(ServiceError::new(
                504,
                "timeout",
                "generation deadline reached",
            ));
        }
        let event = time::timeout_at(deadline, body::next(owner))
            .await
            .map_err(|_| ServiceError::new(504, "timeout", "generation deadline reached"))?
            .map_err(|_| {
                ServiceError::new(503, "worker_unavailable", "model worker disconnected")
            })?;
        match event {
            ResponseEvent::Text(text) => {
                replay.delta(delta_event(identity, &text))?;
                partial.push_str(&text);
                // Cooperate with the listener even when the original
                // worker already has the next event ready.
                tokio::task::consume_budget().await;
            }
            ResponseEvent::TokenText { .. } => {
                return Err(ServiceError::new(
                    500,
                    "unexpected_token_observation",
                    "unrequested token observation",
                ));
            }
            ResponseEvent::Error(error) => return Err(error),
            ResponseEvent::Done {
                output,
                prompt_tokens,
                cached_prompt_tokens,
                completion_tokens,
                finish_reason,
            } => {
                if *partial != output {
                    return Err(ServiceError::new(
                        500,
                        "invalid_stream",
                        "text deltas differ from terminal output",
                    ));
                }
                return Ok(Terminal {
                    output,
                    status: terminal_status(finish_reason)?,
                    usage: Usage::new(prompt_tokens, cached_prompt_tokens, completion_tokens)?,
                });
            }
        }
    }
}

pub(in crate::http) fn start(
    jobs: &Jobs,
    request: RequestEvents,
    options: Options,
    request_id: &str,
    pending: Pending,
    errors: ErrorReporting,
) -> Result<Response<Reply>> {
    // The current-thread listener performs admission and this publication
    // without an await, so no other submitter can consume the checked slot.
    jobs.require_capacity()?;
    let identity = Identity::new(request_id);
    let owner = Arc::new(request);
    // Prepare every mandatory small snapshot before publishing an active ID.
    // "in_progress" denotes accepted protocol processing, not GPU activity.
    let initial_view = options.view(&identity, &owner.model, "in_progress", &[], None, None);
    let initial = pending.record(&initial_view, "", false)?;
    let cancelled = pending.record(
        &options.view(&identity, &owner.model, "cancelled", &[], None, None),
        "",
        false,
    )?;
    let failure =
        json!({"code":"server_error","message":"The background response could not complete."});
    let fallback_view = options.view(&identity, &owner.model, "failed", &[], None, Some(&failure));
    let fallback = pending.record(&fallback_view, "", false)?;
    let replay = pending.replay.clone();
    let response = if let Some(replay) = &replay {
        // Refuse before publishing the response ID if even the opening
        // events and mandatory failure projection cannot fit.
        replay.initialize(
            initial_events(&identity, &initial_view),
            StreamEvent::Response("response.failed", &fallback_view),
        )?;
        replay.reply(None)?
    } else {
        body::retained_json(initial.clone().bytes(), None)
    };
    let active = pending.publish_active(&owner, initial, cancelled);
    jobs.spawn(async move {
        let mut partial = String::new();
        // A stored notification survives cancellation before the task's first
        // poll. Once cancelled the store is terminal; late Done cannot win.
        let result = tokio::select! {
            biased;
            _ = active.wake.notified() => return,
            result = async {
                match &replay {
                    Some(replay) => streamed_terminal(&owner, &identity, replay, &mut partial).await,
                    None => terminal(&owner).await,
                }
            } => result,
        };
        let result = result.and_then(
            |Terminal { output, status, usage }| {
                let content = [TextPart::new(&output)];
                let items = [Message::new(&identity, status, &content)];
                let view = options.view(&identity, &owner.model, status, &items, Some(usage), None);
                let record = pending.record(&view, &output, true)?;
                // Prepare the entire terminal group before publishing either
                // it or a successful record/context append.
                let batch = replay.as_ref().map(|replay| {
                    replay.prepare_terminal(done_events(&identity, &view, &output, status, &content))
                }).transpose()?;
                Ok((record, Some(status), batch))
            },
        );
        let (record, status, batch) = match result {
            Ok(result) => result,
            Err(error) => {
                owner.cancel();
                // Retained events are shared by all authorized readers;
                // do not retain request-specific diagnostic details.
                let error = response_error(&errors, &error, false);
                let content = [TextPart::new(&partial)];
                let items = [Message::new(&identity, "incomplete", &content)];
                let items = if partial.is_empty() { &[][..] } else { &items[..] };
                let view = options.view(&identity, &owner.model, "failed", items, None, Some(&error));
                let failed = pending.record(&view, &partial, false).and_then(|record| {
                    let batch = replay.as_ref().map(|replay| {
                        replay.prepare_terminal([StreamEvent::Response("response.failed", &view)])
                    }).transpose()?;
                    Ok((record, None, batch))
                });
                failed.unwrap_or_else(|_| {
                    // Opening admission reserved space at the largest
                    // sequence number for this deterministic fallback.
                    let view = options.view(&identity, &owner.model, "failed", &[], None, Some(&failure));
                    let batch = replay.as_ref().map(|replay| {
                        replay.fallback(StreamEvent::Response("response.failed", &view))
                    });
                    (fallback, None, batch)
                })
            }
        };
        pending.finish_active(&active, record, status, batch);
        // Retire the original event receipt and projection reservation now;
        // no polling response, replay reader or terminal record keeps them.
        drop(owner);
    });
    Ok(response)
}
