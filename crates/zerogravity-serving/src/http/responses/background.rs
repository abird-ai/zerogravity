//! Joined background protocol work; the engine still owns every inference.
//! Polling records never retain a strong request receipt.
use super::{
    Identity, Message, Options, Pending, Result, Terminal, TextPart, response_error, terminal,
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
};
use zerogravity_engine::router::{RequestEvents, ServiceError};

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
    let initial = pending.record(
        &options.view(&identity, &owner.model, "in_progress", &[], None, None),
        "",
        false,
    )?;
    let cancelled = pending.record(
        &options.view(&identity, &owner.model, "cancelled", &[], None, None),
        "",
        false,
    )?;
    let failure =
        json!({"code":"server_error","message":"The background response could not complete."});
    let fallback = pending.record(
        &options.view(&identity, &owner.model, "failed", &[], None, Some(&failure)),
        "",
        false,
    )?;
    let active = pending.publish_active(&owner, initial.clone(), cancelled);
    let response = body::retained_json(initial.bytes(), None);
    jobs.spawn(async move {
        // A stored notification survives cancellation before the task's first
        // poll. Once cancelled the store is terminal; late Done cannot win.
        let result = tokio::select! {
            biased;
            _ = active.wake.notified() => return,
            result = terminal(&owner) => result,
        };
        let result = result.and_then(
            |Terminal {
                 output,
                 status,
                 usage,
             }| {
                let content = [TextPart::new(&output)];
                let items = [Message::new(&identity, status, &content)];
                pending
                    .record(
                        &options.view(&identity, &owner.model, status, &items, Some(usage), None),
                        &output,
                        true,
                    )
                    .map(|record| (record, Some(status)))
            },
        );
        let (record, status) = match result {
            Ok(record) => record,
            Err(error) => {
                owner.cancel();
                // Report only safe diagnostics. The pre-priced fallback
                // cannot fail even if the terminal output exceeded its cap.
                let error = response_error(&errors, &error, false);
                (
                    pending
                        .record(
                            &options.view(
                                &identity,
                                &owner.model,
                                "failed",
                                &[],
                                None,
                                Some(&error),
                            ),
                            "",
                            false,
                        )
                        .unwrap_or(fallback),
                    None,
                )
            }
        };
        pending.finish_active(&active, record, status);
        // Retire the original event receipt and projection reservation now;
        // no polling response or terminal record can keep them alive.
        drop(owner);
    });
    Ok(response)
}
