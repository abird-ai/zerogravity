//! Deliberate wire projection of original owner observations. Never serialize
//! the full runtime diagnostic graph, raw errors, credentials, or file paths.
use crate::control_host::{Actions, Selection};
use serde_json::{Value, json};
use zerogravity::{Result, Runtime, control::ServeSummary};

pub(super) fn snapshot(
    runtime: &Runtime,
    selection: &Selection,
    actions: &Actions,
) -> Result<Value> {
    let owner = runtime.control().snapshot()?;
    let selected = selection.snapshot();
    let models: Vec<_> = owner.runtime.catalog.iter().map(|model| {
        let capability = model.capabilities.as_ref().map(|cap| json!({
            "context": cap.max_context,
            "vocabulary": cap.vocabulary_size,
            "chat": cap.chat_allowed,
            "provider": cap.details.execution.as_ref().map(|execution| format!("{:?}", execution.selected.provider)),
            "family": cap.details.artifact.as_ref().map(|artifact| &artifact.family),
            "snapshot": cap.details.artifact.as_ref().and_then(|artifact| artifact.cache_snapshot.as_ref()),
            "source_host_reserved_bytes": cap.details.source_host_reserved_bytes,
        }));
        json!({
            // Exact opaque IDs cannot round-trip through a JS number.
            "registration_id": model.registration_id.to_string(),
            "aliases": model.aliases,
            "lifecycle": model.lifecycle,
            "admitted_requests": model.admitted_requests,
            "capabilities": capability,
        })
    }).collect();
    let scheduler = owner.runtime.scheduler.as_ref().map(|scheduler| {
        json!({
            "context": scheduler.config.context,
            "output": scheduler.config.output,
            "active_limit": scheduler.active_limit,
            "active": scheduler.counters.active,
            "queued": scheduler.queue.pending,
            "output_tokens": scheduler.counters.output_tokens,
        })
    });
    let serving = match owner.serving {
        ServeSummary::Unavailable => json!({"state": "unavailable"}),
        ServeSummary::Idle => json!({"state": "idle"}),
        ServeSummary::Transitioning => json!({"state": "transitioning"}),
        ServeSummary::Listening {
            binding,
            accepted_requests,
            active_connections,
        } => json!({
            "state": "listening",
            "generation": binding.generation.to_string(),
            "address": binding.address.to_string(),
            "accepted_requests": accepted_requests,
            "active_connections": active_connections,
        }),
        ServeSummary::Finished { binding } => json!({
            "state": "finished",
            "generation": binding.generation.to_string(),
            "address": binding.address.to_string(),
        }),
    };
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    let download = owner.download.as_ref().map(|row| {
        json!({
            "id": row.id.to_string(),
            "phase": row.phase,
            "status": row.status,
            "terminal": row.status.terminal(),
            "cancel_requested": row.cancel_requested,
            "selected_files": row.selected_files,
            "completed_files": row.completed_files,
            "total_bytes": row.total_bytes,
            "total_expected_bytes": row.total_expected_bytes,
            "total_maximum_bytes": row.total_maximum_bytes,
            "cached_bytes": row.cached_bytes,
            "resumed_bytes": row.resumed_bytes,
            "checkpoint_bytes": row.checkpoint_bytes,
            "logical_bytes_per_second": row.logical_bytes_per_second,
            "snapshot": row.snapshot,
        })
    });
    #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
    let download: Option<Value> = None;
    let (_, action, last_web) = actions.observe();
    let action = action.map(|job| {
        json!({
            "id": job.id.to_string(), "origin": job.origin, "title": job.title,
            "cancel_requested": job.cancel_requested,
        })
    });
    Ok(json!({
        "schema": "zerogravity.control.v1",
        "capabilities": {
            "serve": cfg!(feature = "serve"),
            "cache": cfg!(all(feature = "cache", target_os = "linux")),
            "acquisition": cfg!(all(feature = "acquisition", target_os = "linux")),
            "actions": true,
        },
        // Named selection only, never a portable path or document bytes.
        "configuration": { "profile": selected.target.profile, "offline": selected.offline, "chat": selected.chat },
        "runtime": { "status": owner.runtime.status, "models": models, "scheduler": scheduler },
        "serving": serving,
        "download": download,
        "action": action,
        "web_action": last_web.map(|id| id.to_string()),
    }))
}
