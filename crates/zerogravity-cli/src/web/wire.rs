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
            "layer_placement": cap.details.layer_placement,
            "memory": cap.details.memory.as_ref().map(|memory| {
                use zerogravity::models::ModelMemory;
                match memory {
                    ModelMemory::Host { session_state_bytes, session_scratch_bytes } =>
                        json!({"host_state_bytes":session_state_bytes,"host_scratch_bytes":session_scratch_bytes}),
                    ModelMemory::Resident(memory) => json!({
                        "device":memory.device,"resident_bytes":memory.resident_bytes,
                        "model_capacity_bytes":memory.model_capacity_bytes,
                    }),
                    ModelMemory::ResidentText{admission,resident_bytes} => json!({
                        "device":admission.device,"resident_bytes":resident_bytes,
                        "resident_weight_bytes":admission.resident_weight_bytes,
                        "state_reserve_bytes":admission.state_reserve_bytes,
                        "model_capacity_bytes":admission.admitted_model_capacity_bytes,
                        "context_device_budget_bytes":admission.context_device_budget_bytes,
                    }),
                }
            }),

        }));
        json!({
            // Exact opaque IDs cannot round-trip through a JS number.
            "registration_id": model.registration_id.to_string(),
            "aliases": model.aliases,
            "lifecycle": model.lifecycle,
            "admitted_requests": model.admitted_requests,
            "performance": {
                "loaded_ms":model.performance.loaded_ms,
                "output_tokens":model.performance.output_tokens,
                "completed_requests":model.performance.completed_requests,
                "avg_decode_tps":model.performance.decode_tps(),
                "avg_ttft_ms":model.performance.mean_ttft_ms(),
                "ttft_samples":model.performance.ttft_samples,
                "decode_intervals":model.performance.decode_intervals,
                "last_completed":model.performance.last_completed.map(|sample|json!({
                    "decode_tps":sample.decode_tps(),"ttft_ms":sample.ttft_ms(),
                    "output_tokens":sample.output_tokens,
                })),
            },
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
        "configuration": { "profile": selected.target.profile, "offline": selected.offline, "chat": selected.chat, "theme": selected.ui.theme.unwrap_or_default() },
        "runtime": { "status": owner.runtime.status, "models": models, "scheduler": scheduler },
        "serving": serving,
        "download": download,
        "action": action,
        "web_action": last_web.map(|id| id.to_string()),
    }))
}

/// Cold inventory is a separate authenticated read. Progress/token events
/// never force cache scans; the browser requests it on entry/invalidation.
pub(super) fn models(runtime: &Runtime) -> Value {
    let catalog: Vec<_> = zerogravity::targets::MODEL_TARGETS.iter().map(|target| {
        #[cfg(all(feature = "acquisition", feature = "model-qwen35", target_os = "linux"))]
        let recipe = (!target.payload_hold).then(|| zerogravity::models::QwenPullRecipe::for_target(target.slug).ok()).flatten().map(|recipe| json!({
            "snapshot": recipe.snapshot(), "bytes": recipe.bytes(),
            "repository": recipe.repository(), "revision": recipe.revision(),
        }));
        #[cfg(not(all(feature = "acquisition", feature = "model-qwen35", target_os = "linux")))]
        let recipe: Option<Value> = None;
        json!({
            "slug": target.slug, "task": target.task.label(),
            "repository": target.repository, "source_layout": target.source_layout.label(),
            "source_status": target.source_status.label(), "execution_status": target.execution_status.label(),
            "payload_hold": target.payload_hold, "note": target.note, "recipe": recipe,
        })
    }).collect();

    let mut result = json!({"cached": [], "catalog": catalog, "partial": [], "warning": null});
    #[cfg(all(feature = "cache", target_os = "linux"))]
    {
        use crate::control_host::inventory;
        match inventory::cache(runtime.control()) {
            Ok(observation) => {
                result["cached"] = json!(observation.cached.iter().map(|model| json!({
                    "digest": model.snapshot,
                    "provider": model.provider,
                    "repository": inventory::public_repository(&model.provider, &model.repository),
                    "revision": inventory::public_revision(&model.provider, &model.revision),
                    "files": model.files, "bytes": model.bytes, "payload_verified": model.payload_verified,
                })).collect::<Vec<_>>());
                match observation.partial {
                    Ok(partial) => result["partial"] = json!(partial),
                    Err(_) => {
                        result["warning"] = json!(
                            "Partial inventory unavailable; cached snapshot metadata remains visible."
                        )
                    }
                }
            }
            Err(_) => {
                result["warning"] = json!(
                    "Cache observation unavailable; inspect the host cache in the native client."
                )
            }
        }
    }
    #[cfg(not(all(feature = "cache", target_os = "linux")))]
    {
        let _ = runtime;
        result["warning"] = json!("Cache inspection is unavailable in this build.");
    }
    result
}

pub(super) fn devices(runtime: &Runtime) -> Result<Value> {
    let inventory = runtime.control().devices()?;
    Ok(json!({
        "observed_unix_ms": inventory.observed_unix_ms.to_string(),
        "devices": inventory.devices.into_iter().map(|row| json!({
            "id": row.id, "class": row.class, "vendor": row.vendor,
            "backends": row.backends, "ceiling": row.ceiling, "available": row.available,
            "provenance": row.provenance, "exclusion": row.exclusion,
        })).collect::<Vec<_>>(),
    }))
}
