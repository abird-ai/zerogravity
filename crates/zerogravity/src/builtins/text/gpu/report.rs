//! Common device observations; no execution, reference cases or source recipes.
use serde_json::Value;
use zerogravity_engine::router::worker::SchedulerSnapshot;
pub(crate) struct Resources {
    pub host: Option<(u64, u64, Option<u64>, u64, u64)>,
    pub fixed: Option<Value>,
    pub hf_source: Option<Value>,
    pub hf_resources: Option<Value>,
}
impl Resources {
    pub fn ordinary(snapshot: &SchedulerSnapshot) -> Self {
        Self {
            host: snapshot.host.as_ref().map(|host| {
                let (limit, reserved, _, serving) = host.totals();
                (limit, reserved, reserved.checked_sub(serving), 0, serving)
            }),
            fixed: None,
            hf_source: None,
            hf_resources: None,
        }
    }
}
pub(crate) fn render(snapshot: &SchedulerSnapshot, resources: Resources) -> Value {
    let m = &snapshot.counters;
    let mut value = serde_json::json!({"kind":"resident_gpu",
                    "physical_accounting":"ResidentService/ResourceArbiter",
                    "batch_mode":if snapshot.cohort_width != 0 {"fused"} else {"interleaved"},
                    "native_batch_rows_max":snapshot.cohort_width.max(1),
                    "gpu_step_attempts":m.attempted,
                    "completed_gpu_quanta":m.completed,
                    "completed_quantum_unit":"request work, including terminal presentation",
                    "gpu_forward_steps_completed":m.gpu_forwards,
                    "gpu_forward_calls_completed":m.gpu_forwards,
                    "completed_model_token_steps":m.model_tokens,
                    "completed_forward_kernel_dispatches":m.kernel_dispatches,
                    "committed_gpu_roots":m.committed_roots,
                    "native_projection_dispatches_completed":m.fused_projections,
                    "fused_sampler_dispatches_completed":m.fused_sampler,
                    "gpu_fused_rows_completed":m.fused_rows,
                    "gpu_fused_state_only_batches":m.fused_state_only_batches,
                    "gpu_fused_state_only_rows":m.fused_state_only_rows,
                    "gpu_fused_greedy_batches":m.fused_greedy_batches,
                    "gpu_fused_batches_by_rows":{"2":m.cohorts_by_width[0],"4":m.cohorts_by_width[1],"8":m.cohorts_by_width[2]},
                    "cohort_admission_refusals":m.cohort_admission_refusals,
                    "discarded_ready_cohorts":m.discarded_cohorts,
                    "accepted_output_tokens":m.output_tokens,
                    "output_credit_parked_requests":m.parked,
                    "failed_gpu_quanta":m.failed,
                    "cancelled_requests":m.cancelled});
    let tail = serde_json::json!({
                    "active_sessions":m.active,
                    "peak_active_sessions":m.peak,
                    "active_sessions_limit":snapshot.active_limit,
                    "fixed-prompt_resources":resources.fixed,
                    "hf_source":resources.hf_source,"hf_resources":resources.hf_resources,
                    "queued_requests":snapshot.queue.pending,
                    "pending_admission_in_progress":snapshot.queue.in_progress,
                    "pending_queue_capacity":snapshot.queue.capacity,
                    "pending_queue_charged_bytes":snapshot.queue.backing_bytes,
                    "pending_queue_consumers":snapshot.queue.consumers,
                    "queued_requests_limit":snapshot.queue.limit,
                    "total_host_limit_bytes":resources.host.map(|h|h.0),
                    "total_host_reserved_bytes":resources.host.map(|h|h.1),
                    "model_host_reserved_bytes":resources.host.map(|h|h.2),
                    "backend_host_reserved_bytes":resources.host.map(|h|h.3),
                    "serving_host_reserved_bytes":resources.host.map(|h|h.4),
                    "scheduling_groups":m.groups,
                    "co_scheduled_members":m.co_scheduled,
                    "state_cohort_publications":m.cohort_publications,
                    "state_cohort_rows":m.cohort_rows,
                    "gpu_fused_batches":m.fused_batches});
    if let (Some(target), Some(source)) = (value.as_object_mut(), tail.as_object()) {
        target.extend(source.iter().map(|(key, item)| (key.clone(), item.clone())));
    }

    value
}
