use super::*;

pub(super) fn run(request: Request) -> Result<()> {
    run_with_discovery(request, devices::discover)
}

pub(super) fn run_with_discovery(
    request: Request,
    mut discover: impl FnMut() -> Result<devices::Inventory>,
) -> Result<()> {
    zerogravity::run::validate_source_request(&request)?;
    let Entrypoint::Model(id) = &request.entrypoint else {
        request.graph.validate_entrypoint(&request.entrypoint)?;
        return Err(Error::Unsupported(
            "graph run adapter is not qualified; graph validation succeeded, routes remain inert"
                .into(),
        ));
    };
    let i = request
        .graph
        .models
        .iter()
        .position(|m| &m.id == id)
        .ok_or_else(|| bad("direct entrypoint model absent"))?;
    let inventory = discover()?;
    #[cfg(all(
        feature = "model-qwen35",
        feature = "cache",
        feature = "text",
        target_os = "linux",
        any(feature = "cuda", feature = "hip")
    ))]
    if let Some(backend) =
        super::qwen_source::selected_backend(&request, &request.graph.models[i], &inventory)?
    {
        return super::qwen_source::run(request, inventory, backend);
    }
    if qwen35_cuda::selected(&request.backend, &request.graph.models[i]) {
        qwen35_cuda::require_route_capability()?;
        return qwen35_cuda::run_route(request, inventory);
    }
    gguf(request, inventory)
}
#[cfg(not(all(feature = "cpu", feature = "model-qwen35", feature = "text")))]
fn gguf(_: Request, _: devices::Inventory) -> Result<()> {
    Err(Error::Unsupported(
        "GGUF run requires compiled features cpu, model-qwen35 and text".into(),
    ))
}
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
fn gguf(request: Request, inventory: devices::Inventory) -> Result<()> {
    let plan = zerogravity::run::cpu::Plan::new(request, inventory)?;
    if plan.dry_run() {
        println!(
            "{}",
            serde_json::to_string_pretty(plan.decision())
                .map_err(|error| Error::State(error.to_string()))?
        );
        return plan.require_proposal();
    }
    if plan.explain() {
        eprintln!(
            "{}",
            serde_json::to_string(plan.decision())
                .map_err(|error| Error::State(error.to_string()))?
        );
    }
    let prepared = plan.prepare()?;
    let options = crate::InferenceOptions {
        prompt: prepared.task.prompt,
        max_tokens: prepared.task.max_tokens,
        raw: prepared.task.raw,
        context: prepared.task.context,
        audit_jsonl: prepared
            .diagnostics
            .audit_logits_jsonl
            .map(|p| p.to_string_lossy().into_owned()),
        audit_layers_jsonl: prepared
            .diagnostics
            .audit_layers_jsonl
            .map(|p| p.to_string_lossy().into_owned()),
        teacher_ids: prepared.task.teacher_ids,
        prefill_chunk: prepared.task.prefill_chunk,
    };
    crate::infer_prepared(&prepared.source_label, prepared.source, options)
}
