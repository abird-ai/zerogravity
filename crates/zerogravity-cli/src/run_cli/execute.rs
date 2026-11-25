use super::*;

#[derive(Clone, Copy, Default)]
pub(super) struct Mode {
    pub interactive: bool,
    pub offer: bool,
    pub pull: bool,
}

pub(super) fn run(request: Request, mode: Mode) -> Result<()> {
    run_impl(request, devices::discover, mode)
}

#[cfg(all(
    test,
    feature = "qwen35-cuda-serving",
    not(feature = "qwen35-cuda-qualification"),
    target_os = "linux"
))]
pub(super) fn run_with_discovery(
    request: Request,
    discover: impl FnMut() -> Result<devices::Inventory>,
) -> Result<()> {
    run_impl(request, discover, Mode::default())
}

fn run_impl(
    mut request: Request,
    mut discover: impl FnMut() -> Result<devices::Inventory>,
    mode: Mode,
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
    crate::output::emit!(
        Verbose,
        "Requested backend: {}; context {}; max output {}",
        request.backend.label(),
        request.task.context,
        request.task.max_tokens
    );
    let mut inventory = discover()?;
    super::source::resolve(&mut request, &inventory, mode)?;
    if zerogravity::devices::fresh(&inventory).is_err() {
        inventory = discover()?;
    }
    crate::output::emit!(
        Diagnostic,
        "Observed {} device(s) for routing",
        inventory.devices.len()
    );
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
        return super::qwen_source::run(request, inventory, backend, mode.interactive);
    }
    if request.numerical.as_deref() == Some("native") {
        return Err(Error::Unsupported(
            "native numerical profile requires canonical Qwen ROCm source".into(),
        ));
    }
    if qwen35_cuda::selected(&request.backend, &request.graph.models[i]) {
        if mode.interactive {
            return Err(Error::Unsupported(
                "finite CUDA diagnostic route supports one explicit prompt only".into(),
            ));
        }
        qwen35_cuda::require_route_capability()?;
        return qwen35_cuda::run_route(request, inventory);
    }
    gguf(request, inventory, mode.interactive)
}
#[cfg(not(all(feature = "cpu", feature = "model-qwen35", feature = "text")))]
fn gguf(_: Request, _: devices::Inventory, _: bool) -> Result<()> {
    Err(Error::Unsupported(
        "GGUF run requires compiled features cpu, model-qwen35 and text".into(),
    ))
}
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
fn gguf(request: Request, inventory: devices::Inventory, interactive: bool) -> Result<()> {
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
        crate::output::emit!(
            Explicit,
            "{}",
            serde_json::to_string(plan.decision())
                .map_err(|error| Error::State(error.to_string()))?
        );
    }
    let target = plan.decision()["target"]
        .as_str()
        .unwrap_or("model")
        .to_owned();
    let mut prepared = plan.prepare()?;
    crate::output::emit!(
        Status,
        "{} · CPU · cpu:0 · context {}",
        target,
        prepared.task.context
    );
    crate::output::emit!(Verbose, "Selected CPU source: {}", prepared.source_label);
    if interactive {
        use zerogravity::diagnostics::cpu_text::{self, Diagnostics, TextOptions};
        let task = &prepared.task;
        return super::input::session(&target, |prompt| {
            cpu_text::generate_repeated(
                &mut prepared.source,
                TextOptions {
                    prompt,
                    context: task.context,
                    max_tokens: task.max_tokens,
                    raw: task.raw,
                    prefill_chunk: task.prefill_chunk,
                    teacher_ids: None,
                },
                Diagnostics::default(),
                |_| Ok(()),
                |output| {
                    println!("{}", output.text());
                    crate::output::emit!(
                        Status,
                        "zerogravity: {} generated tokens, {} total consumed (CPU)",
                        output.generated().len(),
                        output.consumed()
                    );
                    Ok(())
                },
            )
        });
    }
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
