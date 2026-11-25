use super::*;
use std::ffi::OsString;
use zerogravity::run::config::{self as sdk, RunOverrides};

pub(super) fn resolve(
    a: Arguments,
    cwd: &Path,
    xdg: Option<OsString>,
    home: Option<OsString>,
) -> Result<Request> {
    use sdk::DocumentSelection;
    let selection = match (&a.config, &a.profile) {
        (Some(_), Some(_)) => return Err(bad("--config and --profile are mutually exclusive")),
        (Some(file), None) => DocumentSelection::File(Path::new(file)),
        (None, Some(profile)) => DocumentSelection::Profile(profile),
        (None, None) if a.model.is_none() && a.graph.is_empty() => DocumentSelection::Default,
        (None, None) => DocumentSelection::None,
    };
    let selected = sdk::select_document(selection, cwd, xdg.as_deref(), home.as_deref())?;
    let graph = if a.graph.is_empty() {
        None
    } else {
        Some(crate::pipeline::patch(&a.graph, true)?)
    };
    sdk::resolve(
        selected.as_deref(),
        RunOverrides {
            model: a.model,
            model_config: a.model_config,
            tokenizer: a.tokenizer,
            snapshot: a.snapshot,
            native_artifacts: a.native_artifacts,
            backend: a.backend,
            devices: a.devices,
            headroom: a.headroom,
            entrypoint: a.entrypoint,
            task: a.task,
            diagnostics: a.diagnostics,
            numerical: a.numerical,
            speculation: a.speculation,
            draft_device: a.draft_device,
            graph,
            vendor: a.vendor,
        },
        cwd,
        Origin::Cli,
    )
}
