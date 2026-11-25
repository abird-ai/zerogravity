use super::*;
use std::ffi::OsString;
use zerogravity::run::config::{self as sdk, RunOverrides};

#[cfg(test)]
pub(super) fn resolve(
    a: Arguments,
    cwd: &Path,
    xdg: Option<OsString>,
    home: Option<OsString>,
) -> Result<Request> {
    resolve_selected(a, cwd, xdg, home).map(|(request, _)| request)
}

pub(super) fn resolve_selected(
    a: Arguments,
    cwd: &Path,
    xdg: Option<OsString>,
    home: Option<OsString>,
) -> Result<(Request, bool)> {
    use sdk::DocumentSelection;
    let selection = match (&a.config, &a.profile) {
        (Some(_), Some(_)) => return Err(bad("--config and --profile are mutually exclusive")),
        (Some(file), None) => DocumentSelection::File(Path::new(file)),
        (None, Some(profile)) => DocumentSelection::Profile(profile),
        (None, None) => DocumentSelection::Default,
    };
    let selected =
        zerogravity::configuration::load_selected(selection, cwd, xdg.as_deref(), home.as_deref())?;
    let preferences = selected
        .as_ref()
        .map(|(_, doc)| {
            (
                doc.ui.unwrap_or_default(),
                doc.downloads.unwrap_or_default(),
            )
        })
        .unwrap_or_default();
    // Selected document defaults apply to direct positional sources, not a
    // hidden config model pin. Explicit --model overrides retain their old contract
    // when a document provides a model entrypoint.
    let mut selected = selected;
    if a.direct_positional
        && a.model.is_some()
        && a.entrypoint.is_none()
        && a.graph.is_empty()
        && let Some((_, doc)) = &mut selected
    {
        doc.model.clear();
        doc.pipeline.clear();
        doc.route.clear();
        if let Some(run) = &mut doc.run {
            run.entrypoint = None;
        }
    }
    let graph = if a.graph.is_empty() {
        None
    } else {
        Some(crate::pipeline::patch(&a.graph, true)?)
    };
    let request = sdk::resolve_run_document(
        selected,
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
    )?;
    #[cfg(not(test))]
    crate::output::install_document(preferences.0);
    Ok((request, preferences.1.ask_before_pull.unwrap_or(true)))
}
