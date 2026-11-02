use super::*;
use std::ffi::OsString;
use zerogravity::run::config::{self as sdk, RunOverrides};

pub(super) fn resolve(
    a: Arguments,
    cwd: &Path,
    xdg: Option<OsString>,
    home: Option<OsString>,
) -> Result<Request> {
    let selector_free = a.config.is_none() && a.model.is_none() && a.graph.is_empty();
    let selected = if selector_free {
        let base = if let Some(p) = xdg.map(PathBuf::from).filter(|p| p.is_absolute()) {
            p
        } else {
            let h =
                PathBuf::from(home.ok_or_else(|| {
                    bad("HOME required when XDG_CONFIG_HOME is unset/nonabsolute")
                })?);
            if !h.is_absolute() {
                return Err(bad("HOME must be absolute"));
            }
            h.join(".config")
        };
        Some(base.join("abird/zerogravity/config.yaml"))
    } else {
        a.config.as_deref().map(|s| absolute(cwd, s))
    };
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
