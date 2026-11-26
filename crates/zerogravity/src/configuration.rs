//! One cold configuration document and its filesystem owner. No source,
//! device, network, runtime, credential or execution authority.
use crate::{
    Error, Result,
    pipelines::{self, ServeDoc},
    run::config::{self, DocumentSelection},
    run_policy::{Devices, Entrypoint, Headroom, Origin, Request, Speculation},
};
use serde_yaml::{Mapping, Value};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    path::{Path, PathBuf},
};

pub use crate::pipelines::{Document as ConfigDocument, DownloadDoc, ThemeMode, UiDoc, UiSwitch};
mod file;
pub use file::ConfigFile;

pub const INITIAL: &str = "# ZeroGravity configuration. Omit fields to use built-in defaults.\n\
# Named profiles use the same schema; no inheritance or credentials.\n\
schema: zerogravity.pipeline.v2\n";

fn bad(message: impl Into<String>) -> Error {
    Error::Malformed(message.into())
}

/// One selected document, read once. Only a missing implicit default falls
/// back to built-ins; named/explicit documents must exist and parse.
pub fn load_selected(
    selection: DocumentSelection<'_>,
    cwd: &Path,
    xdg: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Result<Option<(PathBuf, ConfigDocument)>> {
    if selection == DocumentSelection::Default
        && !xdg.is_some_and(|p| Path::new(p).is_absolute())
        && !home.is_some_and(|p| Path::new(p).is_absolute())
    {
        // No usable config namespace. Direct explicit requests retain
        // built-ins; editing or explicit profile selection needs a root.
        return Ok(None);
    }
    let Some(path) = config::select_document(selection, cwd, xdg, home)? else {
        return Ok(None);
    };
    load_path(path, selection == DocumentSelection::Default)
}

/// Read a previously resolved absolute document path using the same policy.
/// Only a caller's implicit-default selection may allow a missing file.
pub fn load_path(
    path: PathBuf,
    allow_missing_default: bool,
) -> Result<Option<(PathBuf, ConfigDocument)>> {
    if !path.is_absolute() {
        return Err(bad("configuration path must be absolute"));
    }
    if path.extension().is_some_and(|ext| ext == "json") {
        return Err(bad(
            "--config selects pipeline YAML; model metadata uses --model-config",
        ));
    }
    match pipelines::load_document(&path) {
        Ok(doc) => Ok(Some((path, doc))),
        Err(Error::Io(error))
            if allow_missing_default
                && error.kind() == std::io::ErrorKind::NotFound
                && matches!(std::fs::symlink_metadata(&path), Err(e) if e.kind() == std::io::ErrorKind::NotFound) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Fully typed, cold validation, including settings-only documents. Existing
/// graph validation stays authoritative; no model/device probing. A valid
/// document may still request an uncompiled or unqualified adapter.
pub fn validate(path: &Path, bytes: &[u8]) -> Result<ConfigDocument> {
    let doc = pipelines::parse_document(bytes)?;
    inspect_policy(path, &doc)?;
    Ok(doc)
}

/// Cold resolved document policy, including settings-only documents.
/// The placeholder entrypoint in an empty document is not a runnable model.
/// No source or device is opened; callers may inspect the returned policy.
pub fn inspect_policy(path: &Path, doc: &ConfigDocument) -> Result<Request> {
    let (request, _) = config::inspect_document(path, doc.clone(), false)?;
    if doc.serve.is_some() {
        let (serving, serve) = config::inspect_document(path, doc.clone(), true)?;
        validate_serve(&serving, &serve)?;
    }
    Ok(request)
}

/// Shared finite serving policy before physical admission.
pub fn validate_serve(request: &Request, settings: &ServeDoc) -> Result<()> {
    let listen = settings.listen.as_deref().unwrap_or("127.0.0.1:8000");
    let address: std::net::SocketAddr = listen
        .parse()
        .map_err(|_| bad("serve.listen requires numeric IP:PORT"))?;
    if !address.ip().is_loopback() {
        return Err(Error::Unsupported("serve.listen must be loopback".into()));
    }
    if settings
        .host_mib
        .is_some_and(|n| !(256..=1_048_576).contains(&n))
    {
        return Err(bad("serve.host_mib must be 256..=1048576"));
    }
    crate::runtime::text::validate_config(
        1,
        settings.workers.unwrap_or(2),
        settings.active_per_worker.unwrap_or(2),
        request.task.context,
        request.task.max_tokens,
        settings.max_pending_requests,
    )
    .map_err(|e| e.into_core())
}

pub fn document_value(path: &Path, bytes: &[u8]) -> Result<Value> {
    validate(path, bytes)?;
    pipelines::parse_document_node(bytes)
}

/// Dotted mapping paths address the real schema (e.g. run.task.context).
/// Lists are whole typed values, not unstable index-addressed records.
fn path_parts(path: &str) -> Result<Vec<&str>> {
    let parts: Vec<_> = path.split('.').collect();
    if parts.len() > 8
        || path.len() > 256
        || parts
            .iter()
            .any(|p| p.is_empty() || !p.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'))
    {
        return Err(bad("config key requires a bounded dotted schema path"));
    }
    Ok(parts)
}

fn lookup<'a>(node: &'a Value, path: &str) -> Result<Option<&'a Value>> {
    let mut value = node;
    for part in path_parts(path)? {
        let Some(next) = value.get(part) else {
            return Ok(None);
        };
        value = next;
    }
    Ok(Some(value))
}

#[derive(Clone, Copy, Debug)]
pub enum Change<'a> {
    Set { path: &'a str, yaml: &'a str },
    Unset { path: &'a str },
}

pub fn changed(path: &Path, bytes: &[u8], edit: Change<'_>) -> Result<Vec<u8>> {
    let mut node = pipelines::parse_document_node(bytes)?;
    // Require current typed shape, but allow a typed semantic mistake to be
    // corrected by a single edit. Unknown fields can be repaired via edit.
    let _ = pipelines::parse_document(bytes)?;
    let (key, value) = match edit {
        Change::Set { path, yaml } => {
            if yaml.len() > pipelines::MAX_CONFIG_BYTES {
                return Err(Error::Limit("configuration value size".into()));
            }
            let value: Value = serde_yaml::from_str(yaml).map_err(|_| bad("invalid YAML value"))?;
            (path, Some(value))
        }
        Change::Unset { path } => (path, None),
    };
    let parts = path_parts(key)?;
    if parts[0] == "schema" {
        return Err(bad("schema version is not an editable setting"));
    }
    let mut current = &mut node;
    for part in &parts[..parts.len() - 1] {
        let map = current
            .as_mapping_mut()
            .ok_or_else(|| bad("config path must address mapping keys"))?;
        let field = Value::String((*part).into());
        if value.is_none() && !map.contains_key(&field) {
            return Err(bad("config key is not set"));
        }
        current = map
            .entry(field)
            .or_insert_with(|| Value::Mapping(Mapping::new()));
    }
    let map = current
        .as_mapping_mut()
        .ok_or_else(|| bad("config path must address mapping keys"))?;
    let field = Value::String(parts.last().unwrap().to_string());
    if let Some(value) = value {
        map.insert(field, value);
    } else if map.remove(field).is_none() {
        return Err(bad("config key is not set"));
    }
    let output = serde_yaml::to_string(&node)
        .map_err(|_| bad("cannot encode configuration"))?
        .into_bytes();
    validate(path, &output)?;
    Ok(output)
}

/// Effective startup setting values with the existing resolver's provenance.
/// No selected device, admission, source verification or execution inferred.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Setting {
    pub value: Value,
    pub origin: String,
}

fn origin(origin: Option<&Origin>, path: &Path, present: bool) -> String {
    match origin {
        Some(Origin::Config(file)) => file.display().to_string(),
        Some(Origin::Cli) => "CLI".into(),
        Some(Origin::Caller) => "caller".into(),
        Some(Origin::Builtin) => "built-in".into(),
        None if present => path.display().to_string(),
        None => "built-in".into(),
    }
}
fn yaml(value: impl serde::Serialize) -> Value {
    serde_yaml::to_value(value).expect("configuration scalar/list serialization")
}

pub fn explain(path: &Path, bytes: &[u8]) -> Result<BTreeMap<String, Setting>> {
    let doc = validate(path, bytes)?;
    let node = pipelines::parse_document_node(bytes)?;
    let (request, _) = config::inspect_document(path, doc.clone(), false)?;
    let mut rows = BTreeMap::new();
    let mut add = |key: &str, value: Value, provenance: Option<&Origin>| -> Result<()> {
        let explicit = lookup(&node, key)?.is_some();
        rows.insert(
            key.into(),
            Setting {
                value,
                origin: origin(provenance, path, explicit),
            },
        );
        Ok(())
    };
    let p = &request.provenance;
    add(
        "run.backend",
        yaml(request.backend.label()),
        p.get("backend"),
    )?;
    let devices = |devices: &Devices| match devices {
        Devices::Auto => yaml(vec!["auto".to_string()]),
        Devices::Ordered(devices) => yaml(
            devices
                .iter()
                .map(|device| device.label())
                .collect::<Vec<_>>(),
        ),
    };
    add("run.device", devices(&request.devices), p.get("device"))?;
    add(
        "run.headroom",
        yaml(match request.headroom {
            Headroom::Default => "auto (10%, minimum 1 GiB host / 256 MiB device)".into(),
            Headroom::Bytes(n) => n.to_string(),
            Headroom::Percent(n) => format!("{n}%"),
        }),
        p.get("headroom"),
    )?;
    add(
        "run.numerical",
        yaml(request.numerical.as_deref().unwrap_or("source")),
        p.get("numerical"),
    )?;
    add(
        "run.speculation",
        yaml(match request.speculation {
            Speculation::Auto => "auto".into(),
            Speculation::Off => "off".into(),
            Speculation::Mtp => "mtp".into(),
            Speculation::Draft(model) => format!("draft:{model}"),
        }),
        p.get("speculation"),
    )?;
    if let Some(dev) = &request.draft_device {
        add("run.draft_device", devices(dev), p.get("draft_device"))?;
    }
    add(
        "run.task.context",
        yaml(request.task.context),
        p.get("task.context"),
    )?;
    add(
        "run.task.max_tokens",
        yaml(request.task.max_tokens),
        p.get("task.max_tokens"),
    )?;
    add("run.task.raw", yaml(request.task.raw), p.get("task.raw"))?;
    add(
        "run.task.prefill_chunk",
        yaml(request.task.prefill_chunk),
        p.get("task.prefill_chunk"),
    )?;
    add(
        "run.diagnostics.explain",
        yaml(request.diagnostics.explain),
        p.get("diagnostics.explain"),
    )?;
    add(
        "run.diagnostics.dry_run",
        yaml(request.diagnostics.dry_run),
        p.get("diagnostics.dry_run"),
    )?;
    if doc.run.as_ref().is_some_and(|run| run.entrypoint.is_some()) {
        add(
            "run.entrypoint",
            yaml(match request.entrypoint {
                Entrypoint::Model(id) => format!("model:{id}"),
                Entrypoint::Pipeline(id) => format!("pipeline:{id}"),
            }),
            p.get("entrypoint"),
        )?;
    }
    for key in ["model", "pipeline", "route"] {
        add(
            key,
            node.get(key)
                .cloned()
                .unwrap_or_else(|| yaml(Vec::<Value>::new())),
            None,
        )?;
    }
    let ui = doc.ui.unwrap_or_default();
    add("ui.theme", yaml(ui.theme.unwrap_or_default()), None)?;
    add("ui.mouse", yaml(ui.mouse.unwrap_or_default()), None)?;
    add("ui.color", yaml(ui.color.unwrap_or_default()), None)?;
    add("ui.progress", yaml(ui.progress.unwrap_or_default()), None)?;
    add(
        "ui.interactive",
        yaml(ui.interactive.unwrap_or_default()),
        None,
    )?;
    add(
        "downloads.ask_before_pull",
        yaml(
            doc.downloads
                .unwrap_or_default()
                .ask_before_pull
                .unwrap_or(true),
        ),
        None,
    )?;
    let serve = doc.serve.unwrap_or_default();
    let run = doc.run.as_ref().and_then(|r| r.task.as_ref());
    add(
        "serve.listen",
        yaml(serve.listen.as_deref().unwrap_or("127.0.0.1:8000")),
        None,
    )?;
    add(
        "serve.workers",
        serve
            .workers
            .map_or_else(|| yaml("auto (CPU 2 / original GPU 1)"), yaml),
        None,
    )?;
    add(
        "serve.active_per_worker",
        yaml(serve.active_per_worker.unwrap_or(2)),
        None,
    )?;
    let inherited = |explicit, run_value| if explicit { None } else { p.get(run_value) };
    add(
        "serve.max_context",
        yaml(
            serve
                .max_context
                .or(run.and_then(|t| t.context))
                .unwrap_or(1024),
        ),
        inherited(serve.max_context.is_some(), "task.context"),
    )?;
    add(
        "serve.max_output",
        yaml(
            serve
                .max_output
                .or(run.and_then(|t| t.max_tokens))
                .unwrap_or(128),
        ),
        inherited(serve.max_output.is_some(), "task.max_tokens"),
    )?;
    add("serve.host_mib", yaml(serve.host_mib.unwrap_or(4096)), None)?;
    add(
        "serve.max_pending_requests",
        yaml(serve.max_pending_requests),
        None,
    )?;
    add(
        "serve.enable_per_request_metrics",
        yaml(serve.enable_per_request_metrics.unwrap_or(false)),
        None,
    )?;
    add(
        "serve.enable_error_details",
        yaml(serve.enable_error_details.unwrap_or(false)),
        None,
    )?;
    Ok(rows)
}

pub fn get(path: &Path, bytes: &[u8], key: &str) -> Result<Setting> {
    let node = document_value(path, bytes)?;
    if let Some(value) = lookup(&node, key)? {
        return Ok(Setting {
            value: value.clone(),
            origin: path.display().to_string(),
        });
    }
    explain(path, bytes)?
        .remove(key)
        .ok_or_else(|| bad("unknown or unset configuration key"))
}

/// Copy one document without retargeting relative local sources. Named
/// profiles share a base; copies involving default may need absolute anchors.
pub fn copy_document(source: &Path, destination: &Path, bytes: &[u8]) -> Result<Vec<u8>> {
    validate(source, bytes)?;
    let base = source
        .parent()
        .ok_or_else(|| bad("configuration parent unavailable"))?;
    if source.parent() == destination.parent() {
        return Ok(bytes.to_vec());
    }
    let mut node = pipelines::parse_document_node(bytes)?;
    let anchor_path = |value: &mut Value| -> Result<()> {
        let path = Path::new(
            value
                .as_str()
                .ok_or_else(|| bad("configuration path must be text"))?,
        );
        if path.is_relative() {
            *value = yaml(base.join(path).to_string_lossy());
        }
        Ok(())
    };
    if let Some(models) = node.get_mut("model").and_then(Value::as_sequence_mut) {
        for model in models {
            if let Some(source) = model.get_mut("source") {
                let mut value = source
                    .as_str()
                    .ok_or_else(|| bad("model source must be text"))?
                    .to_owned();
                config::anchor_source(&mut value, base)?;
                *source = yaml(value);
            }
            for key in ["model_config", "tokenizer", "native_artifacts"] {
                if let Some(value) = model.get_mut(key) {
                    anchor_path(value)?;
                }
            }
        }
    }
    if let Some(diagnostics) = node
        .get_mut("run")
        .and_then(|run| run.get_mut("diagnostics"))
    {
        for key in ["audit_logits_jsonl", "audit_layers_jsonl"] {
            if let Some(value) = diagnostics.get_mut(key) {
                anchor_path(value)?;
            }
        }
    }
    let output = serde_yaml::to_string(&node)
        .map_err(|_| bad("cannot encode copied configuration"))?
        .into_bytes();
    validate(destination, &output)?;
    Ok(output)
}

/// Named profiles have the same schema. No persisted selected-profile pointer.
pub fn profiles(root: &Path) -> Result<Vec<String>> {
    let mut names = vec!["default".into()];
    let entries = match std::fs::read_dir(root.join("profiles")) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(names),
        Err(error) => return Err(error.into()),
    };
    for (index, entry) in entries.enumerate() {
        let entry = entry?;
        if index >= 4096 {
            return Err(Error::Limit("profile inventory extent".into()));
        }
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str().and_then(|n| n.strip_suffix(".yaml")) else {
            continue;
        };
        if name != "default" && config::validate_profile_name(name).is_ok() {
            names.push(name.into());
        }
    }
    names[1..].sort_unstable();
    Ok(names)
}
