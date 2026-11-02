//! Shared run-document precedence and source-path policy. No environment,
//! stdin, credentials, model/device loading, admission or output-file creation.
use crate::pipelines::{
    Config, DiagnosticDoc, EntryDoc, ModelInstance, PipelinePatch, RunDoc, TaskDoc,
};
use crate::{Error, Result, run_policy::*};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// Explicit caller overrides. None inherits configuration/defaults; Some(false),
/// Some("") and present empty graph lists remain meaningful overrides.
#[derive(Clone, Debug, Default)]
pub struct RunOverrides {
    pub model: Option<String>,
    pub model_config: Option<String>,
    pub tokenizer: Option<String>,
    pub snapshot: Option<String>,
    pub native_artifacts: Option<String>,
    pub backend: Option<Backend>,
    pub devices: Option<Vec<String>>,
    pub headroom: Option<Headroom>,
    pub entrypoint: Option<Entrypoint>,
    pub task: TaskDoc,
    pub diagnostics: DiagnosticDoc,
    pub numerical: Option<String>,
    pub speculation: Option<Speculation>,
    pub draft_device: Option<Vec<String>>,
    pub graph: Option<PipelinePatch>,
    pub vendor: Option<String>,
}

/// Resolve an explicitly selected bounded YAML file and typed overrides.
/// No implicit default-file discovery: callers supply their own XDG policy.
/// Provenance is descriptive, never source authentication or execution authority.
/// Only the configuration file is read; every model/source/device stays cold.
pub fn resolve(
    config: Option<&Path>,
    mut a: RunOverrides,
    cwd: &Path,
    override_origin: Origin,
) -> Result<Request> {
    if !cwd.is_absolute() {
        return Err(bad("run base directory must be absolute"));
    }
    let selected = config.map(|path| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            cwd.join(path)
        }
    });
    let mut provenance = BTreeMap::new();
    let (mut graph, run) = if let Some(path) = &selected {
        if path.extension().is_some_and(|e| e == "json") {
            return Err(bad(
                "--config selects pipeline YAML; old model metadata moved to --model-config",
            ));
        }
        let mut doc = crate::pipelines::load_document(
            path.to_str()
                .ok_or_else(|| bad("config path must be UTF-8"))?,
        )?;
        let base = path
            .parent()
            .ok_or_else(|| bad("config parent unavailable"))?;
        for m in &mut doc.model {
            if let Some(s) = &mut m.source {
                anchor_source(s, base)?;
            }
            for p in [
                &mut m.model_config,
                &mut m.tokenizer,
                &mut m.native_artifacts,
            ]
            .into_iter()
            .flatten()
            {
                *p = absolute(base, p).to_string_lossy().into_owned();
            }
        }
        let run = doc.run.clone().unwrap_or_default();
        let mut run = run;
        if let Some(d) = &mut run.diagnostics {
            for p in [&mut d.audit_logits_jsonl, &mut d.audit_layers_jsonl]
                .into_iter()
                .flatten()
            {
                *p = absolute(base, p).to_string_lossy().into_owned();
            }
        }
        (doc.compile()?, run)
    } else {
        (
            Config {
                models: Vec::new(),
                pipelines: Vec::new(),
                routes: Vec::new(),
            },
            RunDoc::default(),
        )
    };
    let origin = selected
        .as_ref()
        .map(|p| Origin::Config(p.clone()))
        .unwrap_or(Origin::Builtin);
    macro_rules! note {
        ($name:expr,$cli:expr,$cfg:expr) => {
            provenance.insert(
                $name.into(),
                if $cli {
                    override_origin.clone()
                } else if $cfg {
                    origin.clone()
                } else {
                    Origin::Builtin
                },
            );
        };
    }
    note!(
        "entrypoint",
        a.entrypoint.is_some() || a.model.is_some(),
        run.entrypoint.is_some()
    );
    note!("device", a.devices.is_some(), run.device.is_some());
    note!("headroom", a.headroom.is_some(), run.headroom.is_some());
    note!("numerical", a.numerical.is_some(), run.numerical.is_some());
    note!(
        "speculation",
        a.speculation.is_some(),
        run.speculation.is_some()
    );
    note!(
        "draft_device",
        a.draft_device.is_some(),
        run.draft_device.is_some()
    );
    let ct = run.task.as_ref();
    note!(
        "task.prompt",
        a.task.prompt.is_some(),
        ct.is_some_and(|t| t.prompt.is_some())
    );
    note!(
        "task.teacher_ids",
        a.task.teacher_ids.is_some(),
        ct.is_some_and(|t| t.teacher_ids.is_some())
    );
    let cd = run.diagnostics.as_ref();
    note!(
        "diagnostics.explain",
        a.diagnostics.explain.is_some(),
        cd.is_some_and(|d| d.explain.is_some())
    );
    note!(
        "diagnostics.dry_run",
        a.diagnostics.dry_run.is_some(),
        cd.is_some_and(|d| d.dry_run.is_some())
    );
    note!(
        "diagnostics.audit_logits_jsonl",
        a.diagnostics.audit_logits_jsonl.is_some(),
        cd.is_some_and(|d| d.audit_logits_jsonl.is_some())
    );
    note!(
        "diagnostics.audit_layers_jsonl",
        a.diagnostics.audit_layers_jsonl.is_some(),
        cd.is_some_and(|d| d.audit_layers_jsonl.is_some())
    );
    for m in &graph.models {
        for (field, present) in [
            ("target", true),
            ("source", m.source.is_some()),
            ("model_config", m.model_config.is_some()),
            ("tokenizer", m.tokenizer.is_some()),
            ("snapshot", m.snapshot.is_some()),
            ("native_artifacts", m.native_artifacts.is_some()),
        ] {
            provenance.insert(
                format!("model.{}.{field}", m.id),
                if present {
                    origin.clone()
                } else {
                    Origin::Builtin
                },
            );
        }
    }
    let mut entrypoint = a
        .entrypoint
        .clone()
        .or_else(|| run.entrypoint.as_ref().map(EntryDoc::typed));
    let mut inferred_target: Option<String> = None;
    if let Some(source) = a.model.take() {
        let id = match &entrypoint {
            Some(Entrypoint::Model(id)) => id.clone(),
            Some(Entrypoint::Pipeline(_)) => {
                return Err(bad("--model cannot override a graph entrypoint"));
            }
            None if selected.is_some() => {
                return Err(bad(
                    "config model override requires an explicit model entrypoint",
                ));
            }
            None => "main".into(),
        };
        let mut source = source;
        anchor_source(&mut source, cwd)?;
        if let Some(m) = graph.models.iter_mut().find(|m| m.id == id) {
            m.source = Some(source);
        } else if selected.is_some() {
            return Err(bad("model override entrypoint is absent from model table"));
        } else {
            graph.models.push(ModelInstance {
                id: id.clone(),
                target: String::new(),
                source: Some(source),
                model_config: None,
                tokenizer: None,
                snapshot: None,
                native_artifacts: None,
            });
            inferred_target = Some(id.clone());
        }
        entrypoint = Some(Entrypoint::Model(id.clone()));
        provenance.insert(format!("model.{id}.source"), override_origin.clone());
    }
    if let Some(patch) = a.graph.take() {
        let replaces_models = patch.models.is_some();
        patch.apply(&mut graph, &mut entrypoint)?;
        if replaces_models {
            provenance.retain(|key, _| !key.starts_with("model."));
            for model in &mut graph.models {
                if let Some(source) = &mut model.source {
                    anchor_source(source, cwd)?;
                }
                for path in [
                    &mut model.model_config,
                    &mut model.tokenizer,
                    &mut model.native_artifacts,
                ]
                .into_iter()
                .flatten()
                {
                    *path = absolute(cwd, path).to_string_lossy().into_owned();
                }
            }

            for m in &graph.models {
                for (field, supplied) in [
                    ("target", true),
                    ("source", true),
                    ("model_config", m.model_config.is_some()),
                    ("tokenizer", m.tokenizer.is_some()),
                    ("snapshot", m.snapshot.is_some()),
                    ("native_artifacts", m.native_artifacts.is_some()),
                ] {
                    provenance.insert(
                        format!("model.{}.{field}", m.id),
                        if supplied {
                            override_origin.clone()
                        } else {
                            Origin::Builtin
                        },
                    );
                }
            }
        }
        provenance.insert("graph".into(), override_origin.clone());
    }
    let entrypoint = entrypoint
        .ok_or_else(|| bad("run requires exactly one explicit model/pipeline entrypoint"))?;
    if a.model_config.is_some()
        || a.tokenizer.is_some()
        || a.snapshot.is_some()
        || a.native_artifacts.is_some()
    {
        let Entrypoint::Model(id) = &entrypoint else {
            return Err(bad(
                "model sidecars/snapshot require a direct model entrypoint",
            ));
        };
        let m = graph
            .models
            .iter_mut()
            .find(|m| &m.id == id)
            .ok_or_else(|| bad("entrypoint model absent"))?;
        for (field, value) in [
            ("model.model_config", a.model_config.take()),
            ("model.tokenizer", a.tokenizer.take()),
            ("model.native_artifacts", a.native_artifacts.take()),
        ] {
            if let Some(v) = value {
                let v = absolute(cwd, &v).to_string_lossy().into_owned();
                if field.ends_with("model_config") {
                    m.model_config = Some(v)
                } else if field.ends_with("tokenizer") {
                    m.tokenizer = Some(v)
                } else {
                    m.native_artifacts = Some(v)
                }
                provenance.insert(
                    format!("model.{id}.{}", field.trim_start_matches("model.")),
                    override_origin.clone(),
                );
            }
        }
        if let Some(s) = a.snapshot.take() {
            m.snapshot = Some(s);
            provenance.insert(format!("model.{id}.snapshot"), override_origin.clone());
        }
    }
    let config_task = run.task.unwrap_or_default();
    let mut task = Task::default();
    macro_rules! merge {
        ($field:ident) => {{
            task.$field = match a.task.$field.take() {
                Some(v) => {
                    provenance.insert(
                        concat!("task.", stringify!($field)).into(),
                        override_origin.clone(),
                    );
                    v
                }
                None => match config_task.$field {
                    Some(v) => {
                        provenance
                            .insert(concat!("task.", stringify!($field)).into(), origin.clone());
                        v
                    }
                    None => {
                        provenance
                            .insert(concat!("task.", stringify!($field)).into(), Origin::Builtin);
                        task.$field
                    }
                },
            };
        }};
    }
    // Optional values retain presence separately from their value.
    task.prompt = a.task.prompt.take().or(config_task.prompt);
    task.teacher_ids = a.task.teacher_ids.take().or(config_task.teacher_ids);
    merge!(context);
    merge!(max_tokens);
    merge!(raw);
    merge!(prefill_chunk);
    let config_diagnostics = run.diagnostics.unwrap_or_default();
    let diagnostics = Diagnostics {
        explain: a
            .diagnostics
            .explain
            .or(config_diagnostics.explain)
            .unwrap_or(false),
        dry_run: a
            .diagnostics
            .dry_run
            .or(config_diagnostics.dry_run)
            .unwrap_or(false),
        audit_logits_jsonl: a
            .diagnostics
            .audit_logits_jsonl
            .or(config_diagnostics.audit_logits_jsonl)
            .map(|s| absolute(cwd, &s)),
        audit_layers_jsonl: a
            .diagnostics
            .audit_layers_jsonl
            .or(config_diagnostics.audit_layers_jsonl)
            .map(|s| absolute(cwd, &s)),
    };
    let backend = match a.backend {
        Some(b) => {
            if let Some(config) = &run.backend {
                let _ = Backend::parse(config)?;
            }
            provenance.insert("backend".into(), override_origin.clone());
            b
        }
        None => {
            provenance.insert(
                "backend".into(),
                if run.backend.is_some() {
                    origin.clone()
                } else {
                    Origin::Builtin
                },
            );
            run.backend
                .as_deref()
                .map(Backend::parse)
                .transpose()?
                .unwrap_or(Backend::Auto)
        }
    };
    let device = a.devices.or(run.device);
    let devices = device
        .as_deref()
        .map(Devices::parse)
        .transpose()?
        .unwrap_or(Devices::Auto);
    let headroom = a
        .headroom
        .or(run.headroom.as_deref().map(Headroom::parse).transpose()?)
        .unwrap_or(Headroom::Default);
    let speculation = a
        .speculation
        .or(run
            .speculation
            .as_deref()
            .map(Speculation::parse)
            .transpose()?)
        .unwrap_or(Speculation::Auto);
    let draft_device = a
        .draft_device
        .or(run.draft_device)
        .as_deref()
        .map(Devices::parse)
        .transpose()?;
    for m in &graph.models {
        for field in [
            "target",
            "source",
            "model_config",
            "tokenizer",
            "snapshot",
            "native_artifacts",
        ] {
            provenance
                .entry(format!("model.{}.{field}", m.id))
                .or_insert(Origin::Builtin);
        }
    }
    let request = Request {
        graph,
        entrypoint,
        task,
        backend,
        devices,
        headroom,
        numerical: a.numerical.or(run.numerical),
        speculation,
        draft_device,
        diagnostics,
        provenance,
        vendor_filter: a.vendor,
    };
    request.validate_policy()?;
    // A direct synthetic source acquires its target only from parsed source
    // geometry. Every declared model target validates now.
    let pending = request
        .graph
        .models
        .iter()
        .filter(|m| m.target.is_empty())
        .map(|m| m.id.as_str())
        .collect::<Vec<_>>();
    match pending.as_slice() {
        [] => {
            request.graph.validate_entrypoint(&request.entrypoint)?;
        }
        [id] if Some(*id) == inferred_target.as_deref() => {}
        _ => return Err(bad("declared model target must be nonempty")),
    }
    Ok(request)
}
fn anchor_source(s: &mut String, base: &Path) -> Result<()> {
    if let SourceSelector::Local(p) = SourceSelector::parse(s)? {
        *s = absolute(
            base,
            p.to_str().ok_or_else(|| bad("source path must be UTF-8"))?,
        )
        .to_string_lossy()
        .into_owned();
    }
    Ok(())
}

fn absolute(base: &Path, s: &str) -> PathBuf {
    let p = Path::new(s);
    if p.is_absolute() {
        p.into()
    } else {
        base.join(p)
    }
}
fn bad(s: impl Into<String>) -> Error {
    Error::Malformed(s.into())
}
