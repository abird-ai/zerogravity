use super::*;
use std::ffi::OsString;
use zerogravity::pipeline::{Config, ModelInstance};

pub(super) fn resolve(
    mut a: Arguments,
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
    let mut provenance = BTreeMap::new();
    let (mut graph, run) = if let Some(path) = &selected {
        if path.extension().is_some_and(|e| e == "json") {
            return Err(bad(
                "--config selects pipeline YAML; old model metadata moved to --model-config",
            ));
        }
        let mut doc = crate::pipeline::load_document(
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
                    Origin::Cli
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
        provenance.insert(format!("model.{id}.source"), Origin::Cli);
    }
    if !a.graph.is_empty() {
        apply_graph(&mut graph, &a.graph, &mut entrypoint, cwd)?;
        if a.graph.iter().any(|s| s == "--define-model") {
            for m in &graph.models {
                for field in ["target", "source"] {
                    provenance.insert(format!("model.{}.{field}", m.id), Origin::Cli);
                }
            }
        }
        provenance.insert("graph".into(), Origin::Cli);
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
                    Origin::Cli,
                );
            }
        }
        if let Some(s) = a.snapshot.take() {
            m.snapshot = Some(s);
            provenance.insert(format!("model.{id}.snapshot"), Origin::Cli);
        }
    }
    let config_task = run.task.unwrap_or_default();
    let mut task = Task::default();
    macro_rules! merge {
        ($field:ident) => {{
            task.$field = match a.task.$field.take() {
                Some(v) => {
                    provenance.insert(concat!("task.", stringify!($field)).into(), Origin::Cli);
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
            provenance.insert("backend".into(), Origin::Cli);
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

fn apply_graph(
    graph: &mut Config,
    args: &[String],
    entry: &mut Option<Entrypoint>,
    cwd: &Path,
) -> Result<()> {
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut it = args.iter();
    while let Some(f) = it.next() {
        let v = if f == "--realtime" {
            "true".into()
        } else {
            it.next()
                .ok_or_else(|| bad("graph flag requires value"))?
                .clone()
        };
        groups
            .entry(if f == "--realtime-value" {
                "--realtime".into()
            } else {
                f.clone()
            })
            .or_default()
            .push(v);
    }
    for (flag, values) in &groups {
        let empty = values.iter().filter(|s| *s == "[]").count();
        if empty == 0 {
            continue;
        }
        // `[]` is an explicit whole-list clear only for graph list fields.
        if !matches!(
            flag.as_str(),
            "--define-model"
                | "--step"
                | "--link"
                | "--route"
                | "--chat-context-source"
                | "--chat-context-match"
                | "--chat-context-decision"
        ) {
            return Err(bad(format!(
                "empty list spelling requires a graph list field; {flag} needs a value"
            )));
        }
        if values.len() != 1 {
            return Err(bad("empty list cannot mix with list members"));
        }
    }
    let name = groups
        .get("--name")
        .and_then(|v| v.first())
        .cloned()
        .or_else(|| match entry {
            Some(Entrypoint::Pipeline(n)) => Some(n.clone()),
            _ => None,
        })
        .ok_or_else(|| bad("graph flags require --name or a pipeline entrypoint"))?;
    match entry {
        Some(Entrypoint::Model(_)) => {
            return Err(bad("graph flags contradict a direct model entrypoint"));
        }
        Some(Entrypoint::Pipeline(selected)) if selected != &name => {
            return Err(bad("--name contradicts selected pipeline entrypoint"));
        }
        _ => {}
    }
    let old = graph.pipelines.iter().find(|p| p.name == name).cloned();
    let mut flags = vec![
        "--name".into(),
        name.clone(),
        "--input".into(),
        groups
            .get("--input")
            .and_then(|v| v.first())
            .cloned()
            .or_else(|| old.as_ref().map(|p| p.input.label().into()))
            .ok_or_else(|| bad("graph input required"))?,
        "--output".into(),
        groups
            .get("--output")
            .and_then(|v| v.first())
            .cloned()
            .or_else(|| old.as_ref().map(|p| p.output.label().into()))
            .ok_or_else(|| bad("graph output required"))?,
    ];
    if groups.keys().any(|k| k.starts_with("--chat-context-"))
        && !groups.contains_key("--chat-context-from")
        && let Some(context) = old.as_ref().and_then(|p| p.chat_context.as_ref())
    {
        flags.extend(["--chat-context-from".into(), context.from_model.clone()]);
    }
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let v = it.next().ok_or_else(|| bad("graph flag value"))?;
        let flag = if flag == "--realtime-value" {
            "--realtime"
        } else {
            flag.as_str()
        };
        if matches!(flag, "--name" | "--input" | "--output") || v == "[]" {
            continue;
        }
        if flag == "--realtime" {
            if v == "true" {
                flags.push(flag.into());
            }
            continue;
        }
        flags.push(if flag == "--define-model" {
            "--model".into()
        } else {
            flag.into()
        });
        flags.push(v.clone());
    }
    let mut compiled = crate::pipeline::flags(&flags)?;
    let mut new = compiled.pipelines.remove(0);
    if let Some(old) = old {
        if !groups.contains_key("--step") {
            new.steps = old.steps;
        }
        if !groups.contains_key("--link") {
            new.links = old.links;
        }
        if !groups.contains_key("--realtime") {
            new.realtime = old.realtime;
        }
        if !groups.keys().any(|k| k.starts_with("--chat-context-")) {
            new.chat_context = old.chat_context;
        } else if let (Some(previous), Some(current)) =
            (old.chat_context, new.chat_context.as_mut())
        {
            if !groups.contains_key("--chat-context-enabled") {
                current.enabled = previous.enabled;
            }
            if !groups.contains_key("--chat-context-max-bytes") {
                current.max_bytes = previous.max_bytes;
            }
            if !groups.contains_key("--chat-context-source") {
                current.selection.sources = previous.selection.sources;
            } else if groups
                .get("--chat-context-source")
                .is_some_and(|v| v == &["[]"])
            {
                current.selection.sources.clear();
            }
            if !groups.contains_key("--chat-context-match")
                && !groups.contains_key("--chat-context-decision")
            {
                current.selection.stages = previous.selection.stages;
            }
        }
    }
    if groups.contains_key("--define-model") {
        graph.models = compiled.models;
        for m in &mut graph.models {
            if let Some(s) = &mut m.source {
                anchor_source(s, cwd)?;
            }
        }
    }
    if groups.contains_key("--route") {
        graph.routes = compiled.routes;
    }
    if let Some(i) = graph.pipelines.iter().position(|p| p.name == name) {
        graph.pipelines[i] = new;
    } else {
        graph.pipelines.push(new);
    }
    *entry = Some(Entrypoint::Pipeline(name));
    Ok(())
}
