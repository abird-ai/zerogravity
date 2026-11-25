//! Native presentation of the shared cold configuration owner.
use super::library::{Key, Row};
use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};
use zerogravity::{
    Error, Result,
    configuration::{self as cfg, Change, ConfigDocument, ConfigFile, INITIAL},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Target {
    pub path: PathBuf,
    pub profile: Option<String>,
    pub allow_missing: bool,
}
impl Target {
    pub fn selected(selection: Option<(&str, &str)>) -> Result<Self> {
        Ok(Self {
            path: crate::config_cli::selected_path(selection)?,
            profile: match selection {
                None => Some("default".into()),
                Some(("--profile", name)) => Some(name.into()),
                _ => None,
            },
            allow_missing: selection.is_none(),
        })
    }
    pub fn profile(name: &str) -> Result<Self> {
        Self::selected(if name == "default" {
            None
        } else {
            Some(("--profile", name))
        })
    }
    pub fn label(&self) -> String {
        self.profile
            .clone()
            .unwrap_or_else(|| self.path.display().to_string())
    }
    fn open(&self) -> Result<ConfigFile> {
        let file = ConfigFile::open(self.path.clone())?;
        if !self.allow_missing {
            file.require_existing()?;
        }
        Ok(file)
    }
    pub fn read(&self) -> Result<Option<(PathBuf, ConfigDocument)>> {
        cfg::load_path(self.path.clone(), self.allow_missing)
    }
}

#[derive(Clone)]
pub(super) struct Snapshot {
    pub target: Target,
    file: Arc<ConfigFile>,
}
impl fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Snapshot")
            .field("target", &self.target)
            .field("exists", &self.file.exists())
            .finish_non_exhaustive()
    }
}
impl PartialEq for Snapshot {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target
            && self.file.exists() == other.file.exists()
            && self.file.bytes() == other.file.bytes()
    }
}
impl Eq for Snapshot {}
impl Snapshot {
    fn open(target: Target) -> Result<Self> {
        Ok(Self {
            file: Arc::new(target.open()?),
            target,
        })
    }
    pub fn exists(&self) -> bool {
        self.file.exists()
    }
    pub fn raw(&self) -> Result<&str> {
        std::str::from_utf8(self.file.bytes()).map_err(|_| {
            Error::Malformed("Configuration is not UTF-8; use config edit to repair".into())
        })
    }
    fn require_current(&self) -> Result<()> {
        if Self::open(self.target.clone())? != *self {
            return Err(Error::State(
                "configuration changed since it was read; reload before saving".into(),
            ));
        }
        Ok(())
    }
}

pub(super) struct Observed {
    pub snapshot: Snapshot,
    pub rows: Vec<Row>,
    pub values: BTreeMap<String, String>,
    pub warning: Option<String>,
}

// Descriptions of the existing typed schema, not new policy or defaults.
fn help(key: &str) -> &'static str {
    match key {
        "run.backend" => "auto, cpu, cuda, rocm or vulkan; original qualification still applies.",
        "run.device" => "Ordered physical device selectors, or [auto]. CPU is terminal.",
        "run.headroom" => "Memory reserve: bytes/size or percentage; unset for automatic policy.",
        "run.numerical" => "source or native; native requires the qualified ROCm route.",
        "run.speculation" => {
            "auto, off, mtp or draft:MODEL; adapter eligibility is checked separately."
        }
        "run.draft_device" => "Explicit ordered draft device selectors.",
        "run.entrypoint" => "Mapping: {model: ID} or {pipeline: NAME}.",
        "run.task.context" => {
            "Requested context tokens; existing model and scheduler bounds apply."
        }
        "run.task.max_tokens" => "Maximum generated tokens per direct run.",
        "run.task.raw" => "true uses raw text instead of a chat template.",
        "run.task.prefill_chunk" => "Requested prefill chunk size.",
        "run.task.prompt" => "Optional example prompt for direct run; not an HTTP request.",
        "run.task.teacher_ids" => "Explicit token ID sequence for supported diagnostics.",
        "run.diagnostics.explain" => "Show the resolved execution decision.",
        "run.diagnostics.dry_run" => "Inspect decision without execution.",
        "run.diagnostics.audit_logits_jsonl" | "run.diagnostics.audit_layers_jsonl" => {
            "Explicit diagnostic output path; supported direct diagnostic routes only."
        }
        "serve.listen" => "Numeric loopback IP:PORT. Port zero requests an ephemeral port.",
        "serve.workers" => "Worker count; unset for the selected backend default.",
        "serve.active_per_worker" => "Concurrent admitted rows per worker.",
        "serve.max_context" => "Serving context capacity; may inherit run.task.context.",
        "serve.max_output" => "Serving output capacity; may inherit run.task.max_tokens.",
        "serve.host_mib" => "Host admission envelope in MiB, not a memory availability claim.",
        "serve.max_pending_requests" => {
            "Optional finite queue count; host admission still applies."
        }
        "serve.enable_per_request_metrics" => "Enable supported per-request inference metrics.",
        "ui.color" | "ui.progress" | "ui.interactive" => {
            "auto, always or never; explicit CLI presentation preferences win."
        }
        "downloads.ask_before_pull" => "Offer preference, not consent or network authority.",
        "model" | "pipeline" | "route" => {
            "Whole typed YAML list; existing graph/source validation remains authoritative."
        }
        _ => "Existing typed document setting; unset to inherit its default.",
    }
}

fn lookup<'a>(node: &'a serde_yaml::Value, key: &str) -> Option<&'a serde_yaml::Value> {
    key.split('.').try_fold(node, |node, part| node.get(part))
}
fn preview(value: &str) -> String {
    let flat = value.replace('\n', " ");
    let mut result: String = flat.chars().take(100).collect();
    if flat.chars().count() > 100 {
        result.push('…');
    }
    result
}
pub(super) fn observe(target: Target) -> Result<Observed> {
    let snapshot = Snapshot::open(target)?;
    let mut result = Observed {
        snapshot,
        rows: Vec::new(),
        values: BTreeMap::new(),
        warning: None,
    };
    let file = &result.snapshot.file;
    let settings = (|| -> Result<_> {
        Ok((
            cfg::document_value(file.path(), file.bytes())?,
            cfg::explain(file.path(), file.bytes())?,
        ))
    })();
    let (node, mut effective) = match settings {
        Ok(pair) => pair,
        Err(error) => {
            result.warning = Some(format!(
                "Invalid document: {error}; raw YAML repair is available"
            ));
            return Ok(result);
        }
    };
    // Optional fields omitted by explain remain discoverable. Their
    // effective value is deliberately not invented.
    for key in [
        "run.entrypoint",
        "run.draft_device",
        "run.task.prompt",
        "run.task.teacher_ids",
        "run.diagnostics.audit_logits_jsonl",
        "run.diagnostics.audit_layers_jsonl",
    ] {
        effective.entry(key.into()).or_insert_with(|| cfg::Setting {
            value: serde_yaml::Value::Null,
            origin: "unset".into(),
        });
    }
    for (key, setting) in effective {
        let explicit = lookup(&node, &key);
        let value = if let Some(value) = explicit {
            let raw = serde_yaml::to_string(value)
                .map_err(|_| Error::State("setting encoding".into()))?;
            result.values.insert(key.clone(), raw.clone());
            raw
        } else {
            serde_yaml::to_string(&setting.value)
                .map_err(|_| Error::State("setting encoding".into()))?
        };
        result.rows.push(Row {
            key: Key::Setting(key.clone()),
            title: format!("{key} = {}", preview(value.trim_end())),
            details: vec![
                key,
                format!("Value: {}", preview(value.trim_end())),
                format!("Origin: {}", setting.origin),
                if explicit.is_some() {
                    "Explicit value".into()
                } else {
                    "Inherited / built-in".into()
                },
            ],
        });
        let row = result.rows.last_mut().expect("added row");
        if let Key::Setting(key) = &row.key {
            row.details.push(help(key).into());
        }
    }
    result.rows.sort_by(|a, b| match (&a.key, &b.key) {
        (Key::Setting(a), Key::Setting(b)) => group(a).cmp(group(b)).then_with(|| a.cmp(b)),
        _ => std::cmp::Ordering::Equal,
    });
    Ok(result)
}

pub(super) fn group(key: &str) -> &str {
    key.rsplit_once('.').map_or("graph", |(group, _)| group)
}

pub(super) fn profiles() -> Result<Vec<Row>> {
    let root = Target::profile("default")?;
    let names = cfg::profiles(root.path.parent().expect("configuration parent"))?;
    names
        .into_iter()
        .map(|name| {
            let target = Target::profile(&name)?;
            Ok(Row {
                key: Key::Profile(name.clone()),
                title: if name == "default" {
                    "default · config.yaml".into()
                } else {
                    format!("{name} · profiles/{name}.yaml")
                },
                details: vec![
                    target.path.display().to_string(),
                    if target.path.is_file() {
                        "Stored document".into()
                    } else {
                        "Built-in defaults (no document created)".into()
                    },
                    "Enter opens editor; activation is explicit.".into(),
                ],
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Action {
    Create(String),
    Copy {
        source: Snapshot,
        name: String,
    },
    Remove(Snapshot),
    Activate(Snapshot),
    Validate(Snapshot),
    Set {
        file: Snapshot,
        key: String,
        value: String,
    },
    Unset {
        file: Snapshot,
        key: String,
    },
    Replace {
        file: Snapshot,
        yaml: String,
    },
}
impl Action {
    pub fn title(&self) -> &'static str {
        match self {
            Self::Create(_) => "Create profile",
            Self::Copy { .. } => "Copy profile",
            Self::Remove(_) => "Remove profile",
            Self::Activate(_) => "Activate profile",
            Self::Validate(_) => "Validate config",
            Self::Set { .. } | Self::Unset { .. } | Self::Replace { .. } => "Save config",
        }
    }
}
pub(super) struct Outcome {
    pub message: String,
    pub editor: Option<Target>,
    pub active: Option<(Target, ConfigDocument)>,
}
impl Outcome {
    fn message(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            editor: None,
            active: None,
        }
    }
}
fn new_profile(name: &str, bytes: &[u8], source: Option<&Path>) -> Result<Outcome> {
    if name == "default" {
        return Err(Error::Malformed(
            "default is config.yaml; edit it instead".into(),
        ));
    }
    let target = Target::profile(name)?;
    let file = ConfigFile::open(target.path.clone())?;
    let copied;
    let bytes = if let Some(source) = source {
        copied = cfg::copy_document(source, file.path(), bytes)?;
        &copied
    } else {
        bytes
    };
    file.create(bytes)?;
    let mut outcome = Outcome::message(format!("Created profile {name}"));
    outcome.editor = Some(target);
    Ok(outcome)
}
fn save(file: Snapshot, bytes: Vec<u8>, active: &Target) -> Result<Outcome> {
    let doc = if file.target.path == active.path {
        Some(zerogravity::pipelines::parse_document(&bytes)?)
    } else {
        None
    };
    file.file.replace(&bytes)?;
    let mut outcome = Outcome::message("Saved configuration");
    if let Some(doc) = doc {
        outcome.active = Some((active.clone(), doc));
    }
    Ok(outcome)
}
pub(super) fn perform(action: Action, active: &Target) -> Result<Outcome> {
    match action {
        Action::Create(name) => new_profile(&name, INITIAL.as_bytes(), None),
        Action::Copy { source, name } => {
            source.require_current()?;
            new_profile(&name, source.file.bytes(), Some(source.file.path()))
        }
        Action::Remove(file) => {
            if file
                .target
                .profile
                .as_deref()
                .is_none_or(|name| name == "default")
                || file.target.path == active.path
            {
                return Err(Error::Unsupported("Select another active profile before removing a named profile; default and portable files are protected".into()));
            }
            file.file.remove()?;
            let mut outcome = Outcome::message("Removed profile");
            outcome.editor = Some(active.clone());
            Ok(outcome)
        }
        Action::Activate(file) => {
            file.require_current()?;
            let doc = cfg::validate(file.file.path(), file.file.bytes())?;
            let mut outcome = Outcome::message(format!(
                "Activated {} for future actions",
                file.target.label()
            ));
            outcome.active = Some((file.target, doc));
            Ok(outcome)
        }
        Action::Validate(file) => {
            file.require_current()?;
            cfg::validate(file.file.path(), file.file.bytes())?;
            Ok(Outcome::message("Valid configuration (cold policy only)"))
        }
        Action::Set { file, key, value } => {
            let bytes = cfg::changed(
                file.file.path(),
                file.file.bytes(),
                Change::Set {
                    path: &key,
                    yaml: &value,
                },
            )?;
            save(file, bytes, active)
        }
        Action::Unset { file, key } => {
            let bytes = cfg::changed(
                file.file.path(),
                file.file.bytes(),
                Change::Unset { path: &key },
            )?;
            save(file, bytes, active)
        }
        Action::Replace { file, yaml } => save(file, yaml.into_bytes(), active),
    }
}
