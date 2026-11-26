//! Shared cold configuration actions and typed observations; no second store.
use super::Target;
use std::{fmt, path::Path, sync::Arc};
use zerogravity::{
    Error, Result,
    configuration::{self as cfg, Change, ConfigDocument, ConfigFile, INITIAL},
};

#[derive(Clone)]
pub(crate) struct Snapshot {
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
    pub(crate) fn open(target: Target) -> Result<Self> {
        Ok(Self {
            file: Arc::new(target.open()?),
            target,
        })
    }
    pub fn exists(&self) -> bool {
        self.file.exists()
    }
    #[cfg(feature = "tui")]
    pub fn raw(&self) -> Result<&str> {
        std::str::from_utf8(self.file.bytes()).map_err(|_| {
            Error::Malformed("Configuration is not UTF-8; use config edit to repair".into())
        })
    }
    #[cfg(feature = "web")]
    pub fn node(&self) -> Result<serde_yaml::Value> {
        cfg::document_value(self.file.path(), self.file.bytes())
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

pub(crate) struct SettingRow {
    pub key: String,
    pub value: serde_yaml::Value,
    pub explicit: bool,
    pub origin: String,
    pub help: &'static str,
}
pub(crate) struct Observed {
    pub snapshot: Snapshot,
    pub rows: Vec<SettingRow>,
    pub warning: Option<String>,
    #[cfg(feature = "web")]
    pub node: Option<serde_yaml::Value>,
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
        "ui.mouse" => {
            "Native mouse capture: auto/always on a capable terminal, never disables it; F8 toggles this session."
        }
        "ui.theme" => "Auto, dark or light appearance; no execution policy changes.",
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

pub(crate) fn observe(target: Target) -> Result<Observed> {
    let snapshot = Snapshot::open(target)?;
    let file = &snapshot.file;
    let (node, mut effective) = match (|| -> Result<_> {
        Ok((
            cfg::document_value(file.path(), file.bytes())?,
            cfg::explain(file.path(), file.bytes())?,
        ))
    })() {
        Ok(pair) => pair,
        Err(error) => {
            return Ok(Observed {
                snapshot,
                rows: Vec::new(),
                #[cfg(feature = "web")]
                node: None,
                warning: Some(format!(
                    "Invalid document: {error}; raw YAML repair is available"
                )),
            });
        }
    };
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
    let mut rows = Vec::with_capacity(effective.len());
    for (key, setting) in effective {
        let explicit = lookup(&node, &key);
        rows.push(SettingRow {
            help: help(&key),
            value: explicit.cloned().unwrap_or(setting.value),
            explicit: explicit.is_some(),
            origin: setting.origin,
            key,
        });
    }
    rows.sort_by(|a, b| {
        group(&a.key)
            .cmp(group(&b.key))
            .then_with(|| a.key.cmp(&b.key))
    });
    Ok(Observed {
        snapshot,
        rows,
        warning: None,
        #[cfg(feature = "web")]
        node: Some(node),
    })
}
pub(crate) fn group(key: &str) -> &str {
    if key.starts_with("ui.") {
        "Appearance"
    } else {
        key.rsplit_once('.').map_or("graph", |(group, _)| group)
    }
}

pub(crate) fn profiles() -> Result<Vec<String>> {
    let root = Target::profile("default")?;
    cfg::profiles(root.path.parent().expect("configuration parent"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Action {
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
pub(crate) struct Outcome {
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
pub(crate) fn perform(action: Action, active: &Target) -> Result<Outcome> {
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
