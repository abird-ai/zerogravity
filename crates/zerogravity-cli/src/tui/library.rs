//! Presentation-only model/device rows and a single coalesced read worker.
//! Selection is an exact observed identity, not a cache lease or admission.
use super::settings::{self, Observed, Snapshot, Target};
use std::thread::{self, JoinHandle};
use zerogravity::{Error, Result, Runtime, control::ControlPlane};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Key {
    Snapshot(String),
    Target(String),
    #[cfg_attr(
        not(all(feature = "cache", target_os = "linux")),
        allow(dead_code, reason = "checkpoint inventory is optional")
    )]
    Partial(String),
    Device(String),
    Profile(String),
    Setting(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Shelf {
    #[default]
    Cached,
    Catalog,
    Partial,
}
impl Shelf {
    pub fn next(self) -> Self {
        match self {
            Self::Cached => Self::Catalog,
            Self::Catalog => Self::Partial,
            Self::Partial => Self::Cached,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Models,
    Devices,
    Profiles,
    Config,
}

#[derive(Clone)]
pub(super) struct Row {
    pub key: Key,
    pub title: String,
    pub details: Vec<String>,
}

#[derive(Default)]
pub(super) struct Rows {
    pub rows: Vec<Row>,
    pub selected: Option<Key>,
}
impl Rows {
    fn replace(&mut self, rows: Vec<Row>) {
        self.rows = rows;
        if !self
            .rows
            .iter()
            .any(|row| self.selected.as_ref() == Some(&row.key))
        {
            self.selected = self.rows.first().map(|row| row.key.clone());
        }
    }
    pub fn selected(&self) -> Option<&Row> {
        self.rows
            .iter()
            .find(|row| self.selected.as_ref() == Some(&row.key))
    }
    pub fn index(&self) -> usize {
        self.rows
            .iter()
            .position(|row| self.selected.as_ref() == Some(&row.key))
            .unwrap_or(0)
    }
    pub fn move_selection(&mut self, delta: i8) -> usize {
        let index = self.index();
        let next = if delta < 0 {
            index.saturating_sub(1)
        } else {
            (index + 1).min(self.rows.len().saturating_sub(1))
        };
        self.selected = self.rows.get(next).map(|row| row.key.clone());
        next
    }
}

#[derive(Default)]
struct Models {
    cached: Vec<Row>,
    partial: Vec<Row>,
    warning: Option<String>,
}
enum Observation {
    Models(Models),
    Devices(Vec<Row>),
    Profiles(Vec<Row>),
    Config(Observed),
}

struct Reader {
    kind: Kind,
    target: Target,
    thread: Option<JoinHandle<Result<Observation>>>,
}
impl Reader {
    fn start(runtime: &Runtime, kind: Kind, target: Target) -> Result<Self> {
        let runtime = runtime.clone();
        let editor = target.clone();
        let worker = thread::Builder::new()
            .name("zg-control-observe".into())
            .spawn(move || {
                let control = runtime.control();
                match kind {
                    Kind::Models => Ok(Observation::Models(models(control))),
                    Kind::Devices => devices(control).map(Observation::Devices),
                    Kind::Profiles => settings::profiles().map(Observation::Profiles),
                    Kind::Config => settings::observe(editor).map(Observation::Config),
                }
            })?;
        Ok(Self {
            kind,
            target,
            thread: Some(worker),
        })
    }
    fn finished(&self) -> bool {
        self.thread
            .as_ref()
            .is_some_and(|thread| thread.is_finished())
    }
    fn join(&mut self) -> Result<Observation> {
        self.thread
            .take()
            .expect("owned reader")
            .join()
            .map_err(|_| Error::State("inventory worker panicked".into()))?
    }
}
impl Drop for Reader {
    fn drop(&mut self) {
        if self.thread.is_some() {
            let _ = self.join();
        }
    }
}

pub(super) struct Library {
    pub shelf: Shelf,
    cached: Rows,
    catalog: Rows,
    partial: Rows,
    devices: Rows,
    profiles: Rows,
    config: Rows,
    pub editor: Target,
    observed: Option<Observed>,
    pub query: String,
    pub raw: bool,
    profile_notice: Option<String>,
    config_notice: Option<String>,
    model_notice: Option<String>,
    device_notice: Option<String>,
    pub detail: Option<(Key, String)>,
    reader: Option<Reader>,
    pending: Option<Kind>,
}

impl Library {
    pub fn new(editor: Target) -> Self {
        let mut catalog = Rows::default();
        catalog.replace(
            zerogravity::targets::MODEL_TARGETS
                .iter()
                .map(|target| Row {
                    key: Key::Target(target.slug.into()),
                    title: format!(
                        "{} · {} · {}",
                        target.slug,
                        target.task.label(),
                        target.source_status.label()
                    ),
                    details: vec![
                        format!("Repository: {}", target.repository.unwrap_or("unverified")),
                        format!("Source layout: {}", target.source_layout.label()),
                        format!("Execution: {}", target.execution_status.label()),
                        if target.payload_hold {
                            "Payload hold; no catalog acquisition.".into()
                        } else {
                            target.note.into()
                        },
                    ],
                })
                .collect(),
        );
        Self {
            shelf: Shelf::Cached,
            cached: Rows::default(),
            catalog,
            partial: Rows::default(),
            devices: Rows::default(),
            profiles: Rows::default(),
            config: Rows::default(),
            editor,
            observed: None,
            query: String::new(),
            raw: false,
            profile_notice: None,
            config_notice: None,
            model_notice: None,
            device_notice: None,
            detail: None,
            reader: None,
            pending: None,
        }
    }
}
impl Library {
    pub fn set_editor(&mut self, target: Target) {
        self.editor = target;
        self.observed = None;
        self.config.replace(Vec::new());
        self.config_notice = None;
        self.query.clear();
        self.raw = false;
    }
    pub fn invalidate_config(&mut self) {
        self.observed = None;
        self.config.rows.clear();
        self.config_notice = None;
    }
    pub fn snapshot(&self) -> Option<Snapshot> {
        self.observed
            .as_ref()
            .map(|observed| observed.snapshot.clone())
    }
    pub fn observed(&self) -> Option<&Observed> {
        self.observed.as_ref()
    }
    pub fn value(&self, key: &str) -> Option<&str> {
        self.observed.as_ref()?.values.get(key).map(String::as_str)
    }
    pub fn filter_config(&mut self) {
        let query = self.query.to_ascii_lowercase();
        self.config
            .replace(self.observed.as_ref().map_or_else(Vec::new, |observed| {
                observed
                    .rows
                    .iter()
                    .filter(|row| {
                        query.is_empty()
                            || row.title.to_ascii_lowercase().contains(&query)
                            || row
                                .details
                                .iter()
                                .any(|detail| detail.to_ascii_lowercase().contains(&query))
                    })
                    .cloned()
                    .collect()
            }));
    }
    pub fn rows(&self, kind: Kind) -> &Rows {
        match (kind, self.shelf) {
            (Kind::Models, Shelf::Cached) => &self.cached,
            (Kind::Models, Shelf::Catalog) => &self.catalog,
            (Kind::Models, Shelf::Partial) => &self.partial,
            (Kind::Devices, _) => &self.devices,
            (Kind::Profiles, _) => &self.profiles,
            (Kind::Config, _) => &self.config,
        }
    }
    pub fn rows_mut(&mut self, kind: Kind) -> &mut Rows {
        match (kind, self.shelf) {
            (Kind::Models, Shelf::Cached) => &mut self.cached,
            (Kind::Models, Shelf::Catalog) => &mut self.catalog,
            (Kind::Models, Shelf::Partial) => &mut self.partial,
            (Kind::Devices, _) => &mut self.devices,
            (Kind::Profiles, _) => &mut self.profiles,
            (Kind::Config, _) => &mut self.config,
        }
    }
    pub fn notice(&self, kind: Kind) -> Option<&str> {
        if self
            .reader
            .as_ref()
            .is_some_and(|reader| reader.kind == kind)
            || self.pending == Some(kind)
        {
            return Some("Refreshing observations…");
        }
        match kind {
            Kind::Models => self.model_notice.as_deref(),
            Kind::Devices => self.device_notice.as_deref(),
            Kind::Profiles => self.profile_notice.as_deref(),
            Kind::Config => self.config_notice.as_deref(),
        }
    }
    /// Coalesce observation invalidation; never queue an effect or open a
    /// second cache owner. Only one read worker exists regardless of refreshes.
    pub fn request(&mut self, runtime: &Runtime, kind: Kind) {
        if self.reader.is_some() {
            self.pending = Some(kind);
        } else {
            match Reader::start(runtime, kind, self.editor.clone()) {
                Ok(reader) => self.reader = Some(reader),
                Err(error) => self.error(kind, error),
            }
        }
    }
    fn error(&mut self, kind: Kind, error: Error) {
        let notice = Some(format!("Observation unavailable: {error}"));
        match kind {
            Kind::Models => self.model_notice = notice,
            Kind::Devices => self.device_notice = notice,
            Kind::Profiles => self.profile_notice = notice,
            Kind::Config => {
                self.config_notice = notice;
                self.observed = None;
                self.config.replace(Vec::new());
            }
        }
    }
    pub fn poll(&mut self, runtime: &Runtime) -> bool {
        if !self.reader.as_ref().is_some_and(Reader::finished) {
            return false;
        }
        let mut reader = self.reader.take().expect("finished observation");
        match reader.join() {
            Ok(Observation::Models(models)) => {
                self.cached.replace(models.cached);
                self.partial.replace(models.partial);
                self.model_notice = models.warning;
            }
            Ok(Observation::Devices(rows)) => {
                self.devices.replace(rows);
                self.device_notice = None;
            }
            Ok(Observation::Profiles(rows)) => {
                self.profiles.replace(rows);
                self.profile_notice = None;
            }
            Ok(Observation::Config(observed)) => {
                if observed.snapshot.target == self.editor {
                    self.config_notice = observed.warning.clone();
                    self.observed = Some(observed);
                    self.filter_config();
                }
            }
            Err(error) if reader.kind != Kind::Config || reader.target == self.editor => {
                self.error(reader.kind, error)
            }
            Err(_) => {}
        }
        if let Some(kind) = self.pending.take() {
            self.request(runtime, kind);
        }
        true
    }
    pub fn join(&mut self) -> Result<()> {
        self.pending = None;
        if let Some(mut reader) = self.reader.take() {
            reader.join().map(|_| ())
        } else {
            Ok(())
        }
    }
}

fn models(control: ControlPlane<'_>) -> Models {
    #[cfg(all(feature = "cache", target_os = "linux"))]
    {
        let result = (|| {
            let observation = crate::control_host::inventory::cache(control)?;
            let cached = observation
                .cached
                .into_iter()
                .map(|model| Row {
                    key: Key::Snapshot(model.snapshot.clone()),
                    title: format!("{} · {} bytes", model.repository, model.bytes),
                    details: vec![
                        format!("Snapshot: sha256:{}", model.snapshot),
                        format!("{}:{}@{}", model.provider, model.repository, model.revision),
                        format!("{} files · {} bytes", model.files, model.bytes),
                        "Metadata only; payloads and execution not verified.".into(),
                    ],
                })
                .collect();
            // A malformed/contended checkpoint record must not hide good
            // canonical snapshots or the static intent catalog.
            let (partial, warning) = match observation.partial {
                Ok(partials) => (
                    partials
                        .into_iter()
                        .map(|partial| Row {
                            key: Key::Partial(partial.key.clone()),
                            title: format!("{} / {} bytes", partial.bytes, partial.total),
                            details: vec![
                                partial.key,
                                "Checkpoint observation only; may be active or invalid.".into(),
                                "Repeat the original pinned pull intent to revalidate and resume."
                                    .into(),
                            ],
                        })
                        .collect(),
                    None,
                ),
                Err(error) => (
                    Vec::new(),
                    Some(format!("Partial inventory unavailable: {error}")),
                ),
            };
            Ok(Models {
                cached,
                partial,
                warning,
            })
        })();
        result.unwrap_or_else(|error: Error| Models {
            warning: Some(format!("Cache observation unavailable: {error}")),
            ..Default::default()
        })
    }
    #[cfg(not(all(feature = "cache", target_os = "linux")))]
    {
        let _ = control;
        Models {
            warning: Some("Cache inspection is unavailable in this build.".into()),
            ..Default::default()
        }
    }
}

fn devices(control: ControlPlane<'_>) -> Result<Vec<Row>> {
    let inventory = control.devices()?;
    Ok(inventory
        .devices
        .into_iter()
        .map(|device| Row {
            key: Key::Device(device.id.clone()),
            title: format!("{} · {} · {}", device.id, device.class, device.vendor),
            details: vec![
                format!("Observed: {} (Unix ms)", inventory.observed_unix_ms),
                format!("Backends: {}", device.backends.join(", ")),
                format!(
                    "Capacity: {}",
                    device
                        .ceiling
                        .map_or_else(|| "unknown".into(), |bytes| format!("{bytes} bytes"))
                ),
                format!(
                    "Available: {}",
                    device
                        .available
                        .map_or_else(|| "unknown".into(), |bytes| format!("{bytes} bytes"))
                ),
                format!("Observation: {}", device.provenance),
                device.exclusion.map_or_else(
                    || "No observed hardware exclusion; not a source-qualified capability.".into(),
                    |reason| format!("Excluded: {reason}"),
                ),
            ],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_selection_survives_reordering_and_stale_rows_clear() {
        let row = |id: &str| Row {
            key: Key::Snapshot(id.into()),
            title: id.into(),
            details: Vec::new(),
        };
        let mut rows = Rows::default();
        rows.replace(vec![row("a"), row("b")]);
        rows.move_selection(1);
        rows.replace(vec![row("b"), row("c")]);
        assert_eq!(rows.selected().unwrap().key, Key::Snapshot("b".into()));
        rows.replace(Vec::new());
        assert!(rows.selected().is_none());
    }
}
