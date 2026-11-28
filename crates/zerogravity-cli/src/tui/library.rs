//! Presentation-only model/device rows and a single coalesced read worker.
//! Selection is an exact observed identity, not a cache lease or admission.
use super::{
    format,
    settings::{self, Observed, Snapshot, Target},
};
use crate::control_host::execution::{self, Choice, Execution};
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
    pub annotation: Option<String>,
    pub details: Vec<String>,
    pub technical: Vec<String>,
}

#[derive(Default)]
pub(super) struct Rows {
    rows: Vec<Row>,
    visible: Vec<usize>,
    query: String,
    pub selected: Option<Key>,
}
impl Rows {
    fn replace(&mut self, rows: Vec<Row>) {
        self.rows = rows;
        self.filter();
    }
    fn invalidate(&mut self) {
        self.rows.clear();
        self.visible.clear();
        // Keep the preferred key until the next observation reconciles it.
    }
    pub fn query(&self) -> &str {
        &self.query
    }
    pub fn set_query(&mut self, query: String) {
        self.query = query;
        self.filter();
    }
    fn filter(&mut self) {
        let query = self.query.to_ascii_lowercase();
        self.visible.clear();
        self.visible
            .extend(self.rows.iter().enumerate().filter_map(|(i, row)| {
                (query.is_empty()
                    || row.title.to_ascii_lowercase().contains(&query)
                    || row
                        .annotation
                        .as_ref()
                        .is_some_and(|value| value.to_ascii_lowercase().contains(&query))
                    || row
                        .details
                        .iter()
                        .chain(&row.technical)
                        .any(|detail| detail.to_ascii_lowercase().contains(&query)))
                .then_some(i)
            }));
        if !self
            .visible
            .iter()
            .any(|&i| self.selected.as_ref() == Some(&self.rows[i].key))
        {
            self.selected = self.visible.first().map(|&i| self.rows[i].key.clone());
        }
    }
    pub fn visible(&self) -> impl ExactSizeIterator<Item = &Row> {
        self.visible.iter().map(|&i| &self.rows[i])
    }
    pub fn selected(&self) -> Option<&Row> {
        self.visible()
            .find(|row| self.selected.as_ref() == Some(&row.key))
    }
    pub fn index(&self) -> usize {
        self.visible()
            .position(|row| self.selected.as_ref() == Some(&row.key))
            .unwrap_or(0)
    }
    pub fn move_selection(&mut self, delta: i8) -> usize {
        let index = self.index();
        let next = if delta < 0 {
            index.saturating_sub(1)
        } else {
            (index + 1).min(self.visible.len().saturating_sub(1))
        };
        self.selected = self.visible.get(next).map(|&i| self.rows[i].key.clone());
        next
    }
}
#[derive(Default)]
struct Models {
    #[cfg(all(feature = "cache", target_os = "linux"))]
    sources: Vec<zerogravity::control::CachedModelSummary>,
    execution_choices: Vec<Choice>,
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
    execution_choices: Vec<Choice>,
    execution: Option<(String, Execution)>,
    cached: Rows,
    #[cfg(all(feature = "cache", target_os = "linux"))]
    sources: Vec<zerogravity::control::CachedModelSummary>,
    catalog: Rows,
    partial: Rows,
    devices: Rows,
    profiles: Rows,
    config: Rows,
    pub editor: Target,
    observed: Option<Observed>,
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
                    title: target.slug.into(),
                    annotation: Some(target.task.label().into()),
                    details: vec![
                        format!("Repository: {}", target.repository.unwrap_or("unverified")),
                        format!("Source: {}", target.source_status.label()),
                        format!("Execution: {}", target.execution_status.label()),
                    ],
                    technical: vec![
                        format!("Task: {}", target.task.label()),
                        format!("Source layout: {}", target.source_layout.label()),
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
            execution_choices: execution::choices(&[]),
            execution: None,
            cached: Rows::default(),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            sources: Vec::new(),
            catalog,
            partial: Rows::default(),
            devices: Rows::default(),
            profiles: Rows::default(),
            config: Rows::default(),
            editor,
            observed: None,
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
    #[cfg(all(feature = "cache", target_os = "linux"))]
    pub fn source(&self, digest: &str) -> Option<&zerogravity::control::CachedModelSummary> {
        self.sources.iter().find(|source| source.snapshot == digest)
    }

    /// Only a structural route refusal, never a cached capacity grant.
    pub fn source_refusal(&self, digest: &str) -> Option<&'static str> {
        #[cfg(all(feature = "cache", target_os = "linux"))]
        {
            self.source(digest).and_then(|model| {
                crate::resident::cached_route(&model.snapshot, model.source_layout).refusal()
            })
        }
        #[cfg(not(all(feature = "cache", target_os = "linux")))]
        {
            let _ = digest;
            None
        }
    }

    pub fn execution_for(&self, digest: &str) -> Execution {
        self.execution
            .as_ref()
            .filter(|(source, _)| source == digest)
            .map(|(_, choice)| choice.clone())
            .unwrap_or_default()
    }
    pub fn set_execution(&mut self, digest: String, execution: Execution) {
        self.execution = (execution != Execution::default()).then_some((digest, execution));
    }
    pub fn execution_choices(&self, digest: &str) -> Vec<Choice> {
        #[cfg(all(feature = "cache", target_os = "linux"))]
        if let Some(model) = self.source(digest) {
            let route = crate::resident::cached_route(&model.snapshot, model.source_layout);
            return self
                .execution_choices
                .iter()
                .filter(|choice| route.permits(&choice.execution))
                .cloned()
                .collect();
        }
        let _ = digest;
        self.execution_choices.clone()
    }
    pub fn set_editor(&mut self, target: Target) {
        self.editor = target;
        self.observed = None;
        self.config.replace(Vec::new());
        self.config_notice = None;
        self.config.set_query(String::new());
        self.raw = false;
    }
    pub fn invalidate_config(&mut self) {
        self.observed = None;
        self.config.invalidate();
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
    pub fn effective_choice(&self, key: &str) -> Option<&str> {
        self.observed
            .as_ref()?
            .effective_choices
            .get(key)
            .map(String::as_str)
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
    #[cfg(test)]
    pub fn test_replace(&mut self, kind: Kind, rows: Vec<Row>) {
        self.rows_mut(kind).replace(rows);
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
                self.execution_choices = if models.execution_choices.is_empty() {
                    execution::choices(&[])
                } else {
                    models.execution_choices
                };
                #[cfg(all(feature = "cache", target_os = "linux"))]
                {
                    self.sources = models.sources;
                }
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
            Ok(Observation::Config(mut observed)) => {
                if observed.snapshot.target == self.editor {
                    self.config_notice = observed.warning.clone();
                    self.config.replace(std::mem::take(&mut observed.rows));
                    self.observed = Some(observed);
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
            let inventory = if observation.cached.is_empty() {
                None
            } else {
                control.devices().ok()
            };
            let execution_choices =
                execution::choices(inventory.as_ref().map_or(&[], |row| row.devices.as_slice()));
            let cached = observation
                .cached
                .iter()
                .map(|model| Row {
                    key: Key::Snapshot(model.snapshot.clone()),
                    title: model.repository.clone(),
                    annotation: Some(if model.has_weights {
                        format::bytes(model.bytes)
                    } else {
                        format!("{} · metadata", format::bytes(model.bytes))
                    }),
                    details: vec![
                        format!(
                            "Size: {} · {} files",
                            format::bytes(model.bytes),
                            model.files
                        ),
                        format!("Source: {}", model.provider),
                        if model.has_weights {
                            "Cached payloads · not verified by inventory".into()
                        } else {
                            "Metadata-only snapshot · no weights".into()
                        },
                        crate::resident::cached_route(&model.snapshot, model.source_layout)
                            .label()
                            .into(),
                        crate::resident::cached_route(&model.snapshot, model.source_layout)
                            .refusal()
                            .unwrap_or("Candidate only · planner revalidates before load")
                            .into(),
                    ],
                    technical: vec![
                        format!("Snapshot: sha256:{}", model.snapshot),
                        format!("Repository: {}", model.repository),
                        format!("Revision: {}", model.revision),
                        format!("Size: {}", format::exact_bytes(model.bytes)),
                        "No execution or remote freshness qualification.".into(),
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
                            title: format!(
                                "Partial · {}",
                                partial.key.chars().take(12).collect::<String>()
                            ),
                            annotation: Some(format!(
                                "{} / {}",
                                format::bytes(partial.bytes),
                                format::bytes(partial.total)
                            )),
                            details: vec![
                                format!(
                                    "Checkpoint: {} / {}",
                                    format::bytes(partial.bytes),
                                    format::bytes(partial.total)
                                ),
                                "Resume the original pull to revalidate.".into(),
                            ],
                            technical: vec![
                                partial.key,
                                format!("Observed bytes: {}", format::exact_bytes(partial.bytes)),
                                format!("Expected bytes: {}", format::exact_bytes(partial.total)),
                                "Checkpoint observation only; may be active or invalid.".into(),
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
                sources: observation.cached,
                execution_choices,
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
            title: format!("{} · {}", device.id, device.vendor),
            annotation: device.available.map(format::bytes),
            details: vec![
                format!("Class: {}", device.class),
                format!("Backends: {}", device.backends.join(", ")),
                format!(
                    "Capacity: {}",
                    device
                        .ceiling
                        .map_or_else(|| "Unknown".into(), format::bytes)
                ),
                format!(
                    "Available: {}",
                    device
                        .available
                        .map_or_else(|| "Unknown".into(), format::bytes)
                ),
                device
                    .exclusion
                    .clone()
                    .unwrap_or_else(|| "Hardware observed · model fit not checked".into()),
            ],
            technical: vec![
                format!("Observed: {} (Unix ms)", inventory.observed_unix_ms),
                format!("Observation: {}", device.provenance),
                format!(
                    "Capacity: {}",
                    device
                        .ceiling
                        .map_or_else(|| "Unknown".into(), format::exact_bytes)
                ),
                format!(
                    "Available: {}",
                    device
                        .available
                        .map_or_else(|| "Unknown".into(), format::exact_bytes)
                ),
                "Physical observation is not execution admission.".into(),
            ],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_search_keeps_exact_visible_selection_without_losing_inventory() {
        let mut library = Library::new(super::super::work::Document::test_default().target);
        library.shelf = Shelf::Catalog;
        let rows = library.rows_mut(Kind::Models);
        let count = rows.visible().len();
        rows.set_query("gemma4".into());
        assert!(rows.visible().len() > 0 && rows.visible().len() < count);
        assert!(rows.selected().unwrap().title.contains("gemma4"));
        rows.set_query("no such source".into());
        assert!(rows.selected().is_none());
        assert_eq!(rows.visible().len(), 0);
        rows.set_query(String::new());
        assert_eq!(rows.visible().len(), count);
        assert!(rows.selected().is_some());
    }
    #[test]
    fn exact_selection_survives_reordering_and_stale_rows_clear() {
        let row = |id: &str| Row {
            key: Key::Snapshot(id.into()),
            title: id.into(),
            annotation: None,
            details: Vec::new(),
            technical: Vec::new(),
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
