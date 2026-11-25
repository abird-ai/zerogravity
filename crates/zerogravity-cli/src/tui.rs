//! Native client shell. Runtime lifetime belongs to the outer host; view
//! selection owns no model, task, listener or download.
mod library;
mod terminal;
mod view;
mod work;

use crate::output;
use crossterm::event::{self, Event, KeyEventKind};
use library::{Key, Kind, Library};
use std::{
    collections::VecDeque,
    path::PathBuf,
    time::{Duration, Instant},
};
use view::{Action, DownloadAction, Form, FormKind, LibraryAction, Navigation, RunTarget, View};
use work::{Document, Job, Owner, Prompt, Request, Response};
use zerogravity::{
    Result, Runtime,
    control::{ControlEvent, ControlPlane, ControlPoll, ControlSnapshot, ServeSummary},
    diagnostics::RuntimePhase,
    models::ModelLifecycle,
};

pub(super) fn launch(selection: Option<(&str, &str)>, offline: bool) -> Result<()> {
    if !output::launch_interaction()? {
        output::usage();
        return Ok(());
    }
    crate::config_cli::apply_presentation(selection)?;
    if !output::reporter().run_interaction()? {
        output::usage();
        return Ok(());
    }
    let config = crate::config_cli::selected_path(selection)?;
    let document = Document::new(selection, offline)?;
    // Per-load worker and queue bounds remain the admitted plan's bounds.
    // The host permits the entire existing finite worker range and original
    // host-credit-backed pending policy, including explicit count ceilings.
    let runtime = Runtime::builder()
        .limits(zerogravity::runtime::SchedulerLimits {
            max_active: 32 * zerogravity::runtime::text::MAX_ACTIVE_PER_WORKER,
            max_queued: None,
            ..Default::default()
        })
        .build()?;
    let stop = terminal::Stop::install()?;
    let result = session(
        &runtime,
        document,
        config,
        output::reporter().color_enabled(),
        &stop,
    );
    work::combine(result, work::retire(&runtime))
}

#[cfg(all(test, feature = "acquisition", target_os = "linux"))]
pub(crate) fn download_fixture_session(runtime: &Runtime, root: &std::path::Path) -> Result<()> {
    let stop = terminal::Stop::install()?;
    session(
        runtime,
        Document::test_default(),
        root.join("config.yaml"),
        false,
        &stop,
    )
}

/// Client-local session over the caller's runtime; effectful actions have
/// explicit ownership, and view changes never retire shared owners.
fn session(
    runtime: &Runtime,
    document: Document,
    config: PathBuf,
    color: bool,
    stop: &terminal::Stop,
) -> Result<()> {
    let control = runtime.control();
    let mut events = control.subscribe();
    let snapshot = control.snapshot()?;
    let mut app = App {
        navigation: Navigation::default(),
        snapshot,
        config,
        cold: Vec::new(),
        events: VecDeque::new(),
        notice: None,
        color,
        document,
        job: None,
        selected: None,
        response: None,
        library: Library::default(),
    };
    app.reconcile_owner();
    let mut terminal = terminal::Session::enter()?;
    let result = (|| {
        let mut dirty = true;
        let mut sampled = Instant::now();
        while !stop.requested() {
            dirty |= app.poll_job(control);
            dirty |= app.library.poll(runtime);
            let mut changed = false;
            let mut cache_changed = false;
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            let mut download_changed = false;
            let mut resync = false;
            // Bounded transient invalidations, not a second runtime history.
            for _ in 0..64 {
                match events.poll(Duration::ZERO) {
                    ControlPoll::Pending => break,
                    ControlPoll::Event(event) => {
                        if app.events.len() == 64 {
                            app.events.pop_front();
                        }
                        use zerogravity::control::ControlEventKind;
                        cache_changed |= matches!(
                            event.kind,
                            ControlEventKind::CacheAliasFinished { .. }
                                | ControlEventKind::CacheReclamationFinished { .. }
                        );
                        #[cfg(all(feature = "acquisition", target_os = "linux"))]
                        {
                            download_changed |=
                                matches!(event.kind, ControlEventKind::DownloadChanged { .. });
                        }
                        app.events.push_back(event);
                        changed = true;
                    }
                    ControlPoll::Resync { missed } => {
                        app.events.clear();
                        app.notice =
                            Some(format!("Missed {missed} events; reloaded owner snapshot"));
                        changed = true;
                        resync = true;
                    }
                }
            }
            if changed
                || ((app.live() || app.job.is_some())
                    && sampled.elapsed() >= Duration::from_secs(1))
            {
                app.refresh(control);
                #[cfg(all(feature = "acquisition", target_os = "linux"))]
                {
                    cache_changed |= download_changed
                        && app
                            .snapshot
                            .download
                            .as_ref()
                            .is_some_and(|download| download.status.terminal());
                }
                if resync || (cache_changed && app.navigation.view() == View::Models) {
                    app.refresh_cold(runtime, control);
                }
                sampled = Instant::now();
                dirty = true;
            }
            if dirty {
                terminal
                    .terminal
                    .draw(|frame| view::render(frame, &mut app))?;
                dirty = false;
            }
            let ready = event::poll(Duration::from_millis(200));
            if stop.requested() {
                break;
            }
            if !ready? {
                continue;
            }
            let input = event::read();
            if stop.requested() {
                break;
            }
            match input? {
                Event::Resize(_, _) => dirty = true,
                Event::Paste(value) => {
                    if !app.navigation.paste(&value) {
                        app.notice = Some(
                            "Paste ignored outside an editable field or outside its input bounds"
                                .into(),
                        );
                    }
                    dirty = true;
                }
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    // Ctrl-C is cooperative cancellation while an action
                    // owns work, otherwise it follows the explicit quit path.
                    if key.code == crossterm::event::KeyCode::Char('c')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                        && app.job.is_some()
                    {
                        app.cancel();
                    } else {
                        let action = app.navigation.key(key);
                        if app.handle(runtime, control, action) {
                            break;
                        }
                    }
                    dirty = true;
                }
                _ => {}
            }
        }
        Ok(())
    })();
    let restored = terminal.restore();
    // Never hold raw terminal state over a synchronous load/retirement join.
    let joined = if let Some(mut job) = app.job.take() {
        job.cancel();
        job.join().map(|_| ())
    } else {
        Ok(())
    };
    let observed = app.library.join();
    work::combine(
        work::combine(work::combine(result, restored), joined),
        observed,
    )
}

struct App {
    navigation: Navigation,
    snapshot: ControlSnapshot,
    config: PathBuf,
    cold: Vec<String>,
    events: VecDeque<ControlEvent>,
    notice: Option<String>,
    color: bool,
    document: Document,
    job: Option<Job>,
    selected: Option<Owner>,
    response: Option<Response>,
    library: Library,
}

impl App {
    fn owners(&self) -> impl Iterator<Item = Owner> + '_ {
        self.snapshot
            .runtime
            .catalog
            .iter()
            .filter(|model| model.lifecycle == ModelLifecycle::Ready)
            .filter_map(|model| {
                model.aliases.first().map(|alias| Owner {
                    alias: alias.clone(),
                    registration: model.registration_id,
                })
            })
    }

    fn reconcile_owner(&mut self) {
        let selected = self.selected.as_ref();
        if selected.is_none() || !self.owners().any(|owner| Some(&owner) == selected) {
            let first = self.owners().next();
            self.selected = first;
        }
    }

    fn move_owner(&mut self, delta: i8) {
        let owners: Vec<_> = self.owners().collect();
        if owners.is_empty() {
            self.selected = None;
            return;
        }
        let index = self
            .selected
            .as_ref()
            .and_then(|selected| owners.iter().position(|owner| owner == selected))
            .unwrap_or(0);
        let next = if delta < 0 {
            index.saturating_sub(1)
        } else {
            (index + 1).min(owners.len() - 1)
        };
        self.selected = Some(owners[next].clone());
        self.navigation.reveal_owner(next);
    }

    fn cancel(&mut self) {
        if let Some(job) = &self.job {
            job.cancel();
            self.notice = Some("Cancellation requested; waiting for the action owner".into());
        }
    }

    fn start(&mut self, runtime: &Runtime, request: Request) {
        if self.job.is_some() {
            self.notice =
                Some("An action is already in progress; wait or request cancellation".into());
            return;
        }
        let generates = request.generates();
        match Job::start(runtime, &self.document, request) {
            Ok(job) => {
                if generates {
                    self.response = Some(Response::default());
                }
                self.notice = None;
                self.job = Some(job);
            }
            Err(error) => self.notice = Some(error.to_string()),
        }
    }

    fn poll_job(&mut self, control: ControlPlane<'_>) -> bool {
        let Some(job) = &self.job else {
            return false;
        };
        let revision = self
            .response
            .as_ref()
            .map_or(0, |response| response.revision);
        let changed = if let Some(response) = job.sample(revision) {
            self.response = Some(response);
            true
        } else {
            false
        };
        if !job.finished() {
            // Redraw output without traversing device or cache owners.
            return changed;
        }
        let mut job = self.job.take().expect("finished owned worker");
        // Join before the final sample, so fast terminal output is never lost.
        let result = job.join();
        let revision = self
            .response
            .as_ref()
            .map_or(0, |response| response.revision);
        if let Some(response) = job.sample(revision) {
            self.response = Some(response);
        }
        let message = match result {
            Ok(message) => message,
            Err(error) => format!("{} failed: {error}", job.title()),
        };
        self.notice = Some(message.lines().next().unwrap_or("").to_owned());
        if let Some(key) = job.detail_key.take() {
            self.library.detail = Some((key, message));
        }
        self.refresh(control);
        true
    }

    fn handle(&mut self, runtime: &Runtime, control: ControlPlane<'_>, action: Action) -> bool {
        match action {
            Action::Quit if self.live() || self.job.is_some() => self.navigation.confirm(
                "Quit this process? Its listener and loaded models will be retired after owned work joins.".into(),
                view::Confirmation::Quit,
            ),
            Action::Quit | Action::Confirmed(view::Confirmation::Quit) => return true,
            Action::Open => self.refresh_cold(runtime, control),
            Action::Refresh => {
                self.refresh(control);
                self.refresh_cold(runtime, control);
            }
            Action::Library(action) => self.library_action(runtime, control, action),
            Action::Download(action) => self.download_action(control, action),
            Action::Redraw => {},
            Action::Cancel => self.cancel(),
            Action::MoveOwner(delta) => self.move_owner(delta),
            // A modal or repeated key cannot queue another effectful operation.
            _ if self.job.is_some() => {
                self.notice = Some("An action is already in progress; wait or request cancellation".into());
            }
            Action::Load => {
                if self.snapshot.runtime.status.phase != RuntimePhase::Empty {
                    self.notice = Some("Retire the current route set before loading another".into());
                } else {
                    self.navigation.form(FormKind::Source { run: false });
                }
            }
            Action::Run => {
                if let Some(owner) = self.selected.clone() {
                    self.navigation.form(FormKind::Prompt {
                        target: RunTarget::Owner(owner),
                        chat: self.document.chat,
                    });
                } else if self.snapshot.runtime.status.phase == RuntimePhase::Empty {
                    self.navigation.form(FormKind::Source { run: true });
                } else {
                    self.notice = Some("No ready model owner".into());
                }
            }
            Action::Unload => {
                if let Some(owner) = self.selected.clone() {
                    self.navigation.confirm(
                        format!("Retire {} (registration {}) and all its aliases? Accepted work drains; the listener is unchanged.", owner.alias, owner.registration),
                        view::Confirmation::Request(Request::Unload(owner)),
                    );
                } else { self.notice = Some("No ready model owner".into()); }
            }
            Action::Start => {
                if self.snapshot.serving == ServeSummary::Unavailable {
                    self.notice = Some("Inference API requires the optional serve feature".into());
                } else if self.snapshot.serving != ServeSummary::Idle {
                    self.notice = Some("Stop the current listener first".into());
                } else if self.snapshot.runtime.status.loaded_owners == 0 {
                    self.notice = Some("Load a model first (l)".into());
                } else {
                    self.navigation.form(FormKind::Listen);
                }
            }
            Action::Stop => {
                match self.snapshot.serving {
                    ServeSummary::Listening { binding, .. } | ServeSummary::Finished { binding } => {
                        self.navigation.confirm(
                            format!("Stop and join listener at {}? Loaded models will be retained.", binding.address),
                            view::Confirmation::Request(Request::Stop(binding.generation)),
                        );
                    }
                    _ => self.notice = Some("No owned listener to stop".into()),
                }
            }
            Action::Submit(Form { kind, value }) => match kind {
                FormKind::Source { run: false } => self.start(runtime, Request::Load { source: value, prompt: None, device: None }),
                FormKind::Source { run: true } => self.navigation.form(FormKind::Prompt { target: RunTarget::Source(value), chat: self.document.chat }),
                FormKind::Prompt { target, chat } => {
                    if self.navigation.view() == View::Models {
                        self.navigation.open(View::Home);
                    }
                    let prompt = Prompt { text: value, chat };
                    let request = match target {
                        RunTarget::Owner(owner) => Request::Invoke { owner, prompt },
                        RunTarget::Source(source) => Request::Load { source, prompt: Some(prompt), device: None },
                    };
                    self.start(runtime, request);
                }
                FormKind::Listen => self.start(runtime, Request::Start(value)),
                FormKind::PullSource => self.pull_source(value),
                FormKind::PullSelection { source } => self.confirm_pull(&source, &value),
                FormKind::Alias { digest } => self.start(runtime, Request::Alias { digest, name: value }),
                FormKind::DeviceSource { device, load } => {
                    if load {
                        self.start(runtime, Request::Load { source: value, prompt: None, device: Some(device) });
                    } else {
                        self.start(runtime, Request::CheckFit { source: value, device });
                    }
                }

            },
            Action::Confirmed(view::Confirmation::Request(request)) => {
                #[cfg(all(feature = "acquisition", target_os = "linux"))]
                if matches!(&request, Request::Pull(_)) {
                    self.navigation.open(View::Downloads);
                }
                self.start(runtime, request);
            }
        }
        false
    }

    fn library_action(
        &mut self,
        runtime: &Runtime,
        control: ControlPlane<'_>,
        action: LibraryAction,
    ) {
        let kind = if self.navigation.view() == View::Devices {
            Kind::Devices
        } else {
            Kind::Models
        };
        match action {
            LibraryAction::Cycle => {
                self.library.shelf = self.library.shelf.next();
                self.navigation
                    .reveal_library(self.library.rows(kind).index());
                return;
            }
            LibraryAction::Move(delta) => {
                let index = self.library.rows_mut(kind).move_selection(delta);
                self.navigation.reveal_library(index);
                return;
            }
            LibraryAction::Inspect if kind == Kind::Devices => {
                self.refresh_cold(runtime, control);
                return;
            }
            _ => {}
        }
        if self.job.is_some() {
            self.notice =
                Some("An action is already in progress; wait or request cancellation".into());
            return;
        }
        if action == LibraryAction::Pull {
            if self.document.offline {
                self.notice = Some("Offline mode prohibits acquisition".into());
                return;
            }
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            {
                if self.navigation.view() == View::Models
                    && let Some(Key::Target(target)) = self.library.rows(kind).selected.clone()
                    && let Ok(intent) = crate::pull::native::Intent::parse(&target, "")
                {
                    self.confirm_intent(intent);
                    return;
                }
                self.navigation.form(FormKind::PullSource);
            }
            #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
            {
                self.notice =
                    Some("Acquisition requires the optional Linux acquisition feature".into());
            }
            return;
        }
        let Some(key) = self.library.rows(kind).selected.clone() else {
            self.notice = Some("No observed selection".into());
            return;
        };
        match (key, action) {
            (Key::Snapshot(digest), LibraryAction::Inspect) => self.start(runtime, Request::Inspect(digest)),
            (Key::Snapshot(digest), LibraryAction::Verify) => self.start(runtime, Request::Verify(digest)),
            (Key::Snapshot(digest), LibraryAction::Alias) => self.navigation.form(FormKind::Alias { digest }),
            (Key::Snapshot(digest), LibraryAction::Reclaim(false)) => self.start(runtime, Request::Reclaim { digest, apply: false }),
            (Key::Snapshot(digest), LibraryAction::Reclaim(true)) => self.navigation.confirm(
                format!("Remove exact cached snapshot sha256:{digest}? Active readers or references refuse removal. No references are deleted; original owner revalidates."),
                view::Confirmation::Request(Request::Reclaim { digest, apply: true }),
            ),
            (Key::Snapshot(digest), LibraryAction::Run) => {
                if self.snapshot.runtime.status.phase != RuntimePhase::Empty {
                    self.notice = Some("Retire the current route set before loading another, or invoke its owner from Running".into());
                } else {
                    self.navigation.form(FormKind::Prompt { target: RunTarget::Source(format!("sha256:{digest}")), chat: self.document.chat });
                }
            }
            (Key::Snapshot(digest), LibraryAction::Load(serve)) => {
                if self.snapshot.runtime.status.phase != RuntimePhase::Empty {
                    self.notice = Some("Retire the current route set before loading another".into());
                } else {
                    if serve { self.navigation.open(View::Serve); }
                    self.start(runtime, Request::Load { source: format!("sha256:{digest}"), prompt: None, device: None });
                }
            }
            (Key::Device(device), LibraryAction::Fit) => self.navigation.form(FormKind::DeviceSource { device, load: false }),
            (Key::Device(device), LibraryAction::Load(false)) => {
                if self.snapshot.runtime.status.phase != RuntimePhase::Empty {
                    self.notice = Some("Retire the current route set before loading another".into());
                } else {
                    self.navigation.form(FormKind::DeviceSource { device, load: true });
                }
            }
            (Key::Target(_) | Key::Partial(_), LibraryAction::Inspect) => {
                self.notice = Some("Selected observation shown; not an execution or resume grant".into());
            }
            _ => self.notice = Some("Action requires a compatible observed selection".into()),
        }
    }

    fn pull_source(&mut self, source: String) {
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        {
            if self.document.offline {
                self.notice = Some("Offline mode prohibits acquisition".into());
            } else if source.trim().starts_with("hf:") || source.trim().starts_with("gh:") {
                self.navigation.form(FormKind::PullSelection { source });
            } else {
                self.confirm_pull(&source, "");
            }
        }
        #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
        {
            let _ = source;
            self.notice =
                Some("Acquisition requires the optional Linux acquisition feature".into());
        }
    }
    fn confirm_pull(&mut self, source: &str, selection: &str) {
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        {
            match crate::pull::native::Intent::parse(source, selection) {
                Ok(intent) => self.confirm_intent(intent),
                Err(error) => self.notice = Some(error.to_string()),
            }
        }
        #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
        {
            let _ = (source, selection);
            self.notice = Some("Acquisition unavailable".into());
        }
    }
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    fn confirm_intent(&mut self, intent: crate::pull::native::Intent) {
        match intent.confirmation() {
            Ok(question) => self
                .navigation
                .confirm(question, view::Confirmation::Request(Request::Pull(intent))),
            Err(error) => self.notice = Some(error.to_string()),
        }
    }

    fn download_action(&mut self, control: ControlPlane<'_>, action: DownloadAction) {
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        {
            use zerogravity::control::DownloadCommand;
            // Exact observed id, never resolve "latest" at action time.
            let Some(download) = self
                .snapshot
                .download
                .as_ref()
                .filter(|download| !download.status.terminal())
            else {
                if action == DownloadAction::Cancel {
                    self.cancel();
                } else {
                    self.notice = Some("No observed active download".into());
                }
                return;
            };
            let command = match action {
                DownloadAction::Pause => DownloadCommand::Pause,
                DownloadAction::Resume => DownloadCommand::Resume,
                DownloadAction::Cancel => DownloadCommand::Cancel,
            };
            self.notice = Some(match control.control_download(download.id, command) {
                Ok(()) => format!(
                    "{:?} requested for operation {}; awaiting owner observation",
                    command, download.id
                ),
                Err(error) => error.to_string(),
            });
            self.refresh(control);
        }
        #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
        {
            let _ = control;
            if action == DownloadAction::Cancel {
                self.cancel();
            } else {
                self.notice = Some("Acquisition unavailable".into());
            }
        }
    }

    fn live(&self) -> bool {
        let runtime = &self.snapshot.runtime.status;
        if runtime.loaded_owners != 0
            || matches!(
                self.snapshot.serving,
                ServeSummary::Listening { .. } | ServeSummary::Transitioning
            )
        {
            return true;
        }
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        if self
            .snapshot
            .download
            .as_ref()
            .is_some_and(|d| !d.status.terminal())
        {
            return true;
        }
        false
    }

    fn refresh(&mut self, control: ControlPlane<'_>) {
        match control.snapshot() {
            Ok(snapshot) => {
                self.snapshot = snapshot;
                self.reconcile_owner();
            }
            Err(error) => self.notice = Some(format!("Observation failed: {error}")),
        }
    }

    fn refresh_cold(&mut self, runtime: &Runtime, control: ControlPlane<'_>) {
        match self.navigation.view() {
            View::Models => {
                self.library.request(runtime, Kind::Models);
                return;
            }
            View::Devices => {
                self.library.request(runtime, Kind::Devices);
                return;
            }
            _ => {}
        }
        self.cold = match self.cold_rows(control) {
            Ok(rows) => rows,
            Err(error) => vec![format!("Observation unavailable: {error}")],
        };
    }

    fn cold_rows(&self, _control: ControlPlane<'_>) -> Result<Vec<String>> {
        let mut rows = Vec::new();
        match self.navigation.view() {
            View::Profiles => {
                let default = crate::config_cli::selected_path(None)?;
                if let Some(root) = default.parent() {
                    for name in zerogravity::configuration::profiles(root)? {
                        rows.push(name);
                    }
                }
            }
            View::Config => {
                let file = zerogravity::configuration::ConfigFile::open(self.config.clone())?;
                rows.push(if file.exists() {
                    self.config.display().to_string()
                } else {
                    "Built-in defaults (no document created)".into()
                });
                for (key, setting) in
                    zerogravity::configuration::explain(file.path(), file.bytes())?
                {
                    if matches!(key.as_str(), "model" | "pipeline" | "route") {
                        continue;
                    }
                    rows.push(format!(
                        "{key}  ·  {}",
                        serde_json::to_string(&setting.value)
                            .unwrap_or_else(|_| "unavailable".into())
                    ));
                }
            }
            _ => {}
        }
        Ok(rows)
    }
}
