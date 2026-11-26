//! Native client shell. Runtime lifetime belongs to the outer host; view
//! selection owns no model, task, listener or download.
pub(crate) mod library;
mod logs;
pub(crate) mod settings;
mod terminal;
mod view;
mod work;

use crate::{
    control_host::{self, ActionSummary, Actions, Origin, Selection},
    output,
};
use crossterm::event::{self, Event, KeyEventKind};
use library::{Key, Kind, Library};
use std::time::{Duration, Instant};
use view::{
    Action, DownloadAction, Form, FormKind, LibraryAction, LogAction, Navigation, RunTarget,
    SettingsAction, View,
};
use work::{Completed, Document, Job, Owner, Prompt, Request, Response};
use zerogravity::{
    Result, Runtime,
    control::{ControlPlane, ControlPoll, ControlSnapshot, ServeSummary},
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
    let document = Document::new(selection, offline)?;
    let runtime = control_host::runtime()?;
    let stop = terminal::Stop::install()?;
    let result = session(
        &runtime,
        Selection::new(document),
        output::reporter().color_enabled(),
        &stop,
    );
    work::combine(result, work::retire(&runtime))
}

#[cfg(all(test, feature = "acquisition", target_os = "linux"))]
pub(crate) fn download_fixture_session(runtime: &Runtime, root: &std::path::Path) -> Result<()> {
    let stop = terminal::Stop::install()?;
    let mut document = Document::test_default();
    document.target.path = root.join("config.yaml");
    session(runtime, Selection::new(document), false, &stop)
}

/// Client-local session over the caller's runtime; effectful actions have
/// explicit ownership, and view changes never retire shared owners.
fn session(
    runtime: &Runtime,
    document: Selection,
    color: bool,
    stop: &terminal::Stop,
) -> Result<()> {
    let control = runtime.control();
    let mut events = control.subscribe();
    let snapshot = control.snapshot()?;
    let actions = Actions::default();
    let mut app = App {
        navigation: Navigation::default(),
        snapshot,
        logs: logs::Logs::new(actions.clone()),
        #[cfg(feature = "web")]
        web: None,
        library: Library::new(document.snapshot().target),
        notice: None,
        color,
        document,
        actions,
        active: None,
        job: None,
        selected: None,
        response: None,
    };
    app.reconcile_owner();
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    app.actions
        .with_logs_mut(|ring| ring.observe_download(app.snapshot.download.as_ref()));
    let mut terminal = terminal::Session::enter()?;
    let result = (|| {
        let mut dirty = true;
        let mut sampled = Instant::now();
        let mut config_revision = app.document.configuration_revision();
        while !stop.requested() {
            dirty |= app.poll_job(runtime, control);
            let active = app.actions.active();
            let action_changed = active != app.active;
            app.active = active;
            dirty |= action_changed;
            dirty |= app.library.poll(runtime);
            let observed = app.document.configuration_revision();
            if config_revision != observed {
                config_revision = observed;
                app.color = output::document_color(app.document.snapshot().ui);
                app.library.invalidate_config();
                if matches!(app.navigation.view(), View::Config | View::Profiles) {
                    app.refresh_cold(runtime, control);
                }
                dirty = true;
            }
            let mut changed = false;
            let mut cache_changed = false;
            let mut devices_changed = false;
            let mut resync = false;
            // Bounded transient invalidations, not a second runtime history.
            for _ in 0..64 {
                match events.poll(Duration::ZERO) {
                    ControlPoll::Pending => break,
                    ControlPoll::Event(event) => {
                        cache_changed |= control_host::inventory::models_changed(event.kind);
                        devices_changed |= control_host::inventory::devices_changed(event.kind);
                        app.actions.with_logs_mut(|ring| ring.event(event));
                        changed = true;
                    }
                    ControlPoll::Resync { missed } => {
                        app.actions.with_logs_mut(|ring| ring.gap(missed));
                        app.notice =
                            Some(format!("Missed {missed} events; reloaded owner snapshot"));
                        changed = true;
                        resync = true;
                    }
                }
            }
            if changed
                || action_changed
                || ((app.live() || app.active.is_some())
                    && sampled.elapsed() >= Duration::from_secs(1))
            {
                app.refresh(control);
                if resync
                    || (cache_changed && app.navigation.view() == View::Models)
                    || (devices_changed && app.navigation.view() == View::Devices)
                {
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
                        && app.active.is_some()
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
    #[cfg(feature = "web")]
    let restored = work::combine(
        restored,
        app.web.take().map_or(Ok(()), crate::web::Host::close),
    );
    // Never hold raw terminal state over a synchronous load/retirement join.
    let joined = app.actions.shutdown();
    let observed = app.library.join();
    work::combine(
        work::combine(work::combine(result, restored), joined),
        observed,
    )
}

struct App {
    navigation: Navigation,
    snapshot: ControlSnapshot,
    logs: logs::Logs,
    #[cfg(feature = "web")]
    web: Option<crate::web::Host>,
    notice: Option<String>,
    color: bool,
    document: Selection,
    actions: Actions,
    active: Option<ActionSummary>,
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
        if let Some(active) = &self.active {
            match self.actions.cancel(active.id, None) {
                Ok(()) => {
                    self.notice =
                        Some("Cancellation requested; waiting for the action owner".into());
                }
                Err(error) => self.notice = Some(error.to_string()),
            }
        }
    }

    fn start(&mut self, runtime: &Runtime, request: Request) {
        if self.job.is_some() || self.active.is_some() {
            self.notice =
                Some("An action is already in progress; wait or request cancellation".into());
            return;
        }
        let generates = request.generates();
        let mut job = Job::ticket(&request);
        match self
            .actions
            .start(runtime, &self.document, request, Origin::Native)
        {
            Ok(id) => {
                job.id = id;
                if generates {
                    self.response = Some(Response::default());
                }
                self.notice = None;
                self.job = Some(job);
                self.active = self.actions.active();
            }
            Err(error) => {
                self.notice = Some(error.to_string());
            }
        }
    }

    fn poll_job(&mut self, runtime: &Runtime, control: ControlPlane<'_>) -> bool {
        let Some(job) = &self.job else {
            return false;
        };
        let revision = self
            .response
            .as_ref()
            .map_or(0, |response| response.revision);
        let (sample, result) = self.actions.poll_native(job.id, revision);
        let changed = sample.is_some();
        if let Some(response) = sample {
            self.response = Some(response);
        }
        let Some(result) = result else {
            return changed;
        };
        let mut job = self.job.take().expect("finished native ticket");
        let message = match result {
            Ok(Completed::Message(message)) => message,
            #[cfg(all(feature = "cache", target_os = "linux"))]
            Ok(Completed::Refused {
                message,
                native_detail,
            }) => native_detail
                .map_or_else(|| message.clone(), |detail| format!("{message}: {detail}")),
            Ok(Completed::Settings(outcome)) => {
                self.color = output::document_color(self.document.snapshot().ui);
                if let Some(editor) = outcome.editor {
                    self.library.set_editor(editor);
                    if matches!(self.navigation.view(), View::Config | View::Profiles) {
                        self.navigation.open(View::Config);
                    }
                } else {
                    self.library.invalidate_config();
                }
                self.refresh_cold(runtime, control);
                outcome.message
            }
            Err(error) => {
                if let Some(retry) = job.retry.take()
                    && self.navigation.view() == View::Config
                    && self.navigation.idle()
                {
                    match retry {
                        settings::Action::Set { file, key, value }
                            if file.target == self.library.editor =>
                        {
                            self.navigation
                                .form_value(FormKind::ConfigValue { file, key }, value)
                        }
                        settings::Action::Replace { file, yaml } => {
                            self.navigation.form_value(FormKind::RawYaml { file }, yaml)
                        }
                        _ => {}
                    }
                }
                format!("{} failed: {error}", job.title)
            }
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
            Action::Quit if self.live() || self.active.is_some() || self.web_owned() => self.navigation.confirm(
                "Quit this process? Its Web/inference listeners and loaded models will be retired after owned work joins.".into(),
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
            Action::Settings(action) => self.settings_action(runtime, action),
            Action::Logs(action) => self.log_action(action),
            Action::OpenWeb => self.open_web(runtime),
            Action::Submit(Form { kind: FormKind::Search { view }, value }) => {
                match view {
                    View::Config => {
                        self.library.query = value;
                        self.library.raw = false;
                        self.library.filter_config();
                    }
                    View::Logs => self.logs.query = value,
                    _ => {}
                }
                self.navigation.reveal_library(0);
            }
            Action::Redraw => {},
            Action::Cancel => self.cancel(),
            Action::MoveOwner(delta) => self.move_owner(delta),
            // A modal or repeated key cannot queue another effectful operation.
            _ if self.active.is_some() || self.job.is_some() => {
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
                        chat: self.document.snapshot().chat,
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
                FormKind::Source { run: true } => self.navigation.form(FormKind::Prompt { target: RunTarget::Source(value), chat: self.document.snapshot().chat }),
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
                FormKind::ProfileCreate => self.start(runtime, Request::Settings(settings::Action::Create(value))),
                FormKind::ProfileCopy { source } => self.start(runtime, Request::Settings(settings::Action::Copy { source, name: value })),
                FormKind::ConfigValue { file, key } => self.start(runtime, Request::Settings(settings::Action::Set { file, key, value })),
                FormKind::RawYaml { file } => self.start(runtime, Request::Settings(settings::Action::Replace { file, yaml: value })),
                FormKind::Search { .. } => unreachable!("search handled before effect guard"),

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
        if self.active.is_some() || self.job.is_some() {
            self.notice =
                Some("An action is already in progress; wait or request cancellation".into());
            return;
        }
        if action == LibraryAction::Pull {
            if self.document.snapshot().offline {
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
                    self.navigation.form(FormKind::Prompt { target: RunTarget::Source(format!("sha256:{digest}")), chat: self.document.snapshot().chat });
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
            if self.document.snapshot().offline {
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
                #[cfg(all(feature = "acquisition", target_os = "linux"))]
                self.actions
                    .with_logs_mut(|ring| ring.observe_download(self.snapshot.download.as_ref()));
                self.reconcile_owner();
            }
            Err(error) => self.notice = Some(format!("Observation failed: {error}")),
        }
    }

    fn refresh_cold(&mut self, runtime: &Runtime, _control: ControlPlane<'_>) {
        let kind = match self.navigation.view() {
            View::Models => Kind::Models,
            View::Devices => Kind::Devices,
            View::Profiles => Kind::Profiles,
            View::Config => Kind::Config,
            _ => return,
        };
        self.library.request(runtime, kind);
    }

    fn settings_action(&mut self, runtime: &Runtime, action: SettingsAction) {
        let kind = if self.navigation.view() == View::Profiles {
            Kind::Profiles
        } else {
            Kind::Config
        };
        match action {
            SettingsAction::Move(delta) => {
                let index = self.library.rows_mut(kind).move_selection(delta);
                self.navigation.reveal_library(index);
                return;
            }
            SettingsAction::Search => {
                self.navigation.form_value(
                    FormKind::Search { view: View::Config },
                    self.library.query.clone(),
                );
                return;
            }
            SettingsAction::RawView => {
                self.library.raw = !self.library.raw;
                self.navigation.reveal_library(0);
                return;
            }
            SettingsAction::Open if kind == Kind::Profiles => {
                if let Some(Key::Profile(name)) = self.library.rows(kind).selected.as_ref() {
                    match settings::Target::profile(name) {
                        Ok(target) => {
                            self.library.set_editor(target);
                            self.navigation.open(View::Config);
                            self.library.request(runtime, Kind::Config);
                        }
                        Err(error) => self.notice = Some(error.to_string()),
                    }
                }
                return;
            }
            _ => {}
        }
        if self.active.is_some() || self.job.is_some() {
            self.notice =
                Some("An action is already in progress; wait or request cancellation".into());
            return;
        }
        if action == SettingsAction::Create {
            self.navigation.form(FormKind::ProfileCreate);
            return;
        }
        let Some(file) = self.library.snapshot() else {
            self.notice = Some("Open a configuration document and wait for its observation".into());
            return;
        };
        match action {
            SettingsAction::Open => {
                if let Some(Key::Setting(key)) = self.library.rows(kind).selected.as_ref() {
                    let value = self.library.value(key).unwrap_or("").to_owned();
                    self.navigation.form_value(
                        FormKind::ConfigValue {
                            file,
                            key: key.clone(),
                        },
                        value,
                    );
                }
            }
            SettingsAction::RawEdit => match file.raw() {
                Ok(yaml) => self
                    .navigation
                    .form_value(FormKind::RawYaml { file: file.clone() }, yaml.to_owned()),
                Err(error) => self.notice = Some(error.to_string()),
            },
            SettingsAction::Activate => {
                self.start(runtime, Request::Settings(settings::Action::Activate(file)))
            }
            SettingsAction::Validate => {
                self.start(runtime, Request::Settings(settings::Action::Validate(file)))
            }
            SettingsAction::Copy => self.navigation.form(FormKind::ProfileCopy { source: file }),
            SettingsAction::Remove => {
                if file
                    .target
                    .profile
                    .as_deref()
                    .is_none_or(|name| name == "default")
                    || file.target.path == self.document.snapshot().target.path
                {
                    self.notice = Some("Activate another profile first; default and portable files cannot be removed here".into());
                } else {
                    self.navigation.confirm(
                        format!(
                            "Remove profile {} at {}? Existing runtime owners remain unchanged.",
                            file.target.label(),
                            file.target.path.display()
                        ),
                        view::Confirmation::Request(Request::Settings(settings::Action::Remove(
                            file,
                        ))),
                    );
                }
            }
            SettingsAction::Unset => {
                if let Some(Key::Setting(key)) = self.library.rows(kind).selected.as_ref() {
                    if self.library.value(key).is_some() {
                        self.navigation.confirm(
                            format!("Unset {key}? Future actions inherit the configured default."),
                            view::Confirmation::Request(Request::Settings(
                                settings::Action::Unset {
                                    file,
                                    key: key.clone(),
                                },
                            )),
                        );
                    } else {
                        self.notice = Some("Setting is already inherited or unset".into());
                    }
                }
            }
            _ => {}
        }
    }

    fn web_owned(&self) -> bool {
        #[cfg(feature = "web")]
        {
            self.web.is_some()
        }
        #[cfg(not(feature = "web"))]
        {
            false
        }
    }

    fn open_web(&mut self, runtime: &Runtime) {
        #[cfg(feature = "web")]
        {
            if self.web.as_ref().is_some_and(crate::web::Host::finished)
                && let Err(error) = self.web.take().expect("finished host").close()
            {
                self.notice = Some(format!("Web host stopped: {error}"));
                return;
            }
            if self.web.is_none() {
                match crate::web::Host::start(
                    runtime,
                    "127.0.0.1:0".parse().expect("loopback default"),
                    self.document.clone(),
                    self.actions.clone(),
                ) {
                    Ok(host) => self.web = Some(host),
                    Err(error) => {
                        self.notice = Some(format!("Web host failed: {error}"));
                        return;
                    }
                }
            }
            self.navigation
                .web_url(self.web.as_ref().expect("owned host").private_url());
        }
        #[cfg(not(feature = "web"))]
        {
            let _ = runtime;
            self.notice = Some("Embedded Web control requires the optional web feature".into());
        }
    }

    fn log_action(&mut self, action: LogAction) {
        match action {
            LogAction::Follow => self.logs.toggle_follow(),
            LogAction::Level => self.logs.cycle_level(),
            LogAction::Component => self.logs.cycle_component(),
            LogAction::Search => self.navigation.form_value(
                FormKind::Search { view: View::Logs },
                self.logs.query.clone(),
            ),
        }
        self.navigation.reveal_library(0);
    }
}
