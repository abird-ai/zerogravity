//! Native client shell. Runtime lifetime belongs to the outer host; view
//! selection owns no model, task, listener or download.
mod terminal;
mod view;
mod work;

use crate::output;
use crossterm::event::{self, Event, KeyEventKind};
use std::{
    collections::VecDeque,
    path::PathBuf,
    time::{Duration, Instant},
};
use view::{Action, Form, FormKind, Navigation, RunTarget, View};
use work::{Document, Job, Owner, Prompt, Request, Response};
use zerogravity::{
    Result, Runtime,
    control::{ControlEvent, ControlPlane, ControlPoll, ControlSnapshot, ServeSummary},
    diagnostics::RuntimePhase,
    models::ModelLifecycle,
};

pub(super) fn launch(selection: Option<(&str, &str)>) -> Result<()> {
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
    let document = Document::new(selection)?;
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
    };
    app.reconcile_owner();
    let mut terminal = terminal::Session::enter()?;
    let result = (|| {
        let mut dirty = true;
        let mut sampled = Instant::now();
        while !stop.requested() {
            dirty |= app.poll_job(control);
            let mut changed = false;
            // Bounded transient invalidations, not a second runtime history.
            for _ in 0..64 {
                match events.poll(Duration::ZERO) {
                    ControlPoll::Pending => break,
                    ControlPoll::Event(event) => {
                        if app.events.len() == 64 {
                            app.events.pop_front();
                        }
                        app.events.push_back(event);
                        changed = true;
                    }
                    ControlPoll::Resync { missed } => {
                        app.events.clear();
                        app.notice =
                            Some(format!("Missed {missed} events; reloaded owner snapshot"));
                        changed = true;
                    }
                }
            }
            if changed
                || ((app.live() || app.job.is_some())
                    && sampled.elapsed() >= Duration::from_secs(1))
            {
                app.refresh(control, changed);
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
    work::combine(work::combine(result, restored), joined)
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
        self.notice = Some(match result {
            Ok(message) => message,
            Err(error) => format!("{} failed: {error}", job.title()),
        });
        self.refresh(control, false);
        true
    }

    fn handle(&mut self, runtime: &Runtime, control: ControlPlane<'_>, action: Action) -> bool {
        match action {
            Action::Quit if self.live() || self.job.is_some() => self.navigation.confirm(
                "Quit this process? Its listener and loaded models will be retired after owned work joins.".into(),
                view::Confirmation::Quit,
            ),
            Action::Quit | Action::Confirmed(view::Confirmation::Quit) => return true,
            Action::Open => self.refresh_cold(control),
            Action::Refresh => self.refresh(control, true),
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
                FormKind::Source { run: false } => self.start(runtime, Request::Load { source: value, prompt: None }),
                FormKind::Source { run: true } => self.navigation.form(FormKind::Prompt { target: RunTarget::Source(value), chat: self.document.chat }),
                FormKind::Prompt { target, chat } => {
                    let prompt = Prompt { text: value, chat };
                    let request = match target {
                        RunTarget::Owner(owner) => Request::Invoke { owner, prompt },
                        RunTarget::Source(source) => Request::Load { source, prompt: Some(prompt) },
                    };
                    self.start(runtime, request);
                }
                FormKind::Listen => self.start(runtime, Request::Start(value)),
            },
            Action::Confirmed(view::Confirmation::Request(request)) => self.start(runtime, request),
        }
        false
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

    fn refresh(&mut self, control: ControlPlane<'_>, cold: bool) {
        match control.snapshot() {
            Ok(snapshot) => {
                self.snapshot = snapshot;
                self.reconcile_owner();
            }
            Err(error) => self.notice = Some(format!("Observation failed: {error}")),
        }
        if cold {
            self.refresh_cold(control);
        }
    }

    fn refresh_cold(&mut self, control: ControlPlane<'_>) {
        self.cold = match self.cold_rows(control) {
            Ok(rows) => rows,
            Err(error) => vec![format!("Observation unavailable: {error}")],
        };
    }

    fn cold_rows(&self, control: ControlPlane<'_>) -> Result<Vec<String>> {
        let mut rows = Vec::new();
        match self.navigation.view() {
            View::Models => {
                #[cfg(all(feature = "cache", target_os = "linux"))]
                {
                    let root = zerogravity::cache::paths::resolve_root(
                        None,
                        std::env::var_os("XDG_CACHE_HOME").as_deref(),
                        std::env::var_os("HOME").as_deref(),
                    )?;
                    let cached = control.cached_models(&root)?;
                    rows.push(format!("{} cached snapshots (metadata only)", cached.len()));
                    for model in cached {
                        rows.push(format!(
                            "{} / {}  ·  {} bytes",
                            model.provider, model.repository, model.bytes
                        ));
                        rows.push(format!("  sha256:{}", model.snapshot));
                    }
                }
                #[cfg(not(all(feature = "cache", target_os = "linux")))]
                rows.push("Cache inspection is unavailable in this build.".into());
            }
            View::Devices => {
                let inventory = control.devices()?;
                rows.push("Observed hardware; model eligibility is checked separately.".into());
                for device in inventory.devices {
                    rows.push(format!(
                        "{} · {} · {}",
                        device.id, device.class, device.vendor
                    ));
                    rows.push(format!("  Backends: {}", device.backends.join(", ")));
                }
            }
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
