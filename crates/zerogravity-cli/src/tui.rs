//! Native client shell. Runtime lifetime belongs to the outer host; view
//! selection owns no model, task, listener or download.
mod terminal;
mod view;

use crate::output;
use crossterm::event::{self, Event, KeyEventKind};
use std::{
    collections::VecDeque,
    path::PathBuf,
    time::{Duration, Instant},
};
use view::{Action, Navigation, View};
use zerogravity::{
    Result, Runtime,
    control::{ControlEvent, ControlPlane, ControlPoll, ControlSnapshot, ServeSummary},
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
    let runtime = Runtime::builder().build()?;
    let result = session(&runtime, config, output::reporter().color_enabled());
    let retired = runtime.shutdown();
    result.and(retired)
}

/// Client-local session over the caller's existing runtime. Navigation does
/// not implicitly load, acquire, serve, unload, stop or shut down.
fn session(runtime: &Runtime, config: PathBuf, color: bool) -> Result<()> {
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
    };
    let stop = terminal::Stop::install()?;
    let mut terminal = terminal::Session::enter()?;
    let result = (|| {
        let mut dirty = true;
        let mut sampled = Instant::now();
        while !stop.requested() {
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
            if changed || (app.live() && sampled.elapsed() >= Duration::from_secs(1)) {
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
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    match app.navigation.key(key) {
                        Action::Quit => break,
                        Action::Open => app.refresh_cold(control),
                        Action::Refresh => app.refresh(control, true),
                        Action::Redraw => {}
                    }
                    dirty = true;
                }
                _ => {}
            }
        }
        Ok(())
    })();
    let restored = terminal.restore();
    result.and(restored)
}

struct App {
    navigation: Navigation,
    snapshot: ControlSnapshot,
    config: PathBuf,
    cold: Vec<String>,
    events: VecDeque<ControlEvent>,
    notice: Option<String>,
    color: bool,
}

impl App {
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
            Ok(snapshot) => self.snapshot = snapshot,
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
