//! Cell-native navigation and rendering. No execution or filesystem effects.
use super::{
    App,
    library::Kind,
    work::{Owner, Request},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Clear, Paragraph, Wrap},
};
use zerogravity::control::ServeSummary;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum View {
    #[default]
    Home,
    Running,
    Serve,
    Models,
    Downloads,
    Devices,
    Profiles,
    Config,
    Logs,
}
impl View {
    const ALL: [Self; 9] = [
        Self::Home,
        Self::Running,
        Self::Serve,
        Self::Models,
        Self::Downloads,
        Self::Devices,
        Self::Profiles,
        Self::Config,
        Self::Logs,
    ];
    fn index(self) -> usize {
        self as usize
    }
    fn name(self) -> &'static str {
        [
            "Home",
            "Running",
            "Serve",
            "Models",
            "Downloads",
            "Devices",
            "Profiles",
            "Config",
            "Logs",
        ][self.index()]
    }
}

#[derive(Debug, Default)]
struct Palette {
    query: String,
    cursor: usize,
}
impl Palette {
    fn matches(&self) -> Vec<View> {
        let query = self.query.to_ascii_lowercase();
        View::ALL
            .into_iter()
            .filter(|view| view.name().to_ascii_lowercase().contains(&query))
            .collect()
    }
}
#[derive(Debug, PartialEq, Eq)]
pub(super) enum RunTarget {
    Source(String),
    Owner(Owner),
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum FormKind {
    Source {
        run: bool,
    },
    Prompt {
        target: RunTarget,
        chat: bool,
    },
    Listen,
    #[cfg_attr(
        not(all(feature = "acquisition", target_os = "linux")),
        allow(dead_code, reason = "acquisition forms are optional")
    )]
    PullSource,
    #[cfg_attr(
        not(all(feature = "acquisition", target_os = "linux")),
        allow(dead_code, reason = "acquisition forms are optional")
    )]
    PullSelection {
        source: String,
    },
    Alias {
        digest: String,
    },
    DeviceSource {
        device: String,
        load: bool,
    },
}
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Form {
    pub kind: FormKind,
    pub value: String,
}
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Confirmation {
    Quit,
    Request(Request),
}
#[derive(Debug, Default)]
enum Modal {
    #[default]
    None,
    Palette(Palette),
    Help,
    Form(Form),
    Confirm {
        question: String,
        action: Confirmation,
        scroll: u16,
    },
}
#[derive(Debug, Default)]
pub(super) struct Navigation {
    view: View,
    rail: bool,
    cursor: usize,
    scroll: [u16; 9],
    modal: Modal,
}
#[derive(Debug, PartialEq, Eq)]
pub(super) enum LibraryAction {
    Cycle,
    Move(i8),
    Inspect,
    Verify,
    Reclaim(bool),
    Alias,
    Load(bool),
    Run,
    Pull,
    Fit,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DownloadAction {
    Pause,
    Resume,
    Cancel,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Action {
    Redraw,
    Open,
    Refresh,
    Quit,
    Load,
    Run,
    Unload,
    Start,
    Stop,
    Cancel,
    MoveOwner(i8),
    Library(LibraryAction),
    Download(DownloadAction),
    Submit(Form),
    Confirmed(Confirmation),
}

impl Navigation {
    pub fn view(&self) -> View {
        self.view
    }
    pub fn form(&mut self, kind: FormKind) {
        self.modal = Modal::Form(Form {
            kind,
            value: String::new(),
        });
    }
    pub fn confirm(&mut self, question: String, action: Confirmation) {
        self.modal = Modal::Confirm {
            question,
            action,
            scroll: 0,
        };
    }
    pub fn reveal_owner(&mut self, index: usize) {
        // One row per registration; keep selection on screen when moving.
        self.scroll[View::Running.index()] = index.saturating_sub(1) as u16;
    }
    pub fn reveal_library(&mut self, _: usize) {
        self.scroll[self.view.index()] = 0;
    }
    /// Paste is data only. A pasted newline or shortcut never activates a
    /// command. Prompt text can be multiline; source/listener fields cannot.
    pub fn paste(&mut self, value: &str) -> bool {
        let (buffer, limit, multiline) = match &mut self.modal {
            Modal::Form(form) => (
                &mut form.value,
                32 * 1024,
                matches!(
                    form.kind,
                    FormKind::Prompt { .. } | FormKind::PullSelection { .. }
                ),
            ),
            Modal::Palette(palette) => (&mut palette.query, 128, false),
            _ => return false,
        };
        if buffer.len().saturating_add(value.len()) > limit
            || value
                .chars()
                .any(|c| c.is_control() && !(multiline && matches!(c, '\n' | '\r' | '\t')))
        {
            return false;
        }
        buffer.push_str(value);
        if let Modal::Palette(palette) = &mut self.modal {
            palette.cursor = 0;
        }
        true
    }

    pub fn open(&mut self, view: View) -> Action {
        self.view = view;
        self.cursor = view.index();
        self.rail = false;
        self.modal = Modal::None;
        Action::Open
    }
    pub fn key(&mut self, key: KeyEvent) -> Action {
        use KeyCode::*;
        if key.code == Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Action::Quit;
        }
        match &mut self.modal {
            Modal::Palette(palette) => {
                match key.code {
                    Esc => self.modal = Modal::None,
                    Enter => {
                        if let Some(view) = palette.matches().get(palette.cursor) {
                            return self.open(*view);
                        }
                    }
                    Up => palette.cursor = palette.cursor.saturating_sub(1),
                    Down => {
                        palette.cursor =
                            (palette.cursor + 1).min(palette.matches().len().saturating_sub(1))
                    }
                    Backspace => {
                        palette.query.pop();
                        palette.cursor = 0;
                    }
                    Char(c)
                        if !c.is_control()
                            && !key
                                .modifiers
                                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                            && palette.query.len() < 128 =>
                    {
                        palette.query.push(c);
                        palette.cursor = 0;
                    }
                    _ => {}
                }
                return Action::Redraw;
            }
            Modal::Help => {
                if matches!(key.code, Esc | Enter) {
                    self.modal = Modal::None;
                }
                return Action::Redraw;
            }
            Modal::Form(form) => {
                match key.code {
                    Esc => self.modal = Modal::None,
                    Enter => {
                        let Modal::Form(form) = std::mem::take(&mut self.modal) else {
                            unreachable!()
                        };
                        return Action::Submit(form);
                    }
                    Backspace => {
                        form.value.pop();
                    }
                    Char('j')
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && matches!(form.kind, FormKind::PullSelection { .. })
                            && form.value.len() < 32 * 1024 =>
                    {
                        form.value.push('\n')
                    }
                    Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if let FormKind::Prompt { chat, .. } = &mut form.kind {
                            *chat = !*chat;
                        }
                    }
                    Char(c)
                        if !c.is_control()
                            && !key
                                .modifiers
                                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                            && form.value.len() + c.len_utf8() <= 32 * 1024 =>
                    {
                        form.value.push(c);
                    }
                    _ => {}
                }
                return Action::Redraw;
            }
            Modal::Confirm { scroll, .. } => {
                match key.code {
                    Up | PageUp => {
                        *scroll = scroll.saturating_sub(if key.code == PageUp { 8 } else { 1 })
                    }
                    Down | PageDown => {
                        *scroll = scroll.saturating_add(if key.code == PageDown { 8 } else { 1 })
                    }
                    Char('y' | 'Y') => {
                        let Modal::Confirm { action, .. } = std::mem::take(&mut self.modal) else {
                            unreachable!()
                        };
                        return Action::Confirmed(action);
                    }
                    Esc | Char('n' | 'N') => self.modal = Modal::None,
                    _ => {}
                }
                return Action::Redraw;
            }
            Modal::None => {}
        }
        match key.code {
            Char('q') => Action::Quit,
            Char(':') => {
                self.modal = Modal::Palette(Palette::default());
                Action::Redraw
            }
            Char('?') => {
                self.modal = Modal::Help;
                Action::Redraw
            }
            Char('d') if !self.rail && matches!(self.view, View::Models | View::Downloads) => {
                Action::Library(LibraryAction::Pull)
            }
            Char('p') if !self.rail && self.view == View::Downloads => {
                Action::Download(DownloadAction::Pause)
            }
            Char('s') if !self.rail && self.view == View::Downloads => {
                Action::Download(DownloadAction::Resume)
            }
            Char('c') if !self.rail && self.view == View::Downloads => {
                Action::Download(DownloadAction::Cancel)
            }
            Char('m') if !self.rail && self.view == View::Models => {
                Action::Library(LibraryAction::Cycle)
            }
            Enter | Char('i')
                if !self.rail && matches!(self.view, View::Models | View::Devices) =>
            {
                Action::Library(LibraryAction::Inspect)
            }
            Char('v') if !self.rail && self.view == View::Models => {
                Action::Library(LibraryAction::Verify)
            }
            Char('a') if !self.rail && self.view == View::Models => {
                Action::Library(LibraryAction::Alias)
            }
            Char('x') if !self.rail && self.view == View::Models => {
                Action::Library(LibraryAction::Reclaim(true))
            }
            Char('X') if !self.rail && self.view == View::Models => {
                Action::Library(LibraryAction::Reclaim(false))
            }
            Char('l') if !self.rail && matches!(self.view, View::Models | View::Devices) => {
                Action::Library(LibraryAction::Load(false))
            }
            Char('s') if !self.rail && self.view == View::Models => {
                Action::Library(LibraryAction::Load(true))
            }
            Char('p') if !self.rail && self.view == View::Models => {
                Action::Library(LibraryAction::Run)
            }
            Char('e') if !self.rail && self.view == View::Devices => {
                Action::Library(LibraryAction::Fit)
            }
            Char('r') => Action::Refresh,
            Char('c') => Action::Cancel,
            Char('l')
                if !self.rail && matches!(self.view, View::Home | View::Running | View::Serve) =>
            {
                Action::Load
            }
            Enter | Char('p') if !self.rail && matches!(self.view, View::Home | View::Running) => {
                Action::Run
            }
            Char('u') if !self.rail && self.view == View::Running => Action::Unload,
            Char('s') if !self.rail && self.view == View::Serve => Action::Start,
            Char('t') if !self.rail && self.view == View::Serve => Action::Stop,
            Char(c @ '1'..='9') => self.open(View::ALL[c as usize - '1' as usize]),
            Tab | BackTab => {
                self.rail = !self.rail;
                Action::Redraw
            }
            Esc if self.rail => {
                self.rail = false;
                self.cursor = self.view.index();
                Action::Redraw
            }
            Esc => self.open(View::Home),
            Enter if self.rail => self.open(View::ALL[self.cursor]),
            Up | Char('k') if self.rail => {
                self.cursor = self.cursor.saturating_sub(1);
                Action::Redraw
            }
            Down | Char('j') if self.rail => {
                self.cursor = (self.cursor + 1).min(View::ALL.len() - 1);
                Action::Redraw
            }
            Up | Char('k') if matches!(self.view, View::Models | View::Devices) => {
                Action::Library(LibraryAction::Move(-1))
            }
            Down | Char('j') if matches!(self.view, View::Models | View::Devices) => {
                Action::Library(LibraryAction::Move(1))
            }
            Up | Char('k') if self.view == View::Running => Action::MoveOwner(-1),
            Down | Char('j') if self.view == View::Running => Action::MoveOwner(1),
            Up | Char('k') => {
                self.scroll[self.view.index()] = self.scroll[self.view.index()].saturating_sub(1);
                Action::Redraw
            }
            Down | Char('j') => {
                self.scroll[self.view.index()] = self.scroll[self.view.index()].saturating_add(1);
                Action::Redraw
            }
            PageUp => {
                self.scroll[self.view.index()] = self.scroll[self.view.index()].saturating_sub(8);
                Action::Redraw
            }
            PageDown => {
                self.scroll[self.view.index()] = self.scroll[self.view.index()].saturating_add(8);
                Action::Redraw
            }
            _ => Action::Redraw,
        }
    }
}

fn accent(color: bool) -> Style {
    let style = Style::default().add_modifier(Modifier::BOLD);
    if color { style.fg(Color::Cyan) } else { style }
}
fn muted(color: bool) -> Style {
    if color {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    }
}
fn clean(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}
fn text(text: impl AsRef<str>) -> Line<'static> {
    Line::raw(clean(text.as_ref()))
}
fn label(text: impl AsRef<str>, color: bool) -> Line<'static> {
    self::text(text).style(accent(color))
}

fn serve(summary: ServeSummary) -> String {
    match summary {
        ServeSummary::Unavailable => "Not compiled".into(),
        ServeSummary::Idle => "Stopped".into(),
        ServeSummary::Transitioning => "Transitioning".into(),
        ServeSummary::Listening {
            binding,
            accepted_requests,
            active_connections,
        } => format!(
            "http://{}/v1 · {} requests · {} connections",
            binding.address, accepted_requests, active_connections
        ),
        ServeSummary::Finished { binding } => {
            format!("Finished at {}; terminal join pending", binding.address)
        }
    }
}
fn downloads(app: &App) -> String {
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    {
        app.snapshot.download.as_ref().map_or_else(
            || "No operation in this runtime".into(),
            |download| {
                format!(
                    "Operation {} · {:?} · {:?} · {} bytes",
                    download.id, download.status, download.phase, download.total_bytes
                )
            },
        )
    }
    #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
    {
        let _ = app;
        "Acquisition not compiled".into()
    }
}

fn download_rows(rows: &mut Vec<Line<'static>>, app: &App) {
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    if let Some(download) = &app.snapshot.download {
        use zerogravity::control::DownloadStatus;
        if let Some(source) = &download.source {
            rows.push(text(format!(
                "{}:{}@{}",
                source.provider(),
                source.repository(),
                source.revision()
            )));
        }
        rows.push(text(format!(
            "Files: {} / {}",
            download.completed_files, download.selected_files
        )));
        let bytes = |current: u64, exact: Option<u64>, maximum: u64| {
            exact.map_or_else(
                || format!("{current} bytes · total unknown (maximum {maximum})"),
                |total| format!("{current} / {total} bytes"),
            )
        };
        rows.push(text(format!(
            "Total: {}",
            bytes(
                download.total_bytes,
                download.total_expected_bytes,
                download.total_maximum_bytes
            )
        )));
        if let Some(path) = &download.path {
            rows.push(text(format!("File: {path}")));
            rows.push(text(bytes(
                download.file_bytes,
                download.file_expected_bytes,
                download.file_maximum_bytes,
            )));
        }
        rows.push(text(format!(
            "Cached: {} bytes · resumed: {} bytes",
            download.cached_bytes, download.resumed_bytes
        )));
        rows.push(text(format!(
            "Last observed checkpoint: {} bytes",
            download.checkpoint_bytes
        )));
        if let Some(total) = download.total_expected_bytes.filter(|total| *total != 0) {
            let percent = (download.total_bytes as u128 * 100 / total as u128).min(100);
            rows.push(text(format!("Progress: {percent}%")));
        }
        if download.status == DownloadStatus::Running
            && !download.cancel_requested
            && let Some(rate) = download.logical_bytes_per_second.filter(|rate| *rate > 0)
        {
            rows.push(text(format!("Accepted logical bytes/s: {rate}")));
            if let Some(total) = download.total_expected_bytes {
                rows.push(text(format!(
                    "Approximate transfer ETA: {}s (not publication)",
                    total.saturating_sub(download.total_bytes).div_ceil(rate)
                )));
            }
        }
        if download.cancel_requested && !download.status.terminal() {
            rows.push(text(
                "Cancellation requested; awaiting original executor outcome.",
            ));
        }
        if let Some(snapshot) = &download.snapshot {
            rows.push(text(format!("Snapshot: {snapshot}")));
        }
        if download.status == DownloadStatus::DurabilityUnknown {
            rows.push(text(
                "Publication durability unknown; inspect before retrying.",
            ));
        }
    }
    #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
    let _ = (rows, app);
}

fn content(app: &App) -> Vec<Line<'static>> {
    let runtime = &app.snapshot.runtime;
    let mut rows = Vec::new();
    match app.navigation.view {
        View::Home => {
            rows.push(label("Local runtime", app.color));
            rows.push(text(format!(
                "{:?} · {} model owners · {} aliases",
                runtime.status.phase, runtime.status.loaded_owners, runtime.status.loaded_models
            )));
            if runtime.status.failed {
                rows.push(text("Runtime reports a failure; inspect diagnostics."));
            }
            if let Some(scheduler) = &runtime.scheduler {
                rows.push(text(format!(
                    "Requests: {} active · {} queued · {} output tokens",
                    scheduler.counters.active,
                    scheduler.queue.pending,
                    scheduler.counters.output_tokens
                )));
            } else {
                rows.push(text("No scheduler installed"));
            }
            rows.push(Line::default());
            rows.push(label("Quick run", app.color));
            rows.push(text(match &app.selected {
                Some(owner) => format!("Enter: prompt for {}", owner.alias),
                None => "Enter: choose a source and prompt".into(),
            }));
            rows.push(text("l load a cached/local model or configured route set"));
            rows.push(Line::default());
            rows.push(label("Inference API", app.color));
            rows.push(text(serve(app.snapshot.serving)));
            rows.push(label("Downloads", app.color));
            rows.push(text(downloads(app)));
        }
        View::Running => {
            rows.push(text(format!(
                "{} registered model owners · Enter run · u unload · l load",
                runtime.status.loaded_owners
            )));
            for model in &runtime.catalog {
                let selected = app
                    .selected
                    .as_ref()
                    .is_some_and(|owner| owner.registration == model.registration_id);
                let line = text(format!(
                    "{} {} · {:?} · {} admitted",
                    if selected { ">" } else { " " },
                    model.aliases.join(", "),
                    model.lifecycle,
                    model.admitted_requests
                ));
                rows.push(if selected {
                    line.style(accent(app.color))
                } else {
                    line
                });
            }
            if let Some(selected) = &app.selected
                && let Some(model) = runtime
                    .catalog
                    .iter()
                    .find(|model| model.registration_id == selected.registration)
            {
                rows.push(Line::default());
                rows.push(label(
                    format!(
                        "{} · registration {}",
                        selected.alias, selected.registration
                    ),
                    app.color,
                ));
                if let Some(capability) = &model.capabilities {
                    rows.push(text(format!(
                        "Context capacity: {} · vocabulary {}",
                        capability.max_context, capability.vocabulary_size
                    )));
                    if let Some(execution) = &capability.details.execution {
                        rows.push(text(format!("Backend: {:?}", execution.selected.provider)));
                    }
                    if let Some(memory) = &capability.details.memory {
                        use zerogravity::models::ModelMemory;
                        match memory {
                            ModelMemory::Host {
                                session_state_bytes,
                                session_scratch_bytes,
                            } => rows.push(text(format!(
                                "Host session state {} bytes · scratch {} bytes",
                                session_state_bytes, session_scratch_bytes
                            ))),
                            ModelMemory::Resident(memory) => {
                                rows.push(text(format!("Device: {}", memory.device.device)))
                            }
                            ModelMemory::ResidentText {
                                admission,
                                resident_bytes,
                            } => rows.push(text(format!(
                                "Device: {} · {} resident bytes",
                                admission.device.device, resident_bytes
                            ))),
                        }
                    }
                    if let Some(bytes) = capability.details.source_host_reserved_bytes {
                        rows.push(text(format!("Source host reservation: {bytes} bytes")));
                    }
                }
            }
            if let Some(scheduler) = &runtime.scheduler {
                rows.push(text(format!(
                    "Runtime requests: {} active / {} queued · {} output tokens",
                    scheduler.counters.active,
                    scheduler.queue.pending,
                    scheduler.counters.output_tokens
                )));
            }
        }
        View::Serve => {
            rows.push(label("OpenAI-compatible inference API", app.color));
            rows.push(text(serve(app.snapshot.serving)));
            rows.push(text("This process only · s start · t stop · l load"));
            if let Some(scheduler) = &runtime.scheduler {
                rows.push(text(format!(
                    "Context {} · output {} · active {}/{} · queued {}",
                    scheduler.config.context,
                    scheduler.config.output,
                    scheduler.counters.active,
                    scheduler.active_limit,
                    scheduler.queue.pending
                )));
            }
            for model in &runtime.catalog {
                if model.capabilities.is_some() {
                    rows.push(text(format!(
                        "Registered aliases: {}",
                        model.aliases.join(", ")
                    )));
                }
            }
        }
        View::Models | View::Devices => {
            unreachable!("library views render their own selected row and detail")
        }
        View::Downloads => {
            rows.push(label("Download operation", app.color));
            rows.push(text(downloads(app)));
            rows.push(text("d new pull · p pause · s resume · c cancel"));
            if app.document.offline {
                rows.push(text("Offline · new acquisition disabled"));
            }
            download_rows(&mut rows, app);
        }
        View::Logs => {
            rows.push(text(
                "Transient control notifications in this client (not model output).",
            ));
            for event in &app.events {
                rows.push(text(format!("{}  {:?}", event.sequence, event.kind)));
            }
            if app.events.is_empty() {
                rows.push(text("No notifications received."));
            }
        }
        _ => {
            rows.extend(app.cold.iter().map(text));
            if rows.is_empty() {
                rows.push(text("No entries."));
            }
            rows.push(Line::default());
            rows.push(text("Observation only · r refresh"));
        }
    }
    if matches!(app.navigation.view, View::Home | View::Running)
        && let Some(response) = &app.response
    {
        rows.push(Line::default());
        rows.push(label(
            response.owner.as_ref().map_or_else(
                || "Response".into(),
                |owner| {
                    format!(
                        "Response · {} · registration {}",
                        owner.alias, owner.registration
                    )
                },
            ),
            app.color,
        ));
        if response.text.is_empty() {
            rows.push(text("No output yet"));
        } else {
            rows.extend(response.text.lines().map(text));
        }
        if response.clipped {
            rows.push(text(
                "[Display limit reached; original output was not altered]",
            ));
        }
    }
    rows
}

pub(super) fn render(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    if area.width < 20 || area.height < 7 {
        frame.render_widget(Paragraph::new("ZeroGravity · resize / q quit"), area);
        return;
    }
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(2),
    ])
    .areas(area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            ratatui::text::Span::styled(" ZeroGravity", accent(app.color)),
            ratatui::text::Span::raw("  /  local control"),
        ])),
        header,
    );
    if body.width >= 70 {
        let rail = if body.width < 100 { 18 } else { 23 };
        let [nav, workspace] =
            Layout::horizontal([Constraint::Length(rail), Constraint::Min(1)]).areas(body);
        render_rail(frame, nav, app);
        render_workspace(frame, workspace, app);
    } else if app.navigation.rail {
        render_rail(frame, body, app);
    } else {
        render_workspace(frame, body, app);
    }
    let status = if let Some(job) = &app.job {
        format!(
            "{} · {} · c cancel",
            job.title(),
            if job.requested() {
                "cancellation requested; waiting"
            } else {
                "in progress"
            }
        )
    } else {
        app.notice
            .clone()
            .unwrap_or_else(|| "Explicit actions only · no implicit acquisition".into())
    };
    frame.render_widget(
        Paragraph::new(vec![
            text(status).style(muted(app.color)),
            text("1–9 views  Tab nav  : palette  r refresh  ? help  q quit"),
        ]),
        footer,
    );
    match &app.navigation.modal {
        Modal::None => {}
        Modal::Help => {
            let popup = center(body, 68, 16);
            frame.render_widget(Clear, popup);
            frame.render_widget(
                Paragraph::new(vec![
                    label("Navigation", app.color),
                    text("1–9     Open a view"),
                    text("Tab     Switch navigation/workspace"),
                    text("↑ ↓ j k Move/scroll"),
                    text("Enter   Open view / run"),
                    text("l       Load on Home / Running / Serve"),
                    text("u       Retire selected Running registration"),
                    text("s / t   Start / stop listener on Serve"),
                    text("c       Request cancellation"),
                    text(":       Destination palette"),
                    text("r       Refresh observations"),
                    text("Esc     Back / close overlay"),
                    text("q       Quit · Ctrl-C cancel/quit"),
                ])
                .block(Block::bordered().title(" Help "))
                .wrap(Wrap { trim: false }),
                popup,
            );
        }
        Modal::Form(form) => {
            let popup = center(body, 76, 12);
            frame.render_widget(Clear, popup);
            let (title, hint, mode) = match &form.kind {
                FormKind::Source { run } => (
                    if *run {
                        " Quick run source "
                    } else {
                        " Load model "
                    },
                    "MODEL or ID=MODEL; blank uses configured model table. Local/cached only.",
                    "",
                ),
                FormKind::Prompt { target: _, chat } => (
                    " Prompt ",
                    "Type a prompt. Ctrl-R switches chat/raw; Enter submits.",
                    if *chat { "Chat" } else { "Raw" },
                ),
                FormKind::Listen => (
                    " Start inference API ",
                    "Numeric loopback IP:PORT; blank uses selected document/default.",
                    "",
                ),
                FormKind::PullSource => (
                    " Pull source ",
                    "Approved catalog target, or hf:/gh:OWNER/REPO@40-character commit SHA.",
                    "",
                ),
                FormKind::PullSelection { .. } => (
                    " Pull selection ",
                    "Exact relative files, one per line (Ctrl-J/paste), or quant:TAG. Explicit confirmation follows.",
                    "",
                ),
                FormKind::Alias { .. } => (
                    " Create alias ",
                    "New lowercase alias for this exact snapshot. No replacement.",
                    "",
                ),
                FormKind::DeviceSource { load, .. } => (
                    if *load {
                        " Load on device "
                    } else {
                        " Check device policy "
                    },
                    "MODEL or ID=MODEL; blank uses selected document. Exact selected physical device overrides device list.",
                    "",
                ),
            };
            // Show the tail while typing long input; keep full value in form.
            let tail: String = form
                .value
                .chars()
                .rev()
                .take(320)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let mut lines = vec![
                text(hint),
                text(mode),
                Line::default(),
                label(format!("› {tail}▏"), app.color),
                Line::default(),
                text("Enter accept · Esc cancel"),
            ];
            if form.value.chars().count() > 320 {
                lines.insert(3, text("… earlier input retained"));
            }
            frame.render_widget(
                Paragraph::new(lines)
                    .wrap(Wrap { trim: false })
                    .block(Block::bordered().title(title)),
                popup,
            );
        }
        Modal::Confirm {
            question, scroll, ..
        } => {
            let popup = center(body, 86, 15);
            frame.render_widget(Clear, popup);
            let block = Block::bordered().title(" Confirm ");
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let [question_area, controls] =
                Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
            frame.render_widget(
                Paragraph::new(question.lines().map(text).collect::<Vec<_>>())
                    .wrap(Wrap { trim: false })
                    .scroll((*scroll, 0)),
                question_area,
            );
            frame.render_widget(
                Paragraph::new(label("y confirm · n/Esc cancel · arrows scroll", app.color)),
                controls,
            );
        }
        Modal::Palette(palette) => {
            let popup = center(body, 56, 14);
            frame.render_widget(Clear, popup);
            let mut rows = vec![
                label(format!("› {}", palette.query), app.color),
                Line::default(),
            ];
            let matches = palette.matches();
            if matches.is_empty() {
                rows.push(text("No matching view"));
            }
            for (i, view) in matches.into_iter().enumerate() {
                let line = text(format!(
                    "{} {}",
                    if i == palette.cursor { ">" } else { " " },
                    view.name()
                ));
                rows.push(if i == palette.cursor {
                    line.style(accent(app.color))
                } else {
                    line
                });
            }
            frame.render_widget(
                Paragraph::new(rows).block(Block::bordered().title(" Go to · Enter / Esc ")),
                popup,
            );
        }
    }
}
fn render_rail(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let rows: Vec<_> = View::ALL
        .into_iter()
        .map(|view| {
            let selected = app.navigation.rail && view.index() == app.navigation.cursor;
            let active = view == app.navigation.view;
            let line = text(format!(
                "{} {} {}",
                if selected {
                    ">"
                } else if active {
                    "●"
                } else {
                    " "
                },
                view.index() + 1,
                view.name()
            ));
            if selected || active {
                line.style(accent(app.color))
            } else {
                line
            }
        })
        .collect();
    let offset = if app.navigation.rail {
        app.navigation
            .cursor
            .saturating_sub(area.height.saturating_sub(3) as usize) as u16
    } else {
        0
    };
    frame.render_widget(
        Paragraph::new(rows)
            .block(Block::bordered().title(if app.navigation.rail {
                " Navigation * "
            } else {
                " Navigation "
            }))
            .scroll((offset, 0)),
        area,
    );
}
fn render_library(frame: &mut Frame<'_>, area: Rect, app: &mut App, kind: Kind) {
    let title = match kind {
        Kind::Models => format!(" Models · {:?} ", app.library.shelf),
        Kind::Devices => " Devices ".into(),
    };
    let block = Block::bordered().title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let collection = app.library.rows(kind);
    let mut header = match kind {
        Kind::Models => vec![
            text("m Cached/Catalog/Partial · j/k select · Enter inspect"),
            text("l load · p run · s load for Serve · d pull · v verify"),
            text("a alias · X preview · x remove"),
        ],
        Kind::Devices => vec![
            text("j/k select · Enter refresh · e check model policy"),
            text("l load on selected physical device · no inferred model eligibility"),
        ],
    };
    if app.document.offline && kind == Kind::Models {
        header.push(text("Offline · acquisition disabled"));
    }
    let status = app.library.notice(kind).unwrap_or("Owner observations");
    header.push(text(format!(
        "{} entries · {status}",
        collection.rows.len()
    )));
    let height = (header.len() as u16).min(inner.height.saturating_sub(2));
    let [head, body] =
        Layout::vertical([Constraint::Length(height), Constraint::Min(1)]).areas(inner);
    frame.render_widget(Paragraph::new(header), head);
    let [list, detail] = if body.width >= 100 {
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(body)
    } else {
        Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(body)
    };
    let lines: Vec<_> = collection
        .rows
        .iter()
        .map(|row| {
            let selected = collection.selected.as_ref() == Some(&row.key);
            let line = text(format!(
                "{} {}",
                if selected { ">" } else { " " },
                row.title
            ));
            if selected {
                line.style(accent(app.color))
            } else {
                line
            }
        })
        .collect();
    let list_offset = collection
        .index()
        .saturating_sub(list.height.saturating_sub(3) as usize)
        .min(u16::MAX as usize) as u16;
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::bordered().title(" Entries "))
            .scroll((list_offset, 0)),
        list,
    );
    let mut details = Vec::new();
    if let Some(row) = collection.selected() {
        details.push(label(&row.title, app.color));
        details.extend(row.details.iter().map(text));
        if let Some((key, result)) = &app.library.detail
            && key == &row.key
        {
            details.push(Line::default());
            details.extend(result.lines().map(text));
        }
    } else {
        details.push(text("No observed selection"));
    }
    let scroll = &mut app.navigation.scroll[app.navigation.view.index()];
    *scroll = (*scroll).min(details.len().saturating_sub(1).min(u16::MAX as usize) as u16);
    frame.render_widget(
        Paragraph::new(details)
            .block(Block::bordered().title(" Selected · PgUp/PgDn "))
            .wrap(Wrap { trim: false })
            .scroll((*scroll, 0)),
        detail,
    );
}

fn render_workspace(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    match app.navigation.view {
        View::Models => {
            render_library(frame, area, app, Kind::Models);
            return;
        }
        View::Devices => {
            render_library(frame, area, app, Kind::Devices);
            return;
        }
        _ => {}
    }
    let rows = content(app);
    let scroll = &mut app.navigation.scroll[app.navigation.view.index()];
    *scroll = (*scroll).min(rows.len().saturating_sub(1).min(u16::MAX as usize) as u16);
    let offset = *scroll;
    frame.render_widget(
        Paragraph::new(rows)
            .wrap(Wrap { trim: false })
            .scroll((offset, 0))
            .block(Block::bordered().title(format!(" {} ", app.navigation.view.name()))),
        area,
    );
}
fn center(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    use zerogravity::Runtime;
    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    #[test]
    fn navigation_preserves_scroll_and_palette_owns_typing() {
        let mut nav = Navigation::default();
        assert_eq!(nav.key(key(KeyCode::Char('8'))), Action::Open);
        assert_eq!(nav.view(), View::Config);
        nav.key(key(KeyCode::Down));
        nav.key(key(KeyCode::Char(':')));
        assert_eq!(nav.key(key(KeyCode::Char('q'))), Action::Redraw);
        assert!(matches!(&nav.modal, Modal::Palette(p) if p.query=="q"));
        nav.key(key(KeyCode::Backspace));
        for c in "logs".chars() {
            nav.key(key(KeyCode::Char(c)));
        }
        assert_eq!(nav.key(key(KeyCode::Enter)), Action::Open);
        assert_eq!(nav.view(), View::Logs);
        nav.key(key(KeyCode::Char('8')));
        assert_eq!(nav.scroll[View::Config.index()], 1);
        assert_eq!(nav.key(key(KeyCode::Char('q'))), Action::Quit);
    }
    #[test]
    fn prompt_and_confirmation_own_keys_and_preserve_registration() {
        let owner = Owner {
            alias: "main".into(),
            registration: 7,
        };
        let mut nav = Navigation::default();
        nav.form(FormKind::Prompt {
            target: RunTarget::Owner(owner.clone()),
            chat: true,
        });
        assert!(nav.paste("q123\n"));
        assert!(!nav.paste("\u{1b}[200~"));
        assert!(matches!(&nav.modal, Modal::Form(_)));
        assert_eq!(
            nav.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL)),
            Action::Redraw
        );
        assert_eq!(
            nav.key(key(KeyCode::Enter)),
            Action::Submit(Form {
                kind: FormKind::Prompt {
                    target: RunTarget::Owner(owner.clone()),
                    chat: false
                },
                value: "q123\n".into(),
            })
        );
        nav.confirm(
            "retire exact owner".into(),
            Confirmation::Request(Request::Unload(owner.clone())),
        );
        assert_eq!(nav.key(key(KeyCode::Char('n'))), Action::Redraw);
        nav.confirm(
            "retire exact owner".into(),
            Confirmation::Request(Request::Unload(owner.clone())),
        );
        assert_eq!(
            nav.key(key(KeyCode::Char('y'))),
            Action::Confirmed(Confirmation::Request(Request::Unload(owner)))
        );
    }

    #[test]
    fn actual_owner_snapshot_renders_wide_and_narrow() {
        let runtime = Runtime::builder().build().unwrap();
        let mut app = App {
            navigation: Navigation::default(),
            snapshot: runtime.control().snapshot().unwrap(),
            config: "/nonexistent/config.yaml".into(),
            cold: Vec::new(),
            events: Default::default(),
            notice: None,
            color: false,
            document: super::super::work::Document::test_default(),
            job: None,
            selected: None,
            response: None,
            library: super::super::library::Library::default(),
        };
        for width in [60, 90, 150] {
            let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
            terminal.draw(|frame| render(frame, &mut app)).unwrap();
            let buffer = terminal.backend().buffer();
            let text = buffer
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();
            assert!(text.contains("ZeroGravity"));
            assert!(text.contains("0 model owners"));
            assert_eq!(text.contains("Navigation"), width >= 70);
            app.navigation.rail = true;
            terminal.draw(|frame| render(frame, &mut app)).unwrap();
            let text = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();
            assert!(text.contains("Navigation"));
            app.navigation.rail = false;
        }
        runtime.shutdown().unwrap();
    }
}
