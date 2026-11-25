//! Cell-native navigation and rendering. No execution or filesystem effects.
use super::App;
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
#[derive(Debug, Default)]
enum Modal {
    #[default]
    None,
    Palette(Palette),
    Help,
}
#[derive(Debug, Default)]
pub(super) struct Navigation {
    view: View,
    rail: bool,
    cursor: usize,
    scroll: [u16; 9],
    modal: Modal,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Action {
    Redraw,
    Open,
    Refresh,
    Quit,
}

impl Navigation {
    pub fn view(&self) -> View {
        self.view
    }
    fn open(&mut self, view: View) -> Action {
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
            Char('r') => Action::Refresh,
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
                    "Requests: {} active · {} queued",
                    scheduler.counters.active, scheduler.queue.pending
                )));
            } else {
                rows.push(text("No scheduler installed"));
            }
            rows.push(Line::default());
            rows.push(label("Inference API", app.color));
            rows.push(text(serve(app.snapshot.serving)));
            rows.push(Line::default());
            rows.push(label("Downloads", app.color));
            rows.push(text(downloads(app)));
            rows.push(Line::default());
            rows.push(text(
                "Read-only shell · use direct commands for execution and management.",
            ));
        }
        View::Running => {
            rows.push(text(format!(
                "{} registered model owners",
                runtime.status.loaded_owners
            )));
            for model in &runtime.catalog {
                rows.push(label(
                    format!("{}  ·  {:?}", model.aliases.join(", "), model.lifecycle),
                    app.color,
                ));
                rows.push(text(format!(
                    "Registration {} · {} admitted requests",
                    model.registration_id, model.admitted_requests
                )));
                if let Some(capability) = &model.capabilities {
                    rows.push(text(format!(
                        "Context capacity: {}",
                        capability.max_context
                    )));
                }
            }
        }
        View::Serve => {
            rows.push(label("OpenAI-compatible inference API", app.color));
            rows.push(text(serve(app.snapshot.serving)));
            rows.push(text(
                "This process only. Use the direct serve command for now.",
            ));
        }
        View::Downloads => rows.push(text(downloads(app))),
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
    let footer_text = app
        .notice
        .as_deref()
        .unwrap_or("Read-only observations · no implicit model or network effects");
    frame.render_widget(
        Paragraph::new(vec![
            text(footer_text).style(muted(app.color)),
            text("1–9 views  Tab nav  : palette  r refresh  ? help  q quit"),
        ]),
        footer,
    );
    match &app.navigation.modal {
        Modal::None => {}
        Modal::Help => {
            let popup = center(body, 64, 12);
            frame.render_widget(Clear, popup);
            frame.render_widget(
                Paragraph::new(vec![
                    label("Navigation", app.color),
                    text("1–9     Open a view"),
                    text("Tab     Switch navigation/workspace"),
                    text("↑ ↓ j k Move/scroll"),
                    text("Enter   Open selected view"),
                    text(":       Destination palette"),
                    text("r       Refresh observations"),
                    text("Esc     Back / close overlay"),
                    text("q       Quit (Ctrl-C also exits)"),
                ])
                .block(Block::bordered().title(" Help "))
                .wrap(Wrap { trim: false }),
                popup,
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
fn render_workspace(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
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
        assert_eq!(nav.key(key(KeyCode::Char('4'))), Action::Open);
        assert_eq!(nav.view(), View::Models);
        nav.key(key(KeyCode::Down));
        nav.key(key(KeyCode::Char(':')));
        assert_eq!(nav.key(key(KeyCode::Char('q'))), Action::Redraw);
        assert!(matches!(&nav.modal, Modal::Palette(p) if p.query=="q"));
        nav.key(key(KeyCode::Backspace));
        for c in "config".chars() {
            nav.key(key(KeyCode::Char(c)));
        }
        assert_eq!(nav.key(key(KeyCode::Enter)), Action::Open);
        assert_eq!(nav.view(), View::Config);
        nav.key(key(KeyCode::Char('4')));
        assert_eq!(nav.scroll[View::Models.index()], 1);
        assert_eq!(nav.key(key(KeyCode::Char('q'))), Action::Quit);
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
