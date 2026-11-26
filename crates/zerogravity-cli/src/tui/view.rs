//! Cell-native navigation and rendering. No execution or filesystem effects.
use super::{
    App,
    input::{Hits, Intent, Pointer, Target},
    layout::{Motion, Pane, Scrollbar, SplitKind, Splitter, Workspace},
    library::{Key, Kind, Shelf},
    settings::Snapshot,
    theme::Theme,
    work::{Owner, Request},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};
use zerogravity::{configuration::ThemeMode, control::ServeSummary};

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
    ProfileCreate,
    ProfileCopy {
        source: Snapshot,
    },
    ConfigValue {
        file: Snapshot,
        key: String,
    },
    RawYaml {
        file: Snapshot,
    },
    Search {
        view: View,
    },
}
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Form {
    pub kind: FormKind,
    pub value: String,
}
impl Form {
    fn multiline(&self) -> bool {
        matches!(
            self.kind,
            FormKind::Prompt { .. }
                | FormKind::PullSelection { .. }
                | FormKind::ConfigValue { .. }
                | FormKind::RawYaml { .. }
        )
    }
    fn limit(&self) -> usize {
        match self.kind {
            FormKind::ConfigValue { .. } | FormKind::RawYaml { .. } => {
                zerogravity::pipelines::MAX_CONFIG_BYTES
            }
            FormKind::Search { .. } => 256,
            _ => 32 * 1024,
        }
    }
    fn insert(&mut self, cursor: &mut usize, c: char) {
        if self.value.len().saturating_add(c.len_utf8()) <= self.limit() {
            self.value.insert(*cursor, c);
            *cursor += c.len_utf8();
        }
    }
}
fn previous_char(value: &str, cursor: usize) -> usize {
    value[..cursor]
        .char_indices()
        .next_back()
        .map_or(0, |(i, _)| i)
}
fn next_char(value: &str, cursor: usize) -> usize {
    cursor + value[cursor..].chars().next().map_or(0, char::len_utf8)
}
fn line_start(value: &str, cursor: usize) -> usize {
    value[..cursor].rfind('\n').map_or(0, |i| i + 1)
}
fn line_end(value: &str, cursor: usize) -> usize {
    value[cursor..]
        .find('\n')
        .map_or(value.len(), |i| cursor + i)
}
fn vertical(value: &str, cursor: usize, down: bool) -> usize {
    let start = line_start(value, cursor);
    let column = value[start..cursor].chars().count();
    let target = if down {
        let end = line_end(value, cursor);
        if end == value.len() {
            return cursor;
        }
        end + 1
    } else {
        if start == 0 {
            return cursor;
        }
        line_start(value, start - 1)
    };
    let end = line_end(value, target);
    value[target..end]
        .char_indices()
        .nth(column)
        .map_or(end, |(i, _)| target + i)
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
    Theme {
        file: Snapshot,
        choice: ThemeMode,
    },
    #[cfg(feature = "web")]
    WebUrl(String),
    Form {
        form: Form,
        cursor: usize,
    },
    Confirm {
        question: String,
        action: Confirmation,
        scroll: u16,
    },
}
#[derive(Debug, Default)]
pub(super) struct Navigation {
    view: View,
    cursor: usize,
    workspace: Workspace,
    hits: Hits,
    pointer: Pointer,
    too_small: bool,
    modal: Modal,
}
#[derive(Debug, PartialEq, Eq)]
pub(super) enum LibraryAction {
    Cycle,
    SelectShelf(Shelf),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SettingsAction {
    Move(i8),
    Open,
    Create,
    Copy,
    Remove,
    Activate,
    Validate,
    Unset,
    RawView,
    RawEdit,
    Search,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LogAction {
    Follow,
    Level,
    Component,
    Search,
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
    SelectOwner(Owner),
    SelectRow(Kind, Key),
    Scroll(Pane, Motion),
    Library(LibraryAction),
    Download(DownloadAction),
    Settings(SettingsAction),
    Logs(LogAction),
    OpenWeb,
    Appearance,
    Submit(Form),
    Confirmed(Confirmation),
}

impl Navigation {
    pub fn view(&self) -> View {
        self.view
    }
    pub fn form(&mut self, kind: FormKind) {
        self.form_value(kind, String::new());
    }
    pub fn form_value(&mut self, kind: FormKind, value: String) {
        let cursor = if matches!(kind, FormKind::ConfigValue { .. }) && value.ends_with('\n') {
            value.len() - 1
        } else {
            value.len()
        };
        self.modal = Modal::Form {
            form: Form { kind, value },
            cursor,
        };
    }
    pub fn edit_setting(&mut self, file: Snapshot, key: String, value: String) {
        if key == "ui.theme" {
            let choice = serde_yaml::from_str(&value).unwrap_or_default();
            self.modal = Modal::Theme { file, choice };
        } else {
            self.form_value(FormKind::ConfigValue { file, key }, value);
        }
    }
    pub fn preview_theme(&self) -> Option<ThemeMode> {
        if let Modal::Theme { choice, .. } = &self.modal {
            Some(*choice)
        } else {
            None
        }
    }
    pub fn idle(&self) -> bool {
        matches!(self.modal, Modal::None)
    }
    #[cfg(feature = "web")]
    pub fn web_url(&mut self, url: String) {
        self.modal = Modal::WebUrl(url);
    }
    pub fn confirm(&mut self, question: String, action: Confirmation) {
        self.modal = Modal::Confirm {
            question,
            action,
            scroll: 0,
        };
    }
    fn rail(&self) -> bool {
        self.workspace.focus == Pane::Rail
    }
    fn focus(&mut self, pane: Pane) {
        self.workspace.focus(self.view.index(), pane);
        if pane == Pane::Rail {
            self.workspace
                .viewport(self.view.index(), Pane::Rail)
                .reveal(self.cursor);
        }
    }
    pub fn focus_main(&mut self) {
        self.focus(Pane::Main);
        self.reveal_library(0);
    }
    pub fn invalidate_pointer(&mut self) {
        self.pointer.invalidate_selection();
    }
    pub fn cancel_pointer(&mut self) {
        self.pointer.cancel();
    }
    pub fn reveal_owner(&mut self, _: usize) {
        self.reveal_library(0);
    }
    pub fn reveal_library(&mut self, _: usize) {
        self.workspace.request_reveal(self.view.index());
        self.workspace
            .viewport(self.view.index(), Pane::Detail)
            .offset = 0;
    }
    pub fn follow_response(&mut self) {
        for view in [View::Home, View::Running] {
            self.workspace.set_follow_output(view.index(), true);
            self.workspace.viewport(view.index(), Pane::Detail).offset = 0;
        }
    }
    pub fn scroll(&mut self, pane: Pane, motion: Motion) -> bool {
        let viewport = self.workspace.viewport(self.view.index(), pane);
        viewport.scroll(motion);
        let at_end = viewport.offset == viewport.max;
        if pane == Pane::Detail {
            self.workspace.set_follow_output(self.view.index(), at_end);
        }
        // Wheel/manual scrollbar deliberately does not reveal the selection.
        if pane == Pane::Main {
            self.workspace.take_reveal(self.view.index());
        }
        at_end
    }
    pub fn mouse(&mut self, event: MouseEvent) -> Option<Action> {
        match self.pointer.event(event, &self.hits)? {
            Intent::ScrollModal(delta, max) => {
                if let Modal::Confirm { scroll, .. } = &mut self.modal {
                    *scroll = scroll.saturating_add_signed(delta).min(max);
                    Some(Action::Redraw)
                } else {
                    None
                }
            }
            Intent::Resize(splitter, column, row) if self.idle() => {
                self.workspace.drag(splitter, column, row);
                Some(Action::Redraw)
            }
            Intent::Scroll(pane, motion) if self.idle() => Some(Action::Scroll(pane, motion)),
            Intent::Click(target) => {
                // A modal registers only its own visible targets. No backdrop
                // click can select an underlying model or approve a mutation.
                if !self.idle() {
                    return match (&mut self.modal, target) {
                        (Modal::Palette(_), Target::Navigate(view)) => Some(self.open(view)),
                        (Modal::Theme { choice, .. }, Target::Theme(mode)) => {
                            *choice = mode;
                            Some(Action::Redraw)
                        }
                        _ => None,
                    };
                }
                Some(match target {
                    Target::Navigate(view) => self.open(view),
                    Target::Navigation => {
                        self.focus(if self.rail() {
                            self.workspace.preferred_focus(self.view.index())
                        } else {
                            Pane::Rail
                        });
                        Action::Redraw
                    }
                    Target::Pane(pane) | Target::Focus(pane) => {
                        self.focus(pane);
                        Action::Redraw
                    }
                    Target::Maximize(pane) => {
                        self.workspace.toggle_maximize(self.view.index(), pane);
                        Action::Redraw
                    }
                    Target::Row(kind, key) => {
                        self.focus(Pane::Main);
                        Action::SelectRow(kind, key)
                    }
                    Target::Owner(owner) => {
                        self.focus(Pane::Main);
                        Action::SelectOwner(owner)
                    }
                    Target::Shelf(shelf) => {
                        self.focus(Pane::Main);
                        Action::Library(LibraryAction::SelectShelf(shelf))
                    }
                    Target::Appearance => Action::Appearance,
                    Target::Help => {
                        self.modal = Modal::Help;
                        Action::Redraw
                    }
                    Target::Palette => {
                        self.modal = Modal::Palette(Palette::default());
                        Action::Redraw
                    }
                    _ => return None,
                })
            }
            _ => None,
        }
    }
    /// Paste is data only, never a shortcut or submit.
    pub fn paste(&mut self, value: &str) -> bool {
        match &mut self.modal {
            Modal::Form { form, cursor } => {
                if form.value.len().saturating_add(value.len()) > form.limit()
                    || value.chars().any(|c| {
                        c.is_control() && !(form.multiline() && matches!(c, '\n' | '\r' | '\t'))
                    })
                {
                    return false;
                }
                form.value.insert_str(*cursor, value);
                *cursor += value.len();
                true
            }
            Modal::Palette(palette) => {
                if palette.query.len().saturating_add(value.len()) > 128
                    || value.chars().any(char::is_control)
                {
                    return false;
                }
                palette.query.push_str(value);
                palette.cursor = 0;
                true
            }
            _ => false,
        }
    }

    pub fn open(&mut self, view: View) -> Action {
        self.pointer.cancel();
        self.view = view;
        self.cursor = view.index();
        self.workspace.focus = self.workspace.preferred_focus(view.index());
        self.modal = Modal::None;
        Action::Open
    }
    pub fn key(&mut self, key: KeyEvent) -> Action {
        use KeyCode::*;
        self.pointer.cancel();
        if self.too_small {
            return match key.code {
                Esc if !self.idle() => {
                    self.modal = Modal::None;
                    Action::Redraw
                }
                Char('q') if self.idle() => Action::Quit,
                Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    if self.idle() {
                        Action::Quit
                    } else {
                        self.modal = Modal::None;
                        Action::Redraw
                    }
                }
                _ => Action::Redraw,
            };
        }
        if key.code == Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            if !self.idle() {
                self.modal = Modal::None;
                return Action::Redraw;
            }
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
            #[cfg(feature = "web")]
            Modal::WebUrl(_) => {
                if matches!(key.code, Esc | Enter) {
                    self.modal = Modal::None;
                }
                return Action::Redraw;
            }
            Modal::Help => {
                if matches!(key.code, Esc | Enter) {
                    self.modal = Modal::None;
                }
                return Action::Redraw;
            }
            Modal::Theme { choice, .. } => {
                const MODES: [ThemeMode; 3] = [ThemeMode::Auto, ThemeMode::Dark, ThemeMode::Light];
                let index = MODES
                    .iter()
                    .position(|mode| mode == choice)
                    .expect("known theme");
                match key.code {
                    Esc => self.modal = Modal::None,
                    Enter if key.kind == crossterm::event::KeyEventKind::Press => {
                        let Modal::Theme { file, choice } = std::mem::take(&mut self.modal) else {
                            unreachable!()
                        };
                        return Action::Submit(Form {
                            kind: FormKind::ConfigValue {
                                file,
                                key: "ui.theme".into(),
                            },
                            value: choice.label().into(),
                        });
                    }
                    Up | Left | Char('k') => *choice = MODES[index.saturating_sub(1)],
                    Down | Right | Char('j') => *choice = MODES[(index + 1).min(2)],
                    _ => {}
                }
                return Action::Redraw;
            }
            Modal::Form { form, cursor } => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                let raw = matches!(form.kind, FormKind::RawYaml { .. });
                if (key.code == Enter && !raw) || (ctrl && key.code == Char('s') && raw) {
                    let Modal::Form { form, .. } = std::mem::take(&mut self.modal) else {
                        unreachable!()
                    };
                    return Action::Submit(form);
                }
                match key.code {
                    Esc => self.modal = Modal::None,
                    Backspace if *cursor > 0 => {
                        let previous = previous_char(&form.value, *cursor);
                        form.value.drain(previous..*cursor);
                        *cursor = previous;
                    }
                    Delete if *cursor < form.value.len() => {
                        let next = next_char(&form.value, *cursor);
                        form.value.drain(*cursor..next);
                    }
                    Left => *cursor = previous_char(&form.value, *cursor),
                    Right => *cursor = next_char(&form.value, *cursor),
                    Home => *cursor = line_start(&form.value, *cursor),
                    End => *cursor = line_end(&form.value, *cursor),
                    Up if form.multiline() => *cursor = vertical(&form.value, *cursor, false),
                    Down if form.multiline() => *cursor = vertical(&form.value, *cursor, true),
                    Char('u') if ctrl => {
                        form.value.clear();
                        *cursor = 0;
                    }
                    Char('a') if ctrl => *cursor = 0,
                    Char('e') if ctrl => *cursor = form.value.len(),
                    Char('r') if ctrl => {
                        if let FormKind::Prompt { chat, .. } = &mut form.kind {
                            *chat = !*chat;
                        }
                    }
                    Enter if raw => form.insert(cursor, '\n'),
                    Char('j') if ctrl && form.multiline() => form.insert(cursor, '\n'),
                    Tab if raw => {
                        form.insert(cursor, ' ');
                        form.insert(cursor, ' ');
                    }
                    Char(c)
                        if !c.is_control()
                            && !key
                                .modifiers
                                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                    {
                        form.insert(cursor, c)
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
        let pane = self.workspace.focus;
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            let split = match key.code {
                Left => Some((SplitKind::Horizontal, -1)),
                Right => Some((SplitKind::Horizontal, 1)),
                Up => Some((SplitKind::Vertical, -1)),
                Down => Some((SplitKind::Vertical, 1)),
                _ => None,
            };
            if let Some((kind, delta)) = split {
                self.workspace.adjust(kind, delta);
                return Action::Redraw;
            }
        }
        if key.modifiers.contains(KeyModifiers::ALT) && matches!(key.code, Left | Right) {
            self.workspace
                .adjust(SplitKind::Rail, if key.code == Left { -1 } else { 1 });
            return Action::Redraw;
        }
        match key.code {
            F(6) if self.workspace.has_detail => {
                self.focus(if pane == Pane::Detail {
                    Pane::Main
                } else {
                    Pane::Detail
                });
                Action::Redraw
            }
            F(7) if self.workspace.has_detail => {
                self.workspace.toggle_maximize(self.view.index(), pane);
                Action::Redraw
            }
            BackTab => {
                self.focus(match pane {
                    Pane::Main => Pane::Rail,
                    Pane::Rail if self.workspace.has_detail => Pane::Detail,
                    Pane::Rail => Pane::Main,
                    Pane::Detail => Pane::Main,
                });
                Action::Redraw
            }
            PageUp => Action::Scroll(pane, Motion::Page(-1)),
            PageDown => Action::Scroll(pane, Motion::Page(1)),
            Home => Action::Scroll(pane, Motion::Start),
            End => Action::Scroll(pane, Motion::End),
            Up | Char('k') if pane == Pane::Detail => Action::Scroll(pane, Motion::Step(-1)),
            Down | Char('j') if pane == Pane::Detail => Action::Scroll(pane, Motion::Step(1)),
            Enter if pane == Pane::Detail => Action::Redraw,
            Char('q') => Action::Quit,
            Char('T') => Action::Appearance,
            Char(':') => {
                self.modal = Modal::Palette(Palette::default());
                Action::Redraw
            }
            Char('?') => {
                self.modal = Modal::Help;
                Action::Redraw
            }
            Enter if !self.rail() && matches!(self.view, View::Profiles | View::Config) => {
                Action::Settings(SettingsAction::Open)
            }
            Char('c') if !self.rail() && self.view == View::Profiles => {
                Action::Settings(SettingsAction::Create)
            }
            Char('a') if !self.rail() && self.view == View::Config => {
                Action::Settings(SettingsAction::Activate)
            }
            Char('y') if !self.rail() && self.view == View::Config => {
                Action::Settings(SettingsAction::Copy)
            }
            Char('x') if !self.rail() && self.view == View::Config => {
                Action::Settings(SettingsAction::Remove)
            }
            Char('z') if !self.rail() && self.view == View::Config => {
                Action::Settings(SettingsAction::Unset)
            }
            Char('t') if !self.rail() && self.view == View::Config => {
                Action::Settings(SettingsAction::Validate)
            }
            Char('v') if !self.rail() && self.view == View::Config => {
                Action::Settings(SettingsAction::RawView)
            }
            Char('w') if !self.rail() && self.view == View::Config => {
                Action::Settings(SettingsAction::RawEdit)
            }
            Char('/') if !self.rail() && self.view == View::Config => {
                Action::Settings(SettingsAction::Search)
            }
            Char('/') if !self.rail() && self.view == View::Logs => Action::Logs(LogAction::Search),
            Char('p') if !self.rail() && self.view == View::Logs => Action::Logs(LogAction::Follow),
            Char('l') if !self.rail() && self.view == View::Logs => Action::Logs(LogAction::Level),
            Char('f') if !self.rail() && self.view == View::Logs => {
                Action::Logs(LogAction::Component)
            }
            Char('b') if !self.rail() && matches!(self.view, View::Home | View::Serve) => {
                Action::OpenWeb
            }
            Char('d') if !self.rail() && matches!(self.view, View::Models | View::Downloads) => {
                Action::Library(LibraryAction::Pull)
            }
            Char('p') if !self.rail() && self.view == View::Downloads => {
                Action::Download(DownloadAction::Pause)
            }
            Char('s') if !self.rail() && self.view == View::Downloads => {
                Action::Download(DownloadAction::Resume)
            }
            Char('c') if !self.rail() && self.view == View::Downloads => {
                Action::Download(DownloadAction::Cancel)
            }
            Char('m') if !self.rail() && self.view == View::Models => {
                Action::Library(LibraryAction::Cycle)
            }
            Enter | Char('i')
                if !self.rail() && matches!(self.view, View::Models | View::Devices) =>
            {
                Action::Library(LibraryAction::Inspect)
            }
            Char('v') if !self.rail() && self.view == View::Models => {
                Action::Library(LibraryAction::Verify)
            }
            Char('a') if !self.rail() && self.view == View::Models => {
                Action::Library(LibraryAction::Alias)
            }
            Char('x') if !self.rail() && self.view == View::Models => {
                Action::Library(LibraryAction::Reclaim(true))
            }
            Char('X') if !self.rail() && self.view == View::Models => {
                Action::Library(LibraryAction::Reclaim(false))
            }
            Char('l') if !self.rail() && matches!(self.view, View::Models | View::Devices) => {
                Action::Library(LibraryAction::Load(false))
            }
            Char('s') if !self.rail() && self.view == View::Models => {
                Action::Library(LibraryAction::Load(true))
            }
            Char('p') if !self.rail() && self.view == View::Models => {
                Action::Library(LibraryAction::Run)
            }
            Char('e') if !self.rail() && self.view == View::Devices => {
                Action::Library(LibraryAction::Fit)
            }
            Char('r') => Action::Refresh,
            Char('c') => Action::Cancel,
            Char('l')
                if !self.rail()
                    && matches!(self.view, View::Home | View::Running | View::Serve) =>
            {
                Action::Load
            }
            Enter | Char('p')
                if !self.rail() && matches!(self.view, View::Home | View::Running) =>
            {
                Action::Run
            }
            Char('u') if !self.rail() && self.view == View::Running => Action::Unload,
            Char('s') if !self.rail() && self.view == View::Serve => Action::Start,
            Char('t') if !self.rail() && self.view == View::Serve => Action::Stop,
            Char(c @ '1'..='9') => self.open(View::ALL[c as usize - '1' as usize]),
            Tab => {
                self.focus(if self.rail() {
                    Pane::Main
                } else if self.workspace.has_detail && self.workspace.focus == Pane::Main {
                    Pane::Detail
                } else {
                    Pane::Rail
                });
                Action::Redraw
            }
            Esc if self.rail() => {
                self.focus(Pane::Main);
                self.cursor = self.view.index();
                Action::Redraw
            }
            Esc if pane == Pane::Detail => {
                self.focus(Pane::Main);
                Action::Redraw
            }
            Esc => self.open(View::Home),
            Enter if self.rail() => self.open(View::ALL[self.cursor]),
            Up | Char('k') if self.rail() => {
                self.cursor = self.cursor.saturating_sub(1);
                self.workspace
                    .viewport(self.view.index(), Pane::Rail)
                    .reveal(self.cursor);
                Action::Redraw
            }
            Down | Char('j') if self.rail() => {
                self.cursor = (self.cursor + 1).min(View::ALL.len() - 1);
                self.workspace
                    .viewport(self.view.index(), Pane::Rail)
                    .reveal(self.cursor);
                Action::Redraw
            }
            Up | Char('k') if matches!(self.view, View::Models | View::Devices) => {
                Action::Library(LibraryAction::Move(-1))
            }
            Down | Char('j') if matches!(self.view, View::Models | View::Devices) => {
                Action::Library(LibraryAction::Move(1))
            }
            Up | Char('k') if matches!(self.view, View::Profiles | View::Config) => {
                Action::Settings(SettingsAction::Move(-1))
            }
            Down | Char('j') if matches!(self.view, View::Profiles | View::Config) => {
                Action::Settings(SettingsAction::Move(1))
            }
            Up | Char('k') if self.view == View::Running => Action::MoveOwner(-1),
            Down | Char('j') if self.view == View::Running => Action::MoveOwner(1),
            Up | Char('k') => Action::Scroll(pane, Motion::Step(-1)),
            Down | Char('j') => Action::Scroll(pane, Motion::Step(1)),
            _ => Action::Redraw,
        }
    }
}

// One restrained rule per pane rather than a nested bright frame.
fn pane<'a>(title: impl Into<Line<'a>>, theme: Theme, focused: bool) -> Block<'a> {
    Block::new()
        .borders(Borders::TOP)
        .border_style(theme.border(focused))
        .style(theme.surface())
        .title(title.into().style(theme.heading()))
}
fn dialog<'a>(title: &'a str, theme: Theme) -> Block<'a> {
    Block::bordered()
        .border_style(theme.border(true))
        .style(theme.raised())
        .title(Line::styled(title, theme.heading()))
}
fn clean(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}
fn text(text: impl AsRef<str>) -> Line<'static> {
    Line::raw(clean(text.as_ref()))
}
fn label(text: impl AsRef<str>, theme: Theme) -> Line<'static> {
    self::text(text).style(theme.heading())
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
fn serve_style(summary: ServeSummary, theme: Theme) -> ratatui::style::Style {
    match summary {
        ServeSummary::Listening { .. } => theme.success(),
        ServeSummary::Transitioning => theme.info(),
        ServeSummary::Finished { .. } => theme.warning(),
        _ => theme.muted(),
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

struct Content {
    rows: Vec<Line<'static>>,
    details: Vec<Line<'static>>,
    owners: Vec<(usize, Owner)>,
}
fn content(app: &App, theme: Theme) -> Content {
    let runtime = &app.snapshot.runtime;
    let mut rows = Vec::new();
    let mut detail_at = None;
    let mut owners = Vec::new();
    match app.navigation.view {
        View::Home => {
            rows.push(label("Local runtime", theme));
            rows.push(
                text(format!(
                    "{:?} · {} model owners · {} aliases",
                    runtime.status.phase,
                    runtime.status.loaded_owners,
                    runtime.status.loaded_models
                ))
                .style(if runtime.status.failed {
                    theme.danger()
                } else {
                    theme.info()
                }),
            );
            if runtime.status.failed {
                rows.push(
                    text("Runtime reports a failure; inspect diagnostics.").style(theme.danger()),
                );
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
            rows.push(text(format!(
                "Active configuration: {}",
                app.document.snapshot().target.label()
            )));
            #[cfg(feature = "web")]
            rows.push(text("b open private Web control URL"));
            rows.push(label("Quick run", theme));
            rows.push(text(match &app.selected {
                Some(owner) => format!("Enter: prompt for {}", owner.alias),
                None => "Enter: choose a source and prompt".into(),
            }));
            rows.push(text("l load a cached/local model or configured route set"));
            rows.push(Line::default());
            rows.push(label("Inference API", theme));
            rows.push(
                text(serve(app.snapshot.serving)).style(serve_style(app.snapshot.serving, theme)),
            );
            rows.push(label("Downloads", theme));
            rows.push(text(downloads(app)));
        }
        View::Running => {
            rows.push(text(format!(
                "{} registered model owners · Enter run · u unload · l load",
                runtime.status.loaded_owners
            )));
            for model in &runtime.catalog {
                if model.lifecycle == zerogravity::models::ModelLifecycle::Ready
                    && let Some(alias) = model.aliases.first()
                {
                    owners.push((
                        rows.len(),
                        Owner {
                            alias: alias.clone(),
                            registration: model.registration_id,
                        },
                    ));
                }
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
                    line.style(theme.selected())
                } else {
                    line
                });
            }
            detail_at = Some(rows.len());
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
                    theme,
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
            rows.push(label("OpenAI-compatible inference API", theme));
            rows.push(
                text(serve(app.snapshot.serving)).style(serve_style(app.snapshot.serving, theme)),
            );
            rows.push(text("This process only · s start · t stop · l load"));
            #[cfg(feature = "web")]
            rows.push(text("b open private Web control URL"));
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
            rows.push(label("Download operation", theme));
            rows.push(text(downloads(app)));
            rows.push(text("d new pull · p pause · s resume · c cancel"));
            if app.document.snapshot().offline {
                rows.push(text("Offline · new acquisition disabled"));
            }
            download_rows(&mut rows, app);
        }
        View::Profiles | View::Config | View::Logs => unreachable!("dedicated workspace renderer"),
    }
    if matches!(app.navigation.view, View::Home | View::Running)
        && let Some(response) = &app.response
    {
        if detail_at.is_none() {
            detail_at = Some(rows.len());
        }
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
            theme,
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
    let details = detail_at.map_or_else(Vec::new, |at| rows.split_off(at));
    Content {
        rows,
        details,
        owners,
    }
}

fn row_rect(area: Rect, index: usize, offset: u16) -> Option<Rect> {
    let y = index.checked_sub(offset as usize)?;
    (y < area.height as usize).then(|| Rect::new(area.x, area.y + y as u16, area.width, 1))
}
fn divider(
    frame: &mut Frame<'_>,
    nav: &mut Navigation,
    theme: Theme,
    track: Option<(Rect, Splitter)>,
) {
    let Some((rect, splitter)) = track else {
        return;
    };
    let mark = if splitter.kind == SplitKind::Vertical {
        "─"
    } else {
        "│"
    };
    for y in rect.y..rect.bottom() {
        for x in rect.x..rect.right() {
            frame.buffer_mut()[(x, y)]
                .set_symbol(mark)
                .set_style(theme.muted());
        }
    }
    nav.hits.add(rect, Target::Split(splitter));
}
fn control(
    frame: &mut Frame<'_>,
    nav: &mut Navigation,
    theme: Theme,
    rect: Rect,
    title: &str,
    target: Target,
) {
    if rect.is_empty() {
        return;
    }
    frame.render_widget(Paragraph::new(text(title)).style(theme.raised()), rect);
    nav.hits.add(rect, target);
}
fn pane_body(
    frame: &mut Frame<'_>,
    area: Rect,
    nav: &mut Navigation,
    theme: Theme,
    which: Pane,
    title: &str,
) -> Rect {
    if area.is_empty() {
        return area;
    }
    nav.hits.add(area, Target::Pane(which));
    let focused = nav.workspace.focus == which;
    let block = pane(
        format!(" {}{} ", title, if focused { " *" } else { "" }),
        theme,
        focused,
    );
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if which != Pane::Rail && nav.workspace.has_detail && area.width >= 20 {
        let other = if which == Pane::Main {
            Pane::Detail
        } else {
            Pane::Main
        };
        let maximized = nav.workspace.maximized(nav.view.index(), which);
        let toggle = if maximized { " [-] " } else { " [+] " };
        let focus = if which == Pane::Main {
            " Details › "
        } else {
            " ‹ List "
        };
        let toggle_width = toggle.chars().count() as u16;
        let focus_width = focus.chars().count() as u16;
        let right = area.right();
        let toggle_rect = Rect::new(right - toggle_width, area.y, toggle_width, 1);
        control(
            frame,
            nav,
            theme,
            toggle_rect,
            toggle,
            Target::Maximize(which),
        );
        if area.width >= 33 {
            control(
                frame,
                nav,
                theme,
                Rect::new(toggle_rect.x - focus_width, area.y, focus_width, 1),
                focus,
                Target::Focus(other),
            );
        } else {
            control(
                frame,
                nav,
                theme,
                Rect::new(toggle_rect.x - 4, area.y, 4, 1),
                " ↔ ",
                Target::Focus(other),
            );
        }
    }
    inner
}
/// The exact Paragraph wrapper computes visual line count (including wide
/// graphemes). Row lists deliberately truncate to one row per identity.
enum ScrollMode {
    Manual,
    Reveal(usize),
    Follow,
}
impl From<Option<usize>> for ScrollMode {
    fn from(row: Option<usize>) -> Self {
        row.map_or(Self::Manual, Self::Reveal)
    }
}
fn scroll_rows(
    frame: &mut Frame<'_>,
    nav: &mut Navigation,
    theme: Theme,
    which: Pane,
    area: Rect,
    paragraph: Paragraph<'_>,
    mode: ScrollMode,
) -> (Rect, u16) {
    if area.is_empty() {
        return (area, 0);
    }
    let body = Rect::new(area.x, area.y, area.width.saturating_sub(1), area.height);
    let track = Rect::new(body.right(), area.y, area.width - body.width, area.height);
    let viewport = nav.workspace.viewport(nav.view.index(), which);
    viewport.measure(paragraph.line_count(body.width), body.height);
    match mode {
        ScrollMode::Manual => {}
        ScrollMode::Reveal(row) => viewport.reveal(row),
        ScrollMode::Follow => viewport.scroll(Motion::End),
    }
    let viewport = *viewport;
    frame.render_widget(paragraph.scroll((viewport.offset, 0)), body);
    if let Some(bar) = Scrollbar::new(which, track, viewport) {
        for y in track.y..track.bottom() {
            let thumb = y >= bar.thumb.y && y < bar.thumb.bottom();
            frame.buffer_mut()[(track.x, y)]
                .set_symbol(if thumb { "┃" } else { "│" })
                .set_style(if thumb {
                    theme.muted()
                } else {
                    theme.border(false)
                });
        }
        nav.hits.add(track, Target::Scrollbar(bar));
    }
    (body, viewport.offset)
}

pub(super) fn render(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    let theme = app.theme();
    app.navigation.hits.clear();
    frame.render_widget(Block::default().style(theme.canvas()), area);
    app.navigation.too_small = area.width < 20 || area.height < 7;
    if app.navigation.too_small {
        frame.render_widget(Paragraph::new("ZeroGravity · resize / q quit"), area);
        return;
    }
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(2),
    ])
    .areas(area);
    let profile = clean(&app.document.snapshot().target.label());
    let appearance = format!(" T Theme: {} ", theme.description());
    let appearance_width = if header.width >= 60 {
        appearance.chars().count() as u16
    } else {
        0
    };
    let nav_width = if area.width < 70 { 6 } else { 0 };
    let [nav_button, identity, appearance_area] = Layout::horizontal([
        Constraint::Length(nav_width),
        Constraint::Min(1),
        Constraint::Length(appearance_width),
    ])
    .areas(header);
    if nav_width > 0 {
        control(
            frame,
            &mut app.navigation,
            theme,
            nav_button,
            " Nav ",
            Target::Navigation,
        );
    }
    let header_context = if area.width >= 100 {
        format!("  ·  {profile}")
    } else {
        String::new()
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ZeroGravity", theme.accent()),
            Span::raw("  /  local control"),
            Span::styled(header_context, theme.muted()),
        ]))
        .style(theme.surface()),
        identity,
    );
    if appearance_width > 0 {
        control(
            frame,
            &mut app.navigation,
            theme,
            appearance_area,
            &appearance,
            Target::Appearance,
        );
    }
    app.navigation.workspace.has_detail = matches!(
        app.navigation.view,
        View::Models | View::Devices | View::Profiles | View::Config | View::Running
    ) || (app.navigation.view == View::Home
        && app.response.is_some());
    if !app.navigation.workspace.has_detail && app.navigation.workspace.focus == Pane::Detail {
        app.navigation.focus(Pane::Main);
    }
    let (rail, workspace, track) = app.navigation.workspace.shell(body);
    if !rail.is_empty() {
        render_rail(frame, rail, app, theme);
    }
    if !workspace.is_empty() {
        render_workspace(frame, workspace, app, theme);
    }
    divider(frame, &mut app.navigation, theme, track);
    let status = if let Some(job) = &app.active {
        format!(
            "{} · {:?} · {} · c cancel",
            job.title,
            job.origin,
            if job.cancel_requested {
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
            text(status).style(if app.active.is_some() {
                theme.info()
            } else {
                theme.muted()
            }),
            text("1–9 views  Tab focus  F6 panes  F7 size  F8 mouse  q quit").style(theme.muted()),
        ])
        .style(theme.surface()),
        footer,
    );
    if footer.width >= 44 {
        let y = footer.bottom() - 1;
        control(
            frame,
            &mut app.navigation,
            theme,
            Rect::new(footer.right() - 8, y, 8, 1),
            " ? Help ",
            Target::Help,
        );
        control(
            frame,
            &mut app.navigation,
            theme,
            Rect::new(footer.right() - 16, y, 8, 1),
            " : Go   ",
            Target::Palette,
        );
    }
    if !app.navigation.idle() {
        app.navigation.hits.clear();
    }
    match &mut app.navigation.modal {
        Modal::None => {}
        #[cfg(feature = "web")]
        Modal::WebUrl(url) => {
            let popup = center(body, 110, 11);
            frame.render_widget(Clear, popup);
            frame.render_widget(
                Paragraph::new(vec![
                    label("Open this private URL in a browser", theme),
                    text(url),
                    Line::default(),
                    text("Same process and runtime. Keep the control credential private."),
                    text(
                        "Esc / Enter closes this view; the listener remains owned by this process.",
                    ),
                ])
                .wrap(Wrap { trim: false })
                .block(dialog(" Embedded Web ", theme)),
                popup,
            );
        }
        Modal::Help => {
            let popup = center(body, 68, 20);
            frame.render_widget(Clear, popup);
            frame.render_widget(
                Paragraph::new(vec![
                    label("Navigation", theme),
                    text("1–9     Open a view"),
                    text("Tab     Switch navigation/workspace"),
                    text("↑ ↓ j k Move/scroll"),
                    text("Enter   Open view / run"),
                    text("l       Load on Home / Running / Serve"),
                    text("u       Retire selected Running registration"),
                    text("s / t   Start / stop listener on Serve"),
                    text("b       Web control URL on Home / Serve"),
                    text("c       Request cancellation"),
                    text(":       Destination palette"),
                    text("r       Refresh observations"),
                    text("T       Appearance setting (Auto / Dark / Light)"),
                    text("F6 panes · F7 maximize · F8 mouse capture"),
                    text("Alt+←/→ rail · Ctrl+arrows split size"),
                    text("Esc     Back / close overlay"),
                    text("q       Quit · Ctrl-C cancel/quit"),
                ])
                .block(dialog(" Help ", theme))
                .wrap(Wrap { trim: false }),
                popup,
            );
        }
        Modal::Theme { file, choice } => {
            let popup = center(body, 54, 11);
            frame.render_widget(Clear, popup);
            let mut rows = Vec::new();
            if popup.height >= 11 {
                rows.push(text(format!("Editing: {}", file.target.label())).style(theme.muted()));
                rows.push(Line::default());
            }
            if popup.height >= 7 {
                for (mode, title) in [
                    (ThemeMode::Auto, "Auto"),
                    (ThemeMode::Dark, "Dark"),
                    (ThemeMode::Light, "Light"),
                ] {
                    let hit_area = dialog(" Appearance ", theme).inner(popup);
                    if let Some(rect) = row_rect(hit_area, rows.len(), 0) {
                        app.navigation.hits.add(rect, Target::Theme(mode));
                    }
                    let selected = *choice == mode;
                    let row = text(format!("{} {title}", if selected { ">" } else { " " }));
                    rows.push(if selected {
                        row.style(theme.selected())
                    } else {
                        row
                    });
                }
            } else {
                rows.push(text(format!("‹ {} ›", choice.label())).style(theme.selected()));
            }
            if popup.height >= 11 {
                rows.push(Line::default());
                rows.push(text("Auto: terminal hint; unknown uses dark.").style(theme.muted()));
            }
            rows.push(
                text(if popup.width >= 41 {
                    "↑ ↓ preview · Enter apply · Esc cancel"
                } else {
                    "↑↓ Enter Esc"
                })
                .style(theme.accent()),
            );
            frame.render_widget(
                Paragraph::new(rows).block(dialog(" Appearance ", theme)),
                popup,
            );
        }
        Modal::Form { form, cursor } => {
            let raw = matches!(form.kind, FormKind::RawYaml { .. });
            let popup = if raw { body } else { center(body, 76, 14) };
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
                FormKind::ProfileCreate => (
                    " Create profile ",
                    "New bounded lowercase name; no overwrite or activation.",
                    "",
                ),
                FormKind::ProfileCopy { .. } => (
                    " Copy profile ",
                    "New name; relative paths preserved by the shared copy owner.",
                    "",
                ),
                FormKind::ConfigValue { .. } => (
                    " Edit setting ",
                    "One YAML value (Ctrl-J/paste for multiple lines). Ctrl-U clears input.",
                    "",
                ),
                FormKind::RawYaml { .. } => (
                    " Raw YAML editor ",
                    "Ctrl-S validates and saves; Enter newline, Tab two spaces, Ctrl-U clear, Esc discard.",
                    "",
                ),
                FormKind::Search { .. } => (
                    " Search ",
                    "Search visible rows by text. Empty clears the filter.",
                    "",
                ),
            };
            let block = dialog(title, theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let header = match &form.kind {
                FormKind::ConfigValue { key, .. } => format!("{key} · {hint}"),
                _ if !mode.is_empty() => format!("{hint} · {mode}"),
                _ => hint.to_owned(),
            };
            let [head, input, controls] = Layout::vertical([
                Constraint::Length(if inner.width < 60 { 3 } else { 2 }),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .areas(inner);
            frame.render_widget(
                Paragraph::new(text(header)).wrap(Wrap { trim: false }),
                head,
            );
            let before = &form.value[..*cursor];
            let after = &form.value[*cursor..];
            let mut prefix = before.rsplit('\n');
            let current = prefix.next().unwrap_or("");
            let keep = (input.height.saturating_sub(2) as usize / 2).max(1);
            let previous: Vec<_> = prefix.take(keep).collect();
            let mut rows: Vec<Line<'static>> = previous.into_iter().rev().map(text).collect();
            let left: String = current
                .chars()
                .rev()
                .take(input.width.saturating_sub(2) as usize)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let mut suffix = after.split('\n');
            let right: String = suffix
                .next()
                .unwrap_or("")
                .chars()
                .take(input.width as usize)
                .collect();
            rows.push(label(format!("{left}▏{right}"), theme));
            rows.extend(suffix.take(input.height as usize).map(text));
            frame.render_widget(Paragraph::new(rows).wrap(Wrap { trim: false }), input);
            frame.render_widget(
                Paragraph::new(text(if raw {
                    "Ctrl-S save · Esc discard"
                } else {
                    "Enter accept · Esc cancel"
                })),
                controls,
            );
        }
        Modal::Confirm {
            question, scroll, ..
        } => {
            let popup = center(body, 86, 15);
            frame.render_widget(Clear, popup);
            let block = dialog(" Confirm ", theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let [question_area, controls] =
                Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
            let paragraph = Paragraph::new(question.lines().map(text).collect::<Vec<_>>())
                .wrap(Wrap { trim: false });
            let max = paragraph
                .line_count(question_area.width)
                .saturating_sub(question_area.height as usize)
                .min(u16::MAX as usize) as u16;
            *scroll = (*scroll).min(max);
            frame.render_widget(paragraph.scroll((*scroll, 0)), question_area);
            app.navigation
                .hits
                .add(question_area, Target::ModalScroll(max));
            frame.render_widget(
                Paragraph::new(label("y confirm · n/Esc cancel · arrows scroll", theme)),
                controls,
            );
        }
        Modal::Palette(palette) => {
            let popup = center(body, 56, 14);
            frame.render_widget(Clear, popup);
            let mut rows = vec![
                label(format!("› {}", palette.query), theme),
                Line::default(),
            ];
            let matches = palette.matches();
            if matches.is_empty() {
                rows.push(text("No matching view"));
            }
            for (i, view) in matches.into_iter().enumerate() {
                if let Some(rect) = row_rect(
                    dialog(" Go to · Enter / Esc ", theme).inner(popup),
                    rows.len(),
                    0,
                ) {
                    app.navigation.hits.add(rect, Target::Navigate(view));
                }
                let line = text(format!(
                    "{} {}",
                    if i == palette.cursor { ">" } else { " " },
                    view.name()
                ));
                rows.push(if i == palette.cursor {
                    line.style(theme.selected())
                } else {
                    line
                });
            }
            frame.render_widget(
                Paragraph::new(rows).block(dialog(" Go to · Enter / Esc ", theme)),
                popup,
            );
        }
    }
}
fn render_rail(frame: &mut Frame<'_>, area: Rect, app: &mut App, theme: Theme) {
    let rows: Vec<_> = View::ALL
        .into_iter()
        .map(|view| {
            let selected = app.navigation.rail() && view.index() == app.navigation.cursor;
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
            if selected {
                line.style(theme.selected())
            } else if active {
                line.style(theme.accent())
            } else {
                line
            }
        })
        .collect();
    let inner = pane_body(
        frame,
        area,
        &mut app.navigation,
        theme,
        Pane::Rail,
        "Navigation",
    );
    let (body, offset) = scroll_rows(
        frame,
        &mut app.navigation,
        theme,
        Pane::Rail,
        inner,
        Paragraph::new(rows),
        ScrollMode::Manual,
    );
    for view in View::ALL {
        if let Some(rect) = row_rect(body, view.index(), offset) {
            app.navigation.hits.add(rect, Target::Navigate(view));
        }
    }
}
fn render_library(frame: &mut Frame<'_>, area: Rect, app: &mut App, kind: Kind, theme: Theme) {
    let title = match kind {
        Kind::Models => format!(" Models · {:?} ", app.library.shelf),
        Kind::Devices => " Devices ".into(),
        Kind::Profiles => " Profiles ".into(),
        Kind::Config => format!(" Config · {} ", app.library.editor.label()),
    };
    let block = pane(title, theme, false);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let collection = app.library.rows(kind);
    let mut header = match kind {
        Kind::Models => vec![
            text(if inner.width >= 30 {
                ""
            } else {
                "m shelf · j/k select"
            }),
            text("l load · p run · s load for Serve · d pull · v verify"),
            text("a alias · X preview · x remove"),
        ],
        Kind::Devices => vec![
            text("j/k select · Enter refresh · e check model policy"),
            text("l load on selected physical device · no inferred model eligibility"),
        ],
        Kind::Profiles => vec![
            text(format!(
                "Active: {}",
                app.document.snapshot().target.label()
            )),
            text("j/k select · Enter open/editor · c create"),
            text("Open is inert; activate, copy or remove in Config."),
        ],
        Kind::Config => vec![
            text(format!(
                "Active: {} · Editing: {}",
                app.document.snapshot().target.label(),
                app.library.editor.path.display()
            )),
            text("j/k select · / search · Enter edit · z unset · t validate"),
            text("a activate · y copy · x remove · v raw view · w raw edit"),
            text(format!(
                "Search: {} · {}",
                if app.library.query.is_empty() {
                    "all"
                } else {
                    &app.library.query
                },
                if app.library.raw {
                    "Raw YAML view"
                } else {
                    "Typed settings"
                }
            )),
        ],
    };
    if app.document.snapshot().offline && kind == Kind::Models {
        header.push(text("Offline · acquisition disabled"));
    }
    if kind == Kind::Config
        && app
            .library
            .observed()
            .is_some_and(|obs| !obs.snapshot.exists())
    {
        header.push(text("Built-in defaults (no document created)"));
    }
    let status = app.library.notice(kind).unwrap_or("Owner observations");
    header.push(text(format!(
        "{} entries · {status}",
        collection.rows.len()
    )));
    let reserved = if inner.height >= 17 { 13 } else { 3 };
    let height = (header.len() as u16).min(inner.height.saturating_sub(reserved));
    let [head, body] =
        Layout::vertical([Constraint::Length(height), Constraint::Min(1)]).areas(inner);
    frame.render_widget(Paragraph::new(header), head);
    if kind == Kind::Models && head.height > 0 && head.width >= 30 {
        let mut x = head.x;
        for (shelf, title) in [
            (Shelf::Cached, " Cached "),
            (Shelf::Catalog, " Catalog "),
            (Shelf::Partial, " Partial "),
        ] {
            let width = title.len() as u16;
            let rect = Rect::new(x, head.y, width, 1);
            frame.render_widget(
                Paragraph::new(text(title)).style(if app.library.shelf == shelf {
                    theme.selected()
                } else {
                    theme.raised()
                }),
                rect,
            );
            app.navigation.hits.add(rect, Target::Shelf(shelf));
            x += width + 1;
        }
    }
    let (list, detail, track) = app
        .navigation
        .workspace
        .inner(body, app.navigation.view.index());
    divider(frame, &mut app.navigation, theme, track);
    let mut lines = Vec::new();
    let mut group = "";
    let mut selected_line = 0;
    let mut keys = Vec::new();
    for row in &collection.rows {
        if let Key::Setting(key) = &row.key {
            let next = super::settings::group(key);
            if next != group {
                group = next;
                lines.push(label(group, theme));
            }
        }
        let selected = collection.selected.as_ref() == Some(&row.key);
        if selected {
            selected_line = lines.len();
        }
        keys.push((lines.len(), &row.key));
        let line = text(format!(
            "{} {}",
            if selected { ">" } else { " " },
            row.title
        ));
        lines.push(if selected {
            line.style(theme.selected())
        } else {
            line
        });
    }
    if !list.is_empty() {
        let inner = pane_body(
            frame,
            list,
            &mut app.navigation,
            theme,
            Pane::Main,
            "Entries",
        );
        let reveal = app
            .navigation
            .workspace
            .take_reveal(app.navigation.view.index())
            .then_some(selected_line);
        let (visible, offset) = scroll_rows(
            frame,
            &mut app.navigation,
            theme,
            Pane::Main,
            inner,
            Paragraph::new(lines),
            reveal.into(),
        );
        for (index, key) in keys {
            if let Some(rect) = row_rect(visible, index, offset) {
                app.navigation
                    .hits
                    .add(rect, Target::Row(kind, key.clone()));
            }
        }
    }
    let mut details = Vec::new();
    if kind == Kind::Config && app.library.raw {
        if let Some(observed) = app.library.observed() {
            match observed.snapshot.raw() {
                Ok(yaml) => details.extend(yaml.lines().map(text)),
                Err(error) => details.push(text(error.to_string())),
            }
        }
    } else if let Some(row) = collection.selected() {
        details.push(label(&row.title, theme));
        details.extend(row.details.iter().map(text));
        if let Some((key, result)) = &app.library.detail
            && key == &row.key
        {
            details.push(Line::default());
            details.extend(result.lines().map(text));
        }
    } else {
        details.push(text(if kind == Kind::Config {
            "No typed selection; w raw edit, v raw view"
        } else {
            "No observed selection"
        }));
    }
    if !detail.is_empty() {
        let inner = pane_body(
            frame,
            detail,
            &mut app.navigation,
            theme,
            Pane::Detail,
            "Selected",
        );
        scroll_rows(
            frame,
            &mut app.navigation,
            theme,
            Pane::Detail,
            inner,
            Paragraph::new(details).wrap(Wrap { trim: false }),
            ScrollMode::Manual,
        );
    }
}

fn render_logs(frame: &mut Frame<'_>, area: Rect, app: &mut App, theme: Theme) {
    let inner = pane_body(
        frame,
        area,
        &mut app.navigation,
        theme,
        Pane::Main,
        "Operational logs",
    );
    let [head, body] =
        Layout::vertical([Constraint::Length(3.min(inner.height)), Constraint::Min(0)])
            .areas(inner);
    frame.render_widget(
        Paragraph::new(vec![
            text(app.logs.heading()),
            text("/ search · l level · f component · p pause/follow · PgUp/PgDn scroll"),
            text(format!(
                "Local operational observations; {} older entries dropped. No model output.",
                app.logs.dropped()
            )),
        ]),
        head,
    );
    let rows = app.logs.lines();
    let lines = if rows.is_empty() {
        vec![text("No notifications received.")]
    } else {
        rows.iter().map(text).collect()
    };
    scroll_rows(
        frame,
        &mut app.navigation,
        theme,
        Pane::Main,
        body,
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        if app.logs.following() {
            ScrollMode::Follow
        } else {
            ScrollMode::Manual
        },
    );
}
fn render_workspace(frame: &mut Frame<'_>, area: Rect, app: &mut App, theme: Theme) {
    match app.navigation.view {
        View::Models => {
            render_library(frame, area, app, Kind::Models, theme);
            return;
        }
        View::Devices => {
            render_library(frame, area, app, Kind::Devices, theme);
            return;
        }
        View::Profiles => {
            render_library(frame, area, app, Kind::Profiles, theme);
            return;
        }
        View::Config => {
            render_library(frame, area, app, Kind::Config, theme);
            return;
        }
        View::Logs => {
            render_logs(frame, area, app, theme);
            return;
        }

        _ => {}
    }
    let Content {
        rows,
        details,
        owners,
    } = content(app, theme);
    let view = app.navigation.view;
    let (main, detail, track) = if app.navigation.workspace.has_detail {
        app.navigation
            .workspace
            .inner(area, app.navigation.view.index())
    } else {
        (area, Rect::default(), None)
    };
    divider(frame, &mut app.navigation, theme, track);
    if !main.is_empty() {
        let title = app.navigation.view.name();
        let inner = pane_body(frame, main, &mut app.navigation, theme, Pane::Main, title);
        let reveal = if app
            .navigation
            .workspace
            .take_reveal(app.navigation.view.index())
        {
            owners
                .iter()
                .find(|(_, owner)| Some(owner) == app.selected.as_ref())
                .map(|(line, _)| *line)
        } else {
            None
        };
        let (visible, offset) = scroll_rows(
            frame,
            &mut app.navigation,
            theme,
            Pane::Main,
            inner,
            if view == View::Running {
                Paragraph::new(rows)
            } else {
                Paragraph::new(rows).wrap(Wrap { trim: false })
            },
            reveal.into(),
        );
        for (line, owner) in owners {
            if let Some(rect) = row_rect(visible, line, offset) {
                app.navigation.hits.add(rect, Target::Owner(owner));
            }
        }
    }
    if !detail.is_empty() {
        let inner = pane_body(
            frame,
            detail,
            &mut app.navigation,
            theme,
            Pane::Detail,
            if view == View::Home {
                "Response"
            } else {
                "Selected"
            },
        );
        let follow = app.response.is_some()
            && app
                .navigation
                .workspace
                .follow_output(app.navigation.view.index());
        scroll_rows(
            frame,
            &mut app.navigation,
            theme,
            Pane::Detail,
            inner,
            Paragraph::new(details).wrap(Wrap { trim: false }),
            if follow {
                ScrollMode::Follow
            } else {
                ScrollMode::Manual
            },
        );
    }
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
        assert_eq!(nav.key(key(KeyCode::Char('5'))), Action::Open);
        assert_eq!(nav.view(), View::Downloads);
        nav.workspace
            .viewport(View::Downloads.index(), Pane::Main)
            .measure(10, 3);
        assert_eq!(
            nav.key(key(KeyCode::Down)),
            Action::Scroll(Pane::Main, Motion::Step(1))
        );
        nav.scroll(Pane::Main, Motion::Step(1));
        nav.key(key(KeyCode::Char(':')));
        assert_eq!(nav.key(key(KeyCode::Char('q'))), Action::Redraw);
        assert!(matches!(&nav.modal, Modal::Palette(p) if p.query=="q"));
        nav.key(key(KeyCode::Backspace));
        for c in "logs".chars() {
            nav.key(key(KeyCode::Char(c)));
        }
        assert_eq!(nav.key(key(KeyCode::Enter)), Action::Open);
        assert_eq!(nav.view(), View::Logs);
        nav.key(key(KeyCode::Char('5')));
        assert_eq!(
            nav.workspace
                .viewport(View::Downloads.index(), Pane::Main)
                .offset,
            1
        );
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
        assert!(matches!(&nav.modal, Modal::Form { .. }));
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
            logs: super::super::logs::Logs::default(),
            #[cfg(feature = "web")]
            web: None,
            notice: None,
            color: false,
            appearance: super::super::theme::Environment::detect(),
            document: crate::control_host::Selection::new(
                super::super::work::Document::test_default(),
            ),
            job: None,
            actions: crate::control_host::Actions::default(),
            active: None,
            selected: None,
            response: None,
            library: super::super::library::Library::new(
                super::super::work::Document::test_default().target,
            ),
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
            app.navigation.focus(Pane::Rail);
            terminal.draw(|frame| render(frame, &mut app)).unwrap();
            let text = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();
            assert!(text.contains("Navigation"));
            app.navigation.focus(Pane::Main);
        }
        app.color = true;
        app.appearance =
            super::super::theme::Environment::from_hints(Some("truecolor"), None, None);
        for (mode, expected) in [
            (ThemeMode::Dark, (23, 26, 32)),
            (ThemeMode::Light, (255, 255, 255)),
        ] {
            let mut doc = app.document.snapshot();
            doc.ui.theme = Some(mode);
            app.document.update(doc);
            let mut terminal = Terminal::new(TestBackend::new(100, 28)).unwrap();
            terminal.draw(|frame| render(frame, &mut app)).unwrap();
            let cell = &terminal.backend().buffer()[(0, 0)];
            assert_eq!(
                cell.bg,
                ratatui::style::Color::Rgb(expected.0, expected.1, expected.2)
            );
        }
        runtime.shutdown().unwrap();
    }
}
