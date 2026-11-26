//! Cell-native navigation and rendering. No execution or filesystem effects.
use super::{
    App,
    choice::Picker,
    controls::{Bound, Menu, Scope},
    editor::{self, Editor},
    format,
    input::{Hits, Intent, Pointer, Target},
    layout::{Anchor, Motion, Pane, Scrollbar, SplitKind, Splitter, Viewport, Workspace},
    library::{Key, Kind, Shelf},
    settings::Snapshot,
    theme::Theme,
    work::{Owner, Request},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;
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
    pub fn library(self) -> Option<Kind> {
        match self {
            Self::Models => Some(Kind::Models),
            Self::Devices => Some(Kind::Devices),
            Self::Profiles => Some(Kind::Profiles),
            Self::Config => Some(Kind::Config),
            _ => None,
        }
    }
    pub fn searchable(self) -> bool {
        self.library().is_some() || matches!(self, Self::Logs | Self::Running)
    }
    fn index(self) -> usize {
        self as usize
    }
    fn horizontal_default(self) -> u16 {
        if self == Self::Running { 380 } else { 550 }
    }
    pub fn name(self) -> &'static str {
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
    scroll: usize,
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
            _ => 32 * 1024,
        }
    }
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
    Help {
        scroll: u16,
    },
    Details {
        title: String,
        lines: Vec<String>,
        scroll: u16,
    },
    Menu(Menu),
    Choice(Picker),
    Theme {
        file: Snapshot,
        choice: ThemeMode,
    },
    #[cfg(feature = "web")]
    WebUrl {
        url: String,
        scroll: u16,
    },
    Form {
        form: Form,
        editor: Editor,
    },
    Confirm {
        question: String,
        action: Confirmation,
        scroll: u16,
    },
}
impl Modal {
    fn text_scroll(&mut self) -> Option<&mut u16> {
        match self {
            Self::Help { scroll } | Self::Details { scroll, .. } | Self::Confirm { scroll, .. } => {
                Some(scroll)
            }
            #[cfg(feature = "web")]
            Self::WebUrl { scroll, .. } => Some(scroll),
            _ => None,
        }
    }
}
#[derive(Debug, Default)]
pub(super) struct Navigation {
    view: View,
    cursor: usize,
    workspace: Workspace,
    hits: Hits,
    pointer: Pointer,
    too_small: bool,
    split_visible: bool,
    search: Option<Editor>,
    search_width: u16,
    library_anchors: [Anchor<Key>; 6],
    running_anchor: Anchor<Owner>,
    log_anchor: Anchor<u64>,
    output_anchors: [Anchor<usize>; 2],
    running_response: bool,
    has_response: bool,
    modal: Modal,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    Choose,
    Create,
    Copy,
    Remove,
    Activate,
    Validate,
    Unset,
    RawView,
    RawEdit,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LogAction {
    Follow,
    Level,
    Component,
    PickLevel,
    PickComponent,
    SetLevel(Option<crate::control_host::logs::Level>),
    SetComponent(Option<crate::control_host::logs::Component>),
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Action {
    Redraw,
    Open,
    ShowDetails,
    OpenActions,
    OpenSessionMenu,
    RowActions(Kind, Key),
    OwnerActions(Owner),
    Search(SearchEdit),
    Bound(Bound),
    FollowOutput,
    Refresh,
    Quit,
    Load,
    Run,
    Unload,
    Start,
    Stop,
    Cancel,
    MoveOwner(i8),
    MoveLog(i8),
    SelectLog(u64),
    ShowLog(Option<u64>),
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

/// Text input is interpreted only by the inline filter, never as an
/// operation shortcut. Actual query storage remains with each observed list.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum SearchEdit {
    Focus,
    Insert(String),
    Delete(bool),
    Move(bool, bool),
    Edge(bool, bool),
    Point {
        start: usize,
        end: usize,
        column: u16,
        extend: bool,
    },
    Clear,
    Escape,
    DragEdge(i16, u16),
}

fn move_text_scroll(scroll: &mut u16, key: KeyCode) {
    match key {
        KeyCode::Up | KeyCode::PageUp => {
            *scroll = scroll.saturating_sub(if key == KeyCode::PageUp { 8 } else { 1 })
        }
        KeyCode::Down | KeyCode::PageDown => {
            *scroll = scroll.saturating_add(if key == KeyCode::PageDown { 8 } else { 1 })
        }
        KeyCode::Home => *scroll = 0,
        KeyCode::End => *scroll = u16::MAX,
        _ => {}
    }
}
impl Navigation {
    pub fn view(&self) -> View {
        self.view
    }
    pub fn inline_search(&self) -> bool {
        self.idle() && self.view.searchable() && self.search.is_some()
    }
    pub fn edit_search(&mut self, value: &mut String, edit: SearchEdit) -> bool {
        if !self.idle() || !self.view.searchable() {
            return false;
        }
        if self.search.is_none() {
            self.focus(Pane::Main);
            self.search = Some(Editor::new(value, value.len()));
        }
        let editor = self.search.as_mut().expect("focused search");
        let escape = edit == SearchEdit::Escape;
        match edit {
            SearchEdit::Focus => {}
            SearchEdit::Insert(text) => {
                if text.chars().any(char::is_control) {
                    return false;
                }
                if !editor.replace(value, &text, 256) {
                    return false;
                }
            }
            SearchEdit::Delete(backward) => editor.delete(value, backward),
            SearchEdit::Move(right, extend) => editor.move_horizontal(value, right, extend),
            SearchEdit::Edge(end, extend) => editor.edge(value, end, true, extend),
            SearchEdit::Point {
                start,
                end,
                column,
                extend,
            } => {
                editor.measure(value, self.search_width.max(1), 1);
                editor.point(value, start..end, column, extend);
            }
            SearchEdit::DragEdge(delta, column) => editor.drag_edge(value, delta, column),
            SearchEdit::Clear | SearchEdit::Escape => editor.clear(value),
        }
        if escape {
            self.search = None;
        }
        true
    }
    pub fn details(&mut self, title: String, lines: Vec<String>) {
        self.pointer.cancel();
        self.search = None;
        self.modal = Modal::Details {
            title,
            lines,
            scroll: 0,
        };
    }
    pub fn menu(&mut self, menu: Menu) {
        self.pointer.cancel();
        self.search = None;
        self.modal = Modal::Menu(menu);
    }
    pub fn following_output(&self) -> bool {
        self.workspace.follow_output(self.view.index())
    }
    pub fn toggle_output_follow(&mut self) {
        if !self.has_response || !matches!(self.view, View::Home | View::Running) {
            return;
        }
        if self.view == View::Running {
            self.running_response = true;
        }
        self.focus(Pane::Detail);
        self.workspace
            .set_follow_output(self.view.index(), !self.following_output());
    }
    fn accept_modal(&mut self) -> Option<Action> {
        self.pointer.cancel();
        match std::mem::take(&mut self.modal) {
            Modal::Form { form, .. } => Some(Action::Submit(form)),
            Modal::Choice(picker) => Some(picker.apply()),
            Modal::Theme { file, choice } => Some(Action::Submit(Form {
                kind: FormKind::ConfigValue {
                    file,
                    key: "ui.theme".into(),
                },
                value: choice.label().into(),
            })),
            Modal::Confirm { action, .. } => Some(Action::Confirmed(action)),
            modal => {
                self.modal = modal;
                None
            }
        }
    }
    fn dismiss_modal(&mut self) -> Action {
        self.pointer.cancel();
        self.modal = Modal::None;
        Action::Redraw
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
        self.pointer.cancel();
        self.search = None;
        self.modal = Modal::Form {
            editor: Editor::new(&value, cursor),
            form: Form { kind, value },
        };
    }
    pub fn restore_form(&mut self, form: Form) {
        match form.kind {
            FormKind::ConfigValue { file, key } => self.edit_setting(file, key, form.value),
            kind => self.form_value(kind, form.value),
        }
    }
    pub fn edit_setting(&mut self, file: Snapshot, key: String, value: String) {
        if key == "ui.theme" {
            let choice = serde_yaml::from_str(&value).unwrap_or_default();
            self.pointer.cancel();
            self.search = None;
            self.modal = Modal::Theme { file, choice };
        } else {
            self.form_value(FormKind::ConfigValue { file, key }, value);
        }
    }
    pub fn pick_setting(
        &mut self,
        file: Snapshot,
        key: String,
        choices: &'static [&'static str],
        effective: &str,
        raw: String,
    ) {
        if let Some(index) = choices.iter().position(|choice| *choice == effective) {
            self.pointer.cancel();
            self.search = None;
            self.modal = Modal::Choice(Picker::setting(file, key, choices, index));
        } else {
            // A custom value (e.g. draft:MODEL) is not coerced into a
            // finite choice; preserve the existing lossless YAML route.
            self.edit_setting(file, key, raw);
        }
    }
    pub fn pick_level(&mut self, level: Option<crate::control_host::logs::Level>) {
        self.pointer.cancel();
        self.search = None;
        self.modal = Modal::Choice(Picker::level(level));
    }
    pub fn pick_component(&mut self, component: Option<crate::control_host::logs::Component>) {
        self.pointer.cancel();
        self.search = None;
        self.modal = Modal::Choice(Picker::component(component));
    }
    fn choice_yaml(&mut self) -> Action {
        if let Modal::Choice(picker) = std::mem::take(&mut self.modal)
            && let Some(form) = picker.into_yaml()
        {
            self.form_value(form.kind, form.value);
        }
        Action::Redraw
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
        self.pointer.cancel();
        self.search = None;
        self.modal = Modal::WebUrl { url, scroll: 0 };
    }
    pub fn confirm(&mut self, question: String, action: Confirmation) {
        self.pointer.cancel();
        self.search = None;
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
        self.search = None;
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
    fn list_slot(kind: Kind, shelf: Shelf) -> usize {
        match kind {
            Kind::Models => match shelf {
                Shelf::Cached => 0,
                Shelf::Catalog => 1,
                Shelf::Partial => 2,
            },
            Kind::Devices => 3,
            Kind::Profiles => 4,
            Kind::Config => 5,
        }
    }
    pub fn shelf_changed(&mut self, shelf: Shelf) {
        self.search = None;
        let view = self.view.index();
        let last = self.library_anchors[Self::list_slot(Kind::Models, shelf)].last();
        self.workspace.viewport(view, Pane::Main).offset = last.unwrap_or(0);
        self.workspace.viewport(view, Pane::Detail).offset = 0;
        if last.is_none() {
            self.workspace.request_reveal(view);
        } else {
            self.workspace.take_reveal(view);
        }
    }
    fn output_visible(&self) -> bool {
        self.has_response
            && (self.view == View::Home || (self.view == View::Running && self.running_response))
    }
    fn viewport(&mut self, pane: Pane) -> &mut Viewport {
        let view = self.view.index();
        if pane == Pane::Detail && self.output_visible() {
            self.workspace.output(view)
        } else {
            self.workspace.viewport(view, pane)
        }
    }
    fn toggle_response(&mut self) {
        if self.view == View::Running && self.has_response {
            self.running_response = !self.running_response;
            self.focus(Pane::Detail);
        }
    }
    pub fn reveal_owner(&mut self, _: usize) {
        self.running_response = false;
        self.reveal_library(0);
    }
    pub fn reveal_library(&mut self, _: usize) {
        self.workspace.request_reveal(self.view.index());
        self.workspace
            .viewport(self.view.index(), Pane::Detail)
            .offset = 0;
    }
    pub fn follow_response(&mut self) {
        self.output_anchors = Default::default();
        self.running_response = true;
        self.has_response = true;
        if matches!(self.view, View::Home | View::Running) {
            self.focus(Pane::Detail);
        }
        for view in [View::Home, View::Running] {
            self.workspace.set_follow_output(view.index(), true);
            *self.workspace.output(view.index()) = Viewport::default();
        }
    }
    pub fn scroll(&mut self, pane: Pane, motion: Motion, shelf: Shelf) -> bool {
        let output = self.output_visible();
        let viewport = self.viewport(pane);
        viewport.scroll(motion);
        let at_end = viewport.offset == viewport.max;
        if pane == Pane::Detail && output {
            self.output_anchors[usize::from(self.view == View::Running)] = Anchor::default();
            self.workspace.set_follow_output(self.view.index(), at_end);
        }
        if pane == Pane::Main {
            if let Some(kind) = self.view.library() {
                self.library_anchors[Self::list_slot(kind, shelf)] = Anchor::default();
            } else if self.view == View::Running {
                self.running_anchor = Anchor::default();
            } else if self.view == View::Logs {
                self.log_anchor = Anchor::default();
            }
        }
        // Wheel/manual scrollbar deliberately does not reveal the selection.
        if pane == Pane::Main {
            self.workspace.take_reveal(self.view.index());
        }
        at_end
    }
    pub fn mouse(&mut self, event: MouseEvent) -> Option<Action> {
        match self.pointer.event(event, &self.hits)? {
            Intent::EditPoint {
                start,
                end,
                column,
                extend,
            } => {
                if let Modal::Form { form, editor } = &mut self.modal {
                    editor.point(&form.value, start..end, column, extend);
                    Some(Action::Redraw)
                } else if self.idle() && self.view.searchable() {
                    Some(Action::Search(SearchEdit::Point {
                        start,
                        end,
                        column,
                        extend,
                    }))
                } else {
                    None
                }
            }
            Intent::EditEdge(delta, column) => {
                if let Modal::Form { form, editor } = &mut self.modal {
                    editor.drag_edge(&form.value, delta, column);
                    Some(Action::Redraw)
                } else if self.inline_search() {
                    Some(Action::Search(SearchEdit::DragEdge(delta, column)))
                } else {
                    None
                }
            }
            Intent::EditScroll(motion) => {
                if let Modal::Form { editor, .. } = &mut self.modal {
                    editor.scroll(motion);
                    Some(Action::Redraw)
                } else {
                    None
                }
            }
            Intent::ScrollModal(delta, max) => {
                if let Some(scroll) = self.modal.text_scroll() {
                    *scroll = scroll.saturating_add_signed(delta).min(max);
                    Some(Action::Redraw)
                } else if let Modal::Form { editor, .. } = &mut self.modal {
                    editor.scroll(Motion::Step(delta));
                    Some(Action::Redraw)
                } else if let Modal::Choice(picker) = &mut self.modal {
                    picker.scroll = picker.scroll.saturating_add_signed(delta).min(max);
                    picker.move_by(delta as isize);
                    Some(Action::Redraw)
                } else if let Modal::Palette(palette) = &mut self.modal {
                    palette.cursor = palette
                        .cursor
                        .saturating_add_signed(delta as isize)
                        .min(palette.matches().len().saturating_sub(1));
                    Some(Action::Redraw)
                } else if let Modal::Menu(menu) = &mut self.modal {
                    menu.scroll = menu.scroll.saturating_add_signed(delta).min(max);
                    menu.cursor = menu
                        .cursor
                        .saturating_add_signed(delta as isize)
                        .min(menu.buttons.len().saturating_sub(1));
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
            Intent::Context(target) if self.idle() => {
                self.search = None;
                match target {
                    Target::Row(kind, key) | Target::RowMenu(kind, key) => {
                        Some(Action::RowActions(kind, key))
                    }
                    Target::Log(id) | Target::LogDetails(id) => Some(Action::ShowLog(Some(id))),
                    Target::Owner(owner) | Target::OwnerMenu(owner) => {
                        Some(Action::OwnerActions(owner))
                    }
                    _ => None,
                }
            }
            Intent::Click(target) => {
                // A modal registers only its own visible targets. No backdrop
                // click can select an underlying model or approve a mutation.
                if !self.idle() {
                    if target == Target::AcceptModal {
                        return self.accept_modal();
                    }
                    if target == Target::DismissModal {
                        return Some(self.dismiss_modal());
                    }
                    if target == Target::ChoiceYaml
                        && matches!(&self.modal,Modal::Choice(p) if p.is_setting())
                    {
                        return Some(self.choice_yaml());
                    }
                    if let Target::Command(ref bound) = target
                        && matches!(self.modal, Modal::Menu(_))
                    {
                        self.dismiss_modal();
                        return Some(Action::Bound(bound.clone()));
                    }
                    return match (&mut self.modal, target) {
                        (Modal::Palette(_), Target::Navigate(view)) => Some(self.open(view)),
                        (Modal::Choice(picker), Target::Choice(index))
                            if index < picker.count() =>
                        {
                            picker.cursor = index;
                            Some(Action::Redraw)
                        }
                        (Modal::Theme { choice, .. }, Target::Theme(mode)) => {
                            *choice = mode;
                            Some(Action::Redraw)
                        }
                        (
                            Modal::Form {
                                form:
                                    Form {
                                        kind: FormKind::Prompt { chat, .. },
                                        ..
                                    },
                                ..
                            },
                            Target::PromptMode,
                        ) => {
                            *chat = !*chat;
                            Some(Action::Redraw)
                        }
                        _ => None,
                    };
                }
                Some(match target {
                    Target::Navigate(view) => self.open(view),
                    Target::More => {
                        self.search = None;
                        Action::OpenActions
                    }
                    Target::SessionMenu => {
                        self.search = None;
                        Action::OpenSessionMenu
                    }
                    Target::SearchFocus => Action::Search(SearchEdit::Focus),
                    Target::SearchClear => Action::Search(SearchEdit::Clear),
                    Target::RowMenu(kind, key) => {
                        self.focus(Pane::Main);
                        Action::RowActions(kind, key)
                    }
                    Target::OwnerMenu(owner) => {
                        self.focus(Pane::Main);
                        Action::OwnerActions(owner)
                    }
                    Target::Command(bound) => Action::Bound(bound),
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
                    Target::ResponseTab => {
                        self.toggle_response();
                        Action::Redraw
                    }
                    Target::Maximize(pane) => {
                        self.search = None;
                        self.workspace.toggle_maximize(self.view.index(), pane);
                        Action::Redraw
                    }
                    Target::Row(kind, key) => {
                        self.focus(Pane::Main);
                        Action::SelectRow(kind, key)
                    }
                    Target::Log(id) => {
                        self.focus(Pane::Main);
                        Action::SelectLog(id)
                    }
                    Target::LogDetails(id) => Action::ShowLog(Some(id)),
                    Target::Owner(owner) => {
                        self.focus(Pane::Main);
                        Action::SelectOwner(owner)
                    }
                    Target::Shelf(shelf) => {
                        self.focus(Pane::Main);
                        Action::Library(LibraryAction::SelectShelf(shelf))
                    }
                    Target::Help => {
                        self.search = None;
                        self.modal = Modal::Help { scroll: 0 };
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
            Modal::Form { form, editor } => {
                if value.chars().any(|c| {
                    c.is_control() && !(form.multiline() && matches!(c, '\n' | '\r' | '\t'))
                }) {
                    return false;
                }
                let limit = form.limit();
                editor.replace(&mut form.value, value, limit)
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
        self.search = None;
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
                Char('q')
                    if self.idle()
                        && (key.modifiers.contains(KeyModifiers::CONTROL)
                            || !self.view.searchable()) =>
                {
                    Action::Quit
                }
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
            Modal::Help { .. } | Modal::Details { .. } => {
                if matches!(key.code, Esc | Enter) {
                    return self.dismiss_modal();
                }
                if let Some(scroll) = self.modal.text_scroll() {
                    move_text_scroll(scroll, key.code);
                }
                return Action::Redraw;
            }
            #[cfg(feature = "web")]
            Modal::WebUrl { .. } => {
                if matches!(key.code, Esc | Enter) {
                    return self.dismiss_modal();
                }
                if let Some(scroll) = self.modal.text_scroll() {
                    move_text_scroll(scroll, key.code);
                }
                return Action::Redraw;
            }
            Modal::Menu(menu) => {
                match key.code {
                    Esc | F(4) | F(10) => {
                        return self.dismiss_modal();
                    }
                    Enter if key.kind == crossterm::event::KeyEventKind::Press => {
                        if let Some(bound) = menu.selected() {
                            self.dismiss_modal();
                            return Action::Bound(bound);
                        }
                    }
                    Up | Char('k') => menu.cursor = menu.cursor.saturating_sub(1),
                    Down | Char('j') => {
                        menu.cursor = (menu.cursor + 1).min(menu.buttons.len().saturating_sub(1))
                    }
                    Home => menu.cursor = 0,
                    End => menu.cursor = menu.buttons.len().saturating_sub(1),
                    PageUp => menu.cursor = menu.cursor.saturating_sub(8),
                    PageDown => {
                        menu.cursor = (menu.cursor + 8).min(menu.buttons.len().saturating_sub(1))
                    }
                    _ => {}
                }
                return Action::Redraw;
            }
            Modal::Choice(picker) => {
                match key.code {
                    Esc => return self.dismiss_modal(),
                    Enter if key.kind == crossterm::event::KeyEventKind::Press => {
                        return self.accept_modal().expect("choice submit");
                    }
                    F(2) if picker.is_setting() => return self.choice_yaml(),
                    Up | Left | Char('k') => picker.move_by(-1),
                    Down | Right | Char('j') => picker.move_by(1),
                    Home => picker.cursor = 0,
                    End => picker.cursor = picker.count() - 1,
                    PageUp => picker.move_by(-8),
                    PageDown => picker.move_by(8),
                    _ => {}
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
                        return self.accept_modal().expect("theme submit");
                    }
                    Up | Left | Char('k') => *choice = MODES[index.saturating_sub(1)],
                    Down | Right | Char('j') => *choice = MODES[(index + 1).min(2)],
                    _ => {}
                }
                return Action::Redraw;
            }
            Modal::Form { form, editor } => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                let extend = key.modifiers.contains(KeyModifiers::SHIFT);
                let raw = matches!(form.kind, FormKind::RawYaml { .. });
                if (key.code == Enter && !raw) || (ctrl && key.code == Char('s') && raw) {
                    return self.accept_modal().expect("form submit");
                }
                let limit = form.limit();
                match key.code {
                    Esc => return self.dismiss_modal(),
                    Backspace => editor.delete(&mut form.value, true),
                    Delete => editor.delete(&mut form.value, false),
                    Left | Right => editor.move_horizontal(&form.value, key.code == Right, extend),
                    Home | End => editor.edge(&form.value, key.code == End, ctrl, extend),
                    Up | Down if form.multiline() => {
                        editor.move_vertical(&form.value, key.code == Down, extend)
                    }
                    PageUp | PageDown => {
                        editor.scroll(Motion::Page(if key.code == PageUp { -1 } else { 1 }))
                    }
                    Char('u') if ctrl => editor.clear(&mut form.value),
                    Char('a' | 'e') if ctrl => {
                        editor.edge(&form.value, key.code == Char('e'), true, extend)
                    }
                    Char('r') if ctrl => {
                        if let FormKind::Prompt { chat, .. } = &mut form.kind {
                            *chat = !*chat;
                        }
                    }
                    Enter if raw => {
                        editor.replace(&mut form.value, "\n", limit);
                    }
                    Char('j') if ctrl && form.multiline() => {
                        editor.replace(&mut form.value, "\n", limit);
                    }
                    Tab if raw => {
                        editor.replace(&mut form.value, "  ", limit);
                    }
                    Char(c)
                        if !c.is_control()
                            && !key
                                .modifiers
                                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                    {
                        editor.replace(&mut form.value, c.encode_utf8(&mut [0; 4]), limit);
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
                        return self.accept_modal().expect("confirmation submit");
                    }
                    Esc | Char('n' | 'N') => self.modal = Modal::None,
                    _ => {}
                }
                return Action::Redraw;
            }
            Modal::None => {}
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        if ctrl && key.code == Char('q') {
            return Action::Quit;
        }
        if self.view.searchable() {
            let editing = self.search.is_some();
            let edit = match key.code {
                Char('f') if ctrl && !alt => Some(SearchEdit::Focus),
                Char('u') if ctrl && editing => Some(SearchEdit::Clear),
                Char('a' | 'e') if ctrl && editing => {
                    Some(SearchEdit::Edge(key.code == Char('e'), shift))
                }
                Esc if editing => Some(SearchEdit::Escape),
                Left | Right if editing && !ctrl && !alt => {
                    Some(SearchEdit::Move(key.code == Right, shift))
                }
                Home | End if editing && !alt => Some(SearchEdit::Edge(key.code == End, shift)),
                Backspace | Delete if !alt && !ctrl => {
                    Some(SearchEdit::Delete(key.code == Backspace))
                }
                Char(c) if !ctrl && !alt && !c.is_control() => {
                    Some(SearchEdit::Insert(c.to_string()))
                }
                _ => None,
            };
            if let Some(edit) = edit {
                return Action::Search(edit);
            }
            if editing && key.code == Tab {
                self.search = None;
                return Action::Redraw;
            }
            // Arrow navigation, menu, Alt commands and explicit view changes
            // leave the inline field but retain the live filter.
            self.search = None;
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
                self.workspace
                    .adjust(kind, delta, self.view.horizontal_default());
                return Action::Redraw;
            }
        }
        if key.modifiers.contains(KeyModifiers::ALT) && matches!(key.code, Left | Right) {
            self.workspace.adjust(
                SplitKind::Rail,
                if key.code == Left { -1 } else { 1 },
                self.view.horizontal_default(),
            );
            return Action::Redraw;
        }
        if ctrl && matches!(key.code, Char(_)) {
            return Action::Redraw;
        }
        match key.code {
            F(1) => {
                self.modal = Modal::Help { scroll: 0 };
                Action::Redraw
            }
            F(3) if self.view == View::Running && self.has_response => {
                self.toggle_response();
                Action::Redraw
            }
            F(4) => Action::OpenActions,
            F(10) => Action::OpenSessionMenu,
            F(9) => Action::FollowOutput,
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
            Enter if pane == Pane::Detail => Action::OpenActions,
            Char('q') => Action::Quit,
            Char('T') => Action::Appearance,
            Char('D')
                if matches!(
                    self.view,
                    View::Models | View::Devices | View::Config | View::Profiles | View::Running
                ) =>
            {
                Action::ShowDetails
            }
            Char(':') => {
                self.modal = Modal::Palette(Palette::default());
                Action::Redraw
            }
            Char('?') => {
                self.modal = Modal::Help { scroll: 0 };
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
            Char('e') if !self.rail() && self.view == View::Config => {
                Action::Settings(SettingsAction::Choose)
            }
            Char('p') if !self.rail() && self.view == View::Logs => Action::Logs(LogAction::Follow),
            Char('l') if !self.rail() && self.view == View::Logs => {
                Action::Logs(LogAction::PickLevel)
            }
            Char('f') if !self.rail() && self.view == View::Logs => {
                Action::Logs(LogAction::PickComponent)
            }
            Char('L') if !self.rail() && self.view == View::Logs => Action::Logs(LogAction::Level),
            Char('F') if !self.rail() && self.view == View::Logs => {
                Action::Logs(LogAction::Component)
            }
            Char('b') if alt || (!self.rail() && matches!(self.view, View::Home | View::Serve)) => {
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
            Enter if !self.rail() && matches!(self.view, View::Models | View::Devices) => {
                Action::OpenActions
            }
            Char('i') if !self.rail() && matches!(self.view, View::Models | View::Devices) => {
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
            Enter if !self.rail() && self.view == View::Running => Action::OpenActions,
            Enter if !self.rail() && self.view == View::Logs => Action::ShowLog(None),
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
            Up if self.view == View::Logs => Action::MoveLog(-1),
            Down if self.view == View::Logs => Action::MoveLog(1),
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
                    "Operation {} · {:?} · {:?}",
                    download.id, download.status, download.phase
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
                || {
                    format!(
                        "{} · total unknown (max {})",
                        format::bytes(current),
                        format::bytes(maximum)
                    )
                },
                |total| format!("{} / {}", format::bytes(current), format::bytes(total)),
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
            "Cached: {} · resumed: {}",
            format::bytes(download.cached_bytes),
            format::bytes(download.resumed_bytes)
        )));
        rows.push(text(format!(
            "Last observed checkpoint: {}",
            format::bytes(download.checkpoint_bytes)
        )));
        if let Some(total) = download.total_expected_bytes.filter(|total| *total != 0) {
            let percent = (download.total_bytes as u128 * 100 / total as u128).min(100);
            rows.push(text(format!("Progress: {percent}%")));
        }
        if download.status == DownloadStatus::Running
            && !download.cancel_requested
            && let Some(rate) = download.logical_bytes_per_second.filter(|rate| *rate > 0)
        {
            rows.push(text(format!("Accepted: {}/s", format::bytes(rate))));
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
fn content(app: &App, theme: Theme, detail: Rect) -> Content {
    let runtime = &app.snapshot.runtime;
    let mut rows = Vec::new();
    let mut detail_at = None;
    let mut owners = Vec::new();
    match app.navigation.view {
        View::Home => {
            rows.push(label("Local runtime", theme));
            let count = runtime.status.loaded_owners;
            rows.push(
                text(if count == 0 {
                    "No models loaded".into()
                } else {
                    format!("{count} model{} loaded", if count == 1 { "" } else { "s" })
                })
                .style(if runtime.status.failed {
                    theme.danger()
                } else if count > 0 {
                    theme.success()
                } else {
                    theme.muted()
                }),
            );
            if runtime.status.failed {
                rows.push(text("Runtime failed · see Logs").style(theme.danger()));
            } else if !matches!(
                runtime.status.phase,
                zerogravity::diagnostics::RuntimePhase::Empty
                    | zerogravity::diagnostics::RuntimePhase::Ready
            ) {
                rows.push(text(format!("{:?}", runtime.status.phase)).style(theme.info()));
            }
            if let Some(scheduler) = &runtime.scheduler {
                rows.push(
                    text(format!(
                        "{} active · {} queued",
                        scheduler.counters.active, scheduler.queue.pending
                    ))
                    .style(theme.muted()),
                );
            }
            rows.push(Line::default());
            rows.push(
                text(match &app.selected {
                    Some(owner) => format!("Quick run with {}", owner.alias),
                    None => "Quick run: choose a local or cached model.".into(),
                })
                .style(theme.muted()),
            );
            if app.snapshot.serving != ServeSummary::Unavailable {
                rows.push(Line::default());
                rows.push(label("Inference API", theme));
                rows.push(
                    text(serve(app.snapshot.serving))
                        .style(serve_style(app.snapshot.serving, theme)),
                );
            }
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            if app.snapshot.download.is_some() {
                rows.push(Line::default());
                rows.push(label("Downloads", theme));
                rows.push(text(downloads(app)));
            }
        }
        View::Running => {
            rows.push(text(format!(
                "{} registered model owners",
                runtime.status.loaded_owners
            )));
            for model in &runtime.catalog {
                if !app.owner_matches(&model.aliases) {
                    continue;
                }
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
                let busy = if model.admitted_requests > 0 {
                    format!(" · {} admitted", model.admitted_requests)
                } else {
                    String::new()
                };
                let line = text(format!(
                    "{} {} · {} · {:?}{busy}",
                    if selected { ">" } else { "●" },
                    super::performance::model_name(model),
                    super::performance::provider(model),
                    model.lifecycle
                ));
                rows.push(if selected {
                    line.style(theme.selected())
                } else {
                    line
                });
            }
            if !app.running_query.is_empty() && rows.len() == 1 {
                rows.push(text("No matches").style(theme.muted()));
            }
            detail_at = Some(rows.len());
            if let Some(selected) = &app.selected
                && let Some(model) = runtime
                    .catalog
                    .iter()
                    .find(|model| model.registration_id == selected.registration)
            {
                rows.push(label(super::performance::model_name(model), theme));
                rows.extend(super::performance::summary(
                    model,
                    &app.performance,
                    theme,
                    detail.width.saturating_sub(2) as usize,
                    detail.height.saturating_sub(1),
                ));
            }
        }
        View::Serve => {
            rows.push(label("OpenAI-compatible inference API", theme));
            rows.push(
                text(serve(app.snapshot.serving)).style(serve_style(app.snapshot.serving, theme)),
            );
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
            if app.document.snapshot().offline {
                rows.push(text("Offline · new acquisition disabled"));
            }
            download_rows(&mut rows, app);
        }
        View::Profiles | View::Config | View::Logs => unreachable!("dedicated workspace renderer"),
    }
    let details = detail_at.map_or_else(Vec::new, |at| rows.split_off(at));
    Content {
        rows,
        details,
        owners,
    }
}

fn row_rect(area: Rect, index: usize, offset: usize) -> Option<Rect> {
    let y = index.checked_sub(offset)?;
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
fn button(
    frame: &mut Frame<'_>,
    hits: &mut Hits,
    rect: Rect,
    title: &str,
    style: Style,
    target: Option<Target>,
) {
    if rect.is_empty() {
        return;
    }
    frame.render_widget(Paragraph::new(text(title)).style(style), rect);
    if let Some(target) = target {
        hits.add(rect, target);
    }
}
fn control(
    frame: &mut Frame<'_>,
    nav: &mut Navigation,
    theme: Theme,
    rect: Rect,
    title: &str,
    target: Target,
) {
    button(
        frame,
        &mut nav.hits,
        rect,
        title,
        theme.raised(),
        Some(target),
    );
}
/// A context affordance when the selected row menu is not visible.
fn pane_actions(frame: &mut Frame<'_>, nav: &mut Navigation, theme: Theme, area: Rect) {
    if area.is_empty() {
        return;
    }
    let reserved = if !nav.workspace.has_detail || area.width < 20 {
        0
    } else if nav.split_visible {
        5
    } else if area.width >= 33 {
        13
    } else {
        9
    };
    let remaining = area.width.saturating_sub(reserved);
    let title = if remaining >= 26 {
        " Actions… "
    } else {
        " … "
    };
    let width = title.chars().count() as u16;
    if remaining >= width + 8 {
        control(
            frame,
            nav,
            theme,
            Rect::new(area.right() - reserved - width, area.y, width, 1),
            title,
            Target::More,
        );
    }
}
/// Render only buttons which actually fit. The explicit confirm is never
/// the default key action of a destructive confirmation.
fn modal_buttons(
    frame: &mut Frame<'_>,
    hits: &mut Hits,
    theme: Theme,
    area: Rect,
    labels: (&str, &str),
    enabled: bool,
    destructive: bool,
) -> Rect {
    if area.is_empty() {
        return area;
    }
    let (accept, cancel) = labels;
    let (accept, cancel) = if accept.chars().count() + cancel.chars().count() > area.width as usize
    {
        (
            if destructive { " confirm " } else { " Apply " },
            " Cancel ",
        )
    } else {
        (accept, cancel)
    };
    let aw = accept.chars().count() as u16;
    let cw = cancel.chars().count() as u16;
    if aw + cw > area.width {
        return area;
    }
    let accept_rect = Rect::new(area.right() - aw, area.y, aw, 1);
    let cancel_rect = Rect::new(accept_rect.x - cw, area.y, cw, 1);
    button(
        frame,
        hits,
        cancel_rect,
        cancel,
        if destructive {
            theme.selected()
        } else {
            theme.raised()
        },
        Some(Target::DismissModal),
    );
    button(
        frame,
        hits,
        accept_rect,
        accept,
        if !enabled {
            theme.raised().patch(theme.muted())
        } else if destructive {
            theme.raised().patch(theme.danger())
        } else {
            theme.primary()
        },
        enabled.then_some(Target::AcceptModal),
    );
    Rect::new(area.x, area.y, cancel_rect.x - area.x, area.height)
}
fn close_button(frame: &mut Frame<'_>, hits: &mut Hits, theme: Theme, area: Rect) {
    let title = " Esc close ";
    let width = (title.len() as u16).min(area.width);
    button(
        frame,
        hits,
        Rect::new(area.right() - width, area.y, width, area.height),
        title,
        theme.raised(),
        Some(Target::DismissModal),
    );
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
        if !nav.split_visible && area.width >= 33 {
            control(
                frame,
                nav,
                theme,
                Rect::new(toggle_rect.x - focus_width, area.y, focus_width, 1),
                focus,
                Target::Focus(other),
            );
        } else if !nav.split_visible {
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
fn paint_bar(frame: &mut Frame<'_>, theme: Theme, bar: Scrollbar) {
    for y in bar.track.y..bar.track.bottom() {
        let thumb = y >= bar.thumb.y && y < bar.thumb.bottom();
        frame.buffer_mut()[(bar.track.x, y)]
            .set_symbol(if thumb { "┃" } else { "│" })
            .set_style(if thumb {
                theme.muted()
            } else {
                theme.border(false)
            });
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
) -> (Rect, usize) {
    if area.is_empty() {
        return (area, 0);
    }
    let body = Rect::new(
        area.x + u16::from(area.width > 1),
        area.y,
        area.width.saturating_sub(2),
        area.height,
    );
    let track = Rect::new(area.right().saturating_sub(1), area.y, 1, area.height);
    let viewport = nav.viewport(which);
    viewport.measure(
        paragraph
            .line_count(body.width)
            .min(u16::MAX as usize + body.height as usize),
        body.height,
    );
    match mode {
        ScrollMode::Manual => {}
        ScrollMode::Reveal(row) => viewport.reveal(row),
        ScrollMode::Follow => viewport.scroll(Motion::End),
    }
    let viewport = *viewport;
    frame.render_widget(paragraph.scroll((viewport.offset as u16, 0)), body);
    if let Some(bar) = Scrollbar::new(which, track, viewport) {
        paint_bar(frame, theme, bar);
        nav.hits.add(track, Target::Scrollbar(bar));
    }
    (body, viewport.offset)
}

fn render_editor(
    frame: &mut Frame<'_>,
    hits: &mut Hits,
    theme: Theme,
    area: Rect,
    value: &str,
    editor: &mut Editor,
    show_cursor: bool,
) {
    if area.is_empty() {
        return;
    }
    let body = Rect::new(area.x, area.y, area.width.saturating_sub(1), area.height);
    if body.width == 0 {
        return;
    }
    let layout = editor.measure(value, body.width, body.height);
    let viewport = editor.viewport;
    let selected = editor.selection();
    hits.add(area, Target::EditorViewport);
    frame.render_widget(Block::default().style(theme.raised()), area);
    let mut lines = Vec::new();
    for (row, range) in layout
        .rows
        .iter()
        .enumerate()
        .skip(viewport.offset)
        .take(body.height as usize)
    {
        let mut spans = Vec::new();
        for (offset, g) in value[range.clone()].grapheme_indices(true) {
            let at = range.start + offset;
            let style = if selected.contains(&at) {
                theme.selected()
            } else {
                theme.raised()
            };
            spans.push(Span::styled(editor::display(g, body.width), style));
        }
        lines.push(Line::from(spans));
        if let Some(rect) = row_rect(body, row, viewport.offset) {
            hits.add(
                rect,
                Target::EditorRow {
                    start: range.start,
                    end: range.end,
                },
            );
        }
    }
    // Blank space below the draft belongs to its final caret.
    for y in body.y + lines.len() as u16..body.bottom() {
        hits.add(
            Rect::new(body.x, y, body.width, 1),
            Target::EditorRow {
                start: value.len(),
                end: value.len(),
            },
        );
    }
    frame.render_widget(Paragraph::new(lines).style(theme.raised()), body);
    let (row, col) = layout.position(value, editor.caret);
    if show_cursor && let Some(rect) = row_rect(body, row, viewport.offset) {
        frame.set_cursor_position((rect.x + col.min(body.width - 1), rect.y));
    }
    let track = Rect::new(body.right(), body.y, 1, body.height);
    if let Some(bar) = Scrollbar::new(Pane::Main, track, viewport) {
        paint_bar(frame, theme, bar);
        hits.add(track, Target::EditorBar(bar));
    }
}

fn modal_text(
    frame: &mut Frame<'_>,
    hits: &mut Hits,
    content: Rect,
    paragraph: Paragraph<'_>,
    scroll: &mut u16,
) {
    let max = paragraph
        .line_count(content.width)
        .saturating_sub(content.height as usize)
        .min(u16::MAX as usize) as u16;
    *scroll = (*scroll).min(max);
    frame.render_widget(paragraph.scroll((*scroll, 0)), content);
    hits.add(content, Target::ModalScroll(max));
}
fn render_search(frame: &mut Frame<'_>, area: Rect, app: &mut App, theme: Theme) {
    if area.is_empty() || area.width < 12 {
        return;
    }
    let value = app.search_query().to_owned();
    let focused = app.navigation.inline_search();
    let clear_width = if value.is_empty() || area.width < 16 {
        0
    } else {
        4
    };
    let [label, field, clear] = Layout::horizontal([
        Constraint::Length(8),
        Constraint::Min(1),
        Constraint::Length(clear_width),
    ])
    .areas(area);
    frame.render_widget(
        Paragraph::new(" Search ").style(if focused {
            theme.accent()
        } else {
            theme.muted()
        }),
        label,
    );
    app.navigation.hits.add(label, Target::SearchFocus);
    let mut inactive = Editor::new(&value, value.len());
    app.navigation.search_width = field.width.saturating_sub(1);
    let editor = app.navigation.search.as_mut().unwrap_or(&mut inactive);
    render_editor(
        frame,
        &mut app.navigation.hits,
        theme,
        field,
        &value,
        editor,
        focused,
    );
    if value.is_empty() && !focused {
        frame.render_widget(
            Paragraph::new("Type to filter…").style(theme.raised().patch(theme.muted())),
            field,
        );
    }
    if clear_width > 0 {
        control(
            frame,
            &mut app.navigation,
            theme,
            clear,
            " × ",
            Target::SearchClear,
        );
    }
}

pub(super) fn render(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    let theme = app.theme();
    app.navigation.hits.clear();
    app.navigation.has_response = app.response.is_some();
    frame.render_widget(Block::default().style(theme.canvas()), area);
    app.navigation.too_small = area.width < 20 || area.height < 7;
    if app.navigation.too_small {
        frame.render_widget(Paragraph::new("ZeroGravity · resize / Ctrl-Q quit"), area);
        return;
    }
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(2),
    ])
    .areas(area);
    let profile = clean(&app.document.snapshot().target.label());
    let nav_width = if area.width < 70 { 6 } else { 0 };
    let [nav_button, identity] =
        Layout::horizontal([Constraint::Length(nav_width), Constraint::Min(1)]).areas(header);
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
    let profile = if area.width >= 45 {
        format!("  ·  {profile}")
    } else {
        String::new()
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ZeroGravity", theme.accent()),
            Span::styled(profile, theme.muted()),
        ]))
        .style(theme.surface()),
        identity,
    );
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
        let [search, content] = Layout::vertical([
            Constraint::Length(u16::from(app.navigation.view.searchable())),
            Constraint::Min(0),
        ])
        .areas(workspace);
        if !content.is_empty() {
            render_workspace(frame, content, app, theme);
        }
        if !search.is_empty() {
            render_search(frame, search, app, theme);
        }
    }
    divider(frame, &mut app.navigation, theme, track);
    let status = if let Some(job) = &app.active {
        format!(
            "{} · {:?} · {}",
            job.title,
            job.origin,
            if job.cancel_requested {
                "cancellation requested; waiting"
            } else {
                "in progress"
            }
        )
    } else {
        app.notice.clone().unwrap_or_else(|| "Ready".into())
    };
    let cancel = app.active.as_ref().is_some_and(|job| !job.cancel_requested)
        && app.navigation.view != View::Downloads
        && footer.width >= 32;
    let status_width = footer.width.saturating_sub(if cancel { 15 } else { 0 });
    frame.render_widget(
        Paragraph::new(text(status).style(if app.active.is_some() {
            theme.info()
        } else {
            theme.muted()
        }))
        .style(theme.surface()),
        Rect::new(footer.x, footer.y, status_width, 1),
    );
    if cancel {
        let scope = std::rc::Rc::new(Scope::capture(app));
        control(
            frame,
            &mut app.navigation,
            theme,
            Rect::new(footer.right() - 15, footer.y, 15, 1),
            " Cancel action ",
            Target::Command(Bound::new(super::controls::Command::Cancel, &scope)),
        );
    }
    frame.render_widget(
        Paragraph::new(vec![
            text(if app.navigation.view.searchable() {
                "Type to search  ·  F4 actions  ·  Ctrl-Q quit"
            } else {
                "F4 actions  ·  Ctrl-Q quit"
            })
            .style(theme.muted()),
        ])
        .style(theme.surface()),
        Rect::new(footer.x, footer.y + 1, footer.width, 1),
    );
    if footer.width >= 20 {
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
            " Menu … ",
            Target::SessionMenu,
        );
    }
    if !app.navigation.idle() {
        app.navigation.hits.clear();
    }
    let menu_current = match &app.navigation.modal {
        Modal::Menu(menu) => menu.scope.as_ref() == &Scope::capture(app),
        _ => true,
    };
    let busy = app.active.is_some() || app.job.is_some();
    let body = if area.height < 12 { area } else { body };
    match &mut app.navigation.modal {
        Modal::None => {}
        #[cfg(feature = "web")]
        Modal::WebUrl { url, scroll } => {
            let popup = center(body, 110, 11);
            frame.render_widget(Clear, popup);
            let block = dialog(" Embedded Web ", theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let [content, controls] =
                Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(inner);
            modal_text(
                frame,
                &mut app.navigation.hits,
                content,
                Paragraph::new(vec![
                    label("Open this private URL in a browser", theme),
                    text(url),
                    Line::default(),
                    text("Same process and runtime. Keep the control credential private."),
                    text(
                        "Esc / Enter closes this view; the listener remains owned by this process.",
                    ),
                ])
                .wrap(Wrap { trim: false }),
                scroll,
            );
            close_button(frame, &mut app.navigation.hits, theme, controls);
        }
        Modal::Help { scroll } => {
            let popup = center(body, 68, 23);
            frame.render_widget(Clear, popup);
            let block = dialog(" Help ", theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let [content, controls] =
                Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(inner);
            modal_text(
                frame,
                &mut app.navigation.hits,
                content,
                Paragraph::new(vec![
                    label("Navigation", theme),
                    text("Alt+1–9   Views · Alt+: go to view"),
                    text("Tab       Cycle pane focus"),
                    text("↑ ↓       Select / scroll"),
                    text("Type      Live filter in list views"),
                    text("Ctrl-F    Focus search · Esc clear"),
                    text("Enter     Selected model / device actions"),
                    text("          Open profile or edit setting"),
                    text("F4 / …    Context actions · right-click row"),
                    text("F10       Session menu · Web / theme / refresh"),
                    text("Alt+key   Listed operation shortcuts"),
                    text("F3 model/response · F6 panes · F7 maximize · F8 mouse"),
                    text("Alt+←/→ rail · Ctrl+arrows split size"),
                    text("Editor    Drag or Shift+arrows select"),
                    text("Esc       Back / close dialog"),
                    text("Ctrl-Q    Quit · Ctrl-C cancel / quit"),
                ])
                .wrap(Wrap { trim: false }),
                scroll,
            );
            close_button(frame, &mut app.navigation.hits, theme, controls);
        }
        Modal::Details {
            title,
            lines,
            scroll,
        } => {
            let popup = center(body, 100, 24);
            frame.render_widget(Clear, popup);
            let title = format!(" {} ", clean(title));
            let block = dialog(&title, theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let [content, controls] =
                Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(inner);
            let paragraph = Paragraph::new(lines.iter().map(text).collect::<Vec<_>>())
                .wrap(Wrap { trim: false });
            modal_text(frame, &mut app.navigation.hits, content, paragraph, scroll);
            close_button(frame, &mut app.navigation.hits, theme, controls);
        }
        Modal::Menu(menu) => {
            let popup = center(
                body,
                56,
                (menu.buttons.len().saturating_add(4)).clamp(7, 26) as u16,
            );
            frame.render_widget(Clear, popup);
            let title = format!(
                " Actions · {} ",
                truncate_cells(&menu.title, popup.width.saturating_sub(14) as usize)
            );
            let block = dialog(&title, theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let [list, reason, footer] = Layout::vertical([
                Constraint::Min(0),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .areas(inner);
            let current = menu_current;
            let max = menu
                .buttons
                .len()
                .saturating_sub(list.height as usize)
                .min(u16::MAX as usize) as u16;
            menu.scroll = menu.scroll.min(max);
            if menu.cursor < menu.scroll as usize {
                menu.scroll = menu.cursor.min(u16::MAX as usize) as u16;
            } else if menu.cursor >= menu.scroll as usize + list.height as usize && list.height > 0
            {
                menu.scroll = (menu.cursor + 1 - list.height as usize).min(max as usize) as u16;
            }
            // Wheel scroll is local to the menu. Move cursor with a wheel
            // event when required to keep keyboard focus in the viewport.
            app.navigation.hits.add(list, Target::ModalScroll(max));
            for (index, item) in menu.buttons.iter().enumerate() {
                let Some(rect) = row_rect(list, index, menu.scroll as usize) else {
                    continue;
                };
                let available = current && item.disabled.is_none();
                let selected = menu.cursor == index;
                let shortcut = if app.navigation.view.searchable() && item.key.chars().count() == 1
                {
                    format!("Alt+{}", item.key)
                } else {
                    item.key.to_owned()
                };
                let title = format!(
                    "{} {}  ({})",
                    if !available {
                        "-"
                    } else if selected {
                        ">"
                    } else {
                        " "
                    },
                    item.label,
                    shortcut
                );
                let style = if !available {
                    theme.muted()
                } else if selected {
                    theme.selected()
                } else {
                    theme.surface()
                };
                button(
                    frame,
                    &mut app.navigation.hits,
                    rect,
                    &title,
                    style,
                    available.then(|| Target::Command(Bound::new(item.command, &menu.scope))),
                );
            }
            let message = if menu.buttons.is_empty() {
                "No available actions"
            } else if !current {
                "Observations changed; reopen actions"
            } else {
                menu.buttons
                    .get(menu.cursor)
                    .and_then(|b| b.disabled)
                    .unwrap_or("↑↓ choose · Enter open")
            };
            frame.render_widget(Paragraph::new(text(message)).style(theme.muted()), reason);
            close_button(frame, &mut app.navigation.hits, theme, footer);
        }
        Modal::Choice(picker) => {
            let popup = center(body, 64, (picker.count() + 4).max(8) as u16);
            frame.render_widget(Clear, popup);
            let title = format!(" {} ", clean(&picker.title()));
            let block = dialog(&title, theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let [list, controls] =
                Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(inner);
            let max = picker.count().saturating_sub(list.height as usize) as u16;
            picker.scroll = picker.scroll.min(max);
            if picker.cursor < picker.scroll as usize {
                picker.scroll = picker.cursor as u16;
            } else if list.height > 0
                && picker.cursor >= picker.scroll as usize + list.height as usize
            {
                picker.scroll = (picker.cursor + 1 - list.height as usize).min(max as usize) as u16;
            }
            app.navigation.hits.add(list, Target::ModalScroll(max));
            for i in 0..picker.count() {
                let Some(rect) = row_rect(list, i, picker.scroll as usize) else {
                    continue;
                };
                let selected = i == picker.cursor;
                let label = format!("{} {}", if selected { ">" } else { " " }, picker.label(i));
                button(
                    frame,
                    &mut app.navigation.hits,
                    rect,
                    &label,
                    if selected {
                        theme.selected()
                    } else {
                        theme.surface()
                    },
                    Some(Target::Choice(i)),
                );
            }
            let space = modal_buttons(
                frame,
                &mut app.navigation.hits,
                theme,
                controls,
                (" Enter apply ", " Esc cancel "),
                !busy || !picker.is_setting(),
                false,
            );
            if picker.is_setting() && space.width >= 11 {
                button(
                    frame,
                    &mut app.navigation.hits,
                    Rect::new(space.x, space.y, 11, 1),
                    " F2 YAML ",
                    theme.raised(),
                    Some(Target::ChoiceYaml),
                );
            }
        }
        Modal::Theme { file, choice } => {
            let popup = center(body, 54, 11);
            frame.render_widget(Clear, popup);
            let block = dialog(" Appearance ", theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let [content, controls] =
                Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(inner);
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
                    let hit_area = content;
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
                rows.push(
                    text(format!(
                        "{} · Auto follows terminal hint",
                        theme.description()
                    ))
                    .style(theme.muted()),
                );
            }
            frame.render_widget(Paragraph::new(rows), content);
            modal_buttons(
                frame,
                &mut app.navigation.hits,
                theme,
                controls,
                (" Enter apply ", " Esc cancel "),
                !busy,
                false,
            );
        }
        Modal::Form { form, editor } => {
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
                    "Type a prompt. Ctrl-R switches chat/raw; Enter submits. Shift+arrows select.",
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
            };
            let title = match &form.kind {
                FormKind::ConfigValue { file, .. } | FormKind::RawYaml { file } => {
                    format!("{}· {} ", title, clean(&file.target.label()))
                }
                FormKind::Prompt {
                    target: RunTarget::Owner(owner),
                    ..
                } => format!(
                    "{}· {} #{} ",
                    title,
                    clean(&owner.alias),
                    owner.registration
                ),
                _ => title.to_owned(),
            };
            let block = dialog(&title, theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let header = match &form.kind {
                FormKind::ConfigValue { key, .. } => format!("{key} · {hint}"),
                _ if !mode.is_empty() => format!("{hint} · {mode}"),
                _ => hint.to_owned(),
            };
            let [head, input, controls] = Layout::vertical([
                Constraint::Length(
                    (if inner.width < 60 { 3 } else { 2 }).min(inner.height.saturating_sub(2)),
                ),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .areas(inner);
            frame.render_widget(
                Paragraph::new(text(header)).wrap(Wrap { trim: false }),
                head,
            );
            render_editor(
                frame,
                &mut app.navigation.hits,
                theme,
                input,
                &form.value,
                editor,
                true,
            );
            let accept = match form.kind {
                FormKind::RawYaml { .. } => " Ctrl-S save ",
                FormKind::Prompt { .. } => " Enter send ",
                FormKind::ConfigValue { .. } => " Enter save ",
                _ => " Enter accept ",
            };
            let available = modal_buttons(
                frame,
                &mut app.navigation.hits,
                theme,
                controls,
                (accept, " Esc cancel "),
                !busy,
                false,
            );
            if !mode.is_empty() {
                let title = format!(" {mode} ↔ ");
                let width = title.chars().count() as u16;
                if available.width >= width {
                    button(
                        frame,
                        &mut app.navigation.hits,
                        Rect::new(available.x, available.y, width, 1),
                        &title,
                        theme.selected(),
                        Some(Target::PromptMode),
                    );
                }
            }
        }
        Modal::Confirm {
            question,
            scroll,
            action,
            ..
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
            modal_text(
                frame,
                &mut app.navigation.hits,
                question_area,
                paragraph,
                scroll,
            );
            modal_buttons(
                frame,
                &mut app.navigation.hits,
                theme,
                controls,
                (" y confirm ", " n/Esc cancel "),
                !busy || matches!(action, Confirmation::Quit),
                true,
            );
        }
        Modal::Palette(palette) => {
            let popup = center(body, 56, 14);
            frame.render_widget(Clear, popup);
            let block = dialog(" Go to · Enter / Esc ", theme);
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let [content, controls] =
                Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(inner);
            let [query, list] =
                Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(content);
            frame.render_widget(
                Paragraph::new(label(format!("› {}", palette.query), theme)),
                query,
            );
            let matches = palette.matches();
            let max = matches.len().saturating_sub(list.height as usize);
            palette.scroll = palette.scroll.min(max);
            if palette.cursor < palette.scroll {
                palette.scroll = palette.cursor;
            } else if list.height > 0 && palette.cursor >= palette.scroll + list.height as usize {
                palette.scroll = (palette.cursor + 1 - list.height as usize).min(max);
            }
            app.navigation
                .hits
                .add(list, Target::ModalScroll(max as u16));
            if matches.is_empty() {
                frame.render_widget(Paragraph::new("No matching view"), list);
            }
            for (i, view) in matches.into_iter().enumerate() {
                if let Some(rect) = row_rect(list, i, palette.scroll) {
                    let line = text(format!(
                        "{} {}",
                        if i == palette.cursor { ">" } else { " " },
                        view.name()
                    ));
                    frame.render_widget(
                        Paragraph::new(if i == palette.cursor {
                            line.style(theme.selected())
                        } else {
                            line
                        }),
                        rect,
                    );
                    app.navigation.hits.add(rect, Target::Navigate(view));
                }
            }
            close_button(frame, &mut app.navigation.hits, theme, controls);
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
fn truncate_cells(value: &str, width: usize) -> String {
    let value = clean(value);
    if value.width() <= width {
        return value;
    }
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut used = 0;
    for grapheme in value.graphemes(true) {
        let n = grapheme.width();
        if used + n >= width {
            break;
        }
        out.push_str(grapheme);
        used += n;
    }
    out.push('…');
    out
}
fn item_line(
    title: &str,
    annotation: Option<&str>,
    width: u16,
    selected: bool,
    theme: Theme,
) -> Line<'static> {
    let width = width as usize;
    let prefix = if selected { "> " } else { "  " };
    let value = annotation.map(clean).filter(|v| v.width() + 5 < width);
    let mut spans = vec![Span::raw(prefix)];
    if let Some(value) = value {
        let name_width = width.saturating_sub(2 + 2 + value.width());
        let title = truncate_cells(title, name_width);
        let pad = name_width.saturating_sub(title.width()) + 2;
        spans.push(Span::raw(title));
        spans.push(Span::raw(" ".repeat(pad)));
        spans.push(Span::styled(
            value,
            if selected {
                theme.selected()
            } else {
                theme.muted()
            },
        ));
    } else {
        spans.push(Span::raw(truncate_cells(title, width.saturating_sub(2))));
    }
    let line = Line::from(spans);
    if selected {
        line.style(theme.selected())
    } else {
        line
    }
}
fn render_library(frame: &mut Frame<'_>, area: Rect, app: &mut App, kind: Kind, theme: Theme) {
    let title = match kind {
        Kind::Models if area.width < 30 => format!(" Models · {:?} ", app.library.shelf),
        Kind::Models => " Models ".into(),
        Kind::Devices => " Devices ".into(),
        Kind::Profiles => " Profiles ".into(),
        Kind::Config => format!(
            " Config · {}{} ",
            app.library.editor.label(),
            if app.library.editor.path == app.document.snapshot().target.path {
                ""
            } else {
                " (inactive)"
            }
        ),
    };
    let block = pane(title, theme, false);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let collection = app.library.rows(kind);
    let query = collection.query();
    let count = collection.visible().len();
    let mut status = match kind {
        Kind::Models => format!("{count} models"),
        Kind::Config if app.library.raw => "Raw YAML".into(),
        Kind::Config
            if app
                .library
                .observed()
                .is_some_and(|obs| !obs.snapshot.exists()) =>
        {
            "Built-in defaults".into()
        }
        Kind::Config => format!("{count} settings"),
        Kind::Profiles => format!("{count} profiles"),
        Kind::Devices => format!("{count} devices"),
    };
    if let Some(notice) = app.library.notice(kind) {
        status.push_str(&format!(" · {notice}"));
    } else if kind == Kind::Models && app.document.snapshot().offline {
        status.push_str(" · Offline");
    }
    let [head, body] = Layout::vertical([
        Constraint::Length(u16::from(inner.height >= 4)),
        Constraint::Min(1),
    ])
    .areas(inner);
    let mut x = head.x;
    if kind == Kind::Models && head.height > 0 && head.width >= 30 {
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
                    theme.surface().patch(theme.muted())
                }),
                rect,
            );
            app.navigation.hits.add(rect, Target::Shelf(shelf));
            x += width + 1;
        }
    }
    if kind == Kind::Models && head.height > 0 && head.width < 30 {
        let title = format!(" {:?} › ", app.library.shelf);
        let width = (title.chars().count() as u16).min(head.width);
        control(
            frame,
            &mut app.navigation,
            theme,
            Rect::new(x, head.y, width, 1),
            &title,
            Target::Shelf(app.library.shelf.next()),
        );
        x += width + 1;
    }
    if x < head.right() {
        frame.render_widget(
            Paragraph::new(text(status)).style(theme.muted()),
            Rect::new(x, head.y, head.right() - x, head.height),
        );
    }
    let (list, detail, track) = app.navigation.workspace.inner(
        body,
        app.navigation.view.index(),
        app.navigation.view.horizontal_default(),
    );
    app.navigation.split_visible = !list.is_empty() && !detail.is_empty();
    divider(frame, &mut app.navigation, theme, track);
    let mut lines = Vec::new();
    let mut group = "";
    let mut selected_line = 0;
    let mut keys = Vec::new();
    for row in collection.visible() {
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
        let active = matches!(&row.key,Key::Profile(name) if app.document.snapshot().target.profile.as_deref()==Some(name.as_str()));
        let annotation = if active {
            Some("Active")
        } else {
            row.annotation.as_deref()
        };
        lines.push(item_line(
            &row.title,
            annotation,
            list.width.saturating_sub(6),
            selected,
            theme,
        ));
    }
    if lines.is_empty() {
        lines.push(
            text(if !query.is_empty() {
                "No matches"
            } else {
                match (kind, app.library.shelf) {
                    (Kind::Models, Shelf::Cached) => "No cached models",
                    (Kind::Models, Shelf::Partial) => "No partial downloads",
                    (Kind::Config, _) => "No settings observed",
                    _ => "No entries",
                }
            })
            .style(theme.muted()),
        );
    }
    let mut menu_visible = false;
    if !list.is_empty() {
        let inner = pane_body(
            frame,
            list,
            &mut app.navigation,
            theme,
            Pane::Main,
            if kind == Kind::Config {
                "Setting"
            } else {
                "Name"
            },
        );
        let reveal = app
            .navigation
            .workspace
            .take_reveal(app.navigation.view.index())
            .then_some(selected_line);
        let slot = Navigation::list_slot(kind, app.library.shelf);
        let viewport = app
            .navigation
            .workspace
            .viewport(app.navigation.view.index(), Pane::Main);
        viewport.offset = app.navigation.library_anchors[slot].restore(
            viewport.offset,
            reveal.is_some(),
            keys.is_empty(),
            |key| {
                keys.iter()
                    .find(|(_, candidate)| *candidate == key)
                    .map(|(row, _)| *row)
            },
            viewport.offset,
        );
        let (visible, offset) = scroll_rows(
            frame,
            &mut app.navigation,
            theme,
            Pane::Main,
            inner,
            Paragraph::new(lines),
            reveal.into(),
        );
        app.navigation.library_anchors[slot].remember(offset, || {
            keys.iter()
                .find(|(row, _)| *row >= offset)
                .map(|(row, key)| (*row, (*key).clone()))
        });
        for (index, key) in keys {
            if let Some(rect) = row_rect(visible, index, offset) {
                app.navigation
                    .hits
                    .add(rect, Target::Row(kind, key.clone()));
                if collection.selected.as_ref() == Some(key) && rect.width >= 7 {
                    let menu = Rect::new(rect.right() - 4, rect.y, 4, 1);
                    control(
                        frame,
                        &mut app.navigation,
                        theme,
                        menu,
                        " … ",
                        Target::RowMenu(kind, key.clone()),
                    );
                    menu_visible = true;
                }
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
            && let Some(summary) = result.lines().next()
            && app.notice.as_deref() != Some(summary)
        {
            details.push(Line::default());
            details.push(text(format!("Last action: {summary}")).style(theme.muted()));
        }
    } else {
        details.push(text(if kind == Kind::Config {
            "Select a setting"
        } else {
            "Select an item"
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
        if !menu_visible {
            pane_actions(frame, &mut app.navigation, theme, detail);
            menu_visible = true;
        }
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
    if !menu_visible && !list.is_empty() {
        pane_actions(frame, &mut app.navigation, theme, list);
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
    pane_actions(frame, &mut app.navigation, theme, area);
    let [head, body] =
        Layout::vertical([Constraint::Length(1.min(inner.height)), Constraint::Min(0)])
            .areas(inner);
    let mut heading = app.logs.heading();
    let dropped = app.logs.dropped();
    if dropped > 0 {
        heading.push_str(&format!(" · {dropped} older entries dropped"));
    }
    frame.render_widget(Paragraph::new(text(heading)).style(theme.muted()), head);
    let rows = app.logs.entries();
    if app
        .logs
        .selected
        .is_some_and(|id| !rows.iter().any(|row| row.sequence == id))
    {
        app.logs.selected = None;
    }
    let selected = app.logs.selected;
    let reveal = app
        .navigation
        .workspace
        .take_reveal(View::Logs.index())
        .then(|| rows.iter().position(|row| Some(row.sequence) == selected))
        .flatten();
    let lines = if rows.is_empty() {
        vec![text("No notifications received.")]
    } else {
        rows.iter()
            .map(|entry| {
                item_line(
                    &entry.line(),
                    None,
                    body.width.saturating_sub(6),
                    Some(entry.sequence) == selected,
                    theme,
                )
            })
            .collect()
    };
    let follow = app.logs.following();
    let viewport = app
        .navigation
        .workspace
        .viewport(View::Logs.index(), Pane::Main);
    if !follow {
        viewport.offset = app.navigation.log_anchor.restore(
            viewport.offset,
            reveal.is_some(),
            rows.is_empty(),
            |id| rows.iter().position(|row| row.sequence == *id),
            0,
        );
    }
    let (visible, offset) = scroll_rows(
        frame,
        &mut app.navigation,
        theme,
        Pane::Main,
        body,
        Paragraph::new(lines),
        if follow {
            ScrollMode::Follow
        } else {
            reveal.into()
        },
    );
    if follow {
        app.navigation.log_anchor = Anchor::default();
    }
    app.navigation.log_anchor.remember(offset, || {
        rows.get(offset).map(|entry| (offset, entry.sequence))
    });
    for (i, entry) in rows.iter().enumerate() {
        if let Some(rect) = row_rect(visible, i, offset) {
            app.navigation.hits.add(rect, Target::Log(entry.sequence));
            if selected == Some(entry.sequence) && rect.width >= 7 {
                control(
                    frame,
                    &mut app.navigation,
                    theme,
                    Rect::new(rect.right() - 4, rect.y, 4, 1),
                    " … ",
                    Target::LogDetails(entry.sequence),
                );
            }
        }
    }
}
fn render_response(frame: &mut Frame<'_>, area: Rect, app: &mut App, theme: Theme) {
    let Some(response) = &app.response else {
        return;
    };
    if area.width < 3 || area.height == 0 {
        return;
    }
    let heading = response.owner.as_ref().map_or_else(
        || "Response".into(),
        |owner| {
            // Long digest aliases must not conceal the response owner.
            let prefix = format!("registration {} · ", owner.registration);
            format!(
                "{prefix}{}",
                truncate_cells(
                    &owner.alias,
                    (area.width as usize).saturating_sub(prefix.width())
                )
            )
        },
    );
    let [head, body] = Layout::vertical([
        Constraint::Length(u16::from(area.height > 1)),
        Constraint::Min(1),
    ])
    .areas(area);
    frame.render_widget(Paragraph::new(text(heading)).style(theme.muted()), head);
    let width = body.width.saturating_sub(2).max(1);
    let value = if response.text.is_empty() {
        "No output yet"
    } else {
        &response.text
    };
    let visual = editor::VisualRows::new(value, width);
    let view = app.navigation.view;
    let slot = usize::from(view == View::Running);
    let follow = app.navigation.following_output();
    let viewport = app.navigation.workspace.output(view.index());
    if !follow {
        viewport.offset = app.navigation.output_anchors[slot].restore(
            viewport.offset,
            false,
            false,
            |byte| {
                Some(
                    visual
                        .rows
                        .partition_point(|row| row.start <= *byte)
                        .saturating_sub(1),
                )
            },
            viewport.offset,
        );
    }
    let viewport = app.navigation.workspace.output(view.index());
    viewport.measure(
        visual.rows.len() + usize::from(response.clipped),
        body.height,
    );
    if follow {
        viewport.scroll(Motion::End);
    }
    let viewport = *viewport;
    let offset = viewport.offset;
    let visible = Rect::new(body.x + 1, body.y, width, body.height);
    let mut lines = Vec::new();
    for row in visual.rows.iter().skip(offset).take(body.height as usize) {
        lines.push(Line::from(
            value[row.clone()]
                .graphemes(true)
                .map(|g| Span::raw(editor::display(g, width)))
                .collect::<Vec<_>>(),
        ));
    }
    if response.clipped
        && lines.len() < body.height as usize
        && offset + lines.len() == visual.rows.len()
    {
        lines.push(
            text("[Display limit reached; original output was not altered]").style(theme.muted()),
        );
    }
    frame.render_widget(Paragraph::new(lines), visible);
    let track = Rect::new(body.right() - 1, body.y, 1, body.height);
    if let Some(bar) = Scrollbar::new(Pane::Detail, track, viewport) {
        paint_bar(frame, theme, bar);
        app.navigation.hits.add(track, Target::Scrollbar(bar));
    }
    if follow {
        app.navigation.output_anchors[slot] = Anchor::default();
    }
    app.navigation.output_anchors[slot].remember(offset, || {
        visual.rows.get(offset).map(|row| (offset, row.start))
    });
}

fn render_workspace(frame: &mut Frame<'_>, area: Rect, app: &mut App, theme: Theme) {
    if app.navigation.view == View::Running {
        app.performance
            .observe(app.selected.as_ref().and_then(|owner| {
                app.snapshot
                    .runtime
                    .catalog
                    .iter()
                    .find(|model| model.registration_id == owner.registration)
            }));
    }
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
    let view = app.navigation.view;
    let (main, detail, track) = if app.navigation.workspace.has_detail {
        app.navigation.workspace.inner(
            area,
            app.navigation.view.index(),
            app.navigation.view.horizontal_default(),
        )
    } else {
        (area, Rect::default(), None)
    };
    let Content {
        rows,
        mut details,
        owners,
    } = content(app, theme, detail);
    app.navigation.split_visible = !main.is_empty() && !detail.is_empty();
    let mut menu_visible = false;
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
        if view == View::Running {
            let viewport = app.navigation.workspace.viewport(view.index(), Pane::Main);
            viewport.offset = app.navigation.running_anchor.restore(
                viewport.offset,
                reveal.is_some(),
                owners.is_empty(),
                |key| {
                    owners
                        .iter()
                        .find(|(_, candidate)| candidate == key)
                        .map(|(row, _)| *row)
                },
                viewport.offset,
            );
        }
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
        if view == View::Running {
            app.navigation.running_anchor.remember(offset, || {
                owners.iter().find(|(row, _)| *row >= offset).cloned()
            });
        }
        for (line, owner) in owners {
            if let Some(rect) = row_rect(visible, line, offset) {
                app.navigation.hits.add(rect, Target::Owner(owner.clone()));
                if app.selected.as_ref() == Some(&owner) && rect.width >= 7 {
                    control(
                        frame,
                        &mut app.navigation,
                        theme,
                        Rect::new(rect.right() - 4, rect.y, 4, 1),
                        " … ",
                        Target::OwnerMenu(owner),
                    );
                    menu_visible = true;
                }
            }
        }
    }
    if !menu_visible && !main.is_empty() {
        pane_actions(frame, &mut app.navigation, theme, main);
        menu_visible = true;
    }
    if !detail.is_empty() {
        let output = app.navigation.output_visible();
        let inner = pane_body(
            frame,
            detail,
            &mut app.navigation,
            theme,
            Pane::Detail,
            if output { "Response" } else { "Selected" },
        );
        if !menu_visible {
            pane_actions(frame, &mut app.navigation, theme, detail);
        }
        if view == View::Running && app.response.is_some() && detail.width >= 20 {
            // A single tab switch, not another toolbar. Title region
            // remains distinct from maximize/restore and pane actions.
            let title = if app.navigation.running_response {
                " Response↔"
            } else {
                " Model ↔ "
            };
            let width = title.chars().count() as u16;
            control(
                frame,
                &mut app.navigation,
                theme,
                Rect::new(detail.x, detail.y, width, 1),
                title,
                Target::ResponseTab,
            );
        }
        if app.navigation.output_visible() {
            render_response(frame, inner, app, theme);
        } else {
            let action = if view == View::Running && !details.is_empty() && app.selected.is_some() {
                // Measure the actual wrapped prefix so the exact read-only
                // selected-owner link follows scroll/resize and hit masking.
                let width = inner.width.saturating_sub(2);
                let line = Paragraph::new(details.clone())
                    .wrap(Wrap { trim: false })
                    .line_count(width);
                let title = if width >= 21 {
                    "Performance details ›"
                } else {
                    "Details ›"
                };
                details.push(Line::from(Span::styled(title, theme.accent())));
                Some((line, title))
            } else {
                None
            };
            let (visible, offset) = scroll_rows(
                frame,
                &mut app.navigation,
                theme,
                Pane::Detail,
                inner,
                Paragraph::new(details).wrap(Wrap { trim: false }),
                ScrollMode::Manual,
            );
            if let Some((line, title)) = action
                && let Some(rect) = row_rect(visible, line, offset)
            {
                let scope = std::rc::Rc::new(Scope::capture(app));
                let rect = Rect::new(
                    rect.x,
                    rect.y,
                    rect.width.min(title.chars().count() as u16),
                    1,
                );
                app.navigation.hits.add(
                    rect,
                    Target::Command(Bound::new(super::controls::Command::Details, &scope)),
                );
            }
        }
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
        nav.scroll(Pane::Main, Motion::Step(1), Shelf::Cached);
        nav.key(key(KeyCode::Char(':')));
        assert_eq!(nav.key(key(KeyCode::Char('q'))), Action::Redraw);
        assert!(matches!(&nav.modal, Modal::Palette(p) if p.query=="q"));
        nav.key(key(KeyCode::Backspace));
        for c in "logs".chars() {
            nav.key(key(KeyCode::Char(c)));
        }
        assert_eq!(nav.key(key(KeyCode::Enter)), Action::Open);
        assert_eq!(nav.view(), View::Logs);
        nav.key(KeyEvent::new(KeyCode::Char('5'), KeyModifiers::ALT));
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
            running_query: String::new(),
            performance: super::super::performance::History::default(),
            library: super::super::library::Library::new(
                super::super::work::Document::test_default().target,
            ),
        };
        let scope = std::rc::Rc::new(Scope::capture(&app));
        let bound = Bound::new(super::super::controls::Command::Load, &scope);
        assert!(bound.valid(&app));
        app.document.configuration_changed();
        assert!(!bound.valid(&app));
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
            assert!(text.contains("No models loaded"));
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
        app.navigation.open(View::Models);
        app.library.test_replace(
            Kind::Models,
            vec![
                super::super::library::Row {
                    key: Key::Snapshot("a".repeat(64)),
                    title: "google/gemma-4-E2B-it".into(),
                    annotation: Some(format::bytes(32_198_128)),
                    details: vec![
                        "Size: 30.71 MiB · 1 file".into(),
                        "Metadata only · payloads not verified".into(),
                    ],
                    technical: vec![format!("Snapshot: sha256:{}", "a".repeat(64))],
                },
                super::super::library::Row {
                    key: Key::Snapshot("b".repeat(64)),
                    title: "Qwen/Qwen3.5-2B".into(),
                    annotation: Some(format::bytes(4_571_272_453)),
                    details: vec![
                        "Size: 4.26 GiB · 12 files".into(),
                        "Metadata only · payloads not verified".into(),
                    ],
                    technical: vec![format!("Snapshot: sha256:{}", "b".repeat(64))],
                },
            ],
        );
        app.library.rows_mut(Kind::Models).move_selection(1);
        let mut terminal = Terminal::new(TestBackend::new(128, 24)).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let buffer = terminal.backend().buffer();
        let row = |y| {
            (0..128)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        };
        let bar = row(1);
        assert!(bar.contains("Search") && bar.contains("Type to filter"));
        for redundant in [
            "Load",
            "Run",
            "More",
            "Inspect",
            "Verify",
            "Alias",
            "Refresh",
            "Appearance",
            "Cached",
            "Catalog",
            "Partial",
            "Load for Serve",
        ] {
            assert!(!bar.contains(redundant), "{bar}");
        }
        let visible = (0..24).map(row).collect::<Vec<_>>();
        let text = visible.join("\n");
        assert!(text.contains("30.71 MiB") && text.contains("4.26 GiB"));
        assert!(!text.contains("sha256:") && !text.contains("4571272453"));
        println!("Toolbar-free Models cells:\n{text}");
        // Ordinary characters, even former operation/navigation/quit
        // keys, go to local filtering. No hidden model can be acted on.
        let scope = std::rc::Rc::new(Scope::capture(&app));
        let old = Bound::new(
            super::super::controls::Command::Library(LibraryAction::Load(false)),
            &scope,
        );
        for c in "gemma".chars() {
            let action = app.navigation.key(key(KeyCode::Char(c)));
            assert!(matches!(action, Action::Search(_)));
            assert!(!app.handle(&runtime, runtime.control(), action));
        }
        assert_eq!(app.search_query(), "gemma");
        assert_eq!(app.library.rows(Kind::Models).visible().len(), 1);
        assert_eq!(
            app.library.rows(Kind::Models).selected().unwrap().key,
            Key::Snapshot("a".repeat(64))
        );
        assert!(!old.valid(&app));
        assert!(app.active.is_none());
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(!text.contains("Qwen/Qwen3.5-2B"));
        for c in "q8:/?".chars() {
            let action = app.navigation.key(key(KeyCode::Char(c)));
            assert!(matches!(action, Action::Search(_)));
            assert!(!app.handle(&runtime, runtime.control(), action));
        }
        assert_eq!(app.search_query(), "gemmaq8:/?");
        assert!(app.library.rows(Kind::Models).selected().is_none());
        assert!(!app.handle(
            &runtime,
            runtime.control(),
            Action::RowActions(Kind::Models, Key::Snapshot("b".repeat(64)))
        ));
        assert!(app.navigation.idle());
        let action = app.navigation.key(key(KeyCode::Esc));
        app.handle(&runtime, runtime.control(), action);
        assert!(app.search_query().is_empty());
        assert_eq!(app.library.rows(Kind::Models).visible().len(), 2);
        // Right-click/ellipsis is selection + menu only, never work.
        app.handle(
            &runtime,
            runtime.control(),
            Action::RowActions(Kind::Models, Key::Snapshot("b".repeat(64))),
        );
        assert!(
            matches!(&app.navigation.modal,Modal::Menu(menu) if menu.title=="Qwen/Qwen3.5-2B" && menu.buttons[0].label=="Load" && menu.buttons[1].label=="Run" && !menu.buttons.iter().any(|b|b.label=="Pull"))
        );
        assert!(app.active.is_none());
        app.navigation.dismiss_modal();
        app.handle(&runtime, runtime.control(), Action::ShowDetails);
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(text.contains(&format!("sha256:{}", "b".repeat(64))));
        app.navigation.dismiss_modal();
        app.library.shelf = Shelf::Catalog;
        let menu = Menu::new(&app, false);
        assert!(menu.buttons.iter().any(|b| b.label == "Pull"));
        assert!(
            !menu
                .buttons
                .iter()
                .any(|b| matches!(b.label, "Load" | "Run" | "Verify payloads" | "Remove"))
        );
        // Actual rendered list anchors, including per-shelf preservation
        // and a temporary viewport clamp after a larger resize.
        app.library.shelf = Shelf::Cached;
        app.navigation.shelf_changed(Shelf::Cached);
        let rows: Vec<_> = (0..40)
            .map(|i| super::super::library::Row {
                key: Key::Snapshot(format!("fixture-{i}")),
                title: format!("fixture-{i:02}"),
                annotation: None,
                details: vec![],
                technical: vec![],
            })
            .collect();
        app.library.test_replace(Kind::Models, rows.clone());
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        app.navigation
            .scroll(Pane::Main, Motion::At(12), Shelf::Cached);
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let mut reordered = rows;
        reordered.rotate_right(2);
        app.library.test_replace(Kind::Models, reordered);
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert_eq!(
            app.navigation
                .workspace
                .viewport(View::Models.index(), Pane::Main)
                .offset,
            14
        );
        let mut tall = Terminal::new(TestBackend::new(128, 70)).unwrap();
        tall.draw(|frame| render(frame, &mut app)).unwrap();
        assert_eq!(
            app.navigation
                .workspace
                .viewport(View::Models.index(), Pane::Main)
                .offset,
            0
        );
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert_eq!(
            app.navigation
                .workspace
                .viewport(View::Models.index(), Pane::Main)
                .offset,
            14
        );
        app.library.shelf = Shelf::Catalog;
        app.navigation.shelf_changed(Shelf::Catalog);
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        app.library.shelf = Shelf::Cached;
        app.navigation.shelf_changed(Shelf::Cached);
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert_eq!(
            app.navigation
                .workspace
                .viewport(View::Models.index(), Pane::Main)
                .offset,
            14
        );

        // Render the actual selected pane and exercise the visible exact-owner
        // details link. Fixture metrics are not an inference observation.
        app.navigation.open(View::Running);
        app.selected = Some(Owner {
            alias: "fixture".into(),
            registration: 9,
        });
        app.snapshot
            .runtime
            .catalog
            .push(zerogravity::models::ModelSnapshot {
                registration_id: 9,
                source: zerogravity::identity::SourceIdentity::unbound(),
                aliases: vec!["fixture".into()],
                lifecycle: zerogravity::models::ModelLifecycle::Ready,
                admitted_requests: 1,
                performance: zerogravity::models::ModelPerformance {
                    loaded_ms: 720_000,
                    output_tokens: 429,
                    completed_requests: 1,
                    ttft_samples: 1,
                    ttft_total_ns: 184_000_000,
                    decode_intervals: 428,
                    decode_total_ns: 10_000_000_000,
                    ..Default::default()
                },
                independent_retirement: true,
                capabilities: None,
            });
        app.snapshot.runtime.status.loaded_owners = 1;
        let mut dashboard = Terminal::new(TestBackend::new(110, 28)).unwrap();
        dashboard.draw(|frame| render(frame, &mut app)).unwrap();
        let buffer = dashboard.backend().buffer();
        let painted = (0..28)
            .map(|y| {
                (0..110)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        assert!(
            painted
                .iter()
                .any(|line| line.contains("42.8 tok/s") && line.contains("184 ms"))
        );
        assert!(
            painted
                .iter()
                .any(|line| line.contains("Avg decode") && line.contains("Avg TTFT"))
        );
        assert!(painted.iter().any(|line| line.contains("1 admitted")));
        let (y, x) = painted
            .iter()
            .enumerate()
            .find_map(|(y, line)| {
                line.find("Performance details")
                    .map(|byte| (y as u16, line[..byte].chars().count() as u16))
            })
            .expect("visible details link");
        let mouse = |kind| MouseEvent {
            kind,
            column: x,
            row: y,
            modifiers: KeyModifiers::NONE,
        };
        app.navigation
            .mouse(mouse(MouseEventKind::Down(MouseButton::Left)));
        let Some(Action::Bound(bound)) = app
            .navigation
            .mouse(mouse(MouseEventKind::Up(MouseButton::Left)))
        else {
            panic!("performance link did not bind the selected owner");
        };
        assert_eq!(bound.command, super::super::controls::Command::Details);
        assert!(bound.valid(&app));
        app.selected = None;
        assert!(!bound.valid(&app));
        app.snapshot = runtime.control().snapshot().unwrap();

        // Render bounded response text on its own viewport. Changing
        // model-detail scroll/focus cannot move or resume that response.
        app.navigation.open(View::Running);
        app.response = Some(super::super::work::Response {
            owner: Some(Owner {
                alias: "fixture".into(),
                registration: 9,
            }),
            text: (0..50).map(|i| format!("response line {i:02}\n")).collect(),
            ..Default::default()
        });
        app.navigation.follow_response();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert!(app.navigation.output_visible());
        assert!(
            app.navigation
                .workspace
                .output(View::Running.index())
                .offset
                > 0
        );
        app.navigation
            .scroll(Pane::Detail, Motion::At(8), Shelf::Cached);
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert!(!app.navigation.following_output());
        app.navigation.toggle_response();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert!(!app.navigation.output_visible());
        app.navigation
            .scroll(Pane::Detail, Motion::End, Shelf::Cached);
        assert!(!app.navigation.following_output());
        app.navigation.toggle_response();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert_eq!(
            app.navigation
                .workspace
                .output(View::Running.index())
                .offset,
            8
        );
        app.response
            .as_mut()
            .unwrap()
            .text
            .push_str("new appended line\n");
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert_eq!(
            app.navigation
                .workspace
                .output(View::Running.index())
                .offset,
            8
        );
        let mut narrow = Terminal::new(TestBackend::new(60, 18)).unwrap();
        narrow.draw(|frame| render(frame, &mut app)).unwrap();
        assert_eq!(
            app.navigation
                .workspace
                .output(View::Running.index())
                .offset,
            8
        );

        // Minimum usable terminal: a menu still has actual visible
        // selectable rows, not just its footer.
        app.navigation.open(View::Home);
        app.response = None;
        app.navigation.menu(Menu::new(&app, false));
        let mut compact = Terminal::new(TestBackend::new(20, 7)).unwrap();
        compact.draw(|frame| render(frame, &mut app)).unwrap();
        use crossterm::event::{MouseButton, MouseEventKind};
        let event = |kind| MouseEvent {
            kind,
            column: 4,
            row: 1,
            modifiers: KeyModifiers::NONE,
        };
        assert!(
            app.navigation
                .mouse(event(MouseEventKind::Down(MouseButton::Left)))
                .is_none()
        );
        assert!(
            matches!(app.navigation.mouse(event(MouseEventKind::Up(MouseButton::Left))),
            Some(Action::Bound(bound)) if bound.command==super::super::controls::Command::Run)
        );
        app.navigation.dismiss_modal();
        runtime.shutdown().unwrap();
    }
}
