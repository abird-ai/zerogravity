//! Visible native actions. Bind to observed identities, not coordinates or
//! simulated keystrokes. The original App handlers remain the effect owners.
#[cfg(all(feature = "acquisition", target_os = "linux"))]
use super::view::DownloadAction;
use super::{
    App,
    library::{Key, Kind, Shelf},
    settings::Snapshot,
    view::{Action, LibraryAction, LogAction, SettingsAction, View},
    work::Owner,
};
use crate::control_host::{ActionSummary, Target};
use std::rc::Rc;
use zerogravity::{control::ServeSummary, diagnostics::RuntimePhase};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Command {
    Load,
    Run,
    Unload,
    Start,
    Stop,
    Cancel,
    Library(LibraryAction),
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    Download(DownloadAction),
    Settings(SettingsAction),
    Logs(LogAction),
    Details,
    LogDetails(u64),
    FollowOutput,
    OpenWeb,
    Appearance,
    Refresh,
    Quit,
}
impl Command {
    pub fn action(self) -> Action {
        match self {
            Self::Load => Action::Load,
            Self::Run => Action::Run,
            Self::Unload => Action::Unload,
            Self::Start => Action::Start,
            Self::Stop => Action::Stop,
            Self::Cancel => Action::Cancel,
            Self::Library(action) => Action::Library(action),
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            Self::Download(action) => Action::Download(action),
            Self::Settings(action) => Action::Settings(action),
            Self::Logs(action) => Action::Logs(action),
            Self::Details => Action::ShowDetails,
            Self::LogDetails(id) => Action::ShowLog(Some(id)),
            Self::FollowOutput => Action::FollowOutput,
            Self::OpenWeb => Action::OpenWeb,
            Self::Appearance => Action::Appearance,
            Self::Refresh => Action::Refresh,
            Self::Quit => Action::Quit,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Listener {
    Unavailable,
    Idle,
    Transitioning,
    Listening(u64),
    Finished(u64),
}
impl From<ServeSummary> for Listener {
    fn from(summary: ServeSummary) -> Self {
        match summary {
            ServeSummary::Unavailable => Self::Unavailable,
            ServeSummary::Idle => Self::Idle,
            ServeSummary::Transitioning => Self::Transitioning,
            ServeSummary::Listening { binding, .. } => Self::Listening(binding.generation),
            ServeSummary::Finished { binding } => Self::Finished(binding.generation),
        }
    }
}

/// One small binding per rendered action surface. No counter/progress
/// snapshots or large text copies. ConfigFile snapshots share their bytes.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Scope {
    view: View,
    selection: Option<Key>,
    source_refusal: Option<&'static str>,
    shelf: Shelf,
    owner: Option<Owner>,
    editor: Option<Snapshot>,
    active_document: Target,
    configuration_revision: u64,
    phase: RuntimePhase,
    listener: Listener,
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    download: Option<(u64, zerogravity::control::DownloadStatus, bool)>,
    active: Option<ActionSummary>,
    native_busy: bool,
    offline: bool,
    valid_document: bool,
    explicit_value: bool,
    raw: bool,
    follow_logs: bool,
    log: Option<u64>,
    follow_output: bool,
    has_response: bool,
}
impl Scope {
    pub fn capture(app: &App) -> Self {
        let view = app.navigation.view();
        let kind = match view {
            View::Models => Some(Kind::Models),
            View::Devices => Some(Kind::Devices),
            View::Profiles => Some(Kind::Profiles),
            View::Config => Some(Kind::Config),
            _ => None,
        };
        let selection =
            kind.and_then(|kind| app.library.rows(kind).selected().map(|row| row.key.clone()));
        let document = app.document.snapshot();
        let editor = if view == View::Config {
            app.library.snapshot()
        } else {
            None
        };
        let explicit_value = match &selection {
            Some(Key::Setting(key)) => app.library.value(key).is_some(),
            _ => false,
        };
        Self {
            view,
            source_refusal: match &selection {
                Some(Key::Snapshot(digest)) => app.library.source_refusal(digest),
                _ => None,
            },
            selection,
            shelf: app.library.shelf,
            owner: app.selected.clone(),
            editor,
            active_document: document.target,
            configuration_revision: app.document.configuration_revision(),
            phase: app.snapshot.runtime.status.phase,
            listener: app.snapshot.serving.into(),
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            download: app
                .snapshot
                .download
                .as_ref()
                .map(|d| (d.id, d.status, d.cancel_requested)),
            active: app.active.clone(),
            native_busy: app.job.is_some(),
            offline: document.offline,
            valid_document: app.library.observed().is_some_and(|o| o.warning.is_none()),
            explicit_value,
            raw: app.library.raw,
            follow_logs: app.logs.following(),
            log: app.logs.selected,
            follow_output: app.navigation.following_output(),
            has_response: app.response.is_some(),
        }
    }
    fn busy(&self) -> Option<&'static str> {
        (self.active.is_some() || self.native_busy).then_some("Action in progress")
    }
    fn load(&self) -> Option<&'static str> {
        self.busy()
            .or((self.phase != RuntimePhase::Empty).then_some("Retire current route set first"))
    }
    fn run(&self) -> Option<&'static str> {
        self.busy()
            .or((self.owner.is_none() && self.phase != RuntimePhase::Empty)
                .then_some("No ready model"))
    }
    fn pull(&self) -> Option<&'static str> {
        self.busy()
            .or((!cfg!(all(feature = "acquisition", target_os = "linux")))
                .then_some("Acquisition not compiled"))
            .or(self.offline.then_some("Offline mode"))
    }
    fn file(&self) -> Option<&'static str> {
        self.busy()
            .or(self.editor.is_none().then_some("Observe document first"))
    }
    fn valid_file(&self) -> Option<&'static str> {
        self.file()
            .or((!self.valid_document).then_some("Invalid document"))
    }
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    fn download(&self, action: DownloadAction) -> Option<&'static str> {
        use zerogravity::control::DownloadStatus;
        let Some((_, status, cancel)) = self.download else {
            return Some("No active transfer");
        };
        if status.terminal() {
            return Some("Transfer finished");
        }
        if cancel {
            return Some("Cancellation requested");
        }
        match action {
            DownloadAction::Pause if status != DownloadStatus::Running => Some("Not running"),
            DownloadAction::Resume if status != DownloadStatus::Paused => Some("Not paused"),
            _ => None,
        }
    }

    /// Hints are based on the same observed state as the existing handlers.
    /// They are not an admission decision; original requests revalidate.
    pub fn buttons(&self) -> Vec<Button> {
        use Command as C;
        use LibraryAction as L;
        use LogAction as G;
        use SettingsAction as S;
        let mut out = Vec::new();
        let mut add = |command, label, key, disabled| {
            out.push(Button {
                command,
                label,
                key,
                disabled,
            })
        };
        let selected = self.selection.is_some();
        let choose_owner = self.owner.is_none().then_some("No ready registration");
        let select_row = (!selected).then_some("Select a row");
        match self.view {
            View::Home => {
                add(C::Run, "Run", "Enter", self.run());
                add(C::Load, "Load", "l", self.load());
            }
            View::Running => {
                if self.owner.is_some() {
                    add(C::Run, "Run", "p", self.run());
                    add(C::Unload, "Unload", "u", self.busy().or(choose_owner));
                    add(C::Details, "Performance details", "D", None);
                } else if self.phase == RuntimePhase::Empty {
                    add(C::Load, "Load", "l", self.load());
                }
            }
            View::Serve => match self.listener {
                Listener::Listening(_) | Listener::Finished(_) => {
                    add(C::Stop, "Stop", "t", self.busy())
                }
                Listener::Idle => {
                    add(C::Start, "Start", "s", self.busy().or(choose_owner));
                    if self.owner.is_none() {
                        add(C::Load, "Load", "l", self.load());
                    }
                }
                Listener::Transitioning | Listener::Unavailable => {}
            },
            View::Models => match &self.selection {
                Some(Key::Snapshot(_)) => {
                    let load = self.load().or(self.source_refusal);
                    add(C::Library(L::Load(false)), "Load", "l", load);
                    add(C::Library(L::Run), "Run", "p", load);
                    add(C::Details, "Details", "D", None);
                    add(C::Library(L::Inspect), "Inspect manifest", "i", self.busy());
                    add(C::Library(L::Verify), "Verify payloads", "v", self.busy());
                    add(C::Library(L::Alias), "Alias", "a", self.busy());
                    add(
                        C::Library(L::Load(true)),
                        "Load for Serve",
                        "Alt-s",
                        load.or((!cfg!(feature = "serve")).then_some("Serving not compiled")),
                    );
                    add(
                        C::Library(L::Reclaim(false)),
                        "Removal preview",
                        "X",
                        self.busy(),
                    );
                    add(C::Library(L::Reclaim(true)), "Remove", "x", self.busy());
                    add(C::Library(L::Execution), "Execution…", "e", self.busy());
                }
                Some(Key::Target(_)) | Some(Key::Partial(_)) => {
                    add(
                        C::Library(L::Pull),
                        if matches!(self.selection, Some(Key::Partial(_))) {
                            "Resume pull…"
                        } else {
                            "Pull"
                        },
                        "d",
                        self.pull(),
                    );
                    add(C::Details, "Details", "D", None);
                }
                _ => add(C::Library(L::Pull), "New pull", "d", self.pull()),
            },
            View::Downloads => {
                #[cfg(all(feature = "acquisition", target_os = "linux"))]
                if let Some((_, status, _)) = self.download
                    && !status.terminal()
                {
                    let action = if status == zerogravity::control::DownloadStatus::Paused {
                        DownloadAction::Resume
                    } else {
                        DownloadAction::Pause
                    };
                    add(
                        C::Download(action),
                        if action == DownloadAction::Pause {
                            "Pause"
                        } else {
                            "Resume"
                        },
                        if action == DownloadAction::Pause {
                            "p"
                        } else {
                            "s"
                        },
                        self.download(action),
                    );
                    add(
                        C::Download(DownloadAction::Cancel),
                        "Cancel transfer",
                        "c",
                        self.download(DownloadAction::Cancel),
                    );
                } else {
                    add(C::Library(L::Pull), "New pull", "d", self.pull());
                }
                #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
                add(C::Library(L::Pull), "New pull", "d", self.pull());
            }
            View::Devices => {
                if selected {
                    add(C::Library(L::Fit), "Check policy", "e", self.busy());
                    add(C::Library(L::Load(false)), "Load", "l", self.load());
                    add(C::Details, "Details", "D", None);
                }
            }
            View::Profiles => {
                add(C::Settings(S::Open), "Open editor", "Enter", select_row);
                add(C::Settings(S::Create), "Create", "c", self.busy());
            }
            View::Config => {
                add(
                    C::Settings(S::Choose),
                    "Edit",
                    "e",
                    self.file().or(select_row),
                );
                add(C::Settings(S::Validate), "Validate", "t", self.valid_file());
                add(
                    C::Settings(S::RawView),
                    if self.raw { "Typed view" } else { "Raw view" },
                    "v",
                    None,
                );
                add(C::Settings(S::RawEdit), "Raw edit", "w", self.file());
                add(
                    C::Settings(S::Activate),
                    "Activate",
                    "a",
                    self.valid_file().or(self
                        .editor
                        .as_ref()
                        .is_some_and(|f| f.target.path == self.active_document.path)
                        .then_some("Already active")),
                );
                add(C::Settings(S::Copy), "Copy", "y", self.valid_file());
                add(
                    C::Settings(S::Unset),
                    "Unset",
                    "z",
                    self.file()
                        .or((!self.explicit_value).then_some("Inherited or unset")),
                );
                let protected = self.editor.as_ref().is_none_or(|f| {
                    f.target.profile.as_deref().is_none_or(|n| n == "default")
                        || f.target.path == self.active_document.path
                });
                add(
                    C::Settings(S::Remove),
                    "Remove profile",
                    "x",
                    self.file()
                        .or(protected.then_some("Active/default/portable document")),
                );
            }
            View::Logs => {
                if let Some(id) = self.log {
                    add(C::LogDetails(id), "Details", "Enter", None);
                }
                add(
                    C::Logs(G::Follow),
                    if self.follow_logs {
                        "Pause view"
                    } else {
                        "Follow"
                    },
                    "p",
                    None,
                );
                add(C::Logs(G::PickLevel), "Level", "l", None);
                add(C::Logs(G::PickComponent), "Component", "f", None);
            }
        }
        if self.has_response && matches!(self.view, View::Home | View::Running) {
            add(
                C::FollowOutput,
                if self.follow_output {
                    "Pause output"
                } else {
                    "Follow output"
                },
                "F9",
                None,
            );
        }
        if let Some(active) = &self.active {
            add(
                C::Cancel,
                "Cancel action",
                "Ctrl-C",
                active.cancel_requested.then_some("Cancellation requested"),
            );
        }
        if selected && matches!(self.view, View::Profiles | View::Config) {
            add(C::Details, "Details", "D", None);
        }
        out
    }
    pub fn session_buttons(&self) -> Vec<Button> {
        let web = (!cfg!(feature = "web")).then_some("Web not compiled");
        vec![
            Button {
                command: Command::OpenWeb,
                label: "Open Web",
                key: "b",
                disabled: web,
            },
            Button {
                command: Command::Appearance,
                label: "Appearance",
                key: "T",
                disabled: None,
            },
            Button {
                command: Command::Refresh,
                label: "Refresh",
                key: "r",
                disabled: None,
            },
            Button {
                command: Command::Quit,
                label: "Quit",
                key: "Ctrl-Q",
                disabled: None,
            },
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Button {
    pub command: Command,
    pub label: &'static str,
    pub key: &'static str,
    pub disabled: Option<&'static str>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Bound {
    pub command: Command,
    scope: Rc<Scope>,
}
impl Bound {
    pub fn new(command: Command, scope: &Rc<Scope>) -> Self {
        Self {
            command,
            scope: Rc::clone(scope),
        }
    }
    pub fn valid(&self, app: &App) -> bool {
        let current = Scope::capture(app);
        self.scope.as_ref() == &current
            && current
                .buttons()
                .into_iter()
                .chain(current.session_buttons())
                .any(|button| button.command == self.command && button.disabled.is_none())
    }
}

#[derive(Debug)]
pub(super) struct Menu {
    pub scope: Rc<Scope>,
    pub title: String,
    pub buttons: Vec<Button>,
    pub cursor: usize,
    pub scroll: u16,
}
impl Menu {
    pub fn new(app: &App, session: bool) -> Self {
        let scope = Rc::new(Scope::capture(app));
        let title = if session {
            "Session".into()
        } else if let Some(kind) = scope.view.library() {
            app.library
                .rows(kind)
                .selected()
                .map_or_else(|| scope.view.name().to_owned(), |row| row.title.clone())
        } else if scope.view == View::Running {
            scope
                .owner
                .as_ref()
                .map_or_else(|| "Running actions".into(), |owner| owner.alias.clone())
        } else {
            scope.view.name().to_owned()
        };
        let buttons = if session {
            scope.session_buttons()
        } else {
            scope.buttons()
        };
        let cursor = buttons
            .iter()
            .position(|b| b.disabled.is_none())
            .unwrap_or(0);
        Self {
            scope,
            title,
            buttons,
            cursor,
            scroll: 0,
        }
    }
    pub fn selected(&self) -> Option<Bound> {
        let button = self.buttons.get(self.cursor)?;
        button
            .disabled
            .is_none()
            .then(|| Bound::new(button.command, &self.scope))
    }
}
