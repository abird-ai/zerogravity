//! Visible native actions. Bind to observed identities, not coordinates or
//! simulated keystrokes. The original App handlers remain the effect owners.
use super::{
    App,
    library::{Key, Kind, Shelf},
    settings::Snapshot,
    view::{Action, DownloadAction, LibraryAction, LogAction, SettingsAction, View},
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
    Download(DownloadAction),
    Settings(SettingsAction),
    Logs(LogAction),
    Navigate(View),
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
            Self::Download(action) => Action::Download(action),
            Self::Settings(action) => Action::Settings(action),
            Self::Logs(action) => Action::Logs(action),
            Self::Navigate(view) => Action::Navigate(view),
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
        let selection = kind.and_then(|kind| app.library.rows(kind).selected.clone());
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
    fn download(&self, action: DownloadAction) -> Option<&'static str> {
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        {
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
        #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
        {
            let _ = action;
            Some("Acquisition not compiled")
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
        let snapshot = matches!(self.selection, Some(Key::Snapshot(_)));
        let choose_snapshot = (!snapshot).then_some("Select a cached snapshot");
        let device = (!matches!(self.selection, Some(Key::Device(_)))).then_some("Select a device");
        let choose_owner = self.owner.is_none().then_some("No ready registration");
        let select_row = (!selected).then_some("Select a row");
        let web = (!cfg!(feature = "web")).then_some("Web not compiled");
        match self.view {
            View::Home => {
                add(C::Run, "Run", "Enter", self.run());
                add(C::Load, "Load", "l", self.load());
                add(C::Navigate(View::Models), "Models", "4", None);
                add(C::Navigate(View::Serve), "Serve", "3", None);
                add(C::Navigate(View::Downloads), "Downloads", "5", None);
                add(C::OpenWeb, "Web", "b", web);
            }
            View::Running => {
                add(C::Run, "Run", "Enter", self.run());
                add(C::Load, "Load", "l", self.load());
                add(C::Unload, "Unload", "u", self.busy().or(choose_owner));
            }
            View::Serve => {
                let start = match self.listener {
                    Listener::Unavailable => Some("Serve not compiled"),
                    Listener::Idle => self.owner.is_none().then_some("Load a model first"),
                    _ => Some("Stop current listener first"),
                };
                add(C::Start, "Start", "s", self.busy().or(start));
                add(
                    C::Stop,
                    "Stop",
                    "t",
                    self.busy().or((!matches!(
                        self.listener,
                        Listener::Listening(_) | Listener::Finished(_)
                    ))
                    .then_some("No owned listener")),
                );
                add(C::Load, "Load", "l", self.load());
                add(C::Navigate(View::Models), "Models", "4", None);
                add(C::OpenWeb, "Web", "b", web);
            }
            View::Models => {
                add(
                    C::Library(L::Load(false)),
                    "Load",
                    "l",
                    self.load().or(choose_snapshot),
                );
                add(
                    C::Library(L::Run),
                    "Run",
                    "p",
                    self.load().or(choose_snapshot),
                );
                add(C::Library(L::Pull), "Pull", "d", self.pull());
                add(
                    C::Library(L::Inspect),
                    "Inspect",
                    "Enter",
                    self.busy().or(select_row),
                );
                add(
                    C::Library(L::Load(true)),
                    "Load for Serve",
                    "s",
                    self.load().or(choose_snapshot),
                );
                add(
                    C::Library(L::Verify),
                    "Verify",
                    "v",
                    self.busy().or(choose_snapshot),
                );
                add(
                    C::Library(L::Alias),
                    "Alias",
                    "a",
                    self.busy().or(choose_snapshot),
                );
                add(
                    C::Library(L::Reclaim(false)),
                    "Removal preview",
                    "X",
                    self.busy().or(choose_snapshot),
                );
                add(
                    C::Library(L::Reclaim(true)),
                    "Remove",
                    "x",
                    self.busy().or(choose_snapshot),
                );
                // Available in overflow even when shelf tabs cannot fit.
                for (shelf, label) in [
                    (Shelf::Cached, "Cached"),
                    (Shelf::Catalog, "Catalog"),
                    (Shelf::Partial, "Partial"),
                ] {
                    add(C::Library(L::SelectShelf(shelf)), label, "m", None);
                }
            }
            View::Downloads => {
                add(
                    C::Download(DownloadAction::Pause),
                    "Pause",
                    "p",
                    self.download(DownloadAction::Pause),
                );
                add(
                    C::Download(DownloadAction::Resume),
                    "Resume",
                    "s",
                    self.download(DownloadAction::Resume),
                );
                add(
                    C::Download(DownloadAction::Cancel),
                    "Cancel transfer",
                    "c",
                    self.download(DownloadAction::Cancel),
                );
                add(C::Library(L::Pull), "New pull", "d", self.pull());
            }
            View::Devices => {
                add(
                    C::Library(L::Fit),
                    "Check policy",
                    "e",
                    self.busy().or(device),
                );
                add(
                    C::Library(L::Load(false)),
                    "Load",
                    "l",
                    self.load().or(device),
                );
            }
            View::Profiles => {
                add(C::Settings(S::Open), "Open editor", "Enter", select_row);
                add(C::Settings(S::Create), "Create", "c", self.busy());
            }
            View::Config => {
                add(
                    C::Settings(S::Open),
                    "Edit",
                    "Enter",
                    self.file().or(select_row),
                );
                add(C::Settings(S::Search), "Search", "/", None);
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
                add(C::Logs(G::Search), "Search", "/", None);
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
                add(C::Logs(G::Level), "Level", "l", None);
                add(C::Logs(G::Component), "Component", "f", None);
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
        add(C::Refresh, "Refresh", "r", None);
        add(C::Appearance, "Appearance", "T", None);
        add(C::Quit, "Quit", "q", None);
        out
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
                .iter()
                .any(|button| button.command == self.command && button.disabled.is_none())
    }
}

#[derive(Debug)]
pub(super) struct Menu {
    pub scope: Rc<Scope>,
    pub buttons: Vec<Button>,
    pub cursor: usize,
    pub scroll: u16,
}
impl Menu {
    pub fn new(app: &App) -> Self {
        let scope = Rc::new(Scope::capture(app));
        let buttons = scope.buttons();
        let cursor = buttons
            .iter()
            .position(|b| b.disabled.is_none())
            .unwrap_or(0);
        Self {
            scope,
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
