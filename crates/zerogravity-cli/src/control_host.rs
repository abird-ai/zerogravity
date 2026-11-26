//! Foreground interactive host policy, shared by native and browser clients.
mod actions;
pub(crate) mod work;
#[cfg(feature = "tui")]
pub(crate) use actions::ActionSummary;
pub(crate) use actions::{Actions, Origin};
use std::{
    path::PathBuf,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
};
#[cfg(feature = "tui")]
use zerogravity::configuration::ConfigFile;
use zerogravity::{
    Error, Result, Runtime,
    configuration::{self as cfg, ConfigDocument},
    control::ServeSummary,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Target {
    pub path: PathBuf,
    pub profile: Option<String>,
    pub allow_missing: bool,
}
impl Target {
    pub fn selected(selection: Option<(&str, &str)>) -> Result<Self> {
        Ok(Self {
            path: crate::config_cli::selected_path(selection)?,
            profile: match selection {
                None => Some("default".into()),
                Some(("--profile", name)) => Some(name.into()),
                _ => None,
            },
            allow_missing: selection.is_none(),
        })
    }
    #[cfg(feature = "tui")]
    pub fn profile(name: &str) -> Result<Self> {
        Self::selected(if name == "default" {
            None
        } else {
            Some(("--profile", name))
        })
    }
    #[cfg(feature = "tui")]
    pub fn label(&self) -> String {
        self.profile
            .clone()
            .unwrap_or_else(|| self.path.display().to_string())
    }
    #[cfg(feature = "tui")]
    pub(crate) fn open(&self) -> Result<ConfigFile> {
        let file = ConfigFile::open(self.path.clone())?;
        if !self.allow_missing {
            file.require_existing()?;
        }
        Ok(file)
    }
    pub fn read(&self) -> Result<Option<(PathBuf, ConfigDocument)>> {
        cfg::load_path(self.path.clone(), self.allow_missing)
    }
}

#[derive(Clone)]
pub(crate) struct Document {
    pub target: Target,
    #[cfg_attr(
        not(feature = "tui"),
        allow(dead_code, reason = "terminal presentation only")
    )]
    pub ui: zerogravity::configuration::UiDoc,
    #[cfg_attr(
        not(feature = "tui"),
        allow(dead_code, reason = "terminal prompt default")
    )]
    pub chat: bool,
    pub offline: bool,
}
impl Document {
    #[cfg(all(test, feature = "tui"))]
    pub fn test_default() -> Self {
        Self {
            target: Target {
                path: "/nonexistent/config.yaml".into(),
                profile: Some("default".into()),
                allow_missing: true,
            },
            chat: true,
            offline: false,
            ui: Default::default(),
        }
    }

    pub fn new(selection: Option<(&str, &str)>, offline: bool) -> Result<Self> {
        let target = Target::selected(selection)?;
        let doc = target.read()?.map(|(_, doc)| doc).unwrap_or_default();
        Ok(Self::from_config(target, doc, offline))
    }
    pub fn from_config(
        target: Target,
        doc: zerogravity::configuration::ConfigDocument,
        offline: bool,
    ) -> Self {
        let chat = doc.run.and_then(|run| run.task).and_then(|task| task.raw) != Some(true);
        Self {
            target,
            chat,
            offline,
            ui: doc.ui.unwrap_or_default(),
        }
    }
    pub(crate) fn read(
        &self,
    ) -> Result<
        Option<(
            std::path::PathBuf,
            zerogravity::configuration::ConfigDocument,
        )>,
    > {
        self.target.read()
    }
}

/// One process-local selection. Workers take a cheap owned snapshot before
/// their cold operation; selection changes cannot reconfigure an admitted owner.
#[derive(Clone)]
pub(crate) struct Selection(Arc<Selected>);
struct Selected {
    document: RwLock<Document>,
    #[cfg(feature = "web")]
    revision: std::sync::atomic::AtomicU64,
}
impl Selection {
    pub fn new(document: Document) -> Self {
        Self(Arc::new(Selected {
            document: RwLock::new(document),
            #[cfg(feature = "web")]
            revision: std::sync::atomic::AtomicU64::new(0),
        }))
    }
    pub fn snapshot(&self) -> Document {
        self.0
            .document
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    #[cfg(feature = "web")]
    pub fn revision(&self) -> u64 {
        self.0.revision.load(Ordering::Acquire)
    }
    #[cfg(feature = "tui")]
    pub fn update(&self, document: Document) {
        *self
            .0
            .document
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = document;
        #[cfg(feature = "web")]
        self.0.revision.fetch_add(1, Ordering::Release);
    }
}

pub(crate) fn runtime() -> Result<Runtime> {
    Runtime::builder()
        .limits(zerogravity::runtime::SchedulerLimits {
            max_active: 32 * zerogravity::runtime::text::MAX_ACTIVE_PER_WORKER,
            max_queued: None,
            ..Default::default()
        })
        .build()
}

/// Foreground process exit, not a client navigation operation. The worker is
/// joined by the caller first; no concurrent start/stop remains in this UI.
pub(crate) fn retire(runtime: &Runtime) -> Result<()> {
    #[cfg(feature = "serve")]
    let stopped = (|| match runtime.control().serving()? {
        ServeSummary::Listening { binding, .. } | ServeSummary::Finished { binding } => {
            runtime.control().stop_serving(binding.generation)
        }
        ServeSummary::Idle | ServeSummary::Unavailable => Ok(()),
        ServeSummary::Transitioning => {
            Err(Error::State("Listener transition still in progress".into()))
        }
    })();
    #[cfg(not(feature = "serve"))]
    let stopped = {
        let _ = ServeSummary::Unavailable;
        Ok(())
    };
    combine(stopped, runtime.shutdown())
}

pub(crate) fn combine(primary: Result<()>, cleanup: Result<()>) -> Result<()> {
    match (primary, cleanup) {
        (Ok(()), result) | (result, Ok(())) => result,
        (Err(primary), Err(cleanup)) => Err(Error::State(format!("{primary}; cleanup: {cleanup}"))),
    }
}

pub(crate) struct Stop {
    flag: Arc<AtomicBool>,
    #[cfg(unix)]
    signals: Vec<signal_hook::SigId>,
}
impl Stop {
    pub fn install() -> Result<Self> {
        let flag = Arc::new(AtomicBool::new(false));
        #[cfg(unix)]
        {
            let mut stop = Self {
                flag,
                signals: Vec::new(),
            };
            for signal in [
                signal_hook::consts::SIGINT,
                signal_hook::consts::SIGTERM,
                signal_hook::consts::SIGHUP,
            ] {
                stop.signals
                    .push(signal_hook::flag::register(signal, stop.flag.clone())?);
            }
            Ok(stop)
        }
        #[cfg(not(unix))]
        Ok(Self { flag })
    }
    pub fn requested(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}
#[cfg(unix)]
impl Drop for Stop {
    fn drop(&mut self) {
        for id in self.signals.drain(..) {
            signal_hook::low_level::unregister(id);
        }
    }
}
