//! Terminal state is scoped to the foreground client, even on error or
//! unwind. Crossterm handles keys/resizes; external Unix signals request exit.
use crossterm::{
    cursor::Show,
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{DefaultTerminal, Terminal, backend::CrosstermBackend};
use std::{
    io::{self, stdout},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity::Result;

pub(super) struct Session {
    pub terminal: DefaultTerminal,
    active: bool,
}
impl Session {
    pub fn enter() -> Result<Self> {
        enable_raw_mode()?;
        if let Err(error) = execute!(stdout(), EnterAlternateScreen) {
            let _ = restore();
            return Err(error.into());
        }
        let terminal = match Terminal::new(CrosstermBackend::new(stdout())) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = restore();
                return Err(error.into());
            }
        };
        Ok(Self {
            terminal,
            active: true,
        })
    }
    pub fn restore(&mut self) -> Result<()> {
        restore()?;
        self.active = false;
        Ok(())
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if self.active {
            let _ = restore();
        }
    }
}
fn restore() -> io::Result<()> {
    let raw = disable_raw_mode();
    let screen = execute!(stdout(), LeaveAlternateScreen, Show);
    raw.and(screen)
}

pub(super) struct Stop {
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
