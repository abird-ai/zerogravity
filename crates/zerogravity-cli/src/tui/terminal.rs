//! Terminal state is scoped to the foreground client, even on error or
//! unwind. Crossterm handles keys/resizes; external Unix signals request exit.
use crossterm::{
    cursor::Show,
    event::{DisableBracketedPaste, EnableBracketedPaste},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{DefaultTerminal, Terminal, backend::CrosstermBackend};
use std::io::{self, stdout};
use zerogravity::Result;

pub(super) struct Session {
    pub terminal: DefaultTerminal,
    active: bool,
}
impl Session {
    pub fn enter() -> Result<Self> {
        enable_raw_mode()?;
        if let Err(error) = execute!(stdout(), EnterAlternateScreen, EnableBracketedPaste) {
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
    let screen = execute!(stdout(), DisableBracketedPaste, LeaveAlternateScreen, Show);
    raw.and(screen)
}

pub(super) use crate::control_host::Stop;
