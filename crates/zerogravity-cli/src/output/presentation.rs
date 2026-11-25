use std::{
    fmt::{self, Display},
    io::{self, IsTerminal, Write},
    sync::OnceLock,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Switch {
    #[default]
    Auto,
    Always,
    Never,
}

impl Switch {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "always" => Some(Self::Always),
            "never" => Some(Self::Never),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Verbosity {
    Quiet,
    #[default]
    Normal,
    Verbose,
    Diagnostic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) plain: bool,
    pub(crate) color: Switch,
    pub(crate) progress: Switch,
    pub(crate) verbosity: Verbosity,
    pub(crate) jsonl: bool,
    pub(crate) interactive: Switch,
}

#[derive(Clone, Copy)]
struct Terminal {
    input: bool,
    diagnostics: bool,
    dumb: bool,
    no_color: bool,
}

impl Terminal {
    fn detect() -> Self {
        Self {
            input: io::stdin().is_terminal(),
            diagnostics: io::stderr().is_terminal(),
            dumb: std::env::var_os("TERM").is_some_and(|term| term == "dumb"),
            no_color: std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Reporter {
    rich: bool,
    color: bool,
    progress: bool,
    verbosity: Verbosity,
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    jsonl: bool,
    interactive: Switch,
    terminal_input: bool,
}

// Human presentation only. Explicit machine diagnostics and primary stdout
// retain their original contracts; events cannot authorize runtime effects.
#[derive(Clone, Copy)]
#[allow(dead_code, reason = "event producers are independently feature-gated")]
pub(crate) enum Event<'a> {
    Status(fmt::Arguments<'a>),
    Warning(fmt::Arguments<'a>),
    Progress(fmt::Arguments<'a>),
    Verbose(fmt::Arguments<'a>),
    Diagnostic(fmt::Arguments<'a>),
    Explicit(fmt::Arguments<'a>),
}

impl Reporter {
    fn new(options: Options, terminal: Terminal) -> Self {
        let rich = !options.plain && terminal.input && terminal.diagnostics && !terminal.dumb;
        Self {
            rich,
            color: match options.color {
                Switch::Always => true,
                Switch::Never => false,
                Switch::Auto => rich && !terminal.no_color,
            },
            progress: options.progress != Switch::Never,
            verbosity: options.verbosity,
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            jsonl: options.jsonl,
            interactive: options.interactive,
            terminal_input: terminal.input && terminal.diagnostics && !terminal.dumb,
        }
    }

    pub(crate) fn progress(self) -> bool {
        self.progress && self.verbosity != Verbosity::Quiet
    }

    #[cfg(feature = "serve")]
    pub(crate) fn serve_status(self) -> bool {
        self.rich && self.progress()
    }

    /// Interactive prompts require terminal input and human-visible output.
    /// Auto keeps explicit plain/quiet operation noninteractive.
    pub(crate) fn run_interaction(self) -> zerogravity::Result<bool> {
        let capable =
            self.terminal_input && io::stdout().is_terminal() && self.verbosity != Verbosity::Quiet;
        match self.interactive {
            Switch::Never => Ok(false),
            Switch::Auto => Ok(capable && self.rich),
            Switch::Always if capable => Ok(true),
            Switch::Always => Err(zerogravity::Error::Unsupported(
                "interactive run requires a capable terminal".into(),
            )),
        }
    }

    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    pub(crate) fn download_jsonl(self) -> bool {
        self.jsonl
    }

    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    pub(crate) fn download_interaction(self) -> Switch {
        self.interactive
    }

    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    pub(crate) fn download_auto_input(self) -> bool {
        self.rich && self.progress() && !self.jsonl && io::stdout().is_terminal()
    }

    fn allows(self, event: Event<'_>) -> bool {
        match event {
            Event::Status(_) => self.verbosity != Verbosity::Quiet,
            Event::Warning(_) | Event::Explicit(_) => true,
            Event::Progress(_) => self.progress(),
            Event::Verbose(_) => {
                matches!(self.verbosity, Verbosity::Verbose | Verbosity::Diagnostic)
            }
            Event::Diagnostic(_) => self.verbosity == Verbosity::Diagnostic,
        }
    }

    pub(crate) fn write(self, event: Event<'_>, out: &mut impl Write) -> io::Result<()> {
        if !self.allows(event) {
            return Ok(());
        }
        let (message, symbol, color) = match event {
            Event::Status(message) => (message, "● ", "36"),
            // Actionable warnings, e.g. uncertain cache retirement, must not
            // disappear just because ordinary progress is suppressed.
            Event::Warning(message) => (message, "! ", "33"),
            Event::Progress(message) => (message, "◐ ", "36"),
            Event::Verbose(message) | Event::Diagnostic(message) => (message, "  ", "2"),
            Event::Explicit(message) => return writeln!(out, "{message}"),
        };
        let prefix = if self.rich {
            symbol
        } else if matches!(event, Event::Warning(_)) {
            "WARNING: "
        } else {
            ""
        };
        if self.color {
            writeln!(out, "\x1b[{color}m{prefix}{message}\x1b[0m")
        } else {
            writeln!(out, "{prefix}{message}")
        }
    }

    pub(crate) fn failure(self, failure: Failure<'_>, out: &mut impl Write) -> io::Result<()> {
        let prefix = if self.rich { "✖ error" } else { "error" };
        if self.color {
            write!(out, "\x1b[31m{prefix}[{}]\x1b[0m: ", failure.code)?;
        } else {
            write!(out, "{prefix}[{}]: ", failure.code)?;
        }
        if failure.command.is_empty() {
            write!(out, "zerogravity: ")?;
        } else {
            write!(out, "zerogravity {}: ", failure.command)?;
        }
        writeln!(out, "{}", failure.summary)?;
        if let Some(hint) = failure.hint {
            writeln!(out, "hint: {hint}")?;
        }
        if self.verbosity == Verbosity::Diagnostic {
            let mut cause = failure.cause;
            while let Some(error) = cause {
                writeln!(out, "cause: {error}")?;
                cause = error.source();
            }
        }
        Ok(())
    }
}

pub(crate) struct Failure<'a> {
    command: &'static str,
    code: &'static str,
    summary: &'a dyn Display,
    hint: Option<&'static str>,
    cause: Option<&'a (dyn std::error::Error + 'static)>,
}

impl<'a> Failure<'a> {
    pub(crate) fn new(
        command: &'static str,
        code: &'static str,
        summary: &'a dyn Display,
        hint: Option<&'static str>,
    ) -> Self {
        Self {
            command,
            code,
            summary,
            hint,
            cause: None,
        }
    }

    pub(crate) fn core(command: &'static str, error: &'a zerogravity::Error) -> Self {
        use std::error::Error as _;
        use zerogravity::Error;
        let (code, hint) = match error {
            Error::Malformed(_) => ("malformed", Some("Check the input and command help.")),
            Error::Unsupported(_) => ("unsupported", None),
            Error::Resource(_) => (
                "resource_exhausted",
                Some("Reduce the requested capacity or free the reported resource."),
            ),
            Error::Limit(_) => ("limit_exceeded", Some("Check the configured bounds.")),
            Error::State(_) => ("invalid_state", None),
            Error::Io(_) => ("io_error", None),
        };
        Self {
            cause: error.source(),
            ..Self::new(command, code, error, hint)
        }
    }
}

static REPORTER: OnceLock<Reporter> = OnceLock::new();

pub(crate) fn install(options: Options) {
    // Set once before dispatch; never consult environment on an event hot path.
    let _ = REPORTER.set(Reporter::new(options, Terminal::detect()));
}

pub(crate) fn reporter() -> Reporter {
    *REPORTER.get_or_init(|| Reporter::new(Options::default(), Terminal::detect()))
}

pub(crate) fn report(event: Event<'_>) {
    // Presentation is best-effort and cannot change an execution outcome.
    let reporter = reporter();
    if reporter.allows(event) {
        let _ = reporter.write(event, &mut io::stderr().lock());
    }
}

pub(crate) fn fail(failure: Failure<'_>) {
    let _ = reporter().failure(failure, &mut io::stderr().lock());
}

macro_rules! emit {
    ($variant:ident, $($arg:tt)*) => {
        $crate::output::report($crate::output::Event::$variant(format_args!($($arg)*)))
    };
}
pub(crate) use emit;

#[cfg(test)]
mod tests {
    use super::*;

    const TTY: Terminal = Terminal {
        input: true,
        diagnostics: true,
        dumb: false,
        no_color: false,
    };

    #[test]
    fn capability_and_explicit_color_policy() {
        let auto = Options::default();
        let rich = Reporter::new(auto, TTY);
        assert!(rich.rich && rich.color);
        for terminal in [
            Terminal {
                input: false,
                ..TTY
            },
            Terminal {
                diagnostics: false,
                ..TTY
            },
            Terminal { dumb: true, ..TTY },
        ] {
            let plain = Reporter::new(auto, terminal);
            assert!(!plain.rich && !plain.color);
        }
        let no_color = Terminal {
            no_color: true,
            ..TTY
        };
        assert!(!Reporter::new(auto, no_color).color);
        assert!(
            Reporter::new(
                Options {
                    color: Switch::Always,
                    ..auto
                },
                no_color
            )
            .color
        );
        assert!(
            !Reporter::new(
                Options {
                    color: Switch::Never,
                    ..auto
                },
                TTY
            )
            .color
        );
        assert!(
            !Reporter::new(
                Options {
                    plain: true,
                    ..auto
                },
                TTY
            )
            .rich
        );
    }

    #[test]
    fn output_levels_progress_and_explicit_diagnostics() {
        let mut out = Vec::new();
        let quiet = Reporter::new(
            Options {
                verbosity: Verbosity::Quiet,
                ..Default::default()
            },
            TTY,
        );
        for event in [
            Event::Status(format_args!("status")),
            Event::Progress(format_args!("progress")),
            Event::Verbose(format_args!("verbose")),
            Event::Diagnostic(format_args!("diagnostic")),
        ] {
            quiet.write(event, &mut out).unwrap();
        }
        assert!(out.is_empty());
        quiet
            .write(Event::Explicit(format_args!("{{\"schema\":1}}")), &mut out)
            .unwrap();
        assert_eq!(out, b"{\"schema\":1}\n");
        out.clear();
        quiet
            .write(
                Event::Warning(format_args!("incomplete retirement")),
                &mut out,
            )
            .unwrap();
        assert!(
            String::from_utf8(out.clone())
                .unwrap()
                .contains("incomplete retirement")
        );

        let verbose = Reporter::new(
            Options {
                verbosity: Verbosity::Verbose,
                progress: Switch::Never,
                plain: true,
                ..Default::default()
            },
            TTY,
        );
        out.clear();
        verbose
            .write(Event::Verbose(format_args!("detail")), &mut out)
            .unwrap();
        verbose
            .write(Event::Progress(format_args!("hidden")), &mut out)
            .unwrap();
        verbose
            .write(Event::Diagnostic(format_args!("hidden")), &mut out)
            .unwrap();
        assert_eq!(out, b"detail\n");
        let diagnostic = Reporter::new(
            Options {
                verbosity: Verbosity::Diagnostic,
                plain: true,
                ..Default::default()
            },
            TTY,
        );
        diagnostic
            .write(Event::Diagnostic(format_args!("diagnostic")), &mut out)
            .unwrap();
        assert_eq!(out, b"detail\ndiagnostic\n");
    }

    #[test]
    fn plain_rich_and_typed_error() {
        let error = zerogravity::Error::resource("VRAM", "load", 8, 4);
        let failure = || Failure::core("run", &error);
        let mut out = Vec::new();
        let plain = Reporter::new(
            Options {
                plain: true,
                ..Default::default()
            },
            TTY,
        );
        plain.failure(failure(), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("error[resource_exhausted]: zerogravity run:"));
        assert!(text.contains("requested 8 bytes, available 4 bytes"));
        assert!(text.contains("hint:"));
        assert!(!text.contains('\x1b'));

        let mut out = Vec::new();
        let rich = Reporter::new(Options::default(), TTY);
        rich.failure(failure(), &mut out).unwrap();
        assert!(
            String::from_utf8(out)
                .unwrap()
                .starts_with("\x1b[31m✖ error[resource_exhausted]\x1b[0m")
        );
    }
}
