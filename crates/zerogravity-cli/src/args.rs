//! Product command grammar. The existing execution and acquisition grammars
//! retain ownership of their arguments and validation.
use crate::output::{Options, Switch, Verbosity};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Format {
    #[default]
    Auto,
    Text,
    Json,
    Jsonl,
}

#[derive(Debug, Default)]
struct Globals {
    presentation: Options,
    format: Format,
    interactive: Switch,
    offline: bool,
    yes: bool,
    document: Option<(&'static str, String)>,
}

impl Globals {
    // Consume one option at a grammar boundary. Never search inside opaque
    // provider argv or a command-owned value (e.g. --prompt "--quiet").
    fn take(&mut self, args: &[String], document: bool) -> Result<usize, &'static str> {
        let Some(raw) = args.first() else {
            return Ok(0);
        };
        let (flag, inline) = raw
            .split_once('=')
            .map_or((raw.as_str(), None), |(a, b)| (a, Some(b)));
        let value = || {
            inline
                .map(|v| (v, 1))
                .or_else(|| args.get(1).map(|v| (v.as_str(), 2)))
                .ok_or("global option requires a value")
        };
        match flag {
            "-q" | "--quiet" if inline.is_none() => {
                if matches!(
                    self.presentation.verbosity,
                    Verbosity::Verbose | Verbosity::Diagnostic
                ) {
                    return Err("quiet and verbose are mutually exclusive");
                }
                self.presentation.verbosity = Verbosity::Quiet;
            }
            "-v" | "--verbose" | "-vv" if inline.is_none() => {
                self.presentation.verbosity = match (self.presentation.verbosity, flag) {
                    (Verbosity::Quiet, _) => {
                        return Err("quiet and verbose are mutually exclusive");
                    }
                    (Verbosity::Normal, "-v" | "--verbose") => Verbosity::Verbose,
                    _ => Verbosity::Diagnostic,
                };
            }
            "--offline" if inline.is_none() => self.offline = true,
            "--yes" if inline.is_none() => self.yes = true,
            "--format" | "--color" | "--progress" | "--interactive" => {
                let (v, used) = value()?;
                match flag {
                    "--format" => {
                        self.format = match v {
                            "auto" => Format::Auto,
                            "text" => Format::Text,
                            "json" => Format::Json,
                            "jsonl" => Format::Jsonl,
                            _ => return Err("format requires auto, text, json or jsonl"),
                        }
                    }
                    "--color" => {
                        self.presentation.color =
                            Switch::parse(v).ok_or("color requires auto, always or never")?
                    }
                    "--progress" => {
                        self.presentation.progress =
                            Switch::parse(v).ok_or("progress requires auto, always or never")?
                    }
                    _ => {
                        self.interactive =
                            Switch::parse(v).ok_or("interactive requires auto, always or never")?
                    }
                }
                return Ok(used);
            }
            "--config" | "--profile" if document => {
                let (v, used) = value()?;
                if self.document.is_some() {
                    return Err("select exactly one --config or --profile");
                }
                self.document = Some((
                    if flag == "--config" {
                        "--config"
                    } else {
                        "--profile"
                    },
                    v.into(),
                ));
                return Ok(used);
            }
            _ => return Ok(0),
        }
        Ok(1)
    }

    fn prefix<'a>(
        &mut self,
        mut args: &'a [String],
        document: bool,
    ) -> Result<&'a [String], &'static str> {
        loop {
            if args.first().is_some_and(|v| v == "--") {
                return Ok(&args[1..]);
            }
            let used = self.take(args, document)?;
            if used == 0 {
                return Ok(args);
            }
            args = &args[used..];
        }
    }

    // The run and serve owners each have a flat grammar with one value per
    // value-taking flag. Skip the entire owner argument, including flag-like
    // values. Other legacy/sensitive grammars are opaque after their prefix.
    fn flat(
        &mut self,
        mut args: &[String],
        takes_value: impl Fn(&str) -> bool,
    ) -> Result<Vec<String>, &'static str> {
        let mut result = Vec::new();
        while let Some(raw) = args.first() {
            let used = self.take(args, false)?;
            if used != 0 {
                args = &args[used..];
                continue;
            }
            result.push(raw.clone());
            let used = if takes_value(raw) && args.len() > 1 {
                result.push(args[1].clone());
                2
            } else {
                1
            };
            args = &args[used..];
        }
        Ok(result)
    }

    fn finish(&mut self) -> Options {
        self.presentation.plain = self.format != Format::Auto;
        if matches!(self.format, Format::Json | Format::Jsonl)
            && self.presentation.progress == Switch::Auto
        {
            self.presentation.progress = Switch::Never;
        }
        self.presentation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Topic {
    Run,
    Serve,
    Model,
    Device,
    Pipeline,
    Profile,
    Config,
    Doctor,
    Web,
    Completion,
    Help,
    Version,
    Cache,
}

pub(super) struct Node {
    pub(super) topic: Topic,
    pub(super) name: &'static str,
    pub(super) description: &'static str,
    pub(super) children: &'static [&'static str],
}

pub(super) const TREE: &[Node] = &[
    Node {
        topic: Topic::Run,
        name: "run",
        description: "Run a model or run document",
        children: &[],
    },
    Node {
        topic: Topic::Serve,
        name: "serve",
        description: "Start a local OpenAI-compatible API",
        children: &[],
    },
    Node {
        topic: Topic::Model,
        name: "model",
        description: "List, inspect and manage cached models",
        children: &[
            "list", "inspect", "search", "pull", "verify", "remove", "import", "alias",
        ],
    },
    Node {
        topic: Topic::Device,
        name: "device",
        description: "Observe available hardware",
        children: &["list", "inspect"],
    },
    Node {
        topic: Topic::Pipeline,
        name: "pipeline",
        description: "Check or run a declared pipeline",
        children: &["check", "run"],
    },
    Node {
        topic: Topic::Profile,
        name: "profile",
        description: "Named run documents (management pending)",
        children: &[],
    },
    Node {
        topic: Topic::Config,
        name: "config",
        description: "Configuration management (pending)",
        children: &[],
    },
    Node {
        topic: Topic::Doctor,
        name: "doctor",
        description: "Local diagnostics (pending)",
        children: &[],
    },
    Node {
        topic: Topic::Web,
        name: "web",
        description: "Local control UI (pending)",
        children: &[],
    },
    Node {
        topic: Topic::Cache,
        name: "cache",
        description: "Advanced cache maintenance",
        children: &["remove", "by-name", "import-hf"],
    },
    Node {
        topic: Topic::Completion,
        name: "completion",
        description: "Generate shell completions",
        children: &["bash", "zsh", "fish", "powershell"],
    },
    Node {
        topic: Topic::Help,
        name: "help",
        description: "Show command help",
        children: &[],
    },
    Node {
        topic: Topic::Version,
        name: "version",
        description: "Print the version",
        children: &[],
    },
];

pub(super) fn node(topic: Topic) -> &'static Node {
    TREE.iter()
        .find(|n| n.topic == topic)
        .expect("known command")
}

fn topic(name: &str) -> Option<Topic> {
    TREE.iter().find(|n| n.name == name).map(|n| n.topic)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shell {
    Bash,
    Zsh,
    Fish,
    Pwsh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ModelRead {
    List,
    Inspect,
    Verify,
    Remove,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ModelCommand {
    Read {
        operation: ModelRead,
        args: Vec<String>,
        format: Format,
        yes: bool,
    },
    Search(Vec<String>),
    Pull(Vec<String>),
    Import(Vec<String>),
    Alias(Vec<String>),
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Command {
    Help,
    HelpTopic(Topic, Option<String>),
    Version,
    Completion(Shell),
    Inspect(String),
    Run {
        args: Vec<String>,
        alias: Option<String>,
        format: Format,
    },
    Devices(Vec<String>),
    DeviceInspect {
        selector: String,
        format: Format,
    },
    Model(ModelCommand),
    Pull(Vec<String>),
    Models(Vec<String>),
    Pipeline(Vec<String>),
    Serve(Vec<String>),
    CacheRun(Vec<String>),
    CacheManage(Vec<String>),
    NotImplemented(String),
    Invalid(String),
    Unknown,
}

fn help(args: &[String]) -> bool {
    matches!(args, [one] if matches!(one.as_str(), "-h" | "--help" | "help"))
}
fn unsupported_format(format: Format) -> Result<(), &'static str> {
    if matches!(format, Format::Json | Format::Jsonl) {
        Err("structured format is unavailable for this command; use its documented output options")
    } else {
        Ok(())
    }
}
fn no_jsonl(format: Format) -> Result<(), &'static str> {
    if format == Format::Jsonl {
        Err("JSONL lifecycle events are not available yet")
    } else {
        Ok(())
    }
}
fn with_prefix(prefix: &str, args: &[String]) -> Vec<String> {
    std::iter::once(prefix.into())
        .chain(args.iter().cloned())
        .collect()
}

fn parse_command(args: &[String], g: &mut Globals) -> Result<Command, &'static str> {
    let args = g.prefix(args, true)?;
    let Some(first) = args.first() else {
        if g.document.is_some() {
            return Err("select a command for --config or --profile");
        }
        return Ok(Command::Help);
    };
    let name = first.as_str();
    let rest = g.prefix(&args[1..], false)?;
    let command = match name {
        "-h" | "--help" => Command::Help,
        "-V" | "--version" | "version" => Command::Version,
        "help" => match rest {
            [] => Command::Help,
            [command, children @ ..] if children.len() <= 1 => {
                let topic = topic(command)
                    .or(match command.as_str() {
                        "devices" => Some(Topic::Device),
                        "pull" | "models" | "inspect" => Some(Topic::Model),
                        "run-cache" | "run-cpu" | "run-cuda" | "run-rocm" | "run-vulkan"
                        | "run-vulkan-amd" | "run-vulkan-nvidia" => Some(Topic::Run),
                        _ => None,
                    })
                    .ok_or("unknown help topic")?;
                if children
                    .first()
                    .is_some_and(|c| !node(topic).children.contains(&c.as_str()))
                {
                    return Err("unknown help topic");
                }
                Command::HelpTopic(topic, children.first().cloned())
            }
            _ => return Err("help accepts a command and optional subcommand"),
        },
        "inspect" => match rest {
            [path] if !path.is_empty() => Command::Inspect(path.clone()),
            _ => return Err("inspect requires exactly one local path"),
        },
        "run" | "run-cpu" | "run-cuda" | "run-rocm" | "run-vulkan" | "run-vulkan-amd"
        | "run-vulkan-nvidia" => {
            let mut args = g.flat(rest, crate::run_cli::takes_value)?;
            if let Some((flag, value)) = g.document.take() {
                args.splice(..0, [flag.into(), value]);
            }
            no_jsonl(g.format)?;
            Command::Run {
                args,
                alias: (name != "run").then(|| name.into()),
                format: g.format,
            }
        }
        "serve" => {
            let args = g.flat(rest, |raw| {
                !matches!(
                    raw,
                    "--enable-per-request-metrics" | "-h" | "--help" | "help"
                )
            })?;
            if help(&args) {
                Command::HelpTopic(Topic::Serve, None)
            } else {
                if let Some((flag, value)) = g.document.take() {
                    if flag != "--config" {
                        return Err("serve named profiles await shared serving configuration");
                    }
                    if !args.is_empty() {
                        return Err("serving --config must stand alone");
                    }
                    Command::Serve(vec![flag.into(), value])
                } else {
                    Command::Serve(args)
                }
            }
        }
        "device" | "devices" => {
            if help(rest) {
                Command::HelpTopic(Topic::Device, None)
            } else {
                let (operation, tail) = rest
                    .split_first()
                    .map_or(("list", rest), |(op, tail)| (op.as_str(), tail));
                let tail = g.flat(tail, |_| false)?;
                no_jsonl(g.format)?;
                if help(&tail) {
                    return Ok(Command::HelpTopic(Topic::Device, Some(operation.into())));
                }
                match operation {
                    "list" if tail.is_empty() || tail == ["--json"] => {
                        let mut args = vec!["list".into()];
                        if g.format == Format::Json || tail == ["--json"] {
                            args.push("--json".into());
                        }
                        Command::Devices(args)
                    }
                    "inspect" if name == "device" => match tail.as_slice() {
                        [selector] => Command::DeviceInspect {
                            selector: selector.clone(),
                            format: g.format,
                        },
                        _ => return Err("device inspect requires exactly one observed device ID"),
                    },
                    _ => return Err("device list [--json] | device inspect ID"),
                }
            }
        }
        "model" => {
            if rest.is_empty() || help(rest) {
                Command::HelpTopic(Topic::Model, None)
            } else {
                let operation = rest[0].as_str();
                if !node(Topic::Model).children.contains(&operation) {
                    return Err("unknown model subcommand");
                }
                let tail = g.prefix(&rest[1..], false)?;
                if help(tail) {
                    Command::HelpTopic(Topic::Model, Some(operation.into()))
                } else {
                    Command::Model(match operation {
                        "list" | "inspect" | "verify" | "remove" => {
                            let args =
                                g.flat(tail, |raw| matches!(raw, "--cache" | "--snapshot"))?;
                            no_jsonl(g.format)?;
                            let operation = match operation {
                                "list" => ModelRead::List,
                                "inspect" => ModelRead::Inspect,
                                "verify" => ModelRead::Verify,
                                _ => ModelRead::Remove,
                            };
                            if operation == ModelRead::Remove {
                                unsupported_format(g.format)?;
                            }
                            ModelCommand::Read {
                                operation,
                                args,
                                format: g.format,
                                yes: g.yes,
                            }
                        }
                        "search" => ModelCommand::Search(tail.to_vec()),
                        "pull" => ModelCommand::Pull(tail.to_vec()),
                        "import" => ModelCommand::Import(tail.to_vec()),
                        "alias" => ModelCommand::Alias(tail.to_vec()),
                        _ => unreachable!(),
                    })
                }
            }
        }
        "pipeline" => {
            if help(rest) {
                Command::HelpTopic(Topic::Pipeline, None)
            } else if let Some((operation, tail)) = rest.split_first() {
                Command::Pipeline(with_prefix(operation, g.prefix(tail, false)?))
            } else {
                Command::Pipeline(Vec::new())
            }
        }
        "profile" | "config" | "doctor" | "web" => {
            if help(rest) {
                Command::HelpTopic(topic(name).expect("known command"), None)
            } else {
                Command::NotImplemented(name.into())
            }
        }
        "completion" => {
            if help(rest) {
                Command::HelpTopic(Topic::Completion, None)
            } else {
                Command::Completion(match rest {
                    [shell] => match shell.as_str() {
                        "bash" => Shell::Bash,
                        "zsh" => Shell::Zsh,
                        "fish" => Shell::Fish,
                        "powershell" => Shell::Pwsh,
                        _ => return Err("completion requires bash, zsh, fish or powershell"),
                    },
                    _ => return Err("completion requires one shell"),
                })
            }
        }
        "pull" => Command::Pull(rest.to_vec()),
        "models" => Command::Models(rest.to_vec()),
        "cache" => Command::CacheManage(rest.to_vec()),
        "run-cache" => Command::CacheRun(rest.to_vec()),
        _ => Command::Unknown,
    };
    if g.document.is_some() {
        return Err("--config/--profile selection is unavailable for this command");
    }
    if g.interactive == Switch::Always {
        return Err("forced interactive mode is not implemented yet");
    }
    if g.yes
        && !matches!(
            command,
            Command::Model(ModelCommand::Read {
                operation: ModelRead::Remove,
                ..
            })
        )
    {
        return Err("--yes is only supported for model remove");
    }
    if !matches!(
        command,
        Command::Devices(_)
            | Command::DeviceInspect { .. }
            | Command::Run { .. }
            | Command::Model(ModelCommand::Read { .. })
    ) {
        unsupported_format(g.format)?;
    }
    // Forward a restriction at the owner's first option boundary, not
    // after a possibly incomplete value-taking argument.
    if g.offline {
        return match command {
            Command::Pull(args) => Ok(Command::Pull(offline_pull(args, false)?)),
            Command::Model(ModelCommand::Pull(args)) => Ok(Command::Model(ModelCommand::Pull(
                offline_pull(args, true)?,
            ))),
            other => Ok(other),
        };
    }
    Ok(command)
}

fn offline_pull(mut args: Vec<String>, canonical: bool) -> Result<Vec<String>, &'static str> {
    if help(&args) || args.get(1..).is_some_and(help) {
        return Ok(args);
    }
    let first = args.first().map(String::as_str);
    let boundary = match first {
        Some("list" | "object" | "oci") => 1,
        Some(
            "hf" | "github" | "inspect" | "verify" | "components" | "xet" | "modelscope" | "azure"
            | "s3" | "gcs",
        ) => 2,
        Some("candidates") => 3,
        Some(locator)
            if canonical && (locator.starts_with("hf:") || locator.starts_with("gh:")) =>
        {
            1
        }
        _ => return Err("offline pull requires an explicit supported provider"),
    };
    if args.len() < boundary {
        return Err("offline pull requires its provider positional arguments");
    }
    args.insert(boundary, "--offline".into());
    Ok(args)
}

pub(super) fn invocation(args: &[String]) -> (Options, Command) {
    let mut globals = Globals::default();
    let command =
        parse_command(args, &mut globals).unwrap_or_else(|detail| Command::Invalid(detail.into()));
    (globals.finish(), command)
}

#[cfg(test)]
pub(super) fn parse<I, S>(args: I) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    invocation(
        &args
            .into_iter()
            .map(|s| s.as_ref().to_owned())
            .collect::<Vec<_>>(),
    )
    .1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_globals_never_consume_prompt_or_provider_values() {
        let args = [
            "-v",
            "run",
            "model.gguf",
            "--prompt",
            "-q",
            "-v",
            "--color=never",
            "--progress",
            "never",
            "--format",
            "text",
        ]
        .map(str::to_owned);
        let (options, command) = invocation(&args);
        assert_eq!(options.verbosity, Verbosity::Diagnostic);
        assert_eq!(options.color, Switch::Never);
        assert_eq!(options.progress, Switch::Never);
        assert!(options.plain);
        assert!(
            matches!(command, Command::Run { args, .. } if args == ["model.gguf", "--prompt", "-q"])
        );
        let args = ["model", "pull", "hf", "repo", "--file", "--quiet"].map(str::to_owned);
        let (options, command) = invocation(&args);
        assert_eq!(options.verbosity, Verbosity::Normal);
        assert!(
            matches!(command, Command::Model(ModelCommand::Pull(args)) if args == ["hf", "repo", "--file", "--quiet"])
        );
    }

    #[test]
    fn restrictions_precede_owner_values() {
        assert_eq!(
            parse(["--offline", "pull", "hf", "org/repo", "--file"]),
            Command::Pull(vec![
                "hf".into(),
                "org/repo".into(),
                "--offline".into(),
                "--file".into()
            ])
        );
        assert_eq!(
            parse(["model", "pull", "--offline", "hf:org/repo@main", "--file"]),
            Command::Model(ModelCommand::Pull(vec![
                "hf:org/repo@main".into(),
                "--offline".into(),
                "--file".into()
            ]))
        );
        assert_eq!(
            parse(["--profile", "laptop", "run", "--prompt"]),
            Command::Run {
                args: vec!["--profile".into(), "laptop".into(), "--prompt".into()],
                alias: None,
                format: Format::Auto
            }
        );
    }

    #[test]
    fn global_format_and_config_selection_are_typed() {
        assert_eq!(
            parse(["device", "list", "--format", "json"]),
            Command::Devices(vec!["list".into(), "--json".into()])
        );
        assert_eq!(
            parse(["--profile", "laptop", "run"]),
            Command::Run {
                args: vec!["--profile".into(), "laptop".into()],
                alias: None,
                format: Format::Auto
            }
        );
        assert!(
            matches!(parse(["--color", "SECRET", "pull"]), Command::Invalid(detail) if !detail.contains("SECRET"))
        );
        assert!(matches!(parse(["-q", "run", "-v"]), Command::Invalid(_)));
        assert!(matches!(
            parse(["--config", "a", "--profile", "b", "run"]),
            Command::Invalid(_)
        ));
        assert!(matches!(
            parse(["--format", "jsonl", "run"]),
            Command::Invalid(_)
        ));
    }
}
