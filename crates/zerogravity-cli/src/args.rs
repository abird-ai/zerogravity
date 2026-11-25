//! CLI argument syntax; no source/model execution.

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Command {
    Help,
    Version,
    Inspect(String),
    Run {
        args: Vec<String>,
        alias: Option<String>,
    },
    Devices(Vec<String>),
    Pull(Vec<String>),
    Models(Vec<String>),
    Pipeline(Vec<String>),
    Serve(Vec<String>),
    CacheRun(Vec<String>),
    CacheManage(Vec<String>),
    NotImplemented(String),
    Invalid(String),
    Unknown(String),
}

pub(super) fn parse<I, S>(args: I) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    match args.next() {
        None => Command::Help,
        Some(first) => match first.as_ref() {
            "-h" | "--help" | "help" => Command::Help,
            "-V" | "--version" | "version" => Command::Version,
            "inspect" => match (args.next(), args.next()) {
                (Some(path), None) if !path.as_ref().is_empty() => {
                    Command::Inspect(path.as_ref().to_owned())
                }
                _ => Command::Invalid("inspect requires exactly one local path".into()),
            },
            "run" | "run-cpu" | "run-cuda" | "run-rocm" | "run-vulkan" | "run-vulkan-amd"
            | "run-vulkan-nvidia" => Command::Run {
                alias: (first.as_ref() != "run").then(|| first.as_ref().to_owned()),
                args: args.map(|arg| arg.as_ref().to_owned()).collect(),
            },
            "devices" => Command::Devices(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "pull" => Command::Pull(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "models" => Command::Models(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "pipeline" => Command::Pipeline(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "cache" => Command::CacheManage(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "run-cache" => Command::CacheRun(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "serve" => Command::Serve(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "config" => Command::NotImplemented(first.as_ref().to_owned()),
            other => Command::Unknown(other.to_owned()),
        },
    }
}

/// Consume presentation options only before the command. A subcommand owns
/// every subsequent token, including literal prompts/paths and credentials
/// mistakenly supplied as values. UX-2 will unify the command grammar.
pub(super) fn presentation(
    args: &[String],
) -> Result<(crate::output::Options, &[String]), &'static str> {
    use crate::output::{Options, Switch, Verbosity};
    let mut options = Options::default();
    let mut index = 0;
    while let Some(raw) = args.get(index) {
        let (flag, inline) = raw
            .split_once('=')
            .map_or((raw.as_str(), None), |(a, b)| (a, Some(b)));
        match flag {
            "-q" | "--quiet" if inline.is_none() => {
                if matches!(
                    options.verbosity,
                    Verbosity::Verbose | Verbosity::Diagnostic
                ) {
                    return Err("quiet and verbose are mutually exclusive");
                }
                options.verbosity = Verbosity::Quiet;
            }
            "-v" | "--verbose" | "-vv" if inline.is_none() => {
                options.verbosity = match (options.verbosity, flag) {
                    (Verbosity::Quiet, _) => {
                        return Err("quiet and verbose are mutually exclusive");
                    }
                    (Verbosity::Normal, "-v" | "--verbose") => Verbosity::Verbose,
                    _ => Verbosity::Diagnostic,
                };
            }
            "--format" | "--color" | "--progress" => {
                let value = if let Some(value) = inline {
                    value
                } else {
                    index += 1;
                    args.get(index)
                        .map(String::as_str)
                        .ok_or("presentation option requires a value")?
                };
                match flag {
                    "--format" => {
                        options.plain = match value {
                            "auto" => false,
                            "text" => true,
                            _ => {
                                return Err(
                                    "global format supports auto or text; use command-specific JSON/JSONL options",
                                );
                            }
                        }
                    }
                    "--color" => {
                        options.color =
                            Switch::parse(value).ok_or("color requires auto, always or never")?
                    }
                    "--progress" => {
                        options.progress =
                            Switch::parse(value).ok_or("progress requires auto, always or never")?
                    }
                    _ => unreachable!(),
                }
            }
            "--" => {
                index += 1;
                break;
            }
            _ => break,
        }
        index += 1;
    }
    Ok((options, &args[index..]))
}

#[cfg(test)]
mod presentation_tests {
    use super::*;
    use crate::output::{Switch, Verbosity};

    #[test]
    fn only_leading_options_are_consumed() {
        let args = [
            "-v",
            "-v",
            "--color=never",
            "--progress",
            "never",
            "--format",
            "text",
            "run",
            "--prompt",
            "-q",
        ]
        .map(str::to_owned);
        let (options, command) = presentation(&args).unwrap();
        assert_eq!(options.verbosity, Verbosity::Diagnostic);
        assert_eq!(options.color, Switch::Never);
        assert_eq!(options.progress, Switch::Never);
        assert!(options.plain);
        assert_eq!(command, &args[7..]);
        assert!(matches!(parse(command), Command::Run { args, .. } if args == ["--prompt", "-q"]));
    }

    #[test]
    fn invalid_presentation_never_echoes_raw_value() {
        for args in [
            vec!["--color", "SECRET"],
            vec!["--format=SECRET"],
            vec!["-q", "-v"],
            vec!["--progress"],
        ] {
            let args = args.into_iter().map(str::to_owned).collect::<Vec<_>>();
            assert!(!presentation(&args).unwrap_err().contains("SECRET"));
        }
    }
}
