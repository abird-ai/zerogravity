//! Process command dispatch over the public SDK.
use super::output::{Failure, fail, topic_usage, usage};
#[cfg(all(feature = "acquisition", target_os = "linux"))]
use super::pull;
#[cfg(feature = "serve")]
use super::serve;
use super::{
    Command, VERSION, execute_cache, execute_cache_manage, inspect, models, pipeline, run_cli,
};
use crate::args::ModelCommand;
use std::{path::Path, process::ExitCode};
use zerogravity::Result;

fn finish(command: &'static str, result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            fail(Failure::core(command, &error));
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            if matches!(command, "pull" | "model pull" | "run")
                && zerogravity::control::DownloadStatus::from_error(&error)
                    == zerogravity::control::DownloadStatus::Cancelled
            {
                return ExitCode::from(130);
            }
            ExitCode::from(2)
        }
    }
}

pub(super) fn run(command: Command) -> ExitCode {
    let (command, selection) = match command {
        Command::WithDocument { command, selection } => (*command, Some(selection)),
        command => (command, None),
    };
    if !matches!(
        &command,
        Command::Home { .. } | Command::Web { .. } | Command::Serve(_)
    ) {
        super::output::operational::start(super::output::operational::Mode::Command);
    }
    if matches!(
        command,
        Command::Model(_) | Command::Devices(_) | Command::DeviceInspect { model: None, .. }
    ) && let Err(error) = crate::config_cli::apply_presentation(
        selection
            .as_ref()
            .map(|(flag, value)| (*flag, value.as_str())),
    ) {
        return finish("config", Err(error));
    }
    match command {
        Command::Home { selection, offline } => {
            #[cfg(feature = "tui")]
            return finish(
                "tui",
                crate::tui::launch(
                    selection
                        .as_ref()
                        .map(|(flag, value)| (*flag, value.as_str())),
                    offline,
                ),
            );
            #[cfg(not(feature = "tui"))]
            {
                let _ = (selection, offline);
                usage();
                ExitCode::SUCCESS
            }
        }
        Command::Help => {
            usage();
            ExitCode::SUCCESS
        }
        Command::HelpTopic(topic, child) => finish("help", topic_usage(topic, child.as_deref())),
        Command::Completion(shell) => {
            crate::completion::write(shell);
            ExitCode::SUCCESS
        }
        Command::Version => {
            println!("zerogravity v{VERSION}");
            ExitCode::SUCCESS
        }
        Command::CacheManage(args) => finish("cache", execute_cache_manage(args)),
        Command::CacheRun(args) => finish("run-cache", execute_cache(args)),
        Command::Run {
            args,
            alias,
            format,
        } => finish("run", run_cli::run(args, alias.as_deref(), format)),
        Command::Devices(args) => finish("device", run_cli::devices(&args)),
        Command::DeviceInspect {
            selector,
            model,
            format,
        } => finish(
            "device inspect",
            run_cli::device_inspect(&selector, model, selection, format),
        ),
        Command::Model(command) => {
            let (label, result) = match command {
                ModelCommand::Read {
                    operation,
                    args,
                    format,
                    yes,
                } => (
                    "model",
                    crate::model_cli::read(operation, &args, format, yes),
                ),
                ModelCommand::Search(args) => ("model search", crate::model_cli::search(&args)),
                ModelCommand::Discovery {
                    operation,
                    args,
                    offline,
                    format,
                } => (
                    "model discovery",
                    crate::model_cli::discovery(
                        &operation,
                        &args,
                        selection
                            .as_ref()
                            .map(|(flag, value)| (*flag, value.as_str())),
                        offline,
                        format,
                    ),
                ),
                ModelCommand::Pull(args) => ("model pull", crate::model_cli::pull(&args)),
                ModelCommand::Import(mut args) => {
                    args.insert(0, "import-hf".into());
                    ("model import", execute_cache_manage(args))
                }
                ModelCommand::Alias(mut args) => {
                    args.insert(0, "by-name".into());
                    ("model alias", execute_cache_manage(args))
                }
            };
            finish(label, result)
        }
        Command::Inspect(path) => finish("inspect", inspect(Path::new(&path))),
        Command::Pull(args) => {
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            let result = pull::run(&args);
            #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
            let result = {
                let _ = args;
                Err(zerogravity::Error::Unsupported(
                    "pull requires the optional acquisition feature on Linux".into(),
                ))
            };
            finish("pull", result)
        }
        Command::Pipeline {
            args,
            document,
            format,
        } => finish("pipeline", pipeline::run(&args, document, format)),
        Command::Doctor { document, format } => {
            finish("doctor", crate::doctor::run(document, format))
        }
        Command::Config {
            args,
            document,
            format,
        } => finish("config", crate::config_cli::config(&args, document, format)),
        Command::Profile { args, format, yes } => {
            finish("profile", crate::config_cli::profile(&args, format, yes))
        }
        Command::Web {
            args,
            selection,
            offline,
        } => {
            #[cfg(feature = "web")]
            let result = crate::web::run(
                &args,
                selection
                    .as_ref()
                    .map(|(flag, value)| (*flag, value.as_str())),
                offline,
            );
            #[cfg(not(feature = "web"))]
            let result = {
                let _ = (args, selection, offline);
                Err(zerogravity::Error::Unsupported(
                    "embedded control UI requires --features web".into(),
                ))
            };
            finish("web", result)
        }
        Command::Serve(args) => {
            #[cfg(feature = "serve")]
            {
                match serve::run(&args) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(error) => {
                        // Preserve the original service code/status, not a lossy
                        // translation back into an engine error category.
                        fail(Failure::service("serve", &error));
                        ExitCode::from(2)
                    }
                }
            }
            #[cfg(not(feature = "serve"))]
            {
                let _ = args;
                finish(
                    "serve",
                    Err(zerogravity::Error::Unsupported(
                        "native HTTP serving requires --features serve".into(),
                    )),
                )
            }
        }
        Command::Models(args) => finish("models", models(&args)),
        Command::WithDocument { .. } => unreachable!("document selection already resolved"),
        Command::Invalid(detail) => {
            fail(Failure::new(
                "",
                "invalid_usage",
                &detail,
                Some("Run zerogravity --help."),
            ));
            ExitCode::from(2)
        }
        // Do not echo arbitrary argv (e.g. a misplaced acquisition token).
        Command::Unknown => {
            fail(Failure::new(
                "",
                "unknown_command",
                &"unknown command",
                Some("Run zerogravity --help."),
            ));
            ExitCode::from(2)
        }
    }
}
