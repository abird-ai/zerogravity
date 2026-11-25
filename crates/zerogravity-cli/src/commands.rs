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
use zerogravity::{Error, Result};

fn finish(command: &'static str, result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            fail(Failure::core(command, &error));
            ExitCode::from(2)
        }
    }
}

pub(super) fn run(command: Command) -> ExitCode {
    match command {
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
        Command::DeviceInspect { selector, format } => {
            finish("device inspect", run_cli::device_inspect(&selector, format))
        }
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
                Err(Error::Unsupported(
                    "pull requires the optional acquisition feature on Linux".into(),
                ))
            };
            finish("pull", result)
        }
        Command::Pipeline(args) => finish("pipeline", pipeline::run(&args)),
        Command::Serve(args) => {
            #[cfg(feature = "serve")]
            {
                match serve::run(&args) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(error) => {
                        // Preserve the original service code/status, not a lossy
                        // translation back into an engine error category.
                        fail(Failure::new("serve", error.code, &error, None));
                        ExitCode::from(2)
                    }
                }
            }
            #[cfg(not(feature = "serve"))]
            {
                let _ = args;
                finish(
                    "serve",
                    Err(Error::Unsupported(
                        "native HTTP serving requires --features serve".into(),
                    )),
                )
            }
        }
        Command::Models(args) => finish("models", models(&args)),
        Command::NotImplemented(name) => finish(
            "",
            Err(Error::Unsupported(format!(
                "'{name}' is not implemented (use explicit run/serve/pipeline configuration)"
            ))),
        ),
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
