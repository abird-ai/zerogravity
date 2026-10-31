//! Command dispatch; existing execution owners migrate in F0–F3.
use super::output::usage;
#[cfg(all(feature = "acquisition", target_os = "linux"))]
use super::pull;
#[cfg(feature = "serve")]
use super::serve;
use super::{
    Command, VERSION, execute_cache, execute_cache_manage, execute_cpu, execute_gpu, inspect,
    models, pipeline,
};
use std::{path::Path, process::ExitCode};
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
use zerogravity::Result;

pub(super) fn run(command: Command) -> ExitCode {
    match command {
        Command::Help => {
            usage();
            ExitCode::SUCCESS
        }
        Command::Version => {
            println!("zerogravity v{VERSION}");
            ExitCode::SUCCESS
        }
        Command::CacheManage(args) => match execute_cache_manage(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("zerogravity cache: {error}");
                ExitCode::from(2)
            }
        },
        Command::CacheRun(args) => match execute_cache(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("zerogravity: {error}");
                ExitCode::from(2)
            }
        },
        command @ Command::Run { .. } => match execute_cpu(command) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("zerogravity run: {e}");
                ExitCode::from(2)
            }
        },
        Command::GpuRun { backend, args } => match execute_gpu(backend, args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("zerogravity {}: {e}", backend.command());
                ExitCode::from(2)
            }
        },
        Command::Inspect(path) => match inspect(Path::new(&path)) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("zerogravity inspect: {e}");
                ExitCode::from(2)
            }
        },
        Command::Pull(args) => {
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            let result = pull::run(&args);
            #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
            let result: Result<()> = {
                let _ = args;
                Err(zerogravity::Error::Unsupported(
                    "pull requires the optional acquisition feature on Linux".into(),
                ))
            };
            match result {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("zerogravity pull: {error}");
                    ExitCode::from(2)
                }
            }
        }
        Command::Pipeline(args) => match pipeline::run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("zerogravity pipeline: {error}");
                ExitCode::from(2)
            }
        },
        Command::Serve(args) => {
            #[cfg(feature = "serve")]
            let result = serve::run(&args);
            #[cfg(not(feature = "serve"))]
            let result: std::result::Result<(), String> = {
                let _ = args;
                Err("native HTTP serving requires --features serve".into())
            };
            match result {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("zerogravity serve: {error:?}");
                    ExitCode::from(2)
                }
            }
        }
        Command::Models(args) => match models(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("zerogravity models: {error}");
                ExitCode::from(2)
            }
        },
        Command::NotImplemented(name) => {
            eprintln!(
                "zerogravity: '{name}' is not implemented (local Qwen3.5 CPU run/inspect exist; router, API and model management are future work)"
            );
            ExitCode::from(2)
        }
        Command::Invalid(detail) => {
            eprintln!("zerogravity: {detail}");
            ExitCode::from(2)
        }
        Command::Unknown(name) => {
            eprintln!("zerogravity: unknown command '{name}'. Try --help.");
            ExitCode::from(2)
        }
    }
}
