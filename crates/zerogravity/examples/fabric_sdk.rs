//! Only the public zerogravity SDK + std. Default invocation is cold.
//! `pull REPO REVISION FILE` explicitly authorizes network/cache IO; other modes
//! never acquire sources. `run MODEL.gguf PROMPT` requires admitted local inputs.
#[cfg(target_os = "linux")]
fn main() -> zerogravity::Result<()> {
    application::run()
}
#[cfg(not(target_os = "linux"))]
fn main() -> zerogravity::Result<()> {
    Err(zerogravity::Error::Unsupported(
        "this cache/acquisition SDK example requires Linux".into(),
    ))
}
#[cfg(target_os = "linux")]
mod application {
    use std::{
        num::NonZeroUsize,
        path::Path,
        sync::{Arc, atomic::AtomicBool},
    };
    use zerogravity::run_policy::{Backend, Entrypoint, Origin, Request, Speculation};
    use zerogravity::{Error, Result, Runtime};

    fn direct_request(source: &str) -> Result<Request> {
        zerogravity::run::config::resolve(
            None,
            zerogravity::run::config::RunOverrides {
                model: Some(source.into()),
                entrypoint: Some(Entrypoint::Model("local".into())),
                backend: Some(Backend::Cpu),
                numerical: Some("source".into()),
                speculation: Some(Speculation::Off),
                diagnostics: zerogravity::pipelines::DiagnosticDoc {
                    dry_run: Some(true),
                    ..Default::default()
                },
                ..Default::default()
            },
            &std::env::current_dir()?,
            Origin::Caller,
        )
    }
    pub fn run() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let runtime = Runtime::builder().build()?;
        let result = (|| -> Result<()> {
            match args
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .as_slice()
            {
                [] | ["--help"] => println!(
                    "fabric_sdk inspect PATH | dry-run MODEL.gguf | run MODEL.gguf PROMPT | pull REPO REVISION FILE"
                ),
                ["inspect", path] => {
                    let info = runtime.artifacts().inspect(Path::new(path))?;
                    println!(
                        "{} bytes, {} source tensors; metadata is not execution eligibility",
                        info.file_len,
                        info.tensors.len()
                    );
                }
                ["dry-run", source] => {
                    let plan = zerogravity::run::cpu::Plan::new(
                        direct_request(source)?,
                        zerogravity::devices::discover()?,
                    )?;
                    println!("{}", plan.decision());
                    plan.require_proposal()?;
                }
                ["run", model, prompt] => {
                    runtime
                        .models()
                        .load(zerogravity::models::ModelLoadRequest::CpuText {
                            declared: vec![("local".into(), (*model).into())],
                            config: zerogravity::runtime::TextConfig {
                                max_context: 512,
                                max_output: 8,
                                ..Default::default()
                            },
                        })
                        .map_err(|error| Error::State(error.to_string()))?;
                    let mut stream = runtime.run(&zerogravity::run::TaskRequest {
                        model: "local".into(),
                        input: zerogravity::run::TaskInput::Text((*prompt).into()),
                        max_output: NonZeroUsize::new(8).expect("nonzero literal"),
                    })?;
                    while let Some(event) = stream.next()? {
                        use zerogravity::runtime::text::ResponseEvent;
                        match event {
                            ResponseEvent::Text(text) | ResponseEvent::TokenText { text, .. } => {
                                print!("{text}")
                            }
                            ResponseEvent::Done { .. } => println!(),
                            ResponseEvent::Error(error) => {
                                return Err(Error::State(error.to_string()));
                            }
                        }
                    }
                }
                ["pull", repository, revision, file] => {
                    use zerogravity::{
                        acquisition::{
                            Provider, PullLimits, PullRequest, Selection,
                            http::{HttpOptions, HttpsClient},
                        },
                        artifacts::{HubSelection, PullMode},
                        cache::{CacheOptions, CacheStore, CommitOutcome},
                    };
                    let request = PullRequest {
                        provider: Provider::HuggingFace,
                        repository: (*repository).into(),
                        revision: (*revision).into(),
                        selection: Selection::Files {
                            paths: vec![(*file).into()],
                            subfolder: None,
                        },
                    };
                    let hub = runtime.artifacts().hub(
                        HubSelection::Files(&request),
                        PullLimits::default(),
                        None,
                        None,
                    )?;
                    // The application chooses the credential variable, never argv secrets.
                    let client = HttpsClient::from_env(
                        "https://huggingface.co",
                        "fabric-example",
                        Some("HF_TOKEN"),
                        HttpOptions::default(),
                    )?;
                    let admission = zerogravity::artifacts::transfer_admission(64 * 1024 * 1024)?;
                    let cancel = Arc::new(AtomicBool::new(false));
                    let prepared = hub.resolve(&client, &admission, &cancel, PullMode::Transfer)?;
                    let store = CacheStore::open(CacheOptions::new(None, 16 * 1024 * 1024 * 1024))?;
                    let receipt =
                        prepared.execute(&store, &client, &admission, &cancel, &mut |_| {})?;
                    match receipt.outcome() {
                        CommitOutcome::Durable(snapshot) => {
                            println!("{}: verified data, not model readiness", snapshot.digest())
                        }
                        CommitOutcome::DurabilityUnknown { .. } => {
                            return Err(Error::State(
                                "visible cache binding; durability unknown, inspect before retry"
                                    .into(),
                            ));
                        }
                    }
                }
                _ => return Err(Error::Malformed("use fabric_sdk --help".into())),
            }
            Ok(())
        })();
        let closed = runtime.shutdown();
        match (result, closed) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(error), Err(close)) => Err(Error::State(format!(
                "operation: {error}; shutdown: {close}"
            ))),
        }
    }
}
