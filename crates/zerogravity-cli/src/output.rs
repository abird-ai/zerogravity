//! CLI presentation.
mod presentation;
pub(crate) use presentation::{
    Event, Failure, Options, Switch, Verbosity, emit, fail, install, report,
};

use super::VERSION;
use crate::args::{TREE, Topic, node};
pub(crate) use presentation::reporter;

pub(super) fn usage() {
    println!(
        "zerogravity v{VERSION} — AI execution fabric\n\nUSAGE:\n    zerogravity [global options] <command> [options]\n\nCOMMANDS:"
    );
    for n in TREE {
        let description = match n.topic {
            Topic::Serve if !cfg!(feature = "serve") => "Local API (requires serve feature)",
            _ => n.description,
        };
        println!("    {:12} {description}", n.name);
    }
    println!(
        "\nGLOBAL OPTIONS:\n    -q, --quiet | -v, --verbose | -vv\n    --format auto|text|json|jsonl  (structured formats where supported)\n    --color auto|always|never --progress auto|always|never\n    --interactive auto|never|always (run, serve and online hub/Xet pull)\n    --config FILE | --profile NAME (where supported)\n    --offline  --yes (model remove only)\n\nRun 'zerogravity help <command>' for details. No implicit acquisition."
    );
}

pub(super) fn topic_usage(topic: Topic, child: Option<&str>) -> zerogravity::Result<()> {
    match topic {
        Topic::Run => crate::run_cli::help(),
        Topic::Serve => {
            #[cfg(feature = "serve")]
            crate::serve::help();
            #[cfg(not(feature = "serve"))]
            println!(
                "USAGE: zerogravity serve MODEL... | --model ID=MODEL | --config FILE | --profile NAME\nRequires the optional serve feature."
            );
        }
        Topic::Model => match child {
            Some("pull") => {
                println!(
                    "USAGE: zerogravity model pull hf:REPO@REV --file PATH [--file PATH ...]\n       zerogravity model pull gh:OWNER/REPO@REV --file PATH\nAdvanced provider forms retain the existing pull grammar. Acquisition is explicit.\nOnline hub/Xet transfers: --format jsonl, --interactive auto|always|never.\nSpace requests pause/resume; c/Ctrl-C requests cancellation."
                );
                #[cfg(all(feature = "acquisition", target_os = "linux"))]
                crate::pull::help();
                #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
                println!("Requires the optional acquisition feature on Linux.");
            }
            Some("list") => println!(
                "USAGE: zerogravity model list [--partial] [--cache ABS_PATH] [--format json]\nRead-only manifest metadata or recorded checkpoints; no payload or execution qualification."
            ),
            Some("inspect") | Some("verify") => {
                let operation = child.expect("known operation");
                println!(
                    "USAGE: zerogravity model {operation} REF [--snapshot DIGEST] [--cache ABS_PATH] [--format json]\nREF: explicit local file (inspect only), canonical reference, sha256:DIGEST, full digest, by-name:NAME, or acquired hf:/gh:/http(s): locator. No remote lookup."
                );
            }
            Some("remove") => println!(
                "USAGE: zerogravity model remove REF [--cache ABS_PATH] --dry-run | --yes\nExact canonical snapshot reclamation; referenced or busy data refuses."
            ),
            Some("search") => println!(
                "USAGE: zerogravity model search [QUERY]\nOffline family intent catalog, not a model readiness or remote search claim."
            ),
            Some("import") => println!(
                "USAGE: zerogravity model import --source PATH --repository REPO --commit SHA --file ROLE:PATH ...\nExisting cache import-hf grammar; see cache import-hf --help."
            ),
            Some("alias") => println!(
                "USAGE: zerogravity model alias list|set|remove [arguments]\nExisting cache by-name grammar; see cache --help."
            ),
            _ => {
                println!("USAGE: zerogravity model <command> [options]\n\nCOMMANDS:");
                for child in node(Topic::Model).children {
                    println!("    {child}");
                }
                println!(
                    "\nRun 'zerogravity help model <command>' for details. Cached source operations require Linux and the cache feature."
                );
            }
        },
        Topic::Device => println!(
            "USAGE: zerogravity device list [--format json]\n       zerogravity device inspect ID [--format json]\nObserved hardware only; use exact PCI, gpu:N, cpu:0 or supported device-node selectors."
        ),
        Topic::Pipeline => crate::pipeline::run(&["--help".into()])?,
        Topic::Cache => {
            #[cfg(all(feature = "cache", target_os = "linux"))]
            crate::cache_manage::help();
            #[cfg(not(all(feature = "cache", target_os = "linux")))]
            println!("Advanced cache maintenance requires Linux and the cache feature.");
        }
        Topic::Completion => println!("USAGE: zerogravity completion bash|zsh|fish|powershell"),
        Topic::Profile => println!(
            "USAGE: zerogravity --profile NAME run [options]\nNamed profile selection uses the existing run YAML schema; profile management is not implemented yet."
        ),
        Topic::Config => println!(
            "USAGE: zerogravity --config FILE run [options]\nConfiguration editing and unified serving documents are not implemented yet."
        ),
        Topic::Doctor | Topic::Web => println!("{} is not implemented yet.", node(topic).name),
        Topic::Version => println!("USAGE: zerogravity version"),
        Topic::Help => usage(),
    }
    Ok(())
}

pub(super) fn json(value: &impl serde::Serialize) -> zerogravity::Result<()> {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut out, value)
        .map_err(|error| zerogravity::Error::State(error.to_string()))?;
    writeln!(out)?;
    Ok(())
}

#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
pub(crate) struct CpuAuditFile {
    path: String,
    writer: Option<std::io::BufWriter<std::fs::File>>,
}
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
impl CpuAuditFile {
    pub(crate) fn new(path: String) -> Self {
        Self { path, writer: None }
    }
}
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
impl zerogravity::diagnostics::cpu_text::AuditSink for CpuAuditFile {
    fn open(&mut self) -> zerogravity::Result<&mut dyn std::io::Write> {
        if self.writer.is_none() {
            self.writer = Some(std::io::BufWriter::new(std::fs::File::create(&self.path)?));
        }
        Ok(self.writer.as_mut().expect("opened audit file"))
    }
}
