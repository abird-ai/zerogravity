//! Explicit managed E2B candidates; original tokenizer/session/private feedback.
use std::{
    io::{self, Read},
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity::cache::{CacheOptions, CacheStore};
#[cfg(feature = "gemma4-hip-resident-candidate")]
use zerogravity::diagnostics::gemma_text::HIP_PROFILE;
use zerogravity::diagnostics::gemma_text::{self, Backend, CPU_PROFILE, Report};
use zerogravity::gemma4::execution::MAX_OUTPUT;
use zerogravity::{Error, Result};
struct Options {
    snapshot: String,
    prompt: Option<String>,
    max_output: usize,
    resident_hip: bool,
}
/// Dispatch only an option, never the same bytes in a legacy option's value.
pub(super) fn selected(args: &[String]) -> bool {
    let mut rest = args.iter().skip(1); // Snapshot/ref is a positional value.
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--source-profile" => return true,
            "--raw" => {}
            "--prompt"
            | "-p"
            | "--max-tokens"
            | "-n"
            | "--audit-logits-jsonl"
            | "--audit-layers-jsonl"
            | "--teacher-ids"
            | "--prefill-chunk"
            | "--cache"
            | "--model-file"
            | "--config-file"
            | "--tokenizer-file"
            | "--config-snapshot"
            | "--tokenizer-snapshot"
            | "--disk-limit-bytes"
            | "--source-memory-limit-bytes" => {
                if rest.next().is_none() {
                    return false;
                }
            }
            // Leave malformed legacy syntax to its original parser.
            _ => return false,
        }
    }
    false
}
fn options(args: &[String]) -> Result<Options> {
    let invalid = || {
        Error::Malformed("Gemma run-cache requires SNAPSHOT --source-profile gemma4-e2b-bf16-cpu-candidate [--prompt TEXT] [--max-tokens 1..8]".into())
    };
    let snapshot = args.first().ok_or_else(invalid)?.clone();
    if snapshot.is_empty() || snapshot.len() > 4096 || snapshot.chars().any(char::is_control) {
        return Err(invalid());
    }
    let (mut profile, mut prompt, mut max) = (None, None, None);
    let mut rest = args[1..].iter();
    while let Some(flag) = rest.next() {
        let value = rest.next().ok_or_else(invalid)?;
        match flag.as_str() {
            "--source-profile" if profile.is_none() => profile = Some(value.as_str()),
            "--prompt" if prompt.is_none() => prompt = Some(value.clone()),
            "--max-tokens" if max.is_none() => {
                max = Some(value.parse::<usize>().map_err(|_| invalid())?)
            }
            _ => return Err(invalid()),
        }
    }
    let max_output = max.unwrap_or(8);
    #[cfg(feature = "gemma4-hip-resident-candidate")]
    let resident_hip = profile == Some(HIP_PROFILE);
    #[cfg(not(feature = "gemma4-hip-resident-candidate"))]
    let resident_hip = false;
    if (profile != Some(CPU_PROFILE) && !resident_hip)
        || !(1..=MAX_OUTPUT).contains(&max_output)
        || prompt.as_ref().is_some_and(|p| p.len() > 4096)
    {
        return Err(invalid());
    }
    Ok(Options {
        snapshot,
        prompt,
        max_output,
        resident_hip,
    })
}
struct Signals(Vec<signal_hook::SigId>);
impl Drop for Signals {
    fn drop(&mut self) {
        for id in self.0.drain(..) {
            signal_hook::low_level::unregister(id);
        }
    }
}
pub(super) fn run(args: &[String]) -> Result<()> {
    let options = options(args)?;
    let flag = Arc::new(AtomicBool::new(false));
    let mut signals = Signals(Vec::new());
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signals
            .0
            .push(signal_hook::flag::register(signal, flag.clone())?);
    }
    let prompt = match options.prompt {
        Some(text) => text,
        None => {
            let mut text = String::new();
            io::stdin().take(4097).read_to_string(&mut text)?;
            if text.len() > 4096 {
                return Err(Error::Limit(
                    "Gemma plain stdin exceeds4096UTF8bytes".into(),
                ));
            }
            text
        }
    };
    let backend = if options.resident_hip {
        #[cfg(feature = "gemma4-hip-resident-candidate")]
        {
            Backend::HipResident
        }
        #[cfg(not(feature = "gemma4-hip-resident-candidate"))]
        {
            return Err(Error::Unsupported(
                "Gemma HIP resident candidate feature is disabled".into(),
            ));
        }
    } else {
        Backend::Cpu
    };
    // Existing approved canonical XDG cache policy; never creates/raises policy.
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * 1024 * 1024 * 1024))?;
    gemma_text::generate(
        &store,
        &options.snapshot,
        &prompt,
        options.max_output,
        backend,
        &flag,
        |event| {
            match event {
                Report::Text(text) => println!("{text}"),
                Report::Finished {
                    backend: Backend::Cpu,
                    emitted,
                    calls,
                    frontier,
                    first_root: Some(root),
                } => eprintln!(
                    "Gemma CPU candidate: {emitted} emitted, {calls} calls, frontier{frontier}, first root{root}; trained/chat/native/HTTP unqualified"
                ),
                #[cfg(feature = "gemma4-hip-resident-candidate")]
                Report::Finished {
                    backend: Backend::HipResident,
                    emitted,
                    calls,
                    frontier,
                    ..
                } => eprintln!(
                    "Gemma HIP resident candidate: {emitted} emitted, {calls} calls, frontier{frontier}; observer=none private_feedback=true trained/HTTP/performance unqualified"
                ),
                _ => {
                    return Err(Error::State(
                        "CPU diagnostic omitted its actual first root".into(),
                    ));
                }
            }
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(tail: &[&str]) -> Vec<String> {
        ["approved"]
            .into_iter()
            .chain(tail.iter().copied())
            .map(str::to_string)
            .collect()
    }
    #[test]
    fn explicit_profile_plain_only_and_real_limits() {
        assert!(
            options(&args(&[
                "--source-profile",
                CPU_PROFILE,
                "--prompt",
                "你好",
                "--max-tokens",
                "8"
            ]))
            .is_ok()
        );
        for tail in [
            vec![],
            vec!["--source-profile", "qwen"],
            vec!["--source-profile", CPU_PROFILE, "--max-tokens", "9"],
            vec!["--source-profile", CPU_PROFILE, "--teacher-ids", "1"],
            vec!["--source-profile", CPU_PROFILE, "--chat", "yes"],
        ] {
            assert!(options(&args(&tail)).is_err());
        }
    }

    #[test]
    fn legacy_literal_profile_text_stays_a_value_for_dispatch() {
        let legacy = args(&[
            "--model-file",
            "model.gguf",
            "--raw",
            "--prompt",
            "--source-profile",
            "--max-tokens",
            "8",
        ]);
        assert!(!selected(&legacy));
        for flag in [
            "--prompt",
            "-p",
            "--model-file",
            "--config-file",
            "--tokenizer-file",
            "--cache",
            "--config-snapshot",
            "--tokenizer-snapshot",
            "--audit-logits-jsonl",
            "--audit-layers-jsonl",
            "--teacher-ids",
            "--prefill-chunk",
            "--max-tokens",
            "-n",
            "--disk-limit-bytes",
            "--source-memory-limit-bytes",
        ] {
            assert!(!selected(&args(&[flag, "--source-profile"])), "{flag}");
        }
        assert!(!selected(&["--source-profile".into()]));
        assert!(!selected(&args(&["--unknown", "--source-profile"])));
    }

    #[test]
    fn real_profile_option_is_selected_after_a_literal_flag_value() {
        let actual = args(&[
            "--prompt",
            "--source-profile",
            "--source-profile",
            CPU_PROFILE,
            "--max-tokens",
            "1",
        ]);
        assert!(selected(&actual));
        assert_eq!(
            options(&actual).unwrap().prompt.as_deref(),
            Some("--source-profile")
        );
        assert!(selected(&args(&["--source-profile"])));
        assert!(options(&args(&["--source-profile"])).is_err());
    }
}
