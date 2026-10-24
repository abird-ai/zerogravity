//! Explicit managed E2B candidates; original tokenizer/session/private feedback.
use std::{
    io::{self, Read},
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity::cache::{CacheOptions, CacheStore};
use zerogravity::gemma4_cpu::{
    Gemma4CpuContext, Gemma4CpuLoaded, MAX_CONTEXT, MAX_OUTPUT, PROFILE,
};
use zerogravity::{Error, Result};
#[cfg(feature = "gemma4-hip-resident-candidate")]
mod gpu;
#[cfg(feature = "gemma4-hip-resident-candidate")]
const HIP_PROFILE: &str = "gemma4-e2b-bf16-hip-resident-candidate";
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
    if (profile != Some(PROFILE) && !resident_hip)
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
fn context(resident_hip: bool) -> Result<Arc<Gemma4CpuContext>> {
    if resident_hip {
        #[cfg(feature = "gemma4-hip-resident-candidate")]
        return Gemma4CpuContext::new_hip_qualification();
        #[cfg(not(feature = "gemma4-hip-resident-candidate"))]
        return Err(Error::Unsupported(
            "Gemma HIP resident candidate feature is disabled".into(),
        ));
    }
    Gemma4CpuContext::new()
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
    let context = context(options.resident_hip)?;
    // Existing approved canonical XDG cache policy. Never creates/raises policy.
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * 1024 * 1024 * 1024))?;
    let snapshot = Arc::new(
        if options.snapshot.len() == 64
            && options
                .snapshot
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            store.open_snapshot_cancellable(&options.snapshot, &flag)?
        } else {
            store.resolve_ref_cancellable(&options.snapshot, &flag)?
        },
    );
    let input = context.prepare_text(&snapshot, &prompt, Some(&flag))?;
    if input.ids().len() + options.max_output - 1 > MAX_CONTEXT {
        return Err(Error::Limit(
            "Gemma prompt+natural outputs exceeds64consumed rows".into(),
        ));
    }
    #[cfg(feature = "gemma4-hip-resident-candidate")]
    if options.resident_hip {
        // Tokenizer owners retire before native preparation; decoder remains admitted.
        let decoder = context.prepare_text(&snapshot, "One", Some(&flag))?;
        let loaded = Gemma4CpuLoaded::open(snapshot.clone(), context.clone(), Some(&flag))?;
        let result = gpu::run(&loaded, input, &decoder, options.max_output, &flag);
        drop(decoder);
        let closed = loaded.close(None);
        drop(snapshot);
        drop(store);
        drop(context);
        result?;
        closed?;
        return Ok(());
    }
    let loaded = Gemma4CpuLoaded::open(snapshot.clone(), context.clone(), Some(&flag))?;
    let mut session = loaded.session(input)?;
    let result = (|| -> Result<()> {
        let first = session.infer_prompt(options.max_output, Some(&flag))?;
        let mut calls = 1;
        while !session.stopped() {
            session.infer_next(Some(&flag))?;
            calls += 1;
        }
        let text = session.decode_generated(true, Some(&flag))?;
        println!("{}", text.as_str());
        eprintln!(
            "Gemma CPU candidate: {} emitted, {} calls, frontier{}, first root{}; trained/chat/native/HTTP unqualified",
            session.generated_ids().len(),
            calls,
            session.binding().frontier(),
            first.root.root()
        );
        Ok(())
    })();
    drop(session);
    // Cooperative cancellation does not skip immutable-source final validation.
    let closed = loaded.close(None);
    drop(snapshot);
    drop(store);
    let reserved = context.reserved_bytes()?;
    drop(context);
    let final_bytes = zerogravity::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&zerogravity::resource_arbiter::PhysicalDeviceKey::new(
            "host",
            "system-ram",
        )?)?
        .reserved;
    result?;
    closed?;
    if reserved != 3 * 1024 * 1024 || final_bytes != 0 {
        return Err(Error::State(
            "Gemma ordinary source/session/decoder owners not retired".into(),
        ));
    }
    Ok(())
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
                PROFILE,
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
            vec!["--source-profile", PROFILE, "--max-tokens", "9"],
            vec!["--source-profile", PROFILE, "--teacher-ids", "1"],
            vec!["--source-profile", PROFILE, "--chat", "yes"],
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
            PROFILE,
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
