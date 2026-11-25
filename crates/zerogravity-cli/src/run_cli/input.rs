//! Process input only. No source, model or backend ownership.
use super::*;
use std::io::{self, IsTerminal, Read};
#[cfg(any(
    all(feature = "cpu", feature = "model-qwen35", feature = "text"),
    all(
        feature = "model-qwen35",
        feature = "cache",
        feature = "text",
        target_os = "linux",
        any(feature = "cuda", feature = "hip")
    )
))]
use std::io::{BufRead, Write};

const MAX_PROMPT: u64 = 1024 * 1024;

fn read_prompt(reader: impl Read) -> Result<String> {
    let mut text = String::new();
    reader.take(MAX_PROMPT + 1).read_to_string(&mut text)?;
    if text.len() as u64 > MAX_PROMPT {
        return Err(Error::Limit("prompt exceeds 1 MiB".into()));
    }
    Ok(text)
}

pub(super) fn prompt_file(path: &Path) -> Result<String> {
    if !std::fs::metadata(path)?.is_file() {
        return Err(bad("prompt file must be regular"));
    }
    let file = std::fs::File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(bad("prompt file must be regular"));
    }
    read_prompt(file)
}

pub(crate) fn interaction() -> Result<bool> {
    crate::output::reporter().run_interaction()
}

/// Configured and explicitly supplied prompts take precedence over stdin.
/// Dry-run never reads process input; a TTY is not an implicit EOF reader.
pub(super) fn prepare(request: &mut Request, interactive: bool) -> Result<bool> {
    if request.diagnostics.dry_run || request.task.prompt.is_some() {
        return Ok(false);
    }
    if io::stdin().is_terminal() {
        if !interactive {
            return Err(bad(
                "provide a prompt or --prompt-file when interactive run is disabled",
            ));
        }
        if request.task.teacher_ids.is_some()
            || request.diagnostics.audit_logits_jsonl.is_some()
            || request.diagnostics.audit_layers_jsonl.is_some()
        {
            return Err(Error::Unsupported("interactive run does not repeat teacher/audit diagnostics; provide one explicit prompt".into()));
        }
        return Ok(true);
    }
    request.task.prompt = Some(read_prompt(io::stdin().lock())?);
    request.provenance.insert("task.prompt".into(), Origin::Cli);
    Ok(false)
}

#[cfg(all(
    feature = "acquisition",
    feature = "model-qwen35",
    feature = "text",
    target_os = "linux",
    any(feature = "cuda", feature = "hip")
))]
pub(crate) fn confirm_download() -> Result<bool> {
    eprint!("Download and use this exact source? [y/N] ");
    io::stderr().flush()?;
    let line = read_line(&mut io::stdin().lock())?;
    Ok(line.is_some_and(|line| matches!(line.trim(), "y" | "Y" | "yes" | "YES")))
}

#[cfg(any(
    all(feature = "cpu", feature = "model-qwen35", feature = "text"),
    all(
        feature = "model-qwen35",
        feature = "cache",
        feature = "text",
        target_os = "linux",
        any(feature = "cuda", feature = "hip")
    )
))]
fn read_line(reader: &mut impl BufRead) -> Result<Option<String>> {
    let mut line = String::new();
    let count = reader.take(MAX_PROMPT + 2).read_line(&mut line)?;
    if count == 0 {
        return Ok(None);
    }
    if line.ends_with('\n') {
        line.pop();
        if line.ends_with('\r') {
            line.pop();
        }
    }
    if line.len() as u64 > MAX_PROMPT {
        return Err(Error::Limit("prompt exceeds 1 MiB".into()));
    }
    Ok(Some(line))
}

/// A small repeated one-shot session, not conversation history. Each
/// request creates fresh model state on the same already prepared owner.
#[cfg(any(
    all(feature = "cpu", feature = "model-qwen35", feature = "text"),
    all(
        feature = "model-qwen35",
        feature = "cache",
        feature = "text",
        target_os = "linux",
        any(feature = "cuda", feature = "hip")
    )
))]
pub(super) fn session(model: &str, mut generate: impl FnMut(String) -> Result<()>) -> Result<()> {
    eprintln!("Independent prompts (no conversation history). /help or /exit.");
    let mut stdin = io::stdin().lock();
    loop {
        eprint!("You › ");
        io::stderr().flush()?;
        let Some(prompt) = read_line(&mut stdin)? else {
            break;
        };
        match prompt.as_str() {
            "/exit" | "/quit" => break,
            "/help" => eprintln!("/help · /model · /new · /exit. Each prompt is independent."),
            "/model" => eprintln!("{model}"),
            "/new" => eprintln!("Each request already starts with fresh state."),
            "" => continue,
            _ => {
                eprint!("Assistant › ");
                io::stderr().flush()?;
                // Report per-request errors without losing the prepared source.
                if let Err(error) = generate(prompt) {
                    crate::output::fail(crate::output::Failure::core("run", &error));
                }
            }
        }
    }
    Ok(())
}
