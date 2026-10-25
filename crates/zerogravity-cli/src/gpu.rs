//! Optional resident GPU text generation using the engine-owned GPU graph.
//! Tokenization/rendering are host operations; model execution and greedy
//! sampling use the existing resident implementation without CPU fallback.
use super::GpuBackend;
use std::{
    io::{self, Read, Write},
    path::PathBuf,
};
use zerogravity::gpu::ResidentModel;
use zerogravity::{Error, Result, qwen35::LoadedQwenModel, tokenize::QwenTokenizer};

const MIB: usize = 1024 * 1024;
const CONTEXT: usize = 512;
const PROMPT_BYTES: usize = MIB;

struct Options {
    model: PathBuf,
    prompt: Option<String>,
    max_tokens: usize,
    raw: bool,
    provider_cap: usize,
    device_cap: u64,
}

fn positive(value: String, flag: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Malformed(format!("{flag} requires a positive integer")))
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Options> {
    let model = args
        .next()
        .filter(|path| !path.is_empty() && !path.starts_with("--"))
        .ok_or_else(|| Error::Malformed("resident GPU run requires a model path".into()))?;
    let mut options = Options {
        model: PathBuf::from(model),
        prompt: None,
        max_tokens: 16,
        raw: false,
        provider_cap: 2048 * MIB,
        device_cap: 4096u64 * MIB as u64,
    };
    while let Some(flag) = args.next() {
        if flag == "--raw" {
            options.raw = true;
            continue;
        }
        match flag.as_str() {
            "--prompt" | "--max-tokens" | "--provider-mib" | "--device-mib" => {
                let value = args
                    .next()
                    .ok_or_else(|| Error::Malformed(format!("{flag} requires a value")))?;
                match flag.as_str() {
                    "--prompt" => options.prompt = Some(value),
                    "--max-tokens" => {
                        options.max_tokens = positive(value, &flag)?;
                        if options.max_tokens > CONTEXT {
                            return Err(Error::Limit(format!(
                                "--max-tokens must be in 1..={CONTEXT}"
                            )));
                        }
                    }
                    "--provider-mib" => {
                        options.provider_cap = positive(value, &flag)?
                            .checked_mul(MIB)
                            .ok_or_else(|| Error::Limit("provider GPU budget overflow".into()))?;
                    }
                    _ => {
                        options.device_cap = u64::try_from(positive(value, &flag)?)
                            .ok()
                            .and_then(|n| n.checked_mul(MIB as u64))
                            .ok_or_else(|| Error::Limit("physical GPU budget overflow".into()))?;
                    }
                }
            }
            _ => {
                return Err(Error::Malformed(format!(
                    "unknown resident GPU option {flag}"
                )));
            }
        }
    }
    if options.provider_cap as u64 > options.device_cap {
        return Err(Error::Limit(
            "provider GPU budget exceeds device budget".into(),
        ));
    }
    Ok(options)
}

pub(crate) async fn run(backend: GpuBackend, args: impl Iterator<Item = String>) -> Result<()> {
    let options = parse(args)?;
    let prompt = match options.prompt {
        Some(prompt) => prompt,
        None => {
            let mut prompt = String::new();
            io::stdin()
                .take(PROMPT_BYTES as u64 + 1)
                .read_to_string(&mut prompt)?;
            if prompt.len() > PROMPT_BYTES {
                return Err(Error::Limit("prompt stdin exceeds 1 MiB".into()));
            }
            prompt
        }
    };
    let parent = options
        .model
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let tokenizer = QwenTokenizer::from_file(&parent.join("tokenizer.json"))?;
    let prompt_tokens = tokenizer.prompt_tokens(&prompt, !options.raw)?;
    if prompt_tokens.is_empty() {
        return Err(Error::Malformed("prompt tokenization empty".into()));
    }
    // The final emitted token is not consumed again. Reserve the entire
    // worst-case request BEFORE model preparation or any GPU allocation.
    if prompt_tokens
        .len()
        .checked_add(options.max_tokens - 1)
        .is_none_or(|n| n > CONTEXT)
    {
        return Err(Error::Limit(format!(
            "resident GPU prompt and generation exceed {CONTEXT} consumed tokens"
        )));
    }
    let model = LoadedQwenModel::open(&options.model, &parent.join("config.json"))?;
    let plan = model.forward_plan(CONTEXT, true, true)?;
    let device = backend.device()?;
    let resident = ResidentModel::load(
        device,
        model,
        plan,
        options.provider_cap,
        options.device_cap,
    )
    .await?;
    let mut session = resident.session().await?;
    eprintln!("zerogravity: GPU prefill {} tokens", prompt_tokens.len());
    let mut readback = session.forward_greedy(&prompt_tokens).await?;
    let mut generated = Vec::with_capacity(options.max_tokens);
    for step in 0..options.max_tokens {
        let token = readback
            .device_token
            .take()
            .ok_or_else(|| Error::State("resident GPU returned no greedy token".into()))?;
        if token.id == tokenizer.eos() {
            break;
        }
        generated.push(token.id);
        if step + 1 < options.max_tokens {
            readback = session.forward_device_token(&token).await?;
        }
    }
    let text = tokenizer.decode(&generated, true)?;
    writeln!(io::stdout().lock(), "{text}")?;
    eprintln!(
        "zerogravity: {} generated tokens, {} total consumed (GPU)",
        generated.len(),
        session.binding().frontier(),
    );
    Ok(())
}
